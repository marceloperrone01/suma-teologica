"""Retrieval from ChromaDB (dense) and optional BM25 (lexical) hybrid via RRF."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from app.bm25_index import tokenize as bm25_tokenize_pt

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "data" / "chroma"
EMBEDDING_MODEL = "BAAI/bge-m3"

_BM25_PATHS = {
    "suma": ROOT / "data" / "bm25.pkl",
    "tcc": ROOT / "data" / "tcc" / "tcc_bm25.pkl",
}
_COLLECTIONS = {
    "suma": "suma_chunks",
    "tcc": "tcc_chunks",
}


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    citacao: str
    secao: str
    titulo_questao: str
    titulo_artigo: str
    parte: str
    questao: int
    artigo: int
    distance: float  # 0.0 for hybrid results (RRF score is in `rrf_score`)
    rrf_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class TccChunk:
    chunk_id: str
    text: str
    citacao: str
    section: str
    titulo: str
    source_file: str
    doc_id: str
    distance: float
    rrf_score: float = 0.0
    rerank_score: float = 0.0


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(EMBEDDING_MODEL, device=device)
    if device == "cuda":
        m = m.half()
    return m


@lru_cache(maxsize=4)
def _get_collection(name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(name)


def _collection() -> chromadb.Collection:
    return _get_collection("suma_chunks")


@lru_cache(maxsize=4)
def _load_bm25_for(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["chunk_ids"], data["bm25"]


def _row_to_chunk(cid: str, doc: str, meta: dict, dist: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        text=doc,
        citacao=meta["citacao"],
        secao=meta["secao"],
        titulo_questao=meta["titulo_questao"],
        titulo_artigo=meta["titulo_artigo"],
        parte=meta["parte"],
        questao=int(meta["questao"]),
        artigo=int(meta["artigo"]),
        distance=float(dist),
    )


def _row_to_tcc_chunk(cid: str, doc: str, meta: dict, dist: float = 0.0) -> TccChunk:
    return TccChunk(
        chunk_id=cid,
        text=doc,
        citacao=meta["citacao"],
        section=meta["section"],
        titulo=meta["titulo"],
        source_file=meta["source_file"],
        doc_id=meta["doc_id"],
        distance=float(dist),
    )


def _tokenize_for(query: str, dataset: str) -> list[str]:
    if dataset == "tcc":
        from app.tcc_bm25 import tokenize as tokenize_en
        return tokenize_en(query)
    return bm25_tokenize_pt(query)


def search(
    query: str,
    top_k: int = 8,
    where: dict | None = None,
    dataset: str = "suma",
) -> list[RetrievedChunk] | list[TccChunk]:
    """Dense search via ChromaDB."""
    model = _model()
    col = _get_collection(_COLLECTIONS[dataset])
    embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    res = col.query(
        query_embeddings=[embedding.tolist()],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    row_fn = _row_to_tcc_chunk if dataset == "tcc" else _row_to_chunk
    out = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append(row_fn(cid, doc, meta, dist))
    return out


def bm25_search(
    query: str,
    top_k: int = 8,
    where: dict | None = None,
    dataset: str = "suma",
) -> list[RetrievedChunk] | list[TccChunk]:
    """Lexical search via BM25. Fetches metadata from Chroma for parity.

    `where` filter is applied AFTER BM25 ranking, so increase top_k upstream when filtering.
    """
    pkl_path = str(_BM25_PATHS[dataset])
    chunk_ids, bm25 = _load_bm25_for(pkl_path)
    tokens = _tokenize_for(query, dataset)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    candidates = ranked[: max(top_k * 4, 32)]
    candidate_ids = [chunk_ids[i] for i in candidates if scores[i] > 0.0]
    if not candidate_ids:
        return []

    col = _get_collection(_COLLECTIONS[dataset])
    fetched = col.get(ids=candidate_ids, include=["documents", "metadatas"])
    by_id = {cid: (doc, meta) for cid, doc, meta in zip(fetched["ids"], fetched["documents"], fetched["metadatas"])}

    row_fn = _row_to_tcc_chunk if dataset == "tcc" else _row_to_chunk
    out = []
    for cid in candidate_ids:
        if cid not in by_id:
            continue
        doc, meta = by_id[cid]
        if where:
            if not all(meta.get(k) == v for k, v in where.items()):
                continue
        out.append(row_fn(cid, doc, meta, dist=0.0))
        if len(out) >= top_k:
            break
    return out


def hybrid_search(
    query: str,
    top_k: int = 8,
    where: dict | None = None,
    dataset: str = "suma",
    *,
    rrf_k: int = 60,
    per_retriever_k: int | None = None,
) -> list[RetrievedChunk] | list[TccChunk]:
    """Hybrid dense + BM25 fused with Reciprocal Rank Fusion.

    RRF score for doc d:  sum_r 1 / (rrf_k + rank_r(d))
    where rank_r is 1-indexed rank in retriever r's list.

    If BM25 index is missing, falls back to dense-only.
    """
    n = per_retriever_k or max(top_k * 3, 24)
    dense = search(query, top_k=n, where=where, dataset=dataset)
    try:
        lex = bm25_search(query, top_k=n, where=where, dataset=dataset)
    except FileNotFoundError:
        return dense[:top_k]

    scores: dict[str, float] = {}
    payload: dict[str, RetrievedChunk | TccChunk] = {}
    for ranked in (dense, lex):
        for rank, c in enumerate(ranked, start=1):
            scores[c.chunk_id] = scores.get(c.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
            payload.setdefault(c.chunk_id, c)

    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out = []
    for cid, sc in ordered[:top_k]:
        c = payload[cid]
        c.rrf_score = sc
        out.append(c)
    return out


RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=1)
def _reranker():
    """Lazy-load the CrossEncoder reranker (~568M params, FP16 on CUDA ≈ 1.1 GB)."""
    import torch
    from sentence_transformers import CrossEncoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rr = CrossEncoder(RERANKER_MODEL, device=device, trust_remote_code=False)
    if device == "cuda":
        rr.model.half()
    return rr


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int = 8) -> list[RetrievedChunk]:
    """Re-score `chunks` against `query` with bge-reranker-v2-m3 and return top-k.

    Useful after a wide hybrid_search (e.g. top_k=24) to extract the most
    semantically relevant passages before passing them to the LLM.
    """
    if not chunks:
        return []
    rr = _reranker()
    scores = rr.predict([(query, c.text) for c in chunks])
    ranked = sorted(zip(chunks, scores), key=lambda kv: float(kv[1]), reverse=True)
    out: list[RetrievedChunk] = []
    for c, s in ranked[:top_k]:
        c.rerank_score = float(s)
        out.append(c)
    return out


def dedupe_by_article(chunks: list[RetrievedChunk], max_per_article: int = 2) -> list[RetrievedChunk]:
    """Avoid flooding a single article: keep at most N chunks per (parte,q,a)."""
    counts: dict[tuple, int] = {}
    out: list[RetrievedChunk] = []
    for c in chunks:
        key = (c.parte, c.questao, c.artigo)
        if counts.get(key, 0) >= max_per_article:
            continue
        counts[key] = counts.get(key, 0) + 1
        out.append(c)
    return out


def dedupe_by_source(chunks: list[TccChunk], max_per_doc: int = 3) -> list[TccChunk]:
    """Keep at most N chunks per (doc_id, section) to avoid flooding from one paper."""
    counts: dict[tuple, int] = {}
    out: list[TccChunk] = []
    for c in chunks:
        key = (c.doc_id, c.section)
        if counts.get(key, 0) >= max_per_doc:
            continue
        counts[key] = counts.get(key, 0) + 1
        out.append(c)
    return out
