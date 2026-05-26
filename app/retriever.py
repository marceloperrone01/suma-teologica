"""Dense retrieval from ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "suma_chunks"
EMBEDDING_MODEL = "BAAI/bge-m3"


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
    distance: float


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(EMBEDDING_MODEL, device=device)
    if device == "cuda":
        m = m.half()
    return m


@lru_cache(maxsize=1)
def _collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(COLLECTION)


def search(query: str, top_k: int = 8, where: dict | None = None) -> list[RetrievedChunk]:
    """Search the collection for the most relevant chunks to `query`.

    Args:
        query: Natural-language question or topic.
        top_k: Number of results to return.
        where: Optional Chroma metadata filter, e.g. {"parte": "II-II"}.
    """
    model = _model()
    col = _collection()
    embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    res = col.query(
        query_embeddings=[embedding.tolist()],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    out: list[RetrievedChunk] = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        out.append(
            RetrievedChunk(
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
        )
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
