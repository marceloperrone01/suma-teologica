"""Embed TCC chunks with BAAI/bge-m3 and persist to ChromaDB collection 'tcc_chunks'.

Run after data/tcc/tcc_chunks.jsonl has been produced by app.tcc_chunker.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_JSONL = ROOT / "data" / "tcc" / "tcc_chunks.jsonl"
CHROMA_DIR = ROOT / "data" / "chroma"
COLLECTION = "tcc_chunks"
EMBEDDING_MODEL = "BAAI/bge-m3"


def load_chunks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def make_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device}…", file=sys.stderr)
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model = model.half()
    return model


def ensure_collection(client: chromadb.ClientAPI, name: str, recreate: bool = False) -> chromadb.Collection:
    if recreate:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=CHUNKS_JSONL)
    ap.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--recreate", action="store_true", help="Drop existing collection first")
    ap.add_argument("--limit", type=int, default=0, help="Limit chunks (for smoke testing)")
    args = ap.parse_args(argv)

    chunks = load_chunks(args.input)
    if args.limit:
        chunks = chunks[: args.limit]

    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for c in chunks:
        if c["chunk_id"] in seen_ids:
            continue
        seen_ids.add(c["chunk_id"])
        deduped.append(c)
    if len(deduped) != len(chunks):
        print(f"Dropped {len(chunks) - len(deduped)} duplicate chunks", file=sys.stderr)
    chunks = deduped
    print(f"Loaded {len(chunks)} chunks", file=sys.stderr)

    args.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(args.chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = ensure_collection(client, COLLECTION, recreate=args.recreate)

    existing: set[str] = set()
    if not args.recreate and collection.count() > 0:
        existing_ids = collection.get(include=[])["ids"]
        existing = set(existing_ids)
        print(f"Found {len(existing)} existing chunks, will skip", file=sys.stderr)

    todo = [c for c in chunks if c["chunk_id"] not in existing]
    if not todo:
        print("Nothing to do. Collection is up to date.", file=sys.stderr)
        return 0

    model = make_model()
    bs = args.batch_size

    with tqdm(total=len(todo), desc="Embedding+upsert") as pbar:
        for i in range(0, len(todo), bs):
            batch = todo[i : i + bs]
            texts = [c["text"] for c in batch]
            embeddings = model.encode(
                texts,
                batch_size=bs,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            collection.upsert(
                ids=[c["chunk_id"] for c in batch],
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=[
                    {
                        "doc_id": c["doc_id"],
                        "source_file": c["source_file"],
                        "section": c["section"],
                        "titulo": c["titulo"],
                        "citacao": c["citacao"],
                    }
                    for c in batch
                ],
            )
            pbar.update(len(batch))

    print(f"Done. Collection '{COLLECTION}' has {collection.count()} chunks.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
