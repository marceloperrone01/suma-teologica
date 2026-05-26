"""Build (or load) a BM25 index over chunks.jsonl for hybrid retrieval.

Run after data/chunks.jsonl exists:

    .venv/bin/python -m app.bm25_index

Produces data/bm25.pkl containing:
    {
        "chunk_ids": [str, ...],   # parallel to BM25 docs
        "tokenized": [[str, ...]], # not stored — only bm25 is
        "bm25": BM25Okapi,
    }
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_JSONL = ROOT / "data" / "chunks.jsonl"
BM25_PKL = ROOT / "data" / "bm25.pkl"

# Conservative PT stopword list. Theological terms are rarely in this list.
_STOPWORDS_PT = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas",
    "por", "pelo", "pela", "pelos", "pelas",
    "para", "com", "sem", "sob", "sobre", "entre",
    "e", "ou", "mas", "que", "se", "sim", "ja",
    "muito", "mais", "menos", "tambem", "tao",
    "ser", "estar", "ter", "haver",
    "e", "sao", "foi", "era", "sera", "sera",
    "isto", "isso", "aquilo", "este", "esta", "esse", "essa", "aquele", "aquela",
    "eu", "tu", "ele", "ela", "nos", "vos", "eles", "elas",
    "me", "te", "lhe", "nos", "vos", "lhes",
    "meu", "minha", "seu", "sua", "nosso", "nossa",
    "ao", "aos",
}


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents, split into word tokens, drop stopwords."""
    norm = _strip_accents(text.lower())
    return [t for t in _TOKEN_RE.findall(norm) if t not in _STOPWORDS_PT and len(t) > 1]


def build(input_path: Path = CHUNKS_JSONL, output_path: Path = BM25_PKL) -> None:
    print(f"Loading {input_path}…", file=sys.stderr)
    chunks: list[dict] = []
    seen: set[str] = set()
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["chunk_id"] in seen:
                continue
            seen.add(c["chunk_id"])
            chunks.append(c)
    print(f"Loaded {len(chunks)} unique chunks. Tokenizing…", file=sys.stderr)

    chunk_ids: list[str] = []
    tokenized: list[list[str]] = []
    for c in tqdm(chunks, desc="Tokenize"):
        chunk_ids.append(c["chunk_id"])
        tokenized.append(tokenize(c["text"]))

    print("Building BM25…", file=sys.stderr)
    bm25 = BM25Okapi(tokenized)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump({"chunk_ids": chunk_ids, "bm25": bm25}, f)
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1024:.1f} KB)", file=sys.stderr)


@lru_cache(maxsize=1)
def load() -> tuple[list[str], BM25Okapi]:
    """Load the serialized BM25 index. Raises FileNotFoundError if missing."""
    with BM25_PKL.open("rb") as f:
        data = pickle.load(f)
    return data["chunk_ids"], data["bm25"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=CHUNKS_JSONL)
    ap.add_argument("--output", type=Path, default=BM25_PKL)
    args = ap.parse_args(argv)
    build(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
