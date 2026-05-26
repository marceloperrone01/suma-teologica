"""Convert articles.jsonl into chunks.jsonl.

Each chunk is a unit of retrieval. Strategy:
  - One chunk per Article section (each objeção, sed contra, respondeo,
    each resposta-à-objeção).
  - One summary chunk per Article: titles + respondeo (for broad queries).
  - Sections longer than MAX_TOKENS (approx) are split into sub-chunks with
    overlap, preserving citation metadata.
  - Articles missing structured sections fall back to a single raw_body chunk.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTICLES_JSONL = ROOT / "data" / "articles.jsonl"
CHUNKS_JSONL = ROOT / "data" / "chunks.jsonl"

# Approx tokens-per-char for Portuguese ≈ 0.25 (4 chars/token).
# We target ~800 tokens / chunk = ~3200 chars.
MAX_CHARS = 3200
OVERLAP_CHARS = 320


@dataclass
class Chunk:
    chunk_id: str
    text: str
    parte: str
    questao: int
    artigo: int
    secao: str  # "objecao_1" | "sed_contra" | "respondeo" | "resposta_1" | "resumo" | "raw"
    citacao: str
    titulo_questao: str
    titulo_artigo: str
    sub_idx: int = 0


def split_long(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Backtrack to a paragraph or sentence boundary if possible
            window_start = max(start + max_chars - 400, start)
            cut = text.rfind("\n\n", window_start, end)
            if cut == -1:
                cut = text.rfind(". ", window_start, end)
            if cut != -1 and cut > start + max_chars // 2:
                end = cut
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, end - max_chars // 4)
    return [p for p in parts if p]


def make_chunks(art: dict) -> list[Chunk]:
    out: list[Chunk] = []
    base = dict(
        parte=art["parte"],
        questao=art["questao"],
        artigo=art["artigo"],
        citacao=art["citacao"],
        titulo_questao=art["titulo_questao"],
        titulo_artigo=art["titulo_artigo"],
    )

    def add(secao: str, text: str) -> None:
        text = text.strip()
        if not text:
            return
        pieces = split_long(text)
        for i, piece in enumerate(pieces):
            cid = f"{art['parte']}|q{art['questao']}|a{art['artigo']}|{secao}|{i}"
            out.append(Chunk(chunk_id=cid, text=piece, secao=secao, sub_idx=i, **base))

    for i, obj in enumerate(art.get("objecoes", []), start=1):
        add(f"objecao_{i}", obj)

    add("sed_contra", art.get("sed_contra", ""))
    add("respondeo", art.get("respondeo", ""))

    for i, rep in enumerate(art.get("respostas_objecoes", []), start=1):
        add(f"resposta_{i}", rep)

    # Summary chunk: title + respondeo (or raw_body if no respondeo)
    summary_body = art.get("respondeo") or art.get("raw_body", "")
    if summary_body:
        summary = f"{art['titulo_questao']} — {art['titulo_artigo']}\n\n{summary_body[:1500]}"
        add("resumo", summary)

    # Fallback for fully unstructured articles: emit raw_body as a single chunk
    if not out and art.get("raw_body"):
        add("raw", art["raw_body"])

    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ARTICLES_JSONL)
    ap.add_argument("--output", type=Path, default=CHUNKS_JSONL)
    args = ap.parse_args(argv)

    articles = [json.loads(line) for line in args.input.open(encoding="utf-8")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    section_counts: dict[str, int] = {}
    with args.output.open("w", encoding="utf-8") as f:
        for art in articles:
            for ch in make_chunks(art):
                f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")
                total += 1
                section_counts[ch.secao] = section_counts.get(ch.secao, 0) + 1
    print(f"Wrote {total} chunks to {args.output}", file=sys.stderr)
    print("By type:", file=sys.stderr)
    # collapse subtypes (objecao_1 → objecao, resposta_1 → resposta)
    rolled: dict[str, int] = {}
    for k, v in section_counts.items():
        prefix = k.split("_")[0]
        rolled[prefix] = rolled.get(prefix, 0) + v
    for k, v in sorted(rolled.items(), key=lambda kv: -kv[1]):
        print(f"  {k}: {v}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
