"""Chunk TCC articles into retrieval units by section.

Run after tcc_articles.jsonl exists:

    .venv/bin/python -m app.tcc_chunker

Produces data/tcc/tcc_chunks.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_JSONL = ROOT / "data" / "tcc" / "tcc_articles.jsonl"
OUT_JSONL = ROOT / "data" / "tcc" / "tcc_chunks.jsonl"

MAX_CHARS = 3200
OVERLAP_CHARS = 320

# Section heading regex — matches lines like "ABSTRACT", "INTRODUCTION", "2. METHODS"
_SECTION_RE = re.compile(
    r"^\s*(?:\d+\.?\s+)?"
    r"(ABSTRACT|INTRODUCTION|BACKGROUND|METHODS?|METHODOLOGY|MATERIALS?\s+AND\s+METHODS?|"
    r"PARTICIPANTS?|PROCEDURE|MEASURES?|RESULTS?|FINDINGS?|DISCUSSION|"
    r"CONCLUSION|CONCLUSIONS|IMPLICATIONS|LIMITATIONS?|ACKNOWLEDGEMENTS?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Stop chunking when we hit references
_REFS_RE = re.compile(
    r"^\s*(?:\d+\.?\s+)?(REFERENCES?|BIBLIOGRAPHY|REFERÊNCIAS)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_SECTION_LABEL_MAP = {
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "introduction",
    "methods": "methods",
    "method": "methods",
    "methodology": "methods",
    "materials and methods": "methods",
    "participants": "methods",
    "procedure": "methods",
    "measures": "methods",
    "results": "results",
    "findings": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "implications": "discussion",
    "limitations": "discussion",
    "acknowledgements": "other",
    "acknowledgments": "other",
}


def _normalize_section(heading: str) -> str:
    key = re.sub(r"\s+", " ", heading.strip().lower())
    # Trim leading numbering
    key = re.sub(r"^\d+\.?\s+", "", key)
    return _SECTION_LABEL_MAP.get(key, "other")


def _doc_id(source_file: str) -> str:
    """Create a slug from source_file for use in chunk IDs."""
    slug = unicodedata.normalize("NFKD", source_file)
    slug = "".join(c for c in slug if not unicodedata.combining(c))
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug).strip("-").lower()
    return slug[:60]


def _split_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split text into overlapping chunks respecting paragraph boundaries."""
    if len(text) <= max_chars:
        return [text]

    # Minimum advance per step to avoid near-infinite loop when boundary is near start
    min_advance = max_chars - overlap

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Try to split at paragraph boundary, searching only in the back half to guarantee advance
        search_start = start + min_advance
        boundary = text.rfind("\n\n", search_start, end)
        if boundary == -1:
            boundary = text.rfind("\n", search_start, end)
        if boundary == -1:
            boundary = text.rfind(". ", search_start, end)
        if boundary == -1:
            boundary = end
        chunks.append(text[start : boundary + 1])
        start = boundary + 1 - overlap
    return chunks


def _extract_sections(full_text: str) -> list[tuple[str, str]]:
    """Return list of (section_label, section_text) from full text."""
    # Find where references start — stop there
    refs_m = _REFS_RE.search(full_text)
    text = full_text[: refs_m.start()] if refs_m else full_text

    # Find all section headings and their positions
    matches = list(_SECTION_RE.finditer(text))

    if not matches:
        return [("other", text.strip())]

    sections: list[tuple[str, str]] = []

    # Text before first heading → treat as abstract/other
    pre_text = text[: matches[0].start()].strip()
    if pre_text:
        sections.append(("other", pre_text))

    for i, m in enumerate(matches):
        heading = m.group(1)
        label = _normalize_section(heading)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((label, body))

    return sections


def chunk_article(article: dict) -> list[dict]:
    doc_id = _doc_id(article["source_file"])
    sections = _extract_sections(article["full_text"])

    chunks: list[dict] = []
    for section_label, body in sections:
        parts = _split_text(body)
        for sub_idx, part in enumerate(parts):
            chunk_id = f"{doc_id}|{section_label}|{sub_idx}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": part.strip(),
                    "doc_id": doc_id,
                    "source_file": article["source_file"],
                    "section": section_label,
                    "titulo": article["title"],
                    "citacao": article["citacao"],
                    "sub_idx": sub_idx,
                }
            )
    return chunks


def build(input_path: Path = IN_JSONL, output_path: Path = OUT_JSONL) -> None:
    articles: list[dict] = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            articles.append(json.loads(line))
    print(f"Loaded {len(articles)} articles.", file=sys.stderr)

    all_chunks: list[dict] = []
    for art in articles:
        chunks = chunk_article(art)
        all_chunks.extend(chunks)
        print(f"  {art['source_file'][:50]}: {len(chunks)} chunks", file=sys.stderr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    section_counts: dict[str, int] = {}
    for c in all_chunks:
        section_counts[c["section"]] = section_counts.get(c["section"], 0) + 1
    print(f"\nTotal chunks: {len(all_chunks)}", file=sys.stderr)
    for sec, cnt in sorted(section_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {sec}: {cnt}", file=sys.stderr)
    print(f"Wrote {output_path}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=IN_JSONL)
    ap.add_argument("--output", type=Path, default=OUT_JSONL)
    args = ap.parse_args(argv)
    build(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
