"""Parse PDFs and DOCX from tcc/ into tcc_articles.jsonl.

Run:
    .venv/bin/python -m app.tcc_parser

Produces data/tcc/tcc_articles.jsonl — one JSON object per document.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TCC_DIR = ROOT / "tcc"
OUT_DIR = ROOT / "data" / "tcc"
OUT_JSONL = OUT_DIR / "tcc_articles.jsonl"


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalize_spaces(text: str) -> str:
    """Collapse spaced-out letters like 'I N T R O D U C T I O N' → 'INTRODUCTION'."""
    return re.sub(r"(?<=[A-Z]) (?=[A-Z])", "", text)


def _extract_title(text: str) -> str:
    """Heuristic: grab first substantial non-boilerplate line (before Abstract/Introduction)."""
    lines = text.splitlines()
    boilerplate = re.compile(
        r"(abstract|introduction|journal|doi|vol\.|available|received|accepted|©|www\.|http|issn)",
        re.IGNORECASE,
    )
    for line in lines[:50]:
        stripped = line.strip()
        if len(stripped) < 15 or len(stripped) > 300:
            continue
        if boilerplate.search(stripped):
            break
        if re.match(r"^\d", stripped):
            continue
        return stripped
    return Path("unknown").stem


def _extract_citacao(text: str, source_file: str) -> str:
    """Best-effort APA-style short citation from first page."""
    year_m = re.search(r"\b(19|20)\d{2}\b", text[:2000])
    year = year_m.group(0) if year_m else "n.d."

    # Try to find "Author, A., Author, B." pattern
    author_m = re.search(
        r"([A-Z][a-záéíóúâêîôûãõçàèìòùü\-]+(?:,\s+[A-Z]\.(?:\s+[A-Z]\.)?)?(?:\s+(?:&|and)\s+[A-Z][a-záéíóúâêîôûãõçàèìòùü\-]+)?)",
        text[:1500],
    )
    if author_m:
        name = author_m.group(1).strip().split(",")[0].strip()
        return f"{name} et al., {year}"

    # Fallback: use filename slug
    slug = re.sub(r"[^a-zA-Z0-9]+", " ", source_file).strip().split()
    name = slug[0].capitalize() if slug else "Autor"
    return f"{name} et al., {year}"


def parse_pdf(path: Path) -> dict | None:
    try:
        result = subprocess.run(
            ["pdftotext", "-enc", "UTF-8", str(path), "-"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            print(f"  [WARN] pdftotext failed for {path.name}: {result.stderr[:200]}", file=sys.stderr)
            return None
        text = _normalize_spaces(result.stdout)
    except FileNotFoundError:
        print("  [ERROR] pdftotext not found. Install poppler-utils.", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  [WARN] Timeout extracting {path.name}", file=sys.stderr)
        return None

    source_file = path.stem
    title = _extract_title(text)
    citacao = _extract_citacao(text, source_file)
    return {
        "source_file": source_file,
        "title": title,
        "citacao": citacao,
        "full_text": text.strip(),
    }


def parse_docx(path: Path) -> dict | None:
    try:
        from docx import Document
    except ImportError:
        print("  [ERROR] python-docx not installed. Run: pip install python-docx", file=sys.stderr)
        return None

    try:
        doc = Document(str(path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs)
    except Exception as e:
        print(f"  [WARN] Could not parse {path.name}: {e}", file=sys.stderr)
        return None

    source_file = path.stem
    title = _extract_title(text)
    citacao = _extract_citacao(text, source_file)
    return {
        "source_file": source_file,
        "title": title,
        "citacao": citacao,
        "full_text": text.strip(),
    }


def parse_all(tcc_dir: Path = TCC_DIR) -> list[dict]:
    articles: list[dict] = []
    files = sorted(tcc_dir.iterdir())
    for path in files:
        if path.suffix.lower() == ".pdf":
            print(f"  Parsing PDF: {path.name}", file=sys.stderr)
            art = parse_pdf(path)
        elif path.suffix.lower() in {".docx", ".doc"}:
            print(f"  Parsing DOCX: {path.name}", file=sys.stderr)
            art = parse_docx(path)
        else:
            continue
        if art:
            articles.append(art)
    return articles


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tcc-dir", type=Path, default=TCC_DIR)
    ap.add_argument("--output", type=Path, default=OUT_JSONL)
    args = ap.parse_args(argv)

    print(f"Parsing documents from {args.tcc_dir}…", file=sys.stderr)
    articles = parse_all(args.tcc_dir)
    print(f"Parsed {len(articles)} documents.", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for art in articles:
            f.write(json.dumps(art, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
