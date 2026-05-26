"""Parses the extracted Suma Teológica text into scholastic structure.

Reads data/extracted.txt and writes data/articles.jsonl, one JSON object per
Article with full section breakdown (objeções, sed contra, respondeo, respostas).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTRACTED = ROOT / "data" / "extracted.txt"
ARTICLES_JSONL = ROOT / "data" / "articles.jsonl"

PARTS = ["I", "I-II", "II-II", "III", "Suplemento", "Apêndice"]

RE_QUESTAO = re.compile(r"^\s*Questão\s+(\d+)\s*[:.\-—–―]\s*(.+?)\s*$")
RE_ARTIGO = re.compile(r"^\s*Art\.?\s*(\d+)\s*[—\-–―‐‑‒−]\s*(.+?)\s*$")
RE_SED_CONTRA = re.compile(r"^\s*Mas,?\s+em\s+contrário\b", re.IGNORECASE | re.MULTILINE)
RE_SOLUCAO = re.compile(r"^\s*SOLU[ÇC][ÃÂA]O\.\s*[—\-–―‐‑‒−]?\s*", re.MULTILINE)
RE_RESPONDEO_INLINE = re.compile(r"\bSOLU[ÇC][ÃÂA]O\.\s*[—\-–―‐‑‒−]?")

ORDINALS = [
    "PRIMEIRA", "SEGUNDA", "TERCEIRA", "QUARTA", "QUINTA",
    "SEXTA", "SÉTIMA", "OITAVA", "NONA", "DÉCIMA",
    "UNDÉCIMA", "DUODÉCIMA",
]
RE_RESPOSTA = re.compile(
    r"(?:DONDE\s+A\s+)?RESPOSTA\s+(?:À|A)\s+(" + "|".join(ORDINALS) + r")(?:\s+OBJEÇÃO)?\.\s*[—\-–―‐‑‒−]"
)

# Lines starting an objection. Variants seen:
#   "1. — Pois..."  "1. Pois..."  "2. Demais. — ..."  "2. Demais — ..."  "1 — Pois..."
# Accept "N." or "N <dash>"; subsequent objections must be sequential
# (validated in split_objections).
RE_OBJ_START = re.compile(r"^\s*(\d+)(?:\.|\s+[—\-–―])\s*")


@dataclass
class Article:
    parte: str
    questao: int
    artigo: int
    titulo_questao: str
    titulo_artigo: str
    citacao: str
    objecoes: list[str] = field(default_factory=list)
    sed_contra: str = ""
    respondeo: str = ""
    respostas_objecoes: list[str] = field(default_factory=list)
    raw_body: str = ""


def find_content_start(lines: list[str]) -> int:
    """Find the line where actual content begins (first body Questão 1).

    The PDF starts with TOC where Questão lines have trailing dots+page nums.
    The body starts where 'Questão 1: ...' appears without trailing dots.
    """
    for i, line in enumerate(lines):
        m = RE_QUESTAO.match(line)
        if not m:
            continue
        if "..." in line:
            continue
        if int(m.group(1)) == 1:
            return i
    raise RuntimeError("Could not locate body content start")


def part_for_index(idx: int) -> str:
    if idx < len(PARTS):
        return PARTS[idx]
    return f"Apêndice-{idx - len(PARTS) + 2}"


def citation(parte: str, q: int, a: int) -> str:
    return f"ST {parte}, q.{q}, a.{a}"


def parse_articles(lines: list[str]) -> list[Article]:
    start = find_content_start(lines)
    articles: list[Article] = []
    part_idx = 0
    seen_first_article = False
    current_q: int | None = None
    current_q_title = ""
    current_article: Article | None = None
    body_buf: list[str] = []

    def flush_article() -> None:
        nonlocal current_article, body_buf
        if current_article is not None:
            current_article.raw_body = "\n".join(body_buf).strip()
            split_sections(current_article)
            articles.append(current_article)
        body_buf = []
        current_article = None

    for line in lines[start:]:
        # Skip TOC-like lines that may have slipped through (dots + page number).
        if "..." in line and re.search(r"\.{6,}\s*\d+\s*$", line):
            continue

        mq = RE_QUESTAO.match(line)
        if mq and "..." not in line:
            q_num = int(mq.group(1))
            q_title = mq.group(2).strip()
            # Detect part transition: Questão 1 after we've already seen articles
            if q_num == 1 and seen_first_article:
                part_idx += 1
            current_q = q_num
            current_q_title = q_title
            flush_article()
            continue

        ma = RE_ARTIGO.match(line)
        if ma and current_q is not None:
            a_num = int(ma.group(1))
            a_title = ma.group(2).strip()
            # Skip false positives: article titles in body should be short-ish and
            # not contain trailing page-number dots
            if "..." in line:
                continue
            flush_article()
            parte = part_for_index(part_idx)
            current_article = Article(
                parte=parte,
                questao=current_q,
                artigo=a_num,
                titulo_questao=current_q_title,
                titulo_artigo=a_title,
                citacao=citation(parte, current_q, a_num),
            )
            seen_first_article = True
            continue

        if current_article is not None:
            body_buf.append(line.rstrip())

    flush_article()
    return articles


def split_sections(art: Article) -> None:
    """Populate objecoes, sed_contra, respondeo, respostas_objecoes from raw_body."""
    body = art.raw_body

    m_sc = RE_SED_CONTRA.search(body)
    m_sol = RE_SOLUCAO.search(body) or RE_RESPONDEO_INLINE.search(body)
    m_resp = RE_RESPOSTA.search(body)

    pre_objs_end = m_sc.start() if m_sc else (m_sol.start() if m_sol else (m_resp.start() if m_resp else len(body)))
    objecoes_text = body[:pre_objs_end]

    # When SOLUÇÃO is present, sed contra ends there. Otherwise, fall back to
    # the next blank line after 'Mas, em contrário' (sed contra is typically a
    # single short paragraph), and respondeo flows from there until the first
    # RESPOSTA À X marker.
    if m_sc:
        if m_sol:
            sc_end = m_sol.start()
        else:
            blank = body.find("\n\n", m_sc.end())
            sc_end = blank if blank != -1 else (m_resp.start() if m_resp else len(body))
        sed_contra_text = body[m_sc.start():sc_end].strip()
    else:
        sc_end = m_sol.start() if m_sol else (m_resp.start() if m_resp else len(body))
        sed_contra_text = ""

    if m_sol:
        sol_end = m_resp.start() if m_resp else len(body)
        respondeo_text = body[m_sol.start():sol_end].strip()
    elif m_sc and m_resp:
        respondeo_text = body[sc_end:m_resp.start()].strip()
    else:
        respondeo_text = ""

    respostas_text = body[m_resp.start():].strip() if m_resp else ""

    art.objecoes = split_objections(objecoes_text)
    art.sed_contra = sed_contra_text
    art.respondeo = respondeo_text
    art.respostas_objecoes = split_respostas(respostas_text)


def split_objections(text: str) -> list[str]:
    """Split text into individual objections.

    Objections are numbered sequentially (1, 2, 3, ...). We accept candidate
    starts only if they continue the expected sequence — this filters out
    stray '12.' or '1.' lines that appear inside an objection's body.
    """
    lines = text.split("\n")
    candidates: list[tuple[int, int]] = []  # (line_idx, number)
    for i, line in enumerate(lines):
        m = RE_OBJ_START.match(line)
        if m:
            candidates.append((i, int(m.group(1))))
    # Find the first candidate that starts with 1, then take subsequent matches
    # that continue the sequence (2, 3, 4, ...).
    starts: list[int] = []
    expected = 1
    for line_idx, num in candidates:
        if num == expected:
            starts.append(line_idx)
            expected += 1
    if not starts:
        return []
    objs: list[str] = []
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else len(lines)
        chunk = "\n".join(lines[s:e]).strip()
        if chunk:
            objs.append(chunk)
    return objs


def split_respostas(text: str) -> list[str]:
    """Split respostas-às-objeções section by RESPOSTA À X markers."""
    if not text:
        return []
    matches = list(RE_RESPOSTA.finditer(text))
    if not matches:
        return []
    out: list[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append(text[start:end].strip())
    return out


def write_jsonl(articles: list[Article], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for a in articles:
            d = asdict(a)
            # Move raw_body out to keep articles.jsonl readable; keep last for debugging.
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="Print N random parsed articles (no write)")
    ap.add_argument("--input", type=Path, default=EXTRACTED)
    ap.add_argument("--output", type=Path, default=ARTICLES_JSONL)
    args = ap.parse_args(argv)

    text = args.input.read_text(encoding="utf-8")
    lines = text.split("\n")
    articles = parse_articles(lines)

    print(f"Parsed {len(articles)} articles", file=sys.stderr)
    by_part: dict[str, int] = {}
    for a in articles:
        by_part[a.parte] = by_part.get(a.parte, 0) + 1
    for p, n in by_part.items():
        print(f"  {p}: {n} articles", file=sys.stderr)

    if args.sample > 0:
        rng = random.Random(42)
        sampled = rng.sample(articles, min(args.sample, len(articles)))
        for a in sampled:
            print("=" * 80)
            print(f"{a.citacao} — {a.titulo_artigo}")
            print(f"  Questão: {a.titulo_questao}")
            print(f"  Objeções: {len(a.objecoes)}")
            print(f"  Sed contra: {len(a.sed_contra)} chars")
            print(f"  Respondeo: {len(a.respondeo)} chars")
            print(f"  Respostas obj.: {len(a.respostas_objecoes)}")
            if a.objecoes:
                print(f"  [obj 1 preview] {a.objecoes[0][:150]}...")
            if a.respondeo:
                print(f"  [respondeo preview] {a.respondeo[:200]}...")
        return 0

    write_jsonl(articles, args.output)
    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
