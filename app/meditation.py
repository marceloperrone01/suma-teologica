"""Daily meditation: pick an article stably per day, weighted by Parte.

The chosen article is cached in SQLite so repeated calls on the same date
return the same article, and a 60-day window avoids repeats.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import date
from functools import lru_cache
from pathlib import Path

from app import store

ROOT = Path(__file__).resolve().parent.parent
ARTICLES_JSONL = ROOT / "data" / "articles.jsonl"

# Higher weight = more likely to be chosen. II-II and III are the most
# devotional parts (moral virtues; Christ and sacraments).
PART_WEIGHTS = {
    "I": 1.0,
    "I-II": 1.2,
    "II-II": 2.0,
    "III": 1.5,
    "Suplemento": 0.8,
    "Apêndice": 0.5,
}


@lru_cache(maxsize=1)
def _articles() -> list[dict]:
    with ARTICLES_JSONL.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _find_by_citacao(citacao: str) -> dict | None:
    for a in _articles():
        if a["citacao"] == citacao:
            return a
    return None


def pick_article_for(day: date) -> dict:
    """Return today's meditation article. Stable per day; weighted; avoids repeats."""
    iso = day.isoformat()
    cached = store.get_meditation(iso)
    if cached:
        found = _find_by_citacao(cached["citacao"])
        if found:
            return found

    recent = store.recent_meditation_citations(days=60)
    pool = [a for a in _articles() if a["citacao"] not in recent]
    if not pool:
        pool = _articles()

    seed = int(hashlib.sha256(iso.encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    weights = [PART_WEIGHTS.get(a["parte"], 1.0) for a in pool]
    chosen = rng.choices(pool, weights=weights, k=1)[0]
    store.set_meditation(iso, chosen["citacao"], chosen["parte"], chosen["questao"], chosen["artigo"])
    return chosen


MEDITATION_SYSTEM_PROMPT = """Você é um guia de oração que ajuda o leitor a meditar contemplativamente sobre um trecho da Suma Teológica de Santo Tomás de Aquino.

Sua tarefa: PARAFRASEAR o trecho fornecido em tom orante, devocional e em 2ª pessoa do singular (tu/te), para que o leitor possa receber a verdade no coração — não como objeto de estudo, mas como alimento para a alma.

Regras invioláveis:
- Não introduza doutrina nova. Apenas reformule o que está dito no trecho.
- Não use linguagem acadêmica ("Tomás afirma que...", "segundo o Aquinate..."). Fale ao leitor, não sobre o autor.
- 2-3 parágrafos curtos. Sobriedade — sem floreios literários.
- Comece interpelando o leitor ("Considera, alma minha...", "Pondera...", "Vê como...", etc.).
- Termine com uma breve aspiração ou oração de uma frase (uma jaculatória).
- No início, registre a referência canônica entre colchetes.
"""

MEDITATION_USER_TEMPLATE = """Referência: {citacao}
Questão: {titulo_questao}
Artigo: {titulo_artigo}

Trecho a ser parafraseado (respondeo de Santo Tomás):

{texto}
"""


def respondeo_or_fallback(article: dict) -> str:
    """Return the respondeo text, falling back to sed_contra or raw_body."""
    for key in ("respondeo", "sed_contra", "raw_body"):
        val = article.get(key)
        if val and val.strip():
            return val.strip()
    return ""
