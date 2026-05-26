"""Daily meditation page."""

from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import store  # noqa: E402
from app.generator import DEFAULT_MODEL, backend_for, paraphrase  # noqa: E402
from app.meditation import pick_article_for  # noqa: E402

MODELS = [
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-opus-4-7",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
]

PT_MONTHS = [
    "janeiro", "fevereiro", "março", "abril", "maio", "junho",
    "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
]
PT_WEEKDAYS = [
    "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira",
    "sexta-feira", "sábado", "domingo",
]


def _pt_date(d: date) -> str:
    return f"{PT_WEEKDAYS[d.weekday()]}, {d.day} de {PT_MONTHS[d.month - 1]} de {d.year}"


def _missing_key_for(model: str) -> str | None:
    backend = backend_for(model)
    if backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY"
    if backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        return "OPENAI_API_KEY"
    return None


st.set_page_config(page_title="Meditação do dia — Suma Teológica", layout="wide")
st.title("📖 Meditação do dia")

with st.sidebar:
    st.subheader("Configurações")
    selected_date = st.date_input("Data", value=date.today(), max_value=date.today())
    model = st.selectbox(
        "Modelo (para paráfrase)",
        options=MODELS,
        index=MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in MODELS else 0,
    )
    missing = _missing_key_for(model)
    if missing:
        st.error(f"Defina `{missing}` no `.env`.")

st.caption(_pt_date(selected_date).capitalize())

article = pick_article_for(selected_date)

st.subheader(f"{article['citacao']}")
st.markdown(f"**Questão {article['questao']} — {article['titulo_questao']}**")
st.markdown(f"**Artigo {article['artigo']} — {article['titulo_artigo']}**")

st.divider()

respondeo = (article.get("respondeo") or "").strip()
sed_contra = (article.get("sed_contra") or "").strip()
objecoes = article.get("objecoes") or []
respostas = article.get("respostas_objecoes") or []

if respondeo:
    with st.expander("☀ Respondeo (Solução)", expanded=True):
        st.markdown(respondeo)
elif sed_contra:
    with st.expander("⚖ Sed contra", expanded=True):
        st.markdown(sed_contra)
else:
    st.warning("Este artigo não tem respondeo estruturado; veja o texto bruto abaixo.")
    st.markdown(article.get("raw_body", ""))

if sed_contra and respondeo:
    with st.expander("⚖ Sed contra"):
        st.markdown(sed_contra)

if objecoes:
    with st.expander(f"✦ Objeções ({len(objecoes)})"):
        for i, obj in enumerate(objecoes, start=1):
            st.markdown(f"**Objeção {i}.** {obj}")
            if i <= len(respostas):
                st.markdown(f"_Ad {i}._ {respostas[i-1]}")
            st.markdown("---")

st.divider()
st.subheader("✨ Paráfrase contemplativa")

iso = selected_date.isoformat()
cached_med = store.get_meditation(iso) or {}
existing_paraphrase = cached_med.get("paraphrase")

if existing_paraphrase:
    st.markdown(existing_paraphrase)
    if st.button("🔄 Regenerar"):
        if missing:
            st.error(f"Defina `{missing}` para regenerar.")
        else:
            with st.spinner("Compondo paráfrase contemplativa…"):
                try:
                    result = paraphrase(article, model=model)
                except KeyError as e:
                    st.error(f"Variável de ambiente faltando: {e}.")
                    st.stop()
                store.save_paraphrase(iso, result.text)
            st.rerun()
else:
    st.caption("Clique abaixo para receber uma paráfrase em tom orante, em 2ª pessoa.")
    if st.button("Gerar paráfrase contemplativa", type="primary", disabled=bool(missing)):
        with st.spinner("Compondo paráfrase contemplativa…"):
            try:
                result = paraphrase(article, model=model)
            except KeyError as e:
                st.error(f"Variável de ambiente faltando: {e}.")
                st.stop()
            store.save_paraphrase(iso, result.text)
            st.session_state["meditation_usage"] = result
        st.rerun()

usage = st.session_state.get("meditation_usage")
if usage:
    st.caption(
        f"Backend: {usage.backend} · Modelo: {usage.model} · "
        f"in={usage.input_tokens} out={usage.output_tokens}"
    )

st.divider()
with st.expander("Histórico recente (últimos 14 dias)"):
    today = date.today()
    rows = []
    for offset in range(14):
        d = today - timedelta(days=offset)
        m = store.get_meditation(d.isoformat())
        if m:
            rows.append((d, m["citacao"]))
    if rows:
        for d, citacao in rows:
            st.markdown(f"- **{d.isoformat()}** — {citacao}")
    else:
        st.caption("Ainda não há meditações registradas.")
