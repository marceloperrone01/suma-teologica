"""Streamlit chat UI for the Suma Teológica RAG."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from app.generator import DEFAULT_MODEL, answer  # noqa: E402
from app.retriever import dedupe_by_article, search  # noqa: E402

PARTS = ["(todas)", "I", "I-II", "II-II", "III", "Suplemento"]

_SECAO_LABELS = {
    "objecao": "objeção",
    "resposta": "resposta",
    "sed_contra": "sed contra",
    "respondeo": "respondeo",
    "resumo": "resumo",
    "raw": "trecho",
}


def _short_secao(secao: str) -> str:
    if "_" in secao and secao not in _SECAO_LABELS:
        base, n = secao.rsplit("_", 1)
        return f"{_SECAO_LABELS.get(base, base)} {n}"
    return _SECAO_LABELS.get(secao, secao)


st.set_page_config(page_title="Suma Teológica — Leitura Devocional", layout="wide")
st.title("📜 Suma Teológica — Leitura Devocional")
st.caption("RAG sobre a Suma Teológica de Santo Tomás de Aquino. Embeddings locais (bge-m3) + Claude.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Configurações")
    model = st.selectbox(
        "Modelo Claude",
        options=["claude-sonnet-4-6", "claude-haiku-4-5"],
        index=0,
    )
    parte_filter = st.selectbox("Filtrar por Parte", PARTS, index=0)
    top_k = st.slider("Passagens recuperadas (top-k)", min_value=4, max_value=16, value=8)
    max_per_article = st.slider("Máx. chunks por artigo", min_value=1, max_value=4, value=2)
    if st.button("Nova conversa"):
        st.session_state.messages = []
        st.rerun()

main_col, sources_col = st.columns([2, 1])

with main_col:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Faça uma pergunta sobre a Suma…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Buscando passagens e meditando…"):
                where: dict | None = None
                if parte_filter != "(todas)":
                    where = {"parte": parte_filter}
                # Over-fetch then dedupe to avoid flooding one article
                raw = search(prompt, top_k=top_k * 2, where=where)
                chunks = dedupe_by_article(raw, max_per_article=max_per_article)[:top_k]

                # Compact history for context (last 4 turns, no tool/user content
                # repetition — we send only assistant-style summaries to keep tokens low).
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1][-4:]
                ]

                result = answer(prompt, chunks, model=model, history=history)
            st.markdown(result.text)
            st.session_state.messages.append({"role": "assistant", "content": result.text})
            st.session_state["last_sources"] = chunks
            st.session_state["last_usage"] = result

with sources_col:
    st.subheader("Fontes citadas")
    chunks = st.session_state.get("last_sources", [])
    if not chunks:
        st.caption("As passagens recuperadas aparecerão aqui após sua pergunta.")
    else:
        for i, c in enumerate(chunks, start=1):
            with st.expander(f"{i}. {c.citacao} — {_short_secao(c.secao)}"):
                st.markdown(f"**Questão:** {c.titulo_questao}")
                st.markdown(f"**Artigo:** {c.titulo_artigo}")
                st.markdown(f"**Distância:** {c.distance:.3f}")
                st.markdown("---")
                st.markdown(c.text)

    usage = st.session_state.get("last_usage")
    if usage:
        st.divider()
        st.caption(
            f"Modelo: {usage.model} · in={usage.input_tokens} out={usage.output_tokens}"
            + (f" · cache_read={usage.cache_read_tokens}" if usage.cache_read_tokens else "")
        )


