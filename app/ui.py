"""Streamlit chat UI for the Suma Teológica RAG — SQLite-backed history."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

import sys  # noqa: E402
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import store  # noqa: E402
from app.generator import DEFAULT_MODEL, answer, backend_for  # noqa: E402
from app.retriever import (  # noqa: E402
    RetrievedChunk,
    dedupe_by_article,
    hybrid_search,
    rerank,
    search,
)

PARTS = ["(todas)", "I", "I-II", "II-II", "III", "Suplemento"]
MODELS = [
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-opus-4-7",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
]

_SECAO_LABELS = {
    "objecao": "objeção",
    "resposta": "resposta",
    "sed_contra": "sed contra",
    "respondeo": "respondeo",
    "resumo": "resumo",
    "raw": "trecho",
}

_CITATION_RE = re.compile(r"\[ST\s+([IVX\-]+),\s*q\.\s*(\d+),\s*a\.\s*(\d+)(?:,\s*([^\]]+))?\]")


def _short_secao(secao: str) -> str:
    if "_" in secao and secao not in _SECAO_LABELS:
        base, n = secao.rsplit("_", 1)
        return f"{_SECAO_LABELS.get(base, base)} {n}"
    return _SECAO_LABELS.get(secao, secao)


def _highlight_citations(text: str) -> str:
    return _CITATION_RE.sub(lambda m: f"**`{m.group(0)}`**", text)


def _missing_key_for(model: str) -> str | None:
    backend = backend_for(model)
    if backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY"
    if backend == "openai" and not os.environ.get("OPENAI_API_KEY"):
        return "OPENAI_API_KEY"
    return None


def _chunks_to_dicts(chunks: list[RetrievedChunk]) -> list[dict]:
    return [asdict(c) for c in chunks]


def _dicts_to_chunks(dicts: list[dict]) -> list[RetrievedChunk]:
    return [RetrievedChunk(**d) for d in dicts]


def _load_conversation(conv_id: int) -> None:
    """Restore a conversation from SQLite into st.session_state."""
    msgs = store.get_messages(conv_id)
    st.session_state.messages = [
        {
            "role": m.role,
            "content": m.content,
            "sources": json.loads(m.sources_json) if m.sources_json else None,
        }
        for m in msgs
    ]
    last_with_sources = next((m for m in reversed(st.session_state.messages) if m.get("sources")), None)
    if last_with_sources:
        st.session_state["last_sources"] = _dicts_to_chunks(last_with_sources["sources"])
    else:
        st.session_state.pop("last_sources", None)
    st.session_state["conversation_id"] = conv_id


def _new_conversation(model: str) -> int:
    conv_id = store.create_conversation(model=model)
    st.session_state.messages = []
    st.session_state["conversation_id"] = conv_id
    st.session_state.pop("last_sources", None)
    st.session_state.pop("last_usage", None)
    return conv_id


st.set_page_config(page_title="Suma Teológica — Leitura Devocional", layout="wide")
st.title("📜 Suma Teológica — Leitura Devocional")
st.caption("RAG sobre a Suma Teológica de Santo Tomás de Aquino. Embeddings locais (bge-m3) + Claude/OpenAI.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Configurações")
    model = st.selectbox(
        "Modelo",
        options=MODELS,
        index=MODELS.index(DEFAULT_MODEL) if DEFAULT_MODEL in MODELS else 0,
        help="Claude usa prompt caching; OpenAI usa prefix caching automático.",
    )
    parte_filter = st.selectbox("Filtrar por Parte", PARTS, index=0)
    top_k = st.slider("Passagens recuperadas (top-k)", min_value=4, max_value=16, value=8)
    max_per_article = st.slider("Máx. chunks por artigo", min_value=1, max_value=4, value=2)
    hybrid = st.toggle(
        "Busca híbrida (denso + BM25)",
        value=True,
        help="Combina embeddings com BM25 lexical via Reciprocal Rank Fusion.",
    )
    use_rerank = st.toggle(
        "Reranker (bge-reranker-v2-m3)",
        value=False,
        help="Refina top-K com cross-encoder. Carrega ~1.1 GB em VRAM. "
             "Custa ~0.5-1s por consulta na GPU; mais preciso para perguntas ambíguas.",
    )
    st.caption(f"Backend: **{backend_for(model)}**")
    missing = _missing_key_for(model)
    if missing:
        st.error(f"Defina `{missing}` no `.env` antes de gerar.")

    st.divider()
    st.subheader("Conversas")
    if st.button("➕ Nova conversa", use_container_width=True):
        _new_conversation(model)
        st.rerun()

    convs = store.list_conversations(limit=30)
    current_id = st.session_state.get("conversation_id")
    for conv in convs:
        label = conv.title or f"Conversa #{conv.id}"
        if len(label) > 40:
            label = label[:37] + "…"
        prefix = "▶ " if conv.id == current_id else "  "
        col_a, col_b = st.columns([4, 1])
        with col_a:
            if st.button(f"{prefix}{label}", key=f"conv_{conv.id}", use_container_width=True):
                _load_conversation(conv.id)
                st.rerun()
        with col_b:
            if st.button("🗑", key=f"del_{conv.id}", help="Apagar conversa"):
                store.delete_conversation(conv.id)
                if conv.id == current_id:
                    st.session_state.pop("conversation_id", None)
                    st.session_state.messages = []
                st.rerun()

def _copy_button(content: str, key: str) -> None:
    """Render a popover that exposes the raw markdown with a built-in copy button."""
    with st.popover("📋 Copiar como markdown", use_container_width=False):
        st.code(content, language="markdown")


main_col, sources_col = st.columns([2, 1])

with main_col:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(_highlight_citations(msg["content"]))
                _copy_button(msg["content"], key=f"copy_{i}")
            else:
                st.markdown(msg["content"])

    if prompt := st.chat_input("Faça uma pergunta sobre a Suma…"):
        if _missing_key_for(model):
            st.stop()
        conv_id = st.session_state.get("conversation_id")
        if conv_id is None:
            title = prompt[:80] + ("…" if len(prompt) > 80 else "")
            conv_id = store.create_conversation(model=model, title=title)
            st.session_state["conversation_id"] = conv_id
        elif not st.session_state.messages:
            title = prompt[:80] + ("…" if len(prompt) > 80 else "")
            store.update_conversation_title(conv_id, title)

        st.session_state.messages.append({"role": "user", "content": prompt, "sources": None})
        store.add_message(conv_id, "user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Buscando passagens e meditando…"):
                where: dict | None = {"parte": parte_filter} if parte_filter != "(todas)" else None
                retrieve = hybrid_search if hybrid else search
                # When reranking, fetch a wider pool (3× top_k) before rerank trims.
                fetch_k = top_k * 3 if use_rerank else top_k * 2
                raw = retrieve(prompt, top_k=fetch_k, where=where)
                if use_rerank:
                    raw = rerank(prompt, raw, top_k=top_k * 2)
                chunks = dedupe_by_article(raw, max_per_article=max_per_article)[:top_k]

                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1][-4:]
                ]

                try:
                    result = answer(prompt, chunks, model=model, history=history)
                except KeyError as e:
                    st.error(f"Variável de ambiente faltando: {e}.")
                    st.stop()

            st.markdown(_highlight_citations(result.text))
            _copy_button(result.text, key=f"copy_new_{len(st.session_state.messages)}")
            sources = _chunks_to_dicts(chunks)
            st.session_state.messages.append({"role": "assistant", "content": result.text, "sources": sources})
            store.add_message(conv_id, "assistant", result.text, sources=sources)
            st.session_state["last_sources"] = chunks
            st.session_state["last_usage"] = result

with sources_col:
    st.subheader("Fontes citadas")
    chunks = st.session_state.get("last_sources", [])
    if not chunks:
        st.caption("As passagens recuperadas aparecerão aqui após sua pergunta.")
    else:
        for i, c in enumerate(chunks, start=1):
            if c.rerank_score:
                score_label = f"rerank={c.rerank_score:+.2f}"
            elif c.rrf_score:
                score_label = f"rrf={c.rrf_score:.4f}"
            else:
                score_label = f"d={c.distance:.3f}"
            with st.expander(f"{i}. {c.citacao} — {_short_secao(c.secao)}  ·  {score_label}"):
                st.markdown(f"**Questão:** {c.titulo_questao}")
                st.markdown(f"**Artigo:** {c.titulo_artigo}")
                st.markdown("---")
                st.markdown(c.text)

    usage = st.session_state.get("last_usage")
    if usage:
        st.divider()
        cache_str = f" · cache_read={usage.cache_read_tokens}" if usage.cache_read_tokens else ""
        st.caption(
            f"Backend: {usage.backend} · Modelo: {usage.model} · "
            f"in={usage.input_tokens} out={usage.output_tokens}{cache_str}"
        )
