"""LLM-side generation: routes between Claude (Anthropic) and OpenAI backends,
with retrieved context, devotional prompt, and canonical citations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from app.meditation import MEDITATION_SYSTEM_PROMPT, MEDITATION_USER_TEMPLATE, respondeo_or_fallback
from app.retriever import RetrievedChunk, TccChunk

DEFAULT_MODEL = "claude-sonnet-4-6"

CLAUDE_MODELS = {
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-opus-4-7",
}
OPENAI_MODELS = {
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
}

SYSTEM_PROMPT = """Você é um companheiro de leitura da Suma Teológica de Santo Tomás de Aquino, em sua tradução para o português. Seu propósito é auxiliar o usuário em leitura DEVOCIONAL e meditativa — não acadêmica fria.

Diretrizes invioláveis:
1. Baseie-se EXCLUSIVAMENTE nas passagens fornecidas como contexto. Não invente conteúdo da Suma nem invoque outras fontes.
2. Toda afirmação extraída de uma passagem deve vir com sua referência canônica entre colchetes: [ST I, q.13, a.2, resp.], [ST II-II, q.23, a.4, ad 1], etc. Use a citação exata fornecida.
3. Distinga claramente o tipo de cada passagem:
   - "objeção" = posição que Tomás vai REFUTAR. Nunca apresente como doutrina.
   - "sed contra" = razão tradicional contra as objeções.
   - "respondeo" / "Solução" = a afirmação doutrinal de Tomás.
   - "resposta" / "ad N" = a refutação tomista a cada objeção.
4. Se as passagens fornecidas não respondem à pergunta, diga isso honestamente e sugira buscas relacionadas — não fabrique.
5. Tom devocional: ao final, ofereça um breve parágrafo de aplicação contemplativa, em 2ª pessoa, que ajude o leitor a meditar sobre a verdade exposta. Mantenha sobriedade — sem floreios.

Formato da resposta:
- Comece com uma exposição clara da doutrina extraída do(s) respondeo, com citações.
- Se relevante, mencione objeções e suas refutações.
- Termine com o parágrafo devocional separado por uma linha em branco e iniciado por "**Para meditação:**".
"""


@dataclass
class GenResult:
    text: str
    model: str
    backend: str  # "anthropic" | "openai"
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


def format_context(chunks: Iterable[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        secao_label = _label_secao(c.secao)
        blocks.append(
            f"[Passagem {i}]\n"
            f"Referência: {c.citacao} ({secao_label})\n"
            f"Questão: {c.titulo_questao}\n"
            f"Artigo: {c.titulo_artigo}\n"
            f"Texto:\n{c.text.strip()}"
        )
    return "\n\n---\n\n".join(blocks)


def _label_secao(secao: str) -> str:
    base = secao.split("_")[0]
    n = secao.split("_")[1] if "_" in secao else ""
    label_map = {
        "objecao": f"objeção {n}",
        "resposta": f"resposta à objeção {n} (ad {n})",
        "sed": "sed contra",
        "respondeo": "respondeo (Solução)",
        "resumo": "resumo do artigo",
        "raw": "trecho do artigo",
    }
    return label_map.get(base, secao)


_anthropic_client = None
_openai_client = None


def _anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        _anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _anthropic_client


def _openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def backend_for(model: str) -> str:
    if model in CLAUDE_MODELS or model.startswith("claude"):
        return "anthropic"
    if model in OPENAI_MODELS or model.startswith("gpt"):
        return "openai"
    raise ValueError(f"Unknown model: {model}")


def _call(
    system_prompt: str,
    user_content: str,
    model: str,
    history: list[dict],
    max_tokens: int,
) -> GenResult:
    backend = backend_for(model)
    if backend == "anthropic":
        return _call_anthropic(system_prompt, user_content, model, history, max_tokens)
    return _call_openai(system_prompt, user_content, model, history, max_tokens)


def _call_anthropic(
    system_prompt: str,
    user_content: str,
    model: str,
    history: list[dict],
    max_tokens: int,
) -> GenResult:
    client = _anthropic()
    messages: list[dict] = list(history)
    messages.append({"role": "user", "content": user_content})
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
        ],
        messages=messages,
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    u = resp.usage
    return GenResult(
        text=text,
        model=model,
        backend="anthropic",
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_creation_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )


def _call_openai(
    system_prompt: str,
    user_content: str,
    model: str,
    history: list[dict],
    max_tokens: int,
) -> GenResult:
    client = _openai()
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
    )
    text = resp.choices[0].message.content or ""
    u = resp.usage
    cached = 0
    details = getattr(u, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return GenResult(
        text=text,
        model=model,
        backend="openai",
        input_tokens=u.prompt_tokens,
        output_tokens=u.completion_tokens,
        cache_read_tokens=cached,
        cache_creation_tokens=0,
    )


def answer(
    question: str,
    chunks: list[RetrievedChunk],
    model: str = DEFAULT_MODEL,
    history: list[dict] | None = None,
    max_tokens: int = 1024,
) -> GenResult:
    """Q&A grounded in retrieved chunks, with devotional aplicação at the end."""
    context = format_context(chunks)
    user_content = f"Pergunta: {question}\n\nPassagens recuperadas:\n\n{context}"
    return _call(SYSTEM_PROMPT, user_content, model, history or [], max_tokens)


TCC_SYSTEM_PROMPT = """Você é um assistente de pesquisa acadêmica especializado em psicologia clínica, psicoterapia e ansiedade social. Seu propósito é responder perguntas com base EXCLUSIVAMENTE nos artigos científicos fornecidos como contexto.

Diretrizes invioláveis:
1. Baseie-se EXCLUSIVAMENTE nas passagens fornecidas como contexto. Não invente dados, resultados ou afirmações.
2. Toda afirmação extraída de um artigo deve vir com sua referência entre colchetes: [Morina et al., 2022], [Krafft et al., 2020], etc. Use a citação exata fornecida nos metadados.
3. Diferencie claramente entre resultados empíricos ("o estudo encontrou que…") e afirmações teóricas dos autores.
4. Se as passagens fornecidas não respondem à pergunta, diga isso honestamente — não fabrique conteúdo.
5. Tom: acadêmico, objetivo, preciso. Sem parágrafo devocional ou de aplicação pessoal.

Formato da resposta:
- Síntese direta dos achados relevantes, com citações entre colchetes.
- Se múltiplos artigos divergem, aponte a divergência explicitamente.
- Mantenha resposta concisa e informativa.
"""


def format_context_tcc(chunks: Iterable[TccChunk]) -> str:
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[Passagem {i}]\n"
            f"Referência: {c.citacao} (seção: {c.section})\n"
            f"Título: {c.titulo}\n"
            f"Texto:\n{c.text.strip()}"
        )
    return "\n\n---\n\n".join(blocks)


def answer_tcc(
    question: str,
    chunks: list[TccChunk],
    model: str = DEFAULT_MODEL,
    history: list[dict] | None = None,
    max_tokens: int = 1024,
) -> GenResult:
    """Academic Q&A grounded in TCC research paper chunks."""
    context = format_context_tcc(chunks)
    user_content = f"Pergunta: {question}\n\nPassagens recuperadas:\n\n{context}"
    return _call(TCC_SYSTEM_PROMPT, user_content, model, history or [], max_tokens)


def paraphrase(article: dict, model: str = DEFAULT_MODEL, max_tokens: int = 700) -> GenResult:
    """Contemplative paraphrase of an article (uses respondeo when available)."""
    user_content = MEDITATION_USER_TEMPLATE.format(
        citacao=article["citacao"],
        titulo_questao=article["titulo_questao"],
        titulo_artigo=article["titulo_artigo"],
        texto=respondeo_or_fallback(article),
    )
    return _call(MEDITATION_SYSTEM_PROMPT, user_content, model, [], max_tokens)
