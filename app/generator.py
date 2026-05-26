"""LLM-side generation: prompts the configured Claude model with retrieved
context and produces a devotional/meditative answer with canonical citations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from anthropic import Anthropic

from app.retriever import RetrievedChunk

DEFAULT_MODEL = "claude-sonnet-4-6"

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
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


def format_context(chunks: Iterable[RetrievedChunk]) -> str:
    """Format retrieved chunks into a structured context block for the LLM."""
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


_client_cache: dict[str, Anthropic] = {}


def _client() -> Anthropic:
    if "default" not in _client_cache:
        _client_cache["default"] = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client_cache["default"]


def answer(
    question: str,
    chunks: list[RetrievedChunk],
    model: str = DEFAULT_MODEL,
    history: list[dict] | None = None,
    max_tokens: int = 1024,
) -> GenResult:
    """Generate a devotional answer grounded in the retrieved chunks.

    The system prompt is sent with a cache_control breakpoint so the SDK
    will cache it across turns (saving tokens in multi-turn sessions).
    """
    client = _client()
    context = format_context(chunks)

    user_content = (
        f"Pergunta: {question}\n\n"
        f"Passagens recuperadas:\n\n{context}"
    )

    messages: list[dict] = []
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=messages,
    )
    text = "".join(block.text for block in resp.content if block.type == "text")
    usage = resp.usage
    return GenResult(
        text=text,
        model=model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )
