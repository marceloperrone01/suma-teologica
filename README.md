# Suma Teológica — RAG devocional

Sistema RAG sobre a Suma Teológica de Santo Tomás de Aquino (trad. Alexandre Correia) para leitura devocional/meditativa.

- **Embeddings locais**: `BAAI/bge-m3` na GPU (FP16).
- **Vector store local**: ChromaDB persistente.
- **Busca híbrida**: denso (bge-m3) + BM25 fundidos com Reciprocal Rank Fusion; reranker opcional `bge-reranker-v2-m3`.
- **Geração via API**: Claude Sonnet/Haiku/Opus (com prompt caching) **ou** OpenAI GPT-4.1/4o, com seletor na UI.
- **Estrutura escolástica preservada**: cada objeção, sed contra, respondeo e resposta é um chunk com metadado de citação canônica (`ST I-II, q.94, a.2`).
- **Meditação diária**: artigo do dia ponderado por Parte, com paráfrase contemplativa LLM em 2ª pessoa.
- **Histórico persistente**: conversas e meditações armazenadas em SQLite local.
- **UI**: Streamlit multi-página (chat + meditação) com painel lateral de fontes e realce de citações.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
# edite .env e adicione ANTHROPIC_API_KEY e/ou OPENAI_API_KEY
```

## Indexação (rodar uma vez)

```bash
.venv/bin/python -m app.parser       # PDF → articles.jsonl  (~30s)
.venv/bin/python -m app.chunker      # articles → chunks.jsonl
.venv/bin/python -m app.ingest       # chunks → ChromaDB com embeddings (~15 min na RTX 4050)
.venv/bin/python -m app.bm25_index   # chunks → data/bm25.pkl (~5s)
```

O texto extraído de `suma-teolc3b3gica.pdf` já vem em `data/extracted.txt` (gerado por `pdftotext -layout`).

## Executar a UI

```bash
.venv/bin/streamlit run app/ui.py
```

Abra http://localhost:8501.

## Estrutura

```
app/
  parser.py       # PDF → articles.jsonl (regex escolástica)
  chunker.py      # articles → chunks com metadados
  ingest.py       # embeddings bge-m3 → ChromaDB
  bm25_index.py   # chunks → data/bm25.pkl (BM25Okapi + tokenizador PT)
  retriever.py    # busca densa, BM25, híbrida (RRF) e reranker
  generator.py    # roteador Claude/OpenAI + prompts (Q&A e meditação)
  meditation.py   # picker do artigo do dia + prompt contemplativo
  store.py        # SQLite (conversas, mensagens, meditações)
  ui.py           # Streamlit (chat)
  pages/
    2_📖_Meditação_do_dia.py
data/
  extracted.txt   # texto bruto do PDF
  articles.jsonl  # 2686 artigos estruturados
  chunks.jsonl    # ~25k chunks com metadados
  chroma/         # índice denso persistente
  bm25.pkl        # índice lexical BM25
  store.sqlite    # histórico de conversas + meditações
```

## Roadmap

- **Fase 1 (MVP)**: extração + parser + chunking + embeddings + Chroma + busca densa + UI Claude. ✅
- **Fase 2**: BM25 híbrido (RRF), OpenAI como modelo alternativo, citações canônicas realçadas na UI. ✅
- **Fase 3 (atual)**: meditação diária, histórico persistente em SQLite, reranker bge-reranker-v2-m3. ✅
