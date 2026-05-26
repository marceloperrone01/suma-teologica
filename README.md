# Suma Teológica — RAG devocional

Sistema RAG sobre a Suma Teológica de Santo Tomás de Aquino (trad. Alexandre Correia) para leitura devocional/meditativa.

- **Embeddings locais**: `BAAI/bge-m3` na GPU (FP16).
- **Vector store local**: ChromaDB persistente.
- **Geração via API**: Claude Sonnet 4.6 (com prompt caching).
- **Estrutura escolástica preservada**: cada objeção, sed contra, respondeo e resposta é um chunk com metadado de citação canônica (`ST I-II, q.94, a.2`).
- **UI**: Streamlit (chat + painel lateral com fontes).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
# edite .env e adicione ANTHROPIC_API_KEY
```

## Indexação (rodar uma vez)

```bash
.venv/bin/python -m app.parser       # PDF → articles.jsonl  (~30s)
.venv/bin/python -m app.chunker      # articles → chunks.jsonl
.venv/bin/python -m app.ingest       # chunks → ChromaDB com embeddings (~15 min na RTX 4050)
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
  parser.py     # PDF → articles.jsonl (regex escolástica)
  chunker.py    # articles → chunks com metadados
  ingest.py     # embeddings bge-m3 → ChromaDB
  retriever.py  # busca densa
  generator.py  # prompt + chamada Claude
  ui.py         # Streamlit
data/
  extracted.txt   # texto bruto do PDF
  articles.jsonl  # 2686 artigos estruturados
  chunks.jsonl    # ~25k chunks com metadados
  chroma/         # índice persistente
```

## Roadmap

- **Fase 1 (MVP — atual)**: extração + parser + chunking + embeddings + Chroma + busca densa + UI Claude.
- **Fase 2**: BM25 híbrido, OpenAI como modelo alternativo, citações canônicas linkadas.
- **Fase 3**: meditação diária, histórico persistente em SQLite, reranker bge-reranker-v2-m3.
