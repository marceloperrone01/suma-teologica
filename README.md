# Biblioteca RAG — Suma Teológica & TCC

Sistema RAG local com duas coleções pesquisáveis via interface unificada:

- **📜 Suma Teológica** — leitura devocional/meditativa da Suma Teológica de Santo Tomás de Aquino (trad. Alexandre Correia)
- **🎓 TCC** — pesquisa acadêmica em artigos científicos de psicologia clínica / CBT / ACT / ansiedade social

**Stack:**
- Embeddings locais `BAAI/bge-m3` na GPU (FP16)
- Vector store ChromaDB persistente (duas coleções: `suma_chunks`, `tcc_chunks`)
- Busca híbrida: denso + BM25 fundidos com Reciprocal Rank Fusion; reranker opcional `bge-reranker-v2-m3`
- Geração via API: Claude Sonnet/Haiku/Opus (com prompt caching) **ou** OpenAI GPT-4.1/4o
- Histórico persistente em SQLite (por coleção)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env
# edite .env: adicione ANTHROPIC_API_KEY e/ou OPENAI_API_KEY
```

## Indexação

### Suma Teológica (rodar uma vez)

```bash
# O texto extraído já existe em data/extracted.txt (gerado por pdftotext -layout)
.venv/bin/python -m app.parser       # extracted.txt → data/articles.jsonl     (~30s)
.venv/bin/python -m app.chunker      # articles.jsonl → data/chunks.jsonl
.venv/bin/python -m app.ingest       # chunks.jsonl → ChromaDB suma_chunks      (~15 min RTX 4050)
.venv/bin/python -m app.bm25_index   # chunks.jsonl → data/bm25.pkl             (~5s)
```

### TCC (rodar uma vez; repetir ao adicionar documentos)

Coloque os PDFs e/ou arquivos `.docx` na pasta `tcc/` e rode:

```bash
.venv/bin/python -m app.tcc_parser    # tcc/*.pdf + tcc/*.docx → data/tcc/tcc_articles.jsonl
.venv/bin/python -m app.tcc_chunker   # tcc_articles.jsonl → data/tcc/tcc_chunks.jsonl
.venv/bin/python -m app.tcc_ingest    # tcc_chunks.jsonl → ChromaDB tcc_chunks   (~1 min RTX 4050)
.venv/bin/python -m app.tcc_bm25      # tcc_chunks.jsonl → data/tcc/tcc_bm25.pkl (~5s)
```

> Para reindexar do zero (ex: após remover documentos): adicione `--recreate` ao `tcc_ingest`.

## Executar a UI

```bash
.venv/bin/streamlit run app/ui.py
```

Abra http://localhost:8501. O seletor de acervo **📜 Suma Teológica / 🎓 TCC** aparece no topo do sidebar.

## Estrutura

```
app/
  # ── Suma Teológica ──────────────────────────────────────────────
  parser.py         # extracted.txt → articles.jsonl (regex escolástica)
  chunker.py        # articles.jsonl → chunks.jsonl (por seção: objeção/respondeo/…)
  ingest.py         # chunks.jsonl → ChromaDB "suma_chunks" (bge-m3)
  bm25_index.py     # chunks.jsonl → data/bm25.pkl (BM25Okapi, stopwords PT)

  # ── TCC ─────────────────────────────────────────────────────────
  tcc_parser.py     # tcc/*.pdf + .docx → data/tcc/tcc_articles.jsonl
  tcc_chunker.py    # tcc_articles.jsonl → data/tcc/tcc_chunks.jsonl (por seção acadêmica)
  tcc_ingest.py     # tcc_chunks.jsonl → ChromaDB "tcc_chunks" (bge-m3)
  tcc_bm25.py       # tcc_chunks.jsonl → data/tcc/tcc_bm25.pkl (BM25, stopwords PT+EN)

  # ── Núcleo compartilhado ─────────────────────────────────────────
  retriever.py      # busca densa/BM25/híbrida (RRF) e reranker — param dataset="suma"|"tcc"
  generator.py      # Claude/OpenAI: answer() devocional, answer_tcc() acadêmico
  meditation.py     # artigo do dia ponderado por Parte + paráfrase contemplativa
  store.py          # SQLite: conversas (com campo dataset), mensagens, meditações
  ui.py             # Streamlit: chat + seletor de acervo
  pages/
    2_📖_Meditação_do_dia.py

data/
  extracted.txt          # texto bruto da Suma (pdftotext)
  articles.jsonl         # 2.686 artigos estruturados
  chunks.jsonl           # ~25k chunks com metadados escolásticos
  bm25.pkl               # índice BM25 da Suma
  chroma/                # ChromaDB (suma_chunks + tcc_chunks)
  store.sqlite           # histórico de conversas + meditações
  tcc/
    tcc_articles.jsonl   # 11 documentos extraídos
    tcc_chunks.jsonl     # 186 chunks com metadados acadêmicos
    tcc_bm25.pkl         # índice BM25 do TCC

tcc/                     # PDFs e DOCX dos artigos científicos (fonte)
```

## Roadmap

- **Fase 1 (MVP)**: extração + parser + chunking + embeddings + Chroma + busca densa + UI Claude. ✅
- **Fase 2**: BM25 híbrido (RRF), OpenAI como modelo alternativo, citações canônicas realçadas na UI. ✅
- **Fase 3**: meditação diária, histórico persistente em SQLite, reranker bge-reranker-v2-m3. ✅
- **Fase 4**: segunda coleção TCC (PDFs acadêmicos), seletor de acervo na UI. ✅
