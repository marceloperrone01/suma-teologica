# Plano: Sistema RAG para a Suma Teológica (uso devocional)

## Contexto

O usuário tem um PDF de 4278 páginas da Suma Teológica de Santo Tomás de Aquino (em português, gerado a partir de Word 2007, com texto extraível — não escaneado) e quer um sistema RAG para **leitura devocional/meditativa**, rodando num laptop com i7-13650HX, 16 GB RAM e RTX 4050 (≈6 GB VRAM), Ubuntu 24.04.

Decisões já confirmadas com o usuário:
- **Arquitetura híbrida**: embeddings e busca rodam localmente; geração de respostas vai por API (Claude **e** OpenAI, com seletor na UI).
- **Modelo de embedding**: `BAAI/bge-m3` (multilingual forte em português, contexto 8192 tokens).
- **Interface**: Web UI local (Streamlit).
- **Indexação preserva a estrutura escolástica** (Parte → Questão → Artigo → Objeções → Sed contra → Respondeo → Respostas).
- **Extras**: meditação diária, citações canônicas (ST I, q.X, a.Y), histórico persistente.
- Inspeção de amostra do PDF confirmou extração limpa via `pdftotext -layout`. Marcadores estruturais detectáveis com regex: `Questão N:`, `Art. N —`, `SOLUÇÃO. —`, `Mas, em contrário`, `RESPOSTA À PRIMEIRA./SEGUNDA./...`, objeções numeradas `1. —`, `2. Demais. —`.

## Arquitetura

```
                 ┌─ Suma Teológica ─────────────────────────────────┐
                 │  PDF → pdftotext → parser escolástico             │
                 │  → chunks.jsonl → bge-m3 → ChromaDB "suma_chunks" │
                 │  → bm25.pkl                                       │
                 └───────────────────────────────────────────────────┘
                 ┌─ TCC ─────────────────────────────────────────────┐
                 │  tcc/*.pdf + .docx → tcc_parser → tcc_articles     │
                 │  → tcc_chunker → tcc_chunks.jsonl                  │
                 │  → bge-m3 → ChromaDB "tcc_chunks"                 │
                 │  → tcc_bm25.pkl                                    │
                 └───────────────────────────────────────────────────┘
                                     ↓ dataset="suma"|"tcc"
Streamlit UI ←→ retriever (híbrido BM25+denso) ←→ rerank ←→ Claude/OpenAI (gerador)
     ↑  [seletor acervo]                                              ↓
     └────────── SQLite (conversas.dataset, mensagens, meditações) ←──┘
```

### Stack

| Camada | Escolha | Motivo |
|---|---|---|
| Extração | `pdftotext -layout` (poppler) | Texto já limpo; rápido (~2 min para 4278 pgs); sem dependência ML |
| Parser estrutural | Python + regex | Marcadores são consistentes e detectáveis |
| Embeddings | `BAAI/bge-m3` via `sentence-transformers` ou `FlagEmbedding` | Português forte, contexto 8192, ~2.3 GB, roda na RTX 4050 com FP16 |
| Vector store | **ChromaDB** (modo persistente local) | Simples de operar, suporta filtros por metadado, sem servidor |
| Busca lexical | `rank-bm25` em paralelo aos embeddings | RAG híbrido (denso + esparso) eleva precisão em termos teológicos raros |
| Rerank (opcional, fase 2) | `BAAI/bge-reranker-v2-m3` | Refina top-K antes de mandar ao LLM |
| LLM (geração) | `anthropic` SDK (Claude Sonnet 4.6 default, Haiku 4.5 como rápido) + `openai` SDK (GPT-4.1/4o) com seletor | Qualidade em português + prompt caching no Claude |
| UI | Streamlit | Curva curta, componentes prontos para chat, fontes citadas e seletor de modelo |
| Persistência | SQLite (`sqlite3` da stdlib) | Histórico, favoritos, meditação do dia |

### Estrutura do projeto

```
suma-teologica/
├── suma-teolc3b3gica.pdf          # PDF original
├── tcc/                           # PDFs e DOCX dos artigos científicos (TCC)
├── data/
│   ├── extracted.txt              # texto bruto da Suma (pdftotext)
│   ├── articles.jsonl             # 2.686 artigos estruturados
│   ├── chunks.jsonl               # ~25k chunks escolásticos
│   ├── bm25.pkl                   # índice BM25 da Suma
│   ├── chroma/                    # ChromaDB (suma_chunks + tcc_chunks)
│   ├── store.sqlite               # conversas, mensagens, meditações
│   └── tcc/
│       ├── tcc_articles.jsonl     # 11 documentos extraídos
│       ├── tcc_chunks.jsonl       # 186 chunks acadêmicos
│       └── tcc_bm25.pkl           # índice BM25 do TCC
├── app/
│   ├── parser.py                  # extracted.txt → articles.jsonl
│   ├── chunker.py                 # articles.jsonl → chunks.jsonl (estrutura escolástica)
│   ├── ingest.py                  # chunks.jsonl → ChromaDB "suma_chunks"
│   ├── bm25_index.py              # chunks.jsonl → bm25.pkl
│   ├── tcc_parser.py              # tcc/*.pdf + .docx → tcc_articles.jsonl
│   ├── tcc_chunker.py             # tcc_articles.jsonl → tcc_chunks.jsonl
│   ├── tcc_ingest.py              # tcc_chunks.jsonl → ChromaDB "tcc_chunks"
│   ├── tcc_bm25.py                # tcc_chunks.jsonl → tcc_bm25.pkl
│   ├── retriever.py               # busca densa/BM25/híbrida, param dataset=
│   ├── generator.py               # answer() devocional + answer_tcc() acadêmico
│   ├── meditation.py              # artigo do dia + paráfrase contemplativa
│   ├── store.py                   # SQLite (conversas.dataset, mensagens, meditações)
│   ├── ui.py                      # Streamlit: seletor acervo + chat
│   └── pages/
│       └── 2_📖_Meditação_do_dia.py
├── .env.example
├── requirements.txt
└── README.md
```

## Pipeline de indexação

### Suma Teológica (rodar uma vez)

```bash
pdftotext -layout suma-teolc3b3gica.pdf data/extracted.txt   # ~2 min
python -m app.parser       # extracted.txt → data/articles.jsonl       (~30s)
python -m app.chunker      # articles.jsonl → data/chunks.jsonl
python -m app.ingest       # chunks.jsonl → ChromaDB "suma_chunks"      (~15 min RTX 4050)
python -m app.bm25_index   # chunks.jsonl → data/bm25.pkl               (~5s)
```

**Detalhes da Suma:**

1. **`app/parser.py`** — detecta estrutura escolástica via regex:
   - `^Questão (\d+):\s*(.+)$` → abre nova Questão
   - `^Art\.\s*(\d+)\s*[—-]\s*(.+)$` → abre novo Artigo
   - Segmentos: objeções (`^\d+\.\s`), sed contra (`Mas, em contrário`), respondeo (`SOLUÇÃO. —`), respostas (`RESPOSTA À PRIMEIRA./…`)
   - Saída: 2.686 artigos em `articles.jsonl` com campos `parte`, `questao`, `artigo`, `citacao`, `objecoes`, `sed_contra`, `respondeo`, `respostas_objecoes`

2. **`app/chunker.py`** — unidade primária = seção do Artigo:
   - 1 chunk por objeção, sed contra, respondeo, resposta + 1 chunk-resumo por Artigo
   - Seções > 12.800 chars partidas com overlap de 320 chars
   - Chunk ID: `{parte}|q{questao}|a{artigo}|{secao}|{sub_idx}`

3. **`app/ingest.py`** — embeddings + ChromaDB:
   - Modelo: `BAAI/bge-m3` (CUDA FP16, ~2.3 GB VRAM, batch 32)
   - Coleção: `suma_chunks`, métrica cosine
   - Metadata indexada: `parte`, `questao`, `artigo`, `secao`, `citacao`, `titulo_questao`, `titulo_artigo`
   - Suporte a `--recreate` e retomada incremental

4. **`app/bm25_index.py`** — índice lexical:
   - `BM25Okapi` sobre todos os chunks, tokenizador PT (strip accents + stopwords)
   - Serializado em `data/bm25.pkl`

---

### TCC (rodar uma vez; repetir ao adicionar documentos)

Coloque os PDFs e/ou arquivos `.docx` na pasta `tcc/` e rode:

```bash
python -m app.tcc_parser    # tcc/*.pdf + *.docx → data/tcc/tcc_articles.jsonl
python -m app.tcc_chunker   # tcc_articles.jsonl → data/tcc/tcc_chunks.jsonl
python -m app.tcc_ingest    # tcc_chunks.jsonl → ChromaDB "tcc_chunks"    (~1 min RTX 4050)
python -m app.tcc_bm25      # tcc_chunks.jsonl → data/tcc/tcc_bm25.pkl    (~5s)
```

> Para reindexar do zero: `python -m app.tcc_ingest --recreate`

**Detalhes do TCC:**

1. **`app/tcc_parser.py`** — extração genérica de documentos acadêmicos:
   - PDFs: `pdftotext` via subprocess (sem `-layout`, melhor para papers multi-coluna)
   - DOCX: `python-docx`
   - Heurística de título: primeiro bloco de texto antes de "Abstract"
   - Heurística de citação: regex de autor+ano no início do documento → `"Morina et al., 2022"`
   - Saída: `tcc_articles.jsonl` com campos `source_file`, `title`, `citacao`, `full_text`

2. **`app/tcc_chunker.py`** — chunking por seção acadêmica:
   - Detecta headings via regex: `ABSTRACT`, `INTRODUCTION`, `METHODS`, `RESULTS`, `DISCUSSION`, `CONCLUSION`, etc.
   - Para antes de `REFERENCES` (evita ruído de citações bibliográficas)
   - Mesmos parâmetros da Suma: MAX_CHARS = 3.200, OVERLAP = 320
   - Sliding window garante avanço mínimo de `MAX_CHARS - OVERLAP` por passo (evita loop)
   - Chunk ID: `{doc_id}|{section}|{sub_idx}`
   - Metadata: `doc_id`, `source_file`, `section`, `titulo`, `citacao`

3. **`app/tcc_ingest.py`** — mesmo pipeline de embeddings da Suma:
   - Modelo compartilhado `BAAI/bge-m3`
   - Coleção: `tcc_chunks` (mesmo ChromaDB, coleção separada)

4. **`app/tcc_bm25.py`** — índice BM25 com stopwords PT + EN (papers em inglês):
   - Serializado em `data/tcc/tcc_bm25.pkl`

**Estatísticas atuais do TCC:** 11 documentos (10 PDFs + 1 DOCX), 186 chunks únicos.

## Pipeline de consulta

1. Usuário envia pergunta na UI.
2. **Retrieval híbrido** (`app/retriever.py`):
   - top-30 do ChromaDB (denso) + top-30 do BM25 (esparso).
   - Fusão por Reciprocal Rank Fusion (RRF, k=60), pegar top-12.
   - (Opcional fase 2) reranquear com `bge-reranker-v2-m3` para top-6.
3. **Geração** (`app/generator.py`):
   - Template de prompt em português, instruindo o LLM a:
     - Citar SEMPRE a localização canônica (ST I, q.X, a.Y) ao usar uma passagem.
     - Distinguir tom: respondeo (afirmação tomista), objeção (posição refutada), resposta (refutação).
     - Tom devocional/meditativo na conclusão, sem inventar passagens.
   - Roteador escolhe `claude-sonnet-4-6` ou `gpt-4.1` conforme seleção da UI.
   - Para Claude: usar **prompt caching** no system prompt + nas instruções fixas (economiza ~80% de tokens em sessões longas — ver `claude-api` skill).
4. UI exibe resposta + painel lateral com as fontes citadas (texto integral do chunk + link "ver artigo completo").

## Recursos extras

### Meditação diária (`app/store.py` + `ui.py`)
- Função `meditacao_do_dia()`: usa `hash(date.today())` como seed; sorteia um Artigo de `articles.jsonl` (peso maior em II-II e III, mais devocionais).
- Página "Meditação de hoje" mostra: título da Questão/Artigo, `respondeo` completo, e um botão "gerar paráfrase contemplativa" que chama o LLM com prompt específico (tom orante, em 2ª pessoa, sem alterar a doutrina).
- Salva qual artigo já foi sorteado em `meditations` (tabela SQLite) para evitar repetição em janela móvel de N dias.

### Citações canônicas
- Já embutidas no metadado de cada chunk; o prompt do gerador exige `[ST I, q.13, a.2, resp.]` ou similar no formato; UI faz parse e linka para o artigo completo.

### Histórico persistente
- Tabela `conversations` (id, started_at, model, title) e `messages` (conversation_id, role, content, sources_json, created_at).
- Sidebar lista conversas anteriores; clicar restaura o contexto.

## Arquivos críticos (a criar)

- `app/parser.py` — regex de detecção estrutural; é o coração da indexação. Validar contra 5-10 artigos manualmente antes de processar tudo.
- `app/chunker.py` — produz `chunks.jsonl` com metadados completos.
- `app/ingest.py` — orquestra extração → parse → chunk → embed → persist.
- `app/retriever.py` — busca híbrida + filtros (ex: `where={"parte": "II-II"}`).
- `app/generator.py` — prompt template + roteador Claude/OpenAI com prompt caching.
- `app/ui.py` — Streamlit (chat, seletor de modelo, painel de fontes, página de meditação, histórico).
- `app/store.py` — SQLite.
- `requirements.txt` — `chromadb`, `sentence-transformers` (ou `FlagEmbedding`), `rank-bm25`, `anthropic`, `openai`, `streamlit`, `pypdf` (fallback), `python-dotenv`.
- `.env.example` — chaves de API.

## Considerações de hardware (i7-13650HX + RTX 4050 6GB + 16GB RAM)

- **bge-m3** em FP16 ocupa ~2.3 GB VRAM — sobra folga na RTX 4050.
- **Tempo de indexação**: pdftotext ~2 min, parse ~30s, embeddings ~10-20 min, total ~25 min (uma vez).
- **Latência de consulta**: retrieval local <500ms; latência total dominada pela API do LLM (~2-5s).
- **Disco**: ChromaDB para ~40k chunks com bge-m3 (1024 dim, float32) ≈ 160-200 MB; tranquilo nos 512 GB.
- **RAM**: Streamlit + Chroma + bge-m3 carregado em VRAM ≈ 4-5 GB de RAM usada. Folga nos 16 GB.

## Verificação (end-to-end)

1. **Parser**: rodar `python -m app.parser --sample 10` e inspecionar manualmente 10 artigos aleatórios em `articles.jsonl`. Confirmar que objeções, sed contra, respondeo e respostas foram separados corretamente.
2. **Indexação**: rodar `python -m app.ingest` e confirmar que `data/chroma/` foi criado e `collection.count()` ≈ número esperado de chunks.
3. **Retrieval isolado**: consultas-teste:
   - "A caridade é uma virtude?" → deve retornar chunks de ST II-II, q.23.
   - "Pode-se provar a existência de Deus?" → ST I, q.2 (especialmente a.3, as cinco vias).
   - "O que é a acédia?" → ST II-II, q.35.
4. **Geração**: rodar UI com `streamlit run app/ui.py`; testar as 3 queries acima com Claude e com OpenAI; verificar se as citações ST aparecem e batem com as fontes do painel lateral.
5. **Meditação diária**: abrir a página, conferir que o artigo do dia muda corretamente entre dias diferentes (forçar via parâmetro `?date=`).
6. **Histórico**: enviar 3 mensagens, fechar o app, reabrir, confirmar que a conversa anterior aparece na sidebar e pode ser retomada.

## Fases de execução

- **Fase 1 (MVP)**: extração + parser + chunking + embeddings + Chroma + retrieval denso + UI Streamlit chat com Claude apenas. ✅
- **Fase 2**: OpenAI + seletor, busca híbrida BM25 (RRF), citações canônicas realçadas na UI. ✅
- **Fase 3**: meditação diária, histórico persistente em SQLite, reranker bge-reranker-v2-m3. ✅
- **Fase 4**: segunda coleção TCC (PDFs acadêmicos), pipeline de ingestão genérico, seletor de acervo na UI, system prompt acadêmico separado. ✅
