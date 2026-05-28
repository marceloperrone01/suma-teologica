# Plano: Adicionar coleção TCC ao sistema RAG

## Contexto

O sistema RAG atual serve exclusivamente a Suma Teológica (coleção ChromaDB `suma_chunks`, BM25 em `data/bm25.pkl`, pipeline de ingestão especializado em estrutura escolástica Parte/Questão/Artigo/Seção). O usuário quer adicionar os PDFs acadêmicos do diretório `tcc/` (artigos de psicoterapia/CBT/ACT) como uma segunda coleção pesquisável, selecionável via rádio na UI sem sair do app.

---

## Abordagem

Princípio: **zero mudança no pipeline da Suma**. Todo código novo vive em arquivos novos. Apenas 4 arquivos existentes recebem cirurgia mínima.

---

## Fase 1 — Pipeline de ingestão do TCC (arquivos novos)

### `app/tcc_parser.py`
- Itera sobre todos PDFs e o `.docx` em `tcc/`
- PDFs: `pdftotext` via subprocess (binário já instalado; sem flag `-layout` para papers multi-coluna)
- `.docx`: `python-docx` (nova dependência em `requirements.txt`)
- Extrai: `source_file`, `title` (heurística: primeiro bloco antes de "Abstract"), `full_text`
- Output: `data/tcc/tcc_articles.jsonl`

### `app/tcc_chunker.py`
- Detecta seções por regex em maiúsculas: `ABSTRACT`, `INTRODUCTION`, `METHOD(S)`, `RESULT(S)`, `DISCUSSION`, `CONCLUSION`; para antes de `REFERENCES` (evita ruído de citações)
- Chunk metadata schema:
  ```json
  {
    "chunk_id": "morina-2022-cbt|introduction|0",
    "text": "...",
    "doc_id": "morina-2022-cbt",
    "source_file": "Clin Psychology...",
    "section": "introduction",
    "titulo": "The effectiveness of CBT for social anxiety",
    "citacao": "Morina et al., 2022",
    "sub_idx": 0
  }
  ```
- MAX_CHARS = 3200, OVERLAP = 320 (igual ao chunker da Suma)
- Output: `data/tcc/tcc_chunks.jsonl`

### `app/tcc_ingest.py`
- Cópia mínima de `app/ingest.py` com `CHUNKS_JSONL = data/tcc/tcc_chunks.jsonl` e `COLLECTION = "tcc_chunks"`
- Mesmo modelo BAAI/bge-m3 (já em memória)
- Metadata upsertada: `doc_id`, `source_file`, `section`, `titulo`, `citacao`

### `app/tcc_bm25.py`
- Cópia mínima de `app/bm25_index.py` com output em `data/tcc/tcc_bm25.pkl`
- Adiciona stopwords em inglês (papers são em inglês)
- Tokenizador local `tokenize_en` (não modifica `app/bm25_index.py`)

---

## Fase 2 — Retriever (modificação cirúrgica)

### `app/retriever.py` — mudanças:

**a) Novo dataclass `TccChunk`** (ao lado de `RetrievedChunk`):
```python
@dataclass
class TccChunk:
    chunk_id: str; text: str; citacao: str; section: str
    titulo: str; source_file: str; doc_id: str
    distance: float; rrf_score: float = 0.0; rerank_score: float = 0.0
```

**b) Substituir `_collection()` por `_get_collection(name: str)`** com `@lru_cache(maxsize=4)`:
- `_collection()` existente vira wrapper que chama `_get_collection("suma_chunks")` para zero quebra de retrocompatibilidade

**c) Parâmetro `dataset="suma"` nas funções `search`, `bm25_search`, `hybrid_search`**:
- `dataset="suma"` → `suma_chunks` collection + `data/bm25.pkl`
- `dataset="tcc"` → `tcc_chunks` collection + `data/tcc/tcc_bm25.pkl`
- Todos os call sites existentes continuam funcionando sem mudança (default)

**d) `_load_bm25_for(pkl_path: str)`** com `@lru_cache(maxsize=4)` — substitui a importação direta de `load_bm25()` para suportar múltiplos arquivos pkl

**e) Nova função `dedupe_by_source`** para TCC:
```python
def dedupe_by_source(chunks: list[TccChunk], max_per_doc: int = 3) -> list[TccChunk]:
    # agrupa por (doc_id, section), mantém max_per_doc por grupo
```

**f) `rerank` aceita `list[RetrievedChunk | TccChunk]`** — o tipo do campo `text` é o mesmo; sem mudança na lógica

---

## Fase 3 — Generator (adição de funções)

### `app/generator.py` — adições sem tocar em nada existente:

```python
TCC_SYSTEM_PROMPT = """Assistente de pesquisa acadêmica em psicologia clínica...
- Base-se EXCLUSIVAMENTE nas passagens fornecidas.
- Cite com [Autor et al., ano] em toda afirmação.
- Tom acadêmico, objetivo. Sem parágrafo devocional.
"""

def format_context_tcc(chunks) -> str: ...
def answer_tcc(question, chunks, model, history, max_tokens) -> GenResult: ...
```

---

## Fase 4 — Store (migração backward-compatible)

### `app/store.py` — mudanças mínimas:

**a) Adicionar `dataset TEXT NOT NULL DEFAULT 'suma'` ao `SCHEMA`** (no CREATE TABLE)

**b) Bloco de migração** no `connect()` logo após `executescript(SCHEMA)`:
```python
try:
    conn.execute("ALTER TABLE conversations ADD COLUMN dataset TEXT NOT NULL DEFAULT 'suma'")
except sqlite3.OperationalError:
    pass  # coluna já existe
```

**c) Atualizar `Conversation` dataclass**: add `dataset: str = "suma"`

**d) Atualizar `create_conversation(model, title=None, dataset="suma")`** — inclui `dataset` no INSERT

**e) Atualizar SELECTs** em `list_conversations` e `get_conversation` para incluir `dataset`

---

## Fase 5 — UI

### `app/ui.py` — mudanças estruturadas:

**a) Seletor de acervo** — primeiro widget do sidebar:
```python
dataset = st.radio("Acervo", ["📜 Suma Teológica", "🎓 TCC"], horizontal=True)
dataset_key = "suma" if "Suma" in dataset else "tcc"
```

**b) Reset de sessão ao trocar acervo** — quando `dataset_key != st.session_state.get("active_dataset")`, limpa mensagens e `conversation_id`

**c) Filtros condicionais**:
- `dataset_key == "suma"` → mostra `parte_filter` selectbox (atual)
- `dataset_key == "tcc"` → esconde o filtro de Parte (não aplicável)

**d) Título e placeholder** variam por `dataset_key`

**e) Retrieval call**: `search(...)` / `hybrid_search(...)` recebem `dataset=dataset_key`

**f) Deduplicação e geração condicionais**:
```python
if dataset_key == "suma":
    chunks = dedupe_by_article(raw, ...)[:top_k]
    result = answer(prompt, chunks, ...)
else:
    chunks = dedupe_by_source(raw, ...)[:top_k]
    result = answer_tcc(prompt, chunks, ...)
```

**g) Painel de fontes** — branch por dataset para renderizar campos corretos (`titulo_questao`/`titulo_artigo` vs `titulo`/`section`)

**h) `create_conversation(model=model, dataset=dataset_key)`**

**i) Badge no histórico de conversas**: `📜` para Suma, `🎓` para TCC

---

## Arquivos a criar (novos)

- `app/tcc_parser.py`
- `app/tcc_chunker.py`
- `app/tcc_ingest.py`
- `app/tcc_bm25.py`
- `data/tcc/` (diretório)

## Arquivos a modificar (cirurgia mínima)

- `app/retriever.py` — +`TccChunk`, `_get_collection`, `dataset=` param, `dedupe_by_source`
- `app/generator.py` — +`TCC_SYSTEM_PROMPT`, `format_context_tcc`, `answer_tcc`
- `app/store.py` — +`dataset` column, migração, `Conversation.dataset`
- `app/ui.py` — seletor de acervo, lógica condicional
- `requirements.txt` — +`python-docx>=1.1.0`

## Arquivos intencionalmente intocados

`app/parser.py`, `app/chunker.py`, `app/ingest.py`, `app/bm25_index.py`, `app/meditation.py`, `app/pages/2_📖_Meditação_do_dia.py`

---

## Verificação

```bash
# 1. Ingestão
python -m app.tcc_parser
python -m app.tcc_chunker
python -m app.tcc_ingest
python -m app.tcc_bm25

# 2. Smoke test retriever (REPL)
from app.retriever import search
assert hasattr(search("CBT social anxiety", dataset="tcc")[0], 'titulo')
assert hasattr(search("caridade", dataset="suma")[0], 'titulo_questao')

# 3. Store migration (REPL)
from app import store
cid = store.create_conversation("claude-sonnet-4-6", dataset="tcc")
assert store.get_conversation(cid).dataset == "tcc"

# 4. UI end-to-end
streamlit run app/ui.py
# - Radio aparece no topo do sidebar
# - Troca para TCC: filtro de Parte some, título muda
# - Pergunta sobre CBT → fontes TCC com [Autor, ano]
# - Volta Suma → pergunta sobre caridade → fontes ST, tom devocional
# - Meditação do dia ainda funciona
```
