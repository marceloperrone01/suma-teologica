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
PDF → extração → parser escolástico → chunks (com metadados) → embeddings bge-m3
                                                                       ↓
                                                                  ChromaDB (local)
                                                                       ↓
Streamlit UI ←→ retriever (híbrido BM25+denso) ←→ rerank ←→ Claude/OpenAI (gerador)
        ↑                                                              ↓
        └────────── SQLite (histórico + meditação do dia) ←────────────┘
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
├── suma-teolc3b3gica.pdf          # já existe
├── data/
│   ├── extracted.txt              # saída de pdftotext
│   ├── articles.jsonl             # 1 linha por Artigo parseado, com seções
│   ├── chunks.jsonl               # chunks com metadados
│   └── chroma/                    # índice ChromaDB persistente
├── app/
│   ├── ingest.py                  # pipeline: PDF → chunks → embeddings → Chroma
│   ├── parser.py                  # regex que detecta Parte/Questão/Artigo/seções
│   ├── chunker.py                 # chunking respeitando fronteiras de seção
│   ├── retriever.py               # busca híbrida BM25+denso, filtros por metadado
│   ├── generator.py               # roteador Claude/OpenAI com prompt template
│   ├── store.py                   # SQLite (histórico, meditação diária)
│   └── ui.py                      # Streamlit
├── .env.example                   # ANTHROPIC_API_KEY, OPENAI_API_KEY
├── requirements.txt
└── README.md
```

## Pipeline de indexação (rodar uma vez)

1. **Extrair texto**: `pdftotext -layout suma-teolc3b3gica.pdf data/extracted.txt`.
2. **Parsear estrutura** (`app/parser.py`):
   - Detectar Parte (I, I-II, II-II, III, Suplemento) — provavelmente por cabeçalho ou range de páginas; inspecionar o sumário (páginas iniciais já analisadas) para mapear.
   - `^Questão (\d+):\s*(.+)$` → abre nova Questão.
   - `^Art\.\s*(\d+)\s*[—-]\s*(.+)$` → abre novo Artigo.
   - Dentro de cada Artigo, segmentar em campos:
     - `objeções`: bloco até `Mas, em contrário` (split por `^\d+\.\s` ou `^\d+\.\s*Demais\.\s*—`).
     - `sed_contra`: do `Mas, em contrário` até `SOLUÇÃO. —`.
     - `respondeo`: do `SOLUÇÃO. —` até a primeira `RESPOSTA À PRIMEIRA.`.
     - `respostas_obj`: lista de respostas a partir de `RESPOSTA À PRIMEIRA./SEGUNDA./TERCEIRA./...`.
   - Saída: `articles.jsonl`, um objeto por Artigo:
     ```json
     {
       "parte": "I", "questao": 13, "artigo": 2,
       "titulo_questao": "Dos nomes de Deus",
       "titulo_artigo": "Se algum nome se predica de Deus substancialmente",
       "citacao": "ST I, q.13, a.2",
       "secoes": {
         "objecoes": ["...", "...", "..."],
         "sed_contra": "...",
         "respondeo": "...",
         "respostas_objecoes": ["...", "...", "..."]
       },
       "pagina_pdf": 200
     }
     ```
3. **Chunking** (`app/chunker.py`):
   - **Unidade primária = seção do Artigo** (objeção individual, sed contra, respondeo, resposta individual). Cada seção vira 1 chunk com metadado completo de citação.
   - Se uma seção exceder ~800 tokens (raro), partir em sub-chunks com overlap de ~80 tokens, preservando `citacao` + `secao` + `sub_idx`.
   - Adicionar um chunk-resumo por Artigo contendo `titulo_questao + titulo_artigo + respondeo` (para queries amplas).
4. **Embeddings + indexação**:
   - Carregar `bge-m3` em GPU (FP16); batch size 32; tempo estimado para ~30-50k chunks: 10-20 minutos na RTX 4050.
   - Persistir em ChromaDB com `collection.add(documents, embeddings, metadatas, ids)`. Metadados: `parte`, `questao`, `artigo`, `secao`, `citacao`, `pagina_pdf`.
5. **Índice BM25 paralelo**: `rank_bm25.BM25Okapi` serializado em `data/bm25.pkl` (tokenização simples lowercase + remoção de stopwords PT).

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

## Fases sugeridas de execução

- **Fase 1 (MVP)**: extração + parser + chunking + embeddings + Chroma + retrieval denso + UI Streamlit chat com Claude apenas. Sem BM25, sem rerank, sem extras.
- **Fase 2**: adicionar OpenAI + seletor, busca híbrida BM25, citações canônicas no prompt.
- **Fase 3**: meditação diária + histórico persistente + (opcional) reranker.
