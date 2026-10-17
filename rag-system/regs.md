# Design: regulation index (RAG over NBRK acts) — MVP

Status: built and measured locally on 2026-10-17 (M1, 8 GB). Companion to `design.md`; the research behind it is in `regulatory_sources.md`. Code: `svc/regs/`.

## 0. What it is for, and the conflict with design.md

`design.md` §4 says *"No search over legal texts. The judge sees only the quotes in the catalogue, and no model writes a citation."* This index keeps the second sentence and changes the first. It is used in two places:

1. **Now (MVP): authoring and checking `rules.yaml`.**
   - The rules workstream searches the acts in the web page.
   - The page copies the verbatim clause, its label and the anchored adilet URL into a ready `citations` block.
   - `python -m regs verify rules.yaml` fails the deploy if any quote in the catalogue does not appear word for word in the acts. This targets criterion 4 (no invented references).
2. **Later (optional): retrieval for the judge (design.md step 6).**
   - The observation is the query, and the top clauses go to the judge next to the candidate rules.
   - The judge may pick a returned **chunk id**, never write text. The citation is then looked up from that chunk, so "no model writes a citation" still holds.
   - This helps the `other` rule most, since it now cites only a whole document.
   - Not wired in: `pipeline.py` belongs to the user.

**Proposed edit to design.md §4.** Replace that bullet with: *"No model writes a citation. Citations come from `rules.yaml`, or from a regulation-index chunk the judge selects by id; `regs verify` checks every catalogue quote against the indexed acts."*

## 1. Pipeline

```
sources.yaml ──fetch──▶ raw/<code>.html ──parse──▶ parsed/<code>.json ──chunk──▶ chunks.jsonl
                (online, once)            clauses + structure + status        513 chunks
                                                                                   │ embed (bge-m3, local)
                                                                                   ▼
   web page /regs ◀── FastAPI /api/regs/* ◀── search.py ◀── Milvus (Lite server process, :19530)
   CLI search/eval/verify ◀────────────────────┘    │      dense COSINE + BM25 sparse
                                                    └── ltr.py: LambdaMART over cosine + BM25 features
```

Everything after `fetch` runs offline. With `HF_HUB_OFFLINE=1` the model loads from `HF_HOME` (section 7 of the brief). To move into a network with no internet, copy `data/regs/` and the model weights.

## 2. Corpus and preprocessing (`fetch.py`, `parse.py`)

- **Sources.** Four NBRK acts, listed with their reasons in `sources.yaml` and `regulatory_sources.md`.
  - `canonical_url` is adilet, and every citation points there.
  - `fetch_url` is the zakon.uchet.kz mirror. adilet.zan.kz is now a JS app whose document API needs a key; the mirror serves adilet's own HTML with the same `#zNN` clause anchors.
- **Structure-aware parsing** of adilet markup inside `<article>`. Nothing is guessed from plain text.
  - `<h3>` is the title of the Rules, a "Глава" (chapter) or a "Параграф" (paragraph).
  - `<p id="zNN">` is one paragraph. It is a new clause when it starts with `N.` or `N-M.` and the number is higher than the last one, so lists inside a clause are not mistaken for clauses. Otherwise it is a subclause `N)` or a continuation paragraph of the current clause.
  - `<span class="note">` and `<p class="note">` are footnotes ("Сноска…"). They are stored on the clause and parsed into `amended_by` (e.g. "№ 71 от 10.11.2025"). "N. Исключен…" becomes a clause with status `repealed`.
  - `<font color=red>` is an editor's note. When it says "Действие пункта N приостановлено … действует в следующей редакции", clause N gets status `suspended` and the interim wording, which is the text in force.
  - The resolution text before the Rules and the annexes (forms) after them are skipped. An optional `include_chapters` limits a long act to its relevant chapters (No. 49: chapters 1, 5, 6).
- **Per-act edition.** The newest amending act found in the footnotes ("№ 78 от 30.06.2026") is shown in the UI. It is written into each copied citation as a reminder to check the clause on adilet.
- **Result.** 253 clauses with no gaps in numbering. All but one clause have their adilet anchor; the exception is No. 231 p. 53, which has no anchor in the source.

| Act | Clauses | Repealed | Suspended | Text includes amendments up to |
|---|---|---|---|---|
| No. 231 (V1900019680) | 154 | 2 | 0 | No. 78 of 30.06.2026 |
| No. 14 (V2000020075) | 24 | 1 | 3 | No. 88 of 24.12.2024 |
| No. 230 (V1700016120) | 39 | 0 | 0 | No. 78 of 30.06.2026 |
| No. 49 (V1900018545), ch. 1, 5, 6 | 36 | 0 | 0 | No. 78 of 30.06.2026 |

## 3. Chunking (`chunk.py`)

- **One clause (пункт) is one chunk.** A clause is the unit a citation names, and clauses are never merged, so every hit maps to exactly one "п. N". Median 75 words; 90 % are under 160.
- **Long clauses** (over 220 words, about 400 bge-m3 tokens) split at subclause boundaries. Each part repeats the opening line ("2. В Правилах используются следующие понятия:"), so a part still says what it is about. Labels look like "п. 2, пп. 10)–17)". This affects 17 chunks.
- **Subclause chunks (small-to-big).** A clause that is a list of duties also gets one chunk per subclause: opening line plus that subclause, labelled "п. 31, пп. 4)". The full clause stays indexed. Search folds the clause and subclause hits back into one result per clause, with the best-matching label. Without this, the one relevant duty in "31. Кассовый работник … 1)…6)" was diluted, and the clause ranked 6th–11th for "выдает деньги без пересчета в присутствии клиента". Result: 265 clause-level and 248 subclause-level chunks.
- **Three texts per chunk:**
  - `text` is the verbatim wording. This is what gets quoted, and what `verify` matches.
  - `embed_text` is `"<act>. <chapter>. <paragraph>. п. N.\n" + text`. The context header lets the vector know, for example, that a clause is about *accepting* cash.
  - `lex` is `embed_text` lowercased, with ё→е, stop words removed and Snowball-stemmed, for BM25.

## 4. Index (`store.py`): Milvus

**Collection `regs_chunks`, one row per chunk.**

| Field | Type | Role |
|---|---|---|
| `id` | VARCHAR PK | `V1900019680:31:pp4`; deterministic, so rebuilds are stable |
| `dense` | FLOAT_VECTOR(1024) | bge-m3 embedding of `embed_text`; COSINE; AUTOINDEX (flat on Lite, HNSW on Standalone) |
| `lex` | VARCHAR, analyzer on | input to the BM25 function |
| `sparse` | SPARSE_FLOAT_VECTOR | **filled inside Milvus by a `BM25` Function from `lex`**; SPARSE_INVERTED_INDEX, metric BM25 (k1 1.2, b 0.75) |
| `doc_code`, `clause`, `clause_label`, `granularity`, `status`, `seq` | scalar | filters (`doc_code in [...]`, `status != "repealed"`), folding, reading neighbouring clauses |
| `text`, `chapter`, `section`, `url`, `anchor`, `notes`, `amended_by`, `doc_short` | VARCHAR | returned with each hit: everything a citation needs |

**Decisions, and why.**
- **Milvus as its own process.** `python -m regs milvus` runs Milvus Lite's gRPC server (`milvus-lite server`) on `127.0.0.1:19530`. It needs no Docker and no database service, in the same spirit as the separate SGLang processes in design.md.
  - Clients connect over gRPC, so the web app, the CLI build and later the pipeline can all use the index at once.
  - The code does not change for Milvus Standalone or Zilliz: set `REGS_MILVUS_URI`.
  - *Why not embedded:* milvus-lite 3.x runs its engine (faiss) inside the calling process. faiss ships its own OpenMP runtime, and torch ships another, so two runtimes end up in one process. On macOS that crashed 6 of 6 runs when the model loaded before Milvus (exit 139), and 1 of 3 runs when Milvus opened first. With Milvus in a separate process, every run so far has been clean: the separate-process tests, two builds, three eval runs and the web app.
  - Embedded mode still works (`REGS_MILVUS_URI=<path>`). It then sets `OMP_NUM_THREADS=1` automatically, which made it stable (3/3).
- **BM25 inside Milvus, stemming in our code.** Milvus Lite accepts a `stemmer` analyzer filter but does not apply it: "видеонаблюдение" did not match "видеонаблюдения". Its only tokenizers are `standard` and `jieba`. So `lexical.py` stems Russian (and English) with Snowball before the text reaches Milvus, and Milvus only splits on whitespace and computes BM25. This behaves the same on Lite and on a Milvus server.
- **Full rebuild instead of upserts.** The corpus is 513 rows. Drop, create and insert takes about 1 minute, nearly all of it embedding on MPS; on the server GPU it takes seconds. A rebuild never leaves stale chunks behind.
- **Embedding model `BAAI/bge-m3`** (MIT, 568M parameters, 1024 dimensions, 8k context). It is multilingual, so English visual criteria (`rules.yaml`) and Russian observations (InternVideo3) both reach Russian clauses. It needs no query prefix. To swap it, set `REGS_EMBED_MODEL` and, for e5-style models, `REGS_QUERY_PREFIX`/`REGS_PASSAGE_PREFIX`.

## 5. Retrieval (`search.py`)

- **Modes:**
  - `hybrid` (default) runs Milvus `hybrid_search` with a dense request and a BM25 request. Each takes the top 40 candidates, then they are fused with `RRFRanker(60)`.
  - `dense` and `bm25` are available for comparison.
  - Filters (`doc_code in […]`, `status != "repealed"`) go into each request.
- **The pool size is fixed** (`max(4k, 40)`), so the top results do not depend on how many results were asked for.
- **Folding.** One result per clause. The best chunk wins, and the other matched labels are listed in `also_matched`.
- **Glossary** (`glossary.yaml`, 10 entries). Observations say "кассир", "смартфон" and "посторонний"; the acts say "кассовый работник", "средства связи … фото и видео фиксации" and "третьи лица". A query word that starts with a `when` prefix adds the legal wording **to the BM25 query only**. The entries describe what words mean; none refers to a specific video.
- **Explain.** Every hit carries its rank and score in the dense list and in the BM25 list on their own. The UI shows them, so you can see why a clause came up.

### 5.1 Ranking with LambdaMART (default, `ltr.py`)

The default mode (`auto`) replaces the fixed RRF fusion with a learned one.

- **Candidates.** The union of the dense top-60 and the BM25 top-60 chunks, about 97 per query. Every candidate gets both scores exactly:
  - a missing cosine comes from a dense search filtered to those ids;
  - BM25 comes from one search that returns every matching chunk, and a chunk outside it scores 0.
- **Features, all derived from cosine and BM25:** `cos`, `bm25`, `cos_rank`, `bm25_rank`, `cos_gap` (minus the query's best cosine) and `bm25_rel` (divided by the query's best BM25). Raw BM25 cannot be compared across queries; the relative and rank versions can.
- **Model.** LambdaMART: 50 scikit-learn regression trees (8 leaves each) fitted to LambdaRank gradients weighted by the |ΔNDCG@10| of each pair swap, with Newton leaf values. **Monotone constraints** mean a higher cosine or BM25 can never lower the score. With 26 training queries this is the main guard against overfitting.
- **Training** (`python -m regs train-ltr`):
  - Dev queries only, labelled at chunk level (1 = a chunk of an expected clause).
  - Hyperparameters chosen by 5-fold cross-validation grouped by query, ranked by out-of-fold top-1, then MRR.
  - Then trained on all dev queries and saved to `DATA_DIR/ltr_lambdamart.pkl`, with a JSON sidecar holding the parameters and CV scores.
  - Retrain after every `build`, since features depend on the embeddings.
- **Why not LightGBM.** Its macOS wheel needs Homebrew's `libomp`. When it borrowed torch's copy instead, the process deadlocked. The scikit-learn implementation is the same algorithm (Burges 2010) with no extra native runtime, and for 50 small trees it predicts in microseconds.

### 5.2 Optional neural reranker (`rerank.py`, off by default)

`REGS_RERANK_MODEL` turns on a second stage after either ranker. It supports cross-encoders (`BAAI/bge-reranker-v2-m3`) and Qwen3-Reranker (an LLM asked yes/no under a task instruction; score = P(yes)). `REGS_RERANK_WEIGHT` blends its score with the first-stage score, and `python -m regs sweep` picks the weight on dev.

It is off because it cost about 4 s per query on the M1 and lowered dev top-1 (0.69 → 0.54) while raising hold-out top-1 (0.33 → 0.44). The Qwen3 run was stopped before it finished.

### 5.3 Milvus Lite quirks found on the way

- **BM25 comes back as a negative distance**: the best hit is around −26 and weak hits are near 0. `store.search_bm25` returns `abs()`, which also fits Milvus Standalone's positive scores.
- **BM25 statistics are computed over the filtered rows.** An `id in [...]` filter changes a chunk's score (п. 22 scored −26.6 with no id filter and −6.2 with one). So BM25 from searches with different filters cannot be compared.

## 6. Measured retrieval quality

`eval_queries.yaml` holds two sets of queries that describe visible events, each labelled with the clause(s) a reviewer would cite. A hit counts if the act and clause number match.
- **dev:** 26 queries, 13 Russian observations and 13 English visual criteria. The subclause chunks, the glossary and LambdaMART were all shaped on these.
- **holdout:** 18 new phrasings, written before the rankers were tried and never used to choose anything.

**Holdout is the number to trust.** Both sets were written by the people who built the index and are small: one query is 5.6 points of holdout top-1.

**Holdout (18 queries)**

| Ranking | top-1 | top-3 | top-5 | MRR |
|---|---|---|---|---|
| dense (cosine) | 0.33 | 0.56 | 0.61 | 0.45 |
| BM25 + glossary | 0.39 | 0.56 | 0.61 | 0.47 |
| hybrid RRF + glossary | 0.33 | 0.56 | 0.67 | 0.44 |
| hybrid + bge-reranker-v2-m3 (optional) | 0.44 | 0.78 | 0.78 | 0.58 |
| **LambdaMART on cosine + BM25 (default)** | **0.39** | **0.56** | **0.78** | **0.52** |

**Dev (26 queries)**

| Ranking | top-1 | top-3 | top-5 | MRR |
|---|---|---|---|---|
| clause chunks only, hybrid (first build) | 0.50 | 0.65 | 0.81 | 0.61 |
| dense (cosine) | 0.46 | 0.69 | 0.77 | 0.58 |
| BM25 + glossary | 0.46 | 0.65 | 0.77 | 0.58 |
| hybrid RRF + glossary | 0.69 | 0.89 | 0.92 | 0.77 |
| hybrid + bge-reranker-v2-m3 | 0.54 | 0.81 | 0.92 | 0.70 |
| LambdaMART, out-of-fold (5-fold CV) | 0.85 | 0.92 | — | 0.89 |

**Reading this.**
- LambdaMART is the best of the rankers that use only cosine and BM25 on holdout: top-5 0.67 → 0.78 and MRR 0.44 → 0.52.
- Its top-1 gain on holdout is small: one more query of 18.
- The gap between 0.85 out-of-fold on dev and 0.39 on holdout means the dev phrasings are easier, or closer to each other, than the holdout ones. Use the holdout figure as the expectation.

**What limits top-1: the signals, not the candidate pool.**
- The right clause is among the ~97 candidates for every holdout query (pool recall 100%).
- For 4 queries, both cosine and BM25 score it low, so LambdaMART ranks it 6th, 33rd, 44th and 55th:
  - "банкноты клиента пересчитываются уже после его ухода" (6th)
  - "один сотрудник упаковывает и пломбирует пачки…" (33rd)
  - "клиент проходит за стойку кассира" (44th)
  - "сумка с деньгами передается инкассаторам в зале…" (55th)
- The rest are near-misses between related clauses: p. 31 against p. 35 (who counts the cash), p. 24 against p. 108 (which side checks the bag).
- Reweighting two signals cannot reach these. They need a signal that reads the query and the clause together (section 9).

**Latency** on the M1 with Milvus Lite: about 0.15–0.45 s per LambdaMART search. That is the query embedding plus three Milvus searches; the trees cost almost nothing. With the cross-encoder on it is about 4 s.

## 7. Web interface and API (`app.py`, `static/index.html`)

**The page** (`/regs`) is one HTML file with plain JavaScript and no build step, like the main UI in design.md. It has:
- a query box with example chips;
- a choice of ranking (LambdaMART, hybrid RRF, dense, BM25), the cross-encoder on or off, the number of results, filters by act, glossary on or off, and repealed clauses shown or hidden;
- for each hit:
  - the clause label;
  - a status badge: in force, interim wording, or repealed;
  - the chapter and paragraph;
  - the LambdaMART score with the hit's cosine and BM25 values, plus its separate dense and BM25 ranks;
  - the verbatim text with matching stems highlighted;
  - the amending acts and footnotes;
  - buttons: **Open on adilet** (anchored to the clause), **Copy quote**, **For rules.yaml** (a `citations` block whose comment shows the indexed edition, with `edition_checked` left for a person to fill in), and **Neighbouring clauses** (the previous and next clause).

**The API.** All endpoints are GET. `X-API-Key` is required only when `REGS_API_KEY` is set.
- `/api/regs/stats`: the acts, their editions, the model and the row counts.
- `/api/regs/search?q=&k=&mode=auto|ltr|hybrid|dense|bm25&doc=&include_repealed=&glossary=&rerank=`
- `/api/regs/chunk/{id}?before=&after=`: one chunk plus the clauses around it.

**Mounting into the main service.** `app.include_router(regs.app.router); app.include_router(regs.app.page_router)` puts the page at `/regs` on the same public URL. Load the embedder at startup there too, as `app.lifespan` does.

## 8. Operations

Run these from `svc/` with env/main active (`pip install -r regs/requirements.txt`).

```bash
python -m regs fetch          # once, online: 4 acts -> data/regs/raw/
python -m regs milvus &       # Milvus Lite server on 127.0.0.1:19530 (keep running)
python -m regs build          # parse -> chunk -> embed -> index  (about 1 min on M1)
python -m regs train-ltr      # LambdaMART on the dev queries (~1 min); rerun after every build
python -m regs serve          # http://127.0.0.1:8010/regs  (19530 is Milvus gRPC, not a web page)
python -m regs eval           # retrieval regression check
python -m regs verify ../rules.yaml   # exit 1 if any catalogue quote is not in the acts
python -m regs search "кассир отходит от рабочего места" -k 5
```

**Server paths (design.md).**
- `REGS_DATA_DIR=/home/jovyan/model_train_amir/regs`
- `HF_HOME=/home/jovyan/model_train_amir/hf`, `HF_HUB_OFFLINE=1`
- In `run.sh`: start `python -m regs milvus` in a restart loop next to the SGLang servers, and before uvicorn.

**GPU memory.** Models load in fp16 on cuda/mps (`REGS_MODEL_DTYPE`). bge-m3 then needs about 1.1 GB, and the optional cross-encoder another 1.1 GB. That is small next to the 61 GB budget, but count it in the memory test with all three models.

**On the M1 laptop (2026-10-17).** The repo is in `~/Documents`, which iCloud Drive syncs. With the disk over 90% full, macOS had evicted 8,307 files of the project `.venv`, the Milvus manifest and the source files to iCloud. Every import then crawled or hung. The fix:
- run from a venv outside iCloud, `~/.venvs/hackathon26`;
- keep generated data there too, with `REGS_DATA_DIR=~/.local/share/hackathon26/regs`;
- never keep a Milvus data directory inside an iCloud-synced folder.

## 9. Limitations and next steps

- **Edition.** The texts come from a mirror. Adilet is the authority, so every clause that goes into `rules.yaml` must be opened on adilet before submission and `edition_checked` filled in. If the adilet API becomes available with a key, `fetch.py` can switch to it, and the parser stays the same.
- **Top-1 needs more than two signals.** LambdaMART can only reweight cosine and BM25. Next steps, in order:
  1. Label real InternVideo3 observations from the training clips as more training queries. LambdaMART improves with labels, and these are the queries that matter.
  2. Add the cross-encoder score as a third LambdaMART feature, so the model learns when to trust it. Alone it had the best holdout top-1 (0.44).
  3. Glossary terms for the vocabulary gaps behind the worst cases ("за стойку" for the cash room, "упаковывает" for "в присутствии контролирующего работника"). Tune these on new dev cases, not on holdout.
- **Russian only.** Kazakh versions of the acts (`/kaz/docs/`) are not indexed. The UI and the jury work in Russian.
- **More acts.** Add an act by appending to `sources.yaml` and running `fetch` and `build`. The anti-terrorism requirements for banks are a candidate once a version for banks is confirmed.
- **The eval set is ours.** Grow it with real InternVideo3 observations from the training clips once the pipeline produces them. Those are the queries that matter.
- **Not wired into `pipeline.py`** (section 0). The judge integration needs a design.md change first.
