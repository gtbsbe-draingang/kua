# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This is a hackathon project (ForteBank Datathon 26). Main documents:
- `Кейс_ForteBank.md`: the case spec, transcribed from the PDF.
- `design.md`: the whole-service architecture.
- `rag.md`: the regulation index.
- `regulatory_sources.md`: the acts, plus the violation types video can check and their clauses.

The only code so far is the regulation index in `svc/regs/`. The video pipeline and the jury API from design.md are not built yet. When you add them, add their commands here.

## Regulation index (`svc/regs/`)

Run commands from `svc/`. Install with `pip install -r regs/requirements.txt`.
- **On the M1 laptop**, use `~/.venvs/hackathon26/bin/python` with `REGS_DATA_DIR=$HOME/.local/share/hackathon26/regs`. Do not use the repo's `.venv`. `~/Documents` is synced by iCloud Drive and the disk is over 90% full, so macOS evicts files from it: imports hang, and a Milvus directory there can break.
- **On the server**, use env/main.

```bash
python -m regs fetch                 # download the acts in regs/sources.yaml (online, once)
python -m regs milvus                # Milvus Lite server on 127.0.0.1:19530; keep it running
python -m regs build                 # parse -> chunk -> embed (bge-m3) -> Milvus; full rebuild
python -m regs train-ltr             # LambdaMART on cosine + BM25 features; rerun after every build
python -m regs serve                 # UI at http://127.0.0.1:8010/regs, API at /api/regs/*; 19530 is Milvus gRPC, not a page
python -m regs eval                  # retrieval regression check on regs/eval_queries.yaml
python -m regs verify rules.yaml     # exit 1 if a citation quote is not verbatim in the acts
python -m regs search "query" -k 5   # one query from the terminal
```

- **Milvus runs as its own process; don't switch to embedded mode by default.** milvus-lite 3.x runs faiss in-process, and faiss's OpenMP runtime clashes with torch's, which crashes macOS runs. If `REGS_MILVUS_URI` is a file path, `config.py` sets `OMP_NUM_THREADS=1` as a fallback.
- **BM25 stemming happens in `lexical.py` (Snowball).** Milvus Lite ignores the analyzer's `stemmer` filter.
- **Texts are fetched from the zakon.uchet.kz mirror.** adilet.zan.kz is a JS app whose API needs a key. The mirror keeps adilet's HTML and `#zNN` anchors. Citations always point to adilet.
- **Chunks are one clause each, plus one per subclause for list-type clauses.** `search()` folds them back to one result per clause. Quotes come from the chunk's verbatim `text` field.
- **Ranking.** The default mode `auto` uses LambdaMART (`ltr.py`, scikit-learn trees, monotone constraints) over cosine + BM25 features, and falls back to hybrid RRF if no model is trained.
  - An optional cross-encoder or Qwen3 stage (`rerank.py`) runs only when `REGS_RERANK_MODEL` is set.
  - Judge any ranking change by top-1 on the **holdout** set in `regs eval`. Train and tune only on the dev `queries`, never on `holdout`.
- **Milvus Lite returns BM25 as a negative distance.** `store.search_bm25` applies `abs()`. Its BM25 statistics also change with the filter, so never compare BM25 scores from searches with different filters.
- **Models load in fp16 on cuda/mps** (`REGS_MODEL_DTYPE`). Rebuild the index if this setting changes.
- **All settings are `REGS_*` environment variables, read in `regs/config.py`.** On the server, set `REGS_DATA_DIR=/home/jovyan/model_train_amir/regs` and `HF_HUB_OFFLINE=1`.
- **design.md §4 still says "no search over legal texts".** `rag.md` §0 proposes the change. Keep the invariant that no model writes a citation.

## What is being built

A self-hosted AI service that analyzes overhead CCTV-style video of a bank cashier's desk and flags visually observable signs of cash-handling discipline violations. It has a web UI and a public API, and the jury scores it automatically through the API.

User flow (spec §4): upload a video → see processing status → get a verdict: «Нарушение не обнаружено» ("no violation found") or «Обнаружены признаки возможных нарушений» ("signs of possible violations found"). For each violation, show its type, timecode, 1–3 evidence frames, a short description of what is visible and why it may be a violation, and the rule it relates to (document name + clause/section).

## Hard constraints (breaking these means disqualification or lost points)

- **Fully local inference.** All models run inside the deployed service. Calling cloud AI APIs (LLMs, multimodal models, recognition services) while processing video is forbidden. This includes sending them frames or text descriptions of frames. Once model weights are downloaded, processing must work offline so the service can move into the bank's internal network.
- **Only open-source models and code whose licenses allow commercial use.** ForteBank keeps commercial rights to the solution.
- **No face recognition or identification.** Detecting people or poses is fine; identifying individuals is not.
- **Visual-only analysis.** The videos have no audio.
- **Don't claim violations that need data the video can't show** (e.g., whether a posting or system reconciliation was correct). Output is a hint for a human reviewer, not a legal or disciplinary conclusion.
- **Rule citations must be real.** Use only Republic of Kazakhstan acts (NBRK, adilet.zan.kz). Public documents from other banks, and references without an exact clause, are also accepted. Inventing citations or making unsupported legal claims is penalized. Keep the rule catalog (violation type → source document → clause) as curated data, never generated text.

## Jury API contract (§10). The scoring is automated, so field names and values must match exactly.

- Auth: header `X-API-Key`. Reject requests that lack a valid key.
- `POST /api/v1/analyze`: multipart upload, form field `file`, `.mov` file of about 29 MB. Returns `job_id`.
- `GET /api/v1/jobs/{job_id}`: `status` is one of `processing` | `done` | `failed`. When done:

```json
{
  "job_id": "3f2a9c1e",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Violation type name",
      "timecode_sec": 44.2,
      "description": "What is visible and why it may be a violation",
      "evidence_frames": [{ "timestamp_sec": 44.2, "image_base64": "/9j/4AAQ..." }]
    }
  ]
}
```

- `verdict` is `no_violation` or `violation`. `violations` is an empty list when there are none. `evidence_frames` holds 1–3 base64 JPEGs.
- **Each video has a 5-minute limit.** A video with no result within 5 minutes counts as unprocessed. Processing must be asynchronous: background job plus status polling.
- The service must be reachable at a public URL. Auth, history, report export, and batch upload are not required for the UI.

## Data characteristics (from the FAQ)

- Training set: at least 40 staged `.mov` clips, 10–145 s long, ≥29 FPS, no audio. Each shows one overhead camera on one cash desk, with the cashier, clients, and cash collectors (инкассаторы) in frame.
- Labels give only a violation flag plus a timecode (e.g., "second 11, violation"). **Violation types are not labeled**, so the team defines the taxonomy.
- Test videos are similar, but **the final one is shot from a different camera angle**. Tests may also include violation types missing from the training data. Favor angle-robust, generalizable signals over overfitting to one scene layout.
- Augmenting with your own recordings, synthetic data, or open datasets is allowed. The dataset may be used only within the hackathon. Test videos must never be used for training or shared.

## Scoring priorities (§9 + FAQ). Use these when making design tradeoffs.

| Criterion | Pts | Note |
|---|---|---|
| Detection | 25 | Scored on the **per-video verdict**, not per event |
| Localization | 15 | Exact frame + timecode; there is no stated tolerance |
| Classification | 15 | Correct violation type |
| Rule grounding | 15 | Accurate source and interpretation, no fabricated references |
| Robustness | 10 | Unseen videos, a different angle |
| Product readiness | 10 | UI usability, upload/processing reliability, clear results |
| Technical solution | 10 | Architecture, practicality, compliance with the locality rule |

Missing a violation is "unacceptable", but false positives should be minimized. Bias toward recall, and use evidence frames plus the description so a human can dismiss false alarms. Omission violations (a required action not performed) are allowed, and the team picks their timecode convention.

## Deliverables (§8)

Service URL plus API URL plus an API key for the jury, run/usage instructions, an architecture and model overview, the regulatory sources used and how rules are mapped to video, a list of detectable violation types with descriptions, and a list of known limitations and failure cases. The deadline is 12:00 (the date isn't given in the spec).
