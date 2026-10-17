# Kuá

English | [Русский](README.ru.md)

Kuá (Kazakh «куә», "witness") reviews CCTV video from a bank cash desk. You upload a recording. The service says
whether it found signs of possible violations of cash-handling rules and, for every flagged moment, shows the time
code, 1-3 evidence frames, a short description of what is visible, and the regulation the moment may concern: act,
clause, verbatim quote and a link to the text on adilet.zan.kz.

It does not accuse anyone. The result is a hint for a human reviewer, not a legal or disciplinary conclusion.
Everything runs on one GPU server with open-source models; no cloud AI service is called while a video is processed.

**Highlights.** Measured on the 40 labelled training clips (the only labelled data, from one camera angle; the settings
were chosen on these clips), details in [section 6](#6-results-on-the-training-clips):

- 20 of 21 violation clips found: every labelled moment has an event within 2 s.
- The time code is within 2 s of the label for 95.5% of the labelled moments, with a median error of 1.0 s.
- 28.3 s per clip on average, one clip at a time.
- Five long clips sent at once finished within 148.9 s of the 240 s deadline, with 0 judge fallbacks.
- Every flag comes with 1-3 evidence frames, a time code and a cited norm. Quotes are taken verbatim from the act text on
  adilet.zan.kz and no model writes a citation (the catch-all type `other` cites the act as a whole).
- It runs fully locally on one GPU server.

Jump to: [1 Address](#1-address-and-access) · [2 Quick start](#2-quick-start) · [3 Architecture](#3-architecture-approach-and-models) ·
[4 Normative sources](#4-normative-sources-and-how-rules-are-matched-to-video) · [5 Violation types](#5-violation-types) ·
[6 Results](#6-results-on-the-training-clips) · [7 Scope and limitations](#7-scope-and-limitations) ·
[8 Running the service](#8-running-the-service) · [9 Repository](#9-repository)

## 1. Address and access

| | |
|---|---|
| Web UI | https://frida-parallelepipedous-cubicly.ngrok-free.dev |
| API base | https://frida-parallelepipedous-cubicly.ngrok-free.dev/api/v1 |
| API key | Sent to the jury separately. Send it in the `X-API-Key` header. A missing or wrong key gets HTTP 401. |
| Health check | `GET /healthz`, no key needed, answers `{"ok":true}` |

In a browser, the free ngrok domain first shows a notice page with a one-click "Visit Site" button. Click it once.
API clients such as curl or scripts do not see that page.

## 2. Quick start

### 2.1 In the browser

The interface is in Russian.

1. Open the web UI and click through ngrok's notice page.
2. Paste the API key into «API-ключ». It is stored only in this browser.
3. Drop a video onto the page or choose a file (`.mov` or `.mp4`, up to 200 MB) and press «Проверить видео».
4. The page shows the stage (в очереди, чтение видео, проверка фрагментов, формирование результата) and the elapsed
   time. It says "up to 4 minutes"; the service's own deadline is 240 s. The address gets `#job=<id>`, so a reload or
   the same link brings the job back.
5. The result shows one of the two verdicts from the brief: «Нарушение не обнаружено» or «Обнаружены признаки
   возможных нарушений», followed by the note that this is a hint for a human reviewer. Below it are the video with a
   timeline of events and one card per event:
   - the time code (a button that jumps the video to that moment), the type, and the fragment it covers;
   - «Что видно»: what the video model saw, in Russian;
   - «Почему это может быть нарушением»: the language model's justification, or the catalogue's cautious text;
   - «Правило и источник»: document, clause, verbatim quote and a link to adilet.zan.kz;
   - the evidence frames (click to enlarge);
   - a tag: «оценка модели» (type and wording from the language model), «слабый сигнал» (weak signal: the language
     model is unsure of a violation, so the catalogue's cautious wording is shown) or «по каталогу» (the language model
     did not answer, so type and text come from the rules catalogue).

### 2.2 Through the API

Two steps, as in the brief: upload, then poll.

```bash
BASE=https://frida-parallelepipedous-cubicly.ngrok-free.dev
KEY=<API_KEY>                      # sent to the jury separately

# 1. Upload (multipart field "file"). The answer is immediate.
curl -X POST "$BASE/api/v1/analyze" -H "X-API-Key: $KEY" -F "file=@video_017.mov"
# {"job_id":"3f2a9c1e7b4d","status":"processing"}

# 2. Poll. status is processing, done or failed.
curl -H "X-API-Key: $KEY" "$BASE/api/v1/jobs/3f2a9c1e7b4d"

# Or wait in a loop and keep the result:
JOB=3f2a9c1e7b4d
until curl -s -H "X-API-Key: $KEY" "$BASE/api/v1/jobs/$JOB" -o result.json \
      && ! grep -q '"status":"processing"' result.json; do sleep 3; done
```

A finished job returns the schema of the brief (section 10) plus extra fields. The values below are illustrative;
`image_base64` is shortened.

```json
{
  "job_id": "3f2a9c1e7b4d",
  "status": "done",
  "verdict": "violation",
  "violations": [
    {
      "type": "Оставление наличных без присмотра",
      "timecode_sec": 44.2,
      "description": "<what is visible> <why it may be a violation> Основание: <document>, п. 20, ч. 2.",
      "evidence_frames": [{ "timestamp_sec": 44.2, "image_base64": "/9j/4AAQ..." }],

      "type_id": "cash_unattended",
      "rule": { "document": "<act title>", "clause": "п. 20, ч. 2", "quote": "<verbatim quote>",
                "url": "https://adilet.zan.kz/rus/docs/V1900019680#z112" },
      "confidence": 0.31,
      "observation": "<what the video model saw>",
      "justification": "<why it may be a violation>",
      "type_source": "judge",
      "t_start": 40.0,
      "t_end": 48.0,
      "weak_signal": false
    }
  ],
  "video_url": "/api/v1/jobs/3f2a9c1e7b4d/video?token=<per-job token>"
}
```

| Field | Meaning |
|---|---|
| `verdict` | `violation` if at least one event was kept, otherwise `no_violation`. There is no third value. |
| `violations` | Kept events, sorted by time; `[]` when the verdict is `no_violation`. |
| `type` | The rule's name (Russian), see [section 5](#5-violation-types). |
| `timecode_sec` | Time of the event in seconds from the start, rounded to 0.1 s. It is the time of the first evidence frame. |
| `description` | `observation` + `justification` + `Основание: <document>, <clause>.`, so the citation is inside the field the brief defines. |
| `evidence_frames` | 1-3 JPEG frames (quality 90), base64: at the time code, at the start and at the end of the fragment; a frame closer than 0.5 s to another is dropped. |
| **Extra fields** | |
| `type_id` | Stable rule id (`cash_unattended`, ..., `other`). |
| `rule` | `document`, `clause`, `quote`, `url` of the cited norm. `clause` is empty for the document-level citation of `other`. |
| `confidence` | The video model's probability of YES for the fragment (0-1, two decimals); 0 when the check gave no answer. Not calibrated; an event is kept from 0.05 upward. |
| `observation` | What the video model saw, in Russian. |
| `justification` | The justification shown: the language model's, or the catalogue's cautious text for a weak signal or a fallback. |
| `type_source` | `judge` (the language model chose the type) or `fallback` (it did not answer; the type is the video model's or `other`). |
| `t_start`, `t_end` | The fragment the event covers (a 4 s cell), in seconds. |
| `weak_signal` | `true` when the language model's own probability of a violation is below 0.5: the event is kept and shown with the catalogue's cautious wording. |
| `video_url` | Added to every state of `GET /api/v1/jobs/{job_id}`. The uploaded video; the token in the URL is valid for this job only. |

Other details of `GET /api/v1/jobs/{job_id}`:

- While `processing` it also returns `stage` (`queued`, `decoding`, `verifying`, `writing`) and `elapsed_sec`.
- When `failed` it returns `error` (for example when the file is not a readable video).
- `?debug=true` adds a `debug` object with every candidate fragment, kept or rejected, their scores, timings and the
  versions of the models and thresholds used (no images).
- `GET /api/v1/jobs/{job_id}/video` returns the uploaded file (range requests work). It accepts the per-job token from
  `video_url` or the `X-API-Key` header.

| HTTP code | When |
|---|---|
| 401 | Missing or wrong `X-API-Key`. Checked before the upload body is read. |
| 413 | Upload larger than 200 MB (the server's `MAX_UPLOAD_BYTES`). |
| 400 / 422 | The request is not `multipart/form-data`, or it has no `file` field. |
| 404 | Unknown or malformed `job_id` (ids are 12 hexadecimal characters). |

## 3. Architecture, approach and models

```
 browser / curl
      |  HTTPS, public URL (ngrok tunnel)
      v
 API + UI  :8000   FastAPI. Key check, queue (2 worker threads), jobs on disk,
      |            240 s deadline per video, counted from the end of the upload
      v  for each job
 1 read    decode; make an upright internal copy if the input needs it
 2 cut     4 s cells; up to 16 (evenly spaced) are checked, each with 2 s of context on both sides
 3 check   InternVideo3-8B, 4 replicas (:30000, :30002-:30004):
           per fragment YES/NO, a rule id, a Russian description
 4 keep    P(YES) >= 0.05, or no answer; merge neighbours of the same type; verdict
 5 frames  1-3 JPEG frames per kept event
 6 match   RAG server :30001: bge-m3 + BM25 + glossary, fused by reciprocal rank fusion
           Judge Qwen3.5-9B :30005 (SGLang): rule, clause, P(violation), justification
 7 write   citation copied from the act text; JSON result
```

### 3.1 Processes

| Process | Port | Python env | Role |
|---|---|---|---|
| API and UI | 8000 | `env/main` | `api.py`, `pipeline/`, `static/`. Job folders on disk; unfinished jobs are queued again after a restart. |
| Video check, 4 replicas | 30000, 30002, 30003, 30004 | `env/iv3` | `servers/iv3_server.py`, InternVideo3-8B-Instruct. |
| Regulation matcher (RAG) | 30001 | `env/main` | `servers/rag_server.py`. Retrieval over the clauses, and the front for the judge. |
| Judge LLM | 30005 | `env/vlm` | Qwen3.5-9B on SGLang 0.5.19. |
| Public URL | - | ngrok agent | `tunnel.sh`, forwards the public address to port 8000. |

There are three Python 3.12 environments because the models need different library versions (`env/main`:
transformers 5.14.1; `env/iv3`: transformers 4.57.3; `env/vlm`: SGLang). The server has one NVIDIA H200 (144 GB); our
processes used about 113 GB at the last check. The GPU server is shared with other jobs.

### 3.2 Models

| Model | Used for |
|---|---|
| InternVideo3-8B-Instruct | Looks at each fragment (4 fps) and answers one question that lists all the rules: `ANSWER: YES/NO`, `TYPE: <rule id or none>`, `OBSERVATION: <1-3 sentences in Russian>`. P(YES) is read from the model's own first answer token. |
| Qwen3.5-9B | The judge. Given the video model's description, the candidate rules and the offered clauses, it returns the rule, the clause, its own probability of a violation and a justification of 1-3 sentences in Russian worded as a possibility. Thinking is off and temperature is 0. **It reads text only, not the frames.** |
| BAAI/bge-m3 | Dense embeddings of the 247 clauses and of the observation. |
| BM25 and a small glossary | The lexical side of retrieval (Russian stemming; the glossary maps everyday words such as "смартфон" to the acts' wording). Fused with the dense side by reciprocal rank fusion (k = 60). |

VideoJev-4B (an event-search model) and an RT-DETR person detector (a zoom on people) are also in `setup.sh` and in the
code, but they are off in the deployed configuration.

The models are used as released; nothing was trained or fine-tuned.

### 3.3 What happens to a video

1. **Accept.** The key is checked before the body is read. The upload is streamed to disk (limit 200 MB) and queued.
   Two jobs run at a time; the others wait, and the wait counts against the 240 s deadline.
2. **Read.** The video is probed. If it is not H.264, has a rotation tag or does not start at time 0, it is copied once
   to an upright internal file (the server has no H.264 encoder, so the copy is MPEG-4).
3. **Cut.** The video is divided into 4 s cells. Up to 16 cells are checked: every cell for a clip of at most 64 s,
   otherwise 16 evenly spaced cells. Each cell is checked on a fragment that extends 2 s before and after it (at most 8 s).
4. **Check.** InternVideo3 gets the fragment and a question that lists the 14 detectable rules (the 13 specific rules and
   the catch-all `other`; the question for each rule comes from the catalogue) with instructions to use only what is
   visible, never to identify or describe a person, and to call people only «кассир», «клиент» or «инкассатор».
5. **Keep.** A fragment is kept when P(YES) >= 0.05. A fragment whose check gave no answer (a late or failed call, or
   no time left) is also kept, because a miss is worse than a false alarm. Kept events of the same type whose time codes
   are within 3 s are merged; with 4 s cells this affects only a short last cell, so neighbouring cells stay separate
   events. The verdict is `violation` if any event is kept.
6. **Evidence.** 1-3 JPEG frames are cut for each kept event.
7. **Match** (for each kept event, in parallel; see [section 4](#4-normative-sources-and-how-rules-are-matched-to-video)).
   The candidate rules are the video model's `TYPE`, up to four rules whose catalogue clauses are among the top 8
   retrieved for the description, and `other`. The judge picks one candidate and a clause. If its answer is missing,
   late or not a valid candidate, the event gets the video model's type (or `other`) and the catalogue's text.
8. **Write.** The citation is copied from the act text (no model writes it), the result is stored with the job, and the
   status becomes `done`.

Every call to a model server has a timeout and never raises: a failed or late call falls back to the cheaper path
instead of failing the job. A job becomes `failed` when the video cannot be read, or on an unexpected error.

### 3.4 Deployed settings (`thresholds.json`)

| Key | Value | Meaning |
|---|---|---|
| `generator` | `grid` | Cut the clip into cells and check them; no search model. |
| `cell_sec` | 4.0 | Cell length in seconds. The time code of an event is the cell's centre. |
| `top_k` | 16 | At most 16 cells are checked per clip. |
| `verify` | 0.05 | Keep a fragment when P(YES) is at least this. |
| `merge_sec` | 3 | Merge same-type events whose time codes are this close. |
| `judge_weak` | 0.5 | Below this violation probability from the judge the event is a weak signal. |
| `verify_mode` | `single` | One question listing all rules (not one question per rule). |
| `zoom` | `false` | No person zoom. |

The file also holds `cand`, `rank`, `strong`, `fallback` and `anchor` keys that the grid generator does not use.

### 3.5 Local operation

- The service calls only servers on `127.0.0.1` (model servers and the matcher). The environment sets
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; the weights are downloaded once by `setup.sh`. After that, processing a
  video needs no internet. The ngrok tunnel is only the way in from outside.
- No audio is used, and no face recognition or person identification is done.
- The repository also has `tools/rag_eval.py`, a development measurement script with an optional cloud-LLM backend. It was
  used for one measurement on a developer laptop. The service never imports or calls it, and the backend refuses to
  start on the server.

## 4. Normative sources and how rules are matched to video

### 4.1 Sources

All quotes in `rules.yaml` are verbatim from the Russian text of the acts on adilet.zan.kz, read on 2026-10-17 through
adilet's public API. Each citation of a clause carries a link with the paragraph anchor.

| Act | Used for | Edition read |
|---|---|---|
| Resolution of the Board of the National Bank of Kazakhstan (НБРК) of 29 November 2019 No. 231, [V1900019680](https://adilet.zan.kz/rus/docs/V1900019680): rules for cash operations and for collection of banknotes, coins and valuables in second-tier banks | The basis of 13 of the 14 detectable rules and of all rules that cannot be checked | Revision of 12.07.2026 |
| НБРК Resolution of 24 February 2020 No. 14, [V2000020075](https://adilet.zan.kz/rus/docs/V2000020075): rules for the protection and the arrangement of bank premises | `unauthorized_person_in_cash_area` (п. 8, ч. 1) and, at document level, `other` | Revision of 24.12.2024 |

The retrieval corpus (`rag/acts/`) holds four acts and 247 in-force clauses: the two above, НБРК No. 49 (exchange
operations with cash foreign currency; chapters 1, 5 and 6, [V1900018545](https://adilet.zan.kz/rus/docs/V1900018545)) and НБРК
No. 230 (payability of banknotes and coins, [V1700016120](https://adilet.zan.kz/rus/docs/V1700016120)). The judge can offer a clause from
any of them; the catalogue cites only No. 231 and No. 14. Clauses that are repealed or suspended are not in the corpus.
An amendment to No. 14 (Resolution No. 88) suspends its new wording of п. 11, 13 (подп. 3), 14 and 16 until 01.01.2027;
the corpus marks п. 11, 14 and 16 as suspended and drops them, and the catalogue cites none of them.

Checked and deliberately not cited: НБРК No. 58 of 03.03.2001 and No. 394 of 15.11.1999 (repealed, replaced by No. 231),
No. 247 of 24.12.2014 (repealed, covers only the National Bank's own units) and No. 120 of 28.09.2020 (also covers the
National Bank itself). Tests check that every cited clause exists in the in-force corpus and that no repealed act is cited.
The full table of sources, the rules, the limitations and the legal caveats are in [RULES.md](RULES.md) (sections 1, 3, 4
and 5); the data is in [rules.yaml](rules.yaml).

### 4.2 From video to citation

1. **Each rule is written as something visible.** An entry of `rules.yaml` holds what must be seen (`visual_criteria`),
   what must never be claimed (`never_claim`: amounts of cash, postings, authenticity of banknotes, the identity of a
   person, and so on), the YES/NO question for the video model, a cautious Russian text («может указывать на ...»), and
   one to three citations with a verbatim quote. Descriptions of actions and roles never use screen positions (a test
   guards this), so the rules do not depend on where the camera stands.
2. **The video model sees the fragment and describes it.** Its description is the only thing that links the picture to
   the text of the law.
3. **Retrieval.** The description is searched in the corpus: bge-m3 dense search and a BM25 search (with the glossary),
   merged by reciprocal rank fusion; the top 8 clauses are kept. On 15 hand-written test queries the correct clause was in
   the top 8 for 12 queries with hybrid search and 10 with dense search alone.
4. **The judge chooses.** It is shown the description, the candidate rules (criteria, `never_claim`, citations) and the
   clauses: first the clauses of the candidates' own catalogue citations, then the retrieved ones. It returns a rule id
   from the candidates, a clause id from the offered clauses (or none), its probability of a violation and a justification.
5. **The citation is copied, never written by a model.** For the clause the judge chose, the document, "п. N", the quote
   (the retrieved passage, or else the catalogue quote) and the link are taken from the act text in the corpus. If the
   judge gave no clause, or the event has no valid judge answer, the rule's first catalogue citation is used.
6. **Weak signals.** When the judge is unsure (its own probability of a violation is below 0.5), the event is kept with
   the judge's rule and citation, but the justification becomes the rule's cautious catalogue text and the event is
   marked as a weak signal. The judge never changes which events are kept or the verdict.

Because of step 5 a flag can be wrong about what happened, but it cannot cite a clause that does not exist.

## 5. Violation types

The service recognises 14 types: 13 specific ones and `other`. `type` in the API is the Russian name. All types are
checked from the picture alone, with no data from the bank's systems. "Norm" gives the clause of НБРК No. 231, except
where No. 14 is named. † marks a type where a required step is missing; its time code points to the moment the step
should have happened.

| `type_id` and name | What the video must show | Norm |
|---|---|---|
| `other`: Иное возможное нарушение порядка кассовых операций | Cash or valuables are handled in a way the cashier does not control, or an action with cash does not match the normal steps of acceptance, issuance or collection. | No. 231 and No. 14, document level (no clause) |
| `cash_unattended`: Оставление наличных без присмотра | Cash lies open on the desk, the cashier walks away or turns away for a long time, and no employee watches it. | п. 20, ч. 2 |
| `cashier_leaves_during_service`: Уход кассира с рабочего места во время обслуживания клиента | A client is at the desk and the operation is not finished; the cashier gets up and leaves; the client waits, maybe with cash or documents on the counter. | п. 22 |
| `previous_cash_on_desk`: Наличные от других операций на рабочем месте при обслуживании клиента | While a client is served, other stacks of cash that are not part of this operation lie open; cash from earlier operations is not put into a drawer or safe; cash of different clients is mixed. | п. 20, ч. 2 |
| `cash_storage_left_open`: Хранилище наличных на рабочем месте оставлено открытым | A drawer, safe, box or trolley with cash is left open; keys are in the lock or on the desk; the cashier is away or busy. | п. 20, ч. 2; п. 80 |
| `cash_into_personal_belongings`: Перемещение наличных в одежду или личные вещи | The cashier or another person puts banknotes into a pocket, sleeve, personal bag or wallet instead of the drawer, safe or collection bag; it may be done discreetly. | п. 20, ч. 2; п. 12, ч. 1 |
| `personal_items_at_workplace`: Личные вещи или смартфон на рабочем месте кассира | The cashier uses a smartphone at the workplace or photographs or films with it; a personal bag, wallet or other personal items lie on the desk. | п. 12, ч. 1; п. 12, ч. 2 |
| † `deposit_not_counted`: Приём наличных без пересчёта | The client hands over banknotes or coins and the cashier puts them straight into a drawer, safe or bag, with no sheet-by-sheet count by hand or on a counting machine. | п. 19, подп. 2); п. 20, ч. 1 |
| `deposit_counted_without_client`: Пересчёт принимаемых наличных вне присутствия клиента | The cashier takes the client's cash away from the counter before counting it, or counts it only after the client has left. | п. 20, ч. 1 |
| † `withdrawal_not_counted`: Выдача наличных без пересчёта в присутствии клиента | The cashier hands cash to the client: loose banknotes or an opened or partial bundle without counting in front of the client. A full bundle in intact bank wrapping is not a violation. | п. 31, подп. 4); п. 32, ч. 1 |
| † `recount_without_controller`: Пересчёт, сортировка или упаковка наличных без контролирующего работника | No client is being served; the cashier counts, sorts, bands or packs cash and no second employee watches. | п. 42 |
| `unauthorized_person_in_cash_area`: Доступ постороннего лица в зону кассы | A person who is not the cashier enters the cashier's workspace, or a client reaches across the counter for cash, drawers or documents, and the cashier does not stop it. | No. 14, п. 8, ч. 1 |
| `collection_cash_not_in_sealed_bag`: Передача наличных инкассатору без опломбированной сумки | Cash passes between the cashier and a cash collector as loose banknotes or open bundles, or the bag is open, not tied or unsealed when handed over. | п. 105; п. 93, ч. 1 |
| † `collection_bag_not_inspected`: Приём инкассаторской сумки без проверки целостности и пломбы | The collector and the cashier exchange a bag and the receiving side puts it away or opens it without looking at the bag and its seal. | п. 24, ч. 2, подп. 3) и 4); п. 107, подп. 2) |

### Rules in the catalogue that the service never judges

Nine rules are written down with their citations but are never emitted, because the facts they need are not visible on
video. This follows the brief: the service must not claim a violation when the conclusion needs data that is not visible.

| `type_id` | Norm | Why it cannot be checked from video |
|---|---|---|
| `client_identification` | п. 19, подп. 1); п. 7 | Identification may use a digital-documents service on the cashier's screen, and face recognition is forbidden for the service. |
| `signatures_and_document_details` | п. 19, подп. 3) | Signatures, comparison with specimens and document details cannot be read from the camera. |
| `amount_matches_document` | п. 19, подп. 4) | The amount of cash and the amount in the document cannot be determined reliably from the camera. |
| `posting_to_client_account` | п. 26, ч. 1 | Crediting and the correctness of the posting are visible only in the bank's accounting system. |
| `banknote_authenticity_check` | п. 57 | The authenticity and payability of banknotes and the result of the equipment check cannot be determined from the camera. |
| `suspicious_banknotes_procedure` | п. 61 | It cannot be established that a banknote is doubtful or counterfeit, nor that the report was drawn up. |
| `cash_transfer_between_cashiers_record` | п. 15 | The record in the transfer log cannot be seen. |
| `end_of_day_reconciliation` | п. 56 | Reconciliation is done on documents and in the accounting system. |
| `collection_documents_and_seal_match` | п. 24, ч. 2, подп. 1); п. 94, ч. 3 | The accompanying document and the match of the seal impression with the specimen cannot be read from the camera. |

## 6. Results on the training clips

Measured on the 40 labelled training clips (21 with a violation, 19 clean; 22 labelled moments, because one clip has two),
all from one camera angle. A clip counts as "found" when every labelled moment has a kept event within 2 s (our own
tolerance; the brief gives none).

- **Run 1:** the 40 clips sent through the API one at a time, with the deployed settings. It was made before the
  Qwen3.5-9B judge was connected; the judge sets the type and the wording, not which events are kept.
- **Run 2:** the five longest labelled clips (X-4, X-5, X-13, X-14, X-39) sent at the same time, after the Qwen3.5-9B
  judge was connected.

| Measure | Value | Run |
|---|---|---|
| Violation clips found (every labelled moment has an event within 2 s) | 20 of 21 | 1 |
| Time-code error, 22 labelled moments | within 1 s: 17 (77%); within 2 s: 21 (95.5%); median 1.0 s; largest 2.0 s | 1 |
| Latency, one clip at a time | mean 28.3 s, 95th percentile 60.4 s, longest 61.5 s (measured before the Qwen judge was connected) | 1 |
| Latency, five long clips at once | slowest 148.9 s (the 142.5 s clip X-14) of the 240 s deadline; the others 59.9-125.7 s; 71 judge calls, 0 fallbacks | 2 |
| Retrieval of the right clause | in the top 8 for 12 of 15 hand-written test queries with hybrid search (10 with dense search alone) | retrieval check, [section 4.2](#42-from-video-to-citation) |

## 7. Scope and limitations

**Scope**

- **A result is a hint** for a human reviewer, not a legal or disciplinary conclusion. There are two verdicts only,
  `violation` and `no_violation`.
- **Strictly visual.** The service uses no audio, does no face recognition or identification of people, and does not
  read amounts or documents.
- **Nine rules are listed but never judged** (the table in [section 5](#5-violation-types)): they need data that is not
  on the video. Anything outside the 14 types can be reported only as `other`.
- **Built for one top-down camera and one cash desk.** The prompts name roles and actions, never positions on the
  screen. All labelled clips are from one angle (464×848 pixels, 11.6-142.5 s).
- **Clips longer than 64 s are sampled.** Up to 64 s every 4 s cell is checked; for a longer clip, 16 evenly spaced
  cells are checked, each with 2 s of context on both sides. This limit follows from the time budget.
- **The legal reading is ours.** The acts were read on adilet.zan.kz on 2026-10-17 through its public API; later
  amendments are not tracked and the Kazakh-language text was not checked. `unauthorized_person_in_cash_area` is an
  inference: neither act says in words that outsiders may not enter the cash desk, so it rests on No. 14 п. 8 ч. 1 (the
  cash unit is isolated by a lockable entrance). `recount_without_controller` is judged on what is visible: the
  controlling employee may be out of frame. `personal_items_at_workplace` is low severity: п. 12 ч. 2 permits phones
  without photo or video functions, which video cannot show. A client who leaves without counting cash is not flagged.

**Operation**

- **Deadline.** A job has 240 s from the end of the upload (the brief allows 5 minutes). Two jobs run at a time; the rest
  queue, and their wait counts against the deadline.
- **The public address needs the tunnel agent running.** Browsers first see ngrok's notice page. After a hard kill of the
  agent, ngrok can refuse a new one for about a minute.
- **Playback.** The page plays the uploaded file. If the browser cannot decode the codec (for example HEVC in Chrome), the
  video does not play, but the frames and time codes are still shown.
- **Uploaded videos and results stay in the job folder on the server.** There is no automatic cleanup.

## 8. Running the service

Everything lives under one root on the GPU server, `/home/jovyan/model_train_amir/` (`<root>` below): `svc/` is this
repository, next to `env/` (Python environments), `hf/` (model weights), `jobs/`, `logs/` and `secrets/`. The scripts
read their paths from `svc/env.sh`, which keeps every cache, download and temporary file inside `<root>` and runs the
models offline.

**One-time setup**

```bash
cd <root>/svc
bash setup.sh tools toolchain main iv3 vlm weights   # any subset; every target can be repeated safely
```

It builds the three environments (wheels only, nothing is compiled on the server) and downloads the weights of
VideoJev-4B, InternVideo3-8B-Instruct, Qwen3.5-9B, the RT-DETR detector and bge-m3 into `hf/`.

Two secrets are created by hand and are never part of the repository:

- the API key: one line, no spaces, in `<root>/secrets/dev_api_key` (mode 600), read by `api_server.sh` into the server's
  environment only;
- the ngrok agent configuration in `<root>/secrets/ngrok.yml` (mode 600), used by `tunnel.sh`.

**Start, in this order**

```bash
cd <root>/svc
bash tools/judge_llm.sh start    # Qwen3.5-9B on :30005. Needs 35 GiB of free GPU memory; loading takes minutes.

# InternVideo3 replicas, one process per port: 30000, 30002, 30003, 30004 (about 19 GB of GPU memory each).
# There is no tracked start script. After `source <root>/env.sh`, start each one detached:
#   $MTA_ROOT/env/iv3/bin/python -m uvicorn servers.iv3_server:app --host 127.0.0.1 --port 30000

bash tools/rag_server.sh start   # :30001. Loads bge-m3 and embeds every clause before it opens the port.
bash tools/api_server.sh start   # :8000. With the grid generator it is ready in about 5 s.
bash tunnel.sh install           # once: the pinned ngrok agent
bash tunnel.sh start && bash tunnel.sh wait
```

Each of `judge_llm.sh`, `rag_server.sh` and `api_server.sh` takes `start`, `stop` or `status`. `status` exits 0 when the
server is healthy, 3 when it runs but is not ready yet, 1 when it is not running. `stop` sends TERM and, after 60 s,
KILL, and only touches a process it recognises as its own. Stopping the API cuts the jobs that are running; the next start
queues them again. `tunnel.sh` takes `install`, `start`, `wait [SECS]` (exit 3 if the endpoint is not online in time),
`stop` and `status`. A loop restarts the agent when it exits.

**Check without a GPU.** The stub pipeline returns one fixed placeholder event (text marked `[заглушка]`), which is enough
to try the UI and the API:

```bash
cd svc && uv sync
API_KEY=<any string> PIPELINE=stub uv run uvicorn api:app --port 8000
```

Environment variables of the API: `API_KEY` (required), `PIPELINE` (`stub`, or `real` on the server), `JOBS_DIR`,
`RULES_PATH`, `MAX_UPLOAD_BYTES` (default 200 MB), `JUDGE_URL` and `IV3_URLS` (set by `api_server.sh`).

## 9. Repository

| Path | Contents |
|---|---|
| `api.py`, `jobs.py` | HTTP API, queue and workers, job folders on disk |
| `pipeline/` | The review pipeline: `real.py` (steps), `logic.py` (keep and merge rules), `clients.py` (prompts and calls to the model servers), `frames.py` (video access), `stub.py` |
| `servers/` | `iv3_server.py` (video check), `rag_server.py` (matcher and judge front) |
| `rag/` | Clause corpus (`acts/`), retrieval, lexical search, glossary, matcher |
| `rules.yaml`, `rules.py`, [RULES.md](RULES.md) | The rules catalogue, its loader and validator, the sources and caveats |
| `thresholds.json` | Deployed settings |
| `static/` | The web page |
| `eval/`, `data/labels.csv` | The evaluation script and the labels of the 40 training clips (the videos are not in the repository) |
| `setup.sh`, `env.sh`, `tools/`, `tunnel.sh` | Server setup and the start scripts |
| `tests/` | Unit tests with all models mocked. At the last recorded run 520 passed. |

Run the tests with `cd svc && uv sync && uv run pytest -q`. They need no GPU; a few need `ffmpeg` and skip without it.
