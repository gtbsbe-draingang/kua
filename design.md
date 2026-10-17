# Design: CCTV cash-desk review service

Status: draft, waiting for approval. The data facts below were checked against `open/` on 2026-10-17. The user chose on 2026-10-17 to add a separate local judge model for rule matching: **Qwen3.5-9B** (Apache-2.0).

Naming: everything on the server lives under `/home/jovyan/model_train_amir/` and uses neutral names (see `CLAUDE.md`). That covers the UI title, the README that ships with the code, and the tunnel and service names. `case.md` and the deck stay on the laptop.

**What the data looks like:**
- **Files:** 42 clips, all H.264, 464×848 portrait, 30 fps, 11.6–142.5 s long. Х-14 (142.5 s, clean) is the worst case for latency.
- **Duplicates:** Х-4 and Х-47 are byte-identical but labelled with different seconds (57 s and 1 min 8 s). They are one clip with two events, so one clip can contain more than one violation. Х-5 and Х-48 are identical too, both clean.
- **Totals after removing duplicates:** 21 clips with a violation, 19 without.
- **File names:** X-1 and X-2 use a Latin "X"; every other file uses a Cyrillic "Х". Normalise names before matching.
- **Labels:** in `open/open_labels.xlsx`. Each row is file, second and "нарушение"/"без нарушения". There is no violation type.

## 1. Architecture (6 components)

```
 jury script / browser ──HTTPS (stable hostname; tunnel or ingress = open item)──┐
┌─ env/main · one process: uvicorn api:app :8000 ──────────────────────────────▼─┐
│ api.py   POST /api/v1/analyze · GET /api/v1/jobs/{id} · /  → static/index.html │
│          2 worker threads → pipeline.analyze()                                 │
│ pipeline.py  VideoJev-4B in-process (behind a lock) · rules.yaml loaded once   │
└──────────────┬──────────────────────────────────────┬─────────────────────────┘
               │ HTTP localhost:30000                 │ HTTP localhost:30001
               │ (OpenAI-compatible, with logprobs)   │ (OpenAI-compatible, JSON schema)
┌─ env/vlm · SGLang · InternVideo3-8B ─┐   ┌─ env/vlm · SGLang · Qwen3.5-9B (judge) ─┐
└──────────────────────────────────────┘   └─────────────────────────────────────────┘
             disk: /home/jovyan/model_train_amir/{jobs,hf,logs}
```

1. **`api.py`** (FastAPI): checks the API key, stores job files, runs the worker threads and serves the UI.
2. **`pipeline.py`**: runs the models.
3. **InternVideo3 server** (`servers/iv3_server.py`, port 30000): confirms or rejects each moment and describes it.
   - It is a small FastAPI wrapper around the Hugging Face transformers model (4.57.3, `trust_remote_code`).
   - It isn't SGLang, because InternVideo3's SGLang model code exists only in a former employer's private image.
4. **SGLang server 2**: serves Qwen3.5-9B, the judge that picks which rule was broken and writes the justification.
5. **`rules.yaml` + `rules.py`**: the rule data plus a small loader and validator. The judge picks a rule id and the citation is looked up by that id. No model writes a citation.
6. **`static/index.html`**: one HTML page with plain JavaScript and no build step.

**Two Python environments, one per transformers version:**
- `/home/jovyan/model_train_amir/env/main` holds FastAPI, transformers 5.14.1, flash-attn and VideoJev.
- `/home/jovyan/model_train_amir/env/iv3` holds transformers 4.57.3 and the InternVideo3 server.
- `/home/jovyan/model_train_amir/env/vlm` holds SGLang, which serves the Qwen3.5-9B judge.
- Each SGLang server is its own process, so no two models share an interpreter. No Docker is needed.
- **GPU memory:**
  - `--mem-fraction-static` is a share of the whole 141 GB GPU, not of the free memory.
  - Start the judge's SGLang server with a fixed cap (start at 0.17), then the InternVideo3 server, then VideoJev.
  - The three models must fit in what the GE-2052 training leaves free: about 61 GB once the yolo1280 run stops, and more after the two 1024 runs finish.

**`run.sh`:**
- Sets `HF_HOME=/home/jovyan/model_train_amir/hf`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, so nothing needs the internet once the weights are downloaded (section 7 of the brief).
- Kills any old copies, starts both SGLang servers, waits for both health checks, then starts uvicorn and the tunnel.
- Each process runs in a restart loop and logs to `/home/jovyan/model_train_amir/logs/`.

**How a job runs:**
- **POST:**
  1. Check `X-API-Key`; return 401 if it is missing or wrong.
  2. Stream the upload to `jobs/<id>/input.mov`; return 413 above 200 MB. Don't trust the file extension; run ffprobe on the file during the job.
  3. Write `state.json` and put the job on the queue.
  4. Return 200 with `{"job_id","status":"processing"}`.
- **Worker:** updates `stage` as it goes and writes `result.json` atomically (temp file, then rename). It sets `done` on success, or `failed` with an `error` field if it raises.
- **After the pod restarts:** every job still marked `processing` goes back on the queue in `created_at` order. The files on disk are the only state.
- **Statuses:** only the three in the brief. A queued job reports `processing` with `stage: "queued"`.

**Pipeline.** The deadline is `created_at + 240 s`, so a result always comes back within the 5-minute limit.
1. **Decode** with PyAV at 2 fps. Keep the aspect ratio and don't assume portrait.
2. **Find when (VideoJev):** one `predict_video` grounding call with every rule's queries, including the generic queries of the `other` rule.
   - A rule's score is its best score over its paraphrased queries.
   - Segments scoring at least `cand` become candidates. Merge overlapping ones and keep the top 6.
3. **Which type (VideoJev):** a `choice` call on each candidate window. The options are each rule's `choice_option` plus "normal cash handling".
4. **Check and describe (InternVideo3):** one SGLang request per candidate, all in parallel.
   - The window is [start − 2 s, end + 2 s] at about 4 fps.
   - The reply puts the answer first: `YES`/`NO`, then `TYPE`, then `OBSERVATION` in Russian.
   - `verify_score` is P(YES), read from the logprobs.
5. **Keep or drop:**
   - Keep an event if `verify_score ≥ verify` or `ground ≥ strong`. InternVideo3 cannot overrule a very strong VideoJev hit, which protects recall.
   - If SGLang errors, times out, or the deadline is close, keep the event if `ground ≥ fallback`.
   - Merge events of the same type that are within 3 s of each other.
   - The verdict is `violation` if any event is kept.
6. **Judge the rule (Qwen3.5-9B):** one request per kept event, all in parallel.
   - **Input:**
     - InternVideo3's observation;
     - the 1–3 evidence frames from step 7;
     - the candidate rules: the top 3 from VideoJev's `type_probs` plus InternVideo3's `TYPE`, with each rule's name, visual criteria, `never_claim` list and full citation quotes.
   - **Output:** JSON, enforced with SGLang's JSON-schema decoding:
     - `rule_id`, which must be one of the candidates or `other`;
     - `justification` in Russian;
     - `confidence`.
   - **Prompt:** the justification may use only facts in the observation and the frames, may not state anything on the `never_claim` list, and is worded as "may indicate". Thinking is off and temperature is 0.
   - **Effect:** the judge sets the type and the justification only. It never changes which events are kept or the verdict, so recall is unaffected.
   - **If it fails:** on an error, a timeout, a rule id that isn't a candidate, or a deadline that is close, use the best VideoJev type and the rule's fixed `why` text instead.
   - **`other`:** if the judge picks `other`, the event gets the `other` rule's document-level citation.
7. **Timecode and evidence frames:**
   - The timecode is the frame inside the segment with the highest grounding score, or the segment start if the evaluation shows that matches the labels better (set by `anchor` in `thresholds.json`).
   - For a violation by omission (FAQ 10), the query describes the visible moment the required step was skipped, e.g. "hands cash over without counting it".
   - Evidence frames are taken at the timecode, the segment start and the segment end. They are decoded by seeking at the native 30 fps and saved as JPEG at quality 90. A frame within 0.5 s of another is dropped.
   - The first evidence frame is always at the timecode.

**Latency budget for a 145 s clip** (to be measured on Х-14):

| Step | Budget |
|---|---|
| Upload and save (depends on the tunnel) | 10 s |
| Decode about 290 frames | 5 s |
| VideoJev grounding plus 6 type choices | 30 s |
| InternVideo3 on up to 6 windows in parallel | 45 s |
| Frames and JSON | 2 s |
| Qwen3.5-9B judge on the kept events in parallel | 15 s |
| **Target total** | **≤ 105 s** (hard fallback at 240 s) |

The evidence frames (step 7) are cut before the judge runs (step 6), because the judge looks at them.

The jury may send 5 videos at once. With 2 workers and a lock around VideoJev, job B's VideoJev step runs while job A waits on InternVideo3.

**Choosing thresholds (recall first):** the evaluation saves every score to `debug.json`, then tries combinations of `cand`, `verify`, `strong` and `fallback` without rerunning the models.
1. Keep only the settings that flag all 21 violation clips within ±2 s of the label.
2. Among those, take the one that flags the fewest clean clips.
3. Then loosen it one step, because all 40 clips come from a single camera angle.

The values go in `thresholds.json` and are not hard-coded.

**Violation types not in the training set:**
- The catalogue includes rules from the research even when no training clip shows them. Each has its own queries, so adding one needs no code change.
- An `other` rule has generic queries such as "cash left unattended" or "a non-cashier reaches into the cash area". Its citation names the document without a clause, which FAQ 31 allows.

**A different camera angle:**
- No zones, masks, crops or detectors.
- Queries and prompts describe roles and actions, never positions on screen like "left" or "top".
- Prompts never ask who a person is (FAQ 29). Descriptions say "кассир", "клиент" or "инкассатор".

## 2. Contracts

### Rules catalogue: `/home/jovyan/model_train_amir/svc/rules.yaml`

Owned by the rules workstream. The entry below only shows the format; it is not a confirmed violation type.

```yaml
version: 1
rules:
  - id: cash_unattended            # stable type_id; exactly one rule has id "other"
    name: "Оставление наличных без присмотра"   # becomes the API "type"
    seen_in_training: true         # false = added from research only
    detectable: true               # false = documented limitation, never emitted, no queries
    severity: high                 # high|medium|low, ordering only
    omission: false
    visual_criteria: ["cash lies exposed on the desk", "cashier leaves or turns away"]   # English, used in prompts
    never_claim: ["amount of cash", "whether the operation was posted"]               # section 7
    grounding_queries: ["the cashier walks away leaving cash on the counter",
                        "cash is left unattended on the desk"]                         # VideoJev
    choice_option: "cash left unattended"
    verify_prompt: "Does the cashier leave cash unattended and out of their control?"   # pipeline wraps it in a shared template
    why: "Наличные оставлены без контроля кассира, что может указывать на нарушение требования о сохранности ценностей."  # hedged wording
    citations:
      - document: "Правила осуществления кассовых операций и операций по инкассации ..., утв. постановлением Правления НБРК от 29.11.2019 № 231"
        clause: "п. __"            # may be empty: document-level citation (FAQ 31)
        quote: "<verbatim from the current adilet edition>"
        url: "https://adilet.zan.kz/rus/docs/..."
        edition_checked: "2026-10-17"
```

The loader refuses to start the service unless:
- every id is unique;
- every detectable rule has at least one query and at least one citation;
- every citation has a non-empty `quote` and a URL on `adilet.zan.kz` or `nationalbank.kz`;
- exactly one `other` rule exists.

### Pipeline: `/home/jovyan/model_train_amir/svc/pipeline.py`

Owned by the user. A stub runs when `PIPELINE=stub`.

```python
@dataclass
class Frame: t: float; jpeg: bytes
@dataclass
class Event:
    rule_id: str; t_start: float; t_end: float; timecode: float   # timecode == evidence[0].t
    ground_score: float                 # VideoJev max sigmoid over the rule's queries
    type_probs: dict[str, float]        # VideoJev choice softmax over rule ids + "none"
    verify_score: float | None          # InternVideo3 P(YES); None = not run
    kept: bool; reason: str             # verified | strong_ground | fallback_no_vlm | rejected
    observation: str                    # Russian, visible facts only, may be ""
    evidence: list[Frame]               # 1-3 when kept
    judge_rule_id: str | None           # Qwen3.5-9B pick; None = judge not run or failed
    justification: str                  # judge text, or the rule's fixed `why` on fallback
    type_source: str                    # judge | fallback
@dataclass
class PipelineResult:
    verdict: Literal["violation", "no_violation"]
    events: list[Event]                 # ALL candidates, including rejected ones (for the threshold sweep)
    timings: dict[str, float]; versions: dict[str, str]
def analyze(video_path: str, deadline: float, on_stage: Callable[[str], None]) -> PipelineResult
# internal wrappers the user can mock. VideoJev and InternVideo3 both read video files, so windows are cut to short clips first:
def ground(video_path: str, queries: list[str]) -> dict[str, list[float]]   # per-time-step probabilities; logic.segments() turns them into (start, end, score)
def choose(window_path: str, options: list[str]) -> dict[str, float]
def verify(window_path: str, rules: list[dict]) -> tuple[float | None, str, str]       # p_yes, type_id, observation
def judge(observation: str, evidence: list[Frame], candidates: list[dict]) -> tuple[str | None, str, float | None]  # rule_id, justification, confidence
```

The stub returns one event at 25% of the clip's length, with real frames, so the UI can be built against it.

### Job files: `/home/jovyan/model_train_amir/jobs/<id>/`

- `input.mov`
- `state.json`: `{"status","stage":"queued|decoding|grounding|verifying|writing","created_at","error"}`
- `result.json`: the exact body of the GET response
- `debug.json`: the full `PipelineResult` without JPEG bytes, also returned by `GET …?debug=1`

### GET response when done: `api.to_api(job_id, result, rules)`

It contains the fields from section 10 of the brief, then fields we add.

```json
{"job_id":"3f2a9c1e7b4d","status":"done","verdict":"violation","violations":[
 {"type":"<rules[id].name>","timecode_sec":44.2,
  "description":"<observation> <justification> Основание: <document>, <clause>.",
  "evidence_frames":[{"timestamp_sec":44.2,"image_base64":"/9j/..."}],
  "type_id":"cash_unattended","rule":{"document":"…","clause":"…","quote":"…","url":"…"},"confidence":0.87}]}
```

- The jury's scoring is automatic and may read only the fields in the brief's schema, so the citation goes inside `description` as well as in `rule`.
- Only kept events are listed, sorted by time.
- Times are rounded to 0.1 s.
- A clean video returns `"violations": []`.

### Labels: `/home/jovyan/model_train_amir/data/labels.csv`

Columns: `clip,has_violation,t_label,t_start,t_end,type_id,angle,source,note`.
- One row per event, so Х-4 gets two rows.
- `source` is `given`, `own` or `augmented`.
- Clip names are normalised to a Latin "X".

### Thresholds: `/home/jovyan/model_train_amir/thresholds.json`

`{"cand","verify","strong","fallback","top_k":6,"merge_sec":3,"anchor":"peak|onset","anchor_offset_sec"}`. The values come from the threshold sweep.

## 3. Workstreams for teammates

Everyone can start once these contracts are agreed. With more people, split a row.

| Workstream | Produces | Can start right away on | Connects through |
|---|---|---|---|
| **1. Rules and labels.** Research and violation types stay together, because the type list is what links a video to a rule. | `labels.csv`, `rules.yaml`, and the README sections on sources, types and "what we cannot check" | Watching the 40 clips; reading adilet | The loader validates the catalogue; the queries feed VideoJev; the visual criteria, `never_claim` lists and quotes feed the judge; the types feed the evaluation |
| **2. Service shell.** API, workers, UI, `run.sh`, deployment. | `api.py`, `static/index.html`, `run.sh`, the environments, the weights for all three models in `hf/`, the public URL | The stub pipeline; deploy the stub first so the evaluation can call the real URL | The user switches it to `PIPELINE=real` |
| **3. Evaluation and extra clips.** | `eval.py`; our own recordings from another angle, including violation types not in training; flipped, rotated and cropped copies made with ffmpeg | Running against the stub URL; it removes duplicates and normalises X/Х | `--sweep` writes `thresholds.json`; the numbers go into the README |
| **4. README and deck** (the deliverables in section 8 of the brief) | A neutral README in `svc/`; the deck stays on the laptop | The architecture and source outline | Final numbers from the evaluation |

**Sources for workstream 1:**
- **Main source:** НБРК Board Resolution No. 231 of 29.11.2019, the rules for cash operations and cash collection in second-tier banks.
- **Also check:** a cash-premises act from 2020, amended in 2025. It is mostly about how the premises are built.
- **Do not cite:** No. 58/2001 or No. 394/1999, which have been replaced, or V14Y0010204, which covers only НБРК's own units.
- **Beyond the training clips:** add rules that can be checked visually even if no training clip shows them.

**What the UI shows (workstream 2):**
- The two verdict phrases exactly as the brief words them.
- A stage indicator with elapsed time.
- A video player where clicking a timecode jumps there.
- For each violation: its evidence frames, the description, and the rule with its link.
- The line from section 7 of the brief saying this is a hint for a human reviewer.
- A field for the API key, saved in localStorage.

**What the evaluation reports (workstream 3):**
- Verdict accuracy and recall. Recall must be 100%.
- False positives on clean clips.
- Timecode error: median, maximum, and the share within 1 s and within 2 s.
- Type accuracy with the judge on and off, to show whether the judge earns its place.
- The longest time from POST to done, including a run with 5 videos sent at once.
- Whether every response matches the schema, and whether every cited rule exists in the catalogue.

**What the public URL needs (still open):**
- One HTTPS address pointing to `localhost:8000`, with the UI and API on the same address.
- A hostname that stays the same after the pod restarts, because the URL is submitted once.
- Uploads of at least 200 MB with a timeout of at least 120 s.
- The `X-API-Key` header passed through, with no login page in front.
- A neutral name.

**Order of work:**
1. Agree the contracts.
2. The labels and the first version of the rules.
3. The user runs the first baseline (models prompted, not trained) through the evaluation.
4. The real pipeline goes onto the deployed shell, and the evaluation runs end to end over the public URL.
5. The threshold sweep sets the thresholds, which are then frozen.
6. The clips from other angles and with new types: tune only the queries and prompts.
7. Freeze the code, then rehearse a restart: kill everything, run `run.sh`, and check that a pending job finishes and the URL is unchanged.
8. Fill in the final numbers and submit the URL and API key.

## 4. Deliberately not building
- No users, sessions, history, export or batch upload (FAQ 27). Authentication is one API key.
- No database server, Redis, Celery or message broker. The job folder is the state.
- No Docker or Kubernetes manifests. Rule matching is the judge call inside the pipeline plus a lookup, not a service with its own API.
- No detector, tracker, pose model or zone outlines, because they are tied to one camera angle.
- No search over legal texts. The judge sees only the quotes in the catalogue, and no model writes a citation.
- No third "insufficient data" verdict (FAQ 15).
- No frontend framework, no websockets, and no translation model.
- No fine-tuning before the baseline is measured. If fine-tuning happens later, it is kept only if the results on the other-angle clips hold.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Three models compete for GPU memory next to the training runs. | Fixed memory caps on both SGLang servers, started before VideoJev, and a test of all three together on Х-14. |
| InternVideo3 won't run under plain transformers on the pod. | Check it first, in the smoke test. If it fails, stop and ask the user whether Qwen3.5-9B should take over the check step on its existing server. |
| The judge writes a justification that isn't supported by the video. | Only candidate rule ids are allowed, it may use only the observation and frames, the `never_claim` list goes in its prompt, it has no say in keep or drop, and the fixed `why` text is the fallback. |
| 5 videos sent at once break the 5-minute limit. | 2 workers, the VideoJev lock, the 240 s fallback, and a test with 5 videos at once. |
| A pod restart changes the URL or loses jobs. | A fixed hostname, all state on disk, jobs resumed at startup, and the restart rehearsal. |
| Thresholds fit too closely to 40 clips from one angle. | Loosen by one step, test on the other-angle clips, and make no assumptions about positions on screen. |
| A citation is wrong or misapplied, which costs points under criterion 4. | Exact quotes from the current adilet edition, a loader that rejects rules without a quote or an approved URL, and hedged wording. |

## Research leads for the rules workstream

None of these is verified yet. The details on Resolution No. 231 come from search snippets, so the rules workstream must confirm the adilet page and its current edition before citing it.

- adilet: [V090005843_](https://www.adilet.zan.kz/rus/docs/V090005843_), [V14Y0010204](https://adilet.zan.kz/rus/docs/V14Y0010204)
- cdb.kz: [Rules No. 231](https://cdb.kz/sistema/pravovaya-baza/ob-utverzhdenii-pravil-osushchestvleniya-kassovykh-operatsiy-i-operatsiy-po-inkassatsii-banknot-mone/)
- prg.kz: [1022942](https://prg.kz/m/amp/document/1022942), [1016205](https://prg.kz/m/amp/document/1016205), [38551339](https://prg.kz/m/amp/document/38551339/1/11000), [1020000](https://prg.kz/m/amp/document/1020000/)
- zakon.uchet.kz: [V010001482_](https://zakon.uchet.kz/rus/docs/V010001482_), [V090005843_](https://zakon.uchet.kz/rus/docs/V090005843_), [V1200007985](https://zakon.uchet.kz/rus/docs/V1200007985), [V990001017_](https://zakon.uchet.kz/rus/docs/V990001017_), [V050004048_](https://zakon.uchet.kz/rus/docs/V050004048_), [V2400035552](https://zakon.uchet.kz/rus/docs/V2400035552)
- nationalbank.kz: [file 107276](https://nationalbank.kz/file/download/107276)
- News: [bizmedia.kz](https://bizmedia.kz/2025-01-20-banki-kazahstana-dolzhny-v-techenii-dvuh-let-obnovit-kassovye-pomeshheniya/), [informburo.kz](https://informburo.kz/novosti/banki-i-obmenniki-kazaxstana-pereoboruduyut-kassy-i-seify-po-novym-pravilam-bezopasnosti), [zakon.kz](https://zakon.kz/pravo/6456224-v-kazakhstane-usilyat-trebovaniya-k-bezopasnosti-pomeshcheniy-bankov-obmennikov-i-pochty.html)
