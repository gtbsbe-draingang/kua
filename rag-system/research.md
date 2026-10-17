# Research: retrieving the violated clause from a per-second action log

Status: measured locally on 2026-10-17. Companion to `regs.md`, which describes the index this builds on. Experiment files are in `research/`.

## 0. Summary

Searching with the whole log as one query does not work: on 12 test logs the current search put the right clause first for 2 of them (top-1 0.17).

What works is splitting the job in two:
1. **Recall.** Turn one log into many small queries (overlapping windows of lines, plus a few "suspicious moments" written by an LLM), search each with the existing LambdaMART search, and combine the lists. The right clause was in the top 10 for all 12 logs.
2. **Precision.** Give an LLM the log and the full text of the top 20 clauses, and let it pick the clause that was broken, by id. Top-1 rose to 0.67 and top-3 to 0.92.

The 12 logs were written by Claude, not produced by the VLM, so read the numbers as a direction, not as the expected accuracy.

## 1. Setup

- **Test set** (`research/timelines.yaml`): 12 logs in the per-second, camera-description style the VLM produces, 10–22 lines each. Each hides one violation among routine actions, described as plain behaviour and never named as a breach:

  | Case | Violation | Expected clause(s) |
  |---|---|---|
  | leave_workplace | cashier leaves the booth while the client waits | No. 231 п. 22 |
  | accept_no_count | accepted cash goes into the drawer uncounted | No. 231 п. 20, 19 |
  | payout_no_count | cash paid out without recounting in front of the client | No. 231 п. 31 |
  | phone_in_cash_room | cashier photographs banknotes with a smartphone | No. 231 п. 12 |
  | stranger_in_cash_room | a person without a badge is let into the exchange booth | No. 49 п. 41; No. 14 п. 8, 16; No. 231 п. 75 |
  | bag_seal_unchecked | collection bag seal cut without checking it | No. 231 п. 24, 107 |
  | no_detector | banknotes counted by hand, detector switched off | No. 231 п. 19, 36, 57, 9, 41 |
  | bundling_alone | counting and packing with no controlling employee | No. 231 п. 42 |
  | keys_left | keys left hanging in the drawer-safe lock | No. 231 п. 80, 76 |
  | prev_client_money_on_desk | previous client's cash left on the desk | No. 231 п. 20 |
  | no_identification | cash accepted without asking for ID | No. 231 п. 19 |
  | door_open | cash booth door propped open | No. 14 п. 16, 8 |

  Expected clauses follow the labels already used in `regs/eval_queries.yaml`. A hit counts if the act and clause number match any of them.
- **Stack.** The live Milvus index (513 chunks), bge-m3 through Ollama, the LambdaMART models saved at 18:10 and 18:12, and `claude-haiku-5-5` as the LLM.
- **Isolation.** It ran in a separate container against a copy of `data/regs` and `rag/regs`, because another session was retraining LambdaMART and restarting the `rag` container at the same time. Nothing in the stack or data was changed.
- **Noise.** Haiku 5.5 rejects the `temperature` parameter, so answers can differ between runs; the script caches every LLM answer so a rerun gives the same numbers. With 12 cases, one case is 0.083 of top-1.

## 2. Why the current search fails on long logs

1. **The violation is averaged away.** A whole log becomes one dense vector. In a 20-line routine log, appending the violating sentence left the vector at cosine 0.97 to the routine-only one, so the violation is a few percent of the signal. The clauses that are generally close to any cash-desk log then win:
   - No. 231 п. 19, 20 and 31 (accepting and paying out cash);
   - No. 49 п. 33-2, about exchange-office opening hours, which reached the top 5 for about 7 of the 12 logs. It is a "hub": a long, generic chunk that is close to many queries.
2. **Long logs are silently truncated.** bge-m3 through Ollama's `/v1/embeddings` ignores everything after roughly 1,500–1,800 Russian words (about 4k tokens), with HTTP 200 and no warning. Measured by appending a probe sentence to padding of growing length: up to 150 lines (~1,350 words) the probe changed the vector; from 200 lines (~1,800 words) the vector was identical to the padding alone. In a log longer than a few minutes, the end is never searched.
3. **Most violations are omissions.** "Did not recount", "did not ask for ID", "did not check the seal": nothing in the log names the missing step, so neither cosine nor BM25 can match it. Only a step that reads the log next to the clause text can notice an absence.
4. **The ranker was tuned on one-sentence queries.** LambdaMART was trained on queries like "кассир выдает деньги клиенту без пересчета". Its relative features (`cos_gap`, `bm25_rel`) behave differently when a 300-word query shares terms with nearly every clause in BM25.
5. **Timestamps are noise.** Removing them alone raised top-1 from 0.17 to 0.33.

## 3. Methods tested

| Id | Method |
|---|---|
| B0 | Whole log as one query, LambdaMART (current default) |
| B1 | Whole log + LLM rewrite (current `/regs` path) |
| B2 | Whole log with timestamps removed |
| S1 / S2 | Windows of 6 lines, step 3, timestamps removed; each window searched, lists combined by best score (S1) or by RRF with k = 10 (S2). No LLM. |
| L1 | LLM compresses the log into 2–6 neutral episodes; each searched; RRF |
| L2 | LLM lists up to 4 "suspicious moments" (what was seen, which mandatory step was missing), without naming clauses; each searched; RRF |
| D1 | "Violation cards" only: the LLM wrote, for every clause, short sentences on how breaking it would look on camera; L2's sentences are matched against those cards by cosine |
| D2 | L2 + D1, RRF |
| D3 | L2 + D1 + windows, RRF |
| R1 | LLM judge over the top 20 of D2 (clause text cut at 900 characters) |
| R2 | LLM judge over the top 20 of B0 |
| R3 | As R1, full clause text |
| R4 | LLM judge over the top 30 of D3, full clause text |
| R5 | No search: the LLM reads all 247 in-force clauses (600 characters each) and picks |

The judge is given the log and a numbered list of clauses, and answers with up to 5 list numbers, most confident first. The rest of the pool keeps its retrieval order behind them.

## 4. Results

| Method | top-1 | top-3 | top-5 | top-10 | MRR |
|---|---|---|---|---|---|
| B0 whole log (current) | 0.17 | 0.58 | 0.67 | 0.83 | 0.40 |
| B1 whole log + rewrite (current /regs) | 0.42 | 0.67 | 0.75 | 0.83 | 0.55 |
| B2 whole log, no timestamps | 0.33 | 0.58 | 0.67 | 0.83 | 0.50 |
| S1 windows, best score | 0.33 | 0.50 | 0.75 | 0.75 | 0.47 |
| S2 windows, RRF | 0.42 | 0.67 | 0.75 | 0.83 | 0.54 |
| L1 LLM episodes | 0.25 | 0.50 | 0.67 | 0.75 | 0.43 |
| L2 LLM suspicious moments | 0.33 | 0.50 | 0.75 | 0.92 | 0.49 |
| D1 violation cards only | 0.17 | 0.42 | 0.58 | 0.67 | 0.34 |
| D2 L2 + cards | 0.25 | 0.58 | 0.67 | 0.92 | 0.45 |
| D3 L2 + cards + windows | 0.33 | 0.67 | 0.83 | **1.00** | 0.55 |
| **R1 judge over D2 top 20** | **0.67** | **0.92** | 0.92 | 1.00 | **0.77** |
| R2 judge over B0 top 20 | 0.42 | 0.58 | 0.67 | 0.75 | 0.54 |
| R3 judge over D2 top 20, full text | 0.67 | 0.83 | 0.83 | 1.00 | 0.75 |
| R4 judge over D3 top 30, full text | 0.50 | 0.75 | 1.00 | 1.00 | 0.67 |
| R5 LLM over the whole catalogue | 0.17 | 0.17 | 0.17 | 0.25 | 0.20 |

Per-case ranks and every LLM output are in `research/run2.txt`.

**Reading this.**
- **No single query finds every clause; a combination does.** Each fan-out source misses some cases, but D3 has the right clause in the top 10 for all 12.
- **Picking needs a model that reads the clause.** Retrieval alone stays at 0.17–0.42 top-1; the judge lifts it to 0.50–0.67. Over the current pool (R2) the judge only reaches 0.42, because that pool misses the right clause for 2 logs and ranks it low for others.
- **R1, R3 and R4 differ by one or two cases**, which is within the noise of 12 logs. Their common result, a judge over a fan-out pool, is the finding; which pool size is best needs more cases.
- **Putting the whole act in the prompt fails** (R5): 247 clauses is too long a list for Haiku to pick from reliably, and it is the most expensive variant.
- **Violation cards alone are weak** (D1), but they add the right clause to the pool in cases the other sources miss (phone, keys).

**Errors that remain after the judge.**
- payout_no_count: п. 19 (accepting cash) is chosen over п. 31 (paying out). The two clauses are near-identical lists of duties; the difference is the type of operation.
- accept_no_count: п. 35 (the client recounts) beats п. 20 (the cashier counts in the client's presence), the same near-miss noted in `regs.md` section 6.
- phone_in_cash_room: п. 69 and п. 33 beat п. 12.

**The suspicious-moments list is often wrong.** For example, it flagged "cash put in a desk drawer instead of storage", while п. 20 explicitly lists desk drawers as allowed storage. Such sentences are fine as search queries, but must never be shown as evidence or justification.

## 5. Recommended design

```
log ─► clean up: remove timestamps, merge repeated lines
     ├─► windows of 6 lines, step 3                 ─┐
     ├─► LLM: 2–4 suspicious moments                ─┼─► LambdaMART search per query ─► RRF (k = 10) ─► top 20–30 clauses
     └─► each query also matched to violation cards ─┘
                                                      ─► LLM judge: log + clause texts (with subclauses) ─► clause ids + timestamps
```

- **The judge returns only ids.** The citation text and URL are then looked up from the chunk, which keeps the `regs.md` rule that no model writes a citation.
- **Ask the judge for timestamps** of the moment it relies on, so a person can check the video.
- **Allow "none".** The judge prompt already accepts НЕТ; combine it with the existing `min_score` threshold for logs with no violation.

### 5.1 Index changes, most useful first

1. **`observable` and `operation` fields per clause.**
   - `observable`: the card generator judged 165 of 247 in-force clauses as visible on video. The other ~82 (definitions, deadlines, reporting, document flow) should leave the pool for log queries; this also removes hubs such as No. 49 п. 33-2.
   - `operation`: receipt, payout, cash collection, counting and packing, storage, keys, premises access, personal items. Tag each log window with an operation (one cheap LLM call, or rules on the window's verbs) and boost clauses with the same tag. This targets the main remaining error, п. 31 against п. 19.
2. **Violation cards as extra index rows** with `granularity="card"`, folded back into their clause in `_fold` the way subclause chunks are now. They are for finding clauses only; a card is never quoted.
   - Generation quality was the weak point: for 93 clauses Haiku packed three sentences into one string, and for 10 it dropped the act code (`"120"` instead of `"V1900019680:120"`).
   - Ask for one sentence per array item, include "did not do X" wording, and have a person review the cards of the ~60 clauses that matter most for the rules catalogue.
3. **Give the judge subclause text** (for example п. 31, пп. 4)), not the opening of a long clause cut at a character limit.
4. **Untested idea: bge-m3 multi-vector (ColBERT) scores.** Matching token by token instead of one vector per text should dilute a long query less. Ollama does not expose this mode; it needs `BGEM3FlagModel` from FlagEmbedding, used to rescore the ~30 candidates in our process.

### 5.2 Training and evaluation

- **Collect real long logs.** Run InternVideo3 on the training clips and label each log with the violated clause and the time span. These, not hand-written logs, should drive every choice.
- **Add multi-query features to LambdaMART.** With many sub-queries per log, new features become available: how many sub-queries returned the clause, its best and mean score across them, its best rank among the windows, and whether a card matched. This extends `ltr.py` without changing its training loop.
- **Keep the 12 logs here as a smoke test** only; do not tune on them.

## 6. Bugs found on the way

1. **Concurrent searches crash.** `lexical.py` keeps the Snowball stemmers in module globals, and they are not thread-safe. Two searches at once failed with `IndexError: string index out of range` inside `snowballstemmer`. FastAPI runs sync endpoints in a thread pool, so `/api/regs/search` can hit this under load. Fix: one stemmer per thread (`threading.local`) or a lock around `stemWord`.
2. **Ollama truncates long input silently** (section 2, point 2). Clauses are not affected (the longest is about 400 tokens), only long queries. Set `OLLAMA_CONTEXT_LENGTH=8192` on the Ollama server; the OpenAI-compatible endpoint ignores a per-request `num_ctx`. Better, never embed a whole log (section 5).
3. **Haiku 5.5 rejects `temperature`** (HTTP 400, "`temperature` is deprecated for this model"). `rewrite.py` does not pass it, so nothing breaks today, but answers cannot be made deterministic that way; cache them instead.

## 7. Cost per log

- 1 LLM call for the suspicious moments (short output).
- About 5–15 searches (windows plus moments), about 0.3 s each on the M1.
- 1 judge call with about 10k input tokens (20 clauses).
- The violation cards are built once with the index: about 25 LLM calls for 247 clauses, about 500 card sentences to embed.

## 8. Next steps, in order

1. Fix the stemmer race and set the Ollama context length.
2. Add `observable` and `operation` to the chunks.
3. Implement the fan-out search and the judge as a `log` mode in `search.py`.
4. Collect and label real long InternVideo3 logs.
5. Generate the violation cards with human review and index them.
6. Retrain LambdaMART with the multi-query features.

## 9. Files

| File | What it is |
|---|---|
| `research/exp.py` | The experiment. Run inside the `rag` image with `EXP_DIR` pointing at this folder. It caches LLM answers in `llm_cache.json` next to it. |
| `research/timelines.yaml` | The 12 test logs and their expected clauses |
| `research/cards.json` | Violation cards generated by Haiku 5.5, keyed by chunk id |
| `research/run2.txt` | Full output: tables, per-case ranks, the LLM's episodes and suspicious moments |

## Sources

- [Exploiting LLMs' Reasoning Capability to Infer Implicit Concepts in Legal Information Retrieval](https://arxiv.org/abs/2410.12154v1): LLM query reformulation for statute retrieval from case facts (COLIEE).
- [NOWJ@COLIEE 2025](https://www.arxiv.org/pdf/2509.08025) and [NOWJ@COLIEE 2026](https://arxiv.org/pdf/2607.16603): retrieve, rerank, then LLM reasoning for statute retrieval.
- [Follow the Rules: Reasoning for Video Anomaly Detection with LLMs (AnomalyRuler)](https://arxiv.org/pdf/2407.10299): rules derived by an LLM, then checked against video descriptions.
- [SmartHome-Bench](https://arxiv.org/pdf/2506.12992): rule-based reasoning over video with multimodal LLMs.
- [Ollama context window truncation](https://www.betterclaw.io/blog/ollama-context-window-truncates-agent-config) and [an embedding pipeline fix for Ollama's num_ctx](https://git.johnwilger.com/jwilger/auto_review/pulls/21): silent truncation and how to raise the limit.
