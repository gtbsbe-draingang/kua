# Research: improving detection, classification and rule grounding

Status: proposals from the 2026-10-17 discussion. Nothing here is implemented yet. The new `rules.yaml` fields in section 2 change the contract in design.md §2, so the team agrees on them first and design.md is updated with them.

Inputs: design.md (the pipeline), regulatory_sources.md (acts, the 13-type issue register, what video cannot check), rag.md (the regulation index), and the case brief: §4 item 7, §5, §7, §9 criteria 1, 3 and 4, §10, and FAQ 9, 16, 30 and 31.

## 0. The short version

1. **Ask the models for visible facts, not verdicts.** Write each rule as a checklist of yes/no facts plus a condition, and let code apply the condition. Models judge by everyday norms, and at a cash desk that misses violations (section 1).
2. **Make the catalogue complete by construction.** Sweep all 253 indexed clauses, put each one in a bucket, and give the judge every card, not the top 3 (section 3).
3. **Separate direct clauses from indirect ones, and write the exceptions in.** Most grounding risk comes from a real clause applied wrongly, not from an invented one (section 4).
4. **Diagnose before tuning.** For each missed clip, find out whether perception or reasoning failed (section 2.7).
5. **Fix the output format.** The jury's schema has no rule field, so the citation goes in `description` in one fixed form (section 4, row 7).

## 1. The problem seen in testing

The model does not flag acts that are violations at a bank cash desk, because elsewhere they are ordinary actions. Four causes:

- **It does not know the bank's procedure.** Unless the prompt states that cash is counted in front of the client, an uncounted handover looks normal.
- **Many violations are omissions** ("handed over without counting"). Models describe what happened, so a missing step is not mentioned unless they are asked about it.
- **A free description leaves out the deciding detail.** "The cashier gets up and leaves" is a violation only if a client is still at the window or cash is still on the counter.
- **Models lean towards "no".** They avoid accusing people, and "violation" is a loaded word.

## 2. Facts, not verdicts

The legal knowledge, exceptions included, moves out of the model's common sense and into curated data. The model only answers what it can see.

### 2.1 Each rule as fact questions plus a condition

New `rules.yaml` fields (a proposal):

```yaml
- id: issue_not_counted
  strength: direct                  # direct | indirect (section 4, row 2)
  facts:
    hands_cash:    "Does the cashier hand banknotes to the client?"
    counted:       "Before handing them over, does the cashier count the banknotes (by hand or with a counting machine) where the client can see?"
    sealed_bundle: "Is the cash handed over a bundle still in its bank band or packaging?"
  violation_if: {all: [hands_cash], none: [counted, sealed_bundle]}   # No. 231 p. 31 sub. 4); p. 32 exception

- id: leaves_workplace
  strength: direct
  facts:
    leaves:          "Does the cashier get up and leave the workplace?"
    client_or_cash:  "At that moment, is a client still at the window, or are cash or documents still on the counter?"
  violation_if: {all: [leaves, client_or_cash]}                        # No. 231 p. 22
```

`violation_if` is structured data (`all` / `none` lists), not a string to `eval`. The loader checks that every name in it is a defined fact.

Draft facts for the strongest types. None of these has been checked against a clip yet.

| Type (register #) | `all` (must be seen) | `none` (clears the cashier) | Clause |
|---|---|---|---|
| `leaves_workplace` (1) | leaves; client or cash still present | | No. 231 p. 22 |
| `receipt_not_counted` (2) | takes cash from the client | counts it while the client is still at the window | No. 231 p. 20; p. 19 sub. 2) |
| `issue_not_counted` (3) | hands cash to the client | counted where the client can see; sealed bank bundle | No. 231 p. 31 sub. 4); exception p. 32 |
| `cash_left_exposed` (4) | cash open on the desk, outside a drawer or safe; cashier away or serving another client | | No. 231 p. 20, part 2 |
| `phone_or_personal_items` (5) | smartphone held or used at the cash desk | | No. 231 p. 12 (see section 4, row 3) |
| `collection_bag_not_checked` (8) | takes a bag from a cash collector | looks over the bag and its seal before opening or storing it | No. 231 p. 24 sub. 3)–4) |

### 2.2 Targeted yes/no questions

- InternVideo3 is asked the fact questions of the candidate rules, not "describe the video" and not "is this a violation".
- Every omission gets its own question ("does the cashier count…?").
- The word "violation" does not appear in these prompts.
- Questions name roles and actions (кассир, клиент, инкассатор), never positions on screen, so they carry over to another camera angle.

### 2.3 Decide in code, biased towards recall

- Read P(YES) for each fact from the logprobs, as design.md step 4 already does for `verify_score`.
- A fact in `all` counts as present when P(YES) ≥ `t_all`, a low threshold.
- A fact in `none` clears the cashier only when P(YES) ≥ `t_none`, a high threshold. The required step has to be seen clearly. If the model is unsure whether the cashier counted, the event is kept, and the evidence frames let a reviewer dismiss it.
- Tune `t_all` and `t_none` with the existing threshold sweep: every one of the 21 violation clips must be caught (FAQ 16), then take the fewest false alarms, then loosen one step.

### 2.4 The bank's procedure in the system prompt

A short preamble resets what "normal" means:

```
You review CCTV from a bank cash desk. Bank procedure is stricter than everyday behaviour:
- cash is counted in front of the client, both when received and when handed over;
- the cashier does not leave while serving a client;
- cash not in use is kept in a locked drawer, never left on the desk;
- no smartphones at the cash desk; personal bags are kept outside the cash area;
- only the cashier and cash collectors are on the cashier's side.
Answer only about what is visible. Describe people by role (cashier, client, cash collector).
```

### 2.5 Examples from the training clips, only if misses remain

- Per type: one clip with the violation and one without, either as short text descriptions or as frames.
- Training clips only. Test videos are never used (brief §10).
- Check the result on the other-angle clips (design.md §3, workstream 3) before keeping it.

### 2.6 The judge does not decide keep or drop

This is already in design.md step 6, and it stays that way: Qwen3.5-9B picks the type and writes the wording, but cannot remove an event. If any current test setup lets an LLM decide yes or no on its own, that is where recall leaks.

### 2.7 Diagnose each miss first

For every missed violation clip, read the model's observation:

| What the observation says | Failure | Fix |
|---|---|---|
| The deciding fact is missing (for example, nothing about counting) | Perception | Targeted question, wider window, higher fps |
| The fact is there, but the answer was still "no" | Reasoning | Facts plus condition (2.1–2.3), procedure preamble (2.4) |

## 3. A complete, closed catalogue

Every possible video situation cannot be listed, but every rule can: the four indexed acts have 253 clauses.

1. **Sweep.** Go through all 253 clauses (`data/regs/parsed/`) and put each one in a bucket. The result is a coverage table, so "all cases" is shown rather than assumed.
   - **Visible** on a desk camera → becomes a card.
   - **Needs data the video does not show** (amounts, documents, postings) → goes on the limitations list and is never reported (brief §7).
   - **Not applicable** (definitions, audits, other parties) → skipped.
2. **Cards** for the visible bucket, with facts, condition, strength, `never_claim`, hedged `why`, and citations. regulatory_sources.md §2 has 13 types as a starting point.
3. **`other` in variants.** One document-level citation per area (cash operations No. 231, premises No. 14, exchange No. 49) instead of always No. 231. FAQ 31 accepts a citation without a clause, but brief §4 item 7 asks for «пункт/раздел», so `other` should be rare.
4. **Give the judge every card.** At 15–20 cards of a few hundred tokens each, the catalogue fits in the prompt (roughly 5–10k tokens). This removes the ceiling of "top 3 from VideoJev plus InternVideo3's type". Searching over cards would only be worth it with hundreds of them.
5. **The regulation index stays a build-time tool** (design.md §4, rag.md §0):
   - Use it to find clauses and exceptions for each card.
   - `regs verify` checks the quotes.
   - Cross-check: search with each card's facts. The cited clauses should come back near the top; if they don't, the citation is wrong or a better clause exists.

## 4. Rule grounding: weak points and fixes

"No model writes a citation" protects against invented references. It does not protect against a real clause being applied wrongly, and criterion 4 scores both «точность источника» and «трактовки».

| # | Weak point | Why it costs points | Fix |
|---|---|---|---|
| 1 | **A wrong type gives a real but wrong citation.** The citation is looked up by type. | Lost on classification (15) and grounding (15) at once. `regs verify` passes, because the quote is genuine. | Facts and condition (section 2). All cards go to the judge. At low confidence, use the document-level `other` rather than a clause that may be wrong. |
| 2 | **Indirect clauses are cited as prohibitions.** No. 14 p. 16 sub. 1) says the room *has* a lockable door, not that it stays shut. No. 231 p. 80 leaves keys to internal documents. No. 14 p. 21 is about the loading area layout. No. 231 p. 19: the check can be done by eye. No. 49 p. 41 applies to **exchange-point** cash rooms only. | «Неподтверждённое юридическое утверждение» | A `strength` field. Fixed wording for indirect rules, «может не соответствовать требованию…», never «нарушение п. …». Consider moving types 7, 12 and 13 to limitations. |
| 3 | **Exceptions in the same act.** p. 32 allows issuing a sealed bundle uncounted; p. 42 exempts serving a client; p. 75 allows entry by order of the head of the bank; p. 12 allows phones *without* a photo or video function. | The citation contradicts the flag. | Exceptions become `none` facts. For phones: flag when a camera or recording is visible, or say "a communication device is visible at the cash desk; p. 12 allows only devices without a photo/video function". |
| 4 | **The catalogue has not been matched to the clips.** The labels have no type (FAQ 2), and the 21 violation clips have not been typed. | Staged violations we have no card for fall to `other`, which has no clause. | Type every violation clip in `labels.csv`, add the missing types, and use `other` variants (section 3). |
| 5 | **`regs verify` checks the quote, not the clause.** `svc/regs/search.py:192` only checks that the quote is somewhere in the act named by the URL. | A real quote from p. 20 labelled «п. 22», a wrong `#zNN` anchor, or a two-word quote all pass. | Require the quote to be in the chunk whose clause label and anchor match the citation. Set a minimum quote length. Reject URLs whose act is not indexed. |
| 6 | **Nothing checks the justification.** Hedged wording and `never_claim` are only asked for in the prompt. InternVideo3's observation can also state things the video does not show. | «кассир нарушил п. 22» or an amount is exactly what criterion 4 penalises. | A check after the judge: reject categorical words (нарушил, виновен, точно), any `never_claim` term and any amount, then fall back to the fixed `why`. |
| 7 | **The jury's API has no rule field** (brief §10: `type`, `timecode_sec`, `description`, `evidence_frames`). Scoring is automatic (FAQ 18). | A rule that is not in `description` may not count. A long act title can bury the clause. | One fixed ending: «Основание: Правила кассовых операций… (пост. Правления НБРК от 29.11.2019 № 231), п. 22.» Keep the text before it short. Use plain Russian `type` names. |
| 8 | **One event, one rule.** Leaving the desk with cash on it breaks No. 231 p. 22 and p. 20 part 2. | The second rule is lost. | Several citations on a card for combined situations, or two events at the same timecode with different types. |
| 9 | **Edition not checked.** The texts come from the zakon.uchet.kz mirror, and `edition_checked` is empty. No. 14 clauses 11, 13 sub. 3, 14 and 16 are suspended until 01.01.2027; only their interim wording is in force. | A shifted clause number or wording that is not in force. | Before submission, open each cited adilet link, confirm the wording and number, and fill in the date. Quote the interim wording. |
| 10 | **The chain is not built.** `rules.yaml`, `labels.csv`, the loader, the judge and the description format exist only in design.md. | None of rows 1–9 can be tested yet. | Write `rules.yaml` from the register first, then run `regs verify` on it. |

## 5. Regulation index accuracy (build-time tool)

Measured on the 18 holdout queries, which are never used for tuning (rag.md §6):

| Ranking | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| Hybrid RRF + glossary | 0.33 | 0.56 | 0.67 | 0.44 |
| **LambdaMART on cosine + BM25 (default)** | **0.39** | **0.56** | **0.78** | **0.52** |
| bge-reranker-v2-m3 (optional) | 0.44 | 0.78 | 0.78 | 0.58 |

LambdaMART reached 0.846 hit@1 out-of-fold on the dev set; the holdout is the honest number. The correct clause is always among the candidates (pool recall 1.0), so the limit is the ranking signals, not retrieval.

Next steps, in order (rag.md §9):
1. Label real InternVideo3 observations from the training clips as new dev queries. These are the queries that matter.
2. Add the cross-encoder score as a third LambdaMART feature.
3. Add glossary terms for the remaining vocabulary gaps, tuned on dev only.
4. Extend `regs verify` (section 4, row 5).

## 6. Open questions

- **Licences.** design.md gives Apache-2.0 for Qwen3.5-9B but no licence for VideoJev-4B or InternVideo3-8B. Commercial-use licences are a hard requirement (FAQ 24).
- **Judge candidates.** Keep the top 3 or give the judge every card (section 3, item 4)?
- **A stage for the judge.** The stages in `state.json` have no step for the judge. The UI would show "verifying" or "writing" while it runs.
- **`other` queries.** Its example query "cash left unattended" overlaps the example `cash_unattended` rule. Generic queries should not repeat a real rule.
- **Current test setup.** Which model made the yes/no decision in the tests that missed violations: InternVideo3, the judge, or something else? Section 2.6 depends on the answer.

## 7. Proposed order of work

1. Type the 21 violation clips in `labels.csv`, and diagnose the current misses (2.7).
2. Do the clause sweep and coverage table (section 3).
3. Agree on the contract change (`facts`, `violation_if`, `strength`), then write cards for types 1–5 and 8.
4. Extend `regs verify` and add the justification check (section 4, rows 5–6).
5. Ask the fact questions in the verify step, and run the threshold sweep with `t_all` and `t_none` (2.3).
6. Fix the description format with «Основание: …» (section 4, row 7).
7. Add examples only if clips are still missed, then test on the other-angle clips (2.5).
8. Before submission: check every cited clause on adilet and fill in `edition_checked` (section 4, row 9).
