# Regulatory sources and visually checkable issues

This covers section 6 of the brief: the public acts we use, the rules in them that video can check, and how each rule links to something visible. It is the input for `rules.yaml` (design.md, section 2) and for the regulation index (`rag.md`).

**How the texts were checked (2026-10-17).**
- Every quote below is copied from the indexed corpus, not retyped, and `python -m regs verify` can confirm them.
- adilet.zan.kz is now a JavaScript app whose document API needs a key. So the HTML was downloaded from zakon.uchet.kz, which mirrors adilet under the same document codes and keeps the same clause anchors (`#zNN`). Every link below points to adilet.
- The mirrored texts include amendments up to the dates in the table. Before submission, open each adilet link and confirm the clause reads the same; that is what `edition_checked` in `rules.yaml` records.
- Only acts of the Republic of Kazakhstan are used (FAQ 30).

## 1. Acts in the index

| Act | adilet | Text includes amendments up to | What it gives us |
|---|---|---|---|
| **Rules for cash operations and cash collection in second-tier banks**, NBRK Board Resolution No. 231 of 29.11.2019 | [V1900019680](https://adilet.zan.kz/rus/docs/V1900019680) | No. 78 of 30.06.2026 | The main source. How a cashier accepts, counts, issues and stores cash, and how cash collectors hand over bags. All 133 clauses are indexed. |
| **Rules for the security and layout of bank premises**, NBRK Board Resolution No. 14 of 24.02.2020 | [V2000020075](https://adilet.zan.kz/rus/docs/V2000020075) | No. 88 of 24.12.2024 | The cash unit: isolation, a lockable door per cash room, video cameras, collection areas hidden from third parties. No. 231 p. 5 points here. Clauses 11, 13 sub. 3, 14 and 16 are **suspended until 01.01.2027**, with interim wording that applies meanwhile. The index stores that wording and marks the clause `suspended`. |
| **Rules for determining the payability of banknotes and coins**, NBRK Board Resolution No. 230 of 29.11.2017 | [V1700016120](https://adilet.zan.kz/rus/docs/V1700016120) | No. 78 of 30.06.2026 | What "payable" and "genuine" mean. No. 231 pp. 41 and 57 point here. |
| **Rules for exchange operations with foreign cash**, NBRK Board Resolution No. 49 of 04.04.2019 | [V1900018545](https://adilet.zan.kz/rus/docs/V1900018545) | No. 78 of 30.06.2026 | Exchange desks: no outsiders in the cash room, and outside money kept elsewhere (p. 41). No. 231 p. 58 points here. Only chapters 1, 5 and 6 are indexed. |

**Looked at and left out:**
- NBRK No. 58/2001 and No. 394/1999: replaced, as design.md already notes. No. 58 lost force when No. 231 took effect on 01.01.2020.
- V14Y0010204: covers only NBRK's own units.
- The Law "On banks and banking activity": no rule about what can be seen at a cash desk. No. 231 cites it only as its legal basis.
- Anti-terrorism protection requirements for vulnerable sites (Government Resolution No. 305 of 06.05.2021): might cover access control, but we have not found or checked a version for banks. It is a candidate for later.

## 2. Issue register: what video can show, and which clause it relates to

**How to read the table.**
- **Strength:**
  - *direct* means the clause names the action.
  - *indirect* means the clause sets a requirement that the visible fact may break. Word these as "may indicate" and keep them at lower confidence.
- **Never claim** lists what the video cannot show (section 7 of the brief). These go into `never_claim` in `rules.yaml`.
- **Ids** are suggestions for `rules.yaml`. No training clip has been checked against any type yet.

| # | Suggested id | What is visible | Clause (verbatim, key part) | Strength | Never claim |
|---|---|---|---|---|---|
| 1 | `leaves_workplace` | The cashier gets up and leaves the workplace while a client is still at the window or cash/documents are still on the counter. | No. 231 [p. 22](https://adilet.zan.kz/rus/docs/V1900019680#z114): «Кассовому работнику не допускается покидать рабочее место до окончания обслуживания клиента.» | direct | why they left; whether the operation was finished in the system |
| 2 | `receipt_not_counted` | The cashier takes the client's cash and puts it away without counting it, or counts it after the client has left. | No. 231 [p. 20](https://adilet.zan.kz/rus/docs/V1900019680#z111): «Наличные деньги принимаются и пересчитываются кассовым работником в присутствии клиента.»; [p. 19 sub. 2)](https://adilet.zan.kz/rus/docs/V1900019680#z104): «принимает банкноты с полистным пересчетом … с проверкой на подлинность и платежность» | direct | the amount; whether the count matched the document |
| 3 | `issue_not_counted` | The cashier hands cash to the client without counting it in front of them. | No. 231 [p. 31 sub. 4)](https://adilet.zan.kz/rus/docs/V1900019680#z144): «пересчитывает подготовленные к выдаче наличные деньги в присутствии клиента, выдает клиенту наличные деньги» | direct, **with an exception** | that a sealed bundle was opened. No. 231 p. 32 allows issuing full or partial **bundles with undamaged packaging** "by the amount on the packaging, without counting every note". Do not flag issuing an intact bank bundle. |
| 4 | `cash_left_exposed` | Cash from earlier operations lies open on the desk, outside the locking drawer or safe, while another client is served or the desk is empty. | No. 231 [p. 20, part 2](https://adilet.zan.kz/rus/docs/V1900019680#z111): «На рабочем месте кассового работника находятся наличные деньги клиента. Все ранее принятые наличные деньги хранятся в специальных устройствах … закрывающихся на ключ» | direct | whose money it is, unless that follows from the sequence of events; the amount |
| 5 | `phone_or_personal_items` | The cashier uses a smartphone at the cash desk, or a personal bag or wallet is at the workplace. | No. 231 [p. 12](https://adilet.zan.kz/rus/docs/V1900019680#z81): «Хранение личных вещей, наличных денег и ценностей работников банка осуществляется вне кассового узла … Допускается внос … средств связи, не имеющих функции фото и видео фиксации.» | direct for a visible smartphone, indirect for bags | that the device can take photos or video, unless a camera is visible; who owns the item |
| 6 | `outsider_in_cash_room` | Someone who is neither the cashier nor a cash collector enters the cashier's side of the counter or the cash booth. | No. 49 [p. 41](https://adilet.zan.kz/rus/docs/V1900018545#z138): «В помещениях операционных касс обменных пунктов не допускается присутствие посторонних лиц.» (exchange desks); No. 14 [p. 8](https://adilet.zan.kz/rus/docs/V2000020075#z49): «Кассовый узел подлежит изолированию … одним закрывающимся служебным входом.»; No. 231 [p. 75](https://adilet.zan.kz/rus/docs/V1900019680#z221) (vault or safe room) | direct for exchange desks, indirect for bank cash rooms | **who the person is** (FAQ 29 bans identification) and whether they are authorised. p. 75 itself allows entry by order of the head of the bank when accompanied. Describe roles only. |
| 7 | `cash_room_door_open` | The cash room door stays open during operations. | No. 14 [p. 16 sub. 1)](https://adilet.zan.kz/rus/docs/V2000020075#z69): «каждая касса имеет отдельный вход с закрывающейся на замок дверью» (interim wording until 01.01.2027) | indirect: the clause describes the room's construction, not a duty to keep the door shut | that the door was unlocked; best used together with #6 |
| 8 | `collection_bag_not_checked` | The cashier takes a bag from a cash collector and opens or stores it without looking at the bag and its seal. | No. 231 [p. 24 sub. 3)–4)](https://adilet.zan.kz/rus/docs/V1900019680#z117): «проверяет целостность сумки; проверяет правильность пломбирования сумки»; the collector's side: [p. 107](https://adilet.zan.kz/rus/docs/V1900019680#z298) | direct, but the check is a short glance and easy to miss on video | whether the seal matches its sample; the cover document |
| 9 | `damaged_bag_without_collector` | A damaged bag is opened and the contents counted after the collector has left. | No. 231 [p. 25](https://adilet.zan.kz/rus/docs/V1900019680#z123): «В случае обнаружения нарушения целостности сумки кассовый работник в присутствии работника подразделения инкассации … вскрывает и пересчитывает» | direct only if the damage is visible | the shortfall or surplus |
| 10 | `counting_without_controller` | Away from clients, one cashier counts, sorts or packs bundles with no second employee present. | No. 231 [p. 42](https://adilet.zan.kz/rus/docs/V1900019680#z163): «… осуществляются кассовым работником в присутствии контролирующего работника, за исключением случаев совершения кассовых операций по приему и выдаче наличных денег клиенту.» | direct, but **not for serving clients**, which is exempt | that nobody is off camera |
| 11 | `no_authenticity_check` | Cash is accepted or exchanged without passing it through a counter or detector and without any visible check. | No. 231 [p. 19 sub. 2)](https://adilet.zan.kz/rus/docs/V1900019680#z104), [p. 36](https://adilet.zan.kz/rus/docs/V1900019680#z155) (exchange), [p. 9](https://adilet.zan.kz/rus/docs/V1900019680#z78) (equipment); definitions in No. 230 | indirect | that no check happened at all, because it can be done by eye. Claim only "the detector was not visibly used". |
| 12 | `keys_unsecured` | Keys to the safe or drawer are left in the lock or handed to another person. | No. 231 [p. 76](https://adilet.zan.kz/rus/docs/V1900019680#z222), [p. 80](https://adilet.zan.kz/rus/docs/V1900019680#z226): «… определяется внутренними документами банка, обеспечивающими исключение их несанкционированного использования.» | indirect: p. 80 leaves the procedure to internal rules | that the use was unauthorised |
| 13 | `collection_in_public_view` | Cash collection takes place in the customer area in view of clients. | No. 14 [p. 21](https://adilet.zan.kz/rus/docs/V2000020075#z90): «обеспечивается ограничение доступа и просмотра третьими лицами за осуществлением инкассации» | indirect: the clause is about the layout of the loading area | anything about the route or schedule |

Types 1–5 and 8 are the strongest candidates for the training clips: one camera above one cash desk, with clients and cash collectors in frame (FAQ 3). Types 6, 7 and 9–13 cover violation types that may appear only in the test set (FAQ 9). Each has its own queries, so adding one to `rules.yaml` needs no code change.

## 3. Rules video cannot check (limitations to list in the README)

These need data the video does not show, so the service must never report them (section 7):
- Customer due diligence before a cash operation (No. 231 p. 7).
- Liability contracts and segregation of duties (pp. 13–14).
- Whether a transfer between cashiers was recorded (p. 15).
- The required details on cash documents (pp. 17–18, 29–30).
- Whether signatures match their samples, and whether amounts match the document (p. 19 sub. 3)–4), p. 31 sub. 2)).
- Same-day crediting (p. 26).
- Whether the packaging is the bank's own (p. 33).
- End-of-day reconciliation and cash documents (pp. 56, 64–68).
- Audits (pp. 81–85).

**Identifying the client by their ID document** (p. 19 sub. 1), p. 31 sub. 1)) would mean comparing a face with a document. FAQ 29 forbids that. Do not flag it.

**Client recount** (p. 35: "the client counts the cash received without leaving the cash desk") limits the *client's* later claims. It places no duty on the cashier, so a client walking away without counting is **not** a cashier violation.

## 4. How a rule links to the video

1. **Video to rule (design.md, steps 2–6).** Each `rules.yaml` entry carries:
   - visual criteria and grounding queries, which find the moment on video;
   - `verify_prompt`, which confirms the moment;
   - `never_claim`, which limits what the justification may say;
   - citations, with the act, clause, verbatim quote and adilet URL.

   The judge picks a rule id, and the citation is looked up by that id. No model writes a citation.
2. **Rule to source (this index).** The rules workstream uses the search page (`/regs`) to find the clause for each type. "Для rules.yaml" copies a ready `citations` block with the verbatim quote and the anchored adilet URL.
3. **Checking (`python -m regs verify rules.yaml`).** Fails if any quote in `rules.yaml` is not word for word in the indexed acts. Run it before every deploy, so a mistyped or invented quote cannot reach the jury. This targets criterion 4.
4. **The `other` rule** cites No. 231 at document level (FAQ 31 allows a citation without a clause) and uses hedged wording.
