# Rules catalogue (`rules.yaml`, version 1)

The service flags *possible* cash-handling violations in bank cash-desk video and cites the
regulation behind each flag. This file lists the sources, explains how a flag gets its
citation, and records what the service cannot check from video.

## 1. Sources

All quotes in `rules.yaml` are verbatim from the Russian text of the act's current edition on
adilet.zan.kz, read on 2026-10-17. The texts were read through adilet's public JSON API (the
document pages are a JavaScript application); `edition_checked` in each citation is the date of
that reading. Each citation URL carries a `#zNNN` paragraph anchor taken from the served text.

| | Act 1 (main) | Act 2 (cash premises) |
|---|---|---|
| Title | Об утверждении Правил осуществления кассовых операций и операций по инкассации банкнот, монет и ценностей в банках второго уровня, филиалах банков-нерезидентов Республики Казахстан, Национальном операторе почты и юридических лицах, исключительной деятельностью которых является инкассация банкнот, монет и ценностей | Об утверждении Правил организации охраны и устройства помещений банков второго уровня, филиалов банков-нерезидентов Республики Казахстан, Национального оператора почты, юридических лиц, исключительной деятельностью которых является инкассация банкнот, монет и ценностей, и юридических лиц, осуществляющих деятельность исключительно через обменные пункты на основании лицензии Национального Банка Республики Казахстан на обменные операции с наличной иностранной валютой |
| Act | Постановление Правления Национального Банка Республики Казахстан от 29 ноября 2019 года № 231 (registered with the Ministry of Justice 3 December 2019, № 19680) | Постановление Правления Национального Банка Республики Казахстан от 24 февраля 2020 года № 14 (registered 27 February 2020, № 20075) |
| URL | https://adilet.zan.kz/rus/docs/V1900019680 | https://adilet.zan.kz/rus/docs/V2000020075 |
| Edition read | revision of 12.07.2026 (current) | revision of 24.12.2024 (current; last amended by № 88 of 24.12.2024) |
| In force | yes (in force, with amendments) | yes (in force, with amendments) |

Act 1 is the basis of 13 of the 14 detectable rules and of all non-detectable ones. Act 2 is used by
`unauthorized_person_in_cash_area` and, at document level, by `other`. Note on Act 2: № 88
suspends until 01.01.2027 its new wording of pp. 11, 13 подп. 3), 14 and 16. The corpus
(`rag/acts/V2000020075.json`) marks pp. 11, 14 and 16 as suspended, so the retrieval corpus drops
them and `rules.yaml` cites none of them.
`tests/test_rules.py::test_catalogue_citations_resolve_to_in_force_clauses` guards this for every
cited clause of every act in the corpus.

Checked and deliberately **not cited**: НБРК № 58 of 03.03.2001 and № 394 of 15.11.1999
(repealed, replaced by № 231); НБРК № 247 of 24.12.2014 (V14Y0010204: repealed and covers only the
National Bank's own units); НБРК № 120 of 28.09.2020 (also covers the National Bank itself).
`tests/test_rules.py::test_no_cited_repealed_acts` guards this.

## 2. How a rule reaches a flag

1. The catalogue's `grounding_queries` (2-4 English phrases per detectable rule, describing
   actions and roles, never screen positions) are given to the grounding model to propose
   candidate moments in the video.
2. The check prompt built from `visual_criteria`, `never_claim` and `verify_prompt` is asked about
   each candidate; `choice_option` is the rule's entry in the multiple-choice question.
3. The judge model picks a rule **id** (or `other`).
4. The citation is looked up from the catalogue by that id (`rules.citation_text`). No model writes
   or edits a citation: a flag can only carry text that was verified against adilet.

Because of step 4, a wrong flag can be wrong about *what happened* but cannot cite an invented
clause. Every flag is a hint for a human reviewer, not a finding.

## 3. Detectable rules

| id | Name | What is visible on video | Citation (Rule 231 unless stated) |
|---|---|---|---|
| `other` | Иное возможное нарушение порядка кассовых операций | cash or valuables are handled in a way that is not controlled by the cashier; an action with cash that does not match normal acceptance, issuance or cash-collection steps | document level; document level |
| `cash_unattended` | Оставление наличных без присмотра | cash lies exposed on the desk or counter; the cashier walks away or turns away for a long time; no employee keeps the exposed cash under control | п. 20, ч. 2 |
| `cashier_leaves_during_service` | Уход кассира с рабочего места во время обслуживания клиента | a client is at the cash desk and the transaction is not finished; the cashier gets up and walks away from the workplace; the client is left waiting, possibly with cash or documents on the counter | п. 22 |
| `previous_cash_on_desk` | Наличные от других операций на рабочем месте при обслуживании клиента | while a client is served, other stacks of cash that are not part of this client's transaction lie open on the desk; cash from earlier transactions is not put into a drawer, safe or cash box; cash of different clients is mixed on the counter | п. 20, ч. 2 |
| `cash_storage_left_open` | Хранилище наличных на рабочем месте оставлено открытым | a cash drawer, safe, cash box or trolley holding cash is left open; keys are left in the lock or lying on the desk; the cashier is away or busy with something else while it stays open | п. 20, ч. 2; п. 80 |
| `cash_into_personal_belongings` | Перемещение наличных в одежду или личные вещи | the cashier or another person puts banknotes into a pocket, sleeve, personal bag or wallet; the cash does not go into the cash drawer, safe or a collection bag; the movement may be done discreetly | п. 20, ч. 2; п. 12, ч. 1 |
| `personal_items_at_workplace` | Личные вещи или смартфон на рабочем месте кассира | the cashier uses a smartphone at the cash workplace; the cashier photographs or films with a phone at the desk; a personal handbag, wallet or other personal items are on the cash desk | п. 12, ч. 1; п. 12, ч. 2 |
| `deposit_not_counted` | Приём наличных без пересчёта | the client hands over banknotes or coins; the cashier puts the cash straight into a drawer, safe or bag; no sheet-by-sheet hand count and no counting machine is used for that cash | п. 19, подп. 2); п. 20, ч. 1 |
| `deposit_counted_without_client` | Пересчёт принимаемых наличных вне присутствия клиента | the cashier takes the client's cash away from the counter before counting it; the cash is counted only after the client has left; the client cannot see the counting | п. 20, ч. 1 |
| `withdrawal_not_counted` | Выдача наличных без пересчёта в присутствии клиента | the cashier hands cash to the client; loose banknotes or an opened or partial bundle are handed over without being counted in front of the client; a full bundle in intact bank wrapping is not a violation: it may be issued by the amount on the wrapper without a sheet count | п. 31, подп. 4); п. 32, ч. 1 |
| `recount_without_controller` | Пересчёт, сортировка или упаковка наличных без контролирующего работника | no client is being served; the cashier counts, sorts, bands or packs cash; no second employee watches the counting | п. 42 |
| `unauthorized_person_in_cash_area` | Доступ постороннего лица в зону кассы | a person who is not the cashier (a client or another visitor) enters the cashier's workspace; a client reaches across the counter and takes or touches cash, drawers or documents on the cashier's side; the cashier does not stop it | п. 8, ч. 1 |
| `collection_cash_not_in_sealed_bag` | Передача наличных инкассатору без опломбированной сумки | cash is handed between the cashier and a cash collector as loose banknotes or open bundles; the collection bag is open, not tied or has no seal when it is handed over; cash is put into the bag after the handover and the bag is not sealed | п. 105; п. 93, ч. 1 |
| `collection_bag_not_inspected` | Приём инкассаторской сумки без проверки целостности и пломбы | a cash collector and the cashier exchange a collection bag; the receiving side puts the bag away or opens it without looking at the bag and its seal; no visible inspection of the seal, the twine or the bag's integrity | п. 24, ч. 2, подп. 3); п. 24, ч. 2, подп. 4); п. 107, подп. 2) |

Clauses of Act 2 (№ 14) are used by `unauthorized_person_in_cash_area` (п. 8 ч. 1)
and by the document-level citation in `other`; every other clause above belongs to № 231.
`other` has an empty `clause`: it cites the act as a whole.

## 4. Rules that cannot be checked from video (`detectable: false`)

These rules are in the catalogue so the README can list what the service does not check. They are
never emitted as flags.

| id | Citation | Limitation |
|---|---|---|
| `client_identification` | п. 19, подп. 1); п. 7 | Не проверяется по видео: идентификация допускается по данным сервиса цифровых документов на экране кассира, а распознавание лиц сервису запрещено. |
| `signatures_and_document_details` | п. 19, подп. 3) | Не проверяется по видео: наличие и подлинность подписей, их сверка с образцами и реквизиты документа не читаются с камеры. |
| `amount_matches_document` | п. 19, подп. 4) | Не проверяется по видео: сумму наличных и сумму в документе нельзя надёжно определить с камеры. |
| `posting_to_client_account` | п. 26, ч. 1 | Не проверяется по видео: зачисление и корректность проводки видны только в учётной системе банка. |
| `banknote_authenticity_check` | п. 57 | Не проверяется по видео: подлинность и платёжность банкнот и результат проверки на оборудовании с камеры не определяются. |
| `suspicious_banknotes_procedure` | п. 61 | Не проверяется по видео: нельзя установить, что банкнота сомнительная или имеет признаки подделки, и проверить составление акта. |
| `cash_transfer_between_cashiers_record` | п. 15 | Не проверяется по видео: факт и содержание записи в документе учёта передачи наличных с камеры не устанавливаются. |
| `end_of_day_reconciliation` | п. 56 | Не проверяется по видео: сверка сумм с данными учёта выполняется по документам и в учётной системе. |
| `collection_documents_and_seal_match` | п. 24, ч. 2, подп. 1); п. 94, ч. 3 | Не проверяется по видео: сведения препроводительного документа и совпадение оттиска пломбы с образцом с камеры не читаются. |

## 5. Caveats from the legal research

- **`unauthorized_person_in_cash_area` is an inference.** Neither act says in words that outsiders
  may not enter the cash desk (№ 231 п. 75 restricts entry only to the vault and safe room). The
  rule rests on № 14 п. 8, which requires the cash unit to be isolated from other premises by one
  lockable service entrance. It deliberately does not cite п. 16 подп. 1) (a separate lockable
  entrance for each cash room): п. 16 is suspended until 01.01.2027. Its `why` text keeps the
  hedge "может указывать на нарушение".
- **`recount_without_controller` has a high false-positive risk.** The controlling employee may be
  out of frame, and preparing a client's withdrawal is excluded by the clause itself (п. 42).
- **`personal_items_at_workplace`:** п. 12 ч. 2 permits communication devices without photo or video
  functions; video cannot show whether a phone has a camera, so the rule is low severity.
- **`withdrawal_not_counted`:** п. 32 ч. 1 allows handing over bundles in intact bank wrapping
  without a sheet count; the visual criteria say so.
- **A client who leaves without counting cash is not flagged** (п. 35 governs the client's later
  claims, not a bank duty).
- **`cash_unattended`, `previous_cash_on_desk`, `cash_storage_left_open`** share п. 20 ч. 2 but
  describe different visible moments; they may be merged if the clips do not separate them.
- **Editions** were read through adilet's public API; the edition label on the rendered page and
  the anchor scrolling in a browser were not checked. The Kazakh-language text was not checked.

## 6. Связь с обучающими клипами / Clips

TODO: filled in after the per-clip labelling. For each rule with `seen_in_training: true`: the
clips that show it, their `t_start`/`t_end`, and how the visible action matches the citation.
