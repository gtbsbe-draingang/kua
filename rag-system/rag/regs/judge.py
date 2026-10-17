"""LLM judge: keep only the retrieved clauses whose requirement applies to the described situation.

Retrieval ranks clauses by similarity, so near-misses (п. 31, paying out, vs п. 19, accepting)
and omissions ("did not ask for ID": nothing in the description names the missing step) stay
hard. The judge reads the description next to the full clause texts of the LambdaMART top
JUDGE_POOL and answers with list numbers, most relevant first, or НЕТ. It picks by number only:
the citation (act, clause label, text, URL) still comes from the index, so no model writes one.
Research behind it: research.md (judge over a fan-out pool, top-1 0.17 -> 0.67 on 12 logs).
The prompt is lenient on purpose: off-topic queries are already stopped by the rewrite gate, so the
judge answers НЕТ only when no clause regulates the described operation. Against a strict prompt
("the action contradicts the clause") it halved missed clauses on dev (2/28 -> 1/28) for 2/30
off-topic queries shown; adding LambdaMART's top-1 as a fallback only added false positives.
"""
import re

from . import config

JUDGE_POOL = 20
MAX_PICKS = 5

SYSTEM = """Ты проверяешь, какие пункты нормативных актов Национального Банка Республики Казахстан \
относятся к описанной ситуации в банке.

Тебе дают описание ситуации (слова пользователя или описание видео с камеры над кассой) и \
пронумерованный список пунктов актов с их полным текстом.

Выбери пункты, которые регулируют описанную ситуацию: устанавливают порядок именно той операции или \
того действия, которые описаны (прием, выдача, пересчет, упаковка, хранение, инкассация, доступ в \
помещения, средства связи и т.п.), или требуют шага, который в описании пропущен, или запрещают \
описанное действие. Выбирай пункт, даже если нарушение в описании явно не названо.
Не выбирай пункты о другой операции (например, выдача вместо приема), пункты-определения и общие \
положения, если есть пункт о самой ситуации, и пункты, совпадающие с описанием только отдельными словами.

Ответ: номера из списка через запятую, самый подходящий первым, не больше {max_picks}, внутри тегов \
<ответ>...</ответ>. Ответь <ответ>НЕТ</ответ>, только если ни один пункт не регулирует описанные действия. \
Всё вне тегов игнорируется."""


def enabled() -> bool:
    return bool(config.REWRITE_BASE_URL and config.REWRITE_MODEL)


def _message(query: str, hits: list[dict]) -> str:
    items = "\n\n".join(f"[{i}] {h['doc_short']}, {h['clause_label']}\n{h['text']}" for i, h in enumerate(hits, 1))
    return f"Ситуация:\n{query}\n\nПункты:\n{items}"


def judge(query: str, hits: list[dict], system: str | None = None, max_tokens: int = 1024) -> list[int]:
    """Indices into `hits` of the clauses that apply, most relevant first; [] for none."""
    if not hits:
        return []
    from openai import OpenAI
    client = OpenAI(base_url=config.REWRITE_BASE_URL, api_key=config.REWRITE_API_KEY)
    r = client.chat.completions.create(model=config.REWRITE_MODEL, max_tokens=max_tokens, messages=[
        {"role": "system", "content": (system or SYSTEM).format(max_picks=MAX_PICKS)},
        {"role": "user", "content": _message(query, hits)}])
    blocks = re.findall(r"<ответ>(.*?)</ответ>", r.choices[0].message.content or "", re.S)
    if not blocks:
        raise RuntimeError("judge answered without <ответ> tags")
    picks = []
    for n in re.findall(r"\d+", blocks[-1]):  # "НЕТ" has no digits -> []
        i = int(n) - 1
        if 0 <= i < len(hits) and i not in picks:
            picks.append(i)
    return picks[:MAX_PICKS]
