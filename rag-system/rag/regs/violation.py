"""From a VLM log of a video to the clause it breaks, in four steps:

    1. the VLM writes a per-second log of the video (outside this service);
    2. an LLM compresses the log into a one- or two-sentence summary in the acts' professional terms;
    3. the summary is searched in the index (it reads like the one-sentence queries LambdaMART was
       trained on, unlike the raw log: research.md, section 2);
    4. an LLM reads the summary next to the top-k clauses and names the ones the actions break.
    5. the result carries those clauses' citations from the index; no model writes a citation.

    from regs.violation import check
    check(log)  # -> {"summary": ..., "violations": [citation, ...], "candidates": [...]}
"""
import re

from . import config, judge
from .search import citation, search

# Both calls make the model reason before answering; at 1024 it sometimes stopped before writing the
# answer (finish_reason=length, empty content).
MAX_TOKENS = 4096

SUMMARY = """Тебе дан посекундный журнал видео с камеры над кассой банка. Сожми его в одно-два предложения \
на профессиональном языке нормативных актов Национального Банка Республики Казахстан (кассовый работник, \
клиент, прием или выдача наличных денег, пересчет, документ, удостоверяющий личность, инкассаторская сумка, \
пломба, третьи лица, средства связи, кассовое помещение, сейф и т.п.).

Назови операцию и действия кассового работника, которые важны для порядка ее проведения. Обязательно \
сохрани: шаги, которые пропущены или сделаны не по порядку (например, деньги приняты без пересчета, документ \
не проверен), уход с рабочего места, посторонних лиц, телефоны, открытые двери, деньги или ключи без присмотра.
Не пиши время, внешность и обычные действия без значения. Не называй пункты и нормы и не оценивай, нарушено ли \
что-то: только факты.

Ответ напиши только внутри тегов <ответ>...</ответ>; всё вне тегов игнорируется."""

COMPARE = """Ты проверяешь работу кассового работника банка по нормативным актам Национального Банка \
Республики Казахстан.

Тебе дают краткое описание того, что произошло, и пронумерованный список пунктов актов с их полным текстом.

Выбери пункты, которые описанные действия нарушают: работник не выполнил требование пункта, выполнил его не \
в том порядке или сделал то, что пункт запрещает. Не выбирай пункт, который только регулирует эту операцию, \
но описанными действиями соблюден, и пункт, исключение из которого прямо описано (например, выдача пачки в \
целой банковской упаковке).

Ответ: номера нарушенных пунктов через запятую, самый явный первым, не больше {max_picks}, внутри тегов \
<ответ>...</ответ>. Ответь <ответ>НЕТ</ответ>, если ни один пункт не нарушен. Всё вне тегов игнорируется."""


def summarize(log: str) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=config.REWRITE_BASE_URL, api_key=config.REWRITE_API_KEY)
    r = client.chat.completions.create(model=config.REWRITE_MODEL, max_tokens=MAX_TOKENS, messages=[
        {"role": "system", "content": SUMMARY}, {"role": "user", "content": log}])
    blocks = re.findall(r"<ответ>(.*?)</ответ>", r.choices[0].message.content or "", re.S)
    if not blocks:
        raise RuntimeError("summary answered without <ответ> tags")
    return " ".join(blocks[-1].split())


def check(log: str, k: int = 3) -> dict:
    summary = summarize(log)
    hits = search(summary, k=k, mode="auto", threshold=False)["hits"]
    picks = judge.judge(summary, hits, system=COMPARE, max_tokens=MAX_TOKENS)
    return {"summary": summary, "violation": bool(picks),
            "violations": [{"key": f"{hits[i]['doc_code']}:{hits[i]['clause']}", **citation(hits[i])} for i in picks],
            "candidates": [{"key": f"{h['doc_code']}:{h['clause']}", "document": h["doc_short"],
                            "clause": h["clause_label"]} for h in hits]}
