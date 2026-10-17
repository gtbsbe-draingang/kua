"""Translate everyday words of a free-form sentence into the professional terms of the NBRK acts.

The acts say "кассовый работник", "третьи лица", "прием наличных денег"; people and video
descriptions say "кассир", "посторонний", "протягивает купюры". bge-m3 and BM25 match better in the
acts' own vocabulary. The model gets two inputs: the user's sentence, and the texts of the clauses a
plain first-pass search found for it (search._context). Those texts are a dictionary of the acts'
real terms ("кассовый узел", "средства связи, не имеющие функции фото и видео фиксации"); they
replace a hand-written glossary. The model does a word-for-word translation into keywords only: it does not
guess which rule applies, does not judge the situation and adds nothing that was not described
(an earlier version that wrote the presumed rule steered the search toward the wrong clause).
The model never names a document or clause; every citation still comes from the index.
"""
import functools
import re

from . import config

SYSTEM = """Ты переводчик бытовых слов на язык нормативных актов Национального Банка Республики Казахстан \
(правила кассовых операций и инкассации, обменных операций с наличной иностранной валютой, определения \
платежности банкнот и монет, охраны и устройства помещений банков).

Пользователь описывает ситуацию своими словами, на любом языке. Переведи ее слова и словосочетания в \
профессиональные термины этих актов, один к одному, как переводчик: кассир -> кассовый работник, \
посторонний -> третье лицо, купюры -> банкноты, телефон -> средства связи, ящик -> металлический шкаф \
или сейф, протягивает деньги -> передача наличных денег, и т.п.

Правила:
- Только ключевые слова и короткие словосочетания через запятую, 4-15 штук. Никаких предложений.
- Переводи только то, что есть в описании. Ничего не добавляй: не угадывай, какое правило или какое \
нарушение здесь может быть, не оценивай, не пиши, что работник обязан или не вправе делать.
- Если описанное не касается того, что регулируют эти акты (обращение с наличными деньгами и ценностями, \
работа кассы и кассового работника, инкассация, обмен валюты, платежность и подлинность банкнот и монет, \
доступ в кассовые помещения, охрана и устройство помещений банка), ответь одним словом НЕТ.
- Пользователь может приложить фрагменты актов: это словарь их терминологии, а не ответ. Если там есть \
термин, который точно называет то, что описано, используй именно его. Не бери из фрагментов понятия, \
которых нет в описании.
- Окончательный ответ напиши только внутри тегов <ответ>...</ответ>; всё вне тегов игнорируется."""


def _message(query: str, context: tuple[str, ...]) -> str:
    if not context:
        return query
    vocab = "\n".join(f"- {c}" for c in context)
    return f"Описание:\n{query}\n\nФрагменты актов (словарь терминов, не ответ):\n{vocab}"


def enabled() -> bool:
    return bool(config.REWRITE_BASE_URL and config.REWRITE_MODEL)


@functools.lru_cache(maxsize=1024)
def rewrite(query: str, context: tuple[str, ...] = ()) -> str:
    """Comma-separated professional terms, or "НЕТ" if the acts do not cover the query.
    `context`: clause texts from a first-pass search, shown as the acts' vocabulary."""
    from openai import OpenAI
    client = OpenAI(base_url=config.REWRITE_BASE_URL, api_key=config.REWRITE_API_KEY)
    # 1024, not less: with a small limit the model can stop before writing anything (finish_reason=length)
    r = client.chat.completions.create(model=config.REWRITE_MODEL, max_tokens=1024, messages=[
        {"role": "system", "content": SYSTEM}, {"role": "user", "content": _message(query, context)}])
    # Only the last <ответ> block counts, so a draft or comment outside it never reaches the search.
    blocks = re.findall(r"<ответ>(.*?)</ответ>", r.choices[0].message.content or "", re.S)
    return " ".join(blocks[-1].split()) if blocks else query


def out_of_scope(rewritten: str | None) -> bool:
    """The model answered that no requirement of the indexed acts applies."""
    return (rewritten or "").strip(" .!\"«»").upper() == "НЕТ"
