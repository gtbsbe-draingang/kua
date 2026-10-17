"""Experiment: conformance checking instead of rule retrieval.
An LLM only labels log lines with a closed vocabulary of events (no rules, no verdicts);
code checks each service against the expected procedure and reports deviations with their clause.
Run like exp.py:  docker run ... rag-system-rag python /exp/proc.py
"""
import hashlib
import json
import os
import re
import sys

import yaml

sys.path.insert(0, "/app")
from regs import config  # noqa: E402

D = os.environ.get("EXP_DIR", "/tmp/exp")
CACHE_PATH = f"{D}/llm_cache.json"
cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}


def llm(system: str, user: str, max_tokens: int = 4000) -> str:
    key = hashlib.sha1((system + "\x00" + user).encode()).hexdigest()
    if key not in cache:
        from openai import OpenAI
        c = OpenAI(base_url=config.REWRITE_BASE_URL, api_key=config.REWRITE_API_KEY)
        r = c.chat.completions.create(model=config.REWRITE_MODEL, max_tokens=max_tokens,
                                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        cache[key] = r.choices[0].message.content or ""
        json.dump(cache, open(CACHE_PATH, "w"), ensure_ascii=False)
    return cache[key]


# Closed event vocabulary. Descriptions say what is visible, never what is allowed.
EVENTS = {
    "service_start": "клиент подошел к окну / начал обслуживаться (arg client = номер клиента 1, 2, ...)",
    "service_end": "клиент забрал документы/деньги и ушел от окна (arg client)",
    "id_check": "кассир смотрит документ клиента, сверяет фото (arg client)",
    "take_cash": "кассир берет наличные, переданные клиентом (arg client)",
    "count_cash": "кассир пересчитывает деньги вручную или на счетной машине (arg client, если деньги этого клиента)",
    "store_cash": "кассир убирает деньги в ящик, сейф, шкаф",
    "hand_cash": "кассир передает наличные клиенту (arg client)",
    "sealed_bundle": "передаваемые клиенту деньги — пачка в целой банковской упаковке/бандероли",
    "cashier_leaves": "кассир встает и уходит с рабочего места / из кабины",
    "cashier_returns": "кассир возвращается на рабочее место",
    "cash_unattended": "деньги лежат открыто на столе (не в ящике/сейфе), пока кассир отошел или обслуживает другого клиента",
    "phone_camera": "в кассе используется смартфон/телефон с камерой (съемка, наведение камеры)",
    "outsider_in_room": "в кассовое помещение заходит человек, не являющийся кассиром или инкассатором при исполнении",
    "keys_unattended": "ключи оставлены в замке без присмотра или переданы постороннему",
    "door_open": "дверь кассового помещения открыта/подперта во время работы",
    "bag_received": "кассир получает инкассаторскую сумку",
    "bag_checked": "кассир осматривает сумку, целостность, пломбу перед вскрытием",
    "bag_opened": "кассир вскрывает сумку / срезает пломбу",
    "packing": "кассир пересчитывает, сортирует, упаковывает, бандеролирует деньги вне обслуживания клиента",
    "controller_present": "рядом с кассиром присутствует второй (контролирующий) работник во время пересчета/упаковки",
}

LABEL = ("Тебе дан посекундный журнал видео из кассы банка. Разметь его событиями ТОЛЬКО из словаря ниже. "
         "Не оценивай и не ищи нарушения, только отмечай, что видно. Одна строка журнала может дать 0, 1 или "
         "несколько событий. Событие, которого в журнале нет, не добавляй.\n\nСЛОВАРЬ:\n"
         + "\n".join(f"- {k}: {v}" for k, v in EVENTS.items())
         + '\n\nОтвет строго JSON-массив: [{"t": "00:00:05", "event": "take_cash", "client": 1}, ...] '
           '(client только для событий с клиентом).')


def label(log: str) -> list[dict]:
    out = llm(LABEL, log)
    m = re.search(r"\[.*\]", out, re.S)
    ev = json.loads(m.group(0)) if m else []
    return [e for e in ev if e.get("event") in EVENTS]


def check(ev: list[dict]) -> list[tuple[str, str, str]]:
    """Conformance check. Returns (time, clause key, reason) for every deviation from the procedures."""
    out = []
    by_client: dict = {}
    for i, e in enumerate(ev):
        if e.get("client") is not None:
            by_client.setdefault(e["client"], []).append((i, e))

    def idx(c, name):
        return [i for i, e in by_client.get(c, []) if e["event"] == name]

    for c, items in by_client.items():
        start = min(i for i, _ in items)
        end = (idx(c, "service_end") or [len(ev)])[-1]
        took, gave = idx(c, "take_cash"), idx(c, "hand_cash")
        if (took or gave) and not idx(c, "id_check"):
            # Receipt (п. 19 пп. 1) or payout (п. 31 пп. 1): identification comes first
            out.append((ev[(took or gave)[0]]["t"], "V1900019680:19" if took else "V1900019680:31", "no ID check"))
        for t in took:  # receipt: counted in the client's presence (п. 20, п. 19 пп. 2)
            counted = [i for i in range(t, end + 1) if ev[i]["event"] == "count_cash" and ev[i].get("client") in (c, None)]
            if not counted:
                out.append((ev[t]["t"], "V1900019680:20", "cash taken, not counted while client present"))
        for g in gave:  # payout: counted before handing over (п. 31 пп. 4), unless full sealed bundle (п. 32)
            counted = any(ev[i]["event"] == "count_cash" for i in range(start, g))
            sealed = any(ev[i]["event"] == "sealed_bundle" for i in range(start, g + 1))
            if not counted and not sealed:
                out.append((ev[g]["t"], "V1900019680:31", "cash handed over uncounted"))
        # п. 22: cashier does not leave before the service ends
        for i in range(start, end):
            if ev[i]["event"] == "cashier_leaves":
                out.append((ev[i]["t"], "V1900019680:22", f"left during service of client {c}"))
    # Bag: integrity and seal checked before opening (п. 24 пп. 3–4)
    for i, e in enumerate(ev):
        if e["event"] == "bag_opened" and not any(x["event"] == "bag_checked" for x in ev[:i]):
            out.append((e["t"], "V1900019680:24", "bag opened without checking seal"))
        if e["event"] == "packing" and not any(x["event"] == "controller_present" for x in ev):
            out.append((e["t"], "V1900019680:42", "packing without controlling employee"))
    # State invariants: any occurrence is a deviation
    invariants = {"cash_unattended": "V1900019680:20", "phone_camera": "V1900019680:12",
                  "outsider_in_room": "V1900018545:41", "keys_unattended": "V1900019680:80",
                  "door_open": "V2000020075:16"}
    for e in ev:
        if e["event"] in invariants:
            out.append((e["t"], invariants[e["event"]], e["event"]))
    seen, uniq = set(), []
    for o in out:  # one report per clause, earliest first
        if o[1] not in seen:
            seen.add(o[1]); uniq.append(o)
    return uniq


cases = yaml.safe_load(open(f"{D}/timelines.yaml"))["cases"]
caught = fp = 0
for c in cases:
    ev = label(c["log"])
    found = check(ev)
    hit = any(k in c["expect"] for _, k, _ in found)
    caught += hit
    extra = [f for f in found if f[1] not in c["expect"]]
    fp += len(extra)
    print(f"{'HIT ' if hit else 'MISS'} {c['id']:26s} found={[(t, k.split(':')[1], r) for t, k, r in found]}")
    if not hit:
        print("      events:", [(e['t'], e['event'], e.get('client')) for e in ev])
print(f"\ncaught {caught}/{len(cases)}  extra (not expected) reports {fp}, {fp / len(cases):.1f} per log")
