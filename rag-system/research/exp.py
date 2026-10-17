"""Experiment: long per-second action logs -> violated clause. Run inside the rag container:
    docker exec -w /app rag-system-rag-1 python /tmp/exp/exp.py
"""
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys

import numpy as np
import yaml

sys.path.insert(0, "/app")
from regs import config, store  # noqa: E402
from regs.chunk import load_chunks  # noqa: E402
from regs.embed import get_embedder  # noqa: E402
from regs.search import search  # noqa: E402

D = os.environ.get("EXP_DIR", "/tmp/exp")
CACHE_PATH = f"{D}/llm_cache.json"
cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}


def llm(system: str, user: str, max_tokens: int = 2048) -> str:
    key = hashlib.sha1((system + "\x00" + user).encode()).hexdigest()
    if key not in cache:
        from openai import OpenAI
        c = OpenAI(base_url=config.REWRITE_BASE_URL, api_key=config.REWRITE_API_KEY)
        r = c.chat.completions.create(model=config.REWRITE_MODEL, max_tokens=max_tokens,
                                      messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        cache[key] = r.choices[0].message.content or ""
    return cache[key]


def save_cache():
    json.dump(cache, open(CACHE_PATH, "w"), ensure_ascii=False)


cases = yaml.safe_load(open(f"{D}/timelines.yaml"))["cases"]
emb = get_embedder()

# ---------------------------------------------------------------- 0. truncation check
probe = "Кассир фотографирует деньги на смартфон."
pad = "\n".join(f"00:{i // 60:02d}:{i % 60:02d} Кассир сидит за столом и смотрит в монитор." for i in range(400))
v = lambda t: np.array(emb.query(t))
a, b, c = v(probe), v(pad), v(pad + "\n" + probe)
cos = lambda x, y: float(x @ y / np.linalg.norm(x) / np.linalg.norm(y))
print(f"[trunc] words(pad)={len(pad.split())}  cos(pad+probe, pad)={cos(b, c):.4f}  cos(pad+probe, probe)={cos(a, c):.4f}"
      f"  cos(pad, probe)={cos(a, b):.4f}  (pad+probe == pad => tail truncated)")


def key(h):
    return f"{h['doc_code']}:{h['clause']}"


def ranked(q, rewrite=False, k=30):
    return search(q, k=k, mode="ltr", rewrite=rewrite, threshold=False)["hits"]


def fuse(lists: list[list[dict]], how="max"):
    """lists of folded hits -> clause ranking. max = best LTR score over sub-queries; rrf = sum 1/(10+rank)."""
    s: dict[str, float] = {}
    for hits in lists:
        for r, h in enumerate(hits, 1):
            k_ = key(h)
            if how == "max":
                s[k_] = max(s.get(k_, -1e9), h["score"])
            else:
                s[k_] = s.get(k_, 0) + 1 / (10 + r)
    return [k_ for k_, _ in sorted(s.items(), key=lambda x: -x[1])]


def rank_of(order, expect):
    return next((i + 1 for i, k_ in enumerate(order) if k_ in expect), None)


def windows(log: str, size=6, stride=3):
    lines = [l for l in log.strip().splitlines() if l.strip()]
    return ["\n".join(lines[i:i + size]) for i in range(0, max(1, len(lines) - size + stride), stride)]


strip_ts = lambda t: re.sub(r"^\d\d:\d\d:\d\d\s*", "", t, flags=re.M)

EPISODES = """Тебе дан посекундный журнал действий работника банка, записанный по видео.
Сожми его в 2-6 эпизодов. Каждый эпизод — одно предложение в настоящем времени: кто, что делает, с чем,
в каком порядке, и чего НЕ происходит между шагами, если это заметно (например: «берет деньги клиента и сразу кладет
в ящик, не пересчитывая»). Не оценивай, не упоминай правила и нарушения. Только наблюдаемые факты из журнала.
Ответ: эпизоды по одному на строке внутри <ответ>...</ответ>."""

DEVIATIONS = """Ты контролер кассовой дисциплины банка в Казахстане (правила НБРК: кассовые операции и инкассация,
обменные операции, платежность банкнот, охрана и устройство помещений). Тебе дан посекундный журнал действий,
записанный по видео. Найди до 4 моментов, где поведение работника или ситуация в кассе могут расходиться с
обязательным порядком работы (пропущен обязательный шаг, неправильный порядок, недопустимый предмет или человек,
оставленные без контроля деньги или ключи и т.п.). Для каждого момента напиши одно короткое предложение-наблюдение:
что видно и какого обязательного шага не было. Не называй номера пунктов и документы. Если ничего подозрительного — НЕТ.
Ответ: по одному предложению на строке внутри <ответ>...</ответ>."""


def tagged_lines(text):
    m = re.findall(r"<ответ>(.*?)</ответ>", text, re.S)
    body = m[-1] if m else text
    return [l.strip(" -•\t0123456789.") for l in body.splitlines() if l.strip(" -•\t") and l.strip().upper() != "НЕТ"]


# ---------------------------------------------------------------- violation cards (doc-side expansion)
CARDS = """Для каждого пункта нормативного акта Национального Банка РК опиши, как его НАРУШЕНИЕ выглядело бы на записи
камеры видеонаблюдения в банке/кассе: 3 коротких предложения бытовым языком, как их написал бы наблюдатель
(«кассир кладет деньги клиента в ящик, не пересчитав их»). Только видимые действия людей, предметы, двери, деньги.
Если нарушение пункта нельзя увидеть на видео (определения, сроки, отчетность, документооборот) — пустой список.
Ответ строго JSON: {"<id>": ["...", "...", "..."], ...} без пояснений."""

chunks = [c for c in load_chunks() if c["granularity"] == "clause" and c["status"] != "repealed"]
cards_path = f"{D}/cards.json"
if os.path.exists(cards_path):
    cards = json.load(open(cards_path))
else:
    batches = [chunks[i:i + 10] for i in range(0, len(chunks), 10)]

    def gen(batch):
        user = "\n\n".join(f"id={c['id']}\n{c['doc_short']}. {c['chapter']}.\n{c['text'][:1500]}" for c in batch)
        out = llm(CARDS, user, 4000)
        m = re.search(r"\{.*\}", out, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}
    cards = {}
    with cf.ThreadPoolExecutor(8) as ex:
        for d in ex.map(gen, batches):
            cards.update(d)
    json.dump(cards, open(cards_path, "w"), ensure_ascii=False, indent=1)
    save_cache()
# LLM sometimes drops the act code (only No. 231 has clauses 120+) or packs 3 sentences into one string
norm_id = lambda cid: ":".join((cid if ":" in cid else "V1900019680:" + cid).split(":")[:2])
card_rows = [(norm_id(cid), sent) for cid, ss in cards.items() for s in ss
             for sent in re.split(r"(?<=[.!?])\s+", s) if len(sent.split()) > 3]
print(f"[cards] {len(cards)} clauses, {sum(bool(v) for v in cards.values())} observable, {len(card_rows)} card sentences")
card_vec = np.array(emb.passages([s for _, s in card_rows]))
card_vec /= np.linalg.norm(card_vec, axis=1, keepdims=True)


def card_search(q, k=30):
    qv = np.array(emb.query(q)); qv /= np.linalg.norm(qv)
    sims = card_vec @ qv
    best: dict[str, float] = {}
    for (cid, _), s in zip(card_rows, sims):
        best[cid] = max(best.get(cid, -1), float(s))
    return [{"doc_code": c.split(":")[0], "clause": c.split(":")[1], "score": s}
            for c, s in sorted(best.items(), key=lambda x: -x[1])[:k]]


# ---------------------------------------------------------------- LLM listwise rerank
text_of = {f"{c['doc_code']}:{c['clause']}": c for c in load_chunks() if c["granularity"] == "clause"}
RERANK = """Тебе дан посекундный журнал действий из видео в банке и пронумерованный список пунктов нормативных актов.
Выбери пункты, которые нарушены в журнале, в порядке убывания уверенности (не больше 5). Нарушение = в журнале
видно действие или пропуск шага, прямо противоречащее тексту пункта. Ответ: номера из списка через запятую внутри
<ответ>...</ответ>, или НЕТ."""


def llm_rerank(log, cand_keys, chars=900, n=20):
    cand_keys = cand_keys[:n]
    lst = "\n\n".join(f"[{i + 1}] {text_of[k_]['doc_short']}, {text_of[k_]['clause_label']}: {text_of[k_]['text'][:chars]}"
                      for i, k_ in enumerate(cand_keys) if k_ in text_of)
    out = llm(RERANK, f"ЖУРНАЛ:\n{log}\n\nПУНКТЫ:\n{lst}", 1024)
    m = re.findall(r"<ответ>(.*?)</ответ>", out, re.S)
    picks = [int(x) - 1 for x in re.findall(r"\d+", m[-1] if m else "") if 0 < int(x) <= len(cand_keys)]
    picked = [cand_keys[i] for i in dict.fromkeys(picks)]
    return picked + [k_ for k_ in cand_keys if k_ not in picked]


catalog = [k_ for k_ in dict.fromkeys(f"{c['doc_code']}:{c['clause']}" for c in chunks) if k_ in text_of]

# ---------------------------------------------------------------- run
def run_case(c):
    log, exp = c["log"], set(c["expect"])
    eps = tagged_lines(llm(EPISODES, log))
    dev = tagged_lines(llm(DEVIATIONS, log))
    R = {}
    R["B0 whole log, ltr"] = [key(h) for h in ranked(log)]
    R["B1 whole log, ltr+rewrite"] = [key(h) for h in ranked(log, rewrite=True)]
    R["B2 whole log no timestamps"] = [key(h) for h in ranked(strip_ts(log))]
    W = [ranked(strip_ts(w)) for w in windows(log)]
    R["S1 windows6/3 max"] = fuse(W, "max")
    R["S2 windows6/3 rrf"] = fuse(W, "rrf")
    E = [ranked(e) for e in eps] or [ranked(log)]
    R["L1 LLM episodes rrf"] = fuse(E, "rrf")
    V = [ranked(d) for d in dev] or [ranked(log)]
    R["L2 LLM deviations rrf"] = fuse(V, "rrf")
    CV = [card_search(d) for d in (dev or [log])]
    R["D1 cards(dev) rrf"] = fuse(CV, "rrf")
    R["D2 L2 + cards rrf"] = fuse(V + CV, "rrf")
    R["D3 L2+cards+windows rrf"] = fuse(V + CV + W, "rrf")
    R["R1 LLM rerank of D2 top-20"] = llm_rerank(log, R["D2 L2 + cards rrf"])
    R["R2 LLM rerank of B0 top-20"] = llm_rerank(log, R["B0 whole log, ltr"])
    R["R3 LLM rerank of D2 full text"] = llm_rerank(log, R["D2 L2 + cards rrf"], chars=4000)
    R["R4 LLM rerank of D3 top-30 full"] = llm_rerank(log, R["D3 L2+cards+windows rrf"], chars=4000, n=30)
    R["R5 LLM over whole catalog"] = llm_rerank(log, catalog, chars=600, n=len(catalog))
    return c["id"], {n: rank_of(o, exp) for n, o in R.items()}, eps, dev, R


with cf.ThreadPoolExecutor(1) as ex:  # snowballstemmer globals are not thread-safe
    results = list(ex.map(run_case, cases))
save_cache()

names = list(results[0][1])
print(f"\n{'method':32s} top1  top3  top5  top10 MRR   | per-case rank")
for n in names:
    rs = [r[1][n] for r in results]
    at = lambda m: sum(bool(x and x <= m) for x in rs) / len(rs)
    mrr = sum(1 / x for x in rs if x) / len(rs)
    print(f"{n:32s} {at(1):.2f}  {at(3):.2f}  {at(5):.2f}  {at(10):.2f}  {mrr:.2f}  | " + " ".join(str(x or '-') for x in rs))
print("cases: " + " ".join(r[0] for r in results))
for cid, _, eps, dev, R in results:
    print(f"\n## {cid}\n  episodes: {eps}\n  deviations: {dev}\n  B0 top5: {R['B0 whole log, ltr'][:5]}\n  R1 top5: {R['R1 LLM rerank of D2 top-20'][:5]}")
