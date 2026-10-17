"""regs.violation.check (summary -> search top-k -> compare) on synthetic logs.
Run like exp.py, with the real rag/regs mounted at /app/regs.
CASES=timelines.yaml,timelines_gen.yaml picks the case files; K=3,10 compares k on the same summaries.
A case with expect [] is a clean log: it is right when no violation is reported. Lenient scoring also accepts
the clauses in `also` (other clauses the generator's verifier saw broken)."""
import functools
import os
import sys

import yaml

sys.path.insert(0, "/app")
from regs import violation  # noqa: E402

D = os.environ.get("EXP_DIR", "/exp")
violation.summarize = functools.lru_cache(violation.summarize)  # same summary for every k
cases = [c for f in os.environ.get("CASES", "timelines.yaml").split(",")
         for c in yaml.safe_load(open(f"{D}/{f}"))["cases"]]
bad, clean = [c for c in cases if c["expect"]], [c for c in cases if not c["expect"]]
for k in map(int, os.environ.get("K", "3").split(",")):
    print(f"\n=== k={k}")
    recall = top1 = lenient = extra = quiet = 0
    for c in cases:
        r = violation.check(c["log"], k=k)
        cand = [x["key"] for x in r["candidates"]]
        got = [v["key"] for v in r["violations"]]
        ok = set(c["expect"]) | set(c.get("also") or [])
        if c["expect"]:
            recall += any(x in c["expect"] for x in cand)
            top1 += bool(got) and got[0] in c["expect"]
            lenient += bool(got) and got[0] in ok
            extra += sum(x not in ok for x in got)
        else:
            quiet += not got
        mark = ("OK  " if got[:1] and got[0] in ok else "MISS") if c["expect"] else ("OK  " if not got else "FP  ")
        print(f"{mark} {c['id']:20s} expect={c['expect']} also={c.get('also') or []}\n"
              f"     summary: {r['summary']}\n     top-{k}: {cand}\n     violations: {got}")
    n = len(bad)
    print(f"k={k}: {n} violation logs: expected clause in top-{k} {recall}/{n}; first violation correct {top1}/{n} "
          f"(lenient {lenient}/{n}); clauses outside expect+also {extra}. {len(clean)} clean logs: no violation "
          f"reported {quiet}/{len(clean)}")
