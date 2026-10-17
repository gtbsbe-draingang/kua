"""Build data/labels.csv from the given spreadsheet.

Clip names are normalised to Latin ``X-<n>`` (no extension). Byte-identical
files (md5) are merged into the lowest-numbered clip, one row per event.
"""
import argparse
import csv
import hashlib
import re
from pathlib import Path

import openpyxl

COLUMNS = ["clip", "has_violation", "t_label", "t_start", "t_end",
           "type_id", "angle", "source", "note"]


def norm_name(name: str) -> str:
    stem = re.sub(r"\.mov$", "", str(name).strip(), flags=re.I)
    return re.sub(r"^[XxХх]", "X", stem)  # Cyrillic Х/х and Latin x -> Latin X


def clip_num(name: str) -> int:
    return int(name.split("-")[1])


def parse_seconds(s: str) -> float:
    s = s.lower()
    m = re.search(r"(\d+)\s*мин", s)
    sec = re.search(r"(\d+)\s*сек", s)
    if not (m or sec):
        raise ValueError(f"cannot parse time: {s!r}")
    return int(m.group(1)) * 60 + int(sec.group(1)) if m and sec else (
        int(m.group(1)) * 60 if m else int(sec.group(1)))


def _md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_labels(xlsx: Path, clips_dir: Path) -> list[dict]:
    # canonical clip per md5 (lowest number)
    by_hash: dict[str, list[str]] = {}
    for p in Path(clips_dir).glob("*.mov"):
        by_hash.setdefault(_md5(p), []).append(norm_name(p.stem))
    canon = {}
    for names in by_hash.values():
        best = min(names, key=clip_num)
        for n in names:
            canon[n] = best

    ws = openpyxl.load_workbook(xlsx).active
    events: dict[str, list[tuple[float | None, str]]] = {}
    for name, ts, verdict in list(ws.iter_rows(values_only=True))[1:]:
        if not name:
            continue
        orig = norm_name(name)
        clip = canon.get(orig, orig)
        note = f"merged from {orig}" if clip != orig else ""
        ev = events.setdefault(clip, [])
        if str(verdict).strip().startswith("наруш") and ts:
            ev.append((parse_seconds(str(ts)), note))

    rows = []
    for clip in sorted(events, key=clip_num):
        ev = sorted(events[clip], key=lambda e: e[0])
        base = {"t_start": "", "t_end": "", "type_id": "", "angle": "main", "source": "given"}
        if ev:
            for t, note in ev:
                t = int(t) if float(t).is_integer() else t
                rows.append({"clip": clip, "has_violation": 1, "t_label": t, **base, "note": note})
        else:
            rows.append({"clip": clip, "has_violation": 0, "t_label": "", **base, "note": ""})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, required=True)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = build_labels(a.xlsx, a.clips)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows, {len({r['clip'] for r in rows})} clips -> {a.out}")


if __name__ == "__main__":
    main()
