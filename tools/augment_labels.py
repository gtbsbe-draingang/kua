"""CSV and file-resolution helpers for tools/augment.sh (stdlib only)."""
import argparse
import csv
import sys
from pathlib import Path

VARIANTS = ("tr", "hf", "crop", "dark")


def norm(name):
    """Normalise Cyrillic Х/х (U+0425/U+0445) to Latin X."""
    return name.replace("Х", "X").replace("х", "X")


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r), r.fieldnames


def unique_clips(rows, only=None):
    seen = []
    for r in rows:
        if r["clip"] not in seen:
            seen.append(r["clip"])
    if only:
        missing = [c for c in only if c not in seen]
        if missing:
            sys.exit(f"augment: --only clips not in labels: {', '.join(missing)}")
        seen = [c for c in seen if c in only]
    return seen


def resolve(clips_dir, clip):
    hits = [p for p in Path(clips_dir).iterdir() if p.suffix.lower() == ".mov" and norm(p.stem) == clip]
    if len(hits) != 1:
        sys.exit(f"augment: {clip}: expected one .mov in {clips_dir}, found {len(hits)}")
    return hits[0]


def augment_rows(rows, only=None):
    keep = set(unique_clips(rows, only))
    out = []
    for r in rows:
        if r["clip"] not in keep:
            continue
        for v in VARIANTS:
            out.append({**r, "clip": f"{r['clip']}-{v}", "angle": "other",
                        "source": "augmented", "note": v})
    return out


def write_labels(src, dst, only=None):
    rows, fields = read_rows(src)
    out = augment_rows(rows, only)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    return len(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="print 'clip<TAB>path' per unique clip")
    p.add_argument("clips_dir"); p.add_argument("labels"); p.add_argument("--only", default="")
    q = sub.add_parser("write", help="write the augmented labels csv")
    q.add_argument("labels"); q.add_argument("out"); q.add_argument("--only", default="")
    a = ap.parse_args(argv)
    only = [c for c in a.only.split(",") if c] or None
    if a.cmd == "plan":
        rows, _ = read_rows(a.labels)
        for c in unique_clips(rows, only):
            print(f"{c}\t{resolve(a.clips_dir, c)}")
    else:
        n = write_labels(a.labels, a.out, only)
        print(f"augment: wrote {n} label rows to {a.out}")


if __name__ == "__main__":
    main()
