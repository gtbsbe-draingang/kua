#!/usr/bin/env bash
# Push this checkout to the pod's /home/jovyan/model_train_amir/svc.
#
#   tools/push.sh                       tracked files (git ls-files) as one tarball
#   tools/push.sh --clips               the same, plus every .mov from the clips
#                                       folder to model_train_amir/data/clips/
#   tools/push.sh --clips-dir DIR       the same as --clips, with another folder
#
# It ships the working tree of the tracked files: uncommitted edits to tracked
# files go along, untracked files never do. It never deletes anything on the
# pod: the tarball is extracted over svc/, so files that exist only on the pod
# (including files since removed from git) stay until removed by hand. The
# root's env.sh is (re)linked to svc/env.sh. Clips keep their file names, and a
# clip whose md5 on the pod already matches is skipped.
#
# The clips folder defaults to <repo root>/../open/videos. The pod helpers
# (jx.py, jup_upload.py) are found in $POD_HELPERS_DIR, which defaults to
# ~/Coding/armeta/ge-2052-rpn-classifier/tools; the helpers read the access
# token themselves.
set -euo pipefail

usage() { sed -n '2,/^set -euo pipefail/{/^#/s/^# \{0,1\}//p;}' "$0"; }

REPO=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
HELPERS=${POD_HELPERS_DIR:-$HOME/Coding/armeta/ge-2052-rpn-classifier/tools}
CLIPS=0
CLIPS_DIR=$REPO/../open/videos

while [ $# -gt 0 ]; do
  case $1 in
    --clips) CLIPS=1 ;;
    --clips-dir) [ $# -ge 2 ] || { usage >&2; exit 2; }; CLIPS=1; CLIPS_DIR=$2; shift ;;
    --clips-dir=*) CLIPS=1; CLIPS_DIR=${1#*=} ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

for f in jx.py jup_upload.py; do
  [ -f "$HELPERS/$f" ] || { echo "missing $HELPERS/$f (set POD_HELPERS_DIR)" >&2; exit 1; }
done
if [ "$CLIPS" = 1 ] && [ ! -d "$CLIPS_DIR" ]; then
  echo "clips folder not found: $CLIPS_DIR" >&2; exit 1
fi

IFS= read -r -d '' PUSH_PY <<'PY' || true
import hashlib, re, subprocess, sys, tarfile, tempfile, unicodedata
from pathlib import Path

repo, helpers, clips_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[4])
want_clips = sys.argv[3] == "1"
POD = "/home/jovyan/model_train_amir"   # absolute, for shell commands on the pod
REL = "model_train_amir"                # the same folder relative to /home/jovyan, for uploads
DONE = "__push_step_done__"


def remote(script, timeout=300):
    """Run a bash script on the pod and return its output; fail unless it ran to the end.
    The root's env.sh is sourced first when it exists, so temp files stay inside the root."""
    prelude = f"set -euo pipefail\nif [ -f {POD}/env.sh ]; then . {POD}/env.sh; fi\n"
    p = subprocess.run(
        [sys.executable, str(helpers / "jx.py"), "-s", "-t", str(timeout)],
        input=f"{prelude}{script}\necho {DONE}\n", capture_output=True, text=True)
    lines = p.stdout.splitlines()
    if p.returncode != 0 or DONE not in lines:
        sys.stderr.write(p.stdout + p.stderr)
        sys.exit("push: a command on the pod failed")
    return [line for line in lines if line != DONE]


def upload(local, remote_rel):
    p = subprocess.run([sys.executable, str(helpers / "jup_upload.py"), str(local), remote_rel],
                       capture_output=True, text=True)
    if p.returncode != 0:
        sys.stderr.write(p.stdout + p.stderr)
        sys.exit(f"push: upload failed: {local}")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def nfc(name):
    return unicodedata.normalize("NFC", name)


def push_code():
    listing = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"],
                             capture_output=True, check=True).stdout.decode()
    files = [f for f in listing.split("\0") if f and ((repo / f).exists() or (repo / f).is_symlink())]

    def owner_neutral(info):
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        return info

    with tempfile.TemporaryDirectory() as tmp:
        tgz = Path(tmp) / "svc.tgz"
        with tarfile.open(tgz, "w:gz", format=tarfile.PAX_FORMAT) as tar:
            for f in files:
                tar.add(repo / f, arcname=f, recursive=False, filter=owner_neutral)
        digest = md5(tgz)
        upload(tgz, f"{REL}/cache/push/svc.tgz")
    out = remote(f"""
cd {POD}
echo "{digest}  cache/push/svc.tgz" | md5sum -c --quiet
mkdir -p svc
tar -xzf cache/push/svc.tgz -C svc --no-same-owner
if [ -e env.sh ] && [ ! -L env.sh ]; then
  echo "{POD}/env.sh is a regular file, not the symlink; left it alone" >&2; exit 1
fi
ln -sfn svc/env.sh env.sh
echo "env.sh -> $(readlink env.sh)"
""")
    print(f"code: {len(files)} tracked files pushed to {POD}/svc (tarball md5 {digest})")
    for line in out:
        print("  " + line)


def remote_md5s():
    out = remote(f"mkdir -p {POD}/data/clips && cd {POD}/data/clips && "
                 "find . -maxdepth 1 -type f -iname '*.mov' -print0 | xargs -0 -r md5sum")
    found = {}
    for line in out:
        m = re.fullmatch(r"([0-9a-f]{32})  \./(.+)", line)
        if m:
            found[nfc(m.group(2))] = m.group(1)
    return found


def push_clips():
    local = sorted(p for p in clips_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mov")
    if not local:
        sys.exit(f"push: no .mov files in {clips_dir}")
    sums = {nfc(p.name): md5(p) for p in local}
    have = remote_md5s()
    todo = [p for p in local if have.get(nfc(p.name)) != sums[nfc(p.name)]]
    print(f"clips: {len(local)} local, {len(local) - len(todo)} already on the pod, {len(todo)} to upload")
    for i, p in enumerate(todo, 1):
        upload(p, f"{REL}/data/clips/{nfc(p.name)}")
        print(f"  [{i}/{len(todo)}] {nfc(p.name)}  {p.stat().st_size / 1e6:.1f} MB", flush=True)
    have = remote_md5s()
    bad = [name for name, digest in sums.items() if have.get(name) != digest]
    if bad:
        sys.exit(f"push: md5 differs on the pod for {len(bad)} clips: {', '.join(bad)}")
    print(f"clips: all {len(sums)} match their local md5 on the pod "
          f"({sum(p.stat().st_size for p in local) / 1e6:.0f} MB)")


push_code()
if want_clips:
    push_clips()
PY

exec uv run --no-project --quiet --with requests --with websocket-client \
  python -c "$PUSH_PY" "$REPO" "$HELPERS" "$CLIPS" "$CLIPS_DIR"
