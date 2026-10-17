#!/usr/bin/env bash
# Make "other camera angle" copies of labelled clips: -tr, -hf, -crop, -dark.
# Usage: bash tools/augment.sh <clips_dir> <labels_csv> <out_dir> <out_labels_csv> [--only X-1,X-3] [-j N]
set -euo pipefail

usage() { sed -n '2,3p' "$0" >&2; exit 2; }
ONLY=""; JOBS=4; POS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="${2:?--only needs a value}"; shift 2 ;;
    -j) JOBS="${2:?-j needs a value}"; shift 2 ;;
    -h|--help) usage ;;
    *) POS+=("$1"); shift ;;
  esac
done
[ ${#POS[@]} -eq 4 ] || usage
CLIPS="${POS[0]}"; LABELS="${POS[1]}"; OUT="${POS[2]}"; OUT_LABELS="${POS[3]}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
mkdir -p "$OUT"

vf_for() {
  case "$1" in
    tr) echo "transpose=1" ;;
    hf) echo "hflip" ;;
    crop) echo "crop=trunc(iw*0.8/2)*2:trunc(ih*0.8/2)*2,scale=1920:1080:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1" ;;
    dark) echo "eq=brightness=-0.15" ;;
  esac
}

make_one() {  # clip src variant filter outdir
  local clip="$1" src="$2" v="$3" vf="$4" out="$5"
  local dst="$out/$clip-$v.mov" tmp="$out/.$clip-$v.part.mov"
  if [ -s "$dst" ] && ffprobe -v error -show_entries format=duration -of csv=p=0 "$dst" >/dev/null 2>&1; then
    echo "skip  $clip-$v"; return 0
  fi
  rm -f "$tmp"
  if ffmpeg -nostdin -hide_banner -loglevel error -y -threads 2 -i "$src" -vf "$vf" \
       -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an -movflags +faststart "$tmp" \
     && mv "$tmp" "$dst"; then
    echo "made  $clip-$v"
  else
    rm -f "$tmp"; echo "augment: FAILED $clip-$v (ffmpeg error on $src)" >&2; return 1
  fi
}
export -f make_one

PLAN="$("$PY" "$HERE/augment_labels.py" plan "$CLIPS" "$LABELS" ${ONLY:+--only "$ONLY"})"
[ -n "$PLAN" ] || { echo "augment: no clips to process" >&2; exit 1; }

jobs_file="$(mktemp)"; trap 'rm -f "$jobs_file"' EXIT
while IFS=$'\t' read -r clip src; do
  for v in tr hf crop dark; do
    printf '%s\0%s\0%s\0%s\0%s\0' "$clip" "$src" "$v" "$(vf_for "$v")" "$OUT" >>"$jobs_file"
  done
done <<<"$PLAN"

if ! xargs -0 -n 5 -P "$JOBS" bash -c 'make_one "$@"' _ <"$jobs_file"; then
  echo "augment: one or more variants failed; labels csv not written" >&2
  exit 1
fi

"$PY" "$HERE/augment_labels.py" write "$LABELS" "$OUT_LABELS" ${ONLY:+--only "$ONLY"}
