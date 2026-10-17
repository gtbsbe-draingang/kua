#!/usr/bin/env bash
# Make "other camera angle" copies of labelled clips: -tr, -hf, -crop, -dark.
# Usage: bash tools/augment.sh <clips_dir> <labels_csv> <out_dir> <out_labels_csv> [--only X-1,X-3] [-j N]
# Env:   FFMPEG=/path/to/ffmpeg   ffmpeg binary (default: ffmpeg)
#        VIDEO_ENCODER=name       force an encoder: libx264, libopenh264 or mpeg4.
#                                 Default: first available of libx264, libopenh264, mpeg4.
#                                 Servers without libx264: VIDEO_ENCODER=mpeg4 bash tools/augment.sh ...
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
case "$JOBS" in ''|*[!0-9]*|0) echo "augment: -j must be a positive integer" >&2; exit 2 ;; esac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
if "$PY" -c 'import os,sys; sys.exit(0 if os.path.realpath(sys.argv[1])==os.path.realpath(sys.argv[2]) else 1)' "$LABELS" "$OUT_LABELS"; then
  echo "augment: out_labels_csv must differ from labels_csv (would overwrite the source labels)" >&2; exit 2
fi
mkdir -p "$OUT"

FFMPEG="${FFMPEG:-ffmpeg}"
encoder_args() {  # encoder name -> ffmpeg video-encoder args
  case "$1" in
    libx264)     echo "-c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p" ;;
    libopenh264) echo "-c:v libopenh264 -b:v 4M -pix_fmt yuv420p" ;;
    mpeg4)       echo "-c:v mpeg4 -q:v 2 -pix_fmt yuv420p" ;;
    *) return 1 ;;
  esac
}
if [ -n "${VIDEO_ENCODER:-}" ]; then
  encoder_args "$VIDEO_ENCODER" >/dev/null || {
    echo "augment: VIDEO_ENCODER must be libx264, libopenh264 or mpeg4 (got '$VIDEO_ENCODER')" >&2; exit 2; }
  ENCODER="$VIDEO_ENCODER"
else
  ENCODER=""
  ENCODERS="$("$FFMPEG" -hide_banner -encoders 2>/dev/null || true)"
  for e in libx264 libopenh264 mpeg4; do
    if printf '%s\n' "$ENCODERS" | awk -v e="$e" '$2==e{found=1} END{exit !found}'; then
      ENCODER="$e"; break
    fi
  done
  [ -n "$ENCODER" ] || { echo "augment: no usable encoder in '$FFMPEG' (need libx264, libopenh264 or mpeg4)" >&2; exit 1; }
fi
echo "augment: encoder $ENCODER" >&2
export FFMPEG ENCODER
export -f encoder_args

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
  local -a enc; read -ra enc <<<"$(encoder_args "$ENCODER")"
  if "$FFMPEG" -nostdin -hide_banner -loglevel error -y -i "$src" -vf "$vf" \
       -threads 2 "${enc[@]}" -an -movflags +faststart "$tmp" \
     && mv "$tmp" "$dst"; then
    echo "made  $clip-$v"
  else
    rm -f "$tmp"; echo "augment: FAILED $clip-$v (ffmpeg error on $src)" >&2; return 1
  fi
}
export -f make_one

PLAN="$("$PY" "$HERE/augment_labels.py" plan "$CLIPS" "$LABELS" ${ONLY:+--only "$ONLY"})"
[ -n "$PLAN" ] || { echo "augment: no clips to process" >&2; exit 1; }

while IFS=$'\t' read -r clip _; do rm -f "$OUT/.$clip"-*.part.mov; done <<<"$PLAN"

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
