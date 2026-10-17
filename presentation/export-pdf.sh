#!/usr/bin/env bash
# Export a deck to a 16:9 PDF (one 1920x1080 page per slide) with headless Chrome.
# Usage: ./export-pdf.sh [deck.html] [out.pdf]     defaults: index.html -> pitch.pdf
set -euo pipefail
cd "$(dirname "$0")"

in="${1:-index.html}"
out="${2:-pitch.pdf}"

if [[ -n "${CHROME:-}" ]]; then
  chrome="$CHROME"
elif [[ -x "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ]]; then
  chrome="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
else
  chrome="$(command -v google-chrome || command -v chromium || command -v chromium-browser || true)"
fi
[[ -n "$chrome" ]] || { echo "Chrome not found; set CHROME=/path/to/chrome" >&2; exit 1; }

"$chrome" --headless=new --disable-gpu --no-pdf-header-footer \
  --virtual-time-budget=10000 --run-all-compositor-stages-before-draw \
  --print-to-pdf="$PWD/$out" "file://$PWD/$in?print" 2>/dev/null

echo "wrote $out"
