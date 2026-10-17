#!/usr/bin/env bash
# Public URL: a Cloudflare named tunnel from this pod to the API on 127.0.0.1:8000.
#
#   bash tunnel.sh install   put the pinned cloudflared in env/tunnel/bin (sha256-checked)
#   bash tunnel.sh start     run it in the background, restarted whenever it exits
#   bash tunnel.sh stop      stop the restart loop and cloudflared
#   bash tunnel.sh status    loop pid, edge connections (metrics /ready), log tail
#
# The tunnel and its DNS record are created on the laptop (cloudflared tunnel
# create / route dns). Only the tunnel token comes to the pod, in
# secrets/tunnel_token (mode 600), read with --token-file so it is never on a
# command line or in the environment. The origin is given with --url, so the pod
# never learns the public hostname; calls made from the pod go to localhost.
#
# Every path stays inside the root: cloudflared runs with HOME=env/tunnel, so its
# default config lookup (~/.cloudflared) never reads or writes the real home.
# Logs go to logs/tunnel.log. Override the metrics address with TUNNEL_METRICS.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SELF="$HERE/$(basename "${BASH_SOURCE[0]}")"
# shellcheck source=env.sh
source "$HERE/env.sh"

VERSION=2026.9.3
SHA256=77e26d8d900e0b8469f416239d14b5f296525fdf79fee6f511ef55609e3fbac2
URL=https://github.com/cloudflare/cloudflared/releases/download/$VERSION/cloudflared-linux-amd64

HOME_DIR=$MTA_ROOT/env/tunnel
BIN=$HOME_DIR/bin/cloudflared
TOKEN=$MTA_ROOT/secrets/tunnel_token
PIDFILE=$HOME_DIR/loop.pid
LOG=$MTA_ROOT/logs/tunnel.log
ORIGIN=http://127.0.0.1:8000
METRICS=${TUNNEL_METRICS:-127.0.0.1:20241}

die() { echo "tunnel.sh: $*" >&2; exit 1; }

install() {
  if [ -x "$BIN" ] && echo "$SHA256  $BIN" | sha256sum -c --quiet - 2>/dev/null; then
    echo "cloudflared $VERSION already installed: $BIN"
    return
  fi
  mkdir -p "$(dirname "$BIN")"
  curl -fsSL --retry 3 -o "$BIN.part" "$URL"
  echo "$SHA256  $BIN.part" | sha256sum -c --quiet - || die "checksum mismatch for $URL"
  chmod 755 "$BIN.part"
  mv -f "$BIN.part" "$BIN"
  "$BIN" --version
}

# The loop's pid, if that process is alive and really is our loop (pids get reused).
loop_pid() {
  local pid
  [ -s "$PIDFILE" ] || return 1
  pid=$(<"$PIDFILE")
  [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' <"/proc/$pid/cmdline" | grep -qF "$SELF _loop" || return 1
  echo "$pid"
}

ready() { curl -fsS -m 3 "http://$METRICS/ready" 2>/dev/null; }

start() {
  [ -x "$BIN" ] || die "cloudflared missing: run 'bash tunnel.sh install' first"
  [ -s "$TOKEN" ] || die "no tunnel token in $TOKEN"
  [ "$(stat -c %a "$TOKEN")" = 600 ] || die "$TOKEN must be mode 600"
  if pid=$(loop_pid); then
    echo "already running (loop pid $pid)"
    return
  fi
  mkdir -p "$(dirname "$LOG")"
  setsid nohup bash "$SELF" _loop >>"$LOG" 2>&1 </dev/null &
  for _ in $(seq 30); do
    if ready >/dev/null; then status; return; fi
    sleep 1
  done
  status
  die "no edge connection after 30 s (see $LOG)"
}

# Runs detached (from start). A TERM from stop reaches cloudflared, which drains
# in-flight requests for its grace period, and then the loop exits.
_loop() {
  local child='' rc
  echo $$ >"$PIDFILE"
  trap '[ -n "$child" ] && kill -TERM "$child" 2>/dev/null; wait "$child" 2>/dev/null; exit 0' TERM INT
  while :; do
    HOME=$HOME_DIR "$BIN" tunnel --no-autoupdate --metrics "$METRICS" \
      run --token-file "$TOKEN" --url "$ORIGIN" &
    child=$!
    rc=0
    wait "$child" || rc=$?
    echo "$(date -Is) cloudflared exited (rc $rc), restarting in 5 s"
    sleep 5
  done
}

stop() {
  local pid
  if pid=$(loop_pid); then
    kill -TERM "$pid"
    for _ in $(seq 40); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
  fi
  # A cloudflared left behind by a killed loop: match only our own binary's path.
  pkill -TERM -f "^$BIN tunnel" 2>/dev/null || true
  if loop_pid >/dev/null || pgrep -f "^$BIN tunnel" >/dev/null; then
    die "still running after 40 s"
  fi
  echo "stopped"
}

status() {
  local pid
  if pid=$(loop_pid); then
    echo "loop pid $pid; cloudflared pid(s): $(pgrep -d ' ' -f "^$BIN tunnel" || echo none)"
  else
    echo "not running"
  fi
  echo "ready: $(ready || echo 'no (metrics unreachable or no edge connection)')"
  [ -f "$LOG" ] && tail -n 5 "$LOG"
  return 0
}

case "${1:-}" in
  install|start|stop|status|_loop) "$1" ;;
  *) sed -n '2,/^set -euo pipefail/{/^#/s/^# \{0,1\}//p;}' "$0"; exit 2 ;;
esac
