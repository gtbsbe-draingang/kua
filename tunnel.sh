#!/usr/bin/env bash
# Public URL: an ngrok endpoint on the account's free dev domain, from this pod
# to the API on 127.0.0.1:8000.
#
#   bash tunnel.sh install       put the pinned ngrok agent in env/tunnel/bin (sha256-checked)
#   bash tunnel.sh start         start the supervised agent in the background; returns at once
#   bash tunnel.sh wait [SECS]   wait until the endpoint is online (default 30 s; exit 3 if not)
#   bash tunnel.sh stop          stop the loop and the agent (TERM, then KILL after 45 s)
#   bash tunnel.sh status        loop, agent, endpoint from the agent's local API, log tail
#
# The authtoken lives in secrets/ngrok.yml (mode 600, holding only "version: 3"
# and "agent: authtoken: ..."), passed with --config, so it is never on a command
# line or in the environment. The agent's local API stays on 127.0.0.1:4040 and
# request inspection is off, so the agent keeps no uploaded video in memory.
#
# The agent runs with HOME=env/tunnel, so nothing is written outside the root.
# Logs go to logs/tunnel.log. A loop restarts the agent when it exits, backing off
# from 5 s to 60 s. run.sh calls `start` once the API is up; a tunnel that cannot
# connect never keeps the API from starting.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
SELF="$HERE/$(basename "${BASH_SOURCE[0]}")"
# shellcheck source=env.sh
source "$HERE/env.sh"

VERSION=3.39.11
SHA256=e51aa234283bcf20a777e21a624b2a3501b058b3d5a1faec59106311dbe30395
DEB=https://ngrok-agent.s3.amazonaws.com/pool/main/n/ngrok/ngrok_${VERSION}-0_amd64.deb

PUBLIC_URL=https://frida-parallelepipedous-cubicly.ngrok-free.dev
ORIGIN=http://127.0.0.1:8000
AGENT_API=127.0.0.1:4040   # the agent's default web_addr

HOME_DIR=$MTA_ROOT/env/tunnel
BIN=$HOME_DIR/bin/ngrok
CONF=$MTA_ROOT/secrets/ngrok.yml
PIDFILE=$HOME_DIR/ngrok.pid
LOCK=$HOME_DIR/ngrok.lock
LOG=$MTA_ROOT/logs/tunnel.log

usage() { sed -n '2,/^set -euo pipefail/{/^#/s/^# \{0,1\}//p;}' "$SELF"; }
die() { echo "tunnel.sh: $*" >&2; exit 1; }

install() {
  local deb=$HOME_DIR/bin/ngrok.deb.part
  if [ -x "$BIN" ] && [ "$(HOME=$HOME_DIR "$BIN" version 2>/dev/null)" = "ngrok version $VERSION" ]; then
    echo "ngrok $VERSION already installed: $BIN"
    return
  fi
  mkdir -p "$HOME_DIR/bin"
  curl -fsSL --retry 3 -o "$deb" "$DEB" || { rm -f "$deb"; die "download failed: $DEB"; }
  echo "$SHA256  $deb" | sha256sum -c --quiet - >/dev/null 2>&1 \
    || { rm -f "$deb"; die "checksum mismatch for $DEB"; }
  dpkg-deb --fsys-tarfile "$deb" | tar -xO ./usr/local/bin/ngrok >"$BIN.part"
  rm -f "$deb"
  chmod 755 "$BIN.part"
  mv -f "$BIN.part" "$BIN"
  HOME=$HOME_DIR "$BIN" version
}

# The loop's pid, if that process is alive and really is our loop (pids get reused).
loop_pid() {
  local pid
  [ -s "$PIDFILE" ] || return 1
  pid=$(<"$PIDFILE")
  [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' <"/proc/$pid/cmdline" | grep -qF "$SELF _loop" || return 1
  echo "$pid"
}

# Agents of our own binary, supervised or not.
agent_pids() { pgrep -d ' ' -f "^$BIN " || true; }

# The public URL the running agent serves, from its local API.
endpoint() {
  curl -fsS --noproxy '*' -m 3 "http://$AGENT_API/api/tunnels" 2>/dev/null \
    | grep -o '"public_url":"[^"]*"' | head -n 1 | cut -d'"' -f4
}

# TERM every agent of our binary, wait up to 45 s, then KILL what is left.
reap_agents() {
  [ -n "$(agent_pids)" ] || return 0
  pkill -TERM -f "^$BIN " 2>/dev/null || true
  for _ in $(seq 45); do
    [ -n "$(agent_pids)" ] || return 0
    sleep 1
  done
  pkill -KILL -f "^$BIN " 2>/dev/null || true
  sleep 1
}

start() {
  local pid
  [ -x "$BIN" ] || die "ngrok missing: run 'bash tunnel.sh install' first"
  [ -s "$CONF" ] || die "no agent config with the authtoken in $CONF"
  [ "$(stat -c %a "$CONF")" = 600 ] || die "$CONF must be mode 600"
  if pid=$(loop_pid); then
    echo "already running (loop pid $pid)"
    return
  fi
  # No loop, so any agent left is an orphan (its loop was killed) or still
  # draining after a stop. It holds the endpoint, so a new agent would be refused.
  reap_agents
  [ -z "$(agent_pids)" ] || die "an old agent survived KILL: $(agent_pids)"
  mkdir -p "$(dirname "$LOG")"
  setsid nohup bash "$SELF" _loop >>"$LOG" 2>&1 </dev/null &
  echo "started (loop pid $!)"
}

# Runs detached (from start). On TERM it empties the pid file at once, so a loop
# that is shutting down never counts as running, then lets the agent drain.
_loop() {
  local child='' stopping=0 delay=5 started rc
  exec 9>"$LOCK"
  flock -w 10 9 || { echo "$(date -Is) another loop holds $LOCK; exiting"; exit 0; }
  echo $$ >"$PIDFILE"
  trap 'stopping=1; : >"$PIDFILE"; [ -z "$child" ] || kill -TERM "$child" 2>/dev/null || true' TERM
  while [ "$stopping" = 0 ]; do
    started=$(date +%s)
    HOME=$HOME_DIR "$BIN" http "$ORIGIN" --url "$PUBLIC_URL" --config "$CONF" \
      --inspect=false --log stdout --log-format logfmt 9>&- &
    child=$!
    # A TERM that landed between the fork and the line above found no child.
    [ "$stopping" = 0 ] || kill -TERM "$child" 2>/dev/null || true
    rc=0
    wait "$child" || rc=$?
    # The trap interrupts wait; wait again for the agent to finish draining.
    [ "$stopping" = 0 ] || { wait "$child" 2>/dev/null || true; break; }
    child=''
    [ $(( $(date +%s) - started )) -lt 60 ] || delay=5
    echo "$(date -Is) ngrok exited (rc $rc), restarting in $delay s"
    sleep "$delay" 9>&- &
    child=$!
    wait "$child" || true
    child=''
    delay=$(( delay * 2 > 60 ? 60 : delay * 2 ))
  done
  exit 0
}

wait_online() {
  local secs=${1:-30}
  for _ in $(seq "$secs"); do
    if [ "$(endpoint || true)" = "$PUBLIC_URL" ]; then
      echo "online: $PUBLIC_URL"
      return
    fi
    sleep 1
  done
  echo "tunnel.sh: not online after $secs s (see $LOG)" >&2
  return 3
}

stop() {
  local pid
  if pid=$(loop_pid); then
    kill -TERM "$pid" 2>/dev/null || true
  fi
  reap_agents
  # The loop exits as soon as its agent is gone.
  for _ in $(seq 5); do
    loop_pid >/dev/null || break
    sleep 1
  done
  if pid=$(loop_pid); then
    kill -KILL "$pid" 2>/dev/null || true
  fi
  if loop_pid >/dev/null || [ -n "$(agent_pids)" ]; then
    die "still running after KILL"
  fi
  echo "stopped"
}

status() {
  local pid url
  if pid=$(loop_pid); then echo "loop pid $pid"; else echo "loop: not running"; fi
  echo "agent pid(s): $(agent_pids)"
  if url=$(endpoint) && [ -n "$url" ]; then echo "endpoint: $url"; else echo "endpoint: offline"; fi
  if [ -f "$LOG" ]; then tail -n 5 "$LOG"; fi
  return 0
}

case "${1:-}" in
  install|start|stop|status|_loop) "$1" ;;
  wait) wait_online "${2:-30}" ;;
  -h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
