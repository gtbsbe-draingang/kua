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
# from 5 s to 60 s. After a hard kill, ngrok keeps the old session for a while and
# refuses the new agent until it expires; the loop's backoff rides that out, so
# `wait` can exit 3 for a minute or so. run.sh calls `start` once the API is up and
# treats a failed `wait` as a warning: the tunnel never keeps the API from starting.
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
LOCK=$HOME_DIR/ngrok.lock   # held by the running loop on fd 9: the real liveness test
CTL=$HOME_DIR/ngrok.ctl     # serialises start and stop (fd 8)
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
  dpkg-deb --fsys-tarfile "$deb" | tar -xO ./usr/local/bin/ngrok >"$BIN.part" \
    || { rm -f "$deb" "$BIN.part"; die "could not extract ngrok from $deb"; }
  rm -f "$deb"
  chmod 755 "$BIN.part"
  mv -f "$BIN.part" "$BIN"
  HOME=$HOME_DIR "$BIN" version
}

# The loop's pid, if that process is alive and really is our loop (pids get reused).
# A loop that is shutting down has emptied the file, so it no longer counts.
loop_pid() {
  local pid
  [ -s "$PIDFILE" ] || return 1
  pid=$(<"$PIDFILE")
  [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' <"/proc/$pid/cmdline" | grep -qF "$SELF _loop" || return 1
  echo "$pid"
}

# Pids of our loops that have the lock file open on fd 9: the holder, plus any
# loop still waiting for it (and a loop's short-lived forks).
lock_openers() {
  local fd pid
  for fd in /proc/[0-9]*/fd/9; do
    [ "$fd" -ef "$LOCK" ] || continue
    pid=${fd#/proc/}
    pid=${pid%%/*}
    tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | grep -qF "$SELF _loop" || continue
    echo "$pid"
  done
}

# Agents of our own binary, supervised or not.
agent_pids() { pgrep -d ' ' -f "^$BIN " || true; }

# The public URL the running agent serves, from its local API.
endpoint() {
  curl -fsS --noproxy '*' -m 3 "http://$AGENT_API/api/tunnels" 2>/dev/null \
    | grep -o '"public_url" *: *"[^"]*"' | sed -n 1p | cut -d'"' -f4
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

ctl_lock() {
  exec 8>"$CTL"
  flock -w 120 8 || die "another start or stop has been running for 2 minutes"
}

start() {
  local pid
  [ -x "$BIN" ] || die "ngrok missing: run 'bash tunnel.sh install' first"
  [ -s "$CONF" ] || die "no agent config with the authtoken in $CONF"
  [ "$(stat -c %a "$CONF")" = 600 ] || die "$CONF must be mode 600"
  ctl_lock
  if pid=$(loop_pid); then
    echo "already running (loop pid $pid)"
    return
  fi
  # No live loop, so any agent left is an orphan (its loop was killed) or still
  # draining after a stop. It holds the endpoint, so a new agent would be refused.
  reap_agents
  [ -z "$(agent_pids)" ] || die "an old agent survived KILL: $(agent_pids)"
  mkdir -p "$(dirname "$LOG")"
  setsid nohup bash "$SELF" _loop 8>&- >>"$LOG" 2>&1 </dev/null &
  # Hold the control lock until the loop has its lock and pid file, so a second
  # start or a stop right behind this one sees it.
  for _ in $(seq 75); do
    if pid=$(loop_pid); then
      echo "started (loop pid $pid)"
      return
    fi
    sleep 0.2
  done
  die "the loop did not come up within 15 s (see $LOG)"
}

# Runs detached (from start). On TERM it empties the pid file at once, so a loop
# that is shutting down never counts as running, then lets the agent drain. It
# keeps the lock until the agent is gone.
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
    if [ "$stopping" = 1 ]; then
      # The trap interrupts wait (a second TERM interrupts it again): keep
      # waiting until the agent has finished draining.
      while kill -0 "$child" 2>/dev/null; do wait "$child" 2>/dev/null || true; done
      break
    fi
    child=''
    [ $(( $(date +%s) - started )) -lt 60 ] || delay=5
    echo "$(date -Is) ngrok exited (rc $rc), restarting in $delay s"
    # A TERM that arrived while no child was set has already ended the loop.
    [ "$stopping" = 0 ] || break
    sleep "$delay" 9>&- &
    child=$!
    [ "$stopping" = 0 ] || kill -TERM "$child" 2>/dev/null || true
    wait "$child" || true
    child=''
    delay=$(( delay * 2 > 60 ? 60 : delay * 2 ))
  done
  exit 0
}

wait_online() {
  local secs=$1 i
  [[ $secs =~ ^[0-9]+$ ]] || die "wait: seconds must be a whole number, got '$secs'"
  secs=$((10#$secs))   # "08" is not octal
  for (( i = 0; i <= secs; i++ )); do
    if [ "$(endpoint || true)" = "$PUBLIC_URL" ]; then
      echo "online: $PUBLIC_URL"
      return
    fi
    if [ "$i" -lt "$secs" ]; then sleep 1; fi
  done
  echo "tunnel.sh: not online after $secs s (see $LOG)" >&2
  return 3
}

stop() {
  local pid
  ctl_lock
  if pid=$(loop_pid); then
    kill -TERM "$pid" 2>/dev/null || true
  fi
  reap_agents
  # The loop exits as soon as its agent is gone and releases the lock.
  if ! flock -w 10 "$LOCK" true; then
    for pid in $(lock_openers); do
      kill -KILL "$pid" 2>/dev/null || true
    done
    reap_agents
    flock -w 5 "$LOCK" true || die "a loop still holds $LOCK"
  fi
  [ -z "$(agent_pids)" ] || die "an agent survived KILL: $(agent_pids)"
  echo "stopped"
}

status() {
  local pid url agents
  if pid=$(loop_pid); then echo "loop pid $pid"; else echo "loop: not running"; fi
  agents=$(agent_pids)
  echo "agent pid(s): ${agents:-none}"
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
