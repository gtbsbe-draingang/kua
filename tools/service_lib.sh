# shellcheck shell=bash
# The part that tools/rag_server.sh and tools/api_server.sh share: a uvicorn server that runs from env/main as a
# detached process, found again through a pid file. Source it from a script that has done `set -euo pipefail` and set
#
#   SELF     the script's own path (usage and error messages)
#   PORT     the port the server listens on (127.0.0.1)
#   APP      the uvicorn app, e.g. api:app
#   PY       the interpreter that runs it: the first word of its command line
#   PIDFILE  where start writes the pid; LOG the server's log; CTL the lock file that serialises start and stop
#
# and define its own start(). This file defines usage, die, is_our_cmdline, server_pid, alive, port_taken,
# wait_gone, ctl_lock, stop and status. tools/judge_llm.sh is a different kind of server and does not use it.

usage() { sed -n '2,/^set -euo pipefail/{/^#/s/^# \{0,1\}//p;}' "$SELF"; }
die() { echo "${SELF##*/}: $*" >&2; exit 1; }

# True when the command line (the arguments joined by spaces, as tr makes it of /proc/PID/cmdline) is this server's:
# our interpreter running `-m uvicorn <app>` with `--port <port>`. Whole words, not substrings: neither another app
# nor --port 300010 matches.
is_our_cmdline() {
  local cmd=$1
  [[ $cmd == "$PY -m uvicorn $APP "* && " $cmd" == *" --port $PORT "* ]]
}

# The pid in the pid file, if that process is alive and really is our server (pids get reused). A zombie, which
# nothing may have reaped (the pod's pid 1 does not), has no command line and so is not the server.
server_pid() {
  local pid cmd
  [ -s "$PIDFILE" ] || return 1
  pid=$(<"$PIDFILE")
  [[ $pid =~ ^[1-9][0-9]*$ ]] || return 1
  [ -r "/proc/$pid/cmdline" ] || return 1
  cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline") || return 1
  is_our_cmdline "$cmd" || return 1
  echo "$pid"
}

# True while pid runs (a zombie does not).
alive() {
  local state
  state=$(sed 's/^.*) //' "/proc/$1/stat" 2>/dev/null | cut -c1) || return 1
  [ -n "$state" ] && [ "$state" != Z ]
}

# True when something listens on the port.
port_taken() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }

# Wait up to $1 seconds until the pid file no longer names a live server of ours: it exited, or its pid now belongs
# to another process.
wait_gone() {
  local _
  for _ in $(seq "$1"); do
    server_pid >/dev/null || return 0
    sleep 1
  done
  ! server_pid >/dev/null
}

ctl_lock() {
  mkdir -p "$(dirname "$CTL")"
  exec 8>"$CTL"
  flock -w 120 8 || die "another start or stop has been running for 2 minutes"
}

stop() {
  local pid='' msg _
  ctl_lock
  if [ -s "$PIDFILE" ]; then pid=$(<"$PIDFILE"); fi
  [[ $pid =~ ^[1-9][0-9]*$ ]] || pid=''   # a signal is only ever sent to a checked pid: never 0 (the caller's group)
  if [ -z "$pid" ]; then
    msg="no server process was running (no usable pid file)"
  elif [ "$(server_pid || true)" = "$pid" ]; then   # alive, and its command line is our server's
    kill -TERM "$pid" 2>/dev/null || true
    if ! wait_gone 60; then
      echo "pid $pid ignored TERM for 60 s: KILL"
      kill -KILL "$pid" 2>/dev/null || true   # wait_gone just found the pid file still naming our server
      wait_gone 10 || die "pid $pid is still running after KILL; pid file kept"
    fi
    msg="stopped pid $pid"
  elif alive "$pid"; then
    msg="no server process was running (pid $pid in the pid file is another process, left alone)"
  else
    msg="no server process was running (pid $pid in the pid file is gone)"
  fi
  rm -f "$PIDFILE"
  for _ in $(seq 15); do
    port_taken || break
    sleep 1
  done
  if port_taken; then
    echo "$msg; port $PORT is still in use by another process"
  else
    echo "$msg; port $PORT is free"
  fi
}

status() {
  local pid body
  if ! pid=$(server_pid); then
    echo "not running"
    return 1
  fi
  if body=$(curl -fsS --noproxy '*' -m 3 "http://127.0.0.1:$PORT/healthz" 2>/dev/null); then
    echo "running (pid $pid), healthy: $body"
    return 0
  fi
  echo "running (pid $pid), not ready yet (GET /healthz fails); log tail:"
  { tail -n 3 "$LOG" 2>/dev/null | cut -c1-300; } || true   # a missing log must not change the exit code
  return 3
}
