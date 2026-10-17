#!/usr/bin/env bash
# The RAG server (the regulation matcher, the pipeline's judge): servers.rag_server:app under uvicorn from env/main,
# on 127.0.0.1:30001, answering with the Qwen3.5-9B judge LLM that tools/judge_llm.sh serves on port 30005.
# Runs offline with every cache inside the root (env.sh).
#
#   bash rag_server.sh start    start the server in the background and return once its process is up. It loads bge-m3
#                               and embeds every clause before it opens the port, which takes a while: poll `status`.
#                               Refuses when something already listens on the port.
#   bash rag_server.sh status   pid and GET /healthz, which names the LLM and the retrieval mode that are really
#                               serving. Exit 0 healthy, 3 running but not ready, 1 not running.
#   bash rag_server.sh stop     TERM the server named in logs/rag.pid (after checking that the process is uvicorn
#                               running servers.rag_server:app on port 30001), KILL after 60 s. A server started by
#                               hand is stopped too when its pid is in the file. A pid file naming any other live
#                               process is removed and that process left alone. Then waits up to 15 s for the port to
#                               be free and says whether it is.
#
# The server's own environment, set here: RAG_LLM_KIND=openai, RAG_LLM_MODEL=qwen3.5-9b (the name judge_llm.sh serves
# the model under), RAG_LLM_URLS=http://127.0.0.1:30005/v1/chat/completions, OMP_NUM_THREADS=4. RAG_HYBRID is not set:
# the committed default (hybrid retrieval) applies, and a RAG_HYBRID in the caller's environment is passed on.
# The server itself is documented in servers/rag_server.py.
#
# Log: logs/rag.log (appended). Pid: logs/rag.pid.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
SELF="$HERE/$(basename "${BASH_SOURCE[0]}")"
SVC=$(cd "$HERE/.." && pwd -P)
MTA_ROOT=/home/jovyan/model_train_amir
# shellcheck source=/dev/null
source "$MTA_ROOT/env.sh"

PORT=30001
APP=servers.rag_server:app
JUDGE_PORT=30005                 # tools/judge_llm.sh
JUDGE_MODEL_NAME=qwen3.5-9b      # judge_llm.sh's SERVED_NAME

PY=$MTA_ROOT/env/main/bin/python
LOG=$MTA_ROOT/logs/rag.log
PIDFILE=$MTA_ROOT/logs/rag.pid
CTL=$MTA_ROOT/logs/rag.lock      # serialises start and stop (fd 8)

usage() { sed -n '2,/^set -euo pipefail/{/^#/s/^# \{0,1\}//p;}' "$SELF"; }
die() { echo "rag_server.sh: $*" >&2; exit 1; }

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

start() {
  local pid
  ctl_lock
  if pid=$(server_pid); then
    echo "already running (pid $pid)"
    return
  fi
  [ -x "$PY" ] || die "no env/main: $PY"
  ! port_taken || die "port $PORT is already in use by a process that logs/rag.pid does not name; find and stop it first"
  mkdir -p "$(dirname "$LOG")"
  echo "[$(date -u '+%F %T')] start rag server" >>"$LOG"
  cd "$SVC"
  # setsid: its own session, so it outlives the shell that started it. 8>&-: the server must not keep the lock.
  RAG_LLM_KIND=openai RAG_LLM_MODEL=$JUDGE_MODEL_NAME \
    RAG_LLM_URLS="http://127.0.0.1:$JUDGE_PORT/v1/chat/completions" OMP_NUM_THREADS=4 \
    setsid nohup "$PY" -m uvicorn "$APP" --host 127.0.0.1 --port "$PORT" --no-access-log \
    >>"$LOG" 2>&1 </dev/null 8>&- &
  pid=$!
  echo "$pid" >"$PIDFILE"
  sleep 3
  server_pid >/dev/null || die "the server did not stay up (see $LOG)"
  echo "started (pid $pid); log $LOG; poll: bash $SELF status"
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

case "${1:-}" in
  start|stop|status) "$1" ;;
  -h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
