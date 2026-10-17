#!/usr/bin/env bash
# The API (api:app under uvicorn from env/main, on 127.0.0.1:8000): the web UI, POST /api/v1/analyze and the video
# pipeline (VideoJev in this process, InternVideo3 on the four replicas, the RAG server as the judge). The public
# URL (tunnel.sh) forwards to it. Runs offline with every cache inside the root (env.sh).
#
#   bash api_server.sh start    start the API in the background and return once its process is up. It loads VideoJev
#                               onto the GPU before it opens the port, which takes a while: poll `status`.
#                               Refuses when something already listens on the port.
#   bash api_server.sh status   pid and GET /healthz. Exit 0 healthy, 3 running but not ready, 1 not running.
#   bash api_server.sh stop     TERM the server named in logs/api.pid (after checking that the process is uvicorn
#                               running api:app on port 8000), KILL after 60 s. A server started by hand is stopped
#                               too when its pid is in the file. A pid file naming any other live process is removed
#                               and that process left alone. Then waits up to 15 s for the port to be free and says
#                               whether it is. Jobs still running are cut off; the next start queues them again.
#
# The API's own environment, set here: PIPELINE=real, JUDGE_URL=http://127.0.0.1:30001 (the RAG server),
# IV3_URLS = the /verify URLs of the replicas on 30000, 30002, 30003 and 30004, JOBS_DIR=<root>/jobs,
# JUDGE_MODEL="Qwen/Qwen3.5-9B via RAG :30001" (the label of the judge in each result's versions), OMP_NUM_THREADS=4,
# and API_KEY, read from secrets/dev_api_key into the server's environment only: never on a command line, never
# printed. Optional settings (RULES_PATH, FFMPEG, VIDEO_ENCODER, ...) are passed on from the caller's environment.
#
# Log: logs/api.log (appended). Pid: logs/api.pid.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
SELF="$HERE/$(basename "${BASH_SOURCE[0]}")"
SVC=$(cd "$HERE/.." && pwd -P)
MTA_ROOT=/home/jovyan/model_train_amir
# shellcheck source=/dev/null
source "$MTA_ROOT/env.sh"

PORT=8000
APP=api:app
RAG_PORT=30001                   # tools/rag_server.sh
IV3_PORTS=(30000 30002 30003 30004)   # the InternVideo3 replicas; 30001 is the RAG server and 30005 the judge LLM

PY=$MTA_ROOT/env/main/bin/python
KEY_FILE=$MTA_ROOT/secrets/dev_api_key
LOG=$MTA_ROOT/logs/api.log
PIDFILE=$MTA_ROOT/logs/api.pid
CTL=$MTA_ROOT/logs/api.lock      # serialises start and stop (fd 8)

usage() { sed -n '2,/^set -euo pipefail/{/^#/s/^# \{0,1\}//p;}' "$SELF"; }
die() { echo "api_server.sh: $*" >&2; exit 1; }

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
  local pid key port iv3_urls=''
  ctl_lock
  if pid=$(server_pid); then
    echo "already running (pid $pid)"
    return
  fi
  [ -x "$PY" ] || die "no env/main: $PY"
  [ -r "$KEY_FILE" ] || die "cannot read the API key file $KEY_FILE"
  command -v "${FFMPEG:-ffmpeg}" >/dev/null || die "no ffmpeg (${FFMPEG:-ffmpeg}): the pipeline cuts its clips with it"
  ! port_taken || die "port $PORT is already in use by a process that logs/api.pid does not name; find and stop it first"
  set +x   # the key below must not reach a trace
  key=$(<"$KEY_FILE")
  [[ -n $key && $key != *[[:space:][:cntrl:]]* ]] || die "$KEY_FILE must hold the key and nothing else"
  for port in "${IV3_PORTS[@]}"; do iv3_urls+="${iv3_urls:+,}http://127.0.0.1:$port/verify"; done
  mkdir -p "$(dirname "$LOG")"
  echo "[$(date -u '+%F %T')] start api" >>"$LOG"
  cd "$SVC"
  # The key goes into the server's environment only (a variable in front of the command is not part of its argument
  # list). setsid: its own session, so it outlives the shell that started it. 8>&-: the server must not keep the lock.
  API_KEY=$key PIPELINE=real JUDGE_URL="http://127.0.0.1:$RAG_PORT" IV3_URLS=$iv3_urls JOBS_DIR=$MTA_ROOT/jobs \
    JUDGE_MODEL="Qwen/Qwen3.5-9B via RAG :$RAG_PORT" OMP_NUM_THREADS=4 \
    setsid nohup "$PY" -m uvicorn "$APP" --host 127.0.0.1 --port "$PORT" --no-access-log \
    >>"$LOG" 2>&1 </dev/null 8>&- &
  pid=$!
  unset key
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
