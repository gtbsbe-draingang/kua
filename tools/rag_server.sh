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

# shellcheck source=service_lib.sh
source "$HERE/service_lib.sh"

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

case "${1:-}" in
  start|stop|status) "$1" ;;
  -h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
