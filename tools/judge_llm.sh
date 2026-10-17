#!/usr/bin/env bash
# The judge LLM: Qwen3.5-9B on SGLang, an OpenAI-style server on 127.0.0.1:30005 for the RAG server
# (RAG_LLM_KIND=openai, RAG_LLM_URL=http://127.0.0.1:30005/v1/chat/completions, model qwen3.5-9b).
# Runs from env/vlm, offline, with every cache inside the root (env.sh).
#
#   bash judge_llm.sh start    refuse when under 35 GiB of GPU memory is free; else start the server in
#                              the background and return once its process is up. Loading the weights and
#                              capturing CUDA graphs takes minutes: poll `status`.
#                              --mem-fraction-static is a fraction of the GPU memory that is FREE at start
#                              (SGLang: slack = free * (1 - fraction), taken from what is left after the
#                              weights), not of the card. The GPU is shared and the free amount varies, so
#                              the fraction is computed to give a fixed pool of 28 GiB: the weights (17.6 GiB)
#                              plus about 10 GiB for the KV cache and the recurrent-state slots.
#   bash judge_llm.sh status   pid and GET /health. Exit 0 healthy, 3 running but not ready, 1 not running.
#   bash judge_llm.sh stop     TERM the server named in logs/judge_llm.pid (after checking that the process
#                              is sglang on port 30005), KILL after 60 s; then any live "sglang" process
#                              left in its session (a scheduler orphaned by a crash) is killed too. A pid
#                              file naming some other live process is removed and that process left alone.
#
# Log: logs/judge_llm.log (appended). Pid: logs/judge_llm.pid.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
SELF="$HERE/$(basename "${BASH_SOURCE[0]}")"
MTA_ROOT=/home/jovyan/model_train_amir
# shellcheck source=/dev/null
source "$MTA_ROOT/env.sh"

PORT=30005
MODEL_PATH=Qwen/Qwen3.5-9B
SERVED_NAME=qwen3.5-9b
MIN_FREE_MIB=$((35 * 1024))
STATIC_MIB=$((28 * 1024))   # weights + KV cache + recurrent-state pool; see the header

PY=$MTA_ROOT/env/vlm/bin/python
PRELOAD_LIBS=("$MTA_ROOT/env/toolchain/lib/libgcc_s.so.1" "$MTA_ROOT/env/toolchain/lib/libstdc++.so.6")
LOG=$MTA_ROOT/logs/judge_llm.log
PIDFILE=$MTA_ROOT/logs/judge_llm.pid
CTL=$MTA_ROOT/logs/judge_llm.lock   # serialises start and stop (fd 8)

usage() { sed -n '2,/^set -euo pipefail/{/^#/s/^# \{0,1\}//p;}' "$SELF"; }
die() { echo "judge_llm.sh: $*" >&2; exit 1; }

# The pid in the pid file, if that process is alive and really is our server (pids get reused).
server_pid() {
  local pid cmd
  [ -s "$PIDFILE" ] || return 1
  pid=$(<"$PIDFILE")
  [[ $pid =~ ^[0-9]+$ ]] || return 1
  [ -r "/proc/$pid/cmdline" ] || return 1
  cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline")
  grep -qF 'sglang.launch_server' <<<"$cmd" && grep -qF -e "--port $PORT " <<<"$cmd" || return 1
  echo "$pid"
}

# True while pid runs (a zombie, which nothing may have reaped, does not).
alive() {
  local state
  state=$(sed 's/^.*) //' "/proc/$1/stat" 2>/dev/null | cut -c1) || return 1
  [ -n "$state" ] && [ "$state" != Z ]
}

gpu_free_mib() {
  local free
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sed -n 1p | tr -d ' ')
  [[ $free =~ ^[0-9]+$ ]] || return 1
  echo "$free"
}

ctl_lock() {
  exec 8>"$CTL"
  flock -w 120 8 || die "another start or stop has been running for 2 minutes"
}

start() {
  local pid free fraction lib
  ctl_lock
  if pid=$(server_pid); then
    echo "already running (pid $pid)"
    return
  fi
  [ -x "$PY" ] || die "no env/vlm: $PY"
  for lib in "${PRELOAD_LIBS[@]}"; do
    [ -r "$lib" ] || die "missing $lib (the toolchain; bash setup.sh toolchain)"
  done
  free=$(gpu_free_mib) || die "cannot read the free GPU memory from nvidia-smi"
  [ "$free" -ge "$MIN_FREE_MIB" ] || die "only $free MiB of GPU memory is free, need $MIN_FREE_MIB"
  # SGLang re-reads the free memory a moment later (and sees ~0.5 GiB less: its own CUDA context, plus whatever
  # the other processes moved meanwhile), so the pool lands within about 1% of 28 GiB, not exactly on it.
  # LC_ALL=C: the decimal point must be a point, whatever the locale.
  fraction=$(LC_ALL=C awk -v pool="$STATIC_MIB" -v free="$free" 'BEGIN { printf "%.3f", pool / free }')
  mkdir -p "$(dirname "$LOG")"
  echo "[$(date -u '+%F %T')] start judge llm ($free MiB of GPU memory free, --mem-fraction-static $fraction)" >>"$LOG"
  # setsid: its own session, so stop can find every process it spawned. 8>&-: the server must not keep the lock.
  # PATH: SGLang compiles its JIT kernels with ninja, which lives in env/vlm/bin (nvcc and gcc come from env.sh).
  # LD_PRELOAD: those kernels are built by the toolchain's gcc 13 and need its libstdc++ (GLIBCXX_3.4.31), newer
  # than the pod's. A preload of just the two runtime libraries, not LD_LIBRARY_PATH: the toolchain's lib dir also
  # holds a libcudart 12.8 that would shadow the one torch uses.
  PATH="$MTA_ROOT/env/vlm/bin:$PATH" OMP_NUM_THREADS=4 \
    LD_PRELOAD="$(IFS=:; echo "${PRELOAD_LIBS[*]}")" \
    setsid nohup "$PY" -m sglang.launch_server \
    --model-path "$MODEL_PATH" --served-model-name "$SERVED_NAME" \
    --host 127.0.0.1 --port "$PORT" \
    --mem-fraction-static "$fraction" --context-length 32768 \
    >>"$LOG" 2>&1 </dev/null 8>&- &
  pid=$!
  echo "$pid" >"$PIDFILE"
  sleep 3
  server_pid >/dev/null || die "the server did not stay up (see $LOG)"
  echo "started (pid $pid); log $LOG; poll: bash $SELF status"
}

# Live processes in the session led by pid (the server's scheduler and helpers) that are ours to kill: a binary
# from inside the root and "sglang" in the command line (the scheduler and detokenizer rename themselves
# "sglang::..."). Zombies have no binary and are skipped.
session_leftovers() {
  local sid=$1 p
  for p in $(pgrep -s "$sid" || true); do
    case "$(readlink "/proc/$p/exe" 2>/dev/null || true)" in
      "$MTA_ROOT"/*) tr '\0' ' ' <"/proc/$p/cmdline" 2>/dev/null | grep -qF 'sglang' && echo "$p" || true ;;
    esac
  done
}

stop() {
  local pid='' p sweep=0 note=''
  ctl_lock
  if [ -s "$PIDFILE" ]; then pid=$(<"$PIDFILE"); fi
  [[ $pid =~ ^[1-9][0-9]*$ ]] || pid=''   # never 0: pgrep -s 0 means the caller's own session
  if [ -n "$pid" ] && server_pid >/dev/null; then
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 60); do
      alive "$pid" || break
      sleep 1
    done
    if alive "$pid"; then
      echo "pid $pid ignored TERM for 60 s: KILL"
      kill -KILL "$pid" 2>/dev/null || true
    fi
    sleep 1
    sweep=1
  elif [ -n "$pid" ] && ! alive "$pid"; then
    # our server is gone (crashed, or the pod restarted); what it left in its session may still hold GPU memory
    note=" (no server process was running)"
    sweep=1
  else
    # no pid, or the pid file's number now belongs to another process: not ours, so nothing is killed
    note=" (no server process was running${pid:+; pid $pid in the pid file is not the server, left alone})"
  fi
  if [ "$sweep" = 1 ]; then
    for p in $(session_leftovers "$pid"); do
      echo "killing leftover pid $p of the server's session"
      kill -KILL "$p" 2>/dev/null || true
    done
  fi
  rm -f "$PIDFILE"
  echo "stopped$note"
}

status() {
  local pid
  if ! pid=$(server_pid); then
    echo "not running"
    return 1
  fi
  if curl -fsS --noproxy '*' -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "running (pid $pid), healthy"
    return 0
  fi
  echo "running (pid $pid), not ready yet (GET /health fails); log tail:"
  tail -n 3 "$LOG" | cut -c1-300
  return 3
}

case "${1:-}" in
  start|stop|status) "$1" ;;
  -h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
