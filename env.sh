# shellcheck shell=bash
# Pod environment. Source it before anything else on the pod:
#   source /home/jovyan/model_train_amir/env.sh
# (the root's env.sh is a symlink to svc/env.sh, created by tools/push.sh)
#
# Every cache, download, build and temp location points inside the root, so
# nothing is written to ~/.cache, ~/.local, ~/.config, ~/.nv, ~/.triton or /tmp.
# Two libraries hard-code ~ with no override, and neither runs for our models:
# humming-kernels (~/.humming, quantized MoE) and TileLang's JIT workspace
# (~/.cache/tilelang, DeepSeek sparse attention).
# Runtime is offline. Download commands override it on the command line only:
#   HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 hf download ...

export MTA_ROOT=/home/jovyan/model_train_amir

# Hugging Face weights and downloaded model code.
export HF_HOME="$MTA_ROOT/hf"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# Generic caches (platformdirs, torch hub, matplotlib, ...).
export XDG_CACHE_HOME="$MTA_ROOT/cache"
export XDG_CONFIG_HOME="$MTA_ROOT/cache/config"
export XDG_DATA_HOME="$MTA_ROOT/cache/share"
export XDG_STATE_HOME="$MTA_ROOT/cache/state"
export TORCH_HOME="$MTA_ROOT/cache/torch"
export MPLCONFIGDIR="$MTA_ROOT/cache/matplotlib"
# Never read or write ~/.local's site-packages, whichever interpreter runs.
export PYTHONNOUSERSITE=1

# Package managers. uv's managed CPython and its shims also stay inside the root.
export PIP_CACHE_DIR="$MTA_ROOT/cache/pip"
export UV_CACHE_DIR="$MTA_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$MTA_ROOT/env/python"
export UV_PYTHON_BIN_DIR="$MTA_ROOT/env/python/bin"
export UV_TOOL_DIR="$MTA_ROOT/env/uv-tools"
export UV_TOOL_BIN_DIR="$MTA_ROOT/env/uv-tools/bin"
export UV_PYTHON_PREFERENCE=only-managed

# GPU kernel and JIT caches (CUDA defaults to ~/.nv, Triton to ~/.triton).
export CUDA_CACHE_PATH="$MTA_ROOT/cache/nv"
export TRITON_HOME="$MTA_ROOT/cache"   # Triton's dump/override dirs (default ~/.triton)
export TRITON_CACHE_DIR="$MTA_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$MTA_ROOT/cache/inductor"
export TORCH_EXTENSIONS_DIR="$MTA_ROOT/cache/torch_extensions"
export NUMBA_CACHE_DIR="$MTA_ROOT/cache/numba"
export FLASHINFER_WORKSPACE_BASE="$MTA_ROOT/cache"
export SGLANG_CACHE_DIR="$MTA_ROOT/cache/sglang"           # default ~/.cache/sglang
export SGLANG_JIT_CACHE_DIR="$MTA_ROOT/cache/sglang/jit"   # default ~/.cache/sglang/jit
export SGLANG_DG_CACHE_DIR="$MTA_ROOT/cache/deep_gemm"
export TVM_FFI_CACHE_DIR="$MTA_ROOT/cache/tvm-ffi"         # default ~/.cache/tvm-ffi
export CUTE_DSL_CACHE_DIR="$MTA_ROOT/cache/cute_dsl"       # default /tmp
export QUACK_HOME="$MTA_ROOT/cache/quack"                  # default ~
export TILELANG_CACHE_DIR="$MTA_ROOT/cache/tilelang"
export OUTLINES_CACHE_DIR="$MTA_ROOT/cache/outlines"

# Temporary files (Python's tempfile, uv, sglang's IPC socket names) too.
export TMPDIR="$MTA_ROOT/cache/tmp"
[ -d "$TMPDIR" ] || mkdir -p "$TMPDIR"

# The pod's CPU quota is 16 cores but nproc reports 192, so every library that
# sizes its thread pool from nproc gets throttled. Cap them.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export RAYON_NUM_THREADS=4

# uv and hf live in env/tools (created by setup.sh).
case ":$PATH:" in
  *":$MTA_ROOT/env/tools/bin:"*) ;;
  *) export PATH="$MTA_ROOT/env/tools/bin:$PATH" ;;
esac
