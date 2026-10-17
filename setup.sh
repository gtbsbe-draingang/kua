#!/usr/bin/env bash
# Build the pod's Python environments and fetch the model weights.
#
#   bash setup.sh main iv3 vlm      # any subset of: tools toolchain main iv3 vlm weights
#
# Every target is idempotent: rerunning it only installs what is missing.
# Nothing is ever compiled. Every install runs with --no-build (and env.sh sets
# UV_NO_BUILD=1), so a package without a prebuilt wheel fails the build instead
# of starting a source build.
# flash-attn and causal-conv1d come from their GitHub release wheels.
#
#   tools      env/tools: uv and the hf CLI, in a venv made by python3 -m venv + pip
#   toolchain  env/toolchain: prebuilt gcc/g++ 13 and nvcc 12.8 from conda-forge
#              (micromamba), for Triton's JIT; env.sh exports CC, CXX, CUDA_HOME
#   main       env/main: API + VideoJev-4B (torch 2.8.0+cu128, transformers 5.14.1)
#   iv3        env/iv3: InternVideo3-8B server (torch 2.8.0+cu128, transformers 4.57.3)
#   vlm        env/vlm: SGLang for the Qwen3.5-9B judge (CUDA 12 build, see below)
#   weights    hf download of the five models into HF_HOME, then a shard check
#
# vlm's architecture check needs the weights; it is deferred (logged) until they
# exist, and the weights target runs it once env/vlm exists.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=env.sh
source "$HERE/env.sh"

ENVS=$MTA_ROOT/env
TOOLS=$ENVS/tools
UV=$TOOLS/bin/uv
PY=3.12

UV_VERSION=0.13.1
HFHUB_VERSION=2.2.0

# The pod has a 16-CPU quota shared with training runs: keep parallelism low.
export UV_CONCURRENT_DOWNLOADS=4 UV_CONCURRENT_INSTALLS=4 UV_CONCURRENT_BUILDS=1
export UV_HTTP_TIMEOUT=300
export UV_LINK_MODE=hardlink
export NO_COLOR=1   # plain logs

# The driver is 570 (CUDA 12.8). env/main and env/iv3 use the exact cu128 build;
# env/vlm uses SGLang's CUDA 12 lane (cu129, runs on 12.8 through CUDA minor
# version compatibility). CUDA 13 builds do not run on this driver.
PT128=https://download.pytorch.org/whl/cu128
PT129=https://download.pytorch.org/whl/cu129
TORCH_28="torch @ $PT128/torch-2.8.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"
TV_23="torchvision @ $PT128/torchvision-0.23.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"
FA_REL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post2
CC_REL=https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0

# SGLang 0.5.19 is the newest release with a CUDA 12 lane (its install docs:
# "SGLang 0.5.19 is the last release with a CUDA 12 lane (PyTorch 2.13)");
# 0.5.20+ need a CUDA 13 driver (>= 580). Its default dependencies are CUDA 13
# builds, so the overrides reproduce, in one resolve, the CUDA 12 lane of
# SGLang's own v0.5.19 Dockerfile (the lmsysorg/sglang:v0.5.19-cu129 image):
#   - torch 2.13.0 / torchvision 0.28.0 / torchaudio 2.11.0, cu129 builds
#   - the cu129 wheels of sglang-kernel, sgl-deep-gemm and sgl-deep-ep
#   - cuda-python 12.x, flashinfer-python[cu12], nvidia-cutlass-dsl without [cu13]
#   - humming-kernels[cu12]: the image keeps [cu13] and strips *-cu13 packages
#     afterwards, which misses CUDA 13's unsuffixed nvcc/runtime wheels
#   - cuda-tile 1.6.0rc5: PyPI has only a wheel-stub sdist, which would fetch
#     this same wheel from pypi.nvidia.com at build time; --no-build needs the
#     wheel itself
# The pod has no CUDA toolkit (no nvcc), so FlashInfer's prebuilt kernels (the
# cubin package and the cu129 AOT jit-cache, both 0.6.18 like flashinfer-python)
# are added instead of compiling them at runtime.
SGLANG_VERSION=0.5.19
SGL_WHL=https://github.com/sgl-project/whl/releases/download
FI_REL=https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.18
VLM_OVERRIDES=(
  "torch @ $PT129/torch-2.13.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl"
  "torchvision @ $PT129/torchvision-0.28.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl"
  "torchaudio @ $PT129/torchaudio-2.11.0%2Bcu129-cp312-cp312-manylinux_2_28_x86_64.whl"
  "sglang-kernel @ $SGL_WHL/v0.4.6.post1/sglang_kernel-0.4.6.post1+cu129-cp310-abi3-manylinux2014_x86_64.whl"
  "sgl-deep-gemm @ $SGL_WHL/v0.1.7/sgl_deep_gemm-0.1.7+cu129-py3-none-manylinux2014_x86_64.whl"
  "sgl-deep-ep @ $SGL_WHL/v0.1.2/sgl_deep_ep-0.1.2+cu129-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
  "cuda-tile @ https://pypi.nvidia.com/cuda-tile/cuda_tile-1.6.0rc5-cp312-cp312-manylinux2014_x86_64.whl"
  "cuda-python>=12,<13"
  "flashinfer-python[cu12]==0.6.18"
  "nvidia-cutlass-dsl==4.6.2"
  "humming-kernels[cu12]==0.1.12"
)
VLM_EXTRAS=(
  "flashinfer-cubin @ $FI_REL/flashinfer_cubin-0.6.18-py3-none-any.whl"
  "flashinfer-jit-cache @ $FI_REL/flashinfer_jit_cache-0.6.18+cu129-cp39-abi3-manylinux_2_28_x86_64.whl"
)

# The pod has no C compiler and no CUDA toolkit. Triton builds a small C launcher
# per kernel signature at first launch (VideoJev's flash-linear-attention kernels
# need it), so env/toolchain holds prebuilt compilers from conda-forge, installed
# by a static micromamba binary. cuda-version=12.8 matches the driver (570).
# All micromamba state (package cache, repodata, rc lookups, environments.txt)
# stays under cache/mamba: it runs with --no-rc, its root prefix and package dirs
# there, and HOME pointed there too, so nothing can land in the real home.
MICROMAMBA_VERSION=2.9.0-0
MICROMAMBA_SHA256=366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3
MICROMAMBA_URL=https://github.com/mamba-org/micromamba-releases/releases/download/$MICROMAMBA_VERSION/micromamba-linux-64
TOOLCHAIN_SPECS=(
  "gcc=13" "gxx=13"
  "cuda-version=12.8"
  "cuda-nvcc=12.8.93" "cuda-cudart-dev=12.8.90" "cuda-cccl=12.8.90"
)

# Models: repo id, then hf download --exclude patterns (junk the author uploaded).
# bge-m3 keeps only what AutoTokenizer + AutoModel need for the dense embedding:
# config, tokenizer files and pytorch_model.bin (the repo has no safetensors file).
MODELS=(
  "ZhuYuhan/VideoJev-4B"
  "yanziang/InternVideo3-8B-Instruct|.claude/*|__pycache__/*"
  "Qwen/Qwen3.5-9B"
  "PekingU/rtdetr_r50vd_coco_o365"
  "BAAI/bge-m3|onnx/*|imgs/*|*.jpg|*.webp|colbert_linear.pt|sparse_linear.pt"
)

log() { printf '[%s] %s\n' "$(date -u '+%F %T')" "$*"; }
die() { log "ERROR: $*"; exit 1; }

# uv pip install into one env, wheels only.
pip_env() {
  local name=$1; shift
  "$UV" pip install --python "$ENVS/$name/bin/python" --no-build "$@"
}

make_venv() {
  local dir=$ENVS/$1
  if [ ! -x "$dir/bin/python" ]; then
    log "creating $dir (Python $PY)"
    "$UV" venv --python "$PY" "$dir"
  fi
}

freeze() {
  "$UV" pip freeze --python "$ENVS/$1/bin/python" > "$MTA_ROOT/logs/freeze-$1.txt"
  log "package list: logs/freeze-$1.txt"
}

# Import checks. They run with the GPU hidden, so nothing can touch it; the
# driver check afterwards only asks CUDA whether it accepts the driver
# (cudaGetDeviceCount), which creates no context and allocates no memory.
check() {
  local name=$1 code=$2
  log "check $name"
  CUDA_VISIBLE_DEVICES='' "$ENVS/$name/bin/python" -c "$code"
  "$ENVS/$name/bin/python" -c 'import torch; print("cuda runtime", torch.version.cuda, "| accepts driver:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())'
}

build_tools() {
  if [ ! -x "$TOOLS/bin/python" ]; then
    log "creating $TOOLS with $(command -v python3)"
    python3 -m venv "$TOOLS"
  fi
  if [ "$("$UV" --version 2>/dev/null | awk '{print $2}')" != "$UV_VERSION" ] \
     || ! "$TOOLS/bin/python" -c "import huggingface_hub as h, sys; sys.exit(h.__version__ != '$HFHUB_VERSION')" 2>/dev/null; then
    "$TOOLS/bin/pip" install --disable-pip-version-check --progress-bar off --only-binary=:all: \
      "uv==$UV_VERSION" "huggingface_hub==$HFHUB_VERSION"
  fi
  "$UV" python install "$PY" --no-bin
  log "tools: $("$UV" --version), hf $("$TOOLS/bin/python" -c 'import huggingface_hub as h; print(h.__version__)'), python $("$UV" python find "$PY")"
}

# micromamba with all of its state under cache/mamba (see MICROMAMBA_* above).
# Usage: mamba_run <subcommand> [args...]
mamba_run() {
  local sub=$1 root=$MTA_ROOT/cache/mamba
  shift
  mkdir -p "$root/home" "$root/pkgs"
  HOME=$root/home MAMBA_ROOT_PREFIX=$root CONDA_PKGS_DIRS=$root/pkgs MAMBA_PKGS_DIRS=$root/pkgs \
    MAMBA_REGISTER_ENVS=false CONDA_REGISTER_ENVS=false \
    "$ENVS/toolchain-bin/micromamba" "$sub" --no-rc -r "$root" "$@"
}

build_toolchain() {
  local bin=$ENVS/toolchain-bin/micromamba prefix=$ENVS/toolchain
  if [ ! -x "$bin" ] || [ "$(sha256sum "$bin" | cut -d' ' -f1)" != "$MICROMAMBA_SHA256" ]; then
    log "download micromamba $MICROMAMBA_VERSION"
    mkdir -p "$(dirname "$bin")"
    curl -fsSL --retry 3 -o "$bin.part" "$MICROMAMBA_URL"
    echo "$MICROMAMBA_SHA256  $bin.part" | sha256sum -c --quiet || die "micromamba checksum mismatch"
    chmod +x "$bin.part"
    mv -f "$bin.part" "$bin"
  fi
  log "micromamba $("$bin" --version)"
  local verb=install
  [ -f "$prefix/conda-meta/history" ] || verb=create
  log "micromamba $verb $prefix: ${TOOLCHAIN_SPECS[*]}"
  mamba_run "$verb" -y -p "$prefix" --override-channels -c conda-forge --strict-channel-priority \
    "${TOOLCHAIN_SPECS[@]}"
  mamba_run list -p "$prefix" > "$MTA_ROOT/logs/freeze-toolchain.txt"
  log "package list: logs/freeze-toolchain.txt"
  check_toolchain
}

# The compilers run, and build what Triton builds: a shared C module that includes
# Python.h (env/main's interpreter) and links libcuda, then Triton's own launcher
# utilities (driver.c). nvcc compiles an sm_90 kernel object. Nothing here runs
# on the GPU: the Triton check imports with the GPU hidden and only loads libcuda.
check_toolchain() {
  local tc=$ENVS/toolchain dir=$MTA_ROOT/cache/setup/toolchain-check inc libcuda
  log "check toolchain"
  [ -x "$tc/bin/gcc" ] && [ -x "$tc/bin/g++" ] && [ -x "$tc/bin/nvcc" ] || die "toolchain incomplete: no gcc, g++ or nvcc in $tc/bin"
  "$tc/bin/gcc" --version | head -n 1
  "$tc/bin/g++" --version | head -n 1
  "$tc/bin/nvcc" --version | tail -n 2
  [ "$("$tc/bin/gcc" -dumpversion | cut -d. -f1)" = 13 ] || die "gcc is not 13"
  "$tc/bin/nvcc" --version | grep -q 'release 12\.8' || die "nvcc is not 12.8"
  mkdir -p "$dir"
  printf '__global__ void k(float *x) { x[threadIdx.x] *= 2.f; }\n' > "$dir/k.cu"
  "$tc/bin/nvcc" -ccbin "$tc/bin/g++" -arch=sm_90 -c "$dir/k.cu" -o "$dir/k.o"
  log "nvcc built an sm_90 object: $(stat -c %s "$dir/k.o") bytes"
  if [ ! -x "$ENVS/main/bin/python" ]; then
    log "Python.h and Triton checks deferred: env/main not built (rerun: setup.sh toolchain)"
    return 0
  fi
  inc=$(CUDA_VISIBLE_DEVICES='' "$ENVS/main/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
  libcuda=$(ldconfig -p | awk '/libcuda\.so\.1 / {print $NF; exit}')
  [ -n "$libcuda" ] || die "libcuda.so.1 not found by ldconfig"
  libcuda=$(dirname "$libcuda")
  cat > "$dir/tc_check.c" <<'C'
#include <Python.h>
static struct PyModuleDef def = {PyModuleDef_HEAD_INIT, "tc_check", NULL, -1, NULL};
PyMODINIT_FUNC PyInit_tc_check(void) { return PyModule_Create(&def); }
C
  "$tc/bin/gcc" "$dir/tc_check.c" -O3 -shared -fPIC -Wno-psabi -I"$inc" -L"$libcuda" -lcuda -o "$dir/tc_check.so"
  CUDA_VISIBLE_DEVICES='' "$ENVS/main/bin/python" -c "import sys; sys.path.insert(0, '$dir'); import tc_check; print('Python.h module built and imported:', tc_check.__file__)"
  CUDA_VISIBLE_DEVICES='' CC="$tc/bin/gcc" "$ENVS/main/bin/python" -c '
import triton
from triton.backends.nvidia.driver import CudaUtils
CudaUtils()
print("triton", triton.__version__, "built and loaded its CUDA launcher utilities")'
}

build_main() {
  make_venv main
  pip_env main "$TORCH_28" "$TV_23"
  # Pick the flash-attn / causal-conv1d wheel that matches torch's C++ ABI.
  local abi
  abi=$(CUDA_VISIBLE_DEVICES='' "$ENVS/main/bin/python" -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')
  log "torch cxx11abi=$abi"
  local fa="flash_attn-2.8.3.post2+cu12torch2.8cxx11abi${abi}-cp312-cp312-linux_x86_64.whl"
  local cc="causal_conv1d-1.7.0+cu12torch2.8cxx11abi${abi}-cp312-cp312-linux_x86_64.whl"
  log "wheels: $fa, $cc"
  pip_env main "$TORCH_28" "$TV_23" \
    "flash-attn @ $FA_REL/$fa" "causal-conv1d @ $CC_REL/$cc" \
    transformers==5.14.1 accelerate safetensors numpy pillow av==18.1.0 \
    flash-linear-attention==0.4.2 fla-core==0.4.2 \
    fastapi uvicorn python-multipart pyyaml httpx openpyxl snowballstemmer
  freeze main
  check main '
import torch, torchvision, transformers, accelerate, safetensors, numpy, PIL, av
import flash_attn, causal_conv1d, fla, fastapi, uvicorn, multipart, yaml, httpx, openpyxl, snowballstemmer
import flash_attn_2_cuda, causal_conv1d_cuda          # the compiled extensions load
from flash_attn import flash_attn_varlen_func
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_vl import video_processing_qwen3_vl
from importlib.metadata import version as v
print("torch", torch.__version__, "cuda", torch.version.cuda, "cxx11abi", torch._C._GLIBCXX_USE_CXX11_ABI)
print("torchvision", torchvision.__version__, "| transformers", transformers.__version__, "| accelerate", accelerate.__version__)
print("flash_attn", flash_attn.__version__, "| causal_conv1d", causal_conv1d.__version__)
print("flash-linear-attention", v("flash-linear-attention"), "| fla-core", v("fla-core"), "| av", av.__version__)
print("numpy", numpy.__version__, "| pillow", PIL.__version__, "| safetensors", safetensors.__version__)
print("fastapi", fastapi.__version__, "| uvicorn", uvicorn.__version__, "| python-multipart", v("python-multipart"), "| pyyaml", yaml.__version__, "| httpx", httpx.__version__)
print("snowballstemmer", v("snowballstemmer"), "| lexical retrieval of the RAG server")
'
}

build_iv3() {
  make_venv iv3
  pip_env iv3 "$TORCH_28" "$TV_23" \
    transformers==4.57.3 accelerate qwen-vl-utils av==18.1.0 fastapi uvicorn
  freeze iv3
  check iv3 '
import torch, torchvision, transformers, accelerate, qwen_vl_utils, av, fastapi, uvicorn
from transformers.models.deepseek_v3.modeling_deepseek_v3 import apply_rotary_pos_emb_interleave
from transformers.models.qwen3_vl import video_processing_qwen3_vl
from qwen_vl_utils import process_vision_info
from importlib.metadata import version as v
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__, "| transformers", transformers.__version__, "| accelerate", accelerate.__version__)
print("qwen-vl-utils", v("qwen-vl-utils"), "| av", av.__version__, "| fastapi", fastapi.__version__, "| uvicorn", uvicorn.__version__)
'
}

build_vlm() {
  make_venv vlm
  local ov=$MTA_ROOT/cache/setup/vlm-overrides.txt
  mkdir -p "$(dirname "$ov")"
  printf '%s\n' "${VLM_OVERRIDES[@]}" > "$ov"
  # No --prerelease=allow (SGLang's docs suggest it for old uv): it would pull
  # release candidates of unrelated packages (pandas, safetensors, grpcio, ...).
  # uv's default still takes a pre-release where a requirement names one
  # (flash-attn-4>=4.0.0b18, cuda-tile==1.6.0rc5).
  pip_env vlm --override "$ov" "sglang==$SGLANG_VERSION" "${VLM_EXTRAS[@]}"
  freeze vlm
  # A CUDA 13 runtime would not run on this driver; none may slip in. CUDA 13
  # wheels are named *-cu13 or unsuffixed with version 13.x. Two exceptions:
  # nvidia-ml-py's 13.x is the NVML binding's own numbering, and
  # nvidia-cuda-nvdisasm (a SASS disassembler, not a runtime library) is pinned
  # to >=13.3 by nvidia-cutlass-dsl-libs on every CUDA lane.
  if grep -i -E -- 'cu13|^nvidia-[a-z0-9-]+==13\.' "$MTA_ROOT/logs/freeze-vlm.txt" \
     | grep -v -i -E '^nvidia-(ml-py|cuda-nvdisasm)=='; then
    die "CUDA 13 packages in env/vlm (listed above)"
  fi
  check vlm '
import torch, torchvision, sglang, sgl_kernel, transformers, xgrammar, flashinfer
from sglang.srt.models.qwen3_5 import EntryClass
from importlib.metadata import version as v
print("torch", torch.__version__, "cuda", torch.version.cuda, "| torchvision", torchvision.__version__, "| torchaudio", v("torchaudio"))
print("sglang", v("sglang"), "| sglang-kernel", v("sglang-kernel"), "| transformers", transformers.__version__, "| xgrammar", v("xgrammar"))
print("flashinfer-python", v("flashinfer-python"), "| flashinfer-cubin", v("flashinfer-cubin"), "| flashinfer-jit-cache", v("flashinfer-jit-cache"))
print("sgl-deep-gemm", v("sgl-deep-gemm"), "| sgl-deep-ep", v("sgl-deep-ep"), "| cuda-python", v("cuda-python"), "| cuda-tile", v("cuda-tile"))
'
  "$ENVS/vlm/bin/python" -c 'from cuda.bindings import driver as d; e, ver = d.cuDriverGetVersion(); print("cuda-bindings sees driver API", ver)'
  check_vlm_arch
}

# The Qwen3.5-9B checkpoint's architectures must have a model class in SGLang.
# It needs both env/vlm and the weights, and setup runs vlm before weights on a
# fresh pod, so both targets call it and it is skipped until both exist.
check_vlm_arch() {
  local base=$HF_HOME/hub/models--Qwen--Qwen3.5-9B ref
  if [ ! -x "$ENVS/vlm/bin/python" ]; then
    log "architecture check deferred: env/vlm not built"
    return 0
  fi
  if [ ! -f "$base/refs/main" ]; then
    log "architecture check deferred: weights not downloaded"
    return 0
  fi
  ref=$(cat "$base/refs/main")
  if [ ! -f "$base/snapshots/$ref/config.json" ]; then
    log "architecture check deferred: weights not downloaded"
    return 0
  fi
  log "check vlm architecture"
  CUDA_VISIBLE_DEVICES='' "$ENVS/vlm/bin/python" - "$base/snapshots/$ref/config.json" <<'PY'
import json, sys
from sglang.srt.models.qwen3_5 import EntryClass
archs = json.load(open(sys.argv[1]))["architectures"]
names = [c.__name__ for c in EntryClass]
print("Qwen3.5-9B architectures", archs, "| sglang.srt.models.qwen3_5 entry classes", names)
sys.exit(0 if set(archs) <= set(names) else "sglang has no model class for this checkpoint")
PY
}

# Download the five models into HF_HOME (resumable), then check every shard
# against its index: each tensor the index names must sit in the shard it names.
build_weights() {
  local spec repo pats args
  for spec in "${MODELS[@]}"; do
    IFS='|' read -r repo pats <<< "${spec}|"
    args=()
    while IFS= read -r p; do [ -n "$p" ] && args+=(--exclude "$p"); done < <(tr '|' '\n' <<< "$pats")
    log "download $repo ${args[*]}"
    HF_HUB_OFFLINE=0 HF_HUB_DISABLE_XET=1 "$TOOLS/bin/hf" download "$repo" "${args[@]}" --max-workers 4
  done
  verify_weights
  backup_iv3
  check_vlm_arch
}

verify_weights() {
  local spec repo
  for spec in "${MODELS[@]}"; do
    repo=${spec%%|*}
    "$TOOLS/bin/python" - "$HF_HOME/hub/models--${repo//\//--}" <<'PY'
import json, struct, sys
from pathlib import Path
base = Path(sys.argv[1])
ref = (base / "refs" / "main").read_text().strip()
snap = base / "snapshots" / ref
incomplete = list((base / "blobs").glob("*.incomplete"))
idx = snap / "model.safetensors.index.json"
if not idx.exists() and not (snap / "model.safetensors").exists():
    # a pickle-zip checkpoint (the bge-m3 embedder): check the archive opens and lists its tensors
    import zipfile
    with zipfile.ZipFile(snap / "pytorch_model.bin") as z:
        entries = [n for n in z.namelist() if "/data/" in n]
    size = sum(p.stat().st_size for p in (base / "blobs").iterdir() if p.is_file())
    print(f"{base.name}: rev {ref[:12]} | pytorch_model.bin, {len(entries)} tensor storages | "
          f"incomplete files {len(incomplete)} | {size/1e9:.2f} GB")
    sys.exit(1 if (incomplete or not entries) else 0)
if idx.exists():
    index = json.loads(idx.read_text())["weight_map"]
else:  # a single-file checkpoint (the person detector): its one file holds every tensor
    with open(snap / "model.safetensors", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        index = {k: "model.safetensors" for k in json.loads(f.read(n)) if k != "__metadata__"}
found = {}
for shard in sorted(set(index.values())):
    with open(snap / shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    for name in header:
        if name != "__metadata__":
            found.setdefault(name, []).append(shard)
missing = [t for t in index if t not in found]
extra = [t for t in found if t not in index]
wrong = [t for t in index if t in found and found[t] != [index[t]]]
size = sum(p.stat().st_size for p in (base / "blobs").iterdir() if p.is_file())
print(f"{base.name}: rev {ref[:12]} | {len(index)} tensors in {len(set(index.values()))} shards | "
      f"missing {len(missing)} extra {len(extra)} wrong-shard {len(wrong)} | "
      f"incomplete files {len(incomplete)} | {size/1e9:.2f} GB")
sys.exit(1 if (missing or extra or wrong or incomplete) else 0)
PY
  done
}

# The InternVideo3 mirror is the only public copy of those weights, so keep a
# second, independent copy (real files, not hardlinks) when the disk allows.
backup_iv3() {
  local base=$HF_HOME/hub/models--yanziang--InternVideo3-8B-Instruct ref dst free_gb
  ref=$(cat "$base/refs/main")
  dst=$HF_HOME/backup/InternVideo3-8B-Instruct-$ref
  free_gb=$(df -P --block-size=1G "$MTA_ROOT" | awk 'NR==2 {print $4}')
  if [ -f "$dst/.complete" ]; then
    log "InternVideo3 backup present: $dst"
  elif [ "$free_gb" -lt 200 ]; then
    log "skip InternVideo3 backup: only ${free_gb} GB free"
  else
    log "copying InternVideo3 snapshot to $dst"
    mkdir -p "$dst"
    cp -rL "$base/snapshots/$ref/." "$dst/"
    (cd "$base/snapshots/$ref" && find -L . -type f -print0 | sort -z | xargs -0 md5sum) > "$dst/.md5"
    (cd "$dst" && md5sum --quiet -c .md5) || die "InternVideo3 backup does not match the snapshot"
    touch "$dst/.complete"
    log "InternVideo3 backup verified: $(du -sh "$dst" | cut -f1)"
  fi
}

usage() { sed -n '2,/^set -euo pipefail/{/^#/s/^# \{0,1\}//p;}' "$0"; }
[ $# -gt 0 ] || { usage; exit 2; }
case $1 in -h|--help) usage; exit 0 ;; esac
for target in "$@"; do
  case $target in tools|toolchain|main|iv3|vlm|weights) ;; *) die "unknown target: $target" ;; esac
done
mkdir -p "$MTA_ROOT/logs"
build_tools
for target in "$@"; do
  [ "$target" = tools ] && continue
  start=$SECONDS
  log "=== $target ==="
  "build_$target"
  log "=== $target done in $((SECONDS - start)) s ==="
done
