"""Paths and settings. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
SOURCES_PATH = PKG_DIR / "sources.yaml"
EVAL_PATH = PKG_DIR / "eval_queries.yaml"
STATIC_DIR = PKG_DIR / "static"

# On the server: REGS_DATA_DIR=/home/jovyan/model_train_amir/regs
DATA_DIR = Path(os.environ.get("REGS_DATA_DIR", PKG_DIR.parent.parent / "data" / "regs"))
RAW_DIR = DATA_DIR / "raw"
PARSED_DIR = DATA_DIR / "parsed"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
INDEX_META_PATH = DATA_DIR / "index_meta.json"

# Milvus runs as its own process: `python -m regs milvus` starts a Milvus Lite server on
# MILVUS_DATA_DIR (no Docker). The app, the CLI and the pipeline connect to it over gRPC, so they
# can share the index while it runs. A Milvus Standalone server at the same URI works unchanged.
MILVUS_DATA_DIR = Path(os.environ.get("REGS_MILVUS_DATA_DIR", DATA_DIR / "milvus.db"))
MILVUS_HOST = os.environ.get("REGS_MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.environ.get("REGS_MILVUS_PORT", "19530"))
MILVUS_URI = os.environ.get("REGS_MILVUS_URI", f"http://{MILVUS_HOST}:{MILVUS_PORT}")
MILVUS_TOKEN = os.environ.get("REGS_MILVUS_TOKEN", "")
COLLECTION = os.environ.get("REGS_COLLECTION", "regs_chunks")

# REGS_MILVUS_URI may also be a local path: Milvus Lite then runs inside this process. Its engine
# uses faiss, which ships its own OpenMP runtime next to torch's; two runtimes in one process crash
# on macOS (seen with milvus-lite 3.2.2 + torch 2.14). One OpenMP thread avoids that. This must be
# set before torch is imported, which is why it is here.
if "://" not in MILVUS_URI:
    os.environ.setdefault("OMP_NUM_THREADS", "1")

EMBED_MODEL = os.environ.get("REGS_EMBED_MODEL", "BAAI/bge-m3")
EMBED_DEVICE = os.environ.get("REGS_EMBED_DEVICE") or None  # None = cuda, then mps, then cpu
# fp16 halves model memory (bge-m3 and bge-reranker-v2-m3 are 2.3 GB each in fp32). "auto" uses
# fp16 on cuda/mps and fp32 on cpu. The index must be built with the same setting it is queried with.
MODEL_DTYPE = os.environ.get("REGS_MODEL_DTYPE", "auto")
EMBED_BATCH = int(os.environ.get("REGS_EMBED_BATCH", "8"))
EMBED_MAX_TOKENS = int(os.environ.get("REGS_EMBED_MAX_TOKENS", "1024"))
# bge-m3 needs no prefixes; e5-style models want "query: " / "passage: ".
QUERY_PREFIX = os.environ.get("REGS_QUERY_PREFIX", "")
PASSAGE_PREFIX = os.environ.get("REGS_PASSAGE_PREFIX", "")

# Ranking. The default mode "auto" ranks with LambdaMART over cosine + BM25 (ltr.py) once
# `python -m regs train-ltr` has saved a model, and falls back to hybrid RRF until then.
LTR_MODEL_PATH = DATA_DIR / "ltr_lambdamart.pkl"
LTR_META_PATH = DATA_DIR / "ltr_lambdamart.json"

# Optional neural stage after that: a reranker reads (query, chunk) pairs for the top candidates
# and reorders them. Off by default: bge-reranker-v2-m3 lowered dev top-1 (0.69 -> 0.54) and
# costs ~4 s per query on an M1. Set REGS_RERANK_MODEL to turn it on. Supported families:
#   cross-encoders, e.g. BAAI/bge-reranker-v2-m3 (Apache-2.0)
#   Qwen3-Reranker-0.6B/4B/8B (Apache-2.0): an LLM that answers yes/no; score = P(yes)
RERANK_MODEL = os.environ.get("REGS_RERANK_MODEL", "")
RERANK_CANDIDATES = int(os.environ.get("REGS_RERANK_CANDIDATES", "30"))  # chunks, before folding
# Final score = w * reranker score + (1 - w) * retrieval score scaled to 0..1 over the candidates.
# 1.0 = reranker only. Choose w with `python -m regs sweep` on the dev queries only.
RERANK_WEIGHT = float(os.environ.get("REGS_RERANK_WEIGHT", "1.0"))
RERANK_MAX_TOKENS = int(os.environ.get("REGS_RERANK_MAX_TOKENS", "1024"))
RERANK_BATCH = int(os.environ.get("REGS_RERANK_BATCH", "8"))
# Task description given to instruction-following rerankers (Qwen3); cross-encoders ignore it.
RERANK_INSTRUCTION = os.environ.get(
    "REGS_RERANK_INSTRUCTION",
    "Given a description of something visible on a bank cash-desk video, or a rule paraphrase, "
    "judge whether this clause of a Kazakhstan banking regulation is the rule that the described "
    "action may violate or that the paraphrase refers to")

# A clause longer than this is split at subclause boundaries (about 350-450 bge-m3 tokens).
MAX_CHUNK_WORDS = int(os.environ.get("REGS_MAX_CHUNK_WORDS", "220"))

# If set, /api/regs/* requires the same X-API-Key header as the jury API.
API_KEY = os.environ.get("REGS_API_KEY", "")

FETCH_TIMEOUT = 60
USER_AGENT = "Mozilla/5.0 (regs-index; research use)"
