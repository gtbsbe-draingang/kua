"""Dense embeddings with a local sentence-transformers model (default BAAI/bge-m3, MIT licence).

bge-m3 is multilingual, so an English query ("cashier leaves the desk") finds Russian clauses.
That matters because rules.yaml keeps visual criteria in English and observations are in Russian.
With HF_HUB_OFFLINE=1 the model loads from HF_HOME and nothing goes to the network.
"""
import threading

from . import config

_lock = threading.RLock()
_instance = None


def _dtype(device: str):
    import torch
    name = config.MODEL_DTYPE
    if name == "auto":
        name = "float16" if device in ("cuda", "mps") else "float32"
    return getattr(torch, name)


def _device() -> str:
    if config.EMBED_DEVICE:
        return config.EMBED_DEVICE
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    def __init__(self, model_name: str = config.EMBED_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.device = _device()
        self.dtype = str(_dtype(self.device)).replace("torch.", "")
        self.model = SentenceTransformer(model_name, device=self.device,
                                         model_kwargs={"dtype": _dtype(self.device)})
        self.model.max_seq_length = config.EMBED_MAX_TOKENS
        self.dim = (getattr(self.model, "get_embedding_dimension", None)
                    or self.model.get_sentence_embedding_dimension)()

    def _encode(self, texts: list[str], batch_size: int) -> list[list[float]]:
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=len(texts) > 32)
        return vecs.astype("float32").tolist()

    def passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode([config.PASSAGE_PREFIX + t for t in texts], config.EMBED_BATCH)

    def query(self, text: str) -> list[float]:
        with _lock:  # web requests run in a thread pool; one forward pass at a time
            return self._encode([config.QUERY_PREFIX + text], 1)[0]


class ApiEmbedder:
    """Same interface as Embedder, backed by an OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(self, model_name: str = config.EMBED_MODEL):
        from openai import OpenAI
        self.model_name = model_name
        self.client = OpenAI(base_url=config.EMBED_BASE_URL, api_key=config.EMBED_API_KEY)
        self.device, self.dtype = config.EMBED_BASE_URL, "api"
        self.dim = len(self.query("dim"))

    def _encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), config.EMBED_BATCH):
            r = self.client.embeddings.create(model=self.model_name, input=texts[i:i + config.EMBED_BATCH])
            out += [d.embedding for d in r.data]
        return out

    def passages(self, texts: list[str]) -> list[list[float]]:
        return self._encode([config.PASSAGE_PREFIX + t for t in texts])

    def query(self, text: str) -> list[float]:
        return self._encode([config.QUERY_PREFIX + text])[0]


def get_embedder() -> Embedder | ApiEmbedder:
    """One shared model per process; loading bge-m3 takes several seconds."""
    global _instance
    with _lock:
        if _instance is None:
            _instance = ApiEmbedder() if config.EMBED_BASE_URL else Embedder()
        return _instance
