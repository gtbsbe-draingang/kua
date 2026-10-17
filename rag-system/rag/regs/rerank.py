"""Second-stage reranking: score (query, chunk) pairs with a model that reads both together.

Retrieval embeds the query and the clause separately, so it ranks by overall similarity. A
reranker reads them as one input and can tell "the cashier must count cash in front of the
client" (p. 31) from "the client counts the cash at the desk" (p. 35), which share most words.
That is what puts the right clause at rank 1.

Two backends, chosen by the model name:
  CrossEncoder   BAAI/bge-reranker-v2-m3 and similar: one relevance logit per pair, sigmoid -> 0..1
  Qwen3Reranker  Qwen/Qwen3-Reranker-*: a causal LM asked "yes" or "no" for the pair under a task
                 instruction (config.RERANK_INSTRUCTION); score = P("yes") from the last logits
Both return scores in 0..1, so a threshold can mean "no clause matches well".
"""
import threading

from . import config
from .embed import _device, _dtype

_lock = threading.RLock()
_instance = None


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder
        self.model_name = model_name
        self.device = _device()
        self.model = CrossEncoder(model_name, device=self.device, max_length=config.RERANK_MAX_TOKENS,
                                  model_kwargs={"dtype": _dtype(self.device)})

    def score(self, query: str, passages: list[str]) -> list[float]:
        import torch
        scores = self.model.predict([(query, p) for p in passages], batch_size=config.RERANK_BATCH,
                                    activation_fn=torch.nn.Sigmoid(), convert_to_numpy=True,
                                    show_progress_bar=False)
        return [float(s) for s in scores]


class Qwen3Reranker:
    """Follows the usage on the Qwen3-Reranker model card."""
    PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
              "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
              "<|im_end|>\n<|im_start|>user\n")
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_name = model_name
        self.device = _device()
        self.tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=_dtype(self.device)).to(self.device).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.prefix_ids = self.tok.encode(self.PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tok.encode(self.SUFFIX, add_special_tokens=False)

    def score(self, query: str, passages: list[str]) -> list[float]:
        import torch
        budget = config.RERANK_MAX_TOKENS - len(self.prefix_ids) - len(self.suffix_ids)
        out: list[float] = []
        for i in range(0, len(passages), config.RERANK_BATCH):
            texts = [f"<Instruct>: {config.RERANK_INSTRUCTION}\n<Query>: {query}\n<Document>: {p}"
                     for p in passages[i:i + config.RERANK_BATCH]]
            enc = self.tok(texts, padding=False, truncation="longest_first", max_length=budget,
                           return_attention_mask=False)
            enc["input_ids"] = [self.prefix_ids + ids + self.suffix_ids for ids in enc["input_ids"]]
            batch = self.tok.pad(enc, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**batch).logits[:, -1, :]
            pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            out.extend(torch.softmax(pair, dim=1)[:, 1].tolist())
        return out


def get_reranker():
    """One shared reranker per process, or None when REGS_RERANK_MODEL is empty."""
    global _instance
    if not config.RERANK_MODEL:
        return None
    with _lock:
        if _instance is None:
            cls = Qwen3Reranker if "qwen3-reranker" in config.RERANK_MODEL.lower() else CrossEncoderReranker
            _instance = cls(config.RERANK_MODEL)
        return _instance


def score(query: str, passages: list[str]) -> list[float]:
    reranker = get_reranker()
    with _lock:  # web requests run in a thread pool; one forward pass at a time
        return reranker.score(query, passages)
