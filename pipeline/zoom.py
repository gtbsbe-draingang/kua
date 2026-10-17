"""Person zoom: find the people in a window and give the padded union box to crop it to."""
from __future__ import annotations

import io
import threading

from PIL import Image

from pipeline import frames

DETECTOR = "PekingU/rtdetr_r50vd_coco_o365"
SCORE_MIN = 0.5
PAD = 0.10
MAX_AREA = 0.85
LONG_SIDE = 896

_lock = threading.Lock()
_model = None
_processor = None


def _even_down(v: float) -> int:
    return 2 * int(v // 2)


def _even_up(v: float) -> int:
    return -2 * int(-v // 2)


def union_box(boxes: list[tuple[float, float, float, float]], width: int, height: int,
              pad: float = PAD) -> tuple[int, int, int, int] | None:
    """Union of boxes padded by `pad` of its own size per side, clamped to the frame, with x0/y0 rounded
    down and x1/y1 up to even. None for no boxes, or when the result covers >= MAX_AREA of the frame."""
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0 = max(_even_down(max(x0 - px, 0.0)), 0)
    y0 = max(_even_down(max(y0 - py, 0.0)), 0)
    x1 = min(_even_up(min(x1 + px, float(width))), width)
    y1 = min(_even_up(min(y1 + py, float(height))), height)
    if x1 <= x0 or y1 <= y0:
        return None
    if (x1 - x0) * (y1 - y0) >= MAX_AREA * width * height:
        return None
    return x0, y0, x1, y1


def _load():
    """The detector, loaded once from the local snapshot (offline) onto CUDA."""
    global _model, _processor
    with _lock:
        if _model is None:
            import torch
            from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
            proc = RTDetrImageProcessor.from_pretrained(DETECTOR, local_files_only=True)
            model = RTDetrForObjectDetection.from_pretrained(DETECTOR, local_files_only=True)
            _model = model.to("cuda").eval()
            _processor = proc
            torch.set_grad_enabled(False)
        return _model, _processor


def _detect(images: list[Image.Image]) -> list[list[tuple[str, float, tuple[float, float, float, float]]]]:
    """Per image, every detection as (label, score, (x0, y0, x1, y1) in pixels of that image)."""
    import torch
    model, proc = _load()
    inputs = proc(images=images, return_tensors="pt").to("cuda")
    with _lock, torch.inference_mode():
        out = model(**inputs)
    sizes = torch.tensor([[im.height, im.width] for im in images])
    results = proc.post_process_object_detection(out, threshold=0.0, target_sizes=sizes)
    id2label = model.config.id2label
    return [[(id2label[int(lab)], float(score), tuple(float(v) for v in box))
             for score, lab, box in zip(r["scores"], r["labels"], r["boxes"])] for r in results]


def person_box(path: str, start: float, end: float) -> tuple[int, int, int, int] | None:
    """Padded union box of the people seen at start, mid and end of the window, in pixels of the frames of
    `path` (the upright copy windows are cut from). Raises if the detector fails."""
    images = []
    for t in (start, (start + end) / 2, end):
        images.append(Image.open(io.BytesIO(frames.jpeg_at(path, t).jpeg)).convert("RGB"))
    w, h = images[0].size
    people = [box for dets in _detect(images) for label, score, box in dets
              if label == "person" and score >= SCORE_MIN]
    return union_box(people, w, h)
