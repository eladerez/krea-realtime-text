"""The live display checker (LIVE.md Changes, 15 Nov 2026; Elad: "look at the calculator screen in real time to see if
the numbers are correct"). After every block the newest frame's display is read by OCR and compared with the number
that should be there (what the viewer typed and the loop wrote).

read_display: RapidOCR (detector + recogniser) on the whole frame (on a crop around the display the detector found
nothing in 3 of 9 saved live frames; on the whole frame it found the digits), keeping the lines in a window around the
display's box, widened by `margin` of the box's size each way, because the calculator moves a little between memory
writes and resets (15 Nov frames: the digits had slid right of the fixed box). Of those lines, the ones with digits
whose height is between `min_h` and 1.5 times the box's and that share at least 60% of their rows with the box (the
keypad's digit labels below the display are dropped) are joined left to right. If the detector finds nothing, the
recogniser alone reads the box as one line (kept only when it is all digits; "how": "rec", logged but never a reason
to fix, code-check-r1 #2: a recogniser gives confident digits on texture). Returns the digits read, the lowest line
score, and the joined box in frame pixels.
verdict: "ok" (reads exactly the number), "wrong" (reads other digits), "unread" (no digits found: display blurred,
covered or out of frame). An OCR misreading is possible either way; every block's display crop is saved for audit.
"""
from __future__ import annotations

import re

import numpy as np

from textbench import ocr_metrics


def digits(text: str) -> str:
    return re.sub(r"[^0-9]", "", text or "")


def _engine():
    """The shared RapidOCR instance (textbench/ocr_metrics.py), loaded once (code-check-r1 #10)."""
    if ocr_metrics._ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        ocr_metrics._ocr = RapidOCR()
    return ocr_metrics._ocr


def clamp(box: tuple[int, int, int, int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    return max(0, x0), max(0, y0), min(w, max(x1, x0 + 1)), min(h, max(y1, y0 + 1))


def read_display(frame: np.ndarray, box: tuple[int, int, int, int], margin: float = 0.5,
                 min_h: float = 0.45) -> dict:
    """frame: RGB uint8 (H, W, 3); box: (x0, y0, x1, y1) of the display's digits."""
    h_img, w_img = frame.shape[:2]
    x0, y0, x1, y1 = clamp(box, frame.shape)
    bw, bh = x1 - x0, y1 - y0
    cx0, cx1 = max(0, int(x0 - margin * bw)), min(w_img, int(x1 + margin * bw))
    cy0, cy1 = max(0, int(y0 - margin * bh)), min(h_img, int(y1 + margin * bh))
    result, _ = _engine()(frame[:, :, ::-1].copy())
    lines = []
    for quad, text, score in result or []:
        d = digits(text)
        if not d:
            continue
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        lx0, ly0, lx1, ly1 = min(xs), min(ys), max(xs), max(ys)
        if lx1 < cx0 or lx0 > cx1 or ly1 < cy0 or ly0 > cy1:  # outside the window around the display
            continue
        overlap = min(ly1, y1) - max(ly0, y0)  # rows shared with the display's box
        if not min_h * bh <= ly1 - ly0 <= 1.5 * bh or overlap < 0.6 * (ly1 - ly0):
            continue
        lines.append((lx0, d, float(score), (int(lx0), int(ly0), int(lx1), int(ly1))))
    if lines:
        lines.sort()
        bx = (min(b[0] for *_, b in lines), min(b[1] for *_, b in lines),
              max(b[2] for *_, b in lines), max(b[3] for *_, b in lines))
        return {"reads": "".join(d for _, d, _, _ in lines), "score": round(min(s for _, _, s, _ in lines), 3),
                "box": bx, "how": "det"}
    res, _ = _engine()(frame[y0:y1, x0:x1, ::-1].copy(), use_det=False, use_cls=False)
    for text, score in [(r[0], float(r[1])) for r in res or []]:
        t = re.sub(r"\s", "", text or "")
        if t and t.isdigit():
            return {"reads": t, "score": round(score, 3), "box": (x0, y0, x1, y1), "how": "rec"}
    return {"reads": "", "score": 0.0, "box": None, "how": "none"}


def verdict(reads: str, want: str) -> str:
    if not reads:
        return "unread"
    return "ok" if reads == digits(want) else "wrong"


def follow(box: tuple[int, int, int, int], found: tuple[int, int, int, int], align: str = "right",
           anchor: tuple[int, int, int, int] | None = None, max_shift: int = 120,
           shape: tuple[int, ...] | None = None) -> tuple[int, int, int, int]:
    """Move the display's box (its size kept) to where the digits were found: right edges aligned for a right-aligned
    display (left edges for left, centres for center), rows centred. If the moved box would be more than max_shift
    pixels from `anchor` (the start image's box; code-check-r1 #9: a cap on the total move, not per block), the box is
    returned unchanged (a misdetection). Clamped to the frame when `shape` is given."""
    x0, y0, x1, y1 = box
    fx0, fy0, fx1, fy1 = found
    dx = {"right": fx1 - x1, "left": fx0 - x0}.get(align, (fx0 + fx1 - x0 - x1) // 2)
    dy = (fy0 + fy1 - y0 - y1) // 2
    new = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
    a = anchor or box
    if abs(new[0] - a[0]) > max_shift or abs(new[1] - a[1]) > max_shift:
        return box
    return clamp(new, shape) if shape is not None else new
