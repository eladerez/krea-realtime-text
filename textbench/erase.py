"""Find where given texts are drawn in a frame and erase them (steps 17-18: change only the text in Krea's memory).

text_boxes: RapidOCR boxes whose reading fuzzily matches one of the targets or one of their words (OCR often splits
"WIFI OFF" or drops a space), as (x0, y0, x1, y1) pixel rectangles, padded.
erase: fills each box with the label's own background colour (box_colours: the majority colour inside the box), so
the layout stays and the letters are gone.
"""
from __future__ import annotations

import numpy as np
from rapidfuzz import fuzz

from textbench import ocr_metrics
from textbench.ocr_metrics import normalize

PAD = 6
RING = 6


def text_boxes(frame: np.ndarray, targets: list[str], min_score: float = 70) -> list[tuple[int, int, int, int]]:
    """frame: RGB uint8 (H, W, 3)."""
    if not targets:
        return []
    ocr_metrics.ocr_frame(frame[:8, :8])  # loads the shared RapidOCR instance
    result, _ = ocr_metrics._ocr(frame[:, :, ::-1].copy())
    keys = {normalize(t) for t in targets} | {normalize(w) for t in targets for w in t.split() if len(normalize(w)) >= 3}
    h, w = frame.shape[:2]
    boxes = []
    for quad, text, _score in result or []:
        n = normalize(text)
        if not n or not any(fuzz.ratio(n, k) >= min_score or (len(n) >= 3 and n in k) or (len(k) >= 3 and k in n)
                            for k in keys):
            continue
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        boxes.append((max(0, int(min(xs)) - PAD), max(0, int(min(ys)) - PAD),
                      min(w, int(max(xs)) + PAD), min(h, int(max(ys)) + PAD)))
    return boxes


def box_colours(frame: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(background, ink) of a text box, taken from inside it: its pixels split into two colour groups (2-means on
    RGB); the larger group is the label's background, the smaller the letters. Sampling a ring outside the box
    instead fails on a tight badge (white page around a black "CART 2" badge -> the box was filled white)."""
    x0, y0, x1, y1 = box
    px = frame[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    lum = px.mean(1)
    c = np.stack([px[lum <= np.percentile(lum, 10)].mean(0), px[lum >= np.percentile(lum, 90)].mean(0)])
    for _ in range(8):
        lab = np.linalg.norm(px[:, None] - c[None], axis=2).argmin(1)
        c = np.stack([px[lab == k].mean(0) if (lab == k).any() else c[k] for k in (0, 1)])
    big = int(np.bincount(lab, minlength=2).argmax())
    return np.median(px[lab == big], 0).astype(np.uint8), np.median(px[lab == 1 - big], 0).astype(np.uint8)


def erase(frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = frame.copy()
    for x0, y0, x1, y1 in boxes:
        out[y0:y1, x0:x1] = box_colours(frame, (x0, y0, x1, y1))[0]
    return out
