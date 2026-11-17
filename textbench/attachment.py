"""Step 31: does the text move with the scene? OCR reads a card glued to the screen as well as text on a panel.

Every STEP frames, two motions are measured on the same pair of frames:
  label  where the label's whole box goes: normalised template matching in a window of SEARCH pixels, with sub-pixel refinement. Not optical
         flow: its coarse levels see the background moving past a small fixed label and then lock onto a neighbouring letter (scripts/attachment_controls.py,
         product page, shake: truth 0, gain 0.8);
  scene  a similarity transform fitted with RANSAC to corners around the label (a ring: the label's box grown by its height is left
         out, and the ring reaches RING box sizes out; the whole frame if the ring has too few corners), evaluated at the label's centre:
         where a point of the scene at that spot would go. Local on purpose: under an orbit, a panel moves against the far background
         (parallax), so a fit to the whole frame calls text printed on the panel detached. A step counts only if the whole frame moves too, at least
         half as far: a hand or a crowd passing a label under a still camera is not the view moving.

follow gain = sum(label . scene) / sum(scene . scene), over the steps of a video: 1 moves with the scene, 0 stays fixed on the
screen. Only motion of the scene counts as evidence: a still scene gives no evidence, whatever the text does.
Limits: it measures motion, not looks, so a card that follows the scene perfectly still passes; hands or flames filling the frame can
take over the scene fit; a scene that the model redraws without camera motion has no single motion to follow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

STEP = 6  # frames between measurements (16 fps: 0.375 s)
WINDOW = 4  # steps per window for the frame verdict (1.5 s)
MIN_SCENE_INLIERS = 12
SEARCH, MIN_NCC = 64, 0.5  # label search window in pixels, and the lowest match score that counts as finding it
RING = 1.0  # outer edge of the ring, in the label's width plus height
LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
Box = tuple[float, float, float, float]


@dataclass
class Step:
    frame: int
    label: tuple[float, float] | None
    scene: tuple[float, float] | None


def _flow(a: np.ndarray, b: np.ndarray, mask: np.ndarray, count: int, quality: float = 0.01) -> tuple[np.ndarray, np.ndarray] | None:
    """Corners of a inside mask and where they go in b, kept when tracking back lands within a pixel of the start."""
    p0 = cv2.goodFeaturesToTrack(a, count, quality, 5, mask=mask)
    if p0 is None:
        return None
    p1, s1, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, **LK)
    back, s2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, **LK)
    ok = (s1[:, 0] == 1) & (s2[:, 0] == 1) & (np.linalg.norm((p0 - back)[:, 0], axis=1) < 1.0)
    return (p0[ok, 0], p1[ok, 0]) if ok.any() else None


def _box_mask(shape: tuple[int, int], box: Box, grow: float = 0.0) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = box
    m = np.zeros((h, w), np.uint8)
    m[max(0, int(y0 - grow)):min(h, int(y1 + grow) + 1), max(0, int(x0 - grow)):min(w, int(x1 + grow) + 1)] = 255
    return m


def measure(a: np.ndarray, b: np.ndarray, box: Box) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Label and scene motion between grey frames a and b, for the label in box on a; None where there is too little to go on."""
    label = _label_motion(a, b, box)
    hole = _box_mask(a.shape, box, box[3] - box[1])
    outer = _box_mask(a.shape, box, RING * (box[2] - box[0] + box[3] - box[1]))
    ring = _scene_at(a, b, outer & (255 - hole), box)
    full = _scene_at(a, b, 255 - hole, box)
    if ring is None or full is None:
        return label, ring or full
    if np.hypot(*full) < 0.5 * np.hypot(*ring):
        return label, None  # only something near the label moved (a hand, a crowd), not the view: no evidence either way
    return label, ring


def _label_motion(a: np.ndarray, b: np.ndarray, box: Box) -> tuple[float, float] | None:
    h, w = a.shape
    x0, y0 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
    x1, y1 = min(w, int(round(box[2]))), min(h, int(round(box[3])))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    wx0, wy0 = max(0, x0 - SEARCH), max(0, y0 - SEARCH)
    wx1, wy1 = min(w, x1 + SEARCH), min(h, y1 + SEARCH)
    r = cv2.matchTemplate(b[wy0:wy1, wx0:wx1], a[y0:y1, x0:x1], cv2.TM_CCOEFF_NORMED)
    _, score, _, (mx, my) = cv2.minMaxLoc(r)
    if score < MIN_NCC:
        return None
    return wx0 + mx + _peak(r[my, max(0, mx - 1):mx + 2], mx) - x0, wy0 + my + _peak(r[max(0, my - 1):my + 2, mx], my) - y0


def _peak(v: np.ndarray, at: int) -> float:
    """Sub-pixel offset of a peak from its three neighbouring scores (0 at the window's edge)."""
    if at == 0 or len(v) < 3:
        return 0.0
    d = float(v[0] - 2 * v[1] + v[2])
    return 0.5 * float(v[0] - v[2]) / d if d < 0 else 0.0


def _scene_at(a: np.ndarray, b: np.ndarray, mask: np.ndarray, box: Box) -> tuple[float, float] | None:
    rest = _flow(a, b, mask, 600, 0.003)
    if rest is None or len(rest[0]) < MIN_SCENE_INLIERS:
        return None
    m, inl = cv2.estimateAffinePartial2D(rest[0], rest[1], method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if m is None or int(inl.sum()) < max(MIN_SCENE_INLIERS, 0.3 * len(rest[0])):
        return None
    c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 1.0])
    d = m @ c - c[:2]
    return float(d[0]), float(d[1])


def track(frames: Iterable[tuple[int, np.ndarray]], boxes: dict[int, Box], first: int, end: int | None = None) -> list[Step]:
    """Motions every STEP frames from `first`, frames given as (index, RGB). The box is the nearest one known at most
    STEP frames before the pair's first frame; a pair with no box gives no measurement."""
    out: list[Step] = []
    prev, prev_i = None, None
    for i, rgb in frames:
        if i < first or (end is not None and i >= end) or (i - first) % STEP:
            continue
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if prev is not None:
            known = [k for k in boxes if prev_i - STEP <= k <= prev_i]
            if known:
                out.append(Step(prev_i, *measure(prev, g, boxes[max(known)])))
            else:
                out.append(Step(prev_i, None, None))
        prev, prev_i = g, i
    return out


def summary(steps: list[Step]) -> dict:
    """Follow gain and how much evidence stands behind it. scene_path_px: length of the scene's path over the steps where both motions are
    known. coverage: the share of steps where they are."""
    both = [(s.label, s.scene) for s in steps if s.label is not None and s.scene is not None]
    if not both:
        return {"steps": len(steps), "coverage": 0.0, "scene_path_px": 0.0, "label_path_px": 0.0, "slip_path_px": 0.0, "gain": None}
    la, sc = np.array([b[0] for b in both]), np.array([b[1] for b in both])
    denom = float((sc * sc).sum())
    return {"steps": len(steps), "coverage": len(both) / max(1, len(steps)),
            "scene_path_px": float(np.linalg.norm(sc, axis=1).sum()), "label_path_px": float(np.linalg.norm(la, axis=1).sum()),
            "slip_path_px": float(np.linalg.norm(la - sc, axis=1).sum()),
            "gain": float((la * sc).sum() / denom) if denom > 0 else None}


def windows(steps: list[Step]) -> list[dict]:
    """Per window of WINDOW steps: first frame, net scene motion, net slip of the label against it (None when a step lacks a measurement)."""
    out = []
    for k in range(0, max(0, len(steps) - WINDOW + 1)):
        part = steps[k:k + WINDOW]
        if any(s.label is None or s.scene is None for s in part):
            out.append({"frame": part[0].frame, "scene_px": None, "slip_px": None})
            continue
        la, sc = np.sum([s.label for s in part], axis=0), np.sum([s.scene for s in part], axis=0)
        out.append({"frame": part[0].frame, "scene_px": float(np.linalg.norm(sc)), "slip_px": float(np.linalg.norm(la - sc))})
    return out
