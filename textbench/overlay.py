"""Step 25, variant B: the letters as a layer of their own (Elad, 1 Nov 2026: "a component that holds the actual
letters ... mixed with the picture at the end"). Krea's memory never sees them: whatever Krea draws on the label (its
own text, a placeholder) is only the surface; on every output frame the label's quad is tracked and the wanted text
is drawn on the surface (textbench/write_surface.py write_on: flattened, old ink inpainted, new line drawn, warped
back, clipped to the panel). A text change is then a change of the layer, with nothing stale in memory to fight.

Tracking: see overlay(). First version (step 25 offline): the detector quad re-read every frame shrank to the visible
part when an arm covered the label or it slid off-screen ("ITEMS" without the "2"), and the new text was drawn small
inside it with the old letters left.
"""
from __future__ import annotations

import cv2
import numpy as np

from textbench.ocr_metrics import normalize
from textbench.write_surface import follow, text_quads, write_on


def full(found: dict[str, np.ndarray], texts: list[str], frame: np.ndarray) -> tuple[str, np.ndarray] | None:
    """A detection of one of the texts read whole (OCR's reading equals it after normalising)."""
    from textbench import ocr_metrics
    for k, q in found.items():
        x0, y0 = np.maximum(q.min(0).astype(int) - 4, 0)
        x1, y1 = q.max(0).astype(int) + 4
        res, _ = ocr_metrics._ocr(frame[y0:y1, x0:x1, ::-1].copy())
        if normalize("".join(r[1] for r in res or [])) == normalize(k):
            return k, q
    return None


def homography(key: np.ndarray, frame: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Key frame -> frame homography from ORB features around the label (the quad grown by its height sideways and
    twice its height up and down: the panel's letters and edges), RANSAC, so a hand over part of the panel does not
    move the fit. Returns the homography (None if fewer than 12 inliers) and the inlier count."""
    a, b = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (key, frame))
    tl, tr, br, bl = quad
    u = (tr - tl) / max(1e-6, float(np.linalg.norm(tr - tl)))
    v = (bl - tl) / max(1e-6, float(np.linalg.norm(bl - tl)))
    h = float(np.linalg.norm(bl - tl))
    grown = np.int32([tl - h * u - 2 * h * v, tr + h * u - 2 * h * v, br + h * u + 2 * h * v, bl - h * u + 2 * h * v])
    mask = np.zeros_like(a)
    cv2.fillConvexPoly(mask, grown, 255)
    orb = cv2.ORB_create(3000)
    ka, da = orb.detectAndCompute(a, mask)
    kb, db = orb.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 12:
        return None, 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
    if len(m) < 12:
        return None, 0
    src = np.float32([ka[x.queryIdx].pt for x in m])
    dst = np.float32([kb[x.trainIdx].pt for x in m])
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    n = int(inl.sum()) if inl is not None else 0
    return (H, n) if H is not None and n >= 12 else (None, n)


def overlay(frames: list[np.ndarray], wanted: list[str], seen: list[str], root: str, start: int = 0,
            min_inliers: int = 40) -> tuple[list[np.ndarray], dict]:
    """frames: RGB, in order. wanted[i]: the text to show on frame i. seen: the texts Krea may have drawn on the
    label (any of the stages). The label is measured on every frame where OCR reads one of them whole (that frame
    becomes the key: Krea redraws its label every frame, and it grows and shifts against the panel as the camera
    moves, so a key kept for long drifts off it); other frames (label covered or cut by the frame edge) get the key
    quad moved by the key -> frame homography (homography()), or by ECC from the previous frame when that fails. The same flattened box is rewritten on every frame (the
    writer's choices from the key frame), so a label that is half covered or half off-screen is still drawn whole."""
    out: list[np.ndarray] = []
    quad = key = kq = kname = boxes = prev = None
    stats: dict = {"written": 0, "keys": 0, "ecc": 0, "lost": 0, "centres": [], "inliers": []}
    for i, f in enumerate(frames):
        if i < start:
            out.append(f)
            prev = f
            continue
        n = 0
        if key is not None:
            H, n = homography(key, f, kq)
            if H is not None:
                quad = cv2.perspectiveTransform(kq[None], H)[0]
            else:
                quad = follow(prev, f, quad)
                stats["ecc"] += 1
            stats["inliers"].append(n)
        hit = full(text_quads(f, seen), seen, f)  # Krea redraws its label every frame: re-measure when read whole
        if hit:
            kname, kq = hit
            key, quad = f, kq
            _, boxes = write_on(f, [(kname, wanted[i])], {kname: kq}, root, "center", None, False)
            stats["keys"] += 1
        prev = f
        if quad is None:
            out.append(f)
            continue
        try:
            g, _ = write_on(f, [(kname, wanted[i])], {kname: quad}, root, "center", boxes, False)
        except Exception:  # a degenerate quad: show Krea's frame
            stats["lost"] += 1
            g = f
        out.append(g)
        stats["written"] += 1
        stats["centres"].append([round(float(v), 1) for v in quad.mean(0)])
    return out, stats
