"""Step 24: write the new text on the surface it is printed on, not on the screen. Before this, the old text's
detector quad was turned into an upright rectangle; on a tilted or receding panel (the salad demo's counter, 22 degrees
in seed 0) the rectangle was filled flat and upright letters drawn in it, so the new text sat on a pale card pasted on
the camera image instead of on the panel (Elad, 31 Oct 2026: "the text look pasted on the camera screen, not the
screen inside the movie").

Here the panel is flattened first: the detector's quad (its 4 corners follow the text's tilt and slant) defines a
homography to a flat canvas; the existing writer (textbench/write_text.py: partial edits, Krea's weight and width,
inpainting) edits the canvas, and only the pixels it changed are warped back onto the frame, feathered, so the rest of
the panel keeps Krea's own pixels, shading and edges. In the older memory frames the quad is followed with a
homography (ECC on the area around it, from the newest frame), so a camera that moves or zooms carries the text along.

Step 31, v2=True. Three faults seen in the 30 s runs and on saved frames (scripts/surface_offline.py):
  incomplete quad   OCR's quad can hold only part of the label: "ITEMS" without its digit (salad seed 1, block 9: the old 4 stayed), or a digit
                    taller than the box (the top of the old 2 stayed beside the new 3). complete_quad grows it to all the ink of the label's line.
  no margin         the new text could reach the panel's edge, where it is clipped (combustion seed 1: ATERIAL MAGNESIUM). room_v2 keeps a margin of a letter height,
                    measures on a canvas wide enough to see the panel's real width, and the text is kept between its edges.
  missed label      OCR reads tilted text worse (salad seed 0, block 6: label found by neither reading, and the write was skipped). find_quads retries on the frame turned.
A label that has left the frame cannot be fixed here: the writer fits the text into what is still visible.
"""
from __future__ import annotations

import cv2
import numpy as np
from rapidfuzz import fuzz

from textbench import ocr_metrics
from textbench.ocr_metrics import normalize
from textbench.write_text import write

MARGIN = 0.6  # canvas margin around the text, in text heights (room for the box padding and a longer new text)
MOST_OF_FRAME = 0.2  # step 31: a label line larger than this share of the frame is not what complete_quad assumes (salad seed 1: 33-38%, and the writer took most of the panel; the other 13 switches: at most 13%)
GAP = 1.2  # step 31: letter heights between ink on the same line that still count as the same label


def text_quads(frame: np.ndarray, targets: list[str], min_score: float = 70) -> dict[str, np.ndarray]:
    """The detector quad (4 x 2, clockwise from top-left) of each target; the largest matching one."""
    ocr_metrics.ocr_frame(frame[:8, :8])  # loads the shared RapidOCR instance
    result, _ = ocr_metrics._ocr(frame[:, :, ::-1].copy())
    out: dict[str, np.ndarray] = {}
    for t in targets:
        keys = {normalize(t)} | {normalize(w) for w in t.split() if len(normalize(w)) >= 3}
        best = None
        for quad, text, _ in result or []:
            n = normalize(text)
            if n and any(fuzz.ratio(n, k) >= min_score or (len(n) >= 3 and n in k) or (len(k) >= 3 and k in n)
                         for k in keys):
                q = np.float32(quad)
                if best is None or cv2.contourArea(q) > cv2.contourArea(best):
                    best = q
        if best is not None:
            out[t] = best
    return out


def find_quads(frame: np.ndarray, targets: list[str], angles: tuple[float, ...] = (0, 24, -24), zoom: float = 1.5) -> dict[str, np.ndarray]:
    """text_quads, and for a target it misses, again on three overlapping views of the frame, enlarged and turned (salad seed 0, block 6: tilted
    letters read as "S" at full size and "ITEMS" enlarged, on the same pixels); the quad is mapped back."""
    out = text_quads(frame, targets)
    h, w = frame.shape[:2]
    for cx in (0.7, 0.3, 0.5):
        for ang in angles:
            missing = [t for t in targets if t not in out]
            if not missing:
                return out
            m = cv2.getRotationMatrix2D((cx * w, h / 2), ang, zoom)
            m[:, 2] += [(0.5 - cx) * w, 0]  # the view's centre goes to the middle of the output
            view = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            back = cv2.invertAffineTransform(m)
            for t, q in text_quads(view, missing).items():
                out[t] = cv2.transform(q[None], back)[0].astype(np.float32)
    return out


def label_colours(flat: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Step 31: (background, ink) of a text box. The two colours come from box_colours, but which is which comes from their shape, not their share:
    a background is one connected piece around the letters, ink is many. By share they swap: on bold letters, and on a box padded past a card's
    edge (combustion seed 0: on 3 of 9 memory frames the black screen outnumbered the white card, the panel collapsed and the old text stayed)."""
    from textbench.erase import box_colours
    a, b = (c.astype(np.float32) for c in box_colours(flat, box))
    x0, y0, x1, y1 = box
    px = flat[y0:y1, x0:x1].astype(np.float32)
    near_a = (np.linalg.norm(px - a, axis=2) < np.linalg.norm(px - b, axis=2)).astype(np.uint8)

    def largest_share(mask: np.ndarray) -> float:
        n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        return float(st[1:, cv2.CC_STAT_AREA].max() / max(1, mask.sum())) if n > 1 else 0.0

    return (a, b) if largest_share(near_a) >= largest_share(1 - near_a) else (b, a)


def complete_quad(frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """The quad grown to all the ink of the label's line, on the flat canvas: ink components touching the quad, then
    those on the same line within GAP letter heights of them, while no taller than 2.5 x the quad and on the panel."""
    Hm, size, inner = canvas_for(quad, grow=3.0)
    x0, y0, x1, y1 = inner
    H = y1 - y0
    flat = cv2.warpPerspective(frame, Hm, size, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    bg, ink = label_colours(flat, inner)
    px = flat.astype(np.float32)
    mask = (np.linalg.norm(px - ink, axis=2) < np.linalg.norm(px - bg, axis=2)).astype(np.uint8)
    panel = cv2.warpPerspective(panel_v2(frame, quad, bg, ink), Hm, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    mask &= panel  # the wall behind a digit that touches the panel's edge is not part of the digit
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    union = [x0, y0, x1, y1]
    used: set[int] = set()
    changed = True
    while changed:
        changed = False
        for i in range(1, n):
            if i in used:
                continue
            cx, cy, cw, ch, area = st[i]
            if ch > 2.5 * H or area < max(4, 0.02 * H * H):
                continue
            overlap = min(cy + ch, y1) - max(cy, y0)
            gap = max(cx - union[2], union[0] - (cx + cw))
            if overlap >= 0.5 * min(ch, H) and gap <= GAP * H:
                used.add(i)
                changed = True
                union = [min(union[0], cx), min(union[1], cy), max(union[2], cx + cw), max(union[3], cy + ch)]
    p = 0.25 * H  # the box keeps some panel around the ink, or bold letters outnumber the panel and the colours swap (combustion, first try)
    union = [union[0] - p, union[1] - p, union[2] + p, union[3] + p]
    u = np.float32([[union[0], union[1]], [union[2], union[1]], [union[2], union[3]], [union[0], union[3]]])
    return cv2.perspectiveTransform(u[None], Hm)[0]


def canvas_for(quad: np.ndarray, grow: float = 1.0) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
    """Homography from a flat canvas to the frame, the canvas size, and the text's box on the canvas. The canvas keeps
    the text's own length and height; grow widens its side margins for a longer new text."""
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    m = MARGIN * h
    mx = m + w * (grow - 1) / 2  # a longer new text grows on both sides (labels are centred)
    W, H = int(round(w + 2 * mx)), int(round(h + 2 * m))
    inner = np.float32([[mx, m], [mx + w, m], [mx + w, m + h], [mx, m + h]])
    return cv2.getPerspectiveTransform(inner, np.float32(quad)), (W, H), (int(mx), int(m), int(mx + w), int(m + h))


def follow(newest: np.ndarray, frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """The quad moved from the newest memory frame to an older one: a homography fitted (ECC) on the area around the
    text, started from the best translation (template matching); the translation alone if ECC does not converge."""
    a, b = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (newest, frame))
    h, w = a.shape
    x0, y0 = np.maximum(quad.min(0), 0).astype(int)
    x1, y1 = np.minimum(quad.max(0), [w, h]).astype(int)
    p = max(8, (y1 - y0) // 2)
    tx0, ty0, tx1, ty1 = max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p)
    s = 48
    sx0, sy0 = max(0, tx0 - s), max(0, ty0 - s)
    win, tpl = b[sy0:min(h, ty1 + s), sx0:min(w, tx1 + s)], a[ty0:ty1, tx0:tx1]
    dx = dy = 0
    if win.shape[0] >= tpl.shape[0] and win.shape[1] >= tpl.shape[1]:
        _, _, _, (mx, my) = cv2.minMaxLoc(cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED))
        dx, dy = sx0 + mx - tx0, sy0 + my - ty0
    warp = np.float32([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    mask = np.zeros_like(a)
    mask[ty0:ty1, tx0:tx1] = 255
    try:  # ECC finds the warp of the older frame's coordinates to the newest's; it maps newest -> older here
        _, warp = cv2.findTransformECC(a.astype(np.float32), b.astype(np.float32), warp, cv2.MOTION_HOMOGRAPHY,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-4), mask, 5)
    except cv2.error:
        pass
    return cv2.perspectiveTransform(quad[None], warp)[0]


def panel_mask(frame: np.ndarray, quad: np.ndarray, bg: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """The surface the text is printed on, as a 0/1 mask: the pixels of the label's background colour that are
    connected to the text, with everything inside its outline (the letters). Edits are kept inside it, so a new text
    never spills past the panel's edge onto the scene behind (step 24, salad seed 1: the slanted quad reached past the
    counter's edge and the dark wall there was painted white)."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = 0, 0, w, h  # the whole frame: a longer new text reaches past the old one's neighbourhood
    px = frame[y0:y1, x0:x1].astype(np.float32)
    near = (np.linalg.norm(px - bg, axis=2) < 0.3 * np.linalg.norm(ink - bg)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(near)
    q = (quad - [x0, y0]).astype(np.int32)
    inside = np.zeros_like(near)
    cv2.fillConvexPoly(inside, q, 1)
    if n < 2:
        return np.ones((h, w), np.uint8)
    best = 1 + int(np.argmax([(inside & (lab == i)).sum() for i in range(1, n)]))
    comp = (lab == best).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    out = np.zeros((h, w), np.uint8)
    out[y0:y1, x0:x1] = filled
    return out


def panel_v2(frame: np.ndarray, quad: np.ndarray, bg: np.ndarray, ink: np.ndarray) -> np.ndarray:
    """Step 31: the surface as a 0/1 mask: the regions of the label's background colour that hold a real share of the text's quad, joined and closed
    over gaps up to the text's height. panel_mask keeps one region and fills only closed holes, so a flame across the text cut the surface in two
    (combustion seed 1: half the old letters could not be edited), and a digit touching the panel's or the frame's edge was left out as a bay in the
    outline (salad, both seeds: the old digit stayed). Not a convex hull: that painted over the bowl beside the panel (first try, seen by eye; OCR passed it)."""
    h, w = frame.shape[:2]
    near = (np.linalg.norm(frame.astype(np.float32) - bg, axis=2) < 0.3 * np.linalg.norm(ink - bg)).astype(np.uint8)
    n, lab = cv2.connectedComponents(near)
    inside = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(inside, quad.astype(np.int32), 1)
    counts = np.bincount(lab[inside == 1], minlength=n)
    counts[0] = 0
    keep = np.flatnonzero(counts >= max(8, 0.10 * counts.sum()))
    if not len(keep):
        return np.ones((h, w), np.uint8)
    comp = np.isin(lab, keep).astype(np.uint8)
    k = max(3, int(0.4 * np.linalg.norm(quad[3] - quad[0])) | 1)  # about a letter's stroke: wider closes the panel across onto the wall (salad seed 1)
    # the border is padded first, so a digit cut by the frame's edge is closed like any other gap
    padded = cv2.copyMakeBorder(comp, k, k, k, k, cv2.BORDER_REPLICATE)
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))[k:-k, k:-k]
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros((h, w), np.uint8)
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out


def clean_label(flat: np.ndarray, box: tuple[int, int, int, int], bg: np.ndarray, ink: np.ndarray, old: str) -> bool:
    """Step 31: no more pieces of ink in the box than the old text has characters. More means fragments of an earlier text are there (salad
    seed 1: a hook of the old 3 beside the digit), and a partial edit would keep them, so the whole line is rewritten instead."""
    x0, y0, x1, y1 = box
    px = flat[max(0, y0):y1, max(0, x0):x1].astype(np.float32)
    mask = (np.linalg.norm(px - ink, axis=2) < np.linalg.norm(px - bg, axis=2)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H = y1 - y0
    pieces = sum(st[i, cv2.CC_STAT_AREA] >= max(4, 0.005 * H * H) for i in range(1, n))
    return pieces <= len(old.replace(" ", ""))


def room(panel: np.ndarray, Hm: np.ndarray, size: tuple[int, int], inner: tuple[int, int, int, int]) -> float:
    """The width the panel offers along the text line, on the flat canvas: the run of panel columns (panel in at
    least half the text's rows) around the text's centre, less a small margin. A longer new text is made smaller to
    fit (step 24, combustion seed 1: MATERIAL COPPER at the old letters' height was wider than the screen)."""
    flat = cv2.warpPerspective(panel, Hm, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    x0, y0, x1, y1 = inner
    cols = flat[y0:y1].mean(0) >= 0.5
    c = (x0 + x1) // 2
    if not cols[c]:
        return float(size[0])
    a, b = c, c
    while a > 0 and cols[a - 1]:
        a -= 1
    while b < len(cols) - 1 and cols[b + 1]:
        b += 1
    return 0.92 * (b - a + 1) - (y1 - y0) * 0.2


def room_v2(panel: np.ndarray, Hm: np.ndarray, size: tuple[int, int], inner: tuple[int, int, int, int]) -> tuple[float, float] | None:
    """Step 31: the panel's run along the text line on the flat canvas, less about a letter's height at each side, as (left, right)
    canvas columns. Off-frame columns are not panel, so a label leaving the frame gets only what is still visible."""
    flat = cv2.warpPerspective(panel, Hm, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
    x0, y0, x1, y1 = inner
    cols = flat[y0:y1].mean(0) >= 0.5
    centre = (x0 + x1) // 2
    on = np.flatnonzero(cols)
    if not len(on):
        return None
    c = int(on[np.argmin(np.abs(on - centre))])
    a, b = c, c
    while a > 0 and cols[a - 1]:
        a -= 1
    while b < len(cols) - 1 and cols[b + 1]:
        b += 1
    m = (y1 - y0) // 2  # the box's height includes its padding, so half of it is about a letter's height
    return (float(a + m), float(b - m)) if b - a > 3 * m else (float(a), float(b))


def write_on(frame: np.ndarray, pairs: list[tuple[str, str]], quads: dict[str, np.ndarray], root: str, align: str,
             boxes: dict | None, partial_edit: bool, v2: bool = False) -> tuple[np.ndarray, dict]:
    """Each old text's panel flattened, edited by the existing writer, and its changed pixels warped back.
    boxes: the writer's choices from the newest frame (canvas boxes, partial or not), reused on the older frames."""
    out = frame.astype(np.float32)
    used: dict = {}
    for old, new in pairs:
        if old not in quads:
            continue
        Hm, size, inner = canvas_for(quads[old], grow=3.0 if v2 else max(1.0, len(new) / max(1, len(old))) + 0.3)
        flat = cv2.warpPerspective(frame, Hm, size, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                   borderMode=cv2.BORDER_REPLICATE)
        from textbench.erase import box_colours
        # colours from the text box padded by half its height: on the tight box bold letters can outnumber the panel
        # (combustion seed 1: white letters on a black screen came out as a white panel)
        e = (inner[3] - inner[1]) // 2
        wide = (max(0, inner[0] - e), max(0, inner[1] - e), min(size[0], inner[2] + e), min(size[1], inner[3] + e))
        if v2:  # the newest frame's colours on every memory frame: they must not swap from frame to frame
            kept = None if boxes is None else boxes.get(f"{old}|_colours:{old}")
            bg0, ink0 = (np.float32(kept[0]), np.float32(kept[1])) if kept else label_colours(flat, inner)
        else:
            bg0, ink0 = (c.astype(np.float32) for c in box_colours(flat, wide))
        panel = (panel_v2 if v2 else panel_mask)(frame, quads[old], bg0, ink0)
        lim = {old: room(panel, Hm, size, inner)}
        span = room_v2(panel, Hm, size, inner) if v2 else None
        if span is not None:
            lim = {old: span[1] - span[0], f"_partial:{old}": max(0.0, span[1] - inner[0])}
        given = None if boxes is None else {k.split("|", 1)[1]: v for k, v in boxes.items() if k.startswith(old + "|")}
        if given is not None and span is not None:
            given[f"_span:{old}"] = span
            lim[old] = min(lim[old], given.get(f"_width:{old}", lim[old]))  # the newest frame's width: the text must not change size across the memory
        if given is None:  # the newest frame: the box is OCR's box on the flat canvas joined with the quad's own (step
            # 25c poster: OCR's canvas box held only "MAY 2", the old 4 was never erased and MAY 2524 came out)
            from textbench.erase import PAD, text_boxes
            ib = (inner[0] - PAD, inner[1] - PAD, inner[2] + PAD, inner[3] + PAD)
            found = text_boxes(flat, [old])
            if found:
                f0 = max(found, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
                ib = (min(ib[0], f0[0]), min(ib[1], f0[1]), max(ib[2], f0[2]), max(ib[3], f0[3]))
            start = {old: ib, f"_lines:{old}": 1}
            if span is not None:
                start[f"_span:{old}"] = span
            if v2 and not clean_label(flat, ib, bg0, ink0, old):
                start[f"_partial:{old}"] = False
            edited, b = write(flat, [(old, new)], root, align, start, partial_edit, "inpaint", lim, check=True)
        else:
            edited, b = write(flat, [(old, new)], root, align, given, partial_edit, "inpaint", lim)
        if v2 and given is None:
            b[f"_width:{old}"] = lim[old]
            b[f"_colours:{old}"] = (bg0.tolist(), ink0.tolist())
        used.update({f"{old}|{k}": v for k, v in b.items()})
        # only what the writer changed goes back, feathered by a pixel or two: the panel keeps Krea's own pixels
        diff = (np.abs(edited.astype(np.int16) - flat.astype(np.int16)).max(2) > 6).astype(np.uint8)
        diff = cv2.dilate(diff, np.ones((3, 3), np.uint8))
        alpha = cv2.GaussianBlur(diff.astype(np.float32), (5, 5), 0)
        fh, fw = frame.shape[:2]
        back = cv2.warpPerspective(edited, Hm, (fw, fh), flags=cv2.INTER_CUBIC)
        a = cv2.warpPerspective(alpha, Hm, (fw, fh), flags=cv2.INTER_LINEAR)
        a = (a * cv2.GaussianBlur(panel.astype(np.float32), (3, 3), 0))[..., None]
        out = out * (1 - a) + back.astype(np.float32) * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8), used


def contained(before: np.ndarray, after: np.ndarray, quads: dict[str, np.ndarray], most: float = 1.5) -> bool:
    """Step 31: did the write stay on the label? Changed pixels at most `most` x the label's area. OCR cannot be the check: it read the right
    text on frames where a first try had painted over the bowl. Edits are already clipped to panel_v2, so this catches a wrong panel, not a spill."""
    changed = int((np.abs(before.astype(np.int16) - after.astype(np.int16)).max(2) > 12).sum())
    return changed <= most * sum(cv2.contourArea(q) for q in quads.values())


def text_width(frame: np.ndarray, quad: np.ndarray, bg: np.ndarray, ink: np.ndarray) -> float | None:
    """Step 31: the width the panel offers the text on one frame, in the flat canvas's pixels."""
    Hm, size, inner = canvas_for(quad, grow=3.0)
    span = room_v2(panel_v2(frame, quad, bg, ink), Hm, size, inner)
    return None if span is None else span[1] - span[0]


def write_cache(cache: list[np.ndarray], pairs: list[tuple[str, str]], root: str, align: str = "center",
                partial_edit: bool = True, v2: bool = False) -> tuple[list[np.ndarray], int]:
    """The memory frames (oldest first, RGB) with each old text rewritten on its surface. Returns the frames and the
    number of texts written (0: the old text was not found on the newest frame; the cache is returned unchanged).
    v2: step 31, see the top of this file."""
    newest = cache[-1]
    targets = [o for o, _ in pairs]
    quads = find_quads(newest, targets) if v2 else text_quads(newest, targets)
    if not quads:
        return cache, 0
    if v2:
        grown = {k: complete_quad(newest, v) for k, v in quads.items()}
        if any(cv2.contourArea(g) > MOST_OF_FRAME * newest.shape[0] * newest.shape[1] for g in grown.values()):
            return write_cache(cache, pairs, root, align, partial_edit, False)  # the step 24b writer instead
        quads = grown
    result, boxes = write_on(newest, pairs, quads, root, align, None, partial_edit, v2)
    if v2 and not contained(newest, result, quads):  # never trust a v2 write that repaints the scene: the step 24b writer instead
        return write_cache(cache, pairs, root, align, partial_edit, False)
    followed = [quads if i == len(cache) - 1 else {k: follow(newest, f, v) for k, v in quads.items()} for i, f in enumerate(cache)]
    if v2:  # one size across the memory: the narrowest width any frame offers (poster seed 1: the sax player narrowed the panel on the older frames)
        for old, _ in pairs:
            bg, ink = (np.float32(c) for c in boxes[f"{old}|_colours:{old}"]) if f"{old}|_colours:{old}" in boxes else (None, None)
            if bg is None:
                continue
            widths = [w for f, q in zip(cache, followed) if old in q and (w := text_width(f, q[old], bg, ink)) is not None]
            if widths:
                boxes[f"{old}|_width:{old}"] = min(widths)
    out = []
    for i, f in enumerate(cache):
        q = followed[i]
        out.append(write_on(f, pairs, q, root, align, boxes, partial_edit, v2)[0])
    return out, len(quads)


def blank_quad(frame: np.ndarray, min_frac: float = 0.01, max_frac: float = 0.45) -> np.ndarray | None:
    """Step 25c: a blank label (no text to find): the most rectangular flat region of the frame. Low-texture pixels
    (Sobel magnitude of the blurred grey image below a threshold) are split by colour into connected regions; each
    region's outline is reduced to 4 corners (approxPolyDP on its convex hull); the score is area x fill (region area /
    quad area), for regions between min_frac and max_frac of the frame that fill their quad to 85% or more.
    Returns the quad clockwise from top-left, or None."""
    h, w = frame.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    flat = (mag < 12).astype(np.uint8)
    light = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)[..., 0].astype(np.int32)
    # candidate regions: the flat pixels alone (a label's edges are not flat), and flat pixels split into brightness
    # bands (a white label against a white wall), the bands also shifted by half a step (salad seed 0, step 25c: a
    # label with a gentle gradient straddled a band edge and split in two)
    masks = [flat]
    for off in (0, 16):
        q = (light + off) // 32
        masks += [((q == v) & (flat == 1)).astype(np.uint8) for v in np.unique(q)]
    best, score = None, 0.0
    for mk in masks:
        m = cv2.morphologyEx(mk, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, labels, st, _ = cv2.connectedComponentsWithStats(m)
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA]
            if not min_frac * h * w <= a <= max_frac * h * w:
                continue
            bx, by, bw, bh = st[i, :4]
            if bx <= 1 or by <= 1 or bx + bw >= w - 1 or by + bh >= h - 1:
                continue  # touching the frame's edge: background, not a label (step 25c combustion seed 2)
            comp = (labels == i).astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
            poly = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
            if len(poly) != 4:
                continue
            qa = cv2.contourArea(poly)
            fill = a / max(qa, 1.0)
            if fill >= 0.85 and a * fill > score:
                best, score = poly.reshape(4, 2).astype(np.float32), a * fill
    if best is None:
        return None
    c = best.mean(0)  # order clockwise from top-left
    ang = np.arctan2(best[:, 1] - c[1], best[:, 0] - c[0])
    best = best[np.argsort(ang)]
    k = int(np.argmin(best.sum(1)))
    return np.roll(best, -k, axis=0)


def place_text(frame: np.ndarray, quad: np.ndarray, text: str, root: str, fill: float = 0.55) -> np.ndarray:
    """Step 25c: draw text on a blank label given by its quad: flattened, the text centred, its capitals `fill` of the
    label's height (and at most 85% of its width), black on a light label and white on a dark one, warped back."""
    from PIL import Image, ImageDraw
    from textbench.write_text import font_for
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    W, H = int(round(w)), int(round(h))
    Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), np.float32(quad))
    flat = cv2.warpPerspective(frame, Hm, (W, H), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)
    ink = (20, 20, 20) if flat.mean() > 128 else (240, 240, 240)
    cap = max(8, int(fill * H))
    f = font_for(cap, root, (700, 100))
    while cap > 8 and (lambda b: b[2] - b[0])(f.getbbox(text)) > 0.85 * W:
        cap -= 1
        f = font_for(cap, root, (700, 100))
    a = Image.new("L", (W, H))
    l, t, r, b = f.getbbox(text)
    ImageDraw.Draw(a).text(((W - (r - l)) / 2 - l, (H - (b - t)) / 2 - t), text, font=f, fill=255)
    alpha = cv2.warpPerspective(np.asarray(a, np.float32) / 255, Hm, frame.shape[1::-1], flags=cv2.INTER_LINEAR)[..., None]
    out = frame.astype(np.float32) * (1 - alpha) + np.float32(ink) * alpha
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def place_cache(cache: list[np.ndarray], text: str, root: str) -> tuple[list[np.ndarray], int]:
    """Step 25c: the memory frames with text placed on the blank label found in the newest one (followed into the
    older ones by homography). Returns the frames and 1, or the cache unchanged and 0 if no label is found."""
    newest = cache[-1]
    quad = blank_quad(newest)
    if quad is None:
        return cache, 0
    out = [place_text(f, quad if i == len(cache) - 1 else follow(newest, f, quad), text, root)
           for i, f in enumerate(cache)]
    return out, 1
