"""A general text writer for Krea's memory (LIVE.md 17 Nov 2026, route 1; Elad: "I want to generalize digits and
letters on many different things"). The calculator's 7-segment writer only works on grey LCDs; this one works on any
flat surface: the calculator runs showed that a plain pasted font in memory is enough, the LoRA makes Krea draw it in
the scene's style.

find_text: the text on a frame by OCR (RapidOCR quads), the line (or neighbouring lines on one row) best matching the
wanted text; returns its quad (tl, tr, br, bl, float pixels) along the text's own slant.
TextTracker: the quad found once (on the start image) and moved to later frames by a homography of ORB features in a
window around it (start -> frame), as the LCD tracker's template match, but for any surface and perspective.
erase: the old text removed: inside the quad (padded) the surface colour is a robust quadratic fit of its pixels (text
strokes fall out as outliers); pixels far from the fit (the strokes, dilated) are filled with the fit plus a fine grain.
paste: the new text in a narrow bold font (PASTE_FONT), the ink colour measured from the old strokes, the size from the
old line's height, aligned to the old line (left, right or centre), slanted with the quad; soft alpha. It stays on its
surface: the surface's extent around the old line is measured on the erased frame (surface_span) and the text is
shrunk and moved to fit inside it with a margin (LIVE.md 18 Nov, Elad on the live door: "looks pasted. The digits
spill out of the sign").
write: erase + paste. Every step returns None rather than guessing when the text is not found.
"""
from __future__ import annotations

import difflib
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pathlib import Path

# code-tscene-r1 blocker 2: the font ships with the repo (assets/fonts; DejaVu fonts licence), not the system's
FONT = str(Path(__file__).resolve().parents[1] / "assets/fonts/DejaVuSans-Bold.ttf")  # find_field's width shares
# LIVE.md 18 Nov: the pasted glyphs in a narrow grotesque, closer to the signs Krea draws than DejaVu's wide bold
# (Liberation Sans Narrow Bold, GPLv2 with the font exception: assets/fonts/LiberationSansNarrow-LICENSE.txt)
PASTE_FONT = str(Path(__file__).resolve().parents[1] / "assets/fonts/LiberationSansNarrow-Bold.ttf")


def norm(text: str) -> str:
    return re.sub(r"[^0-9A-Z$%.]", "", (text or "").upper())


def same(read: str, want: str) -> bool:
    """A reading equals the wanted text, dots aside (code-tscene-r1 should-know 1: the calculator's printed "%"/dot
    marks read as "68.")."""
    return norm(read).replace(".", "") == norm(want).replace(".", "")


def _engine():
    from textbench.display_check import _engine as e
    return e()


def ocr_lines(frame: np.ndarray) -> list[tuple[np.ndarray, str, float]]:
    result, _ = _engine()(np.ascontiguousarray(frame[:, :, ::-1]))
    return [(np.asarray(q, np.float32), t, float(s)) for q, t, s in (result or [])]


def _order(pts: np.ndarray) -> np.ndarray:
    """4 corners of the min-area rectangle around pts, as tl, tr, br, bl with the long side as the text direction."""
    rect = cv2.minAreaRect(pts.astype(np.float32))
    box = cv2.boxPoints(rect)
    c = box.mean(0)
    # the text runs along the longer side, left to right
    e = [box[(i + 1) % 4] - box[i] for i in range(4)]
    i = int(np.argmax([np.linalg.norm(v) for v in e[:2]]))
    u = e[i] / (np.linalg.norm(e[i]) + 1e-6)
    if u[0] < 0:
        u = -u
    v = np.array([-u[1], u[0]])  # down, perpendicular
    if v[1] < 0:
        v = -v
    pu, pv = (box - c) @ u, (box - c) @ v
    hw, hh = np.abs(pu).max(), np.abs(pv).max()
    return np.stack([c - hw * u - hh * v, c + hw * u - hh * v, c + hw * u + hh * v, c - hw * u + hh * v]).astype(np.float32)


def find_text(frame: np.ndarray, want: str, lines=None, min_ratio: float = 0.6) -> dict | None:
    """The OCR line, or run of up to 3 lines on one row (left to right), that best matches `want`."""
    lines = ocr_lines(frame) if lines is None else lines
    w = norm(want)
    if not w or not lines:
        return None
    lines = sorted(lines, key=lambda x: x[0][:, 0].min())
    best = None
    for i in range(len(lines)):
        run = [lines[i]]
        for j in range(i, min(i + 3, len(lines))):
            if j > i:
                q0, q1 = run[-1][0], lines[j][0]
                y0a, y1a, y0b, y1b = q0[:, 1].min(), q0[:, 1].max(), q1[:, 1].min(), q1[:, 1].max()
                if min(y1a, y1b) - max(y0a, y0b) < 0.5 * min(y1a - y0a, y1b - y0b):
                    break
                run = run + [lines[j]]
            text = "".join(norm(t) for _, t, _ in run)
            r = difflib.SequenceMatcher(None, text, w).ratio()
            if best is None or r > best["ratio"]:
                best = {"quad": _order(np.concatenate([q for q, _, _ in run])), "read": text, "ratio": r,
                        "score": min(s for _, _, s in run)}
    return best if best and best["ratio"] >= min_ratio else None


def find_field(frame: np.ndarray, line: str, field: str, lines=None) -> dict | None:
    """The quad of `field`, a part of the text line `line` ("42" in "ROUTE 42"): the field alone when OCR read it as
    its own line (signs over several rows), else the line's quad cut at the field's share of the line's width (from the
    font's advance widths, the line as written in `line`)."""
    lines = ocr_lines(frame) if lines is None else lines
    alone = find_text(frame, field, lines, min_ratio=0.99)
    whole = find_text(frame, line, lines)
    if alone is not None and (whole is None or norm(line) == norm(field) or whole["ratio"] < 0.9):
        return alone
    if whole is None:
        return alone
    if norm(line) == norm(field):
        return whole
    # the cut is placed by what OCR read on the line (it may hold quotes, a "%", a trailing mark the prompt does not),
    # else by the prompt's line
    ref, fld = whole["read"], norm(field)
    i = ref.rfind(fld) if fld and fld in ref else -1
    if i < 0:
        ref, fld = line.upper(), field.upper()
        i = ref.find(fld)
        if i < 0:
            return whole
    f = ImageFont.truetype(FONT, 100)
    total = f.getlength(ref)
    a0, a1 = f.getlength(ref[:i]) / total, f.getlength(ref[:i + len(fld)]) / total
    tl, tr, br, bl = whole["quad"]
    q = np.stack([tl + a0 * (tr - tl), tl + a1 * (tr - tl), bl + a1 * (br - bl), bl + a0 * (br - bl)]).astype(np.float32)
    return {**whole, "quad": q, "line_quad": whole["quad"]}


class TextTracker:
    """The start image's text quad moved to later frames by an ORB homography in a window around it."""

    def __init__(self, start: np.ndarray, quad: np.ndarray, pad: float = 1.5) -> None:
        self.quad = quad.astype(np.float32)
        h = float(np.linalg.norm(quad[3] - quad[0]))
        x0, y0 = quad.min(0) - pad * h
        x1, y1 = quad.max(0) + pad * h
        H, W = start.shape[:2]
        self.win = (max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1)))
        self.orb = cv2.ORB_create(1500)
        g = cv2.cvtColor(start, cv2.COLOR_RGB2GRAY)
        mask = np.zeros_like(g)
        mask[self.win[1]:self.win[3], self.win[0]:self.win[2]] = 255
        self.kp, self.des = self.orb.detectAndCompute(g, mask)

    def locate(self, frame: np.ndarray, min_inliers: int = 25) -> tuple[np.ndarray | None, int]:
        """The quad in `frame` and the homography's inlier count (None when too few)."""
        if self.des is None or len(self.kp) < min_inliers:
            return None, 0
        g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        x0, y0, x1, y1 = self.win
        m = 80  # the surface may have moved: a wider window in the frame
        mask = np.zeros_like(g)
        mask[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m] = 255
        kp, des = self.orb.detectAndCompute(g, mask)
        if des is None or len(kp) < min_inliers:
            return None, 0
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(self.des, des)
        if len(matches) < min_inliers:
            return None, 0
        src = np.float32([self.kp[mm.queryIdx].pt for mm in matches])
        dst = np.float32([kp[mm.trainIdx].pt for mm in matches])
        hmat, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        n = int(inl.sum()) if inl is not None else 0
        if hmat is None or n < min_inliers:
            return None, n
        q = cv2.perspectiveTransform(self.quad[None], hmat)[0]
        # code-tscene-r1 should-know 2: a wrong homography passes a low inlier count; reject jumps in size or place
        a0 = cv2.contourArea(self.quad.astype(np.float32))
        a1 = cv2.contourArea(q.astype(np.float32))
        h0 = float(np.linalg.norm(self.quad[3] - self.quad[0]))
        # code-tscene-r2: an area band of 0.5-2 rejected normal push-ins (bus, door); 0.25-9 = 0.5-3x in side length
        if not 0.25 <= a1 / max(a0, 1e-6) <= 9.0 or np.linalg.norm(q.mean(0) - self.quad.mean(0)) > 160 + 2 * h0:
            return None, -n
        return q, n


def _region(shape, quad: np.ndarray, grow_u: float, grow_v: float) -> np.ndarray:
    tl, tr, br, bl = quad
    u, v = tr - tl, bl - tl
    h = np.linalg.norm(v)
    uu, vv = u / (np.linalg.norm(u) + 1e-6), v / (h + 1e-6)
    pts = np.stack([tl - grow_u * h * uu - grow_v * h * vv, tr + grow_u * h * uu - grow_v * h * vv,
                    br + grow_u * h * uu + grow_v * h * vv, bl - grow_u * h * uu + grow_v * h * vv])
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.round(pts).astype(np.int32), 1)
    return m.astype(bool)


def _fit(img: np.ndarray, region: np.ndarray, iters: int = 4):
    """A robust quadratic fit (x, y) of each colour over `region`; returns (fit image over region, residual magnitude,
    MAD)."""
    ys, xs = np.nonzero(region)
    a = np.stack([np.ones_like(xs), xs, ys, xs * xs, ys * ys, xs * ys], 1).astype(np.float64)
    col = img[region].astype(np.float64)
    use = np.ones(len(xs), bool)
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(a[use], col[use], rcond=None)
        res = np.linalg.norm(col - a @ coef, axis=1)
        mad = np.median(res[use]) + 1e-3
        use = res < 3 * 1.4826 * mad
    return a @ coef, res, mad, (ys, xs), col


def auto_ink(frame: np.ndarray, quad: np.ndarray) -> str:
    """"dark" or "light": the text's polarity. Inside the tight quad (OCR's own box, no padding) the pixels are split
    into two colour clusters; the strokes are the smaller one."""
    region = _region(frame.shape, quad, 0.0, 0.0)
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(lab[region], 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    small = int(np.argmin(np.bincount(labels.ravel(), minlength=2)))
    return "dark" if centers[small, 0] < centers[1 - small, 0] else "light"


def erase(frame: np.ndarray, quad: np.ndarray, seed: int = 0, grow_u: float = 0.12,
          ink_hint: str | None = None, grow_v: float = 0.18, inside: bool = False) -> tuple[np.ndarray, dict]:
    """The old text's strokes replaced by the surface: inside the padded quad the pixels are split into two colour
    clusters (k-means); the surface is the cluster that fills the region's rim, the ink the other one (bold text can
    cover half the region, so a robust fit of one surface colour fails, as on the first check). The ink pixels,
    dilated, are inpainted from the surface around them (OpenCV Telea). Returns (frame, info: ink colour, stroke share)."""
    region = _region(frame.shape, quad, grow_u, grow_v)  # grow_v: LIVE.md 20 Nov (field writer)
    rim = region & ~cv2.erode(region.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
    px = lab[region]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(px, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    lab_img = np.full(frame.shape[:2], -1, np.int32)
    lab_img[region] = labels
    rim_votes = np.bincount(lab_img[rim][lab_img[rim] >= 0], minlength=2)
    surf = int(np.argmax(rim_votes))
    if ink_hint == "auto":  # LIVE.md 17 Nov (v3): inside the tight (unpadded) quad the strokes are the minority
        ink_hint = auto_ink(frame, quad)
    if ink_hint in ("dark", "light"):  # code-tscene-r1 blocker 1: the rim can fall on a dark bezel or sky
        darker = int(np.argmin(centers[:, 0]))  # L of Lab
        surf = 1 - darker if ink_hint == "dark" else darker
    ink_k = 1 - surf
    stroke = (lab_img == ink_k)
    # code-tscene-r1 (calculator): ink-coloured pieces touching the region's rim are not text (a bezel, the night sky
    # around a sign): the strokes of the text sit inside the padded region
    n_cc, cc, st_cc, _ = cv2.connectedComponentsWithStats(stroke.astype(np.uint8), connectivity=8)
    touch = np.unique(cc[rim & stroke])
    big = [t for t in touch if t > 0 and st_cc[t, cv2.CC_STAT_AREA] > 0.15 * region.sum()]  # a glyph is smaller
    if inside:  # PlaneWriter: the region lies inside the surface's own edges, so nothing in it is a bezel (a glowing
        big = []  # word is one big piece touching the rim)
    stroke &= ~np.isin(cc, big)
    share = float(stroke.sum() / max(1, region.sum()))
    h = float(np.linalg.norm(quad[3] - quad[0]))
    k = max(3, int(round(h * 0.06)) | 1)
    grown = cv2.dilate(stroke.astype(np.uint8), np.ones((k, k), np.uint8)).astype(np.uint8) * region.astype(np.uint8)
    out = cv2.inpaint(np.ascontiguousarray(frame), grown, max(3, int(h * 0.15)), cv2.INPAINT_TELEA)
    ink = np.median(frame[stroke], 0) if stroke.sum() > 10 else None
    # the core of the strokes (far from the surface colour), not their anti-aliased edges
    if stroke.sum() > 10:
        d = np.linalg.norm(lab[stroke] - centers[surf], axis=1)
        ink = np.median(frame[stroke][d >= np.percentile(d, 50)], 0)
    return out, {"ink": ink, "stroke_share": round(share, 3)}


def _render(text: str, height_px: float) -> np.ndarray:
    """The text's ink as an alpha mask (float 0-1), cropped to its ink box, cap height ~ height_px."""
    size = 200
    f = ImageFont.truetype(PASTE_FONT, size)
    l, t, r, b = f.getbbox(text)
    im = Image.new("L", (r - l + 20, b - t + 20), 0)
    ImageDraw.Draw(im).text((10 - l, 10 - t), text, font=f, fill=255)
    a = np.asarray(im, np.float32) / 255
    ys, xs = np.nonzero(a > 0.05)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sc = height_px / a.shape[0]
    return cv2.resize(a, (max(1, int(a.shape[1] * sc)), max(1, int(height_px))), interpolation=cv2.INTER_AREA)


def surface_span(frame: np.ndarray, quad: np.ndarray, reach: float = 4.0, tol: float = 22.0,
                 run: int = 3) -> tuple[float, float, float, float]:
    """The surface's extent around the (erased) line, in the quad's own axes: (left, right) along the text from tl, in
    pixels (left may be negative), and (top, bottom) across it. From the middle of the line, rows at 30/50/70% of its
    height (and columns at 30/50/70% of its width) are walked outwards until `run` pixels in a row differ from the
    surface colour (Lab distance > tol), at most `reach` line heights; the narrowest row bounds the text."""
    tl, tr, br, bl = quad.astype(np.float64)
    u, v = tr - tl, bl - tl
    w, h = np.linalg.norm(u), np.linalg.norm(v)
    uu, vv = u / (w + 1e-6), v / (h + 1e-6)
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
    H, W = frame.shape[:2]
    c = tl + 0.5 * u + 0.5 * v
    core = _region(frame.shape, quad, -0.1, -0.2) if w > 0.4 * h else _region(frame.shape, quad, 0.0, 0.0)
    surf = np.median(lab[core], 0) if core.any() else lab[int(np.clip(c[1], 0, H - 1)), int(np.clip(c[0], 0, W - 1))]

    def walk(p0: np.ndarray, d: np.ndarray, limit: float) -> float:
        bad = 0
        for s in range(1, int(limit)):
            x, y = p0 + s * d
            if not (0 <= x < W and 0 <= y < H):
                return s - bad
            bad = bad + 1 if np.linalg.norm(lab[int(y), int(x)] - surf) > tol else 0
            if bad >= run:
                return s - bad
        return limit

    lim_u, lim_v = w / 2 + reach * h, h / 2 + reach * h
    lefts, rights, tops, bots = [], [], [], []
    for f in (0.3, 0.5, 0.7):
        p = tl + 0.5 * u + f * v
        lefts.append(w / 2 - walk(p, -uu, lim_u))
        rights.append(w / 2 + walk(p, uu, lim_u))
        q = tl + f * u + 0.5 * v
        tops.append(h / 2 - walk(q, -vv, lim_v))
        bots.append(h / 2 + walk(q, vv, lim_v))
    return max(lefts), min(rights), max(tops), min(bots)


def paste(frame: np.ndarray, quad: np.ndarray, text: str, ink, align: str = "center", fill_h: float = 0.72,
          margin: float = 0.25, fit: bool = True) -> np.ndarray:
    """margin: the clear space kept between the text and the surface's edge, in line heights (at least 6% of the
    surface's width). fit=False keeps the old rule (never wider than the old line + 10%)."""
    tl, tr, br, bl = quad
    u, v = tr - tl, bl - tl
    w_old, h = np.linalg.norm(u), np.linalg.norm(v)
    uu, vv = u / (w_old + 1e-6), v / (h + 1e-6)
    a = _render(text, h * fill_h)
    if fit:
        left, right, top, bot = surface_span(frame, quad)
        m = max(margin * h, 0.06 * (right - left))
        lo, hi = left + m, right - m
        fit = hi - lo >= 0.5 * w_old and bot - top >= 0.8 * h  # else a stray edge through the line: the old rule
    if fit:
        max_wpx, max_hpx = min(hi - lo, 1.6 * w_old), (bot - top) - 2 * 0.15 * h  # code-paste-fit-r1 #3: a missed edge
    else:
        lo, hi = -0.05 * w_old, 1.05 * w_old
        max_wpx, max_hpx = 1.1 * w_old, h
    sc = min(1.0, max_wpx / a.shape[1], max_hpx / a.shape[0])
    if sc < 1.0:
        a = cv2.resize(a, (max(1, int(a.shape[1] * sc)), max(1, int(a.shape[0] * sc))), interpolation=cv2.INTER_AREA)
    ah, aw = a.shape
    off_v = (h - ah) / 2
    off_u = {"left": 0.0, "right": w_old - aw}.get(align, (w_old - aw) / 2)
    off_u = float(np.clip(off_u, lo, max(lo, hi - aw)))  # inside the surface
    origin = tl + off_u * uu + off_v * vv
    m2 = np.float32([[uu[0], vv[0], origin[0]], [uu[1], vv[1], origin[1]]])  # glyph (x, y) -> frame
    H, W = frame.shape[:2]
    alpha = cv2.warpAffine(a, m2, (W, H), flags=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.5)[..., None]
    col = np.asarray(ink if ink is not None else (20, 20, 20), np.float32)
    return np.clip(frame * (1 - alpha) + col * alpha, 0, 255).astype(np.uint8)


def write(frame: np.ndarray, quad: np.ndarray, text: str, align: str = "center", seed: int = 0,
          grow_u: float = 0.12, ink_hint: str | None = None) -> tuple[np.ndarray, dict]:
    """grow_u: how far past the quad's ends the old strokes are erased, in line heights (a field cut from a line by
    width shares: 0.25, its edges are estimates)."""
    erased, info = erase(frame, quad, seed, grow_u, ink_hint)
    return paste(erased, quad, text, info["ink"], align), info


# ---- the plane writer (LIVE.md 20 Nov; Elad on the field writer's stills: "the angling is wrong. It looks like the
# letters are not on the plane of where they are printed") ----
# The start image is measured once: the surface's plane (a homography, from the surface's own border: sign frame, tag
# sides, display bezel), the old text's place and size on it, the font nearer its width, and a leading symbol drawn
# smaller than the letters (Krea's raised "$"). In a later frame the plane is the start's moved by the tracker's
# homography, so it does not jump from frame to frame. The new text is drawn in the rectified plane and warped back.

PLANE_FONTS = {"narrow": PASTE_FONT, "wide": FONT}


def glyphs(text: str, font: str) -> np.ndarray:
    """The text's ink as an alpha mask (float 0-1), cropped to its ink box, at a 200 px font size."""
    f = ImageFont.truetype(font, 200)
    l, t, r, b = f.getbbox(text)
    im = Image.new("L", (r - l + 20, b - t + 20), 0)
    ImageDraw.Draw(im).text((10 - l, 10 - t), text, font=f, fill=255)
    a = np.asarray(im, np.float32) / 255
    ys, xs = np.nonzero(a > 0.05)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def cap_metrics(text: str, font: str) -> tuple[float, float]:
    """At a 200 px font size: the cap height (of "H"; digits and capitals share it) and how far the text's ink top
    sits below the capitals' top (negative when a symbol, as "$", reaches above them)."""
    f = ImageFont.truetype(font, 200)
    h = f.getbbox("H")
    return float(h[3] - h[1]), float(f.getbbox(text)[1] - h[1])


def _corners(p: np.ndarray) -> np.ndarray:
    """4 points as tl, tr, br, bl (by x+y and y-x), keeping their shape (not a rectangle, unlike _order)."""
    p = p.reshape(-1, 2).astype(np.float32)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.float32([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]])


def ink_mask(frame: np.ndarray, quad: np.ndarray, ink, grow_u: float = 0.05, grow_v: float = 0.1,
             tol: float = 30.0) -> np.ndarray:
    """Pixels of the ink's colour (Lab distance < tol) in the padded quad, pieces under 2% of the ink left out."""
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
    ink_lab = cv2.cvtColor(np.uint8([[np.clip(ink, 0, 255)]]), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    d = ((np.linalg.norm(lab - ink_lab, axis=2) < tol) & _region(frame.shape, quad, grow_u, grow_v)).astype(np.uint8)
    n, cc, st, _ = cv2.connectedComponentsWithStats(d, connectivity=8)
    return np.isin(cc, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 0.02 * d.sum()])


def _line(a, b) -> np.ndarray:
    return np.cross(np.r_[a, 1.0], np.r_[b, 1.0])


def _meet(l1, l2) -> np.ndarray:
    p = np.cross(l1, l2)
    return p[:2] / p[2]


def surface_edges(frame: np.ndarray, q: np.ndarray, erased: np.ndarray, lines=None, max_ang: float = 20.0,
                  reach_v: float = 3.0, reach_u: float = 1.5, tol: float = 30.0) -> dict:
    """The line segments (OpenCV LSD) that bound the text's surface, nearest on each side: "top"/"bottom" within
    max_ang deg of the text's direction, above / below it; "left"/"right" within max_ang deg of its normal, beside it.
    Left out: segments under 0.5x the text's height (along the text: also under 0.4x its width and 1.5x its height), on
    another OCR text's letters, and segments that are not a border of the surface: just inside (towards the text) must
    be nearer the surface's colour (the old text's box after the erase) than just outside, at 2 of 3 points (a relative
    test: a glowing display is brighter by its text than at its frame). Returns {side: (a, b)}."""
    tl, tr, br, bl = q.astype(np.float64)
    u, v = (tr - tl + br - bl) / 2, (bl - tl + br - tr) / 2
    w, h = np.linalg.norm(u), np.linalg.norm(v)
    uu, vv, c = u / w, v / h, q.mean(0)
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
    surf = np.median(cv2.cvtColor(erased, cv2.COLOR_RGB2LAB).astype(np.float32)[_region(frame.shape, q, -0.05, -0.1)], 0)
    lines = ocr_lines(frame) if lines is None else lines
    boxes = [(cv2.convexHull(_corners(np.asarray(b))), bool(np.linalg.norm(np.asarray(b).mean(0) - c) < 0.25 * h))
             for b, _, _ in lines]
    segs = cv2.createLineSegmentDetector().detect(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))[0]
    H, W = frame.shape[:2]
    best: dict = {}
    for s in (segs.reshape(-1, 4) if segs is not None else []):  # OpenCV 4: (N, 1, 4); 5: (N, 4)
        a, b = s[:2].astype(np.float64), s[2:].astype(np.float64)
        d = b - a
        L = np.linalg.norm(d)
        if L < 0.5 * h:
            continue
        ang = np.degrees(np.arccos(min(1.0, abs(d @ uu) / L)))
        horiz, vert = ang <= max_ang, ang >= 90 - max_ang
        if not (horiz or vert) or (horiz and L < 0.4 * w and L < 1.5 * h):
            continue
        m = (a + b) / 2
        # on another text's letters (its OCR box, 3 px grown), or well inside this text's own box (a border may touch it)
        if any(cv2.pointPolygonTest(bx, (float(m[0]), float(m[1])), True) > (3 if own else -3) for bx, own in boxes):
            continue
        du, dv = (m - c) @ uu, (m - c) @ vv
        if horiz:
            side, dist = ("top" if dv < 0 else "bottom"), abs(dv) - h / 2
            if not (-0.1 * h < dist < reach_v * h) or abs(du) > w / 2 + reach_u * w:
                continue
            inward = vv if side == "top" else -vv
        else:
            side, dist = ("left" if du < 0 else "right"), abs(du) - w / 2
            if not (-0.1 * h < dist < reach_u * w + h) or abs(dv) > h:  # beside the text, not below or above it
                continue
            inward = uu if side == "left" else -uu
        okc = 0
        for t in (0.25, 0.5, 0.75):
            pi, po = a + t * d + 4 * inward, a + t * d - 4 * inward
            if all(0 <= p[0] < W and 0 <= p[1] < H for p in (pi, po)):
                di = np.linalg.norm(lab[int(pi[1]), int(pi[0])] - surf)
                do = np.linalg.norm(lab[int(po[1]), int(po[0])] - surf)
                okc += int(di < 2 * tol and di < do - 10)
        if okc < 2:
            continue
        score = dist - 0.05 * L  # nearest first, longer breaks ties
        if side not in best or score < best[side][0]:
            best[side] = (score, a, b)
    return {k: (x[1], x[2]) for k, x in best.items()}


def surface_quad(frame: np.ndarray, q: np.ndarray, erased: np.ndarray, lines=None) -> tuple[np.ndarray, list]:
    """The surface's quad (tl, tr, br, bl) from its edges: two opposite edges give that direction's vanishing point;
    one edge gives its direction (the missing side parallel to it, 0.75x the text's height beyond the text); none, the
    text quad's own direction. Returns (quad, the sides found)."""
    tl, tr, br, bl = q.astype(np.float64)
    e = surface_edges(frame, q, erased, lines)
    u, v = (tr - tl + br - bl) / 2, (bl - tl + br - tr) / 2
    uu, vv, h, w, c = u / np.linalg.norm(u), v / np.linalg.norm(v), np.linalg.norm(v), np.linalg.norm(u), q.mean(0)
    L = {k: _line(*e[k]) for k in e}
    for s1, s2, dflt, off in (("top", "bottom", u, vv * (h / 2 + 0.25 * h)), ("left", "right", v, uu * (w / 2 + 0.25 * h))):
        if s1 in L and s2 in L:
            continue
        found = [k for k in (s1, s2) if k in L]
        dirn = (e[found[0]][1] - e[found[0]][0]) if found else dflt
        for s, sign in ((s1, -1), (s2, 1)):
            if s not in L:
                p = c + sign * off
                L[s] = _line(p, p + dirn)
    quad = np.float32([_meet(L["top"], L["left"]), _meet(L["top"], L["right"]), _meet(L["bottom"], L["right"]),
                       _meet(L["bottom"], L["left"])])
    return quad, sorted(e)


def _plane(quad: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """The homography from a W x H rectangle (the quad's mean side lengths) to the quad."""
    tl, tr, br, bl = quad
    W = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    H = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    rect = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
    return cv2.getPerspectiveTransform(rect, quad.astype(np.float32)).astype(np.float64), (int(W), int(H))


class PlaneWriter:
    """The field's text written on its surface's plane. Measured once on the start image (field_quad: the field's own
    OCR quad; track_quad: the tracker's start quad, the field's or its line's); write() then takes the tracked quad of a
    later frame and the new field text."""

    def __init__(self, start: np.ndarray, field_quad: np.ndarray, old: str, align: str = "center",
                 track_quad: np.ndarray | None = None, ink_hint: str | None = None, lines=None,
                 margin: float = 0.08) -> None:
        q = np.asarray(field_quad, np.float32)
        self.track0 = np.asarray(track_quad if track_quad is not None else q, np.float32)
        self.align, self.ink_hint = align, ink_hint
        erased, info = erase(start, q, 0, 0.05, ink_hint, grow_v=0.03)
        if info["ink"] is None:
            raise ValueError("the old text's ink was not found on the start image")
        self.ink0 = np.asarray(info["ink"], np.float32)
        self.surface, self.edges = surface_quad(start, q, erased, lines)
        self.Hm, (W, H) = _plane(self.surface)
        r = (cv2.warpPerspective(ink_mask(start, q, info["ink"]).astype(np.uint8) * 255, np.linalg.inv(self.Hm), (W, H),
                                 flags=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        n, _cc, st, _ = cv2.connectedComponentsWithStats(r, connectivity=8)
        comps = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 0.02 * r.sum()]
        if not comps:
            raise ValueError("the old text's ink was not found in the surface's plane")
        left = {i: st[i, cv2.CC_STAT_LEFT] for i in comps}
        right = {i: st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] for i in comps}
        top = {i: st[i, cv2.CC_STAT_TOP] for i in comps}
        hgt = {i: st[i, cv2.CC_STAT_HEIGHT] for i in comps}
        med = float(np.median(list(hgt.values())))
        core = [i for i in comps if 0.8 * med <= hgt[i] <= 1.25 * med] or comps
        v0 = float(np.median([top[i] for i in core]))
        v1 = float(np.median([top[i] + hgt[i] for i in core]))
        self.hpx = max(4.0, v1 - v0)
        self.v0 = v0
        u0, u1 = float(min(left.values())), float(max(right.values()))
        self.box = (u0, u1, float(min(top.values())), float(max(top[i] + hgt[i] for i in comps)))  # all the old ink
        # a leading symbol drawn smaller than the letters (Krea's raised "$"): its size, height and gap kept
        self.lead = None
        body0, ub0 = old, u0
        first = min(comps, key=lambda i: left[i])
        if old[:1] and not old[0].isalnum() and len(comps) > 1 and hgt[first] < 0.85 * med:
            nxt = min((i for i in comps if i != first), key=lambda i: left[i])
            self.lead = {"ch": old[0], "h": hgt[first] / self.hpx, "top": (top[first] - v0) / self.hpx,
                         "gap": max(0.0, left[nxt] - right[first]) / self.hpx}
            body0, ub0 = old[1:], float(left[nxt])
        best = None
        for name, path in PLANE_FONTS.items():  # the font nearer the old text's width, then a stretch towards it
            a = glyphs(body0, path)
            sx = (u1 - ub0) / (a.shape[1] * self.hpx / cap_metrics(body0, path)[0])  # sized by the capitals' height
            if best is None or abs(np.log(sx)) < abs(np.log(best[2])):
                best = (name, path, sx)
        self.font_name, self.font, self.sx = best[0], best[1], float(np.clip(best[2], 0.8, 1.25))
        # room: the surface's width in the plane less a margin (a side not found: 0.4x the old text beyond it); never
        # less than 1.1x the old text
        mg, ow = margin * W, u1 - u0
        self.lo = mg if "left" in self.edges else u0 - 0.4 * ow
        self.hi = W - mg if "right" in self.edges else u1 + 0.4 * ow
        self.u0, self.u1, self.size = u0, u1, (W, H)
        self.widest = (u0, u1)  # the widest text written so far, in the plane: what an erase covers
        self.pad = int(max(W, H))  # the canvas' margin around the plane (text may run past the surface quad)
        self._cache: dict = {}
        # the text's polarity on its surface (erase_strays): its ink darker or lighter than the surface around it
        lab = lambda c: cv2.cvtColor(np.uint8([[c]]), cv2.COLOR_RGB2LAB)[0, 0, 0]
        self.polarity = "dark" if lab(np.clip(self.ink0, 0, 255)) < np.median(
            cv2.cvtColor(erased, cv2.COLOR_RGB2LAB)[_region(start.shape, q, 0.05, 0.03)][:, 0]) else "light"

    def info(self) -> dict:
        return {"edges": self.edges, "font": self.font_name, "sx": round(self.sx, 2), "h": round(self.hpx, 1),
                "lead": self.lead, "surface": np.round(self.surface).tolist()}

    def layout(self, text: str) -> tuple[np.ndarray, tuple[float, float]]:
        """The text's alpha on the plane's canvas (margin self.pad) and its span (x0, x1) in plane coordinates."""
        if text in self._cache:
            return self._cache[text]
        W, H = self.size
        hp, sx, lead = self.hpx, self.sx, self.lead
        lead = lead if lead and text[:1] == lead["ch"] and len(text) > 1 else None
        btext = text[1:] if lead else text
        body = glyphs(btext, self.font)
        cap, top_off = cap_metrics(btext, self.font)  # the capitals and digits at the old text's height; a taller
        k = hp / cap                                   # symbol ("$" on the gas pump) reaches past them, as Krea's
        wb, hb, yb = body.shape[1] * k * sx, body.shape[0] * k, top_off * k
        if lead:
            g = glyphs(lead["ch"], self.font)
            hl = lead["h"] * hp
            wl, gap = g.shape[1] * hl / g.shape[0] * sx, lead["gap"] * hp
        else:
            wl = gap = 0.0
        total = wl + gap + wb
        cu = (self.u0 + self.u1) / 2
        room = 2 * min(cu - self.lo, self.hi - cu) if self.align == "center" else self.hi - self.u0
        sc = min(1.0, max(room, 1.1 * (self.u1 - self.u0)) / total)
        x0 = self.u0 if self.align == "left" else cu - total * sc / 2
        yc = self.v0 + hp / 2
        canvas = np.zeros((H + 2 * self.pad, W + 2 * self.pad), np.float32)

        def put(a, x, y, w, h):
            w, h = max(1, int(round(w))), max(1, int(round(h)))
            x, y = int(round(x)) + self.pad, int(round(y)) + self.pad
            g = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
            ys, xs = slice(max(0, y), min(canvas.shape[0], y + h)), slice(max(0, x), min(canvas.shape[1], x + w))
            canvas[ys, xs] = np.maximum(canvas[ys, xs], g[ys.start - y:ys.stop - y, xs.start - x:xs.stop - x])

        if lead:
            put(g, x0, yc - hp * sc / 2 + lead["top"] * hp * sc, wl * sc, hl * sc)
        put(body, x0 + (wl + gap) * sc, yc - hp * sc / 2 + yb * sc, wb * sc, hb * sc)
        self._cache[text] = (canvas, (x0, x0 + total * sc))
        return self._cache[text]

    def write(self, frame: np.ndarray, track_quad: np.ndarray, text: str, seed: int = 0) -> tuple[np.ndarray, dict]:
        """The old text (the widest written so far) erased and `text` drawn, at the plane the tracked quad gives."""
        hs = cv2.getPerspectiveTransform(self.track0, np.asarray(track_quad, np.float32)).astype(np.float64)
        m = hs @ self.Hm  # plane -> this frame
        canvas, (x0, x1) = self.layout(text)
        # the erase spans the surface inside its margins (Krea may have drawn a wider word than any written); a field
        # after a label (align "left") from the field's left edge only, so the label stays
        lo = min(self.widest[0], x0, self.lo if self.align == "center" else self.u0)
        hi = max(self.widest[1], x1, self.hi)
        _, _, b0, b1 = self.box
        # padded past a glow (a box tight on the letters lets erase take the glow for a bezel and keep it), and kept
        # inside the surface (past its edge, erase smears the background in)
        ex = 0.2 * self.hpx
        W, H = self.size
        e_in = 4.0 + 0.05 * self.hpx  # erase grows the box a little more (grow 0.02): kept this far inside the edges
        lo, hi = max(e_in, lo - ex) + ex, min(W - e_in, hi + ex) - ex
        b0, b1 = max(e_in, b0 - ex) + ex, min(H - e_in, b1 + ex) - ex
        box = np.float32([[lo - ex, b0 - ex], [hi + ex, b0 - ex], [hi + ex, b1 + ex], [lo - ex, b1 + ex]])
        erased, info = erase(frame, cv2.perspectiveTransform(box[None], m)[0], seed, 0.02, self.ink_hint, grow_v=0.02,
                             inside=True)
        self.widest = (lo, hi)
        ink = self.ink0 if info["ink"] is None or info["stroke_share"] < 0.02 else np.asarray(info["ink"], np.float32)
        t = np.float64([[1, 0, self.pad], [0, 1, self.pad], [0, 0, 1]])
        al = cv2.warpPerspective(canvas, m @ np.linalg.inv(t), frame.shape[1::-1], flags=cv2.INTER_LINEAR)
        al = cv2.GaussianBlur(al, (0, 0), 0.5)[..., None]
        out = np.clip(erased * (1 - al) + ink * al, 0, 255).astype(np.uint8)
        return out, {"ink": ink.round().tolist(), "stroke_share": info["stroke_share"], "span": (round(x0), round(x1))}

    def surface_at(self, track_quad: np.ndarray) -> np.ndarray | None:
        """The surface's quad in a frame, from the tracked quad; None when the tracker's homography folds or flips it
        (a convex quad, its area 0.3-3x the start's, as TextTracker.locate's own band)."""
        hs = cv2.getPerspectiveTransform(self.track0, np.asarray(track_quad, np.float32)).astype(np.float64)
        s = cv2.perspectiveTransform(self.surface[None].astype(np.float64), hs)[0].astype(np.float32)
        a0 = cv2.contourArea(np.asarray(self.surface, np.float32))
        if not cv2.isContourConvex(s.round().astype(np.int32)) or not 0.3 <= cv2.contourArea(s) / max(a0, 1e-6) <= 3.0:
            return None
        return s

    def band_at(self, track_quad: np.ndarray, reach: float = 2.5) -> np.ndarray:
        """The text's row in a frame, past the surface on both sides by `reach` surface widths: rows 0.4x the cap height
        above to 1.4x below its top, in the surface's plane (where Krea leaves copies of the text as it moves)."""
        hs = cv2.getPerspectiveTransform(self.track0, np.asarray(track_quad, np.float32)).astype(np.float64)
        W, _ = self.size
        b = np.float32([[-reach * W, self.v0 - 0.4 * self.hpx], [(1 + reach) * W, self.v0 - 0.4 * self.hpx],
                        [(1 + reach) * W, self.v0 + 1.4 * self.hpx], [-reach * W, self.v0 + 1.4 * self.hpx]])
        return cv2.perspectiveTransform(b[None].astype(np.float64), hs @ self.Hm)[0].astype(np.float32)


def stray_boxes(lines, surface: np.ndarray, band: np.ndarray, chars: set, min_h: float, shape: tuple,
                off_share: float = 0.25) -> list[np.ndarray]:
    """LIVE.md 21 Nov (Elad: "The number on the ticket price moved out of the plane"; fix 2): OCR lines that are (in
    part) copies of the written text off its surface: in the text's row (band), at least min_h tall, with a character
    of the texts written so far (chars). OCR often reads a copy and the real text as one line; the part on the surface
    is left to the caller's mask."""
    out = []
    for l, t, _s in lines:
        l = np.asarray(l, np.float32)
        if not any(c in chars for c in norm(t)):
            continue
        if min(cv2.minAreaRect(l)[1]) < min_h:
            continue
        c = tuple(float(v) for v in l.mean(0))
        if cv2.pointPolygonTest(band, c, False) < 0:
            continue
        if cv2.pointPolygonTest(surface, c, False) >= 0:
            # a line centred on the surface is a copy only when off_share of it runs past it (the merged "$4$649"; the
            # real text's box often pokes a few px past a narrow tag): kept, the caller's mask leaves the surface alone
            lm, sm = np.zeros(shape[:2], np.uint8), np.zeros(shape[:2], np.uint8)
            cv2.fillConvexPoly(lm, np.round(cv2.convexHull(l)).astype(np.int32), 1)
            cv2.fillConvexPoly(sm, np.round(surface).astype(np.int32), 1)
            if (lm & (1 - sm)).sum() < off_share * max(1, lm.sum()):
                continue
        out.append(l)
    return out


def erase_strays(frame: np.ndarray, surface: np.ndarray, boxes: list[np.ndarray], polarity: str = "dark",
                 keep: int = 4, delta: float = 20.0) -> tuple[np.ndarray, int]:
    """The strokes inside `boxes` (grown 15%) but off the surface (its quad grown `keep` px, never touched) inpainted from
    the background around them (OpenCV Telea): strokes are pixels darker ("dark" text) or lighter than the region's
    median by `delta` in Lab L. Faint copies OCR does not read stay (a blob search without OCR took denim shadows on
    frames without copies: LIVE.md 21 Nov). Returns (frame, stroke pixels changed)."""
    H, W = frame.shape[:2]
    region = np.zeros((H, W), np.uint8)
    for b in boxes:
        c = b.mean(0)
        cv2.fillConvexPoly(region, np.round(c + 1.15 * (b - c)).astype(np.int32), 1)
    surf = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(surf, np.round(surface).astype(np.int32), 1)
    off = ~cv2.dilate(surf, np.ones((2 * keep + 1, 2 * keep + 1), np.uint8)).astype(bool)
    region = region.astype(bool) & off
    if not region.any():
        return frame, 0
    L = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32) * (100 / 255)
    med = float(np.median(L[region]))
    stroke = region & ((L < med - delta) if polarity == "dark" else (L > med + delta))
    if not stroke.any():
        return frame, 0
    stroke = cv2.dilate(stroke.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & off
    n = int(stroke.sum())
    if not n:
        return frame, 0
    return cv2.inpaint(frame, stroke.astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA), n
