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
paste: the new text in a plain bold font, the ink colour measured from the old strokes, the size from the old line's
height, aligned to the old line (left, right or centre), slanted with the quad; soft alpha.
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
FONT = str(Path(__file__).resolve().parents[1] / "assets/fonts/DejaVuSans-Bold.ttf")


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


def erase(frame: np.ndarray, quad: np.ndarray, seed: int = 0, grow_u: float = 0.12,
          ink_hint: str | None = None) -> tuple[np.ndarray, dict]:
    """The old text's strokes replaced by the surface: inside the padded quad the pixels are split into two colour
    clusters (k-means); the surface is the cluster that fills the region's rim, the ink the other one (bold text can
    cover half the region, so a robust fit of one surface colour fails, as on the first check). The ink pixels,
    dilated, are inpainted from the surface around them (OpenCV Telea). Returns (frame, info: ink colour, stroke share)."""
    region = _region(frame.shape, quad, grow_u, 0.18)
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
    f = ImageFont.truetype(FONT, size)
    l, t, r, b = f.getbbox(text)
    im = Image.new("L", (r - l + 20, b - t + 20), 0)
    ImageDraw.Draw(im).text((10 - l, 10 - t), text, font=f, fill=255)
    a = np.asarray(im, np.float32) / 255
    ys, xs = np.nonzero(a > 0.05)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sc = height_px / a.shape[0]
    return cv2.resize(a, (max(1, int(a.shape[1] * sc)), max(1, int(height_px))), interpolation=cv2.INTER_AREA)


def paste(frame: np.ndarray, quad: np.ndarray, text: str, ink, align: str = "center", fill_h: float = 0.72,
          max_w: float = 1.1) -> np.ndarray:
    tl, tr, br, bl = quad
    u, v = tr - tl, bl - tl
    w_old, h = np.linalg.norm(u), np.linalg.norm(v)
    uu, vv = u / (w_old + 1e-6), v / (h + 1e-6)
    a = _render(text, h * fill_h)
    if a.shape[1] > max_w * w_old:  # never wider than the old line (+10%): it stays on its surface
        sc = max_w * w_old / a.shape[1]
        a = cv2.resize(a, (max(1, int(a.shape[1] * sc)), max(1, int(a.shape[0] * sc))), interpolation=cv2.INTER_AREA)
    ah, aw = a.shape
    off_v = (h - ah) / 2
    off_u = {"left": 0.0, "right": w_old - aw}.get(align, (w_old - aw) / 2)
    origin = tl + off_u * uu + off_v * vv
    m = np.float32([[uu[0], vv[0], origin[0]], [uu[1], vv[1], origin[1]]])  # glyph (x, y) -> frame
    H, W = frame.shape[:2]
    alpha = cv2.warpAffine(a, m, (W, H), flags=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.5)[..., None]
    col = np.asarray(ink if ink is not None else (20, 20, 20), np.float32)
    return np.clip(frame * (1 - alpha) + col * alpha, 0, 255).astype(np.uint8)


def write(frame: np.ndarray, quad: np.ndarray, text: str, align: str = "center", seed: int = 0,
          grow_u: float = 0.12, ink_hint: str | None = None) -> tuple[np.ndarray, dict]:
    """grow_u: how far past the quad's ends the old strokes are erased, in line heights (a field cut from a line by
    width shares: 0.25, its edges are estimates)."""
    erased, info = erase(frame, quad, seed, grow_u, ink_hint)
    return paste(erased, quad, text, info["ink"], align), info
