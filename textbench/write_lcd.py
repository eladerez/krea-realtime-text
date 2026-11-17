"""Write a number on a 7-segment LCD the way the LCD draws it (LIVE.md Changes, 15 Nov 2026, after session
textbench-run-20261115-1909: klein drew large generic digits over the display, and Krea turned 7-segment digits into
8s). An LCD's digits are a font, so the number is drawn in DSEG7 Classic (OFL, keshikan/DSEG v0.46), not in our sans
font on a grey patch.

find_lcd: the LCD on a clean frame (the start image), by a flood fill over its own colour (a brightness threshold
found the white desk or the light frame around the LCD instead; a colour threshold failed on Krea's frames).
LcdTracker: that LCD followed in later frames by template matching (on Krea's frames a fill stops partway along
strong gradients and misses the digits at the far end, 15 Nov session crops).
analyse: the old digits are the pixels darker than the LCD's robust brightness fit inside it. The panel's angle comes from its
principal axis; the digits' height, right edge and ink colour come from the holes (else defaults from the
panel's size and its darkest pixels).
write_lcd: the old digits are inpainted from the LCD's own neighbouring pixels (outside the LCD is first replaced by
its fitted colour, so the bezel cannot bleed in), plus a fine grain, so there is no patch; the number is drawn in DSEG7, right-
aligned at the old digits' right edge with every digit in its LCD place, slanted (`slant`: horizontal shear, as
Krea's digits lean), rotated with the panel, as a soft alpha mask in the measured ink colour. Optional `ghost`: the
unlit segments at a few percent of the contrast.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "assets/fonts/DSEG7Classic-Bold.ttf"


def _fill(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, cnts, -1, 1, -1)
    return out


def _quad(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(xs), xs, ys, xs * xs, ys * ys, xs * ys], 1).astype(np.float64)


def find_lcd(frame: np.ndarray, box: tuple[int, int, int, int], margin: float = 0.6,
             tol: int = 6, fixed: bool = False) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """The LCD on a clean frame (the start image): a flood fill over its own smooth colour from seeds of plain LCD
    background in the box's middle rows, its largest region's convex hull (an LCD is a convex quadrilateral, so
    digits at its edge stay inside). Returns the window (x0, y0, x1, y1) and the LCD mask in it. On Krea's later
    frames a fill stops partway along strong gradients, so the live loop finds the LCD once and tracks it
    (LcdTracker)."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    bw, bh = x1 - x0, y1 - y0
    wx0, wy0 = max(0, int(x0 - margin * bw)), max(0, int(y0 - margin * bh))
    wx1, wy1 = min(w, int(x1 + margin * bw)), min(h, int(y1 + margin * bh))
    win = np.ascontiguousarray(frame[wy0:wy1, wx0:wx1])
    gray = cv2.cvtColor(win, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(win, (7, 7), 0)
    by0, by1 = y0 - wy0 + bh // 4, y1 - wy0 - bh // 4
    band = gray[by0:by1, x0 - wx0:x1 - wx0].astype(np.float32)
    if band.size == 0:
        return None
    cand = np.argwhere(np.abs(band - np.percentile(band, 70)) < 4)
    best, best_area = None, 0
    for k in np.linspace(0, len(cand) - 1, min(5, len(cand))).astype(int):
        sy, sx = cand[k]
        m = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), np.uint8)
        cv2.floodFill(blur.copy(), m, (int(sx + x0 - wx0), int(sy + by0)), (0, 0, 0), (tol,) * 3, (tol,) * 3,
                      4 | cv2.FLOODFILL_MASK_ONLY | (cv2.FLOODFILL_FIXED_RANGE if fixed else 0) | (255 << 8))
        reg = m[1:-1, 1:-1]
        if best_area < reg.sum() / 255 < 0.6 * reg.size:
            best, best_area = reg, reg.sum() / 255
    if best is None:
        return None
    reg = cv2.morphologyEx((best > 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n_f, lab_f, st_f, _ = cv2.connectedComponentsWithStats(reg, 8)
    if n_f < 2:
        return None
    big = (lab_f == 1 + int(np.argmax(st_f[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    if fixed:  # LIVE.md 16 Nov (start image s0): digits filling the LCD's height split the fill; all its pieces count
        big = np.isin(lab_f, 1 + np.nonzero(st_f[1:, cv2.CC_STAT_AREA] >= 50)[0]).astype(np.uint8)
    filled = np.zeros_like(big)
    cv2.fillConvexPoly(filled, cv2.convexHull(cv2.findNonZero(big)), 1)
    return (wx0, wy0, wx1, wy1), cv2.erode(filled, np.ones((3, 3), np.uint8))


def analyse(frame: np.ndarray, window: tuple[int, int, int, int], filled: np.ndarray,
            old_digits: int | None = None, digits_box: tuple[int, int, int, int] | None = None,
            rim: float = 6.0, edge: float = 3.0) -> dict | None:
    """Given the LCD mask in a window of this frame, find the old digits and measure them (see the module text)."""
    wx0, wy0, wx1, wy1 = window
    win = np.ascontiguousarray(frame[wy0:wy1, wx0:wx1])
    gray = cv2.cvtColor(win, cv2.COLOR_RGB2GRAY)
    filled = filled.astype(np.uint8)
    lcd = filled.astype(bool)
    if lcd.sum() < 200:
        return None
    # the digits: pixels darker than the LCD's own brightness there, by a third of the LCD-to-ink range. The
    # brightness is a robust quadratic fit of the LCD's pixels (ink dropped as outliers), so a gradient is not ink
    fy_, fx_ = np.where(lcd)
    a = _quad(fx_, fy_)
    gv = gray[lcd].astype(np.float64)
    use = np.ones(len(gv), bool)
    for _ in range(4):  # robust fit: drop pixels well below the fit (ink), refit
        coef, *_ = np.linalg.lstsq(a[use], gv[use], rcond=None)
        res = gv - a @ coef
        mad = np.median(np.abs(res[use])) + 1e-3
        use = res > -3 * 1.4826 * mad
    expect = np.zeros(gray.shape, np.float64)
    expect[fy_, fx_] = a @ coef
    dark = np.where(lcd, expect - gray, 0)
    delta = max(18.0, 0.33 * float(np.percentile(dark[lcd], 99.5)))
    holes = (lcd & (dark > delta)).astype(np.uint8)
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # drop pixels within `rim` px of the LCD's border (its shadowed inner edge is not ink; the digits sit further in)
    inner = cv2.distanceTransform(filled, cv2.DIST_L2, 5) > rim
    holes = (holes.astype(bool) & inner).astype(np.uint8)
    if digits_box is not None:  # where OCR found the digits (display_check.read_display), widened a little
        dx0, dy0, dx1, dy1 = digits_box
        lim = np.zeros_like(holes)
        lim[max(0, dy0 - wy0 - 6):max(0, dy1 - wy0 + 6), max(0, dx0 - wx0 - 8):max(0, dx1 - wx0 + 8)] = 1
        holes = holes & lim
    # the LCD's axes: angle and height (in its own frame, not the tilted bounding box)
    ev, evec = np.linalg.eigh(np.cov(np.stack([fx_ - fx_.mean(), fy_ - fy_.mean()])))
    ux, uy = evec[:, 1]
    if ux < 0:
        ux, uy = -ux, -uy
    ang0 = float(np.degrees(np.arctan2(uy, ux)))
    lh = float(np.ptp(-(fx_ - fx_.mean()) * uy + (fy_ - fy_.mean()) * ux)) + 1
    # remove long straight runs along the LCD's long axis (its edge shading joins digits into one shape); a digit's
    # horizontal segment is about half a digit wide, far shorter than `line` of the LCD's height
    k = max(9, int(0.6 * lh))
    cxw, cyw = holes.shape[1] / 2, holes.shape[0] / 2
    to = cv2.getRotationMatrix2D((cxw, cyw), ang0, 1.0)
    back = cv2.getRotationMatrix2D((cxw, cyw), -ang0, 1.0)
    r = cv2.warpAffine(holes * 255, to, holes.shape[::-1], flags=cv2.INTER_NEAREST)
    lines = cv2.morphologyEx(r, cv2.MORPH_OPEN, np.ones((1, k), np.uint8))
    lines = cv2.dilate(cv2.warpAffine(lines, back, holes.shape[::-1], flags=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8))
    holes = (holes.astype(bool) & ~(lines > 0)).astype(np.uint8)
    n_h, lab_h, st_h, _ = cv2.connectedComponentsWithStats(holes, 8)
    keep = np.zeros_like(holes)
    for i in range(1, n_h):
        hw = st_h[i, cv2.CC_STAT_WIDTH] * st_h[i, cv2.CC_STAT_HEIGHT]
        # a digit, or a piece of one (Krea's thin segments break apart), is compact and shorter than the LCD
        if (st_h[i, cv2.CC_STAT_AREA] >= 12 and 0.15 * lh <= st_h[i, cv2.CC_STAT_HEIGHT] < 0.9 * lh
                and st_h[i, cv2.CC_STAT_AREA] >= 0.15 * hw):
            keep[lab_h == i] = 1
    holes = keep
    # what to erase: the digits and any pixel near them noticeably darker than the LCD (their soft edges)
    near = cv2.dilate(holes, np.ones((17, 17), np.uint8)).astype(bool)  # wide: a digit's segment at the LCD's rim
    soft = lcd & near & (dark > 0.35 * delta)
    erase = cv2.dilate((holes.astype(bool) | soft).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & lcd
    panel = (lcd & ~cv2.dilate(erase.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)).astype(np.uint8)
    fy, fx = np.where(filled)  # the panel's long axis: principal axis of its filled pixels
    cx, cy = fx.mean(), fy.mean()
    ev, evec = np.linalg.eigh(np.cov(np.stack([fx - cx, fy - cy])))
    ux, uy = evec[:, 1]
    if ux < 0:
        ux, uy = -ux, -uy
    ang = float(np.degrees(np.arctan2(uy, ux)))
    t0 = np.deg2rad(ang)
    u = (fx - cx) * np.cos(t0) + (fy - cy) * np.sin(t0)
    v = -(fx - cx) * np.sin(t0) + (fy - cy) * np.cos(t0)
    rw, rh = float(np.ptp(u)), float(np.ptp(v))
    # "dark": the darkness map for the 7-segment reader, zero within `edge` px of the LCD's border (its shaded edge
    # line is not ink; a wider band cut real digits near the edge: 30/91 session crops read right instead of 90)
    geo = {"window": (wx0, wy0, wx1, wy1), "panel": panel.astype(bool), "filled": filled.astype(bool),
           "holes": holes.astype(bool), "erase": erase, "dark": (dark / delta * (cv2.distanceTransform(filled, cv2.DIST_L2, 5) > edge)).astype(np.float32), "angle": float(ang), "center": (cx + wx0, cy + wy0), "panel_hw": (rw, rh)}
    ys, xs = np.where(holes)
    if len(xs) > 20:
        # the digits' extent in the panel's own axes (rotate the hole pixels by -angle about the panel's centre)
        t = np.deg2rad(ang)
        px, py = xs + wx0 - geo["center"][0], ys + wy0 - geo["center"][1]
        u, v = px * np.cos(t) + py * np.sin(t), -px * np.sin(t) + py * np.cos(t)
        hp = win[holes.astype(bool)].reshape(-1, 3).astype(np.float32)
        hg = gray[holes.astype(bool)]
        cnt, _ = cv2.findContours(holes, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        perim = sum(cv2.arcLength(c, True) for c in cnt)
        left, right = float(np.percentile(u, 0.5)), float(np.percentile(u, 99.5))
        geo.update(right=right, top=float(np.percentile(v, 1)), bottom=float(np.percentile(v, 99)),
                   ink=np.median(hp[hg <= np.percentile(hg, 30)], 0),  # the digits' core, not their soft edges
                   stroke=2 * len(xs) / max(perim, 1.0),  # area / half the perimeter ~ stroke width
                   pitch=(right - left) / old_digits if old_digits else None)
    else:
        geo.update(right=0.42 * rw, top=-0.3 * rh, bottom=0.3 * rh, stroke=None, pitch=None,
                   ink=np.percentile(win[panel.astype(bool)].reshape(-1, 3), 2, axis=0).astype(np.float32) * 0.35)
    return geo




def lcd_geometry(frame: np.ndarray, box: tuple[int, int, int, int], margin: float = 0.6,
                 old_digits: int | None = None, digits_box: tuple[int, int, int, int] | None = None) -> dict | None:
    """find_lcd + analyse on one frame (a clean frame: the start image)."""
    found = find_lcd(frame, box, margin)
    return analyse(frame, found[0], found[1], old_digits, digits_box) if found else None


class LcdTracker:
    """The LCD found once on the start image, then followed in each later frame by template matching over position
    and scale (the LCD with a ring of bezel, within `search` px; scales `scales`): Krea's camera drifts sideways (up
    to ~70 px) and closer (the LCD grew in later blocks of the 15 Nov session). `affine` maps start-image pixels to
    this frame's; the LCD mask and a kept digit layout (transform_layout) move with it."""

    def __init__(self, start: np.ndarray, box: tuple[int, int, int, int], search: int = 90,
                 scales: tuple[float, ...] = tuple(np.round(np.arange(0.8, 1.41, 0.04), 2))) -> None:
        # the fill's tolerance that fits this LCD's gradient differs per start image (seed 1: 6, seed 2: 10): try a
        # few, keep the one whose LCD holds the most digit pixels (a fill that leaks past the LCD is refused)
        best = None
        for tol in (4, 6, 8, 10, 12):
            found = find_lcd(start, box, tol=tol)
            g = analyse(start, found[0], found[1]) if found else None
            if g is not None and (best is None or g["holes"].sum() > best[0]):
                best = (int(g["holes"].sum()), found, tol)
        if best is None or best[0] == 0:
            # LIVE.md 16 Nov (start image s0): an LCD with a brightness gradient: a fill that follows neighbours stops
            # partway at tight tolerances and leaks onto the desk at loose ones; a fill against the seed's own
            # colour (fixed range) holds. Only when the fills above find no LCD with digits (s1 and s2 keep theirs)
            for tol in (15, 20, 25, 30):
                found = find_lcd(start, box, tol=tol, fixed=True)
                g = analyse(start, found[0], found[1]) if found else None
                if g is not None and (best is None or g["holes"].sum() > best[0]):
                    best = (int(g["holes"].sum()), found, -tol)
        if best is None or best[0] == 0:
            raise ValueError("no LCD with digits found on the start image")
        (self.window, self.filled), self.tol = best[1], best[2]
        wx0, wy0, wx1, wy1 = self.window
        ys, xs = np.where(self.filled)
        pad = 14  # the LCD with a ring of bezel: texture for matching
        self.t = (wx0 + max(0, xs.min() - pad), wy0 + max(0, ys.min() - pad), wx0 + xs.max() + pad, wy0 + ys.max() + pad)
        g = cv2.cvtColor(np.ascontiguousarray(start), cv2.COLOR_RGB2GRAY)
        self.tmpl = g[self.t[1]:self.t[3], self.t[0]:self.t[2]]
        self.search, self.scales, self.wide = search, scales, 220
        self.affine = np.array([[1.0, 0, 0], [0, 1.0, 0]])

    def reset(self) -> None:
        """Back to the start image's placement (a new session starts from it)."""
        self.affine = np.array([[1.0, 0, 0], [0, 1.0, 0]])

    def locate(self, frame: np.ndarray, near: bool = True, min_score: float = 0.6) -> tuple[np.ndarray, float]:
        """The start -> frame affine (scale s about the template's corner, then shift) and its match score. First
        near the last placement (scales within 0.08 of its scale, `search` px around it); if that matches below
        `min_score`, everywhere within `wide` px of the start position at every scale (replay 2103: Krea panned the
        camera up to 113 px over a few blocks, past a fixed search around the start)."""
        if near:
            a, sc_ = self._match(frame, self.affine, [s for s in self.scales if abs(s - self.affine[0, 0]) <= 0.081],
                                 self.search // 2)
            if sc_ >= min_score:
                return a, sc_
        return self._match(frame, np.array([[1.0, 0, 0], [0, 1.0, 0]]), self.scales, self.wide)

    def _match(self, frame: np.ndarray, around: np.ndarray, scales, search: int) -> tuple[np.ndarray, float]:
        g = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2GRAY)
        h, w = g.shape
        # where the template's corner was under `around`; scales are about that corner
        tx0, ty0 = self.t[0], self.t[1]
        px0 = around[0, 0] * tx0 + around[0, 2]
        py0 = around[1, 1] * ty0 + around[1, 2]
        best = (-1.0, None)
        for sc in scales:
            tm = cv2.resize(self.tmpl, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
            th, tw = tm.shape
            m = search + int(abs(sc - around[0, 0]) * max(th, tw))
            sx0, sy0 = max(0, int(px0) - m), max(0, int(py0) - m)
            sx1, sy1 = min(w, int(px0) + tw + m), min(h, int(py0) + th + m)
            area = g[sy0:sy1, sx0:sx1]
            if area.shape[0] < th or area.shape[1] < tw:
                continue
            _, score, _, (mx, my) = cv2.minMaxLoc(cv2.matchTemplate(area, tm, cv2.TM_CCOEFF_NORMED))
            if score > best[0]:
                nx, ny = sx0 + mx, sy0 + my  # where the template's corner lands
                best = (score, np.array([[sc, 0, nx - sc * tx0], [0, sc, ny - sc * ty0]]))
        return (best[1] if best[1] is not None else self.affine), float(best[0])

    def geometry(self, frame: np.ndarray, old_digits: int | None = None,
                 digits_box: tuple[int, int, int, int] | None = None, min_score: float = 0.4,
                 affine: np.ndarray | None = None) -> dict | None:
        """`affine`: a placement already found (e.g. on the block's last frame) instead of matching this frame."""
        if affine is None:
            a, score = self.locate(frame)
            if score < min_score:
                return None
        else:
            a, score = affine, float("nan")
        h, w = frame.shape[:2]
        wx0, wy0, wx1, wy1 = self.window
        corners = a @ np.array([[wx0, wx1, wx0, wx1], [wy0, wy0, wy1, wy1], [1, 1, 1, 1]], np.float64)
        cx0, cy0 = max(0, int(corners[0].min())), max(0, int(corners[1].min()))
        cx1, cy1 = min(w, int(np.ceil(corners[0].max()))), min(h, int(np.ceil(corners[1].max())))
        # the start's LCD mask (window coords) -> this frame's window coords
        to_win = np.vstack([a, [0, 0, 1]]) @ np.array([[1, 0, wx0], [0, 1, wy0], [0, 0, 1]], np.float64)
        to_win[0, 2] -= cx0
        to_win[1, 2] -= cy0
        filled = cv2.warpAffine(self.filled.astype(np.uint8), to_win[:2], (cx1 - cx0, cy1 - cy0), flags=cv2.INTER_NEAREST)
        if filled.sum() < 200:
            return None
        self.affine = a
        geo = analyse(frame, (cx0, cy0, cx1, cy1), filled, old_digits, digits_box)
        if geo is not None:
            geo["track"] = {"scale": round(float(a[0, 0]), 3), "dx": round(float(a[0, 2]), 1),
                            "dy": round(float(a[1, 2]), 1), "score": round(score, 3)}
            geo["affine"] = a
        return geo


def transform_layout(lay: dict, affine: np.ndarray, window: tuple[int, int, int, int]) -> dict:
    """A layout kept from the start image (start frame coordinates via its window) moved by a start -> frame affine
    into another window: the same LCD places on the moved, scaled LCD."""
    ox, oy = lay["window"][:2]
    a3 = np.vstack([affine, [0, 0, 1]])
    win_old = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], np.float64)
    win_new = np.array([[1, 0, -window[0]], [0, 1, -window[1]], [0, 0, 1]], np.float64)
    m = (win_new @ a3 @ win_old @ np.vstack([lay["m"], [0, 0, 1]]))[:2]
    return {**lay, "m": m, "window": window, "shape": (window[3] - window[1], window[2] - window[0]),
            # LIVE.md (16 Nov, push-in run): the stroke stays in the glyph canvas's units: render() thickens there and
            # its warp by `m` applies the scale (scaling it here too drew 2x-zoomed digits about twice too bold)
            "stroke": lay.get("stroke")}


def _background(win: np.ndarray, panel: np.ndarray, region: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The panel's colour at `region`'s pixels: a quadratic fit (in x, y) of its bright pixels, plus its residual noise."""
    ys, xs = np.where(panel)
    a = np.stack([np.ones_like(xs), xs, ys, xs * xs, ys * ys, xs * ys], 1).astype(np.float64)
    col = win[panel].astype(np.float64)
    coef, *_ = np.linalg.lstsq(a, col, rcond=None)
    resid = col - a @ coef
    ry, rx = np.where(region)
    b = np.stack([np.ones_like(rx), rx, ry, rx * rx, ry * ry, rx * ry], 1).astype(np.float64)
    noise = rng.normal(0, 1, (len(rx), 1)) * np.minimum(resid.std(0) * 0.3, 2.0)  # a fine grain, never speckle
    return np.clip(b @ coef + noise, 0, 255)


def _font(height: float, root: str) -> ImageFont.FreeTypeFont:
    size = max(8, int(height))
    for _ in range(3):  # fit the digit "8"'s ink height to the wanted height
        f = ImageFont.truetype(f"{root}/{FONT}", size)
        t, b = f.getbbox("8")[1], f.getbbox("8")[3]
        size = max(8, round(size * height / max(1, b - t)))
    return ImageFont.truetype(f"{root}/{FONT}", size)


def layout(geo: dict, root: str, n_chars: int = 5, slant: float = 0.18) -> dict:
    """Where the digits go: DSEG7 at the old digits' height, right-aligned at their right edge, scaled to their pitch,
    slanted and rotated with the LCD (window coordinates). Kept by the live loop after a write, so later writes and the
    7-segment reader (textbench/seg7.py) use the same places."""
    height = geo["bottom"] - geo["top"]
    f = _font(height, root)
    wx0, wy0, wx1, wy1 = geo["window"]
    cx, cy = geo["center"][0] - wx0, geo["center"][1] - wy0
    sx = float(np.clip(geo["pitch"] / max(1.0, f.getlength("8")), 0.6, 1.3)) if geo.get("pitch") else 1.0
    xr, yc = cx + geo["right"], cy + (geo["top"] + geo["bottom"]) / 2
    scale = np.array([[sx, 0, xr * (1 - sx)], [0, 1, 0], [0, 0, 1]], np.float64)  # about the right edge
    shear = np.array([[1, -slant, slant * yc], [0, 1, 0], [0, 0, 1]], np.float64)  # lean right about the middle row
    rot = cv2.getRotationMatrix2D((cx, cy), -geo["angle"], 1.0)
    return {"root": root, "size": f.size, "height": height, "n": n_chars, "x": cx + geo["right"],
            "y": cy + geo["top"], "m": rot @ shear @ scale, "shape": (wy1 - wy0, wx1 - wx0), "window": geo["window"],
            "stroke": geo.get("stroke")}


def _draw(lay: dict, f: ImageFont.FreeTypeFont, s: str) -> np.ndarray:
    ch, cw = lay["shape"]
    canvas = Image.new("L", (cw, ch), 0)
    ImageDraw.Draw(canvas).text((lay["x"] - f.getbbox(s)[2], lay["y"] - f.getbbox("8")[1]), s, font=f, fill=255)
    return np.asarray(canvas, np.uint8)


def render(lay: dict, text: str, thicken: bool = True) -> np.ndarray:
    """The digits' ink as a 0..1 mask in the window (no blur). DSEG's 7 has a top-left segment (it read as a 9 at
    small size in replay 2103): each 7 is drawn as its top bar and right side only (7 minus the segment that 4
    shares with it but 1 does not)."""
    f = ImageFont.truetype(f"{lay['root']}/{FONT}", lay["size"])
    s = text[-lay["n"]:].rjust(lay["n"], "!")  # DSEG: "!" is a blank one digit wide, so each digit keeps its place
    c = _draw(lay, f, s)
    for i, ch_ in enumerate(s):
        if ch_ == "7":
            def one(d: str) -> np.ndarray:
                return _draw(lay, f, "!" * i + d + "!" * (len(s) - 1 - i)) > 127
            seg_f = one("7") & one("4") & ~one("1")
            c = np.where(cv2.dilate(seg_f.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0, 0, c).astype(np.uint8)
    if thicken and lay.get("stroke"):  # DSEG's segments thickened to Krea's stroke width (measured the same way)
        b = (c > 127).astype(np.uint8)
        if b.any():
            cnt, _ = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            mine = 2 * b.sum() / max(1.0, sum(cv2.arcLength(q, True) for q in cnt))
            k = int(round(min(lay["stroke"] - mine, 0.12 * lay["height"])))
            if k > 0:
                c = cv2.dilate(c, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k + 1, k + 1)))
    ch, cw = lay["shape"]
    return cv2.warpAffine(c.astype(np.float32) / 255, lay["m"], (cw, ch), flags=cv2.INTER_LINEAR)


def write_lcd(frame: np.ndarray, box: tuple[int, int, int, int], text: str, root: str, n_chars: int = 5,
              slant: float = 0.18, ghost: float = 0.0, blur: float = 0.8, geo: dict | None = None,
              seed: int = 0, lay: dict | None = None) -> np.ndarray:
    """frame: RGB uint8; box: the display's digit area. Returns the frame with `text` on the LCD (unchanged if the
    panel is not found). `lay`: a kept layout (else one from this frame's old digits)."""
    geo = geo or lcd_geometry(frame, box)
    if geo is None:
        return frame
    wx0, wy0, wx1, wy1 = geo["window"]
    out = frame.copy()
    win = out[wy0:wy1, wx0:wx1]
    erase = geo["erase"]
    if erase.any():
        # fill from the LCD's own neighbouring pixels (its local shading), with everything outside the LCD first set
        # to the LCD's fitted colour so the dark bezel cannot bleed in; then a fine grain
        src = frame[wy0:wy1, wx0:wx1]
        outside = ~geo["filled"]
        tmp = np.ascontiguousarray(src.copy())
        if outside.any():
            tmp[outside] = _background(src, geo["panel"], outside, np.random.default_rng(seed)).astype(np.uint8)
        filled_in = cv2.inpaint(tmp, erase.astype(np.uint8), 7, cv2.INPAINT_TELEA)
        grain = np.random.default_rng(seed + 1).normal(0, 1.2, (int(erase.sum()), 1))
        win[erase] = np.clip(filled_in[erase].astype(np.float64) + grain, 0, 255).astype(np.uint8)
    if lay is not None:  # a kept layout (start image coordinates): moved with the tracked LCD
        lay = transform_layout(lay, geo.get("affine", np.array([[1.0, 0, 0], [0, 1.0, 0]])), geo["window"])
    else:
        lay = layout(geo, root, n_chars, slant)
    mask = render(lay, text)
    if ghost:
        mask = np.maximum(mask, render(lay, "8" * n_chars) * ghost)
    if blur:
        mask = cv2.GaussianBlur(mask, (0, 0), blur)
    # keep the ink on the LCD, with a few px of margin: the mask was eroded and the last digit sits near its edge
    on_lcd = cv2.dilate(geo["filled"].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(np.float32)
    mask = (mask * on_lcd)[..., None]
    win[:] = np.clip(win.astype(np.float32) * (1 - mask) + geo["ink"] * mask, 0, 255).astype(np.uint8)
    return out


def shift_layout(lay: dict, window: tuple[int, int, int, int]) -> dict:
    """A kept layout moved into another window (the tracked LCD's): the same frame positions."""
    ox, oy = lay["window"][0] - window[0], lay["window"][1] - window[1]
    m = lay["m"].copy()
    m[0, 2] += ox
    m[1, 2] += oy
    return {**lay, "m": m, "window": window, "shape": (window[3] - window[1], window[2] - window[0])}
