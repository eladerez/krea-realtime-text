"""Read a 7-segment LCD by its segments (LIVE.md, 15 Nov 2026): OCR misreads these slanted LCD digits both ways (Krea's
"68" as "88" is right, but DSEG's "23456" as "952" and as nothing is wrong), so the live checker and the scores need a
reader made for them.

segment_masks: for a layout (textbench/write_lcd.py layout: the LCD places, height, pitch, slant, angle), each digit
0-9 is rendered alone in each place; a pixel of "8" belongs to the segment whose on/off pattern over the ten digits
it follows (a..g, standard 7-segment table; best agreement, at least 9 of 10). Each segment's mask is eroded to its core.
read: for each place, each segment's mean darkness (the LCD's darkness map: how far below its fitted brightness, in
units of the ink threshold, from write_lcd.analyse) decides on/off (on: above `on`, and above `rel` of the brightest
segment on the display); the 7-bit pattern is decoded by the table (exact; or "?" when it is no digit). A place with no
segment on is blank. Returns the digits read (blanks dropped), per place the pattern and a confidence (the smallest
margin of a segment from the on/off threshold).
"""
from __future__ import annotations

import cv2
import numpy as np

from textbench.write_lcd import render

SEGS = "abcdefg"
TABLE = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg", "5": "acdfg", "6": "acdefg", "7": "abc",
         "8": "abcdefg", "9": "abcdfg"}  # as textbench/write_lcd.py render draws them (its 7 without DSEG's segment f)
DECODE = {frozenset(v): k for k, v in TABLE.items()}
# the other common LCD forms: 7 with f (as DSEG draws it), 6 without a, 9 without d
DECODE.update({frozenset("abcf"): "7", frozenset("cdefg"): "6", frozenset("abcfg"): "9"})


def segment_masks(lay: dict) -> list[dict[str, np.ndarray]]:
    """Per place (left to right), segment -> boolean mask in the layout's window."""
    n = lay["n"]
    places = []
    for p in range(n):
        lit = {}
        for d in TABLE:
            text = d + "!" * (n - 1 - p)  # the digit in place p (right alignment pads the left with blanks)
            lit[d] = render(lay, text, thicken=False) > 0.5
        eight = lit["8"]
        votes = np.zeros(eight.shape + (7,), np.int16)
        for si, sname in enumerate(SEGS):
            for d, segs in TABLE.items():
                votes[..., si] += (lit[d] == (sname in segs))
        best = votes.argmax(-1)
        agree = votes.max(-1)
        masks = {}
        for si, sname in enumerate(SEGS):
            m = (eight & (best == si) & (agree >= 9)).astype(np.uint8)
            core = cv2.erode(m, np.ones((2, 2), np.uint8))
            masks[sname] = (core if core.sum() >= 4 else m).astype(bool)
        places.append(masks)
    return places


def read(dark: np.ndarray, places: list[dict[str, np.ndarray]], on: float = 0.6, rel: float = 0.35) -> dict:
    """dark: the LCD's darkness map in the layout's window (write_lcd.analyse geo["dark"])."""
    vals = [{s: float(dark[m].mean()) if m.any() else 0.0 for s, m in segs.items()} for segs in places]
    top = max((v for pv in vals for v in pv.values()), default=0.0)
    thr = max(on, rel * top)
    out, pats, conf = "", [], []
    for pv in vals:
        lit = frozenset(s for s, v in pv.items() if v > thr)
        pats.append("".join(sorted(lit)))
        conf.append(round(min(abs(v - thr) for v in pv.values()), 3))
        if not lit:
            continue
        out += DECODE.get(lit, "?")
    return {"reads": out, "patterns": pats, "confidence": conf, "threshold": round(thr, 3)}


class MaskCache:
    """Segment masks for a kept layout (textbench/write_lcd.py), per tracked placement, fast: computed once per scale
    on the whole frame (placement = scale only) and shifted to each placement's offset, then cut to its window
    (replay 2103: the lock must read every frame; segment_masks takes ~0.1 s)."""

    def __init__(self, lay0: dict, frame_hw: tuple[int, int] = (480, 832)) -> None:
        self.lay0, self.hw, self.base = lay0, frame_hw, {}

    def for_geo(self, g: dict) -> list[dict[str, np.ndarray]]:
        from textbench.write_lcd import transform_layout
        a = g["affine"]
        sc = round(float(a[0, 0]), 2)
        if sc not in self.base:
            # code-klein-r2 blocker: the canvas grows with the scale (an 832x480 canvas cut the right-hand digit places
            # off past ~1.5x); never smaller than the frame, so scales up to 1 are drawn exactly as before
            h, w = max(self.hw[0], int(np.ceil(self.hw[0] * sc))), max(self.hw[1], int(np.ceil(self.hw[1] * sc)))
            self.base[sc] = segment_masks(transform_layout(self.lay0, np.array([[sc, 0, 0], [0, sc, 0]]), (0, 0, w, h)))
        tx, ty = int(round(float(a[0, 2]))), int(round(float(a[1, 2])))
        x0, y0, x1, y1 = g["window"]
        out = []
        for place in self.base[sc]:
            cut = {}
            for k, m in place.items():
                h, w = m.shape  # the base canvas's own size
                c = np.zeros((y1 - y0, x1 - x0), bool)
                sy0, sy1, sx0, sx1 = y0 - ty, y1 - ty, x0 - tx, x1 - tx  # source rows/cols in the base mask
                cy0, cx0 = max(0, -sy0), max(0, -sx0)
                sy0c, sx0c, sy1c, sx1c = max(0, sy0), max(0, sx0), min(h, sy1), min(w, sx1)
                if sy1c > sy0c and sx1c > sx0c:
                    c[cy0:cy0 + sy1c - sy0c, cx0:cx0 + sx1c - sx0c] = m[sy0c:sy1c, sx0c:sx1c]
                cut[k] = c
            out.append(cut)
        return out
