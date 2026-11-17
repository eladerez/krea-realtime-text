"""The klein restyle of the 7-segment font write (LIVE.md, 16 Nov 2026; Elad: "I want the digits to be part of the
scene"): the square crop around the LCD of a font-written frame goes to FLUX.2 klein 4B with PROMPT (the "short"
prompt of runner/restyle_test.py); its answer goes back inside the LCD only (paste) and keeps klein's lightness but
takes the font frame's colour (recolour, as scripts/restyle_fix.py). Offline on 19 frames: OCR read the wanted number
on 89.5% of seed 0's answers (data/scores/restyle_fix.json).
The live loop (runner/gen_krea.py KreaLive.restyle) restyles the newest memory frame, keeps klein's answer only when
OCR reads the wanted number on it, and puts the same answer into each memory frame.
"""
from __future__ import annotations

import cv2
import numpy as np

EDGE = 512  # klein's input and output side (step 40a)
PROMPT = ('Make this calculator display look real and natural, like the rest of the photo. It shows "{n}"; do not '
          "change the digits.")


def lcd_mask(g: dict, shape: tuple) -> np.ndarray:
    """The LCD's mask (uint8 0/255) in frame pixels, from the tracker's geometry (window and filled)."""
    m = np.zeros(shape[:2], np.uint8)
    x0, y0, x1, y1 = g["window"]
    m[y0:y1, x0:x1] = g["filled"].astype(np.uint8) * 255
    return m


def square_crop(lcd: np.ndarray) -> list[int]:
    """A square of side 1.5x the LCD's larger extent around its centre, inside the frame (scripts/restyle_inputs.py)."""
    ys, xs = np.nonzero(lcd)
    h, w = lcd.shape
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = min(int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.5), h, w)
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    return [x0, y0, x0 + side, y0 + side]


def soft(lcd: np.ndarray) -> np.ndarray:
    """The LCD mask dilated 4 px and feathered 3 px, as float (H, W, 1)."""
    return cv2.GaussianBlur(cv2.dilate(lcd, np.ones((9, 9), np.uint8)).astype(np.float32) / 255, (0, 0), 3)[..., None]


def paste(frame: np.ndarray, back: np.ndarray, crop: list[int], lcd: np.ndarray) -> np.ndarray:
    """klein's crop (already at the crop's size) back into the frame, only inside the LCD (runner/restyle_test.py)."""
    x0, y0, x1, y1 = crop
    m = soft(lcd)[y0:y1, x0:x1]
    out = frame.copy()
    out[y0:y1, x0:x1] = np.clip(frame[y0:y1, x0:x1] * (1 - m) + back * m, 0, 255).astype(np.uint8)
    return out


def recolour(klein: np.ndarray, font: np.ndarray, lcd: np.ndarray) -> np.ndarray:
    """klein's lightness (CIELAB L) with the font frame's colour (a, b), inside the LCD (scripts/restyle_fix.py)."""
    lk = cv2.cvtColor(klein, cv2.COLOR_RGB2LAB).astype(np.float32)
    mix = cv2.cvtColor(font, cv2.COLOR_RGB2LAB).astype(np.float32)
    mix[..., 0] = lk[..., 0]
    out = cv2.cvtColor(np.clip(mix, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    m = soft(lcd)
    return np.clip(klein * (1 - m) + out * m, 0, 255).astype(np.uint8)


def put(font: np.ndarray, back: np.ndarray, crop: list[int], lcd: np.ndarray) -> np.ndarray:
    """A font-written frame with klein's answer in its LCD, colour-fixed."""
    return recolour(paste(font, back, crop, lcd), font, lcd)


def box_at(box: tuple[int, int, int, int], affine: np.ndarray) -> tuple[int, int, int, int]:
    """The start image's digit box moved by the tracker's start -> frame affine (for OCR's search window)."""
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0, 1.0], [x1, y1, 1.0]]) @ np.asarray(affine, float).T
    return int(pts[0, 0]), int(pts[0, 1]), int(pts[1, 0]), int(pts[1, 1])
