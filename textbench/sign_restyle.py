"""LIVE.md 19 Nov 2026 (Elad: "Do a, b and c"; "It's kind of blurry and it doesn't look part of the scene"): FLUX.2
klein restyles the general writer's pasted text (textbench/paste_writer.py) on any surface, as textbench/restyle.py does
for the calculator's LCD. The square crop around the text region (lora_targets.text_region of the tracked quad) goes to
klein (runner/edit_server.py mode "restyle") at 512 px with PROMPT; its answer goes back inside the region only
(feathered), and is kept only when OCR reads the wanted line at the quad on the result (lora_targets.read_text). Used
by option B (the memory's paste restyled once per write, then held by the LoRA) and option C (the teacher's memory).
"""
from __future__ import annotations

import time

import cv2
import numpy as np
from PIL import Image

EDGE = 512
PROMPT = ('Make the text "{t}" look real and natural, part of this photo: the same material, print, lighting, focus '
          "and grain as the surface it is on. Keep exactly the same letters and numbers, in the same place and size.")


def region_mask(shape: tuple, quad: np.ndarray) -> np.ndarray:
    """uint8 0/255: the text region around the quad (lora_targets.text_region, the training targets' region)."""
    from textbench.lora_targets import text_region
    return (text_region(shape, quad) > 0).astype(np.uint8) * 255


def square_crop(mask: np.ndarray) -> list[int]:
    """A square of side 1.6x the region's larger extent around its centre, inside the frame."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = min(max(int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.6), 64), h, w)
    x0 = int(np.clip(cx - side / 2, 0, w - side))
    y0 = int(np.clip(cy - side / 2, 0, h - side))
    return [x0, y0, x0 + side, y0 + side]


def put(frame: np.ndarray, back: np.ndarray, crop: list[int], mask: np.ndarray) -> np.ndarray:
    """klein's crop (at the crop's size) back into the frame inside the region only, feathered 3 px."""
    x0, y0, x1, y1 = crop
    m = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), 3)[y0:y1, x0:x1, None]
    out = frame.copy()
    out[y0:y1, x0:x1] = np.clip(frame[y0:y1, x0:x1] * (1 - m) + back * m, 0, 255).astype(np.uint8)
    return out


def restyle(frame: np.ndarray, quad: np.ndarray, line: str, editor, seeds=(0, 1)) -> tuple[dict | None, dict]:
    """editor(img512, prompt, seed) -> img512. Returns ({"crop", "mask", "back"} to apply with apply(), or None when no
    seed's answer reads `line`) and an info dict (klein_s, reads, how)."""
    from textbench.lora_targets import read_text
    from textbench.paste_writer import same
    info: dict = {"line": line, "how": "font", "reads": [], "klein_s": 0.0}
    mask = region_mask(frame.shape, quad)
    if not mask.any():
        info["error"] = "empty region"
        return None, info
    crop = square_crop(mask)
    x0, y0, x1, y1 = crop
    inp = np.asarray(Image.fromarray(np.ascontiguousarray(frame[y0:y1, x0:x1])).resize((EDGE, EDGE), Image.Resampling.LANCZOS))
    for sd in seeds:
        t = time.perf_counter()
        ans = editor(inp, PROMPT.format(t=line), sd)
        info["klein_s"] = round(info["klein_s"] + time.perf_counter() - t, 3)
        back = np.asarray(Image.fromarray(ans).resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS))
        r = read_text(put(frame, back, crop, mask), quad)
        info["reads"].append(r)
        if r is not None and same(r, line):
            info["how"] = f"klein s{sd}"
            return {"crop": crop, "mask": mask, "back": back}, info
    return None, info


def apply(frame: np.ndarray, r: dict) -> np.ndarray:
    """The kept answer put into another frame at the same place (memory frames share the newest frame's placement)."""
    return put(frame, r["back"], r["crop"], r["mask"])
