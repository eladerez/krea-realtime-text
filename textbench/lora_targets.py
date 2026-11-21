"""Training targets for the LoRA pilot (research/plans/LORA_PILOT.md; Elad: "Yes", 16 Nov 2026): a block of Krea's frames
with its display made to show the wanted number, in Krea's own style where possible.

Per frame the LCD is tracked and read (textbench/write_lcd.py LcdTracker, textbench/seg7.py, as the lock does):
- frames that read the wanted number stay as Krea made them;
- a frame that reads wrong gets its LCD from the nearest frame of the same block that reads right (Krea's own pixels),
  moved by the two frames' tracked placements (start -> frame affines) and put in inside the LCD (dilated 4 px,
  feathered 3 px, textbench/restyle.py soft): Krea's drift within a block (23476 -> 29476) is undone with its own drawing;
- if no frame of the block reads right, the font write of the middle frame restyled by klein (textbench/restyle.py,
  kept only when OCR reads the number) is the source instead, when an editor is given;
- every changed frame is read again (7-segment reader for Krea's pixels, OCR for klein's); a frame not confirmed is
  marked unverified, and the trainer leaves its latent's display out of the loss.
"""
from __future__ import annotations

import cv2
import numpy as np

from textbench.display_check import digits, read_display
from textbench.restyle import EDGE, PROMPT, box_at, lcd_mask, put, soft, square_crop
from textbench.seg7 import read as seg7_read
from textbench.write_lcd import write_lcd


def _a3(a: np.ndarray) -> np.ndarray:
    return np.vstack([np.asarray(a, np.float64), [0, 0, 1]])


def transplant(dst: np.ndarray, src: np.ndarray, a_dst: np.ndarray, a_src: np.ndarray, lcd_dst: np.ndarray) -> np.ndarray:
    """src's LCD moved onto dst's LCD (src -> start -> dst placement) and put in inside dst's LCD, feathered."""
    m = (_a3(a_dst) @ np.linalg.inv(_a3(a_src)))[:2]
    h, w = dst.shape[:2]
    moved = cv2.warpAffine(src, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    k = soft(lcd_dst)
    return np.clip(dst * (1 - k) + moved * k, 0, 255).astype(np.uint8)


def klein_source(frame: np.ndarray, g: dict, text: str, root: str, lay0: dict, start_box, editor, seeds=(0, 1)) -> tuple:
    """The font write of `text` on `frame`, restyled by klein (editor(img512, prompt, seed) -> img), colour-fixed, kept
    when OCR reads the number; else (None, info)."""
    from PIL import Image
    font = write_lcd(frame, None, text, root, geo=g, lay=lay0)
    lcd = lcd_mask(g, font.shape)
    crop = square_crop(lcd)
    x0, y0, x1, y1 = crop
    inp = np.asarray(Image.fromarray(np.ascontiguousarray(font[y0:y1, x0:x1])).resize((EDGE, EDGE), Image.Resampling.LANCZOS))
    box = box_at(start_box, g["affine"])
    reads = []
    for sd in seeds:
        ans = editor(inp, PROMPT.format(n=text), sd)
        back = np.asarray(Image.fromarray(ans).resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS))
        out = put(font, back, crop, lcd)
        r = read_display(out, box)["reads"]
        reads.append(r)
        if r == digits(text):
            return out, {"klein_seed": sd, "reads": reads}
    return None, {"reads": reads}


def build(frames: list[np.ndarray], want_text: str, tracker, masks, root: str, lay0: dict, start_box,
          editor=None) -> dict:
    """frames: the block's RGB frames as Krea made them. Returns {"frames": corrected list, "verified": [bool],
    "lcd": [uint8 mask or None], "how": [str], "info": {...}}. The tracker's state moves as in the lock (near search)."""
    want = digits(want_text)
    n = len(frames)
    geo, reads = [None] * n, [None] * n
    for i, f in enumerate(frames):
        a, sc = tracker.locate(f, min_score=0.4)
        if sc < 0.4:
            continue
        g = tracker.geometry(f, affine=a)
        if g is None:
            continue
        geo[i] = g
        reads[i] = seg7_read(g["dark"], masks.for_geo(g))["reads"]
    right = [r == want for r in reads]
    out, verified, how = list(frames), [False] * n, ["lost"] * n
    lcd = [lcd_mask(g, f.shape) if g is not None else None for g, f in zip(geo, frames)]
    for i in range(n):
        if right[i]:
            verified[i], how[i] = True, "krea"
    src_frame, src_a, src_how = None, None, None
    if not any(right):
        mid = next((i for i in sorted(range(n), key=lambda i: abs(i - n // 2)) if geo[i] is not None), None)
        if editor is not None and mid is not None:
            try:
                src_frame, kinfo = klein_source(frames[mid], geo[mid], want_text, root, lay0, start_box, editor)
            except Exception as ex:  # an editor error leaves the block unverified, never stops the collection
                src_frame, kinfo = None, {"error": f"{type(ex).__name__}: {ex}"[:200]}
            if src_frame is not None:
                src_a, src_how = geo[mid]["affine"], "klein"
        info = {"source": src_how or "none", "klein": kinfo if editor is not None and mid is not None else None}
    else:
        # code-lora-r1 should-fix 3: OCR (independent of the 7-segment reader that chose it) on the block's main source
        src_i = max((i for i in range(n) if right[i]), key=lambda i: sum(1 for k in range(n) if not right[k] and
                    min((j for j in range(n) if right[j]), key=lambda j: abs(j - k)) == i), default=None)
        info = {"source": "krea", "source_frame": src_i,
                "source_ocr": read_display(frames[src_i], box_at(start_box, geo[src_i]["affine"]))["reads"]
                if src_i is not None and any(not r for r in right) else None}
    for i in range(n):
        if right[i] or geo[i] is None:
            continue
        if any(right):
            j = min((j for j in range(n) if right[j]), key=lambda j: abs(j - i))
            out[i] = transplant(frames[i], frames[j], geo[i]["affine"], geo[j]["affine"], lcd[i])
            g2 = tracker.geometry(out[i], affine=geo[i]["affine"])
            ok = g2 is not None and seg7_read(g2["dark"], masks.for_geo(g2))["reads"] == want
            how[i] = "from frame %d" % j
        elif src_frame is not None:
            out[i] = transplant(frames[i], src_frame, geo[i]["affine"], src_a, lcd[i])
            ok = read_display(out[i], box_at(start_box, geo[i]["affine"]))["reads"] == want
            how[i] = "klein"
        else:
            ok, how[i] = False, "no source"
        verified[i] = bool(ok)
    info.update(reads=reads, right=sum(right), verified=sum(verified), n=n)
    return {"frames": out, "verified": verified, "lcd": lcd, "how": how, "info": info}


def latent_weights(lcd: list, verified: list, frames_per_latent: int = 4, hw: tuple[int, int] = (60, 104),
                   inside: float = 1.0, outside: float = 0.1) -> np.ndarray:
    """Loss weights per latent cell [n_lat, 60, 104]: `inside` on the LCD (union of its frames' masks, one cell
    dilated), `outside` elsewhere; a latent whose frames are not all verified gets 0 on its LCD cells."""
    n_lat = len(lcd) // frames_per_latent
    w = np.full((n_lat, *hw), outside, np.float32)
    for j in range(n_lat):
        fr = range(j * frames_per_latent, (j + 1) * frames_per_latent)
        ms = [lcd[i] for i in fr if lcd[i] is not None]
        if not ms:
            continue
        u = np.maximum.reduce([m.astype(np.float32) / 255 for m in ms])
        cells = cv2.dilate((cv2.resize(u, (hw[1], hw[0]), interpolation=cv2.INTER_AREA) > 0.05).astype(np.uint8),
                           np.ones((3, 3), np.uint8)).astype(bool)
        ok = all(verified[i] for i in fr)
        w[j][cells] = inside if ok else 0.0
    return w


# LIVE.md 17 Nov (v3: any surface, letters, a moving camera): the same targets for the general writer's scenes. The
# text line is tracked from the start image (textbench/paste_writer.TextTracker) and read by OCR in each frame; a frame
# that reads wrong gets the text region of the nearest right frame of its block, moved by the two frames' tracked quads
# (a perspective warp), put in inside a padded text region, feathered; every changed frame is read again.

def text_region(shape, quad: np.ndarray, grow_u: float = 0.35, grow_v: float = 0.35) -> np.ndarray:
    """The tracked line's region (padded by line heights along and across it), as a uint8 mask (0/255)."""
    from textbench.paste_writer import _region
    return (_region(shape, quad, grow_u, grow_v) * 255).astype(np.uint8)


def read_text(frame: np.ndarray, quad: np.ndarray) -> str:
    """OCR's lines whose centre lies in the quad (padded by a third of its height along it, 0.15 across), joined."""
    from textbench.paste_writer import norm, ocr_lines
    h = float(np.linalg.norm(quad[3] - quad[0]))
    x0, y0 = quad.min(0) - np.array([h / 3, h * 0.15])
    x1, y1 = quad.max(0) + np.array([h / 3, h * 0.15])
    got = sorted((q[:, 0].min(), t) for q, t, _ in ocr_lines(frame)
                 if x0 <= q[:, 0].mean() <= x1 and y0 <= q[:, 1].mean() <= y1)
    return "".join(norm(t) for _, t in got)


def build_text(frames: list[np.ndarray], want_line: str, tracker) -> dict:
    """frames: the block's RGB frames as Krea made them; want_line: the whole line expected (e.g. "ROUTE 47").
    Returns the same dict as build() ("lcd" holds the text region masks)."""
    from textbench.paste_writer import same
    n = len(frames)
    quads, reads = [None] * n, [None] * n
    for i, f in enumerate(frames):
        q, _n = tracker.locate(f)
        if q is None:
            continue
        quads[i] = q
        reads[i] = read_text(f, q)
    right = [r is not None and same(r, want_line) for r in reads]
    out, verified, how = list(frames), [False] * n, ["lost"] * n
    masks = [text_region(f.shape, q) if q is not None else None for q, f in zip(quads, frames)]
    for i in range(n):
        if right[i]:
            verified[i], how[i] = True, "krea"
    src_i = None
    for i in range(n):
        if right[i] or quads[i] is None:
            continue
        if not any(right):
            how[i] = "no source"
            continue
        j = min((j for j in range(n) if right[j]), key=lambda j: abs(j - i))
        src_i = j if src_i is None else src_i
        m = cv2.getPerspectiveTransform(quads[j].astype(np.float32), quads[i].astype(np.float32))
        h, w = frames[i].shape[:2]
        moved = cv2.warpPerspective(frames[j], m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        k = soft(masks[i])
        out[i] = np.clip(frames[i] * (1 - k) + moved * k, 0, 255).astype(np.uint8)
        how[i] = "from frame %d" % j
        verified[i] = same(read_text(out[i], quads[i]), want_line)
    info = {"source": "krea" if any(right) else "none", "source_frame": src_i,
            "source_ocr": reads[src_i] if src_i is not None else None,
            "reads": reads, "right": sum(right), "verified": sum(verified), "n": n}
    return {"frames": out, "verified": verified, "lcd": masks, "how": how, "info": info}
