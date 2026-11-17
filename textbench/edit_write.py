"""Step 36 (Elad, 9 Nov 2026: "Is there a way to change just the letters but keep the letters part of the scene?"; "Yes"):
the memory write with an image editor drawing the new letters instead of our font.

The memory write (step 20) finds the old label on the newest memory frame, draws the new text over it in IBM Plex and
re-encodes the 9 memory frames; Krea copies the flat font forward, so the text looks pasted on. Here the newest memory
frame goes to Qwen-Image-Edit with step 27's instruction ("Change the text "A" to "B", in exactly the same font, size,
color, angle and position..."), which redrew labels in the scene's own lettering. Only what the editor changed near
the old label is kept (change_mask); that patch, colour-matched to the frame around it, is pasted into all 9 memory
frames at the label's tracked position (write_text.track_boxes), and Krea continues from it as in the memory write.

The editor is any callable (rgb uint8 HxWx3, prompt, seed) -> rgb of the same size; on the VM a separate process
(runner/edit_server.py) through FileEditor.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np

from textbench.write_text import track_boxes, write

Editor = Callable[[np.ndarray, str, int], np.ndarray]
DIFF = 28.0      # a pixel counts as changed by the editor above this RGB distance (0-441)
REACH = 2.0      # changed pixels count only within this many label heights of the old label's box
CROP_MIN = 256   # step 38: the crop sent to the editor is at least this many pixels on each side (in the frame)


def edit_prompt(pairs: list[tuple[str, str]]) -> str:
    """Step 27's instruction (prompts/pause2d.jsonl), for one or more labels."""
    change = " and ".join(f'the text "{old}" to "{new}"' for old, new in pairs)
    return (f"Change {change}, in exactly the same font, size, color, angle and position. "
            "Keep everything else in the image exactly the same.")


def label_boxes(frame: np.ndarray, pairs: list[tuple[str, str]], root: str) -> dict[str, tuple[int, int, int, int]]:
    """The old labels' boxes as the memory write finds them (write_text.write, OCR on the newest frame)."""
    _, boxes = write(frame, pairs, root)
    return {old: boxes[old] for old, _ in pairs if isinstance(boxes.get(old), tuple)}


def change_mask(before: np.ndarray, after: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Where the editor changed the label: pixels that differ by more than DIFF, closed and dilated by a quarter label
    height so whole letters and their edges are kept, then feathered. A changed region is kept whole if it touches the
    old box (widened by half a height); regions reach at most REACH heights from the box on every side (a longer new
    text may run past it). Changes not connected to a label (an editor's drift elsewhere) are left out.
    Step 36 fix: the first version cut the changes at 1 height above and below the box; on the ATM the OCR box held the
    small word AMOUNT and the tops of the big digits, so the new $65 was cut and Krea kept $25.
    Returns float32 HxW in [0, 1]."""
    import cv2
    h, w = before.shape[:2]
    diff = np.linalg.norm(after.astype(np.float32) - before.astype(np.float32), axis=2) > DIFF
    m = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in boxes:
        lh = max(4, y1 - y0)
        r, c = int(REACH * lh), lh // 2
        ya, yb, xa, xb = max(0, y0 - r), min(h, y1 + r), max(0, x0 - r), min(w, x1 + r)
        part = diff[ya:yb, xa:xb].astype(np.uint8)
        k = max(3, lh // 4) | 1
        part = cv2.dilate(cv2.morphologyEx(part, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)), np.ones((k, k), np.uint8))
        n, lab = cv2.connectedComponents(part)
        core = lab[max(0, y0 - c) - ya:min(h, y1 + c) - ya, max(0, x0 - c) - xa:min(w, x1 + c) - xa]
        keep = np.isin(lab, [v for v in np.unique(core) if v])
        m[ya:yb, xa:xb] |= keep.astype(np.uint8)
    if not m.any():
        return m.astype(np.float32)
    return np.clip(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5) * 1.5, 0, 1)


def crop_box(shape: tuple[int, ...], boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    """Step 38: the part of the frame the editor sees: the labels' union widened by REACH label heights on every side
    (so it holds everything change_mask can keep) and to at least CROP_MIN on each side, inside the frame."""
    h, w = shape[:2]
    lh = max(4, max(y1 - y0 for _, y0, _, y1 in boxes))
    x0, y0 = min(b[0] for b in boxes) - int(REACH * lh), min(b[1] for b in boxes) - int(REACH * lh)
    x1, y1 = max(b[2] for b in boxes) + int(REACH * lh), max(b[3] for b in boxes) + int(REACH * lh)
    for lo, hi, size in ((0, 2, w), (1, 3, h)):
        box = [x0, y0, x1, y1]
        need = min(size, CROP_MIN) - (box[hi] - box[lo])
        if need > 0:
            box[lo] -= need // 2
            box[hi] += need - need // 2
        shift = max(0, -box[lo]) - max(0, box[hi] - size)
        box[lo], box[hi] = max(0, box[lo] + shift), min(size, box[hi] + shift)
        x0, y0, x1, y1 = box
    return x0, y0, x1, y1


def edit_crop(frame: np.ndarray, prompt: str, seed: int, editor: Editor,
              boxes: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Step 38: the editor redraws only the crop around the labels; the rest of the frame is left as it was."""
    x0, y0, x1, y1 = crop_box(frame.shape, boxes)
    out = frame.copy()
    out[y0:y1, x0:x1] = editor(np.ascontiguousarray(frame[y0:y1, x0:x1]), prompt, seed)
    return out, (x0, y0, x1, y1)


def colour_match(before: np.ndarray, after: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The editor may shift the whole image's tone slightly; the patch gets the frame's tone back: the mean RGB
    difference on a ring just outside the mask (where nothing should have changed) is removed from the edited image."""
    import cv2
    hard = (mask > 0.05).astype(np.uint8)
    ring = cv2.dilate(hard, np.ones((9, 9), np.uint8)) - hard
    if ring.sum() < 20:
        return after
    shift = (before[ring > 0].astype(np.float32) - after[ring > 0].astype(np.float32)).mean(0)
    return np.clip(after.astype(np.float32) + shift, 0, 255).astype(np.uint8)


def shift(a: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """a moved by (dx, dy) pixels, zeros where it leaves the frame."""
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = a[ys, xs]
    return out


def paste(frame: np.ndarray, patch: np.ndarray, mask: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    m = shift(mask, dx, dy)[..., None]
    return np.clip(frame * (1 - m) + shift(patch, dx, dy) * m, 0, 255).astype(np.uint8)


def edit_cache(cache: list[np.ndarray], pairs: list[tuple[str, str]], editor: Editor, root: str, seed: int = 0,
               track: bool = True, crop: bool = False,
               boxes_by_old: dict[str, tuple[int, int, int, int]] | None = None) -> tuple[list[np.ndarray], int, dict]:
    """The 9 memory frames (oldest first) with the old labels redrawn by the editor. Returns (frames, labels found,
    info: prompt, boxes, editor seconds, share of the box area the editor changed)."""
    pairs = [(o, n) for o, n in pairs if o and n and o != n]
    newest = cache[-1]
    # LIVE.md: a live session passes the label's known box (OCR cannot find a lone digit by its old text)
    if boxes_by_old:  # code-live-r2 #4: only the boxes of pairs still to write
        boxes = {o: tuple(b) for o, b in boxes_by_old.items() if b and any(o == p for p, _ in pairs)}
    else:
        boxes = label_boxes(newest, pairs, root)
    info: dict = {"found": len(boxes)}
    if not boxes:
        return cache, 0, info
    prompt = edit_prompt([(o, n) for o, n in pairs if o in boxes])
    t = time.perf_counter()
    if crop:  # step 38
        edited, info["crop"] = edit_crop(newest, prompt, seed, editor, list(boxes.values()))
    elif boxes_by_old and hasattr(editor, "boxes"):
        editor.boxes = {k: list(v) for k, v in boxes.items()}  # sent with the request (runner/edit_server.py)
        try:
            edited = editor(newest, prompt, seed)
        finally:
            editor.boxes = None
    else:
        edited = editor(newest, prompt, seed)
    info.update(prompt=prompt, boxes={k: list(v) for k, v in boxes.items()}, editor_s=time.perf_counter() - t)
    mask = change_mask(newest, edited, list(boxes.values()))
    area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes.values())
    info["changed"] = float((mask > 0.5).sum() / max(1, area))
    if not mask.any():
        return cache, 0, info
    patch = colour_match(newest, edited, mask)
    out = []
    for c in cache:
        if not track:
            out.append(paste(c, patch, mask))
            continue
        moved = track_boxes(newest, c, boxes)
        k = max(boxes, key=lambda b: (boxes[b][2] - boxes[b][0]) * (boxes[b][3] - boxes[b][1]))  # the largest label leads
        out.append(paste(c, patch, mask, moved[k][0] - boxes[k][0], moved[k][1] - boxes[k][1]))
    return out, len(boxes), info


class FileEditor:
    """The Krea process's side of runner/edit_server.py: a request is <n>.png + <n>.json in the folder, the answer
    <n>.out.png + <n>.out.json. before/after are called around the wait (Krea frees the GPU for the editor)."""

    def __init__(self, folder: Path, before: Callable[[], None] = lambda: None, after: Callable[[], None] = lambda: None,
                 timeout: float = 900.0) -> None:
        self.folder, self.before, self.after, self.timeout, self.n = Path(folder), before, after, timeout, 0
        self.last: dict = {}
        self.mode = "full"  # step 37: "fast", the 8-step Lightning editor (runner/edit_server.py); step 38: "resident"
        self.boxes: dict | None = None  # LIVE.md: known label boxes, sent with the request

    def wait_ready(self, timeout: float = 3600.0) -> None:
        t = time.time()
        while not (self.folder / "ready").exists():
            if (self.folder / "failed").exists() or time.time() - t > timeout:
                raise RuntimeError(f"editor server not ready: {(self.folder / 'failed').read_text() if (self.folder / 'failed').exists() else 'timeout'}")
            time.sleep(2)

    def __call__(self, rgb: np.ndarray, prompt: str, seed: int) -> np.ndarray:
        from PIL import Image
        self.n += 1
        # code-live-r2 #5: unique across processes; code-klein-r1 blocker: and across FileEditor instances of one process
        # (a replay makes one per arm and seed; with pid_n alone seed 2's requests met seed 1's old answers)
        req = self.folder / f"{os.getpid()}_{id(self):x}_{time.time_ns()}_{self.n:05d}"
        Image.fromarray(rgb).save(req.with_suffix(".png"))
        tmp = req.with_suffix(".json.part")
        tmp.write_text(json.dumps({"prompt": prompt, "seed": seed, "mode": self.mode,
                                   **({"boxes": self.boxes} if self.boxes else {})}))
        self.before()
        try:
            tmp.rename(req.with_suffix(".json"))  # the server only picks up complete requests
            out, t = Path(f"{req}.out.json"), time.time()
            while not out.exists():
                if time.time() - t > self.timeout:
                    raise TimeoutError(f"no edit for {req.name} in {self.timeout:.0f} s")
                time.sleep(0.02)  # step 40b: was 0.2 (with the server's 0.2, ~0.4 s added to a ~0.4 s edit)
            self.last = json.loads(out.read_text())
            if "error" in self.last:
                raise RuntimeError(f"editor failed on {req.name}: {self.last['error']}")
            return np.asarray(Image.open(f"{req}.out.png").convert("RGB"))
        finally:
            self.after()
