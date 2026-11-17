"""Step 39 (PLAN.md): a small model that restyles our font write into the scene's lettering, in the latent space of
Wan's VAE (the VAE Krea draws through; diffusers AutoencoderKLWan, run in the gen environment).

Input, for a square window around the label resized to WIN pixels: the window as it was (O), the same window with the new
text drawn in our font (F, textbench/write_text.py), both encoded by the VAE to 16 x WIN/8 x WIN/8 latents, and the label's
mask at latent size. Output: the latent of the window as the editor write makes it (E), predicted as a change to F's latent.
Decoded, it goes through the editor write's paste (textbench/edit_write.py), so to the rest of the pipeline it is an editor.
The numpy helpers below have no torch dependency; the model is built only when torch is imported (VM).
"""
from __future__ import annotations

import numpy as np

WIN = 256         # the window's side in pixels after resizing (32 x 32 latents)
WIN_MIN = 256     # the window is at least this many frame pixels on a side
LABEL_SHARE = 0.6  # the label takes at most this share of the window's width


def window(shape: tuple[int, ...], box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """A square region of the frame around the label: wide enough that the label is at most LABEL_SHARE of it, at
    least 4 label heights and WIN_MIN pixels, inside the frame (moved, then cut at the frame's smaller side)."""
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    side = int(max(WIN_MIN, (x1 - x0) / LABEL_SHARE, 4 * (y1 - y0)))
    side = min(side, h, w)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    wx = min(max(0, cx - side // 2), w - side)
    wy = min(max(0, cy - side // 2), h - side)
    return wx, wy, wx + side, wy + side


def cut(frame: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """The window, resized to WIN x WIN."""
    import cv2
    x0, y0, x1, y1 = rect
    return cv2.resize(frame[y0:y1, x0:x1], (WIN, WIN), interpolation=cv2.INTER_AREA)


def put(frame: np.ndarray, patch: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    """The frame with the window replaced by the patch, resized back to the window's size."""
    import cv2
    x0, y0, x1, y1 = rect
    out = frame.copy()
    out[y0:y1, x0:x1] = cv2.resize(patch, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    return out


def label_mask(o: np.ndarray, f: np.ndarray, e: np.ndarray | None = None, grow: int = 9) -> np.ndarray:
    """Where the label changes (O vs F, and O vs E when given), dilated: the loss's focus. float32 WIN x WIN in [0, 1]."""
    import cv2
    d = np.abs(f.astype(np.int16) - o.astype(np.int16)).max(2) > 24
    if e is not None:
        d |= np.abs(e.astype(np.int16) - o.astype(np.int16)).max(2) > 24
    return cv2.dilate(d.astype(np.uint8), np.ones((grow, grow), np.uint8)).astype(np.float32)


def build_model(width: int = 96):
    """A small U-Net on latents: in 16 (O) + 16 (F) + 1 (mask) channels at 32 x 32, out 16 channels (E - F), three
    levels (32, 16, 8), GroupNorm + SiLU residual blocks; 11.6M parameters at width 96, ~20M at 128."""
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, cin: int, cout: int) -> None:
            super().__init__()
            self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
            self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
            self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

        def forward(self, x):
            h = self.c1(nn.functional.silu(self.n1(x)))
            return self.skip(x) + self.c2(nn.functional.silu(self.n2(h)))

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            c1, c2, c3 = width, 2 * width, 4 * width
            self.inp = nn.Conv2d(33, c1, 3, padding=1)
            self.d1, self.d2, self.d3 = nn.Sequential(Block(c1, c1), Block(c1, c1)), nn.Sequential(Block(c1, c2), Block(c2, c2)), \
                nn.Sequential(Block(c2, c3), Block(c3, c3), Block(c3, c3))
            self.down = nn.AvgPool2d(2)
            self.u2, self.u1 = nn.Sequential(Block(c3 + c2, c2), Block(c2, c2)), nn.Sequential(Block(c2 + c1, c1), Block(c1, c1))
            self.out = nn.Sequential(nn.GroupNorm(8, c1), nn.SiLU(), nn.Conv2d(c1, 16, 3, padding=1))
            nn.init.zeros_(self.out[-1].weight)
            nn.init.zeros_(self.out[-1].bias)  # starts as the identity on F: the font write

        def forward(self, o, f, m):
            x = self.inp(torch.cat([o, f, m], 1))
            a = self.d1(x)
            b = self.d2(self.down(a))
            c = self.d3(self.down(b))
            up = nn.functional.interpolate
            b = self.u2(torch.cat([up(c, scale_factor=2.0), b], 1))
            a = self.u1(torch.cat([up(b, scale_factor=2.0), a], 1))
            return f + self.out(a)

    return Net()



def window_box(box: tuple[int, int, int, int], rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """The label's frame box in the WIN x WIN window's pixels."""
    x0, y0, x1, y1 = rect
    s = WIN / (x1 - x0)
    return (max(0, int((box[0] - x0) * s)), max(0, int((box[1] - y0) * s)),
            min(WIN, int(round((box[2] - x0) * s))), min(WIN, int(round((box[3] - y0) * s))))


def hide_label(o: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Step 39 v2: the window with the old letters painted out but the label's panel kept (textbench/write_text.py
    inpaint_ink, step 24's eraser), so the model gets the label's look from O and the new glyphs only from F (the first
    model redrew the old text: 3.4% of held-out windows read the new one, PLAN.md). box: window_box of the label."""
    from textbench.write_text import inpaint_ink
    return inpaint_ink(o, [box])
