"""LIVE.md 21 Nov (Elad: "Go for the before"): one side-by-side clip for the reel's opening, the same scene, typing
script and noise seed: base Krea with the typed text only in the prompt (left) and the method (right), at the reel's
frame size and bar (scripts/scene_reel.py decorate), written as an .mp4 that scene_reel takes as an {"mp4": ...} clip.
Usage: python scripts/before_after.py --before <run>/<subdir>/<arm>-s101 --after <run>/<subdir>/<arm>-s101
       --frames 30-93 --title "Airport gate" --out research/demo/before_after_gate.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from scene_reel import BAR, BOLD, FOOT, INK, MUTED, decorate, read, span, typed_per_frame, write  # noqa: E402

LEFT, RIGHT = "Krea alone: text in the prompt", "With my method"


def pair(b: np.ndarray, a: np.ndarray, left: str = LEFT, right: str = RIGHT) -> np.ndarray:
    """The two frames at half size, side by side, centred in a frame of the original size, labelled above."""
    h, w = b.shape[:2]
    can = Image.new("RGB", (w, h), (18, 19, 22))
    hw, hh = w // 2, h // 2
    y = (h - hh) // 2 + 14
    can.paste(Image.fromarray(b).resize((hw - 4, hh)), (0, y))
    can.paste(Image.fromarray(a).resize((hw - 4, hh)), (hw + 4, y))
    d = ImageDraw.Draw(can)
    f = ImageFont.truetype(BOLD, 18)
    d.text((8, y - 30), left, font=f, fill=MUTED)
    d.text((hw + 12, y - 30), right, font=f, fill=INK)
    return np.asarray(can)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--note", default="every frame drawn by Krea Realtime 14B; same scene, typing and noise seed")
    ap.add_argument("--left", default=LEFT)
    ap.add_argument("--right", default=RIGHT)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    b_src, a_src = ROOT / a.before, ROOT / a.after
    bf, af = read(b_src.with_suffix(".mp4")), read(a_src.with_suffix(".mp4"))
    tb, ta = typed_per_frame(b_src.with_suffix(".jsonl")), typed_per_frame(a_src.with_suffix(".jsonl"))
    lo, hi = span(a.frames, min(len(bf), len(af)))
    out = []
    for k in range(lo, hi + 1):
        assert tb[k] == ta[k], f"frame {k}: typed {tb[k]!r} (before) vs {ta[k]!r} (after)"
        out.append(decorate(pair(bf[k], af[k], a.left, a.right), ta[k], a.title, a.note))
    assert out[0].shape[0] == bf[0].shape[0] + BAR + FOOT
    write(out, a.out)
    print(f"wrote {a.out}: {len(out)} frames")


if __name__ == "__main__":
    main()
