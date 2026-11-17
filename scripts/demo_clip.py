"""Step 4 (LIVE.md, 17 Nov 2026; Elad: "Build the LinkedIn demo on the LoRA"): a clip of base Krea and Krea + the
digit LoRA on the same noise seed and the same typed numbers, stacked (top: base, bottom: LoRA) for a phone screen,
with the number typed so far in a bar above. Frames are Krea's own (the replay arms run with live_lock_frames false:
no frame is edited; only the memory gets the typed number, as in setup E).

Usage: python scripts/demo_clip.py --base <run>/replay_V_base-s101 --lora <run>/replay_V_lora2-s101 --out demo.mp4
       [--blocks 2-41] [--layout stack|side]
(--base/--lora: path without extension; <path>.mp4 and <path>.jsonl are read)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runner"))
from common import FPS  # noqa: E402

BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
INK, BG, MUTED, ACCENT = (240, 240, 236), (18, 19, 22), (150, 152, 158), (255, 196, 61)


def frames(path: Path):
    with av.open(str(path)) as c:
        for f in c.decode(video=0):
            yield f.to_ndarray(format="rgb24")


def typed_per_frame(jsonl: Path) -> list[str]:
    out = []
    for line in jsonl.read_text().splitlines():
        b = json.loads(line)
        out += [b["typed"]] * b["n"]
    return out


def label(img: Image.Image, text: str, sub: str, xy=(14, 12)) -> None:
    d = ImageDraw.Draw(img, "RGBA")
    f, fs = ImageFont.truetype(BOLD, 20), ImageFont.truetype(REG, 15)
    w = max(d.textlength(text, font=f), d.textlength(sub, font=fs)) + 20
    d.rounded_rectangle((xy[0], xy[1], xy[0] + w, xy[1] + 52), 8, fill=(0, 0, 0, 165))
    d.text((xy[0] + 10, xy[1] + 6), text, font=f, fill=INK)
    d.text((xy[0] + 10, xy[1] + 30), sub, font=fs, fill=MUTED)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--lora", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocks", default="", help="first-last block to keep, e.g. 2-41")
    ap.add_argument("--layout", default="stack", choices=["stack", "side"])
    ap.add_argument("--lora-name", default="+ digit LoRA (ours)")
    ap.add_argument("--title", default="", help="shown at the right of the top bar (e.g. the surface)")
    ap.add_argument("--note", default="same noise seed and typed numbers; every frame drawn by Krea, none edited")
    a = ap.parse_args()
    typed = typed_per_frame(a.lora.with_suffix(".jsonl"))
    assert typed == typed_per_frame(a.base.with_suffix(".jsonl")), "the two runs must type the same script"
    blocks = [json.loads(line) for line in a.lora.with_suffix(".jsonl").read_text().splitlines()]
    owner = [b["b"] for b in blocks for _ in range(b["n"])]
    lo, hi = (int(x) for x in a.blocks.split("-")) if a.blocks else (0, 10 ** 9)
    bar_h, foot_h = 92, 40
    fmono, fsmall = ImageFont.truetype(MONO, 46), ImageFont.truetype(REG, 17)
    out = None
    with av.open(str(a.out), mode="w") as c:
        for i, (fb, fl) in enumerate(zip(frames(a.base.with_suffix(".mp4")), frames(a.lora.with_suffix(".mp4")))):
            if i >= len(owner) or not lo <= owner[i] <= hi:
                continue
            h, w = fb.shape[:2]
            tb, tl = Image.fromarray(fb), Image.fromarray(fl)
            label(tb, "Krea Realtime 14B", "as released")
            label(tl, "Krea Realtime 14B", a.lora_name)
            if a.layout == "stack":
                W, H = w, bar_h + 2 * h + 4 + foot_h
                can = Image.new("RGB", (W, H), BG)
                can.paste(tb, (0, bar_h))
                can.paste(tl, (0, bar_h + h + 4))
            else:
                W, H = 2 * w + 4, bar_h + h + foot_h
                can = Image.new("RGB", (W, H), BG)
                can.paste(tb, (0, bar_h))
                can.paste(tl, (w + 4, bar_h))
            d = ImageDraw.Draw(can)
            d.text((18, 14), "TYPED", font=fsmall, fill=MUTED)
            d.text((18, 34), typed[i], font=fmono, fill=ACCENT)
            note = a.note
            if a.title:
                ft = ImageFont.truetype(BOLD, 24)
                d.text((W - 18 - d.textlength(a.title, font=ft), 40), a.title, font=ft, fill=INK)
            d.text((18, H - foot_h + 11), note, font=fsmall, fill=MUTED)
            if out is None:
                out = c.add_stream("libx264", rate=FPS)
                out.width, out.height, out.pix_fmt = W - W % 2, H - H % 2, "yuv420p"
                out.options = {"crf": "18"}
            arr = np.asarray(can)[: H - H % 2, : W - W % 2]
            for p in out.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(arr), format="rgb24")):
                c.mux(p)
        if out is not None:
            for p in out.encode():
                c.mux(p)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
