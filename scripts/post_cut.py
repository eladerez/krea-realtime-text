"""LIVE.md 17 Nov (the LinkedIn movie): a post-ready cut. Stacked clips (top: base Krea, bottom: Krea + LoRA, same noise
seed and typed text; scripts/demo_clip.py) of chosen segments, between a title card and a closing card with the
numbers, all 832 wide at 16 fps, joined with ffmpeg.
Usage: python scripts/post_cut.py --plan plan.json --out research/demo/post_v2.mp4
plan.json: {"title": [...lines], "end": [...lines], "lora_name": "...", "note": "...",
            "clips": [{"base": "<run>/<arm>-s101", "lora": "<run>/<arm>-s101", "blocks": "1-12", "title": "Door sign"}]}
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runner"))
from common import FPS  # noqa: E402

BOLD = str(ROOT / "assets/fonts/DejaVuSans-Bold.ttf")
REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
INK, BG, MUTED, ACCENT = (240, 240, 236), (18, 19, 22), (150, 152, 158), (255, 196, 61)


def card(lines: list[str], size: tuple[int, int], seconds: float, out: Path) -> None:
    """lines: the first is large; a line starting with '>' is an accent figure, '~' a muted small one."""
    w, h = size
    img = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(img)
    rows = []
    for i, ln in enumerate(lines):
        if i == 0:
            rows.append((ln, ImageFont.truetype(BOLD, 44), INK, 26))
        elif ln.startswith(">"):
            rows.append((ln[1:].strip(), ImageFont.truetype(BOLD, 32), ACCENT, 18))
        elif ln.startswith("~"):
            rows.append((ln[1:].strip(), ImageFont.truetype(REG, 22), MUTED, 14))
        else:
            rows.append((ln, ImageFont.truetype(REG, 26), INK, 16))
    wrapped = []
    for text, f, col, gap in rows:  # wrap to the card's width
        words, cur = text.split(), ""
        for wd in words:
            t = (cur + " " + wd).strip()
            if d.textlength(t, font=f) > w - 96 and cur:
                wrapped.append((cur, f, col, 4))
                cur = wd
            else:
                cur = t
        wrapped.append((cur, f, col, gap))
    heights = [f.getbbox("Ag")[3] + g for _, f, _, g in wrapped]
    y = (h - sum(heights)) // 2
    for (text, f, col, _), hh in zip(wrapped, heights):
        d.text((48, y), text, font=f, fill=col)
        y += hh
    arr = np.asarray(img)
    with av.open(str(out), mode="w") as c:
        s = c.add_stream("libx264", rate=FPS)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"crf": "18"}
        for _ in range(int(seconds * FPS)):
            for p in s.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    plan = json.loads(a.plan.read_text())
    tmp = Path(tempfile.mkdtemp(prefix="post_cut_", dir=str(a.out.parent.resolve())))
    parts = []
    for i, cl in enumerate(plan["clips"]):
        p = tmp / f"clip{i}.mp4"
        subprocess.run([sys.executable, str(ROOT / "scripts/demo_clip.py"), "--base", cl["base"], "--lora", cl["lora"],
                        "--out", str(p), "--blocks", cl["blocks"], "--layout", "stack", "--title", cl["title"],
                        "--lora-name", plan["lora_name"], "--note", plan["note"]], check=True)
        parts.append(p)
    with av.open(str(parts[0])) as c:
        size = (c.streams.video[0].width, c.streams.video[0].height)
    t, e = tmp / "title.mp4", tmp / "end.mp4"
    card(plan["title"], size, plan.get("title_s", 3.0), t)
    card(plan["end"], size, plan.get("end_s", 5.0), e)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in [t, *parts, e]))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst.resolve()), "-c", "copy",
                    str(a.out.resolve())], check=True, cwd=tmp)
    print(f"wrote {a.out} ({len(parts)} clips) from {tmp}")


if __name__ == "__main__":
    main()
