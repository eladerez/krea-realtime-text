"""LIVE.md 19 Nov (Elad: "make a movie all all the scenes where you change the text and digits on them"; "This movie
will be in the post"): one reel of many scenes, each a stretch of a scripted replay (<run>/<arm>-s<seed>.mp4 + .jsonl,
runner/gen_krea.py live_replay) or of a live session (scripts/live_cut.py output, an .mp4 with its own bar), between a
title card and a closing card (scripts/post_cut.py card()). Replay frames get live_cut's layout: a bar above with the
text typed so far and the scene's name, a footer note; frames are Krea's own (the typed text reaches the model through
the prompt and its memory).
Usage: python scripts/scene_reel.py --plan plan.json --out research/demo/reel.mp4
plan.json: {"title": [...], "end": [...], "note": "...", "title_s": 3, "end_s": 5,
            "clips": [{"replay": "<run>/<subdir>/<arm>-s101", "frames": "120-200", "title": "Gas pump"},
                      {"mp4": "research/demo/live_door.mp4", "frames": "0-160"}]}
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
sys.path.insert(0, str(ROOT / "scripts"))
from common import FPS  # noqa: E402
from post_cut import card  # noqa: E402

BOLD = str(ROOT / "assets/fonts/DejaVuSans-Bold.ttf")
INK, BG, MUTED, ACCENT = (240, 240, 236), (18, 19, 22), (150, 152, 158), (255, 196, 61)
BAR, FOOT = 80, 34


def typed_per_frame(jsonl: Path) -> list[str]:
    out: list[str] = []
    for line in jsonl.read_text().splitlines():
        if line.strip():
            b = json.loads(line)
            out += [b["typed"]] * int(b.get("n", b.get("n_frames", 12)))
    return out


def decorate(img: np.ndarray, typed: str, title: str, note: str) -> np.ndarray:
    h, w = img.shape[:2]
    can = Image.new("RGB", (w, h + BAR + FOOT), BG)
    can.paste(Image.fromarray(img), (0, BAR))
    d = ImageDraw.Draw(can)
    f_big, f_small = ImageFont.truetype(BOLD, 40), ImageFont.truetype(BOLD, 16)
    d.text((16, 10), "TYPED", font=f_small, fill=MUTED)
    d.text((16, 28), typed, font=f_big, fill=ACCENT)
    if title:
        d.text((w - 16 - d.textlength(title, font=f_small), 40), title, font=f_small, fill=INK)
    d.text((16, h + BAR + 8), note, font=f_small, fill=MUTED)
    return np.asarray(can)


def span(s: str, n: int) -> tuple[int, int]:
    if not s:
        return 0, n - 1
    a, b = s.split("-")
    return int(a), min(int(b), n - 1)


def write(frames: list[np.ndarray], out: Path) -> None:
    with av.open(str(out), mode="w") as c:
        st = c.add_stream("libx264", rate=FPS)
        st.height, st.width = frames[0].shape[0] - frames[0].shape[0] % 2, frames[0].shape[1] - frames[0].shape[1] % 2
        st.pix_fmt, st.options = "yuv420p", {"crf": "18"}
        for fr in frames:
            fr = np.ascontiguousarray(fr[: st.height, : st.width])
            for p in st.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                c.mux(p)
        for p in st.encode():
            c.mux(p)


def read(mp4: Path) -> list[np.ndarray]:
    with av.open(str(mp4)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    plan = json.loads(a.plan.read_text())
    note = plan.get("note", "every frame drawn by Krea Realtime 14B; the typed text reaches it through the prompt and "
                            "its memory")
    tmp = Path(tempfile.mkdtemp(prefix="scene_reel_", dir=str(a.out.parent.resolve())))
    parts, size = [], None
    for i, cl in enumerate(plan["clips"]):
        if "replay" in cl:
            src = ROOT / cl["replay"] if not Path(cl["replay"]).is_absolute() else Path(cl["replay"])
            fr = read(src.with_suffix(".mp4"))
            ty = typed_per_frame(src.with_suffix(".jsonl"))
            lo, hi = span(cl.get("frames", ""), len(fr))
            out = [decorate(fr[k], ty[min(k, len(ty) - 1)], cl.get("title", ""), note) for k in range(lo, hi + 1)]
        else:
            fr = read(ROOT / cl["mp4"])
            lo, hi = span(cl.get("frames", ""), len(fr))
            out = fr[lo:hi + 1]
        size = size or (out[0].shape[1], out[0].shape[0])
        if (out[0].shape[1], out[0].shape[0]) != size:  # one size for the concat: pad or crop to the first clip's
            out = [np.asarray(Image.fromarray(x).resize(size)) for x in out]
        p = tmp / f"clip{i:02d}.mp4"
        write(out, p)
        parts.append(p)
        print(f"clip {i}: {cl.get('title', cl.get('mp4'))} {len(out)} frames")
    w, h = size[0] - size[0] % 2, size[1] - size[1] % 2
    t, e = tmp / "title.mp4", tmp / "end.mp4"
    card(plan["title"], (w, h), plan.get("title_s", 3.0), t)
    card(plan["end"], (w, h), plan.get("end_s", 5.0), e)
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in [t, *parts, e]))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst.resolve()),
                    "-c", "copy", str(a.out.resolve())], check=True, cwd=tmp)
    print(f"wrote {a.out} ({len(parts)} clips) from {tmp}")


if __name__ == "__main__":
    main()
