"""Offline check of textbench/lora_targets.py on a replay (no GPU, no klein): per block, how many frames read right,
how many wrong frames the transplant fixes and verifies; a sheet of wrong frames before and after.
Usage: uv run --with fastapi python scripts/lora_targets_check.py --run logs/live/textbench-run-20261116-0542 --arm replay_E_memlock --seed 1"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runner"))
from common import load_config  # noqa: E402
from textbench import lora_targets  # noqa: E402
from textbench.seg7 import MaskCache  # noqa: E402
from textbench.write_lcd import LcdTracker, layout  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--arm", default="replay_E_memlock")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    cfg = load_config(a.arm)
    start = np.asarray(Image.open(ROOT / cfg["start_frame"] / f"{cfg['start_id']}-s1.png").convert("RGB"))
    from live_server import LabelBox
    box = LabelBox(5, "right").update(start, cfg["live_text"])
    tracker = LcdTracker(start, box)
    lay0 = layout(tracker.geometry(start, old_digits=len(cfg["live_text"])), str(ROOT), 5)
    masks = MaskCache(lay0)
    blocks = [json.loads(l) for l in (a.run / f"{a.arm}-s{a.seed}.jsonl").read_text().splitlines()]
    with av.open(str(a.run / f"{a.arm}-s{a.seed}.mp4")) as c:
        fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    starts = np.cumsum([0] + [b["n"] for b in blocks])
    tot = {"blocks": 0, "frames": 0, "right": 0, "verified": 0, "all_wrong_blocks": 0, "latents_ok": 0, "latents": 0}
    tiles = []
    for bi, b in enumerate(blocks):
        if b["n"] != 12:
            continue
        frames = fr[starts[bi]:starts[bi] + 12]
        res = lora_targets.build(frames, b["shown"], tracker, masks, str(ROOT), lay0, box, editor=None)
        inf = res["info"]
        w = lora_targets.latent_weights(res["lcd"], res["verified"])
        tot["blocks"] += 1
        tot["frames"] += inf["n"]
        tot["right"] += inf["right"]
        tot["verified"] += inf["verified"]
        tot["all_wrong_blocks"] += inf["right"] == 0
        tot["latents"] += 3
        tot["latents_ok"] += sum(all(res["verified"][j * 4:(j + 1) * 4]) for j in range(3))
        for i in range(12):
            if res["how"][i].startswith("from") and len(tiles) < 12:
                pair = []
                for img, lab in ((frames[i], f"b{bi} f{i} Krea reads {inf['reads'][i]} (want {b['shown']})"),
                                 (res["frames"][i], f"target: {res['how'][i]}, verified {res['verified'][i]}")):
                    t = Image.fromarray(img[180:360, 300:680]).resize((380, 180))
                    ImageDraw.Draw(t).rectangle((0, 0, 379, 14), fill=(0, 0, 0))
                    ImageDraw.Draw(t).text((3, 1), lab, fill=(255, 255, 255))
                    pair.append(np.asarray(t))
                tiles.append(np.concatenate(pair, 1))
    print(json.dumps(tot))
    if tiles:
        Image.fromarray(np.concatenate(tiles, 0)).save(ROOT / f"research/lcd/lora_targets_{a.arm}_s{a.seed}.png")


if __name__ == "__main__":
    main()
