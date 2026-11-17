"""The LoRA pilot's sample sheet for Elad (research/plans/LORA_PILOT.md: targets approved by eye before the training
spend): random before/after pairs saved by runner/gen_krea.py live_collect (<name>_f<i>_<how>.jpg: Krea's frame | the
target), display crops, labelled with the number shown and the repair's source.
Usage: uv run python scripts/lora_sample_sheet.py --run logs/live/textbench-run-20261116-1313 --n-krea 16 --n-klein 8"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--n-krea", type=int, default=16)
    ap.add_argument("--n-klein", type=int, default=8)
    a = ap.parse_args()
    shown = {}
    for f in a.run.glob("replay_L_collect-s*.jsonl"):
        sd = f.stem.split("-s")[-1]
        for l in f.read_text().splitlines():
            r = json.loads(l)
            c = r.get("collect") or {}
            if c.get("saved"):
                shown[c["name"]] = (sd, r["shown"], c["right"])
    jpgs = sorted((a.run / "lora_samples").glob("*.jpg"))
    rng = random.Random(16)
    krea = [j for j in jpgs if "fromframe" in j.name]
    klein = [j for j in jpgs if j.name.endswith("_klein.jpg")]
    pick = rng.sample(krea, min(a.n_krea, len(krea))) + rng.sample(klein, min(a.n_klein, len(klein)))
    tiles = []
    for j in pick:
        name = "_".join(j.stem.split("_")[:3])
        sd, num, right = shown.get(name, ("?", "?", "?"))
        im = np.asarray(Image.open(j).convert("RGB"))
        w = im.shape[1] // 2
        pair = np.concatenate([im[150:370, 250:720], im[150:370, w + 250:w + 720]], 1)
        t = Image.fromarray(pair)
        d = ImageDraw.Draw(t)
        d.rectangle((0, 0, t.width, 16), fill=(0, 0, 0))
        src = "klein" if "klein" in j.name else "Krea's own frame " + j.stem.split("fromframe")[-1]
        d.text((4, 2), f"seed {sd}  typed {num}  ({right}/12 frames right as drawn)   left: Krea   right: target (from {src})",
               fill=(255, 255, 255))
        tiles.append(np.asarray(t))
    half = (len(tiles) + 1) // 2
    for k, part in enumerate((tiles[:half], tiles[half:])):
        cols = [np.concatenate(part[i::2], 0) for i in range(2)] if len(part) > 1 else [part[0]]
        h = max(c.shape[0] for c in cols)
        cols = [np.pad(c, ((0, h - c.shape[0]), (0, 0), (0, 0))) for c in cols]
        Image.fromarray(np.concatenate(cols, 1)).save(ROOT / f"research/lcd/lora_targets_sheet_{k + 1}.png")
    print(len(krea), "Krea-sourced pairs,", len(klein), "klein pairs; sheet of", len(tiles))


if __name__ == "__main__":
    main()
