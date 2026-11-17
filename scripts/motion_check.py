"""LIVE.md 17 Nov (Elad: "The camera is not moving for the lora results"): how much a replay's video moves.
Per video: the median frame-to-frame change and the median change over 12 frames (mean absolute grey difference at
208x120), and the camera's path within each reset segment (blocks 1-12 of each 13: a similarity transform by ORB
matches from the segment's first full block to its last, as dx, dy in pixels and zoom). Prints one line per video and
the mean per arm; writes --out JSON.
Usage: python scripts/motion_check.py --run logs/live/<launch>/<subdir> --arms replay_M_base_taxi,replay_M_lora2_taxi
       --out data/scores/motion_<name>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np


def similarity(a: np.ndarray, b: np.ndarray):
    orb = cv2.ORB_create(2000)
    k1, d1 = orb.detectAndCompute(a, None)
    k2, d2 = orb.detectAndCompute(b, None)
    if d1 is None or d2 is None:
        return None
    m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2)
    if len(m) < 15:
        return None
    src = np.float32([k1[x.queryIdx].pt for x in m])
    dst = np.float32([k2[x.trainIdx].pt for x in m])
    a_, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC)
    if a_ is None or inl.sum() < 15:
        return None
    return {"dx": float(a_[0, 2]), "dy": float(a_[1, 2]), "zoom": float(np.hypot(a_[0, 0], a_[1, 0])), "inliers": int(inl.sum())}


def video(mp4: Path, jsonl: Path, reset: int = 13) -> dict:
    blocks = [json.loads(x) for x in jsonl.read_text().splitlines()]
    starts = np.cumsum([0] + [b["n"] for b in blocks])  # first frame of each block
    grey, small = [], []
    with av.open(str(mp4)) as c:
        for f in c.decode(video=0):
            g = cv2.cvtColor(f.to_ndarray(format="rgb24"), cv2.COLOR_RGB2GRAY)
            grey.append(g)
            small.append(cv2.resize(g, (208, 120), interpolation=cv2.INTER_AREA).astype(np.float32))
    d1 = [np.abs(small[i + 1] - small[i]).mean() for i in range(len(small) - 1)]
    d12 = [np.abs(small[i + 12] - small[i]).mean() for i in range(0, len(small) - 12, 6)]
    paths = []
    for s0 in range(0, len(blocks), reset):  # each segment: its first full block's first frame -> its last block's last
        b0, b1 = s0 + 1, min(s0 + reset, len(blocks)) - 1
        if b1 <= b0 or starts[b1 + 1] > len(grey):
            continue
        p = similarity(grey[starts[b0]], grey[starts[b1 + 1] - 1])
        if p:
            paths.append({"blocks": [b0, b1], **{k: round(v, 3) if isinstance(v, float) else v for k, v in p.items()}})
    return {"frame_to_frame": round(float(np.median(d1)), 3), "over_12_frames": round(float(np.median(d12)), 3),
            "segments": paths}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {}
    for arm in a.arms.split(","):
        vids = {}
        for mp4 in sorted(a.run.glob(f"{arm}-s*.mp4")):
            if mp4.name.endswith(".part.mp4") or not mp4.with_suffix(".jsonl").exists():
                continue
            v = video(mp4, mp4.with_suffix(".jsonl"))
            vids[mp4.stem] = v
            seg = v["segments"]
            print(f"{mp4.stem:34s} frame {v['frame_to_frame']:.2f}  12 frames {v['over_12_frames']:.1f}  segments "
                  + " ".join(f"(dx {s['dx']:.0f}, zoom {s['zoom']:.2f})" for s in seg), flush=True)
        if vids:
            zs = [s["zoom"] for v in vids.values() for s in v["segments"]]
            res[arm] = {"videos": vids, "frame_to_frame": round(float(np.mean([v["frame_to_frame"] for v in vids.values()])), 3),
                        "over_12_frames": round(float(np.mean([v["over_12_frames"] for v in vids.values()])), 3),
                        "zoom_mean": round(float(np.mean(zs)), 3) if zs else None,
                        "abs_dx_mean": round(float(np.mean([abs(s["dx"]) for v in vids.values() for s in v["segments"]])), 1) if zs else None}
            r = res[arm]
            print(f"  {arm}: frame {r['frame_to_frame']:.2f}  12 frames {r['over_12_frames']:.1f}  zoom per segment "
                  f"{r['zoom_mean']}  |dx| per segment {r['abs_dx_mean']}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
