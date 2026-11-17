"""LIVE.md 17 Nov (route 1, step 2): score replays of any text scene (live_writer "paste" arms). As runner/gen_krea.py
sets the writer up: the field (live_text) found by OCR on the start image inside its line (live_line), the line tracked
by an ORB homography (textbench/paste_writer.TextTracker). For every `stride`-th frame: OCR's lines at the tracked
line, joined, against the line expected (live_line with the field set to the block's shown text). Also the share of
frames after the first switch that still read the start text.
Usage: uv run --with fastapi python scripts/score_text.py --run logs/live/<launch>/<subdir> --config <replay config>
       --out data/scores/<name>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "runner")]
from common import load_config  # noqa: E402
from textbench.paste_writer import TextTracker, find_field, norm, ocr_lines, same  # noqa: E402


def read_line(frame: np.ndarray, tracker: TextTracker) -> str | None:
    q, _ = tracker.locate(frame)
    if q is None:
        return None
    h = float(np.linalg.norm(q[3] - q[0]))
    x0, y0 = q.min(0) - np.array([h / 3, h * 0.15])  # rows above or below are other lines
    x1, y1 = q.max(0) + np.array([h / 3, h * 0.15])
    got = sorted((l[:, 0].min(), t) for l, t, _ in ocr_lines(frame)
                 if x0 <= l[:, 0].mean() <= x1 and y0 <= l[:, 1].mean() <= y1)
    return "".join(norm(t) for _, t in got)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()
    rep = load_config(a.config)["live_replay"]
    arm0 = load_config(rep["arms"][0])
    start = np.asarray(Image.open(ROOT / arm0["start_frame"] / f"{arm0['start_id']}-s{rep['start_seed']}.png").convert("RGB"))
    line, fld = arm0.get("live_line", arm0["live_text"]), arm0["live_text"]
    hit = find_field(start, line, fld)
    if hit is None:
        raise SystemExit(f"{line!r} not found on the start image")
    whole = "line_quad" in hit
    q = hit["line_quad"] if whole else hit["quad"]
    tmpl = line.upper().replace(fld.upper(), "{text}", 1) if whole else "{text}"
    res: dict = {"line": line, "field": fld, "tmpl": tmpl, "arms": {}}
    for arm in rep["arms"]:
        per_seed = {}
        for sd in rep["seeds"]:
            mp4, log = a.run / f"{arm}-s{sd}.mp4", a.run / f"{arm}-s{sd}.jsonl"
            if not mp4.exists() or not log.exists():
                continue
            blocks = [json.loads(x) for x in log.read_text().splitlines()]
            owner = [i for i, b in enumerate(blocks) for _ in range(b["n"])]
            tracker = TextTracker(start, q)
            rows, ok, ok_typed, held, n, n_after, lost = [], 0, 0, 0, 0, 0, 0
            with av.open(str(mp4)) as c:
                for k, f in enumerate(c.decode(video=0)):
                    if k % a.stride or k >= len(owner):
                        continue
                    b = blocks[owner[k]]
                    want = tmpl.format(text=str(b["shown"]).upper())
                    read = read_line(f.to_ndarray(format="rgb24"), tracker)
                    lost += read is None
                    read = read or ""
                    ok += same(read, want)
                    ok_typed += same(read, tmpl.format(text=str(b["typed"]).upper()))  # r1: a failed switch keeps shown
                    n += 1
                    if b["shown"] != fld:
                        n_after += 1
                        held += same(read, tmpl.format(text=fld.upper()))
                    rows.append({"frame": k, "block": owner[k], "want": norm(want), "read": read})
            bs = sorted(b["block_s"] for b in blocks)
            locks = [b["lock"] for b in blocks if isinstance(b.get("lock"), dict)]
            per_seed[sd] = {"frames_read": n, "right": round(ok / max(n, 1), 3),
                            "right_vs_typed": round(ok_typed / max(n, 1), 3), "tracker_lost": lost,
                            "blocks_typed_not_shown": sum(b["typed"] != b["shown"] for b in blocks),
                            "memory_rewrites": sum(bool(l.get("memory_fixed")) for l in locks),
                            "lock_lost": sum((l.get("inliers") or 0) <= 0 for l in locks),
                            "reads_start_text": round(held / max(n_after, 1), 3),
                            "block_s_median": bs[len(bs) // 2], "rows": rows}
        if per_seed:
            res["arms"][arm] = {"seeds": per_seed,
                                "right": round(float(np.mean([v["right"] for v in per_seed.values()])), 3),
                                "reads_start_text": round(float(np.mean([v["reads_start_text"] for v in per_seed.values()])), 3)}
            r = res["arms"][arm]
            print(f"{arm:32s} right {r['right']:.0%}  still the start text {r['reads_start_text']:.0%}  "
                  f"per seed {[(s, v['right'], v['right_vs_typed'], v['blocks_typed_not_shown'], v['memory_rewrites'], v['tracker_lost']) for s, v in per_seed.items()]}"
                  "  (right, vs typed, blocks typed!=shown, lock rewrites, tracker lost)", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
