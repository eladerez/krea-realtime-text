"""Score the lock vs pin replay (LIVE.md, 15 Nov 2026; runner/configs.yaml replay_calc): for each arm and seed, every
`stride`-th frame of its mp4 is read by two readers and compared with the number that should be on the display:
- OCR (textbench/display_check.py read_display): independent of the 7-segment writer, but it misreads its slanted
  LCD digits at times (offline: "23456" as "952"), so it undercounts the LCD arms;
- the 7-segment reader (textbench/seg7.py) at the start image's digit places, the LCD tracked in each frame: made for
  these digits, but it shares the writer's layout, so it is not independent.
"Should show" is the number the loop applied (the log's `shown`); also scored against `typed` (code-replay-r1 #2: a
failed write keeps `shown` at the old number, which Krea keeps drawing), with failed writes counted per arm. For the
lock arm, how often Krea alone was right before the lock (code-replay-r1 #1: the lock rewrites what reads wrong, so
its own 7-segment score is high by design): the last frame's raw reading per block and the share of frames rewritten.
Per arm: the share of read frames that match by each reader (mean over seeds, and each seed), the median block time,
the median lock time. Also a 2x2 clip of the four arms (seed `--clip-seed`) with labels.
Usage: uv run python scripts/score_replay.py --run logs/live/<launch> --out data/scores/replay_calc.json
"""
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
from common import load_config, write_mp4  # noqa: E402
from textbench.display_check import digits, follow, read_display  # noqa: E402
from textbench.seg7 import MaskCache, read as seg7_read  # noqa: E402
from textbench.write_lcd import LcdTracker, layout  # noqa: E402

LABEL = {"replay_A_klein": "A  klein (as today)", "replay_B_lcd": "B  7-seg writer + anchor",
         "replay_C_lock": "C  B + per-block lock", "replay_D_pin": "D  B + latent pin",
         "replay_E_memlock": "E  memory-only lock", "replay_E2_memalways": "E2 memory rewritten every block",
         "replay_K_klein": "K  E + klein-restyled memory", "replay_C_push": "C  push-in camera",
         "replay_V_base": "V  base Krea (= E), held-out", "replay_V_lora": "V  Krea + LoRA, held-out",
         "replay_V_lora_merged": "V  Krea + LoRA merged (bf16)",
         "replay_V_lora2": "V  Krea + LoRA v2 merged"}
LABEL.update({f"replay_P_{a}_{k}": f"P  {'base' if a == 'base' else 'v2 merged'} {k} (prompt-only)" for a in ("base", "lora2")
              for k in ("calc_s0", "kitchen_scale", "bathroom_scale", "multimeter", "timer")})


def restyle_stats(blocks: list[dict]) -> dict | None:
    """code-klein-r1 should-fix 3: the klein restyle calls (runner/gen_krea.py KreaLive.restyle) of a K run: how many,
    how often klein's answer was kept (OCR read the number), errors, klein and OCR time, and klein-free block time."""
    calls = [c for b in blocks for c in b.get("restyle", [])]
    if not calls:
        return None
    kept = [c for c in calls if c.get("how", "font").startswith("klein")]
    plain = sorted(b["block_s"] for b in blocks if not b.get("restyle"))
    return {"calls": len(calls), "klein_kept": len(kept), "kept_share": round(len(kept) / len(calls), 3),
            "kept_seed1": sum(c["how"] == "klein s1" for c in kept),
            "errors": sum(bool(c.get("error")) for c in calls), "skipped": sum(bool(c.get("skipped")) for c in calls),
            "klein_s_total": round(sum(c.get("klein_s", 0) for c in calls), 1),
            "ocr_s_total": round(sum(c.get("ocr_s", 0) for c in calls), 1),
            "block_s_median_without_restyle": plain[len(plain) // 2] if plain else None}


def frames_of(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="logs/live/<launch> with <arm>-s<seed>.mp4 and .jsonl")
    ap.add_argument("--config", default="replay_calc")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--clip-seed", type=int, default=1)
    ap.add_argument("--ocr-only", action="store_true", help="LIVE.md 17 Nov (new scenes): no 7-segment reader (it "
                                                             "needs a found LCD); OCR's box follows the digits")
    ap.add_argument("--arms", default="", help="comma-separated arms instead of the config's (e.g. runs of several launches"
                                               " linked into one folder; LIVE.md 16 Nov, K vs E and C)")
    a = ap.parse_args()
    rep = load_config(a.config)["live_replay"]
    if a.arms:
        rep = {**rep, "arms": a.arms.split(",")}
    arm0 = load_config(rep["arms"][0])
    start = np.asarray(Image.open(ROOT / arm0["start_frame"] / f"{arm0['start_id']}-s{rep['start_seed']}.png").convert("RGB"))
    from live_server import LabelBox
    box = LabelBox(arm0.get("live_max_chars", 5), arm0.get("live_align", "right")).update(start, arm0["live_text"])
    tracker = masks = None
    if not a.ocr_only:
        top = arm0.get("live_track_max_scale")  # LIVE.md (16 Nov, push-in): the arm's tracker scales
        tracker = LcdTracker(start, box, **({"scales": tuple(np.round(np.arange(0.8, top + 0.01, 0.04), 2))} if top else {}))
        # as runner/gen_krea.py (LIVE.md 16 Nov, start images s0, s2): the old digits only inside OCR's box of them
        lay0 = layout(tracker.geometry(start, old_digits=len(arm0["live_text"]), digits_box=read_display(start, box)["box"]),
                      str(ROOT), arm0.get("live_max_chars", 5))
        masks = MaskCache(lay0)
    res: dict = {"arms": {}, "stride": a.stride, "box": box}
    clips: dict = {}
    for arm in rep["arms"]:
        per_seed = {}
        for sd in rep["seeds"]:
            mp4, log = a.run / f"{arm}-s{sd}.mp4", a.run / f"{arm}-s{sd}.jsonl"
            if not mp4.exists():
                continue
            blocks = [json.loads(l) for l in log.read_text().splitlines()]
            if tracker is not None:
                tracker.reset()  # each run starts from the start image's placement
            run_box = box  # --ocr-only: OCR's box follows the digits it finds (capped at 120 px from the start's)
            frames = frames_of(mp4)
            owner = [i for i, b in enumerate(blocks) for _ in range(b["n"])]  # frame -> block
            ocr_ok = seg_ok = seg_typed = n = 0
            rows = []
            for k in range(0, min(len(frames), len(owner)), a.stride):
                want = digits(blocks[owner[k]]["shown"])
                if a.ocr_only:
                    r = read_display(frames[k], run_box)
                    o = r["reads"]
                    if r["box"]:
                        run_box = follow(run_box, r["box"], anchor=box, shape=frames[k].shape)
                    sg = ""
                else:
                    o = read_display(frames[k], box)["reads"]
                    g = tracker.geometry(frames[k])
                    sg = seg7_read(g["dark"], masks.for_geo(g))["reads"] if g else ""
                ocr_ok += o == want
                seg_ok += sg == want
                seg_typed += sg == digits(blocks[owner[k]]["typed"])
                n += 1
                rows.append({"frame": k, "block": owner[k], "want": want, "ocr": o, "seg7": sg})
            bs = sorted(b["block_s"] for b in blocks)
            ls = sorted(b.get("lock_s", 0) for b in blocks)
            locks = [b["lock"] for b in blocks if isinstance(b.get("lock"), dict) and "raw_last" in b["lock"]]
            pre = {"blocks": len(locks),
                   "raw_last_right": round(sum(l["raw_last"] == digits(b["shown"]) for l, b in
                                               zip(locks, [b for b in blocks if isinstance(b.get("lock"), dict) and "raw_last" in b["lock"]]))
                                           / max(len(locks), 1), 3),
                   "frames_rewritten": round(sum(l["frames_fixed"] for l in locks) / max(sum(b["n"] for b in blocks), 1), 3),
                   # frames the lock read wrong (rewritten or, with live_lock_frames false, shown as Krea made them)
                   "frames_read_wrong": round(sum(l.get("frames_wrong", l["frames_fixed"]) for l in locks)
                                              / max(sum(b["n"] for b in blocks), 1), 3)} if locks else None
            per_seed[sd] = {"frames_read": n, "ocr_right": round(ocr_ok / max(n, 1), 3), "seg7_right": round(seg_ok / max(n, 1), 3),
                            "seg7_right_vs_typed": round(seg_typed / max(n, 1), 3), "before_lock": pre,
                            "write_failed": sum(bool(b.get("write_failed")) for b in blocks),
                            "edit_errors": sum(bool(b.get("edit_error")) for b in blocks),
                            "block_s_median": bs[len(bs) // 2], "lock_s_median": ls[len(ls) // 2],
                            "restyle": restyle_stats(blocks), "rows": rows}
            if sd == a.clip_seed:
                clips[arm] = (frames, per_seed[sd])
        if per_seed:
            res["arms"][arm] = {"seeds": per_seed,
                                "ocr_right": round(float(np.mean([v["ocr_right"] for v in per_seed.values()])), 3),
                                "seg7_right": round(float(np.mean([v["seg7_right"] for v in per_seed.values()])), 3)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    for arm, r in res["arms"].items():
        bs = [v["block_s_median"] for v in r["seeds"].values()]
        print(f"{LABEL.get(arm, arm):28s} right by OCR {r['ocr_right']:.0%}  by 7-seg reader {r['seg7_right']:.0%}  "
              f"block {np.median(bs):.2f} s  per seed {[(s, v['ocr_right'], v['seg7_right'], v['seg7_right_vs_typed']) for s, v in r['seeds'].items()]}"
              f"  failed writes {sum(v['write_failed'] for v in r['seeds'].values())}"
              + (f"  before the lock: {[v['before_lock'] for v in r['seeds'].values()]}" if any(v['before_lock'] for v in r['seeds'].values()) else ""))
    if len(clips) >= 2:  # a grid clip, two tiles per row, each tile half size, labelled
        n = min(len(f) for f, _ in clips.values())
        tiles = []
        for arm in [a_ for a_ in rep["arms"] if a_ in clips]:
            fr, _ = clips[arm]
            lab = []
            for f in fr[:n]:
                im = Image.fromarray(f).resize((416, 240))
                ImageDraw.Draw(im).rectangle((0, 0, 250, 16), fill=(0, 0, 0))
                ImageDraw.Draw(im).text((4, 2), LABEL.get(arm, arm), fill=(255, 255, 255))
                lab.append(np.asarray(im))
            tiles.append(lab)
        if len(tiles) % 2:
            tiles.append([np.zeros_like(t) for t in tiles[0]])
        grid = [np.concatenate([np.concatenate([tiles[r][i], tiles[r + 1][i]], 1) for r in range(0, len(tiles), 2)], 0)
                for i in range(n)]
        write_mp4(np.stack(grid), a.out.with_name(a.out.stem + f"_grid_s{a.clip_seed}.mp4"))


if __name__ == "__main__":
    main()
