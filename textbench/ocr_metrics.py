"""Per-frame OCR text metrics for generated videos that should show one known text.

For every frame (or every `stride`-th frame) RapidOCR finds text boxes. Boxes are put in reading order and every run
of up to 3 neighbouring boxes is a candidate reading. Matching ignores case, punctuation and spaces (OCR often drops
or adds spaces). Per frame:
  cer     character error rate of the best candidate vs the target: Levenshtein distance / target length, capped at 1
  exact   cer == 0 (the frame shows the target, readable)
  read    the best candidate string (what a viewer would read there)
Per video:
  exact_rate   share of frames where the target is readable
  mean_cer     mean of the per-frame cer
  flicker      share of consecutive scored frames whose `read` string changes (text appears, vanishes or mutates)
  persistence  after the first exact frame, share of later frames that are still exact (None if never exact)
  cer_by_sec   mean cer per second of video (decay curve), and its least-squares slope per second
  missing / wrong  the cer split by kind of error: target letters absent from the read (cropped, occluded, faded:
               the scene drifting) vs letters substituted or added (the text itself breaking, a "convincing wrong
               answer"); per frame each is an edit count / target length, reported as means

Usage: uv run python -m textbench.ocr_metrics --videos <dir of .mp4> --prompts prompts/prompts.jsonl --out scores.json
Videos must be named <prompt id>-<anything>.mp4 or <prompt id>.mp4.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from rapidfuzz.distance import Levenshtein

MAX_RUN = 3
_ocr = None


def normalize(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def ocr_frame(frame: np.ndarray) -> list[tuple[float, float, float, str]]:
    """Return (center y, left x, box height, text) for every box RapidOCR finds in an RGB frame."""
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    result, _ = _ocr(frame[:, :, ::-1].copy())  # RapidOCR expects BGR like OpenCV
    boxes = []
    for quad, text, _score in result or []:
        ys = [p[1] for p in quad]
        xs = [p[0] for p in quad]
        boxes.append((sum(ys) / 4, min(xs), max(ys) - min(ys), text))
    return boxes


def reading_order(boxes: list[tuple[float, float, float, str]]) -> list[str]:
    """Sort boxes into lines (centers closer than half a box height) and left to right within a line."""
    lines: list[list[tuple]] = []
    for b in sorted(boxes):
        if lines and abs(b[0] - lines[-1][0][0]) < 0.5 * max(b[2], lines[-1][0][2]):
            lines[-1].append(b)
        else:
            lines.append([b])
    return [b[3] for line in lines for b in sorted(line, key=lambda b: b[1])]


def best_match(texts: list[str], target: str) -> tuple[float, str]:
    """Lowest cer over every run of up to MAX_RUN consecutive texts; ('', 1.0) when nothing was read."""
    t = normalize(target)
    best = (1.0, "")
    for i in range(len(texts)):
        for j in range(i + 1, min(i + MAX_RUN, len(texts)) + 1):
            cand = normalize("".join(texts[i:j]))
            if not cand:
                continue
            cer = min(1.0, Levenshtein.distance(t, cand) / len(t))
            if cer < best[0] or (cer == best[0] and not best[1]):
                best = (cer, cand)
    return best


def verdict(texts: list[str], targets: list[str], wrong_max: float = 0.5) -> str:
    """One frame, for the OCR checker in the loop (verify-and-retry, abstain):
      correct  every target reads exactly
      wrong    some target is visible but misspelled: 0 < cer <= wrong_max (a "convincing wrong answer")
      absent   otherwise (text not on screen yet, cut off far more than a letter, or unreadable)"""
    cers = [best_match(texts, t)[0] for t in targets]
    if all(c == 0 for c in cers):
        return "correct"
    if any(0 < c <= wrong_max for c in cers):
        return "wrong"
    return "absent"


def error_split(target: str, read: str) -> tuple[float, float]:
    """(missing, wrong) for one frame: deletions vs substitutions + insertions, each / target length."""
    t = normalize(target)
    if not read:
        return 1.0, 0.0
    ops = Levenshtein.editops(t, read)
    missing = sum(op.tag == "delete" for op in ops)
    return min(1.0, missing / len(t)), min(1.0, (len(ops) - missing) / len(t))


def score_frames(frames: list[np.ndarray], target: str, fps: float, stride: int = 1) -> dict:
    idx = list(range(0, len(frames), stride))
    cers, reads = [], []
    for i in idx:
        cer, read = best_match(reading_order(ocr_frame(frames[i])), target)
        cers.append(cer)
        reads.append(read)
    return summarize(np.array(cers), reads, np.array(idx) / fps, target)


def summarize(cers: np.ndarray, reads: list[str], times: np.ndarray, target: str | None = None) -> dict:
    exact = cers == 0
    first = int(np.argmax(exact)) if exact.any() else None
    persistence = float(exact[first + 1:].mean()) if first is not None and first + 1 < len(exact) else None
    secs = np.floor(times).astype(int)
    by_sec = [float(cers[secs == s].mean()) for s in range(secs.max() + 1)]
    slope = float(np.polyfit(np.arange(len(by_sec)), by_sec, 1)[0]) if len(by_sec) > 1 else 0.0
    changes = [a != b for a, b in zip(reads, reads[1:])]
    return {
        "frames_scored": len(cers),
        "exact_rate": float(exact.mean()),
        "mean_cer": float(cers.mean()),
        "flicker": float(np.mean(changes)) if changes else 0.0,
        "persistence": persistence,
        "cer_by_sec": by_sec,
        "cer_slope_per_sec": slope,
        "cer": [round(float(c), 4) for c in cers],
        "read": reads,
        **({"mean_missing": float(np.mean([error_split(target, r)[0] for r in reads])),
            "mean_wrong": float(np.mean([error_split(target, r)[1] for r in reads]))} if target else {}),
    }


def score_frames_multi(frames: list[np.ndarray], targets: list[str], fps: float, stride: int = 1,
                       final_target: str | None = None) -> dict:
    """The hard set (prompts/hard.jsonl): several strings per frame, each scored like a single target, plus
      all_exact_rate  share of frames where every target is readable at once
      final_exact     for changing text: share of the last second's frames that show `final_target`"""
    idx = list(range(0, len(frames), stride))
    texts = [reading_order(ocr_frame(frames[i])) for i in idx]
    times = np.array(idx) / fps
    per = {}
    for t in targets:
        cers, reads = zip(*(best_match(x, t) for x in texts))
        per[t] = summarize(np.array(cers), list(reads), times, t)
    out = {"frames_scored": len(idx), "targets": {t: {k: v for k, v in s.items() if k not in ("cer", "read")}
                                                  for t, s in per.items()}}
    if per:
        exact = np.array([[c == 0 for c in s["cer"]] for s in per.values()])  # targets x frames
        out.update({
            "exact_rate": float(exact.mean()),
            "all_exact_rate": float(exact.all(axis=0).mean()),
            "mean_cer": float(np.mean([s["mean_cer"] for s in per.values()])),
            "flicker": float(np.mean([s["flicker"] for s in per.values()])),
            "mean_missing": float(np.mean([s["mean_missing"] for s in per.values()])),
            "mean_wrong": float(np.mean([s["mean_wrong"] for s in per.values()])),
            "cer_by_sec": np.mean([s["cer_by_sec"] for s in per.values()], axis=0).tolist(),
        })
    if final_target:
        last = [best_match(x, final_target)[0] == 0 for x, tm in zip(texts, times) if tm >= times[-1] - 1.0]
        out["final_exact"] = float(np.mean(last))
        out["final_read"] = [best_match(x, final_target)[1] for x in texts[-4:]]
    return out


def read_video(path: Path) -> tuple[list[np.ndarray], float]:
    fps = float(iio.immeta(path, plugin="pyav").get("fps", 16.0))
    return list(iio.imiter(path, plugin="pyav")), fps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, default=Path("prompts/prompts.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    rows = {r["id"]: r for r in map(json.loads, args.prompts.read_text().splitlines())}
    scores = json.loads(args.out.read_text()) if args.out.exists() else {}  # resumable
    for path in sorted(args.videos.glob("*.mp4")):
        if path.stem in scores:
            continue
        pid = path.stem.split("-")[0]
        if pid not in rows:
            print(f"skip {path.name}: no prompt id {pid}")
            continue
        frames, fps = read_video(path)
        r = rows[pid]
        if "targets" in r:  # hard set
            s = score_frames_multi(frames, r["targets"], fps, args.stride, r.get("final_target"))
            scores[path.stem] = {"prompt_id": pid, "kind": r["kind"], "fps": fps, **s}
            print(f"{path.stem}: exact {s.get('exact_rate', float('nan')):.2f} "
                  f"all {s.get('all_exact_rate', float('nan')):.2f} final {s.get('final_exact', '-')}")
        else:
            s = score_frames(frames, r["target"], fps, args.stride)
            scores[path.stem] = {"prompt_id": pid, "target": r["target"], "fps": fps, **s}
            print(f"{path.stem}: exact {s['exact_rate']:.2f} cer {s['mean_cer']:.3f} flicker {s['flicker']:.3f}")
        args.out.write_text(json.dumps(scores, indent=1))


if __name__ == "__main__":
    main()
