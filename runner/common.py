"""Shared by the generation runners: config and prompt selection, the job list, video writing and timing lines.
Standard library + numpy, PyYAML and PyAV only, so it imports in both the Krea and the Wan environment."""
from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
FPS = 16
REF20_TARGETS = {"OPEN", "COFFEE", "BOOK NOW", "ADD TO CART"}


def load_config(name: str) -> dict:
    return yaml.safe_load((ROOT / "runner/configs.yaml").read_text())[name]


def jobs(cfg: dict, preflight: bool) -> list[tuple[dict, int]]:
    """(prompt row, seed) pairs to generate, in a fixed order."""
    rows = [json.loads(line) for line in (ROOT / cfg.get("prompt_file", "prompts/prompts.jsonl")).read_text().splitlines()]
    if preflight:
        wanted, seeds = cfg["preflight"]["prompts"], cfg["preflight"]["seeds"]
        rows = [r for r in rows if r["id"] in wanted]
    else:
        seeds = cfg["seeds"]
        if cfg.get("prompts") == "ref20":
            rows = [r for r in rows if r["target"] in REF20_TARGETS]
        if cfg.get("kinds"):
            rows = [r for r in rows if r.get("kind") in cfg["kinds"]]
    return [(r, s) for r in rows for s in seeds]


def video_path(out: Path, prompt_id: str, seed: int) -> Path:
    return out / f"{prompt_id}-s{seed}.mp4"


def write_mp4(frames: np.ndarray, path: Path, fps: int = FPS) -> None:
    """frames: uint8 (T, H, W, 3). H.264, CRF 18; written to a temp name first so a half-written file never counts."""
    tmp = path.with_suffix(".part.mp4")
    with av.open(str(tmp), mode="w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        s.options = {"crf": "18"}
        for f in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    tmp.rename(path)


def log_timing(out: Path, row: dict) -> None:
    with open(out / "timing.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
