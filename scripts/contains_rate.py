"""LIVE.md 20 Nov (fix test): a second reading of score_text.py's rows. score_text counts a frame right only when OCR's
line at the tracked place equals the expected line; on the gas pump Krea draws its own "UNLEADED" label inside the
display beside the price, OCR joins it into the line ("UNLEADED$34"), and the frame counts wrong although the price is
right. Here a frame counts when the expected text is inside the reading (norm(want) in norm(read)) or the full-frame
fallback matched; and how many of those also hold the start text (a ghost of the old text). Reported beside
right_fallback, never instead of it.
Usage: python scripts/contains_rate.py data/scores/<a>_fb.json [...]  (prints and writes <a>_contains.json)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def norm(t: str) -> str:  # as textbench/paste_writer.norm, dots dropped as in same()
    return re.sub(r"[^0-9A-Z$%]", "", (t or "").upper())


def rates(scores: dict) -> dict:
    start = norm(scores.get("field", ""))
    out = {}
    for arm, v in scores["arms"].items():
        for sd, s in v["seeds"].items():
            rows = s["rows"]
            hit = [r for r in rows if (norm(r["want"]) and norm(r["want"]) in norm(r["read"])) or r.get("fallback")]
            ghost = [r for r in hit if start and norm(r["want"]) != start and start in norm(r["read"])]
            out[f"{arm}/s{sd}"] = {"frames": len(rows), "right_fallback": s.get("right_fallback"),
                                   "contains": round(len(hit) / max(len(rows), 1), 3),
                                   "contains_with_start_text": len(ghost)}
    return out


def main() -> None:
    for p in map(Path, sys.argv[1:]):
        r = rates(json.loads(p.read_text()))
        p.with_name(p.name.replace("_fb.json", "_contains.json")).write_text(json.dumps(r, indent=1))
        for k, v in r.items():
            print(f"{k}: exact {v['right_fallback']}, contains {v['contains']}, ghost {v['contains_with_start_text']}")


if __name__ == "__main__":
    main()
