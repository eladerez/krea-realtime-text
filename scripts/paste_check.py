"""LIVE.md 17 Nov (route 1, step 1): the general text writer (textbench/paste_writer.py) on start images of many
surfaces (prompts/fresh.jsonl, prompts/heldout.jsonl; data/start_frames_fresh, data/start_frames_heldout): find each
scene's "before" text by OCR, write its "after" text (and a longer one), read back by OCR. Prints one line per image and
the share written right; writes a sheet of crops (start | write) to --out.
Usage: uv run --with fastapi python scripts/paste_check.py --seeds 0 --out sheet.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from textbench.paste_writer import find_field, find_text, norm, ocr_lines, write  # noqa: E402

SETS = [("prompts/fresh.jsonl", "data/start_frames_fresh"), ("prompts/heldout.jsonl", "data/start_frames_heldout")]


def field(before: str, after: str) -> tuple[str, str]:
    """The changing field: the words the two texts share at the start and end are dropped ("FREE SPACES 42" ->
    "FREE SPACES 39" gives "42" -> "39"); a typed field is one line, as the calculator's number."""
    b, a = before.split(), after.split()
    while len(b) > 1 and len(a) > 1 and b[0] == a[0]:
        b, a = b[1:], a[1:]
    while len(b) > 1 and len(a) > 1 and b[-1] == a[-1]:
        b, a = b[:-1], a[:-1]
    return " ".join(b), " ".join(a)


def read_at(img: np.ndarray, quad: np.ndarray, tight: bool = False) -> str:
    """OCR's lines whose centre lies within the quad (padded by a third of its height; tight: by a tenth, for a field
    that is its own row of a sign), joined left to right."""
    h = float(np.linalg.norm(quad[3] - quad[0]))
    x0, y0 = quad.min(0) - h * (0.1 if tight else 1 / 3)
    x1, y1 = quad.max(0) + h * (0.1 if tight else 1 / 3)
    got = [(q[:, 0].min(), t) for q, t, _ in ocr_lines(img) if x0 <= q[:, 0].mean() <= x1 and y0 <= q[:, 1].mean() <= y1]
    return "".join(norm(t) for _, t in sorted(got))


def crop(img: np.ndarray, quad: np.ndarray, pad: float = 2.0) -> Image.Image:
    h = float(np.linalg.norm(quad[3] - quad[0]))
    x0, y0 = (quad.min(0) - pad * h).astype(int)
    x1, y1 = (quad.max(0) + pad * h).astype(int)
    H, W = img.shape[:2]
    c = img[max(0, y0):min(H, y1), max(0, x0):min(W, x1)]
    im = Image.fromarray(np.ascontiguousarray(c))
    return im.resize((300, max(1, int(300 * im.height / max(1, im.width)))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows, n, ok, found = [], 0, 0, 0
    for pf, d in SETS:
        for line in (ROOT / pf).read_text().splitlines():
            p = json.loads(line)
            before, after = p.get("before", [None])[0], p.get("after", [None])[0]
            if not before or not after:
                continue
            line_before = before
            before, after = field(before, after)  # only the part that changes, as a typed field
            for sd in a.seeds.split(","):
                f = ROOT / d / f"{p['id']}-s{sd}.png"
                if not f.exists():
                    continue
                n += 1
                img = np.asarray(Image.open(f).convert("RGB"))
                lines = ocr_lines(img)
                hit = find_field(img, line_before, before, lines)
                if hit is None:
                    print(f"not found  {f.name}: want {before!r}", flush=True)
                    continue
                found += 1
                line_after = line_before.upper().replace(before.upper(), after.upper(), 1)
                if "line_quad" in hit:  # the field sits inside an OCR line: the whole line is rewritten with it
                    out, info = write(img, hit["line_quad"], line_after)
                else:
                    out, info = write(img, hit["quad"], after)
                read = read_at(out, hit.get("line_quad", hit["quad"]), tight="line_quad" not in hit)
                right = (read == norm(line_after) if "line_quad" in hit else read == norm(after))
                ok += right
                print(f"{'RIGHT' if right else 'wrong'}      {f.name}: {before!r} (read {hit['read']!r}) -> {after!r} "
                      f"reads {read!r}", flush=True)
                t = Image.new("RGB", (600, 22 + crop(img, hit["quad"]).height), "white")
                ImageDraw.Draw(t).text((4, 4), f"{p['id']} s{sd}: {before} -> {after} | OCR {read}", fill="black")
                t.paste(crop(img, hit["quad"]), (0, 22))
                t.paste(crop(out, hit["quad"]), (300, 22))
                rows.append(t)
    print(f"{n} images, text found on {found}, written and read back right on {ok}", flush=True)
    W, H = 1200, 0
    cols = [rows[i::2] for i in range(2)]
    H = max(sum(r.height for r in c) for c in cols)
    sheet = Image.new("RGB", (W, H), "white")
    for ci, c in enumerate(cols):
        y = 0
        for r in c:
            sheet.paste(r, (ci * 600, y))
            y += r.height
    sheet.save(a.out)


if __name__ == "__main__":
    main()
