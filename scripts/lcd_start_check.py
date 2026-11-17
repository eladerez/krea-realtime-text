"""LIVE.md 17 Nov (option 1, new scenes): can the 7-segment pipeline run on a start image? As runner/gen_krea.py sets
it up: OCR's box of the start number (live_server.LabelBox), the LCD found and tracked (write_lcd.LcdTracker), the
digit layout from the old digits inside OCR's box; then test numbers written (write_lcd.write_lcd) and read back by
the 7-segment reader and OCR. A start image is usable when every test write reads back right by the 7-segment reader.
Writes a sheet (start with the LCD mask and digit places marked, then the writes) and prints one line per image.
Usage: uv run --with fastapi python scripts/lcd_start_check.py data/start_frames_newscenes/*-s?.png --out sheet.png
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "runner")]
from live_server import LabelBox  # noqa: E402
from textbench.display_check import read_display  # noqa: E402
from textbench.seg7 import MaskCache, read as seg7_read  # noqa: E402
from textbench.write_lcd import LcdTracker, layout, write_lcd  # noqa: E402

TESTS = ("23456", "907", "18")


def check(path: Path, text: str, n_chars: int = 5, tests=TESTS) -> tuple[bool, str, list[np.ndarray]]:
    im = np.asarray(Image.open(path).convert("RGB"))
    box = LabelBox(n_chars, "right").update(im, text)
    r0 = read_display(im, box)
    tr = LcdTracker(im, box)
    g = tr.geometry(im, old_digits=len(text), digits_box=r0["box"])
    if g is None:
        return False, f"OCR {r0['reads']!r}: no geometry", [im]
    lay = layout(g, str(ROOT), n_chars)
    masks = MaskCache(lay)
    vis = im.copy()
    wx0, wy0, wx1, wy1 = g["window"]
    ov = vis[wy0:wy1, wx0:wx1]
    ov[g["filled"]] = (0.6 * ov[g["filled"]] + 0.4 * np.array([0, 120, 255])).astype(np.uint8)
    for pl in masks.for_geo(g):
        for seg in pl.values():
            ov[seg.astype(bool)] = (0.5 * ov[seg.astype(bool)] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
    outs, reads = [vis], []
    for want in tests:
        out = write_lcd(im, None, want, str(ROOT), geo=g, lay=lay)
        g2 = tr.geometry(out, affine=tr.affine)
        s7 = seg7_read(g2["dark"], masks.for_geo(g2))["reads"] if g2 else None
        reads.append((want, s7, read_display(out, box)["reads"]))
        outs.append(out)
    ok = all(w == s for w, s, _ in reads)
    start7 = seg7_read(g["dark"], masks.for_geo(g))["reads"]
    return ok, (f"OCR {r0['reads']!r} box {r0['box']} tol {tr.tol} angle {g['angle']:.1f} pitch "
                f"{g['pitch'] and round(g['pitch'], 1)} 7-seg on start {start7!r} writes "
                + " ".join(f"{w}->{s}/{o}" for w, s, o in reads)), outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", type=Path, nargs="+")
    ap.add_argument("--text", default="68")
    ap.add_argument("--max-chars", type=int, default=5, help="digits the display holds (live_max_chars)")
    ap.add_argument("--tests", default=",".join(TESTS))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    for p in a.images:
        try:
            ok, msg, outs = check(p, a.text, a.max_chars, a.tests.split(","))
        except Exception as ex:
            ok, msg, outs = False, f"{type(ex).__name__}: {ex}"[:160], [np.asarray(Image.open(p).convert("RGB"))]
            traceback.print_exc(limit=1)
        print(f"{'USABLE' if ok else 'no    '} {p.name}: {msg}", flush=True)
        tiles = [Image.fromarray(o).resize((416, 240)) for o in outs] + [Image.new("RGB", (416, 240))] * (4 - len(outs))
        row = Image.new("RGB", (4 * 416, 262), "white")
        ImageDraw.Draw(row).text((4, 4), f"{'USABLE' if ok else 'no'}  {p.name}  {msg[:200]}", fill="black")
        for i, t in enumerate(tiles):
            row.paste(t, (i * 416, 22))
        rows.append(np.asarray(row))
    Image.fromarray(np.concatenate(rows, 0)).save(a.out)


if __name__ == "__main__":
    main()
