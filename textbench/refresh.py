"""Step 34: re-write text that has slipped. Before each block, OCR reads the newest frame in Krea's memory; every text the
screen should show that reads close to, but not exactly as, the wanted text is written again (memory write, step 20),
from what is shown to what is wanted. Shared by runner/gen_krea.py and scripts/refresh_offline.py, so the GPU run and
the offline test make the same decisions.
"""
from __future__ import annotations

from textbench.ocr_metrics import best_match, normalize

MAX_CER = 0.5  # as the OCR checker's "wrong" verdict: further off, the label is not taken as found
# OCR reads Krea's letter O as the digit 0 (held-out marquee "SH0W", museum "RO0M": re-writing the O did not change the
# reading, 26 of 87 offline re-writes); the decision treats these as the same character. The score does not.
LOOK_ALIKE = str.maketrans("01", "OI")


def shown_as(reading: str, want: str) -> str | None:
    """The reading, spaced and punctuated like want, so the writer can find the label and change only what differs.
    best_match returns a normalized run (letters and digits only, no spaces): "TEMP73" for want "TEMP 75" -> "TEMP 73".
    None when the lengths differ (a letter missing or added): no safe alignment, so no re-write."""
    if len(reading) != len(normalize(want)):
        return None
    it = iter(reading)
    return "".join(next(it) if normalize(c) else c for c in want)


def refresh_pairs(texts: list[str], wanted: list[str]) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """texts: OCR's reading of the newest memory frame. Returns (old shown, wanted) pairs to write, and counts:
    ok (reads exactly, or only O/0 or I/1 differ), far (not found or cer > MAX_CER), unaligned (found, but a letter missing or added)."""
    pairs, n = [], {"ok": 0, "far": 0, "unaligned": 0, "written": 0}
    for want in wanted:
        cer, got = best_match(texts, want)
        if cer == 0 or (got and got.translate(LOOK_ALIKE) == normalize(want).translate(LOOK_ALIKE)):
            n["ok"] += 1
        elif not got or cer > MAX_CER:
            n["far"] += 1
        elif (old := shown_as(got, want)) is None:
            n["unaligned"] += 1
        else:
            pairs.append((old, want))
            n["written"] += 1
    return pairs, n


def wanted_by_block(row: dict, blocks: int) -> dict[int, list[str]]:
    """The texts the screen should show while block b is made: the start text until the first switch, then each
    switch's new text, plus texts that never change (keep)."""
    sched = sorted(row.get("schedule", []), key=lambda x: x["block"])
    out = {}
    for b in range(blocks):
        cur = list(row.get("before", []))
        for x in sched:
            if x["block"] <= b and "after" in x:
                cur = list(x["after"])
        out[b] = cur + list(row.get("keep", []))
    return out
