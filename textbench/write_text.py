"""Write the new text where the old text was (step 20: put the language model's exact string into Krea's memory).
The language model of a Solaris-style system knows the string ("CART 3"); here it is drawn into the frame in place of
the old one: the old text's box is erased with the label's own background (textbench/erase.py), then the new text is
drawn in bold IBM Plex Sans, in the old text's colour, scaled so its capitals fill the old box's height, centred on
the old box (labels) or left-aligned in it (a search box being typed into).
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from textbench.erase import PAD, box_colours, erase, text_boxes
from textbench import ocr_metrics
from textbench.ocr_metrics import normalize, ocr_frame, reading_order

FONT = "assets/fonts/IBMPlexSans.ttf"


def ink_colour(frame: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """The text colour: the minority colour inside the box (textbench/erase.py box_colours)."""
    return tuple(int(c) for c in box_colours(frame, box)[1])


def font_for(cap_height: int, root: str, style: tuple[int, int] = (700, 100)) -> ImageFont.FreeTypeFont:
    """style: (weight 100-700, width 75-100) of the variable font; default bold, normal width."""
    def load(size: int) -> ImageFont.FreeTypeFont:
        f = ImageFont.truetype(f"{root}/{FONT}", size)
        f.set_variation_by_axes(list(style))
        return f
    size = max(8, cap_height)
    for _ in range(3):  # capitals are about 0.7 of the font size: correct towards the wanted cap height
        f = load(size)
        top, bottom = f.getbbox("H")[1], f.getbbox("H")[3]
        size = max(8, round(size * cap_height / max(1, bottom - top)))
    return load(size)


def match_style(frame: np.ndarray, box: tuple[int, int, int, int], text: str, span: tuple[int, int],
                cap: int, root: str) -> tuple[int, int]:
    """Step 22c: the weight and width of the font closest to Krea's own letters. Krea's kept letters (text, drawn
    between columns span, cap height cap) are measured for width and ink density (ink pixels per area: stroke
    weight); the same text is rendered at that cap height in each weight 400-700 and width 75-100, and the closest
    is chosen (relative errors added). Before this the new letters were always bold, normal width: heavier and wider
    than Krea's on the try-on mirror's SIZE badge."""
    x0, y0, x1, y1 = box
    bg, ink = (c.astype(np.float32) for c in box_colours(frame, box))
    px = frame[y0:y1, span[0]:span[1]].astype(np.float32)
    mask = np.linalg.norm(px - ink, axis=2) < np.linalg.norm(px - bg, axis=2)
    rows = np.where(mask.any(1))[0]
    if not len(rows) or not text:
        return 700, 100
    krea_w, krea_d = span[1] - span[0], mask[rows.min():rows.max() + 1].mean()
    best, err = (700, 100), float("inf")
    for weight in range(400, 701, 100):  # lighter weights read as hairlines at these sizes
        for width in (75, 87, 100):
            f = font_for(cap, root, (weight, width))
            l, t, r, b = f.getbbox(text)
            img = Image.new("L", (r - l + 4, b - t + 4))
            ImageDraw.Draw(img).text((2 - l, 2 - t), text, font=f, fill=255)
            a = np.asarray(img) > 127
            rr = np.where(a.any(1))[0]
            d = a[rr.min():rr.max() + 1].mean() if len(rr) else 0
            e = abs((r - l) - krea_w) / krea_w + abs(d - krea_d) / max(krea_d, 1e-3)
            if e < err:
                best, err = (weight, width), e
    return best


def ink_rows(frame: np.ndarray, box: tuple[int, int, int, int], xa: int, xb: int) -> tuple[int, int]:
    """Top and bottom ink rows inside the box between columns xa and xb (the height of those letters)."""
    x0, y0, x1, y1 = box
    bg, ink = (c.astype(np.float32) for c in box_colours(frame, box))
    px = frame[y0:y1, xa:xb].astype(np.float32)
    rows = np.where((np.linalg.norm(px - ink, axis=2) < np.linalg.norm(px - bg, axis=2)).any(1))[0]
    return (y0 + int(rows.min()), y0 + int(rows.max())) if len(rows) else (y0, y1)


def glyph_runs(frame: np.ndarray, box: tuple[int, int, int, int]) -> tuple[list[tuple[int, int]], int, int]:
    """Letters in a text box as column runs: pixels nearer the ink colour than the background, columns that contain
    any, grouped where they touch. Returns the runs (x start, x end, frame coordinates) and the ink's top and bottom
    rows (the glyphs' measured height)."""
    x0, y0, x1, y1 = box
    bg, ink = (c.astype(np.float32) for c in box_colours(frame, box))
    px = frame[y0:y1, x0:x1].astype(np.float32)
    mask = np.linalg.norm(px - ink, axis=2) < np.linalg.norm(px - bg, axis=2)
    cols = mask.sum(0) > max(1, int(0.04 * (y1 - y0)))
    runs, start = [], None
    for i, c in enumerate(list(cols) + [False]):
        if c and start is None:
            start = i
        elif not c and start is not None:
            # a run touching the box's side is not a letter (the box is padded around the text): it is the edge of
            # what is around the label (step 22: the white mirror screen beside the dark SIZE M badge)
            if i - start >= 2 and start > 0 and i < len(cols):
                runs.append((x0 + start, x0 + i))
            else:
                cols[start:i] = False
            start = None
    rows = np.where(mask[:, cols].any(1) if cols.any() else mask.any(1))[0]
    top, bottom = (y0 + int(rows.min()), y0 + int(rows.max())) if len(rows) else (y0, y1)
    return runs, top, bottom


def grow(height: int) -> int:
    """Erase margin around old ink, in px (odd): 5, or a tenth of the text box's height for big letters (step 24,
    salad seed 1: the anti-aliased edges of 120 px letters stayed as a grey ghost of the old text)."""
    k = max(5, height // 10)
    return k + 1 - k % 2


def word_groups(frame: np.ndarray, box: tuple[int, int, int, int],
                n_words: int) -> tuple[list[list[int]], int, int] | None:
    """The label's words as ink spans [x start, x end] (frame coordinates), and the ink's top and bottom rows; None if
    the words cannot be told apart. Word gaps: the (words - 1) widest gaps between ink runs, if they stand out (1.5x
    the widest other gap); a fixed threshold fails on condensed letters (salad seed 1: 180 px tall, the word gap only
    12 px, letter gaps 3 px). Split out of partial_words (step 24) for step 40a's per-word editor."""
    runs, top, bottom = glyph_runs(frame, box)
    if not runs:
        return None
    nb = n_words - 1
    gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
    if len(gaps) < nb:
        return None
    order = sorted(range(len(gaps)), key=lambda i: -gaps[i])
    cut, rest = sorted(order[:nb]), [gaps[i] for i in order[nb:]]
    if nb and (min(gaps[i] for i in cut) < 2 or (rest and min(gaps[i] for i in cut) < 1.5 * max(rest))):
        return None
    groups, a = [], 0
    for i in cut + [len(runs) - 1]:
        groups.append([runs[a][0], runs[i][1]])
        a = i + 1
    return groups, top, bottom


def partial_words(frame: np.ndarray, old: str, new: str, box: tuple[int, int, int, int],
                  root: str, max_w: float | None = None) -> np.ndarray | None:
    """Step 24: change only the words that differ, when letters touch and cannot be told apart (salad seed 1: Krea's
    tall condensed ITEMS 2, its letters merged, so partial() failed and the whole label was redrawn in a smaller,
    wider and lighter font). Words are found by the widest gaps between ink runs; the kept words stay as Krea drew them; the old tail is inpainted; a new tail of the same length is
    stretched into the old tail's own box (Krea's size, width and position: its digit was 1.4x the letters' height),
    otherwise drawn after the kept words at their height."""
    import cv2
    k = 0
    while k < min(len(old), len(new)) and old[k] == new[k]:
        k += 1
    kp = old[:k].rfind(" ") + 1  # the kept words, up to and including the last shared space
    if kp == 0 or not new[kp:].strip():
        return None
    n_keep = old[:kp].count(" ")
    old_tail, new_tail = old[kp:], new[kp:].strip()
    found = word_groups(frame, box, len(old.split(" ")))
    if found is None:
        return None
    groups, top, bottom = found
    cap = bottom - top + 1
    x0, y0, x1, y1 = box
    tx0, tx1 = groups[n_keep][0], groups[-1][1]
    ty0, ty1 = ink_rows(frame, box, tx0, tx1)
    bg, ink = box_colours(frame, box)
    out = frame.copy()
    xa = groups[n_keep - 1][1] + max(2, cap // 8)
    region = np.ascontiguousarray(out[y0:y1, xa:x1])
    px = region.astype(np.float32)
    m = (np.linalg.norm(px - ink.astype(np.float32), axis=2) < np.linalg.norm(px - bg.astype(np.float32), axis=2))
    out[y0:y1, xa:x1] = cv2.inpaint(region, cv2.dilate(m.astype(np.uint8), np.ones((grow(y1 - y0),) * 2, np.uint8)),
                                    5, cv2.INPAINT_TELEA)
    th = ty1 - ty0 + 1
    if len(old_tail.replace(" ", "")) == len(new_tail.replace(" ", "")):
        style = match_style(frame, box, old_tail, (tx0, tx1), th, root)
        f = font_for(th, root, style)
        l, t, r, b = f.getbbox(new_tail)
        img = Image.new("L", (r - l + 4, b - t + 4))
        ImageDraw.Draw(img).text((2 - l, 2 - t), new_tail, font=f, fill=255)
        a = np.asarray(img)
        rows, cols = np.where(a.any(1))[0], np.where(a.any(0))[0]
        a = a[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
        a = cv2.resize(a, (tx1 - tx0, th), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255
        sl = out[ty0:ty0 + th, tx0:tx1].astype(np.float32)
        out[ty0:ty0 + th, tx0:tx1] = np.clip(sl * (1 - a) + ink.astype(np.float32) * a + 0.5, 0, 255).astype(np.uint8)
        return out
    letters_top, letters_bottom = ink_rows(frame, box, groups[0][0], groups[n_keep - 1][1])
    f = font_for(letters_bottom - letters_top + 1, root)
    img = Image.fromarray(out)
    l, t, r, b = ImageDraw.Draw(img).textbbox((0, 0), new_tail, font=f)
    if max_w is not None and tx0 + (r - l) - x0 > max_w + 2 * PAD:
        return None  # longer than the panel allows: the whole-label rewrite shrinks it to fit
    ImageDraw.Draw(img).text((tx0 - l, letters_bottom - b), new_tail, font=f, fill=tuple(int(c) for c in ink))
    return np.asarray(img)


def partial(frame: np.ndarray, old: str, new: str, box: tuple[int, int, int, int], root: str,
            max_w: float | None = None) -> np.ndarray | None:
    """Step 21: change only the characters that differ. The old and new text share a prefix ("CART " in CART 2 ->
    CART 3, "RUN" in RUN -> RUNN); the letters of that prefix are left exactly as Krea drew them, the rest of the box
    is filled with the label's background and the new tail is drawn after them, at the measured height of Krea's own
    letters. Letters are found as column runs of ink (glyph_runs); if their count matches neither the old text nor its
    prefix (touching or broken letters), returns None and the caller rewrites the whole text."""
    k = 0
    while k < min(len(old), len(new)) and old[k] == new[k]:
        k += 1
    keep, tail = old[:k].replace(" ", ""), new[k:]
    if not keep or not tail.strip():
        return None
    runs, top, bottom = glyph_runs(frame, box)
    n_old = len(old.replace(" ", ""))
    if len(runs) not in (n_old, len(keep)):
        return None
    widths = [b - a for a, b in runs]
    if len(runs) == len(keep) >= 2 and widths[-1] > 1.6 * float(np.median(widths[:-1])):
        # the last "kept" letter is two merged ones (step 25c poster seed 1: Krea's 2 and 4 touched, the 4 was taken
        # as not drawn and the 5 was added after it: MAY 245); leave it to the word-level edit
        return None
        return None
    bg, ink = box_colours(frame, box)
    x0, y0, x1, y1 = box
    kept_end = runs[len(keep) - 1][1]
    gaps = [b[0] - a[1] for a, b in zip(runs[:len(keep)], runs[1:len(keep)])] or [max(2, (bottom - top) // 6)]
    letter_gap = int(np.median(gaps))
    if len(runs) == n_old and len(runs) > len(keep):
        start = runs[len(keep)][0]  # the first changed letter starts here
    else:  # Krea did not draw the old tail at all ("CART" for "CART 2"): add after the kept letters
        start = kept_end + letter_gap
    if new[k - 1:k] == " " and len(runs) == len(keep):
        start += int(1.8 * letter_gap) + (bottom - top) // 4  # the space that was missing too
    # height: of the kept letters other than symbols like "$" (step 22c: measuring the replaced letters made the N of
    # WIFI ON taller than Krea's O beside it); of the replaced ones only when no kept letter is alphanumeric
    letters = [r for r, ch in zip(runs, keep) if ch.isalnum()]
    if letters:
        top, bottom = ink_rows(frame, box, letters[0][0], letters[-1][1])
    else:
        top, bottom = ink_rows(frame, box, runs[0][0], runs[-1][1])
    cap = bottom - top + 1
    style = match_style(frame, box, keep, (runs[0][0], kept_end), cap, root)
    f = font_for(cap, root, style)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    l, t, r, b = probe.textbbox((0, 0), tail.strip(), font=f)
    if len(runs) == n_old and len(runs) > len(keep):  # replacing drawn letters: fit their width (Krea's digits are
        old_tail = old[k:].replace(" ", "")           # narrower than Plex's), scaled if the new tail is longer
        max_w = 1.3 * (runs[-1][1] - runs[len(keep)][0]) * max(1.0, len(tail.strip()) / max(1, len(old_tail)))
        while cap > 8 and r - l > max_w:
            cap -= 1
            f = font_for(cap, root, style)
            l, t, r, b = probe.textbbox((0, 0), tail.strip(), font=f)
    end = start + (r - l)
    if max_w is not None and end - box[0] > max_w + 2 * PAD:
        return None  # longer than the panel allows (step 25c combustion seed 1: MAGNESIUM cut at the label's edge)
    out = frame.copy()
    # the label grows if the new tail is longer than what it replaces (the badge is extended in its own colour)
    out[y0:y1, max(x0, x1):max(x1, end + PAD)] = bg
    # the old tail's letters are painted over from the badge around them (inpainting), so its shading and reflections
    # stay (step 22: a flat fill showed as a grey block on the try-on mirror's SIZE badge)
    import cv2
    xa, xb = kept_end + 1, min(x1, out.shape[1])
    if xb > xa:
        region = np.ascontiguousarray(out[y0:y1, xa:xb])
        px = region.astype(np.float32)
        ink_px = (np.linalg.norm(px - ink.astype(np.float32), axis=2)
                  < np.linalg.norm(px - bg.astype(np.float32), axis=2)).astype(np.uint8)
        ink_px = cv2.dilate(ink_px, np.ones((grow(y1 - y0),) * 2, np.uint8))
        out[y0:y1, xa:xb] = cv2.inpaint(region, ink_px, 5, cv2.INPAINT_TELEA)
    img = Image.fromarray(out)
    ImageDraw.Draw(img).text((start - l, bottom - b), tail.strip(), font=f, fill=tuple(int(c) for c in ink))
    return np.asarray(img)


def track_boxes(newest: np.ndarray, frame: np.ndarray, boxes: dict, search: int = 48) -> dict:
    """Step 22b: the boxes found on the newest memory frame, moved to where their area is in an older one. With a
    moving camera the 9 memory frames see the label at different places (11-17 px apart on the station board); pasting
    at one position made the label stay still on screen while the scene moved. Each box, padded by its height for
    context, is found in the older frame by template matching within +-search px."""
    import cv2
    a, b = (cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in (newest, frame))
    h, w = a.shape
    out = dict(boxes)
    for k, box in boxes.items():
        if not isinstance(box, tuple):
            continue
        x0, y0, x1, y1 = box
        p = y1 - y0
        tx0, ty0, tx1, ty1 = max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p)
        sx0, sy0 = max(0, tx0 - search), max(0, ty0 - search)
        win = b[sy0:min(h, ty1 + search), sx0:min(w, tx1 + search)]
        tpl = a[ty0:ty1, tx0:tx1]
        if win.shape[0] < tpl.shape[0] or win.shape[1] < tpl.shape[1]:
            continue
        _, _, _, (mx, my) = cv2.minMaxLoc(cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED))
        dx, dy = sx0 + mx - tx0, sy0 + my - ty0
        out[k] = (max(0, x0 + dx), max(0, y0 + dy), min(w, x1 + dx), min(h, y1 + dy))
    return out


def reads(frame: np.ndarray, box: tuple[int, int, int, int], text: str) -> bool:
    """Does OCR read text in the box (widened to the right for a longer label)? RapidOCR's text detector, and else
    its recogniser alone on the box as one line: the detector often skips a lone changed letter (SIZE L read as SIZE)."""
    x0, y0, x1, y1 = box
    crop = frame[max(0, y0 - 8):y1 + 8, max(0, x0 - 8):min(frame.shape[1], x1 + 4 * (y1 - y0))]
    if normalize(text) in normalize("".join(reading_order(ocr_frame(crop)))):
        return True
    line = frame[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):min(frame.shape[1], x1 + 2 * (y1 - y0))]
    res, _ = ocr_metrics._ocr(line[:, :, ::-1].copy(), use_det=False, use_cls=False)
    return any(normalize(text) in normalize(r[0]) for r in res or [])


def inpaint_ink(frame: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Step 24: erase the letters in each box by inpainting their pixels (nearer the ink colour than the background,
    dilated) from the label around them, so the label's shading stays."""
    import cv2
    out = frame.copy()
    for x0, y0, x1, y1 in boxes:
        bg, ink = (c.astype(np.float32) for c in box_colours(frame, (x0, y0, x1, y1)))
        region = np.ascontiguousarray(out[y0:y1, x0:x1])
        px = region.astype(np.float32)
        m = (np.linalg.norm(px - ink, axis=2) < np.linalg.norm(px - bg, axis=2)).astype(np.uint8)
        out[y0:y1, x0:x1] = cv2.inpaint(region, cv2.dilate(m, np.ones((grow(y1 - y0),) * 2, np.uint8)), 5,
                                        cv2.INPAINT_TELEA)
    return out


def write(frame: np.ndarray, pairs: list[tuple[str, str]], root: str, align: str = "center",
          boxes_by_old: dict[str, tuple[int, int, int, int]] | None = None,
          partial_edit: bool = False, fill: str = "flat",
          limit: dict[str, float] | None = None, check: bool | None = None) -> tuple[np.ndarray, dict]:
    """pairs: (old text, new text). Returns the edited frame and the boxes used per old text (to reuse on the other
    cached frames, where OCR might miss). partial_edit: change only the differing characters where possible.
    fill: how a whole label is erased before the new text is drawn: "flat" (its background colour, steps 20-23) or
    "inpaint" (only the old letters, painted over from the label around them; step 24, no pale block).
    limit: the widest a new text may be, per old text (step 24: the width of the panel it is printed on)."""
    boxes = dict(boxes_by_old or {})
    # the newest frame decides; the other cached frames reuse its boxes and choice (check: force the OCR check of a
    # partial edit even when boxes are given)
    first_frame = boxes_by_old is None if check is None else check
    for old, _ in pairs:
        if old not in boxes:
            found = text_boxes(frame, [old])
            if found:
                main = max(found, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                h = main[3] - main[1]
                # a label split over lines ("WIFI" above "OFF"): join the pieces stacked within two line heights
                parts = [b for b in found if abs((b[1] + b[3]) / 2 - (main[1] + main[3]) / 2) < 2 * h
                         and min(b[2], main[2]) > max(b[0], main[0])]
                boxes[old] = (min(b[0] for b in parts), min(b[1] for b in parts),
                              max(b[2] for b in parts), max(b[3] for b in parts))
                boxes[f"_lines:{old}"] = len(parts)
    if partial_edit:
        rest = []
        for old, new in pairs:
            key = f"_partial:{old}"
            edited = None
            if old in boxes and boxes.get(f"_lines:{old}", 1) == 1 and boxes.get(key, True):
                # step 31: a partial edit starts where the old text starts, so its own limit can be tighter than a centred rewrite's
                lim = (limit or {}).get(f"_partial:{old}", (limit or {}).get(old))
                edited = partial(frame, old, new, boxes[old], root, lim)
                if edited is None:  # step 24: letters touch; change whole words only
                    edited = partial_words(frame, old, new, boxes[old], root, lim)
                if edited is not None and first_frame and not reads(edited, boxes[old], new):
                    edited = None  # keep a partial edit only if OCR reads the new text there
                boxes[key] = edited is not None
            if edited is None:
                rest.append((old, new))
            else:
                frame = edited
        pairs = rest
        if not pairs:
            return frame, boxes
    colours = {old: ink_colour(frame, boxes[old]) for old, _ in pairs if old in boxes}  # helper keys are ignored
    out = (erase if fill == "flat" else inpaint_ink)(frame, [boxes[o] for o, _ in pairs if o in boxes])
    img = Image.fromarray(out)
    d = ImageDraw.Draw(img)
    for old, new in pairs:
        if old not in boxes:
            continue
        x0, y0, x1, y1 = boxes[old]
        cap = int((y1 - y0 - 2 * PAD) * 0.92)
        style = (700, 100)
        if boxes.get(f"_lines:{old}", 1) == 1:  # step 22c: Krea's weight and width, measured on the old text
            runs, top, bottom = glyph_runs(frame, boxes[old])
            if runs:
                style = match_style(frame, boxes[old], old, (runs[0][0], runs[-1][1]), bottom - top + 1, root)
        f = font_for(cap, root, style)
        # no wider than the old text, scaled by the length change (HOME -> PROFILE may grow; INBOX 3 -> 4 may not)
        max_w = (x1 - x0 - 2 * PAD) * max(1.0, len(new) / max(1, len(old)))
        max_w = min(max_w, (limit or {}).get(old, max_w))
        while cap > 8 and d.textbbox((0, 0), new, font=f)[2] - d.textbbox((0, 0), new, font=f)[0] > max_w:
            cap -= 1
            f = font_for(cap, root, style)
        l, t, r, b = d.textbbox((0, 0), new, font=f)
        cy = (y0 + y1) / 2
        # "right" (LIVE.md, 15 Nov: a calculator display) ends the text at the box's right edge
        x = ((x0 + x1) / 2 - (r - l) / 2 - l if align == "center"
             else x1 - PAD - r if align == "right" else x0 + PAD - l)
        span = boxes.get(f"_span:{old}")
        if span is not None:  # step 31: the text stays between the panel's edges, even if the old text sat off centre
            x = min(max(x, span[0] - l), max(span[0], span[1] - (r - l)) - l)
        d.text((x, cy - (b + t) / 2), new, font=f, fill=colours[old])
    return np.asarray(img), boxes
