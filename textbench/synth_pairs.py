"""Step 40a fine-tune (SPEED.md): synthetic label pairs for RS-STE. Each pair is one label drawn twice in exactly the
same style (background, panel, font, size, colour, outline, shadow, glow, bevel, slant, blur, noise, JPEG) with two
different texts, cropped to the same box around the first text: (image A, text A, image B, text B). RS-STE learns to
turn A into B given text B (its stage-2 'synth_pair' training). Many texts hold digits (RS-STE read 0 of 45 "WORD NN"
labels as published).

Backgrounds: crops of the step 39 training stills (data/swap_scenes, 43 objects in no test prompt) or flat/gradient
colour. Fonts: open fonts from the OS packages infra/startup.sh installs (FONT_DIRS), plus our IBM Plex variable font
at random weights and widths. Words: those of prompts/swap_train.jsonl and random letter strings; any word of
prompts/swap_test.jsonl is never drawn (that file is 40b's and step 39's test), nor any word of the four held-out
objects' prompts (40a's windows; Critic r1).
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIRS = ("/usr/share/fonts/opentype/urw-base35", "/usr/share/fonts/truetype/dejavu",
             "/usr/share/fonts/truetype/liberation2", "/usr/share/fonts/truetype/liberation",
             "/usr/share/fonts/truetype/noto", "/usr/share/fonts/opentype/noto", "/usr/share/fonts/truetype/ubuntu",
             "/usr/share/fonts/truetype/dseg", "/usr/share/fonts/opentype/dseg")
SKIP = re.compile(r"Dingbat|Symbol|Emoji|Math|CJK|Arabic|Hebrew|Thai|Armenian|Georgian|Ethiopic|Khmer|Lao|Sinhala|"
                   r"Tamil|Telugu|Kannada|Malayalam|Devanagari|Bengali|Gujarati|Gurmukhi|Oriya|Myanmar|Tibetan|Z003|"
                   r"Color|Mono-?CJK|D050000L|StandardSymbols|Dingbats", re.I)
OUT_H, OUT_W = 64, 256  # rendered crop size (RS-STE resizes to 32x128)
HELD_OUT_OBJECTS = ("rice_cooker", "highway_sign", "book_spine", "door_number")  # runner/train_swap.py HELD_OUT


def font_files(root: str) -> list[str]:
    files = [str(p) for d in FONT_DIRS for p in sorted(Path(d).glob("*")) if p.suffix.lower() in (".ttf", ".otf")
             and not SKIP.search(p.name)]
    return files + [f"{root}/assets/fonts/IBMPlexSans.ttf"]


def words(root: str) -> tuple[list[str], set[str]]:
    """Words of the training prompts (texts in quotes and the targets), minus every word of the test prompts."""
    def of(path: Path) -> set[str]:
        out: set[str] = set()
        for line in path.read_text().splitlines():
            for w in re.findall(r"[A-Za-z][A-Za-z']+", line):
                out.add(w.upper())
        return out
    banned = of(Path(root) / "prompts/swap_test.jsonl")
    # every word of the held-out objects' lines (Critic r1: 45 of their 57 new words were in the vocabulary)
    held = [line for line in (Path(root) / "prompts/swap_train.jsonl").read_text().splitlines()
            if any(o in line for o in HELD_OUT_OBJECTS)]
    for line in held:
        banned |= {w.upper() for w in re.findall(r"[A-Za-z][A-Za-z']+", line)}
    keep = sorted(w for w in of(Path(root) / "prompts/swap_train.jsonl") if 2 <= len(w) <= 10 and w not in banned)
    return keep, banned


class Pairs:
    def __init__(self, root: str, backgrounds: list[np.ndarray]):
        self.root, self.backgrounds = root, backgrounds
        self.fonts = font_files(root)
        self.words, self.banned = words(root)
        self.letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def word(self, rng: np.random.Generator) -> str:
        if self.words and rng.random() < 0.7:
            w = self.words[rng.integers(len(self.words))]
        else:
            w = "".join(rng.choice(list(self.letters), int(rng.integers(2, 9))))
        return w if w not in self.banned else "".join(rng.choice(list(self.letters), len(w)))

    def number(self, rng: np.random.Generator) -> str:
        return str(int(rng.integers(0, 10 ** int(rng.integers(1, 5)))))

    def text(self, rng: np.random.Generator, kind: str | None = None) -> tuple[str, str]:
        kind = kind or rng.choice(["word_num", "num", "word", "mixed", "two_words"], p=[0.3, 0.2, 0.25, 0.1, 0.15])
        if kind == "word_num":
            t = f"{self.word(rng)[:6]} {self.number(rng)}"
        elif kind == "num":
            t = self.number(rng)
        elif kind == "word":
            t = self.word(rng)
        elif kind == "mixed":
            t = rng.choice([f"{self.letters[rng.integers(26)]}{self.number(rng)}", f"${self.number(rng)}.{rng.integers(10, 100)}",
                            f"{rng.integers(1, 13)}:{rng.integers(0, 60):02d}", f"{self.number(rng)}%"])
        else:
            t = f"{self.word(rng)[:6]} {self.word(rng)[:6]}"
        case = rng.random()
        t = t if case < 0.75 else (t.title() if case < 0.9 else t.lower())
        return str(t), str(kind)

    def font(self, rng: np.random.Generator, size: int) -> ImageFont.FreeTypeFont:
        path = self.fonts[rng.integers(len(self.fonts))]
        f = ImageFont.truetype(path, size)
        if path.endswith("IBMPlexSans.ttf"):
            f.set_variation_by_axes([int(rng.integers(300, 701)), int(rng.integers(75, 101))])
        return f

    def background(self, rng: np.random.Generator, w: int, h: int) -> Image.Image:
        if self.backgrounds and rng.random() < 0.6:  # a region of a still, scaled to the canvas (stills are 832x480)
            b = self.backgrounds[rng.integers(len(self.backgrounds))]
            cw = int(rng.integers(min(160, b.shape[1]), b.shape[1] + 1))
            ch = max(8, min(b.shape[0], int(cw * h / w)))
            y, x = rng.integers(0, b.shape[0] - ch + 1), rng.integers(0, b.shape[1] - cw + 1)
            return Image.fromarray(b[y:y + ch, x:x + cw]).resize((w, h), Image.Resampling.BICUBIC)
        c0, c1 = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
        t = np.linspace(0, 1, w if rng.random() < 0.5 else h)[None, :, None] if rng.random() < 0.5 else np.zeros((1, 1, 1))
        g = (c0 * (1 - t) + c1 * t).astype(np.uint8)
        return Image.fromarray(np.broadcast_to(g if g.shape[1] == w else g.transpose(1, 0, 2), (h, w, 3)).copy())

    def pair(self, rng: np.random.Generator) -> tuple[np.ndarray, str, np.ndarray, str]:
        ta, kind = self.text(rng)
        tb, _ = self.text(rng, kind if rng.random() < 0.7 else None)
        while tb == ta:
            tb, _ = self.text(rng, kind)
        style = {k: rng.random() for k in ("panel", "outline", "shadow", "glow", "bevel", "slant", "blur", "noise", "jpeg")}
        size = int(rng.integers(40, 110))
        font = self.font(rng, size)
        ink = tuple(int(v) for v in rng.integers(0, 256, 3))
        canvas_w, canvas_h = 1400, 3 * size
        bg = self.background(rng, canvas_w, canvas_h)
        if style["panel"] < 0.4:  # a sign panel behind the text, in a colour far from the ink
            pc = tuple(int((c + 128 + rng.integers(-40, 41)) % 256) for c in ink)
            ImageDraw.Draw(bg).rounded_rectangle([0, 0, canvas_w, canvas_h], radius=int(rng.integers(0, 20)), fill=pc)
        outline = (int(rng.integers(1, max(2, size // 12))), tuple(int(v) for v in rng.integers(0, 256, 3))) \
            if style["outline"] < 0.2 else None
        shadow = (int(rng.integers(-6, 7)), int(rng.integers(1, 7))) if style["shadow"] < 0.25 else None
        glow = float(rng.uniform(2, 8)) if style["glow"] < 0.2 else None
        bevel = style["bevel"] < 0.15
        shear = float(rng.uniform(-0.25, 0.25)) if style["slant"] < 0.2 else 0.0
        blur = float(rng.uniform(0.3, 1.5)) if style["blur"] < 0.4 else 0.0
        noise = float(rng.uniform(2, 10)) if style["noise"] < 0.4 else 0.0
        jpeg = int(rng.integers(35, 95)) if style["jpeg"] < 0.5 else None

        def draw(text: str, width: int | None) -> tuple[Image.Image, tuple[int, int, int, int]]:
            l, t, r, b = font.getbbox(text)
            mask = Image.new("L", (r - l + 40, b - t + 40))
            ImageDraw.Draw(mask).text((20 - l, 20 - t), text, font=font, fill=255)
            if width is not None and mask.width != width:  # text B fills text A's box (as the editor's box)
                mask = mask.resize((width, mask.height), Image.Resampling.LANCZOS)
            if shear:
                mask = mask.transform(mask.size, Image.Transform.AFFINE, (1, shear, -shear * mask.height / 2, 0, 1, 0),
                                      Image.Resampling.BICUBIC)
            img = bg.copy()
            x, y = (canvas_w - mask.width) // 2, (canvas_h - mask.height) // 2
            m = np.zeros((canvas_h, canvas_w), np.float32)
            m[y:y + mask.height, x:x + mask.width] = np.asarray(mask, np.float32)[:canvas_h - y, :canvas_w - x] / 255
            a = np.asarray(img, np.float32)

            def put(layer: np.ndarray, colour: tuple, arr: np.ndarray) -> np.ndarray:
                return arr * (1 - layer[..., None]) + np.array(colour, np.float32) * layer[..., None]
            if shadow:
                a = put(np.roll(m, shadow, (1, 0)) * 0.6, (0, 0, 0), a)
            if glow:
                g = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(glow)), np.float32) / 255
                a = np.clip(a + g[..., None] * np.array(ink, np.float32) * 0.8, 0, 255)
            if outline:
                o = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * outline[0] + 1)),
                               np.float32) / 255
                a = put(o, outline[1], a)
            a = put(m, ink, a)
            if bevel:
                hi, lo = np.roll(m, (-1, -1), (0, 1)) - m, np.roll(m, (1, 1), (0, 1)) - m
                a = np.clip(a + 60 * np.clip(hi, 0, 1)[..., None] - 60 * np.clip(lo, 0, 1)[..., None], 0, 255)
            return Image.fromarray(a.astype(np.uint8)), (x + 20, y + 20, x + mask.width - 20, y + mask.height - 20)

        img_a, (x0, y0, x1, y1) = draw(ta, None)
        img_b, _ = draw(tb, x1 - x0 + 40)
        pad_x, pad_y = int(rng.uniform(0.05, 0.3) * (y1 - y0)), int(rng.uniform(0.05, 0.3) * (y1 - y0))
        box = (max(0, x0 - pad_x), max(0, y0 - pad_y), min(canvas_w, x1 + pad_x), min(canvas_h, y1 + pad_y))
        out, noise_seed = [], int(rng.integers(1 << 30))  # the same grain on A and B
        for img in (img_a, img_b):
            c = img.crop(box).resize((OUT_W, OUT_H), Image.Resampling.BICUBIC)
            if blur:
                c = c.filter(ImageFilter.GaussianBlur(blur))
            arr = np.asarray(c, np.float32)
            if noise:
                arr = arr + np.random.default_rng(noise_seed).normal(0, noise, arr.shape)
            c = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            if jpeg:
                buf = io.BytesIO()
                c.save(buf, "JPEG", quality=jpeg)
                c = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
            out.append(np.asarray(c))
        return out[0], ta, out[1], tb


def load_backgrounds(folder: Path, limit: int = 200) -> list[np.ndarray]:
    """The step 39 training stills at 832x480 (not the -full ones), never those of the held-out objects (40a's
    windows come from them)."""
    files = [p for p in sorted(folder.glob("*-s0.png")) if not any(o in p.name for o in HELD_OUT_OBJECTS)]
    files = files[::max(1, len(files) // limit)][:limit]  # spread over the objects
    return [np.asarray(Image.open(p).convert("RGB")) for p in files]


if __name__ == "__main__":  # a contact sheet to look at: python -m textbench.synth_pairs <out.png> [n]
    import sys
    root = str(Path(__file__).resolve().parents[1])
    gen = Pairs(root, load_backgrounds(Path(root) / "data/swap_scenes"))
    rng = np.random.default_rng(0)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    rows = [np.concatenate(gen.pair(rng)[::2], 1) for _ in range(n)]
    Image.fromarray(np.concatenate(rows, 0)).save(sys.argv[1])
    print(json.dumps({"fonts": len(gen.fonts), "words": len(gen.words), "banned": len(gen.banned)}))
