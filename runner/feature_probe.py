"""Feature steering, step 1 (LIVE.md, 16 Nov 2026; Elad: "If we know the feature vector of a certain digit ... can we
shift it to represent the 8 digit"): does Krea's transformer hold each display digit in a form a linear probe can read?

During a replay (runner/gen_krea.py live(), cfg live_probe), forward hooks on chosen transformer blocks keep their
output hidden states for the 4 denoising calls of every block (t = 1000, 909, 714, 0 for steps 4; the call before them,
the KV-cache rebuild from memory, is skipped). After the block is decoded, the lock's 7-segment reader gives each frame's
LCD placement and the reading of each digit place; for each latent frame (4 video frames) and digit place, the hidden
states of the tokens under the place's box (16 x 16 px each, weighted by overlap) are averaged into one vector per layer
and call. Saved per arm and seed: <arm>-s<seed>_probe.npz (feats float16 [rows, layers, calls, dim], labels, meta).
scripts/probe_digits.py fits the probes on the Jetson.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TOKEN = 16  # pixels per token side: VAE 8x, patch 2x2
LAT_FRAMES = 4  # video frames per latent (Wan's causal VAE, after the first)
# code-probe-r1 blocker (a): a control box per place, the place's box moved onto the keypad (on the start image, 80 px
# down and 50 px left lands on the keys, at least one token below the LCD): a digit read as well there is not held at
# its place (the prompt names the whole number)
CONTROL = (-50, 80)


class FeatureProbe:
    def __init__(self, transformer, layers: list[int], calls: int = 4) -> None:
        import torch
        self.torch = torch
        model = transformer.model  # WanDiffusionWrapper -> CausalWanModel
        self.layers, self.calls = list(layers), calls
        self.on, self.call, self.store = False, -1, {}
        self.skips: dict = {}  # code-probe-r1 should-fix 2: why blocks were skipped
        self.rows: list[dict] = []
        self.feats: list[np.ndarray] = []
        self.handles = [transformer.register_forward_pre_hook(self._pre)]
        for li in self.layers:
            self.handles.append(model.blocks[li].register_forward_hook(self._hook(li)))

    def _pre(self, _module, _args) -> None:
        if self.on:
            self.call += 1  # 0: the KV-cache rebuild; 1..calls: the denoising steps

    def _hook(self, li: int):
        def hook(_module, _inp, out) -> None:
            if self.on and 1 <= self.call <= self.calls:
                self.store[(li, self.call - 1)] = out.detach()[0].clone()  # [tokens, dim], on the GPU until pool()
        return hook

    def arm(self) -> None:
        self.on, self.call, self.store = True, -1, {}

    def disarm(self) -> None:
        self.on = False

    def skip(self, why: str) -> int:
        self.skips[why] = self.skips.get(why, 0) + 1
        self.store = {}
        return 0

    def pool(self, places: list, block: int, typed: str, shown: str, hw: tuple[int, int] = (480, 832)) -> int:
        """places[i]: for video frame i of the block, None (LCD not found) or a list of (bbox x0, y0, x1, y1 in frame
        pixels, digit label: "0".."9", " " blank, "?" unreadable) per digit place. One row per latent frame and place,
        labelled by the latent's last frame with a reading (all 4 labels kept)."""
        torch = self.torch
        gh, gw = hw[0] // TOKEN, hw[1] // TOKEN
        per_lat = gh * gw
        # exactly the rebuild and `calls` denoising calls, each on whole latent frames; else the block is skipped
        if self.call != self.calls:
            return self.skip(f"calls {self.call}")
        if len(self.store) != len(self.layers) * self.calls or any(v.shape[0] % per_lat for v in self.store.values()):
            return self.skip("tokens")
        n_lat = self.store[(self.layers[0], 0)].shape[0] // per_lat
        if len(places) != n_lat * LAT_FRAMES:  # should-fix 1: a session's first block (1 + 4 + 4 frames, 3 dropped)
            return self.skip(f"frames {len(places)}")
        added = 0
        for j in range(n_lat):
            fr = [places[i] for i in range(j * LAT_FRAMES, min((j + 1) * LAT_FRAMES, len(places)))]
            ref = next((p for p in reversed(fr) if p is not None), None)
            if ref is None:
                continue
            for k, (bbox, lab) in enumerate(ref):
                for control in (False, True):
                    box = (bbox[0] + CONTROL[0], bbox[1] + CONTROL[1], bbox[2] + CONTROL[0], bbox[3] + CONTROL[1]) \
                        if control else bbox
                    w = self._weights(box, gh, gw)
                    if w is None:
                        continue
                    wt = torch.from_numpy(w).to(device=self.store[(self.layers[0], 0)].device)  # float32 (r1 nit)
                    vec = torch.stack([torch.stack([wt @ self.store[(li, c)][j * per_lat:(j + 1) * per_lat].float()
                                                    for c in range(self.calls)]) for li in self.layers])
                    self.feats.append(vec.cpu().numpy().astype(np.float16))
                    self.rows.append({"block": block, "latent": j, "place": k, "label": lab, "control": control,
                                      "labels4": [p[k][1] if p is not None and k < len(p) else None for p in fr],
                                      "bbox": [int(v) for v in box], "typed": typed, "shown": shown})
                    added += 1
        self.store = {}
        return added

    @staticmethod
    def _weights(bbox, gh: int, gw: int) -> np.ndarray | None:
        """Each token's share of the box (overlap area), normalised to sum 1."""
        x0, y0, x1, y1 = bbox
        if x1 <= x0 or y1 <= y0:
            return None
        xs, ys = np.arange(gw) * TOKEN, np.arange(gh) * TOKEN
        ox = np.clip(np.minimum(xs + TOKEN, x1) - np.maximum(xs, x0), 0, None)
        oy = np.clip(np.minimum(ys + TOKEN, y1) - np.maximum(ys, y0), 0, None)
        w = np.outer(oy, ox).astype(np.float32).ravel()
        return w / w.sum() if w.sum() > 0 else None

    def save(self, path: Path, meta: dict) -> None:
        feats = np.stack(self.feats) if self.feats else np.zeros((0, len(self.layers), self.calls, 1), np.float16)
        np.savez(path, feats=feats, labels=np.array([r["label"] for r in self.rows]),
                 rows=json.dumps(self.rows), meta=json.dumps({**meta, "layers": self.layers, "calls": self.calls,
                                                              "skips": self.skips, "control": CONTROL}))

    def close(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
