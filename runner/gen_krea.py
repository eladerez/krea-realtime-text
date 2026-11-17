"""Generate videos with Krea Realtime 14B (GPU VM), through Krea's own inference code in third_party/krea-realtime.
Runs in Krea's environment (its uv.lock), from any directory. Resumable: videos that exist are skipped.

Two changes around Krea's loader, neither touching the model or sampling:
  - the transformer is built from Wan2.1-T2V-14B's config.json instead of from_pretrained, which would first load
    57 GB of Wan weights that Krea's checkpoint then overwrites (so only config.json has to be downloaded);
  - Krea reads the T5 encoder as .safetensors while Wan ships it as .pth; it is converted once, unchanged.
Usage: python runner/gen_krea.py --configs krea_rt,krea_rt_long --models /opt/models [--preflight] [--out ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, jobs, load_config, log_timing, video_path, write_mp4  # noqa: E402

sys.path.insert(0, str(ROOT))  # textbench (OCR) for the D2 text anchor

KREA = ROOT / "third_party/krea-realtime"


def prepare_t5(models: Path) -> None:
    import torch
    from safetensors.torch import save_file
    folder = models / "Wan-AI/Wan2.1-T2V-1.3B"
    dst = folder / "models_t5_umt5-xxl-enc-bf16.safetensors"
    if not dst.exists():
        state = torch.load(folder / "models_t5_umt5-xxl-enc-bf16.pth", map_location="cpu")
        save_file({k: v.contiguous() for k, v in state.items()}, str(dst))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="", help="comma-separated; all run on one model load")
    ap.add_argument("--live", default=None, help="LIVE.md: a live config; serves runner/live_server.py instead of videos")
    ap.add_argument("--models", type=Path, required=True, help="folder with Wan-AI/... and krea/...")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "data/videos")
    args = ap.parse_args()
    runs = []
    for name in [n for n in args.configs.split(",") if n]:
        cfg = load_config(name)
        assert cfg["backend"] == "krea", name
        out = args.out / name
        out.mkdir(parents=True, exist_ok=True)
        todo = [(p, s) for p, s in jobs(cfg, args.preflight) if not video_path(out, p["id"], s).exists()]
        print(f"{name}: {len(todo)} videos to make", flush=True)
        runs.append((name, cfg, out, todo))
    if not any(todo for *_, todo in runs) and not args.live:
        return

    os.environ["MODEL_FOLDER"] = str(args.models / "Wan-AI")
    os.environ["DO_COMPILE"] = "false"
    os.chdir(KREA)  # Krea reads configs/ and templates by relative path
    sys.path.insert(0, str(KREA))
    import numpy as np
    import torch
    from wan.modules.causal_model import CausalWanModel

    CausalWanModel.from_pretrained = classmethod(
        lambda cls, path, **kw: cls.from_config(cls.load_config(path), **kw))
    prepare_t5(args.models)
    from release_server import GenerateParams, GenerationSession, load_all, load_merge_config

    config = load_merge_config("configs/self_forcing_server_14b.yaml")
    config.checkpoint_path = str(args.models / "krea/krea-realtime-video-14b.safetensors")
    t = time.perf_counter()
    models = load_all(config)
    print(f"models loaded in {time.perf_counter() - t:.0f} s", flush=True)
    krea_commit = os.popen("git rev-parse --short HEAD").read().strip()

    class AnchoredSession(GenerationSession):
        """D2, the text anchor. Past the first blocks, Krea re-encodes the oldest frame of its rolling pixel cache
        (frame_context_cache[0], ~9 frames back) as the first context latent. Here, once OCR reads every target in
        a decoded frame, that frame is kept and used in the same slot for the rest of the video."""
        anchor = None

        def get_clean_context_frames(self, models):
            if self.anchor is None or not self.frame_context_cache:
                return super().get_clean_context_frames(models)
            saved, self.frame_context_cache[0] = self.frame_context_cache[0], self.anchor
            try:
                return super().get_clean_context_frames(models)
            finally:
                self.frame_context_cache[0] = saved

    from textbench.erase import erase, text_boxes
    from textbench.write_text import track_boxes, write as write_text
    from textbench.ocr_metrics import best_match, ocr_frame, reading_order, verdict
    from v2v import encode_video_latent
    import wan.modules.causal_model as causal_model

    # Step 18, option 3: hide the old text's tokens from attention. In Krea's denoising pass, self-attention is one
    # call, attention(query = new block, key/value = [context latents | new block]), with no mask. While MASK holds
    # token positions (0-1559 within a latent frame: 30 x 52 patches of 16 x 16 pixels, row-major), those positions
    # are masked out in every context frame, so the new frames cannot copy the old text from memory. Cross-attention
    # (keys = 512 prompt tokens) and the KV-recompute pass (flex_attention) are not touched.
    FRAME_TOKENS, GRID_W = 1560, 52
    MASK: dict = {"tokens": None}
    _attention = causal_model.attention

    def masked_attention(q, k, v, *args, **kwargs):
        n_ctx = k.shape[1] - q.shape[1]
        if MASK["tokens"] is None or n_ctx <= 0 or args or kwargs:
            return _attention(q, k, v, *args, **kwargs)
        keep = torch.ones(k.shape[1], dtype=torch.bool, device=q.device)
        idx = (torch.arange(n_ctx // FRAME_TOKENS, device=q.device)[:, None] * FRAME_TOKENS
               + MASK["tokens"].to(q.device)[None, :]).flatten()
        keep[idx[idx < n_ctx]] = False
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=keep[None, None, None, :])
        return out.transpose(1, 2).type_as(q)

    causal_model.attention = masked_attention

    # Step 20, option 3: steer the new frames towards the written text while they are drawn. Between Krea's denoising
    # passes the predicted clean latent goes through scheduler.add_noise before the next pass; while GUIDE is set,
    # the text box of that prediction is replaced by the latent of the frame with the new text written in, for every
    # pass but the last, so the final pass still draws the box in the model's own way. Memory is not edited.
    # Patched on the class (step 24 fix): every GenerationSession builds a new FlowMatchScheduler (release_server.py
    # init_models), so the instance patch used in step 20 was replaced before any block ran and guide_written was the
    # plain switch.
    GUIDE: dict = {"lat": None, "mask": None}
    from utils.scheduler import FlowMatchScheduler
    _add_noise = FlowMatchScheduler.add_noise

    def guided_add_noise(self, x0, noise, timestep):
        # not into the last pass: the server runs t = 1000, 909, 714, 0, and the t=0 pass barely changes its input, so
        # steering it would paste the latent; steered inputs are those of the 909 and 714 passes
        # LIVE.md (15 Nov, the latent pin): with GUIDE["final"] the t=0 pass's input is steered too, so the region's
        # output is (nearly) the target latent: the display is pinned, not only guided
        if GUIDE["lat"] is not None and x0.shape[0] == GUIDE["lat"].shape[0] and (
                float(timestep.max()) > 0 or GUIDE.get("final")):
            x0 = x0.clone()
            x0[:, :, GUIDE["mask"]] = GUIDE["lat"][:, :, GUIDE["mask"]].to(x0)
        return _add_noise(self, x0, noise, timestep)

    FlowMatchScheduler.add_noise = guided_add_noise

    def box_tokens(boxes: list[tuple[int, int, int, int]]) -> torch.Tensor | None:
        """Token positions (within one latent frame) covered by pixel boxes at 832x480: 16 px per token."""
        ids = {r * GRID_W + c for x0, y0, x1, y1 in boxes
               for r in range(y0 // 16, min(30, (y1 - 1) // 16 + 1)) for c in range(x0 // 16, min(GRID_W, (x1 - 1) // 16 + 1))}
        return torch.tensor(sorted(ids), dtype=torch.long) if ids else None

    def ocr_texts(frame: torch.Tensor) -> list[str]:
        img = frame[0, 0].float().add(1).mul(127.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return reading_order(ocr_frame(img))

    def snapshot(s) -> dict:
        """Everything a block changes in Krea's session (the KV cache is rebuilt from all_latents every block)."""
        return {"current_start_frame": s.current_start_frame, "block_idx": s.block_idx,
                "total_frames_sent": s.total_frames_sent, "last_pred": s.last_pred,
                "decode_vae_cache": [c.clone() if isinstance(c, torch.Tensor) else c for c in s.decode_vae_cache],
                "frame_context_cache": list(s.frame_context_cache)}

    def restore(s, snap: dict) -> None:
        for k in ("current_start_frame", "block_idx", "total_frames_sent", "last_pred"):
            setattr(s, k, snap[k])
        s.decode_vae_cache = [c.clone() if isinstance(c, torch.Tensor) else c for c in snap["decode_vae_cache"]]
        s.frame_context_cache = deque(snap["frame_context_cache"], maxlen=s.frame_context_cache.maxlen)

    def to_rgb(x: torch.Tensor) -> np.ndarray:
        return x[0, 0].float().add(1).mul(127.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()

    def encode9(frames: list[torch.Tensor]) -> torch.Tensor:
        """Krea's own encoder call (as in get_clean_context_frames / setup_start_frame): 9 pixel frames -> 3 latents."""
        return encode_video_latent(models.vae_encoder, [None] * 55, resample_to=16, max_frames=81,
                                   video_path_or_url=None, frames=torch.cat([c[0] for c in frames]).half(),
                                   height=480, width=832, stream=False)[0].transpose(0, 1)[None]

    def to_tensor(rgb: np.ndarray, like: torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(127.5).sub(1)[None, None] \
            .to(dtype=like.dtype, device=like.device)

    def written_cache(s, pairs: list[tuple[str, str]], align: str) -> tuple[list[torch.Tensor], int]:
        """Step 20: the 9 cached frames with each old text replaced by its new text (textbench/write_text.py); boxes
        are found on the newest frame and reused on the older ones (step 22b, cfg track_boxes: moved to where their
        area is in each older frame, so with a moving camera the written text moves with the scene)."""
        cache = list(s.frame_context_cache)
        if WRITE["editor"]:  # step 36: Qwen-Image-Edit redraws the label on the newest frame (textbench/edit_write.py)
            from textbench.edit_write import edit_cache
            out, n, info = edit_cache([to_rgb(c) for c in cache], pairs, EDITOR["fn"], str(ROOT), seed=WRITE["seed"],
                                      track=WRITE["track"], crop=WRITE["crop"], boxes_by_old=WRITE.get("fixed_boxes"))
            info["server_s"] = EDITOR["fn"].last.get("seconds") if n or "editor_s" in info else None
            WRITE["edits"].append(info)
            return ([to_tensor(o, c) for o, c in zip(out, cache)], n) if n else (cache, 0)
        part = WRITE["partial"]  # step 21: change only the differing characters (textbench/write_text.py partial)
        places = [new for old, new in pairs if old == ""]
        if places:  # step 25c: no old text, a blank label: find it and place the text on it
            from textbench.write_surface import place_cache
            out, n = place_cache([to_rgb(c) for c in cache], places[0], str(ROOT))
            return ([to_tensor(o, c) for o, c in zip(out, cache)], n) if n else (cache, 0)
        if WRITE["surface"]:  # step 24: on the panel's own plane (textbench/write_surface.py), tracked by homography
            from textbench.write_surface import write_cache
            out, n = write_cache([to_rgb(c) for c in cache], pairs, str(ROOT), align, partial_edit=part, v2=WRITE["surface_v2"])
            return ([to_tensor(o, c) for o, c in zip(out, cache)], n) if n else (cache, 0)
        # LIVE.md: a live session passes the label's box (runner/live_server.py LabelBox) instead of finding it by OCR
        _, boxes = write_text(to_rgb(cache[-1]), pairs, str(ROOT), align, WRITE.get("fixed_boxes"), partial_edit=part)
        n = sum(isinstance(v, tuple) for v in boxes.values())
        if not n:
            return cache, 0
        newest = to_rgb(cache[-1])
        out = []
        for c in cache:
            rgb = to_rgb(c)
            bx = track_boxes(newest, rgb, boxes) if WRITE["track"] else boxes
            out.append(to_tensor(write_text(rgb, pairs, str(ROOT), align, bx, partial_edit=part)[0], c))
        return out, n

    def free_text_encoder(models, runs) -> None:
        """Step 38: Krea's text encoder (UMT5-XXL, ~11 GB) leaves the GPU so the resident editor fits beside Krea. Every
        prompt the run will use (each video's prompt and its switch prompts) is encoded first, on the GPU as always;
        models.text_encoder then answers from those embeddings (the same tensors Krea would have computed)."""
        prompts = {p["prompt"] for _, _, _, todo in runs for p, _ in todo}
        prompts |= {x["prompt"] for _, _, _, todo in runs for p, _ in todo for x in p.get("schedule", [])}
        enc = models.text_encoder
        with torch.no_grad():
            cache = {q: {k: v.detach().clone() for k, v in enc(text_prompts=[q]).items()} for q in sorted(prompts)}
        with torch.inference_mode(False), torch.no_grad():
            enc.to("cpu")
        torch.cuda.empty_cache()

        class CachedText:
            def __call__(self, text_prompts: list[str]) -> dict:
                (q,) = text_prompts
                return {k: v.clone() for k, v in cache[q].items()}
        models.text_encoder = CachedText()
        print(f"text encoder off the GPU; {len(cache)} prompts encoded", flush=True)

    WRITE = {"partial": False, "track": False, "surface": False, "surface_v2": False, "editor": False, "seed": 0,
             "edits": [], "crop": False}
    EDITOR: dict = {"fn": None}
    if any(cfg.get("editor") for _, cfg, _, todo in runs if todo):  # step 36: the editor process (runner/edit_server.py)
        from textbench.edit_write import FileEditor
        parked = [(m, next(m.parameters()).device) for m in (models.transformer, models.text_encoder)]  # both on cuda

        def park(away: bool) -> None:
            # Krea's weights leave the GPU while the editor runs (both do not fit: 69 + 67 GB peaks on 96 GB) and go back
            # where they were, outside inference mode so the parameters stay normal tensors (moving them inside it
            # makes inference tensors; tested on torch 2.8, Krea's pin)
            with torch.inference_mode(False), torch.no_grad():
                for m, home in parked:
                    m.to("cpu" if away else home)
            torch.cuda.synchronize()
            if away:
                torch.cuda.empty_cache()

        # step 40b: the 40a fast editors (FLUX.2 klein, Qwen-Image-Edit-2511 in fp8) stay on the GPU as step 38's did;
        # step 40c: RS-STE too. Only the big ones (step 38's, Qwen-2511 fp8: ~20-31 GB) need Krea's text encoder off the
        # GPU (and so every switch prompt encoded in advance); klein (~17 GB) and RS-STE (~0.3 GB) fit beside it.
        resident = [cfg.get("editor") in ("qwen_edit_resident", "flux2_klein", "qwen_edit_2511_fp8", "rsste")
                    for _, cfg, _, todo in runs if todo and cfg.get("editor")]
        big = any(cfg.get("editor") in ("qwen_edit_resident", "qwen_edit_2511_fp8") for _, cfg, _, todo in runs if todo)
        if any(resident):  # step 38: the editor stays on the GPU (runner/edit_server.py --resident); nothing is parked
            assert all(resident), "a resident editor run cannot also hold full or fast editor configs"
            if big:
                free_text_encoder(models, runs)
            EDITOR["fn"] = FileEditor(Path(os.environ.get("EDIT_FOLDER", "/opt/textbench/edits")))
        else:
            # step 39: the swap model is small and stays on the GPU; nothing to park. Named park_if_needed: as
            # `parked` (until 12 Nov 2026) it shadowed the list of Krea's weights above, and park() then iterated over
            # this function (TypeError on the first step 37 edit; textbench-preflight-20261112-1220).
            def park_if_needed(away: bool) -> None:
                if EDITOR["fn"].mode not in ("swap", "rsste"):  # step 40b: RS-STE is small too
                    park(away)
            EDITOR["fn"] = FileEditor(Path(os.environ.get("EDIT_FOLDER", "/opt/textbench/edits")),
                                      before=lambda: park_if_needed(True), after=lambda: park_if_needed(False))
        t = time.perf_counter()
        EDITOR["fn"].wait_ready()
        print(f"editor ready after {time.perf_counter() - t:.0f} s more", flush=True)

    def write_memory(s, pairs: list[tuple[str, str]], align: str) -> tuple[int, torch.Tensor | None]:
        """Step 20, option 1: write the new text into Krea's memory (pixel cache + last two latents) and continue."""
        # step 40e (code review research/reviews/code-40e-r1.md): an unchanged text (old == new, the digit demo's 7 -> 7
        # control) is not written; the editor path already skipped it (textbench/edit_write.py edit_cache), the font
        # path redrew it. Step 25c's ("", new) pairs are kept.
        pairs = [(o, n) for o, n in pairs if o != n]
        if not pairs or len(s.frame_context_cache) != s.frame_context_cache.maxlen or s.current_start_frame < 2:
            return 0, None
        edited, n = written_cache(s, pairs, align)
        if not n:
            return 0, None
        s.frame_context_cache = type(s.frame_context_cache)(edited, maxlen=s.frame_context_cache.maxlen)
        st = s.current_start_frame
        s.all_latents[:, st - 2:st] = encode9(edited)[:, 1:3].to(s.all_latents)
        return n, edited[-1]

    def refresh_memory(s, wanted: list[str], align: str) -> dict:
        """Step 34: before a block, re-write any wanted text that has slipped in the newest memory frame
        (textbench/refresh.py decides; write_memory writes, from what is shown to what is wanted)."""
        from textbench.refresh import refresh_pairs
        pairs, n = refresh_pairs(ocr_texts(s.frame_context_cache[-1]), wanted)
        n["boxes"] = write_memory(s, pairs, align)[0] if pairs else 0
        n["pairs"] = pairs
        return n

    def edit_memory(s, texts: list[str]) -> int:
        """Step 17, option 2: erase the old text inside Krea's memory and continue, no restart. The memory a block
        reads is the pixel cache (its oldest frame is re-encoded as the first context latent) plus the last two
        latents; the pixel cache holds exactly the 1 + 2 x 4 frames those three latents come from. The text's boxes
        (found by OCR on the newest frame) are erased in all 9 cached frames, which are then encoded again with Krea's
        own encoder call (as in get_clean_context_frames) to replace the last two latents."""
        cache = list(s.frame_context_cache)
        boxes = text_boxes(to_rgb(cache[-1]), texts)
        if not boxes or len(cache) != s.frame_context_cache.maxlen or s.current_start_frame < 2:
            return len(boxes)
        edited = [torch.from_numpy(erase(to_rgb(c), boxes)).permute(2, 0, 1).float().div(127.5).sub(1)[None, None]
                  .to(dtype=c.dtype, device=c.device) for c in cache]
        s.frame_context_cache = type(s.frame_context_cache)(edited, maxlen=s.frame_context_cache.maxlen)
        frames9 = torch.cat([c[0] for c in edited]).half()
        lat = encode_video_latent(models.vae_encoder, [None] * 55, resample_to=16, max_frames=81,
                                  video_path_or_url=None, frames=frames9, height=480, width=832,
                                  stream=False)[0].transpose(0, 1)[None]
        st = s.current_start_frame
        s.all_latents[:, st - 2:st] = lat[:, 1:3].to(s.all_latents)
        return len(boxes)

    @torch.inference_mode()  # Krea's session tensors are inference tensors; the retry edits them in place
    def generate(cfg: dict, prompt: str, seed: int, blocks: int, targets: list[str] | None = None,
                 verify: list[str] | None = None, start=None, anchor_mode: str = "text",
                 schedule: dict[int, str] | None = None,
                 old_texts: dict[int, list[str]] | None = None,
                 pairs: dict[int, list[tuple[str, str]]] | None = None, align: str = "center",
                 wanted: dict[int, list[str]] | None = None) -> tuple[np.ndarray, dict]:
        """targets: D2 text anchor. verify: the OCR checker in the loop: after each block (~0.75 s) OCR reads its
        last frame; if a target is visible but misspelled (verdict "wrong"), the block is generated again from the
        same state with new noise, up to cfg["verify_tries"] times; the last try is kept either way.
        schedule: step 15, {block: prompt}: before that block the prompt switches, the way Solaris's language model
        rewrites the prompt after a click. Done as Krea's own prompt change does it (release_server.py,
        interpolated_prompt_embeds): new embeddings and a cleared cross-attention cache; set before the block so its
        KV-cache recompute already sees the new prompt.
        old_texts: steps 17-18, {block: texts shown before that switch}, for the erase variants.
        wanted: step 34, {block: texts the screen should show}; with cfg refresh, slipped text is re-written before
        every block from block 1 on, except at a switch (its own write happens there)."""
        frames: list[torch.Tensor] = []
        WRITE["partial"] = cfg.get("write_partial", False)
        WRITE["track"] = cfg.get("track_boxes", False)
        WRITE["surface"] = cfg.get("write_surface", False)
        WRITE["surface_v2"] = cfg.get("write_surface_v2", False)  # step 31: textbench/write_surface.py v2
        WRITE["editor"] = bool(cfg.get("editor"))  # step 36
        if EDITOR["fn"] is not None:  # step 37: qwen_edit_fast, the 8-step editor; step 38: resident, on a crop
            EDITOR["fn"].mode = {"qwen_edit_fast": "fast", "qwen_edit_resident": "resident",
                                 "swap_latent": "swap", "rsste": "rsste", "flux2_klein": "klein",
                                 "qwen_edit_2511_fp8": "qwen2511fp8"}.get(cfg.get("editor"), "full")  # 39, 40b
        WRITE["crop"] = cfg.get("editor") == "qwen_edit_resident"
        WRITE["seed"], WRITE["edits"] = seed, []

        def on_frames(pixels, _ids, event):
            event.synchronize()
            frames.append(pixels[0].float().add(1).mul(127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu())

        # start: step 9, a PIL frame rendered by the image model; Krea encodes it as the context it continues from
        # (its own setup_start_frame). It fills the first latent block, so one extra block of room is allocated.
        params = GenerateParams(prompt=prompt, seed=seed, num_blocks=blocks + (start is not None),
                                num_denoising_steps=cfg["steps"], kv_cache_num_frames=cfg["kv_cache_num_frames"])
        cls = AnchoredSession if targets or cfg.get("memory_edit") == "write_anchor" else GenerationSession
        session = cls(params=params, config=config, frame_callback=on_frames, models=models)
        if start is not None:
            session.setup_start_frame(start, models)
        stats = {"anchor_block": -1, "retries": 0, "blocks_wrong_kept": 0, "blocks_wrong_first_try": 0}
        tries = cfg.get("verify_tries", 4) if verify else 1
        n = models.pipeline.num_frame_per_block
        best_ok = 0
        mask_tokens, mask_until = None, -1
        guide_until = -1
        block_t0 = time.perf_counter()  # step 40c: each block's start (s), so the whole stall at a switch can be read:
        stats["block_t"] = []           # the switch block's length minus a normal block's (frames are synced per block)
        for b in range(blocks):
            stats["block_t"].append(round(time.perf_counter() - block_t0, 4))
            if schedule and b in schedule:
                stats.setdefault("switch_frames", []).append(sum(f.shape[0] for f in frames))
            if schedule and b in schedule and cfg.get("reset"):
                # Step 16: clear Krea's memory at the switch so the old text is not copied forward. A fresh session on
                # the new prompt for the remaining blocks: "full" starts from nothing; "blur" starts from the current
                # frame blurred (layout and colours kept, letters unreadable), through Krea's own start-frame path.
                from PIL import Image, ImageFilter
                last = Image.fromarray(frames[-1][-1].numpy())
                session.dispose()
                blur = cfg["reset"] in ("blur", "erase")
                p2 = GenerateParams(prompt=schedule[b], seed=seed * 1009 + b, num_blocks=blocks - b + blur,
                                    num_denoising_steps=cfg["steps"], kv_cache_num_frames=cfg["kv_cache_num_frames"])
                session = GenerationSession(params=p2, config=config, frame_callback=on_frames, models=models)
                if cfg["reset"] == "erase":  # step 17, option 1: only the old text's area is cleared, the rest stays sharp
                    arr = np.asarray(last)
                    boxes = text_boxes(arr, (old_texts or {}).get(b, []))
                    stats.setdefault("erased_boxes", []).append(len(boxes))
                    session.setup_start_frame(Image.fromarray(erase(arr, boxes)), models)
                elif blur:
                    session.setup_start_frame(last.filter(ImageFilter.GaussianBlur(cfg.get("blur_radius", 8))), models)
                stats.setdefault("reset_blocks", []).append(b)
            elif schedule and b in schedule:
                prev_emb = session.current_prompt_embeds
                emb = models.text_encoder(text_prompts=[schedule[b]])["prompt_embeds"].to(dtype=torch.bfloat16)
                if session.current_prompt_embeds is not None:
                    emb = emb.to(session.current_prompt_embeds.device)
                session.current_prompt_embeds = emb
                if hasattr(session, "conditional_dict"):
                    session.conditional_dict["prompt_embeds"] = emb
                models.pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=session.gpu)
                stats.setdefault("prompt_switch_blocks", []).append(b)
                if cfg.get("memory_edit") == "erase":
                    stats.setdefault("erased_boxes", []).append(edit_memory(session, (old_texts or {}).get(b, [])))
                elif cfg.get("memory_edit") in ("write", "write_anchor"):  # step 20, options 1 and 2
                    n, newest = write_memory(session, (pairs or {}).get(b, []), align)
                    stats.setdefault("written_boxes", []).append(n)
                    if not n and any(o == "" for o, _ in (pairs or {}).get(b, [])) and prev_emb is not None:
                        # step 25c: no blank label found to place the text on: keep the blank-label prompt, or Krea
                        # draws its own garbled text (salad seed 0: "2 ITEMTS") and later edits stack on it
                        session.current_prompt_embeds = prev_emb
                        if hasattr(session, "conditional_dict"):
                            session.conditional_dict["prompt_embeds"] = prev_emb
                        models.pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16,
                                                                    device=session.gpu)
                        stats.setdefault("place_failed_blocks", []).append(b)
                    if cfg["memory_edit"] == "write_anchor" and newest is not None:
                        session.anchor = newest  # kept in the oldest memory slot for the rest of the video
                elif cfg.get("guide") == "written_text":  # step 20, option 3
                    edited, n = written_cache(session, (pairs or {}).get(b, []), align) if (pairs or {}).get(b) else ([], 0)
                    stats.setdefault("guided_boxes", []).append(n)
                    if n:
                        lat = encode9(edited)[0, -1:].repeat(models.pipeline.num_frame_per_block, 1, 1, 1)
                        _, boxes = write_text(to_rgb(session.frame_context_cache[-1]), (pairs or {}).get(b, []),
                                              str(ROOT), align)
                        m = torch.zeros(lat.shape[-2:], dtype=torch.bool, device=lat.device)
                        for x0, y0, x1, y1 in (v for v in boxes.values() if isinstance(v, tuple)):  # 8 x 8 px cells
                            m[max(0, y0 // 8 - 1):(y1 - 1) // 8 + 2, max(0, x0 // 8 - 1):(x1 - 1) // 8 + 2] = True
                        GUIDE["lat"], GUIDE["mask"] = lat, m
                        guide_until = b + cfg.get("guide_blocks", 2)
                elif cfg.get("memory_edit") == "mask":
                    boxes = text_boxes(to_rgb(session.frame_context_cache[-1]), (old_texts or {}).get(b, []))
                    mask_tokens, mask_until = box_tokens(boxes), b + cfg.get("mask_blocks", 2)
                    stats.setdefault("masked_tokens", []).append(0 if mask_tokens is None else len(mask_tokens))
            if cfg.get("refresh") and wanted and b >= 1 and not (schedule and b in schedule):
                stats.setdefault("refresh", []).append([b, refresh_memory(session, wanted[b], align)])
            if cfg.get("memory_edit") == "mask":
                MASK["tokens"] = mask_tokens if schedule and b < mask_until else None
            if cfg.get("guide") == "written_text" and b >= guide_until:
                GUIDE["lat"] = None
            snap = snapshot(session) if verify else None
            for a in range(tries):
                n_frames = len(frames)
                session.generate_block(models)
                if not verify or verdict(ocr_texts(session.frame_context_cache[-1]), verify) != "wrong":
                    break
                stats["blocks_wrong_first_try"] += a == 0
                if a == tries - 1:
                    stats["blocks_wrong_kept"] += 1
                    break
                del frames[n_frames:]
                restore(session, snap)
                st = session.current_start_frame
                g = torch.Generator(session.noise.device).manual_seed(seed * 100003 + b * 101 + a + 1)
                session.noise[:, st:st + n] = torch.randn(session.noise[:, st:st + n].shape, generator=g,
                                                          device=session.noise.device, dtype=session.noise.dtype)
                stats["retries"] += 1
            if targets and (session.anchor is None or anchor_mode == "best"):
                texts = ocr_texts(session.frame_context_cache[-1])
                n_ok = sum(best_match(texts, t)[0] == 0 for t in targets)
                # "text": the first frame where every target reads exactly, kept for good.
                # "best": the frame with the most targets reading exactly so far, replaced when a better one appears.
                if (anchor_mode == "text" and n_ok == len(targets)) or (anchor_mode == "best" and n_ok > best_ok):
                    session.anchor = session.frame_context_cache[-1].clone()
                    stats["anchor_block"] = b
                    stats["anchor_targets_ok"] = best_ok = n_ok
                    stats["anchor_updates"] = stats.get("anchor_updates", 0) + 1
        if WRITE["editor"]:
            stats["edits"] = WRITE["edits"]
        MASK["tokens"] = None
        GUIDE["lat"] = None
        session.dispose()
        return torch.cat(frames).numpy(), stats

    # Untimed warm-up: compiles flex_attention (used for the KV-cache recompute) and selects kernels.
    if any(todo for *_, todo in runs):  # code-live-r1 #1: a live-only run has no videos (it warms up on its first block)
        first = next((name, cfg, todo) for name, cfg, _, todo in runs if todo)
        generate(first[1], first[2][0][0]["prompt"], 12345, 3)

    def run_config(cfg: dict, out: Path, todo: list) -> None:
        for p, seed in todo:
            texts = p.get("targets") or ([p["target"]] if "target" in p else [])
            targets = texts if cfg.get("anchor") in ("text", "best") else None
            verify = texts if cfg.get("verify") else None
            torch.cuda.synchronize()
            t = time.perf_counter()
            start = None
            if cfg.get("start_frame"):  # true: step 9's Qwen-Image frames; a path: that folder (step 10, Kontext)
                from PIL import Image
                folder = "data/start_frames" if cfg["start_frame"] is True else cfg["start_frame"]
                # step 40e: a row may reuse another row's start image (prompts/demo_digits.jsonl, start_id)
                start = Image.open(ROOT / folder / f"{p.get('start_id', p['id'])}-s{seed}.png").convert("RGB")
            schedule = {int(x["block"]): x["prompt"] for x in p.get("schedule", [])} or None
            old_texts = None
            if schedule:  # the text on screen just before each switch: the "before" text, or the previous typed stage
                blocks_sorted = sorted(schedule)
                old_texts = {blocks_sorted[0]: p.get("before", [])}
                for k, blk in enumerate(blocks_sorted[1:]):
                    old_texts[blk] = [p["stages"][k]] if p.get("stages") else []
            pairs = None  # step 20: (old text, new text) at each switch; a typed stage replaces the previous one
            if schedule and p.get("stages"):
                bs = sorted(schedule)
                pairs = {blk: [(p["stages"][k - 1], p["stages"][k])] for k, blk in enumerate(bs) if k > 0}
            elif schedule:
                pairs = {min(schedule): list(zip(p.get("before", []), p.get("after", [])))}
            if any("after" in x for x in p.get("schedule", [])):  # step 23: each switch names its own text change
                pairs = {int(x["block"]): list(zip(x.get("before", []), x.get("after", []))) for x in p["schedule"]}
                old_texts = {int(x["block"]): x.get("before", []) for x in p["schedule"]}
            wanted = None
            if cfg.get("refresh"):  # step 34
                from textbench.refresh import wanted_by_block
                wanted = wanted_by_block(p, cfg["blocks"])
            frames, stats = generate(cfg, p["prompt"], seed, cfg["blocks"], targets, verify, start,
                                     cfg.get("anchor", "text"), schedule, old_texts, pairs,
                                     "left" if p.get("stages") else "center", wanted)
            torch.cuda.synchronize()
            total = time.perf_counter() - t
            path = video_path(out, p["id"], seed)
            write_mp4(frames, path)
            log_timing(out, {"video": path.name, "prompt_id": p["id"], "seed": seed, "frames": len(frames),
                             "total_s": total, "fps": len(frames) / total, **stats,
                             "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9})
            print(f"{path.name}: {len(frames)} frames in {total:.1f} s ({len(frames) / total:.1f} fps)", flush=True)

    failed: list[str] = []
    def live_generator(cfg: dict):
        """LIVE.md Phase 1: Krea generates block after block while the viewer types. Before each block, if the typed
        text differs from what the sign shows, the prompt switches and the text is written into Krea's memory with
        write_memory, exactly as a scheduled switch in the tests (font write, tracked boxes). runner/live_server.py
        serves the frames and takes the keys; this is its generator."""
        sys.path.insert(0, str(ROOT / "runner"))
        from PIL import Image
        WRITE.update(partial=cfg.get("write_partial", False), track=cfg.get("track_boxes", False),
                     editor=bool(cfg.get("editor")), surface=False, crop=False, seed=cfg.get("seed", 0), edits=[])
        if cfg.get("editor"):  # LIVE.md: FLUX.2 klein beside Krea (runner/edit_server.py --fast-editors), as in 40b
            from textbench.edit_write import FileEditor, edit_prompt
            # code-live-r2 #6: Qwen-2511 fp8 would also need Krea's text encoder off the GPU (not done in live)
            assert cfg["editor"] == "flux2_klein", "live supports the klein editor only"
            EDITOR["fn"] = FileEditor(Path(os.environ.get("EDIT_FOLDER", "/opt/textbench/edits")))
            EDITOR["fn"].mode = "klein"
            EDITOR["fn"].wait_ready()
        assert not cfg.get("live_restyle") or cfg.get("editor") == "flux2_klein", "live_restyle needs editor: flux2_klein"  # code-klein-r1 nit 3
        seed, tmpl = cfg.get("seed", 0), cfg["live_prompt"]
        switch_tmpl = cfg.get("live_switch_prompt", tmpl)  # code-live-r1 #12: as 40e's tested switch prompts
        # start_seed: the start image's seed when it differs from Krea's noise seed (LIVE.md, 15 Nov: the replay runs two
        # noise seeds from the one start image whose LCD the 7-segment writer and reader handle)
        start = Image.open(ROOT / cfg["start_frame"] / f"{cfg['start_id']}-s{cfg.get('start_seed', seed)}.png").convert("RGB")

        def set_prompt(s, emb) -> None:  # as a scheduled switch in generate()
            s.current_prompt_embeds = emb
            if hasattr(s, "conditional_dict"):
                s.conditional_dict["prompt_embeds"] = emb
            models.pipeline._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=s.gpu)

        def dual_view(raw: list, shown: list, reads: list, want: str) -> list:
            """LIVE.md (16 Nov, Elad: "I want to see those options in rt visualizer"): each frame as Krea made it
            (option 2: every digit Krea's; the lock's reader's verdict on top) beside the frame after the lock (setup C:
            wrong digits replaced by the 7-segment writer). Krea's generation is the same for both (replay 0542: raw
            frames byte-identical with and without the on-screen rewrite)."""
            from PIL import ImageDraw, ImageFont
            from textbench.display_check import digits as only_digits
            try:
                font = ImageFont.load_default(size=20)
            except TypeError:  # Pillow < 10.1
                font = ImageFont.load_default()
            out = []
            for i, (a, b) in enumerate(zip(raw, shown)):
                r = reads[i] if i < len(reads) else None
                ok = r is not None and r == only_digits(want)
                tiles = []
                for img, label, colour in (
                        (a, f"Krea alone: reads {r or '-'}" if r is not None else "Krea alone (display not found)",
                         (40, 170, 70) if ok else (130, 130, 130) if r is None else (220, 50, 40)),  # code-dual-r1 #2
                        (b, "With the lock" + (" (corrected)" if b is not a else ""), (40, 110, 220))):
                    im = Image.fromarray(img)
                    d = ImageDraw.Draw(im)
                    d.rectangle((0, 0, im.width, 30), fill=(0, 0, 0))
                    d.rectangle((0, 0, 10, 30), fill=colour)
                    d.text((18, 4), label, fill=(255, 255, 255), font=font)
                    tiles.append(np.asarray(im))
                out.append(np.concatenate(tiles, 1))
            return out

        class KreaLive:
            def __init__(self) -> None:
                self.shown, self.applied, self.blocks = cfg["live_text"], 0, cfg.get("live_blocks", 400)
                from live_server import LabelBox
                self.label = LabelBox(cfg.get("live_max_chars", 5), cfg.get("live_align", "center"))
                self.label.update(np.asarray(start), self.shown)
                if WRITE["editor"] and self.label.box is not None:  # code-live-r2 #3: load klein before the first key
                    from textbench.edit_write import edit_prompt
                    t = time.perf_counter()
                    EDITOR["fn"].boxes = {self.shown: list(self.label.box)}
                    try:
                        EDITOR["fn"](np.asarray(start), edit_prompt([(self.shown, "0")]), seed)
                    finally:
                        EDITOR["fn"].boxes = None
                    EDITOR["fn"].timeout = 60.0  # warm now: a stuck edit fails the write, not the session
                    print(f"klein warmed up in {time.perf_counter() - t:.1f} s", flush=True)
                self.chunks: list = []
                self.last = None
                # LIVE.md Changes (15 Nov, the display checker): after every block OCR reads the display on the block's
                # last frame (textbench/display_check.py); a display read wrong on 2 blocks in a row is fixed before
                # the next block by writing the shown number over what it reads (klein; our font only if
                # `live_fix_font_after` is set, which Elad has not chosen). Every block's display crop is saved.
                self.start_box = self.label.box  # the reset writes on the start image, at the start's box
                self.check = {"ok": 0, "wrong": 0, "unread": 0}
                self.by_ctx: dict = {}  # code-check-r1 #6: verdicts by block kind (reset / write / fix / plain) and how
                self.fix, self.fix_tries, self.n_checked, self.fixes = None, 0, 0, 0
                self.wrong_prev = self.last_wrong = None  # r1 #1: the previous block's sure wrong reading; #7: the last
                self.crops = ROOT / "logs/live" / f"{os.environ.get('LAUNCH', 'local')}_crops"
                # LIVE.md (15 Nov, the 7-segment writer): with live_writer "lcd" the display is written by
                # textbench/write_lcd.py, not klein: the LCD found once on the start image and tracked, the digits drawn
                # in DSEG7 in the start's places. live_lock: after every block, frames whose display reads wrong
                # (textbench/seg7.py) are rewritten before they are shown, and so is the memory if its newest frame
                # reads wrong. live_pin: the display's latents are steered to the written memory's in every denoising
                # pass, the last (t=0) included. live_anchor: the written newest frame is kept in the oldest memory
                # slot (step 28's anchor).
                self.lcd = None
                self.klein_off = False  # set after a klein timeout (restyle_safe)
                if cfg.get("live_writer") == "lcd":
                    from textbench.write_lcd import LcdTracker, layout
                    st = np.asarray(start)
                    # LIVE.md (16 Nov, the push-in run): live_track_max_scale widens the tracker's scales (default 1.4)
                    top = cfg.get("live_track_max_scale")
                    self.lcd = LcdTracker(st, self.start_box, **({"scales": tuple(np.round(np.arange(0.8, top + 0.01, 0.04), 2))}
                                                                 if top else {}))
                    # LIVE.md 16 Nov (start images s0, s2): the old digits only inside OCR's box of them (a "%" or a
                    # stray mark beside them widened the pitch); s1's layout is identical either way (checked offline)
                    from textbench.display_check import read_display
                    g0 = self.lcd.geometry(st, old_digits=len(self.shown), digits_box=read_display(st, self.start_box)["box"])
                    if g0 is None:
                        raise RuntimeError("the LCD's digits were not found on the start image")
                    self.lay0 = layout(g0, str(ROOT), cfg.get("live_max_chars", 5))
                    self.identity = np.array([[1.0, 0, 0], [0, 1.0, 0]])
                    from textbench.seg7 import MaskCache
                    self.masks = MaskCache(self.lay0)  # replay 2103 fix: read every frame cheaply
                # LIVE.md 17 Nov (route 1): live_writer "paste": the general text writer (textbench/paste_writer.py)
                # for any surface: the field (live_text) found by OCR on the start image inside its line (live_line,
                # default the field alone); a field inside an OCR line is written as the whole line from live_line
                # with the field replaced; the line tracked by an ORB homography; no 7-segment reader
                self.paster = None
                if cfg.get("live_writer") == "paste":
                    from textbench.paste_writer import TextTracker, find_field
                    st = np.asarray(start)
                    line = cfg.get("live_line", cfg["live_text"])
                    hit = find_field(st, line, cfg["live_text"])
                    if hit is None:
                        raise RuntimeError(f"the text {line!r} was not found on the start image")
                    whole = "line_quad" in hit
                    q = hit["line_quad"] if whole else hit["quad"]
                    self.paster = {"tracker": TextTracker(st, q), "quad": q,
                                   "tmpl": line.upper().replace(cfg["live_text"].upper(), "{text}", 1) if whole else "{text}",
                                   "align": cfg.get("live_paste_align", "center"), "ink": cfg.get("live_paste_ink")}
                self.need_pin = False
                self.lora = None
                # LIVE.md (16 Nov, feature steering step 1): hidden states at the digit places (runner/feature_probe.py)
                self.probe, self.frame_places, self.steps_done = None, [], 0  # steps_done: the replay's block number
                # LIVE.md (16 Nov, the LoRA pilot, research/plans/LORA_PILOT.md): live_collect saves each block's
                # transformer inputs and a corrected target (textbench/lora_targets.py) for runner/lora_train.py
                self.collect, self.calls, self.prev_raw = cfg.get("live_collect"), [], None
                if self.collect:
                    assert self.lcd is not None and cfg.get("live_lock") and not cfg.get("live_lock_frames", True), \
                        "live_collect needs the 7-segment reader and Krea's own frames (live_lock_frames false)"
                    self.collect_dir = ROOT / "logs/live" / os.environ.get("LAUNCH", "local") / "lora_samples"
                    self.collect_dir.mkdir(parents=True, exist_ok=True)
                    self.collect_on, self.prompts_seen = False, {}
                    self.collect_hook = models.transformer.register_forward_pre_hook(self.on_call, with_kwargs=True)
                if cfg.get("live_probe"):
                    assert self.lcd is not None and cfg.get("live_lock"), "live_probe needs the 7-segment lock's reader"
                    from feature_probe import FeatureProbe
                    self.probe = FeatureProbe(models.transformer, cfg["live_probe"]["layers"], calls=cfg["steps"])
                self.new_session(start)  # sets need_pin (self.lcd exists by now)
                # LIVE.md (16 Nov, the LoRA pilot's evaluation): live_lora attaches the adapters (runner/lora_train.py)
                # to Krea's layers for this arm; live() puts the original layers back when the arm ends
                # (last in __init__: nothing after the merge can fail before live() holds the generator)
                self.lora_orig = None
                if cfg.get("live_lora"):
                    from lora_train import attach_lora, merge_lora
                    ck = torch.load(ROOT / cfg["live_lora"], map_location="cpu")
                    self.lora = (ck["state"], ck["scale"])
                    if cfg.get("live_lora_merge"):
                        # LIVE.md 16 Nov (next steps 1): the adapters added into Krea's bf16 weights (no extra cost per
                        # block); the original weights kept on the CPU and put back when the arm ends
                        n, self.lora_orig, err = merge_lora(models.transformer.model, *self.lora, keep=True)
                        print(f"LoRA {cfg['live_lora']} (step {ck['step']}): {n} layers merged into bf16; "
                              f"update kept after rounding: {err}", flush=True)
                    else:
                        print(f"LoRA {cfg['live_lora']} (step {ck['step']}): {attach_lora(models.transformer.model, *self.lora)} layers with adapters", flush=True)


            def new_session(self, img) -> None:
                self.prompt_now = tmpl.format(text=self.shown)
                params = GenerateParams(prompt=tmpl.format(text=self.shown), seed=seed, num_blocks=self.blocks + 1,
                                        num_denoising_steps=cfg["steps"], kv_cache_num_frames=cfg["kv_cache_num_frames"])
                cls = AnchoredSession if cfg.get("live_anchor") else GenerationSession
                with torch.inference_mode():
                    self.session = cls(params=params, config=config, frame_callback=self.on_frames, models=models)
                    self.session.setup_start_frame(img, models)
                self.b = 0
                GUIDE.update(lat=None, mask=None, final=False)  # a new session: the pin is set again by a write
                self.prev_raw = None  # live_collect: a target's encode needs the same session's previous frame
                # code-replay-r1 #5: every LCD arm writes the shown number into memory once after a (re)start (the pin
                # needs it for its target), so arms B, C and D differ only in the lock and the pin
                self.need_pin = (getattr(self, "lcd", None) is not None) or (getattr(self, "paster", None) is not None)

            def on_frames(self, pixels, _ids, event) -> None:
                event.synchronize()
                self.chunks.append(pixels[0].float().add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
                                   .permute(0, 2, 3, 1).cpu().numpy())

            def check_display(self, frame: np.ndarray, kind: str) -> dict:
                """Read the display on the block's last frame; count the verdict; follow the display if it moved;
                queue a fix when it read other digits, surely (detector, score), on this block and the one before."""
                from textbench.display_check import digits, follow, read_display, verdict
                r = read_display(frame, self.label.box)
                v = verdict(r["reads"], self.shown)
                self.check[v] += 1
                self.n_checked += 1
                for key in (f"{kind}:{v}", f"{r['how']}:{v}"):
                    self.by_ctx[key] = self.by_ctx.get(key, 0) + 1
                sure = r["score"] >= cfg.get("live_fix_score", 0.8) and r["how"] == "det"  # r1 #2: never from "rec"
                if r["box"] is not None and sure and (v == "ok" or len(r["reads"]) == len(digits(self.shown))):
                    self.label.box = follow(self.label.box, r["box"], self.label.align, anchor=self.start_box,
                                            shape=frame.shape)  # r1 #1, #9: only a plausible reading moves the box
                if v == "ok":
                    self.fix_tries, self.last_wrong = 0, None
                wrong = v == "wrong" and sure
                if wrong:
                    self.last_wrong = r["reads"]
                confirmed = wrong and self.wrong_prev is not None  # r1 #1: one OCR read never starts a rewrite
                self.wrong_prev = r["reads"] if wrong else None
                fixes = cfg.get("live_fix_max", 4)
                self.fix = (r["reads"], self.label.box) if confirmed and self.fix_tries < fixes else None
                x0, y0, x1, y1 = self.label.box
                self.crops.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.ascontiguousarray(frame[max(0, y0 - 30):y1 + 30, max(0, x0 - 110):x1 + 110])).save(
                    self.crops / f"{self.n_checked:05d}_{r['reads'] or 'none'}_{v}.jpg", quality=85)
                total = sum(self.check.values())
                return {"reads": r["reads"], "score": r["score"], "how": r["how"], "verdict": v, "kind": kind,
                        "want": self.shown, "ok_pct": round(100 * self.check["ok"] / total, 1), "checked": total,
                        "counts": dict(self.check), "by_kind": dict(self.by_ctx), "fixes": self.fixes,
                        "fix_queued": self.fix is not None}

            def fix_display(self, s) -> dict:
                """Write the shown number over what the display reads (as a typed change, with the prompt unchanged);
                our font instead of klein once `live_fix_font_after` fixes in a row have not made it right."""
                reads, box = self.fix
                self.fix = None
                after = cfg.get("live_fix_font_after")
                font = bool(WRITE["editor"]) and after is not None and self.fix_tries >= after
                saved = WRITE["editor"]
                WRITE["editor"] = False if font else saved
                WRITE["fixed_boxes"] = {reads: box}
                out: dict = {"from": reads, "to": self.shown, "font": font, "try": self.fix_tries + 1}
                try:
                    out["boxes"] = write_memory(s, [(reads, self.shown)], self.label.align)[0]
                except Exception as ex:  # an editor error is a failed fix, not the end of the session
                    out["boxes"], out["error"] = 0, f"{type(ex).__name__}: {ex}"[:300]
                finally:
                    WRITE["editor"], WRITE["fixed_boxes"] = saved, None
                if WRITE["edits"]:
                    e = WRITE["edits"][-1]
                    out["edit"] = {k: e.get(k) for k in ("editor_s", "server_s", "changed", "found")}
                    WRITE["edits"].clear()
                self.fix_tries += 1
                self.fixes += 1
                self.wrong_prev = None  # the fixed display needs 2 more wrong readings before another fix
                return out

            def lcd_read(self, rgb: np.ndarray, affine=None, masks=None) -> tuple[str, dict | None, list | None]:
                """The display read by its segments at the start's digit places (moved with the tracked LCD)."""
                from textbench.seg7 import read as seg7_read
                g = self.lcd.geometry(rgb, affine=affine)
                if g is None:
                    return "", None, masks
                return seg7_read(g["dark"], self.masks.for_geo(g))["reads"], g, None

            def lcd_draw(self, rgb: np.ndarray, text: str, g: dict) -> np.ndarray:
                from textbench.write_lcd import write_lcd
                return write_lcd(rgb, None, text, str(ROOT), geo=g, lay=self.lay0)

            def lcd_memory(self, s, text: str) -> int:
                """Write `text` on the display in all 9 memory frames (the LCD placed once, on the newest) and
                re-encode the last two latents, as write_memory does; then the anchor and the pin."""
                cache = list(s.frame_context_cache)
                if len(cache) != s.frame_context_cache.maxlen or s.current_start_frame < 2:
                    return 0
                newest = to_rgb(cache[-1])
                a, score = self.lcd.locate(newest)
                if score < 0.4:
                    return 0
                outs, geos = [], []
                for c in cache:
                    rgb = to_rgb(c)
                    g = self.lcd.geometry(rgb, affine=a)
                    outs.append(self.lcd_draw(rgb, text, g) if g is not None else rgb)
                    geos.append(g)
                drawn = sum(g is not None for g in geos)
                if not drawn:  # code-live-lcd-r1 #3: the LCD matched but no frame could be drawn: not a write
                    return 0
                if cfg.get("live_restyle"):  # LIVE.md (16 Nov): klein restyles the font write, kept if OCR reads it
                    outs = self.restyle_safe(outs, geos, text, a)
                edited = [to_tensor(o, c) if g is not None else c for o, g, c in zip(outs, geos, cache)]
                g_new = next((g for g in reversed(geos) if g is not None), None)
                s.frame_context_cache = type(s.frame_context_cache)(edited, maxlen=s.frame_context_cache.maxlen)
                st = s.current_start_frame
                lat = encode9(edited)
                s.all_latents[:, st - 2:st] = lat[:, 1:3].to(s.all_latents)
                if cfg.get("live_anchor"):
                    s.anchor = edited[-1]
                if cfg.get("live_pin") and g_new is not None:
                    self.set_pin(lat, g_new)
                return drawn

            def paste_memory(self, s, text: str) -> int:
                """live_writer "paste": `text` written (textbench/paste_writer.write) at the tracked line in all 9
                memory frames (placed once, on the newest), the last two latents re-encoded, as lcd_memory."""
                from textbench.paste_writer import write
                cache = list(s.frame_context_cache)
                if len(cache) != s.frame_context_cache.maxlen or s.current_start_frame < 2:
                    return 0
                q, _n = self.paster["tracker"].locate(to_rgb(cache[-1]))
                if q is None:
                    return 0
                words = self.paster["tmpl"].format(text=text.upper())
                outs = [write(to_rgb(c), q, words, self.paster["align"], seed=i, ink_hint=self.paster["ink"])[0]
                        for i, c in enumerate(cache)]
                edited = [to_tensor(o, c) for o, c in zip(outs, cache)]
                s.frame_context_cache = type(s.frame_context_cache)(edited, maxlen=s.frame_context_cache.maxlen)
                st = s.current_start_frame
                lat = encode9(edited)
                s.all_latents[:, st - 2:st] = lat[:, 1:3].to(s.all_latents)
                if cfg.get("live_anchor"):
                    s.anchor = edited[-1]
                return len(edited)

            def paste_read(self, frame: np.ndarray) -> tuple[str, bool, int]:
                """OCR's reading at the tracked line (lines whose centre lies in it, padded by a third of its height
                along it, 0.15 across), whether it is the shown text, and the tracker's inliers (<= 0: lost)."""
                from textbench.paste_writer import norm, ocr_lines, same
                q, n_in = self.paster["tracker"].locate(frame)
                if q is None:
                    return "", False, n_in
                h = float(np.linalg.norm(q[3] - q[0]))
                x0, y0 = q.min(0) - np.array([h / 3, h * 0.15])  # rows above or below are other lines
                x1, y1 = q.max(0) + np.array([h / 3, h * 0.15])
                got = sorted((l[:, 0].min(), t) for l, t, _ in ocr_lines(frame)
                             if x0 <= l[:, 0].mean() <= x1 and y0 <= l[:, 1].mean() <= y1)
                read = "".join(norm(t) for _, t in got)
                return read, same(read, self.paster["tmpl"].format(text=self.shown.upper())), n_in

            def paste_lock(self, frames: list, s) -> dict:
                """The memory-only lock (setup E's) for live_writer "paste": the newest frame read by OCR at the
                tracked line; when it is not the shown text, the memory is written again. Frames are shown as made."""
                read, ok, n_in = self.paste_read(frames[-1])
                mem = 0 if ok or n_in <= 0 else self.paste_memory(s, self.shown)  # lost: no blind rewrite
                return {"raw_last": read, "ok": ok, "memory_fixed": bool(mem), "inliers": n_in}

            def restyle_safe(self, outs: list, geos: list, text: str, a) -> list:
                """restyle, any error keeping the font write (code-klein-r1 should-fix 2); after an editor timeout klein
                is not asked again in this run (should-fix 1: a dead server would cost 2 x 60 s per write)."""
                if self.klein_off:
                    self.restyles.append({"text": text, "how": "font", "skipped": "klein off after a timeout"})
                    return outs
                try:
                    return self.restyle(outs, geos, text, a)
                except Exception as ex:
                    self.restyles.append({"text": text, "how": "font", "error": f"{type(ex).__name__}: {ex}"[:300]})
                    return outs

            def restyle(self, outs: list, geos: list, text: str, a) -> list:
                """LIVE.md (16 Nov, Elad: "I want the digits to be part of the scene"; textbench/restyle.py): klein
                (runner/edit_server.py mode "restyle") restyles the square around the LCD of the newest font-written
                frame; seed by seed (live_restyle_seeds), the first answer on which OCR reads the wanted number goes
                into every drawn frame's LCD (colour-fixed per frame, the LCD placed once as the font's); none: the font
                write stays. Each call is logged in self.restyles."""
                from textbench.display_check import digits, read_display
                from textbench.restyle import EDGE, PROMPT, box_at, lcd_mask, put, square_crop
                k = max(i for i, g in enumerate(geos) if g is not None)
                font, want = outs[k], digits(text)
                lcd = lcd_mask(geos[k], font.shape)
                info: dict = {"text": text, "how": "font", "reads": [], "klein_s": 0.0, "ocr_s": 0.0}
                self.restyles.append(info)
                if not lcd.any():
                    info["error"] = "empty LCD mask"
                    return outs
                crop = square_crop(lcd)
                x0, y0, x1, y1 = crop
                inp = np.asarray(Image.fromarray(np.ascontiguousarray(font[y0:y1, x0:x1]))
                                 .resize((EDGE, EDGE), Image.Resampling.LANCZOS))
                box = box_at(self.start_box, a)
                saved, chosen = EDITOR["fn"].mode, None
                EDITOR["fn"].mode = "restyle"
                try:
                    for sd in cfg.get("live_restyle_seeds", [0, 1]):
                        t = time.perf_counter()
                        ans = EDITOR["fn"](inp, PROMPT.format(n=text), sd)
                        info["klein_s"] += time.perf_counter() - t
                        info.setdefault("server_s", []).append(round(EDITOR["fn"].last.get("seconds", -1), 3))  # nit 4
                        back = np.asarray(Image.fromarray(ans).resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS))
                        t = time.perf_counter()
                        r = read_display(put(font, back, crop, lcd), box)["reads"]
                        info["ocr_s"] += time.perf_counter() - t
                        info["reads"].append(r)
                        if r == want:
                            chosen, info["how"] = back, f"klein s{sd}"
                            break
                except Exception as ex:  # an editor or OCR error keeps the font write, never ends the session
                    info["error"] = f"{type(ex).__name__}: {ex}"[:300]
                    if isinstance(ex, TimeoutError):
                        self.klein_off = True
                finally:
                    EDITOR["fn"].mode = saved
                    info["klein_s"], info["ocr_s"] = round(info["klein_s"], 3), round(info["ocr_s"], 3)
                if chosen is None:
                    return outs
                return [put(o, chosen, crop, lcd_mask(g, o.shape)) if g is not None else o for o, g in zip(outs, geos)]

            def on_call(self, _module, _args, kwargs) -> None:
                """live_collect: every transformer call of the block (0: the KV-cache rebuild, 1..steps: denoising)."""
                if self.collect_on:
                    self.calls.append({"x": kwargs["noisy_image_or_video"].detach().to("cpu", torch.bfloat16).clone(),
                                       "t": kwargs["timestep"].detach().cpu().clone(),
                                       "start": int(kwargs.get("current_start") or 0)})

            def save_sample(self, raw_frames: list, typed: str, info: dict) -> dict:
                """The block's inputs (rebuild context, the denoising inputs and timesteps, Krea's output latents) and
                its target: Krea's latents plus the change the corrected frames make in the VAE's latent space
                (enc(corrected) - enc(as made), both from the previous block's last frame + these 12), so the VAE's
                own round-trip error cancels outside the display; loss weights per latent cell."""
                from textbench import lora_targets
                s = self.session
                out = {"saved": False}
                if len(self.calls) != 1 + cfg["steps"] or len(raw_frames) != 12 or self.prev_raw is None:
                    out["why"] = f"calls {len(self.calls)} frames {len(raw_frames)} prev {self.prev_raw is not None}"
                    return out
                editor = None
                if WRITE["editor"] and not self.klein_off:
                    def editor(img, prompt, seed):
                        saved = EDITOR["fn"].mode
                        EDITOR["fn"].mode = "restyle"
                        try:
                            return EDITOR["fn"](img, prompt, seed)
                        except TimeoutError:
                            self.klein_off = True  # code-lora-r1 should-fix 1: a dead server costs one timeout
                            raise
                        finally:
                            EDITOR["fn"].mode = saved
                res = lora_targets.build(raw_frames, self.shown, self.lcd, self.masks, str(ROOT), self.lay0,
                                         self.start_box, editor=editor)
                base = s.last_pred.detach()
                changed = any(r is not o for r, o in zip(res["frames"], raw_frames))
                target = base.clone()
                if changed:
                    like = s.frame_context_cache[-1]
                    def enc(frames):
                        t = torch.cat([to_tensor(f, like)[0] for f in [self.prev_raw] + list(frames)]).half()
                        return encode_video_latent(models.vae_encoder, [None] * 55, resample_to=16, max_frames=81,
                                                   video_path_or_url=None, frames=t, height=480,
                                                   width=832, stream=False)[0].transpose(0, 1)[None][:, 1:]
                    delta = enc(res["frames"]) - enc(raw_frames)
                    target = (base.float() + delta.float().to(base.device)).to(base.dtype)
                w = lora_targets.latent_weights(res["lcd"], res["verified"], inside=1.0,
                                                outside=self.collect.get("outside", 0.1))
                p = s.current_prompt_embeds
                key = self.prompt_now
                if key not in self.prompts_seen:
                    self.prompts_seen[key] = p.detach().to("cpu", torch.bfloat16).clone()
                k = len(list(self.collect_dir.glob("*.pt")))
                name = f"{os.getpid()}_{self.steps_done:04d}_{k:05d}"
                torch.save({"ctx": self.calls[0], "steps": self.calls[1:], "base": base.to("cpu", torch.bfloat16),
                            "target": target.to("cpu", torch.bfloat16), "weight": torch.from_numpy(w),
                            "prompt": key, "typed": typed, "shown": self.shown, "info": res["info"],
                            "how": res["how"], "verified": res["verified"]}, self.collect_dir / f"{name}.pt")
                if changed:  # a before/after picture of the most changed frame, for the sample sheet
                    i = max(range(12), key=lambda i: int(np.abs(res["frames"][i].astype(int) - raw_frames[i]).sum()))
                    Image.fromarray(np.concatenate([raw_frames[i], res["frames"][i]], 1)).save(
                        self.collect_dir / f"{name}_f{i}_{res['how'][i].replace(' ', '')}.jpg", quality=88)
                out.update(saved=True, name=name, changed=changed, right=res["info"]["right"],
                           verified=res["info"]["verified"], source=res["info"]["source"])
                return out

            def place_labels(self, g: dict) -> list:
                """Each digit place's box in frame pixels (its 7 segments' masks) and the digit read there ("0".."9",
                " " nothing lit, "?" a pattern that is no digit), for runner/feature_probe.py."""
                from textbench.seg7 import DECODE, read as seg7_read
                places = self.masks.for_geo(g)
                pats = seg7_read(g["dark"], places)["patterns"]
                x0, y0 = g["window"][:2]
                out = []
                for pl, pat in zip(places, pats):
                    m = np.zeros_like(next(iter(pl.values())))
                    for seg in pl.values():
                        m |= seg
                    ys, xs = np.nonzero(m)
                    if not len(xs):
                        out.append(((0, 0, 0, 0), "?"))
                        continue
                    lab = DECODE.get(frozenset(pat), "?") if pat else " "
                    out.append(((x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1), lab))
                return out

            def pin_follow(self, s) -> bool:
                """The pin's target moved to where the LCD is now (replay 2103: a target fixed where it was written
                pasted the digits beside the drifting LCD): the shown number written on copies of the 9 memory frames
                at the LCD's current placement, encoded, and set as the target over the LCD's current cells; the
                memory itself is not changed."""
                cache = list(s.frame_context_cache)
                if len(cache) != s.frame_context_cache.maxlen:
                    return False
                a, score = self.lcd.locate(to_rgb(cache[-1]))
                if score < 0.4:
                    GUIDE.update(lat=None, mask=None, final=False)  # LCD lost: no pin rather than a wrong one
                    return False
                edited, g_new = [], None
                for c in cache:
                    rgb = to_rgb(c)
                    g = self.lcd.geometry(rgb, affine=a)
                    edited.append(to_tensor(self.lcd_draw(rgb, self.shown, g), c) if g is not None else c)
                    g_new = g or g_new
                if g_new is None:
                    GUIDE.update(lat=None, mask=None, final=False)  # code-replay-r2 #3: no stale pin
                    return False
                self.set_pin(encode9(edited), g_new)
                return True

            def set_pin(self, lat: torch.Tensor, g: dict) -> None:
                """The pin's target: the written newest memory latent over the LCD's latent cells (8 px; inside the
                LCD by one cell, so the edge with the bezel stays Krea's), for every latent of the next blocks."""
                import cv2
                full = np.zeros((480, 832), np.float32)
                x0, y0, x1, y1 = g["window"]
                full[y0:y1, x0:x1] = g["filled"]
                cells = cv2.erode((cv2.resize(full, (104, 60), interpolation=cv2.INTER_AREA) > 0.6).astype(np.uint8),
                                  np.ones((3, 3), np.uint8)).astype(bool)
                target = lat[0, -1:].repeat(models.pipeline.num_frame_per_block, 1, 1, 1)
                GUIDE.update(lat=target, mask=torch.from_numpy(cells).to(target.device), final=True)
                self.need_pin = False

            def lock(self, frames: list, s) -> dict:
                """After a block: every frame whose display reads other than the shown number is rewritten before it
                is shown, the LCD tracked in each frame (replay 2103: placed once per block, the frames of a block
                where Krea panned got their digits beside the LCD), and the memory too if its newest frame reads
                wrong.
                LIVE.md (16 Nov, option 2): live_lock_frames false: the frames are read but shown as Krea made them
                (only the memory is corrected); live_lock_memory "always": the memory is rewritten after every block,
                also when it reads right (each block starts from clean digits)."""
                from textbench.display_check import digits
                want = digits(self.shown)
                rewrite = cfg.get("live_lock_frames", True)
                fixed, wrong, raw, lost, scores, raw7 = 0, 0, "", 0, [], ""
                self.frame_reads = [None] * len(frames)  # the reader's reading of each raw frame (None: LCD lost)
                self.frame_places = [None] * len(frames)  # with live_probe: per digit place, (box, digit read)
                import hashlib  # code-memlock-r1 #1: Krea's raw newest frame, before any rewrite (C vs E per block)
                raw_md5 = hashlib.md5(np.ascontiguousarray(frames[-1]).tobytes()).hexdigest()[:12] if frames else ""
                # code-best-r1 blocker: the 7-segment reader misses Krea's klein-style digits (the K replay: 77-84% of
                # frames "wrong" against OCR's 48% right), so with live_restyle OCR reads the block's last frame once;
                # if it reads the wanted number, the block's frames are taken as right (not rewritten; labelled so)
                ocr_ok = None
                if cfg.get("live_restyle") and frames:
                    try:
                        from textbench.display_check import read_display
                        from textbench.restyle import box_at
                        a_last, sc_last = self.lcd.locate(frames[-1], min_score=0.4)
                        if sc_last >= 0.4:
                            ocr_ok = read_display(frames[-1], box_at(self.start_box, a_last))["reads"] == want
                    except Exception:
                        ocr_ok = None
                for i, f in enumerate(frames):
                    # code-replay-r2 #1: once a frame of this block is placed, a near match >= 0.4 is kept for the
                    # next (a wide search per frame would cost ~0.1-0.3 s each)
                    a, score = self.lcd.locate(f, min_score=0.4 if i > lost else 0.6)
                    scores.append(score)
                    if score < 0.4:
                        lost += 1
                        continue
                    r, g, _ = self.lcd_read(f, affine=a)
                    if i == len(frames) - 1:
                        raw7 = r  # code-best-r2 nit 1: the 7-segment reading itself, before any OCR rescue
                    if ocr_ok and r != want:
                        r = want  # OCR read the block's last frame right: the 7-segment miss is the reader's
                    self.frame_reads[i] = r
                    if self.probe is not None and g is not None:
                        self.frame_places[i] = self.place_labels(g)
                    if i == len(frames) - 1:
                        raw = r
                    if g is not None and r != want:
                        wrong += 1
                        if rewrite:
                            frames[i] = self.lcd_draw(f, self.shown, g)
                            fixed += 1
                a = self.lcd.affine
                score = min(scores) if scores else 0.0
                mem_rgb = to_rgb(s.frame_context_cache[-1])
                mem_read, mem_g, _ = self.lcd_read(mem_rgb)
                mem_ok, mem_ocr = mem_read == want, None
                if not mem_ok and cfg.get("live_restyle") and mem_g is not None:
                    # klein's digits sit off the font's segment places, so the 7-segment reader can miss them in
                    # Krea's continuation: the memory also counts as right when OCR reads the wanted number
                    from textbench.display_check import read_display
                    from textbench.restyle import box_at
                    mem_ocr = read_display(mem_rgb, box_at(self.start_box, mem_g["affine"]))["reads"]
                    mem_ok = mem_ocr == want
                always = cfg.get("live_lock_memory", "wrong") == "always"
                mem = self.lcd_memory(s, self.shown) if always or not mem_ok else 0
                return {"raw_last": raw, "raw_md5": raw_md5, "frames_fixed": fixed, "frames_wrong": wrong,
                        "ocr_block_ok": ocr_ok, "raw_last_seg7": raw7,
                        "memory_read": mem_read, "memory_ocr": mem_ocr,
                        "memory_fixed": bool(mem),
                        "frames_lost": lost, "score": round(score, 3), "scale": round(float(a[0, 0]), 3),
                        "dx": round(float(a[0, 2]), 1), "dy": round(float(a[1, 2]), 1)}

            def step(self, typed: str, seq: int):
                info: dict = {"wrote": False}
                self.restyles: list = []  # klein restyle calls this block (KreaLive.restyle)
                self.frame_places = []  # code-probe-r1 nit: per block, as frame_reads
                with torch.inference_mode():
                    reset = cfg.get("live_reset_blocks")
                    if reset and self.b >= reset:
                        # LIVE.md Changes (15 Nov): Krea drifts away from the scene after ~30 blocks (the calculator
                        # turned and its display left the frame), so every `reset` blocks it starts again from the
                        # start image with the shown text written on it at the display's box
                        img = np.asarray(start)
                        if self.paster is not None and self.shown != cfg["live_text"]:
                            from textbench.paste_writer import write as paste_write
                            img = paste_write(img, self.paster["quad"], self.paster["tmpl"].format(text=self.shown.upper()),
                                              self.paster["align"], ink_hint=self.paster["ink"])[0]
                            info["reset_write"] = "paste"
                        elif self.lcd is not None and self.shown != cfg["live_text"]:
                            g = self.lcd.geometry(img, affine=self.identity)
                            if g is not None:
                                img = self.lcd_draw(img, self.shown, g)
                                info["reset_write"] = "lcd"
                                if cfg.get("live_restyle"):  # the start image's digits restyled as the memory's
                                    img = self.restyle_safe([img], [g], self.shown, self.identity)[0]
                                    info["reset_restyle"] = self.restyles[-1]
                        elif self.shown != cfg["live_text"] and self.start_box is not None and not cfg.get("live_prompt_only"):  # code-pscene-r1
                            done = False
                            if WRITE["editor"]:  # klein restyles it, as every live change (code-live-r2 #1: through
                                from textbench.edit_write import edit_cache  # edit_cache: mask, colour match, check)
                                try:
                                    out, k, e = edit_cache([img], [(cfg["live_text"], self.shown)], EDITOR["fn"],
                                                           str(ROOT), seed, track=False,
                                                           boxes_by_old={cfg["live_text"]: self.start_box})
                                    info["reset_edit"] = {"n": k, "editor_s": e.get("editor_s")}
                                    if k:
                                        img, done = out[0], True
                                except Exception as ex:  # code-live-r2 #2: an editor error does not end the session
                                    info["edit_error"] = f"{type(ex).__name__}: {ex}"[:300]
                            if not done:
                                img = write_text(img, [(cfg["live_text"], self.shown)], str(ROOT), self.label.align,
                                                 {cfg["live_text"]: self.start_box})[0]
                        self.session.dispose()
                        self.new_session(Image.fromarray(img))
                        info["reset"] = True
                        self.label.box, self.fix = self.start_box, None  # the start image's display, freshly written
                        if self.lcd is not None:
                            self.lcd.reset()
                        self.fix_tries, self.wrong_prev, self.last_wrong = 0, None, None  # code-check-r1 #5
                    elif self.b >= self.blocks - 1 and self.last is not None:  # a long session continues afresh
                        self.session.dispose()
                        self.new_session(Image.fromarray(self.last))
                        info["restarted"] = True
                    t0 = time.perf_counter()
                    s = self.session
                    # code-live-r1 #2: no switch before the session's first block (Krea builds conditional_dict there;
                    # write_memory needs a full memory anyway)
                    if self.lcd is not None:  # the 7-segment display holds digits only, at most live_max_chars of them
                        typed = "".join(ch for ch in typed if ch in "0123456789")[-cfg.get("live_max_chars", 5):]
                    if typed != self.shown and typed.strip() and s.block_idx >= 1:  # r2 #4: no empty display
                        prev = s.current_prompt_embeds
                        new_prompt = switch_tmpl.format(text=typed)
                        changed = new_prompt != getattr(self, "prompt_now", None)  # code-live-r2 #7: same prompt (no
                        if changed:                                                  # number in it): no re-encode
                            emb = models.text_encoder(text_prompts=[new_prompt])["prompt_embeds"].to(dtype=torch.bfloat16)
                            if prev is not None:
                                emb = emb.to(prev.device)
                            set_prompt(s, emb)
                        box = self.label.update(to_rgb(s.frame_context_cache[-1]), self.shown)
                        # the display checker's last reading, when it was wrong, is what the display shows: the edit
                        # changes that (klein's prompt names it) rather than the number the loop believed was there
                        old = self.last_wrong or self.shown  # code-check-r1 #7: kept even when the fix budget is out
                        if cfg.get("live_prompt_only"):
                            # LIVE.md 17 Nov (option 1, new scenes): only the prompt names the new number; nothing is
                            # written into memory (no display reader needed: any scene)
                            n = 1
                        elif self.paster is not None:
                            try:
                                n = self.paste_memory(s, typed)
                            except Exception as ex:
                                n = 0
                                info["edit_error"] = f"{type(ex).__name__}: {ex}"[:300]
                        elif self.lcd is not None:  # the 7-segment writer (no klein, no OCR)
                            try:
                                n = self.lcd_memory(s, typed)
                            except Exception as ex:
                                n = 0
                                info["edit_error"] = f"{type(ex).__name__}: {ex}"[:300]
                        elif self.wrong_prev == typed:  # r1 #8 (r2 #1: read so on the block just before): accept as is
                            n, info["already_shown"] = 1, True
                        else:
                            old = old if old != typed else self.shown
                            WRITE["fixed_boxes"] = {old: box} if box is not None else None
                            try:
                                n, _ = write_memory(s, [(old, typed)], self.label.align)
                            except Exception as ex:  # code-live-r2 #2: an editor error is a failed write, not the end
                                n = 0
                                info["edit_error"] = f"{type(ex).__name__}: {ex}"[:300]
                            WRITE["fixed_boxes"] = None
                        if WRITE["edits"]:  # klein's call: editor_s (round trip), server_s, share of the box changed
                            e = WRITE["edits"][-1]
                            info["edit"] = {k: e.get(k) for k in ("editor_s", "server_s", "changed", "found")}
                            WRITE["edits"].clear()
                        info["box"] = box
                        info["boxes"] = n
                        if n:  # applied only when written
                            self.shown, self.applied, info["wrote"] = typed, seq, True
                            self.prompt_now = new_prompt
                            self.fix, self.fix_tries, self.wrong_prev, self.last_wrong = None, 0, None, None
                            self.need_pin = False  # the typed write is this (re)start's memory write, in every arm
                        elif prev is not None and changed:  # code-live-r1 #4: not written: back to the shown prompt
                            set_prompt(s, prev)  # (as step 25c), or Krea would draw the new number itself
                            info["write_failed"] = True
                    elif self.fix is not None and s.block_idx >= 1:
                        info["fix"] = self.fix_display(s)
                    elif self.lcd is not None and cfg.get("live_pin") and not self.need_pin and s.block_idx >= 1:
                        tp = time.perf_counter()
                        try:  # replay 2103 fix: the pin follows the tracked LCD every block
                            info["pin_follow"] = self.pin_follow(s)
                        except Exception as ex:
                            GUIDE.update(lat=None, mask=None, final=False)
                            info["edit_error"] = f"{type(ex).__name__}: {ex}"[:300]
                        info["pin_s"] = round(time.perf_counter() - tp, 3)
                    if (self.lcd is not None or self.paster is not None) and self.need_pin and s.block_idx >= 1 \
                            and not info.get("wrote"):
                        try:  # after a (re)start: the shown number written once into memory (the pin's target)
                            info["post_reset_write"] = (self.paste_memory(s, self.shown) if self.paster is not None
                                                        else self.lcd_memory(s, self.shown))
                            self.need_pin = False
                        except Exception as ex:
                            info["edit_error"] = f"{type(ex).__name__}: {ex}"[:300]
                    t1 = time.perf_counter()
                    self.chunks = []
                    if self.probe is not None:
                        self.probe.arm()
                    if self.collect:
                        self.calls, self.collect_on = [], True
                    try:
                        self.session.generate_block(models)
                    finally:
                        if self.probe is not None:
                            self.probe.disarm()
                        if self.collect:
                            self.collect_on = False
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()  # code-live-r1 #9: block timing ends here, before the OCR below
                    self.b += 1
                    frames = [f for c in self.chunks for f in c]
                    if frames:
                        self.last = frames[-1]
                    if info["wrote"] and self.lcd is None:
                        info["reads_after"] = " | ".join(ocr_texts(self.session.frame_context_cache[-1]))
                    info["ocr_s"] = round(time.perf_counter() - t2, 3)
                    if self.paster is not None and frames and cfg.get("live_lock"):
                        t4 = time.perf_counter()
                        try:  # a lock or reader error is logged, never the end of the session
                            info["lock"] = self.paste_lock(frames, self.session)
                        except Exception as ex:
                            info["lock_error"] = f"{type(ex).__name__}: {ex}"[:300]
                        info["lock_s"] = round(time.perf_counter() - t4, 3)
                    if self.lcd is not None and frames:
                        t4 = time.perf_counter()
                        raw_frames = list(frames)  # the lock replaces list items (copies), never Krea's arrays
                        self.frame_reads = []
                        try:  # a lock or reader error is logged, never the end of the session
                            if cfg.get("live_lock"):
                                info["lock"] = self.lock(frames, self.session)
                            else:
                                r7, g7, _ = self.lcd_read(frames[-1])
                                info["seg7_last"] = r7
                                info["track"] = g7.get("track") if g7 else None
                        except Exception as ex:
                            info["lock_error"] = f"{type(ex).__name__}: {ex}"[:300]
                        info["lock_s"] = round(time.perf_counter() - t4, 3)
                        if self.collect:
                            tc = time.perf_counter()
                            try:  # a collection error is logged, never the end of the run
                                info["collect"] = self.save_sample(raw_frames, typed, info)
                            except Exception as ex:
                                info["collect_error"] = f"{type(ex).__name__}: {ex}"[:300]
                            info["collect_s"] = round(time.perf_counter() - tc, 3)
                            self.prev_raw = raw_frames[-1]
                        if self.probe is not None:
                            try:  # a probe error is logged, never the end of the run
                                info["probe_rows"] = self.probe.pool(self.frame_places, self.steps_done, typed, self.shown)
                                info["probe_skips"] = dict(self.probe.skips)
                            except Exception as ex:
                                self.probe.store = {}
                                info["probe_error"] = f"{type(ex).__name__}: {ex}"[:300]
                        self.last = frames[-1]
                        if cfg.get("live_dual"):  # LIVE.md (16 Nov): Krea alone | with the lock, side by side
                            td = time.perf_counter()
                            try:
                                frames = dual_view(raw_frames, frames, self.frame_reads, self.shown)
                            except Exception as ex:
                                info["dual_error"] = f"{type(ex).__name__}: {ex}"[:300]
                            info["dual_s"] = round(time.perf_counter() - td, 3)  # code-dual-r1 #1
                    if cfg.get("live_check") and self.label.box is not None and frames:
                        t3 = time.perf_counter()
                        kind = ("reset" if info.get("reset") else "write" if info["wrote"]
                                else "fix" if info.get("fix", {}).get("boxes") else "plain")
                        try:  # code-check-r1 #4: a checker error is logged, never the end of a paid session
                            info["display"] = self.check_display(frames[-1], kind)
                        except Exception as ex:
                            self.fix = None
                            info["check_error"] = f"{type(ex).__name__}: {ex}"[:300]
                        info["check_s"] = round(time.perf_counter() - t3, 3)
                if self.restyles:
                    info["restyle"] = self.restyles
                self.steps_done += 1
                info.update(write_s=round(t1 - t0, 3), gen_s=round(t2 - t1, 3), shown=self.shown,
                            applied_seq=self.applied, block=self.b, n_frames=len(frames))
                return frames, info

        return KreaLive

    def live(cfg: dict) -> None:
        """The live page (runner/live_server.py), or with `live_replay` a scripted typing replay of several arms in
        one process (LIVE.md, 15 Nov: the lock vs pin test): each arm and seed runs `blocks` blocks with the typed text
        from `script` ([block, text] pairs), its frames saved as an mp4 and its blocks logged; then the process ends."""
        rep = cfg.get("live_replay")
        launch = os.environ.get("LAUNCH", "local")
        if not rep:
            sys.path.insert(0, str(ROOT / "runner"))
            from live_server import serve
            serve(live_generator(cfg), cfg["live_text"], cfg.get("port", 8000), ROOT / "logs/live" / f"{launch}.jsonl",
                  title=cfg.get("live_title", "Live Sign"))
            return
        out = ROOT / "logs/live" / launch
        if rep.get("subdir"):  # LIVE.md 16 Nov: several replay configs of the same arms on one VM keep their own files
            out = out / rep["subdir"]
        out.mkdir(parents=True, exist_ok=True)

        def script_for(sd: int) -> list:
            """LIVE.md (16 Nov, the LoRA pilot): random_script draws each seed's own numbers (1-5 digits, no leading
            zero, none of `exclude`, the evaluation's held-out numbers); else the config's fixed script."""
            rs = rep.get("random_script")
            if not rs:
                return rep["script"]
            rng = np.random.default_rng(int(rs.get("rng", 0)) * 1000 + sd)
            excl, nums = {str(x) for x in rs.get("exclude", [])}, []
            while len(nums) < rs["numbers"]:
                k = int(rng.integers(1, 6))
                n = str(int(rng.integers(1 if k > 1 else 0, 10))) + "".join(str(int(d)) for d in rng.integers(0, 10, k - 1))
                if n not in excl and (not nums or n != nums[-1]):
                    nums.append(n)
            return [[rs["first"] + rs["every"] * i, n] for i, n in enumerate(nums)]

        for arm in rep["arms"]:
            for sd in rep["seeds"]:
                acfg = {**load_config(arm), "seed": sd}
                if rep.get("start_seed") is not None:
                    acfg["start_seed"] = rep["start_seed"]
                print(f"=== replay {arm} seed {sd} ===", flush=True)
                frames_all: list = []
                gen = None
                try:  # code-replay-r1 #3: a failed run is logged; the other runs go on
                    gen = live_generator(acfg)()
                    script = script_for(sd)
                    seq, prev = 0, None
                    with (out / f"{arm}-s{sd}.jsonl").open("w") as log:
                        for b in range(rep["blocks"]):
                            typed = acfg["live_text"]
                            for blk, text in script:
                                if b >= blk:
                                    typed = str(text)
                            if typed != prev:
                                seq, prev = seq + 1, typed
                            t = time.time()
                            frames, info = gen.step(typed, seq)
                            log.write(json.dumps({"b": b, "typed": typed, "seq": seq, "block_s": round(time.time() - t, 3),
                                                  "n": len(frames), **info}, default=str) + "\n")
                            log.flush()
                            frames_all.extend(frames)
                except Exception:
                    traceback.print_exc()
                    print(f"replay {arm} seed {sd} failed", flush=True)
                finally:
                    if frames_all:
                        write_mp4(np.stack(frames_all), out / f"{arm}-s{sd}.mp4")
                    if gen is not None and getattr(gen, "lora", None):
                        from lora_train import detach_lora, unmerge_lora
                        if getattr(gen, "lora_orig", None) is not None:
                            print(f"LoRA merge undone: {unmerge_lora(models.transformer.model, gen.lora_orig)} layers restored", flush=True)
                        else:
                            print(f"LoRA taken out: {detach_lora(models.transformer.model)} layers", flush=True)
                    if gen is not None and getattr(gen, "collect", None):
                        try:  # the prompts' embeddings, by prompt text, for runner/lora_train.py
                            torch.save(gen.prompts_seen, gen.collect_dir / f"prompts_{arm}_s{sd}.pt")
                        except Exception:
                            traceback.print_exc()
                        gen.collect_hook.remove()
                    if gen is not None and getattr(gen, "probe", None) is not None:
                        try:
                            gen.probe.save(out / f"{arm}-s{sd}_probe.npz", {"arm": arm, "seed": sd})
                        except Exception:
                            traceback.print_exc()
                        gen.probe.close()
                    if gen is not None:
                        gen.session.dispose()
                    GUIDE.update(lat=None, mask=None, final=False)
                    del frames_all, gen
                    import gc
                    gc.collect()  # code-replay-r1 #6: the session and its generator reference each other
                    torch.cuda.empty_cache()
        print("replay done", flush=True)

    if args.live:
        live(load_config(args.live))
        return

    for name, cfg, out, todo in runs:
        if not todo:
            continue
        print(f"=== {name} ===", flush=True)
        (out / "env.json").write_text(json.dumps({
            "config": name, "cfg": cfg, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "krea_commit": krea_commit}, indent=1))
        try:
            run_config(cfg, out, todo)
        except Exception:  # one failing config must not stop the others in this process
            traceback.print_exc()
            print(f"config {name} failed", flush=True)
            failed.append(name)
    if failed:
        sys.exit(f"failed configs: {','.join(failed)}")


if __name__ == "__main__":
    main()
