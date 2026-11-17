"""Step 36: the editor process beside Krea (textbench/edit_write.py). Runs in our gen environment (diffusers 0.35.1,
the QwenImageEditPipeline step 27 used); Krea runs in its own environment, so the two talk through a folder.

Qwen-Image-Edit is held in CPU memory with diffusers' model offload: each part (text encoder 16 GB, transformer
41 GB, VAE) is on the GPU only while it runs, and the Krea process moves its own weights off the GPU while it waits
(runner/gen_krea.py), so the two fit on one 96 GB card (peaks: Krea 69 GB, the editor 67 GB, data/videos/*/timing.jsonl
and data/start_frames_qwenedit_pause2d/timing.jsonl).
Settings as step 27 (model card: 50 steps, true CFG 4, negative " "), input and output at 1344x768 (~1 MP, as
runner/gen_kontext.py), scaled back to the frame's size.
Step 37: a request's mode "fast" runs the 8-step Lightning LoRA (below) instead, same pipeline and offload.
Step 38, --resident: the fast editor stays on the GPU beside Krea (no offload, no weight moves): the Lightning LoRA is
fused, the transformer's and the text encoder's weights are stored in fp8 (diffusers' layerwise casting, compute in bf16:
20 GB + 8 GB instead of 41 + 16), and requests are crops (textbench/edit_write.py crop_box) edited at RESIDENT_AREA pixels
(the pipeline sizes its input to 1 MP by itself; patched for these requests) instead of 1344x768.
Step 40b, --fast-editors: also serves modes "klein" and "qwen2511fp8", step 40a's FLUX.2 klein 4B and
Qwen-Image-Edit-2511 4-step (fp8 storage) on the window around each label (load_fast_editors).
Step 40b, --rsste: also serves mode "rsste", RS-STE restyling each label's box in one pass (load_rsste_editor).
Step 39, --swap: also serves mode "swap", the latent text-swap model (textbench/swap_model.py, data/swap_model/model.pt):
for each label of the prompt, the window around it (as in training) in our font is restyled in Wan's latent space and
put back; the request's answer then goes through the same change mask, colour match and paste as the editor's.
Requests: <n>.png + <n>.json {prompt, seed, mode}; answers <n>.out.png + <n>.out.json {seconds}. Stops when the stop file
appears. --stub draws with textbench/write_text.py instead (an offline test of the transport, no model).
Usage: python runner/edit_server.py --folder /opt/textbench/edits --models /opt/models [--stub]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
import traceback
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SIZE = (1344, 768)
RESIDENT_AREA = 512 * 768  # step 38: pixels the crop is edited at (aspect kept, sides multiples of 32)
# Step 37, the fast editor: lightx2v/Qwen-Image-Lightning's 8-step LoRA for Qwen-Image-Edit (Apache-2.0) and the scheduler from
# its model card (ModelTC/Qwen-Image-Lightning generate_with_diffusers.py: shift 3 used in distillation).
LIGHTNING_REPO, LIGHTNING_FILE = "lightx2v/Qwen-Image-Lightning", "Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors"
LIGHTNING_SCHEDULER = {"base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False, "max_image_seq_len": 8192,
                       "max_shift": math.log(3), "num_train_timesteps": 1000, "shift": 1.0, "shift_terminal": None,
                       "stochastic_sampling": False, "time_shift_type": "exponential", "use_beta_sigmas": False,
                       "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False}


def area_size(w: int, h: int, area: int = RESIDENT_AREA) -> tuple[int, int]:
    """Step 38: (w, h) scaled to about `area` pixels, aspect kept, both multiples of 32 (as the pipeline's own calculate_dimensions)."""
    s = (area / (w * h)) ** 0.5
    return max(32, round(w * s / 32) * 32), max(32, round(h * s / 32) * 32)


def load_resident(models: Path, fp8: bool = True, full: bool = False):
    """Step 38: the fused 8-step editor on the GPU with fp8 weight storage; edits at the request's own size.
    Step 39's training data (fp8=False, full=True): the same fused editor in bf16 on whole frames at 1344x768, i.e. step 37's
    fast editor without the offload (no Krea on the GPU then). Step 38 showed the crop edits move the scene far more."""
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline
    from diffusers.hooks import apply_layerwise_casting
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit as qe
    pipe = QwenImageEditPipeline.from_pretrained(models / "Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16)
    lora = models / LIGHTNING_REPO / LIGHTNING_FILE
    pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="lightning")
    pipe.fuse_lora(lora_scale=1.0)
    pipe.unload_lora_weights()
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
    for m in (pipe.transformer, pipe.text_encoder) if fp8 else ():
        apply_layerwise_casting(m, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipeline_dims = qe.calculate_dimensions  # (named `full` until 10 Nov 2026, which shadowed the argument: see RESIDENT.md Changes)
    print(f"resident editor on GPU: {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    def edit(img: Image.Image, prompt: str, seed: int, mode: str = "resident") -> Image.Image:
        w, h = SIZE if full else area_size(*img.size)
        qe.calculate_dimensions = lambda _area, ratio: pipeline_dims(w * h, ratio)  # the condition image at the crop's size too
        try:
            out = pipe(image=img.resize((w, h), Image.Resampling.LANCZOS), prompt=prompt, width=w, height=h,
                       negative_prompt=" ", generator=torch.Generator("cuda").manual_seed(seed),
                       num_inference_steps=8, true_cfg_scale=1.0).images[0]
        finally:
            qe.calculate_dimensions = pipeline_dims
        return out.resize(img.size, Image.Resampling.LANCZOS)
    return edit


PAIRS = r'the text "([^"]+)" to "([^"]+)"'  # textbench/edit_write.py edit_prompt


def load_swap(models: Path, weights: Path, stub: bool = False):
    """Step 39: the swap model and Wan's VAE (fp32, as in runner/train_swap.py) on the GPU, ~1 GB."""
    import numpy as np
    from textbench.edit_write import label_boxes
    from textbench.swap_model import cut, hide_label, label_mask, put, window, window_box
    from textbench.write_text import write
    hide = False
    if stub:  # transport test: the window in our font, no model
        def enc(x): return x
        def dec(z): return z
        def net(o, f, m): return f
        def pool(m): return m
    else:
        import torch
        from diffusers import AutoencoderKLWan
        from textbench.swap_model import build_model
        vae = AutoencoderKLWan.from_pretrained(models / "Wan-AI/Wan2.1-T2V-14B-Diffusers", subfolder="vae",
                                               torch_dtype=torch.float32).to("cuda").eval()
        mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, 16, 1, 1)
        std = torch.tensor(vae.config.latents_std, device="cuda").view(1, 16, 1, 1)
        ck = torch.load(weights, map_location="cuda")
        model = build_model(ck["width"]).to("cuda").eval()
        model.load_state_dict(ck["state"])
        hide = bool(ck.get("hide_label", False))

        def enc(x: np.ndarray):
            t = torch.from_numpy(x).to("cuda").permute(2, 0, 1)[None].float().div(127.5).sub(1)[:, :, None]
            return (vae.encode(t).latent_dist.mode()[:, :, 0] - mean) / std

        def dec(z) -> np.ndarray:
            x = vae.decode((z.float() * std + mean)[:, :, None]).sample[:, :, 0]
            return x.add(1).mul(127.5).clamp(0, 255).byte()[0].permute(1, 2, 0).cpu().numpy()

        def pool(m: np.ndarray):
            return torch.nn.functional.max_pool2d(torch.from_numpy(m).to("cuda")[None, None], 8)

        def net(o, f, m):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(o, f, m)
        print(f"swap model on GPU: {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

    def swap(img: Image.Image, prompt: str) -> Image.Image:
        import contextlib
        frame = np.asarray(img.convert("RGB"))
        pairs = re.findall(PAIRS, prompt)
        boxes = label_boxes(frame, pairs, str(ROOT))
        out = frame
        ctx = contextlib.nullcontext()
        if not stub:
            import torch
            ctx = torch.no_grad()
        with ctx:
            for old, new in pairs:
                if old not in boxes:
                    continue
                rect = window(frame.shape, boxes[old])
                o = cut(frame, rect)
                f = cut(write(frame, [(old, new)], str(ROOT))[0], rect)  # as runner/gen_swap_data.py made F
                o_in = hide_label(o, window_box(boxes[old], rect)) if hide else o  # v2 models were trained with the old letters painted out
                out = put(out, dec(net(enc(o_in), enc(f), pool(label_mask(o, f)))), rect)
        return Image.fromarray(out)
    return swap


def load_rsste_editor(weights: Path, numbers: str = "rsste", stub: bool = False):
    """Step 40b (SPEED.md): RS-STE (third_party/RS-STE, runner/rsste_bakeoff.py) as the editor: each label's box (found
    as the memory write finds it, textbench/edit_write.py label_boxes) is edited to its new text in one forward pass and
    put back; the answer then goes through the editor write's change mask, colour match and paste in gen_krea.py.
    numbers="font": a new text with a digit is drawn in our font instead (textbench/write_text.py write); RS-STE as
    published drew no digits (SPEED.md Results). Stays on the GPU (~0.5 GB): no weight moves."""
    import cv2
    import numpy as np
    from textbench.edit_write import label_boxes
    from textbench.write_text import write
    if stub:
        def edit_box(crop: np.ndarray, text: str) -> np.ndarray:
            return crop
    else:
        sys.path.insert(0, str(ROOT / "runner"))
        from rsste_bakeoff import load_rsste, rsste_edit
        model, torch = load_rsste(weights)
        print(f"RS-STE on GPU: {torch.cuda.memory_allocated() / 1e9:.1f} GB ({weights})", flush=True)

        def edit_box(crop: np.ndarray, text: str) -> np.ndarray:
            return rsste_edit(model, torch, crop, text)

    def run(img: Image.Image, prompt: str) -> Image.Image:
        frame = np.asarray(img.convert("RGB"))
        pairs = re.findall(PAIRS, prompt)
        boxes = label_boxes(frame, pairs, str(ROOT))
        out = frame.copy()
        for old, new in pairs:
            if old not in boxes:
                continue
            if numbers == "font" and any(c.isdigit() for c in new):
                out = write(out, [(old, new)], str(ROOT))[0]
                continue
            x0, y0, x1, y1 = boxes[old]
            patch = edit_box(frame[y0:y1, x0:x1], new)
            out[y0:y1, x0:x1] = cv2.resize(patch, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(out)
    return run


FAST_MODES = ("klein", "qwen2511fp8", "restyle")  # step 40b (SPEED.md): the 40a editors, on the GPU beside Krea;
# LIVE.md (16 Nov 2026): "restyle", klein on the request's whole image (a square crop around the LCD, textbench/restyle.py)


def load_fast_editors(models: Path, stub: bool = False):
    """Step 40b: FLUX.2 klein 4B and Qwen-Image-Edit-2511 4-step (fp8 storage) as in step 40a (runner/edit_bakeoff.py):
    for each label, the window around it (textbench/swap_model.py window, 256x256) resized to EDGE x EDGE, edited with
    the editor write's instruction, resized back and put into the frame; the answer then goes through gen_krea.py's
    change mask, colour match and paste. Each model loads when first asked and the other is freed first (both do not
    fit beside Krea; the Krea configs of one VM use one of them at a time). Needs diffusers 0.38 (infra/startup.sh)."""
    import gc
    import numpy as np
    from textbench.edit_write import edit_prompt, label_boxes
    from textbench.swap_model import cut, put, window
    current: dict = {"name": None, "edit": None, "pipe": None}

    def get(name: str):
        if current["name"] == name:
            return current["edit"]
        import torch
        current.update(name=None, edit=None, pipe=None)
        gc.collect()
        torch.cuda.empty_cache()
        sys.path.insert(0, str(ROOT / "runner"))
        from edit_bakeoff import klein_editor, qwen_editor
        t = time.perf_counter()
        edit, pipe, _ = klein_editor(models) if name == "klein" else qwen_editor(models, plus=True, fp8=True)
        current.update(name=name, edit=edit, pipe=pipe)
        print(f"{name} loaded in {time.perf_counter() - t:.0f} s, GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
        return edit

    def run(img: Image.Image, prompt: str, seed: int, name: str, given: dict | None = None) -> Image.Image:
        from edit_bakeoff import EDGE
        if name == "restyle":  # the prompt as given, on the whole image (the caller crops and pastes back)
            o = img.convert("RGB").resize((EDGE, EDGE), Image.Resampling.LANCZOS)
            return o if stub else get("klein")(o, prompt, seed)
        frame = np.asarray(img.convert("RGB"))
        pairs = re.findall(PAIRS, prompt)
        # LIVE.md: the request may carry the label's box (a lone digit cannot be found by OCR of its old text)
        boxes = {k: tuple(v) for k, v in given.items()} if given else label_boxes(frame, pairs, str(ROOT))
        out = frame
        for k, (old, new) in enumerate(pairs):
            if old not in boxes:
                continue
            rect = window(frame.shape, boxes[old])
            o = Image.fromarray(cut(frame, rect)).resize((EDGE, EDGE), Image.Resampling.LANCZOS)
            e = o if stub else get(name)(o, edit_prompt([(old, new)]), seed + k)
            out = put(out, np.asarray(e.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)), rect)
        return Image.fromarray(out)
    return run


def load(models: Path, stub: bool, resident: bool = False):
    if stub:
        import numpy as np
        from textbench.write_text import write

        def draw(img: Image.Image, prompt: str, seed: int, mode: str = "full") -> Image.Image:
            pairs = re.findall(r'the text "([^"]+)" to "([^"]+)"', prompt)
            return Image.fromarray(write(np.asarray(img), pairs, str(ROOT))[0])
        return draw
    if resident:
        return load_resident(models)
    import torch
    from diffusers import QwenImageEditPipeline
    pipe = QwenImageEditPipeline.from_pretrained(models / "Qwen/Qwen-Image-Edit", torch_dtype=torch.bfloat16)
    full = pipe.scheduler
    fast = None
    lora = models / LIGHTNING_REPO / LIGHTNING_FILE
    if lora.exists():  # step 37: a failure here only fails the fast requests
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler
            pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name="lightning")
            pipe.disable_lora()
            fast = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)
            print("lightning LoRA loaded", flush=True)
        except Exception:
            traceback.print_exc()
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)

    def edit(img: Image.Image, prompt: str, seed: int, mode: str = "full") -> Image.Image:
        g = torch.Generator("cuda").manual_seed(seed)
        if mode == "fast":  # the model card's settings: 8 steps, true CFG 1 (one pass per step), shift-3 scheduler
            if fast is None:
                raise RuntimeError("fast mode asked but the Lightning LoRA did not load")
            pipe.enable_lora()
            pipe.scheduler, settings = fast, {"num_inference_steps": 8, "true_cfg_scale": 1.0}
        else:
            pipe.disable_lora() if fast is not None else None
            pipe.scheduler, settings = full, {"num_inference_steps": 50, "true_cfg_scale": 4.0}
        out = pipe(image=img.resize(SIZE, Image.Resampling.LANCZOS), prompt=prompt, width=SIZE[0], height=SIZE[1],
                   negative_prompt=" ", generator=g, **settings).images[0]
        torch.cuda.empty_cache()  # give the memory back to Krea
        return out.resize(img.size, Image.Resampling.LANCZOS)
    return edit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, required=True)
    ap.add_argument("--models", type=Path, default=Path("/opt/models"))
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--resident", action="store_true", help="step 38: fused 8-step editor kept on the GPU, crop requests")
    ap.add_argument("--swap", type=Path, default=None, help="step 39: the swap model's weights (data/swap_model/model.pt)")
    ap.add_argument("--rsste", type=Path, default=None, help="step 40b: RS-STE weights (data/rsste/state_dict.pt or a fine-tune)")
    ap.add_argument("--rsste-numbers", choices=["rsste", "font"], default="rsste", help="step 40b: who draws texts with digits")
    ap.add_argument("--fast-editors", action="store_true", help="step 40b: modes klein and qwen2511fp8 (load_fast_editors)")
    ap.add_argument("--no-qwen", action="store_true", help="LIVE.md: klein only, Qwen-Image-Edit not loaded (its modes fail)")
    a = ap.parse_args()
    a.folder.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    try:
        edit = None if a.no_qwen else load(a.models, a.stub, a.resident)
        swap = load_swap(a.models, a.swap, a.stub) if a.swap else None
        rsste = load_rsste_editor(a.rsste, a.rsste_numbers, a.stub) if a.rsste else None
        fast = load_fast_editors(a.models, a.stub) if a.fast_editors else None
    except Exception:
        (a.folder / "failed").write_text(traceback.format_exc())
        raise
    print(f"editor loaded in {time.perf_counter() - t:.0f} s", flush=True)
    (a.folder / "ready").write_text("1")
    done: set[str] = set()
    while not (a.folder / "stop").exists():
        todo = sorted(p for p in a.folder.glob("*.json") if not p.name.endswith(".out.json") and p.stem not in done)
        if not todo:
            time.sleep(0.02)  # step 40b: 0.2 s polls on both sides added up to ~0.4 s, as long as klein's edit
            continue
        req = todo[0]
        done.add(req.stem)
        out = a.folder / f"{req.stem}.out"
        try:
            r = json.loads(req.read_text())
            t = time.perf_counter()
            src = Image.open(req.with_suffix(".png")).convert("RGB")
            if r.get("mode") == "swap":
                if swap is None:
                    raise RuntimeError("swap mode asked but the server was started without --swap")
                img = swap(src, r["prompt"])
            elif r.get("mode") in FAST_MODES:
                if fast is None:
                    raise RuntimeError("a fast editor mode asked but the server was started without --fast-editors")
                img = fast(src, r["prompt"], int(r["seed"]), r["mode"], r.get("boxes"))
            elif r.get("mode") == "rsste":
                if rsste is None:
                    raise RuntimeError("rsste mode asked but the server was started without --rsste")
                img = rsste(src, r["prompt"])
            else:
                if edit is None:
                    raise RuntimeError(f"mode {r.get('mode', 'full')} asked but the server was started with --no-qwen")
                img = edit(src, r["prompt"], int(r["seed"]), r.get("mode", "full"))
            img.save(f"{out}.png")
            info = {"seconds": time.perf_counter() - t, "mode": r.get("mode", "full"), "size": list(img.size)}
            if a.resident and not a.stub:
                import torch
                info["vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
            print(f"{req.stem}: {info['seconds']:.1f} s  {r['prompt'][:80]}", flush=True)
        except Exception:
            info = {"error": traceback.format_exc()}
            print(info["error"], flush=True)
        Path(f"{out}.json.part").write_text(json.dumps(info))
        Path(f"{out}.json.part").rename(f"{out}.json")


if __name__ == "__main__":
    main()
