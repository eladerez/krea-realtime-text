"""The LoRA pilot's trainer (research/plans/LORA_PILOT.md, stage 2; Elad: "Yes", 16 Nov 2026), on the GPU VM in Krea's
environment. Krea's transformer stays frozen in bf16; low-rank adapters are added to the self-attention projections
(fused q/k/v and output) of chosen blocks (the feature probe, data/scores/probe_digits.json, read the digit best in
blocks 10-20).

Data: the samples runner/gen_krea.py live_collect saved (logs/live/<launch>/lora_samples/*.pt): for each block, the
KV-cache rebuild's input (context latents), the 4 denoising calls' inputs and timesteps, Krea's output latents and the
corrected target (textbench/lora_targets.py), with loss weights per latent cell (LCD 1, elsewhere 0.1, an unverified
LCD 0); prompts_*.pt hold the prompts' embeddings.

A step (as Krea's own Self-Forcing training, which back-propagates through one random denoising step per block, but
with a regression loss instead of DMD and no teacher): rebuild the KV cache from the sample's context without
gradients; pick one of the 4 denoising calls at random; run it with gradients from its stored input (the base model's
trajectory: off-policy, the pilot's simplification); loss = weighted squared error of Krea's x0 prediction against the
target. AdamW on the adapters only; gradient checkpointing through the blocks (Krea's own path for training). The KV
cache's writes during a denoising call land in the same slot on every call (causal_model.py), so the checkpoint's
recompute sees the same cache.

Writes <out>/lora_<step>.pt (adapter weights + settings) every --save steps and <out>/train_log.jsonl.
Usage: python runner/lora_train.py --data logs/live/<launch>/lora_samples --models /opt/models --out data/lora/pilot
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT  # noqa: E402

KREA = ROOT / "third_party/krea-realtime"


class LoRALinear:
    """Wraps an nn.Linear: y = W x + b + scale * B(A x). A: normal with std 1/r, B = 0, so training starts from Krea."""

    @staticmethod
    def wrap(linear, rank: int, alpha: float):
        import torch
        import torch.nn as nn

        class Wrapped(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = linear
                dev = linear.weight.device
                self.A = nn.Parameter(torch.randn(rank, linear.in_features, device=dev, dtype=torch.float32) / rank)
                self.B = nn.Parameter(torch.zeros(linear.out_features, rank, device=dev, dtype=torch.float32))
                self.scale = alpha / rank

            def forward(self, x):
                y = self.base(x)
                return y + (x.float() @ self.A.t() @ self.B.t() * self.scale).to(y.dtype)

        return Wrapped()


def add_lora(model, layers: list[int], rank: int, alpha: float) -> list:
    params = []
    for i in layers:
        att = model.blocks[i].self_attn
        assert att.fused_projections, "Krea's loader fuses q/k/v (release_server.py); the adapters go on to_qkv"
        att.to_qkv = LoRALinear.wrap(att.to_qkv, rank, alpha)
        att.o = LoRALinear.wrap(att.o, rank, alpha)
        params += [att.to_qkv.A, att.to_qkv.B, att.o.A, att.o.B]
    return params


def lora_state(model, layers: list[int]) -> dict:
    out = {}
    for i in layers:
        att = model.blocks[i].self_attn
        for name in ("to_qkv", "o"):
            m = getattr(att, name)
            out[f"blocks.{i}.self_attn.{name}.A"] = m.A.detach().cpu()
            out[f"blocks.{i}.self_attn.{name}.B"] = m.B.detach().cpu()
    return out


def attach_lora(model, state: dict, scale: float) -> int:
    """code-lora-r1 should-fix 2: evaluation runs the adapters as trained (fp32, beside the bf16 weights) instead of
    merging them (bf16 rounding lost most of a small update); detach_lora puts the original modules back exactly."""
    import torch
    n = 0
    for key, a in state.items():
        if not key.endswith(".A"):
            continue
        path = key[:-2]
        _, i, _, name = path.split(".")
        att = model.blocks[int(i)].self_attn
        lin = getattr(att, name)
        w = LoRALinear.wrap(lin, a.shape[0], 1.0)
        with torch.no_grad():
            w.A.copy_(a.to(w.A.device, torch.float32))
            w.B.copy_(state[path + ".B"].to(w.B.device, torch.float32))
        w.scale = scale
        w.requires_grad_(False)
        setattr(att, name, w)
        n += 1
    return n


def detach_lora(model) -> int:
    n = 0
    for blk in model.blocks:
        for name in ("to_qkv", "o"):
            m = getattr(blk.self_attn, name)
            if hasattr(m, "base"):
                setattr(blk.self_attn, name, m.base)
                n += 1
    return n


def merge_lora(model, state: dict, scale: float, keep: bool = False):
    """Add scale * B @ A into the base weights in place (inference: no extra cost), rounded to the weights' dtype once
    (W + delta in fp32, then bf16). keep: also return the original weights (CPU) for unmerge_lora and how much of the
    update survives the rounding: 1 - |bf16(W + d) - W - d| / |d| over all merged layers (1.0: all of it). With keep,
    a failure part way puts the changed layers back before raising (a half-merged model never reaches the next arm)."""
    import torch
    n, orig, res2, d2 = 0, {}, 0.0, 0.0
    try:
        with torch.no_grad():
            for key, a in state.items():
                if not key.endswith(".A"):
                    continue
                path = key[:-2]
                b = state[path + ".B"]
                _, i, _, name = path.split(".")
                lin = getattr(model.blocks[int(i)].self_attn, name)
                assert not hasattr(lin, "base"), "merge into plain layers (adapters attached)"
                w0 = lin.weight.float()
                d = scale * (b.float().to(w0.device) @ a.float().to(w0.device))
                if keep:
                    orig[path] = lin.weight.detach().to("cpu", copy=True)
                lin.weight.copy_((w0 + d).to(lin.weight.dtype))
                res2 += float(((lin.weight.float() - w0 - d) ** 2).sum())
                d2 += float((d ** 2).sum())
                del w0, d
                n += 1
    except Exception:
        if keep:
            unmerge_lora(model, orig)
        raise
    if not keep:
        return n
    return n, orig, round(1 - (res2 / max(d2, 1e-30)) ** 0.5, 4)


def unmerge_lora(model, orig: dict) -> int:
    """Put back the weights merge_lora(keep=True) kept (exactly)."""
    import torch
    with torch.no_grad():
        for path, w in orig.items():
            _, i, _, name = path.split(".")
            lin = getattr(model.blocks[int(i)].self_attn, name)
            lin.weight.copy_(w.to(lin.weight.device))
    return len(orig)


def load_models(models: Path):
    os.environ["MODEL_FOLDER"] = str(models / "Wan-AI")
    os.environ["DO_COMPILE"] = "false"
    os.chdir(KREA)
    sys.path.insert(0, str(KREA))
    from wan.modules.causal_model import CausalWanModel
    CausalWanModel.from_pretrained = classmethod(lambda cls, path, **kw: cls.from_config(cls.load_config(path), **kw))
    from gen_krea import prepare_t5
    prepare_t5(models)
    from release_server import load_all, load_merge_config
    config = load_merge_config("configs/self_forcing_server_14b.yaml")
    config.checkpoint_path = str(models / "krea/krea-realtime-video-14b.safetensors")
    return load_all(config), config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, nargs="+", required=True, help="folders of lora_samples")
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", default="0-39", help="blocks with adapters, e.g. 0-39 or 5-25")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--save", type=int, default=500)
    ap.add_argument("--hold-out", default="", help="comma-separated numbers whose samples are never trained on")
    ap.add_argument("--max-minutes", type=float, default=240.0, help="stop (and save) after this long")
    ap.add_argument("--sources", default="krea", help="repair sources to train on (Elad, 16 Nov: Krea-sourced only)")
    # LIVE.md 16 Nov (v2): the last denoising call (t=0) maps its already clean input to ~itself (grad ~0 in the pilot's
    # log): a quarter of the steps taught nothing; --skip-last trains only the calls before it
    ap.add_argument("--skip-last", action="store_true", help="never pick the last denoising call (t=0)")
    # LIVE.md 17 Nov (v2 audit): on s2's small display the 7-segment reader took a 9 for a 3 or 5 in a few copy
    # sources (79798 for 79738, 48996 for 48936), all where OCR of the source disagreed: drop those blocks
    ap.add_argument("--source-ocr-agree", default="",
                    help="per --data folder, 1 to skip Krea-sourced samples whose copy source OCR reads as another "
                         "number (e.g. 0,1: code-v2-r2, round 1 is s1 only, where the reader is reliable)")
    ap.add_argument("--repeat", default="", help="per --data folder, how often its samples are listed (e.g. 1,2): "
                                                 "code-v2-r1 nit 3, the on-policy round's samples drawn more often")
    ap.add_argument("--init", type=Path, default=None,
                    help="start the adapters from this checkpoint (another run's), when --out has none of its own")
    a = ap.parse_args()
    a.data = [d.resolve() for d in a.data]
    a.out = a.out.resolve()
    a.init = a.init.resolve() if a.init else None  # code-v2-r1 blocker: load_models() changes directory
    a.out.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in a.layers.split("-"))
    layers = list(range(lo, hi + 1))
    held = set(x for x in a.hold_out.split(",") if x)

    reps = [int(x) for x in a.repeat.split(",")] if a.repeat else [1] * len(a.data)
    assert len(reps) == len(a.data) and min(reps) >= 1, "--repeat: one count >= 1 per --data folder"
    agree = [x == "1" for x in a.source_ocr_agree.split(",")] if a.source_ocr_agree else [False] * len(a.data)
    assert len(agree) == len(a.data), "--source-ocr-agree: one 0/1 per --data folder"
    files, prompts, rep_of, agree_of = [], {}, {}, {}
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before CUDA starts
    import torch
    for d, r, ag in zip(a.data, reps, agree):
        for f in sorted(d.glob("*.pt")):
            files.append(f)
            rep_of[f], agree_of[f] = r, ag
        for p in d.glob("prompts_*.pt"):
            prompts.update(torch.load(p))
    files = [f for f in files if not f.name.startswith("prompts_")]
    samples, n_ocr_drop = [], 0
    for f in files:
        s = torch.load(f)
        if s["shown"] in held or s["prompt"] not in prompts or not bool((s["weight"] >= 1).any()):
            continue  # no verified display cell: nothing for the display to learn
        if s["info"].get("source") not in a.sources.split(","):
            continue
        so = s["info"].get("source_ocr")
        if agree_of[f] and so is not None and so != "".join(ch for ch in s["shown"] if ch.isdigit()):
            n_ocr_drop += 1
            continue
        samples += [f] * rep_of[f]
    print(f"{len(set(samples))} samples, {len(samples)} listed with --repeat {reps}, {n_ocr_drop} dropped by --source-ocr-agree ({len(files)} files, {len(prompts)} prompts, held out {sorted(held)})", flush=True)
    if not samples:
        sys.exit("no samples")

    models, config = load_models(a.models)
    # stage 1's smoke test (textbench-run-20261116-1313): release_server.py turns autograd off for the whole process
    # at import (torch.set_grad_enabled(False), its line 2), so the loss had no graph; on again here (the KV-cache
    # rebuild below runs under no_grad explicitly)
    torch.set_grad_enabled(True)
    pipe, tr = models.pipeline, models.transformer
    model = tr.model
    try:  # the prompts' embeddings are saved; Krea's text encoder (~11 GB) leaves the GPU
        models.text_encoder.to("cpu")
    except Exception as ex:
        print("text encoder stays:", ex, flush=True)
    torch.cuda.empty_cache()
    params = add_lora(model, layers, a.rank, a.alpha)
    model.gradient_checkpointing = True
    first = 1
    done_ck = sorted(a.out.glob("lora_[0-9]*.pt"))
    src = done_ck[-1] if done_ck else a.init
    if src is not None:  # resume the adapters (not the optimizer state) from the latest checkpoint, or --init
        ck = torch.load(src, map_location="cpu")
        assert ck["settings"]["rank"] == a.rank and ck["scale"] == a.alpha / a.rank, "--init: other rank or alpha"
        for i in layers:
            att = model.blocks[i].self_attn
            for name in ("to_qkv", "o"):
                m = getattr(att, name)
                m.A.data.copy_(ck["state"][f"blocks.{i}.self_attn.{name}.A"].to(m.A.device))
                m.B.data.copy_(ck["state"][f"blocks.{i}.self_attn.{name}.B"].to(m.B.device))
        first = ck["step"] + 1 if done_ck else 1  # --init: this run's own step count starts at 1
        print(f"resumed from {src}", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    n_par = sum(p.numel() for p in params)
    print(f"adapters on blocks {lo}-{hi}: {n_par / 1e6:.1f} M parameters", flush=True)
    dev = torch.device("cuda")
    settings = {"layers": layers, "rank": a.rank, "alpha": a.alpha, "lr": a.lr, "held_out": sorted(held),
                "samples": len(set(samples)), "listed": len(samples), "repeat": reps, "source_ocr_agree": agree, "ocr_dropped": n_ocr_drop, "skip_last": a.skip_last, "init": str(a.init) if a.init else None}
    (a.out / "settings.json").write_text(json.dumps(settings, indent=1))
    log = (a.out / "train_log.jsonl").open("a")
    t_start = time.time()
    rng = random.Random(first)
    for step in range(first, a.steps + 1):
        f = rng.choice(samples)
        torch.cuda.reset_peak_memory_stats()  # gpu_gb: this step's peak (code-lora-r2 r3 nit 2)
        s = torch.load(f)
        emb = prompts[s["prompt"]].to(dev, torch.bfloat16)
        cond = {"prompt_embeds": emb}
        ctx = s["ctx"]
        t0 = time.perf_counter()
        with torch.no_grad():
            # textbench-run-20261116-1544: out of memory at step 2. Krea re-initialises its caches by zeroing them in
            # place (pipeline/causal_inference.py), but after a step the KV cache holds that step's autograd history
            # (the denoising call's writes with grad), so step 2 kept step 1's graph alive. Fresh caches every step.
            pipe.kv_cache1, pipe.crossattn_cache = None, None
            pipe._initialize_kv_cache(batch_size=1, dtype=torch.bfloat16, device=dev)
            pipe._initialize_crossattn_cache(batch_size=1, dtype=torch.bfloat16, device=dev)
            x_ctx = ctx["x"].to(dev)
            model.block_mask = model._prepare_blockwise_causal_attn_mask(
                device=str(dev), num_frames=x_ctx.shape[1], frame_seqlen=pipe.frame_seq_length,
                num_frame_per_block=pipe.num_frame_per_block, local_attn_size=-1)
            tr(noisy_image_or_video=x_ctx, conditional_dict=cond, timestep=ctx["t"].to(dev),
               kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache, current_start=ctx["start"])
            model.block_mask = None
        k = rng.randrange(len(s["steps"]) - 1 if a.skip_last else len(s["steps"]))
        call = s["steps"][k]
        _, x0 = tr(noisy_image_or_video=call["x"].to(dev), conditional_dict=cond, timestep=call["t"].to(dev),
                   kv_cache=pipe.kv_cache1, crossattn_cache=pipe.crossattn_cache, current_start=call["start"])
        target = s["target"].to(dev).float()
        w = s["weight"].to(dev)[None, :, None]  # [1, latents, 1, 60, 104]
        err = (x0.float() - target) ** 2
        loss = (w * err).sum() / (w.sum() * err.shape[2])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        with torch.no_grad():  # the display's error alone (what the pilot is for)
            lcd = (s["weight"] >= 1).to(dev)[None, :, None]
            lcd_err = float((err * lcd).sum() / max(1.0, float(lcd.sum()) * err.shape[2]))
        rec = {"step": step, "loss": round(float(loss.detach()), 5), "lcd_err": round(lcd_err, 5), "k": k, "t": int(call["t"].flatten()[0]),
               "grad_norm": round(float(gn), 4), "s": round(time.perf_counter() - t0, 2), "file": f.name,
               "gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        del x0, _, err, loss, target, w, lcd  # nothing of this step's graph reaches the next
        if step % 25 == 0 or step == 1:
            print(rec, flush=True)
        done = step == a.steps or (time.time() - t_start) / 60 > a.max_minutes
        if step % a.save == 0 or done:
            ck = {"state": lora_state(model, layers), "settings": settings, "step": step, "scale": a.alpha / a.rank}
            torch.save(ck, a.out / f"lora_{step:05d}.pt")
            torch.save(ck, a.out / "last.pt")  # the evaluation replay's live_lora (runner/configs.yaml)
            print(f"saved lora_{step:05d}.pt", flush=True)
        if done:
            break
    if not all(math.isfinite(float(p.detach().abs().sum())) for p in params):
        sys.exit("non-finite adapter weights")


if __name__ == "__main__":
    main()
