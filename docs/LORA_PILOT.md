# Pilot: fine-tune Krea (LoRA) to keep the typed digits

Status: proposal for Elad's decision, 16 Nov 2026. Nothing here has been spent or built. This is a new phase beyond
PLAN.md: it starts only on Elad's go, and each VM stage needs its own code review, as always.

## Why

The calculator demo's accuracy comes from a controller that rewrites frames: setup C has 91-95% of frames right by the
7-segment reader (data/scores/replay4_klein.json, replay5_push.json). Krea alone, with the typed number written into its
memory, is right on 35-48% of frames by OCR (E 35%, K 48%). It changes digits within a block, for example 23476 ->
29476. Engineers at Runway and Decart would spot the pasted font. A LoRA that makes Krea itself keep the number would
make "Krea draws the digits you type" an honest claim. Merged into the weights, it costs nothing at inference: the same
3.8 s per block.

## Success criteria (decided before training)

On numbers and a start image never seen in training, with the lock off (Krea's frames as made, memory written as in E):
- **Accuracy:** Krea alone right on at least 75% of frames by both readers (7-segment and OCR); baseline 35-59%.
- **Quality:** no visible damage outside the display. Judged by eye on a side-by-side clip, and measured as the mean
  pixel change outside the display versus the base model on the same noise.
- **Speed:** block time unchanged (LoRA merged).
- **Kill rule:** after stage 3, stop if Krea alone is below 60%, or if the scene degrades visibly.

## Method

**Loss.** Krea is a 4-step model distilled with Self-Forcing. Its own training (third_party/krea-realtime/pipeline/
self_forcing_training.py) runs the sampler on its own outputs and back-propagates through one randomly chosen step per
block. The pilot uses the same scheme with a plain regression loss instead of DMD. No teacher model is needed, so it
fits one GPU.
1. Roll out a block from a real memory state (as in the replays), without gradients except at one random step.
2. **Target:** the same block as Krea made it, with the display corrected where it reads wrong. The correction is the
   font write restyled by klein, colour-fixed and kept only when OCR reads the number: the pipeline already built and
   reviewed (textbench/restyle.py). Re-encode the corrected frames with Krea's VAE.
3. **Loss:** x0-prediction error against the target latents, weighted on the display's latent cells (from the tracker),
   with a smaller weight elsewhere, where the target is Krea's own output. So the LoRA learns to change the digits,
   not the scene.

**LoRA.** Rank 16-32 on the attention projections of the transformer blocks. Which blocks is informed by the feature
probe (step 1): train where the digit is readable.

**Data.** Contexts from replays of random 5-digit numbers (as replay_probe), on the calculator's seed-1 start image.
Held out for testing: a set of numbers and the seed-2 start image. About 600 corrected blocks for the pilot.

## Stages and cost (estimates: RTX PRO 6000 spot at $2.04/h, block times from our replays)

| Stage | Where | What | Time | Cost |
|---|---|---|---|---|
| 0 | Jetson | data generator and trainer; stub tests; code review | 1-2 days | $0 |
| 1 | VM | generate ~600 corrected blocks: block 3.8 s, klein ~0.5 s per frame, OCR | ~2 h | ~$4 |
| 2 | VM | LoRA training, ~3000 steps at ~3-4 s, checkpoints every 500 | ~3-4 h | ~$8 |
| 3 | VM | evaluation replays: base vs LoRA, held-out numbers and start image | ~0.5 h | ~$1 |
| 4 | later | other displays (clock, scoreboard) to test generalisation | — | — |

Pilot cap: **$30** (stages 1-3, about $13 expected). Stop and report if exceeded.

## Risks

- **Fitting.** 14B in bf16 (~28 GB) plus activations with gradient checkpointing, on 96 GB. Krea's text encoder can be
  moved off the GPU after the prompts are encoded (done before in step 38). To be checked in stage 1 before training.
- **Re-encoding.** The target latents come from re-encoding decoded frames, so they carry VAE error. Check that the
  round trip is small next to the digit change, as the memory writes already rely on.
- **klein's look.** The targets' corrected digits have klein's look (natural, slightly faint). The LoRA may learn that
  look; acceptable, and better than our font.
- **Overfitting to one scene.** That is why stage 3 holds out numbers and a start image, and stage 4 adds other displays.
- **Within-block drift.** Not every frame of the target is corrected if klein or OCR fails; such frames are dropped
  from the loss (masked), not trained on wrong.
