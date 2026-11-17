# Plan: Krea writes typed text (letters and digits) on any surface

Status: proposal for Elad's decision, 17 Nov 2026 (Elad: "I want to generalize digits and letters on many different
things"). Nothing here is spent or built. Each VM stage gets its own code review.

## What we know (from files)

- The calculator LoRA v2 (data/lora/v2/last.pt) makes Krea keep the typed number when our 7-segment writer puts it into
  Krea's memory: right on 87-94% of frames on held-out numbers, also on an unseen calculator (s0); base 52-61%
  (data/scores/lora_eval_v2_s*.json, lora_eval_s0.json). Merged, it costs no speed.
- With the prompt alone (no memory write), neither base nor v2 writes the typed number on any of 5 scenes (OCR right
  4-5% = the start blocks only), and v2 holds the old number more than base (27-57% of frames vs 4-28%;
  data/scores/pscene_*.json). v2 learned "keep what memory shows", not "read the number from the prompt".
- The feature probe: the digit is held at its own place in blocks 10-20 from the first denoising step, in a code
  shared across places (data/scores/probe_digits.json).
- Our reader/writer/target builder only works on grey 7-segment LCDs; on 4 new LCD scenes it failed
  (scripts/lcd_start_check.py). OCR found the number on all of them.

## Goal and test

Typed text (words and numbers) shown on many kinds of surfaces (signs, screens, labels, packaging, chalkboards, LED and
LCD displays), drawn by Krea, at Krea's speed. Test on surface types held out from training, with words and numbers
never trained on: Krea's frames as made, read by OCR, plus a by-eye check. Bar (decided now): on held-out surfaces,
right on at least 2x base's share of frames and at least 50%; no visible scene damage; no speed cost (merged).

## Method: train prompt-following, not memory-copying

At a switch, the memory still shows the old text and the prompt names the new one. Training samples have exactly that:
context = the memory as it is (old text), prompt = new text, target = Krea's next block with the text changed to the
new one (in Krea's own style). The adapters go on cross-attention (the prompt path) plus self-attention in blocks
10-20 (where the probe reads the text). At run time nothing is written into memory: any surface works without a
per-surface tool.

Targets need a corrected block where Krea did not write the new text. Sources, in order:
1. Krea's own frames: blocks where Krea did write the new text (under the prompt switch, some do, partly); a frame
   that reads right is copied to its neighbours by a tracked homography of the text region (generalises our LCD
   transplant: feature tracking, no LCD finder).
2. A text-editing image model (offline, not at run time) rewrites the text in a frame, kept only when two readers
   agree (OCR + a vision-language reader), then copied across the block as in 1.
Every target frame is re-read; a frame not confirmed is masked out of the loss (as in the pilot).

## Stages and cost (spot RTX PRO 6000 at $2.04/h; estimates from this week's runs)

| Stage | Where | What | Time | Cost |
|---|---|---|---|---|
| 0 | Jetson | general reader (OCR box + second reader), text-region tracker, target builder; ~15 surface types x prompts (digits and words), 10 train / 5 held out; stub tests; review | 1-2 days | $0 |
| 1 | VM | start images (Qwen-Image), 3 per surface | ~0.5 h | ~$1 |
| 2 | VM | collection: base Krea, prompt switches, transformer inputs saved; targets built (Krea frames, editor where needed) | ~1.5-2 h | ~$4 |
| 3 | VM | train (cross-attn + blocks 10-20), evaluate on held-out surfaces, base vs LoRA | ~2.5 h | ~$5 |
| 4 | VM | one on-policy round (collect with the LoRA, retrain, evaluate) | ~3 h | ~$6 |

Cap: $25 for stages 1-4 (about $16 expected). Stop and report if exceeded, or if stage 3 shows no gain over base.

## Risks

- Targets: where Krea never writes the new text, the editor's look becomes the target; its quality per surface is the
  main risk (klein's restyles were poor). Masking unconfirmed frames limits damage, but too few targets on a surface
  means it teaches little.
- Readers: OCR misreads some fonts (it garbled 7-segment digits); the second reader is the guard.
- Letters are harder than digits (more classes, words); start with short words (3-6 letters) and numbers.
- A prompt-following LoRA could weaken scene consistency; the loss keeps a small weight outside the text region, and
  the test checks scenes by eye.
