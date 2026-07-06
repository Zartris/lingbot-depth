# distill/ — status & next steps

Working notes for continuing the LingBot-Depth distillation auto-research harness.
Branch: `worktree-distill-scaffold` → PR #1 on `Zartris/lingbot-depth`.

## METHODOLOGY (current — supersedes earlier "ViT-S student" framing)

The approach is **incremental compression from the teacher**, not a from-scratch small
student. Full detail is in `program.md` (the agent's manual). Key decisions:

- **Seed = the teacher** (`STUDENT_BACKBONE="dinov2_vitl14"`, `build_student` inherits ALL
  matching teacher weights). Shrink one small, weight-inheriting step at a time.
- **Warm-start from the best student** each iteration (cascade fresh<teacher<best); the
  teacher is the fallback, not a per-iteration reseed.
- **Objective** (rewritten): scored ACROSS resolution levels [0,3,6,9]. Accuracy measured
  vs the **TEACHER (fixed anchor)**, never best-student (no ratcheting). `AbsRel` may
  exceed the teacher by ≤ `ACCURACY_BAND_ABSREL=0.01` (else rejected); within band,
  `score = W_SPEED·mean_latency + W_ACC·mean(AbsRel excess)` with `W_ACC≫W_SPEED`
  (accuracy valued higher than speed). Degenerate output → worst score.
- **Token dial (resolution_level 0-9) is a preserved USER FEATURE, not a search knob** —
  `num_tokens_range` fixed to the teacher's; you can't "win" by using fewer tokens.
- **Search space is OPEN** (smaller backbones allowed) but big low-transfer moves are
  budget-limited: the proxy will reject them unfairly. Default budget `TRAIN_MINUTES=90`;
  give a big bet a longer run with `--budget-min <N>`. No stop — runs forever.

✅ **GPU-re-validated** (found + fixed 2 bugs while doing it):
- Teacher per-level curve is sane (L0: 61ms/absrel 0.0082 → L9: 179ms/0.0063).
- **Seed reproduces the teacher exactly** (excess = -0.0000, speedup 1.00x, feat loss
  literally 0) — "start at teacher" works.
- **Bug: training a teacher-init model collapsed it** (absrel 0.007→0.70 in 9 steps).
  Two causes, both fixed: (1) the depth distillation target was computed at the teacher's
  default token count (3600) while the student trains at 1200 — an inconsistent objective;
  now both are at FEATURE_TOKENS. (2) LR 2e-4 / WD 0.05 is too aggressive for a good init
  (AdamW takes ~full-LR steps even at ~0 loss); now LR 5e-5 + 100-step warmup + WD 0.01.
  After the fix, training PRESERVES the init (absrel 0.007→0.012, within band, δ1 0.998).
- Loop runs end-to-end with across-levels scoring; results.csv logs `budget_min`.

The methodology + harness are validated. Remaining work is research, not plumbing.

## Where we are

The harness is built and **verified end-to-end on the GPU** (`--test-run` passed on an
RTX PRO 4000 Blackwell). What works:

- Full pipeline: teacher load → student build → DINOv2 warm-start (best→teacher→fresh
  cascade) → teacher-target cache → distill steps → save + code snapshot → reload →
  eval → objective/score → champion promote.
- Teacher baseline + resolution sweep on the GPU (confirmed the "free speed lever":
  teacher ~117 ms @ L0 → ~354 ms @ L9).
- HF **streaming** of the real `.tar.zst` shards — verified extracting valid sim
  triplets (rgb 960×1280, sane depth ranges).

## Environment (IMPORTANT — the repo's pins don't work on Blackwell)

This Blackwell GPU (sm_120) needs a newer stack than `pyproject.toml` pins:

- `torch 2.11.0+cu128`  (pinned `torch==2.6.0` predates Blackwell — will not run)
- `xformers 0.0.35`  — **required, not optional**: the RGB-D encoder's nested-tensor
  path (`dinov2_rgbd/layers/block.py`) hard-asserts xformers. Match it to torch.
- `requests`, `zstandard`  — for streaming/decompressing the dataset shards.
- plus numpy, opencv, huggingface_hub, trimesh, scipy, matplotlib, pillow.

Run from the repo root. `mdm` imports without `pip install -e .` since the root is on
the path.

## The dataset is tar shards, not a file tree (key finding)

`robbyant/mdm_depth` ships as WebDataset-style `.tar.zst` shards, **47–320 GB each**
(smallest is `RobbySimVal_batch_0001` at 47 GB). We never download a whole shard — we
stream it, zstd-decompress on the fly, and pull the first N complete `(rgb, raw, gt)`
triplets off the front (disk-buffered, cached in `.cache/shards/`).

Verified internal layouts (parsers in `prepare.py`):

| shard family | rgb | raw | gt | ordering |
|---|---|---|---|---|
| `RobbySimVal` | `_rgb.left.jpg` | `_rawdepth.left.png` | `_depth_left.png` | interleaved (cheap) |
| `RobbySim_*_view` | `_left.jpg` | `_rmd2c.png` | `_depth.png` | interleaved (cheap) |
| `RobbyReal` | `color/…` | `rawdepth/…` | `gtdepth/…` | **grouped by modality** |

**Default = sim** (`N_*_REAL = 0`). RobbySimVal has *perfect* GT and streams cheaply.
Real is wired but off by default because its modality-grouped layout means ~3 GB must
be streamed per camera before the first triplet completes.

⚠️ Two things about real still UNVERIFIED:
1. The `color` / `gtdepth` dir names are assumed (I only observed `rawdepth/` when
   streaming). Confirm before enabling real.
2. The `_rmd2c.png` = "raw depth" mapping for sim-train is inferred by elimination.

## Last validated

- `--setup-baseline --hf` (exit 0): streamed 100 RobbySimVal samples, teacher baseline
  `latency=402 ms @ L9, 321.2M params`. Streamed-HF **eval** path works.
- `--test-run --hf` (exit 0): streamed 1200 RobbySim train samples, ran 6 real training
  steps with all three losses active incl. **GT loss** (`feat/out/gt`), saved+snapshotted,
  reloaded, scored (37.2M params, 4.48× faster than teacher). Streamed-HF **train+eval
  together** works — the full loop is validated end-to-end on real data. (Accuracy is
  noise: 6 steps is plumbing, not training.)

`runs/` is git-ignored so these JSONs stay local; re-run to regenerate (fast, cached).

## Next steps, in order

1. `python -m distill.run --setup-baseline --hf`
   → regenerate `runs/teacher_baseline.json` (validated; ~fast, samples cached).
2. `python -m distill.run --test-run --hf`
   → tiny end-to-end on streamed real (sim) data with GT loss.
3. A real iteration: `python -m distill.run --hf` (20 min), sanity-check the score,
   then `--compare` (teacher vs best across resolution levels).
4. Probe a `RobbyReal` shard to confirm `color`/`gtdepth` dir names; if good, enable
   real (`N_*_REAL > 0` in `prepare.py`) and accept the streaming floor.
5. Hand to auto-research: the agent edits `train.py` / `student_model/`, loop with
   `--hf --accept`.

## Ready for auto-research? YES — validated end-to-end, two real bugs found & fixed

Ran real `--hf` iterations. Everything now works AND two critical bugs were caught and
fixed (this is exactly what the validation was for):

- ✅ **Convergence**: with the fixed loss, the student converges to valid positive depth
  (range 1.15–2.31 m, 100% valid; was −268 m garbage before) and losses drop cleanly
  (out 6.7→0.2, gt 3.3→0.1) in ~6 min.
- ✅ **Warm-start cascade** proven: iter 2 inherited 305 tensors from the champion
  (vs iter 1's "0 from best").
- ✅ **`--accept` → promote → committed champion** and keep-or-revert work.
- ✅ **`--compare`** across resolution levels works.

**Bug 1 (objective, CRITICAL — fixed).** A degenerate student (negative depth →
absrel=inf, delta1=0) scored ≈ latency with ZERO penalty and got promoted as champion.
Cause: baseline real-domain metrics are `nan` (real off by default) → nan-poisoned the
penalty to 0. Fix: `_wavg` renormalises the baseline over finite domains, plus a hard
degeneracy guard (non-finite absrel/delta1 or delta1<0.05 → ×1000 penalty). Now garbage
scores ~1000× worse; verified.

**Bug 2 (recipe, fixed).** The model's output remap is `linear` (head emits metric depth
directly, can be negative), but the loss was log-space — `log(clamp(neg,1e-3))` gives
ZERO gradient, so the student drifted to negative depth. Fix: `_depth_l1` is now linear
L1, which penalises negatives directly. Also eval now uses `apply_mask=False` so a
student can't inflate its score by masking the pixels it gets wrong.

**Tuning note (not a blocker):** the sim teacher is near-perfect (absrel 0.0063), so the
accuracy tolerance is essentially unreachable and the objective is accuracy-dominated
(scores are large, ~1e5). Ordering is correct (better student → lower score), so the
research signal is valid; may want to loosen `ACC_TOLERANCE` / cap the absrel violation
later. Also the mask head is NOT distilled (no mask loss) — fine for scoring
(apply_mask=False), but add a mask loss if deployment needs the confidence mask.

## Known gaps / watch-list

- Real (RobbyReal) domain is off by default; dir names unverified (see above).
- **Cross-machine scores aren't comparable** (latency is GPU-specific). Ranking stays
  machine-local (`runs/results.csv`); the committed champion (`distill/best/`, fp16,
  no LFS) records its `gpu` in `best.json` — re-benchmark it locally elsewhere.
- **Noise-blind acceptance** — a within-jitter "improvement" can be accepted; consider
  an acceptance margin once real run-to-run variance is measured.
- **Augmentation vs the teacher-target cache** — cached targets are from clean inputs,
  so geometry-changing augmentation desyncs them (RGB-only jitter is fine).
- **Latency is batch size 1, always** (one frame = one camera read). Don't batch it.

## Command cheat-sheet

```
python -m distill.run --smoke              # logic self-test, no GPU/downloads
python -m distill.run --test-run           # tiny end-to-end on GPU (local examples)
python -m distill.run --setup-baseline --hf# teacher baseline on streamed eval set
python -m distill.run --cache-targets --hf # warm the teacher-target cache
python -m distill.run --hf --accept        # one real iteration; commit+promote if better
python -m distill.run --hf --compare       # teacher vs best vs current across levels
```
