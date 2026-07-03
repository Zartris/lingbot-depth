# distill/ — status & next steps

Working notes for continuing the LingBot-Depth distillation auto-research harness.
Branch: `worktree-distill-scaffold` → PR #1 on `Zartris/lingbot-depth`.

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

## Last validated (this session)

`python -m distill.run --setup-baseline --hf` **completed successfully** (exit 0):
streamed 100 RobbySimVal samples from the 47 GB shard and measured the teacher
baseline — `latency=402 ms @ L9, params=321.2M` (real metrics `nan` because real is
off by default; sim metrics are in `runs/teacher_baseline.json`). So the streamed-HF
eval path works end-to-end. `runs/` is git-ignored, so the baseline JSON stays local —
re-run `--setup-baseline --hf` next session to regenerate it (fast; samples are cached).

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

## Known gaps / watch-list

- **Nothing has run a full real iteration yet** — only `--test-run` (6 steps). Real
  training numbers are unproven.
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
