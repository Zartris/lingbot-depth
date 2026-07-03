# program.md — LingBot-Depth distillation auto-research

You are an ML research agent. Your job: **distill the frozen LingBot-Depth teacher
into a much faster student, while keeping accuracy within the tolerance band.** You
work by repeatedly editing `distill/train.py`, running an experiment, reading the
score, and iterating — the [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
loop, adapted to depth-model distillation.

## The one number you optimise

`score` (lower is better), computed by `distill/prepare.py::objective`:

```
score = latency_ms * (1 + PENALTY * <accuracy-tolerance violations vs teacher>)
```

- **latency_ms** — median single-frame forward time on THIS GPU (PyTorch/CUDA), the
  thing we are trying to reduce.
- **accuracy** — AbsRel & δ1 measured against **ground-truth depth**, weighted
  `0.7*real + 0.3*sim`. You are penalised only once accuracy leaves the band
  (`ACC_TOLERANCE=5%` on AbsRel, `1pt` on δ1) around the teacher.

So the only way to lower `score` is: **get faster without breaking accuracy.** You
cannot lower it by changing how it is measured — see the rules.

Secondary signals logged for you (not in the score, but watch them — the real target
is a robot, possibly non-NVIDIA/edge): **params**, and add FLOPs if useful. Prefer
wins that also cut params/FLOPs, since those port across hardware.

## Rules — what you may and may not touch

**You may edit `distill/train.py` and `distill/student_model/`.**
- `distill/train.py` — student config + the distillation recipe: `STUDENT_BACKBONE`
  (`dinov2_vits14` / `vitb14`), `intermediate_layers`, `STUDENT_NUM_TOKENS_RANGE`,
  neck/head widths; loss terms & weights (`W_OUTPUT_DISTILL`, `W_FEATURE_DISTILL`,
  `W_GT`), optimiser, LR schedule, augmentations, `BATCH_SIZE`, sampling/curriculum.
- `distill/student_model/` — the MUTABLE copy of the model stack (its own `v2.py`,
  encoder, decoder, `dinov2_rgbd/`). Edit the network ITSELF here: attention, patch
  embed, block structure, `forward`/`infer`, output heads, token merging/pruning,
  structural pruning, fused ops. This is where code-level speedups live.

**You may NOT touch** (they define the problem and keep the score honest):
- `distill/prepare.py` — data, teacher, metrics, latency, objective. Frozen. This
  includes the **training pool**, the **eval set** (your exam — you cannot change what
  you're scored on), and the **compute budget** (`TRAIN_MINUTES` / `MAX_STEPS`). Every
  experiment therefore runs on equal data at equal compute, so a better score means a
  genuinely better student — not more data or more time. You control how the data is
  *used* (batching, sampling, augmentation), not how much there is or how long you train.
- the `mdm/` package — defines the frozen teacher. Frozen.

**Hard contract:** whatever `build_student()` returns MUST expose
`infer(image, depth_in=..., intrinsics=...) -> {"depth", "points", "mask"}` with the
same tensor shapes as the teacher, so `prepare.evaluate()` can score it. If you write
a custom student, keep that method signature and the log-in/exp-out depth remap.

## The loop

```
python -m distill.run --setup-baseline   # ONCE: measure the teacher (anchors the band)
# then each iteration:
#   1. read distill/runs/results.csv (what's been tried, what scored well)
#   2. edit distill/train.py — change ONE thing, form a hypothesis
#   3. python -m distill.run --accept     # trains, evals, logs, commits if improved
#   4. read the printed score; keep going
```

Proxy runs are short (`TRAIN_MINUTES≈20`) so you can rank ideas fast. When a config
clearly wins, a **human** promotes it to a longer run by editing the frozen budget in
`prepare.py` (raise `TRAIN_MINUTES`, enlarge the data subset) — the agent never does
this itself. This mirrors autoresearch's "rank cheaply on short runs, scale the winner".

## Warm-start (weight inheritance across iterations)

Each experiment warm-starts as a priority cascade (`WARM_START_FROM` in `train.py`,
default `"best"`): `fresh stock-DINOv2 < teacher decoder < best student so far`, higher
priority winning on name+shape overlap. This matters because the teacher's 1024-d
encoder can't load into the 384-d student — so **only the best student can warm-start
the student encoder**, the main thing distillation teaches; otherwise every run
re-distills the encoder from scratch. When you change one layer, that layer falls
through to teacher/fresh while everything else inherits from the best student.

Consequence: the search is cumulative (evolutionary), so a recipe-only tweak can look
good just from inheriting a fine-tuned parent. Set `WARM_START_FROM="teacher"` for a
clean fixed-init A/B when you need to isolate a change, and validate promoted winners
from a fixed init.

## Suggested idea backlog (roughly cheap→deep)

1. **Fewer tokens** — lower `STUDENT_NUM_TOKENS_RANGE` / training `FEATURE_TOKENS`.
   Token count is THE dominant cost (measured on the teacher, RTX2080ti: L0~128ms →
   L9~307ms, a 2.4x range) and it's independent of input image resolution — the model
   interpolates to a token grid set by `resolution_level` and aspect ratio. Only tiny
   details are lost when lowering it (e.g. thin fence wires). `STUDENT_NUM_TOKENS_RANGE`
   is the student's `resolution_level` mapping, so this is the fastest lever by far.
   NB: `setup-baseline` prints the teacher's own level sweep — the student needs to beat
   that curve, not just the slow level-9 point.
2. **Smaller backbone** — `dinov2_vits14` (384-d, 12 layers) vs teacher L (1024, 24).
   Lean hard on feature + decoder warm-start to recover accuracy.
3. **Fewer `intermediate_layers`** taken from the backbone.
4. **Feature-distillation recipe** — cosine vs L2 weighting, which layer(s) to match,
   attention-transfer. (See ViTKD: matching the *right* features matters a lot.)
5. **Slimmer neck/heads** — reduce `dim_res_blocks` / `num_res_blocks` in the config.
6. **Token merging / pruning** inside the student forward (custom module).
7. **Structured pruning / low-rank** of the warm-started encoder.
8. Later (post-search, human): quantization, torch.compile, ONNX/TensorRT export.

## Log of what worked / didn't

_(Append findings here as you go — one line each. This is your memory across
iterations; the score CSV is the raw data, this is the interpretation.)_

- (baseline) teacher: see `runs/teacher_baseline.json`.
