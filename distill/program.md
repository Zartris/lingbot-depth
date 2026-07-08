# program.md — LingBot-Depth distillation auto-research

You are an ML research agent. Read this whole file before you start — it defines the
method, and the methodology is not obvious. Your job runs forever: keep finding a
**faster** version of the LingBot-Depth model that stays **as accurate as the teacher**.

This is a **distillation / progressive-compression** task, **not** a from-scratch
rewrite. The search STARTS AT THE TEACHER and shrinks it in small, weight-inheriting
steps. "Anyone can make a smaller network" — but a smaller network is not the goal; a
smaller network that *keeps the teacher's accuracy* is, and that only happens if you
transfer as much as possible and change little at a time.

## The core method (read this twice)

1. **Start at the teacher.** The seed student IS the teacher backbone (`dinov2_vitl14`),
   and `build_student()` inherits ~all teacher weights → it begins at ~teacher accuracy.
2. **Shrink incrementally.** Each iteration, make **ONE isolated change** and test it:
   drop/merge a transformer block, prune some attention heads, narrow a dimension,
   lighten a head. Small changes = most weights still transfer = the effect is *visible*
   in one budget.
3. **Inherit from the best student.** Every run warm-starts from the best student so far
   (see Warm-start). As the net shrinks, the best student stays near-teacher accuracy, so
   it is the highest-overlap source of weights for the next small change.
4. **Keep it only if it's faster at ~teacher accuracy.** That's the objective below.
5. **Never stop.** There is no target to reach and stop at — always look for the next
   improvement.

Why incremental: the per-iteration budget can only judge a change *in proportion to how
much weight it transfers*. A high-transfer change (drop 2 layers) shows its true effect
quickly. A big low-transfer change (swap to a fresh small backbone) needs far more
training than one budget to recover accuracy, so the proxy will reject it **regardless of
its real potential**. Staying incremental keeps you in the regime the proxy can judge.

## The one number you minimise

`score` (lower is better), computed by the FROZEN `distill/prepare.py::objective`,
**scored across the resolution levels** `[0,3,6,9]`:

```
per level:  excess = student_AbsRel − teacher_AbsRel   (accuracy vs the TEACHER, fixed anchor)
degenerate (no usable depth: non-finite AbsRel, or δ1 < 0.05)      -> DEGENERATE_SCORE (worst)
excess > ACCURACY_BAND_ABSREL (0.01) at any level                 -> REJECT band (rejected)
otherwise   score = W_SPEED·mean(latency)  +  W_ACC·mean(excess)   ,  W_ACC ≫ W_SPEED
```

Consequences you must internalise:
- **Accuracy is measured against the TEACHER, always — never the best student.** If it
  were vs the best student, the bar would ratchet down every iteration and students would
  drift worse and worse. Teacher is the fixed anchor.
- **Accuracy is valued higher than speed** (`W_ACC ≫ W_SPEED`). A more-accurate, slightly
  slower student BEATS a faster, less-accurate one. So you take only *accuracy-preserving*
  speedups; you never trade accuracy cheaply for speed.
- **You cannot win by lowering the token count.** `resolution_level` 0–9 (the token dial)
  is a user feature preserved on the student and **scored across all levels** — the user
  already chooses it at runtime. `num_tokens_range` is fixed to the teacher's. Real
  speedups must make the network *cheaper per token* (fewer layers/heads/width), not use
  fewer tokens.
- **`AbsRel` is domain-weighted (real 0.7, sim 0.3)** — the exam is both simulated
  (`RobbySimVal`, perfect GT) and real (`RobbyReal`, physical Orbbec sensor; GT holes are
  masked out). Real is weighted higher because it is the deployment target, so a change
  that helps sim but hurts real will usually *lose*. Real is drawn from 5 cameras/scenes
  per split — decent signal, limited diversity (see STATUS.md).
- **Latency is measured at batch size 1** (one camera frame) and is noisy on this GPU —
  prefer changes that also cut **params/FLOPs** (deterministic, portable), logged for you.

## What you may edit
- `distill/train.py` — student config + the distillation recipe: `STUDENT_BACKBONE`,
  `intermediate_layers`, neck/head widths; losses & weights (`W_OUTPUT_DISTILL`,
  `W_FEATURE_DISTILL`, `W_GT`), optimiser, LR schedule, `BATCH_SIZE`, sampling, which
  layers to **freeze**, and `BUDGET_MINUTES` (how long THIS experiment trains — see
  Budget). (NOT `num_tokens_range` — fixed to the teacher's.)
- `distill/student_model/` — the MUTABLE copy of the network. Edit the architecture here:
  drop/merge blocks, prune heads, narrow dims, token merging, fused ops. This is where
  structural speedups live.

## What you may NOT edit
- `distill/prepare.py` — data, teacher, metrics, latency, **objective**, the eval set, and
  the budget DEFAULT + safety CAP (`MAX_BUDGET_MINUTES`). Frozen — it is your exam; you
  cannot change how you're graded. (You DO set your own per-run budget within the cap via
  `train.BUDGET_MINUTES` — see Budget.)
- the `mdm/` package — the frozen teacher. Editing it corrupts your targets and reference.

**Hard contract:** whatever `build_student()` returns MUST expose
`infer(image, depth_in=..., intrinsics=...) -> {"depth", "points", "mask"}`, same tensor
shapes as the teacher, linear output remap (the depth head emits metric depth directly —
supervise it in LINEAR space; a log-space loss lets it drift negative).

## The search space is OPEN

You MAY try anything — including a smaller backbone (`vitb14`/`vits14`) or a radical
restructure. Two rules:
1. **One isolated change per iteration**, so its effect is attributable.
2. A **big low-transfer change won't converge in the default budget** — it would be
   rejected because it can't recover accuracy in time, not because it's a bad idea. So
   when you make a big change, **raise its budget yourself** (`BUDGET_MINUTES` /
   `--budget-min`, up to the cap) to give it a fair, converged shot — that's your call to
   make, weighing the compute cost. Don't let a too-short run convince you a big idea is
   dead; note "budget-limited" in the log.

## Setup — do this once when a run starts

Adapted from autoresearch's setup to our harness:

1. **Branch — never `main`.** All work happens on a long-lived research branch (currently
   `worktree-distill-scaffold`). Unlike vanilla autoresearch (a fresh `autoresearch/<tag>`
   branch per run that you *advance*), here each ACCEPTED experiment is a commit on this
   branch, so the branch history IS the champion lineage. Starting a brand-new run? Branch
   from the current champion, not from `main`.
2. **Read the in-scope files:** this file, `STATUS.md`, `distill/prepare.py` (your frozen
   exam — read it to know how you're graded, never edit it), and `distill/train.py` +
   `distill/student_model/` (what you edit).
3. **Environment:** a Blackwell-class GPU needs torch 2.11+cu128 + xformers 0.0.35 (the
   repo's pins predate Blackwell). Run from the repo root with `XFORMERS_DISABLED` UNSET —
   it hard-disables the required nested-tensor path and the model won't run. See STATUS.md.
4. **Verify data / baseline:** run `python -m distill.run --setup-baseline --hf` ONCE. It
   measures the teacher + per-level curve and writes `runs/teacher_baseline.json` (the
   accuracy anchor). The first call streams ~6 GB of real eval, then caches it. Iterations
   refuse to run without this baseline.
5. **Confirm the seed, then go:** run one iteration UNCHANGED — it should score ~teacher
   (excess ≈ 0). That is the champion to beat. Then start shrinking.

## The loop

```
python -m distill.run --setup-baseline --hf     # ONCE: measure the teacher + its per-level curve
# then each iteration:
#   1. read distill/runs/results.csv + this log — what's been tried, what scored well
#   2. make ONE isolated change (train.py and/or student_model/), form a hypothesis
#   3. python -m distill.run --hf --accept       # trains, evals across levels, commits+promotes if better
#   4. read the score; append a one-line finding to the Log below; repeat
```

- **Budget: YOU set it.** Rate how big your change is and budget TO CONVERGENCE — set
  `BUDGET_MINUTES` in `train.py` (or pass `--budget-min <N>`). Small high-transfer change
  (drop a block) → leave it (default 90 min). Big low-transfer change (fresh smaller
  backbone) → raise it, up to the frozen safety cap `prepare.MAX_BUDGET_MINUTES` (8 h).
  Budget to convergence, **not beyond**: over-budgeting a change that has already plateaued
  just burns compute AND confounds the comparison with shorter runs (a longer run can
  score better purely from more training). The effective budget is logged in results.csv,
  so size it honestly. When a big change looks promising under a long budget, re-run the
  current champion at the SAME budget for a clean apples-to-apples comparison.
- **Freezing:** when you change one region, freeze the untouched layers and fine-tune the
  changed/adjacent ones first — it's cheaper and more stable. If accuracy doesn't recover,
  unfreeze more (downstream layers were trained expecting the old behaviour).
- Teacher targets are cached per sample (`.cache/teacher_targets/`), so the teacher runs
  once per `(sample, FEATURE_TOKENS)`, not every step. Training uses a fixed canvas
  (`prepare.TRAIN_HW`) so the cache is sound; geometry-changing augmentation desyncs it
  (RGB-only jitter is fine).

### Launching a run and reading it
- Launch redirected — do NOT `tee` or let training output flood your context:
  `python -m distill.run --hf --accept > run.log 2>&1`. Then read only the outcome:
  `grep "^\[run\] score=" run.log` (score, mean_latency, speedup, mean_absrel, min_delta1,
  params) and `grep -E "IMPROVED|no improvement" run.log`.
- `runs/results.csv` gets exactly ONE row per run: `run_id, backbone, budget_min, score,
  params, mean_latency_ms, mean_absrel, min_delta1, speedup_vs_teacher`. Read it at the
  start of every iteration — it's the raw history of what's been tried and what scored.

### Keep-or-revert — our harness only auto-KEEPS (read carefully)
`--accept` commits (`train.py` + `student_model/` + `results.csv`) **and** promotes the
champion (`distill/best/`) ONLY when the score improves on the best so far. On a REJECTED
run it commits nothing — **but your working-tree edits stay.** There is no auto-`git
reset`. So YOU must revert a rejected change before the next one, or edits pile up and
"one isolated change" silently breaks:
```
git checkout -- distill/train.py distill/student_model   # discard a rejected change
```
Revert ONLY those two paths — leave `results.csv` alone (its freshly-appended row is your
memory of the failed attempt; it's uncommitted, keep it). To deliberately BUILD ON a
rejected change instead of reverting, keep it and note in the Log that the next recorded
score bundles both edits.

### Timeout / kill (hang guard)
The budget stops *training* on a wall clock inside `train.py`, but a bad edit can still
hang (infinite loop, a wedged eval or stream). Launch each run under an external guard of
~2× your budget + ~10 min overhead, e.g. for a 45-min budget:
```
timeout 100m python -m distill.run --hf --accept > run.log 2>&1
```
If it gets killed, treat it as a crash: revert (above) and move on.

### Crashes — use judgment
Dumb and easy (typo, missing import, an obvious shape mismatch you can fix) → fix and
re-run. Fundamentally broken idea → skip it, revert, and log one line as a "crash" in the
Log below so you don't retry it. Don't sink a whole session into resurrecting a bad idea.

## Warm-start (weight inheritance)

`build_student()` warm-starts as a priority cascade (`WARM_START_FROM`, default `"best"`),
highest priority applied last so it wins on name+shape overlap:

```
fresh (stock DINOv2)  <  teacher (encoder + decoder, all matching)  <  best student so far
```

Because the seed is the teacher backbone, the teacher copy inherits the FULL teacher
(encoder + decoder). As you shrink, only the changed layers stop matching; everything else
keeps inheriting from the best student — which stays near-teacher accuracy, so it's a
high-overlap source. This is why "load from best student" matters: a best student that is
almost as accurate as the teacher is *like loading the teacher, but with far more of the
weights actually transferable to your shrunk architecture*.

Set `WARM_START_FROM="teacher"` for a clean fixed-init A/B when you need to isolate a
recipe change from the cumulative inheritance.

## Shrink axes (roughly, best transfer first)

1. **Drop / merge transformer blocks** (24 → fewer). Kept blocks inherit teacher weights
   directly. Highest transfer, biggest per-step speedup — start here.
2. **Prune attention heads / MLP channels** with importance scoring; inherit the kept ones.
3. **Narrow width** (structured pruning). Harder to inherit cleanly — real work.
4. **Lighter neck/heads** (`dim_res_blocks`, `num_res_blocks`).
5. **Token merging / pruning inside the encoder** (code in `student_model/`).
6. Feature-distillation recipe: which layers to match, cosine vs L2, attention transfer.
7. Big bets (budget-limited): a smaller backbone from a warm start — needs `--budget-min`.

## Log of what worked / didn't

_(Append one line per experiment. This is your memory across iterations; results.csv is
the raw data, this is the interpretation — especially "budget-limited, needs promotion".)_

- (seed) student = teacher backbone, full inheritance → baseline ≈ teacher accuracy/speed;
  first real move is a small shrink from here.
- 2026-07-08: run 20260708-100812 (1-min seed check) logged latency 4.7x WORSE than teacher
  at L9 (844 vs 179 ms) — FALSE ALARM: interleaved re-benchmark (teacher/student/teacher/
  student, one process) gives 177 vs 177 ms (1.00x). The run's latency was contaminated
  (GPU contention/thermal during that session). Lesson: treat single-run latency swings
  with suspicion; re-benchmark interleaved before believing a big latency delta. Its
  score 420.97 is latency-inflated but its accuracy row is valid (mean_absrel 0.0394 ≈ teacher).
