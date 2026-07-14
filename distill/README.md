# distill/ — LingBot-Depth distillation auto-research harness

Speed up LingBot-Depth inference (for robot deployment) by **distilling** the
ViT-L teacher into a smaller/faster student, while holding accuracy — driven by an
agent loop in the style of [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Files

| File | Mutable? | Role |
|------|----------|------|
| `prepare.py` | **frozen** | Teacher, dataset (streamed subset of `robbyant/mdm_depth`), metrics, latency benchmark, and the single scalar **objective**. |
| `train.py` | **agent edits this** | Student config + distillation recipe + training loop. |
| `run.py` | runner | `train → eval → log → keep/revert`. |
| `program.md` | steering | The agent's rules, the metric, the idea backlog. |
| `runs/` | output (git-ignored) | Per-experiment `student.pt` + a copy of the student **code snapshot** (so a past student rebuilds exactly even after the live code changes), plus `results.csv`, `teacher_baseline.json`. |
| `best/` | **committed** | The champion that travels across machines: `student.pt` (weights, via **Git LFS**), its code snapshot, and `best.json` (score, metrics, config, `gpu`, `git_sha`). Updated + committed on each accepted improvement (`--accept`). |
| `.cache/` | output | Downloaded dataset frames + cached teacher targets. |

## The metric (one number, lower is better)

```
score = latency_ms * (1 + PENALTY * <accuracy violations vs teacher, on GT depth>)
```

Latency = median single-frame forward on this GPU. Accuracy = AbsRel/δ1 vs
ground-truth depth, weighted `0.7*real + 0.3*sim` (deployment is a real sensor; sim
is a clean-GT cross-check). Params/FLOPs are logged as portability signals.

## Environment

**xformers is required** — the RGB-D encoder's nested-tensor path (`block.py`) hard-asserts
it; it is not an optional acceleration. Match xformers to your torch/GPU.

Verified working end-to-end on an **NVIDIA RTX PRO 4000 Blackwell (sm_120)**:
`torch 2.11.0+cu128` + `xformers 0.0.35`. Note the repo's pinned `torch==2.6.0` /
`xformers==0.0.29.post2` predate Blackwell and won't run on it — use a cu128 torch there.

## Quick start

```bash
# 0. env (from repo root)
python -m pip install -e .

# 1. sanity-check the harness logic — no GPU, no downloads
python -m distill.run --smoke

# 2. tiny end-to-end pipeline check on GPU (~1-2 min, local examples, no dataset download)
#    build -> warm-start -> cache -> a few steps -> eval -> save/reload -> score
python -m distill.run --test-run

# 3. measure the teacher once (needs GPU + downloads the teacher ckpt)
python -m distill.run --setup-baseline            # add --hf to use the real dataset

# 4. one research iteration: train a student, eval, log the score
python -m distill.run                              # local examples (plumbing)
python -m distill.run --hf --accept                # real data; commit train.py if better

# compare teacher vs best-student-so-far vs a given student across resolution levels
python -m distill.run --hf --compare               # (optionally pass a student.pt path)

# (optional) warm the teacher-target cache up front, so the first run isn't slow
python -m distill.run --hf --cache-targets
```

## Teacher-target cache

The ViT-L teacher is the expensive part of each step, so its per-sample outputs
(encoder features at `FEATURE_TOKENS` + best-quality depth) are cached to
`.cache/teacher_targets/` and reused across every step **and every later experiment** —
the teacher runs once per `(sample, FEATURE_TOKENS)`, not every step. It fills lazily
during training, or eagerly via `--cache-targets`. For this to be sound, training
processes each sample at a fixed canvas (`prepare.TRAIN_HW`, default 480×640); eval
still runs at native resolution. Budget ~3 MB/sample (fp16); set
`prepare.CACHE_TEACHER_FEATURES=False` to cache depth only if disk-constrained.

The agent loop: read `program.md` + `runs/results.csv` → edit `train.py` → run →
read score → repeat.

## Data

The full `robbyant/mdm_depth` is **2.71 TB** — never fully downloaded. `prepare.py`
lists the repo and streams a small balanced subset of `(rgb, rawdepth, gtdepth)`
triplets (real from `RobbyReal`, sim from `RobbySim*`, which has perfect GT). The
folder regexes in `prepare.py` are written from the dataset card; confirm them once
against `HfApi().list_repo_files("robbyant/mdm_depth", repo_type="dataset")` — that is
the single spot that may need a one-line tweak.

Without `--hf`, the harness runs on the 8 bundled `../examples` scenes (no GT, so it
scores fidelity-to-teacher) — enough to validate all plumbing before committing GPU time.

## Champion & cross-machine notes

Accepted winners are promoted to the committed `best/` so the team's best student
travels with the repo. Weights are saved **fp16** (~44 MB for ViT-S — under GitHub's
100 MB blob limit, no LFS needed); if a bigger backbone exceeds
`prepare.MAX_COMMIT_WEIGHTS_MB`, only the code snapshot + `best.json` are committed and
the weights stay local (a warning tells you).

⚠️ **Scores are machine-specific.** Latency (and thus `score`/`speedup`) depend on the
GPU, so `best.json` records the `gpu` it was measured on. Keep-or-revert **ranking**
stays local (`runs/results.csv`, one machine); the committed champion is a portable
warm-start + comparison artifact, not a cross-machine score to rank against. On a new
machine, re-benchmark the champion locally (`--compare`) to get comparable numbers.

## What the student is

`MDMModel` is fully config-instantiated, so a student is just a smaller `model_config`
+ trained weights. `prepare.derive_student_config()` builds a valid one from the
teacher's: swap the DINOv2 backbone (L→S/B), which sets `dim_out` and the neck's first
input width; everything downstream transfers, so the student warm-starts its decoder
straight from the teacher and only the encoder must be distilled.
