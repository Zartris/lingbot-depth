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
| `runs/` | output | Per-experiment checkpoints, `results.csv`, `teacher_baseline.json`. |
| `.cache/` | output | Downloaded dataset frames + cached teacher targets. |

## The metric (one number, lower is better)

```
score = latency_ms * (1 + PENALTY * <accuracy violations vs teacher, on GT depth>)
```

Latency = median single-frame forward on this GPU. Accuracy = AbsRel/δ1 vs
ground-truth depth, weighted `0.7*real + 0.3*sim` (deployment is a real sensor; sim
is a clean-GT cross-check). Params/FLOPs are logged as portability signals.

## Quick start

```bash
# 0. env (from repo root)
python -m pip install -e .

# 1. sanity-check the harness logic — no GPU, no downloads
python -m distill.run --smoke

# 2. measure the teacher once (needs GPU + downloads the teacher ckpt)
python -m distill.run --setup-baseline            # add --hf to use the real dataset

# 3. one research iteration: train a student, eval, log the score
python -m distill.run                              # local examples (plumbing)
python -m distill.run --hf --accept                # real data; commit train.py if better
```

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

## What the student is

`MDMModel` is fully config-instantiated, so a student is just a smaller `model_config`
+ trained weights. `prepare.derive_student_config()` builds a valid one from the
teacher's: swap the DINOv2 backbone (L→S/B), which sets `dim_out` and the neck's first
input width; everything downstream transfers, so the student warm-starts its decoder
straight from the teacher and only the encoder must be distilled.
