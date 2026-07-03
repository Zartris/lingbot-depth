"""
run.py  —  the auto-research runner (the keep-or-revert loop around train.py).

One iteration:
  1. train a student  (train.main -> runs/<id>/student.pt)
  2. eval it on the frozen eval set  (prepare.evaluate -> score + accuracy + latency)
  3. append the result to runs/results.csv
  4. if --accept and the score improved on the best so far, git-commit train.py

The research agent's loop is: read program.md + results.csv -> edit train.py -> run
this -> read the new score -> repeat.  Because prepare.py and mdm/ are frozen, the
only way the score can improve is a genuinely better student.

Usage:
  python -m distill.run --setup-baseline   # measure the teacher once (do this first)
  python -m distill.run                     # one train+eval iteration, log the score
  python -m distill.run --accept            # ... and git-commit train.py if it improved
  python -m distill.run --smoke             # tiny local (no-download) plumbing check
  python -m distill.run --eval-only PATH    # score an existing student.pt
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

# Make `distill` importable whether launched via `-m distill.run` or `distill/run.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill import prepare  # noqa: E402


def _load_student(ckpt_path: Path):
    """Rebuild a student from a saved checkpoint (uses train.build_student's config)."""
    import torch
    from mdm.model.v2 import MDMModel

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = prepare.derive_student_config(
        backbone=ckpt["student_backbone"],
        num_tokens_range=ckpt.get("num_tokens_range"),
    )
    model = MDMModel(**cfg).to(prepare.device())
    model.load_state_dict(ckpt["model"], strict=False)
    return model.eval()


def _eval_set(use_hf: bool):
    # The eval set is frozen in prepare.py and disjoint from the training pool.
    return prepare.eval_dataset(use_hf=use_hf)


def _append_result(row: dict) -> None:
    path = prepare.RUNS_DIR / "results.csv"
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def setup_baseline(use_hf: bool) -> None:
    print("[run] measuring teacher baseline (accuracy + latency) ...")
    base = prepare.measure_teacher_baseline(_eval_set(use_hf))
    print(f"[run] teacher: latency={base.latency_ms:.1f}ms  "
          f"absrel_real={base.absrel_real:.4f}  delta1_real={base.delta1_real:.3f}  "
          f"params={base.params/1e6:.1f}M")
    print(f"[run] wrote {prepare.RUNS_DIR/'teacher_baseline.json'}")


def one_iteration(accept: bool, use_hf: bool) -> dict:
    from distill import train

    base = prepare.load_baseline()
    if base is None:
        print("[run] no teacher baseline yet — run `--setup-baseline` first.")
        sys.exit(2)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = prepare.RUNS_DIR / run_id
    print(f"[run] === iteration {run_id} ===")

    ckpt = train.main(run_dir, use_hf=use_hf)
    student = _load_student(ckpt)

    print("[run] evaluating ...")
    res = prepare.evaluate(student, _eval_set(use_hf), base=base, teacher=prepare.load_teacher())

    row = {"run_id": run_id, "backbone": train.STUDENT_BACKBONE,
           "tokens": str(train.STUDENT_NUM_TOKENS_RANGE), **{k: round(v, 5) for k, v in res.items()}}
    _append_result(row)

    print(f"[run] score={res['score']:.2f}  latency={res['latency_ms']:.1f}ms  "
          f"speedup={res.get('speedup_vs_teacher', 0):.2f}x  "
          f"absrel_w={res['absrel_weighted']:.4f}  delta1_w={res['delta1_weighted']:.3f}  "
          f"params={res['params']/1e6:.1f}M")

    prev_best = _best_score_excluding(run_id)
    improved = res["score"] < prev_best
    print(f"[run] {'IMPROVED' if improved else 'no improvement'} "
          f"(best-so-far={prev_best:.2f})")

    if accept and improved:
        _git_commit(run_id, res)
    return res


def _best_score_excluding(run_id: str) -> float:
    path = prepare.RUNS_DIR / "results.csv"
    if not path.exists():
        return float("inf")
    best = float("inf")
    with path.open() as f:
        for r in csv.DictReader(f):
            if r.get("run_id") == run_id:
                continue
            try:
                best = min(best, float(r["score"]))
            except (KeyError, ValueError):
                pass
    return best


def _git_commit(run_id: str, res: dict) -> None:
    msg = (f"distill: accept {run_id} "
           f"(score={res['score']:.1f}, {res.get('speedup_vs_teacher',0):.2f}x, "
           f"absrel_w={res['absrel_weighted']:.4f})")
    subprocess.run(["git", "add", "distill/train.py", "distill/runs/results.csv"], cwd=prepare.REPO_ROOT)
    subprocess.run(["git", "commit", "-m", msg], cwd=prepare.REPO_ROOT)
    print(f"[run] committed: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup-baseline", action="store_true", help="measure teacher, then exit")
    ap.add_argument("--accept", action="store_true", help="git-commit train.py if score improved")
    ap.add_argument("--smoke", action="store_true", help="local no-download plumbing check")
    ap.add_argument("--eval-only", type=str, default=None, help="score an existing student.pt")
    ap.add_argument("--hf", action="store_true", help="use the streamed HF dataset (default: local)")
    args = ap.parse_args()

    use_hf = args.hf and not args.smoke

    if args.eval_only:
        base = prepare.load_baseline()
        student = _load_student(Path(args.eval_only))
        res = prepare.evaluate(student, _eval_set(use_hf), base=base, teacher=prepare.load_teacher())
        print(res)
        return

    if args.setup_baseline:
        setup_baseline(use_hf)
        return

    if args.smoke:
        # Minimal: derive a student config and run the download-free selftest.
        print("[smoke] running prepare self-test ...")
        prepare._selftest()
        print("[smoke] To exercise the full loop on GPU: "
              "`python -m distill.run --setup-baseline` then `python -m distill.run`.")
        return

    one_iteration(accept=args.accept, use_hf=use_hf)


if __name__ == "__main__":
    main()
