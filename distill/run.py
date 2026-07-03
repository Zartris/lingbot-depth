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
  python -m distill.run --compare [PATH]    # teacher vs best-student vs [PATH] across
                                            #   resolution levels (latency/acc/speedup)
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

# Make `distill` importable whether launched via `-m distill.run` or `distill/run.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill import prepare  # noqa: E402


def _student_class_from_snapshot(ckpt_path: Path, snap: str):
    """Import the student's MDMModel class from the per-checkpoint CODE SNAPSHOT next to
    it, so we rebuild the exact architecture that was trained — independent of whatever
    the live distill/student_model/ code looks like now."""
    import importlib
    parent = str(Path(ckpt_path).resolve().parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    for m in list(sys.modules):            # ensure a clean import of this snapshot
        if m == snap or m.startswith(snap + "."):
            del sys.modules[m]
    return importlib.import_module(f"{snap}.v2").MDMModel


def _load_student(ckpt_path: Path):
    """Rebuild a student from a checkpoint, using the code snapshot saved with it so the
    exact trained architecture is reconstructed regardless of later edits to the live
    student code. Falls back to the live stack for old (pre-snapshot) checkpoints."""
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("student_config") or prepare.derive_student_config(
        backbone=ckpt["student_backbone"],
        num_tokens_range=ckpt.get("num_tokens_range"),
    )
    snap = ckpt.get("snapshot_pkg")
    if snap and (Path(ckpt_path).resolve().parent / snap).exists():
        StudentMDMModel = _student_class_from_snapshot(ckpt_path, snap)
    else:
        if snap:
            print(f"[run] WARNING: code snapshot '{snap}' missing next to {ckpt_path}; "
                  "falling back to live student code (may mis-load if it changed).")
        from distill.student_model.v2 import MDMModel as StudentMDMModel

    model = StudentMDMModel(**cfg).to(prepare.device())
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[run] WARNING: loading {ckpt_path}: {len(missing)} missing / "
              f"{len(unexpected)} unexpected keys — checkpoint/code may be out of sync.")
    return model.eval()


def _eval_set(use_hf: bool):
    # The eval set is frozen in prepare.py and disjoint from the training pool.
    return prepare.eval_dataset(use_hf=use_hf)


def find_best_student() -> "Path | None":
    """Path to the best-scoring student.pt so far (delegates to the frozen helper)."""
    return prepare.best_student_ckpt()


def _print_level_table(rows: list) -> None:
    print(f"      {'level':>5} {'model':>8} {'latency_ms':>11} {'speedup':>8} "
          f"{'absrel_w':>9} {'delta1_w':>9}")
    for r in rows:
        sp = f"{r['speedup_vs_teacher']:.2f}x" if r["speedup_vs_teacher"] else "   -  "
        print(f"      {r['resolution_level']:>5} {r['model']:>8} {r['latency_ms']:>11.1f} "
              f"{sp:>8} {r['absrel_weighted']:>9.4f} {r['delta1_weighted']:>9.4f}")


def compare(current_ckpt: "str | None", use_hf: bool) -> None:
    """Compare teacher vs best-student-so-far vs a given/current student across
    resolution levels (latency, accuracy, speedup at each level)."""
    models = {"teacher": prepare.load_teacher()}
    best = find_best_student()
    if best is not None:
        models["best"] = _load_student(best)
        print(f"[run] best student: {best}")
    if current_ckpt:
        models["current"] = _load_student(Path(current_ckpt))
        print(f"[run] current student: {current_ckpt}")
    if "best" not in models and not current_ckpt:
        print("[run] no student checkpoints yet — showing teacher sweep only.")
    print("[run] latency/accuracy/speedup across resolution_level:")
    rows = prepare.compare_across_levels(models, _eval_set(use_hf))
    _print_level_table(rows)
    print(f"[run] wrote {prepare.RUNS_DIR/'level_comparison.json'}")


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
    eval_set = _eval_set(use_hf)
    base = prepare.measure_teacher_baseline(eval_set)
    print(f"[run] teacher @ L{prepare.BENCH_RESOLUTION_LEVEL}: latency={base.latency_ms:.1f}ms  "
          f"absrel_real={base.absrel_real:.4f}  delta1_real={base.delta1_real:.3f}  "
          f"params={base.params/1e6:.1f}M")

    print("[run] teacher latency-vs-accuracy across resolution_level (free speed lever):")
    rows = prepare.teacher_resolution_report(eval_set)
    print(f"      {'level':>5} {'latency_ms':>11} {'absrel_w':>9} {'delta1_w':>9}")
    for r in rows:
        print(f"      {r['resolution_level']:>5} {r['latency_ms']:>11.1f} "
              f"{r['absrel_weighted']:>9.4f} {r['delta1_weighted']:>9.4f}")
    print(f"[run] wrote {prepare.RUNS_DIR/'teacher_baseline.json'} and "
          f"{prepare.RUNS_DIR/'teacher_resolution_sweep.json'}")


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
        promote_best(run_id, res)     # copy champion into the committed distill/best/
        _git_commit(run_id, res)
    return res


def promote_best(run_id: str, res: dict) -> None:
    """Copy the winning run's weights + code snapshot + metadata into the committed
    distill/best/, so the champion travels across machines (weights via LFS)."""
    import shutil
    import torch
    from distill import train

    src = prepare.RUNS_DIR / run_id
    dst = prepare.BEST_DIR
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # Save weights in fp16 (halves size; inference runs bf16 autocast anyway) so the
    # champion fits under GitHub's 100MB blob limit without LFS. Guard the size: if it's
    # still too big, keep only the code snapshot + metadata (weights stay local) so the
    # push is never rejected.
    ck = torch.load(src / "student.pt", map_location="cpu", weights_only=False)
    ck_half = {**ck, "model": {k: (v.half() if torch.is_floating_point(v) else v)
                               for k, v in ck["model"].items()}, "saved_dtype": "float16"}
    torch.save(ck_half, dst / "student.pt")
    size_mb = (dst / "student.pt").stat().st_size / 1e6
    weights_committed = size_mb <= prepare.MAX_COMMIT_WEIGHTS_MB
    if not weights_committed:
        (dst / "student.pt").unlink()
        print(f"[run] champion weights are {size_mb:.0f}MB > {prepare.MAX_COMMIT_WEIGHTS_MB}MB "
              "limit — committing code snapshot + metadata only (weights stay local; "
              "use a smaller backbone or LFS to make them travel).")

    snap = ck.get("snapshot_pkg")
    if snap and (src / snap).exists():
        shutil.copytree(src / snap, dst / snap,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      cwd=prepare.REPO_ROOT).decode().strip()
    except Exception:
        sha = None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    meta = {"run_id": run_id, "gpu": gpu, "git_sha": sha,
            "weights_committed": weights_committed, "weights_mb": round(size_mb, 1),
            "backbone": train.STUDENT_BACKBONE, "tokens": str(train.STUDENT_NUM_TOKENS_RANGE),
            **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()}}
    (dst / "best.json").write_text(json.dumps(meta, indent=2))
    subprocess.run(["git", "add", "distill/best"], cwd=prepare.REPO_ROOT)
    print(f"[run] promoted {run_id} -> distill/best/  (score={res.get('score', float('nan')):.2f}, "
          f"{size_mb:.0f}MB fp16, gpu={gpu})  — scores are machine-specific; re-benchmark elsewhere")


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
    subprocess.run(["git", "add", "distill/train.py", "distill/student_model",
                    "distill/runs/results.csv"], cwd=prepare.REPO_ROOT)
    subprocess.run(["git", "commit", "-m", msg], cwd=prepare.REPO_ROOT)
    print(f"[run] committed: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup-baseline", action="store_true", help="measure teacher, then exit")
    ap.add_argument("--accept", action="store_true", help="git-commit train.py if score improved")
    ap.add_argument("--smoke", action="store_true", help="local no-download plumbing check")
    ap.add_argument("--eval-only", type=str, default=None, help="score an existing student.pt")
    ap.add_argument("--compare", nargs="?", const="", default=None,
                    help="teacher vs best-student vs [optional student.pt] across "
                         "resolution levels, then exit")
    ap.add_argument("--cache-targets", action="store_true",
                    help="precompute the teacher-target cache over the training pool, then exit")
    ap.add_argument("--hf", action="store_true", help="use the streamed HF dataset (default: local)")
    args = ap.parse_args()

    use_hf = args.hf and not args.smoke

    if args.cache_targets:
        from distill import train
        print("[run] precomputing teacher targets (this runs the teacher once per sample) ...")
        n = prepare.precompute_teacher_targets(train.FEATURE_TOKENS, use_hf=use_hf)
        print(f"[run] cached teacher targets for {n} samples "
              f"at FEATURE_TOKENS={train.FEATURE_TOKENS}")
        return

    if args.compare is not None:
        compare(args.compare or None, use_hf)
        return

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
