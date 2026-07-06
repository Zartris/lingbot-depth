"""
train.py  —  THE MUTABLE FILE.  This is the only model/training file the research
agent is allowed to rewrite (see distill/program.md for the rules).

Goal: distill the frozen teacher (LingBot-Depth ViT-L) into a student that is much
faster on our GPU while staying within the accuracy band defined in prepare.py.

What you (the agent) may change here, freely:
  * STUDENT_CONFIG / build_student() — backbone size, intermediate_layers, neck & head
    widths. (num_tokens_range is FIXED to the teacher's — the resolution 0-9 dial is a
    preserved user feature, not a search knob; see below.)
  * The distillation recipe — which losses (output / feature / gt), their weights,
    optimizer, LR schedule, augmentations, batch size, steps.
  * Anything else in THIS file.
  * distill/student_model/ — the MUTABLE copy of the model stack. Edit the network
    itself here (attention, patch embed, blocks, forward/infer, output heads, token
    merging/pruning, ...). Keep the infer() contract below.

What you may NOT change:
  * distill/prepare.py (data, teacher, metrics, latency, objective) — frozen.
  * the mdm/ package (defines the frozen teacher) — frozen. Editing it would change
    the teacher too and corrupt the targets + eval reference. Use distill/student_model/.
  * The contract: whatever build_student() returns MUST expose
        infer(image, depth_in=..., intrinsics=...) -> {"depth", "points", "mask"}
    with the same tensor shapes as the teacher, so prepare.evaluate() can score it.

Run one experiment with:   python distill/run.py            (train + eval + log)
This file's `main()` does the training and writes runs/<id>/student.pt.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

# Make `distill` importable whether launched via `-m distill.run`, `distill/train.py`,
# or imported by run.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill import prepare  # noqa: E402
from distill.prepare import (  # noqa: E402
    device, load_teacher, derive_student_config,
)

# --------------------------------------------------------------------------- #
#  Experiment knobs  (the agent edits these)                                   #
# --------------------------------------------------------------------------- #

# --- student architecture --------------------------------------------------- #
# The search STARTS AT THE TEACHER and shrinks incrementally (see program.md), so the
# seed IS the teacher backbone — it inherits ~all teacher weights and starts at ~teacher
# accuracy. You make the network smaller one small, weight-inheriting step at a time
# (drop/merge blocks, prune heads/width — edit distill/student_model/), NOT by jumping
# to a small backbone from scratch.
STUDENT_BACKBONE = "dinov2_vitl14"     # start = teacher; shrink from here
# The search space is OPEN — you MAY switch to a smaller backbone (vitb14/vits14) or any
# other big change, isolated and tested. But a big low-transfer move (e.g. a fresh ViT-S
# encoder) won't recover accuracy inside the default budget, so the proxy will reject it
# regardless of its real potential; give such a bet a longer run with `--budget-min`
# (see program.md). The DEFAULT path is incremental: shrink the teacher-sized net one
# small, weight-inheriting step at a time (drop/merge blocks, prune heads/width).
#
# num_tokens_range is FIXED to the teacher's (None -> derive keeps the teacher's range).
# The resolution_level 0-9 / token dial is a PRESERVED USER FEATURE, not a search knob:
# you must NOT "win" by using fewer tokens (the user already chooses that at runtime, and
# the score is measured ACROSS levels). Speedups must come from a cheaper-per-token net.
STUDENT_NUM_TOKENS_RANGE = None

# --- distillation recipe ---------------------------------------------------- #
W_OUTPUT_DISTILL = 1.0     # match teacher refined depth (linear-space L1)
W_FEATURE_DISTILL = 1.0    # match teacher encoder features (via a learned 1x1 projector)
W_GT = 0.5                 # supervise on real ground-truth depth where available
FEATURE_TOKENS = 1200      # token grid used during training (speed vs fidelity)

# --- budget: YOU rate the change and set the time ---------------------------- #
# Rate how big a change this experiment is and budget TO CONVERGENCE (until the loss /
# accuracy plateaus), not beyond:
#   * small, high-transfer change (drop a block, prune heads) -> converges fast; leave
#     None (uses the default prepare.TRAIN_MINUTES = 90).
#   * big, low-transfer change (e.g. a fresh smaller backbone) -> needs much longer to
#     show its real potential; raise this, up to prepare.MAX_BUDGET_MINUTES (8 h cap).
# Over-budgeting a converged change just burns compute AND confounds comparison with
# shorter runs — size it honestly. The effective budget is logged in results.csv.
BUDGET_MINUTES = None      # None -> prepare.TRAIN_MINUTES; else minutes for THIS experiment

# --- optimisation ----------------------------------------------------------- #
# The student is initialised FROM the teacher (or best student), i.e. near a good
# optimum, so fine-tune GENTLY: AdamW takes ~full-LR steps even at near-zero loss, so a
# high LR / high weight-decay walks a good init straight off its optimum (observed: a
# teacher-perfect seed collapsed in ~9 steps at LR 2e-4). Low LR + warmup + low WD.
LR = 5e-5                  # peak LR (fine-tuning, not from-scratch)
WARMUP_STEPS = 100         # linear LR warmup from 0 -> LR (lets Adam's variance settle)
WEIGHT_DECAY = 0.01        # low: high WD pulls inherited weights toward 0 (destructive)
BATCH_SIZE = 2
INHERIT_TEACHER = True     # copy ALL matching teacher weights (encoder + decoder). At the
                           # teacher-sized seed this inherits the FULL teacher; as you
                           # shrink, whatever still matches by name+shape is inherited.
# Where the student's initial weights come from, as a priority cascade (later wins on
# name+shape overlap):  fresh(stock DINOv2) < teacher (all matching) < best student.
#   "best"    -> inherit the best student so far (encoder + decoder),
#                falling back to teacher decoder / fresh for anything that doesn't match.
#                On iteration 1 (no best yet) this is identical to "teacher".
#   "teacher" -> stock DINOv2 encoder + teacher decoder only (fixed init; clean A/B).
#   "fresh"   -> stock DINOv2 encoder only, nothing from teacher/best.
# NB: the teacher's 1024-d encoder never matches the student's 384-d encoder, so ONLY
# "best" can warm-start the student encoder — otherwise it re-distills from scratch
# every run. Downside of "best": the search becomes cumulative, so a recipe-only tweak
# can look good just from inheriting a fine-tuned parent — use "teacher" for a clean
# isolated test when that matters.
WARM_START_FROM = "best"

# NOTE: the training data pool, the eval set, and the compute budget
# (TRAIN_MINUTES / MAX_STEPS) are FROZEN in distill/prepare.py — not here — so every
# experiment runs on equal data at equal compute and the agent can't change its own
# exam. You control how the data is *used* (BATCH_SIZE, sampling, augmentation), not
# how much there is or how long you train.


# --------------------------------------------------------------------------- #
#  Student                                                                      #
# --------------------------------------------------------------------------- #

def build_student() -> "tuple[torch.nn.Module, dict]":
    """Instantiate the student and warm-start it. Returns (model, config).

    The student is the MUTABLE copy of the model stack in `distill/student_model/` —
    edit that tree to change the network itself (attention, patch embed, blocks,
    forward/infer, output heads), not just the config here. The teacher's `mdm/model/`
    stays frozen.

    Warm-start is a priority cascade (see WARM_START_FROM), highest priority applied
    last so it wins on name+shape overlap:

        fresh (stock DINOv2 encoder, via init_weights)
          < teacher (encoder + decoder, ALL tensors that match by name+shape)
            < best student so far (encoder + decoder, ALL matching tensors)

    Because the seed IS the teacher backbone, the teacher copy inherits the FULL teacher
    (encoder + decoder) -> the student starts at ~teacher accuracy. As you shrink the net,
    only the changed layers stop matching; the rest keep inheriting from the best student
    (which stays near-teacher accuracy, so it's a high-overlap source) or the teacher.
    """
    from distill.student_model.v2 import MDMModel as StudentMDMModel

    cfg = derive_student_config(
        backbone=STUDENT_BACKBONE,
        num_tokens_range=STUDENT_NUM_TOKENS_RANGE,
    )
    student = StudentMDMModel(**cfg).to(device())
    student.init_weights()  # base: encoder <- stock DINOv2, everything else fresh

    n_teacher = n_best = 0
    if WARM_START_FROM in ("best", "teacher") and INHERIT_TEACHER:
        n_teacher = _copy_matching(
            student, load_teacher().state_dict(), prefixes=None)  # ALL matching (enc+dec)
    if WARM_START_FROM == "best":
        best = prepare.best_student_ckpt()
        if best is not None:
            best_sd = torch.load(best, map_location="cpu", weights_only=False)["model"]
            n_best = _copy_matching(student, best_sd, prefixes=None)  # inherit ALL matching
            print(f"[train] warm-start parent: {best}")
    print(f"[train] warm-start: {n_teacher} tensors from teacher (enc+dec), "
          f"{n_best} from best student ({WARM_START_FROM})")
    return student, cfg


def _copy_matching(dst: torch.nn.Module, src_state: Dict[str, torch.Tensor],
                   prefixes=None) -> int:
    """Copy tensors from a source state_dict into dst where the name exists, the shape
    matches, and (if `prefixes` is given) the name starts with one of them. Params with
    no match are left untouched, so successive calls compose as a priority cascade.
    Returns the number of tensors copied."""
    dsd = dst.state_dict()
    n = 0
    for k, v in dsd.items():
        if k in src_state and src_state[k].shape == v.shape and (
                prefixes is None or any(k.startswith(p) for p in prefixes)):
            dsd[k] = src_state[k].clone()
            n += 1
    dst.load_state_dict(dsd, strict=False)
    return n


# --------------------------------------------------------------------------- #
#  Batch prep                                                                   #
# --------------------------------------------------------------------------- #

def _depth_l1(pred, target, valid):
    """Masked L1 in LINEAR (metric) depth space. The model's output remap is 'linear',
    so the head can emit negative depth; a log-space loss would clamp those to a
    constant and give ZERO gradient (the student then drifts to negative depth — this
    actually happened). Linear L1 penalises negatives directly and pulls them back up."""
    if valid.sum() == 0:
        return pred.new_zeros(())
    return (F.l1_loss(pred, target, reduction="none") * valid).sum() / valid.sum().clamp_min(1)


def distill_step(student, teacher, samples, dev, feat_proj=None) -> Dict[str, torch.Tensor]:
    # Teacher targets come from prepare.teacher_targets — a per-sample disk cache, so the
    # ViT-L teacher runs once per (sample, FEATURE_TOKENS) instead of every step. It also
    # returns the canonical (TRAIN_HW) inputs the student trains on.
    #   - t_feat: teacher encoder features at FEATURE_TOKENS (grid matches the student's).
    #   - t_depth: teacher BEST-quality refined depth (infer() default level) — the
    #     student learns the teacher's best output while running its own smaller budget.
    imgs, raws, Ks, t_feat, t_depth = prepare.teacher_targets(samples, FEATURE_TOKENS, teacher, dev)

    # Student forward (with grad). Feature + output heads.
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
        s_feat, s_cls = student.forward_feat(imgs, num_tokens=FEATURE_TOKENS, depth=raws)
        s_out = student.forward(imgs, num_tokens=FEATURE_TOKENS, depth=raws)
    s_depth = s_out["depth_reg"]

    losses: Dict[str, torch.Tensor] = {}

    # Feature distillation — the main lever for keeping accuracy. The student encoder
    # emits `dim_out` channels (e.g. 384 for ViT-S) and the teacher a different count
    # (1024 for ViT-L), so a learnable 1x1 projector (feat_proj) maps the student
    # features up to the teacher's channel dim before matching (cosine + L2). The
    # projector is a training-only helper — it is NOT part of the student at inference.
    if W_FEATURE_DISTILL > 0:
        sf, tf = s_feat.float(), t_feat.float()
        if feat_proj is not None:
            sf = feat_proj(sf)
        if sf.shape[-2:] != tf.shape[-2:]:
            sf = F.interpolate(sf, tf.shape[-2:], mode="bilinear", align_corners=False)
        cos = 1 - F.cosine_similarity(sf, tf, dim=1).mean()
        losses["feat"] = W_FEATURE_DISTILL * (cos + 0.1 * F.mse_loss(sf, tf))

    # Output distillation — match the teacher's refined depth.
    if W_OUTPUT_DISTILL > 0:
        valid = (t_depth > prepare.MIN_DEPTH) & (t_depth < prepare.MAX_DEPTH) & torch.isfinite(t_depth)
        losses["out"] = W_OUTPUT_DISTILL * _depth_l1(s_depth.float(), t_depth.float(), valid.float())

    # Ground-truth supervision — only where real gt is present.
    if W_GT > 0:
        gts = [s.gt_depth for s in samples]
        if any(g is not None for g in gts):
            import cv2
            H, W = imgs.shape[-2:]
            g_stack, m_stack = [], []
            for g in gts:
                if g is None:
                    g_stack.append(np.zeros((H, W), np.float32)); m_stack.append(np.zeros((H, W), np.float32))
                else:
                    gg = cv2.resize(g, (W, H), interpolation=cv2.INTER_NEAREST)
                    g_stack.append(gg); m_stack.append((gg > prepare.MIN_DEPTH).astype(np.float32))
            gt = torch.tensor(np.stack(g_stack), device=dev)
            mask = torch.tensor(np.stack(m_stack), device=dev)
            if mask.sum() > 0:
                losses["gt"] = W_GT * _depth_l1(s_depth.float(), gt, mask)

    losses["total"] = sum(losses.values()) if losses else s_depth.new_zeros(())
    return losses


# --------------------------------------------------------------------------- #
#  Train                                                                        #
# --------------------------------------------------------------------------- #

def resolve_budget(minutes: float = None) -> float:
    """Effective training budget (minutes) for a run, clamped to the frozen safety cap.
    Precedence: explicit `minutes` (CLI --budget-min) > BUDGET_MINUTES knob > default."""
    b = minutes if minutes is not None else (
        BUDGET_MINUTES if BUDGET_MINUTES is not None else prepare.TRAIN_MINUTES)
    return min(float(b), prepare.MAX_BUDGET_MINUTES)


def main(run_dir: Path, use_hf: bool = False,
         minutes: float = None, max_steps: int = None) -> Path:
    # Budget: the agent rates the change and sets BUDGET_MINUTES (or a human/CLI passes
    # --budget-min); clamped to prepare.MAX_BUDGET_MINUTES. See the BUDGET_MINUTES knob.
    budget_min = resolve_budget(minutes)
    budget_steps = prepare.MAX_STEPS if max_steps is None else max_steps

    torch.manual_seed(prepare.SEED)
    dev = device()
    print(f"[train] device={dev} backbone={STUDENT_BACKBONE} tokens={STUDENT_NUM_TOKENS_RANGE} "
          f"budget={budget_min:.2f}min/{budget_steps}steps")

    teacher = load_teacher()
    student, student_cfg = build_student()
    student.train()

    train_set = prepare.train_dataset(use_hf=use_hf)   # FROZEN pool (see prepare.py)
    n = len(train_set)
    print(f"[train] {n} training samples ({'HF stream' if use_hf else 'local examples'})")

    # Feature-distillation projector: student dim_out -> teacher dim_out (training-only).
    feat_proj = None
    if W_FEATURE_DISTILL > 0:
        s_dim = student.encoder.output_projections[0].out_channels
        t_dim = teacher.encoder.output_projections[0].out_channels
        if s_dim != t_dim:
            feat_proj = torch.nn.Conv2d(s_dim, t_dim, kernel_size=1).to(dev)
            print(f"[train] feature-distill projector: {s_dim} -> {t_dim}")

    params = [p for p in student.parameters() if p.requires_grad]
    if feat_proj is not None:
        params += list(feat_proj.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)

    rng = np.random.default_rng(prepare.SEED)
    t_start = time.time()
    step = 0
    while (time.time() - t_start) < budget_min * 60 and step < budget_steps:
        idx = rng.integers(0, n, size=BATCH_SIZE)
        samples = [train_set[int(i)] for i in idx]
        losses = distill_step(student, teacher, samples, dev, feat_proj=feat_proj)
        lr_now = LR * min(1.0, (step + 1) / max(1, WARMUP_STEPS))   # linear warmup
        for g in opt.param_groups:
            g["lr"] = lr_now
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 20 == 0:
            msg = " ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
            print(f"[train] step {step:5d} t={time.time()-t_start:6.1f}s {msg}")
        step += 1

    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_pkg = _snapshot_student_code(run_dir)   # freeze the student CODE with the weights
    ckpt_path = run_dir / "student.pt"
    torch.save({
        "model": student.state_dict(),
        "student_config": student_cfg,          # rebuild exactly, without re-deriving
        "snapshot_pkg": snapshot_pkg,           # the code that defines this architecture
        "student_backbone": STUDENT_BACKBONE,
        "num_tokens_range": STUDENT_NUM_TOKENS_RANGE,
        "steps": step,
    }, ckpt_path)
    print(f"[train] done: {step} steps, saved {ckpt_path} (+ code snapshot {snapshot_pkg}/)")
    return ckpt_path


def _snapshot_student_code(run_dir: Path) -> str:
    """Copy the current distill/student_model/ source next to the checkpoint, so this
    student's exact architecture can be rebuilt later even after the live student code
    changes. Returns the snapshot's package name (a valid importable identifier).

    A checkpoint stores weights + config but NOT the code that defines the architecture;
    since that code is mutable, the snapshot is what makes a past student reproducible."""
    import shutil
    snap_name = "student_snap_" + "".join(c if c.isalnum() else "_" for c in run_dir.name)
    dest = run_dir / snap_name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(prepare.REPO_ROOT / "distill" / "student_model", dest,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return snap_name


if __name__ == "__main__":
    main(prepare.RUNS_DIR / "manual")
