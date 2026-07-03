"""
prepare.py  —  IMMUTABLE harness for the LingBot-Depth distillation auto-research loop.

Inspired by karpathy/autoresearch: this file holds everything the research agent is
NOT allowed to touch — the frozen teacher, the dataset, the evaluation metrics, the
latency benchmark, and the single scalar objective. Freezing this file is what keeps
the metric honest: the agent can only improve the score by producing a genuinely
faster-yet-accurate student, not by editing how it is measured.

The agent edits `train.py`. It may `from distill.prepare import ...` but must never
modify this file (nor the `mdm/` package, which defines the frozen teacher).

Contents
--------
  Constants            budgets, tolerances, the sim/real accuracy weighting
  load_teacher()       frozen teacher MDMModel (cached)
  derive_student_config()  build a valid smaller MDMModel config from the teacher's
  train_dataset()/eval_dataset()  frozen, disjoint subset of robbyant/mdm_depth
  cache_teacher_targets()  precompute + cache teacher depth/features for fast inner loop
  depth_metrics()      AbsRel / RMSE / SILog / delta1..3 / masked-completion error
  benchmark_latency()  warmup + CUDA-synced median ms + params
  evaluate()           run the frozen eval set -> dict(score, latency_ms, absrel, ...)
  objective()          collapse (latency, accuracy) into ONE scalar (lower is better)

Nothing here imports train.py; the dependency only goes the other way.
"""
from __future__ import annotations

import copy
import json
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ----------------------------------------------------------------------------- #
#  Constants  (the agent must NOT change these — they define the problem)        #
# ----------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "distill" / ".cache"        # teacher targets, downloaded frames
RUNS_DIR = REPO_ROOT / "distill" / "runs"           # per-experiment ckpts + results
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Frozen teacher.
TEACHER_MODEL_ID = "robbyant/lingbot-depth-pretrain-vitl-14-v0.5"

# HF dataset (2.71 TB in full — we only ever stream a subset).
DATASET_ID = "robbyant/mdm_depth"

# How the two eval domains are weighted into the accuracy term.
# Deployment target is a real robot sensor, so real dominates; sim (perfect GT) is a
# low-noise cross-check.
ACC_WEIGHT_REAL = 0.7
ACC_WEIGHT_SIM = 0.3

# Accuracy tolerance band vs. the teacher, evaluated against ground-truth depth.
# The student is allowed to be this much worse than the teacher before it gets
# penalised. Tighten these to demand more accuracy, loosen to buy more speed.
ACC_TOLERANCE = 0.05          # 5% — absrel may rise by up to 5% relative to teacher
DELTA_TOLERANCE = 0.01        # delta1 may drop by up to 1 abs point vs teacher
PENALTY = 50.0                # multiplier on tolerance violations (in ms-equivalent)

# Depth eval range (metres). Pixels outside are ignored in the metrics.
MIN_DEPTH = 0.1
MAX_DEPTH = 10.0

# Latency benchmark settings (measured on THIS machine's GPU, PyTorch/CUDA).
BENCH_INPUT_HW = (480, 640)   # fixed input size so latency numbers are comparable
BENCH_WARMUP = 10
BENCH_ITERS = 50
# resolution_level (0-9) scales the token grid in the model's pre-processing and
# DOMINATES latency (measured on an RTX2080ti: L0~128ms, L3~150ms, L6~216ms, L9~307ms),
# while the input image resolution does NOT affect it. Latency and accuracy are always
# read at the SAME level (BENCH_RESOLUTION_LEVEL) so they describe one operating point.
BENCH_RESOLUTION_LEVEL = 9    # the operating point the score is computed at
# Reported once for the teacher to expose its latency-vs-accuracy Pareto curve — the
# curve the student ultimately has to beat (lowering the teacher's level is ~free speed).
RESOLUTION_SWEEP = [0, 3, 6, 9]

# ---- Dataset subset (FROZEN) -------------------------------------------------
# The same training pool and the same eval set for EVERY experiment. Frozen here so
# a score difference reflects a better student, not more/different data, and so the
# agent can never change what it is scored on (the eval set is its exam).
N_TRAIN_REAL = 4000
N_TRAIN_SIM = 1500
N_EVAL_REAL = 200
N_EVAL_SIM = 100

# ---- Compute budget per experiment (FROZEN) ----------------------------------
# Equal compute per run => comparable scores (autoresearch's fixed-budget premise).
# The agent optimises what to do WITHIN this budget, not the budget itself.
TRAIN_MINUTES = 20.0          # proxy-run wall-clock; humans promote winners to longer runs
MAX_STEPS = 100_000           # hard cap

SEED = 1234


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- #
#  Teacher                                                                       #
# ----------------------------------------------------------------------------- #

_TEACHER_CACHE: Dict[str, Any] = {}


def load_teacher(model_id: str = TEACHER_MODEL_ID) -> "torch.nn.Module":
    """Load the frozen teacher once and memoise it. Eval mode, grads off."""
    from mdm.model.v2 import MDMModel

    if model_id not in _TEACHER_CACHE:
        model = MDMModel.from_pretrained(model_id).to(device()).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _TEACHER_CACHE[model_id] = model
    return _TEACHER_CACHE[model_id]


def teacher_config(model_id: str = TEACHER_MODEL_ID) -> Dict[str, Any]:
    """Return the teacher's `model_config` dict (as stored in the checkpoint)."""
    from huggingface_hub import hf_hub_download

    ckpt_path = model_id
    if not Path(model_id).exists():
        ckpt_path = hf_hub_download(repo_id=model_id, repo_type="model", filename="model.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return ckpt["model_config"]


# ----------------------------------------------------------------------------- #
#  Student config derivation                                                     #
# ----------------------------------------------------------------------------- #
# The whole model is config-instantiated: MDMModel(**config). A "student" is just a
# smaller config + trained weights. Because forward() does `features + cls_token`,
# the encoder's dim_out MUST equal its backbone width (dim_features). Swapping the
# backbone therefore changes exactly two things: encoder.dim_out and the neck's first
# input width (dim_out + UV channels). Everything downstream transfers unchanged, so
# the student can warm-start its decoder straight from the teacher.

# DINOv2 backbone -> embedding width. Small/Base give the big speedups.
BACKBONE_WIDTH = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14_reg": 1024,
}
# Transformer depth per backbone family (used to remap intermediate_layers).
BACKBONE_DEPTH = {384: 12, 768: 12, 1024: 24}


def derive_student_config(
    backbone: str = "dinov2_vits14",
    num_tokens_range: Optional[List[int]] = None,
    teacher_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Produce a valid MDMModel config for a smaller student from the teacher's config.

    This is a *starting point*. The agent is free to further shrink the neck/heads,
    change `num_tokens_range`, pick different `intermediate_layers`, etc. — inside
    train.py.  Only the invariants required for a runnable model are enforced here.
    """
    cfg = copy.deepcopy(teacher_cfg if teacher_cfg is not None else teacher_config())

    if backbone not in BACKBONE_WIDTH:
        raise ValueError(f"Unknown backbone {backbone!r}; choices: {list(BACKBONE_WIDTH)}")
    student_w = BACKBONE_WIDTH[backbone]
    student_depth = BACKBONE_DEPTH[student_w]

    enc = cfg["encoder"]
    teacher_dim_out = enc["dim_out"]
    enc["backbone"] = backbone
    enc["dim_out"] = student_w          # must match dim_features for the cls_token add
    enc["strict"] = False               # tolerant warm-start from stock DINOv2 weights

    # Remap intermediate_layers to be valid for the (possibly shallower) student.
    il = enc.get("intermediate_layers")
    if isinstance(il, (list, tuple)):
        teacher_depth = BACKBONE_DEPTH[teacher_dim_out] if teacher_dim_out in BACKBONE_DEPTH else 24
        enc["intermediate_layers"] = [
            min(student_depth - 1, round(i * (student_depth - 1) / max(1, teacher_depth - 1)))
            for i in il
        ]
    # if it's an int (== "last n layers") it is already valid for any depth.

    # The neck's first input block consumes (encoder dim_out + UV channels). Preserve
    # the UV delta measured from the teacher so this stays correct regardless of how
    # many UV channels the model concatenates.
    neck = cfg["neck"]
    if isinstance(neck.get("dim_in"), list) and neck["dim_in"] and neck["dim_in"][0] is not None:
        uv_delta = neck["dim_in"][0] - teacher_dim_out
        neck["dim_in"][0] = student_w + uv_delta

    if num_tokens_range is not None:
        cfg["num_tokens_range"] = list(num_tokens_range)

    return cfg


# ----------------------------------------------------------------------------- #
#  Dataset                                                                       #
# ----------------------------------------------------------------------------- #
# Each sample is a (rgb, raw_depth, gt_depth, intrinsics) tuple.
#   input  -> raw_depth  (what the sensor gives; what the model refines)
#   target -> teacher output (distillation) and/or gt_depth (real supervision)
#   eval   -> gt_depth    (the ground truth the score is measured against)
#
# The full dataset is a 2.71 TB file tree (color/ gtdepth/ rawdepth/ per camera).
# We NEVER pull it all — _all_triplets() lists the repo and we cache a small subset.
#
# For instant plumbing checks with no download, LocalExamplesDataset uses the 8
# scenes bundled in ./examples (which have rgb + raw_depth + intrinsics but no gt,
# so the local smoke test scores fidelity-to-teacher only).


@dataclass
class Sample:
    rgb: np.ndarray            # (H, W, 3) uint8, RGB
    raw_depth: np.ndarray      # (H, W) float32, metres, 0 = invalid
    intrinsics: np.ndarray     # (3, 3) float32, UNnormalised (pixel units)
    gt_depth: Optional[np.ndarray] = None   # (H, W) float32, metres, 0 = invalid
    domain: str = "real"       # "real" | "sim"
    key: str = ""


def _load_depth_png(path: Path, scale: float = 1000.0) -> np.ndarray:
    import cv2
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise ValueError(f"failed to read depth {path}")
    d = d.astype(np.float32) / scale
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


class LocalExamplesDataset:
    """The 8 bundled examples. No gt_depth -> for smoke-testing plumbing only."""

    def __init__(self, root: Path = REPO_ROOT / "examples"):
        self.items: List[Path] = sorted(p for p in root.iterdir() if p.is_dir())

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> Sample:
        import cv2
        d = self.items[i]
        rgb_path = next((d / f"rgb{e}" for e in (".png", ".jpg", ".jpeg") if (d / f"rgb{e}").exists()))
        rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)
        raw = _load_depth_png(d / "raw_depth.png")
        K = np.loadtxt(d / "intrinsics.txt", dtype=np.float32)
        return Sample(rgb=rgb, raw_depth=raw, intrinsics=K, gt_depth=None,
                      domain="real", key=d.name)


# --- HF-streamed subset ------------------------------------------------------ #
#
# NOTE: the exact folder regexes below are written against the dataset card
# (color/ gtdepth/ rawdepth/ per camera, RobbySim*/ for simulated).  Confirm them
# against `HfApi().list_repo_files(DATASET_ID, repo_type="dataset")` the first time
# you stream — that is the single place that may need a one-line tweak, and it lives
# in this immutable file on purpose.

_RAW_DIR, _GT_DIR, _RGB_DIR = "rawdepth", "gtdepth", "color"


def _all_triplets(seed: int = SEED) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """List the dataset repo once; return (real, sim) triplet lists, shuffled
    deterministically. The listing + shuffle is cached so the split is stable."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(DATASET_ID, repo_type="dataset")
    file_set = set(files)
    raws = [f for f in files if f"/{_RAW_DIR}/" in f]

    def triplet(raw_path: str) -> Optional[Dict[str, str]]:
        rgb = raw_path.replace(f"/{_RAW_DIR}/", f"/{_RGB_DIR}/")
        gt = raw_path.replace(f"/{_RAW_DIR}/", f"/{_GT_DIR}/")
        # rgb may be jpg while depth is png
        rgb_candidates = [rgb, str(Path(rgb).with_suffix(".jpg")), str(Path(rgb).with_suffix(".png"))]
        rgb = next((c for c in rgb_candidates if c in file_set), None)
        if rgb is None or gt not in file_set:
            return None
        domain = "sim" if "RobbySim" in raw_path else "real"
        return {"raw": raw_path, "rgb": rgb, "gt": gt, "domain": domain, "key": raw_path}

    triplets = [t for t in (triplet(r) for r in raws) if t is not None]
    rng = np.random.default_rng(seed)
    real = [t for t in triplets if t["domain"] == "real"]
    sim = [t for t in triplets if t["domain"] == "sim"]
    rng.shuffle(real); rng.shuffle(sim)
    return real, sim


_SPLIT: Dict[str, List[Dict[str, str]]] = {}


def _split() -> Dict[str, List[Dict[str, str]]]:
    """The FROZEN train/eval split. Eval is reserved from the front of the shuffled
    lists and training from immediately after, so the two are guaranteed disjoint."""
    if not _SPLIT:
        real, sim = _all_triplets()
        _SPLIT["eval"] = real[:N_EVAL_REAL] + sim[:N_EVAL_SIM]
        _SPLIT["train"] = (
            real[N_EVAL_REAL:N_EVAL_REAL + N_TRAIN_REAL]
            + sim[N_EVAL_SIM:N_EVAL_SIM + N_TRAIN_SIM]
        )
    return _SPLIT


class HFStreamedDataset:
    """Downloads (and disk-caches) a fixed subset of triplets from the HF dataset."""

    def __init__(self, index: List[Dict[str, str]]):
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Sample:
        import cv2
        from huggingface_hub import hf_hub_download
        rec = self.index[i]

        def get(path: str) -> str:
            return hf_hub_download(DATASET_ID, path, repo_type="dataset", cache_dir=str(CACHE_DIR / "hf"))

        rgb = cv2.cvtColor(cv2.imread(get(rec["rgb"])), cv2.COLOR_BGR2RGB)
        raw = _load_depth_png(Path(get(rec["raw"])))
        gt = _load_depth_png(Path(get(rec["gt"])))
        # intrinsics: many scenes ship a per-folder intrinsics.txt; fall back to None-safe identity
        K = np.eye(3, dtype=np.float32)
        intr_path = str(Path(rec["raw"]).parents[1] / "intrinsics.txt")
        try:
            K = np.array(np.loadtxt(get(intr_path)), dtype=np.float32)
        except Exception:
            warnings.warn(f"no intrinsics for {rec['key']}; using identity (point cloud invalid)")
        return Sample(rgb=rgb, raw_depth=raw, intrinsics=K, gt_depth=gt,
                      domain=rec["domain"], key=rec["key"])


def train_dataset(use_hf: bool = True):
    """The frozen training pool. `use_hf=False` -> local ./examples (smoke test)."""
    if not use_hf:
        return LocalExamplesDataset()
    return HFStreamedDataset(_split()["train"])


def eval_dataset(use_hf: bool = True):
    """The frozen eval set — the student's exam. Disjoint from the training pool.
    `use_hf=False` -> local ./examples (smoke test)."""
    if not use_hf:
        return LocalExamplesDataset()
    return HFStreamedDataset(_split()["eval"])


# ----------------------------------------------------------------------------- #
#  Metrics                                                                       #
# ----------------------------------------------------------------------------- #

def depth_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Standard monocular-depth metrics on the valid, in-range region.

    `pred` may contain inf where the model masked out pixels; those are treated as
    invalid and excluded (they neither help nor are penalised beyond coverage).
    """
    valid = np.isfinite(pred) & np.isfinite(gt) & (gt > MIN_DEPTH) & (gt < MAX_DEPTH) & (pred > 0)
    coverage = float(valid.mean()) if valid.size else 0.0
    if valid.sum() < 100:
        return {"absrel": float("inf"), "rmse": float("inf"), "silog": float("inf"),
                "delta1": 0.0, "delta2": 0.0, "delta3": 0.0, "coverage": coverage}

    p, g = pred[valid], gt[valid]
    absrel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    err_log = np.log(p) - np.log(g)
    silog = float(np.sqrt(np.mean(err_log ** 2) - np.mean(err_log) ** 2) * 100.0)
    ratio = np.maximum(p / g, g / p)
    return {
        "absrel": absrel, "rmse": rmse, "silog": silog,
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25 ** 2)),
        "delta3": float(np.mean(ratio < 1.25 ** 3)),
        "coverage": coverage,
    }


def completion_metrics(pred: np.ndarray, gt: np.ndarray, raw: np.ndarray) -> Dict[str, float]:
    """Accuracy specifically on pixels the sensor left empty (raw==0) but gt has —
    i.e. the depth-completion job this model exists to do."""
    hole = (raw <= 0.01) & np.isfinite(gt) & (gt > MIN_DEPTH) & (gt < MAX_DEPTH) & np.isfinite(pred) & (pred > 0)
    if hole.sum() < 50:
        return {"hole_absrel": float("nan"), "hole_frac": float(((raw <= 0.01)).mean())}
    p, g = pred[hole], gt[hole]
    return {"hole_absrel": float(np.mean(np.abs(p - g) / g)),
            "hole_frac": float((raw <= 0.01).mean())}


# ----------------------------------------------------------------------------- #
#  Latency + size                                                               #
# ----------------------------------------------------------------------------- #

def count_params(model: "torch.nn.Module") -> int:
    return sum(p.numel() for p in model.parameters())


@torch.inference_mode()
def benchmark_latency(model: "torch.nn.Module", hw: Tuple[int, int] = BENCH_INPUT_HW,
                      resolution_level: int = BENCH_RESOLUTION_LEVEL) -> float:
    """Median forward latency in ms for one frame at a given resolution_level.

    Measures the real `infer()` path (the token-grid pre-processing + encoder, which
    dominate). Note `hw` barely affects latency — the model interpolates to a token
    grid set by resolution_level and aspect ratio — but the aspect ratio does matter,
    so keep it representative. Portable secondary signals (params/FLOPs) are reported
    alongside by evaluate(); latency is the primary metric."""
    dev = device()
    H, W = hw
    img = torch.rand(1, 3, H, W, device=dev, dtype=model.dtype if hasattr(model, "dtype") else torch.float32)
    depth = torch.rand(1, H, W, device=dev) * 3.0 + 0.5
    K = torch.tensor([[0.9, 0, 0.5], [0, 1.2, 0.5], [0, 0, 1]], device=dev, dtype=torch.float32)[None]

    def one():
        model.infer(img, depth_in=depth, intrinsics=K, resolution_level=resolution_level)

    for _ in range(BENCH_WARMUP):
        one()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(BENCH_ITERS):
        t0 = time.perf_counter()
        one()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


# ----------------------------------------------------------------------------- #
#  Objective  —  the single scalar the agent minimises                          #
# ----------------------------------------------------------------------------- #

@dataclass
class Baseline:
    """Teacher reference numbers, measured once, that define the tolerance band."""
    latency_ms: float
    absrel_real: float
    absrel_sim: float
    delta1_real: float
    delta1_sim: float
    params: int


def objective(latency_ms: float, acc: Dict[str, float], base: Baseline) -> float:
    """Collapse (latency, accuracy) into ONE number. Lower is better.

    score = latency_ms * (1 + PENALTY * <accuracy-tolerance violations>)

    The student can only lower the score by getting faster while staying within the
    accuracy band around the teacher. `acc` carries weighted absrel/delta1 over the
    real+sim eval domains.
    """
    absrel = acc["absrel_weighted"]
    delta1 = acc["delta1_weighted"]
    base_absrel = ACC_WEIGHT_REAL * base.absrel_real + ACC_WEIGHT_SIM * base.absrel_sim
    base_delta1 = ACC_WEIGHT_REAL * base.delta1_real + ACC_WEIGHT_SIM * base.delta1_sim

    absrel_violation = max(0.0, absrel / max(base_absrel, 1e-6) - (1.0 + ACC_TOLERANCE))
    delta1_violation = max(0.0, base_delta1 * (1.0 - DELTA_TOLERANCE) - delta1)

    penalty = PENALTY * (absrel_violation + delta1_violation)
    return float(latency_ms * (1.0 + penalty))


# ----------------------------------------------------------------------------- #
#  Evaluation                                                                    #
# ----------------------------------------------------------------------------- #

@torch.inference_mode()
def _predict(model: "torch.nn.Module", s: Sample,
             resolution_level: int = BENCH_RESOLUTION_LEVEL) -> np.ndarray:
    """Run a model's infer() on one Sample at `resolution_level`; return refined depth
    as (H, W) float32. The accuracy path uses the SAME level the latency is timed at."""
    dev = device()
    H, W = s.rgb.shape[:2]
    img = torch.tensor(s.rgb / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)[None]
    depth = torch.tensor(s.raw_depth, dtype=torch.float32, device=dev)[None]
    K = s.intrinsics.copy().astype(np.float32)
    K[0] /= W
    K[1] /= H
    K = torch.tensor(K, device=dev)[None]
    out = model.infer(img, depth_in=depth, intrinsics=K, resolution_level=resolution_level)
    return out["depth"].squeeze().float().cpu().numpy()


def evaluate(model: "torch.nn.Module", eval_set, base: Optional[Baseline] = None,
             teacher: Optional["torch.nn.Module"] = None,
             resolution_level: int = BENCH_RESOLUTION_LEVEL) -> Dict[str, float]:
    """Score a model on the frozen eval set at a given operating point. Returns latency
    + per-domain accuracy (both measured at `resolution_level`) and, if a baseline is
    given, the single objective `score`.

    If a sample has no gt_depth (e.g. the local smoke set), the teacher's BEST-quality
    prediction (level 9) is used as the reference so the harness still produces a
    fidelity signal — the reference is always the best teacher, independent of the
    level the model under test runs at."""
    per_domain: Dict[str, List[Dict[str, float]]] = {"real": [], "sim": []}
    for i in range(len(eval_set)):
        s = eval_set[i]
        pred = _predict(model, s, resolution_level=resolution_level)
        gt = s.gt_depth
        if gt is None:
            if teacher is None:
                teacher = load_teacher()
            gt = _predict(teacher, s, resolution_level=9)  # reference = best teacher
            gt = np.where(np.isfinite(gt), gt, 0.0)
        m = depth_metrics(pred, gt)
        m.update(completion_metrics(pred, gt, s.raw_depth))
        per_domain.setdefault(s.domain, []).append(m)

    def agg(key: str) -> Dict[str, float]:
        rows = per_domain.get(key, [])
        if not rows:
            return {}
        return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}

    real, sim = agg("real"), agg("sim")
    # Weighted accuracy across domains (fall back to whichever exists).
    wr, ws = (ACC_WEIGHT_REAL, ACC_WEIGHT_SIM) if (real and sim) else (
        (1.0, 0.0) if real else (0.0, 1.0))
    absrel_w = wr * real.get("absrel", np.nan) + ws * sim.get("absrel", np.nan)
    delta1_w = wr * real.get("delta1", np.nan) + ws * sim.get("delta1", np.nan)

    latency = benchmark_latency(model, resolution_level=resolution_level)
    result: Dict[str, float] = {
        "latency_ms": latency,
        "params": count_params(model),
        "absrel_weighted": absrel_w,
        "delta1_weighted": delta1_w,
        "absrel_real": real.get("absrel", float("nan")),
        "absrel_sim": sim.get("absrel", float("nan")),
        "delta1_real": real.get("delta1", float("nan")),
        "delta1_sim": sim.get("delta1", float("nan")),
        "hole_absrel_real": real.get("hole_absrel", float("nan")),
    }
    if base is not None:
        result["score"] = objective(latency, result, base)
        result["speedup_vs_teacher"] = base.latency_ms / latency if latency > 0 else 0.0
    return result


def measure_teacher_baseline(eval_set, teacher: Optional["torch.nn.Module"] = None) -> Baseline:
    """Measure the teacher's own accuracy + latency to anchor the tolerance band."""
    teacher = teacher or load_teacher()
    res = evaluate(teacher, eval_set, base=None, teacher=teacher)
    base = Baseline(
        latency_ms=res["latency_ms"],
        absrel_real=res["absrel_real"], absrel_sim=res["absrel_sim"],
        delta1_real=res["delta1_real"], delta1_sim=res["delta1_sim"],
        params=int(res["params"]),
    )
    (RUNS_DIR / "teacher_baseline.json").write_text(json.dumps(asdict(base), indent=2))
    return base


def teacher_resolution_report(eval_set, teacher: Optional["torch.nn.Module"] = None,
                              levels: List[int] = RESOLUTION_SWEEP) -> List[Dict[str, float]]:
    """Measure the teacher's latency + accuracy across resolution_levels — its
    latency-vs-accuracy Pareto curve. This is pure telemetry (it does NOT change the
    objective): it shows how much speed the teacher can buy for free by lowering the
    level, so the student's speedup can be read against a fairly-tuned teacher rather
    than the slowest (level-9) point. Saved to runs/teacher_resolution_sweep.json."""
    teacher = teacher or load_teacher()
    rows: List[Dict[str, float]] = []
    for lvl in levels:
        r = evaluate(teacher, eval_set, base=None, teacher=teacher, resolution_level=lvl)
        rows.append({
            "resolution_level": lvl,
            "latency_ms": round(r["latency_ms"], 3),
            "absrel_weighted": round(r["absrel_weighted"], 5),
            "delta1_weighted": round(r["delta1_weighted"], 5),
        })
    (RUNS_DIR / "teacher_resolution_sweep.json").write_text(json.dumps(rows, indent=2))
    return rows


def load_baseline() -> Optional[Baseline]:
    p = RUNS_DIR / "teacher_baseline.json"
    if p.exists():
        return Baseline(**json.loads(p.read_text()))
    return None


# ----------------------------------------------------------------------------- #
#  Self-test  —  runs the download-free logic so the harness can be sanity       #
#  checked without a GPU or the teacher/dataset. `python distill/prepare.py`     #
# ----------------------------------------------------------------------------- #

def _selftest() -> None:
    print("[selftest] metrics ...")
    gt = np.random.rand(64, 64).astype(np.float32) * 5 + 0.5
    pred = gt * 1.02 + np.random.randn(64, 64).astype(np.float32) * 0.01
    m = depth_metrics(pred, gt)
    assert 0 <= m["delta1"] <= 1 and m["absrel"] < 0.1, m
    print("           absrel=%.4f delta1=%.3f silog=%.2f" % (m["absrel"], m["delta1"], m["silog"]))

    print("[selftest] student-config derivation (mock teacher cfg) ...")
    mock = {
        "encoder": {"backbone": "dinov2_vitl14", "intermediate_layers": [4, 11, 17, 23],
                    "dim_out": 1024, "strict": True},
        "neck": {"dim_in": [1026, 2, 2, 2, 2], "dim_res_blocks": [256, 128, 64, 32, 16],
                 "dim_out": [None, None, None, None, 64], "resamplers": "pixel_shuffle"},
        "num_tokens_range": [1200, 3600],
    }
    for bb, w, depth in [("dinov2_vits14", 384, 12), ("dinov2_vitb14", 768, 12),
                         ("dinov2_vitl14", 1024, 24)]:
        cfg = derive_student_config(backbone=bb, teacher_cfg=mock)
        assert cfg["encoder"]["dim_out"] == w
        assert cfg["neck"]["dim_in"][0] == w + 2, cfg["neck"]["dim_in"]
        assert all(0 <= i < depth for i in cfg["encoder"]["intermediate_layers"])
        assert cfg["encoder"]["strict"] is False
        print("           %-16s dim_out=%4d neck_in0=%4d layers=%s"
              % (bb, cfg["encoder"]["dim_out"], cfg["neck"]["dim_in"][0],
                 cfg["encoder"]["intermediate_layers"]))

    print("[selftest] objective monotonic in latency & penalises accuracy loss ...")
    base = Baseline(latency_ms=100.0, absrel_real=0.05, absrel_sim=0.04,
                    delta1_real=0.97, delta1_sim=0.98, params=300_000_000)
    good = {"absrel_weighted": 0.047, "delta1_weighted": 0.972}
    bad = {"absrel_weighted": 0.20, "delta1_weighted": 0.80}
    assert objective(40, good, base) < objective(80, good, base)          # faster is better
    assert objective(40, bad, base) > objective(40, good, base)           # accuracy loss hurts
    assert objective(40, good, base) < base.latency_ms                    # a real speedup wins
    print("           score(fast,ok)=%.1f  score(fast,bad)=%.1f"
          % (objective(40, good, base), objective(40, bad, base)))
    print("[selftest] OK")


if __name__ == "__main__":
    _selftest()
