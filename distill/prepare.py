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
import csv
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
RUNS_DIR = REPO_ROOT / "distill" / "runs"           # per-experiment ckpts + results (local)
BEST_DIR = REPO_ROOT / "distill" / "best"           # committed champion (travels via git/LFS)
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
# Sim by default (RobbySimVal has perfect GT and streams cheaply). Real shards are
# wired but default to 0 — their modality-grouped layout has a ~3 GB/camera streaming
# floor, so enable real deliberately (set N_*_REAL > 0) once you accept that cost.
N_TRAIN_SIM = 1200
N_TRAIN_REAL = 0
N_EVAL_SIM = 100
N_EVAL_REAL = 0

# ---- Compute budget per experiment (FROZEN) ----------------------------------
# Equal compute per run => comparable scores (autoresearch's fixed-budget premise).
# The agent optimises what to do WITHIN this budget, not the budget itself.
TRAIN_MINUTES = 20.0          # proxy-run wall-clock; humans promote winners to longer runs
MAX_STEPS = 100_000           # hard cap

# ---- Teacher-target cache ----------------------------------------------------
# The ViT-L teacher is expensive; caching its per-sample outputs turns the inner loop
# from "run the teacher every step" into "read a tensor". Caching is only sound if each
# sample is always processed at the same resolution, so TRAINING uses a fixed canvas
# (eval still runs at native resolution). Pick a canvas near your data's aspect ratio.
TRAIN_HW = (480, 640)                 # fixed training resolution (H, W)
TARGET_CACHE_DIR = CACHE_DIR / "teacher_targets"
CACHE_TEACHER_FEATURES = True         # False -> cache depth only (less disk, less speedup)
# Roughly ~3 MB/sample (fp16) at TRAIN_HW + FEATURE_TOKENS=1200 with features cached.

# ---- Champion weight commit guard --------------------------------------------
MAX_COMMIT_WEIGHTS_MB = 95            # keep committed champion under GitHub's 100MB limit

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


# --- HF-streamed subset (WebDataset-style .tar.zst shards) ------------------- #
#
# The dataset ships as huge compressed tar shards (47-320 GB each), NOT a browsable
# file tree. So we STREAM a shard, decompress on the fly, and pull only the first N
# complete (rgb, raw, gt) triplets off the front — downloading a few hundred MB, not
# the whole shard. Extracted samples are disk-cached, so re-runs are free.
#
# Shard layouts differ by family (verified by streaming the real tree):
#   RobbySimVal      <stem>_rgb.left.jpg | _rawdepth.left.png | _depth_left.png   (interleaved)
#   RobbySim *_view  <stem>_left.jpg     | _rmd2c.png (raw)    | _depth.png        (interleaved)
#   RobbyReal        <cam>/{color,rawdepth,gtdepth}/<frame>.<ext>                 (grouped by modality)
# Needs `requests` + `zstandard` installed.

HF_RESOLVE = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/main"
SHARD_CACHE = CACHE_DIR / "shards"
_MAX_SCAN_MEMBERS = 400_000   # safety cap on how deep we stream into a shard


def _parse_simval(name: str):
    if name.endswith("_rgb.left.jpg"):      return name[:-13], "rgb"
    if name.endswith("_rawdepth.left.png"): return name[:-18], "raw"
    if name.endswith("_depth_left.png"):    return name[:-15], "gt"
    return None


def _parse_sim(name: str):
    if name.endswith("_left.jpg"):  return name[:-9], "rgb"
    if name.endswith("_rmd2c.png"): return name[:-10], "raw"
    if name.endswith("_depth.png"): return name[:-10], "gt"
    return None


def _parse_real(name: str):
    parts = name.split("/")
    if len(parts) < 2:
        return None
    modality, frame = parts[-2], parts[-1].rsplit(".", 1)[0]
    stem = "/".join(parts[:-2]) + "/" + frame
    return {"color": (stem, "rgb"), "rawdepth": (stem, "raw"),
            "gtdepth": (stem, "gt")}.get(modality)


# shard file -> (member-name parser, domain)
SHARDS: Dict[str, Tuple[Any, str]] = {
    "RobbySimVal_batch_0001.tar.zst": (_parse_simval, "sim"),
    "RobbySim_object_view_batch_0001.tar.zst": (_parse_sim, "sim"),
    "RobbyReal_batch_0001.tar.zst": (_parse_real, "real"),
    "RobbyReal_batch_0002.tar.zst": (_parse_real, "real"),
}

# Which shards feed which split. Train and eval use DIFFERENT shards -> guaranteed
# disjoint. Real shards are wired but default to 0 samples (N_*_REAL): their
# modality-grouped layout means ~3 GB must be streamed per camera before the first
# triplet completes, so enable real deliberately once you accept that cost.
EVAL_SIM_SHARD, EVAL_REAL_SHARD = "RobbySimVal_batch_0001.tar.zst", "RobbyReal_batch_0001.tar.zst"
TRAIN_SIM_SHARD, TRAIN_REAL_SHARD = "RobbySim_object_view_batch_0001.tar.zst", "RobbyReal_batch_0002.tar.zst"


def _cached_sample_dirs(dest: Path) -> List[Path]:
    return sorted(p for p in dest.glob("*")
                  if p.is_dir() and p.name != "_staging" and (p / "rgb").exists())


def _stream_shard_samples(shard: str, n: int, cache_subdir: str) -> List[Path]:
    """Stream `shard`, extract the first `n` complete (rgb, raw, gt) triplets to a disk
    cache, and return their directories. Disk-buffered via a staging dir, so the
    modality-grouped real shards don't blow up RAM. Cached, so re-runs are free."""
    if n <= 0:
        return []
    import requests, zstandard, tarfile, shutil

    dest = SHARD_CACHE / cache_subdir
    dest.mkdir(parents=True, exist_ok=True)
    ready = _cached_sample_dirs(dest)
    if len(ready) >= n:
        return ready[:n]
    if shard not in SHARDS:
        raise ValueError(f"unknown shard {shard!r}")
    parser, domain = SHARDS[shard]

    staging = dest / "_staging"
    staging.mkdir(exist_ok=True)
    seen: Dict[str, set] = {}
    complete = list(ready)
    print(f"[data] streaming {shard} for {n - len(complete)} more '{domain}' samples -> {dest}")

    with requests.get(f"{HF_RESOLVE}/{shard}", stream=True, timeout=(30, 300)) as r:
        r.raise_for_status()
        reader = zstandard.ZstdDecompressor().stream_reader(r.raw)
        tar = tarfile.open(fileobj=reader, mode="r|")
        scanned = 0
        for m in tar:
            if not m.isfile():
                continue
            scanned += 1
            if scanned > _MAX_SCAN_MEMBERS:
                warnings.warn(f"{shard}: scan cap reached with {len(complete)}/{n} samples")
                break
            pk = parser(m.name)
            if pk is None:
                continue
            stem, kind = pk
            sid = _safe_key(stem)
            if (dest / sid).is_dir():
                continue
            (staging / f"{sid}.{kind}").write_bytes(tar.extractfile(m).read())
            seen.setdefault(sid, set()).add(kind)
            if {"rgb", "raw", "gt"} <= seen[sid]:
                sdir = dest / sid
                sdir.mkdir()
                for k in ("rgb", "raw", "gt"):
                    shutil.move(str(staging / f"{sid}.{k}"), str(sdir / k))
                (sdir / "domain").write_text(domain)
                complete.append(sdir)
                del seen[sid]
                if len(complete) >= n:
                    break
    print(f"[data] {shard}: {len(complete)} samples ready")
    return complete[:n]


class ShardDataset:
    """Samples extracted from streamed shards (raw encoded bytes on disk, decoded lazily).
    Intrinsics are identity — not shipped per-frame, and they don't affect depth metrics
    or the distillation target (only the point cloud)."""

    def __init__(self, sample_dirs: List[Path]):
        self.dirs = sample_dirs

    def __len__(self) -> int:
        return len(self.dirs)

    def __getitem__(self, i: int) -> Sample:
        import cv2
        d = self.dirs[i]

        def dec(name, flags):
            return cv2.imdecode(np.frombuffer((d / name).read_bytes(), np.uint8), flags)

        rgb = cv2.cvtColor(dec("rgb", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        raw = np.nan_to_num(dec("raw", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0,
                            nan=0.0, posinf=0.0, neginf=0.0)
        gt = np.nan_to_num(dec("gt", cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0,
                           nan=0.0, posinf=0.0, neginf=0.0)
        return Sample(rgb=rgb, raw_depth=raw, intrinsics=np.eye(3, dtype=np.float32),
                      gt_depth=gt, domain=(d / "domain").read_text().strip(), key=d.name)


def train_dataset(use_hf: bool = True):
    """The frozen training pool. `use_hf=False` -> local ./examples (smoke test)."""
    if not use_hf:
        return LocalExamplesDataset()
    dirs = (_stream_shard_samples(TRAIN_SIM_SHARD, N_TRAIN_SIM, "train_sim")
            + _stream_shard_samples(TRAIN_REAL_SHARD, N_TRAIN_REAL, "train_real"))
    return ShardDataset(dirs)


def eval_dataset(use_hf: bool = True):
    """The frozen eval set — the student's exam. Streamed from DIFFERENT shards than the
    training pool, so the two are disjoint. `use_hf=False` -> local ./examples."""
    if not use_hf:
        return LocalExamplesDataset()
    dirs = (_stream_shard_samples(EVAL_SIM_SHARD, N_EVAL_SIM, "eval_sim")
            + _stream_shard_samples(EVAL_REAL_SHARD, N_EVAL_REAL, "eval_real"))
    return ShardDataset(dirs)


# ----------------------------------------------------------------------------- #
#  Teacher-target cache                                                          #
# ----------------------------------------------------------------------------- #
# distill_step needs, per sample: the teacher encoder features (at FEATURE_TOKENS) and
# the teacher's best-quality refined depth. Recomputing those with the ViT-L teacher
# every step would eat the whole proxy budget, so we cache them to disk, keyed by
# (FEATURE_TOKENS, sample key). Populated lazily on first touch and reused across every
# step AND every later experiment. Because the cache is per-sample, training must
# process each sample at a fixed resolution (TRAIN_HW) — otherwise the targets wouldn't
# be a stable function of the sample.

def _safe_key(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in key)[:180]


def _canonical_batch(samples: List[Sample], dev) -> Tuple[Any, Any, Any]:
    """Resize a batch of Samples to the fixed TRAIN_HW canvas and stack into tensors.
    Intrinsics are normalised by each sample's NATIVE size (normalised intrinsics are
    resolution-independent), so they stay valid after the resize."""
    import cv2
    H, W = TRAIN_HW
    imgs, raws, Ks = [], [], []
    for s in samples:
        H0, W0 = s.rgb.shape[:2]
        rgb = cv2.resize(s.rgb, (W, H))
        raw = cv2.resize(s.raw_depth, (W, H), interpolation=cv2.INTER_NEAREST)
        imgs.append(torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1))
        raws.append(torch.tensor(raw, dtype=torch.float32))
        K = s.intrinsics.copy().astype(np.float32); K[0] /= W0; K[1] /= H0
        Ks.append(torch.tensor(K))
    return (torch.stack(imgs).to(dev), torch.stack(raws).to(dev), torch.stack(Ks).to(dev))


@torch.no_grad()
def teacher_targets(samples: List[Sample], feature_tokens: int, teacher, dev):
    """Return (imgs, raws, Ks, t_feat, t_depth) for a batch at TRAIN_HW, using a
    per-sample disk cache so the teacher runs once per (sample, feature_tokens) instead
    of every step. imgs/raws are the canonical inputs the student should train on too.

    t_feat: teacher encoder features at `feature_tokens`.  t_depth: teacher best-quality
    refined depth (infer() default level). Cached fp16; recomputed only on a miss."""
    imgs, raws, Ks = _canonical_batch(samples, dev)
    cdir = TARGET_CACHE_DIR / f"tok{feature_tokens}"
    cdir.mkdir(parents=True, exist_ok=True)

    n = len(samples)
    feats: List[Any] = [None] * n
    depths: List[Any] = [None] * n
    miss = []
    for i, s in enumerate(samples):
        f = cdir / (_safe_key(s.key) + ".pt")
        if f.exists():
            d = torch.load(f, map_location=dev)
            depths[i] = d["depth"].float()
            if CACHE_TEACHER_FEATURES and d.get("feat") is not None:
                feats[i] = d["feat"].float()
        else:
            miss.append(i)

    if miss:
        mi, mr, mk = imgs[miss], raws[miss], Ks[miss]
        tf, _ = teacher.infer_feat(mi, depth_in=mr, num_tokens=feature_tokens)
        td = teacher.infer(mi, depth_in=mr, intrinsics=mk, apply_mask=False)["depth"]
        for j, i in enumerate(miss):
            depths[i] = td[j]
            feats[i] = tf[j]     # use the freshly computed feature (avoid recompute below)
            torch.save({"depth": td[j].half().cpu(),
                        "feat": tf[j].half().cpu() if CACHE_TEACHER_FEATURES else None},
                       cdir / (_safe_key(samples[i].key) + ".pt"))

    # Any features still missing (hits whose cached file predates feature caching) are
    # recomputed batched — depth still came from the cache.
    need = [i for i in range(n) if feats[i] is None]
    if need:
        tf, _ = teacher.infer_feat(imgs[need], depth_in=raws[need], num_tokens=feature_tokens)
        for j, i in enumerate(need):
            feats[i] = tf[j]

    return imgs, raws, Ks, torch.stack(feats), torch.stack(depths)


def precompute_teacher_targets(feature_tokens: int, use_hf: bool = True,
                               batch_size: int = 8, teacher=None) -> int:
    """Warm the cache over the whole training pool up front (optional — the cache also
    fills lazily during training). Returns the number of samples processed."""
    teacher = teacher or load_teacher()
    ds = train_dataset(use_hf=use_hf)
    dev = device()
    n = len(ds)
    done = 0
    for start in range(0, n, batch_size):
        batch = [ds[i] for i in range(start, min(start + batch_size, n))]
        teacher_targets(batch, feature_tokens, teacher, dev)
        done += len(batch)
        if done % (batch_size * 10) == 0 or done == n:
            print(f"[cache] teacher targets {done}/{n}")
    return done


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
    # BATCH SIZE 1, ALWAYS: a real camera delivers one frame at a time, so per-frame
    # latency is what deployment sees. Never batch the latency measurement.
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
    # Weighted accuracy across whichever domains are present, renormalising the weights
    # over them (so a real-only or sim-only eval set doesn't get nan-poisoned by the
    # absent domain, and 0*nan can never leak in).
    parts = ([(ACC_WEIGHT_REAL, real)] if real else []) + ([(ACC_WEIGHT_SIM, sim)] if sim else [])
    wsum = sum(w for w, _ in parts) or 1.0
    absrel_w = sum(w * d["absrel"] for w, d in parts) / wsum
    delta1_w = sum(w * d["delta1"] for w, d in parts) / wsum

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


def compare_across_levels(models: Dict[str, "torch.nn.Module"], eval_set,
                          levels: List[int] = RESOLUTION_SWEEP) -> List[Dict[str, float]]:
    """Evaluate several named models (e.g. {'teacher', 'best', 'current'}) at each
    resolution_level and report per-level latency, accuracy, and speedup vs the
    'teacher' model AT THAT SAME LEVEL. Returns flat rows (one per level×model) and
    saves runs/level_comparison.json.

    This is the apples-to-apples view: it shows whether the student actually beats the
    teacher's speed/accuracy trade-off at every operating point, not just at level 9,
    and how the current student stacks up against the best one so far."""
    teacher = models.get("teacher")
    flat: List[Dict[str, float]] = []
    for lvl in levels:
        measured = []
        for name, m in models.items():
            if m is None:
                continue
            r = evaluate(m, eval_set, base=None, teacher=teacher, resolution_level=lvl)
            measured.append((name, r))
        t_lat = next((r["latency_ms"] for n, r in measured if n == "teacher"), None)
        for name, r in measured:
            lat = r["latency_ms"]
            flat.append({
                "resolution_level": lvl,
                "model": name,
                "latency_ms": round(lat, 3),
                "speedup_vs_teacher": round(t_lat / lat, 3) if (t_lat and lat) else None,
                "absrel_weighted": round(r["absrel_weighted"], 5),
                "delta1_weighted": round(r["delta1_weighted"], 5),
            })
    (RUNS_DIR / "level_comparison.json").write_text(json.dumps(flat, indent=2))
    return flat


def load_baseline() -> Optional[Baseline]:
    p = RUNS_DIR / "teacher_baseline.json"
    if p.exists():
        return Baseline(**json.loads(p.read_text()))
    return None


def best_student_ckpt() -> Optional[Path]:
    """Path to the best student checkpoint to use as warm-start / comparison parent.

    Prefers the best LOCAL run (min `score` in results.csv) because those scores were
    measured on THIS machine and are directly comparable. Falls back to the committed
    champion in distill/best/ (which seeds a fresh clone and lets the team's best
    student travel across machines). Ranking/keep-revert stays machine-local; the
    committed champion is a portable warm-start + comparison artifact, not a
    cross-machine score to rank against (latency is machine-specific — see best.json)."""
    path = RUNS_DIR / "results.csv"
    if path.exists():
        best_row, best = None, float("inf")
        with path.open() as f:
            for r in csv.DictReader(f):
                try:
                    s = float(r["score"])
                except (KeyError, ValueError):
                    continue
                if s < best:
                    best, best_row = s, r
        if best_row is not None:
            ckpt = RUNS_DIR / best_row["run_id"] / "student.pt"
            if ckpt.exists():
                return ckpt
    committed = BEST_DIR / "student.pt"     # portable champion (fresh clone / cross-machine)
    return committed if committed.exists() else None


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
