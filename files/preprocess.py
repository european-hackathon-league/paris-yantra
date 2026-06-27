"""Load + normalize + resample + mask 3D NIfTI volumes, with a disk cache.
STUB — implement per docstrings and config.py. See CLAUDE.md §5."""
from __future__ import annotations
from pathlib import Path
import torch
from config import CFG


def preprocess_volume(path: str | Path, cfg=CFG) -> torch.Tensor:
    """path(.nii.gz) -> tensor [1, R, R, R] (R=cfg.resolution).
    Steps: load (RAS), resample to cfg.spacing then to cfg.resolution isotropic,
    clip to cfg.clip_pct percentiles, z-score, brain-mask (SynthStrip if cfg.skull_strip).
    MUST handle variable input shapes (never assume a fixed shape)."""
    raise NotImplementedError("TODO(claude-code): MONAI LoadImage/Orientation/Spacing/Resize/ScaleIntensity + mask")


def build_cache(manifest: list[dict], cfg=CFG):
    """manifest: [{'id': str, 'path': Path}]. Build a MONAI PersistentDataset
    so preprocessed tensors are cached once and reused across runs."""
    raise NotImplementedError("TODO(claude-code): MONAI PersistentDataset over manifest")
