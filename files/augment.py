"""Domain-randomization transforms — the engine. Synthesize L2/L3 conditions from L1.
Concrete MONAI skeleton; tune magnitudes in config.py against the real example pairs.
Applied INDEPENDENTLY to each scan in a pair (CLAUDE.md §7)."""
from __future__ import annotations
from config import CFG

# NOTE: import MONAI transforms lazily inside functions so this file imports on CPU.


def l2_transforms(cfg=CFG):
    """Independent rigid (rot ~15-25°, translation) + non-linear elastic. Destroys
    voxel correspondence on purpose -> teaches L2 invariance."""
    from monai.transforms import Compose, RandAffined, Rand3DElasticd
    a = cfg.l2_aug
    rot = a["rotate_deg"] * 3.14159 / 180.0
    return Compose([
        RandAffined(keys="image", prob=a["prob"],
                    rotate_range=(rot, rot, rot),
                    translate_range=(a["translate_vox"],) * 3, padding_mode="zeros"),
        Rand3DElasticd(keys="image", prob=a["prob"],
                       sigma_range=a["elastic_sigma"], magnitude_range=a["elastic_magnitude"]),
    ])


def l3_transforms(cfg=CFG):
    """Bias field + gamma/contrast (scanner/hospital) + coarse dropout (surgery).
    KEEP rough alignment (only small rotation) — L3 is co-located."""
    from monai.transforms import (Compose, RandBiasFieldd, RandAdjustContrastd,
                                   RandCoarseDropoutd, RandAffined)
    a = cfg.l3_aug
    rot = a["small_rotate_deg"] * 3.14159 / 180.0
    return Compose([
        RandBiasFieldd(keys="image", prob=a["prob"], coeff_range=(0.0, a["bias_field_coeff"])),
        RandAdjustContrastd(keys="image", prob=a["prob"], gamma=a["gamma"]),
        RandCoarseDropoutd(keys="image", prob=a["prob"], holes=a["dropout_holes"],
                           spatial_size=(a["dropout_size"],) * 3, fill_value=0.0),
        RandAffined(keys="image", prob=a["prob"], rotate_range=(rot, rot, rot)),
    ])


def train_transforms(cfg=CFG):
    """Stack used during contrastive training: mix L2 + L3 distortions so the model
    sees both. TODO(claude-code): combine l2+l3, add intensity randomization, flips."""
    from monai.transforms import Compose
    return Compose([*l2_transforms(cfg).transforms, *l3_transforms(cfg).transforms])
