"""Branch B: MIND structural descriptor (Heinrich et al. 2012, MIND-SSC variant).
Training-free, modality-invariant. Anchor for L1 & L3; ~useless on L2.
STUB — implement the descriptor + cross-modal distance."""
from __future__ import annotations
import torch
from typing import Sequence


def mind_descriptor(volume: torch.Tensor) -> torch.Tensor:
    """volume [1,D,H,W] -> MIND feature field [C,D,H,W] (per-voxel self-similarity
    over a small neighbourhood; C = number of offsets). Modality-invariant by design."""
    raise NotImplementedError("TODO(claude-code): MIND/MIND-SSC self-similarity descriptor")


def mind_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Distance between two MIND fields (mean SSD over voxels). Lower = more similar."""
    raise NotImplementedError("TODO(claude-code): masked SSD between descriptor fields")


def mind_rank(query_vol: torch.Tensor, gallery: Sequence[tuple[str, torch.Tensor]]) -> list[str]:
    """Rank gallery (id, volume) by MIND similarity to the query. Best->worst.
    On L2 the result is near-random (expected) — fusion down-weights B there."""
    raise NotImplementedError("TODO(claude-code): compute mind_distance to each, argsort ascending")
