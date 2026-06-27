"""⭐ THE KEYSTONE. Build offline L1/L2/L3 evaluation from the 350 labelled d1 pairs
by SYNTHESIZING fake-L2 and fake-L3, so fusion weights and the veto are tuned on
real numbers instead of guesses. See CLAUDE.md §6. Mostly orchestration — fill TODOs."""
from __future__ import annotations
import random
from typing import Callable, Sequence
import torch

from config import CFG
from metrics import mrr
import augment

# A rank_fn ranks a gallery (list of volumes) for a query volume, returning gallery
# indices best->worst. Works for MIND (mind_rank) and the encoder (cosine) alike.
RankFn = Callable[[torch.Tensor, Sequence[torch.Tensor]], list[int]]


def make_holdout(pairs: list[dict], n: int = CFG.holdout_pairs, seed: int = CFG.seed):
    """pairs: the 350 labelled d1 pairs (each with preprocessed query+target tensors).
    Return (train_pairs, holdout_pairs). IMPORTANT: train the encoder on train_pairs
    and tune w/thresholds on holdout_pairs — no double-dipping."""
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    hold = {i for i in idx[:n]}
    return [p for i, p in enumerate(pairs) if i not in hold], [pairs[i] for i in idx[:n]]


def _apply(transform, vol: torch.Tensor) -> torch.Tensor:
    """Apply a MONAI dict-transform to one volume tensor [1,R,R,R]."""
    return transform({"image": vol})["image"]


def build_level(holdout: list[dict], level: str, cfg=CFG):
    """Return (query_vols, target_vols) for a level. Index i is the matching pair.
    l1: identity. l2: independent l2_transforms on q and target. l3: l3_transforms."""
    if level == "l1":
        return [p["query"] for p in holdout], [p["target"] for p in holdout]
    tf = augment.l2_transforms(cfg) if level == "l2" else augment.l3_transforms(cfg)
    # independent draws for query and target (they must NOT share the transform)
    q = [_apply(tf, p["query"]) for p in holdout]
    g = [_apply(tf, p["target"]) for p in holdout]
    return q, g


def evaluate(rank_fn: RankFn, holdout: list[dict], cfg=CFG) -> dict:
    """Offline MRR per level + mean. THE number that gates every change/submit."""
    out = {}
    for level in ("l1", "l2", "l3"):
        q_vols, g_vols = build_level(holdout, level, cfg)
        rankings, truth = {}, {}
        for i, qv in enumerate(q_vols):
            order = rank_fn(qv, g_vols)      # gallery indices best->worst
            rankings[i] = order
            truth[i] = i                     # matching target shares the index
        out[level] = mrr(rankings, truth)
    out["mean"] = sum(out[l] for l in ("l1", "l2", "l3")) / 3.0
    return out


def tune_fusion_weights(rank_fn_a: RankFn, rank_fn_b: RankFn, holdout: list[dict],
                        cfg=CFG, grid=(0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0)):
    """For each level, sweep w (weight on Branch A) and keep the w with best MRR.
    Expected shape: L2 -> high w (MIND useless), L1/L3 -> lower w (MIND anchors).
    TODO(claude-code): for each level build A & B rankings, fuse via fuse.rank_fusion,
    score with metrics.mrr, return best w per level."""
    raise NotImplementedError("TODO(claude-code): per-level w sweep using fuse.rank_fusion + mrr")
