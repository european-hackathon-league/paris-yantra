"""Train Branch A with InfoNCE + domain-randomization augmentation. STUB."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from config import CFG


def info_nce_loss(q_emb: torch.Tensor, t_emb: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE with in-batch negatives. q_emb/t_emb: [B,D] L2-normed.
    Positives are on the diagonal; everything else in the batch is a negative
    (bigger batch = more negatives = stronger signal)."""
    logits = (q_emb @ t_emb.t()) / temperature
    labels = torch.arange(len(q_emb), device=q_emb.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def train(cfg=CFG):
    """Loop: load 350 d1 pairs -> apply augment.train_transforms INDEPENDENTLY to
    query and target -> encode both -> info_nce_loss -> AdamW + AMP. Checkpoint each
    epoch. After training, report oracle MRR per level. TODO(claude-code): full loop."""
    raise NotImplementedError("TODO(claude-code): dataloader + AMP loop + checkpointing")
