"""Branch A: shared 3D contrastive encoder. STUB — implement backbone + head.
SwinUNETR is a SEGMENTATION net: use its ENCODER and pool features (CLAUDE.md §9)."""
from __future__ import annotations
import torch
from torch import nn
from config import CFG


class Encoder(nn.Module):
    """Shared 3D backbone (+ optional modality token) -> projection head -> L2-normed embedding.
    Both ceT1 and T2 map into ONE space so they can be compared."""

    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg
        # TODO(claude-code): build backbone per cfg.backbone:
        #   - "swinunetr_encoder": monai.networks.nets.SwinUNETR encoder, global-pool the bottleneck
        #   - "resnet3d": monai.networks.nets.resnet (e.g. resnet18/3D)
        #   - "brainmvp": load pretrained brain-MRI encoder
        # then a projection MLP -> cfg.embedding_dim, F.normalize at the end.
        raise NotImplementedError("TODO(claude-code): backbone + projection head")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,1,R,R,R]
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, volume: torch.Tensor) -> torch.Tensor:
        """Single volume [1,R,R,R] -> L2-normed embedding [D]."""
        raise NotImplementedError
