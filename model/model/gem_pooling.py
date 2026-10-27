"""
gem_pooling.py — Generalized Mean Pooling for FairDermNet
---------------------------------------------------------
GeM pooling learns an optimal pooling exponent p during training.
When p=1: average pooling. As p→∞: max pooling.
For dermoscopic features, p≈3 empirically outperforms avg/max.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling.

    Args:
        p (float): Initial pooling exponent. Default 3.0.
        eps (float): Numerical stability epsilon.
        trainable (bool): Whether to learn p during training.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, trainable: bool = True):
        super().__init__()
        self.eps = eps
        if trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, C, H, W]
        Returns:
            Pooled features [B, C]
        """
        p = self.p.clamp(min=1.0) if isinstance(self.p, nn.Parameter) else self.p
        # Global pooling over the spatial dims. `mean` over (H, W) is numerically
        # identical to `avg_pool2d(kernel_size=(H, W))` but exports to ONNX as a
        # plain ReduceMean (v11), which onnx-tf supports — the dynamic kernel_size
        # form cannot be exported by the legacy TorchScript exporter.
        pooled = x.clamp(min=self.eps).pow(p).mean(dim=(-2, -1), keepdim=True)
        return pooled.pow(1.0 / p).squeeze(-1).squeeze(-1)

    def __repr__(self) -> str:
        p_val = self.p.item() if isinstance(self.p, nn.Parameter) else self.p
        return f"GeMPooling(p={p_val:.2f}, trainable={isinstance(self.p, nn.Parameter)})"
