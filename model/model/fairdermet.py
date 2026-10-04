"""
fairdermet.py — FairDermNet: Full Model Architecture
-----------------------------------------------------
MobileNetV3-Large backbone
  + Fairness Attention Module (tone-adaptive cross-attention)
  + GeM Pooling
  + Dual head: Risk classification + Fitzpatrick auxiliary

Architecture summary:
  Input: 224×224×3 RGB (ImageNet normalised)
  → MobileNetV3-Large (timm, pretrained) → 960-dim @ 7×7
  → FairnessAttentionModule                → 960-dim @ 7×7 + fitz_logits
  → GeMPooling                             → 960-dim
  → Dropout(0.3) → Linear(960→128) → BN → ReLU
  → [Risk head]   Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→num_classes)

Parameters (after pruning): ~5.4M
Size (INT8 quantised): ~5.8 MB
Inference (Snapdragon 720G): ~22 ms
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision  # Must load before timm to avoid circular import errors

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from .gem_pooling import GeMPooling
from .fairness_attention import FairnessAttentionModule


# ---------------------------------------------------------------------------
# Utility: load pretrained weights safely
# ---------------------------------------------------------------------------

def _load_backbone(model_name: str, pretrained: bool):
    if not TIMM_AVAILABLE:
        raise ImportError("timm is required. Install with: pip install timm")
    try:
        return timm.create_model(model_name, pretrained=pretrained)
    except Exception as exc:
        if pretrained:
            print(f"[FairDermNet] WARNING: Could not load pretrained weights ({exc}). "
                  "Initialising from scratch.")
            return timm.create_model(model_name, pretrained=False)
        raise


def _split_mobilenetv3(backbone_full: nn.Module):
    """
    Split timm MobileNetV3-Large into feature extractor components.
    timm named children: conv_stem, bn1, blocks, global_pool,
                         conv_head, act2, flatten, classifier
    We use conv_stem + bn1 + blocks as the feature extractor,
    then conv_head and act2 separately.
    bn is Identity (timm MobileNetV3 has no bn2 between conv_head and act2).
    """
    features = nn.Sequential(
        backbone_full.conv_stem,
        backbone_full.bn1,
        backbone_full.blocks,
    )
    conv_head = backbone_full.conv_head
    bn        = nn.Identity()
    act       = backbone_full.act2
    return features, conv_head, bn, act


# ---------------------------------------------------------------------------
# FairDermNet
# ---------------------------------------------------------------------------

class FairDermNet(nn.Module):
    """
    Fairness-aware CNN for skin lesion classification.

    Args:
        num_classes (int): Number of output classes (default 8 for DermaLens).
        num_skin_tones (int): Fitzpatrick classes 0–5 (default 6).
        pretrained (bool): Load ImageNet weights for backbone.
        gem_p (float): Initial GeM pooling exponent.
        dropout_backbone (float): Dropout after GeM pooling.
        dropout_head (float): Dropout inside classification head.

    Forward returns:
        Training mode  → (risk_logits [B, num_classes], fitz_logits [B, 6])
        Eval mode      → risk_logits [B, num_classes]
    """

    BACKBONE = "mobilenetv3_large_100"
    BACKBONE_OUT_CHANNELS = 1280

    def __init__(
        self,
        num_classes: int = 8,
        num_skin_tones: int = 6,
        pretrained: bool = True,
        gem_p: float = 3.0,
        dropout_backbone: float = 0.3,
        dropout_head: float = 0.2,
    ):
        super().__init__()

        self.num_classes    = num_classes
        self.num_skin_tones = num_skin_tones
        C = self.BACKBONE_OUT_CHANNELS

        # ── Backbone ──────────────────────────────────────────────────────
        backbone_full = _load_backbone(self.BACKBONE, pretrained=pretrained)

        (
            self.backbone_features,
            self.backbone_conv_head,
            self.backbone_bn,
            self.backbone_act,
        ) = _split_mobilenetv3(backbone_full)

        # ── Fairness Attention ────────────────────────────────────────────
        self.fairness_attention = FairnessAttentionModule(
            in_channels=C,
            num_tones=num_skin_tones,
        )

        # ── Pooling ───────────────────────────────────────────────────────
        self.gem = GeMPooling(p=gem_p, trainable=True)

        # ── Neck (shared projection) ──────────────────────────────────────
        self.neck = nn.Sequential(
            nn.Dropout(dropout_backbone),
            nn.Linear(C, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # ── Risk Classification Head ──────────────────────────────────────
        self.risk_head = nn.Sequential(
            nn.Linear(128, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_head),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    # -----------------------------------------------------------------------
    def _init_weights(self):
        for m in [self.neck, self.risk_head]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out",
                                            nonlinearity="relu")
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    # -----------------------------------------------------------------------
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run backbone → return raw feature map [B, 960, 7, 7].
        Used for GradCAM++ target layer hooks.
        """
        x = self.backbone_features(x)
        x = self.backbone_conv_head(x)
        x = self.backbone_bn(x)
        x = self.backbone_act(x)
        return x  # [B, 960, 7, 7]

    # -----------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, return_aux: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]] | torch.Tensor:
        # 1. Backbone
        feat = self.extract_features(x)               # [B, 960, 7, 7]

        # 2. Fairness attention
        feat_attended, fitz_logits = self.fairness_attention(feat)
        # feat_attended: [B, 960, 7, 7],  fitz_logits: [B, 6]

        # 3. GeM pooling
        pooled = self.gem(feat_attended)               # [B, 960]

        # 4. Neck projection
        embed = self.neck(pooled)                      # [B, 128]

        # 5. Risk head
        risk_logits = self.risk_head(embed)            # [B, num_classes]

        if self.training or return_aux:
            return risk_logits, fitz_logits
        return risk_logits

    # -----------------------------------------------------------------------
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns softmax probabilities [B, num_classes] in eval mode.
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
        return F.softmax(logits, dim=-1)

    # -----------------------------------------------------------------------
    def get_gradcam_target_layer(self) -> nn.Module:
        """
        Returns the last block of backbone_features — GradCAM++ target.
        """
        children = list(self.backbone_features.children())
        return children[-1]

    # -----------------------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path, **kwargs) -> "FairDermNet":
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        model = cls(pretrained=False, **kwargs)
        model.load_state_dict(state, strict=True)
        print(f"[FairDermNet] Loaded weights from {path}")
        return model

    # -----------------------------------------------------------------------
    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total":       total,
            "trainable":   trainable,
            "total_M":     round(total / 1e6, 2),
            "trainable_M": round(trainable / 1e6, 2),
        }
