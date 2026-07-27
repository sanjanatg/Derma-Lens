"""
loss_functions.py — Multi-Task Fairness-Aware Loss for FairDermNet (8-class)
-----------------------------------------------------------------------------
Total Loss = α × L_class + β × L_fitzpatrick + γ × L_fairness

  L_class       = FocalCrossEntropy(logits [B,8], class_labels [B], class_weights [8])
  L_fitzpatrick = CrossEntropy(fitz_logits [B,6], skin_tone_labels [B])
  L_fairness    = max(Sensitivity_i) − min(Sensitivity_i)
                  across Fitzpatrick groups i ∈ {0,1,2,3,4,5}
                  where Sensitivity = soft-TPR for malignant classes (0–3)

Hyperparameters:
  α = 1.0   (primary task weight)
  β = 0.15  (auxiliary Fitzpatrick head weight)
  γ = 0.25  (fairness regularisation weight)

Notes:
  - Malignant classes: indices 0–3 (MEL, BCC, SCC, AK)
  - Benign classes:    indices 4–7 (NV, BKL, DF, VASC)
  - L_fairness is computed as soft (differentiable) approximation.
  - When a batch has <2 skin tone groups, L_fairness = 0.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_MALIGNANT = 4   # class indices 0–3 are malignant (MEL, BCC, SCC, AK)


# ---------------------------------------------------------------------------
# Focal Loss component
# ---------------------------------------------------------------------------

def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Focal loss: FL = -α_t × (1 − p_t)^γ × log(p_t)
    Reduces loss contribution from easy well-classified examples.

    Args:
        logits: [B, C] raw logits (C = 8 for 8-class)
        labels: [B] long tensor of class indices
        gamma:  focusing parameter (2.0 recommended)
        weight: [C] per-class weights (for class imbalance)
    """
    ce = F.cross_entropy(logits, labels, weight=weight, reduction="none")  # [B]
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma) * ce
    return focal.mean()


# ---------------------------------------------------------------------------
# Per-group sensitivity (soft, differentiable) — 8-class aware
# ---------------------------------------------------------------------------

def _soft_sensitivity_per_group(
    risk_probs: torch.Tensor,    # [B, 8] softmax probs
    class_labels: torch.Tensor,  # [B] class index 0–7
    skin_tones: torch.Tensor,    # [B] 0–5 Fitzpatrick index
    num_tones: int = 6,
    n_malignant: int = N_MALIGNANT,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute soft (differentiable) sensitivity (TPR) per skin tone group.
    Sensitivity = P(predicted malignant | truly malignant).

    Malignant probability = sum of softmax probs for classes 0–(n_malignant-1).

    Returns:
        sens: [num_tones] tensor; groups absent from batch get value -1.
    """
    malignant_prob = risk_probs[:, :n_malignant].sum(dim=1)  # [B]
    pos_mask = (class_labels < n_malignant).float()          # 1 if truly malignant

    sensitivities = []
    for t in range(num_tones):
        group_mask = (skin_tones == t).float()               # [B]
        soft_tp = (malignant_prob * pos_mask * group_mask).sum()
        n_pos   = (pos_mask * group_mask).sum()
        if n_pos < eps:
            sensitivities.append(torch.tensor(-1.0, device=risk_probs.device))
        else:
            sensitivities.append(soft_tp / (n_pos + eps))

    return torch.stack(sensitivities)  # [num_tones]


def _equalized_odds_gap(
    sensitivities: torch.Tensor,  # [num_tones], -1 means group absent
) -> torch.Tensor:
    """max − min sensitivity gap across present groups. 0.0 if <2 groups."""
    present = sensitivities[sensitivities >= 0]
    if present.numel() < 2:
        return torch.tensor(0.0, device=sensitivities.device)
    return present.max() - present.min()


# ---------------------------------------------------------------------------
# Main Loss Class
# ---------------------------------------------------------------------------

class FairnessAwareLoss(nn.Module):
    """
    Multi-task fairness-aware loss for 8-class skin lesion classification.

    Args:
        alpha (float):        Weight for primary class loss.
        beta (float):         Weight for Fitzpatrick auxiliary head loss.
        gamma (float):        Weight for fairness regularisation term.
        focal_gamma (float):  Focusing parameter (0 = vanilla CE).
        class_weights:        [8] per-class weights. Pass result of
                              compute_class_weights_from_counts().
        label_smoothing:      Label smoothing for CE loss.
        num_classes (int):    Number of output classes (8).
        num_tones (int):      Number of Fitzpatrick skin tone groups (6).

    Usage:
        criterion = FairnessAwareLoss(
            alpha=1.0, beta=0.15, gamma=0.25,
            class_weights=weights.to(DEVICE),
            num_classes=8,
        )
        loss, breakdown = criterion(risk_logits, fitz_logits, class_labels, tones)
        loss.backward()
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.15,
        gamma: float = 0.25,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
        num_classes: int = 8,
        num_tones: int = 6,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.num_tones = num_tones
        self.n_malignant = N_MALIGNANT

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    # -----------------------------------------------------------------------
    def compute_risk_loss(
        self,
        logits: torch.Tensor,   # [B, num_classes]
        labels: torch.Tensor,   # [B] class indices 0–8
    ) -> torch.Tensor:
        """Focal CE + vanilla CE blend for class-imbalanced 8-class problem."""
        ce = F.cross_entropy(
            logits, labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        if self.focal_gamma > 0:
            fl = focal_loss(logits, labels,
                            gamma=self.focal_gamma,
                            weight=self.class_weights)
            return 0.5 * ce + 0.5 * fl
        return ce

    # -----------------------------------------------------------------------
    def compute_fitzpatrick_loss(
        self,
        fitz_logits: torch.Tensor,  # [B, 6]
        fitz_labels: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        return F.cross_entropy(fitz_logits, fitz_labels)

    # -----------------------------------------------------------------------
    def compute_fairness_loss(
        self,
        risk_logits: torch.Tensor,   # [B, 9]
        class_labels: torch.Tensor,  # [B] class index 0–8
        skin_tones: torch.Tensor,    # [B]
    ) -> torch.Tensor:
        """Equalized-odds gap across Fitzpatrick groups (soft/differentiable)."""
        risk_probs = F.softmax(risk_logits, dim=-1)
        sensitivities = _soft_sensitivity_per_group(
            risk_probs, class_labels, skin_tones,
            self.num_tones, n_malignant=self.n_malignant,
        )
        return _equalized_odds_gap(sensitivities)

    # -----------------------------------------------------------------------
    def forward(
        self,
        risk_logits: torch.Tensor,       # [B, 8]
        fitz_logits: torch.Tensor,       # [B, 6]
        class_labels: torch.Tensor,      # [B] — class index 0–7
        skin_tone_labels: torch.Tensor,  # [B] — 0–5 Fitzpatrick index
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            total_loss: scalar (backpropagatable)
            breakdown:  dict with individual loss values for logging
        """
        l_risk = self.compute_risk_loss(risk_logits, class_labels)
        l_fitz = self.compute_fitzpatrick_loss(fitz_logits, skin_tone_labels)
        l_fair = self.compute_fairness_loss(risk_logits, class_labels, skin_tone_labels)

        total = self.alpha * l_risk + self.beta * l_fitz + self.gamma * l_fair

        breakdown = {
            "loss_total":      total.item(),
            "loss_risk":       l_risk.item(),
            "loss_fitzpatrick": l_fitz.item(),
            "loss_fairness":   l_fair.item(),
        }
        return total, breakdown

    # -----------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: dict) -> "FairnessAwareLoss":
        return cls(
            alpha=cfg.get("alpha", 1.0),
            beta=cfg.get("beta", 0.15),
            gamma=cfg.get("gamma", 0.25),
        )


# ---------------------------------------------------------------------------
# Class weight helper (kept for backward compatibility)
# ---------------------------------------------------------------------------

def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int = 8,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from a label tensor.
    For the 8-class model, prefer compute_class_weights_from_counts()
    from class_taxonomy which uses actual per-class counts.
    """
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = counts.clamp(min=1)
    weights = counts.sum() / (num_classes * counts)
    return weights
