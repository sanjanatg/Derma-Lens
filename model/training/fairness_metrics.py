import torch
import numpy as np
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FITZPATRICK_NAMES = {
    0: "Type I (Very Fair)",
    1: "Type II (Fair)",
    2: "Type III (Medium)",
    3: "Type IV (Olive)",
    4: "Type V (Brown)",
    5: "Type VI (Dark)",
}

# ---------------------------------------------------------------------------
# Existing low-level functions (unchanged)
# ---------------------------------------------------------------------------

def group_sensitivity(
    preds: torch.Tensor,
    labels: torch.Tensor,
    groups: torch.Tensor,
    positive_class: int = 1,
) -> Dict[int, float]:
    pred_cls = preds.argmax(dim=1)
    out = {}
    for g in torch.unique(groups):
        g_int = int(g.item())
        if g_int < 0:   # unknown tone (-1) — excluded from fairness metrics
            continue
        m = groups == g
        y_true = labels[m]
        y_pred = pred_cls[m]
        positives = (y_true == positive_class).sum().float()
        if positives.item() == 0:
            out[g_int] = 0.0
            continue
        tp = ((y_true == positive_class) & (y_pred == positive_class)).sum().float()
        out[g_int] = float(tp / (positives + 1e-8))
    return out


def equalized_odds_gap(preds, labels, groups) -> float:
    sens = group_sensitivity(preds, labels, groups)
    values = list(sens.values())
    if len(values) < 2:
        return 0.0
    return max(values) - min(values)


def demographic_parity_gap(preds, groups, positive_class: int = 1) -> float:
    pred_cls = preds.argmax(dim=1)
    rates = []
    for g in torch.unique(groups):
        g_int = int(g.item())
        if g_int < 0:   # unknown tone (-1) — excluded
            continue
        m = groups == g
        rate = (pred_cls[m] == positive_class).float().mean().item()
        rates.append(rate)
    if len(rates) < 2:
        return 0.0
    return max(rates) - min(rates)


def performance_parity_gap(metric_by_group: Dict) -> float:
    if len(metric_by_group) < 2:
        return 0.0
    vals = list(metric_by_group.values())
    return max(vals) - min(vals)


# ---------------------------------------------------------------------------
# compute_epoch_metrics
# Called every epoch inside the training loop to get scalar summary metrics.
# Returns a dict of floats safe to log / compare for early stopping.
# ---------------------------------------------------------------------------

def compute_epoch_metrics(
    all_preds: torch.Tensor,   # [N, num_classes] logits or softmax
    all_labels: torch.Tensor,  # [N] integer class indices
    all_groups: torch.Tensor,  # [N] Fitzpatrick 0-5
    num_classes: int = 8,
) -> Dict[str, float]:
    """
    Compute per-epoch summary metrics:
      - overall accuracy
      - balanced accuracy (mean per-class recall)
      - equalized odds gap across Fitzpatrick groups
      - per-class accuracy dict
    """
    pred_cls = all_preds.argmax(dim=1)
    correct = (pred_cls == all_labels).float()
    accuracy = correct.mean().item()

    # Per-class recall → balanced accuracy
    per_class_recall = []
    per_class_acc = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() == 0:
            per_class_acc[c] = 0.0
            continue
        recall = correct[mask].mean().item()
        per_class_recall.append(recall)
        per_class_acc[c] = recall
    balanced_acc = float(np.mean(per_class_recall)) if per_class_recall else 0.0

    # Fairness: equalized odds gap (binary: malignant = classes 0,1,2)
    malignant_classes = {0, 1, 2}
    binary_labels = torch.tensor(
        [1 if l.item() in malignant_classes else 0 for l in all_labels],
        dtype=torch.long,
    )
    # Build binary preds: sum of malignant class probs
    if all_preds.shape[1] >= 3:
        malignant_prob = all_preds[:, :3].sum(dim=1, keepdim=True)
        benign_prob = 1.0 - malignant_prob
        binary_preds = torch.cat([benign_prob, malignant_prob], dim=1)
    else:
        binary_preds = all_preds

    eo_gap = equalized_odds_gap(binary_preds, binary_labels, all_groups)

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "equalized_odds_gap": eo_gap,
        "per_class_accuracy": per_class_acc,
    }


# ---------------------------------------------------------------------------
# compute_fairness_metrics
# Detailed per-group breakdown — used in Cell 9b (DDI eval) and Cell 10
# (fairness bar charts).
# ---------------------------------------------------------------------------

def compute_fairness_metrics(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    all_groups: torch.Tensor,
    num_classes: int = 8,
) -> Dict:
    """
    Returns per-Fitzpatrick-group accuracy, sensitivity (malignant recall),
    and specificity, plus overall equalized odds gap.
    """
    malignant_classes = {0, 1, 2, 3}   # MEL, BCC, SCC, AK
    pred_cls = all_preds.argmax(dim=1)

    group_metrics = {}
    for g in torch.unique(all_groups):
        g_int = int(g.item())
        if g_int < 0:   # unknown tone (-1) — excluded from fairness breakdown
            continue
        mask = all_groups == g

        y_true = all_labels[mask]
        y_pred = pred_cls[mask]

        # Overall accuracy for this group
        acc = (y_true == y_pred).float().mean().item()

        # Binary sensitivity (malignant recall)
        mal_mask = torch.tensor(
            [l.item() in malignant_classes for l in y_true], dtype=torch.bool
        )
        if mal_mask.sum() > 0:
            mal_pred_mask = torch.tensor(
                [p.item() in malignant_classes for p in y_pred], dtype=torch.bool
            )
            sensitivity = (mal_mask & mal_pred_mask).sum().float() / (
                mal_mask.sum().float() + 1e-8
            )
            sensitivity = sensitivity.item()
        else:
            sensitivity = 0.0

        # Binary specificity (benign correct rate)
        ben_mask = ~mal_mask
        if ben_mask.sum() > 0:
            ben_pred_mask = torch.tensor(
                [p.item() not in malignant_classes for p in y_pred], dtype=torch.bool
            )
            specificity = (ben_mask & ben_pred_mask).sum().float() / (
                ben_mask.sum().float() + 1e-8
            )
            specificity = specificity.item()
        else:
            specificity = 0.0

        group_metrics[g_int] = {
            "name": FITZPATRICK_NAMES.get(g_int, f"Type {g_int}"),
            "n": int(mask.sum().item()),
            "accuracy": acc,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

    eo_gap = equalized_odds_gap(
        all_preds,
        torch.tensor(
            [1 if l.item() in malignant_classes else 0 for l in all_labels],
            dtype=torch.long,
        ),
        all_groups,
    )

    return {
        "group_metrics": group_metrics,
        "equalized_odds_gap": eo_gap,
    }


# ---------------------------------------------------------------------------
# build_fairness_report
# Human-readable summary dict — used in Cell 13 (final metric summary).
# ---------------------------------------------------------------------------

def build_fairness_report(
    fairness_metrics_dict: Dict,
    overall_accuracy: Optional[float] = None,
    overall_balanced_accuracy: Optional[float] = None,
) -> Dict:
    """
    Wraps compute_fairness_metrics output into a clean report dict
    suitable for printing or saving as JSON.
    """
    group_metrics = fairness_metrics_dict.get("group_metrics", {})
    eo_gap = fairness_metrics_dict.get("equalized_odds_gap", 0.0)

    rows = []
    for g_int, gm in sorted(group_metrics.items()):
        rows.append(
            {
                "fitzpatrick": gm["name"],
                "n": gm["n"],
                "accuracy": round(gm["accuracy"] * 100, 2),
                "sensitivity": round(gm["sensitivity"] * 100, 2),
                "specificity": round(gm["specificity"] * 100, 2),
            }
        )

    report = {
        "per_group": rows,
        "equalized_odds_gap_pct": round(eo_gap * 100, 2),
        "fairness_target_met": eo_gap < 0.05,
    }
    if overall_accuracy is not None:
        report["overall_accuracy_pct"] = round(overall_accuracy * 100, 2)
    if overall_balanced_accuracy is not None:
        report["overall_balanced_accuracy_pct"] = round(
            overall_balanced_accuracy * 100, 2
        )

    return report