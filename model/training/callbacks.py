from pathlib import Path
import torch


class EarlyStopping:
    def __init__(self, patience=8, mode="max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            return False
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def save_checkpoint(model, optimizer, epoch, metric, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric": metric,
        },
        path,
    )


class ModelCheckpoint:
    def __init__(self, path, monitor="balanced_accuracy", mode="max"):
        self.path = Path(path)
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def step(self, model, optimizer, epoch, metrics: dict):
        value = metrics.get(self.monitor, 0.0)
        if self.best is None:
            improved = True
        else:
            improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            save_checkpoint(model, optimizer, epoch, value, self.path)
            print(f"  💾 Checkpoint saved: {self.monitor}={value:.4f}")
            return True
        return False
