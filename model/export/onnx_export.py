"""
onnx_export.py — PyTorch → ONNX export for FairDermNet (8-class)
=================================================================
Exports the model in inference mode (eval, single-head output).
The Fitzpatrick auxiliary branch is NOT traced — it is inactive in eval mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch


def export_to_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    input_size: int | Tuple[int, int, int, int] = 224,
    opset_version: int = 11,
    simplify: bool = True,
) -> Path:
    """
    Export FairDermNet to ONNX.

    Args:
        model:         FairDermNet instance (will be set to eval mode).
        output_path:   Destination .onnx path.
        input_size:    Either an int (square spatial size) or a full (N, C, H, W) tuple.
        opset_version: ONNX opset. Pinned to 11 for onnx-tf compatibility:
                       onnx-tf 1.10.0 dispatches each op to a handler whose version
                       equals the model opset, and only implements Squeeze/Unsqueeze
                       (axes-as-attribute) and ReduceMean v11+ up to opset 11. At
                       opset 11 torch also auto-decomposes HardSwish (opset 14+
                       only), which onnx-tf has no handler for.
        simplify:      Run onnx-simplifier post-export if available.

    Returns:
        Path to the exported .onnx file.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    if isinstance(input_size, int):
        shape = (1, 3, input_size, input_size)
    else:
        shape = tuple(input_size)

    dummy = torch.randn(*shape)

    print(f"[ONNX Export] Tracing model with input {shape} ...")
    torch.onnx.export(
        model,
        dummy,
        str(output),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["risk_logits"],
        dynamic_axes={
            "input":       {0: "batch"},
            "risk_logits": {0: "batch"},
        },
        do_constant_folding=True,
        # Use the legacy TorchScript exporter: the default dynamo exporter in
        # torch 2.13 refuses opsets < 18, but onnx-tf needs opset 11 (see above).
        dynamo=False,
    )

    size_mb = output.stat().st_size / 1e6
    print(f"[ONNX Export] ✓ Saved → {output}  ({size_mb:.2f} MB)")

    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            model_onnx, ok = onnx_simplify(onnx.load(str(output)))
            if ok:
                onnx.save(model_onnx, str(output))
                print("[ONNX Export] ✓ Simplified with onnxsim")
            else:
                print("[ONNX Export] ⚠ onnxsim returned check=False; keeping original.")
        except ImportError:
            print("[ONNX Export] ⚠ onnxsim not installed — skipping simplification.")
            print("              pip install onnxsim")

    return output
