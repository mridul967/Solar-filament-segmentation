"""Small CPU smoke test for the notebook-style V4 model."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def load_v4_symbols():
    source = Path("heavy-models/V4-heavy-model.py").read_text()
    source = "\n".join("" if line.startswith("!") else line for line in source.splitlines())
    wanted = {
        "AttentionGate", "RSU", "CognitiveRefinerFeatures", "MagAgentU2NetV4",
        "masked_bce_dice", "V4Loss", "pq_metrics", "assert_pq_sanity",
    }
    tree = ast.parse(source)
    nodes = [node for node in tree.body if getattr(node, "name", None) in wanted]
    config = SimpleNamespace(AUX_WARMUP_EPOCHS=3, BOUNDARY_WEIGHT=0.20,
                             SPINE_WEIGHT=0.20, PQ_IOU_THRESHOLD=0.50)
    namespace = {"torch": torch, "nn": nn, "F": F, "np": np, "Config": config,
                 "defaultdict": __import__("collections").defaultdict}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<v4>", "exec"), namespace)
    return namespace


def main():
    torch.manual_seed(0)
    symbols = load_v4_symbols()
    model = symbols["MagAgentU2NetV4"]().train()
    inputs = torch.randn(2, 3, 32, 32)
    targets = {
        "foreground": (torch.rand(2, 1, 32, 32) > 0.95).float(),
        "boundary": (torch.rand(2, 1, 32, 32) > 0.98).float(),
        "spine": (torch.rand(2, 1, 32, 32) > 0.98).float(),
        "disk": torch.ones(2, 1, 32, 32),
        "spine_valid": torch.ones(2, 1, 32, 32),
    }
    outputs = model(inputs)
    assert set(outputs) == {"foreground", "boundary", "spine"}
    assert all(value.shape == targets["foreground"].shape for value in outputs.values())
    loss, parts = symbols["V4Loss"]({"boundary": 5.0, "spine": 5.0})(outputs, targets, epoch=2)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(torch.as_tensor(value)) for value in parts.values())
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters() if parameter.requires_grad)

    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    assert symbols["pq_metrics"]([mask], [mask])["pq"] == 1.0
    assert symbols["pq_metrics"]([], [mask])["fn"] == 1
    symbols["assert_pq_sanity"]()
    print(f"PASS: V4 outputs, finite multi-task loss ({loss.item():.4f}), gradients, and PQ sanity")


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
