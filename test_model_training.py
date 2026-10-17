"""Run with python3 test_model_training.py; requires only PyTorch."""

import ast
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def test_model_training(filename='solar_filament_segmentation_mag_agent_u_net.py'):
    # ponytail: extract model classes to preserve notebook syntax and avoid dataset setup.
    source = Path(__file__).with_name(filename).read_text()
    source = '\n'.join('' if line.startswith('!') else line for line in source.splitlines())
    names = {'AttentionGate', 'RSU', 'CognitiveRefiner', 'MagAgentU2Net', 'DiceBCELoss'}
    classes = [node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name in names]
    namespace = {'torch': torch, 'nn': nn, 'F': F}
    exec(compile(ast.Module(body=classes, type_ignores=[]), '<notebook model>', 'exec'), namespace)

    torch.manual_seed(0)
    model = namespace['MagAgentU2Net']().train()
    if filename == 'lightweight_experiment.py':
        assert sum(p.numel() for p in model.parameters()) < 3_000_000
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    images = torch.randn(2, 3, 32, 32)
    masks = (torch.rand(2, 1, 32, 32) > 0.95).float()
    before = model.cognitive.out_conv.weight.detach().clone()
    logits = model(images)
    assert logits.shape == masks.shape, logits.shape
    loss = namespace['DiceBCELoss']()(logits, masks)
    assert torch.isfinite(loss), loss
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    optimizer.step()
    assert not torch.equal(before, model.cognitive.out_conv.weight)
    print(f'PASS {filename}: {sum(p.numel() for p in model.parameters()):,} parameters, '
          f'output {tuple(logits.shape)}, finite loss {loss.item():.4f}, gradients, optimizer update')
    if filename == 'lightweight_experiment.py':
        # Large half-precision predictions must not overflow Dice reductions.
        half_logits = torch.zeros(2, 1, 512, 512, dtype=torch.float16, requires_grad=True)
        loss = namespace['DiceBCELoss']()(half_logits, torch.zeros_like(half_logits).float())
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(half_logits.grad).all()


if __name__ == '__main__':
    torch.set_num_threads(1)
    test_model_training()
    test_model_training('lightweight_experiment.py')
