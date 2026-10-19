"""Run with python3 test_model_training.py; requires only PyTorch."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def test_panoptic_metric():
    source = Path(__file__).with_name('lightweight_experiment.py').read_text()
    source = '\n'.join('' if line.startswith('!') else line for line in source.splitlines())
    names = {'panoptic_stats', 'finalize_panoptic'}
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]

    class Cv2Stub:
        @staticmethod
        def connectedComponents(mask):
            labels = np.zeros_like(mask, dtype=np.int32)
            label = 0
            for y, x in zip(*np.nonzero(mask)):
                if labels[y, x]:
                    continue
                label += 1
                stack = [(y, x)]
                while stack:
                    yy, xx = stack.pop()
                    if not (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]):
                        continue
                    if not mask[yy, xx] or labels[yy, xx]:
                        continue
                    labels[yy, xx] = label
                    stack.extend((yy + dy, xx + dx) for dy in (-1, 0, 1)
                                 for dx in (-1, 0, 1) if dy or dx)
            return label + 1, labels

    namespace = {
        'np': np,
        'cv2': Cv2Stub,
        'Config': SimpleNamespace(MIN_AREA=1, PQ_IOU_THRESHOLD=0.5,
                                  DIAGNOSTIC_IOU_THRESHOLD=0.1),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<PQ functions>', 'exec'), namespace)
    first = np.zeros((8, 8), dtype=np.uint8)
    second = np.zeros_like(first)
    first[1:3, 1:3] = 1
    second[5:7, 5:7] = 1

    class CocoStub:
        masks = {1: first, 2: second}

        def annToMask(self, annotation):
            return self.masks[annotation['id']]

    coco = CocoStub()
    annotations = [{'id': 1}, {'id': 2}]
    perfect = namespace['finalize_panoptic'](
        namespace['panoptic_stats'](first | second, annotations, coco))
    assert perfect['pq'] == perfect['sq'] == perfect['rq'] == 1.0

    merged = first | second
    for position in range(2, 6):
        merged[position, position] = 1
    failure = namespace['finalize_panoptic'](
        namespace['panoptic_stats'](merged, annotations, coco))
    assert failure['pq'] == 0.0 and failure['merges'] == 1
    print('PASS V1 PQ: perfect instances score 1; merged instances are diagnosed')


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
    test_panoptic_metric()
