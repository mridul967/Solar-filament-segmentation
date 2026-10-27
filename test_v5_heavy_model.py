"""CPU verification of V5. Kaggle's real T4 checks use V5_MODE=smoke/diagnose.

Run with torch, numpy, opencv-python-headless, pycocotools, pandas,
albumentations and matplotlib installed. No Kaggle data/downloads required.
"""
import importlib.util
import json
import os
import runpy
import shutil
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/v5-matplotlib')
os.environ.setdefault('MPLBACKEND', 'Agg')

import cv2
import numpy as np
import torch


def load_module():
    path = Path(__file__).parent / 'heavy-models/V5-heavy-model.py'
    spec = importlib.util.spec_from_file_location('v5_notebook', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(root):
    images, annotations = [], []
    root.mkdir()
    for i in range(10):
        name = f'solar_{i}.jpeg'
        image = np.tile(np.arange(64, dtype=np.uint8), (64, 1)) * 3 + 30
        image[20:30, 12:48] = 20
        assert cv2.imwrite(str(root / name), image)
        for annotator in range(2):
            image_id = f'{i}_{annotator}'
            images.append({'id': image_id, 'file_name': name, 'width': 64, 'height': 64})
            annotations.append({'id': len(annotations) + 1, 'image_id': image_id,
                                'category_id': 1, 'segmentation': [[12, 20, 48, 20, 48, 30, 12, 30]],
                                'bbox': [12, 20, 36, 10], 'area': 360, 'iscrowd': 0})
    path = root / 'annotations.json'
    path.write_text(json.dumps({'images': images, 'annotations': annotations,
                                'categories': [{'id': 1, 'name': 'filament'}]}))
    return path


def main():
    torch.set_num_threads(1)
    v5 = load_module()
    c = v5.Config
    c.DEVICE, c.USE_DP, c.NUM_WORKERS = 'cpu', False, 0
    c.IMG_SIZE, c.ORIGINAL_SIZE, c.TILE_STRIDE, c.GAUSSIAN_SIGMA = 32, 64, 16, 8.0
    c.MIN_AREA, c.VAL_CACHE_ITEMS = 1, 1
    c.PERSISTENT_WORKERS, c.EVAL_SERIAL = False, False
    c.MODE, c.PREFLIGHT = 'smoke', True
    v5.seed_everything()
    assert not torch.backends.mha.get_fastpath_enabled()
    model = v5.MagAgentU2Net()
    assert sum(p.numel() for p in model.parameters()) == 21_133_419, 'Architecture changed'
    images = torch.randn(2, 3, 32, 32)
    targets = (torch.rand(2, 1, 32, 32) > .85).float()
    model.train()
    logits, aux = v5.forward_model(model, images, True)
    criterion = v5.DiceBCELoss()
    first, p1 = criterion(logits, targets, targets, aux, epoch=1, return_parts=True)
    second, p2 = criterion(logits, targets, targets, aux, epoch=2, return_parts=True)
    assert p1['lovasz'] == 0 and p2['lovasz'] > 0 and second > first
    second.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        reference = v5.forward_model(model, images).clone()
        torch.testing.assert_close(reference[:1], v5.forward_model(model, images[:1]), rtol=1e-4, atol=1e-5)
        c.EVAL_SERIAL = True
        serial, serial_aux = v5.forward_model(model, images, True)
        torch.testing.assert_close(reference, serial, rtol=1e-4, atol=1e-5)
        assert serial_aux.shape[0] == 2
        c.EVAL_SERIAL = False
    # Unrelated equal-shape weights must not be transplanted.
    bridge = model.vlm_bridge
    before = bridge.stage1.norm1.weight.detach().clone()
    assert not bridge._copy_matching_tensors({'unrelated.weight': torch.zeros_like(before)}, 'unrelated')
    torch.testing.assert_close(before, bridge.stage1.norm1.weight)
    try:
        bridge._copy_matching_tensors({'a': torch.zeros_like(before), 'b': torch.zeros(7)}, 'invalid',
                                     {'stage1.norm1.weight': 'a', 'stage2.norm1.weight': 'b'})
        raise AssertionError('Invalid transfer accepted')
    except ValueError:
        torch.testing.assert_close(before, bridge.stage1.norm1.weight)
    # Native bridge transfer is strict and complete.
    assert bridge._copy_matching_tensors(bridge.state_dict(), 'native bridge')
    with tempfile.TemporaryDirectory(prefix='v5-test-') as tmp:
        root = Path(tmp)
        model_path, bridge_path = root / 'best.pth', root / 'bridge.pth'
        metadata = {'run_id': 'test', 'experiment': v5.experiment_config(),
                    'split_sha256': 'split', 'annotation_sha256': 'annotations'}
        v5.save_v5_checkpoints(model, model_path, bridge_path, metadata)
        # Recovery inference must choose embedded BEST, never the latest weights.
        saved_best = torch.load(model_path, weights_only=True)
        latest = dict(saved_best['model'])
        latest['cognitive.out_conv.bias'] = latest['cognitive.out_conv.bias'] + 100
        recovery_path = root / 'recover_best.pth'
        v5.atomic_torch_save({'format_version': 2, 'kind': 'training', 'epoch': 21,
                             'phase': 'validated', 'model': latest,
                             'best_model': {'model': saved_best['model'],
                                            'metadata': saved_best['metadata']}}, recovery_path)
        cell = runpy.run_path(str(Path(__file__).parent / 'heavy-models/V5-inference-cell.py'),
                             run_name='recovery_cell_test')
        recovered, _ = cell['load_saved_best'](vars(v5), recovery_path)
        with torch.no_grad():
            torch.testing.assert_close(reference, v5.forward_model(recovered, images))
        del recovered, saved_best, latest
        with torch.no_grad():
            model.cognitive.out_conv.weight.add_(10)
        v5.load_v5_checkpoints(model, model_path, bridge_path, expected=metadata)
        with torch.no_grad():
            torch.testing.assert_close(reference, v5.forward_model(model, images))
        corrupted = torch.load(bridge_path, weights_only=True)
        corrupted['metadata']['checkpoint_id'] = 'wrong'
        v5.atomic_torch_save(corrupted, bridge_path)
        try:
            v5.load_v5_checkpoints(model, model_path, bridge_path)
            raise AssertionError('Mismatched checkpoint pair accepted')
        except ValueError:
            pass
        # A complete model does not depend on the separate bridge export.
        v5.load_v5_checkpoints(model, model_path)
        checkpoint = torch.load(model_path, weights_only=True)
        checkpoint['model'].pop('vlm_bridge.stage1.norm1.weight')
        v5.atomic_torch_save(checkpoint, root / 'incomplete.pth')
        try:
            v5.load_v5_checkpoints(model, root / 'incomplete.pth')
            raise AssertionError('Missing bridge tensor accepted')
        except RuntimeError:
            pass
        # RNG capture is weights_only-safe and replays Python/numpy/torch.
        rng = v5.capture_rng()
        expected = (v5.random.random(), np.random.rand(), torch.rand(3))
        v5.atomic_torch_save(rng, root / 'rng.pth')
        v5.restore_rng(torch.load(root / 'rng.pth', weights_only=True))
        assert expected[:2] == (v5.random.random(), np.random.rand())
        torch.testing.assert_close(expected[2], torch.rand(3))
        annotation_path = fixture(root / 'images')
        c.TRAIN_DIR, c.TEST_DIR = str(annotation_path.parent), str(annotation_path.parent)
        c.ANNOTATIONS_FILE, c.OUT_DIR = str(annotation_path), str(root / 'run')
        c.RESUME_PATH = c.CHECKPOINT_PATH = ''
        # No download, real data processing, training, validation and tiled/RLE path.
        del model, bridge, logits, aux, first, second, checkpoint, corrupted
        original_save = v5.save_training_checkpoint

        def capture_pending(path, *args, **kwargs):
            original_save(path, *args, **kwargs)
            if args[5:7] == (2, 'trained'):
                shutil.copyfile(path, root / 'pending.pth')

        v5.save_training_checkpoint = capture_pending
        v5.main()
        v5.save_training_checkpoint = original_save
        run = root / 'run' / 'smoke'
        history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
        assert len(history) == 2
        assert history[0]['train']['lovasz'] == 0 and history[1]['train']['lovasz'] > 0
        assert all(row['validation']['records'] == 3 for row in history)
        saved = torch.load(run / 'last_training.pth', weights_only=True)
        assert saved['phase'] == 'validated' and saved['epoch'] == 2
        assert abs(saved['scheduler']['last_epoch'] - 2) == 0
        assert saved['best_model'] is not None and saved['optimizer']['state']
        # Pending validation must replay without an optimizer/scheduler double step.
        learning_rate = saved['optimizer']['param_groups'][0]['lr']
        del saved
        c.OUT_DIR, c.RESUME_PATH = str(root / 'resumed'), str(root / 'pending.pth')
        v5.main()
        resumed = [json.loads(line) for line in (root / 'resumed/smoke/history.jsonl').read_text().splitlines()]
        assert [row['epoch'] for row in resumed] == [2]
        assert (root / 'resumed/smoke/best_model.pth').is_file()
        assert abs(resumed[0]['validation']['loss'] - history[1]['validation']['loss']) < 1e-6
        resumed_checkpoint = torch.load(root / 'resumed/smoke/last_training.pth', weights_only=True)
        assert resumed_checkpoint['scheduler']['last_epoch'] == 2
        assert resumed_checkpoint['optimizer']['param_groups'][0]['lr'] == learning_rate
        del resumed_checkpoint
        # Cache is bounded by physical filename, with unchanged tensors.
        coco = v5.COCO(str(annotation_path))
        ids = list(coco.imgs)
        ds = v5.SolarDataset(coco, ids, c.TRAIN_DIR, 32, v5.val_transforms, True)
        a = ds[0][0]
        b = ds[1][0]
        torch.testing.assert_close(a, b)
        ds[2]
        assert len(ds._image_cache) == 1
        train = v5.SolarDataset(coco, ids, c.TRAIN_DIR, 32, v5.train_transforms, True, True)
        train.epoch.fill_(3)
        one, two = train[0], train[0]
        for left, right in zip(one[:3], two[:3]):
            torch.testing.assert_close(left, right)
        try:
            v5.load_image(str(root / 'missing.jpeg'))
            raise AssertionError('Missing image became random noise')
        except FileNotFoundError:
            pass
        # Matching sanity uses instance masks, including empty predictions.
        m = np.zeros((8, 8), dtype=np.uint8)
        m[1:4, 1:4] = 1
        assert v5.pq_metrics([m], [m])['pq'] == 1
        assert v5.pq_metrics([], [m])['fn'] == 1
        assert v5.pq_metrics([m], [])['fp'] == 1
        audit = json.loads((run / 'smoke_submission.csv.audit.json').read_text())
        assert audit['rle_roundtrip_verified'] and audit['input_images'] == 1
        # Remaining Kaggle entry points share the saved fixed weights and split.
        c.RESUME_PATH = ''
        c.CHECKPOINT_PATH = str(run / 'best_model.pth')
        c.MODE, c.OUT_DIR = 'diagnose', str(root / 'diagnostic')
        v5.main()
        capture = root / 'diagnostic/diagnose/diagnostic_batch_1.pth'
        assert capture.is_file()
        c.REPLAY_BATCH, c.OUT_DIR = str(capture), str(root / 'replay')
        v5.main()
        replay = json.loads((root / 'replay/diagnose/diagnostic_results.json').read_text())
        assert replay['records'] == 2
        c.REPLAY_BATCH = ''
        c.MODE, c.OUT_DIR = 'benchmark', str(root / 'timing')
        c.PROFILE_WARMUP, c.PROFILE_BATCHES, c.PROFILE_EPOCH = 1, 1, 2
        v5.main()
        timing = json.loads((root / 'timing/benchmark/history.jsonl').read_text().splitlines()[0])
        assert timing['train']['objective_epoch'] == 2 and timing['train']['lovasz'] > 0
        assert set(timing['train']['timings_seconds']) == {
            'data_wait', 'transfer', 'forward', 'criterion', 'backward', 'optimizer', 'batch'}
        c.MODE, c.OUT_DIR = 'evaluate', str(root / 'evaluate')
        v5.main()
        native = json.loads((root / 'evaluate/native_validation.json').read_text())
        assert native['physical_files'] == 2 and native['annotation_records'] == 4
        c.MODE, c.OUT_DIR, c.EXPECTED_TEST_IMAGES = 'submit', str(root / 'submit'), 10
        v5.main()
        submission = json.loads((root / 'submit/submission.csv.audit.json').read_text())
        assert submission['input_images'] == 10 and submission['rle_roundtrip_verified']
    print('PASS: architecture, finite gradients, warmup, strict transfer/checkpoints, RNG/resume, '
          'real two-epoch smoke pipeline, singleton eval, tiles/TTA/RLE, cache, image failure and PQ')


if __name__ == '__main__':
    main()
