# Paste this file into ONE Kaggle cell after restarting the kernel.
# Attach the SAME updated V5 source, saved outputs, and competition test images.
# If needed, first run: %pip install 'albumentations>=2.0' pycocotools -q

from pathlib import Path
import runpy
import torch


def load_saved_best(v5, checkpoint_path):
    """Select saved validation-best weights, including the matching visual bridge."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 2:
        raise ValueError('Expected the updated V5 checkpoint format; legacy split weights need separate handling.')
    if checkpoint.get('kind') == 'training':
        print(f"Recovery file: epoch {checkpoint['epoch']}, phase={checkpoint['phase']}")
        selected = checkpoint.get('best_model')
        if selected is None:
            raise ValueError('This training checkpoint contains no validated best model.')
    elif checkpoint.get('kind') == 'inference':
        selected = checkpoint
    else:
        raise ValueError('Use best_model.pth or last_training.pth, not the bridge-only export.')
    metadata, state = selected['metadata'], selected['model']
    if not any(key.startswith('vlm_bridge.') for key in state):
        raise ValueError('Checkpoint is missing the trained visual bridge.')
    config = v5['Config']
    for key, value in metadata.get('experiment', {}).items():
        setattr(config, key, value)
    model = v5['MagAgentU2Net'](config.IN_CHANNELS, config.OUT_CHANNELS)
    v5['check_checkpoint']({'format_version': 2, 'metadata': metadata}, model)
    model.load_state_dict(state, strict=True)
    print(f"Loaded validation-best epoch: {metadata.get('epoch', 'not recorded')}")
    print('Saved validation loss:', metadata.get('validation', {}).get('loss', 'not recorded'))
    return model.eval(), metadata


def main():
    # EDIT THESE THREE PATHS. Use last_training.pth if best_model.pth is missing.
    source = Path('/kaggle/input/YOUR-CODE-DATASET/V5-heavy-model.py')
    checkpoint = Path('/kaggle/input/YOUR-SAVED-OUTPUTS/best_model.pth')
    test_dir = Path('/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images')

    if not source.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f'Check source/checkpoint paths: {source}, {checkpoint}')
    test_paths = sorted(str(path) for path in test_dir.glob('*.jpeg'))
    if len(test_paths) != 180:
        raise ValueError(f'Expected 180 test images, found {len(test_paths)} in {test_dir}')
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a GPU accelerator for this inference run.')

    # A non-__main__ name imports definitions WITHOUT calling the training pipeline.
    v5 = runpy.run_path(str(source), run_name='v5_inference')
    config = v5['Config']
    config.GPU, config.DEVICE = 0, 'cuda:0'
    config.MHA_FASTPATH, config.DEBUG_SYNC = False, False
    config.EVAL_AMP, config.EVAL_SERIAL = True, False
    config.BRIDGE_AMP, config.SDP_BACKEND = False, 'math'
    config.CUDNN, config.CUDNN_BENCHMARK = True, False
    config.USE_DP = torch.cuda.device_count() > 1
    config.OUT_DIR = '/kaggle/working'
    v5['seed_everything'](config.SEED)
    model, metadata = load_saved_best(v5, checkpoint)
    model = model.to(config.DEVICE)
    if config.USE_DP:
        model = torch.nn.DataParallel(model)
    model.eval()
    v5['generate_submission'](model, test_paths, '/kaggle/working/submission.csv')
    print('DONE: /kaggle/working/submission.csv (no training was run)')


if __name__ == '__main__':
    main()
