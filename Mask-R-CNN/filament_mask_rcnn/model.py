"""The one canonical vanilla Detectron2 configuration."""

from __future__ import annotations

from pathlib import Path


BASE_CONFIG = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"


def build_vanilla_cfg(
    train_dataset: str,
    val_dataset: str,
    output_dir: str | Path,
    num_classes: int,
    *,
    score_threshold: float = 0.20,
    device: str = "cuda",
):
    """Reproduce the audited vanilla Mask R-CNN architecture and solver values."""
    from detectron2 import model_zoo
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.DATASETS.TRAIN = (train_dataset,)
    cfg.DATASETS.TEST = (val_dataset,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(BASE_CONFIG)
    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 28
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.2, 0.5, 1.0, 2.0, 5.0]]
    cfg.INPUT.MIN_SIZE_TRAIN = (1024, 1200)
    cfg.INPUT.MAX_SIZE_TRAIN = 1344
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.WARMUP_ITERS = 300
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = (3500, 4500)
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = str(output_dir)
    return cfg


def load_run_cfg(
    config_path: str | Path,
    weights_path: str | Path,
    *,
    score_threshold: float = 0.25,
    device: str = "cuda",
):
    """Load a saved run configuration without relying on previous notebook state."""
    from detectron2.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
    cfg.MODEL.DEVICE = device
    return cfg
