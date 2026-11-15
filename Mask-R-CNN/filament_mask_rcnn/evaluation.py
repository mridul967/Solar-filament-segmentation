"""Validation evaluation for saved Detectron2 runs."""

from __future__ import annotations

from pathlib import Path


def evaluate(cfg, dataset_name: str, output_dir: str | Path) -> dict:
    """Evaluate cfg.MODEL.WEIGHTS on a registered COCO dataset."""
    from detectron2.data import build_detection_test_loader
    from detectron2.engine import DefaultPredictor
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator(dataset_name, output_dir=str(output_dir), use_fast_impl=False)
    loader = build_detection_test_loader(cfg, dataset_name)
    return inference_on_dataset(predictor.model, loader, evaluator)
