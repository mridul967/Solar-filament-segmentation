"""Validation evaluation for saved Detectron2 runs."""

from __future__ import annotations

from pathlib import Path


def evaluate(cfg, dataset_name: str, output_dir: str | Path, *, tiling=None) -> dict:
    """Evaluate a full-image or tiled predictor on a registered COCO dataset."""
    from detectron2.data import build_detection_test_loader
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = COCOEvaluator(dataset_name, output_dir=str(output_dir), use_fast_impl=False)
    if tiling is not None:
        import cv2
        from detectron2.data import DatasetCatalog
        from .inference import build_predictor

        predictor = build_predictor(cfg, tiling=tiling)
        evaluator.reset()
        for record in DatasetCatalog.get(dataset_name):
            image = cv2.imread(record["file_name"])
            if image is None:
                raise ValueError(f"could not read validation image: {record['file_name']}")
            evaluator.process([record], [predictor(image)])
        return evaluator.evaluate()

    from detectron2.engine import DefaultPredictor

    predictor = DefaultPredictor(cfg)
    loader = build_detection_test_loader(cfg, dataset_name)
    return inference_on_dataset(predictor.model, loader, evaluator)
