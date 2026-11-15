"""Checkpoint inference and competition CSV creation."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import numpy as np

Postprocessor = Callable[[np.ndarray, np.ndarray, np.ndarray], list[np.ndarray]]
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def build_predictor(cfg, *, tiling=None):
    """Build either the standard full-image predictor or tiled wrapper."""
    if tiling is None:
        from detectron2.engine import DefaultPredictor

        return DefaultPredictor(cfg)
    from .tiling import build_tiled_predictor

    return build_tiled_predictor(cfg, tiling)


def encode_rle(binary_mask: np.ndarray) -> str:
    """Encode one boolean mask using COCO's compressed Fortran-order RLE."""
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return rle["counts"].decode("utf-8")


def _image_paths(image_dir: str | Path) -> list[Path]:
    return sorted(
        path for path in Path(image_dir).iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_submission(
    predictor,
    image_dir: str | Path,
    output_csv: str | Path,
    *,
    postprocessor: Postprocessor | None = None,
) -> Path:
    """Run predictions and write exactly filament_id, segmentation_rle."""
    import cv2

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str]] = []

    for image_path in _image_paths(image_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"could not read image: {image_path}")
        instances = predictor(image)["instances"].to("cpu")
        if not len(instances) or not instances.has("pred_masks"):
            continue
        masks = instances.pred_masks.numpy()
        scores = instances.scores.numpy()
        selected_masks = postprocessor(image, masks, scores) if postprocessor else list(masks.astype(bool))
        image_id = image_path.stem
        for index, mask in enumerate(selected_masks, start=1):
            rows.append((f"{image_id}_{index}", encode_rle(mask)))

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("filament_id", "segmentation_rle"))
        writer.writerows(rows)
    return output_csv


def write_submission_pair(
    predictor,
    image_dir: str | Path,
    raw_output_csv: str | Path,
    postprocessed_output_csv: str | Path,
    postprocessor: Postprocessor,
) -> tuple[Path, Path]:
    """Write raw and processed submissions from one predictor pass per image."""
    import cv2

    raw_rows: list[tuple[str, str]] = []
    processed_rows: list[tuple[str, str]] = []
    for image_path in _image_paths(image_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"could not read image: {image_path}")
        instances = predictor(image)["instances"].to("cpu")
        if not len(instances) or not instances.has("pred_masks"):
            continue
        masks = instances.pred_masks.numpy()
        scores = instances.scores.numpy()
        image_id = image_path.stem
        raw_rows.extend(
            (f"{image_id}_{index}", encode_rle(mask))
            for index, mask in enumerate(masks.astype(bool), start=1)
        )
        processed_rows.extend(
            (f"{image_id}_{index}", encode_rle(mask))
            for index, mask in enumerate(postprocessor(image, masks, scores), start=1)
        )

    outputs = (Path(raw_output_csv), Path(postprocessed_output_csv))
    for output_csv, rows in zip(outputs, (raw_rows, processed_rows), strict=True):
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("filament_id", "segmentation_rle"))
            writer.writerows(rows)
    return outputs
