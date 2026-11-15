"""COCO preparation and Detectron2 registration."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SplitPaths:
    train_json: Path
    val_json: Path
    classes: list[str]


def _load_coco(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        coco = json.load(handle)
    required = {"images", "annotations", "categories"}
    missing = required - coco.keys()
    if missing:
        raise ValueError(f"COCO file is missing keys: {sorted(missing)}")
    if not coco["images"] or not coco["categories"]:
        raise ValueError("COCO file must contain images and categories")
    return coco


def _normalise_annotation_ids(coco: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with integer, unique annotation identifiers."""
    normalised = dict(coco)
    normalised["annotations"] = [
        {**annotation, "id": index}
        for index, annotation in enumerate(coco["annotations"], start=1)
    ]
    return normalised


def _split_payload(coco: dict[str, Any], images: list[dict[str, Any]]) -> dict[str, Any]:
    image_ids = {image["id"] for image in images}
    payload = dict(coco)
    payload["images"] = images
    payload["annotations"] = [
        annotation for annotation in coco["annotations"] if annotation["image_id"] in image_ids
    ]
    return payload


def prepare_grouped_splits(
    coco_json: str | Path,
    split_dir: str | Path,
    *,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> SplitPaths:
    """Split COCO records by physical filename to prevent annotator leakage."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")

    coco = _normalise_annotation_ids(_load_coco(coco_json))
    by_filename: dict[str, list[dict[str, Any]]] = {}
    for image in coco["images"]:
        filename = image.get("file_name")
        if not filename:
            raise ValueError(f"image {image.get('id')} has no file_name")
        by_filename.setdefault(filename, []).append(image)

    filenames = sorted(by_filename)
    random.Random(seed).shuffle(filenames)
    val_start = int(len(filenames) * (1 - val_fraction))
    train_filenames, val_filenames = filenames[:val_start], filenames[val_start:]
    if not train_filenames or not val_filenames:
        raise ValueError("split produced an empty train or validation set")

    train_images = [image for filename in train_filenames for image in by_filename[filename]]
    val_images = [image for filename in val_filenames for image in by_filename[filename]]
    destination = Path(split_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train_json, val_json = destination / "train.json", destination / "val.json"
    train_json.write_text(json.dumps(_split_payload(coco, train_images)), encoding="utf-8")
    val_json.write_text(json.dumps(_split_payload(coco, val_images)), encoding="utf-8")

    classes = [category["name"] for category in sorted(coco["categories"], key=lambda item: item["id"])]
    return SplitPaths(train_json=train_json, val_json=val_json, classes=classes)


def register_splits(
    splits: SplitPaths,
    image_root: str | Path,
    *,
    train_name: str = "filament_train",
    val_name: str = "filament_val",
) -> tuple[str, str, list[str]]:
    """Register fresh Detectron2 datasets and attach ordered class metadata."""
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances

    for name in (train_name, val_name):
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)

    image_root = str(image_root)
    register_coco_instances(train_name, {}, str(splits.train_json), image_root)
    register_coco_instances(val_name, {}, str(splits.val_json), image_root)
    for name in (train_name, val_name):
        MetadataCatalog.get(name).set(thing_classes=splits.classes)
    return train_name, val_name, splits.classes
