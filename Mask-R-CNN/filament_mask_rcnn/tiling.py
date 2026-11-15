"""Optional native-resolution tiled training and inference for Mask R-CNN."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TilingSettings:
    tile_size: int = 1024
    overlap: float = 0.25
    full_image_prob: float = 0.25
    merge_iou: float = 0.5
    merge_ios: float = 0.8
    edge_margin: int = 8

    def __post_init__(self) -> None:
        tile_starts(1, self.tile_size, self.overlap)
        if not 0 <= self.full_image_prob <= 1:
            raise ValueError("full_image_prob must be between 0 and 1")
        if not 0 <= self.merge_iou <= 1 or not 0 <= self.merge_ios <= 1:
            raise ValueError("merge thresholds must be between 0 and 1")

    def as_dict(self) -> dict:
        return asdict(self)


def tile_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    """Return overlapping starts that cover both image edges."""
    if length <= 0 or tile_size <= 0 or not 0 <= overlap < 1:
        raise ValueError("require length/tile_size > 0 and overlap in [0, 1)")
    if length <= tile_size:
        return [0]
    step = max(1, round(tile_size * (1 - overlap)))
    starts = list(range(0, length - tile_size + 1, step))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def mask_match(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Return mask IoU and intersection over the smaller mask."""
    intersection = int(np.count_nonzero(first & second))
    if not intersection:
        return 0.0, 0.0
    first_area, second_area = np.count_nonzero(first), np.count_nonzero(second)
    return (
        intersection / (first_area + second_area - intersection),
        intersection / min(first_area, second_area),
    )


def merge_tile_candidates(candidates: list[dict], settings: TilingSettings) -> list[dict]:
    """Greedily union likely duplicate masks seen in different views."""
    pending = sorted(candidates, key=lambda item: (item["cut"], -item["score"]))
    merged: list[dict] = []
    while pending:
        representative = pending.pop(0)
        union = representative["mask"].copy()
        sources = {representative["source"]}
        remaining = []
        for candidate in pending:
            iou, ios = mask_match(representative["mask"], candidate["mask"])
            is_duplicate = iou >= settings.merge_iou or ios >= settings.merge_ios
            if candidate["source"] not in sources and is_duplicate:
                union |= candidate["mask"]
                sources.add(candidate["source"])
            else:
                remaining.append(candidate)
        merged.append({**representative, "mask": union})
        pending = remaining
    return sorted(merged, key=lambda item: -item["score"])


class GridTileMapper:
    """Sample one overlapping native-resolution tile, or a baseline full image."""

    def __init__(self, cfg, settings: TilingSettings):
        from detectron2.data import DatasetMapper

        if cfg.MODEL.KEYPOINT_ON or cfg.MODEL.LOAD_PROPOSALS:
            raise ValueError("tiled mapper supports instance masks, not keypoints or proposals")
        self.settings = settings
        self.image_format = cfg.INPUT.FORMAT
        self.flip = cfg.INPUT.RANDOM_FLIP
        self.full_mapper = DatasetMapper(cfg, is_train=True)

    def __call__(self, record: dict) -> dict:
        if np.random.random() < self.settings.full_image_prob:
            return self.full_mapper(record)

        from detectron2.data import detection_utils as utils
        from detectron2.data import transforms as transforms
        from detectron2.structures import BitMasks, Instances
        import torch

        record = copy.deepcopy(record)
        image = utils.read_image(record["file_name"], format=self.image_format)
        utils.check_image_size(record, image)
        height, width = image.shape[:2]
        x = int(np.random.choice(tile_starts(width, self.settings.tile_size, self.settings.overlap)))
        y = int(np.random.choice(tile_starts(height, self.settings.tile_size, self.settings.overlap)))
        crop_width = min(self.settings.tile_size, width - x)
        crop_height = min(self.settings.tile_size, height - y)
        applied = [transforms.CropTransform(x, y, crop_width, crop_height)]
        if self.flip != "none" and np.random.random() < 0.5:
            applied.append(
                transforms.HFlipTransform(crop_width)
                if self.flip == "horizontal"
                else transforms.VFlipTransform(crop_height)
            )
        applied = transforms.TransformList(applied)
        crop = applied.apply_image(image)
        annotations = [annotation for annotation in record.pop("annotations", []) if not annotation.get("iscrowd", 0)]
        if any("segmentation" not in annotation for annotation in annotations):
            raise ValueError("every training instance must have a segmentation")
        original = utils.annotations_to_instances(annotations, (height, width), mask_format="bitmask")
        if len(original):
            masks = np.stack(
                [applied.apply_segmentation(mask.numpy().astype(np.uint8)) for mask in original.gt_masks.tensor]
            ).astype(bool)
        else:
            masks = np.zeros((0, crop_height, crop_width), dtype=bool)
        instances = Instances((crop_height, crop_width))
        instances.gt_classes = original.gt_classes
        instances.gt_masks = BitMasks(torch.from_numpy(np.ascontiguousarray(masks)))
        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        record["image"] = torch.as_tensor(np.ascontiguousarray(crop.transpose(2, 0, 1)))
        record["instances"] = utils.filter_empty_instances(instances)
        record["height"], record["width"] = crop_height, crop_width
        return record


def build_tiling_trainer(cfg, settings: TilingSettings):
    """Build DefaultTrainer with only its training mapper changed."""
    from detectron2.data import build_detection_train_loader
    from detectron2.engine import DefaultTrainer

    class TileTrainer(DefaultTrainer):
        @classmethod
        def build_train_loader(cls, loader_cfg):
            return build_detection_train_loader(loader_cfg, mapper=GridTileMapper(loader_cfg, settings))

    return TileTrainer(cfg)


def check_training_batch(trainer) -> None:
    """Run one real finite-loss check before a long GPU training job."""
    import torch

    batch = next(iter(trainer.data_loader))
    for sample in batch:
        height, width = sample["image"].shape[-2:]
        instances = sample["instances"]
        assert instances.image_size == (height, width)
        if len(instances):
            assert tuple(instances.gt_masks.tensor.shape[-2:]) == (height, width)
    was_training = trainer.model.training
    trainer.model.train()
    try:
        with torch.no_grad():
            losses = trainer.model(batch)
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise RuntimeError(f"non-finite preflight loss: {losses}")
    finally:
        trainer.model.train(was_training)


def save_training_preview(cfg, settings: TilingSettings, output_path) -> None:
    """Save one transformed tile and its visible ground-truth masks for review."""
    import cv2
    from detectron2.data import DatasetCatalog

    records = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
    mapper = GridTileMapper(cfg, settings)
    sample = mapper(records[np.random.randint(len(records))])
    for _ in range(39):
        if len(sample["instances"]):
            break
        sample = mapper(records[np.random.randint(len(records))])
    image = sample["image"].permute(1, 2, 0).numpy().copy()
    masks = sample["instances"].gt_masks.tensor.numpy()
    if len(masks):
        image[masks.any(axis=0)] = (0.4 * image[masks.any(axis=0)] + 0.6 * np.array([0, 255, 0])).astype(np.uint8)
    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"could not write preview: {output_path}")


class TiledFilamentPredictor:
    """Full-view plus native tile inference, reconciled in original coordinates."""

    def __init__(self, base_predictor, settings: TilingSettings):
        self.base = base_predictor
        self.settings = settings
        self.last_stats: dict[str, int] = {}

    def _predict_native_tile(self, bgr_tile: np.ndarray) -> dict:
        import torch

        image = bgr_tile[:, :, ::-1] if self.base.input_format == "RGB" else bgr_tile
        tensor = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)), dtype=torch.float32)
        height, width = bgr_tile.shape[:2]
        with torch.inference_mode():
            return self.base.model([{"image": tensor, "height": height, "width": width}])[0]

    def __call__(self, bgr_image: np.ndarray) -> dict:
        import torch
        from detectron2.structures import Boxes, Instances

        height, width = bgr_image.shape[:2]
        candidates: list[dict] = []

        def collect(output: dict, x: int, y: int, view_width: int, view_height: int, source: object) -> None:
            instances = output["instances"].to("cpu")
            margin = max(1, min(self.settings.edge_margin, view_height, view_width))
            for mask, score, class_id in zip(
                instances.pred_masks.numpy(), instances.scores.numpy(), instances.pred_classes.numpy()
            ):
                if mask.shape != (view_height, view_width):
                    raise ValueError("unexpected mask shape; refusing silent rescaling")
                if not mask.any():
                    continue
                cut = (
                    (x > 0 and mask[:, :margin].any())
                    or (y > 0 and mask[:margin, :].any())
                    or (x + view_width < width and mask[:, -margin:].any())
                    or (y + view_height < height and mask[-margin:, :].any())
                )
                canvas = np.zeros((height, width), dtype=bool)
                canvas[y : y + view_height, x : x + view_width] = mask
                candidates.append(
                    {"mask": canvas, "score": float(score), "class_id": int(class_id), "source": source, "cut": cut}
                )

        collect(self.base(bgr_image), 0, 0, width, height, "full")
        tile_count = 0
        for y in tile_starts(height, self.settings.tile_size, self.settings.overlap):
            for x in tile_starts(width, self.settings.tile_size, self.settings.overlap):
                tile = bgr_image[y : y + self.settings.tile_size, x : x + self.settings.tile_size]
                tile_height, tile_width = tile.shape[:2]
                collect(self._predict_native_tile(tile), x, y, tile_width, tile_height, (x, y))
                tile_count += 1

        merged = merge_tile_candidates(candidates, self.settings)
        masks = np.stack([item["mask"] for item in merged]) if merged else np.zeros((0, height, width), dtype=bool)
        result = Instances((height, width))
        result.pred_masks = torch.from_numpy(masks)
        result.scores = torch.tensor([item["score"] for item in merged], dtype=torch.float32)
        result.pred_classes = torch.tensor([item["class_id"] for item in merged], dtype=torch.int64)
        boxes = []
        for mask in masks:
            ys, xs = np.where(mask)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        result.pred_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4))
        self.last_stats = {"tiles": tile_count, "raw_instances": len(candidates), "merged_instances": len(merged)}
        return {"instances": result}


def build_tiled_predictor(cfg, settings: TilingSettings):
    """Wrap one standard Detectron2 predictor; model weights are loaded once."""
    from detectron2.engine import DefaultPredictor

    return TiledFilamentPredictor(DefaultPredictor(cfg), settings)
