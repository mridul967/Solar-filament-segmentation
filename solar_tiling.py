"""Modular tiled training/inference for the uploaded Detectron2 notebook.
Load once using: %run /content/solar_tiling.py
No training or prediction starts when this module is loaded.
"""
import copy
import json
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from pycocotools import mask as mask_utils
from detectron2.data import (DatasetMapper, DatasetCatalog,
                            build_detection_train_loader, detection_utils as utils)
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.structures import Instances, Boxes, BitMasks

def tile_starts(length, tile_size, overlap):
    """Full coverage, including a full-sized tile aligned to the last edge."""
    if tile_size <= 0 or not 0 <= overlap < 1:
        raise ValueError("Require tile_size > 0 and 0 <= overlap < 1")
    if length <= tile_size:
        return [0]
    step = max(1, round(tile_size * (1 - overlap)))
    starts = list(range(0, length - tile_size + 1, step))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def mask_match(a, b):
    """Pixel-mask IoU and intersection-over-smaller (not bounding-box IoU)."""
    intersection = np.count_nonzero(a & b)
    if intersection == 0:
        return 0.0, 0.0
    area_a, area_b = np.count_nonzero(a), np.count_nonzero(b)
    return (intersection / (area_a + area_b - intersection),
            intersection / min(area_a, area_b))


def merge_tile_candidates(candidates, iou_threshold=0.5, ios_threshold=0.8):
    """Greedy, class-agnostic instance union for this class-agnostic benchmark.

    Only different views can merge. Compare against the ORIGINAL representative,
    not its growing union, to limit transitive chains. This is a heuristic:
    containment can still over-merge an erroneous broad prediction.
    """
    pending = sorted(candidates, key=lambda c: (c['cut'], -c['score']))
    merged = []
    while pending:
        representative = pending.pop(0)
        union = representative['mask'].copy()
        sources = {representative['source']}
        remaining = []
        for candidate in pending:
            iou, ios = mask_match(representative['mask'], candidate['mask'])
            if (candidate['source'] not in sources and
                    (iou >= iou_threshold or ios >= ios_threshold)):
                union |= candidate['mask']
                sources.add(candidate['source'])
            else:
                remaining.append(candidate)
        merged.append(dict(representative, mask=union))
        pending = remaining
    return sorted(merged, key=lambda c: -c['score'])


class TiledFilamentPredictor:
    def __init__(self, base_predictor, tile_size=1024, overlap=0.25,
                 enabled=True, include_full_image=True,
                 iou_threshold=0.5, ios_threshold=0.8, edge_margin=8):
        tile_starts(1, tile_size, overlap)  # Validate configuration immediately.
        if not include_full_image:
            raise ValueError("Keep full-image context for the first experiment")
        self.base = base_predictor
        self.tile_size = tile_size
        self.overlap = overlap
        self.enabled = enabled
        self.include_full_image = include_full_image
        self.iou_threshold = iou_threshold
        self.ios_threshold = ios_threshold
        self.edge_margin = edge_margin
        self.last_stats = {}

    def _native_tile_prediction(self, bgr_crop):
        # Bypass DefaultPredictor's resizing ONLY for tiles. Reuse its model,
        # input format, internal normalization, weights and thresholds.
        image = bgr_crop
        if self.base.input_format == 'RGB':
            image = image[:, :, ::-1]
        tensor = torch.as_tensor(np.ascontiguousarray(
            image.transpose(2, 0, 1)), dtype=torch.float32)
        h, w = bgr_crop.shape[:2]
        with torch.inference_mode():
            return self.base.model([{'image': tensor, 'height': h, 'width': w}])[0]

    def __call__(self, bgr_image):
        if not self.enabled:
            return self.base(bgr_image)  # Exact original predictor path.
        h, w = bgr_image.shape[:2]
        candidates = []

        def collect(output, x, y, crop_w, crop_h, source):
            instances = output['instances'].to('cpu')
            if len(instances) == 0:
                return
            masks = instances.pred_masks.numpy()
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            m = max(1, min(self.edge_margin, crop_h, crop_w))
            for mask, score, class_id in zip(masks, scores, classes):
                if mask.shape != (crop_h, crop_w):
                    raise ValueError('Unexpected mask shape; refusing silent rescaling')
                if not mask.any():
                    continue
                # Penalize masks touching artificial crop boundaries only.
                cut = ((x > 0 and mask[:, :m].any()) or
                       (y > 0 and mask[:m, :].any()) or
                       (x + crop_w < w and mask[:, -m:].any()) or
                       (y + crop_h < h and mask[-m:, :].any()))
                canvas = np.zeros((h, w), dtype=bool)
                canvas[y:y + crop_h, x:x + crop_w] = mask
                candidates.append({'mask': canvas, 'score': float(score),
                                   'class_id': int(class_id), 'source': source,
                                   'cut': bool(cut)})

        collect(self.base(bgr_image), 0, 0, w, h, 'full')
        xs = tile_starts(w, self.tile_size, self.overlap)
        ys = tile_starts(h, self.tile_size, self.overlap)
        tile_count = 0
        for y in ys:
            for x in xs:
                crop = bgr_image[y:y + self.tile_size, x:x + self.tile_size]
                ch, cw = crop.shape[:2]
                collect(self._native_tile_prediction(crop), x, y, cw, ch,
                        ('tile', x, y))
                tile_count += 1

        merged = merge_tile_candidates(candidates, self.iou_threshold,
                                       self.ios_threshold)
        result = Instances((h, w))
        mask_array = (np.stack([c['mask'] for c in merged]) if merged
                      else np.zeros((0, h, w), dtype=bool))
        result.pred_masks = torch.from_numpy(mask_array)
        result.scores = torch.tensor([c['score'] for c in merged], dtype=torch.float32)
        result.pred_classes = torch.tensor([c['class_id'] for c in merged], dtype=torch.int64)
        boxes = []
        for mask in mask_array:
            yy, xx = np.where(mask)
            boxes.append([xx.min(), yy.min(), xx.max() + 1, yy.max() + 1])
        result.pred_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4))
        self.last_stats = {'tiles': tile_count, 'raw_instances': len(candidates),
                           'merged_instances': len(merged)}
        return {'instances': result}



class GridTileMapper:
    """One randomly sampled overlapping-grid crop per sampled dataset record.

    Rasterize original masks BEFORE cropping: polygon and COCO RLE supported.
    Keep every nonempty visible instance, including truncated objects. Dropping
    visible fragments without ignore labels would incorrectly teach background.
    No positive-only resampling; empty crops remain useful negative examples.
    """
    def __init__(self, cfg, tile_size=1024, overlap=0.25, full_image_prob=0.25):
        tile_starts(1, tile_size, overlap)
        if not 0 <= full_image_prob <= 1:
            raise ValueError('full_image_prob must be in [0, 1]')
        self.tile_size = tile_size
        self.overlap = overlap
        self.full_image_prob = full_image_prob
        self.image_format = cfg.INPUT.FORMAT
        self.flip = cfg.INPUT.RANDOM_FLIP
        self.full_mapper = DatasetMapper(cfg, is_train=True)
        if cfg.MODEL.KEYPOINT_ON or cfg.MODEL.LOAD_PROPOSALS:
            raise ValueError('This mapper is for instance masks without keypoints/precomputed proposals')

    def __call__(self, record):
        if np.random.random() < self.full_image_prob:
            return self.full_mapper(record)
        record = copy.deepcopy(record)
        image = utils.read_image(record['file_name'], format=self.image_format)
        utils.check_image_size(record, image)
        h, w = image.shape[:2]
        x = int(np.random.choice(tile_starts(w, self.tile_size, self.overlap)))
        y = int(np.random.choice(tile_starts(h, self.tile_size, self.overlap)))
        cw, ch = min(w, self.tile_size), min(h, self.tile_size)
        transforms = [T.CropTransform(x, y, cw, ch)]
        if self.flip != 'none' and np.random.random() < 0.5:
            transforms.append(T.HFlipTransform(cw) if self.flip == 'horizontal'
                              else T.VFlipTransform(ch))
        transforms = T.TransformList(transforms)
        crop = transforms.apply_image(image)
        annotations = [a for a in record.pop('annotations', [])
                       if not a.get('iscrowd', 0)]
        if any('segmentation' not in a for a in annotations):
            raise ValueError('Every training instance must have a segmentation')
        original = utils.annotations_to_instances(annotations, (h, w), mask_format='bitmask')
        if len(original):
            cropped = np.stack([transforms.apply_segmentation(m.numpy().astype(np.uint8))
                                for m in original.gt_masks.tensor]).astype(bool)
        else:
            cropped = np.zeros((0, ch, cw), dtype=bool)
        instances = Instances((ch, cw))
        instances.gt_classes = original.gt_classes
        instances.gt_masks = BitMasks(torch.from_numpy(np.ascontiguousarray(cropped)))
        instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        instances = utils.filter_empty_instances(instances)
        record['image'] = torch.as_tensor(np.ascontiguousarray(crop.transpose(2, 0, 1)))
        record['instances'] = instances
        record['height'], record['width'] = ch, cw
        # Diagnostic metadata only; Detectron2 ignores these extra keys.
        record['tile_xyxy'] = (x, y, x + cw, y + ch)
        return record


def build_tiling_trainer(cfg, enabled=True, tile_size=1024, overlap=0.25,
                         full_image_prob=0.25):
    if not enabled:
        return DefaultTrainer(cfg)

    class TileTrainer(DefaultTrainer):
        @classmethod
        def build_train_loader(cls, loader_cfg):
            return build_detection_train_loader(loader_cfg, mapper=GridTileMapper(
                loader_cfg, tile_size, overlap, full_image_prob))

    return TileTrainer(cfg)


def check_training_batch(trainer):
    """One real forward pass before spending time on a training run."""
    batch = next(iter(trainer.data_loader))
    for sample in batch:
        h, w = sample['image'].shape[-2:]
        gt = sample['instances']
        assert gt.image_size == (h, w)
        if len(gt):
            assert tuple(gt.gt_masks.tensor.shape[-2:]) == (h, w)
            assert (gt.gt_masks.tensor.flatten(1).sum(1) > 0).all()
    was_training = trainer.model.training
    trainer.model.train()
    try:
        with torch.no_grad():
            losses = trainer.model(batch)
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise RuntimeError(f'Non-finite preflight loss: {losses}')
        print('Forward-pass check:', {k: float(v.detach().cpu())
                                      for k, v in losses.items()})
    finally:
        trainer.model.train(was_training)


def prepare_tiling_config(base_cfg, output_dir, enabled=True, tile_size=1024,
                          overlap=0.25, full_image_prob=0.25):
    """Clone the config: do not mutate the caller's baseline or its architecture."""
    cfg = base_cfg.clone()
    cfg.defrost()
    cfg.OUTPUT_DIR = str(output_dir)
    if (Path(output_dir) / 'model_final.pth').exists():
        raise FileExistsError('Choose a fresh output directory to protect the previous run')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Save provenance and split record identities, not only network settings.
    manifest = dict(train_tiling=enabled, tile_size=tile_size, overlap=overlap,
                    full_image_prob=full_image_prob, max_iter=cfg.SOLVER.MAX_ITER,
                    batch_size=cfg.SOLVER.IMS_PER_BATCH,
                    initial_weights=cfg.MODEL.WEIGHTS,
                    training_records=[dict(image_id=d['image_id'], file_name=d['file_name'])
                                      for name in cfg.DATASETS.TRAIN
                                      for d in DatasetCatalog.get(name)],
                    validation_records=[dict(image_id=d['image_id'], file_name=d['file_name'])
                                        for name in cfg.DATASETS.TEST
                                        for d in DatasetCatalog.get(name)])
    with open(Path(output_dir) / 'tiling_settings.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    with open(Path(output_dir) / 'config.yaml', 'w') as f:
        f.write(cfg.dump())
    return cfg


def preview_training_crop(cfg, tile_size=1024, overlap=0.25):
    """Run before allocating a trainer/GPU: render one foreground crop and GT."""
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    records = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
    mapper = GridTileMapper(cfg, tile_size, overlap, full_image_prob=0.0)
    # Try several random views; no effect on the actual training sampling policy.
    for _ in range(40):
        sample = mapper(records[np.random.randint(len(records))])
        if len(sample['instances']):
            break
    image = sample['image'].permute(1, 2, 0).numpy()
    rgb = image[:, :, ::-1] if cfg.INPUT.FORMAT == 'BGR' else image
    gt = sample['instances']
    vis = Visualizer(rgb, metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    result = vis.overlay_instances(boxes=gt.gt_boxes,
                                  masks=gt.gt_masks.tensor.numpy(),
                                  labels=[str(int(v)) for v in gt.gt_classes])
    print('Crop:', sample['tile_xyxy'], 'tensor:', tuple(sample['image'].shape),
          'visible instances:', len(gt))
    try:
        from google.colab.patches import cv2_imshow
        cv2_imshow(result.get_image()[:, :, ::-1])
    except ImportError:
        from IPython.display import display
        from PIL import Image
        display(Image.fromarray(result.get_image()))
    return sample


def make_tiling_predictor(train_cfg, enabled=True, tile_size=1024, overlap=0.25):
    infer_cfg = train_cfg.clone()
    infer_cfg.defrost()
    infer_cfg.MODEL.WEIGHTS = str(Path(train_cfg.OUTPUT_DIR) / 'model_final.pth')
    if not Path(infer_cfg.MODEL.WEIGHTS).is_file():
        raise FileNotFoundError(infer_cfg.MODEL.WEIGHTS)
    base = DefaultPredictor(infer_cfg)
    predictor = TiledFilamentPredictor(base, enabled=enabled, tile_size=tile_size,
                                       overlap=overlap, iou_threshold=0.5,
                                       ios_threshold=0.8)
    predictor._filament_tiling_wrapper = True
    return predictor


def solar_disk_mask(gray, margin=18):
    # Matches the solar postprocessing in the supplied notebook.
    _, threshold = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(gray, dtype=bool)
    (x, y), radius = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    disk = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(disk, (int(x), int(y)), max(0, int(radius - margin)), 255, -1)
    return disk > 0


def generate_solar_submission(predictor, test_dir, output_csv,
                              min_area=200, max_disjoint_blobs=3):
    """Same solar cleanup: disk clip, largest component, score-priority exclusion.
    Accept every common extension, fail on unreadable/wrong-sized images.
    """
    paths = sorted(p for p in Path(test_dir).iterdir()
                   if p.suffix.lower() in {'.jpeg', '.jpg', '.png'})
    if not paths:
        raise FileNotFoundError(f'No test images found in {test_dir}')
    rows, diagnostics = [], []
    for path in tqdm(paths, desc='Solar inference'):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot read {path}')
        h, w = image.shape[:2]
        if (h, w) != (2048, 2048):
            raise ValueError(f'{path.name}: expected 2048x2048, got {(h, w)}')
        disk = solar_disk_mask(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        instances = predictor(image)['instances'].to('cpu')
        occupied = np.zeros((h, w), dtype=bool)
        count = 0
        for index in np.argsort(-instances.scores.numpy(), kind='stable'):
            mask = instances.pred_masks[index].numpy().astype(np.uint8) & disk
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if n <= 1 or n - 1 > max_disjoint_blobs:
                continue
            clean = labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            exclusive = clean & ~occupied
            if exclusive.sum() < min_area:
                continue
            occupied |= exclusive
            count += 1
            rle = mask_utils.encode(np.asfortranarray(exclusive.astype(np.uint8)))
            rows.append(dict(filament_id=f'{path.stem}_{count}',
                             segmentation_rle=rle['counts'].decode('utf-8')))
        diagnostics.append(dict(image=path.name, raw=len(instances), kept=count))
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=['filament_id', 'segmentation_rle'])
    if frame['filament_id'].duplicated().any():
        raise ValueError('Duplicate filament IDs; check duplicate image stems')
    frame.to_csv(output_csv, index=False)
    pd.DataFrame(diagnostics).to_csv(output_csv.with_suffix('.diagnostics.csv'), index=False)
    with open(output_csv.with_suffix('.settings.json'), 'w') as f:
        json.dump(dict(checkpoint=predictor.base.cfg.MODEL.WEIGHTS,
                       tiled_inference=predictor.enabled,
                       tile_size=predictor.tile_size, overlap=predictor.overlap,
                       merge_iou=predictor.iou_threshold, merge_ios=predictor.ios_threshold,
                       score_threshold=predictor.base.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
                       nms_threshold=predictor.base.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
                       min_area=min_area, max_disjoint_blobs=max_disjoint_blobs), f, indent=2)
    print(f'Saved {output_csv}: {len(frame)} instances across {len(paths)} images')
    return frame
