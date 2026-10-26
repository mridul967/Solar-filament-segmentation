# V4: Audited instance-aware solar-filament segmentation for Kaggle.
# Run this notebook-style script in Kaggle after attaching the competition data.

!pip install -q pycocotools

import gc
import glob
import json
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


class Config:
    BASE_DIR = "/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"
    TRAIN_DIR = os.path.join(BASE_DIR, "train/train_images")
    TEST_DIR = os.path.join(BASE_DIR, "test/test_images")
    ANNOTATIONS_FILE = os.path.join(BASE_DIR, "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
    OUT_DIR = "/kaggle/working"

    IMG_SIZE = 1024
    ORIGINAL_SIZE = 2048
    SEED = 42
    TRAIN_FRACTION = 0.80
    DEV_FRACTION = 0.10
    AUDIT_FRACTION = 0.10

    BATCH_SIZE = 2
    ACCUM_STEPS = 2
    EPOCHS = 50
    LR = 2e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    LR_PATIENCE = 4
    EARLY_STOPPING_WARMUP = 12
    EARLY_STOPPING_PATIENCE = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    BOUNDARY_WEIGHT = 0.20
    SPINE_WEIGHT = 0.20
    AUX_WARMUP_EPOCHS = 3
    MAX_AUX_POS_WEIGHT = 20.0
    POS_WEIGHT_SAMPLES = 128
    SPINE_VALIDITY_FLOOR = 0.90

    PQ_IOU_THRESHOLD = 0.50
    FOREGROUND_THRESHOLD_CANDIDATES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
    MIN_AREA_CANDIDATES = (50, 100, 200, 300, 500)
    AUX_THRESHOLD_CANDIDATES = (0.35, 0.50, 0.65)
    DEFAULT_FOREGROUND_THRESHOLD = 0.40
    DEFAULT_MIN_AREA = 200
    MIN_SPINE_SEED_PIXELS = 5

    SPLIT_NAME = "v4_group_split.json"
    CHECKPOINT_NAME = "best-v4-instance-aware.pth"
    HISTORY_NAME = "v4-history.csv"
    CACHE_NAME = "v4-dev-logits.npz"
    DECODER_NAME = "v4-decoder-settings.json"
    AUDIT_NAME = "v4-audit-metrics.json"
    SUBMISSION_NAME = "submission-v4-instance-aware.csv"
    RUN_TRAINING = True


@dataclass(frozen=True)
class DecoderSettings:
    method: str = "components"
    foreground_threshold: float = Config.DEFAULT_FOREGROUND_THRESHOLD
    min_area: int = Config.DEFAULT_MIN_AREA
    boundary_threshold: float = 0.50
    spine_threshold: float = 0.50


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


def solar_disk_mask(height, width):
    y, x = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    radius = min(cx, cy) * 0.98
    return (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2


def remove_limb_darkening(image):
    """Stable baseline normalization; radial-profile correction is deferred."""
    disk = solar_disk_mask(*image.shape)
    result = np.zeros_like(image, dtype=np.float32)
    pixels = image[disk]
    low, high = np.percentile(pixels, (1, 99))
    result[disk] = np.clip((pixels - low) / (high - low + 1e-8), 0, 1)
    return result, disk


def get_magnetic_proxy(image):
    image_u8 = np.clip(image, 0, 1).astype(np.float32) * 255
    dx = cv2.Sobel(image_u8, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(image_u8, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(dx ** 2, (15, 15), 5)
    jxy = cv2.GaussianBlur(dx * dy, (15, 15), 5)
    jyy = cv2.GaussianBlur(dy ** 2, (15, 15), 5)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    return (coherence - coherence.min()) / (coherence.max() - coherence.min() + 1e-8)


def prepare_input(image):
    """The only image preprocessing path used by train, validation, and test."""
    corrected, disk = remove_limb_darkening(image)
    proxy = get_magnetic_proxy(corrected)
    proxy[~disk] = 0
    return np.stack([corrected, corrected, proxy], axis=-1).astype(np.float32), disk


def build_physical_records(coco, image_dir):
    records = defaultdict(list)
    for image_id, image in coco.imgs.items():
        path = os.path.join(image_dir, image["file_name"])
        if os.path.exists(path):
            records[image["file_name"]].append(image_id)
    return {name: sorted(ids) for name, ids in records.items()}


def load_or_create_split(physical_records):
    path = os.path.join(Config.OUT_DIR, Config.SPLIT_NAME)
    available = sorted(physical_records)
    if os.path.exists(path):
        with open(path) as handle:
            split = json.load(handle)
        expected = set(available)
        actual = set(split["train"]) | set(split["dev"]) | set(split["audit"])
        if actual != expected:
            raise ValueError("Saved V4 split does not match the available physical images.")
    else:
        rng = random.Random(Config.SEED)
        rng.shuffle(available)
        train_end = int(len(available) * Config.TRAIN_FRACTION)
        dev_end = train_end + int(len(available) * Config.DEV_FRACTION)
        split = {"train": available[:train_end], "dev": available[train_end:dev_end],
                 "audit": available[dev_end:]}
        with open(path, "w") as handle:
            json.dump(split, handle, indent=2)

    sets = {name: set(files) for name, files in split.items()}
    assert not (sets["train"] & sets["dev"] or sets["train"] & sets["audit"] or sets["dev"] & sets["audit"])
    return split


def mask_from_annotation(coco, annotation):
    return (coco.annToMask(annotation) > 0).astype(np.uint8)


def rasterize_spine(annotation, height, width):
    spine = np.asarray(annotation.get("spine", []), dtype=np.float32)
    result = np.zeros((height, width), dtype=np.uint8)
    if spine.ndim != 1 or len(spine) < 4 or len(spine) % 2:
        return result, False
    points = spine.reshape(-1, 2)
    coordinates_valid = np.all((points[:, 0] >= 0) & (points[:, 0] < width) &
                               (points[:, 1] >= 0) & (points[:, 1] < height))
    if not coordinates_valid:
        return result, False
    cv2.polylines(result, [np.rint(points).astype(np.int32)], False, 1, 1)
    return result, True


def build_targets(coco, image_id):
    image = coco.imgs[image_id]
    height, width = image["height"], image["width"]
    disk = solar_disk_mask(height, width).astype(np.uint8)
    foreground = np.zeros((height, width), dtype=np.uint8)
    instance_id = np.zeros((height, width), dtype=np.int32)
    overlap = np.zeros((height, width), dtype=bool)
    boundary = np.zeros((height, width), dtype=np.uint8)
    spine = np.zeros((height, width), dtype=np.uint8)
    spine_valid = disk.copy()

    for index, annotation in enumerate(coco.imgToAnns.get(image_id, []), start=1):
        mask = mask_from_annotation(coco, annotation)
        already_assigned = instance_id > 0
        overlap |= (mask.astype(bool) & already_assigned)
        instance_id[(mask > 0) & ~already_assigned] = index
        foreground |= mask

        eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        boundary |= mask & ~eroded

        spine_mask, coords_valid = rasterize_spine(annotation, height, width)
        dilated_mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        containment = (spine_mask & dilated_mask).sum() / max(spine_mask.sum(), 1)
        valid = coords_valid and containment >= Config.SPINE_VALIDITY_FLOOR
        if valid:
            spine |= spine_mask
        else:
            spine_valid[mask > 0] = 0

    instance_id[overlap] = -1
    boundary[overlap] = 0
    spine[spine_valid == 0] = 0
    categorical = {
        "foreground": foreground,
        "boundary": boundary,
        "spine": spine,
        "disk": disk,
        "spine_valid": spine_valid,
        "instance_id": instance_id,
    }
    return categorical


def resize_target(target, size=Config.IMG_SIZE):
    return cv2.resize(target, (size, size), interpolation=cv2.INTER_NEAREST)


def spatial_augment(image, targets):
    if random.random() < 0.5:
        image = np.flip(image, axis=1)
        targets = {key: np.flip(value, axis=1) for key, value in targets.items()}
    if random.random() < 0.5:
        image = np.flip(image, axis=0)
        targets = {key: np.flip(value, axis=0) for key, value in targets.items()}
    rotations = random.randrange(4)
    if rotations:
        image = np.rot90(image, rotations)
        targets = {key: np.rot90(value, rotations) for key, value in targets.items()}
    return np.ascontiguousarray(image), {key: np.ascontiguousarray(value) for key, value in targets.items()}


class SolarInstanceDataset(Dataset):
    def __init__(self, coco, physical_records, file_names, image_dir, training):
        self.coco = coco
        self.physical_records = physical_records
        self.file_names = list(file_names)
        self.image_dir = image_dir
        self.training = training

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        records = self.physical_records[file_name]
        image_id = random.choice(records) if self.training else records[0]
        image = load_image(os.path.join(self.image_dir, file_name))
        image, _ = prepare_input(image)
        targets = build_targets(self.coco, image_id)

        image = cv2.resize(image, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        targets = {key: resize_target(value) for key, value in targets.items() if key != "instance_id"}
        targets["boundary"] = cv2.dilate(targets["boundary"], np.ones((3, 3), np.uint8), iterations=1)
        if self.training:
            image, targets = spatial_augment(image, targets)

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        output = {key: torch.from_numpy(value).unsqueeze(0).float() for key, value in targets.items()}
        return image, output


def estimate_aux_pos_weights(coco, physical_records, train_files):
    """Bounded estimates prevent sparse targets from dominating BCE gradients."""
    rng = random.Random(Config.SEED)
    files = list(train_files)
    rng.shuffle(files)
    positives = {"boundary": 0.0, "spine": 0.0}
    totals = {"boundary": 0.0, "spine": 0.0}
    for file_name in tqdm(files[:Config.POS_WEIGHT_SAMPLES], desc="Estimating auxiliary weights"):
        targets = build_targets(coco, physical_records[file_name][0])
        for name in positives:
            valid = targets["disk"] if name == "boundary" else targets["spine_valid"]
            positives[name] += float((targets[name] * valid).sum())
            totals[name] += float(valid.sum())
    return {
        name: float(np.clip((totals[name] - positives[name]) / max(positives[name], 1), 1, Config.MAX_AUX_POS_WEIGHT))
        for name in positives
    }


class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1), nn.BatchNorm2d(f_int))
        self.w_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, gate, skip):
        gate = F.interpolate(self.w_g(gate), size=skip.shape[2:], mode="bilinear", align_corners=True)
        return skip * self.psi(F.relu(gate + self.w_x(skip), inplace=True))


class RSU(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, out_ch, 1)
        self.enc1 = nn.Sequential(nn.Conv2d(out_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(mid_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(mid_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(mid_ch * 2, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        residual = self.conv_in(x)
        encoder = self.enc1(residual)
        decoder = self.dec2(self.enc2(self.pool(encoder)))
        decoder = self.dec1(torch.cat([F.interpolate(decoder, size=encoder.shape[2:], mode="bilinear", align_corners=True), encoder], 1))
        return decoder + residual


class CognitiveRefinerFeatures(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.bridge1 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels)
        self.bridge2 = nn.Conv2d(channels, channels, 3, padding=4, dilation=4, groups=channels)
        self.gate = nn.Sequential(nn.Conv2d(channels + 1, channels, 1), nn.Sigmoid())

    def forward(self, features, proxy):
        bridged = features + F.relu(self.bridge1(features)) + F.relu(self.bridge2(features))
        return bridged * self.gate(torch.cat([bridged, proxy], dim=1))


class MagAgentU2NetV4(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.enc1, self.pool1 = RSU(in_ch, 32, 64), nn.MaxPool2d(2)
        self.enc2, self.pool2 = RSU(64, 64, 128), nn.MaxPool2d(2)
        self.enc3, self.pool3 = RSU(128, 128, 256), nn.MaxPool2d(2)
        self.bottleneck = RSU(256, 256, 512)
        self.att3, self.dec3 = AttentionGate(512, 256, 256), RSU(768, 256, 256)
        self.att2, self.dec2 = AttentionGate(256, 128, 128), RSU(384, 128, 128)
        self.att1, self.dec1 = AttentionGate(128, 64, 64), RSU(192, 64, 64)
        self.refiner = CognitiveRefinerFeatures(64)
        self.foreground_head = nn.Conv2d(64, 1, 1)
        self.boundary_head = nn.Conv2d(64, 1, 1)
        self.spine_head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        proxy = x[:, 2:3]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bottleneck = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([F.interpolate(bottleneck, size=e3.shape[2:], mode="bilinear", align_corners=True), self.att3(bottleneck, e3)], 1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=True), self.att2(d3, e2)], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=True), self.att1(d2, e1)], 1))
        features = self.refiner(d1, proxy)
        return {"foreground": self.foreground_head(features), "boundary": self.boundary_head(features), "spine": self.spine_head(features)}


def masked_bce_dice(logits, target, valid, pos_weight=1.0):
    logits, target, valid = logits.float(), target.float(), valid.float()
    weight = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp_min(1.0)
    probs = torch.sigmoid(logits) * valid
    target = target * valid
    dice = 1 - (2 * (probs * target).sum() + 1) / (probs.sum() + target.sum() + 1)
    return bce + dice


class V4Loss(nn.Module):
    def __init__(self, aux_pos_weights):
        super().__init__()
        self.aux_pos_weights = aux_pos_weights

    def forward(self, outputs, targets, epoch):
        foreground = masked_bce_dice(outputs["foreground"], targets["foreground"], targets["disk"])
        boundary = masked_bce_dice(outputs["boundary"], targets["boundary"], targets["disk"], self.aux_pos_weights["boundary"])
        spine = masked_bce_dice(outputs["spine"], targets["spine"], targets["spine_valid"], self.aux_pos_weights["spine"])
        warmup = min(1.0, (epoch + 1) / Config.AUX_WARMUP_EPOCHS)
        total = foreground + warmup * (Config.BOUNDARY_WEIGHT * boundary + Config.SPINE_WEIGHT * spine)
        return total, {"foreground_loss": foreground.detach(), "boundary_loss": boundary.detach(), "spine_loss": spine.detach(), "aux_scale": warmup}


def train_epoch(model, loader, optimizer, criterion, scaler, epoch):
    model.train()
    totals = defaultdict(float)
    optimizer.zero_grad(set_to_none=True)
    remainder = len(loader) % Config.ACCUM_STEPS
    for step, (images, targets) in enumerate(tqdm(loader, desc=f"Train {epoch + 1}")):
        images = images.to(Config.DEVICE, non_blocking=True)
        targets = {key: value.to(Config.DEVICE, non_blocking=True) for key, value in targets.items()}
        divisor = remainder if remainder and step >= len(loader) - remainder else Config.ACCUM_STEPS
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=Config.DEVICE == "cuda"):
            outputs = model(images)
            loss, pieces = criterion(outputs, targets, epoch)
            scaled_loss = loss / divisor
        scaler.scale(scaled_loss).backward()
        if (step + 1) % Config.ACCUM_STEPS == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        totals["train_loss"] += loss.item()
        for key, value in pieces.items():
            totals[key] += float(value)
    return {key: value / len(loader) for key, value in totals.items()}


def predict_logits(model, image_path):
    image = load_image(image_path)
    input_image, _ = prepare_input(image)
    height, width = input_image.shape[:2]
    resized = cv2.resize(input_image, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(Config.DEVICE)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=Config.DEVICE == "cuda"):
        output = model(tensor)
    return {key: F.interpolate(value.float(), size=(height, width), mode="bilinear", align_corners=False).sigmoid().cpu().numpy()[0, 0]
            for key, value in output.items()}


def components_from_probability(foreground, settings):
    binary = (foreground >= settings.foreground_threshold).astype(np.uint8)
    count, labels = cv2.connectedComponents(binary)
    return [(labels == label).astype(np.uint8) for label in range(1, count)
            if (labels == label).sum() >= settings.min_area]


def watershed_from_probability(foreground, boundary, spine, settings):
    base = components_from_probability(foreground, DecoderSettings("components", settings.foreground_threshold, settings.min_area))
    output = []
    for component in base:
        seed_binary = (spine >= settings.spine_threshold).astype(np.uint8) & component
        seed_count, seed_labels = cv2.connectedComponents(seed_binary)
        seeds = [label for label in range(1, seed_count) if (seed_labels == label).sum() >= Config.MIN_SPINE_SEED_PIXELS]
        if len(seeds) < 2:
            output.append(component)
            continue
        y, x = np.where(component)
        y0, y1 = max(0, y.min() - 1), min(component.shape[0], y.max() + 2)
        x0, x1 = max(0, x.min() - 1), min(component.shape[1], x.max() + 2)
        local_component = component[y0:y1, x0:x1]
        markers = np.ones(local_component.shape, dtype=np.int32)
        markers[local_component > 0] = 0
        for marker_id, seed_label in enumerate(seeds, start=2):
            markers[seed_labels[y0:y1, x0:x1] == seed_label] = marker_id
        # High boundary probability becomes a watershed ridge; decoding remains
        # constrained to the original foreground component.
        surface = np.clip((1 - foreground[y0:y1, x0:x1]) + boundary[y0:y1, x0:x1], 0, 1)
        local_image = np.repeat((surface * 255).astype(np.uint8)[..., None], 3, axis=2)
        labels = cv2.watershed(local_image, markers)
        children = []
        for marker_id in range(2, 2 + len(seeds)):
            child = np.zeros_like(component)
            child[y0:y1, x0:x1] = (labels == marker_id) & (local_component > 0)
            if child.sum() >= settings.min_area:
                children.append(child)
        output.extend(children if len(children) >= 2 else [component])
    return output


def decode_instances(probabilities, settings):
    if settings.method == "watershed":
        return watershed_from_probability(probabilities["foreground"], probabilities["boundary"], probabilities["spine"], settings)
    return components_from_probability(probabilities["foreground"], settings)


def pq_metrics(predictions, targets):
    if not predictions and not targets:
        return {"pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "fp": 0, "fn": 0, "splits": 0, "merges": 0}
    overlaps = np.zeros((len(predictions), len(targets)), dtype=np.float32)
    for p, prediction in enumerate(predictions):
        for t, target in enumerate(targets):
            union = np.logical_or(prediction, target).sum()
            overlaps[p, t] = np.logical_and(prediction, target).sum() / max(union, 1)
    candidates = sorted(((overlaps[p, t], p, t) for p in range(len(predictions)) for t in range(len(targets))), reverse=True)
    matched_predictions, matched_targets, matched_iou = set(), set(), []
    for iou, p, t in candidates:
        if iou < Config.PQ_IOU_THRESHOLD or p in matched_predictions or t in matched_targets:
            continue
        matched_predictions.add(p)
        matched_targets.add(t)
        matched_iou.append(iou)
    tp = len(matched_iou)
    fp, fn = len(predictions) - tp, len(targets) - tp
    denominator = tp + 0.5 * fp + 0.5 * fn
    sq = float(np.mean(matched_iou)) if matched_iou else 0.0
    rq = tp / denominator if denominator else 0.0
    nonzero = overlaps > 0
    # Several predictions touching one target is a split; one prediction
    # touching several targets is a merge.
    splits = int(np.maximum(nonzero.sum(axis=0) - 1, 0).sum())
    merges = int(np.maximum(nonzero.sum(axis=1) - 1, 0).sum())
    return {"pq": sq * rq, "sq": sq, "rq": rq, "tp": tp, "fp": fp, "fn": fn,
            "splits": splits, "merges": merges}


def evaluate_files(model, files, physical_records, coco, settings):
    per_pass = []
    model.eval()
    for file_name in tqdm(files, desc="Development PQ"):
        probabilities = predict_logits(model, os.path.join(Config.TRAIN_DIR, file_name))
        predictions = decode_instances(probabilities, settings)
        for image_id in physical_records[file_name]:
            targets = [mask_from_annotation(coco, annotation) for annotation in coco.imgToAnns.get(image_id, [])]
            per_pass.append(pq_metrics(predictions, targets))
    frame = pd.DataFrame(per_pass)
    metrics = {key: float(frame[key].mean()) for key in ("pq", "sq", "rq", "tp", "fp", "fn", "splits", "merges")}
    metrics["pq_median"] = float(frame["pq"].median())
    return metrics


def cache_logits(model, files):
    names, foreground, boundary, spine = [], [], [], []
    for file_name in tqdm(files, desc="Caching development logits"):
        image = load_image(os.path.join(Config.TRAIN_DIR, file_name))
        input_image, _ = prepare_input(image)
        small = cv2.resize(input_image, (Config.IMG_SIZE, Config.IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(small).permute(2, 0, 1).unsqueeze(0).to(Config.DEVICE)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=Config.DEVICE == "cuda"):
            outputs = model(tensor)
        names.append(file_name)
        foreground.append(outputs["foreground"].float().cpu().numpy()[0, 0].astype(np.float16))
        boundary.append(outputs["boundary"].float().cpu().numpy()[0, 0].astype(np.float16))
        spine.append(outputs["spine"].float().cpu().numpy()[0, 0].astype(np.float16))
    path = os.path.join(Config.OUT_DIR, Config.CACHE_NAME)
    np.savez_compressed(path, file_names=np.asarray(names), foreground=np.stack(foreground), boundary=np.stack(boundary), spine=np.stack(spine))
    return path


def calibrate_decoder(cache_path, physical_records, coco):
    cache = np.load(cache_path)
    cached = {name: {key: cache[key][index] for key in ("foreground", "boundary", "spine")}
              for index, name in enumerate(cache["file_names"])}
    best_settings, best_pq = None, -1.0
    candidates = [DecoderSettings("components", threshold, area)
                  for threshold in Config.FOREGROUND_THRESHOLD_CANDIDATES for area in Config.MIN_AREA_CANDIDATES]
    candidates += [DecoderSettings("watershed", threshold, area, boundary, spine)
                   for threshold in Config.FOREGROUND_THRESHOLD_CANDIDATES for area in Config.MIN_AREA_CANDIDATES
                   for boundary in Config.AUX_THRESHOLD_CANDIDATES for spine in Config.AUX_THRESHOLD_CANDIDATES]
    for settings in tqdm(candidates, desc="Calibrating decoder"):
        scores = []
        for file_name, logits in cached.items():
            probabilities = {}
            for key, value in logits.items():
                native_logits = cv2.resize(value.astype(np.float32), (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE),
                                            interpolation=cv2.INTER_LINEAR)
                probabilities[key] = 1 / (1 + np.exp(-np.clip(native_logits, -30, 30)))
            predictions = decode_instances(probabilities, settings)
            for image_id in physical_records[file_name]:
                targets = [mask_from_annotation(coco, ann) for ann in coco.imgToAnns.get(image_id, [])]
                scores.append(pq_metrics(predictions, targets)["pq"])
        score = float(np.mean(scores))
        if score > best_pq:
            best_pq, best_settings = score, settings
    with open(os.path.join(Config.OUT_DIR, Config.DECODER_NAME), "w") as handle:
        json.dump(asdict(best_settings), handle, indent=2)
    return best_settings, best_pq


def pycoco_rle_encode(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("utf-8")


def generate_submission(model, test_paths, settings):
    rows = []
    model.eval()
    for path in tqdm(test_paths, desc="Generating submission"):
        probabilities = predict_logits(model, path)
        for index, mask in enumerate(decode_instances(probabilities, settings), start=1):
            if not rows:
                round_trip = mask_utils.decode(mask_utils.encode(np.asfortranarray(mask.astype(np.uint8))))
                assert np.array_equal(round_trip, mask), "RLE round-trip failed"
            rows.append({"filament_id": f"{os.path.splitext(os.path.basename(path))[0]}_{index}",
                         "segmentation_rle": pycoco_rle_encode(mask)})
    output = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    output.to_csv(os.path.join(Config.OUT_DIR, Config.SUBMISSION_NAME), index=False)
    return output


def assert_pq_sanity():
    target = np.zeros((8, 8), dtype=np.uint8)
    target[2:5, 2:5] = 1
    assert pq_metrics([target], [target])["pq"] == 1.0
    assert pq_metrics([], [target])["fn"] == 1
    assert pq_metrics([target], []) ["fp"] == 1
    left, right = target.copy(), target.copy()
    left[:, 4:] = 0
    right[:, :4] = 0
    assert pq_metrics([left, right], [target])["splits"] == 1
    first, second = target.copy(), target.copy()
    first[:, 5:] = 0
    second[:, :1] = 0
    assert pq_metrics([first | second], [first, second])["merges"] == 1


def save_visual_diagnostics(model, files, physical_records, coco, settings, count=4):
    """Save fixed development examples for manual target/prediction inspection."""
    for index, file_name in enumerate(files[:count]):
        image = load_image(os.path.join(Config.TRAIN_DIR, file_name))
        targets = build_targets(coco, physical_records[file_name][0])
        probabilities = predict_logits(model, os.path.join(Config.TRAIN_DIR, file_name))
        prediction = np.zeros_like(targets["foreground"])
        for mask in decode_instances(probabilities, settings):
            prediction |= mask
        figure, axes = plt.subplots(2, 3, figsize=(15, 9))
        panels = [
            (image, "input"), (targets["foreground"], "GT foreground"), (prediction, "predicted instances"),
            (targets["boundary"], "GT boundary"), (targets["spine"], "GT spine"),
            (probabilities["spine"], "predicted spine probability"),
        ]
        for axis, (panel, title) in zip(axes.ravel(), panels):
            axis.imshow(panel, cmap="gray")
            axis.set_title(title)
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(os.path.join(Config.OUT_DIR, f"v4-diagnostic-{index}.png"), dpi=140)
        plt.close(figure)


def main():
    seed_everything(Config.SEED)
    assert_pq_sanity()
    coco = COCO(Config.ANNOTATIONS_FILE)
    physical_records = build_physical_records(coco, Config.TRAIN_DIR)
    split = load_or_create_split(physical_records)
    print({name: len(files) for name, files in split.items()})

    train_dataset = SolarInstanceDataset(coco, physical_records, split["train"], Config.TRAIN_DIR, training=True)
    loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS,
                        pin_memory=True, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(Config.SEED))
    aux_weights = estimate_aux_pos_weights(coco, physical_records, split["train"])
    print("Auxiliary positive weights:", aux_weights)

    model = MagAgentU2NetV4().to(Config.DEVICE)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=Config.LR_PATIENCE)
    scaler = torch.amp.GradScaler("cuda", enabled=Config.DEVICE == "cuda")
    criterion = V4Loss(aux_weights)
    baseline_settings = DecoderSettings()
    model_path = os.path.join(Config.OUT_DIR, Config.CHECKPOINT_NAME)
    history, best_pq, stale = [], -1.0, 0

    if Config.RUN_TRAINING:
        for epoch in range(Config.EPOCHS):
            train_metrics = train_epoch(model, loader, optimizer, criterion, scaler, epoch)
            dev_metrics = evaluate_files(model, split["dev"], physical_records, coco, baseline_settings)
            scheduler.step(dev_metrics["pq"])
            row = {"epoch": epoch + 1, "learning_rate": optimizer.param_groups[0]["lr"], **train_metrics, **{f"dev_{key}": value for key, value in dev_metrics.items()}}
            history.append(row)
            pd.DataFrame(history).to_csv(os.path.join(Config.OUT_DIR, Config.HISTORY_NAME), index=False)
            print(row)
            if dev_metrics["pq"] > best_pq:
                best_pq, stale = dev_metrics["pq"], 0
                unwrapped = model.module if isinstance(model, nn.DataParallel) else model
                torch.save({"model": unwrapped.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                            "config": {key: value for key, value in Config.__dict__.items() if key.isupper()}}, model_path)
            else:
                stale += 1
                if epoch + 1 >= Config.EARLY_STOPPING_WARMUP and stale >= Config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping on development PQ.")
                    break
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No V4 checkpoint found at {model_path}")

    unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    unwrapped.load_state_dict(torch.load(model_path, map_location=Config.DEVICE, weights_only=True)["model"])
    cache_path = cache_logits(model, split["dev"])
    settings, calibration_pq = calibrate_decoder(cache_path, physical_records, coco)
    audit_metrics = evaluate_files(model, split["audit"], physical_records, coco, settings)
    with open(os.path.join(Config.OUT_DIR, Config.AUDIT_NAME), "w") as handle:
        json.dump({"best_dev_pq": best_pq, "calibrated_dev_pq": calibration_pq, "decoder": asdict(settings), "audit": audit_metrics}, handle, indent=2)
    print("Frozen audit metrics:", audit_metrics)
    save_visual_diagnostics(model, split["dev"], physical_records, coco, settings)

    submission = generate_submission(model, sorted(glob.glob(os.path.join(Config.TEST_DIR, "*.jpeg"))), settings)
    print(f"Saved {len(submission)} instances to {os.path.join(Config.OUT_DIR, Config.SUBMISSION_NAME)}")
    gc.collect()


main()
