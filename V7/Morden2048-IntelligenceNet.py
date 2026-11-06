# CELL 1: Downloading important binaries and importing libraries
# V7: Solar filament instance segmentation model
# Key upgrades:
# - Lightweight MORDEN Decoder (5x5 Convolutions, InstanceNorm2d, LeakyReLU(0.01), Dropout2d(0.10))
# - Hierarchical Adaptive Pyramid Pooling (APP, depth=5) and dynamic Patch Attention Gates (PAG)
# - Shallow 2-stage CNN stem (preserving high-res edge barbs at strides 4 and 1)
# - IntelligenceRefiner with multi-scale dilated convolutions, FiLM conditioning, magnetic gating, and explicit decision heads
# - Compound foreground/topology loss plus boundary, spine, seed, uncertainty, and auxiliary foreground supervision
# - Post-Processing Cascade: mandatory 2K DenseCRF -> learned seed/boundary watershed -> topology-safe instance export

# Optional Kaggle shell setup (uncomment in notebook cells):
# !pip install optuna albumentations pycocotools timm pydensecrf -q

import os
import sys
import glob
import hashlib
import json
import math
import time
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

# Must be set before importing torch to reduce memory allocator fragmentation
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
cv2.setNumThreads(1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax


print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# %% [code]
# CELL 2: Configuration Class for V7

class Config:
    # === EXACT KAGGLE DATASET PATHS ===
    BASE_DIR = '/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026'
    TRAIN_DIR = os.path.join(BASE_DIR, 'train/train_images')
    TEST_DIR = os.path.join(BASE_DIR, 'test/test_images')
    ANNOTATIONS_FILE = os.path.join(BASE_DIR, 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    OUT_DIR = '/kaggle/working/'

    # === IMAGE DIMENSIONS ===
    # Native GONG H-alpha images stay at 2048x2048 throughout training and inference.
    IMG_SIZE = 2048
    ORIGINAL_SIZE = 2048
    SEED = 42

    # === MODEL ARCHITECTURE PARAMETERS ===
    IN_CHANNELS = 3       # [Limb-corrected H-alpha, Filament Valley Map, Magnetic Structure Tensor Coherence]
    OUT_CHANNELS = 1      # Binary segmentation mask
    
    # MORDEN Hyperparameters (Hu et al., 2026)
    BASE_CHANNELS = 40
    MID_CHANNELS = 40
    BRANCH_CHANNELS = 20
    APP_POOLING_DEPTH = 5
    DROPOUT_RATE = 0.10
    LEAKY_RELU_SLOPE = 0.01

    # === COMPOUND LOSS HYPERPARAMETERS ===
    DICE_WEIGHT = 1.0
    FOCAL_WEIGHT = 1.0
    FOCAL_ALPHA = 0.66    # Long-tail foreground balance for filaments (<10% of solar disk)
    FOCAL_GAMMA = 2.0
    LOVASZ_MAX_WEIGHT = 0.50
    LOVASZ_WARMUP_EPOCHS = 5
    SKEL_RECALL_WEIGHT = 0.25
    AUX_BCE_WEIGHT = 0.15
    BOUNDARY_WEIGHT = 0.20
    SPINE_WEIGHT = 0.25
    SEED_WEIGHT = 0.20
    UNCERTAINTY_WEIGHT = 0.10

    # === TRAINING HYPERPARAMETERS ===
    BATCH_SIZE = 2
    ACCUM_STEPS = 2       # Effective batch size = 4
    EPOCHS = 25
    LR = 2e-4             # AdamW learning rate
    WEIGHT_DECAY = 1e-4   # L2 Regularization
    NUM_WORKERS = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    AMP_DTYPE = torch.bfloat16
    COMPILE_MODE = 'max-autotune'
    METRICS_FILE = 'v7_training_metrics.csv'
    DECODER_CACHE_DIR = 'v7_decoder_cache'
    DECODER_SETTINGS_FILE = 'v7_decoder_settings.json'

    # === POST-PROCESSING & SUBMISSION HYPERPARAMETERS ===
    THRESHOLD = 0.40      # Base probability threshold
    MIN_AREA = 200        # Minimum pixel area filtering out granulation noise
    BOUNDARY_THRESHOLD = 0.50
    SPINE_THRESHOLD = 0.50
    SEED_THRESHOLD = 0.50
    UNCERTAINTY_THRESHOLD = 0.60
    MIN_SEED_AREA = 9
    PQ_IOU_THRESHOLD = 0.50
    
    # 2K DenseCRF Settings
    CRF_W1 = 40.0
    CRF_W2 = 3.0
    CRF_THETA_ALPHA_SQ = 8.0   # Bilateral spatial std^2
    CRF_THETA_BETA_SQ = 12.0   # Bilateral intensity std^2
    CRF_THETA_GAMMA_SQ = 3.0   # Gaussian spatial smoothness std^2
    CRF_ITERATIONS = 10


print(f"Train dir exists: {os.path.exists(Config.TRAIN_DIR)}")
print(f"Test dir exists:  {os.path.exists(Config.TEST_DIR)}")
print(f"Annotations exist: {os.path.exists(Config.ANNOTATIONS_FILE)}")


@dataclass(frozen=True)
class DecoderSettings:
    foreground_threshold: float = Config.THRESHOLD
    min_area: int = Config.MIN_AREA
    boundary_threshold: float = Config.BOUNDARY_THRESHOLD
    spine_threshold: float = Config.SPINE_THRESHOLD
    seed_threshold: float = Config.SEED_THRESHOLD
    uncertainty_threshold: float = Config.UNCERTAINTY_THRESHOLD
    min_seed_area: int = Config.MIN_SEED_AREA


# %% [code]
# CELL 3: Physics-Grounded Preprocessing Engine

def load_image(path):
    """Load one valid native-resolution MAGFiLO image without synthetic fallbacks."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f'MAGFiLO image is missing: {path}')

    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f'MAGFiLO image cannot be decoded: {path}')
    if img.ndim != 2:
        raise ValueError(f'MAGFiLO image must be single-channel, got shape {img.shape}: {path}')
    if img.shape != (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE):
        raise ValueError(
            f'MAGFiLO image must be {Config.ORIGINAL_SIZE}x{Config.ORIGINAL_SIZE}, '
            f'got {img.shape}: {path}'
        )
    return img


def remove_limb_darkening(img):
    """
    Fits a circular mask around the solar disk and normalizes intensities inside the disk.
    Eliminates radial limb darkening so disk borders are not misidentified as filaments.
    """
    h, w = img.shape
    Y, X = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_r = np.min([cx, cy]) * 0.98

    mask = r < max_r
    img_masked = img * mask
    valid_pixels = img_masked[mask]
    if len(valid_pixels) == 0:
        return img.astype(np.float32), mask

    img_min = float(np.min(valid_pixels))
    img_max = float(np.max(valid_pixels))
    img_norm = (img_masked - img_min) / (img_max - img_min + 1e-8)
    img_norm[~mask] = 0.0
    return img_norm.astype(np.float32), mask


def get_filament_valley_map(img_limb, disk_mask):
    """
    Computes a multi-scale dark-valley enhancement map inside the solar disk.
    Subtracts the original image from a large Gaussian background estimate,
    amplifying dark absorption ribbons without enhancing background granulation.
    """
    bg_smooth = cv2.GaussianBlur(img_limb.astype(np.float32), (31, 31), 7)
    valley = np.clip(bg_smooth - img_limb, 0.0, None) * disk_mask
    v_max = valley[disk_mask].max() if np.any(disk_mask) else 1.0
    valley = valley / (v_max + 1e-8)
    valley[~disk_mask] = 0.0
    return valley.astype(np.float32)


def get_magnetic_proxy(img_limb):
    """
    Computes a Structure Tensor coherence map as a proxy for magnetic Polarity Inversion Lines.
    Filaments are highly elongated -> HIGH coherence.
    Quiet-Sun granulation is isotropic -> LOW coherence.
    """
    img_uint8 = np.clip(img_limb * 255.0, 0, 255).astype(np.uint8)
    dx = cv2.Sobel(img_uint8, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_uint8, cv2.CV_32F, 0, 1, ksize=3)

    ksize, sigma = 15, 5
    Jxx = cv2.GaussianBlur(dx**2, (ksize, ksize), sigma)
    Jxy = cv2.GaussianBlur(dx * dy, (ksize, ksize), sigma)
    Jyy = cv2.GaussianBlur(dy**2, (ksize, ksize), sigma)

    coherence = np.sqrt((Jxx - Jyy)**2 + 4 * Jxy**2)
    c_min, c_max = coherence.min(), coherence.max()
    coherence = (coherence - c_min) / (c_max - c_min + 1e-8)
    return coherence.astype(np.float32)


def prepare_input_3ch(img):
    """
    Unified 3-channel physics tensor construction:
    Channel 0: Limb-corrected H-alpha
    Channel 1: Filament Valley Map
    Channel 2: Structure Tensor Magnetic Coherence Proxy
    """
    img_f = img.astype(np.float32, copy=False)
    img_f = (img_f - float(img_f.min())) / (float(img_f.max() - img_f.min()) + 1e-8)
    img_limb, disk_mask = remove_limb_darkening(img_f)
    valley_map = get_filament_valley_map(img_limb, disk_mask)
    mag_map = get_magnetic_proxy(img_limb)
    mag_map[~disk_mask] = 0.0
    return np.stack([img_limb, valley_map, mag_map], axis=-1).astype(np.float32), img_limb


def extract_morphological_centerline(mask_bin):
    """Fast CPU Euclidean Distance Transform ridge detection for filament centerlines."""
    work = (mask_bin > 0).astype(np.uint8)
    if work.max() == 0:
        return np.zeros_like(mask_bin, dtype=np.float32)
    dist = cv2.distanceTransform(work, cv2.DIST_L2, 3)
    local_max = cv2.dilate(dist, np.ones((3, 3), dtype=np.uint8))
    ridge = (work > 0) & (dist >= np.maximum(local_max - 0.35, 0.8))
    if not np.any(ridge):
        return work.astype(np.float32)
    return ridge.astype(np.float32)


def rasterize_dilated_spine(anns, mask_2048, img_size=None):
    """Precomputes 1px-dilated centerline targets on CPU for SkeletonRecall loss."""
    spine_1px = np.zeros(mask_2048.shape, dtype=np.uint8)
    h, w = mask_2048.shape
    for ann in anns:
        raw_spine = np.asarray(ann.get('spine', []), dtype=np.float32)
        if raw_spine.ndim == 1 and len(raw_spine) >= 4 and len(raw_spine) % 2 == 0:
            pts = raw_spine.reshape(-1, 2)
            if np.all((pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)):
                cv2.polylines(spine_1px, [np.rint(pts).astype(np.int32)], False, 1, thickness=1)
    if spine_1px.max() == 0 and mask_2048.max() > 0:
        spine_1px = extract_morphological_centerline(mask_2048).astype(np.uint8)
    dilate_kernel = np.ones((3, 3), dtype=np.uint8)
    spine_dilated = cv2.dilate(spine_1px, dilate_kernel, iterations=1).astype(np.float32)
    spine_dilated = np.minimum(spine_dilated, (mask_2048 > 0).astype(np.float32))
    if img_size is not None and (img_size != h or img_size != w):
        spine_resized = cv2.resize(spine_dilated, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        return (spine_resized > 0.15).astype(np.float32)
    return spine_dilated.astype(np.float32)


# %% [code]
# CELL 4: PyTorch Dataset with MORDEN Augmentations & Physics Guardrails

def _spine_for_instance(ann, mask):
    spine = rasterize_dilated_spine([ann], mask)
    return spine if spine.max() else extract_morphological_centerline(mask)


def build_instance_targets(coco, anns, shape):
    """Build native-resolution semantic and instance-aware supervision for one annotation pass."""
    masks = [(coco.annToMask(ann) > 0).astype(np.uint8) for ann in anns]
    foreground = np.zeros(shape, dtype=np.float32)
    boundary = np.zeros(shape, dtype=np.float32)
    spine = np.zeros(shape, dtype=np.float32)
    seed = np.zeros(shape, dtype=np.float32)
    instance_id = np.zeros(shape, dtype=np.int32)
    overlap_count = np.zeros(shape, dtype=np.uint8)

    for index, (ann, mask) in enumerate(zip(anns, masks), start=1):
        foreground = np.maximum(foreground, mask)
        overlap_count += mask
        instance_id[mask > 0] = index
        inner_boundary = mask - cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        boundary = np.maximum(boundary, cv2.dilate(inner_boundary, np.ones((3, 3), np.uint8), iterations=1))
        instance_spine = _spine_for_instance(ann, mask)
        spine = np.maximum(spine, instance_spine)
        points = np.argwhere(instance_spine > 0)
        if len(points):
            y, x = points[len(points) // 2]
            cv2.circle(seed, (int(x), int(y)), 2, 1, thickness=-1)

    ambiguous = overlap_count > 1
    instance_id[ambiguous] = -1
    boundary_valid = (~ambiguous).astype(np.float32)
    return {
        'foreground': foreground.astype(np.float32),
        'boundary': boundary.astype(np.float32),
        'spine': spine.astype(np.float32),
        'seed': np.minimum(seed, foreground).astype(np.float32),
        'uncertainty': np.zeros(shape, dtype=np.float32),
        'uncertainty_valid': np.zeros(shape, dtype=np.float32),
        'boundary_valid': boundary_valid,
        'instance_id': instance_id,
    }


class SolarDataset(Dataset):
    def __init__(self, coco, records_by_filename, file_names, img_dir, img_size=2048, transforms=None, training=False):
        self.coco = coco
        self.records_by_filename = records_by_filename
        self.file_names = file_names
        self.img_dir = img_dir
        self.img_size = img_size
        self.transforms = transforms
        self.training = training

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image_ids = self.records_by_filename[file_name]
        img_id = random.choice(image_ids) if self.training else image_ids[0]
        img_path = os.path.join(self.img_dir, file_name)

        img = load_image(img_path)

        # 3-channel physics tensor
        img_3ch, _ = prepare_input_3ch(img)
        if img_3ch.shape[:2] != (self.img_size, self.img_size):
            raise ValueError(f'Expected {self.img_size}x{self.img_size} image, got {img_3ch.shape[:2]}.')

        anns = self.coco.imgToAnns.get(img_id, [])
        targets = build_instance_targets(self.coco, anns, img.shape)
        if len(image_ids) > 1:
            pass_foregrounds = [
                build_instance_targets(self.coco, self.coco.imgToAnns.get(pass_id, []), img.shape)['foreground']
                for pass_id in image_ids
            ]
            targets['uncertainty'] = (np.max(pass_foregrounds, axis=0) != np.min(pass_foregrounds, axis=0)).astype(np.float32)
            targets['uncertainty_valid'] = np.ones(img.shape, dtype=np.float32)

        if self.transforms is not None:
            aug = self.transforms(image=img_3ch, mask=targets['foreground'], boundary=targets['boundary'],
                                  spine=targets['spine'], seed=targets['seed'], uncertainty=targets['uncertainty'],
                                  uncertainty_valid=targets['uncertainty_valid'], boundary_valid=targets['boundary_valid'])
            img_3ch = aug['image']
            for key in ('foreground', 'boundary', 'spine', 'seed', 'uncertainty', 'uncertainty_valid', 'boundary_valid'):
                targets[key] = aug['mask'] if key == 'foreground' else aug[key]

        if isinstance(img_3ch, np.ndarray):
            img_3ch = torch.from_numpy(img_3ch).permute(2, 0, 1).float()
        else:
            img_3ch = img_3ch.float()

        tensors = {}
        for key, value in targets.items():
            tensor = torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            tensors[key] = tensor.unsqueeze(0).float() if key != 'instance_id' else tensor.unsqueeze(0).long()
        return img_3ch, tensors


def build_morden_train_transforms():
    """
    Table 1 of Hu et al. (2026):
    - Horizontal & Vertical flip (50%)
    - Continuous Rotation [-45, +45] deg & Scaling [0.5, 1.0] (90%)
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(
            scale=(0.5, 1.0),
            rotate=(-45, 45),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0.0,
            fill_mask=0.0,
            p=0.9,
        ),
        ToTensorV2(),
    ], additional_targets={key: 'mask' for key in ('boundary', 'spine', 'seed', 'uncertainty', 'uncertainty_valid', 'boundary_valid')})


train_transforms = build_morden_train_transforms()
val_transforms = A.Compose([ToTensorV2()], additional_targets={key: 'mask' for key in ('boundary', 'spine', 'seed', 'uncertainty', 'uncertainty_valid', 'boundary_valid')})


# %% [code]
# CELL 5: V6 Model Architecture

class MordenDoubleConv(nn.Module):
    """
    Figure 1(b) of Hu et al. (2026):
    5x5 Conv -> InstanceNorm2d -> LeakyReLU(0.01) (x2) + Dropout2d(0.10).
    Uses depthwise-separable 5x5 convolutions when channels >= 64 to keep params low.
    """
    def __init__(self, in_ch, out_ch, kernel_size=5, dropout=Config.DROPOUT_RATE,
                 slope=Config.LEAKY_RELU_SLOPE, separable=False):
        super().__init__()
        pad = kernel_size // 2

        def _conv(c_in, c_out):
            if separable and c_in >= 64 and kernel_size > 1:
                return nn.Sequential(
                    nn.Conv2d(c_in, c_in, kernel_size, padding=pad, groups=c_in, bias=False),
                    nn.Conv2d(c_in, c_out, 1, bias=False),
                )
            return nn.Conv2d(c_in, c_out, kernel_size, padding=pad, bias=False)

        self.block = nn.Sequential(
            _conv(in_ch, out_ch),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            _conv(out_ch, out_ch),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x):
        return self.block(x)


class AdaptivePyramidPooling(nn.Module):
    """
    Figure 1(c) & Equation (1) of Hu et al. (2026):
    Hierarchical arithmetic progression of pooling scales with depth L=5.
    """
    def __init__(self, in_ch, out_ch, depth=Config.APP_POOLING_DEPTH,
                 dropout=Config.DROPOUT_RATE, slope=Config.LEAKY_RELU_SLOPE):
        super().__init__()
        self.depth = depth
        mid_ch = max(in_ch // depth, 16)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.InstanceNorm2d(mid_ch, affine=True),
                nn.LeakyReLU(negative_slope=slope, inplace=True),
            )
            for _ in range(depth)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch * depth, out_ch, 1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x, target_size=None):
        _, _, H, W = x.shape if target_size is None else (None, None, target_size[0], target_size[1])
        pyramid = []
        max_pool_dim = min(H, 32)
        for i, branch in enumerate(self.branches, start=1):
            hi = max(int(math.floor(max_pool_dim * i / self.depth)), 1)
            wi = max(int(math.floor(max_pool_dim * i / self.depth)), 1)
            pooled = F.adaptive_avg_pool2d(x, output_size=(hi, wi))
            proj = branch(pooled)
            upsampled = F.interpolate(proj, size=(H, W), mode='bilinear', align_corners=False)
            pyramid.append(upsampled)
        return self.fuse(torch.cat(pyramid, dim=1))


class PatchAttentionGate(nn.Module):
    """
    Figure 1(d) & Equation (2) of Hu et al. (2026):
    Dual max & avg patch-pooling with dynamic patch size S_patch = 2^(floor(log2(sqrt(S)))).
    """
    def __init__(self, in_ch, out_ch, slope=Config.LEAKY_RELU_SLOPE, dropout=Config.DROPOUT_RATE):
        super().__init__()
        self.proj_in = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.norm_max = nn.LayerNorm(out_ch)
        self.norm_avg = nn.LayerNorm(out_ch)
        self.act = nn.LeakyReLU(negative_slope=slope, inplace=True)
        self.linear_max = nn.Linear(out_ch, out_ch)
        self.linear_avg = nn.Linear(out_ch, out_ch)
        self.norm_out1 = nn.LayerNorm(out_ch)
        self.norm_out2 = nn.LayerNorm(out_ch)
        self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x1, x2):
        x1_p = self.proj_in(x1)
        B, C, H, W = x2.shape
        if x1_p.shape[2:] != (H, W):
            x1_p = F.interpolate(x1_p, size=(H, W), mode='bilinear', align_corners=False)

        s_x1 = max(min(H, W), 4)
        s_patch = max(2 ** int(math.floor(math.log2(math.sqrt(s_x1)))), 2)
        grid_h, grid_w = max(H // s_patch, 1), max(W // s_patch, 1)

        p_max = F.adaptive_max_pool2d(x1_p, (grid_h, grid_w)).permute(0, 2, 3, 1)
        p_avg = F.adaptive_avg_pool2d(x1_p, (grid_h, grid_w)).permute(0, 2, 3, 1)

        p_max = self.norm_out1(self.linear_max(self.act(self.norm_max(p_max))))
        p_avg = self.norm_out2(self.linear_avg(self.act(self.norm_avg(p_avg))))

        gate_grid = torch.sigmoid(p_max + p_avg).permute(0, 3, 1, 2).contiguous()
        gate = F.interpolate(gate_grid, size=(H, W), mode='bilinear', align_corners=False)
        return self.drop(x2 * gate)


class MordenBranchBlock(nn.Module):
    """Hierarchical shortcut from full-res input into multi-scale decoder stages."""
    def __init__(self, in_ch, branch_ch):
        super().__init__()
        self.dc = MordenDoubleConv(in_ch, branch_ch, kernel_size=5, separable=False)
        self.pag = PatchAttentionGate(in_ch, branch_ch)

    def forward(self, embedded_info, target_size):
        pooled = F.adaptive_avg_pool2d(embedded_info, output_size=target_size)
        feat = self.dc(pooled)
        return self.pag(pooled, feat)


class IntelligenceRefiner(nn.Module):
    """Produces shared topology-aware features for explicit filament decision maps."""
    def __init__(self, in_ch=64, slope=Config.LEAKY_RELU_SLOPE):
        super().__init__()
        self.bridge1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, dilation=2, padding=2, groups=in_ch, bias=False),
            nn.InstanceNorm2d(in_ch, affine=True),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
        )
        self.bridge2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, dilation=4, padding=4, groups=in_ch, bias=False),
            nn.InstanceNorm2d(in_ch, affine=True),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
        )
        self.bridge3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, dilation=8, padding=8, groups=in_ch, bias=False),
            nn.InstanceNorm2d(in_ch, affine=True),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch + 2, in_ch, 1, bias=False),
            nn.InstanceNorm2d(in_ch, affine=True),
            nn.Sigmoid(),
        )

    def forward(self, x, mag_map, vlm_prior=None, gamma=None, beta=None):
        b1 = self.bridge1(x)
        b2 = self.bridge2(x)
        b3 = self.bridge3(x)
        bridged = x + b1 + b2 + 0.5 * b3

        if gamma is not None and beta is not None:
            bridged = bridged * (1.0 + 0.25 * gamma) + 0.25 * beta
        if vlm_prior is None:
            vlm_prior = torch.full_like(mag_map, 0.5)

        g = self.gate(torch.cat([bridged, mag_map, vlm_prior], dim=1))
        return bridged * g


class MagAgentU2Net(nn.Module):
    """
    V7 Core Network:
    - Trainable HRNet-W18 encoder initialized from scratch on MAGFiLO
    - Shallow High-Resolution Stem & Branch Blocks
    - MORDEN multiscale decoder (5x5 convs, InstanceNorm2d, APP, PAG)
    - Explicit foreground, boundary, spine, seed, and uncertainty decision maps
    """
    def __init__(self, in_ch=Config.IN_CHANNELS, out_ch=Config.OUT_CHANNELS):
        super().__init__()
        base_ch = Config.BASE_CHANNELS      # 40
        branch_ch = Config.BRANCH_CHANNELS  # 20
        fpn_ch = 96
        refiner_ch = 64

        # 1. HRNet-W18 native branches: 18/36/72/144 channels at strides 4/8/16/32.
        self.hrnet = timm.create_model(
            'hrnet_w18',
            pretrained=False,
            features_only=True,
            in_chans=in_ch,
            out_indices=(1, 2, 3, 4),
            feature_cfg={'feature_location': ''},
        )
        self.hr_s4 = nn.Sequential(nn.Conv2d(18, fpn_ch, 1, bias=False), nn.InstanceNorm2d(fpn_ch, affine=True))
        self.hr_s8 = nn.Sequential(nn.Conv2d(36, fpn_ch, 1, bias=False), nn.InstanceNorm2d(fpn_ch, affine=True))
        self.hr_s16 = nn.Sequential(nn.Conv2d(72, fpn_ch, 1, bias=False), nn.InstanceNorm2d(fpn_ch, affine=True))
        self.hr_s32 = nn.Sequential(nn.Conv2d(144, fpn_ch, 1, bias=False), nn.InstanceNorm2d(fpn_ch, affine=True))

        # 2. Shallow High-Resolution Physics Stem & Branch Blocks
        self.embed_stem = MordenDoubleConv(in_ch, base_ch, kernel_size=5, separable=False)
        self.branch_s4 = MordenBranchBlock(base_ch, branch_ch)
        self.branch_s8 = MordenBranchBlock(base_ch, branch_ch)

        # 3. HRNet-aware Multiscale Decoder (APP + PAG + Separable 5x5 Convolutions)
        self.app_deep = AdaptivePyramidPooling(fpn_ch * 2, fpn_ch, depth=5)
        self.dec_s16 = MordenDoubleConv(fpn_ch * 2, fpn_ch, kernel_size=5, separable=True)

        self.pag_s8 = PatchAttentionGate(fpn_ch, fpn_ch)
        self.dec_s8 = MordenDoubleConv(fpn_ch + branch_ch, fpn_ch, kernel_size=5, separable=True)
        self.app_s8 = AdaptivePyramidPooling(fpn_ch, fpn_ch, depth=5)

        self.pag_s4 = PatchAttentionGate(fpn_ch, fpn_ch)
        self.dec_s4 = MordenDoubleConv(fpn_ch + branch_ch, refiner_ch, kernel_size=5, separable=True)
        self.app_s4 = AdaptivePyramidPooling(refiner_ch, refiner_ch, depth=5)

        self.dec_s1 = MordenDoubleConv(refiner_ch + base_ch, refiner_ch, kernel_size=5, separable=True)

        # 4. Auxiliary foreground supervision, FiLM generation, and intelligence heads.
        self.aux_foreground_head = nn.Conv2d(fpn_ch, out_ch, 1)
        self.film_head = nn.Sequential(
            nn.Linear(fpn_ch, refiner_ch * 2),
            nn.Tanh(),
        )
        self.intelligence = IntelligenceRefiner(refiner_ch)
        self.decision_heads = nn.ModuleDict({
            'foreground': nn.Conv2d(refiner_ch, out_ch, 1),
            'boundary': nn.Conv2d(refiner_ch, out_ch, 1),
            'spine': nn.Conv2d(refiner_ch, out_ch, 1),
            'seed': nn.Conv2d(refiner_ch, out_ch, 1),
            'uncertainty': nn.Conv2d(refiner_ch, out_ch, 1),
        })

    def forward(self, x):
        _, _, H, W = x.shape
        H4, W4 = H // 4, W // 4
        H8, W8 = H // 8, W // 8
        H16, W16 = H // 16, W // 16
        mag_map = x[:, 2:3]

        s1_embed = self.embed_stem(x)
        b_s4 = self.branch_s4(s1_embed, (H4, W4))
        b_s8 = self.branch_s8(s1_embed, (H8, W8))

        h_s4, h_s8, h_s16, h_s32 = self.hrnet(x)
        p_s4 = self.hr_s4(h_s4)
        p_s8 = self.hr_s8(h_s8)
        p_s16 = self.hr_s16(h_s16)
        p_s32 = self.hr_s32(h_s32)

        p_s32_up16 = F.interpolate(p_s32, size=(H16, W16), mode='bilinear', align_corners=False)
        d16 = self.app_deep(torch.cat([p_s16, p_s32_up16], dim=1), target_size=(H16, W16))
        d16 = self.dec_s16(torch.cat([d16, p_s16], dim=1))

        d16_up8 = F.interpolate(d16, size=(H8, W8), mode='bilinear', align_corners=False)
        gated_s8 = self.pag_s8(p_s8, d16_up8)
        d8 = self.dec_s8(torch.cat([gated_s8, b_s8], dim=1))
        d8 = d8 + self.app_s8(d8, target_size=(H8, W8))

        aux_logits = self.aux_foreground_head(d8)
        pooled_context = F.adaptive_avg_pool2d(d8, 1).flatten(1)
        gamma, beta = torch.chunk(self.film_head(pooled_context).unsqueeze(-1).unsqueeze(-1), 2, dim=1)

        d8_up4 = F.interpolate(d8, size=(H4, W4), mode='bilinear', align_corners=False)
        gated_s4 = self.pag_s4(p_s4, d8_up4)
        d4 = self.dec_s4(torch.cat([gated_s4, b_s4], dim=1))
        d4 = d4 + self.app_s4(d4, target_size=(H4, W4))

        d4_up1 = F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=False)
        d1 = self.dec_s1(torch.cat([d4_up1, s1_embed], dim=1))

        vlm_prior = torch.sigmoid(F.interpolate(aux_logits, size=(H, W), mode='bilinear', align_corners=False))
        refined = self.intelligence(d1, mag_map, vlm_prior=vlm_prior, gamma=gamma, beta=beta)
        outputs = {name: head(refined) for name, head in self.decision_heads.items()}
        outputs['aux_foreground'] = aux_logits
        return outputs


# %% [code]
# CELL 6: Compound Loss (Dice + Focal + Lovasz + SkeletonRecall + AuxBCE)

def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits_flat, labels_flat):
    if logits_flat.numel() == 0:
        return logits_flat.sum() * 0.0
    signs = 2.0 * labels_flat.float() - 1.0
    errors = 1.0 - logits_flat * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels_flat[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge_loss(logits, targets, per_image=True):
    with torch.amp.autocast('cuda', enabled=False):
        logits_f = logits.float()
        targets_f = targets.float()
        if per_image:
            losses = [
                _lovasz_hinge_flat(log_i.reshape(-1), tgt_i.reshape(-1))
                for log_i, tgt_i in zip(logits_f, targets_f)
            ]
            return torch.stack(losses).mean()
        return _lovasz_hinge_flat(logits_f.reshape(-1), targets_f.reshape(-1))


class IntelligenceLoss(nn.Module):
    def __init__(
        self,
        dice_weight=Config.DICE_WEIGHT,
        focal_weight=Config.FOCAL_WEIGHT,
        focal_alpha=Config.FOCAL_ALPHA,
        focal_gamma=Config.FOCAL_GAMMA,
        lovasz_max_weight=Config.LOVASZ_MAX_WEIGHT,
        lovasz_warmup_epochs=Config.LOVASZ_WARMUP_EPOCHS,
        skel_weight=Config.SKEL_RECALL_WEIGHT,
        aux_weight=Config.AUX_BCE_WEIGHT,
    ):
        super().__init__()
        self.bce_none = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_mean = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.lovasz_max_weight = lovasz_max_weight
        self.lovasz_warmup_epochs = lovasz_warmup_epochs
        self.skel_weight = skel_weight
        self.aux_weight = aux_weight

    def masked_bce_dice(self, logits, targets, valid=None):
        valid = torch.ones_like(targets) if valid is None else valid.float()
        denominator = valid.sum().clamp_min(1.0)
        bce = (self.bce_none(logits.float(), targets.float()) * valid).sum() / denominator
        probs = torch.sigmoid(logits.float()) * valid
        target = targets.float() * valid
        dice = 1.0 - (2.0 * (probs * target).sum() + 1.0) / (probs.sum() + target.sum() + 1.0)
        return bce + dice

    def get_lovasz_weight(self, epoch=1):
        if self.lovasz_max_weight <= 0:
            return 0.0
        if self.lovasz_warmup_epochs <= 1:
            return float(self.lovasz_max_weight)
        progress = float(epoch - 1) / float(self.lovasz_warmup_epochs - 1)
        return float(self.lovasz_max_weight * min(1.0, max(0.0, progress)))

    def forward(self, outputs, targets, epoch=1):
        with torch.amp.autocast('cuda', enabled=False):
            logits_f = outputs['foreground'].float()
            targets_f = targets['foreground'].float()

            # 1. Soft Dice Loss
            probs = torch.sigmoid(logits_f)
            smooth = 1.0
            iflat = probs.reshape(-1)
            tflat = targets_f.reshape(-1)
            intersection = (iflat * tflat).sum()
            dice_loss = 1.0 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

            # 2. Long-Tail Focal Loss (alpha=0.66, gamma=2.0)
            bce_per_pixel = self.bce_none(logits_f, targets_f)
            p_t = probs * targets_f + (1.0 - probs) * (1.0 - targets_f)
            alpha_t = self.focal_alpha * targets_f + (1.0 - self.focal_alpha) * (1.0 - targets_f)
            focal_map = alpha_t * torch.pow((1.0 - p_t).clamp_min(1e-6), self.focal_gamma)
            focal_loss = (focal_map * bce_per_pixel).mean()

            total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss

            # 3. Lovasz-Hinge Loss with Warmup
            w_lovasz = self.get_lovasz_weight(epoch)
            if w_lovasz > 0.0:
                lovasz_loss = lovasz_hinge_loss(logits_f, targets_f, per_image=True)
                total_loss = total_loss + w_lovasz * lovasz_loss

            # 4. Skeleton recall keeps the main foreground decision continuous.
            if self.skel_weight > 0:
                sflat = targets['spine'].float().reshape(-1)
                skel_recall = ((iflat * sflat).sum() + smooth) / (sflat.sum() + smooth)
                total_loss = total_loss + self.skel_weight * (1.0 - skel_recall)

            boundary_loss = self.masked_bce_dice(outputs['boundary'], targets['boundary'], targets['boundary_valid'])
            spine_loss = self.masked_bce_dice(outputs['spine'], targets['spine'])
            seed_loss = self.masked_bce_dice(outputs['seed'], targets['seed'])
            uncertainty_valid = targets['uncertainty_valid']
            uncertainty_loss = (self.bce_none(outputs['uncertainty'].float(), targets['uncertainty'].float()) * uncertainty_valid).sum()
            uncertainty_loss = uncertainty_loss / uncertainty_valid.sum().clamp_min(1.0)
            total_loss = total_loss + Config.BOUNDARY_WEIGHT * boundary_loss
            total_loss = total_loss + Config.SPINE_WEIGHT * spine_loss
            total_loss = total_loss + Config.SEED_WEIGHT * seed_loss
            total_loss = total_loss + Config.UNCERTAINTY_WEIGHT * uncertainty_loss

            # 5. Intermediate foreground supervision.
            if self.aux_weight > 0:
                aux_logits = outputs['aux_foreground']
                aux_target = F.adaptive_max_pool2d(targets_f, output_size=aux_logits.shape[2:])
                aux_loss = self.bce_mean(aux_logits.float(), aux_target)
                total_loss = total_loss + self.aux_weight * aux_loss

            return total_loss, {
                'foreground': (self.dice_weight * dice_loss + self.focal_weight * focal_loss).detach(),
                'boundary': boundary_loss.detach(), 'spine': spine_loss.detach(), 'seed': seed_loss.detach(),
                'uncertainty': uncertainty_loss.detach(), 'total': total_loss.detach(),
            }


# %% [code]
# CELL 7: Training & Validation Harness

def base_model(model):
    """Returns the original module when torch.compile wraps it."""
    return getattr(model, '_orig_mod', model)


def unpack_batch(batch):
    return batch[0], batch[1]


def train_epoch(model, loader, optimizer, criterion, epoch=1):
    model.train()
    totals = defaultdict(float)
    optimizer.zero_grad(set_to_none=True)
    remainder = len(loader) % Config.ACCUM_STEPS

    for step, batch in enumerate(tqdm(loader, desc=f"Training Epoch {epoch}")):
        images, targets = unpack_batch(batch)
        images = images.to(Config.DEVICE, non_blocking=True)
        targets = {key: value.to(Config.DEVICE, non_blocking=True) for key, value in targets.items()}

        divisor = remainder if remainder and step >= len(loader) - remainder else Config.ACCUM_STEPS

        with torch.autocast(device_type='cuda', dtype=Config.AMP_DTYPE):
            outputs = model(images)
            loss, pieces = criterion(outputs, targets, epoch=epoch)
            loss = loss / divisor

        loss.backward()

        if (step + 1) % Config.ACCUM_STEPS == 0 or step + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for key, value in pieces.items():
            totals[key] += float(value)

    return {key: value / len(loader) for key, value in totals.items()}


def validate(model, loader, criterion, epoch=1):
    model.eval()
    totals = defaultdict(float)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images, targets = unpack_batch(batch)
            images = images.to(Config.DEVICE, non_blocking=True)
            targets = {key: value.to(Config.DEVICE, non_blocking=True) for key, value in targets.items()}

            with torch.autocast(device_type='cuda', dtype=Config.AMP_DTYPE):
                outputs = model(images)
                _, pieces = criterion(outputs, targets, epoch=epoch)

            for key, value in pieces.items():
                totals[key] += float(value)

    return {key: value / len(loader) for key, value in totals.items()}


# %% [code]
# CELL 8: Post-Processing Cascade (DenseCRF -> Learned Watershed) & Submission

def apply_densecrf_2k(prob_2k, img_limb_2k):
    """Applies mandatory DenseCRF boundary refinement at 2048x2048."""
    H, W = prob_2k.shape
    prob_fg = np.clip(prob_2k.astype(np.float32), 1e-6, 1.0 - 1e-6)
    img_u8 = np.clip(img_limb_2k * 255.0, 0, 255).astype(np.uint8)

    softmax_probs = np.stack([1.0 - prob_fg, prob_fg], axis=0)
    unary = unary_from_softmax(softmax_probs)
    crf = dcrf.DenseCRF2D(W, H, 2)
    crf.setUnaryEnergy(unary)

    sxy_gamma = math.sqrt(Config.CRF_THETA_GAMMA_SQ)
    crf.addPairwiseGaussian(sxy=(sxy_gamma, sxy_gamma), compat=Config.CRF_W2)

    sxy_alpha = math.sqrt(Config.CRF_THETA_ALPHA_SQ)
    srgb_beta = math.sqrt(Config.CRF_THETA_BETA_SQ)
    img_rgb = np.repeat(img_u8[:, :, np.newaxis], 3, axis=2)
    crf.addPairwiseBilateral(
        sxy=(sxy_alpha, sxy_alpha),
        srgb=(srgb_beta, srgb_beta, srgb_beta),
        rgbim=np.ascontiguousarray(img_rgb),
        compat=Config.CRF_W1,
    )

    Q = crf.inference(Config.CRF_ITERATIONS)
    return np.asarray(Q, dtype=np.float32)[1].reshape((H, W))


def resolve_instances(densecrf_foreground, boundary, spine, seed, uncertainty, settings):
    """Split only foreground components with multiple reliable learned markers."""
    mask = (densecrf_foreground > settings.foreground_threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    instances = []
    for label in range(1, count):
        component = (labels == label).astype(np.uint8)
        if stats[label, cv2.CC_STAT_AREA] < settings.min_area:
            continue
        markers_binary = ((seed >= settings.seed_threshold) & (spine >= settings.spine_threshold) &
                          (boundary < settings.boundary_threshold) & (uncertainty < settings.uncertainty_threshold) &
                          (component > 0)).astype(np.uint8)
        marker_count, marker_labels, marker_stats, _ = cv2.connectedComponentsWithStats(markers_binary, connectivity=8)
        valid_markers = [i for i in range(1, marker_count) if marker_stats[i, cv2.CC_STAT_AREA] >= settings.min_seed_area]
        if len(valid_markers) < 2:
            instances.append(component)
            continue
        y, x = np.where(component)
        y0, y1 = max(0, y.min() - 1), min(component.shape[0], y.max() + 2)
        x0, x1 = max(0, x.min() - 1), min(component.shape[1], x.max() + 2)
        local_component = component[y0:y1, x0:x1]
        markers = np.ones(local_component.shape, dtype=np.int32)
        markers[local_component > 0] = 0
        for marker_id, seed_id in enumerate(valid_markers, start=2):
            markers[marker_labels[y0:y1, x0:x1] == seed_id] = marker_id
        surface = np.clip((1.0 - densecrf_foreground[y0:y1, x0:x1]) + boundary[y0:y1, x0:x1] +
                          0.25 * uncertainty[y0:y1, x0:x1], 0.0, 1.0)
        watershed = cv2.watershed(np.repeat((surface * 255).astype(np.uint8)[..., None], 3, axis=2), markers)
        children = []
        for marker_id in range(2, 2 + len(valid_markers)):
            child = np.zeros_like(component)
            child[y0:y1, x0:x1] = ((watershed == marker_id) & (local_component > 0)).astype(np.uint8)
            if child.sum() >= min_area:
                children.append(child)
        instances.extend(children if len(children) >= 2 else [component])
    return instances


def resolve_components(densecrf_foreground, settings):
    mask = (densecrf_foreground > settings.foreground_threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(labels == label).astype(np.uint8) for label in range(1, count)
            if stats[label, cv2.CC_STAT_AREA] >= settings.min_area]


def postprocess_probability_v7(
    probabilities,
    img_limb_2048=None,
    settings=DecoderSettings(),
):
    """Mandatory DenseCRF followed by learned boundary/seed-aware instance decoding."""
    if img_limb_2048 is None:
        raise ValueError('DenseCRF requires the native limb-corrected image.')
    foreground = apply_densecrf_2k(probabilities['foreground'], img_limb_2048)
    return resolve_instances(foreground, probabilities['boundary'], probabilities['spine'],
                             probabilities['seed'], probabilities['uncertainty'], settings)


def pycoco_rle_encode(img):
    """Encodes a binary mask to pycocotools RLE ASCII string."""
    rle = maskUtils.encode(np.asfortranarray(img.astype(np.uint8)))
    return rle['counts'].decode('utf-8')


def assert_rle_round_trip(mask, rle_string):
    decoded = maskUtils.decode({'size': list(mask.shape), 'counts': rle_string.encode('utf-8')})
    if not np.array_equal(decoded.astype(np.uint8), mask.astype(np.uint8)):
        raise ValueError('Submission RLE does not round-trip through pycocotools.')


def predict_intelligence(model, img_path):
    """Return the five native-resolution decision probabilities for one image."""
    img = load_image(img_path)
    prepared, limb = prepare_input_3ch(img)
    tensor = torch.from_numpy(prepared).permute(2, 0, 1).unsqueeze(0).to(Config.DEVICE, non_blocking=True)
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=Config.AMP_DTYPE):
        outputs = model(tensor)
    probabilities = {
        name: torch.sigmoid(outputs[name]).squeeze().float().cpu().numpy()
        for name in ('foreground', 'boundary', 'spine', 'seed', 'uncertainty')
    }
    return probabilities, limb


def get_overlap_matrices(gt_layers, pred_layers):
    """Exact in-memory counterpart of the Kaggle self-evaluation notebook."""
    n_gt, height, width = gt_layers.shape
    n_pred = pred_layers.shape[0]
    gt_flat = torch.from_numpy(gt_layers.astype(np.float32, copy=False)).reshape(n_gt, height * width)
    pred_flat = torch.from_numpy(pred_layers.astype(np.float32, copy=False)).reshape(n_pred, height * width)
    intersection = torch.matmul(gt_flat, pred_flat.t())
    gt_areas = gt_flat.sum(dim=1).view(-1, 1)
    pred_areas = pred_flat.sum(dim=1).view(1, -1)
    union = gt_areas + pred_areas - intersection
    iou = torch.where(union == 0, torch.tensor(0.0), intersection / union)
    dice = torch.where(union == 0, torch.tensor(0.0), 2.0 * intersection / (gt_areas + pred_areas))
    return iou, dice


def panoptic_metrics(predictions, targets):
    """Metric-identical matching and FP/FN accounting to Kaggle's self-evaluator."""
    shape = (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE)
    gt_layers = np.stack(targets).astype(np.float32) if targets else np.zeros((0, *shape), dtype=np.float32)
    pred_layers = np.stack(predictions).astype(np.float32) if predictions else np.zeros((0, *shape), dtype=np.float32)
    iou, dice = get_overlap_matrices(gt_layers, pred_layers)
    hit = iou > Config.PQ_IOU_THRESHOLD
    matched_iou = iou[hit].tolist()
    matched_dice = dice[hit].tolist()
    gt_degree = hit.sum(dim=1).numpy() if len(targets) else np.zeros(0, dtype=np.int64)
    pred_degree = hit.sum(dim=0).numpy() if len(predictions) else np.zeros(0, dtype=np.int64)
    fp = int((pred_degree == 0).sum())
    fn = int((gt_degree == 0).sum())
    pred_foreground = np.logical_or.reduce(predictions) if predictions else np.zeros(shape, dtype=np.uint8)
    gt_foreground = np.logical_or.reduce(targets) if targets else np.zeros(shape, dtype=np.uint8)
    positive_iou = iou[iou > 0].tolist()
    positive_dice = dice[dice > 0].tolist()
    return {
        'sum_iou': sum(matched_iou), 'tp': len(matched_iou), 'fp': fp, 'fn': fn,
        'pred_instances': len(predictions), 'gt_instances': len(targets),
        'splits': int(np.maximum(gt_degree - 1, 0).sum()),
        'merges': int(np.maximum(pred_degree - 1, 0).sum()),
        'gt_miss_count': int((gt_degree == 0).sum()), 'pred_miss_count': int((pred_degree == 0).sum()),
        'gt_multi_count': int((gt_degree > 1).sum()), 'pred_multi_count': int((pred_degree > 1).sum()),
        'foreground_intersection': int(np.logical_and(pred_foreground, gt_foreground).sum()),
        'pred_foreground_pixels': int(pred_foreground.sum()), 'gt_foreground_pixels': int(gt_foreground.sum()),
        '_positive_iou': positive_iou, '_positive_dice': positive_dice, '_matched_dice': matched_dice,
    }


def summarize_panoptic(totals):
    denominator = totals['tp'] + 0.5 * totals['fp'] + 0.5 * totals['fn']
    dice_denominator = totals['pred_foreground_pixels'] + totals['gt_foreground_pixels']
    positive_iou = totals.get('_positive_iou', [])
    positive_dice = totals.get('_positive_dice', [])
    matched_dice = totals.get('_matched_dice', [])
    numeric_totals = {key: value for key, value in totals.items() if not key.startswith('_')}
    return {
        'pq': totals['sum_iou'] / denominator if denominator else 0.0,
        'sq': totals['sum_iou'] / totals['tp'] if totals['tp'] else 0.0,
        'rq': totals['tp'] / denominator if denominator else 0.0,
        'dice': 2.0 * totals['foreground_intersection'] / dice_denominator if dice_denominator else 0.0,
        'fragmentation_rate': totals['splits'] / totals['gt_instances'] if totals['gt_instances'] else 0.0,
        'overmerge_rate': totals['merges'] / totals['pred_instances'] if totals['pred_instances'] else 0.0,
        'false_positive_rate': totals['fp'] / (totals['tp'] + totals['fp']) if totals['tp'] + totals['fp'] else 0.0,
        'false_negative_rate': totals['fn'] / (totals['tp'] + totals['fn']) if totals['tp'] + totals['fn'] else 0.0,
        'gt_miss_rate': totals['gt_miss_count'] / totals['gt_instances'] if totals['gt_instances'] else 0.0,
        'pred_miss_rate': totals['pred_miss_count'] / totals['pred_instances'] if totals['pred_instances'] else 0.0,
        'one_to_many_rate': totals['gt_multi_count'] / totals['gt_instances'] if totals['gt_instances'] else 0.0,
        'many_to_one_rate': totals['pred_multi_count'] / totals['pred_instances'] if totals['pred_instances'] else 0.0,
        'pair_iou_mean': float(np.mean(positive_iou)) if positive_iou else 0.0,
        'pair_iou_median': float(np.median(positive_iou)) if positive_iou else 0.0,
        'pair_dice_mean': float(np.mean(positive_dice)) if positive_dice else 0.0,
        'pair_dice_median': float(np.median(positive_dice)) if positive_dice else 0.0,
        'matched_dice_mean': float(np.mean(matched_dice)) if matched_dice else 0.0,
        **numeric_totals,
    }


def accumulate_pass_metrics(totals, predictions, coco, image_ids):
    for image_id in image_ids:
        targets = [(coco.annToMask(ann) > 0).astype(np.uint8) for ann in coco.imgToAnns.get(image_id, [])]
        for key, value in panoptic_metrics(predictions, targets).items():
            if key.startswith('_'):
                totals.setdefault(key, []).extend(value)
            else:
                totals[key] += value


def evaluate_panoptic(model, coco, records_by_filename, file_names, settings=DecoderSettings()):
    """Evaluate every physical image prediction against every available annotation pass."""
    totals = defaultdict(float)
    for file_name in tqdm(file_names, desc='Validating PQ'):
        probabilities, limb = predict_intelligence(model, os.path.join(Config.TRAIN_DIR, file_name))
        predictions = postprocess_probability_v7(probabilities, limb, settings)
        accumulate_pass_metrics(totals, predictions, coco, records_by_filename[file_name])
    return summarize_panoptic(totals)


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_decoder_cache(model, file_names, checkpoint_path):
    """Run GPU inference and DenseCRF exactly once per grouped development observation."""
    cache_dir = os.path.join(Config.OUT_DIR, Config.DECODER_CACHE_DIR)
    os.makedirs(cache_dir, exist_ok=True)
    records = []
    for index, file_name in enumerate(tqdm(file_names, desc='Caching decoder maps')):
        probabilities, limb = predict_intelligence(model, os.path.join(Config.TRAIN_DIR, file_name))
        densecrf_foreground = apply_densecrf_2k(probabilities['foreground'], limb)
        paths = {}
        for name, array in {**probabilities, 'densecrf_foreground': densecrf_foreground}.items():
            relative_path = f'{index:04d}_{name}.npy'
            np.save(os.path.join(cache_dir, relative_path), np.asarray(array, dtype=np.float32))
            paths[name] = relative_path
        records.append({'file_name': file_name, 'maps': paths})
    manifest = {
        'version': 1,
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha256': checkpoint_sha256(checkpoint_path),
        'split_seed': Config.SEED,
        'file_names': list(file_names),
        'records': records,
    }
    with open(os.path.join(cache_dir, 'manifest.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2)
    return cache_dir, manifest


def load_decoder_cache(cache_dir, checkpoint_path, expected_file_names):
    with open(os.path.join(cache_dir, 'manifest.json')) as handle:
        manifest = json.load(handle)
    if manifest['checkpoint_sha256'] != checkpoint_sha256(checkpoint_path) or manifest['file_names'] != list(expected_file_names):
        raise ValueError('Decoder cache does not match the selected checkpoint or grouped development split.')
    return manifest


def evaluate_cached_decoder(cache_dir, manifest, coco, records_by_filename, settings, components_only=False):
    """Pure NumPy/OpenCV grouped-PQ evaluation; never invokes the model or DenseCRF."""
    totals = defaultdict(float)
    for record in manifest['records']:
        maps = {name: np.load(os.path.join(cache_dir, relative_path)) for name, relative_path in record['maps'].items()}
        predictions = resolve_components(maps['densecrf_foreground'], settings) if components_only else resolve_instances(
            maps['densecrf_foreground'], maps['boundary'], maps['spine'], maps['seed'], maps['uncertainty'], settings)
        accumulate_pass_metrics(totals, predictions, coco, records_by_filename[record['file_name']])
    return summarize_panoptic(totals)


def save_decoder_settings(path, settings, metrics, manifest):
    payload = {'settings': asdict(settings), 'metrics': metrics, 'cache_version': manifest['version'],
               'checkpoint_sha256': manifest['checkpoint_sha256'], 'split_seed': manifest['split_seed'],
               'file_names': manifest['file_names']}
    with open(path, 'w') as handle:
        json.dump(payload, handle, indent=2)


def load_decoder_settings(path):
    with open(path) as handle:
        return DecoderSettings(**json.load(handle)['settings'])


def decoder_rank(metrics):
    return (-metrics['pq'], -metrics['dice'], metrics['fragmentation_rate'], metrics['overmerge_rate'],
            metrics['false_positive_rate'], metrics['false_negative_rate'])


def calibrate_decoder_cache(cache_dir, manifest, coco, records_by_filename):
    """Staged grouped-PQ search over cached arrays; never runs model inference or DenseCRF."""
    stage_a = []
    for threshold in np.linspace(0.25, 0.55, 7):
        for min_area in (50, 100, 200, 400):
            settings = DecoderSettings(foreground_threshold=float(threshold), min_area=min_area)
            metrics = evaluate_cached_decoder(cache_dir, manifest, coco, records_by_filename, settings, components_only=True)
            stage_a.append((settings, metrics))
    stage_a.sort(key=lambda item: decoder_rank(item[1]))

    best_settings = stage_a[0][0]
    best_metrics = evaluate_cached_decoder(cache_dir, manifest, coco, records_by_filename, best_settings)
    coarse_values = {
        'boundary_threshold': np.linspace(0.35, 0.70, 6),
        'spine_threshold': np.linspace(0.35, 0.70, 6),
        'seed_threshold': np.linspace(0.35, 0.70, 6),
        'uncertainty_threshold': np.linspace(0.40, 0.80, 5),
        'min_seed_area': (5, 9, 16),
    }
    for initial_settings, _ in stage_a[:3]:
        current = initial_settings
        current_metrics = evaluate_cached_decoder(cache_dir, manifest, coco, records_by_filename, current)
        for field, values in coarse_values.items():
            candidates = [(replace(current, **{field: float(value) if 'threshold' in field else int(value)}), None) for value in values]
            scored = [(candidate, evaluate_cached_decoder(cache_dir, manifest, coco, records_by_filename, candidate)) for candidate, _ in candidates]
            current, current_metrics = min(scored, key=lambda item: decoder_rank(item[1]))
        if decoder_rank(current_metrics) < decoder_rank(best_metrics):
            best_settings, best_metrics = current, current_metrics

    fine_ranges = {
        'foreground_threshold': (0.25, 0.55), 'boundary_threshold': (0.35, 0.70),
        'spine_threshold': (0.35, 0.70), 'seed_threshold': (0.35, 0.70),
        'uncertainty_threshold': (0.40, 0.80),
    }
    for field, (lower, upper) in fine_ranges.items():
        center = getattr(best_settings, field)
        values = [value for value in (center - 0.04, center - 0.02, center, center + 0.02, center + 0.04)
                  if lower <= value <= upper]
        scored = []
        for value in values:
            candidate = replace(best_settings, **{field: float(value)})
            scored.append((candidate, evaluate_cached_decoder(cache_dir, manifest, coco, records_by_filename, candidate)))
        best_settings, best_metrics = min(scored, key=lambda item: decoder_rank(item[1]))
    return best_settings, best_metrics


def generate_submission(model, test_paths, out_path, settings):
    """Runs inference on test images and generates Kaggle submission.csv."""
    model.eval()
    submissions = []

    with torch.no_grad():
        for img_path in tqdm(test_paths, desc="Generating Submission"):
            base_id = os.path.basename(img_path).split('.')[0]
            if '_' in base_id:
                raise ValueError(f'Image ID contains an underscore and is incompatible with Kaggle self-eval parsing: {base_id}')
            probabilities, img_limb_2048 = predict_intelligence(model, img_path)

            instances = postprocess_probability_v7(
                probabilities,
                img_limb_2048=img_limb_2048,
                settings=settings,
            )

            for i, inst_mask in enumerate(instances, start=1):
                rle_str = pycoco_rle_encode(inst_mask)
                assert_rle_round_trip(inst_mask, rle_str)
                fil_id = f"{base_id}_{i}"
                submissions.append({
                    'filament_id': fil_id,
                    'segmentation_rle': rle_str
                })

    df = pd.DataFrame(submissions)
    if df.empty:
        df = pd.DataFrame(columns=['filament_id', 'segmentation_rle'])
    df.to_csv(out_path, index=False)
    print(f"\nSubmission saved to {out_path}")
    print(f"Total filaments detected: {len(df)}")
    if not df.empty and 'filament_id' in df:
        print(f"Unique images: {df['filament_id'].str.rsplit('_', n=1).str[0].nunique()}")


# %% [code]
# CELL 9: Main Execution Pipeline

def main():
    print("=" * 60)
    print("Initializing V7 HRNet-W18 MORDEN Pipeline")
    print("=" * 60)

    # 1. Load COCO Annotations
    coco = COCO(Config.ANNOTATIONS_FILE)
    all_img_ids = list(coco.imgs.keys())
    print(f"Total images in JSON: {len(all_img_ids)}")

    orphan_annotation_ids = sorted({
        ann['image_id']
        for ann in coco.dataset.get('annotations', [])
        if ann['image_id'] not in coco.imgs
    })
    if orphan_annotation_ids:
        preview = ', '.join(map(str, orphan_annotation_ids[:10]))
        raise ValueError(f'Annotations reference image IDs absent from the COCO images registry: {preview}')

    missing_train_images = []
    for img_id in all_img_ids:
        file_name = coco.imgs[img_id]['file_name']
        img_path = os.path.join(Config.TRAIN_DIR, file_name)
        if not os.path.isfile(img_path):
            missing_train_images.append(file_name)

    if missing_train_images:
        preview = ', '.join(missing_train_images[:10])
        raise FileNotFoundError(
            f'{len(missing_train_images)} COCO image entries are missing from {Config.TRAIN_DIR}: {preview}'
        )

    records_by_filename = defaultdict(list)
    for image_id in all_img_ids:
        records_by_filename[coco.imgs[image_id]['file_name']].append(image_id)
    physical_files = sorted(records_by_filename)

    print(f"Valid training images on disk: {len(physical_files)} physical observations")
    print(f"Total ground truth annotations: {len(coco.dataset.get('annotations', []))}")

    # 2. Group-safe Train / Development / Audit Split by physical filename.
    random.seed(Config.SEED)
    random.shuffle(physical_files)
    train_end = int(0.8 * len(physical_files))
    dev_end = int(0.9 * len(physical_files))
    train_files = physical_files[:train_end]
    dev_files = physical_files[train_end:dev_end]
    audit_files = physical_files[dev_end:]

    print(f"Training: {len(train_files)} | Development: {len(dev_files)} | Audit: {len(audit_files)} physical images")

    # 3. Datasets and Loaders
    train_ds = SolarDataset(coco, records_by_filename, train_files, Config.TRAIN_DIR, Config.IMG_SIZE, train_transforms, training=True)
    val_ds = SolarDataset(coco, records_by_filename, dev_files, Config.TRAIN_DIR, Config.IMG_SIZE, val_transforms)

    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )

    # 4. Initialize and compile the single-A100 model
    model = MagAgentU2Net(in_ch=Config.IN_CHANNELS, out_ch=Config.OUT_CHANNELS)
    model = model.to(Config.DEVICE)
    model = torch.compile(model, mode=Config.COMPILE_MODE)

    # 5. Filter Optimizer Parameters (Trainable Only)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"\nModel Parameters: {total_count:,} total, {trainable_count:,} trainable")

    optimizer = torch.optim.AdamW(trainable_params, lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    criterion = IntelligenceLoss()

    # 6. Training Loop with PQ-based model selection.
    best_pq = float('-inf')
    model_path = os.path.join(Config.OUT_DIR, 'best_model.pth')
    metrics_path = os.path.join(Config.OUT_DIR, Config.METRICS_FILE)
    metric_history = []

    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch}/{Config.EPOCHS}")
        print(f"{'='*40}")

        train_metrics = train_epoch(model, train_loader, optimizer, criterion, epoch=epoch)
        val_metrics = validate(model, val_loader, criterion, epoch=epoch)
        pq_metrics = evaluate_panoptic(model, coco, records_by_filename, dev_files)

        print(f"Train Total: {train_metrics['total']:.4f} | Val Total: {val_metrics['total']:.4f} | "
              f"PQ: {pq_metrics['pq']:.4f} | SQ: {pq_metrics['sq']:.4f} | RQ: {pq_metrics['rq']:.4f} | "
              f"Dice: {pq_metrics['dice']:.4f} | split rate: {pq_metrics['fragmentation_rate']:.4f} | "
              f"merge rate: {pq_metrics['overmerge_rate']:.4f} | FP rate: {pq_metrics['false_positive_rate']:.4f} | "
              f"FN rate: {pq_metrics['false_negative_rate']:.4f}")

        metric_history.append({
            'epoch': epoch, 'split': 'development',
            **{f'train_{key}': value for key, value in train_metrics.items()},
            **{f'validation_{key}': value for key, value in val_metrics.items()},
            **pq_metrics,
        })
        pd.DataFrame(metric_history).to_csv(metrics_path, index=False)

        if pq_metrics['pq'] > best_pq:
            best_pq = pq_metrics['pq']
            torch.save(base_model(model).state_dict(), model_path)
            print(f"✓ Saved new best model checkpoint! (development PQ: {best_pq:.4f})")

    # 7. Generate Submission on Test Images
    print("\n" + "=" * 60)
    print("Loading best checkpoint for Kaggle submission...")
    base_model(model).load_state_dict(torch.load(model_path, map_location=Config.DEVICE, weights_only=True))
    cache_dir, manifest = build_decoder_cache(model, dev_files, model_path)
    manifest = load_decoder_cache(cache_dir, model_path, dev_files)
    decoder_settings, calibration_metrics = calibrate_decoder_cache(cache_dir, manifest, coco, records_by_filename)
    settings_path = os.path.join(Config.OUT_DIR, Config.DECODER_SETTINGS_FILE)
    save_decoder_settings(settings_path, decoder_settings, calibration_metrics, manifest)
    decoder_settings = load_decoder_settings(settings_path)
    print(f"Frozen decoder settings: {asdict(decoder_settings)} | development PQ: {calibration_metrics['pq']:.4f}")

    audit_metrics = evaluate_panoptic(model, coco, records_by_filename, audit_files, settings=decoder_settings)
    metric_history.append({'epoch': 'best', 'split': 'decoder_audit', **audit_metrics, **{f'decoder_{key}': value for key, value in asdict(decoder_settings).items()}})
    pd.DataFrame(metric_history).to_csv(metrics_path, index=False)
    print(f"Frozen audit PQ: {audit_metrics['pq']:.4f} | SQ: {audit_metrics['sq']:.4f} | RQ: {audit_metrics['rq']:.4f} | "
          f"Dice: {audit_metrics['dice']:.4f}")

    test_imgs = sorted(glob.glob(os.path.join(Config.TEST_DIR, '*.jpeg')))
    print(f"Found {len(test_imgs)} test images.")

    submission_path = os.path.join(Config.OUT_DIR, 'submission.csv')
    generate_submission(model, test_imgs, submission_path, decoder_settings)
    print("=" * 60)
    print("V7 PIPELINE COMPLETE!")
    print(f"Submission written to: {submission_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
