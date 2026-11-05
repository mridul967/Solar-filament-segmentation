# CELL 1: Downloading important binaries and importing libraries
# V7: Solar filament instance segmentation model
# Key upgrades:
# - Lightweight MORDEN Decoder (5x5 Convolutions, InstanceNorm2d, LeakyReLU(0.01), Dropout2d(0.10))
# - Hierarchical Adaptive Pyramid Pooling (APP, depth=5) and dynamic Patch Attention Gates (PAG)
# - Shallow 2-stage CNN stem (preserving high-res edge barbs at strides 4 and 1)
# - CognitiveRefinerV6 with multi-scale dilated convolutions (d=2,4,8), FiLM conditioning & magnetic gating
# - Compound Loss: 1.0 * Dice + 1.0 * Focal(alpha=0.66, gamma=2.0) + w_lovasz * LovaszHinge + 0.25 * SkeletonRecall + 0.15 * AuxBCE
# - Post-Processing Cascade: 2K DenseCRF boundary snapping -> OpenCV hole filling -> KD-Tree/DBSCAN fragment clustering

# Optional Kaggle shell setup (uncomment in notebook cells):
# !pip install optuna albumentations pycocotools timm pydensecrf -q

import os
import sys
import glob
import math
import time
import random
from collections import defaultdict
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

from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

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

    # === TRAINING HYPERPARAMETERS ===
    BATCH_SIZE = 2
    ACCUM_STEPS = 2       # Effective batch size = 4
    EPOCHS = 25
    LR = 2e-4             # AdamW learning rate
    WEIGHT_DECAY = 1e-4   # L2 Regularization
    NUM_WORKERS = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    AMP_DTYPE = torch.bfloat16
    COMPILE_MODE = 'max-autotune'

    # === POST-PROCESSING & SUBMISSION HYPERPARAMETERS ===
    THRESHOLD = 0.40      # Base probability threshold
    MIN_AREA = 200        # Minimum pixel area filtering out granulation noise
    
    # 2K DenseCRF Settings
    CRF_W1 = 40.0
    CRF_W2 = 3.0
    CRF_THETA_ALPHA_SQ = 8.0   # Bilateral spatial std^2
    CRF_THETA_BETA_SQ = 12.0   # Bilateral intensity std^2
    CRF_THETA_GAMMA_SQ = 3.0   # Gaussian spatial smoothness std^2
    CRF_ITERATIONS = 10

    # DBSCAN Fragment Integration Settings
    USE_DBSCAN = True
    DBSCAN_EPS = 28.0          # Spatial distance threshold (pixels at 2K resolution)
    DBSCAN_MIN_PTS = 140       # Minimum cluster points

print(f"Train dir exists: {os.path.exists(Config.TRAIN_DIR)}")
print(f"Test dir exists:  {os.path.exists(Config.TEST_DIR)}")
print(f"Annotations exist: {os.path.exists(Config.ANNOTATIONS_FILE)}")


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

class SolarDataset(Dataset):
    def __init__(self, coco, img_ids, img_dir, img_size=1024, transforms=None):
        self.coco = coco
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        file_name = img_info['file_name']
        img_path = os.path.join(self.img_dir, file_name)

        img = load_image(img_path)

        # 3-channel physics tensor
        img_3ch, _ = prepare_input_3ch(img)
        if img_3ch.shape[:2] != (self.img_size, self.img_size):
            raise ValueError(f'Expected {self.img_size}x{self.img_size} image, got {img_3ch.shape[:2]}.')

        # Ground truth 2D binary mask
        anns = self.coco.imgToAnns.get(img_id, [])
        mask_2048 = np.zeros(img.shape[:2], dtype=np.float32)
        for ann in anns:
            m = self.coco.annToMask(ann)
            mask_2048 = np.maximum(mask_2048, m)

        mask = (mask_2048 > 0.5).astype(np.float32)

        # 1px-dilated spine target for skeleton recall
        spine = rasterize_dilated_spine(anns, mask_2048)

        if self.transforms is not None:
            aug = self.transforms(image=img_3ch, mask=mask, spine=spine)
            img_3ch = aug['image']
            mask = aug['mask']
            spine = aug['spine']

        if isinstance(img_3ch, np.ndarray):
            img_3ch = torch.from_numpy(img_3ch).permute(2, 0, 1).float()
        else:
            img_3ch = img_3ch.float()

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).unsqueeze(0).float()
        else:
            mask = mask.unsqueeze(0).float()

        if isinstance(spine, np.ndarray):
            spine = torch.from_numpy(spine).unsqueeze(0).float()
        else:
            spine = spine.unsqueeze(0).float()

        return img_3ch, mask, spine


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
    ], additional_targets={'spine': 'mask'})


train_transforms = build_morden_train_transforms()
val_transforms = A.Compose([ToTensorV2()], additional_targets={'spine': 'mask'})


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


class CognitiveRefinerV6(nn.Module):
    """
    Topology-preserving refiner with dilated convolutions (d=2,4,8),
    InstanceNorm2d, FiLM modulation, and magnetic proxy gating.
    """
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
        self.out_conv = nn.Conv2d(in_ch, 1, 1)

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
        return self.out_conv(bridged * g)


class MagAgentU2Net(nn.Module):
    """
    V7 Core Network:
    - Trainable HRNet-W18 encoder initialized from scratch on MAGFiLO
    - Shallow High-Resolution Stem & Branch Blocks
    - MORDEN multiscale decoder (5x5 convs, InstanceNorm2d, APP, PAG)
    - Intermediate Auxiliary Head + FiLM Generator
    - CognitiveRefinerV6 for anti-fragmentation
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

        # 4. Auxiliary Head & FiLM Generator
        self.aux_head = nn.Conv2d(fpn_ch, out_ch, 1)
        self.film_head = nn.Sequential(
            nn.Linear(fpn_ch, refiner_ch * 2),
            nn.Tanh(),
        )
        self.cognitive = CognitiveRefinerV6(refiner_ch)

    def forward(self, x, return_aux=False):
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

        aux_logits = self.aux_head(d8)
        pooled_context = F.adaptive_avg_pool2d(d8, 1).flatten(1)
        gamma, beta = torch.chunk(self.film_head(pooled_context).unsqueeze(-1).unsqueeze(-1), 2, dim=1)

        d8_up4 = F.interpolate(d8, size=(H4, W4), mode='bilinear', align_corners=False)
        gated_s4 = self.pag_s4(p_s4, d8_up4)
        d4 = self.dec_s4(torch.cat([gated_s4, b_s4], dim=1))
        d4 = d4 + self.app_s4(d4, target_size=(H4, W4))

        d4_up1 = F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=False)
        d1 = self.dec_s1(torch.cat([d4_up1, s1_embed], dim=1))

        vlm_prior = torch.sigmoid(F.interpolate(aux_logits, size=(H, W), mode='bilinear', align_corners=False))
        out = self.cognitive(d1, mag_map, vlm_prior=vlm_prior, gamma=gamma, beta=beta)

        if return_aux:
            return out, aux_logits
        return out


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


class DiceBCELoss(nn.Module):
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

    def get_lovasz_weight(self, epoch=1):
        if self.lovasz_max_weight <= 0:
            return 0.0
        if self.lovasz_warmup_epochs <= 1:
            return float(self.lovasz_max_weight)
        progress = float(epoch - 1) / float(self.lovasz_warmup_epochs - 1)
        return float(self.lovasz_max_weight * min(1.0, max(0.0, progress)))

    def forward(self, logits, targets, spine_targets=None, aux_logits=None, epoch=1):
        with torch.amp.autocast('cuda', enabled=False):
            logits_f = logits.float()
            targets_f = targets.float()

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

            # 4. Skeleton Recall Loss on 1px-dilated spine
            if spine_targets is not None and self.skel_weight > 0:
                sflat = spine_targets.float().reshape(-1)
                skel_recall = ((iflat * sflat).sum() + smooth) / (sflat.sum() + smooth)
                total_loss = total_loss + self.skel_weight * (1.0 - skel_recall)

            # 5. Intermediate Auxiliary Supervision
            if aux_logits is not None and self.aux_weight > 0:
                aux_target = F.adaptive_max_pool2d(targets_f, output_size=aux_logits.shape[2:])
                aux_loss = self.bce_mean(aux_logits.float(), aux_target)
                total_loss = total_loss + self.aux_weight * aux_loss

            return total_loss


# %% [code]
# CELL 7: Training & Validation Harness

def base_model(model):
    """Returns the original module when torch.compile wraps it."""
    return getattr(model, '_orig_mod', model)


def unpack_batch(batch):
    images, masks = batch[0], batch[1]
    spines = batch[2] if len(batch) >= 3 else None
    return images, masks, spines


def train_epoch(model, loader, optimizer, criterion, epoch=1):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    remainder = len(loader) % Config.ACCUM_STEPS

    for step, batch in enumerate(tqdm(loader, desc=f"Training Epoch {epoch}")):
        images, masks, spines = unpack_batch(batch)
        images = images.to(Config.DEVICE, non_blocking=True)
        masks = masks.to(Config.DEVICE, non_blocking=True)
        if spines is not None:
            spines = spines.to(Config.DEVICE, non_blocking=True)

        divisor = remainder if remainder and step >= len(loader) - remainder else Config.ACCUM_STEPS

        with torch.autocast(device_type='cuda', dtype=Config.AMP_DTYPE):
            logits, aux_logits = model(images, return_aux=True)
            loss = criterion(logits, masks, spine_targets=spines, aux_logits=aux_logits, epoch=epoch)
            loss = loss / divisor

        loss.backward()

        if (step + 1) % Config.ACCUM_STEPS == 0 or step + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss += loss.item() * divisor

    return epoch_loss / len(loader)


def validate(model, loader, criterion, epoch=1):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images, masks, spines = unpack_batch(batch)
            images = images.to(Config.DEVICE, non_blocking=True)
            masks = masks.to(Config.DEVICE, non_blocking=True)
            if spines is not None:
                spines = spines.to(Config.DEVICE, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=Config.AMP_DTYPE):
                logits, aux_logits = model(images, return_aux=True)
                loss = criterion(logits, masks, spine_targets=spines, aux_logits=aux_logits, epoch=epoch)

            epoch_loss += loss.item()

    return epoch_loss / len(loader)


# %% [code]
# CELL 8: Post-Processing Cascade (DenseCRF -> Hole Filling -> DBSCAN) & Submission

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


def fill_fragment_holes(mask_u8):
    """Fills internal holes to ensure each filament is a simply connected domain."""
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask_u8, dtype=np.uint8)
    if contours:
        cv2.drawContours(filled, contours, -1, color=1, thickness=cv2.FILLED)
    return filled


def cluster_fragments_dbscan(mask_u8, eps=Config.DBSCAN_EPS, min_pts=Config.DBSCAN_MIN_PTS, min_area=Config.MIN_AREA):
    """
    Integrates broken filament fragments into unified instance masks using DBSCAN/KD-Tree,
    preventing over-fragmentation that penalizes Panoptic Quality (PQ).
    """
    num_cc, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_cc <= 1:
        return []

    valid_ids = [i for i in range(1, num_cc) if stats[i, cv2.CC_STAT_AREA] >= 12]
    if not valid_ids:
        return []

    comp_contours = []
    comp_areas = {}
    for cid in valid_ids:
        comp_mask = (labels_cc == cid).astype(np.uint8)
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = np.vstack([c.reshape(-1, 2) for c in cnts]) if cnts else np.argwhere(comp_mask > 0)[:, ::-1]
        comp_contours.append((cid, pts))
        comp_areas[cid] = int(stats[cid, cv2.CC_STAT_AREA])

    parent = {cid: cid for cid in valid_ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    trees = {cid: cKDTree(pts) for cid, pts in comp_contours}
    for idx_a in range(len(comp_contours)):
        cid_a, pts_a = comp_contours[idx_a]
        for idx_b in range(idx_a + 1, len(comp_contours)):
            cid_b, pts_b = comp_contours[idx_b]
            if find(cid_a) == find(cid_b):
                continue
            ax, ay, aw, ah = stats[cid_a, :4]
            bx, by, bw, bh = stats[cid_b, :4]
            if (ax > bx + bw + eps) or (bx > ax + aw + eps) or (ay > by + bh + eps) or (by > ay + ah + eps):
                continue
            dists, _ = trees[cid_b].query(pts_a, k=1, distance_upper_bound=eps)
            if np.any(np.isfinite(dists)):
                union(cid_a, cid_b)

    clusters = {}
    for cid in valid_ids:
        root = find(cid)
        clusters.setdefault(root, []).append(cid)

    min_cluster_area = max(int(min_pts), int(min_area))
    instances = []
    for cids in clusters.values():
        total_area = sum(comp_areas[c] for c in cids)
        if total_area >= min_cluster_area:
            inst_mask = np.isin(labels_cc, cids).astype(np.uint8)
            instances.append(inst_mask)
    return instances


def postprocess_probability_v6(
    probability,
    img_limb_2048=None,
    threshold=Config.THRESHOLD,
    min_area=Config.MIN_AREA,
    use_dbscan=Config.USE_DBSCAN,
):
    """
    3-stage mandatory post-processing cascade:
    1. 2K DenseCRF boundary refinement
    2. Thresholding + OpenCV Hole Filling
    3. DBSCAN fragment integration for Panoptic Quality (PQ)
    """
    if isinstance(probability, torch.Tensor):
        prob_2k = probability.detach().float().cpu().numpy().squeeze()
    else:
        prob_2k = np.asarray(probability, dtype=np.float32).squeeze()

    if img_limb_2048 is None:
        raise ValueError('DenseCRF requires the native limb-corrected image.')
    prob_2k = apply_densecrf_2k(prob_2k, img_limb_2048)

    mask_u8 = (prob_2k > threshold).astype(np.uint8)
    mask_u8 = fill_fragment_holes(mask_u8)

    if use_dbscan:
        return cluster_fragments_dbscan(mask_u8, eps=Config.DBSCAN_EPS, min_pts=Config.DBSCAN_MIN_PTS, min_area=min_area)

    num_cc, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    return [
        (labels_cc == i).astype(np.uint8)
        for i in range(1, num_cc)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]


def pycoco_rle_encode(img):
    """Encodes a binary mask to pycocotools RLE ASCII string."""
    rle = maskUtils.encode(np.asfortranarray(img.astype(np.uint8)))
    return rle['counts'].decode('utf-8')


def generate_submission(model, test_paths, out_path):
    """Runs inference on test images and generates Kaggle submission.csv."""
    model.eval()
    submissions = []

    with torch.no_grad():
        for img_path in tqdm(test_paths, desc="Generating Submission"):
            base_id = os.path.basename(img_path).split('.')[0]
            img = load_image(img_path)

            prepared_3ch, img_limb_2048 = prepare_input_3ch(img)
            if prepared_3ch.shape[:2] != (Config.IMG_SIZE, Config.IMG_SIZE):
                raise ValueError(f'Expected {Config.IMG_SIZE}x{Config.IMG_SIZE} image, got {prepared_3ch.shape[:2]}.')
            tensor = torch.from_numpy(prepared_3ch).permute(2, 0, 1).unsqueeze(0).to(Config.DEVICE, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=Config.AMP_DTYPE):
                logits = model(tensor)

            prob_2048 = torch.sigmoid(logits).squeeze()

            instances = postprocess_probability_v6(
                prob_2048,
                img_limb_2048=img_limb_2048,
                threshold=Config.THRESHOLD,
                min_area=Config.MIN_AREA,
                use_dbscan=Config.USE_DBSCAN
            )

            for i, inst_mask in enumerate(instances, start=1):
                rle_str = pycoco_rle_encode(inst_mask)
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

    valid_img_ids = all_img_ids

    print(f"Valid training images on disk: {len(valid_img_ids)}")
    print(f"Total ground truth annotations: {len(coco.dataset.get('annotations', []))}")

    # 2. Train / Val Split (80 / 20)
    random.seed(Config.SEED)
    random.shuffle(valid_img_ids)
    split_idx = int(0.8 * len(valid_img_ids))
    train_ids = valid_img_ids[:split_idx]
    val_ids = valid_img_ids[split_idx:]

    print(f"Training images: {len(train_ids)} | Validation images: {len(val_ids)}")

    # 3. Datasets and Loaders
    train_ds = SolarDataset(coco, train_ids, Config.TRAIN_DIR, Config.IMG_SIZE, train_transforms)
    val_ds = SolarDataset(coco, val_ids, Config.TRAIN_DIR, Config.IMG_SIZE, val_transforms)

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
    criterion = DiceBCELoss()

    # 6. Training Loop with Early Best Model Saving
    best_loss = float('inf')
    model_path = os.path.join(Config.OUT_DIR, 'best_model.pth')

    for epoch in range(1, Config.EPOCHS + 1):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch}/{Config.EPOCHS}")
        print(f"{'='*40}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch=epoch)
        val_loss = validate(model, val_loader, criterion, epoch=epoch)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(base_model(model).state_dict(), model_path)
            print(f"✓ Saved new best model checkpoint! (val_loss: {val_loss:.4f})")

    # 7. Generate Submission on Test Images
    print("\n" + "=" * 60)
    print("Loading best checkpoint for Kaggle submission...")
    base_model(model).load_state_dict(torch.load(model_path, map_location=Config.DEVICE, weights_only=True))

    test_imgs = sorted(glob.glob(os.path.join(Config.TEST_DIR, '*.jpeg')))
    print(f"Found {len(test_imgs)} test images.")

    submission_path = os.path.join(Config.OUT_DIR, 'submission.csv')
    generate_submission(model, test_imgs, submission_path)
    print("=" * 60)
    print("V7 PIPELINE COMPLETE!")
    print(f"Submission written to: {submission_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
