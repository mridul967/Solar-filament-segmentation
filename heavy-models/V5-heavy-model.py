# V5: unchanged model architecture, guarded CUDA execution and recoverable training.
# See docs/V5-Kaggle-runbook.md for modes, checkpoint migration and diagnostics.

# Kaggle setup cell if needed: %pip install albumentations pycocotools -q
# This file is also ordinary Python: python V5-heavy-model.py

import os
import glob
import random
import json
import hashlib
import inspect
import platform
import subprocess
import time
import uuid
from pathlib import Path
from collections import defaultdict, OrderedDict

# Set before importing torch / initializing CUDA. Restart after a device fault.
if os.getenv('V5_LAUNCH_BLOCKING', '0') == '1':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ.setdefault('NO_ALBUMENTATIONS_UPDATE', '1')
import numpy as np
import pandas as pd
import cv2
cv2.setNumThreads(1)
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T10:17:42.375853Z","iopub.execute_input":"2026-10-17T10:17:42.376262Z","iopub.status.idle":"2026-10-17T10:17:42.388188Z","shell.execute_reply.started":"2026-10-17T10:17:42.376225Z","shell.execute_reply":"2026-10-17T10:17:42.387494Z"}}
# CELL 2: Configuration class with exact Kaggle dataset paths
# These paths were verified from the directory scan output:
#   /kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/
#       train/train_images/       -> .jpeg training images
#       train/MAGFiLO_1.0_Annotations_kaggle2026_train.json -> ground truth masks
#       test/test_images/         -> .jpeg test images

class Config:
    # === EXACT PATHS (verified from directory scan) ===
    BASE_DIR = '/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026'
    TRAIN_DIR = os.path.join(BASE_DIR, 'train/train_images')
    TEST_DIR = os.path.join(BASE_DIR, 'test/test_images')
    ANNOTATIONS_FILE = os.path.join(BASE_DIR, 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    OUT_DIR = os.getenv('V5_OUT_DIR', '/kaggle/working/')
    
    # === IMAGE DIMENSIONS ===
    # GONG H-alpha images are 2048x2048.
    # We downscale to 1024x1024 for training (to fit GPU VRAM),
    # then upscale predictions back to 2048x2048 for submission.
    IMG_SIZE = 1024
    ORIGINAL_SIZE = 2048
    
    # === TRAINING HYPERPARAMETERS ===
    # Two T4s receive one image each; two optimizer steps are accumulated so
    # the effective batch size remains 4 without changing the architecture.
    BATCH_SIZE = 2
    ACCUM_STEPS = 2
    EPOCHS = 25
    LR = 2e-4             # Learning rate (Adam optimizer)
    WEIGHT_DECAY = 1e-4   # L2 regularization
    NUM_WORKERS = int(os.getenv('V5_WORKERS', '2'))
    GPU = int(os.getenv('V5_GPU', '0'))
    DEVICE = f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu'
    
    # === MODEL PARAMETERS ===
    IN_CHANNELS = 3       # [Limb-corrected H-alpha, Filament Valley Map, Magnetic Proxy Map]
    OUT_CHANNELS = 1      # Single binary segmentation mask
    SEED = 42
    SAM2_VARIANT = 'tiny' # 'tiny' (facebook/sam2-hiera-tiny) or 'base_plus' (sam2_hiera_base_plus)
    VLM_MODEL_NAME = (
        'facebook/sam2-hiera-base-plus'
        if SAM2_VARIANT == 'base_plus'
        else 'facebook/sam2-hiera-tiny'
    )
    MODEL_WEIGHTS_NAME = 'best_model.pth'
    VLM_WEIGHTS_NAME = 'best_vlm.pth'

    # Architecture-preserving runtime controls. One-variable diagnosis: restart
    # the Kaggle kernel between variants, especially after any CUDA fault.
    MODE = os.getenv('V5_MODE', 'train')  # train, smoke, diagnose, benchmark, evaluate, submit
    USE_DP = os.getenv('V5_USE_DP', '1') == '1'
    MHA_FASTPATH = os.getenv('V5_MHA_FASTPATH', '0') == '1'
    EVAL_SERIAL = os.getenv('V5_EVAL_SERIAL', '0') == '1'
    EVAL_AMP = os.getenv('V5_EVAL_AMP', '1') == '1'
    BRIDGE_AMP = os.getenv('V5_BRIDGE_AMP', '0') == '1'
    CUDNN = os.getenv('V5_CUDNN', '1') == '1'
    CUDNN_BENCHMARK = os.getenv('V5_CUDNN_BENCHMARK', '0') == '1'
    SDP_BACKEND = os.getenv('V5_SDP', 'math')
    DEBUG_SYNC = os.getenv('V5_DEBUG_SYNC', '0') == '1'
    PREFLIGHT = os.getenv('V5_PREFLIGHT', '1') == '1'
    PERSISTENT_WORKERS = os.getenv('V5_PERSISTENT_WORKERS', '0') == '1'
    VAL_CACHE_ITEMS = int(os.getenv('V5_VAL_CACHE_ITEMS', '8'))  # bounded, per worker
    PROGRESS_SECONDS = 10.0
    PROFILE_WARMUP = 3
    PROFILE_BATCHES = int(os.getenv('V5_PROFILE_BATCHES', '12'))
    PROFILE_EPOCH = int(os.getenv('V5_PROFILE_EPOCH', '1'))
    CAPTURE_BATCHES = int(os.getenv('V5_CAPTURE_BATCHES', '2'))
    RESUME_PATH = os.getenv('V5_RESUME', '')
    CHECKPOINT_PATH = os.getenv('V5_CHECKPOINT', '')
    REPLAY_BATCH = os.getenv('V5_REPLAY_BATCH', '')
    PRETRAINED_SOURCE = os.getenv('V5_PRETRAINED', '')
    PRETRAINED_MAP = os.getenv('V5_PRETRAINED_MAP', '')  # JSON destination -> source keys
    VAL_LOSS_EPOCH = 5  # fixed full objective; never change with training warmup
    NATIVE_EVAL_LIMIT = int(os.getenv('V5_NATIVE_EVAL_LIMIT', '0'))  # 0 = entire holdout
    NATIVE_EVAL_AFTER_TRAIN = os.getenv('V5_NATIVE_EVAL', '1') == '1'
    EXPECTED_TEST_IMAGES = int(os.getenv('V5_EXPECTED_TEST_IMAGES', '180'))
    PQ_IOU_THRESHOLD = 0.5

    # === MULTI-SCALE DATASET & TILING PARAMETERS ===
    CROP_PROB = 0.50              # 50% full-disk 1024x1024 views, 50% native 1024x1024 crops from 2048x2048
    FILAMENT_CENTER_PROB = 0.70   # Within native crops: 70% filament-centered (with jitter), 30% disk/limb bg
    CROP_JITTER = 256             # Random spatial jitter around filament pixels in native crops
    TILE_STRIDE = 512             # 3x3 sliding window of 1024x1024 native tiles (50% overlap)
    GAUSSIAN_SIGMA = 256.0        # 2D Gaussian window sigma for seamless tile stitching
    GLOBAL_GATE_THRESH = 0.05     # Hard global probability cutoff (preserved)
    LOCAL_BLEND_WEIGHT = 0.85     # High-resolution P_local weight
    GLOBAL_BLEND_WEIGHT = 0.15    # Global full-disk P_global weight

    # === TOPOLOGY-PRESERVING COMPOUND LOSS PARAMETERS ===
    DICE_WEIGHT = 1.0
    FOCAL_WEIGHT = 1.0
    FOCAL_ALPHA = 0.50            # Neutral foreground/background polarity (configurable 0.50-0.65)
    FOCAL_GAMMA = 2.0
    LOVASZ_MAX_WEIGHT = 0.50      # Target LovaszHinge weight after linear warmup
    LOVASZ_WARMUP_EPOCHS = 5      # Linearly warm up w_lovasz from 0.0 to 0.50 across epochs 1 to 5
    SKEL_RECALL_WEIGHT = 0.25     # 1px-dilated morphological centerline recall weight
    VLM_PRIOR_WEIGHT = 0.15
    
    # === SUBMISSION PARAMETERS ===
    MIN_AREA = 200        # Minimum pixel area to count as a valid filament (filters noise)
    THRESHOLD = 0.4       # Probability threshold for binary mask
    USE_TTA = os.getenv('V5_TTA', '1') == '1'  # global TTA; native tiles remain unaugmented

# Verify paths exist
print(f"Train dir exists: {os.path.exists(Config.TRAIN_DIR)}")
print(f"Test dir exists:  {os.path.exists(Config.TEST_DIR)}")
print(f"Annotations exist: {os.path.exists(Config.ANNOTATIONS_FILE)}")
print(f"Train images: {len(glob.glob(os.path.join(Config.TRAIN_DIR, '*.jpeg')))}")
print(f"Test images:  {len(glob.glob(os.path.join(Config.TEST_DIR, '*.jpeg')))}")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T10:17:42.390316Z","iopub.execute_input":"2026-10-17T10:17:42.390722Z","iopub.status.idle":"2026-10-17T10:17:42.407983Z","shell.execute_reply.started":"2026-10-17T10:17:42.390676Z","shell.execute_reply":"2026-10-17T10:17:42.407201Z"}}
# CELL 3: Preprocessing Engine
# Two key functions:
# 1. remove_limb_darkening(): Removes the darkening effect at the Sun's edge
#    by fitting a radial mask and normalizing pixel intensities.
#    This prevents the network from confusing limb darkening with filaments.
#
# 2. get_magnetic_proxy(): Computes the Structure Tensor coherence map.
#    Filaments form along magnetic Polarity Inversion Lines (PILs).
#    Since we can't use actual magnetogram data (competition rules),
#    we extract directional/anisotropic features from the H-alpha image
#    to create a "pseudo-magnetic" orientation map.
#    High coherence = filament-like structure (elongated, directional)
#    Low coherence = isotropic noise (sunspots, granulation)

def load_image(path):
    """Load image handling both 16-bit and 8-bit formats."""
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Cannot decode image: {path}')
    if img.ndim != 2 or img.size == 0 or not np.isfinite(img).all():
        raise ValueError(f'Invalid grayscale image: {path}, shape={img.shape}')
    return img

_DISK_MASK_CACHE = {}

def remove_limb_darkening(img):
    """
    Removes limb darkening by creating a circular mask around the solar disk
    and normalizing intensities within the disk.
    
    The Sun exhibits limb darkening: brightness decreases from center to edge.
    Dark edges can be mistaken for filaments, so we mask them out.
    
    Returns:
        img_norm: Normalized image with limb darkening removed
        mask: Binary mask of the solar disk
    """
    h, w = img.shape
    mask = _DISK_MASK_CACHE.get((h, w))
    if mask is None:
        Y, X = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        r = np.sqrt((X - cx)**2 + (Y - cy)**2)
        max_r = np.min([cx, cy]) * 0.98  # 98% of radius to cleanly cut the limb
        mask = r < max_r
        _DISK_MASK_CACHE[(h, w)] = mask
    
    # Normalize within the mask in float32 (avoids float64 promotion)
    img_f = img.astype(np.float32, copy=False)
    valid_pixels = img_f[mask]
    img_min = np.float32(np.min(valid_pixels))
    img_max = np.float32(np.max(valid_pixels))
    
    img_norm = (img_f - img_min) / (img_max - img_min + np.float32(1e-8))
    img_norm[~mask] = 0.0
    return img_norm, mask

def get_magnetic_proxy(img):
    """
    Computes a 'pseudo-magnetic field' orientation map using the Structure Tensor.
    
    The Structure Tensor captures local gradient direction and strength.
    The 'coherence' (anisotropy) measures how directional a region is:
    - Filaments are elongated -> HIGH coherence
    - Sunspots/noise are round -> LOW coherence
    
    This map guides the Attention Gates to focus on magnetically-aligned structures.
    
    Returns:
        coherence: Normalized 0-1 map where bright = filament-like structure
    """
    img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    
    # Compute image gradients (Sobel operators)
    dx = cv2.Sobel(img_uint8, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(img_uint8, cv2.CV_32F, 0, 1, ksize=3)
    
    # Structure Tensor components (blurred to capture local neighborhood)
    ksize, sigma = 15, 5
    Jxx = cv2.GaussianBlur(dx**2, (ksize, ksize), sigma)
    Jxy = cv2.GaussianBlur(dx * dy, (ksize, ksize), sigma)
    Jyy = cv2.GaussianBlur(dy**2, (ksize, ksize), sigma)
    
    # Coherence (Anisotropy) = sqrt((Jxx - Jyy)^2 + 4*Jxy^2)
    # This measures how "directional" the local structure is
    coherence = np.sqrt((Jxx - Jyy)**2 + 4 * Jxy**2)
    
    # Normalize to 0-1
    coherence = (coherence - coherence.min()) / (coherence.max() - coherence.min() + 1e-8)
    return coherence

def get_filament_valley_map(img_limb, disk_mask):
    """
    Computes a multi-scale dark-valley enhancement map inside the solar disk.
    Replaces the redundant duplicate H-alpha channel from the 0.31 baseline
    without clipping dark filament cores or amplifying background granulation.
    """
    bg_smooth = cv2.GaussianBlur(img_limb.astype(np.float32), (31, 31), 7)
    valley = np.clip(bg_smooth - img_limb, 0.0, None) * disk_mask
    v_max = valley[disk_mask].max() if np.any(disk_mask) else 1.0
    valley = valley / (v_max + 1e-8)
    valley[~disk_mask] = 0.0
    return valley.astype(np.float32)

def prepare_input_3ch(img):
    """Unified 3-channel preprocessing for train, val, and test inference."""
    img_f = img.astype(np.float32, copy=False)
    img_f = (img_f - np.float32(img_f.min())) / (np.float32(img_f.max() - img_f.min()) + np.float32(1e-8))
    img_limb, disk_mask = remove_limb_darkening(img_f)
    valley_map = get_filament_valley_map(img_limb, disk_mask)
    mag_map = get_magnetic_proxy(img_limb)
    mag_map[~disk_mask] = 0.0
    return np.stack([img_limb, valley_map, mag_map], axis=-1).astype(np.float32, copy=False), img_limb

def extract_morphological_centerline(mask_bin):
    """
    Fast single-pass CPU morphological centerline extraction via Euclidean Distance Transform
    ridge detection (< 2 ms in C++ OpenCV vs 32 iterative morphology loops).
    """
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
    """
    Precomputes 1px-dilated morphological centerline targets on CPU at 2048x2048.
    Uses annotated COCO spine centerlines (1px + 1px dilation = 3x3 cross/box) or
    morphological skeletonization fallback, then optionally resizes to img_size.
    """
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
    # 1px dilation around the 1px centerline
    dilate_kernel = np.ones((3, 3), dtype=np.uint8)
    spine_dilated = cv2.dilate(spine_1px, dilate_kernel, iterations=1).astype(np.float32)
    spine_dilated = np.minimum(spine_dilated, (mask_2048 > 0).astype(np.float32))
    if img_size is not None and (img_size != h or img_size != w):
        spine_resized = cv2.resize(spine_dilated, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        return (spine_resized > 0.15).astype(np.float32)
    return spine_dilated.astype(np.float32)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T10:17:42.408949Z","iopub.execute_input":"2026-10-17T10:17:42.409278Z","iopub.status.idle":"2026-10-17T10:17:42.431490Z","shell.execute_reply.started":"2026-10-17T10:17:42.409260Z","shell.execute_reply":"2026-10-17T10:17:42.430710Z"}}
# CELL 4: PyTorch Dataset class (FIXED FOR STRING IDs)
# FIX: Bypasses pycocotools' buggy loadImgs() and getAnnIds() functions
# by directly accessing the underlying coco.imgs and coco.imgToAnns dictionaries.
# This prevents the "isArrayLike" bug where strings are iterated character-by-character.

class SolarDataset(Dataset):
    def __init__(
        self, coco, img_ids, img_dir, img_size=1024, transforms=None,
        return_spine=False, training=False
    ):
        self.coco = coco
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.img_size = img_size
        self.transforms = transforms
        self.return_spine = return_spine
        self.training = training
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self._image_cache = OrderedDict()

    def __len__(self):
        return len(self.img_ids)

    def _sample_native_crop_coords(self, mask_2048, anns=None):
        """
        Samples top-left (y0, x0) for a native-scale 1024x1024 crop from a 2048x2048 image:
        - 70% centered on filament foreground pixels with random spatial jitter
        - 30% random solar disk / limb background to suppress sunspot & limb false positives
        """
        h, w = mask_2048.shape
        crop_sz = self.img_size
        max_y = max(0, h - crop_sz)
        max_x = max(0, w - crop_sz)
        if max_y == 0 and max_x == 0:
            return 0, 0

        if random.random() < Config.FILAMENT_CENTER_PROB:
            yc, xc = None, None
            # Fast O(1) foreground point sampling from annotation spine/polygon/bbox before scanning 4.2M pixels
            if anns:
                ann = anns[random.randrange(len(anns))]
                raw_spine = ann.get('spine')
                if isinstance(raw_spine, list) and len(raw_spine) >= 2:
                    idx_pt = random.randrange(len(raw_spine) // 2) * 2
                    xc, yc = int(raw_spine[idx_pt]), int(raw_spine[idx_pt + 1])
                elif isinstance(ann.get('bbox'), list) and len(ann['bbox']) == 4:
                    bx, by, bw, bh = ann['bbox']
                    xc, yc = int(bx + bw * 0.5), int(by + bh * 0.5)
            if yc is None or xc is None:
                fg_coords = np.argwhere(mask_2048 > 0)
                if len(fg_coords) > 0:
                    yc, xc = fg_coords[random.randrange(len(fg_coords))]
            if yc is not None and xc is not None:
                jitter = getattr(Config, 'CROP_JITTER', 256)
                dy = random.randint(-jitter, jitter)
                dx = random.randint(-jitter, jitter)
                y0 = int(np.clip(yc - crop_sz // 2 + dy, 0, max_y))
                x0 = int(np.clip(xc - crop_sz // 2 + dx, 0, max_x))
                return y0, x0

        # 30% random solar disk / limb background crop
        cy, cx = h // 2, w // 2
        radius = min(cx, cy) * 0.95
        angle = random.uniform(0.0, 2.0 * np.pi)
        dist = radius * np.sqrt(random.uniform(0.0, 1.0))
        yc = int(cy + dist * np.sin(angle))
        xc = int(cx + dist * np.cos(angle))
        y0 = int(np.clip(yc - crop_sz // 2, 0, max_y))
        x0 = int(np.clip(xc - crop_sz // 2, 0, max_x))
        return y0, x0

    def __getitem__(self, idx):
        # Epoch/index seeds make crop + augmentation replay independent of worker
        # scheduling, including persistent workers and a resumed epoch boundary.
        if self.training:
            seed = (Config.SEED + int(self.epoch) * 1_000_003 + idx) % (2**32)
            random.seed(seed)
            np.random.seed(seed)
            if self.transforms is not None:
                self.transforms.set_random_seed(seed)
        img_id = self.img_ids[idx]
        
        # FIX 1: Direct dictionary access (avoids loadImgs string bug)
        img_info = self.coco.imgs[img_id]
        file_name = img_info['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        
        # Load and normalize image
        img = load_image(img_path) if self.training or file_name not in self._image_cache else None
        if img is not None and img.shape != (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE):
            raise ValueError(f'Unexpected image dimensions: {img_path}: {img.shape}')
        
        # FIX 2: Direct dictionary access for annotations
        # coco.imgToAnns maps image_id directly to a list of annotation dicts
        anns = self.coco.imgToAnns.get(img_id, [])
        
        # Create combined binary mask at original 2048x2048 resolution in uint8 (4x less memory bandwidth)
        mask_u8 = np.zeros((Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE), dtype=np.uint8)
        for ann in anns:
            # annToMask handles the polygon_list format perfectly
            mask_u8 = np.maximum(mask_u8, self.coco.annToMask(ann))
        mask_2048 = mask_u8.astype(np.float32)
        
        # Precompute 1px-dilated morphological centerline on CPU at 2048x2048
        spine_2048 = rasterize_dilated_spine(anns, mask_2048, img_size=None)
        
        # Multi-Scale Sampling: 50% full-disk 1024x1024 views, 50% native 1024x1024 crops
        # A 32px halo covers the filters; patch-local normalization is preserved.
        if self.training and random.random() < Config.CROP_PROB:
            y0, x0 = self._sample_native_crop_coords(mask_u8, anns=anns)
            img_f = img.astype(np.float32, copy=False)
            img_f = (img_f - np.float32(img_f.min())) / (np.float32(img_f.max() - img_f.min()) + np.float32(1e-8))
            img_limb, disk_mask = remove_limb_darkening(img_f)
            pad = 32
            h, w = img_limb.shape
            yp0, xp0 = max(0, y0 - pad), max(0, x0 - pad)
            yp1, xp1 = min(h, y0 + self.img_size + pad), min(w, x0 + self.img_size + pad)
            limb_patch = img_limb[yp0:yp1, xp0:xp1]
            disk_patch = disk_mask[yp0:yp1, xp0:xp1]
            valley_patch = get_filament_valley_map(limb_patch, disk_patch)
            mag_patch = get_magnetic_proxy(limb_patch)
            mag_patch[~disk_patch] = 0.0
            dy, dx = y0 - yp0, x0 - xp0
            img_3ch = np.stack([
                limb_patch[dy:dy + self.img_size, dx:dx + self.img_size],
                valley_patch[dy:dy + self.img_size, dx:dx + self.img_size],
                mag_patch[dy:dy + self.img_size, dx:dx + self.img_size],
            ], axis=-1).astype(np.float32, copy=False)
            mask = (mask_2048[y0:y0 + self.img_size, x0:x0 + self.img_size] > 0.5).astype(np.float32)
            spine = (spine_2048[y0:y0 + self.img_size, x0:x0 + self.img_size] > 0.15).astype(np.float32)
        else:
            if not self.training and file_name in self._image_cache:
                img_3ch = self._image_cache.pop(file_name)
                self._image_cache[file_name] = img_3ch
                img_3ch = img_3ch.copy()
            else:
                img_3ch_2048, _ = prepare_input_3ch(img)
                img_3ch = cv2.resize(img_3ch_2048, (self.img_size, self.img_size))
                if not self.training and Config.VAL_CACHE_ITEMS > 0:
                    self._image_cache[file_name] = img_3ch.copy()
                    while len(self._image_cache) > Config.VAL_CACHE_ITEMS:
                        self._image_cache.popitem(last=False)
            mask = cv2.resize(mask_2048, (self.img_size, self.img_size))
            mask = (mask > 0.5).astype(np.float32)
            spine = cv2.resize(spine_2048, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            spine = (spine > 0.15).astype(np.float32)
            # Re-dilate 1px on CPU if full-disk downsampling thinned any centerline segment
            if spine.max() > 0:
                spine = cv2.dilate(spine.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)
        spine = np.maximum(spine * mask, 0.0).astype(np.float32)
        
        # Apply augmentations
        if self.transforms:
            aug = self.transforms(image=img_3ch, mask=mask, spine=spine)
            img_3ch = aug['image']
            mask = aug['mask']
            spine = aug['spine']
        
        # Ensure correct tensor shapes regardless of whether transforms were applied
        # After ToTensorV2(): img is already (C, H, W), mask is (H, W)
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
        
        if self.return_spine:
            return img_3ch, mask, spine, str(img_id)
        return img_3ch, mask

# Define augmentation pipelines (Keep these below the class)
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    ToTensorV2()
], additional_targets={'spine': 'mask'})

val_transforms = A.Compose([
    ToTensorV2()
], additional_targets={'spine': 'mask'})

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T10:17:42.432413Z","iopub.execute_input":"2026-10-17T10:17:42.432981Z","iopub.status.idle":"2026-10-17T10:17:42.454441Z","shell.execute_reply.started":"2026-10-17T10:17:42.432955Z","shell.execute_reply":"2026-10-17T10:17:42.453675Z"}}
# CELL 5: Mag-Agent U²-Net Model Architecture
# Based on the paper "Automated High-Precision Recognition of Solar Filaments
# Based on an Improved U2-Net" (Jiang & Li, 2024)
#
# Architecture components:
# 1. AttentionGate: Suppresses irrelevant features, enhances filament response
# 2. RSU (Residual U-block): Multi-scale feature extraction (from U²-Net)
# 3. CognitiveRefiner: Bridges fragmented filament pieces using magnetic map
# 4. MagAgentU2Net: Full encoder-decoder combining all components

# --- 1. Attention Gate ---
# From Attention U-Net (Oktay et al.)
# Receives low-level features (encoder) and high-level features (decoder).
# Computes attention weights to suppress background and enhance filaments.
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Channels of gating signal (from decoder)
            F_l: Channels of input features (from encoder skip connection)
            F_int: Intermediate channels for attention computation
        """
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal (from deeper decoder layer)
            x: Skip connection features (from encoder)
        Returns:
            Attention-weighted features
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # Sigmoid -> attention weights [0, 1]
        return x * psi  # Weight the skip connection features


# --- 2. RSU Block (Residual U-block) ---
# From U²-Net (Qin et al., 2020)
# A "small U-Net" inside the main U-Net. Captures multi-scale context.
# Each RSU has an encoder-decoder structure with residual connection.
class RSU(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        """
        Args:
            in_ch: Input channels
            mid_ch: Intermediate channels (bottleneck)
            out_ch: Output channels
        """
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, out_ch, 1)
        
        # Encoder path of the mini U-Net
        self.enc1 = nn.Sequential(
            nn.Conv2d(out_ch, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)
        )
        
        # Decoder path of the mini U-Net
        self.dec2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = nn.Sequential(
            nn.Conv2d(mid_ch * 2, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x_in = self.conv_in(x)
        
        # Encode
        e1 = self.enc1(x_in)
        e2 = self.enc2(self.pool(e1))
        
        # Decode
        d2 = self.dec2(e2)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        
        # Residual connection: local + multi-scale features
        return d1 + x_in


# --- 2.5 Custom SAM-inspired visual bridge (self.vlm_bridge) ---
# Four window stages + FPN + learned query decoder. The historical class/field
# names are retained for state-dict compatibility. This is not SAM 2's backbone
# and has no language encoder. Full-model checkpoints include the bridge.
class HieraWindowStage(nn.Module):
    """Hierarchical Vision Transformer (Hiera) stage with local windowed self-attention + FFN."""
    def __init__(self, in_ch, out_ch, stride=1, window_size=8, num_heads=4):
        super().__init__()
        self.proj = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=stride, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )
            if stride > 1 or in_ch != out_ch
            else nn.Identity()
        )
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(out_ch)
        self.attn = nn.MultiheadAttention(out_ch, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(out_ch)
        self.mlp = nn.Sequential(
            nn.Linear(out_ch, out_ch * 2),
            nn.GELU(),
            nn.Linear(out_ch * 2, out_ch),
        )
        self.dw_conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch, bias=False)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        ws = min(self.window_size, H, W)
        # Pad if H or W is not divisible by window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x_pad = F.pad(x, (0, pad_w, 0, pad_h)) if (pad_h or pad_w) else x
        _, _, Hp, Wp = x_pad.shape
        nh, nw = Hp // ws, Wp // ws

        # Partition into non-overlapping [B * nh * nw, ws * ws, C] windows
        win = x_pad.view(B, C, nh, ws, nw, ws).permute(0, 2, 4, 3, 5, 1).contiguous().reshape(B * nh * nw, ws * ws, C)
        win_norm = self.norm1(win).contiguous()
        attn_out, _ = self.attn(win_norm, win_norm, win_norm, need_weights=False)
        win = win + attn_out
        win = win + self.mlp(self.norm2(win))

        # Unpartition back to [B, C, H, W] + local depthwise positional residual
        out = win.view(B, nh, nw, ws, ws, C).permute(0, 5, 1, 3, 2, 4).contiguous().reshape(B, C, Hp, Wp)
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out + self.dw_conv(out)


class VLMSemanticBridge(nn.Module):
    SAM2_STAGE_CHANNELS = {
        'tiny': (96, 192, 384, 768),
        'sam2_hiera_tiny': (96, 192, 384, 768),
        'facebook/sam2-hiera-tiny': (96, 192, 384, 768),
        'base_plus': (112, 224, 448, 896),
        'sam2_hiera_base_plus': (112, 224, 448, 896),
        'facebook/sam2-hiera-base-plus': (112, 224, 448, 896),
    }

    def __init__(
        self,
        in_ch=3,
        e3_ch=256,
        vis_ch=512,
        refiner_ch=64,
        fpn_dim=256,
        num_queries=4,
        variant=None,
    ):
        super().__init__()
        variant_key = variant or getattr(Config, 'SAM2_VARIANT', 'tiny')
        c1, c2, c3, c4 = self.SAM2_STAGE_CHANNELS.get(variant_key, (96, 192, 384, 768))
        self.variant = variant_key
        self.stage_channels = (c1, c2, c3, c4)
        self.fpn_dim = fpn_dim

        # 1. Multi-Scale Hiera Backbone Stages [C1, C2, C3, C4] at strides [4, 8, 16, 32]
        # For 1024x1024 input -> [256x256, 128x128, 64x64, 32x32]
        self.stage1 = HieraWindowStage(in_ch, c1, stride=4, window_size=16, num_heads=4)
        self.stage2 = HieraWindowStage(c1, c2, stride=2, window_size=8, num_heads=4)
        self.stage3 = HieraWindowStage(c2, c3, stride=2, window_size=8, num_heads=8)
        self.stage4 = HieraWindowStage(c3, c4, stride=2, window_size=8, num_heads=8)

        # Skip adapters aligning MagAgentU2Net encoder (e3 @ stride 4, b @ stride 8) with C1, C2
        self.e3_to_c1 = nn.Conv2d(e3_ch, c1, 1, bias=False)
        self.b_to_c2 = nn.Conv2d(vis_ch, c2, 1, bias=False)

        # 2. Custom top-down FPN (projects [C1, C2, C3, C4] -> a 256-d pyramid)
        self.lat1 = nn.Conv2d(c1, fpn_dim, 1, bias=False)
        self.lat2 = nn.Conv2d(c2, fpn_dim, 1, bias=False)
        self.lat3 = nn.Conv2d(c3, fpn_dim, 1, bias=False)
        self.lat4 = nn.Conv2d(c4, fpn_dim, 1, bias=False)
        self.fpn_smooth1 = nn.Sequential(nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False), nn.BatchNorm2d(fpn_dim), nn.GELU())
        self.fpn_smooth2 = nn.Sequential(nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False), nn.BatchNorm2d(fpn_dim), nn.GELU())

        # 3. Random normalized anchors plus learned foreground/background queries
        gen = torch.Generator().manual_seed(42)
        fg_init = F.normalize(torch.randn(num_queries, fpn_dim, generator=gen), dim=-1)
        bg_init = F.normalize(torch.randn(num_queries, fpn_dim, generator=gen), dim=-1)
        self.register_buffer("pos_anchors", fg_init)
        self.register_buffer("neg_anchors", bg_init)
        self.fg_queries = nn.Parameter(torch.zeros(num_queries, fpn_dim))
        self.bg_queries = nn.Parameter(torch.zeros(num_queries, fpn_dim))

        # Two-Way Token-to-Image and Image-to-Token Cross-Attention (SAM 2 Mask Decoder style)
        self.token_to_img_attn = nn.MultiheadAttention(fpn_dim, num_heads=8, batch_first=True)
        self.img_to_token_attn = nn.MultiheadAttention(fpn_dim, num_heads=8, batch_first=True)

        # Residual fusion gates into MagAgentU2Net e3 (256-ch @ H/4) and bottleneck b (512-ch @ H/8)
        self.e3_gate = nn.Sequential(
            nn.Conv2d(fpn_dim, e3_ch, 1, bias=False),
            nn.BatchNorm2d(e3_ch),
            nn.Sigmoid(),
        )
        self.b_out = nn.Sequential(
            nn.Conv2d(fpn_dim + 2, vis_ch, 1, bias=False),
            nn.BatchNorm2d(vis_ch),
            nn.Sigmoid(),
        )
        # FiLM conditioning head for the full-resolution 64-ch CognitiveRefiner
        self.film_head = nn.Sequential(
            nn.Linear(fpn_dim, refiner_ch * 2),
            nn.Tanh(),
        )
        self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    @torch.no_grad()
    def _copy_matching_tensors(self, src_sd, source_label, mapping=None):
        """Load a native bridge strictly, or an explicitly reviewed key mapping.

        Equal shapes alone never authorize a pretrained transfer. This custom
        bridge is SAM-inspired; it is neither SAM 2's trunk nor a text encoder.
        """
        if not isinstance(src_sd, dict):
            raise ValueError('Expected a state dictionary')
        for key in ('model', 'state_dict'):
            if key in src_sd and isinstance(src_sd[key], dict):
                src_sd = src_sd[key]
                break
        dst = self.state_dict()
        if mapping is None:
            native = {k.removeprefix('module.').removeprefix('vlm_bridge.'): v
                      for k, v in src_sd.items()
                      if k.removeprefix('module.').startswith('vlm_bridge.')}
            if native:
                src_sd = native
            if set(src_sd) != set(dst):
                print(f'No verified native bridge mapping for {source_label}; '
                      'keeping intentional initialization (0 pretrained parameters).')
                return False
            mapping = {key: key for key in dst}
        if not isinstance(mapping, dict) or not mapping or len(set(mapping.values())) != len(mapping):
            raise ValueError('Pretrained mapping must be nonempty destination -> unique source keys')
        # Validate ALL mappings before mutating any weights.
        for target, source in mapping.items():
            if target not in dst or source not in src_sd:
                raise ValueError(f'Unknown pretrained key mapping: {target} <- {source}')
            value = src_sd[source]
            if not isinstance(value, torch.Tensor) or value.shape != dst[target].shape:
                raise ValueError(f'Pretrained shape mismatch: {target} <- {source}')
            if not torch.isfinite(value).all():
                raise ValueError(f'Nonfinite pretrained tensor: {source}')
        for target, source in mapping.items():
            dst[target].copy_(src_sd[source].to(dst[target]))
        parameters = dict(self.named_parameters())
        loaded = sum(parameters[k].numel() for k in mapping if k in parameters)
        total = sum(p.numel() for p in parameters.values())
        self.preload_provenance = {
            'source': str(source_label), 'mapping': mapping,
            'parameter_count': loaded, 'parameter_coverage': loaded / total,
            'missing_keys': sorted(set(dst) - set(mapping)),
            'unused_source_keys': sorted(set(src_sd) - set(mapping.values())),
        }
        print(f'Loaded {len(mapping)} explicitly mapped bridge tensors; '
              f'{loaded:,}/{total:,} parameters ({loaded / total:.2%}) from {source_label}')
        return True

    @torch.no_grad()
    def preload_sam2_hiera(self, model_name=None, device='cpu'):
        """Explicit optional initialization; no downloads or stale-file reuse by default."""
        source = model_name or Config.PRETRAINED_SOURCE
        if not source:
            print('Custom visual bridge: intentional initialization; no pretrained/text claim.')
            return False
        mapping = None
        if Config.PRETRAINED_MAP:
            with open(Config.PRETRAINED_MAP) as handle:
                mapping = json.load(handle)
        if os.path.isfile(source):
            if source.endswith('.safetensors'):
                from safetensors.torch import load_file
                state = load_file(source, device='cpu')
            else:
                state = torch.load(source, map_location='cpu', weights_only=True)
            if isinstance(state, dict) and state.get('kind') == 'bridge':
                state = state['state_dict']
            return self._copy_matching_tensors(state, source, mapping)
        if mapping is None:
            raise ValueError('An external SAM/HF model requires V5_PRETRAINED_MAP; '
                             'the custom bridge is not a compatible SAM 2 backbone.')
        from transformers import AutoModel
        backbone = AutoModel.from_pretrained(source).cpu().eval()
        try:
            return self._copy_matching_tensors(backbone.state_dict(), source, mapping)
        finally:
            del backbone

    def forward(self, b, x=None, e3=None):
        """
        Args:
            b: Bottleneck features [B, 512, H/8, W/8]
            x: Optional raw 3-channel input [B, 3, H, W] (1024x1024)
            e3: Optional stride-4 encoder skip [B, 256, H/4, W/4] (256x256)
        Returns:
            If e3 is provided: (e3_mod, b_mod, vlm_logits, gamma, beta)
            If e3 is None:     (b_mod, vlm_logits, gamma, beta)
        """
        with torch.autocast(device_type=b.device.type, dtype=torch.float16,
                            enabled=Config.BRIDGE_AMP and b.is_cuda):
            b = b.float()
            x = x.float() if x is not None else None
            e3 = e3.float() if e3 is not None else None
            B, _, Hb, Wb = b.shape
            H4, W4 = Hb * 2, Wb * 2

            # Extract multi-scale Hiera stages [C1, C2, C3, C4] at strides [4, 8, 16, 32]
            if x is not None:
                c1 = self.stage1(x)
            elif e3 is not None:
                c1 = self.stage1.dw_conv(self.e3_to_c1(e3))
            else:
                b_up = F.interpolate(b, size=(H4, W4), mode='bilinear', align_corners=False)
                c1 = self.stage1.dw_conv(self.b_to_c2(b_up)[:, :self.stage_channels[0]])

            if e3 is not None:
                if c1.shape[2:] != e3.shape[2:]:
                    c1 = F.interpolate(c1, size=e3.shape[2:], mode='bilinear', align_corners=False)
                c1 = c1 + self.e3_to_c1(e3)

            c2 = self.stage2(c1)
            if c2.shape[2:] != (Hb, Wb):
                c2 = F.interpolate(c2, size=(Hb, Wb), mode='bilinear', align_corners=False)
            c2 = c2 + self.b_to_c2(b)
            c3 = self.stage3(c2)
            c4 = self.stage4(c3)

            # Top-down FPN fusion across [C1, C2, C3, C4]
            p4 = self.lat4(c4)
            p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False)
            p2 = self.fpn_smooth2(
                self.lat2(c2) + F.interpolate(p3, size=c2.shape[2:], mode='bilinear', align_corners=False)
            )
            p1 = self.fpn_smooth1(
                self.lat1(c1) + F.interpolate(p2, size=c1.shape[2:], mode='bilinear', align_corners=False)
            )

            # SAM 2 Mask-Query Token Decoder at stride-8 FPN feature map P2
            fg_q = F.normalize(self.pos_anchors + self.fg_queries, dim=-1)  # [Q, D]
            bg_q = F.normalize(self.neg_anchors + self.bg_queries, dim=-1)  # [Q, D]
            all_queries = torch.cat([fg_q, bg_q], dim=0).unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, 2Q, D]

            img_tokens = p2.flatten(2).transpose(1, 2).contiguous()  # [B, Hb*Wb, D]
            refined_q, _ = self.token_to_img_attn(all_queries, img_tokens, img_tokens, need_weights=False)
            all_queries = (all_queries + refined_q).contiguous()
            attn_out, _ = self.img_to_token_attn(img_tokens, all_queries, all_queries, need_weights=False)
            attn_map = p2 + attn_out.transpose(1, 2).contiguous().reshape(B, self.fpn_dim, Hb, Wb)

            # Mask-Query dot-product similarity maps (foreground filament spines vs background sunspots)
            Q = fg_q.shape[0]
            fg_refined = F.normalize(all_queries[:, :Q, :], dim=-1)
            bg_refined = F.normalize(all_queries[:, Q:, :], dim=-1)
            p2_norm = F.normalize(attn_map, dim=1)
            scale = self.logit_scale.exp().clamp(1.0, 50.0)

            sim_fg = torch.einsum("bdhw,bqd->bqhw", p2_norm, fg_refined).mean(dim=1, keepdim=True) * scale
            sim_bg = torch.einsum("bdhw,bqd->bqhw", p2_norm, bg_refined).mean(dim=1, keepdim=True) * scale
            vlm_logits = sim_fg - sim_bg

            # Modulate bottleneck b [B, 512, H/8, W/8] and high-res skip e3 [B, 256, H/4, W/4]
            b_gate = self.b_out(torch.cat([attn_map, torch.sigmoid(sim_fg), torch.sigmoid(sim_bg)], dim=1))
            b_mod = b * (1.0 + b_gate)

            # Global query context -> FiLM (gamma, beta) for full-res CognitiveRefiner
            query_pool = all_queries.mean(dim=1)  # [B, D]
            gamma_beta = self.film_head(query_pool).unsqueeze(-1).unsqueeze(-1)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=1)

            if e3 is not None:
                e3_mod = e3 * (1.0 + 0.5 * self.e3_gate(p1))
                return e3_mod, b_mod, vlm_logits, gamma, beta
            return b_mod, vlm_logits, gamma, beta


# --- 3. Agentic Cognitive Layer (Topology Refiner) ---
# This is our custom addition to solve the Panoptic Quality fragmentation problem.
# Standard CNNs often break filaments into multiple pieces.
# This layer uses dilated convolutions to "see" larger context and
# bridges gaps between fragmented pieces, guided by the magnetic map and VLM prior.
class CognitiveRefiner(nn.Module):
    def __init__(self, in_ch):
        """
        Args:
            in_ch: Number of input channels (features from decoder)
        """
        super().__init__()
        # Dilated convolutions to capture long-range spine continuity
        # These "see" a larger receptive field to connect broken pieces
        self.bridge1 = nn.Conv2d(in_ch, in_ch, 3, dilation=2, padding=2, groups=in_ch)
        self.bridge2 = nn.Conv2d(in_ch, in_ch, 3, dilation=4, padding=4, groups=in_ch)
        self.bridge3 = nn.Conv2d(in_ch, in_ch, 3, dilation=8, padding=8, groups=in_ch)
        
        # Agent Reasoning Gate: uses magnetic map + VLM contrastive prior to decide WHERE to connect
        # Learned visual-query features condition connections alongside the magnetic proxy.
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch + 2, in_ch, 1),  # +1 magnetic map, +1 VLM semantic prior
            nn.Sigmoid()
        )
        self.out_conv = nn.Conv2d(in_ch, 1, 1)

    def forward(self, x, mag_map, vlm_prior=None, gamma=None, beta=None):
        """
        Args:
            x: Feature map from decoder [B, C, H, W]
            mag_map: Magnetic orientation map [B, 1, H, W]
            vlm_prior: Upsampled VLM contrastive probability map [B, 1, H, W]
            gamma, beta: Optional VLM FiLM conditioning parameters [B, C, 1, 1]
        Returns:
            Refined single-channel prediction [B, 1, H, W]
        """
        # Bridge gaps using multi-scale dilated convolutions
        b1 = F.relu(self.bridge1(x))
        b2 = F.relu(self.bridge2(x))
        b3 = F.relu(self.bridge3(x))
        bridged = x + b1 + b2 + 0.5 * b3  # Residual: original + local + medium + long bridge

        if gamma is not None and beta is not None:
            bridged = bridged * (1.0 + 0.25 * gamma) + 0.25 * beta

        if vlm_prior is None:
            vlm_prior = torch.full_like(mag_map, 0.5)
        
        # Gate based on magnetic alignment + VLM contrastive semantic probability
        # The agent "reasons": if magnetic flow & VLM filament concept connect two regions, merge them
        g = self.gate(torch.cat([bridged, mag_map, vlm_prior], dim=1))
        refined = bridged * g
        
        return self.out_conv(refined)


# --- 4. Full Mag-Agent U²-Net (V5 with VLM Semantic Bridge) ---
class MagAgentU2Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        """
        Args:
            in_ch: Input channels (3: H-alpha, Filament Valley Map, Magnetic Map)
            out_ch: Output channels (1: binary segmentation mask)
        """
        super().__init__()
        
        # === ENCODER ===
        self.enc1 = RSU(in_ch, 32, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = RSU(64, 64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = RSU(128, 128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # === BOTTLENECK ===
        self.bottleneck = RSU(256, 256, 512)
        self.vlm_bridge = VLMSemanticBridge(in_ch=in_ch, e3_ch=256, vis_ch=512, refiner_ch=64)
        
        # === DECODER with Attention Gates ===
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att3 = AttentionGate(512, 256, 256)
        self.dec3 = RSU(768, 256, 256)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att2 = AttentionGate(256, 128, 128)
        self.dec2 = RSU(384, 128, 128)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att1 = AttentionGate(128, 64, 64)
        self.dec1 = RSU(192, 64, 64)
        
        # === AGENTIC COGNITIVE LAYER ===
        self.cognitive = CognitiveRefiner(64)

    def forward(self, x, return_aux=False):
        """
        Args:
            x: Input tensor [B, 3, H, W]
            return_aux: If True, also returns bottleneck VLM contrastive logits
        Returns:
            Segmentation logits [B, 1, H, W] (or tuple (out, vlm_logits) when return_aux=True)
        """
        # Extract magnetic map (channel 2) for the Cognitive Layer
        mag_map = x[:, 2:3, :, :]
        
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck + SAM 2 Hiera Multi-Scale FPN & Mask-Query Conditioning
        b = self.bottleneck(self.pool3(e3))
        e3, b, vlm_logits, gamma, beta = self.vlm_bridge(b, x=x, e3=e3)
        
        # Decoder path with Attention Gates
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, self.att3(b, e3)], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self.att2(d3, e2)], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self.att1(d2, e1)], dim=1))
        
        # Upsample VLM semantic prior to full resolution for CognitiveRefiner
        vlm_prior = torch.sigmoid(
            F.interpolate(vlm_logits, size=d1.shape[2:], mode='bilinear', align_corners=False)
        )
        
        # Agentic Cognitive Refinement
        out = self.cognitive(d1, mag_map, vlm_prior=vlm_prior, gamma=gamma, beta=beta)
        
        if return_aux or getattr(self, '_dp_return_aux', False):
            return out, vlm_logits
        return out

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T10:17:42.455368Z","iopub.execute_input":"2026-10-17T10:17:42.455610Z","iopub.status.idle":"2026-10-17T10:17:42.470128Z","shell.execute_reply.started":"2026-10-17T10:17:42.455593Z","shell.execute_reply":"2026-10-17T10:17:42.469456Z"}}
# CELL 6: Topology-Preserving Compound Loss Function (DiceBCELoss)
# Total Loss = 1.0 * Dice + 1.0 * Focal(alpha=0.50, gamma=2.0) + w_lovasz * LovaszHinge + 0.25 * SkeletonRecall
# - Focal Polarity Fix: alpha=0.50 (neutral foreground/background balance; avoids alpha=0.25 thin-neck snapping
#   and avoids >0.75 1-2px halo merging between adjacent filaments).
# - FP16 AMP Stability: LovaszHinge (Berman et al., CVPR 2018) is computed strictly inside
#   `torch.amp.autocast('cuda', enabled=False)` with logits cast to `.float()`.
#   `w_lovasz` linearly warms up from 0.0 to 0.50 across epochs 1 to 5.
# - Skeleton Recall (Kirchhoff et al., ECCV 2024): Soft recall against 1px-dilated morphological
#   centerlines precomputed on CPU in SolarDataset to preserve spine continuity for Panoptic Quality (PQ).

def _lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors (Berman et al., CVPR 2018)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1e-6)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits_flat, labels_flat):
    """Binary Lovasz hinge loss on flattened 1D tensors in float32."""
    if logits_flat.numel() == 0:
        return logits_flat.sum() * 0.0
    signs = 2.0 * labels_flat.float() - 1.0
    errors = 1.0 - logits_flat * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels_flat[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge_loss(logits, targets, per_image=True):
    """
    Per-image Lovasz-Hinge loss strictly evaluated in FP32 (`autocast('cuda', enabled=False)`).
    Directly optimizes the submodular Jaccard / IoU surface for thin solar filament structures.
    """
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
        vlm_weight=Config.VLM_PRIOR_WEIGHT,
    ):
        super().__init__()
        self.bce_none = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_mean = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_alpha = float(np.clip(focal_alpha, 0.50, 0.65))
        self.focal_gamma = focal_gamma
        self.lovasz_max_weight = lovasz_max_weight
        self.lovasz_warmup_epochs = lovasz_warmup_epochs
        self.skel_weight = skel_weight
        self.vlm_weight = vlm_weight

    def get_lovasz_weight(self, epoch=1):
        """
        Linearly warms up w_lovasz from 0.0 to 0.50 across epochs 1 to 5
        (1-indexed epoch 1 -> 0.00, 2 -> 0.125, 3 -> 0.25, 4 -> 0.375, >=5 -> 0.50).
        """
        if self.lovasz_max_weight <= 0:
            return 0.0
        if self.lovasz_warmup_epochs <= 1:
            return float(self.lovasz_max_weight)
        ep = max(1, int(epoch))
        progress = float(ep - 1) / float(self.lovasz_warmup_epochs - 1)
        return float(self.lovasz_max_weight * min(1.0, max(0.0, progress)))

    def forward(self, logits, targets, spine_targets=None, vlm_logits=None, epoch=5,
                return_parts=False):
        """
        Args:
            logits: Raw model output (before sigmoid) [B, 1, H, W]
            targets: Ground truth binary mask [B, 1, H, W]
            spine_targets: Precomputed 1px-dilated morphological centerline mask [B, 1, H, W]
            vlm_logits: Optional bottleneck SAM 2 Hiera query logits [B, 1, H/8, W/8]
            epoch: 1-indexed training epoch (1..EPOCHS) for linear Lovasz warmup (defaults to 5 = full 0.50 weight)
        Returns:
            Combined topology-preserving loss scalar
        """
        with torch.amp.autocast('cuda', enabled=False):
            logits_f = logits.float()
            targets_f = targets.float()

            # 1. Soft Dice Loss (weight = 1.0)
            probs = torch.sigmoid(logits_f)
            smooth = 1.0
            iflat = probs.reshape(-1)
            tflat = targets_f.reshape(-1)
            intersection = (iflat * tflat).sum()
            dice_loss = 1.0 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

            # 2. Binary Focal Loss with neutral alpha=0.50, gamma=2.0 (weight = 1.0)
            bce_per_pixel = self.bce_none(logits_f, targets_f)
            p_t = probs * targets_f + (1.0 - probs) * (1.0 - targets_f)
            alpha_t = self.focal_alpha * targets_f + (1.0 - self.focal_alpha) * (1.0 - targets_f)
            focal_weight_map = alpha_t * torch.pow((1.0 - p_t).clamp_min(1e-6), self.focal_gamma)
            focal_loss = (focal_weight_map * bce_per_pixel).mean()

            total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
            zero = logits_f.new_zeros(())
            lovasz_loss, skeleton_loss, vlm_loss = zero, zero, zero

            # 3. Lovasz-Hinge Loss in strict FP32 with linear warmup across epochs 1..5
            w_lovasz = self.get_lovasz_weight(epoch)
            if w_lovasz > 0.0:
                lovasz_loss = lovasz_hinge_loss(logits_f, targets_f, per_image=True)
                total_loss = total_loss + w_lovasz * lovasz_loss

            # 4. Skeleton Recall Loss (weight = 0.25): soft recall on 1px-dilated centerlines
            if spine_targets is not None and self.skel_weight > 0:
                sflat = spine_targets.float().reshape(-1)
                skel_recall = ((iflat * sflat).sum() + smooth) / (sflat.sum() + smooth)
                skeleton_loss = 1.0 - skel_recall
                total_loss = total_loss + self.skel_weight * skeleton_loss

            # 5. Auxiliary SAM 2 Hiera Query Supervision
            if vlm_logits is not None and self.vlm_weight > 0:
                vlm_target = F.adaptive_max_pool2d(targets_f, output_size=vlm_logits.shape[2:])
                vlm_loss = self.bce_mean(vlm_logits.float(), vlm_target)
                total_loss = total_loss + self.vlm_weight * vlm_loss

        if return_parts:
            return total_loss, {k: v.detach() for k, v in {
                'dice': dice_loss, 'focal': focal_loss, 'lovasz': lovasz_loss,
                'skeleton': skeleton_loss, 'bridge': vlm_loss,
            }.items()}
        return total_loss

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T10:17:42.470986Z","iopub.execute_input":"2026-10-17T10:17:42.471330Z","iopub.status.idle":"2026-10-17T10:17:42.487967Z","shell.execute_reply.started":"2026-10-17T10:17:42.471300Z","shell.execute_reply":"2026-10-17T10:17:42.487125Z"}}
# CELL 7: Training and Validation Loops
# Uses Automatic Mixed Precision (AMP) for:
#   - FP16 execution where supported (measure actual throughput)
#   - Reduced activation storage; the bridge stays FP32 by default
# Gradient clipping prevents exploding gradients with BCE loss.

def forward_model(model, images, return_aux=False):
    """
    Safe forward wrapper for nn.DataParallel:
    1. When batch_size < num_gpus (e.g., B=1 on the last validation batch or test inference),
       runs directly on the primary module.
    2. Even when B >= num_gpus, avoids passing keyword arguments (`**kwargs`) through DataParallel
       by temporarily toggling an attribute on the unwrapped module.
    """
    if not model.training and Config.EVAL_SERIAL:
        unwrapped = unwrap_model(model)
        outputs = [unwrapped(chunk, return_aux=return_aux) for chunk in images.split(1)]
        if return_aux:
            return tuple(torch.cat([output[i] for output in outputs]) for i in range(2))
        return torch.cat(outputs)
    if isinstance(model, nn.DataParallel):
        unwrapped = model.module
        if images.shape[0] < len(model.device_ids):
            return unwrapped(images, return_aux=return_aux)
        prev_flag = getattr(unwrapped, '_dp_return_aux', False)
        unwrapped._dp_return_aux = bool(return_aux)
        try:
            return model(images)
        finally:
            unwrapped._dp_return_aux = prev_flag
    return model(images, return_aux=return_aux)


def synchronize_devices():
    if str(Config.DEVICE).startswith('cuda'):
        devices = list(range(torch.cuda.device_count())) if Config.USE_DP else [Config.GPU]
        for device in devices:
            torch.cuda.synchronize(device)


def profile_mark(timings, name, started, step):
    if Config.MODE == 'benchmark':
        synchronize_devices()
    now = time.perf_counter()
    if Config.MODE == 'benchmark' and step >= Config.PROFILE_WARMUP:
        timings[name].append(now - started)
    return now


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temporary, path)


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def finite_tensor(value, label):
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f'Nonfinite {label}; stop and inspect the recorded batch IDs')


def install_diagnostic_hooks(model):
    """Device-local boundaries; hooks replicate with DataParallel. Debug only."""
    handles = []
    if not Config.DEBUG_SYNC:
        return handles
    selected = ('vlm_bridge.stage', 'vlm_bridge.fpn_smooth', 'att1.', 'att2.', 'att3.')
    for name, module in model.named_modules():
        if not name.startswith(selected) or not isinstance(
                module, (nn.Conv2d, nn.BatchNorm2d, nn.MultiheadAttention, nn.Sigmoid)):
            continue

        def before(layer, inputs, label=name):
            tensor = inputs[0]
            if tensor.is_cuda:
                torch.cuda.synchronize(tensor.device)
            print(f'DEBUG enter {label}: device={tensor.device}, dtype={tensor.dtype}, '
                  f'shape={tuple(tensor.shape)}, stride={tensor.stride()}, '
                  f'offset={tensor.storage_offset()}, ptr_mod16={tensor.data_ptr() % 16}', flush=True)
            finite_tensor(tensor, f'{label} input')

        def after(layer, inputs, outputs, label=name):
            values = outputs if isinstance(outputs, tuple) else (outputs,)
            for value in values:
                if isinstance(value, torch.Tensor):
                    if value.is_cuda:
                        torch.cuda.synchronize(value.device)
                    finite_tensor(value, f'{label} output')
            print(f'DEBUG exit {label}', flush=True)

        handles += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
    return handles


def unpack_batch(batch):
    images, masks = batch[:2]
    spines = batch[2] if len(batch) >= 3 else None
    ids = list(batch[3]) if len(batch) >= 4 else ['unknown'] * len(images)
    return images, masks, spines, ids


def epoch_report(totals, count, started, timings, label):
    if count == 0:
        raise ValueError(f'{label} loader has no usable records')
    result = {key: float(value.item() / count) for key, value in totals.items()}
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError(f'Nonfinite {label} epoch totals: {result}')
    result['records'] = count
    result['seconds'] = time.perf_counter() - started
    if str(Config.DEVICE).startswith('cuda'):
        result['memory'] = {
            str(i): {'allocated_peak_mib': torch.cuda.max_memory_allocated(i) / 2**20,
                     'reserved_peak_mib': torch.cuda.max_memory_reserved(i) / 2**20}
            for i in (range(torch.cuda.device_count()) if Config.USE_DP else [Config.GPU])}
    if timings:
        result['timings_seconds'] = {
            key: {'median': float(np.median(values)), 'p95': float(np.percentile(values, 95))}
            for key, values in timings.items() if values}
        batches = [sum(values) for values in zip(*timings.values())]
        if batches:
            result['timings_seconds']['batch'] = {
                'median': float(np.median(batches)), 'p95': float(np.percentile(batches, 95))}
    print(f'{label}: {json.dumps(result)}', flush=True)
    return result


def train_epoch(model, loader, optimizer, criterion, scaler, epoch=1):
    model.train()
    totals, timings = {}, defaultdict(list)
    count = 0
    optimizer.zero_grad(set_to_none=True)
    remainder = len(loader) % Config.ACCUM_STEPS
    if Config.MODE == 'benchmark':
        synchronize_devices()
    started = tick = time.perf_counter()
    for step, batch in enumerate(tqdm(loader, desc=f'Training (Ep {epoch})',
                                      mininterval=Config.PROGRESS_SECONDS)):
        tick = profile_mark(timings, 'data_wait', tick, step)
        images, masks, spines, ids = unpack_batch(batch)
        if Config.DEBUG_SYNC:
            print(f'TRAIN epoch={epoch} step={step} records={ids}', flush=True)
        images = images.to(Config.DEVICE, non_blocking=True)
        masks = masks.to(Config.DEVICE, non_blocking=True)
        spines = spines.to(Config.DEVICE, non_blocking=True) if spines is not None else None
        tick = profile_mark(timings, 'transfer', tick, step)
        divisor = remainder if remainder and step >= len(loader) - remainder else Config.ACCUM_STEPS
        with torch.autocast(device_type=images.device.type, dtype=torch.float16,
                            enabled=images.is_cuda):
            logits, auxiliary = forward_model(model, images, return_aux=True)
        tick = profile_mark(timings, 'forward', tick, step)
        loss, parts = criterion(logits, masks, spine_targets=spines, vlm_logits=auxiliary,
                                epoch=epoch, return_parts=True)
        tick = profile_mark(timings, 'criterion', tick, step)
        scaler.scale(loss / divisor).backward()
        tick = profile_mark(timings, 'backward', tick, step)
        if (step + 1) % Config.ACCUM_STEPS == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        tick = profile_mark(timings, 'optimizer', tick, step)
        size = images.shape[0]
        for key, value in {'loss': loss.detach(), **parts}.items():
            totals[key] = totals.get(key, 0) + value.float() * size
        count += size
    return epoch_report(totals, count, started, timings, f'Train epoch {epoch}')


@torch.no_grad()
def validate(model, loader, criterion, epoch=5):
    """Fixed full objective, record-weighted reporting, full batch semantics."""
    model.eval()
    totals, timings, count = {}, defaultdict(list), 0
    started = tick = time.perf_counter()
    for step, batch in enumerate(tqdm(loader, desc='Validating',
                                      mininterval=Config.PROGRESS_SECONDS)):
        tick = profile_mark(timings, 'data_wait', tick, step)
        images, masks, spines, ids = unpack_batch(batch)
        if Config.MODE == 'diagnose':
            print(f'VALIDATE epoch={epoch} step={step} records={ids}', flush=True)
            if step < Config.CAPTURE_BATCHES:
                atomic_torch_save({'images': images.cpu(), 'masks': masks.cpu(),
                                   'spines': spines.cpu() if spines is not None else None,
                                   'ids': ids, 'epoch': epoch},
                                  Path(Config.OUT_DIR) / f'diagnostic_batch_{step}.pth')
        images = images.to(Config.DEVICE, non_blocking=True)
        masks = masks.to(Config.DEVICE, non_blocking=True)
        spines = spines.to(Config.DEVICE, non_blocking=True) if spines is not None else None
        tick = profile_mark(timings, 'transfer', tick, step)
        with torch.autocast(device_type=images.device.type, dtype=torch.float16,
                            enabled=images.is_cuda and Config.EVAL_AMP):
            logits, auxiliary = forward_model(model, images, return_aux=True)
        tick = profile_mark(timings, 'forward', tick, step)
        # Always the same objective for best-checkpoint selection, including Ep 1.
        loss, parts = criterion(logits, masks, spine_targets=spines, vlm_logits=auxiliary,
                                epoch=Config.VAL_LOSS_EPOCH, return_parts=True)
        tick = profile_mark(timings, 'criterion', tick, step)
        size = len(images)
        for key, value in {'loss': loss.detach(), **parts}.items():
            totals[key] = totals.get(key, 0) + value.float() * size
        count += size
    return epoch_report(totals, count, started, timings, 'Validation (fixed objective)')


@torch.no_grad()
def preflight(model, loader):
    """Check real batch 2 + singleton and auxiliary/plain routes before training."""
    rng = capture_rng()
    was_training = model.training
    try:
        model.eval()
        batch = next(iter(loader))
        images, _, _, ids = unpack_batch(batch)
        images = images.to(Config.DEVICE)
        with torch.autocast(device_type=images.device.type, dtype=torch.float16,
                            enabled=images.is_cuda and Config.EVAL_AMP):
            logits, auxiliary = forward_model(model, images, return_aux=True)
            plain = forward_model(model, images)
            single = forward_model(model, images[:1])
        for name, tensor in [('logits', logits), ('auxiliary', auxiliary), ('singleton', single)]:
            finite_tensor(tensor, name)
        torch.testing.assert_close(logits, plain, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(logits[:1], single, rtol=3e-2, atol=3e-2)
        print(f'PREFLIGHT PASS: batch={len(images)}, singleton, auxiliary/plain; records={ids}')
    finally:
        model.train(was_training)
        restore_rng(rng)


# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T10:17:42.489156Z","iopub.execute_input":"2026-10-17T10:17:42.489499Z","iopub.status.idle":"2026-10-17T10:17:42.510437Z","shell.execute_reply.started":"2026-10-17T10:17:42.489473Z","shell.execute_reply":"2026-10-17T10:17:42.509794Z"}}
# CELL 8: Inference and Kaggle Submission Generator
# This cell:
# 1. Runs the trained model on all test images
# 2. Upscales predictions from 1024x1024 back to 2048x2048
# 3. Uses Connected Components to separate individual filaments
# 4. Filters out tiny noise (< MIN_AREA pixels)
# 5. Encodes each filament mask as RLE using pycocotools
# 6. Saves submission.csv in the exact format required by the competition
#
# Competition format:
#   filament_id | segmentation_rle
#   20150125172714Mh_1 | <RLE string>
#   20150125172714Mh_2 | <RLE string>

def pycoco_rle_encode(img):
    """
    Encodes a binary mask to pycocotools RLE format (ASCII string).
    This is the exact format required by the competition.
    
    Args:
        img: Binary numpy array (0/1) of shape [H, W]
    Returns:
        RLE counts as ASCII string
    """
    rle = maskUtils.encode(np.asfortranarray(img.astype(np.uint8)))
    return rle['counts'].decode('utf-8')

def build_gaussian_window(size=Config.IMG_SIZE, sigma=Config.GAUSSIAN_SIGMA, device='cpu'):
    """Builds a 2D Gaussian weighting window [1, 1, size, size] for seamless tile stitching."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g1d = torch.exp(-(coords ** 2) / (2.0 * (sigma ** 2)))
    g2d = torch.outer(g1d, g1d)
    g2d = g2d / g2d.max().clamp_min(1e-8)
    return g2d.view(1, 1, size, size)


def _predict_single_view(model, tensor, use_tta=False):
    """Predicts a probability map for a [B, 3, H, W] tensor with optional 4-fold TTA (DataParallel-safe)."""
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=str(Config.DEVICE).startswith('cuda') and Config.EVAL_AMP):
        p0 = torch.sigmoid(forward_model(model, tensor, return_aux=False).float())
        if not use_tta:
            return p0
        ph = torch.flip(
            torch.sigmoid(forward_model(model, torch.flip(tensor, dims=[-1]), return_aux=False).float()),
            dims=[-1],
        )
        pv = torch.flip(
            torch.sigmoid(forward_model(model, torch.flip(tensor, dims=[-2]), return_aux=False).float()),
            dims=[-2],
        )
        pr = torch.rot90(
            torch.sigmoid(forward_model(model, torch.rot90(tensor, 1, dims=[-2, -1]), return_aux=False).float()),
            -1,
            dims=[-2, -1],
        )
        return 0.25 * (p0 + ph + pv + pr)


def predict_probability(model, tensor, use_tta=Config.USE_TTA):
    """
    Predicts full-resolution [1, 1, 2048, 2048] probability map using:
    1. Global full-disk context view (P_global): downsampled to 1024x1024 (with TTA) and upsampled to 2048x2048.
    2. High-resolution 3x3 sliding window of 1024x1024 native tiles (stride=512, 50% overlap)
       weighted by a 2D Gaussian window (sigma=256) to form P_local.
    3. Anti-Breakage Gated Fusion:
       P_final = (P_global > 0.05).float() * (0.85 * P_local + 0.15 * P_global)
    """
    _, _, H, W = tensor.shape
    tile_size = Config.IMG_SIZE
    orig_size = Config.ORIGINAL_SIZE

    # If caller passes an already-downscaled 1024x1024 tensor, evaluate and upsample cleanly
    if H < orig_size or W < orig_size:
        probs_1024 = _predict_single_view(model, tensor, use_tta=use_tta)
        return F.interpolate(probs_1024, size=(orig_size, orig_size), mode='bilinear', align_corners=False)

    # 1. Global full-disk view P_global
    tensor_global = F.interpolate(tensor, size=(tile_size, tile_size), mode='bilinear', align_corners=False)
    p_global_1024 = _predict_single_view(model, tensor_global, use_tta=use_tta)
    p_global = F.interpolate(p_global_1024, size=(H, W), mode='bilinear', align_corners=False)

    # 2. 3x3 Sliding Window of 1024x1024 native tiles (stride=512, 50% overlap) weighted by 2D Gaussian
    stride = getattr(Config, 'TILE_STRIDE', 512)
    sigma = getattr(Config, 'GAUSSIAN_SIGMA', 256.0)
    gauss_win = build_gaussian_window(size=tile_size, sigma=sigma, device=tensor.device)

    y_starts = sorted(set(list(range(0, H - tile_size + 1, stride)) + [H - tile_size]))
    x_starts = sorted(set(list(range(0, W - tile_size + 1, stride)) + [W - tile_size]))

    accum_prob = torch.zeros((tensor.shape[0], 1, H, W), dtype=torch.float32, device=tensor.device)
    accum_weight = torch.zeros((1, 1, H, W), dtype=torch.float32, device=tensor.device)

    coords = [(y0, x0) for y0 in y_starts for x0 in x_starts]
    if tensor.shape[0] == 1 and len(coords) > 1:
        tile_batch_size = max(2, getattr(Config, 'BATCH_SIZE', 2))
        for i in range(0, len(coords), tile_batch_size):
            batch_coords = coords[i:i + tile_batch_size]
            batch_tiles = torch.cat(
                [tensor[:, :, y0:y0 + tile_size, x0:x0 + tile_size] for (y0, x0) in batch_coords],
                dim=0,
            )
            p_tiles = _predict_single_view(model, batch_tiles, use_tta=False)
            for idx, (y0, x0) in enumerate(batch_coords):
                accum_prob[:, :, y0:y0 + tile_size, x0:x0 + tile_size] += p_tiles[idx:idx + 1] * gauss_win
                accum_weight[:, :, y0:y0 + tile_size, x0:x0 + tile_size] += gauss_win
    else:
        for (y0, x0) in coords:
            tile = tensor[:, :, y0:y0 + tile_size, x0:x0 + tile_size]
            p_tile = _predict_single_view(model, tile, use_tta=False)
            accum_prob[:, :, y0:y0 + tile_size, x0:x0 + tile_size] += p_tile * gauss_win
            accum_weight[:, :, y0:y0 + tile_size, x0:x0 + tile_size] += gauss_win

    p_local = accum_prob / accum_weight.clamp_min(1e-6)

    # 3. Hard global cutoff and weighted high-resolution fusion (preserved)
    gate_thresh = getattr(Config, 'GLOBAL_GATE_THRESH', 0.05)
    w_local = getattr(Config, 'LOCAL_BLEND_WEIGHT', 0.85)
    w_global = getattr(Config, 'GLOBAL_BLEND_WEIGHT', 0.15)
    p_final = (p_global > gate_thresh).float() * (w_local * p_local + w_global * p_global)
    return p_final

def postprocess_probability(probability):
    """Shared native postprocessing for holdout and submission."""
    array = probability.detach().float().cpu().numpy().squeeze()
    if array.ndim != 2 or not np.isfinite(array).all() or array.min() < 0 or array.max() > 1.00001:
        raise ValueError('Invalid native probability map')
    mask = (array > Config.THRESHOLD).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = [(labels == i).astype(np.uint8) for i in range(1, count)
                  if stats[i, cv2.CC_STAT_AREA] >= Config.MIN_AREA]
    return components


def pq_metrics(predictions, targets):
    # Reused from V4's matching routine; IoU > 0.5 and disjoint predictions make
    # accepted matches unique. Counts are diagnostics, not the official evaluator.
    overlaps = np.zeros((len(predictions), len(targets)), dtype=np.float32)
    for p, prediction in enumerate(predictions):
        for t, target in enumerate(targets):
            union = np.logical_or(prediction, target).sum()
            overlaps[p, t] = np.logical_and(prediction, target).sum() / max(union, 1)
    candidates = sorted(((overlaps[p, t], p, t) for p in range(len(predictions))
                         for t in range(len(targets))), reverse=True)
    matched_predictions, matched_targets, matched_iou = set(), set(), []
    for iou, p, t in candidates:
        if iou <= Config.PQ_IOU_THRESHOLD or p in matched_predictions or t in matched_targets:
            continue
        matched_predictions.add(p)
        matched_targets.add(t)
        matched_iou.append(float(iou))
    tp = len(matched_iou)
    fp, fn = len(predictions) - tp, len(targets) - tp
    denominator = tp + 0.5 * fp + 0.5 * fn
    # A 0.1 IoU diagnostic threshold avoids counting a one-pixel touch as a split.
    touching = overlaps > 0.1
    return {'pq': sum(matched_iou) / denominator if denominator else 0.0,
            'sum_iou': sum(matched_iou), 'tp': tp, 'fp': fp, 'fn': fn,
            'splits': int(np.maximum(touching.sum(axis=0) - 1, 0).sum()),
            'merges': int(np.maximum(touching.sum(axis=1) - 1, 0).sum())}


@torch.no_grad()
def evaluate_native(model, coco, validation_ids, limit=0):
    """Predict once/file; score each annotation record separately, then file-macro."""
    model.eval()
    grouped = defaultdict(list)
    for image_id in validation_ids:
        grouped[coco.imgs[image_id]['file_name']].append(image_id)
    records, file_scores = [], []
    started = time.perf_counter()
    items = list(grouped.items())
    if limit:
        items = items[:limit]
    for filename, ids in tqdm(items, desc='Native holdout', mininterval=Config.PROGRESS_SECONDS):
        image = load_image(str(Path(Config.TRAIN_DIR) / filename))
        prepared, _ = prepare_input_3ch(image)
        tensor = torch.from_numpy(prepared).permute(2, 0, 1).unsqueeze(0).to(Config.DEVICE)
        probability = predict_probability(model, tensor, use_tta=Config.USE_TTA)
        predictions = postprocess_probability(probability)
        binary = np.zeros(image.shape, dtype=np.uint8)
        for prediction in predictions:
            binary |= prediction
        current = []
        for image_id in ids:
            targets = [coco.annToMask(ann).astype(bool) for ann in coco.imgToAnns.get(image_id, [])]
            target = np.zeros(image.shape, dtype=bool)
            for instance in targets:
                target |= instance
            denominator = int(binary.sum() + target.sum())
            dice = 2 * np.logical_and(binary, target).sum() / denominator if denominator else 1.0
            metrics = {'image_id': str(image_id), 'file_name': filename,
                       'dice': float(dice), **pq_metrics(predictions, targets)}
            records.append(metrics)
            current.append(metrics)
        file_scores.append({key: float(np.mean([r[key] for r in current])) for key in ('dice', 'pq')})
    if not records:
        raise ValueError('No native validation records')
    totals = {key: sum(r[key] for r in records) for key in ('sum_iou', 'tp', 'fp', 'fn', 'splits', 'merges')}
    denominator = totals['tp'] + 0.5 * totals['fp'] + 0.5 * totals['fn']
    result = {'policy': 'each annotation record scored independently; equal weight per physical file',
              'metric_note': 'local PQ diagnostic (IoU > 0.5), not a verified competition score',
              'physical_files': len(items), 'annotation_records': len(records),
              'dice_file_macro': float(np.mean([r['dice'] for r in file_scores])),
              'pq_file_macro': float(np.mean([r['pq'] for r in file_scores])),
              'pq_record_micro': totals['sum_iou'] / denominator if denominator else 0.0,
              'totals': totals, 'seconds': time.perf_counter() - started,
              'threshold': Config.THRESHOLD, 'min_area': Config.MIN_AREA,
              'global_cutoff': Config.GLOBAL_GATE_THRESH, 'tta': Config.USE_TTA,
              'records': records}
    write_json(Path(Config.OUT_DIR) / 'native_validation.json', result)
    print(f'Native holdout: Dice={result["dice_file_macro"]:.4f}, '
          f'local PQ={result["pq_file_macro"]:.4f}; {len(items)} files, '
          f'{result["seconds"]:.1f}s. Per-record details saved.')
    return result


def generate_submission(model, test_paths, out_path):
    """Use exactly the native holdout decoder; validate every exported COCO RLE."""
    model.eval()
    submissions, empty_images = [], []
    stems = [Path(path).stem for path in test_paths]
    if len(stems) != len(set(stems)):
        raise ValueError('Duplicate input filenames would produce duplicate filament IDs')
    started = time.perf_counter()
    with torch.no_grad():
        for img_path in tqdm(test_paths, desc='Generating Submission', mininterval=Config.PROGRESS_SECONDS):
            image = load_image(img_path)
            prepared, _ = prepare_input_3ch(image)
            tensor = torch.from_numpy(prepared).permute(2, 0, 1).unsqueeze(0).to(Config.DEVICE)
            probability = predict_probability(model, tensor, use_tta=Config.USE_TTA)
            if tuple(probability.shape[-2:]) != image.shape:
                raise ValueError(f'Output dimensions do not match {img_path}: {probability.shape}')
            components = postprocess_probability(probability)
            if not components:
                empty_images.append(Path(img_path).stem)
            for i, component in enumerate(components, 1):
                rle = pycoco_rle_encode(component)
                decoded = maskUtils.decode({'size': list(component.shape), 'counts': rle.encode('utf-8')})
                if not np.array_equal(decoded, component):
                    raise ValueError(f'RLE round-trip failed for {img_path}, instance {i}')
                submissions.append({'filament_id': f'{Path(img_path).stem}_{i}', 'segmentation_rle': rle})
    frame = pd.DataFrame(submissions, columns=['filament_id', 'segmentation_rle'])
    if frame['filament_id'].duplicated().any():
        raise ValueError('Duplicate submission filament IDs')
    temporary = str(out_path) + '.tmp'
    frame.to_csv(temporary, index=False)
    os.replace(temporary, out_path)
    result = {'input_images': len(test_paths), 'images_with_components': len(test_paths) - len(empty_images),
              'empty_image_ids': empty_images, 'filaments': len(frame),
              'seconds': time.perf_counter() - started, 'rle_roundtrip_verified': True,
              'schema': list(frame.columns)}
    write_json(str(out_path) + '.audit.json', result)
    print(f'SUBMISSION VERIFIED: {out_path}; {json.dumps(result)}')


def seed_everything(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(Config.GPU)
        torch.cuda.manual_seed_all(seed)
    torch.backends.mha.set_fastpath_enabled(Config.MHA_FASTPATH)
    torch.backends.cudnn.enabled = Config.CUDNN
    torch.backends.cudnn.benchmark = Config.CUDNN_BENCHMARK
    if Config.SDP_BACKEND not in {'math', 'auto', 'efficient', 'flash', 'cudnn'}:
        raise ValueError(f'Unknown V5_SDP={Config.SDP_BACKEND}')
    for name, selected in [('flash', 'flash'), ('mem_efficient', 'efficient'),
                           ('cudnn', 'cudnn'), ('math', 'math')]:
        setter = getattr(torch.backends.cuda, f'enable_{name}_sdp', None)
        if setter:
            setter(Config.SDP_BACKEND in {'auto', selected})


def capture_rng():
    numpy_state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    algorithm, keys, position, gaussian, cached = state['numpy']
    np.random.set_state((algorithm, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume GPU count differs; RNG replay would be invalid')
        torch.cuda.set_rng_state_all([item.cpu() for item in state['cuda']])


def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def model_signature(model):
    schema = [(name, list(value.shape), str(value.dtype))
              for name, value in unwrap_model(model).state_dict().items()]
    return hashlib.sha256(json.dumps(schema).encode()).hexdigest()


def experiment_config():
    names = ('IMG_SIZE', 'ORIGINAL_SIZE', 'BATCH_SIZE', 'ACCUM_STEPS', 'EPOCHS', 'LR',
             'WEIGHT_DECAY', 'IN_CHANNELS', 'OUT_CHANNELS', 'SEED', 'SAM2_VARIANT',
             'CROP_PROB', 'FILAMENT_CENTER_PROB', 'CROP_JITTER', 'DICE_WEIGHT',
             'FOCAL_WEIGHT', 'FOCAL_ALPHA', 'FOCAL_GAMMA', 'LOVASZ_MAX_WEIGHT',
             'LOVASZ_WARMUP_EPOCHS', 'SKEL_RECALL_WEIGHT', 'VLM_PRIOR_WEIGHT', 'VAL_LOSS_EPOCH')
    return {name: getattr(Config, name) for name in names}


def runtime_manifest(model, train_ids, val_ids, coco):
    runtime = {name: getattr(Config, name) for name in vars(Config)
               if name.isupper() and isinstance(getattr(Config, name), (str, bool, int, float))}
    fingerprint_kind = 'file'
    if '__file__' in globals() and Path(__file__).is_file():
        source = Path(__file__).read_bytes()
    else:
        try:
            source = (inspect.getsource(MagAgentU2Net) + inspect.getsource(VLMSemanticBridge)).encode()
            fingerprint_kind = 'model_definitions'
        except (OSError, TypeError):
            source = model_signature(model).encode()
            fingerprint_kind = 'schema_only_source_unavailable'
    annotation_hash = hashlib.sha256(Path(Config.ANNOTATIONS_FILE).read_bytes()).hexdigest()
    split = {'train': [str(i) for i in train_ids], 'validation': [str(i) for i in val_ids],
             'files': {str(i): coco.imgs[i]['file_name'] for i in train_ids + val_ids}}
    split_hash = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    try:
        driver = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version',
                                 '--format=csv,noheader'], capture_output=True, text=True,
                                timeout=10, check=False).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        driver = 'not available'
    return {'run_id': uuid.uuid4().hex, 'source_sha256': hashlib.sha256(source).hexdigest(),
            'source_fingerprint_kind': fingerprint_kind,
            'model_signature': model_signature(model), 'experiment': experiment_config(),
            'annotation_sha256': annotation_hash, 'split_sha256': split_hash,
            'runtime': runtime, 'python': platform.python_version(),
            'packages': {'torch': str(torch.__version__), 'numpy': np.__version__,
                         'opencv': cv2.__version__, 'albumentations': A.__version__,
                         'pandas': pd.__version__},
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'driver': driver, 'cuda_launch_blocking': os.getenv('CUDA_LAUNCH_BLOCKING', '0'),
            'gpus': [{'index': i, 'name': torch.cuda.get_device_name(i),
                      'capability': list(torch.cuda.get_device_capability(i)),
                      'memory_bytes': torch.cuda.get_device_properties(i).total_memory}
                     for i in range(torch.cuda.device_count())],
            'preload': getattr(unwrap_model(model).vlm_bridge, 'preload_provenance',
                               {'source': 'intentional initialization', 'parameter_coverage': 0.0}),
            'split': split}


def check_checkpoint(checkpoint, model, expected=None):
    if checkpoint.get('format_version') != 2 or 'metadata' not in checkpoint:
        raise ValueError('Legacy/unverified checkpoint: use explicit pretrained transfer; not resume/inference')
    metadata = checkpoint['metadata']
    if metadata.get('model_signature') != model_signature(model):
        raise ValueError('Checkpoint architecture/schema mismatch')
    if expected:
        for key in ('experiment', 'split_sha256', 'annotation_sha256'):
            if metadata.get(key) != expected.get(key):
                raise ValueError(f'Checkpoint {key} mismatch; use its original config/split')
    return metadata


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in unwrap_model(model).state_dict().items()}


def save_v5_checkpoints(model, model_path, vlm_path=None, metadata=None):
    """best_model is a single authoritative, complete model (bridge included)."""
    metadata = dict(metadata or {})
    metadata.update(model_signature=model_signature(model), checkpoint_id=uuid.uuid4().hex)
    state = cpu_state(model)
    atomic_torch_save({'format_version': 2, 'kind': 'inference',
                       'metadata': metadata, 'model': state}, model_path)
    if vlm_path:
        bridge = {key.removeprefix('vlm_bridge.'): value for key, value in state.items()
                  if key.startswith('vlm_bridge.')}
        atomic_torch_save({'format_version': 2, 'kind': 'bridge', 'metadata': metadata,
                           'state_dict': bridge}, vlm_path)


def load_v5_checkpoints(model, model_path, vlm_path=None, device='cpu', expected=None):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    metadata = check_checkpoint(checkpoint, model, expected)
    if checkpoint.get('kind') not in {'inference', 'training'}:
        raise ValueError('Expected a complete model checkpoint')
    if vlm_path is not None:
        bridge = torch.load(vlm_path, map_location='cpu', weights_only=True)
        if bridge.get('kind') != 'bridge' or bridge.get('metadata', {}).get('checkpoint_id') != metadata.get('checkpoint_id'):
            raise ValueError('Mismatched model/bridge checkpoint pair')
        saved_bridge = {key.removeprefix('vlm_bridge.'): value for key, value in checkpoint['model'].items()
                        if key.startswith('vlm_bridge.')}
        if set(bridge['state_dict']) != set(saved_bridge) or any(
                not torch.equal(value, bridge['state_dict'][key]) for key, value in saved_bridge.items()):
            raise ValueError('Bridge export contents do not match the full model')
    unwrap_model(model).load_state_dict(checkpoint['model'], strict=True)
    return metadata


def save_training_checkpoint(path, model, optimizer, scaler, scheduler, metadata,
                             epoch, phase, best_loss, train_report, best_model_state=None):
    atomic_torch_save({'format_version': 2, 'kind': 'training', 'metadata': metadata,
                       'model': cpu_state(model), 'optimizer': optimizer.state_dict(),
                       'scaler': scaler.state_dict(), 'scheduler': scheduler.state_dict(),
                       'epoch': epoch, 'phase': phase, 'best_loss': best_loss,
                       'train_report': train_report, 'rng': capture_rng(),
                       'best_model': best_model_state}, path)


def make_loader(dataset, training):
    generator = torch.Generator().manual_seed(Config.SEED)
    return DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=training,
                      drop_last=training, num_workers=Config.NUM_WORKERS,
                      pin_memory=str(Config.DEVICE).startswith('cuda'),
                      persistent_workers=Config.PERSISTENT_WORKERS and Config.NUM_WORKERS > 0,
                      generator=generator)


def main():
    if Config.MODE not in {'train', 'smoke', 'diagnose', 'benchmark', 'evaluate', 'submit'}:
        raise ValueError(f'Unknown V5_MODE={Config.MODE}')
    if Config.NUM_WORKERS < 0 or Config.VAL_CACHE_ITEMS < 0 or Config.PROFILE_BATCHES < 1 or Config.PROFILE_EPOCH < 1:
        raise ValueError('Invalid worker/cache/profile configuration')
    seed_everything()
    os.makedirs(Config.OUT_DIR, exist_ok=True)
    # Smoke/benchmark artifacts cannot overwrite full training checkpoints.
    if Config.MODE in {'smoke', 'benchmark', 'diagnose'}:
        Config.OUT_DIR = str(Path(Config.OUT_DIR) / Config.MODE)
        os.makedirs(Config.OUT_DIR, exist_ok=True)
    coco = COCO(Config.ANNOTATIONS_FILE)
    files = defaultdict(list)
    for image_id, info in coco.imgs.items():
        path = Path(Config.TRAIN_DIR) / info['file_name']
        if not path.is_file():
            raise FileNotFoundError(f'Missing annotated image: {path}')
        files[info['file_name']].append(image_id)
    filenames = sorted(files)
    random.Random(Config.SEED).shuffle(filenames)
    split = int(0.8 * len(filenames))
    train_ids = [i for name in filenames[:split] for i in files[name]]
    val_ids = [i for name in filenames[split:] for i in files[name]]
    if not train_ids or not val_ids:
        raise ValueError('Need nonempty physical-file train/validation splits')
    assert not set(filenames[:split]) & set(filenames[split:])
    print(f'Train: {len(train_ids)} records / {split} physical files; '
          f'validation: {len(val_ids)} records / {len(filenames) - split} files')
    full_train_ids, full_val_ids = list(train_ids), list(val_ids)
    if Config.MODE == 'smoke':
        train_ids, val_ids = train_ids[:4], val_ids[:3]
    elif Config.MODE == 'benchmark':
        limit = (Config.PROFILE_WARMUP + Config.PROFILE_BATCHES) * Config.BATCH_SIZE
        train_ids, val_ids = train_ids[:limit], val_ids[:limit + 1]
    train_ds = SolarDataset(coco, train_ids, Config.TRAIN_DIR, Config.IMG_SIZE,
                            train_transforms, return_spine=True, training=True)
    val_ds = SolarDataset(coco, val_ids, Config.TRAIN_DIR, Config.IMG_SIZE,
                          val_transforms, return_spine=True)
    train_loader, val_loader = make_loader(train_ds, True), make_loader(val_ds, False)
    if len(train_loader) == 0:
        raise ValueError('Training split has fewer records than BATCH_SIZE')
    model = MagAgentU2Net(Config.IN_CHANNELS, Config.OUT_CHANNELS)
    if not Config.RESUME_PATH and not Config.CHECKPOINT_PATH:
        model.vlm_bridge.preload_sam2_hiera()
    manifest = runtime_manifest(model, full_train_ids, full_val_ids, coco)
    manifest['training_protocol'] = Config.MODE
    manifest['selected_records'] = {'train': [str(i) for i in train_ids],
                                    'validation': [str(i) for i in val_ids]}
    model = model.to(Config.DEVICE)
    if Config.USE_DP and torch.cuda.device_count() > 1:
        devices = [Config.GPU] + [i for i in range(torch.cuda.device_count()) if i != Config.GPU]
        model = nn.DataParallel(model, device_ids=devices, output_device=Config.GPU)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}; '
          f'device={Config.DEVICE}; DataParallel={isinstance(model, nn.DataParallel)}; '
          f'MHA fastpath={Config.MHA_FASTPATH}; SDP={Config.SDP_BACKEND}; '
          f'bridge AMP={Config.BRIDGE_AMP}; eval AMP={Config.EVAL_AMP}; '
          f'eval serial={Config.EVAL_SERIAL}; effective batch={Config.BATCH_SIZE * Config.ACCUM_STEPS}')
    write_json(Path(Config.OUT_DIR) / 'run_manifest.json', manifest)
    handles = install_diagnostic_hooks(unwrap_model(model))
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=str(Config.DEVICE).startswith('cuda'))
    model_path = str(Path(Config.OUT_DIR) / Config.MODEL_WEIGHTS_NAME)
    bridge_path = str(Path(Config.OUT_DIR) / Config.VLM_WEIGHTS_NAME)
    last_path = str(Path(Config.OUT_DIR) / 'last_training.pth')
    start_epoch, phase, best_loss, last_train = 1, 'ready', float('inf'), None
    best_model_state = None
    if Config.RESUME_PATH:
        checkpoint = torch.load(Config.RESUME_PATH, map_location='cpu', weights_only=True)
        saved = check_checkpoint(checkpoint, model, manifest)
        if checkpoint.get('kind') != 'training' or checkpoint.get('phase') not in {'trained', 'validated'}:
            raise ValueError('Resume requires a training checkpoint with an epoch phase')
        if saved.get('training_protocol') != Config.MODE:
            raise ValueError('Cannot resume a smoke/benchmark checkpoint as full training (or vice versa)')
        unwrap_model(model).load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        phase = checkpoint['phase']
        start_epoch = checkpoint['epoch'] + (phase == 'validated')
        best_loss, last_train = checkpoint['best_loss'], checkpoint['train_report']
        best_model_state = checkpoint.get('best_model')
        manifest['resumed_from'] = Config.RESUME_PATH
        manifest['run_id'] = saved['run_id']
        manifest['preload'] = saved.get('preload', manifest['preload'])
        restore_rng(checkpoint['rng'])
        print(f'Resuming epoch {start_epoch}, phase={phase}; source training epoch is not repeated')
    elif Config.CHECKPOINT_PATH:
        load_v5_checkpoints(model, Config.CHECKPOINT_PATH, expected=manifest)
    elif Config.MODE in {'evaluate', 'submit'}:
        raise ValueError('Set V5_CHECKPOINT to a new-format best_model.pth')
    write_json(Path(Config.OUT_DIR) / 'run_manifest.json', manifest)
    if Config.MODE == 'diagnose':
        if Config.REPLAY_BATCH:
            captured = torch.load(Config.REPLAY_BATCH, map_location='cpu', weights_only=True)
            diagnostic_loader = [(captured['images'], captured['masks'], captured['spines'], captured['ids'])]
        else:
            diagnostic_loader = val_loader
        # Full holdout by default, so the last singleton is included.
        report = validate(model, diagnostic_loader, criterion)
        write_json(Path(Config.OUT_DIR) / 'diagnostic_results.json', report)
        return
    if Config.PREFLIGHT:
        preflight(model, val_loader)
    if Config.MODE in {'train', 'smoke', 'benchmark'}:
        final_epoch = 2 if Config.MODE == 'smoke' else (1 if Config.MODE == 'benchmark' else Config.EPOCHS)
        history_path = Path(Config.OUT_DIR) / 'history.jsonl'
        for epoch in range(start_epoch, final_epoch + 1):
            if str(Config.DEVICE).startswith('cuda'):
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
            if phase != 'trained':
                train_ds.epoch.fill_(epoch)
                train_loader.generator.manual_seed(Config.SEED + epoch)
                objective_epoch = Config.PROFILE_EPOCH if Config.MODE == 'benchmark' else epoch
                last_train = train_epoch(model, train_loader, optimizer, criterion, scaler, objective_epoch)
                last_train['objective_epoch'] = objective_epoch
                # Written BEFORE eval; scheduler has not stepped for this epoch.
                save_training_checkpoint(last_path, model, optimizer, scaler, scheduler, manifest,
                                         epoch, 'trained', best_loss, last_train, best_model_state)
            validation = validate(model, val_loader, criterion, epoch)
            if validation['loss'] < best_loss:
                best_loss = validation['loss']
                metadata = {**manifest, 'epoch': epoch, 'validation': validation}
                save_v5_checkpoints(model, model_path, bridge_path, metadata)
                best_model_state = {'model': cpu_state(model), 'metadata': metadata}
            scheduler.step()
            save_training_checkpoint(last_path, model, optimizer, scaler, scheduler, manifest,
                                     epoch, 'validated', best_loss, last_train, best_model_state)
            with history_path.open('a') as handle:
                handle.write(json.dumps({'run_id': manifest['run_id'], 'epoch': epoch, 'train': last_train,
                                         'validation': validation, 'best_loss': best_loss}) + '\n')
            phase = 'ready'
        if best_model_state is None:
            raise ValueError('No finite best checkpoint was produced or restored')
        # Re-export saved best when resuming to a fresh Kaggle output directory.
        unwrap_model(model).load_state_dict(best_model_state['model'], strict=True)
        save_v5_checkpoints(model, model_path, bridge_path, best_model_state['metadata'])
        load_v5_checkpoints(model, model_path, bridge_path, expected=manifest)
        if Config.MODE == 'benchmark':
            print(f'BENCHMARK COMPLETE: reports in {Config.OUT_DIR}; no full training/submission run')
            return
        if Config.MODE == 'smoke':
            # Real native tiled + TTA path, RLE encoding/decoding, and visualization.
            evaluate_native(model, coco, full_val_ids, limit=1)
            smoke_paths = [str(Path(Config.TRAIN_DIR) / coco.imgs[full_val_ids[0]]['file_name'])]
            generate_submission(model, smoke_paths, str(Path(Config.OUT_DIR) / 'smoke_submission.csv'))
            visualize_predictions(model, coco, full_val_ids, Config.TRAIN_DIR, num_samples=1)
            print('SMOKE PASS: two epochs, Lovasz activation, singleton eval, checkpoint reload, '
                  'native tiles/TTA, RLE and visualization. Full 25-epoch/T4 verification remains.')
            return
    if Config.MODE == 'evaluate' or (Config.MODE == 'train' and Config.NATIVE_EVAL_AFTER_TRAIN):
        evaluate_native(model, coco, full_val_ids, limit=Config.NATIVE_EVAL_LIMIT)
    if Config.MODE in {'train', 'submit'}:
        test_paths = sorted(glob.glob(os.path.join(Config.TEST_DIR, '*.jpeg')))
        if not test_paths:
            raise ValueError(f'No test images in {Config.TEST_DIR}')
        if Config.EXPECTED_TEST_IMAGES and len(test_paths) != Config.EXPECTED_TEST_IMAGES:
            raise ValueError(f'Expected {Config.EXPECTED_TEST_IMAGES} test images, found {len(test_paths)}')
        generate_submission(model, test_paths, str(Path(Config.OUT_DIR) / 'submission.csv'))
    for handle in handles:
        handle.remove()
    print(f'PIPELINE COMPLETE: {Config.OUT_DIR}')


# CELL 10: Optional - Visualize Sample Predictions
# This cell shows a few example predictions overlaid on the original images
# so you can visually verify the model quality before submitting.

def visualize_predictions(model, coco, img_ids, img_dir, num_samples=4):
    """Display sample predictions for visual verification."""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6 * num_samples))
    
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    
    with torch.no_grad():
        for i, img_id in enumerate(img_ids[:num_samples]):
            img_info = coco.imgs[img_id]
            img_path = os.path.join(img_dir, img_info['file_name'])
            
        # Load and preprocess
            img = load_image(img_path)
            img_3ch, img_limb = prepare_input_3ch(img)
            img_1024 = cv2.resize(img_3ch, (Config.IMG_SIZE, Config.IMG_SIZE))
            tensor = torch.tensor(img_1024).permute(2, 0, 1).unsqueeze(0).to(
                Config.DEVICE, non_blocking=True
            )
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=str(Config.DEVICE).startswith('cuda') and Config.EVAL_AMP):
                logits = forward_model(model, tensor, return_aux=False)
            
            pred = (torch.sigmoid(logits) > Config.THRESHOLD).cpu().numpy().squeeze()
            
            # Get ground truth mask
            anns = coco.imgToAnns.get(img_id, [])
            gt_mask = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE), dtype=np.float32)
            for ann in anns:
                m = coco.annToMask(ann)
                m = cv2.resize(m, (Config.IMG_SIZE, Config.IMG_SIZE))
                gt_mask = np.maximum(gt_mask, m)
            
            # Plot
            axes[i, 0].imshow(img_limb, cmap='gray')
            axes[i, 0].set_title(f'Original: {img_info["file_name"]}')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(gt_mask, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUT_DIR, 'predictions.png'), dpi=150)
    if plt.get_backend().lower() != 'agg':
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()
