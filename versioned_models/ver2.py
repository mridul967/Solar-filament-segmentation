# CELL 1: Downloading important binaries and importing libraries
# This cell installs all required packages and imports them.
# - pycocotools: For reading COCO JSON annotations and RLE encoding for submission
# - albumentations: For data augmentation (flips, rotations)
# - optuna: For short, sequential learning-rate/weight-decay trials

!pip install optuna albumentations pycocotools -q

import os
import glob
import gc
import json
import random
import time
import optuna
import numpy as np
import pandas as pd
# pyrefly: ignore[missing-import]
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T12:57:29.034503Z","iopub.execute_input":"2026-10-17T12:57:29.035098Z","iopub.status.idle":"2026-10-17T12:57:29.048350Z","shell.execute_reply.started":"2026-10-17T12:57:29.035072Z","shell.execute_reply":"2026-10-17T12:57:29.047500Z"},"jupyter":{"outputs_hidden":false}}
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
    OUT_DIR = '/kaggle/working/'
    
    # === IMAGE DIMENSIONS ===
    # GONG H-alpha images are 2048x2048.
    # We downscale to 768x768 for the V2 resolution experiment,
    # then upscale predictions back to 2048x2048 for submission.
    IMG_SIZE = 768
    ORIGINAL_SIZE = 2048
    
    # === TRAINING HYPERPARAMETERS ===
    BATCH_SIZE = 2
    EPOCHS = 2           # Fresh training of the best short-trial configuration
    N_TRIALS = 1          # Includes the baseline configuration as trial 0
    TRIAL_EPOCHS = 2
    SEED = 42
    LR = 2e-4             # Learning rate (Adam optimizer)
    WEIGHT_DECAY = 1e-4   # L2 regularization
    NUM_WORKERS = 2       # DataLoader workers
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # === MODEL PARAMETERS ===
    IN_CHANNELS = 3       # [Limb-corrected H-alpha, Limb-corrected copy, Magnetic Proxy Map]
    OUT_CHANNELS = 1      # Single binary segmentation mask
    
    # === SUBMISSION PARAMETERS ===
    MIN_AREA = 200        # Minimum pixel area to count as a valid filament (filters noise)
    THRESHOLD = 0.4       # Probability threshold for binary mask
    CLOSING_RADIUS = 0    # 0 disables morphological closing
    PQ_IOU_THRESHOLD = 0.5
    DIAGNOSTIC_IOU_THRESHOLD = 0.1

# Verify paths exist
print(f"Train dir exists: {os.path.exists(Config.TRAIN_DIR)}")
print(f"Test dir exists:  {os.path.exists(Config.TEST_DIR)}")
print(f"Annotations exist: {os.path.exists(Config.ANNOTATIONS_FILE)}")
print(f"Train images: {len(glob.glob(os.path.join(Config.TRAIN_DIR, '*.jpeg')))}")
print(f"Test images:  {len(glob.glob(os.path.join(Config.TEST_DIR, '*.jpeg')))}")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T12:57:29.049323Z","iopub.execute_input":"2026-10-17T12:57:29.049638Z","iopub.status.idle":"2026-10-17T12:57:29.065775Z","shell.execute_reply.started":"2026-10-17T12:57:29.049607Z","shell.execute_reply":"2026-10-17T12:57:29.064984Z"},"jupyter":{"outputs_hidden":false}}
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
    return img

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
    Y, X = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    
    # Distance from center of solar disk
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_r = np.min([cx, cy]) * 0.98  # 98% of radius to cleanly cut the limb
    
    # Create circular mask
    mask = r < max_r
    
    # Normalize within the mask
    img_masked = img * mask
    valid_pixels = img_masked[mask]
    img_min = np.min(valid_pixels)
    img_max = np.max(valid_pixels)
    
    img_norm = (img_masked - img_min) / (img_max - img_min + 1e-8)
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
    img_uint8 = (img * 255).astype(np.uint8)
    
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

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T12:57:29.068021Z","iopub.execute_input":"2026-10-17T12:57:29.068326Z","iopub.status.idle":"2026-10-17T12:57:29.085043Z","shell.execute_reply.started":"2026-10-17T12:57:29.068307Z","shell.execute_reply":"2026-10-17T12:57:29.084303Z"},"jupyter":{"outputs_hidden":false}}
# CELL 4: PyTorch Dataset class (FIXED FOR STRING IDs)
# FIX: Bypasses pycocotools' buggy loadImgs() and getAnnIds() functions
# by directly accessing the underlying coco.imgs and coco.imgToAnns dictionaries.
# This prevents the "isArrayLike" bug where strings are iterated character-by-character.

class SolarDataset(Dataset):
    def __init__(self, coco, img_ids, img_dir, img_size=512, transforms=None):
        self.coco = coco
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        # FIX 1: Direct dictionary access (avoids loadImgs string bug)
        img_info = self.coco.imgs[img_id]
        file_name = img_info['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        
        # Load and normalize image
        img = load_image(img_path)
        if img is None:
            raise FileNotFoundError(f'Cannot read training image: {img_path}')
        
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # Apply preprocessing
        img_limb, _ = remove_limb_darkening(img)
        mag_map = get_magnetic_proxy(img_limb)
        
        # Stack into 3 channels: [H-alpha, H-alpha, Magnetic Map]
        img_3ch = np.stack([img_limb, img_limb, mag_map], axis=-1).astype(np.float32)
        img_3ch = cv2.resize(img_3ch, (self.img_size, self.img_size))
        
        # FIX 2: Direct dictionary access for annotations
        # coco.imgToAnns maps image_id directly to a list of annotation dicts
        anns = self.coco.imgToAnns.get(img_id, [])
        
        # Create combined binary mask at original 2048x2048 resolution
        mask_2048 = np.zeros((2048, 2048), dtype=np.float32)
        for ann in anns:
            # annToMask handles the polygon_list format perfectly
            m = self.coco.annToMask(ann)
            mask_2048 = np.maximum(mask_2048, m)
        
        # Resize mask to training size
        mask = cv2.resize(mask_2048, (self.img_size, self.img_size))
        mask = (mask > 0.5).astype(np.float32)
        
        # Apply augmentations
        if self.transforms:
            aug = self.transforms(image=img_3ch, mask=mask)
            img_3ch = aug['image']
            mask = aug['mask']
        
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
        
        return img_3ch, mask

# Define augmentation pipelines (Keep these below the class)
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    ToTensorV2()
])

val_transforms = A.Compose([
    ToTensorV2()
])

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T12:57:29.085996Z","iopub.execute_input":"2026-10-17T12:57:29.086316Z","iopub.status.idle":"2026-10-17T12:57:29.110368Z","shell.execute_reply.started":"2026-10-17T12:57:29.086283Z","shell.execute_reply":"2026-10-17T12:57:29.109732Z"},"jupyter":{"outputs_hidden":false}}
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


# --- 3. Agentic Cognitive Layer (Topology Refiner) ---
# This is our custom addition to solve the Panoptic Quality fragmentation problem.
# Standard CNNs often break filaments into multiple pieces.
# This layer uses dilated convolutions to "see" larger context and
# bridges gaps between fragmented pieces, guided by the magnetic map.
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
        
        # Agent Reasoning Gate: uses magnetic map to decide WHERE to connect
        # If magnetic flow is strong between two fragments, connect them
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch + 1, in_ch, 1),  # +1 for magnetic map channel
            nn.Sigmoid()
        )
        self.out_conv = nn.Conv2d(in_ch, 1, 1)

    def forward(self, x, mag_map):
        """
        Args:
            x: Feature map from decoder [B, C, H, W]
            mag_map: Magnetic orientation map [B, 1, H, W]
        Returns:
            Refined single-channel prediction [B, 1, H, W]
        """
        # Bridge gaps using dilated convolutions
        b1 = F.relu(self.bridge1(x))
        b2 = F.relu(self.bridge2(x))
        bridged = x + b1 + b2  # Residual: original + local bridge + long bridge
        
        # Gate based on magnetic alignment
        # The agent "reasons": if magnetic flow connects two regions, merge them
        g = self.gate(torch.cat([bridged, mag_map], dim=1))
        refined = bridged * g
        
        return self.out_conv(refined)


# --- 4. Full Mag-Agent U²-Net ---
class MagAgentU2Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        """
        Args:
            in_ch: Input channels (3: H-alpha, H-alpha, Magnetic Map)
            out_ch: Output channels (1: binary segmentation mask)
        """
        super().__init__()
        
        # === ENCODER ===
        # ponytail: fixed half-width backbone; compare wider models only after this baseline.
        self.enc1 = RSU(in_ch, 16, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = RSU(32, 32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = RSU(64, 64, 128)
        self.pool3 = nn.MaxPool2d(2)
        
        # === BOTTLENECK ===
        self.bottleneck = RSU(128, 128, 256)
        
        # === DECODER with Attention Gates ===
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att3 = AttentionGate(256, 128, 128)
        self.dec3 = RSU(384, 128, 128)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att2 = AttentionGate(128, 64, 64)
        self.dec2 = RSU(192, 64, 64)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att1 = AttentionGate(64, 32, 32)
        self.dec1 = RSU(96, 32, 32)
        
        # === AGENTIC COGNITIVE LAYER ===
        self.cognitive = CognitiveRefiner(32)

    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 3, H, W]
        Returns:
            Segmentation logits [B, 1, H, W]
        """
        # Extract magnetic map (channel 2) for the Cognitive Layer
        mag_map = x[:, 2:3, :, :]
        
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(e3))
        
        # Decoder path with Attention Gates
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, self.att3(b, e3)], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self.att2(d3, e2)], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self.att1(d2, e1)], dim=1))
        
        # Agentic Cognitive Refinement
        out = self.cognitive(d1, mag_map)
        
        return out

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T12:57:29.111399Z","iopub.execute_input":"2026-10-17T12:57:29.111762Z","iopub.status.idle":"2026-10-17T12:57:29.132426Z","shell.execute_reply.started":"2026-10-17T12:57:29.111730Z","shell.execute_reply":"2026-10-17T12:57:29.131805Z"},"jupyter":{"outputs_hidden":false}}
# CELL 6: Combined Dice + BCE Loss Function
# Dice Loss: Handles severe class imbalance (filaments are <5% of pixels)
# BCE Loss: Provides stable gradients for precise boundary detection
# Combined: dice_loss + bce_loss gives best results for thin structures

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model output (before sigmoid) [B, 1, H, W]
            targets: Ground truth binary mask [B, 1, H, W]
        Returns:
            Combined loss scalar
        """
        # Dice Loss
        probs = torch.sigmoid(logits.float())
        smooth = 1.0
        iflat = probs.view(-1)
        tflat = targets.float().view(-1)
        intersection = (iflat * tflat).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / 
                         (iflat.sum() + tflat.sum() + smooth))
        
        # BCE Loss (with logits - numerically stable)
        bce_loss = self.bce(logits, targets)
        
        return dice_loss + bce_loss

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T12:57:29.133389Z","iopub.execute_input":"2026-10-17T12:57:29.133819Z","iopub.status.idle":"2026-10-17T12:57:29.166477Z","shell.execute_reply.started":"2026-10-17T12:57:29.133783Z","shell.execute_reply":"2026-10-17T12:57:29.165714Z"},"jupyter":{"outputs_hidden":false}}
# CELL 7: Training and Validation Loops
# Uses Automatic Mixed Precision (AMP) for:
#   - 2x faster training on modern GPUs
#   - Reduced activation memory for the 512x512 experiment
# Gradient clipping prevents exploding gradients with BCE loss.

def train_epoch(model, loader, optimizer, criterion, scaler):
    """Train for one epoch. Returns average loss."""
    model.train()
    epoch_loss = 0
    
    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(Config.DEVICE)
        masks = masks.to(Config.DEVICE)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            logits = model(images)
            loss = criterion(logits, masks)
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping (prevents exploding gradients)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(loader)


def create_fixed_split(coco, img_ids, path, val_fraction=0.2):
    """Create or reload a seeded split stratified by instance count and foreground area."""
    if os.path.exists(path):
        with open(path) as f:
            split = json.load(f)
        if set(split['train_ids'] + split['val_ids']) == set(img_ids):
            return split['train_ids'], split['val_ids']

    counts = np.array([len(coco.imgToAnns.get(img_id, [])) for img_id in img_ids])
    areas = np.array([
        sum(ann.get('area', 0) for ann in coco.imgToAnns.get(img_id, [])) /
        (coco.imgs[img_id]['height'] * coco.imgs[img_id]['width'])
        for img_id in img_ids
    ])

    def bins(values):
        edges = np.unique(np.quantile(values, [0.25, 0.5, 0.75]))
        return np.digitize(values, edges, right=True)

    rng = random.Random(Config.SEED)
    strata = {}
    for img_id, count_bin, area_bin in zip(img_ids, bins(counts), bins(areas)):
        strata.setdefault((int(count_bin), int(area_bin)), []).append(img_id)

    val_ids = []
    train_ids = []
    for group in strata.values():
        rng.shuffle(group)
        n_val = round(len(group) * val_fraction)
        val_ids.extend(group[:n_val])
        train_ids.extend(group[n_val:])

    target = round(len(img_ids) * val_fraction)
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    while len(val_ids) < target:
        val_ids.append(train_ids.pop())
    while len(val_ids) > target:
        train_ids.append(val_ids.pop())
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)

    with open(path, 'w') as f:
        json.dump({'seed': Config.SEED, 'train_ids': train_ids, 'val_ids': val_ids}, f, indent=2)
    return train_ids, val_ids


def panoptic_stats(pred_mask, annotations, coco, min_area=None):
    """Standard one-class PQ matching plus split/merge diagnostics for one image."""
    min_area = Config.MIN_AREA if min_area is None else min_area
    num_labels, labels = cv2.connectedComponents(pred_mask.astype(np.uint8))
    pred_areas = np.bincount(labels.ravel(), minlength=num_labels)
    pred_ids = np.flatnonzero(pred_areas >= min_area)
    pred_ids = pred_ids[pred_ids != 0]
    pred_index = {label: i for i, label in enumerate(pred_ids)}
    ious = np.zeros((len(annotations), len(pred_ids)), dtype=np.float32)

    for gt_idx, ann in enumerate(annotations):
        gt = coco.annToMask(ann).astype(bool)
        intersections = np.bincount(labels[gt], minlength=num_labels)
        gt_area = gt.sum()
        for label, pred_idx in pred_index.items():
            intersection = intersections[label]
            union = gt_area + pred_areas[label] - intersection
            ious[gt_idx, pred_idx] = intersection / union if union else 0.0

    candidates = np.argwhere(ious > Config.PQ_IOU_THRESHOLD)
    candidates = sorted(candidates, key=lambda pair: ious[tuple(pair)], reverse=True)
    matched_gt, matched_pred, iou_sum = set(), set(), 0.0
    for gt_idx, pred_idx in candidates:
        if gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(int(gt_idx))
            matched_pred.add(int(pred_idx))
            iou_sum += float(ious[gt_idx, pred_idx])

    diagnostic = ious > Config.DIAGNOSTIC_IOU_THRESHOLD
    return {
        'iou_sum': iou_sum,
        'tp': len(matched_gt),
        'fp': len(pred_ids) - len(matched_pred),
        'fn': len(annotations) - len(matched_gt),
        'splits': int((diagnostic.sum(axis=1) > 1).sum()) if len(annotations) else 0,
        'merges': int((diagnostic.sum(axis=0) > 1).sum()) if len(pred_ids) else 0,
    }


def finalize_panoptic(stats):
    denominator = stats['tp'] + 0.5 * stats['fp'] + 0.5 * stats['fn']
    return {
        **stats,
        'pq': stats['iou_sum'] / denominator if denominator else 1.0,
        'sq': stats['iou_sum'] / stats['tp'] if stats['tp'] else 0.0,
        'rq': stats['tp'] / denominator if denominator else 1.0,
    }

def probability_to_mask(probability, threshold, closing_radius=0):
    """Apply shared validation/submission thresholding at original image size."""
    mask = cv2.resize(
        (probability > threshold).astype(np.uint8),
        (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE),
        interpolation=cv2.INTER_NEAREST)
    if closing_radius:
        diameter = 2 * closing_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def validate(model, loader, criterion):
    """Return loss, foreground Dice, and one-class instance PQ diagnostics."""
    model.eval()
    epoch_loss = 0
    dice_sum = 0.0
    image_count = 0
    pq_totals = dict(iou_sum=0.0, tp=0, fp=0, fn=0, splits=0, merges=0)
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating"):
            images = images.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)
            
            with autocast():
                logits = model(images)
                loss = criterion(logits, masks)
            
            epoch_loss += loss.item()
            probabilities = torch.sigmoid(logits.float())
            preds = (probabilities > Config.THRESHOLD).float()
            dims = (1, 2, 3)
            dice = (2 * (preds * masks).sum(dims) + 1e-6) / (
                preds.sum(dims) + masks.sum(dims) + 1e-6)
            dice_sum += dice.sum().item()
            batch_ids = loader.dataset.img_ids[image_count:image_count + images.size(0)]
            for probability, img_id in zip(probabilities.cpu().numpy(), batch_ids):
                pred_2048 = probability_to_mask(
                    probability.squeeze(), Config.THRESHOLD, Config.CLOSING_RADIUS)
                stats = panoptic_stats(
                    pred_2048, loader.dataset.coco.imgToAnns.get(img_id, []), loader.dataset.coco)
                for key in pq_totals:
                    pq_totals[key] += stats[key]
            image_count += images.size(0)
    
    metrics = finalize_panoptic(pq_totals)
    metrics['foreground_dice'] = dice_sum / image_count
    return epoch_loss / len(loader), metrics


def tune_postprocessing(model, loader):
    """Cache validation probabilities once, then select settings by exact local PQ."""
    model.eval()
    cached = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc='Caching validation predictions'):
            with autocast():
                logits = model(images.to(Config.DEVICE))
            cached.extend(torch.sigmoid(logits.float()).cpu().numpy().astype(np.float16))

    best = None
    # Small coarse grid: 12 candidates, including V1's baseline settings.
    for threshold in (0.3, 0.4, 0.5):
        for min_area in (100, 200):
            for closing_radius in (0, 2):
                totals = dict(iou_sum=0.0, tp=0, fp=0, fn=0, splits=0, merges=0)
                for probability, img_id in zip(cached, loader.dataset.img_ids):
                    pred = probability_to_mask(
                        probability.squeeze(), threshold, closing_radius)
                    stats = panoptic_stats(
                        pred, loader.dataset.coco.imgToAnns.get(img_id, []),
                        loader.dataset.coco, min_area=min_area)
                    for key in totals:
                        totals[key] += stats[key]
                metrics = finalize_panoptic(totals)
                candidate = (metrics['pq'], threshold, min_area, closing_radius, metrics)
                if best is None or candidate[0] > best[0]:
                    best = candidate

    _, Config.THRESHOLD, Config.MIN_AREA, Config.CLOSING_RADIUS, metrics = best
    settings = dict(threshold=Config.THRESHOLD, min_area=Config.MIN_AREA,
                    closing_radius=Config.CLOSING_RADIUS, validation_pq=metrics['pq'])
    with open(os.path.join(Config.OUT_DIR, 'postprocessing_settings.json'), 'w') as f:
        json.dump(settings, f, indent=2)
    print(f"Selected post-processing: {settings}; validation PQ={metrics['pq']:.4f}, "
          f"SQ={metrics['sq']:.4f}, RQ={metrics['rq']:.4f}, "
          f"splits/merges={metrics['splits']}/{metrics['merges']}")
    return settings


def fit_model(train_loader, val_loader, lr, weight_decay, epochs, trial=None):
    """Use identical initialization for each trial; save weights only for the final run."""
    random.seed(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    model = MagAgentU2Net().to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = DiceBCELoss()
    scaler = GradScaler(enabled=Config.DEVICE == 'cuda')
    best_pq = -1.0
    history = []
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}; '
          f'size: {Config.IMG_SIZE}; batch: {Config.BATCH_SIZE}; lr: {lr:.2g}; wd: {weight_decay:.2g}')
    try:
        for epoch in range(epochs):
            if Config.DEVICE == 'cuda':
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
            val_loss, metrics = validate(model, val_loader, criterion)
            if not np.isfinite([train_loss, val_loss, metrics['pq']]).all():
                raise RuntimeError('Non-finite training or validation result')
            peak_gib = torch.cuda.max_memory_allocated() / 2**30 if Config.DEVICE == 'cuda' else 0.0
            seconds = time.perf_counter() - started
            print(f'Epoch {epoch + 1}/{epochs}: train={train_loss:.4f}, val={val_loss:.4f}, '
                  f"Dice={metrics['foreground_dice']:.4f}, PQ={metrics['pq']:.4f}, "
                  f"SQ={metrics['sq']:.4f}, RQ={metrics['rq']:.4f}, "
                  f"TP/FP/FN={metrics['tp']}/{metrics['fp']}/{metrics['fn']}, "
                  f"splits/merges={metrics['splits']}/{metrics['merges']}, "
                  f'{seconds:.1f}s, peak allocated={peak_gib:.2f} GiB')
            history.append(dict(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                                **metrics, seconds=seconds, peak_allocated_gib=peak_gib))
            if metrics['pq'] > best_pq:
                best_pq = metrics['pq']
                if trial is None:
                    torch.save(model.state_dict(), os.path.join(Config.OUT_DIR, 'best_lightweight.pth'))
            if trial is not None:
                trial.report(metrics['pq'], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            else:
                pd.DataFrame(history).to_csv(os.path.join(Config.OUT_DIR, 'lightweight_history.csv'), index=False)
        return best_pq
    finally:
        del model, optimizer, scaler
        gc.collect()
        if Config.DEVICE == 'cuda':
            torch.cuda.empty_cache()

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T12:57:29.167496Z","iopub.execute_input":"2026-10-17T12:57:29.167894Z","iopub.status.idle":"2026-10-17T12:57:29.187788Z","shell.execute_reply.started":"2026-10-17T12:57:29.167865Z","shell.execute_reply":"2026-10-17T12:57:29.187195Z"},"jupyter":{"outputs_hidden":false}}
# CELL 8: Inference and Kaggle Submission Generator
# This cell:
# 1. Runs the trained model on all test images
# 2. Upscales predictions from Config.IMG_SIZE back to 2048x2048
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

def generate_submission(model, test_paths, out_path):
    """
    Generate Kaggle submission CSV from trained model.
    
    Args:
        model: Trained MagAgentU2Net model
        test_paths: List of test image file paths
        out_path: Output path for submission.csv
    """
    model.eval()
    submissions = []
    
    with torch.no_grad():
        for img_path in tqdm(test_paths, desc="Generating Submission"):
            # Extract image ID from filename (e.g., "20220621185332Bh")
            base_id = os.path.basename(img_path).split('.')[0]
            
            # Load and preprocess
            img = load_image(img_path)
            if img is None:
                raise FileNotFoundError(f'Cannot read test image: {img_path}')
            
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img_limb, _ = remove_limb_darkening(img)
            mag_map = get_magnetic_proxy(img_limb)
            
            # Create 3-channel input
            img_3ch = np.stack([img_limb, img_limb, mag_map], axis=-1).astype(np.float32)
            img_1024 = cv2.resize(img_3ch, (Config.IMG_SIZE, Config.IMG_SIZE))
            
            # Convert to tensor and predict
            tensor = torch.tensor(img_1024).permute(2, 0, 1).unsqueeze(0).to(Config.DEVICE)
            
            with autocast():
                logits = model(tensor)
            
            # Threshold and upscale to original 2048x2048
            probability = torch.sigmoid(logits.float()).cpu().numpy().squeeze()
            pred_mask_2048 = probability_to_mask(
                probability, Config.THRESHOLD, Config.CLOSING_RADIUS)
            
            # Connected Components: separate individual filaments
            # This is CRITICAL for Panoptic Quality (avoids over-merging)
            num_labels, labels_im = cv2.connectedComponents(pred_mask_2048)
            
            for i in range(1, num_labels):  # Skip 0 (background)
                component_mask = (labels_im == i).astype(np.uint8)
                
                # Filter out tiny noise fragments
                # This improves PQ by reducing false positives
                if component_mask.sum() < Config.MIN_AREA:
                    continue
                
                # Encode to RLE format required by competition
                rle_str = pycoco_rle_encode(component_mask)
                fil_id = f"{base_id}_{i}"
                submissions.append({
                    'filament_id': fil_id,
                    'segmentation_rle': rle_str
                })
    
    # Save submission CSV
    df = pd.DataFrame(submissions, columns=['filament_id', 'segmentation_rle'])
    df.to_csv(out_path, index=False)
    print(f"\nSubmission saved to {out_path}")
    print(f"Total filaments detected: {len(df)}")
    print(f"Unique images: {df['filament_id'].str.rsplit('_', n=1).str[0].nunique()}")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T12:57:29.188680Z","iopub.execute_input":"2026-10-17T12:57:29.188989Z","iopub.status.idle":"2026-10-17T13:20:48.986374Z","shell.execute_reply.started":"2026-10-17T12:57:29.188957Z","shell.execute_reply":"2026-10-17T13:20:48.985452Z"},"jupyter":{"outputs_hidden":false}}
# CELL 9: Main Execution Pipeline
# This cell orchestrates the entire workflow:
# 1. Load COCO annotations JSON
# 2. Split images into train (80%) and validation (20%)
# 3. Create DataLoaders with augmentations
# 4. Initialize Mag-Agent U²-Net model
# 5. Run short Optuna trials, then train the winner and save its best PQ checkpoint
# 6. Load best model and generate Kaggle submission

# CELL 9: Main Execution Pipeline (FIXED FOR STRING IDs)
# FIX: Extracts string IDs directly from coco.imgs.keys() instead of getImgIds()

def main():
    os.makedirs(Config.OUT_DIR, exist_ok=True)
    # === 1. Load COCO Annotations ===
    print("=" * 60)
    print("Loading COCO annotations...")
    coco = COCO(Config.ANNOTATIONS_FILE)
    
    # FIX 3: Get string IDs directly from the imgs dictionary keys
    all_img_ids = list(coco.imgs.keys())
    print(f"Total images in JSON: {len(all_img_ids)}")
    
    # Filter to only images that actually exist on disk
    valid_img_ids = []
    for img_id in all_img_ids:
        file_name = coco.imgs[img_id]['file_name']
        img_path = os.path.join(Config.TRAIN_DIR, file_name)
        if os.path.exists(img_path):
            valid_img_ids.append(img_id)
            
    print(f"Found {len(valid_img_ids)} valid training images on disk.")
    print(f"Total annotations: {len(coco.dataset.get('annotations', []))}")
    print("=" * 60)
    
    # === 2. Fixed, seeded, stratified Train/Val Split (80/20) ===
    if len(valid_img_ids) < 2:
        raise ValueError('Need at least two readable training images for train/validation split')
    split_path = os.path.join(Config.OUT_DIR, 'split_seed_42.json')
    train_ids, val_ids = create_fixed_split(coco, valid_img_ids, split_path)
    
    print(f"Training images: {len(train_ids)}")
    print(f"Validation images: {len(val_ids)}")
    print(f"Fixed split: {split_path}")
    
    # === 3. Create Datasets and DataLoaders ===
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
    
    # === 4. Short sequential search; trial 0 is the untuned baseline ===
    # ponytail: tune only optimizer settings; keep architecture/resolution/split fixed.
    study = optuna.create_study(
        direction='maximize', sampler=optuna.samplers.TPESampler(seed=Config.SEED, n_startup_trials=3),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=1))
    study.enqueue_trial({'lr': Config.LR, 'weight_decay': Config.WEIGHT_DECAY})

    def objective(trial):
        lr = trial.suggest_float('lr', 5e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        return fit_model(train_loader, val_loader, lr, weight_decay, Config.TRIAL_EPOCHS, trial)

    study.optimize(objective, n_trials=Config.N_TRIALS, n_jobs=1, gc_after_trial=True)
    study.trials_dataframe().to_csv(os.path.join(Config.OUT_DIR, 'optuna_trials.csv'), index=False)
    print(f'Best short-trial PQ: {study.best_value:.4f}; {study.best_params}')

    # === 5. Fresh, longer training with the selected optimizer settings ===
    fit_model(train_loader, val_loader, **study.best_params, epochs=Config.EPOCHS)
    model_path = os.path.join(Config.OUT_DIR, 'best_lightweight.pth')
    
    # === 6. Generate Submission ===
    print("\n" + "=" * 60)
    print("Generating Kaggle submission...")
    model = MagAgentU2Net().to(Config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE, weights_only=True))

    # Tune post-processing on the held-out validation set; reuse these settings for test output.
    tune_postprocessing(model, val_loader)
    
    test_imgs = sorted(glob.glob(os.path.join(Config.TEST_DIR, '*.jpeg')))
    print(f"Found {len(test_imgs)} test images.")
    
    submission_path = os.path.join(Config.OUT_DIR, 'submission.csv')
    generate_submission(model, test_imgs, submission_path)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print(f"Submission file: {submission_path}")
    print("=" * 60)

# Run the pipeline
main()

# %% [code] {"jupyter":{"outputs_hidden":false}}


# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T13:20:48.987833Z","iopub.execute_input":"2026-10-17T13:20:48.988249Z","iopub.status.idle":"2026-10-17T13:21:02.648067Z","shell.execute_reply.started":"2026-10-17T13:20:48.988223Z","shell.execute_reply":"2026-10-17T13:21:02.647340Z"},"jupyter":{"outputs_hidden":false}}
# CELL 10: Visualize Sample Predictions

def visualize_predictions(model, coco, img_ids, img_dir, num_samples=4):
    """Display original images, ground-truth masks, and predictions."""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, 6 * num_samples))

    if num_samples == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for i, img_id in enumerate(img_ids[:num_samples]):
            img_info = coco.imgs[img_id]
            img_path = os.path.join(img_dir, img_info['file_name'])

            img = load_image(img_path)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {img_path}")

            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img_limb, _ = remove_limb_darkening(img)
            mag_map = get_magnetic_proxy(img_limb)

            img_3ch = np.stack(
                [img_limb, img_limb, mag_map],
                axis=-1
            ).astype(np.float32)

            resized_img = cv2.resize(
                img_3ch,
                (Config.IMG_SIZE, Config.IMG_SIZE)
            )

            tensor = (
                torch.from_numpy(resized_img)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(Config.DEVICE)
            )

            with autocast():
                logits = model(tensor)

            probability = torch.sigmoid(logits.float()).cpu().numpy().squeeze()
            prediction = probability_to_mask(
                probability, Config.THRESHOLD, Config.CLOSING_RADIUS)

            annotations = coco.imgToAnns.get(img_id, [])
            ground_truth = np.zeros(
                (Config.IMG_SIZE, Config.IMG_SIZE),
                dtype=np.float32
            )

            for annotation in annotations:
                mask = coco.annToMask(annotation)
                mask = cv2.resize(
                    mask,
                    (Config.IMG_SIZE, Config.IMG_SIZE),
                    interpolation=cv2.INTER_NEAREST
                )
                ground_truth = np.maximum(ground_truth, mask)

            axes[i, 0].imshow(img_limb, cmap='gray')
            axes[i, 0].set_title(f'Original: {img_info["file_name"]}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(ground_truth, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(cv2.resize(
                prediction, (Config.IMG_SIZE, Config.IMG_SIZE),
                interpolation=cv2.INTER_NEAREST), cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

    plt.tight_layout()

    output_path = os.path.join(Config.OUT_DIR, 'predictions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f'Visualization saved to: {output_path}')


model = MagAgentU2Net(
    in_ch=Config.IN_CHANNELS,
    out_ch=Config.OUT_CHANNELS
).to(Config.DEVICE)

checkpoint_path = os.path.join(
    Config.OUT_DIR,
    'best_lightweight.pth'
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=Config.DEVICE,
        weights_only=True
    )
)

coco = COCO(Config.ANNOTATIONS_FILE)
sample_ids = list(coco.imgs.keys())[:4]

visualize_predictions(
    model=model,
    coco=coco,
    img_ids=sample_ids,
    img_dir=Config.TRAIN_DIR,
    num_samples=4
)

# Run visualization (uses the trained model)
# Uncomment after training is complete:
# model = MagAgentU2Net(in_ch=Config.IN_CHANNELS).to(Config.DEVICE)
# model.load_state_dict(torch.load(os.path.join(Config.OUT_DIR, 'best_lightweight.pth')))
# coco = COCO(Config.ANNOTATIONS_FILE)
# visualize_predictions(model, coco, coco.getImgIds()[:4], Config.TRAIN_DIR)
