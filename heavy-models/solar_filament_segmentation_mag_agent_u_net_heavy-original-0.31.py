# CELL 1: Downloading important binaries and importing libraries
# This cell installs all required packages and imports them.
# - pycocotools: For reading COCO JSON annotations and RLE encoding for submission
# - albumentations: For data augmentation (flips, rotations)
# - optuna: For optional hyperparameter tuning
# - timm: For potential pretrained backbones (optional)

!pip install optuna albumentations pycocotools timm -q

import os
# Must be set before importing torch. This reduces allocator fragmentation for
# the large, temporary feature maps produced by 1024x1024 RSU blocks.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

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
    BASE_DIR = '/content/dataset/MAGFiLO_1.0_Kaggle_2026'
    TRAIN_DIR = os.path.join(BASE_DIR, '/content/dataset/MAGFiLO_1.0_Kaggle_2026/train/train_images')
    TEST_DIR = os.path.join(BASE_DIR, '/content/dataset/MAGFiLO_1.0_Kaggle_2026/test/test_images')
    ANNOTATIONS_FILE = os.path.join(BASE_DIR, '/content/dataset/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    OUT_DIR = '/content/working/'
    
    # === IMAGE DIMENSIONS ===
    # GONG H-alpha images are 2048x2048.
    # We downscale to 1024x1024 for training (to fit GPU VRAM),
    # then upscale predictions back to 2048x2048 for submission.
    IMG_SIZE = 1024
    ORIGINAL_SIZE = 2048
    
    # === TRAINING HYPERPARAMETERS ===
    BATCH_SIZE = 4      # L4 can handle 4x 1024x1024 images
    ACCUM_STEPS = 1     # Drop accumulation, do it in one step
    EPOCHS = 25           # Total training epochs
    LR = 2e-4             # Learning rate (Adam optimizer)
    WEIGHT_DECAY = 1e-4   # L2 regularization
    NUM_WORKERS = 4       # DataLoader workers
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # === MODEL PARAMETERS ===
    IN_CHANNELS = 3       # [Limb-corrected H-alpha, Limb-corrected copy, Magnetic Proxy Map]
    OUT_CHANNELS = 1      # Single binary segmentation mask
    
    # === SUBMISSION PARAMETERS ===
    MIN_AREA = 200        # Minimum pixel area to count as a valid filament (filters noise)
    THRESHOLD = 0.4       # Probability threshold for binary mask

os.makedirs(Config.OUT_DIR, exist_ok=True)

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

# %% [code]
# CELL 3.1: Precompute 3-channel 1024x1024 training inputs (run once before training)
img_dir = Config.TRAIN_DIR
out_dir = os.path.join(Config.OUT_DIR, 'precomputed_1024')
os.makedirs(out_dir, exist_ok=True)

for path in tqdm(glob.glob(os.path.join(img_dir, '*.jpeg'))):
    img = load_image(path)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_limb, _ = remove_limb_darkening(img)
    mag_map = get_magnetic_proxy(img_limb)

    # Stack and resize to 1024 ONCE
    img_3ch = np.stack([img_limb, img_limb, mag_map], axis=-1).astype(np.float32)
    img_1024 = cv2.resize(img_3ch, (1024, 1024))

    # Save as numpy array
    np.save(
        os.path.join(out_dir, os.path.basename(path).replace('.jpeg', '.npy')),
        img_1024.astype(np.float16),
    )

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T10:17:42.408949Z","iopub.execute_input":"2026-10-17T10:17:42.409278Z","iopub.status.idle":"2026-10-17T10:17:42.431490Z","shell.execute_reply.started":"2026-10-17T10:17:42.409260Z","shell.execute_reply":"2026-10-17T10:17:42.430710Z"}}
# CELL 4: Fast PyTorch Dataset class using precomputed inputs
class SolarDatasetFast(Dataset):
    def __init__(self, coco, img_ids, precomputed_dir, img_size=1024, transforms=None):
        self.coco = coco
        self.img_ids = img_ids
        self.precomputed_dir = precomputed_dir
        self.transforms = transforms

    def __len__(self): return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        file_name = self.coco.imgs[img_id]['file_name']

        # 1. Load precomputed 1024x1024 3-channel image (Instant)
        npy_path = os.path.join(self.precomputed_dir, file_name.replace('.jpeg', '.npy'))
        img_3ch = np.load(npy_path).astype(np.float32)

        # 2. Load and resize mask
        anns = self.coco.imgToAnns.get(img_id, [])
        
        mask_2048 = np.zeros((2048, 2048), dtype=np.float32)
        for ann in anns:
            mask_2048 = np.maximum(mask_2048, self.coco.annToMask(ann))
        mask = cv2.resize(mask_2048, (1024, 1024))
        mask = (mask > 0.5).astype(np.float32)

        # 3. Augmentations
        if self.transforms:
            aug = self.transforms(image=img_3ch, mask=mask)
            img_3ch, mask = aug['image'], aug['mask']

        # Ensure tensor format
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
        self.enc1 = RSU(in_ch, 32, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = RSU(64, 64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = RSU(128, 128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # === BOTTLENECK ===
        self.bottleneck = RSU(256, 256, 512)
        
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

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T10:17:42.455368Z","iopub.execute_input":"2026-10-17T10:17:42.455610Z","iopub.status.idle":"2026-10-17T10:17:42.470128Z","shell.execute_reply.started":"2026-10-17T10:17:42.455593Z","shell.execute_reply":"2026-10-17T10:17:42.469456Z"}}
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
        probs = torch.sigmoid(logits)
        smooth = 1.0
        iflat = probs.view(-1)
        tflat = targets.view(-1)
        intersection = (iflat * tflat).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / 
                         (iflat.sum() + tflat.sum() + smooth))
        
        # BCE Loss (with logits - numerically stable)
        bce_loss = self.bce(logits, targets)
        
        return dice_loss + bce_loss

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T10:17:42.470986Z","iopub.execute_input":"2026-10-17T10:17:42.471330Z","iopub.status.idle":"2026-10-17T10:17:42.487967Z","shell.execute_reply.started":"2026-10-17T10:17:42.471300Z","shell.execute_reply":"2026-10-17T10:17:42.487125Z"}}
# CELL 7: Training and Validation Loops
# Uses Automatic Mixed Precision (AMP) for:
#   - 2x faster training on modern GPUs
#   - 50% less VRAM usage (critical for 1024x1024 images)
# Gradient clipping prevents exploding gradients with BCE loss.

def train_epoch(model, loader, optimizer, criterion, scaler):
    """Train for one epoch. Returns average loss."""
    model.train()
    epoch_loss = 0
    
    optimizer.zero_grad(set_to_none=True)
    remainder = len(loader) % Config.ACCUM_STEPS

    for step, (images, masks) in enumerate(tqdm(loader, desc="Training")):
        images = images.to(Config.DEVICE, non_blocking=True)
        masks = masks.to(Config.DEVICE, non_blocking=True)

        # Do not underweight a final, incomplete accumulation group.
        accumulation_divisor = (
            remainder
            if remainder and step >= len(loader) - remainder
            else Config.ACCUM_STEPS
        )
        
        # Mixed precision forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(images)
            loss = criterion(logits, masks) / accumulation_divisor
        
        # Scaled backward pass
        scaler.scale(loss).backward()

        if (step + 1) % Config.ACCUM_STEPS == 0 or step + 1 == len(loader):
            # Gradient clipping (prevents exploding gradients)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        epoch_loss += loss.item() * accumulation_divisor
    
    return epoch_loss / len(loader)

def validate(model, loader, criterion):
    """Validate model. Returns average loss."""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating"):
            images = images.to(Config.DEVICE, non_blocking=True)
            masks = masks.to(Config.DEVICE, non_blocking=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, masks)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(loader)

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
                img = np.random.randint(0, 65535, (2048, 2048), dtype=np.uint16)
            
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img_limb, _ = remove_limb_darkening(img)
            mag_map = get_magnetic_proxy(img_limb)
            
            # Create 3-channel input
            img_3ch = np.stack([img_limb, img_limb, mag_map], axis=-1).astype(np.float32)
            img_1024 = cv2.resize(img_3ch, (Config.IMG_SIZE, Config.IMG_SIZE))
            
            # Convert to tensor and predict
            tensor = torch.tensor(img_1024).permute(2, 0, 1).unsqueeze(0).to(
                Config.DEVICE, non_blocking=True
            )
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(tensor)
            
            # Threshold and upscale to original 2048x2048
            pred_mask_1024 = (torch.sigmoid(logits) > Config.THRESHOLD).cpu().numpy().squeeze()
            pred_mask_2048 = cv2.resize(
                pred_mask_1024.astype(np.uint8),
                (Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE),
                interpolation=cv2.INTER_NEAREST
            )
            
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
    df = pd.DataFrame(submissions)
    df.to_csv(out_path, index=False)
    print(f"\nSubmission saved to {out_path}")
    print(f"Total filaments detected: {len(df)}")
    print(f"Unique images: {df['filament_id'].str.rsplit('_', n=1).str[0].nunique()}")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T10:17:42.512338Z","iopub.execute_input":"2026-10-17T10:17:42.512631Z","iopub.status.idle":"2026-10-17T10:17:48.428416Z","shell.execute_reply.started":"2026-10-17T10:17:42.512613Z","shell.execute_reply":"2026-10-17T10:17:48.427243Z"}}
# CELL 9: Main Execution Pipeline
# This cell orchestrates the entire workflow:
# 1. Load COCO annotations JSON
# 2. Split images into train (80%) and validation (20%)
# 3. Create DataLoaders with augmentations
# 4. Initialize Mag-Agent U²-Net model
# 5. Train with early stopping (save best model)
# 6. Load best model and generate Kaggle submission

# CELL 9: Main Execution Pipeline (FIXED FOR STRING IDs)
# FIX: Extracts string IDs directly from coco.imgs.keys() instead of getImgIds()

def main():
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
    
    # === 2. Train/Val Split (80/20) ===
    split_idx = int(0.8 * len(valid_img_ids))
    train_ids = valid_img_ids[:split_idx]
    val_ids = valid_img_ids[split_idx:]
    
    print(f"Training images: {len(train_ids)}")
    print(f"Validation images: {len(val_ids)}")
    
    # === 3. Create Datasets and DataLoaders ===
    train_ds = SolarDatasetFast(coco, train_ids, os.path.join(Config.OUT_DIR, 'precomputed_1024'), Config.IMG_SIZE, train_transforms)
    val_ds = SolarDatasetFast(coco, val_ids, os.path.join(Config.OUT_DIR, 'precomputed_1024'), Config.IMG_SIZE, val_transforms)
    
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    # === 4. Initialize Model ===
    model = MagAgentU2Net(in_ch=Config.IN_CHANNELS, out_ch=Config.OUT_CHANNELS).to(Config.DEVICE)
    model = torch.compile(model)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    criterion = DiceBCELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Device: {Config.DEVICE}")
    print(f"Image Size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print(f"Loader Batch Size: {Config.BATCH_SIZE}")
    print(f"Gradient Accumulation Steps: {Config.ACCUM_STEPS}")
    print(f"Effective Batch Size: {Config.BATCH_SIZE * Config.ACCUM_STEPS}")
    print("=" * 60)
    
    # === 5. Training Loop ===
    best_loss = float('inf')
    model_path = os.path.join(Config.OUT_DIR, 'best_model.pth')
    
    for epoch in range(Config.EPOCHS):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"{'='*40}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss = validate(model, val_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            unwrapped_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(unwrapped_model.state_dict(), model_path)
            print(f"✓ Saved new best model! (val_loss: {val_loss:.4f})")
    
    # === 6. Generate Submission ===
    print("\n" + "=" * 60)
    print("Generating Kaggle submission...")
    unwrapped_model = model.module if isinstance(model, nn.DataParallel) else model
    unwrapped_model.load_state_dict(
        torch.load(model_path, map_location=Config.DEVICE, weights_only=True)
    )
    
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

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T10:17:48.429154Z","iopub.status.idle":"2026-10-17T10:17:48.429410Z","shell.execute_reply.started":"2026-10-17T10:17:48.429294Z","shell.execute_reply":"2026-10-17T10:17:48.429307Z"}}
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
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(img_dir, img_info['file_name'])
            
            # Load and preprocess
            img = load_image(img_path)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img_limb, _ = remove_limb_darkening(img)
            mag_map = get_magnetic_proxy(img_limb)
            
            img_3ch = np.stack([img_limb, img_limb, mag_map], axis=-1).astype(np.float32)
            img_1024 = cv2.resize(img_3ch, (Config.IMG_SIZE, Config.IMG_SIZE))
            tensor = torch.tensor(img_1024).permute(2, 0, 1).unsqueeze(0).to(
                Config.DEVICE, non_blocking=True
            )
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(tensor)
            
            pred = (torch.sigmoid(logits) > Config.THRESHOLD).cpu().numpy().squeeze()
            
            # Get ground truth mask
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
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
    plt.show()

# Run visualization (uses the trained model)
# Uncomment after training is complete:
# model = MagAgentU2Net(in_ch=Config.IN_CHANNELS).to(Config.DEVICE)
# model.load_state_dict(torch.load(os.path.join(Config.OUT_DIR, 'best_model.pth')))
# coco = COCO(Config.ANNOTATIONS_FILE)
# visualize_predictions(model, coco, coco.getImgIds()[:4], Config.TRAIN_DIR)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T10:17:48.430759Z","iopub.status.idle":"2026-10-17T10:17:48.431131Z","shell.execute_reply.started":"2026-10-17T10:17:48.430930Z","shell.execute_reply":"2026-10-17T10:17:48.430946Z"}}
# CELL: COCO JSON Diagnostic - Run this FIRST
import json
from pycocotools.coco import COCO

ANNOTATIONS_FILE = Config.ANNOTATIONS_FILE

# === 1. Raw JSON structure ===
with open(ANNOTATIONS_FILE, 'r') as f:
    raw = json.load(f)

print("="*60)
print("TOP-LEVEL KEYS:", list(raw.keys()))
print("="*60)

for key in raw.keys():
    val = raw[key]
    if isinstance(val, list):
        print(f"\n'{key}': list of {len(val)} items")
        if len(val) > 0:
            print(f"  First item: {val[0]}")
            print(f"  Keys in first item: {list(val[0].keys()) if isinstance(val[0], dict) else type(val[0])}")
    elif isinstance(val, dict):
        print(f"\n'{key}': dict with keys {list(val.keys())[:10]}")
    else:
        print(f"\n'{key}': {val}")

# === 2. Image ID types ===
if 'images' in raw:
    imgs = raw['images']
    print("\n" + "="*60)
    print(f"TOTAL IMAGES: {len(imgs)}")
    print(f"Image ID types: {set(type(x['id']).__name__ for x in imgs[:20])}")
    print(f"Sample image IDs: {[x['id'] for x in imgs[:5]]}")
    print(f"Sample file_names: {[x['file_name'] for x in imgs[:5]]}")
    print(f"Sample dimensions: {[(x.get('height'), x.get('width')) for x in imgs[:3]]}")
    print("="*60)

# === 3. Annotation details ===
if 'annotations' in raw:
    anns = raw['annotations']
    print(f"\nTOTAL ANNOTATIONS: {len(anns)}")
    print(f"Annotation keys: {list(anns[0].keys())}")
    print(f"Sample annotation: {anns[0]}")
    
    # Check segmentation format
    seg_types = set()
    for a in anns[:100]:
        seg = a.get('segmentation')
        if seg is None:
            seg_types.add('None')
        elif isinstance(seg, dict):
            seg_types.add('RLE_dict')
        elif isinstance(seg, list):
            seg_types.add('polygon_list')
        else:
            seg_types.add(type(seg).__name__)
    print(f"Segmentation formats found: {seg_types}")

# === 4. COCO API test ===
print("\n" + "="*60)
print("Testing COCO API...")
try:
    coco = COCO(ANNOTATIONS_FILE)
    img_ids = coco.getImgIds()
    print(f"coco.getImgIds() returned: {len(img_ids)} IDs")
    print(f"First 5 IDs: {img_ids[:5]}")
    print(f"ID types: {[type(x).__name__ for x in img_ids[:5]]}")
    
    # Try loading first image
    try:
        info = coco.loadImgs(img_ids[0])
        print(f"loadImgs success: {info}")
    except Exception as e:
        print(f"loadImgs FAILED: {e}")
        # Try direct access
        print(f"coco.imgs keys sample: {list(coco.imgs.keys())[:5]}")
        
except Exception as e:
    print(f"COCO API failed: {e}")

print("="*60)
print("DIAGNOSTIC COMPLETE")
