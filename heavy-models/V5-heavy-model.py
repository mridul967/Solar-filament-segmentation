# CELL 1: Downloading important binaries and importing libraries
# This cell installs all required packages and imports them.
# - pycocotools: For reading COCO JSON annotations and RLE encoding for submission
# - albumentations: For data augmentation (flips, rotations)
# - optuna: For optional hyperparameter tuning
# - timm: For potential pretrained backbones (optional)

!pip install optuna albumentations pycocotools timm transformers -q

import os
# Must be set before importing torch. This reduces allocator fragmentation for
# the large, temporary feature maps produced by 1024x1024 RSU blocks.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import glob
import random
from collections import defaultdict
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

try:
    from transformers import CLIPTokenizer, CLIPTextModel
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

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
    OUT_DIR = '/kaggle/working/'
    
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
    EPOCHS = 25           # Total training epochs
    LR = 2e-4             # Learning rate (Adam optimizer)
    WEIGHT_DECAY = 1e-4   # L2 regularization
    NUM_WORKERS = 2       # DataLoader workers
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # === MODEL PARAMETERS ===
    IN_CHANNELS = 3       # [Limb-corrected H-alpha, Filament Valley Map, Magnetic Proxy Map]
    OUT_CHANNELS = 1      # Single binary segmentation mask
    SEED = 42
    VLM_MODEL_NAME = 'openai/clip-vit-base-patch32'
    MODEL_WEIGHTS_NAME = 'best_model.pth'
    VLM_WEIGHTS_NAME = 'best_vlm.pth'
    SKEL_RECALL_WEIGHT = 0.25
    VLM_PRIOR_WEIGHT = 0.15
    
    # === SUBMISSION PARAMETERS ===
    MIN_AREA = 200        # Minimum pixel area to count as a valid filament (filters noise)
    THRESHOLD = 0.4       # Probability threshold for binary mask
    USE_TTA = True        # 4-fold test-time augmentation in probability space

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
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_limb, disk_mask = remove_limb_darkening(img)
    valley_map = get_filament_valley_map(img_limb, disk_mask)
    mag_map = get_magnetic_proxy(img_limb)
    mag_map[~disk_mask] = 0.0
    return np.stack([img_limb, valley_map, mag_map], axis=-1).astype(np.float32), img_limb

def rasterize_dilated_spine(anns, mask_2048, img_size=1024):
    """
    Rasterizes annotated COCO spine polylines with thickness=3 at 2048x2048
    before bilinear resizing to img_size so 1px spines never vanish under INTER_NEAREST.
    Falls back to eroded mask centerlines when an annotation has no valid spine coordinates.
    """
    spine_2048 = np.zeros((2048, 2048), dtype=np.float32)
    for ann in anns:
        raw_spine = np.asarray(ann.get('spine', []), dtype=np.float32)
        if raw_spine.ndim == 1 and len(raw_spine) >= 4 and len(raw_spine) % 2 == 0:
            pts = raw_spine.reshape(-1, 2)
            if np.all((pts[:, 0] >= 0) & (pts[:, 0] < 2048) & (pts[:, 1] >= 0) & (pts[:, 1] < 2048)):
                cv2.polylines(spine_2048, [np.rint(pts).astype(np.int32)], False, 1.0, thickness=3)
    if spine_2048.max() == 0 and mask_2048.max() > 0:
        spine_2048 = cv2.erode(mask_2048, np.ones((3, 3), np.uint8), iterations=1)
    spine_resized = cv2.resize(spine_2048, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    return (spine_resized > 0.2).astype(np.float32)

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T10:17:42.408949Z","iopub.execute_input":"2026-10-17T10:17:42.409278Z","iopub.status.idle":"2026-10-17T10:17:42.431490Z","shell.execute_reply.started":"2026-10-17T10:17:42.409260Z","shell.execute_reply":"2026-10-17T10:17:42.430710Z"}}
# CELL 4: PyTorch Dataset class (FIXED FOR STRING IDs)
# FIX: Bypasses pycocotools' buggy loadImgs() and getAnnIds() functions
# by directly accessing the underlying coco.imgs and coco.imgToAnns dictionaries.
# This prevents the "isArrayLike" bug where strings are iterated character-by-character.

class SolarDataset(Dataset):
    def __init__(self, coco, img_ids, img_dir, img_size=1024, transforms=None, return_spine=False):
        self.coco = coco
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.img_size = img_size
        self.transforms = transforms
        self.return_spine = return_spine

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
            img = np.random.randint(0, 65535, (2048, 2048), dtype=np.uint16)
        
        # Stack into 3 channels: [H-alpha, Filament Valley Map, Magnetic Map]
        img_3ch, _ = prepare_input_3ch(img)
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
        spine = rasterize_dilated_spine(anns, mask_2048, self.img_size)
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
            return img_3ch, mask, spine
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


# --- 2.5 Vision-Language Semantic Bridge (CRIS / DenseCLIP / CLIPSeg + CoOp) ---
# Aligns bottleneck visual features with domain-specific solar physics text prompts
# ("filament channel with barbs along PIL" vs "isotropic sunspot / limb darkening").
# Weights and preloaded CLIP text anchors are stored separately in `best_vlm.pth`.
class VLMSemanticBridge(nn.Module):
    POSITIVE_PROMPTS = (
        "an H-alpha solar observation showing an elongated dark solar filament plasma channel",
        "a thin sinuous solar filament spine with lateral barbs along a magnetic polarity inversion line",
        "dark anisotropic thread-like chromospheric filament structure on the solar disk",
        "continuous curved solar filament ribbon suspended above the solar photosphere",
    )
    NEGATIVE_PROMPTS = (
        "a circular isotropic dark sunspot umbra and penumbra in a solar active region",
        "background solar chromospheric granulation noise and bright plage regions",
        "smooth radial solar limb darkening edge artifact near the solar horizon",
        "uniform quiet sun photospheric background without filament channels",
    )

    def __init__(self, vis_ch=512, refiner_ch=64, embed_dim=512, proj_dim=128):
        super().__init__()
        # Deterministic orthogonal initialization for text anchors (overwritten when CLIP preloads)
        gen = torch.Generator().manual_seed(42)
        pos_init = F.normalize(torch.randn(len(self.POSITIVE_PROMPTS), embed_dim, generator=gen), dim=-1)
        neg_init = F.normalize(torch.randn(len(self.NEGATIVE_PROMPTS), embed_dim, generator=gen), dim=-1)
        self.register_buffer("pos_anchors", pos_init)
        self.register_buffer("neg_anchors", neg_init)

        # Learnable CoOp context adaptation tokens (Zhou et al., IJCV 2022)
        self.pos_ctx = nn.Parameter(torch.zeros(len(self.POSITIVE_PROMPTS), embed_dim))
        self.neg_ctx = nn.Parameter(torch.zeros(len(self.NEGATIVE_PROMPTS), embed_dim))

        # Dense visual-text projection (DenseCLIP / CRIS)
        self.txt_proj = nn.Linear(embed_dim, proj_dim, bias=False)
        self.vis_proj = nn.Sequential(
            nn.Conv2d(vis_ch, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.GELU(),
        )
        self.cross_attn = nn.MultiheadAttention(proj_dim, num_heads=4, batch_first=True)
        self.vis_out = nn.Sequential(
            nn.Conv2d(proj_dim + 2, vis_ch, 1),
            nn.BatchNorm2d(vis_ch),
            nn.Sigmoid(),
        )
        # FiLM modulation generator (CLIPSeg, CVPR 2022) for the 64-ch CognitiveRefiner
        self.film_head = nn.Sequential(
            nn.Linear(proj_dim, refiner_ch * 2),
            nn.Tanh(),
        )
        self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    @torch.no_grad()
    def preload_clip_anchors(self, model_name=Config.VLM_MODEL_NAME, device='cpu'):
        """Preload frozen CLIP text embeddings before training and store in buffers."""
        if not HAS_TRANSFORMERS:
            print("transformers not installed; using deterministic VLM anchor buffers.")
            return False
        try:
            tokenizer = CLIPTokenizer.from_pretrained(model_name)
            text_Encoder = CLIPTextModel.from_pretrained(model_name).to(device).eval()
            for prompts, buf_name in ((self.POSITIVE_PROMPTS, "pos_anchors"), (self.NEGATIVE_PROMPTS, "neg_anchors")):
                tokens = tokenizer(list(prompts), padding=True, truncation=True, return_tensors="pt").to(device)
                pooled = text_Encoder(**tokens).pooler_output.float()
                pooled = F.normalize(pooled, dim=-1)
                if pooled.shape[-1] == getattr(self, buf_name).shape[-1]:
                    getattr(self, buf_name).copy_(pooled.to(getattr(self, buf_name).device))
            del text_Encoder
            print(f"✓ Preloaded VLM text anchors from {model_name}")
            return True
        except Exception as exc:
            print(f"VLM preload fallback (offline/cached anchors used): {exc}")
            return False

    def forward(self, b):
        """
        Args:
            b: Bottleneck features [B, 512, H_b, W_b]
        Returns:
            b_mod: VLM-modulated bottleneck [B, 512, H_b, W_b]
            vlm_logits: Contrastive semantic prior logits [B, 1, H_b, W_b]
            gamma, beta: FiLM modulation tensors [B, 64, 1, 1]
        """
        B, C, H, W = b.shape
        pos_emb = F.normalize(self.txt_proj(F.normalize(self.pos_anchors + self.pos_ctx, dim=-1)), dim=-1)
        neg_emb = F.normalize(self.txt_proj(F.normalize(self.neg_anchors + self.neg_ctx, dim=-1)), dim=-1)

        vis_feat = self.vis_proj(b)  # [B, D, H, W]
        vis_norm = F.normalize(vis_feat, dim=1)
        scale = self.logit_scale.exp().clamp(1.0, 50.0)

        # Dense Pixel-Text Contrastive Similarity Maps (DenseCLIP)
        sim_pos = torch.einsum("bdhw,kd->bkhw", vis_norm, pos_emb).mean(dim=1, keepdim=True) * scale
        sim_neg = torch.einsum("bdhw,kd->bkhw", vis_norm, neg_emb).mean(dim=1, keepdim=True) * scale
        vlm_logits = sim_pos - sim_neg  # Positive where filament > sunspot/limb distractor

        # Vision-Language Cross-Attention (CRIS)
        tokens_q = vis_feat.flatten(2).transpose(1, 2)  # [B, H*W, D]
        all_txt = torch.cat([pos_emb, neg_emb], dim=0).unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        attn_out, _ = self.cross_attn(tokens_q, all_txt, all_txt, need_weights=False)
        attn_map = attn_out.transpose(1, 2).reshape(B, -1, H, W)

        gate = self.vis_out(torch.cat([attn_map, torch.sigmoid(sim_pos), torch.sigmoid(sim_neg)], dim=1))
        b_mod = b * (1.0 + gate)

        # Global VLM context -> FiLM (gamma, beta) for high-res CognitiveRefiner
        vlm_pool = attn_out.mean(dim=1)  # [B, D]
        gamma_beta = self.film_head(vlm_pool).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
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
        # If magnetic flow and VLM filament semantics are strong between two fragments, connect them
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
        self.vlm_bridge = VLMSemanticBridge(vis_ch=512, refiner_ch=64)
        
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
        
        # Bottleneck + VLM Semantic Conditioning
        b = self.bottleneck(self.pool3(e3))
        b, vlm_logits, gamma, beta = self.vlm_bridge(b)
        
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
        
        if return_aux:
            return out, vlm_logits
        return out

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T10:17:42.455368Z","iopub.execute_input":"2026-10-17T10:17:42.455610Z","iopub.status.idle":"2026-10-17T10:17:42.470128Z","shell.execute_reply.started":"2026-10-17T10:17:42.455593Z","shell.execute_reply":"2026-10-17T10:17:42.469456Z"}}
# CELL 6: Combined Dice + BCE + Skeleton Recall + VLM Prior Loss Function
# Dice Loss: Handles severe class imbalance (filaments are <5% of pixels)
# BCE Loss: Provides stable gradients for precise boundary detection
# Skeleton Recall Loss (Kirchhoff et al., ECCV 2024): Preserves spine continuity on a single head
# VLM Contrastive Prior Loss: Grounds bottleneck VLM similarity on true filament regions

class DiceBCELoss(nn.Module):
    def __init__(self, skel_weight=Config.SKEL_RECALL_WEIGHT, vlm_weight=Config.VLM_PRIOR_WEIGHT):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.skel_weight = skel_weight
        self.vlm_weight = vlm_weight

    def forward(self, logits, targets, spine_targets=None, vlm_logits=None):
        """
        Args:
            logits: Raw model output (before sigmoid) [B, 1, H, W]
            targets: Ground truth binary mask [B, 1, H, W]
            spine_targets: Optional dilated ground-truth filament spine mask [B, 1, H, W]
            vlm_logits: Optional bottleneck VLM contrastive logits [B, 1, H/8, W/8]
        Returns:
            Combined loss scalar
        """
        logits_f = logits.float()
        targets_f = targets.float()

        # Dice Loss (computed in float32 for AMP stability)
        probs = torch.sigmoid(logits_f)
        smooth = 1.0
        iflat = probs.view(-1)
        tflat = targets_f.view(-1)
        intersection = (iflat * tflat).sum()
        dice_loss = 1 - ((2. * intersection + smooth) / 
                         (iflat.sum() + tflat.sum() + smooth))
        
        # BCE Loss (with logits - numerically stable)
        bce_loss = self.bce(logits_f, targets_f)
        total_loss = dice_loss + bce_loss

        # Skeleton Recall Loss (ECCV 2024): penalizes false breaks along filament spines
        if spine_targets is not None and self.skel_weight > 0:
            sflat = spine_targets.float().view(-1)
            skel_recall = ((iflat * sflat).sum() + smooth) / (sflat.sum() + smooth)
            total_loss = total_loss + self.skel_weight * (1.0 - skel_recall)

        # Bottleneck VLM Contrastive Prior Supervision
        if vlm_logits is not None and self.vlm_weight > 0:
            vlm_target = F.adaptive_max_pool2d(targets_f, output_size=vlm_logits.shape[2:])
            vlm_loss = self.bce(vlm_logits.float(), vlm_target)
            total_loss = total_loss + self.vlm_weight * vlm_loss
        
        return total_loss

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

    for step, batch in enumerate(tqdm(loader, desc="Training")):
        if len(batch) == 3:
            images, masks, spines = batch
            spines = spines.to(Config.DEVICE, non_blocking=True)
        else:
            images, masks = batch
            spines = None
        images = images.to(Config.DEVICE, non_blocking=True)
        masks = masks.to(Config.DEVICE, non_blocking=True)

        # Do not underweight a final, incomplete accumulation group.
        accumulation_divisor = (
            remainder
            if remainder and step >= len(loader) - remainder
            else Config.ACCUM_STEPS
        )
        
        # Mixed precision forward pass
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=Config.DEVICE == 'cuda'):
            logits, vlm_logits = model(images, return_aux=True)
            loss = criterion(logits, masks, spine_targets=spines, vlm_logits=vlm_logits) / accumulation_divisor
        
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
        for batch in tqdm(loader, desc="Validating"):
            if len(batch) == 3:
                images, masks, spines = batch
                spines = spines.to(Config.DEVICE, non_blocking=True)
            else:
                images, masks = batch
                spines = None
            images = images.to(Config.DEVICE, non_blocking=True)
            masks = masks.to(Config.DEVICE, non_blocking=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=Config.DEVICE == 'cuda'):
                logits, vlm_logits = model(images, return_aux=True)
                loss = criterion(logits, masks, spine_targets=spines, vlm_logits=vlm_logits)
            
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

def predict_probability(model, tensor, use_tta=Config.USE_TTA):
    """Predict full-resolution [1, 1, 2048, 2048] probability map with optional 4-fold TTA."""
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=Config.DEVICE == 'cuda'):
        p0 = torch.sigmoid(model(tensor).float())
        if not use_tta:
            probs_1024 = p0
        else:
            ph = torch.flip(torch.sigmoid(model(torch.flip(tensor, dims=[-1])).float()), dims=[-1])
            pv = torch.flip(torch.sigmoid(model(torch.flip(tensor, dims=[-2])).float()), dims=[-2])
            pr = torch.rot90(torch.sigmoid(model(torch.rot90(tensor, 1, dims=[-2, -1])).float()), -1, dims=[-2, -1])
            probs_1024 = 0.25 * (p0 + ph + pv + pr)
    return F.interpolate(probs_1024, size=(Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE), mode='bilinear', align_corners=False)

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
    close_kernel = np.ones((3, 3), dtype=np.uint8)
    
    with torch.no_grad():
        for img_path in tqdm(test_paths, desc="Generating Submission"):
            # Extract image ID from filename (e.g., "20220621185332Bh")
            base_id = os.path.basename(img_path).split('.')[0]
            
            # Load and preprocess
            img = load_image(img_path)
            if img is None:
                img = np.random.randint(0, 65535, (2048, 2048), dtype=np.uint16)
            
            # Create 3-channel input
            img_3ch, _ = prepare_input_3ch(img)
            img_1024 = cv2.resize(img_3ch, (Config.IMG_SIZE, Config.IMG_SIZE))
            
            # Convert to tensor and predict
            tensor = torch.tensor(img_1024).permute(2, 0, 1).unsqueeze(0).to(
                Config.DEVICE, non_blocking=True
            )
            
            # Smooth bilinear probability upscale to 2048x2048 before thresholding
            prob_2048 = predict_probability(model, tensor, use_tta=Config.USE_TTA)
            pred_mask_2048 = (prob_2048 > Config.THRESHOLD).cpu().numpy().squeeze().astype(np.uint8)
            pred_mask_2048 = cv2.morphologyEx(pred_mask_2048, cv2.MORPH_CLOSE, close_kernel)
            
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
    if not df.empty:
        print(f"Unique images: {df['filament_id'].str.rsplit('_', n=1).str[0].nunique()}")

# %% [code] {"execution":{"iopub.status.busy":"2026-10-17T10:17:42.512338Z","iopub.execute_input":"2026-10-17T10:17:42.512631Z","iopub.status.idle":"2026-10-17T10:17:48.428416Z","shell.execute_reply.started":"2026-10-17T10:17:42.512613Z","shell.execute_reply":"2026-10-17T10:17:48.427243Z"}}
# CELL 9: Main Execution Pipeline
# This cell orchestrates the entire workflow:
# 1. Load COCO annotations JSON
# 2. Split images into train (80%) and validation (20%)
# 3. Create DataLoaders with augmentations
# 4. Initialize Mag-Agent U²-Net model + Preload VLM (`best_vlm.pth`)
# 5. Train with early stopping (save `best_model.pth` and `best_vlm.pth`)
# 6. Load best model + best VLM and generate Kaggle submission

# CELL 9: Main Execution Pipeline (FIXED FOR STRING IDs)
# FIX: Extracts string IDs directly from coco.imgs.keys() instead of getImgIds()

def seed_everything(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_v5_checkpoints(model, model_path, vlm_path):
    """Save segmentation weights to best_model.pth and VLM weights to best_vlm.pth."""
    unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    seg_state = {k: v for k, v in unwrapped.state_dict().items() if not k.startswith("vlm_bridge.")}
    torch.save(seg_state, model_path)
    torch.save(unwrapped.vlm_bridge.state_dict(), vlm_path)

def load_v5_checkpoints(model, model_path, vlm_path, device):
    """Load segmentation weights from best_model.pth and VLM weights from best_vlm.pth."""
    unwrapped = model.module if isinstance(model, nn.DataParallel) else model
    unwrapped.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True), strict=False
    )
    if os.path.exists(vlm_path):
        unwrapped.vlm_bridge.load_state_dict(
            torch.load(vlm_path, map_location=device, weights_only=True)
        )

def main():
    seed_everything(Config.SEED)
    os.makedirs(Config.OUT_DIR, exist_ok=True)
    # === 1. Load COCO Annotations ===
    print("=" * 60)
    print("Loading COCO annotations...")
    coco = COCO(Config.ANNOTATIONS_FILE)
    
    # FIX 3: Get string IDs directly from the imgs dictionary keys
    all_img_ids = list(coco.imgs.keys())
    print(f"Total images in JSON: {len(all_img_ids)}")
    
    # Filter to only images that actually exist on disk and group by physical file_name
    valid_img_ids = []
    file_to_ids = defaultdict(list)
    for img_id in all_img_ids:
        file_name = coco.imgs[img_id]['file_name']
        img_path = os.path.join(Config.TRAIN_DIR, file_name)
        if os.path.exists(img_path):
            valid_img_ids.append(img_id)
            file_to_ids[file_name].append(img_id)
            
    print(f"Found {len(valid_img_ids)} valid training records across {len(file_to_ids)} physical images.")
    print(f"Total annotations: {len(coco.dataset.get('annotations', []))}")
    print("=" * 60)
    
    # === 2. Group-Safe Train/Val Split (80/20 by physical file, keeping all annotation passes) ===
    unique_files = sorted(file_to_ids.keys())
    rng = random.Random(Config.SEED)
    rng.shuffle(unique_files)
    split_idx = int(0.8 * len(unique_files))
    train_files = set(unique_files[:split_idx])
    val_files = set(unique_files[split_idx:])
    train_ids = [img_id for f in unique_files[:split_idx] for img_id in file_to_ids[f]]
    val_ids = [img_id for f in unique_files[split_idx:] for img_id in file_to_ids[f]]
    assert not (train_files & val_files), "Physical file leak detected between train and val!"
    
    print(f"Training records: {len(train_ids)} ({len(train_files)} physical images)")
    print(f"Validation records: {len(val_ids)} ({len(val_files)} physical images)")
    
    # === 3. Create Datasets and DataLoaders ===
    train_ds = SolarDataset(
        coco, train_ids, Config.TRAIN_DIR, Config.IMG_SIZE, train_transforms, return_spine=True
    )
    val_ds = SolarDataset(
        coco, val_ids, Config.TRAIN_DIR, Config.IMG_SIZE, val_transforms, return_spine=True
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    # === 4. Initialize Model & Preload VLM Before Training ===
    model = MagAgentU2Net(in_ch=Config.IN_CHANNELS, out_ch=Config.OUT_CHANNELS)
    model_path = os.path.join(Config.OUT_DIR, Config.MODEL_WEIGHTS_NAME)
    vlm_path = os.path.join(Config.OUT_DIR, Config.VLM_WEIGHTS_NAME)
    if os.path.exists(vlm_path):
        model.vlm_bridge.load_state_dict(torch.load(vlm_path, map_location='cpu', weights_only=True))
        print(f"✓ Loaded existing VLM weights from {vlm_path}")
    else:
        model.vlm_bridge.preload_clip_anchors(Config.VLM_MODEL_NAME, device=Config.DEVICE)
        torch.save(model.vlm_bridge.state_dict(), vlm_path)
        print(f"✓ Saved initial VLM weights to {vlm_path}")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    model = model.to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
    criterion = DiceBCELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=Config.DEVICE == 'cuda')
    
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
    
    for epoch in range(Config.EPOCHS):
        print(f"\n{'='*40}")
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"{'='*40}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
        val_loss = validate(model, val_loader, criterion)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model & best VLM separately
        if val_loss < best_loss:
            best_loss = val_loss
            save_v5_checkpoints(model, model_path, vlm_path)
            print(f"✓ Saved new best model ({model_path}) and VLM ({vlm_path})! (val_loss: {val_loss:.4f})")
    
    # === 6. Generate Submission ===
    print("\n" + "=" * 60)
    print("Generating Kaggle submission...")
    load_v5_checkpoints(model, model_path, vlm_path, Config.DEVICE)
    
    test_imgs = sorted(glob.glob(os.path.join(Config.TEST_DIR, '*.jpeg')))
    print(f"Found {len(test_imgs)} test images.")
    
    submission_path = os.path.join(Config.OUT_DIR, 'submission.csv')
    generate_submission(model, test_imgs, submission_path)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print(f"Submission file: {submission_path}")
    print("=" * 60)

# Run the pipeline
if __name__ == "__main__":
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
            img_info = coco.imgs[img_id]
            img_path = os.path.join(img_dir, img_info['file_name'])
            
            # Load and preprocess
            img = load_image(img_path)
            img_3ch, img_limb = prepare_input_3ch(img)
            img_1024 = cv2.resize(img_3ch, (Config.IMG_SIZE, Config.IMG_SIZE))
            tensor = torch.tensor(img_1024).permute(2, 0, 1).unsqueeze(0).to(
                Config.DEVICE, non_blocking=True
            )
            
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=Config.DEVICE == 'cuda'):
                logits = model(tensor)
            
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

ANNOTATIONS_FILE = '/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'

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
