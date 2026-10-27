# V6 (`DinoV3MordenNet`) Phase-Wise Implementation Specification

This phase-wise implementation specification and companion code file are designed for your coding agent (Codex) to upgrade your `V5-heavy-model.py` script into **V6 (`DinoV3MordenNet`)**. 

The upgraded pipeline eliminates the 21M-parameter from-scratch overfitting bottleneck by freezing a **DINOv3 ViT-Small** backbone (`0` trainable encoder parameters) and routing its multi-scale semantic tokens into a **$\sim 2.3\text{M}$ trainable parameter decoder and refiner**. This combines the MORDEN paper’s architectural insights—`5x5` convolutions, `InstanceNorm2d`, `LeakyReLU(0.01)`, `Dropout(0.1)`, and hierarchical `AdaptivePyramidPooling` (APP)—with your existing 3-channel physics input (`[img_limb, valley_map, mag_map]`), `CognitiveRefiner`, `SkeletonRecall` + `LovaszHinge` compound loss, and a physics-grounded **DenseCRF $\rightarrow$ Hole Filling $\rightarrow$ DBSCAN** post-processing cascade.

---

## Part 1: Phase-Wise Implementation Plan for Coding Agent (Codex)

### Phase 1: Input Pipeline & MORDEN Augmentation Upgrade (`CELL 2` & `CELL 4`)
*   **Objective:** Retain the 3-channel physics tensor (`[img_limb, valley_map, mag_map]`) and 1px-dilated morphological `spine` target from V5, while upgrading the augmentation and loss hyperparameters to match the empirical findings of the MORDEN paper.
*   **Agent Tasks:**
    1. In `Config` (`CELL 2`), add `DINO_MODEL_NAME = 'facebook/dinov3-vits16-pretrain-lvd1689m'` (or `timm` equivalent `'vit_small_patch16_dinov3.lvd1689m'`) and set `FOCAL_ALPHA = 0.66` and `FOCAL_GAMMA = 2.0`, matching the paper's long-tail focal loss configuration for solar filaments that occupy less than $10\%$ of the solar disk.
    2. In `CELL 4` (`train_transforms`), expand Albumentations to include the paper's validated augmentation regime: vertical flipping (`p=0.5`), continuous rotation (`limit=45, p=0.9`), random gamma contrast adjustment (`gamma_limit=(50, 150), p=0.9`), and random scaling (`scale_limit=(-0.5, 0.0), p=0.5`).
    3. **Crucial Physics Guardrail:** Apply `RandomGamma` **only** to Channel 0 (`img_limb`) or before recomputing `valley_map` and `mag_map`, or restrict `RandomBrightnessContrast`/`RandomGamma` to intensity channels so the normalized `mag_map` (Channel 2) remains bounded in `[0, 1]`.

### Phase 2: Frozen DINOv3 Multi-Scale Feature Extractor (`CELL 5A`)
*   **Objective:** Replace the from-scratch `RSU` encoder and `VLMSemanticBridge` with a strictly frozen DINOv3 ViT-Small backbone (`patch_size=16`, `embed_dim=384`).
*   **Agent Tasks:**
    1. Load DINOv3 (`vit_small_patch16_dinov3.lvd1689m` via `timm` or `AutoModel.from_pretrained('facebook/dinov3-vits16-pretrain-lvd1689m')`). Note that `patch_size=16` divides `1024x1024` cleanly into a `64x64` token grid without interpolation artifacts.
    2. Set `param.requires_grad = False` for all parameters in the DINOv3 backbone and lock it in `.eval()` mode inside `train()` so dropout/normalization statistics inside the frozen transformer never drift.
    3. Extract intermediate patch token representations from 4 transformer depths—blocks `(2, 5, 8, 11)` (0-indexed)—and reshape each from `[B, N_patches, 384]` to `[B, 384, H/16, W/16]` (`[B, 384, 64, 64]`).

### Phase 3: MORDEN-Inspired Lightweight FPN Decoder (`CELL 5B`)
*   **Objective:** Build a $\sim 1.9\text{M}$ parameter multiscale pyramid decoder incorporating the MORDEN paper's core ablations.
*   **Agent Tasks:**
    1. **Instance Normalization (`InstanceNorm2d`) + `5x5` Convolutions:** Replace all `BatchNorm2d` layers in the feature extraction and decoding blocks with `nn.InstanceNorm2d(..., affine=True)` because Instance Normalization is sensitive to local intensity contrast variations rather than global statistics. Use `kernel_size=5, padding=2` for primary spatial convolutions, `nn.LeakyReLU(negative_slope=0.01, inplace=True)`, and `nn.Dropout2d(p=0.1)` after every block.
    2. **Adaptive Pyramid Pooling (`AdaptivePyramidPooling`):** Implement Equation (1) from the MORDEN paper:
       $$S = \left\{(h_i, w_i) \;\middle|\; h_i = \max\left(\left\lfloor \frac{H \cdot i}{L} \right\rfloor, 1\right),\; w_i = \max\left(\left\lfloor \frac{W \cdot i}{L} \right\rfloor, 1\right),\; i = 1, 2, \dots, L\right\}$$
       Use pooling depth $L = 5$. Deploy `AdaptivePyramidPooling` hierarchically across the multi-scale decoder stages rather than only at the bottleneck, as hierarchical APP outperforms single-stage ASPP/PPM.
    3. **Patch Attention Gate (`PatchAttentionGate`):** Implement the paper's global-local gating mechanism $\mathcal{G}(x_1, x_2) = g(x_1) \cdot x_2$ with dynamic patch size $S_{\text{patch}} = 2^{\lfloor \log_2 \sqrt{S_{x_1}} \rfloor}$ using dual `AdaptiveMaxPool2d` and `AdaptiveAvgPool2d`, `LayerNorm`, `LeakyReLU(0.01)`, and `Sigmoid` gating.
    4. **High-Resolution Shallow Stem (Branch Block):** Because DINOv3 operates at stride 16 (`64x64`), add a lightweight 2-stage shallow CNN stem (`base_channels=40`, `branch_channels=20`) directly on the `1024x1024` 3-channel input using MORDEN's branch-block concept to reintroduce fine pixel-level barb details at `H/4` (`256x256`) and `H/1` (`1024x1024`).

### Phase 4: Topology-Preserving `CognitiveRefiner` & Auxiliary Supervision (`CELL 5C`)
*   **Objective:** Preserve V5's anti-fragmentation `CognitiveRefiner` ($\sim 0.4\text{M}$ parameters, keeping total trainable parameters at $\sim 2.3\text{M}$) and maintain full API compatibility with `forward_model(model, images, return_aux=True)`.
*   **Agent Tasks:**
    1. Keep a lightweight 1x1 auxiliary head at the stride-8 (`128x128`) decoder stage to emit `aux_logits` (replacing `vlm_logits` in `DiceBCELoss` and providing the `vlm_prior` + global FiLM `gamma, beta` conditioning into `CognitiveRefiner`).
    2. Upgrade `CognitiveRefiner` to use `InstanceNorm2d` and `LeakyReLU(0.01)` internally while keeping its multi-scale dilated convolutions (`dilation=2, 4, 8`) and magnetic proxy (`mag_map`) + semantic prior gating.

### Phase 5: Differential Optimizer & Training Loop (`CELL 6` & `CELL 7`)
*   **Objective:** Ensure the optimizer only updates the $\sim 2.3\text{M}$ trainable parameters and never passes frozen DINOv3 tensors to `AdamW`.
*   **Agent Tasks:**
    1. In `main()` (`CELL 7`), filter optimizer parameters:
       `trainable_params = [p for p in model.parameters() if p.requires_grad]`
       Verify via assertion that `sum(p.numel() for p in trainable_params)` is `< 3_000_000`.
    2. Keep `DiceBCELoss` (`1.0 * Dice + 1.0 * Focal(alpha=0.66, gamma=2.0) + w_lovasz * LovaszHinge + 0.25 * SkeletonRecall + 0.15 * AuxBCE`) in strict FP32 (`torch.amp.autocast('cuda', enabled=False)`).

### Phase 6: Post-Processing Cascade: DenseCRF $\rightarrow$ Hole Filling $\rightarrow$ DBSCAN (`CELL 8`)
*   **Objective:** Replace V5's simple `3x3` morphological close + connected components with the MORDEN paper's 2K-resolution physics-guided edge snapping and spatial fragment clustering.
*   **Agent Tasks:**
    1. **2K DenseCRF Boundary Snapping:** Pass the `2048x2048` probability map `P_final` (unary potential) alongside the `2048x2048` limb-corrected H-alpha image `img_limb` (pairwise intensity potential) into DenseCRF using the paper's exact 2K hyperparameters from Table 3: $\omega^{(1)} = 40$, $\omega^{(2)} = 3$, $\theta_\alpha^2 = 8$, $\theta_\beta^2 = 12$, $\theta_\gamma^2 = 3$, and `iterations = 10`. Provide an automatic bilateral-solver fallback if `pydensecrf` is not installed in the Kaggle environment.
    2. **OpenCV Hole Filling:** Fill internal voids inside each detected filament region using contour filling (`cv2.findContours` / flood-fill) so every fragment is a simply connected domain.
    3. **DBSCAN Fragment Integration:** Instead of treating every disconnected contour as a separate filament instance (which destroys Panoptic Quality via False Positive splits), run `sklearn.cluster.DBSCAN` on foreground pixel coordinates (or fragment contour boundary coordinates for $O(N)$ speed) to merge nearby fragments along the same filament channel into a single instance ID. Use resolution-scaled parameters ($\epsilon \approx 22\text{--}45\text{ px}$, $\text{minPts} = 140$ or `Config.MIN_AREA = 200`).

---

## Part 2: Companion Code Snippets File (`v6_dinov3_morden_snippets.py`)

Save the following file as `v6_dinov3_morden_snippets.py` (or instruct your coding agent to splice these drop-in replacements directly into `CELL 2`, `CELL 4`, `CELL 5`, `CELL 7`, and `CELL 8` of `V5-heavy-model.py`). It preserves the exact signature of `MagAgentU2Net(in_ch, out_ch)` and `postprocess_probability(probability, img_limb_2048)` so your checkpointing, preflight checks, and native evaluation work seamlessly.

```python
# ==============================================================================
# FILE: v6_dinov3_morden_snippets.py
# Drop-in replacement modules for V5-heavy-model.py implementing:
#   1. Frozen DINOv3 ViT-Small Backbone (0 trainable params)
#   2. MORDEN Decoder (5x5 Conv, InstanceNorm2d, LeakyReLU, Dropout, APP, PAG)
#   3. Topology-Preserving CognitiveRefiner + Magnetic Proxy Gating
#   4. Post-Processing: 2K DenseCRF -> OpenCV Hole Filling -> DBSCAN Clustering
# ==============================================================================

import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Optional dependencies for Phase 2 (DINOv3) and Phase 6 (DenseCRF + DBSCAN)
try:
    import timm
except ImportError:
    timm = None

try:
    from transformers import AutoModel
except ImportError:
    AutoModel = None

try:
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
except ImportError:
    dcrf = None

from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree


# ==============================================================================
# PHASE 1: CONFIG & AUGMENTATION UPGRADES (Drop into CELL 2 & CELL 4)
# ==============================================================================

class V6ConfigMixin:
    """Add/override these attributes inside Config in CELL 2."""
    # DINOv3 Backbone settings
    DINO_TIMM_NAME = 'vit_small_patch16_dinov3.lvd1689m'
    DINO_HF_NAME = 'facebook/dinov3-vits16-pretrain-lvd1689m'
    DINO_FALLBACK_TIMM = 'vit_small_patch14_dinov2.lvd142m'
    DINO_OUT_INDICES = (2, 5, 8, 11)  # 4 multi-scale transformer stages

    # MORDEN Hyperparameters (Table 2 of Hu et al., 2026)
    BASE_CHANNELS = 40
    MID_CHANNELS = 40
    BRANCH_CHANNELS = 20
    APP_POOLING_DEPTH = 5
    DROPOUT_RATE = 0.10
    LEAKY_RELU_SLOPE = 0.01
    FOCAL_ALPHA = 0.66
    FOCAL_GAMMA = 2.0

    # DenseCRF 2048x2048 Hyperparameters (Table 3 of Hu et al., 2026)
    USE_DENSECRF = True
    CRF_W1 = 40.0
    CRF_W2 = 3.0
    CRF_THETA_ALPHA_SQ = 8.0   # spatial std^2 for bilateral kernel at 2K
    CRF_THETA_BETA_SQ = 12.0   # intensity std^2 for bilateral kernel at 2K
    CRF_THETA_GAMMA_SQ = 3.0   # spatial std^2 for Gaussian smoothness kernel at 2K
    CRF_ITERATIONS = 10

    # DBSCAN Fragment Integration Hyperparameters (Section 3.1.2 of Hu et al., 2026)
    # Note: Set eps=25.0 to 45.0 depending on GONG vs CHASE pixel scale
    USE_DBSCAN = True
    DBSCAN_EPS = 28.0
    DBSCAN_MIN_PTS = 140


def build_morden_train_transforms():
    """
    Upgraded CELL 4 augmentation pipeline incorporating Table 1 of Hu et al. (2026):
    - Vertical flip (50%)
    - Rotation [-45, +45] deg (90%)
    - Scaling [0.5, 1.0) (50%)
    Note: Gamma transformation is applied to channel 0 in SolarDataset before
    computing valley_map and mag_map so magnetic proxy bounds remain intact.
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


# ==============================================================================
# PHASE 2: FROZEN DINOv3 MULTI-SCALE BACKBONE (Drop into CELL 5)
# ==============================================================================

class FrozenDINOv3Encoder(nn.Module):
    """
    Strictly frozen DINOv3 (or DINOv2 fallback) multi-scale feature extractor.
    0 trainable parameters. Extracts 4 intermediate spatial feature maps.
    """
    def __init__(
        self,
        timm_name='vit_small_patch16_dinov3.lvd1689m',
        hf_name='facebook/dinov3-vits16-pretrain-lvd1689m',
        out_indices=(2, 5, 8, 11),
        pretrained=True,
    ):
        super().__init__()
        self.out_indices = out_indices
        self.backend = None
        self.embed_dim = 384
        self.patch_size = 16

        # Register ImageNet normalization buffers for DINO input
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if timm is not None:
            for candidate in (timm_name, 'vit_small_patch14_dinov2.lvd142m'):
                try:
                    self.backbone = timm.create_model(
                        candidate,
                        pretrained=pretrained,
                        features_only=True,
                        out_indices=out_indices,
                        dynamic_img_size=True,
                    )
                    self.backend = 'timm'
                    self.embed_dim = self.backbone.feature_info.channels()[-1]
                    if 'patch14' in candidate:
                        self.patch_size = 14
                    print(f"[DINO Encoder] Loaded frozen timm backbone: {candidate} (dim={self.embed_dim})")
                    break
                except Exception as exc:
                    print(f"[DINO Encoder] timm {candidate} unavailable ({exc}); trying fallback...")

        if self.backend is None and AutoModel is not None:
            self.backbone = AutoModel.from_pretrained(hf_name, output_hidden_states=True)
            self.backend = 'hf'
            self.embed_dim = self.backbone.config.hidden_size
            self.patch_size = getattr(self.backbone.config, 'patch_size', 16)
            print(f"[DINO Encoder] Loaded frozen HF backbone: {hf_name} (dim={self.embed_dim})")

        if self.backend is None:
            raise RuntimeError("Neither timm nor transformers could load DINOv3/v2 backbone.")

        # Strictly freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        """Override train() so frozen DINOv3 always stays in eval mode."""
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = (x - self.mean) / self.std

        # Ensure dimensions are divisible by patch_size (16 for DINOv3, 14 for DINOv2)
        ps = self.patch_size
        if H % ps != 0 or W % ps != 0:
            target_h = int(math.ceil(H / ps) * ps)
            target_w = int(math.ceil(W / ps) * ps)
            x_in = F.interpolate(x_norm, size=(target_h, target_w), mode='bilinear', align_corners=False)
        else:
            target_h, target_w = H, W
            x_in = x_norm

        if self.backend == 'timm':
            feats = self.backbone(x_in)  # List of 4 tensors [B, 384, Hp, Wp]
            return [f.detach().float() for f in feats]

        # HuggingFace transformers path
        outputs = self.backbone(pixel_values=x_in, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of [B, seq_len, D]
        hp, wp = target_h // ps, target_w // ps
        num_patches = hp * wp
        feats = []
        for idx in self.out_indices:
            hs = hidden_states[idx + 1]
            # Strip CLS + register tokens at the front of the sequence
            patch_tokens = hs[:, -num_patches:, :]
            fmap = patch_tokens.transpose(1, 2).contiguous().reshape(B, self.embed_dim, hp, wp)
            feats.append(fmap.detach().float())
        return feats


# ==============================================================================
# PHASE 3: MORDEN MODULES (5x5 Conv + InstanceNorm2d + APP + PAG + Branch Block)
# ==============================================================================

class MordenDoubleConv(nn.Module):
    """
    Figure 1(b) & Section 2.1.1 of Hu et al. (2026):
    5x5 Conv -> InstanceNorm2d -> LeakyReLU(0.01) (x2) + Dropout2d(0.1).
    Uses depthwise-separable 5x5 option when channels > 64 to keep params < 2M.
    """
    def __init__(self, in_ch, out_ch, kernel_size=5, dropout=0.10, slope=0.01, separable=False):
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
    S = {(h_i, w_i) | h_i = max(floor(H * i / L), 1), w_i = max(floor(W * i / L), 1), i=1..L}
    """
    def __init__(self, in_ch, out_ch, depth=5, dropout=0.10, slope=0.01):
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
        # Arithmetic progression of pooling scales (capped to avoid excessive memory at 256x256)
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
    G(x1, x2) = g(x1) * x2 with patch size S_patch = 2^(floor(log2(sqrt(S_x1)))).
    """
    def __init__(self, in_ch, out_ch, slope=0.01, dropout=0.10):
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

        # Equation (2): S_patch = 2^(floor(log2(sqrt(S_x1))))
        s_x1 = max(min(H, W), 4)
        s_patch = max(2 ** int(math.floor(math.log2(math.sqrt(s_x1)))), 2)
        grid_h, grid_w = max(H // s_patch, 1), max(W // s_patch, 1)

        # Dual pooling per patch -> channel/spatial gate
        p_max = F.adaptive_max_pool2d(x1_p, (grid_h, grid_w)).permute(0, 2, 3, 1)  # [B, gh, gw, C]
        p_avg = F.adaptive_avg_pool2d(x1_p, (grid_h, grid_w)).permute(0, 2, 3, 1)

        p_max = self.norm_out1(self.linear_max(self.act(self.norm_max(p_max))))
        p_avg = self.norm_out2(self.linear_avg(self.act(self.norm_avg(p_avg))))

        gate_grid = torch.sigmoid(p_max + p_avg).permute(0, 3, 1, 2).contiguous()
        gate = F.interpolate(gate_grid, size=(H, W), mode='bilinear', align_corners=False)
        return self.drop(x2 * gate)


class MordenBranchBlock(nn.Module):
    """
    Section 2.1.4 of Hu et al. (2026):
    Hierarchical shortcut from embedded input into multi-scale stages using
    Adaptive Pooling + Double Convolution + Patch Attention Gate.
    """
    def __init__(self, in_ch, branch_ch):
        super().__init__()
        self.dc = MordenDoubleConv(in_ch, branch_ch, kernel_size=5, separable=False)
        self.pag = PatchAttentionGate(in_ch, branch_ch)

    def forward(self, embedded_info, target_size):
        pooled = F.adaptive_avg_pool2d(embedded_info, output_size=target_size)
        feat = self.dc(pooled)
        return self.pag(pooled, feat)


# ==============================================================================
# PHASE 4: COGNITIVE REFINER & FULL DINOV3-MORDEN NETWORK (Replaces CELL 5)
# ==============================================================================

class CognitiveRefinerV6(nn.Module):
    """
    Upgraded CognitiveRefiner preserving V5's dilated topology bridging (d=2, 4, 8)
    and magnetic proxy gating, updated with InstanceNorm2d and LeakyReLU(0.01).
    """
    def __init__(self, in_ch=64, slope=0.01):
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
    Drop-in replacement for V5's MagAgentU2Net (~2.1M trainable parameters).
    - Frozen DINOv3 ViT-Small Encoder (0 trainable parameters)
    - MORDEN Decoder (5x5 Convs, InstanceNorm2d, APP, PAG, Branch Blocks)
    - CognitiveRefinerV6 with Magnetic Proxy & Auxiliary Semantic Prior
    """
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        base_ch = getattr(V6ConfigMixin, 'BASE_CHANNELS', 40)
        branch_ch = getattr(V6ConfigMixin, 'BRANCH_CHANNELS', 20)
        fpn_ch = 96
        refiner_ch = 64

        # 1. Frozen DINOv3 Backbone (0 trainable params)
        self.dino = FrozenDINOv3Encoder()
        dino_dim = self.dino.embed_dim

        # 2. Shallow High-Resolution Stem & MORDEN Branch Blocks
        self.embed_stem = MordenDoubleConv(in_ch, base_ch, kernel_size=5, separable=False)
        self.branch_s4 = MordenBranchBlock(base_ch, branch_ch)  # H/4 (256x256)
        self.branch_s8 = MordenBranchBlock(base_ch, branch_ch)  # H/8 (128x128)

        # 3. Lateral 1x1 Adapters from 4 Frozen DINOv3 Layers
        self.lat1 = nn.Sequential(nn.Conv2d(dino_dim, fpn_ch, 1, bias=False), nn.InstanceNorm2d(fpn_ch, affine=True))
        self.lat2 = nn.Sequential(nn.Conv2d(dino_dim, fpn_ch, 1, bias=False), nn.InstanceNorm2d(fpn_ch, affine=True))
        self.lat3 = nn.Sequential(nn.Conv2d(dino_dim, fpn_ch, 1, bias=False), nn.InstanceNorm2d(fpn_ch, affine=True))
        self.lat4 = nn.Sequential(nn.Conv2d(dino_dim, fpn_ch, 1, bias=False), nn.InstanceNorm2d(fpn_ch, affine=True))

        # 4. Hierarchical Adaptive Pyramid Pooling (APP) + PAG Decoder
        self.app_deep = AdaptivePyramidPooling(fpn_ch * 2, fpn_ch, depth=5)
        self.dec_s16 = MordenDoubleConv(fpn_ch * 2, fpn_ch, kernel_size=5, separable=True)

        self.pag_s8 = PatchAttentionGate(fpn_ch, fpn_ch)
        self.dec_s8 = MordenDoubleConv(fpn_ch + branch_ch, fpn_ch, kernel_size=5, separable=True)
        self.app_s8 = AdaptivePyramidPooling(fpn_ch, fpn_ch, depth=5)

        self.pag_s4 = PatchAttentionGate(fpn_ch, fpn_ch)
        self.dec_s4 = MordenDoubleConv(fpn_ch + branch_ch, refiner_ch, kernel_size=5, separable=True)
        self.app_s4 = AdaptivePyramidPooling(refiner_ch, refiner_ch, depth=5)

        # Full-resolution fusion with shallow stem before CognitiveRefiner
        self.dec_s1 = MordenDoubleConv(refiner_ch + base_ch, refiner_ch, kernel_size=5, separable=True)

        # 5. Auxiliary Head & FiLM Conditioning for CognitiveRefiner
        self.aux_head = nn.Conv2d(fpn_ch, out_ch, 1)
        self.film_head = nn.Sequential(
            nn.Linear(fpn_ch, refiner_ch * 2),
            nn.Tanh(),
        )
        self.cognitive = CognitiveRefinerV6(refiner_ch)

    def forward(self, x, return_aux=False):
        B, _, H, W = x.shape
        H4, W4 = H // 4, W // 4
        H8, W8 = H // 8, W // 8
        H16, W16 = H // 16, W // 16
        mag_map = x[:, 2:3, :, :]

        # Shallow full-res embedding + MORDEN branch shortcuts
        s1_embed = self.embed_stem(x)                       # [B, 40, H, W]
        b_s4 = self.branch_s4(s1_embed, (H4, W4))           # [B, 20, H/4, W/4]
        b_s8 = self.branch_s8(s1_embed, (H8, W8))           # [B, 20, H/8, W/8]

        # Frozen DINOv3 multi-scale features
        f1, f2, f3, f4 = self.dino(x)
        p1 = F.interpolate(self.lat1(f1), size=(H16, W16), mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.lat2(f2), size=(H16, W16), mode='bilinear', align_corners=False)
        p3 = F.interpolate(self.lat3(f3), size=(H16, W16), mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.lat4(f4), size=(H16, W16), mode='bilinear', align_corners=False)

        # Bottleneck APP fusion at H/16 (64x64)
        d16 = self.app_deep(torch.cat([p3, p4], dim=1), target_size=(H16, W16))
        d16 = self.dec_s16(torch.cat([d16, p2], dim=1))

        # Decode to H/8 (128x128) with PAG + Branch Block + APP
        p1_up8 = F.interpolate(p1, size=(H8, W8), mode='bilinear', align_corners=False)
        d16_up8 = F.interpolate(d16, size=(H8, W8), mode='bilinear', align_corners=False)
        gated_s8 = self.pag_s8(p1_up8, d16_up8)
        d8 = self.dec_s8(torch.cat([gated_s8, b_s8], dim=1))
        d8 = d8 + self.app_s8(d8, target_size=(H8, W8))

        # Auxiliary semantic logits & FiLM parameters at H/8
        aux_logits = self.aux_head(d8)
        pooled_context = F.adaptive_avg_pool2d(d8, 1).flatten(1)
        gamma, beta = torch.chunk(self.film_head(pooled_context).unsqueeze(-1).unsqueeze(-1), 2, dim=1)

        # Decode to H/4 (256x256) with PAG + Branch Block + APP
        d8_up4 = F.interpolate(d8, size=(H4, W4), mode='bilinear', align_corners=False)
        p1_up4 = F.interpolate(p1, size=(H4, W4), mode='bilinear', align_corners=False)
        gated_s4 = self.pag_s4(p1_up4, d8_up4)
        d4 = self.dec_s4(torch.cat([gated_s4, b_s4], dim=1))
        d4 = d4 + self.app_s4(d4, target_size=(H4, W4))

        # Upsample to H/1 (1024x1024) and fuse with shallow stem
        d4_up1 = F.interpolate(d4, size=(H, W), mode='bilinear', align_corners=False)
        d1 = self.dec_s1(torch.cat([d4_up1, s1_embed], dim=1))

        # CognitiveRefiner guided by Magnetic Proxy and Auxiliary Prior
        vlm_prior = torch.sigmoid(F.interpolate(aux_logits, size=(H, W), mode='bilinear', align_corners=False))
        out = self.cognitive(d1, mag_map, vlm_prior=vlm_prior, gamma=gamma, beta=beta)

        if return_aux or getattr(self, '_dp_return_aux', False):
            return out, aux_logits
        return out


# ==============================================================================
# PHASE 6: POST-PROCESSING CASCADE: DENSECRF -> HOLE FILLING -> DBSCAN (CELL 8)
# ==============================================================================

def apply_densecrf_2k(prob_2k, img_limb_2k):
    """
    Applies DenseCRF at 2048x2048 using Table 3 parameters from Hu et al. (2026):
    w1=40, w2=3, theta_alpha^2=8, theta_beta^2=12, theta_gamma^2=3, iterations=10.
    Falls back to edge-preserving bilateral filtering if pydensecrf is unavailable.
    """
    H, W = prob_2k.shape
    prob_fg = np.clip(prob_2k.astype(np.float32), 1e-6, 1.0 - 1e-6)
    img_u8 = np.clip(img_limb_2k * 255.0, 0, 255).astype(np.uint8)

    if dcrf is None:
        # Edge-preserving Joint Bilateral Fallback when pydensecrf Cython wheel is absent
        guide_f = img_u8.astype(np.float32) / 255.0
        smoothed = cv2.bilateralFilter(prob_fg, d=7, sigmaColor=0.10, sigmaSpace=3.0)
        # Snap edges toward high local contrast gradient
        grad_x = cv2.Sobel(guide_f, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(guide_f, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        grad_norm = grad_mag / (grad_mag.max() + 1e-6)
        return np.clip(smoothed - 0.05 * grad_norm * (0.5 - smoothed), 0.0, 1.0)

    # Exact Gibbs Energy formulation (Equations 3-7 & Table 3 of Hu et al., 2026)
    softmax_probs = np.stack([1.0 - prob_fg, prob_fg], axis=0)  # [2, H, W]
    unary = unary_from_softmax(softmax_probs)
    crf = dcrf.DenseCRF2D(W, H, 2)
    crf.setUnaryEnergy(unary)

    # Smoothness kernel: w^(2) * exp(-|p_i - p_j|^2 / (2 * theta_gamma^2))
    sxy_gamma = math.sqrt(getattr(V6ConfigMixin, 'CRF_THETA_GAMMA_SQ', 3.0))
    crf.addPairwiseGaussian(
        sxy=(sxy_gamma, sxy_gamma),
        compat=getattr(V6ConfigMixin, 'CRF_W2', 3.0),
    )

    # Appearance kernel: w^(1) * exp(-|p_i - p_j|^2 / (2 * theta_alpha^2) - |I_i - I_j|^2 / (2 * theta_beta^2))
    sxy_alpha = math.sqrt(getattr(V6ConfigMixin, 'CRF_THETA_ALPHA_SQ', 8.0))
    srgb_beta = math.sqrt(getattr(V6ConfigMixin, 'CRF_THETA_BETA_SQ', 12.0))
    img_rgb = np.repeat(img_u8[:, :, np.newaxis], 3, axis=2)
    crf.addPairwiseBilateral(
        sxy=(sxy_alpha, sxy_alpha),
        srgb=(srgb_beta, srgb_beta, srgb_beta),
        rgbim=np.ascontiguousarray(img_rgb),
        compat=getattr(V6ConfigMixin, 'CRF_W1', 40.0),
    )

    Q = crf.inference(getattr(V6ConfigMixin, 'CRF_ITERATIONS', 10))
    return np.asarray(Q, dtype=np.float32)[1].reshape((H, W))


def fill_fragment_holes(mask_u8):
    """
    Section 2.2 of Hu et al. (2026):
    Ensures every extracted filament fragment is a simply connected domain
    by filling internal holes via external contour rasterization.
    """
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask_u8, dtype=np.uint8)
    if contours:
        cv2.drawContours(filled, contours, -1, color=1, thickness=cv2.FILLED)
    return filled


def cluster_fragments_dbscan(mask_u8, eps=28.0, min_pts=140, min_area=200):
    """
    Section 2.3 & 3.1.2 of Hu et al. (2026):
    Integrates broken filament fragments into unified instance masks using DBSCAN
    while filtering small hallucination noise clusters.
    Uses contour-point KD-Tree / DBSCAN to run in <15ms on 2048x2048 images.
    """
    num_cc, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_cc <= 1:
        return []

    # Collect boundary coordinates for each connected component
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

    # Build adjacency graph of components whose minimum boundary distance <= eps
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
            # Fast bounding-box pre-reject before KD-tree query
            ax, ay, aw, ah = stats[cid_a, :4]
            bx, by, bw, bh = stats[cid_b, :4]
            if (ax > bx + bw + eps) or (bx > ax + aw + eps) or (ay > by + bh + eps) or (by > ay + ah + eps):
                continue
            dists, _ = trees[cid_b].query(pts_a, k=1, distance_upper_bound=eps)
            if np.any(np.isfinite(dists)):
                union(cid_a, cid_b)

    # Group merged components into final filament instances and filter by min_pts / min_area
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
    threshold=0.40,
    min_area=200,
    use_densecrf=True,
    use_dbscan=True,
):
    """
    Drop-in replacement for V5's `postprocess_probability`:
    1. 2K DenseCRF boundary refinement (if img_limb_2048 is passed)
    2. Thresholding + OpenCV Hole Filling (simply connected domain enforcement)
    3. DBSCAN fragment integration for Panoptic Quality (PQ) preservation
    """
    prob_2k = probability.detach().float().cpu().numpy().squeeze()
    if prob_2k.ndim != 2 or not np.isfinite(prob_2k).all():
        raise ValueError("Invalid native probability map")

    if use_densecrf and img_limb_2048 is not None:
        prob_2k = apply_densecrf_2k(prob_2k, img_limb_2048)

    mask_u8 = (prob_2k > threshold).astype(np.uint8)
    mask_u8 = fill_fragment_holes(mask_u8)

    if use_dbscan:
        return cluster_fragments_dbscan(
            mask_u8,
            eps=getattr(V6ConfigMixin, 'DBSCAN_EPS', 28.0),
            min_pts=getattr(V6ConfigMixin, 'DBSCAN_MIN_PTS', 140),
            min_area=min_area,
        )

    num_cc, labels_cc, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    return [
        (labels_cc == i).astype(np.uint8)
        for i in range(1, num_cc)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]
```

---

## Part 3: Integration Checklist for the Coding Agent

1. **Optimizer Parameter Filtering (`CELL 7`):**
   In `main()`, replace:
   ```python
   optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
   ```
   with:
   ```python
   trainable_params = [p for p in model.parameters() if p.requires_grad]
   print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,} / Total: {sum(p.numel() for p in model.parameters()):,}")
   optimizer = torch.optim.AdamW(trainable_params, lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
   ```
2. **Pass `img_limb` to `postprocess_probability` (`CELL 8`):**
   In `evaluate_native` and `generate_submission`, unpack `prepared, img_limb = prepare_input_3ch(image)` and pass `img_limb_2048=img_limb` into `postprocess_probability_v6(probability, img_limb_2048=img_limb)` so DenseCRF uses the original 2K resolution intensities.
3. **Tune `DBSCAN_EPS` on Validation Holdout (`CELL 8`):**
   Run `V5_MODE=evaluate` on your validation split across `DBSCAN_EPS in [15.0, 22.0, 28.0, 36.0, 45.0]` and inspect `splits` vs. `merges` in `native_validation.json` to maximize `pq_file_macro` for GONG's pixel resolution.