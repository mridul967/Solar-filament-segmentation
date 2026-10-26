Yes — you can attach **two types** of Vision Transformers depending on whether you want **Text-Only VLM Anchors** (current) or **Full Vision-Transformer (ViT) + Text VLM Fusion**:

### 1. Drop-In right now (Just change [`Config.VLM_MODEL_NAME`](file:///Users/vinodpandey/solar-filament-segmentation/heavy-models/V5-heavy-model.py#L85))
[`VLMSemanticBridge`](file:///Users/vinodpandey/solar-filament-segmentation/heavy-models/V5-heavy-model.py#L425-L518) uses `embed_dim=512`, so any `512-dim` CLIP model works out of the box with **zero code changes**:
- `'openai/clip-vit-base-patch32'` *(current default — fast, lightweight)*
- `'openai/clip-vit-base-patch16'` *(recommended upgrade — trained with $16\times16$ finer patch granularity, better for thin structures)*
- `'laion/CLIP-ViT-B-16-laion2B-s34B-b88K'` *(trained on 2B pairs, stronger texture/shape alignment)*

*(If you want `'openai/clip-vit-large-patch14'`, just set `embed_dim=768` in [`VLMSemanticBridge.__init__`](file:///Users/vinodpandey/solar-filament-segmentation/heavy-models/V5-heavy-model.py#L440)).*

---

### 2. Can you attach the **Vision Transformer Image Encoder** (`CLIPVisionModel` / `DINOv2` / `SigLIP`) too?
Right now, [`VLMSemanticBridge`](file:///Users/vinodpandey/solar-filament-segmentation/heavy-models/V5-heavy-model.py#L469-L488) uses `CLIPTextModel` + `MagAgentU2Net`'s CNN bottleneck (`b`).

If you want a **true Vision Transformer image backbone** attached **without running out of VRAM on $1024\times1024$ images**:
- **How to attach it safely:** Run a frozen **ViT (`CLIPVisionModel` from `openai/clip-vit-base-patch16` or `facebook/dinov2-base`)** on a downsampled $224\times224$ view of the input inside `VLMSemanticBridge`, and fuse its dense ViT patch tokens (`14×14`) with the U²-Net bottleneck (`128×128`) and text prompts.
- **Why this works on Kaggle T4s:** Running a ViT directly at $1024\times1024$ ($4096$ tokens) causes $O(N^2)$ CUDA OOM on 16 GB T4s; running the frozen ViT at $224\times224$ inside `VLMSemanticBridge` takes **<300 MB VRAM**, gives global ViT self-attention over the entire solar disk, and keeps `best_model.pth` 100% unchanged.

### Which ViT perform best here?
| Vision Transformer | HuggingFace ID | Why it helps Solar Filaments | VRAM Overhead |
|---|---|---|---|
| **CLIP ViT-B/16** *(Recommended)* | `openai/clip-vit-base-patch16` | Shared image-text space; patch tokens directly align with the solar physics text prompts | ~280 MB |
| **DINOv2 ViT-B/14** | `facebook/dinov2-base` | Best self-supervised dense boundary/curvilinear feature extractor in computer vision | ~320 MB |
| **SigLIP ViT-B/16** | `google/siglip-base-patch16-224` | Sigmoid contrastive loss (designed for multi-label dense presence rather than 1-of-N softmax) | ~290 MB |

Want me to wire `openai/clip-vit-base-patch16`'s **Vision Transformer (`CLIPVisionModel`)** (or `DINOv2`) into [`VLMSemanticBridge`](file:///Users/vinodpandey/solar-filament-segmentation/heavy-models/V5-heavy-model.py#L425-L518) so both ViT visual tokens and text prompts condition the U²-Net?

---
*Ponytail check: No files modified in this turn; risk: attaching a full-resolution $1024\times1024$ ViT without downsampling the ViT branch to $224\times224$ will exceed T4 16 GB VRAM.*
