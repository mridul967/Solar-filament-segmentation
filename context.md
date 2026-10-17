Save the following as **`context.md`** and give it to your Codex / coding agent.

```md
# context.md — Solar Filament Segmentation Challenge 2026

## 1. Project Purpose

This repository / notebook is for the Kaggle competition:

**Solar Filament Segmentation Challenge 2026**  
Automatic segmentation of solar filaments from GONG H-alpha observations.

The goal is to produce pixel-level segmentation masks for individual solar filaments. Predicted masks should:

- Capture the full extent of each filament.
- Preserve fine structures such as barbs.
- Avoid including non-filament dark regions such as sunspots, plage, background noise, or limb-darkening artifacts.
- Avoid fragmentation, where one physical filament is predicted as multiple disconnected components.
- Avoid over-merging, where multiple distinct filaments are merged into one prediction.

The competition is not a simple binary segmentation task. The final scoring emphasizes instance-level structural quality, especially through the Panoptic Quality metric.

---

## 2. Competition Metrics

The competition evaluation mentions both Dice score and Panoptic Quality.

### 2.1 Dice Score

Dice measures pixel-level overlap between predicted and ground-truth masks.

The competition references:

```text
torchmetrics.segmentation.DiceScore
```

The public scoreboard shows the mean Dice score for approximately 50% of the test images.

### 2.2 Panoptic Quality

The primary leaderboard ranking metric is Panoptic Quality, PQ.

Reference:

```text
Panoptic Segmentation / Panoptic Quality metric
DOI: 10.1109/CVPR.2019.00963
```

Conceptually:

```text
PQ = sum IoU over true positives / (|TP| + 0.5 * |FP| + 0.5 * |FN|)
```

Where:

- TP: predicted segment matched to a ground-truth segment with sufficient IoU.
- FP: predicted segment that does not match a ground-truth segment.
- FN: ground-truth segment that is not matched.

### 2.3 Important Metric Implication

A model can have good pixel-wise Dice but poor PQ if it:

- Breaks one filament into many fragments.
- Merges multiple filaments into one mask.
- Produces many tiny false-positive islands.
- Misses thin structures such as barbs.

Therefore, post-processing and topology-aware refinement are important.

---

## 3. Core Strategy: Mag-Agent U²-Net

The solution being developed is called:

```text
Mag-Agent U²-Net
```

It combines:

1. **Attention U²-Net backbone**
   - Based on the paper:
     - Jiang, W.; Li, Z.
     - Automated High-Precision Recognition of Solar Filaments Based on an Improved U2-Net.
     - Universe 2024, 10, 381.
     - DOI: `10.3390/universe10100381`

2. **Pseudo-magnetic field proxy**
   - Derived from H-alpha images only.
   - Uses structure tensor coherence / orientation information.
   - Helps distinguish elongated filament structures from isotropic dark noise.

3. **Agentic Cognitive Layer**
   - Current implementation is a lightweight topology refinement module.
   - Uses dilated convolutions and magnetic-map-guided gating.
   - Intended to reconnect fragmented filament pieces.
   - Intended to improve structural continuity and Panoptic Quality.

---

## 4. Competition Constraints

Important rules from the competition:

- For inference, only H-alpha images in the test directory are used.
- The model may not use other ground-truth metadata during inference.
- External data may be used only under competition rules, but current implementation does not use external data.
- Submission must be a CSV with:
  - `filament_id`
  - `segmentation_rle`

The RLE must be generated using `pycocotools`.

---

## 5. Kaggle Dataset Structure

The dataset path discovered in the Kaggle environment is:

```text
/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/
```

Inside it:

```text
train/
    MAGFiLO_1.0_Annotations_kaggle2026_train.json
    train_images/
        *.jpeg

test/
    test_images/
        *.jpeg
```

The exact config paths used are:

```python
BASE_DIR = '/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026'
TRAIN_DIR = os.path.join(BASE_DIR, 'train/train_images')
TEST_DIR = os.path.join(BASE_DIR, 'test/test_images')
ANNOTATIONS_FILE = os.path.join(BASE_DIR, 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json')
OUT_DIR = '/kaggle/working/'
```

Dataset statistics discovered:

```text
Total training images in JSON: 1154
Valid training images found on disk: 1154
Total training annotations: 8199
Test images: approximately 180
Image size: 2048 x 2048
Image format: .jpeg
```

---

## 6. COCO Annotation Structure

The training annotations are in COCO format.

Top-level JSON keys:

```text
info
licenses
categories
images
annotations
```

### 6.1 Images

Each image entry has keys:

```text
license
file_name
url
height
width
date_captured
id
```

Important details:

- `id` is a **string**, not an integer.
- Example image id:

```text
040301-20140609195854Bh
```

- Example file name:

```text
20140609195854Bh.jpeg
```

- Image dimensions:

```text
height = 2048
width = 2048
```

### 6.2 Annotations

Each annotation entry has keys:

```text
segmentation
area
iscrowd
spine
image_id
bbox
category_id
id
```

Important details:

- `image_id` is also a **string**.
- `segmentation` is polygon format:

```text
polygon_list
```

Example structure:

```python
annotation["segmentation"] = [[x1, y1, x2, y2, ...]]
```

- `spine` is available but currently not used.
- `category_id` exists but is currently ignored.
- All annotations for an image are combined into one binary mask during training.

### 6.3 Critical COCO API Bug Avoidance

Do **not** use:

```python
coco.loadImgs(img_id)
```

when `img_id` is a bare string.

Reason:

`pycocotools` can treat strings as array-like and iterate through characters, causing errors such as:

```text
KeyError: '0'
```

Correct approach:

```python
img_info = coco.imgs[img_id]
anns = coco.imgToAnns.get(img_id, [])
```

This bypasses the buggy behavior and is faster.

---

## 7. Current Notebook Cell Map

The notebook is organized into the following cells.

### CELL 1: Install Dependencies and Imports

Installs:

```text
optuna
albumentations
pycocotools
timm
```

Imports:

```python
os
glob
numpy
pandas
cv2
matplotlib
tqdm
torch
torch.nn
torch.nn.functional
DataLoader
autocast
GradScaler
albumentations
COCO
maskUtils
```

### CELL 2: Configuration

Defines:

```python
class Config:
    BASE_DIR = ...
    TRAIN_DIR = ...
    TEST_DIR = ...
    ANNOTATIONS_FILE = ...
    OUT_DIR = ...

    IMG_SIZE = 1024
    ORIGINAL_SIZE = 2048

    BATCH_SIZE = 4
    EPOCHS = 25
    LR = 2e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    IN_CHANNELS = 3
    OUT_CHANNELS = 1

    MIN_AREA = 200
    THRESHOLD = 0.4
```

Important:

- Training is done at `1024 x 1024` to fit Kaggle GPU memory.
- Final submission masks must be upscaled to `2048 x 2048`.

### CELL 3: Preprocessing Engine

Contains:

```python
load_image(path)
remove_limb_darkening(img)
get_magnetic_proxy(img)
```

### CELL 4: PyTorch Dataset

Defines:

```python
class SolarDataset(Dataset)
```

Also defines:

```python
train_transforms
val_transforms
```

### CELL 5: Model Architecture

Defines:

```python
class AttentionGate(nn.Module)
class RSU(nn.Module)
class CognitiveRefiner(nn.Module)
class MagAgentU2Net(nn.Module)
```

### CELL 6: Loss Function

Defines:

```python
class DiceBCELoss(nn.Module)
```

Combines:

```text
Dice Loss + BCEWithLogitsLoss
```

### CELL 7: Training and Validation Loops

Defines:

```python
train_epoch(...)
validate(...)
```

Uses:

```python
autocast()
GradScaler()
gradient clipping
```

### CELL 8: Inference and Submission Generator

Defines:

```python
pycoco_rle_encode(img)
generate_submission(model, test_paths, out_path)
```

Outputs:

```text
/kaggle/working/submission.csv
```

### CELL 9: Main Execution Pipeline

Runs:

1. Load COCO annotations.
2. Filter valid image IDs.
3. Split train / validation.
4. Create datasets and loaders.
5. Initialize model.
6. Train.
7. Save best model.
8. Load best model.
9. Generate submission CSV.

### CELL 10: Optional Visualization

Used to visually inspect:

- Original image.
- Ground-truth mask.
- Predicted mask.

---

## 8. Preprocessing Details

### 8.1 Image Loading

```python
img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
if img is None:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
```

Then normalized:

```python
img = (img - img.min()) / (img.max() - img.min() + 1e-8)
```

### 8.2 Limb-Darkening Removal

The Sun exhibits limb darkening: edges appear darker and can be confused with filaments.

Current implementation:

- Finds image center.
- Computes radial distance.
- Uses a circular mask with radius:

```python
max_r = min(cx, cy) * 0.98
```

- Normalizes only inside the solar disk.

This is a simplified version of the limb-darkening removal idea from the referenced paper.

The paper describes fitting the solar disk edge using a second-order cosine function. The current code uses a simpler radial mask.

Potential future improvement:

- More accurate limb fitting.
- Polynomial or cosine fitting.
- Edge-aware normalization.

### 8.3 Magnetic Proxy Map

The competition does not provide magnetogram data for inference. Therefore, a pseudo-magnetic proxy is computed from the H-alpha image.

Current method:

1. Compute Sobel gradients:

```python
dx = cv2.Sobel(img_uint8, cv2.CV_32F, 1, 0, ksize=3)
dy = cv2.Sobel(img_uint8, cv2.CV_32F, 0, 1, ksize=3)
```

2. Compute structure tensor components:

```python
Jxx = GaussianBlur(dx**2)
Jxy = GaussianBlur(dx * dy)
Jyy = GaussianBlur(dy**2)
```

3. Compute coherence / anisotropy:

```python
coherence = sqrt((Jxx - Jyy)**2 + 4 * Jxy**2)
```

4. Normalize to `[0, 1]`.

Interpretation:

- High coherence indicates directional, elongated structures.
- Filaments tend to be elongated and aligned with magnetic structures.
- Sunspots and noise are often more isotropic.

This magnetic proxy is used as the third input channel.

---

## 9. Input Tensor Format

The model receives a 3-channel input:

```text
Channel 0: limb-corrected H-alpha image
Channel 1: limb-corrected H-alpha image
Channel 2: magnetic proxy / structure tensor coherence map
```

Current code stacks as:

```python
img_3ch = np.stack([img_limb, img_limb, mag_map], axis=-1)
```

So NumPy shape is:

```text
(H, W, 3)
```

After resizing:

```text
(1024, 1024, 3)
```

After Albumentations `ToTensorV2()`:

```text
(3, 1024, 1024)
```

This is the shape expected by the model.

---

## 10. Critical Tensor Shape Rule

The model expects:

```text
images: [B, 3, H, W]
masks:  [B, 1, H, W]
```

For current settings:

```text
images: [B, 3, 1024, 1024]
masks:  [B, 1, 1024, 1024]
```

A previous error occurred:

```text
RuntimeError: Given groups=1, weight of size [64, 3, 1, 1],
expected input[4, 1024, 3, 1024] to have 3 channels, but got 1024 channels instead
```

Cause:

- Albumentations `ToTensorV2()` already converts `(H, W, C)` to `(C, H, W)`.
- The code then incorrectly applied `.permute(2, 0, 1)` again.
- This changed shape from `(3, 1024, 1024)` to `(1024, 3, 1024)`.

Correct rule:

- If image is NumPy `(H, W, C)`, convert to tensor and permute to `(C, H, W)`.
- If image is already a PyTorch tensor from `ToTensorV2()`, do **not** permute again.

Recommended robust return logic in `SolarDataset.__getitem__`:

```python
if self.transforms:
    aug = self.transforms(image=img_3ch, mask=mask)
    img_3ch = aug['image']
    mask = aug['mask']

if isinstance(img_3ch, np.ndarray):
    img_3ch = torch.from_numpy(img_3ch).permute(2, 0, 1).float()
else:
    img_3ch = img_3ch.float()

if isinstance(mask, np.ndarray):
    mask = torch.from_numpy(mask).unsqueeze(0).float()
else:
    mask = mask.unsqueeze(0).float()

return img_3ch, mask
```

---

## 11. Dataset Class Behavior

`SolarDataset` does the following for each item:

1. Gets COCO image id.
2. Gets image info directly from:

```python
self.coco.imgs[img_id]
```

3. Loads `.jpeg` image from `train_images/`.
4. Normalizes image.
5. Removes limb darkening.
6. Computes magnetic proxy map.
7. Stacks 3 channels.
8. Resizes image to `Config.IMG_SIZE`.
9. Gets annotations directly from:

```python
self.coco.imgToAnns.get(img_id, [])
```

10. Converts each annotation polygon to mask using:

```python
self.coco.annToMask(ann)
```

11. Combines all annotations into one binary mask.
12. Resizes mask to `Config.IMG_SIZE`.
13. Binarizes mask:

```python
mask = (mask > 0.5).astype(np.float32)
```

14. Applies augmentations.
15. Returns image tensor and mask tensor.

---

## 12. Augmentations

Current training augmentations:

```python
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    ToTensorV2()
])
```

Validation augmentations:

```python
val_transforms = A.Compose([
    ToTensorV2()
])
```

Important:

- `ToTensorV2()` is always present in both train and validation transforms.
- Therefore, after augmentation, the image should already be a PyTorch tensor of shape `(3, H, W)`.

---

## 13. Model Architecture Details

The model is:

```python
class MagAgentU2Net(nn.Module)
```

It is inspired by Attention U²-Net from the referenced paper.

### 13.1 Attention Gate

```python
class AttentionGate(nn.Module)
```

Based on Attention U-Net.

Inputs:

- `g`: gating signal from decoder / deeper layer.
- `x`: skip connection from encoder.

Operation:

```text
1x1 conv on g
1x1 conv on x
add
ReLU
1x1 conv
Sigmoid
multiply with x
```

Purpose:

- Suppress irrelevant background features.
- Enhance filament-like features.
- Improve attention on small or thin structures.

### 13.2 RSU Block

```python
class RSU(nn.Module)
```

RSU means Residual U-block.

Based on U²-Net.

Current implementation is lightweight:

- Input convolution.
- Mini encoder path.
- Pooling.
- Mini decoder path.
- Upsampling.
- Concatenation.
- Residual addition.

Purpose:

- Capture multi-scale context.
- Handle both large filament bodies and thin barbs.
- Improve feature richness compared to plain convolution blocks.

The original paper uses deeper RSU blocks with heights ranging from 7 to 4. Current implementation is simplified to reduce GPU memory usage on Kaggle.

Potential future improvement:

- Implement full-height RSU blocks.
- Use dilation inside RSU blocks.
- Add deep supervision.

### 13.3 Cognitive Refiner

```python
class CognitiveRefiner(nn.Module)
```

This is the current implementation of the Agentic Cognitive Layer.

It receives:

```python
x: decoder feature map
mag_map: magnetic proxy map
```

Operations:

1. Dilated depthwise convolution with dilation 2.
2. Dilated depthwise convolution with dilation 4.
3. Residual addition.
4. Concatenate magnetic proxy map.
5. Learnable gating with sigmoid.
6. Final 1x1 convolution to produce one output channel.

Purpose:

- Bridge small gaps in filament structures.
- Reconnect fragmented pieces.
- Use magnetic proxy as guidance.
- Improve structural continuity.
- Improve Panoptic Quality.

This is currently a lightweight CNN module, not a full graph neural network or transformer agent.

Potential future improvements:

- Graph-based component merging.
- CRF post-processing.
- Deformable attention.
- Transformer decoder.
- Topology-aware loss.
- Use annotation `spine` information for supervision.

### 13.4 Full Network Flow

```text
Input: [B, 3, H, W]
    |
    |--- extract magnetic map: x[:, 2:3, :, :]
    |
Encoder:
    enc1 -> pool1 -> enc2 -> pool2 -> enc3 -> pool3
    |
Bottleneck:
    bottleneck
    |
Decoder:
    up3 + attention gate + dec3
    up2 + attention gate + dec2
    up1 + attention gate + dec1
    |
CognitiveRefiner:
    uses decoder features + magnetic map
    |
Output:
    logits [B, 1, H, W]
```

Current parameter count:

```text
10,762,602 total parameters
10,762,602 trainable parameters
```

---

## 14. Loss Function

Current loss:

```python
class DiceBCELoss(nn.Module)
```

Formula used:

```text
loss = DiceLoss + BCEWithLogitsLoss
```

Reason:

- BCE provides stable pixel-wise training.
- Dice helps with class imbalance.
- Filaments occupy a small fraction of the full solar disk.

The referenced paper used BCELoss. The current implementation uses Dice + BCE because the competition dataset has severe foreground/background imbalance and thin structures.

Potential future improvements:

- Focal loss.
- Boundary loss.
- clDice or topology-preserving loss.
- Panoptic-style matching loss.
- Instance-aware loss.
- Barb-specific loss.

---

## 15. Training Details

Current training configuration:

```python
BATCH_SIZE = 4
EPOCHS = 25
LR = 2e-4
WEIGHT_DECAY = 1e-4
OPTIMIZER = AdamW
DEVICE = cuda if available
```

Current split:

```text
Training images: 923
Validation images: 231
```

Training uses:

```python
torch.cuda.amp.autocast()
GradScaler()
gradient clipping
```

Gradient clipping:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Best model is saved based on validation loss:

```python
/kaggle/working/best_model.pth
```

---

## 16. Inference and Submission Pipeline

For each test image:

1. Load `.jpeg` image.
2. Normalize.
3. Remove limb darkening.
4. Compute magnetic proxy.
5. Stack into 3 channels.
6. Resize to `1024 x 1024`.
7. Run model.
8. Apply sigmoid.
9. Threshold at:

```python
Config.THRESHOLD = 0.4
```

10. Resize prediction back to:

```text
2048 x 2048
```

using nearest-neighbor interpolation.

11. Run connected component analysis:

```python
cv2.connectedComponents(pred_mask_2048)
```

12. Discard components with area smaller than:

```python
Config.MIN_AREA = 200
```

13. Encode each remaining component using pycocotools RLE.

14. Create row:

```text
filament_id = f"{image_file_stem}_{component_label}"
segmentation_rle = pycoco_rle_encode(component_mask)
```

15. Save:

```text
/kaggle/working/submission.csv
```

---

## 17. RLE Encoding Rules

The competition expects RLE counts generated by `pycocotools`.

Correct encoding:

```python
def pycoco_rle_encode(img):
    rle = maskUtils.encode(np.asfortranarray(img.astype(np.uint8)))
    return rle['counts'].decode('utf-8')
```

Important:

- Mask must be binary `uint8`.
- Mask must be Fortran-contiguous.
- Mask must be `2048 x 2048`.
- Do not store RLE size.
- Do not include quotes in CSV if possible.
- The size is fixed:

```text
2048 x 2048
```

---

## 18. Submission File Format

Expected CSV columns:

```text
filament_id
segmentation_rle
```

Example:

```text
filament_id,segmentation_rle
20150125172714Mh_1,<RLE counts>
20150125172714Mh_2,<RLE counts>
```

Important:

- `filament_id` is constructed from the test image file name stem plus a unique suffix.
- Example test image:

```text
20220621185332Bh.jpeg
```

- Base id:

```text
20220621185332Bh
```

- If three components are predicted:

```text
20220621185332Bh_1
20220621185332Bh_2
20220621185332Bh_3
```

Do not use the COCO training image id such as:

```text
040301-20140609195854Bh
```

for test submission ids.

---

## 19. Research Basis and Mapping to Code

### 19.1 Main Paper

```text
Automated High-Precision Recognition of Solar Filaments Based on an Improved U2-Net
Wendong Jiang and Zhengyang Li
Universe 2024, 10, 381
DOI: 10.3390/universe10100381
```

Key ideas used:

- U-Net encoder-decoder segmentation.
- Attention U-Net attention gates.
- U²-Net residual U-blocks.
- Attention U²-Net combination.
- Limb-darkening removal.
- Multi-scale feature extraction.
- Importance of preserving fine filament detail.

Paper results for Attention U²-Net:

```text
Accuracy: 0.9987
Precision: 0.8221
Recall: 0.8469
IoU: 0.7139
F1-score: 0.8323
```

These metrics were reported on the paper’s own CHASE-based dataset, not the Kaggle MAGFiLO test set.

### 19.2 Paper Preprocessing

The paper used:

1. Limb-darkening removal.
2. Grayscale transformation.
3. K-means clustering.
4. Particle erosion.
5. Multiple closing operations.
6. Hole filling.

Our code:

- Uses limb-darkening removal.
- Does not use K-means for label creation because MAGFiLO already provides expert annotations.
- Uses structure tensor magnetic proxy instead of K-means-based label generation.
- Uses connected components and area filtering during inference.
- May benefit from morphological closing in future versions.

### 19.3 Paper Architecture

The paper uses:

```text
Attention U²-Net = U²-Net + Attention Gates
```

Our implementation:

```text
Mag-Agent U²-Net = simplified Attention U²-Net + magnetic proxy input + cognitive refiner
```

Differences:

- Our RSU blocks are lightweight.
- We add a magnetic proxy channel.
- We add CognitiveRefiner.
- We use Dice + BCE loss instead of pure BCE.
- We train at 1024 x 1024 instead of 512 x 512.
- We generate instance-like submission masks using connected components.

---

## 20. Known Bugs That Were Fixed

### 20.1 Dummy File Bug

Earlier, dummy paths caused:

```text
AttributeError: 'NoneType' object has no attribute 'min'
```

Cause:

- `cv2.imread()` returned `None` because dummy files did not exist.

Fix:

- Use real dataset paths.
- Optionally fallback to random array for local code validation.

### 20.2 COCO String ID Bug

Error:

```text
KeyError: '0'
```

Cause:

- COCO image IDs are strings.
- `coco.loadImgs(img_id)` with a string can be interpreted as iterable characters.

Fix:

```python
img_info = coco.imgs[img_id]
anns = coco.imgToAnns.get(img_id, [])
```

### 20.3 Channel Permutation Bug

Error:

```text
RuntimeError: Given groups=1, weight of size [64, 3, 1, 1],
expected input[4, 1024, 3, 1024] to have 3 channels, but got 1024 channels instead
```

Cause:

- `ToTensorV2()` already converts `(H, W, C)` to `(C, H, W)`.
- Additional `.permute(2, 0, 1)` scrambled dimensions.

Fix:

- Do not permute if the image is already a tensor.
- Use robust return logic shown in Section 10.

---

## 21. Coding Rules for Future Agents

When modifying this notebook, follow these rules:

### 21.1 Preserve Kaggle Paths

Always use:

```python
Config.BASE_DIR
Config.TRAIN_DIR
Config.TEST_DIR
Config.ANNOTATIONS_FILE
```

Do not hardcode paths outside the config unless debugging.

### 21.2 Respect COCO String IDs

Use:

```python
coco.imgs[img_id]
coco.imgToAnns.get(img_id, [])
```

Avoid:

```python
coco.loadImgs(img_id)
coco.getAnnIds(imgIds=img_id)
```

unless wrapping string IDs carefully in lists and testing.

### 21.3 Preserve Tensor Shape Contract

Model expects:

```text
image batch: [B, 3, H, W]
mask batch:  [B, 1, H, W]
```

Dataset must return:

```text
image: [3, H, W]
mask:  [1, H, W]
```

### 21.4 Do Not Change Submission Size

Submission masks must be:

```text
2048 x 2048
```

Even if training happens at:

```text
1024 x 1024
```

or smaller.

### 21.5 Use pycocotools for RLE

Do not use custom integer-run RLE unless competition explicitly allows it.

Current required method:

```python
maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
```

### 21.6 Keep Inference Independent of Ground Truth

Test-time code must not read ground-truth annotations.

It may only use:

```text
test_images/*.jpeg
trained model weights
```

### 21.7 Maintain Reproducibility

Any major change should update:

- Config.
- Dataset.
- Model.
- Loss.
- Training loop.
- Inference.
- Submission generator.

Do not silently change thresholds or dimensions without noting them.

---

## 22. Current Hyperparameters

```python
IMG_SIZE = 1024
ORIGINAL_SIZE = 2048
BATCH_SIZE = 4
EPOCHS = 25
LR = 2e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2
THRESHOLD = 0.4
MIN_AREA = 200
IN_CHANNELS = 3
OUT_CHANNELS = 1
```

Potential tuning targets:

```text
LR
BATCH_SIZE
THRESHOLD
MIN_AREA
IMG_SIZE
Loss weights
Dilation sizes in CognitiveRefiner
Augmentation strength
```

---

## 23. Expected Output Shapes

### Dataset output

```text
image: torch.Tensor, shape [3, 1024, 1024]
mask:  torch.Tensor, shape [1, 1024, 1024]
```

### Batch

```text
images: [B, 3, 1024, 1024]
masks:  [B, 1, 1024, 1024]
```

### Model output

```text
logits: [B, 1, 1024, 1024]
```

### Inference prediction before submission

```text
pred_mask_1024: [1024, 1024]
pred_mask_2048: [2048, 2048]
```

### Submission component mask

```text
component_mask: uint8, shape [2048, 2048], values {0, 1}
```

---

## 24. Current Limitations

### 24.1 Semantic Training, Instance Submission

The model is currently trained as a binary semantic segmentation model.

It predicts:

```text
filament vs non-filament
```

But the competition evaluates individual filament instances via PQ.

Instances are currently created only during inference using:

```python
cv2.connectedComponents
```

This is a simple approach and may fail if:

- Two filaments touch.
- One filament is fragmented.
- Thin barbs disconnect from the main body.

Future improvement:

- Instance-aware training.
- Connected-component-aware loss.
- Graph merging.
- Panoptic segmentation head.

### 24.2 Magnetic Proxy Is Approximate

The magnetic proxy is not a real magnetogram.

It is only a directional coherence feature derived from H-alpha.

Future improvement:

- Gabor filters.
- Steerable filters.
- Learned orientation branch.
- Use annotation spines to supervise orientation.

### 24.3 Cognitive Layer Is Lightweight

Current CognitiveRefiner is a small CNN module.

It does not yet perform true graph reasoning.

Future improvement:

- Component graph construction.
- Edge prediction between fragments.
- Spatial attention.
- Transformer decoder.
- Morphological closing guided by orientation.

### 24.4 No Local PQ Evaluation Yet

The notebook currently trains using Dice + BCE loss.

It does not yet compute full Panoptic Quality locally.

Future improvement:

- Add local evaluator.
- Use competition self-evaluation notebook as reference.
- Track Dice, IoU, fragmentation, one-to-many, many-to-one relations.

---

## 25. High-Value Future Improvements

Ranked by likely impact:

1. **Local PQ/Dice evaluation**
   - Needed to tune threshold and post-processing.

2. **Threshold tuning**
   - Try `0.30` to `0.50`.
   - Optimize for PQ, not only Dice.

3. **Minimum area tuning**
   - Try `100`, `200`, `300`, `500`.

4. **Morphological closing**
   - Connect small gaps before connected components.
   - Use orientation-aware kernels if possible.

5. **Use spine annotations**
   - Train auxiliary spine prediction.
   - Use spine continuity as loss.
   - Use spine for fragment merging.

6. **Improve RSU blocks**
   - Closer to original U²-Net.
   - Add dilations.
   - Preserve multi-scale context.

7. **Test-time augmentation**
   - Horizontal flip.
   - Vertical flip.
   - 90-degree rotations.
   - Average logits.

8. **Instance-aware head**
   - Predict filament centers.
   - Predict instance boundaries.
   - Use panoptic-style segmentation.

9. **Optuna hyperparameter search**
   - Learning rate.
   - Threshold.
   - Min area.
   - Batch size.
   - Loss weights.

10. **Better limb-darkening correction**
   - Fit solar disk more accurately.
   - Avoid suppressing real limb filaments.

---

## 26. Important Scientific Context

Solar filaments are dense, relatively cool plasma structures suspended above the solar photosphere by magnetic field lines.

They are important because they are associated with:

- Solar flares.
- Coronal mass ejections.
- Solar energetic particle events.
- Space weather hazards.

Filaments often form along polarity inversion lines in the photospheric magnetic field.

In H-alpha observations, filaments appear as dark elongated structures. Their fine structures include:

- Main spine.
- Barbs.
- Threads.
- Fragmented dark plasma lanes.

The main computer-vision challenges are:

- Thin structures.
- Low contrast.
- Background noise.
- Limb darkening.
- Fragmentation.
- Similar-looking dark features such as sunspots.
- Maintaining physical continuity.

---

## 27. Current Definition of Success

A successful implementation should:

1. Train without tensor shape errors.
2. Produce a valid `submission.csv`.
3. Use `pycocotools` RLE encoding.
4. Predict masks at `2048 x 2048`.
5. Preserve filament continuity.
6. Minimize fragmented predictions.
7. Achieve competitive Dice and PQ scores on Kaggle.
8. Be reproducible from the notebook and source code.

According to competition announcements, a PQ score greater than:

```text
0.35
```

is considered valuable by the organizers.

---

## 28. Final Deliverables

For competition submission, the following are needed:

1. Kaggle submission CSV:

```text
submission.csv
```

2. Technical report:

```text
4-page PDF
```

3. Public Git repository containing:

```text
requirements.txt
Jupyter notebook illustrating full pipeline
source code
documentation
```

4. Reproducible prediction pipeline.

---

## 29. References

### Primary architecture paper

```text
Jiang, W.; Li, Z.
Automated High-Precision Recognition of Solar Filaments Based on an Improved U2-Net.
Universe 2024, 10, 381.
DOI: 10.3390/universe10100381
```

### Dataset / ground truth

```text
MAGFiLO: Manually Annotated GONG Filaments from H-Alpha Observations
DOI: 10.1038/s41597-024-03876-y
```

### Panoptic Quality / Panoptic Segmentation

```text
DOI: 10.1109/CVPR.2019.00963
```

### U-Net

```text
Ronneberger, O.; Fischer, P.; Brox, T.
U-Net: Convolutional Networks for Biomedical Image Segmentation.
MICCAI 2015.
```

### Attention U-Net

```text
Oktay, O.; Schlemper, J.; Folgoc, L.L.; et al.
Attention U-Net: Learning Where to Look for the Pancreas.
MIDL 2018.
```

### U²-Net

```text
Qin, X.; Zhang, Z.; Huang, C.; et al.
U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection.
Pattern Recognition 2020.
```

### torchmetrics Dice

```text
torchmetrics.segmentation.DiceScore
```

---

## 30. Short Summary for Coding Agent

You are working on a Kaggle solar filament segmentation notebook.

The model is a custom PyTorch network called **Mag-Agent U²-Net**, inspired by **Attention U²-Net**.

The input is a 3-channel tensor:

```text
[H-alpha limb-corrected, H-alpha limb-corrected, magnetic proxy map]
```

The dataset loads images from `.jpeg` files and masks from a COCO JSON file.

COCO IDs are strings. Use direct dictionary access:

```python
coco.imgs[img_id]
coco.imgToAnns.get(img_id, [])
```

Do not call `coco.loadImgs(img_id)` directly with a string.

Dataset must return:

```text
image: [3, 1024, 1024]
mask:  [1, 1024, 1024]
```

Model output is:

```text
[B, 1, 1024, 1024]
```

Final submission masks must be resized to:

```text
2048 x 2048
```

and encoded using:

```python
pycocotools.mask.encode(np.asfortranarray(mask.astype(np.uint8)))
```

The final CSV must contain:

```text
filament_id
segmentation_rle
```

The main research goal is not only pixel accuracy but also structural continuity, because the competition uses Panoptic Quality and penalizes fragmentation and over-merging.
```