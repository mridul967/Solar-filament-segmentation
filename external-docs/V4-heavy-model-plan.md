# V4 Heavy Model — Audited Implementation Plan

## 1. Decision and hypothesis

V4 is an **instance-aware U2-style segmentation model** built from
`heavy-models/V4-heavy-model.py`, the untouched original heavy model.

The single hypothesis is:

> Training foreground, boundary, and annotated-spine targets per annotation
> pass—and selecting the decoder by held-out instance PQ—will reduce the
> merge/split errors that a semantic union mask plus connected components cannot
> learn to avoid.

This is deliberately different from V3. V3 changed preprocessing,
regularization, and a semantic topology loss while retaining the same union
target and connected-component instance reconstruction.

## 2. Audit evidence that defines V4

| Finding | Evidence | V4 consequence |
|---|---:|---|
| Physical observations repeat as annotation records | 707 JPEGs; 1,154 COCO image records; 296 JPEGs have 2–3 records | Split and sample by `file_name`, never raw COCO record. |
| Annotators substantially disagree | Median per-image union IoU: 0.472; range 0.000–0.804 | Train one annotation pass at a time; never fuse annotators into a synthetic target. |
| Spines are valid supervision | 8,199/8,199 coordinate-valid; median spine-in-mask coverage 1.0 | Add a spine head; mask out spine loss for the 8 examples below 0.90 containment. |
| Downsampling creates no instance collisions | Zero records gained a new overlap at 1024² | Do not make tile sampling the primary V4 variable. |
| Downsampling causes limited fragmentation | 42/1,154 records gained components at 1024² | Keep 1024²; revisit tiles only if V4 diagnostics show fragmentation dominates. |
| Train/test image quality is matched | Median Laplacian variance: train 0.00819, test 0.00827 | Do not add image-quality filtering or domain adaptation. |
| Radial correction can amplify limb/ring artefacts | Visual audit | Keep original preprocessing, with safe clipping only. |

The earlier raw area-ratio observation is not a resolution-loss signal: halving
both dimensions correctly reduces area to one quarter. Nearest-neighbour
resizing preserves median normalized area of 1.00; linear interpolation instead
erodes masks by median 5.8%.

## 3. Scope

### Included

1. Group-safe train/development/audit split.
2. Annotation-pass-aware data loader.
3. Foreground, boundary, and spine targets.
4. Shared U2 decoder with three small prediction heads.
5. Multi-task loss and masked invalid-spine loss.
6. Organizer-compatible local PQ evaluator and diagnostics.
7. Connected-components baseline and conservative seeded-watershed comparator.
8. Cached development outputs and a bounded decoder grid search.
9. Checkpointing, reproducibility, visual diagnostics, and RLE validation.

### Deferred

* Native-resolution crop/tile training and tile blending.
* Pretrained backbones, DETR, Mask2Former, or a second detector.
* Per-pixel instance embeddings and clustering.
* clDice, Skeleton Recall Loss, or another new topology loss.
* Histogram matching or fitted radial limb correction.
* Cross-validation ensembles and test-time augmentation.

Each deferred item tests a separate hypothesis. Reconsider it only from V4's
held-out PQ error breakdown.

## 4. Configuration

Use explicit V4 settings:

```python
IMG_SIZE = 1024
ORIGINAL_SIZE = 2048
SEED = 42
TRAIN_FRACTION, DEV_FRACTION, AUDIT_FRACTION = 0.80, 0.10, 0.10
BATCH_SIZE, ACCUM_STEPS = 2, 2
EPOCHS, LR, WEIGHT_DECAY = 50, 2e-4, 1e-4
LR_PATIENCE, EARLY_STOPPING_WARMUP, EARLY_STOPPING_PATIENCE = 4, 12, 10
FOREGROUND_THRESHOLD_CANDIDATES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
MIN_AREA_CANDIDATES = (50, 100, 200, 300, 500)
BOUNDARY_WEIGHT, SPINE_WEIGHT = 0.20, 0.20
SPINE_VALIDITY_FLOOR = 0.90
```

All artifacts receive a V4-specific name:

```text
v4_group_split.json
best-v4-instance-aware.pth
v4-history.csv
v4-dev-logits.npz
v4-decoder-settings.json
v4-audit-metrics.json
submission-v4-instance-aware.csv
```

## 5. Inputs and transforms

### 5.1 Shared preprocessing

Create `prepare_input(image)` and call it from training, validation,
visualization, and test inference. It must:

1. read grayscale data;
2. min/max normalize safely;
3. apply the current disk masking/limb-normalization method;
4. clip to `[0, 1]` before `uint8` conversion;
5. zero pixels outside the disk; and
6. produce the original three channels: corrected H-alpha, duplicated corrected
   H-alpha, and structure-tensor proxy.

Do not import V3's local-contrast channel: V3 did not establish value, and V4
must keep its causal change focused on instance supervision.

### 5.2 Spatial transforms

Keep only horizontal flip, vertical flip, and 90-degree rotation. Apply the
same sampled transform to input and all categorical targets. Use nearest
neighbour for masks, IDs, boundaries, and spines; use bilinear only for image
channels.

## 6. Grouping and persistent split

Build `file_name -> [COCO image record / annotation pass]`. Assign each
physical filename once, using a seeded shuffled 80/10/10 split, persist the
split JSON, and assert all three filename sets are disjoint.

The V4 dataset's unit is a physical image. For each training sample:

1. select the physical filename;
2. randomly select one COCO image record for that filename;
3. rasterize only that record's annotations; and
4. return the image and its selected target set.

Predict every development/audit JPEG once, then evaluate it separately against
every available annotation pass. Log mean and median pass-level results, and
use organizer-compatible aggregation for checkpoint selection once parity is
verified.

## 7. Targets

Build at 2048² before resizing:

| Target | Construction | Loss mask |
|---|---|---|
| `foreground` | Union of individual masks in one annotation pass | Solar disk |
| `instance_id` | Integer map; mark ambiguous overlaps as ignore | Diagnostics only |
| `boundary` | Inner contour of each mask, gently dilated after resize | Ignore overlaps/outside disk |
| `spine` | Rasterized COCO polyline, one pixel wide before resize | Per-instance mask if containment < 0.90 |

Render six fixed examples showing input, foreground, boundary, spine, and ID
maps before full training. Verify alignment after augmentation.

## 8. Architecture

Retain the original encoder, RSU blocks, attention gates, and channel widths.
Refactor only the final refinement output:

```text
decoder features -> CognitiveRefinerFeatures (64 channels)
                         ├─ foreground head: 1×1 conv -> one logit
                         ├─ boundary head:   1×1 conv -> one logit
                         └─ spine head:      1×1 conv -> one logit
```

The refiner retains its depthwise dilated convolutions and proxy gate, but
returns refined features. Each head is one `nn.Conv2d(64, 1, 1)`. The forward
method returns:

```python
{"foreground": fg_logits, "boundary": boundary_logits, "spine": spine_logits}
```

Retain notebook-compatible `DataParallel`; save/load the unwrapped state dict.

## 9. Objective

Compute all reductions in float32 under AMP:

```text
foreground_loss = BCEWithLogits + Dice
boundary_loss   = masked BCEWithLogits + Dice
spine_loss      = masked BCEWithLogits + Dice

total_loss = foreground_loss + 0.20 * boundary_loss + 0.20 * spine_loss
```

Use bounded positive weights calculated on training groups, not unbounded
inverse-frequency weights. Warm auxiliary weights linearly from zero over the
first three epochs. Log every term separately.

Do not add clDice: V3 did not establish its value, and audited spines offer
direct, cheaper continuity supervision.

## 10. Training and selection

1. Smoke-test one batch: all outputs/losses/gradients finite, optimizer updates.
2. Overfit two physical images until all enabled losses fall materially.
3. Train with AdamW, AMP, gradient clipping after unscaling, and accumulation.
4. After each epoch, predict development images, decode the baseline, and log
   PQ/SQ/RQ/TP/FP/FN/splits/merges.
5. Select best checkpoint by development PQ, not total loss or foreground Dice.
6. Reduce LR after four stagnant PQ epochs; after epoch 12, early-stop after
   ten stagnant checks.
7. Persist model, optimizer, scaler, scheduler, configuration, split, history,
   and decoder settings for resumption.

## 11. Evaluator and instance decoder

### 11.1 Evaluator parity

Reproduce the organizer evaluator before the long run. Assert expected results
for: exact match, empty masks, FP, FN, one split, one merge, and overlapping
predictions if the organizer permits them. Record PQ, SQ, RQ, TP, FP, FN,
split count, and merge count.

### 11.2 Cached maps

For the selected checkpoint, cache foreground, boundary, and spine probability
maps at 2048² per development JPEG. Reuse them for all decoder trials without
retraining.

### 11.3 Decoder candidates

**A. Connected components**

Threshold foreground, filter small native-resolution components, and export
each remaining component.

**B. Conservative spine-seeded watershed**

Threshold foreground; suppress high-boundary regions; use sufficiently long
spine components as markers; run watershed inside foreground; retain an unsplit
component when markers are unreliable; and never apply distance-peak seeding.

If watershed increases development splits or reduces PQ, select connected
components.

### 11.4 Calibration

Search only foreground threshold, minimum native-pixel area, decoder choice,
and—with watershed—boundary/spine thresholds `(0.35, 0.50, 0.65)`. Freeze the
development winner, then evaluate it exactly once on the audit group.

## 12. Test submission

1. Load best unwrapped checkpoint plus frozen decoder settings.
2. Preprocess test image, resize channels to 1024², infer three logits, and
   upsample logits (never thresholded masks) to 2048².
3. Decode once at 2048².
4. Give each instance a deterministic ID and Fortran-order pycocotools RLE.
5. Write `filament_id` and `segmentation_rle`, including headers if empty.
6. Decode sample RLE strings and assert exact pixel round-trip equality.

## 13. Verification checklist

- [ ] Split has no filename overlap.
- [ ] One augmented batch has aligned targets.
- [ ] AMP losses and gradients are finite.
- [ ] Two-image overfit works.
- [ ] Checkpoint reload has identical logits.
- [ ] Evaluator passes synthetic assertions.
- [ ] Cached and on-the-fly decoder results agree.
- [ ] RLE round-trip is exact.
- [ ] Audit is evaluated once after decoder settings freeze.

## 14. Success and V5 decision

V4 succeeds only if its development PQ beats original/V3 and that improvement
persists on the frozen audit group. Always report SQ/RQ and split/merge counts.

| Dominant V4 error | V5 experiment |
|---|---|
| Fragmentation / FN | Native-resolution crop fine-tuning and overlapping tiles |
| Merges | Embedding head or boundary-decoder refinement |
| Poor masks, normal recognition | Preprocessing ablation |
| High variation across annotation passes | Robust/consensus objective |

## 15. Research basis

* Ahmadzadeh et al. (2024), MAGFiLO, *Scientific Data*,
  doi:10.1038/s41597-024-03876-y.
* Jiang & Li (2024), Attention U2-Net, *Universe*,
  doi:10.3390/universe10100381.
* Shit et al. (2021), clDice, CVPR,
  doi:10.1109/CVPR46437.2021.01629.
* Castagnino et al. (2026), GONG framework, *Acta Astronautica*.
