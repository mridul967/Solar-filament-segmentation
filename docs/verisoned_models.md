The next version should become an **instance-aware, topology-aware model evaluated directly with local PQ**. Improving the current binary Dice model alone will probably plateau because the leaderboard penalizes splitting and merging at the instance level.

The paper’s reported `0.9987` accuracy is not comparable to your `0.14` PQ. It used a different CHASE-based dataset, evaluated semantic metrics, and background dominates full-disk images. Its more meaningful reported numbers were F1 `0.8323` and IoU `0.7139`; it did not report instance PQ. [Jiang & Li, 2024](https://www.mdpi.com/2218-1997/10/10/381)

## Proposed plan

### 1. Implement the exact local competition metric first

Preserve every COCO annotation as an individual ground-truth mask and calculate:

- PQ
- Segmentation Quality, `SQ`
- Recognition Quality, `RQ`
- True positives, false positives, false negatives
- Splits: one ground-truth filament matched by several predictions
- Merges: one prediction overlapping several ground-truth filaments

PQ decomposes into segmentation quality and recognition quality. That tells us whether `0.14` comes mainly from poor masks, missing instances, fragmentation, or merging. [Original Panoptic Segmentation paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf)

This metric becomes:

- Checkpoint-selection metric
- Early-stopping metric
- Optuna objective
- Post-processing tuning objective

Foreground Dice remains diagnostic only.

### 2. Repair the validation design

The current split takes the first 80% and last 20% of the JSON. That could produce chronological or observatory distribution shift.

Change to:

- Seeded randomized split
- Stratification by number of filament instances and foreground area
- Fixed split saved to disk
- Eventually, three-fold cross-validation for reliable comparisons

Every experiment must use the same split. Otherwise small PQ changes are not trustworthy.

### 3. Restore useful spatial resolution

The 512 model used only `1.96 GiB`, so it was reduced too aggressively.

Test in this order:

1. `768×768`, batch 2
2. `1024×1024`, batch 1
3. `1024×1024`, batch 1 with gradient accumulation if needed

Keep the 2.72M-parameter model initially. Resolution is likely more valuable than width because thin structures vanish when downsampled. Increase channels only after measuring the resolution ablation.

### 4. Stop collapsing annotations into one training mask

The loader currently unions all filament polygons. That destroys the exact information PQ needs.

For each image, construct:

- Semantic foreground mask
- Integer instance-ID map
- Per-instance boundary map
- Per-instance spine or skeleton map
- Optional instance embedding targets

This is the pivotal dataset change.

### 5. Add lightweight instance-aware output heads

Keep the shared lightweight U²-Net decoder and add three small heads:

1. **Foreground head**  
   Predicts filament versus background.

2. **Boundary head**  
   Predicts pixels separating nearby filament instances. Boundary supervision directly targets over-merging.

3. **Spine head**  
   Predicts filament centerlines using the dataset’s existing `spine` annotations. These can seed instance extraction and encourage continuity.

A fourth **instance embedding head** is the strongest follow-up. It maps pixels from the same filament close together and different filaments apart. The discriminative embedding approach was designed specifically to turn semantic backbones into instance segmenters through clustering. [Discriminative instance loss paper](https://arxiv.org/abs/1708.02551)

I prefer boundary + spine + embeddings over center-offset regression for this dataset. Center-offset methods such as Panoptic-DeepLab are proven and lightweight, but a long curved filament’s centroid may lie far away or even outside its body. [Panoptic-DeepLab](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Panoptic-DeepLab_A_Simple_Strong_and_Fast_Baseline_for_Bottom-Up_Panoptic_CVPR_2020_paper.pdf)

### 6. Replace the loss with a topology-aware multi-task objective

Proposed initial loss:

```text
L =
    1.0 × BCE
  + 1.0 × Dice
  + 0.3 × soft-clDice
  + 0.5 × boundary loss
  + 0.5 × spine loss
  + 0.5 × instance embedding loss
```

Reasoning:

- BCE stabilizes pixel classification.
- Dice handles foreground imbalance.
- **soft-clDice** rewards connected centerlines and was developed for tubular structures such as vessels, roads, and neurons. [clDice paper](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html)
- Boundary loss emphasizes contours in highly imbalanced segmentation. [Boundary Loss paper](https://arxiv.org/abs/1812.07032)
- Spine loss uses supervision already present in MAGFiLO.
- Embedding loss separates touching instances.

Do not add every loss blindly. Ablate them in this order:

1. BCE + Dice baseline
2. Add clDice
3. Add boundary head/loss
4. Add spine head/loss
5. Add embedding head/loss

Keep a component only if validation PQ improves.

### 7. Replace plain connected components

The new post-processing pipeline should be:

```text
foreground probability
        +
spine seeds / embedding clusters
        +
boundary barriers
        ↓
marker-controlled watershed or seeded region growing
        ↓
individual filament instances
```

This directly addresses both failure modes:

- Broken semantic regions can reconnect through predicted spines.
- Touching filaments can separate through boundary and embedding predictions.

HoVer-Net demonstrates how auxiliary spatial predictions can separate touching instances and materially improve PQ, although its centroid-distance formulation is designed around compact nuclei. [HoVer-Net paper](https://arxiv.org/abs/1812.06499)

### 8. Tune post-processing directly for PQ

Cache validation probability maps once. Then optimize without retraining:

- Foreground threshold
- Boundary threshold
- Spine threshold
- Morphological closing radius
- Minimum area
- Minimum spine length
- Watershed seed distance
- Embedding clustering threshold
- Maximum gap-bridging distance

This search will be much cheaper than neural-network Optuna trials. Optimize local PQ, with constraints on split and merge counts.

### 9. Change the Optuna objective

Neural Optuna trials should optimize:

```text
validation PQ
```

Track `SQ` and `RQ` separately.

Search only a few impactful variables:

- Learning rate
- Loss weights
- Resolution: 768 or 1024
- Foreground/boundary/spine thresholds
- Gap-closing distance

Use pruning based on PQ after a warm-up. Do not use three-epoch foreground Dice to choose the final model.

### 10. Use augmentations that preserve solar morphology

Keep flips and rotations because disk orientation is not fixed. Add conservative:

- Brightness and contrast variation
- Gamma variation
- Mild Gaussian noise
- Mild blur/sharpen
- Small affine scale/translation

Avoid aggressive elastic deformation because it may alter filament topology and invalidate spine geometry.

### 11. Add inference improvements after the base model works

Use:

- Horizontal/vertical flip TTA
- 90° rotation TTA
- Average logits before instance extraction
- Threshold and watershed tuning on validation
- Optional ensemble of two folds

These are later-stage improvements because they multiply inference time but require no new architecture.

### 12. Reproduce useful parts of the solar paper

The paper used:

- Attention U²-Net
- Limb-darkening removal
- K-means segmentation
- Erosion
- Repeated closing
- Hole filling

The expert COCO masks make K-means label creation unnecessary, but its morphological operations can become post-processing candidates. Tune them by PQ; fixed aggressive closing could merge distinct filaments. [Solar-filament paper](https://www.mdpi.com/2218-1997/10/10/381)

## Recommended implementation sequence

| Version | Change | Purpose |
|---|---|---|
| V1 | Exact local PQ + randomized fixed split | Establish trustworthy measurement |
| V2 | 768 resolution + tune current threshold/area/closing | Cheap stronger baseline |
| V3 | Preserve instance maps + boundary and spine heads | Reduce merges and fragmentation |
| V4 | Add clDice and instance embedding loss | Improve continuity and separation |
| V5 | Watershed using spine seeds and boundary barriers | Produce PQ-oriented instances |
| V6 | Optuna on cached post-processing outputs | Tune cheaply for PQ |
| V7 | 1024 resolution, TTA, folds | Final accuracy push |

I would not jump immediately to Mask2Former. It is a strong panoptic architecture—reported at `57.8 PQ` on COCO—but it is substantially more complex, and its paper explicitly notes difficulty with small objects and incomplete use of multiscale features. [Mask2Former paper](https://arxiv.org/abs/2112.01527) The boundary/spine/embedding upgrade reuses your current network, matches the available annotations, and targets the exact errors hurting this competition.
