# Documentation Index

Repository documentation navigation. Paths are relative to this index; line anchors refer to the current source. Hidden folders and `.agents/` are excluded. Context statements describe the supplied project brief and are not independent verification of competition rules or successful execution.

## A. Document Registry Matrix

| Document | Location | Domain / Category | Line Count | Summary |
|---|---|---|---:|---|
| [context.md](../context.md) | `context.md` | Research, Data & Datasets, ML Workflow | 1690 | Competition goals, data contracts, Mag-Agent U²-Net pipeline, known issues, and deliverables. |
| [README.md](../README.md) | `README.md` | Project Overview | 0 | Empty project README. |
| [reductions.md](reductions.md) | `docs/reductions.md` | ML Workflow, Experiment Plan | 26 | Verbatim proposal for cheaper models, short Optuna trials, and staged validation. |
| [INDEX.md](INDEX.md) | `docs/INDEX.md` | Documentation Navigation | 161 | Registry, heading map, and topic lookup for repository Markdown. |

## B. Hierarchical Section & Heading Map

### context.md

| Level | Section / Exact Line | Topics |
|---|---|---|
| Document | [context.md — Solar Filament Segmentation Challenge 2026](../context.md#L4) | Full project brief. |
| Section | [1. Project Purpose](../context.md#L6) | Competition objective: preserve individual filament extent and fine structure while avoiding fragmentation and merging. |
| Section | [2. Competition Metrics](../context.md#L25) | Dice overlap and instance matching through Panoptic Quality (PQ). |
| Subsection | [2.1 Dice Score](../context.md#L29) | Dice Score. |
| Subsection | [2.2 Panoptic Quality](../context.md#L41) | Panoptic Quality. |
| Subsection | [2.3 Important Metric Implication](../context.md#L64) | Important Metric Implication. |
| Section | [3. Core Strategy: Mag-Agent U²-Net](../context.md#L77) | Attention U²-Net, H-alpha structure tensor proxy, and a convolutional topology refiner. |
| Section | [4. Competition Constraints](../context.md#L107) | H-alpha-only inference and required pycocotools submission encoding. |
| Section | [5. Kaggle Dataset Structure](../context.md#L122) | Kaggle paths, JPEG layout, image counts, and native 2048 × 2048 resolution. |
| Section | [6. COCO Annotation Structure](../context.md#L166) | COCO polygons, string image identifiers, and direct dictionary access. |
| Subsection | [6.1 Images](../context.md#L180) | Images. |
| Subsection | [6.2 Annotations](../context.md#L216) | Annotations. |
| Subsection | [6.3 Critical COCO API Bug Avoidance](../context.md#L250) | Critical COCO API Bug Avoidance. |
| Section | [7. Current Notebook Cell Map](../context.md#L279) | Ten notebook cells from installation through training, submission, and visualization. |
| Subsection | [CELL 1: Install Dependencies and Imports](../context.md#L283) | Install Dependencies and Imports. |
| Subsection | [CELL 2: Configuration](../context.md#L315) | Configuration. |
| Subsection | [CELL 3: Preprocessing Engine](../context.md#L349) | Preprocessing Engine. |
| Subsection | [CELL 4: PyTorch Dataset](../context.md#L359) | PyTorch Dataset. |
| Subsection | [CELL 5: Model Architecture](../context.md#L374) | Model Architecture. |
| Subsection | [CELL 6: Loss Function](../context.md#L385) | Loss Function. |
| Subsection | [CELL 7: Training and Validation Loops](../context.md#L399) | Training and Validation Loops. |
| Subsection | [CELL 8: Inference and Submission Generator](../context.md#L416) | Inference and Submission Generator. |
| Subsection | [CELL 9: Main Execution Pipeline](../context.md#L431) | Main Execution Pipeline. |
| Subsection | [CELL 10: Optional Visualization](../context.md#L445) | Optional Visualization. |
| Section | [8. Preprocessing Details](../context.md#L455) | Image loading, circular disk normalization, and structure tensor feature extraction. |
| Subsection | [8.1 Image Loading](../context.md#L457) | Image Loading. |
| Subsection | [8.2 Limb-Darkening Removal](../context.md#L471) | Limb-Darkening Removal. |
| Subsection | [8.3 Magnetic Proxy Map](../context.md#L497) | Magnetic Proxy Map. |
| Section | [9. Input Tensor Format](../context.md#L536) | Three input channels and HWC-to-CHW conversion. |
| Section | [10. Critical Tensor Shape Rule](../context.md#L574) | Dataset and batch shape contracts; avoid permuting ToTensorV2 output twice. |
| Section | [11. Dataset Class Behavior](../context.md#L631) | Image lookup, polygon union, resizing, augmentation, and tensor returns. |
| Section | [12. Augmentations](../context.md#L673) | Flips, right-angle rotations, and ToTensorV2. |
| Section | [13. Model Architecture Details](../context.md#L701) | Attention gates, lightweight residual U-blocks, cognitive refiner, and encoder-decoder flow. |
| Subsection | [13.1 Attention Gate](../context.md#L711) | Attention Gate. |
| Subsection | [13.2 RSU Block](../context.md#L742) | RSU Block. |
| Subsection | [13.3 Cognitive Refiner](../context.md#L776) | Cognitive Refiner. |
| Subsection | [13.4 Full Network Flow](../context.md#L819) | Full Network Flow. |
| Section | [14. Loss Function](../context.md#L853) | Dice loss plus BCEWithLogitsLoss for sparse foreground masks. |
| Section | [15. Training Details](../context.md#L886) | AdamW, split sizes, mixed precision, gradient clipping, and best checkpoint. |
| Section | [16. Inference and Submission Pipeline](../context.md#L928) | Prediction, thresholding, native-size upsampling, connected components, and CSV creation. |
| Section | [17. RLE Encoding Rules](../context.md#L983) | Fortran-contiguous uint8 masks encoded as pycocotools compressed counts. |
| Section | [18. Submission File Format](../context.md#L1010) | filament_id and segmentation_rle columns with filename-based instance identifiers. |
| Section | [19. Research Basis and Mapping to Code](../context.md#L1060) | Research inspiration and differences between the cited paper and this implementation. |
| Subsection | [19.1 Main Paper](../context.md#L1062) | Main Paper. |
| Subsection | [19.2 Paper Preprocessing](../context.md#L1093) | Paper Preprocessing. |
| Subsection | [19.3 Paper Architecture](../context.md#L1112) | Paper Architecture. |
| Section | [20. Known Bugs That Were Fixed](../context.md#L1137) | Historical missing-image, COCO identifier, and channel permutation failures. |
| Subsection | [20.1 Dummy File Bug](../context.md#L1139) | Dummy File Bug. |
| Subsection | [20.2 COCO String ID Bug](../context.md#L1156) | COCO String ID Bug. |
| Subsection | [20.3 Channel Permutation Bug](../context.md#L1176) | Channel Permutation Bug. |
| Section | [21. Coding Rules for Future Agents](../context.md#L1197) | Preserve configuration paths, identifiers, shapes, output size, and inference independence. |
| Subsection | [21.1 Preserve Kaggle Paths](../context.md#L1201) | Preserve Kaggle Paths. |
| Subsection | [21.2 Respect COCO String IDs](../context.md#L1214) | Respect COCO String IDs. |
| Subsection | [21.3 Preserve Tensor Shape Contract](../context.md#L1232) | Preserve Tensor Shape Contract. |
| Subsection | [21.4 Do Not Change Submission Size](../context.md#L1248) | Do Not Change Submission Size. |
| Subsection | [21.5 Use pycocotools for RLE](../context.md#L1264) | Use pycocotools for RLE. |
| Subsection | [21.6 Keep Inference Independent of Ground Truth](../context.md#L1274) | Keep Inference Independent of Ground Truth. |
| Subsection | [21.7 Maintain Reproducibility](../context.md#L1285) | Maintain Reproducibility. |
| Section | [22. Current Hyperparameters](../context.md#L1301) | Current dimensions, optimizer settings, thresholds, and tuning candidates. |
| Section | [23. Expected Output Shapes](../context.md#L1332) | Dataset, batch, logits, inference, and component mask dimensions. |
| Subsection | [Dataset output](../context.md#L1334) | Dataset output. |
| Subsection | [Batch](../context.md#L1341) | Batch. |
| Subsection | [Model output](../context.md#L1348) | Model output. |
| Subsection | [Inference prediction before submission](../context.md#L1354) | Inference prediction before submission. |
| Subsection | [Submission component mask](../context.md#L1361) | Submission component mask. |
| Section | [24. Current Limitations](../context.md#L1369) | Semantic-to-instance limitations, approximate proxy, lightweight refiner, and absent PQ evaluation. |
| Subsection | [24.1 Semantic Training, Instance Submission](../context.md#L1371) | Semantic Training, Instance Submission. |
| Subsection | [24.2 Magnetic Proxy Is Approximate](../context.md#L1402) | Magnetic Proxy Is Approximate. |
| Subsection | [24.3 Cognitive Layer Is Lightweight](../context.md#L1415) | Cognitive Layer Is Lightweight. |
| Subsection | [24.4 No Local PQ Evaluation Yet](../context.md#L1429) | No Local PQ Evaluation Yet. |
| Section | [25. High-Value Future Improvements](../context.md#L1443) | Future evaluation, threshold tuning, morphology, spine supervision, and architecture options. |
| Section | [26. Important Scientific Context](../context.md#L1495) | Solar filament structure, scientific motivation, and recognition challenges. |
| Section | [27. Current Definition of Success](../context.md#L1527) | Successful training, valid submission, continuity, reproducibility, and competition objectives. |
| Section | [28. Final Deliverables](../context.md#L1550) | Submission CSV, four-page report, public repository, and prediction pipeline. |
| Section | [29. References](../context.md#L1579) | Architecture, dataset, panoptic quality, U-Net variants, and metric references. |
| Subsection | [Primary architecture paper](../context.md#L1581) | Primary architecture paper. |
| Subsection | [Dataset / ground truth](../context.md#L1590) | Dataset / ground truth. |
| Subsection | [Panoptic Quality / Panoptic Segmentation](../context.md#L1597) | Panoptic Quality / Panoptic Segmentation. |
| Subsection | [U-Net](../context.md#L1603) | U-Net. |
| Subsection | [Attention U-Net](../context.md#L1611) | Attention U-Net. |
| Subsection | [U²-Net](../context.md#L1619) | U²-Net. |
| Subsection | [torchmetrics Dice](../context.md#L1627) | torchmetrics Dice. |
| Section | [30. Short Summary for Coding Agent](../context.md#L1635) | Condensed implementation constraints and structural segmentation objective. |

### README.md

Empty; no headings.

### docs/reductions.md

No Markdown headings; navigate by topic:

- [Lightweight baseline and reduction options](reductions.md#L1): Half-width model, resolution, batch size, and trade-offs.
- [Optuna experiment sequence](reductions.md#L15): Baseline, short trials, pruning, and longer comparisons.
- [Evaluation and hardware decision](reductions.md#L23): Foreground Dice, PQ, and validation before renting.

### docs/INDEX.md

- [A. Document Registry Matrix](#L5): Registry and document line counts.
- [B. Hierarchical Section & Heading Map](#L14): Exact source heading locations and topics.
- [C. Master Keyword & Topic Search Matrix](#L122): Cross-document topic shortcuts.

## C. Master Keyword & Topic Search Matrix

### Architecture & System Design

| Keywords | Source | Coverage |
|---|---|---|
| Mag-Agent U²-Net, AttentionGate, RSU, CognitiveRefiner | [Architecture](../context.md#L701) | Backbone and refinement modules. |
| Configuration, notebook cells, orchestration | [Cell map](../context.md#L279) | Installation through submission. |
| Kaggle paths, train_images, test_images | [Dataset layout](../context.md#L122) | Input directories and output location. |

### APIs & Database Models

| Keywords | Source | Coverage |
|---|---|---|
| COCO, images, annotations, polygons, string IDs | [Annotation schema](../context.md#L166) | Image records, annotation fields, safe lookups. |
| filament_id, segmentation_rle, CSV | [Submission schema](../context.md#L1010) | Output columns and naming. |
| pycocotools, maskUtils.encode, RLE | [Encoding](../context.md#L983) | Binary mask serialization. |
| Services, databases, migrations | — | No services or database models documented. |

### Algorithms, Math & Science

| Keywords | Source | Coverage |
|---|---|---|
| Dice, PQ, TP, FP, FN, IoU | [Metrics](../context.md#L25) | Pixel overlap and instance quality objectives. |
| Sobel, structure tensor, coherence, limb darkening | [Preprocessing](../context.md#L455) | H-alpha-derived input features. |
| DiceBCELoss, BCEWithLogitsLoss | [Loss](../context.md#L853) | Training objective. |
| Connected components, threshold, MIN_AREA | [Inference](../context.md#L928) | Instance extraction and filtering. |
| Solar physics, spine, barbs, fragmentation | [Scientific context](../context.md#L1495) | Morphology and recognition challenges. |

### Machine Learning & Workflows

| Keywords | Source | Coverage |
|---|---|---|
| ToTensorV2, CHW, masks, channel permutation | [Tensor rules](../context.md#L574) | Shape contracts and existing fixes. |
| AdamW, AMP, GradScaler, checkpoint | [Training](../context.md#L886) | Baseline optimization and validation. |
| H-alpha-only inference, ground truth restrictions | [Coding constraints](../context.md#L1197) | Required implementation invariants. |
| Semantic vs instance, approximate proxy, local PQ | [Limitations](../context.md#L1369) | Current design ceilings. |
| Threshold tuning, spine supervision, TTA, Optuna | [Future work](../context.md#L1443) | Deferred improvement candidates. |
| submission.csv, report, repository | [Deliverables](../context.md#L1550) | Competition artifacts. |
| Lightweight baseline, half-width, Optuna pruning, staged experiments | [Reduction plan](reductions.md#L1) | Proposed low-cost experiments; later comparisons remain optional. |
