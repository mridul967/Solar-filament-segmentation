My recommendation is a ~10.85M-parameter model with foreground, boundary, and spine supervision, trained substantially longer with early stopping. But the audit found two measurement problems we should fix before using Optuna to choose models.

I’ve made no code changes. This plan targets your six-hour Kaggle session budget.

1. Correct the evaluation and data split first

   I need to correct my earlier guidance: V1’s local PQ implementation is not an exact replica of the organizer’s scorer.

   I retrieved version 6 of the organizer’s self-evaluation notebook and compared it with ver3-7.py (versioned_models/ver3-7.py:598).

   | Finding | Current behavior | Planned change |
   |---|---|---|
   | PQ matching | Greedily enforces one-to-one matches | Match the organizer’s qualifying-pair calculation |
   | Empty denominator | Returns PQ = 1 | Match the organizer’s PQ = 0 |
   | Split/merge diagnostics | Uses IoU > 0.1 | Also reproduce the organizer’s IoU > 0 diagnostics |
   | Prediction representation | Converts a binary mask into connected components | Evaluate explicit instance masks, preserving the instances produced by the decoder |

   These matching differences matter when masks overlap; they do not imply that every previous PQ value was wrong. We will verify parity on synthetic examples and real validation predictions. Organizer’s evaluation notebook

   The saved notebook also reports 707 actual JPEGs but 1,154 annotation-image records. MAGFiLO permits several annotators to label the same observation. Our current split randomizes annotation IDs, so the same photograph can enter both training and validation under different annotations. The repository’s older YOLO pipeline already recognizes this issue. MAGFiLO dataset documentation

   Planned data partition:
   - Group records by physical image; group observations from the same date together where the filename provides that date.
   - Approximately 80% training, 10% development, 10% final audit, with group boundaries respected.
   - Use development data for checkpoint selection, early stopping, Optuna, and post-processing.
   - Evaluate the final audit partition after freezing those choices.
   - Save a new split file and explicitly reject any filename overlap.

   V1’s 0.19 remains our public-leaderboard baseline. New local scores will not be directly comparable with the old local validation scores.
2. Preserve instance information and use the existing spine annotations

   The current dataset creates one foreground mask by combining all filaments. That is sufficient for semantic segmentation, but it cannot directly teach the network which neighboring structures belong to different instances.

   Each training example will provide:

   | Target | Purpose |
   |---|---|
   | Foreground mask | Learn the full filament extent |
   | Individual instance masks / ID map | Preserve filament identity |
   | Per-instance boundary map | Learn separators between neighboring filaments |
   | Annotated spine map | Learn the main filament centerlines |
   | Mask-derived skeleton target | Emphasize thin structures, including branches absent from the annotated main spine |

   For multiple annotators, we will retain each annotation set separately. We should not combine different annotators’ masks into a collection of supposedly distinct physical filaments. A practical training sampler will select a physical image and then one of its annotator sets. Validation will predict that image once and evaluate against its annotation sets using the organizer’s semantics.

   Additional implementation details:
   - Apply identical spatial augmentation to images and every target.
   - Use nearest-neighbor interpolation for categorical masks and instance IDs.
   - Rasterize spine coordinates at the chosen training resolution.
   - Check for missing, malformed, or out-of-mask spines; missing supervision must not become a false all-background target.
   - Preserve overlapping ground-truth masks for evaluation. Mark ambiguous overlap pixels appropriately when constructing a single training ID map.
   - Count how many small instances disappear during downsampling.

   These checks make the extra heads learn meaningful targets instead of amplifying annotation inconsistencies.
3. Restore the wider backbone and add small prediction heads

   Restore the original channel widths:

   64 → 128 → 256 → 512, with the corresponding decoder widths.

   This returns the backbone to approximately 10.85M parameters, with only a small additional cost for boundary and spine heads.

   The model will produce three outputs:
   - Foreground logits.
   - Boundary logits.
   - Spine logits.

   The heads will share the decoder features. We will retain the existing convolutional refiner for the foreground branch initially, while treating it as a learned feature-refinement module—not as a proven magnetic or topology model.

   I recommend GroupNorm in the main feature blocks, because our per-GPU batch will remain small. GroupNorm does not depend on batch statistics. The attention gate’s single-channel output will remain a simple sigmoid gate rather than receiving an inappropriate multi-group normalization. Group Normalization paper

   My resolution decision is 512×512 for the first wider-model experiment. V2’s 0.05 does not establish that 768 is intrinsically worse: it changed resolution and post-processing while still training for only two epochs.

   A later 768 fine-tuning stage becomes worthwhile once the 512 model has a useful learning curve. Full 1024 training is outside the default first-session configuration.
4. Use a topology objective selected for the compute budget

   The roadmap proposes soft-clDice. It is relevant: its authors show improved connectivity on tubular segmentation tasks. However, its differentiable skeletonization adds computation. clDice paper

   For our first six-hour experiment, I recommend Skeleton Recall Loss instead. It uses a ground-truth skeleton target and avoids repeatedly skeletonizing the prediction inside the gradient calculation. The ECCV 2024 paper specifically targets connectivity with lower computational overhead. Its benchmark improvements are evidence for trying it, not a guarantee for solar filaments. Skeleton Recall paper, authors’ implementation

   Proposed starting objective:

   foreground BCE + foreground Dice
+ 0.2 × skeleton recall loss
+ 0.2 × boundary supervision
+ 0.2 × spine supervision

   These weights are starting values, not established optima.

   Implementation decisions:
   - Compute Dice and other large reductions in float32.
   - Use BCE/Dice supervision for sparse auxiliary maps, with bounded positive weighting if needed.
   - Introduce the auxiliary weights gradually during the initial epochs.
   - Keep foreground BCE/Dice: skeleton recall alone does not adequately penalize excessive foreground.
   - Record each loss separately so a poorly scaled auxiliary term is visible.

   I would defer the embedding head until the simpler model has been evaluated. Discriminative embeddings can group pixels by instance, but introduce another loss, clustering parameters, overlap handling, and inference cost. They become justified if fragmentation or incorrect instance association remains a major error after the spine/boundary model. Discriminative instance segmentation paper
5. Train long enough, with early stopping and a tightly bounded search

   Your saved configuration has EPOCHS=2, TRIAL_EPOCHS=2, and N_TRIALS=1. Its pruner requires two completed startup trials, so pruning cannot activate in that configuration.

   Proposed initial settings:

   | Setting | Decision |
   |---|---|
   | Maximum training epochs | 40 total, including any continued trial epochs |
   | Early-stopping warm-up | First 10 epochs |
   | Early-stopping patience | 8 validation checks |
   | Meaningful improvement for patience | 0.001 absolute PQ |
   | Checkpoint selection | Highest development PQ |
   | Optimizer | AdamW |
   | Initial learning rate | 2e-4 |
   | Learning-rate reduction | Halve after 3 stagnant validation checks |
   | Neural Optuna trials | Up to 3, including the baseline optimizer settings |
   | Maximum epochs per trial | 8 |
   | Pruning grace period | No pruning during the first 4 epochs |
   | Neural search time allowance | Approximately 75 minutes |

   Early stopping and pruning have different jobs. Early stopping ends a run that has stopped improving; pruning ends a trial that is doing poorly compared with earlier trials at comparable training progress. Optuna’s median pruner supports this warm-up and startup behavior. Optuna documentation

   For this small search, vary only:
   - Learning rate: approximately 1e-4 to 4e-4.
   - Weight decay: 1e-6 to 1e-3.

   Keep the architecture, resolution, split, loss weights, and checkpoint decoder fixed during these trials. Three trials cannot meaningfully optimize a large search space.

   Save complete trial checkpoints. Continue the selected trial’s training state, preserving its best checkpoint, so the trial training contributes to the final run.

   A wall-clock controller will check the remaining budget during training. Optuna’s study timeout alone is insufficient for controlling the duration of an already-running trial. If profiling shows that the search would consume too much time, reduce the number of trials and preserve the main training allocation.
6. Handle both T4s and memory explicitly

   For the notebook-oriented implementation, start with:
   - Two T4s: total batch 4, approximately 2 images per GPU.
   - Single-GPU fallback: batch 2 with accumulation over 2 steps.
   - FP16 automatic mixed precision.
   - Gradient clipping after accumulated gradients have been unscaled.
   - A complete forward/backward/optimizer-step memory test before the search.

   Gradient accumulation must also handle the final incomplete accumulation group correctly. PyTorch AMP guidance

   A notebook-compatible DataParallel wrapper is the smallest initial way to use both GPUs. We will benchmark it rather than assume it improves throughput. PyTorch recommends DDP for efficient multi-GPU training; that becomes the preferred execution path if we move to a standalone training launcher. Neither approach automatically pools both GPUs’ VRAM into one larger allocation. PyTorch parallel-training guidance

   Activation checkpointing is a fallback if measured activation memory requires it; it trades additional computation for memory. PyTorch checkpointing documentation

   We will also cache deterministic preprocessing and annotation targets. The current loader repeatedly reads and processes the same observations, including repeated annotator views.
7. Fix preprocessing defects and keep augmentation controlled

   Two concrete code issues deserve attention:
   - remove_limb_darkening() performs circular masking and min/max normalization; it does not actually estimate a radial limb-darkening profile.
   - Its normalization can produce negative values outside the disk. The subsequent unsigned-byte conversion in get_magnetic_proxy() can then create artificial values.

   Planned fixes are to keep preprocessing finite, explicitly clip the image before byte conversion, and correctly handle the outside-disk region. Training, validation, visualization, and submission must call the same preprocessing function.

   I would initially retain the existing input-channel arrangement to limit simultaneous experimental changes. Replacing the duplicate grayscale channel with raw H-alpha, or implementing a fitted radial correction, should be a separate measured experiment.

   Augmentation will keep flips and right-angle rotations, then add mild brightness/contrast, gamma, and noise variation. Appearance augmentation should affect the H-alpha image coherently; we should not independently distort the derived structure channel as if it were an RGB color channel.
8. Use instance decoding cautiously, then calibrate it

   The strongest warning from the roadmap audit is that watershed can worsen fragmentation if one long filament receives too many seeds.

   We will implement two comparable decoding paths:
   - Connected components with area filtering.
   - A conservative marker-controlled watershed using predicted spines as candidate markers and predicted boundaries as barriers.

   The watershed path needs explicit handling for:
   - Foreground components with no valid spine marker.
   - Several disconnected spine predictions inside one filament.
   - Tiny resulting regions.
   - Adjacent instance labels that must remain distinct during export.

   We should avoid simply applying a compact-object distance-peak recipe to long, curved filaments. HoVer-Net supports the general usefulness of auxiliary predictions for instance separation, but its nuclei-specific geometry is not a direct solar-filament recipe. HoVer-Net paper

   Also, watershed restricted to a foreground mask cannot grow across pixels excluded from that mask. Any gap recovery must come from improved predictions or a separately tested permissive foreground region. Watershed documentation

   After selecting the model, cache its development-set output maps once and run a small post-processing search over:
   - Foreground threshold.
   - Minimum area, measured in native 2048×2048 pixels.
   - Connected components versus watershed.
   - Spine and boundary thresholds for the watershed branch.
   - Closing disabled versus a small closing operation.

   Include the current threshold 0.4, minimum area 200, and no closing as a reference. Expand area candidates above 200 as well as below it; V2’s search only tested 100 and 200.

   Keep post-processing tuning separate from neural trials. Record all candidates, then evaluate the frozen winner on the audit partition. Submission generation must load the saved settings rather than depend on modified notebook globals.
9. Apply the V3–V7 roadmap in stages

   | Roadmap stage | Decision for this implementation |
   |---|---|
   | V3: instance targets, boundary/spine heads | Core implementation |
   | V4: topology and embeddings | Skeleton recall first; clDice comparison and embeddings later |
   | V5: instance reconstruction | Implement conservative watershed alongside the component baseline |
   | V6: cached-output optimization | Include a bounded post-processing search |
   | V7: higher resolution, TTA, folds | Optional 768 fine-tuning and limited flip TTA; 1024 and fold ensembles require later runs |

   For TTA, reverse the transforms and average aligned prediction maps before instance extraction. Tune and evaluate with the same TTA setting used for submission.

   Native-resolution patch training is another later option if the target audit shows substantial loss of thin structures at 512. It would require overlapping inference, probability blending, and instance extraction after stitching; independently labeling each tile would introduce artificial fragments.

   This staged approach follows the broader lesson of nnU-Net: preprocessing, training, validation, and post-processing choices can matter as much as adding architectural complexity. nnU-Net paper
10. Reserve time for producing a trustworthy submission

Proposed six-hour allocation:

| Phase | Target allowance |
|---|---:|
| Data checks, caching, GPU benchmark | 15 minutes |
| Small Optuna search | 75 minutes |
| Continue the selected model | 210 minutes |
| Calibration, audit evaluation, inference | 45 minutes |
| Buffer and artifact saving | 15 minutes |

These are budget allocations, not measured runtime predictions. The first benchmark determines how many epochs and trials are feasible.

Each run will save its own best weights, resumable training checkpoint, configuration, grouped split, metric history, Optuna study, post-processing settings, example predictions, and submission CSV.

Before a full run, checks should cover scorer parity, image-group separation, target alignment, finite gradients for every enabled head, tiny-sample overfitting, checkpoint reload, and an RLE encode/decode round trip.

The first success criterion is a reproducible improvement over V1’s 0.19 public score, supported by better held-out instance segmentation. The solar paper’s 0.9987 pixel accuracy is a different metric on a different dataset; its reported F1 of 0.8323 and IoU of 0.7139 are more informative, but still do not predict our Kaggle PQ. Original solar-filament paper
