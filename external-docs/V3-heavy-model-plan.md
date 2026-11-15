Understood. V3 is for making the model identify filaments better, not for local PQ tuning, group-split diagnostics, or metric engineering. You will train and judge it on Kaggle.

We keep the magnetic proxy, cognitive/agent layer, GPU behavior, and `DataParallel` intact.

## V3 implementation plan

### Phase 1 — Improve the three input channels

Keep three channels, but remove the wasted duplicate H-alpha input:

```text
Current:  [limb-corrected H-alpha, same H-alpha copy, magnetic proxy]
V3:       [limb-corrected H-alpha, local-contrast H-alpha, magnetic proxy]
```

The new local-contrast channel will make faint thin filaments, barbs, and dark boundaries easier to distinguish from smooth background. The magnetic proxy and agent continue operating unchanged.

Also improve limb correction so the disk edge is less likely to resemble a dark filament. This is preprocessing only; model input shape stays `[B, 3, H, W]`.

### Phase 2 — Stronger, physically safe augmentation

Keep flips and 90° rotations. Add only image conditions that can realistically vary between solar observations:

- brightness/contrast shift
- gamma variation
- small Gaussian noise
- mild blur

No random crops, aggressive elastic deformations, or heavy distortion: those can damage long filament geometry.

### Phase 3 — Improve the loss without changing output heads

Keep one segmentation output and replace the simple objective with:

```text
Dice + BCE + low-weight soft-clDice
```

soft-clDice explicitly rewards continuous filament centerlines and penalizes fragmentation. It will be computed from the existing single output, so:

- no auxiliary heads
- no boundary/spine output conflicts
- no change to `DataParallel`
- no change to the proxy or agent architecture

The implementation will use float32 probability calculations for numerical safety under mixed precision.

### Phase 4 — Train longer and more stably

The current 25 epochs is likely short for a 10.8M-parameter model trained from scratch.

Add:

- fixed random seed
- longer training budget, initially 60 epochs
- `ReduceLROnPlateau` scheduler using validation loss
- actual early stopping with patience
- save the best validation-loss checkpoint
- version-specific checkpoint and history filenames

This is model-training stabilization, not a local metrics system.

### Phase 5 — Stronger inference

Add 8-way test-time augmentation:

```text
4 rotations × original / horizontal-flip
```

Each prediction is inverted back to its original orientation, then probability maps are averaged before thresholding. This keeps weak, orientation-sensitive filament regions more consistent.

Use the same proxy preprocessing for every transformed input.

### Phase 6 — Preserve full-resolution detail safely

Do not immediately use native-resolution tiled inference in V3. A model trained on globally downsampled 1024² images may not recognize native-resolution 1024² crops at the same scale; that is a train/inference mismatch.

Instead, V3 will first strengthen the 1024² model. Native tiled training and inference will become a separate copied version after this one.

### Phase 7 — Versioned outputs

V3 will write only version-specific artifacts:

```text
best-heavy-ver-3-improved-proxy-agent.pth
heavy-ver-3-improved-history.csv
submission-heavy-ver-3-improved-proxy-agent.csv
```

## Explicitly not included in V3

- Local PQ evaluator, grid search, or cross-validation
- Group-split refactor
- Five-fold ensemble
- Pretrained `timm` backbone; Kaggle’s disabled internet makes weight availability uncertain
- Architecture replacement or removal of proxy/agent
- Native-resolution tile inference

Those are separate experiments, not safe additions to one training version.

## Grill-Me conclusion

The biggest likely weaknesses in the current model are: duplicated input information, limited appearance augmentation, too-short training, and a loss that does not directly reward connectivity. V3 attacks all four while retaining the parts that already proved useful: the magnetic proxy and cognitive refiner.

If you say “implement V3,” I’ll edit only `heavy-ver-3-improved-accuracy-with-proxy-agent.py` and leave every existing version untouched.
