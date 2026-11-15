# Tiled training and inference contract

This experiment preserves the vanilla Mask R-CNN architecture, grouped split,
optimizer, 5,000 iterations, and solar cleanup. It changes only the training
view and inference view.

## Fixed first experiment

- Training tiles: 1024 pixels with 25% overlap.
- Training distribution: 75% random native tile; 25% unchanged baseline full-image mapper.
- Inference: one baseline full-image view plus all native tiles, reconciled by
  cross-view mask IoU >= 0.5 or intersection-over-smaller >= 0.8.
- Solar cleanup: existing disk clipping, largest-component selection, minimum
  area 200, and confidence-priority exclusive-pixel assignment.

## Commands

```bash
python scripts/train.py --coco-json <train.json> --train-image-root <images> --tiled-training
python scripts/evaluate.py --run-dir <run> --train-image-root <images>
python scripts/evaluate.py --run-dir <run> --train-image-root <images> --tiled-inference
python scripts/predict.py --run-dir <run> --test-image-root <images> --inference-modes full tiled
```

`predict.py` writes `submission_full_raw.csv`,
`submission_full_postprocessed.csv`, `submission_tiled_raw.csv`, and
`submission_tiled_postprocessed.csv`. Every file has only `filament_id` and
`segmentation_rle` columns.

The run directory records training, evaluation, and inference tiling settings
as JSON so every submission can be traced to its view policy.
