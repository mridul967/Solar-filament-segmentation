# Modular vanilla Mask R-CNN

Run these commands from this directory after the Colab setup cells:

```bash
python scripts/train.py \
  --coco-json /content/dataset/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --train-image-root /content/dataset/MAGFiLO_1.0_Kaggle_2026/train/train_images \
  --tiled-training --tile-size 1024 --tile-overlap 0.25 --full-image-prob 0.25

python scripts/evaluate.py \
  --run-dir runs/<timestamp>-vanilla \
  --train-image-root /content/dataset/MAGFiLO_1.0_Kaggle_2026/train/train_images

python scripts/evaluate.py \
  --run-dir runs/<timestamp>-vanilla \
  --train-image-root /content/dataset/MAGFiLO_1.0_Kaggle_2026/train/train_images \
  --tiled-inference

python scripts/predict.py \
  --run-dir runs/<timestamp>-vanilla \
  --test-image-root /content/dataset/MAGFiLO_1.0_Kaggle_2026/test/test_images \
  --inference-modes full tiled
```

Tiled training uses one random 1024-pixel crop 75% of the time and an unchanged baseline full-image mapper 25% of the time. The training command performs a real mapper/loss preflight before the 5,000-step run.

`predict.py --inference-modes full tiled` writes four files: `submission_full_raw.csv`, `submission_full_postprocessed.csv`, `submission_tiled_raw.csv`, and `submission_tiled_postprocessed.csv`. Every CSV contains only `filament_id,segmentation_rle`.

Run the lightweight checks with:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```
