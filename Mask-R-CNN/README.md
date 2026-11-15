# Modular vanilla Mask R-CNN

Run these commands from this directory after the Colab setup cells:

```bash
python scripts/train.py \
  --coco-json /content/dataset/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
  --train-image-root /content/dataset/MAGFiLO_1.0_Kaggle_2026/train/train_images

python scripts/evaluate.py \
  --run-dir runs/<timestamp>-vanilla \
  --train-image-root /content/dataset/MAGFiLO_1.0_Kaggle_2026/train/train_images

python scripts/predict.py \
  --run-dir runs/<timestamp>-vanilla \
  --test-image-root /content/dataset/MAGFiLO_1.0_Kaggle_2026/test/test_images
```

`predict.py` writes `submission_raw.csv` and `submission_postprocessed.csv` to the run directory. The latter uses the optional solar-disk cleanup ablation; both contain only `filament_id,segmentation_rle`.

Run the lightweight checks with:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```
