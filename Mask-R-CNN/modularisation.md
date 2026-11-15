Shared design is now settled. I will implement this exact stack next—no hidden notebook state, no duplicate inference variants.

```text
src/filament_mask_rcnn/
  data.py          # COCO ID cleanup, grouped split, Detectron2 registration
  model.py         # exact vanilla Detectron2 R50-FPN configuration
  training.py      # DefaultTrainer only
  evaluation.py    # COCOEvaluator on the validation split
  inference.py     # predictor, RLE encoding, raw/postprocessed CSVs
  postprocess.py   # opt-in solar cleanup only
  artifacts.py     # timestamped run-directory paths and JSON/YAML writes

scripts/
  train.py
  evaluate.py
  predict.py

colab/
  vanilla_mask_rcnn.ipynb  # clone → install → Kaggle download → train/eval/predict/download

tests/
  test_data.py
  test_postprocess.py
```

The public flow will be:

```python
splits = prepare_grouped_splits(...)
train_name, val_name, classes = register_splits(...)
cfg = build_vanilla_cfg(...)
train(cfg)
evaluate(cfg, ...)
write_submission(..., postprocessor=None)               # submission_raw.csv
write_submission(..., postprocessor=solar_postprocess)  # submission_postprocessed.csv
```

Every run will produce:

```text
runs/<timestamp>-vanilla/
  config.yaml
  classes.json
  splits/train.json
  splits/val.json
  model_final.pth
  validation_metrics.json
  submission_raw.csv
  submission_postprocessed.csv
```

Both CSVs will contain only:

```csv
filament_id,segmentation_rle
```

The Colab notebook—not the package—will use `google.colab.files.download()` to download both CSVs. Dataset download, Kaggle credentials, Drive mounting, and Detectron2 installation will remain Colab-only.

Detectron2 will be used directly and consistently: `get_cfg()`, `model_zoo`’s `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml`, `DefaultTrainer`, `DefaultPredictor`, `COCOEvaluator`, and `build_detection_test_loader`. The original vanilla geometry and optimizer values will remain unchanged. The strip-head experiment will be excluded because its custom head is absent.

Ponytail call: I will use stdlib `csv` rather than Pandas solely to write submissions, and retain just one raw and one solar-postprocessed path. No registry or plugin system.

Reply **“implement this plan”** when you want me to make the files.
