"""Evaluate model_final.pth from one vanilla run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filament_mask_rcnn.artifacts import write_json
from filament_mask_rcnn.data import SplitPaths, register_splits
from filament_mask_rcnn.evaluation import evaluate
from filament_mask_rcnn.model import load_run_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--train-image-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = json.loads((args.run_dir / "classes.json").read_text(encoding="utf-8"))["thing_classes"]
    splits = SplitPaths(args.run_dir / "splits/train.json", args.run_dir / "splits/val.json", classes)
    _, val_name, _ = register_splits(splits, args.train_image_root)
    cfg = load_run_cfg(args.run_dir / "config.yaml", args.run_dir / "model_final.pth", device=args.device)
    metrics = evaluate(cfg, val_name, args.run_dir / "eval_val")
    write_json(args.run_dir / "validation_metrics.json", metrics)


if __name__ == "__main__":
    main()
