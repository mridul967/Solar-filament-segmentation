"""Create one vanilla training run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filament_mask_rcnn.artifacts import create_run_directory, write_json
from filament_mask_rcnn.data import prepare_grouped_splits, register_splits
from filament_mask_rcnn.model import build_vanilla_cfg
from filament_mask_rcnn.training import train
from filament_mask_rcnn.tiling import TilingSettings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-json", required=True, type=Path)
    parser.add_argument("--train-image-root", required=True, type=Path)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tiled-training", action="store_true")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--full-image-prob", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = create_run_directory(args.runs_root)
    splits = prepare_grouped_splits(args.coco_json, run_dir / "splits")
    train_name, val_name, classes = register_splits(splits, args.train_image_root)
    cfg = build_vanilla_cfg(train_name, val_name, run_dir, len(classes), device=args.device)
    (run_dir / "config.yaml").write_text(cfg.dump(), encoding="utf-8")
    write_json(run_dir / "classes.json", {"thing_classes": classes})
    tiling = TilingSettings(args.tile_size, args.tile_overlap, args.full_image_prob) if args.tiled_training else None
    write_json(run_dir / "tiling_settings.json", {"training_enabled": bool(tiling), **(tiling.as_dict() if tiling else {})})
    if tiling:
        from filament_mask_rcnn.tiling import save_training_preview

        save_training_preview(cfg, tiling, run_dir / "training_tile_preview.png")
    train(cfg, resume=args.resume, tiling=tiling, preflight=bool(tiling))


if __name__ == "__main__":
    main()
