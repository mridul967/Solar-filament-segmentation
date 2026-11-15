"""Write raw and solar-postprocessed submission CSVs for one trained run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filament_mask_rcnn.inference import build_predictor, write_submission_pair
from filament_mask_rcnn.model import load_run_cfg
from filament_mask_rcnn.tiling import TilingSettings
from filament_mask_rcnn.artifacts import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--test-image-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--inference-modes", choices=("full", "tiled"), nargs="+", default=("full",))
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from filament_mask_rcnn.postprocess import solar_postprocess

    cfg = load_run_cfg(
        args.run_dir / "config.yaml",
        args.run_dir / "model_final.pth",
        score_threshold=args.score_threshold,
        device=args.device,
    )
    for mode in dict.fromkeys(args.inference_modes):
        tiling = TilingSettings(args.tile_size, args.tile_overlap) if mode == "tiled" else None
        predictor = build_predictor(cfg, tiling=tiling)
        raw, processed = write_submission_pair(
            predictor,
            args.test_image_root,
            args.run_dir / f"submission_{mode}_raw.csv",
            args.run_dir / f"submission_{mode}_postprocessed.csv",
            solar_postprocess,
        )
        print(f"{mode} raw submission: {raw}")
        print(f"{mode} postprocessed submission: {processed}")
        write_json(
            args.run_dir / f"inference_tiling_settings_{mode}.json",
            {"enabled": bool(tiling), **(tiling.as_dict() if tiling else {})},
        )
        del predictor


if __name__ == "__main__":
    main()
