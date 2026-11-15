"""Write raw and solar-postprocessed submission CSVs for one trained run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filament_mask_rcnn.inference import build_predictor, write_submission_pair
from filament_mask_rcnn.model import load_run_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--test-image-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-threshold", type=float, default=0.25)
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
    predictor = build_predictor(cfg)
    raw, processed = write_submission_pair(
        predictor,
        args.test_image_root,
        args.run_dir / "submission_raw.csv",
        args.run_dir / "submission_postprocessed.csv",
        solar_postprocess,
    )
    print(f"Raw submission: {raw}")
    print(f"Postprocessed submission: {processed}")


if __name__ == "__main__":
    main()
