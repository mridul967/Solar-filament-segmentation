"""Detectron2 training lifecycle."""

from __future__ import annotations


def train(cfg, *, resume: bool = False, tiling=None, preflight: bool = False) -> None:
    """Train with the baseline loader or the optional tiled training mapper."""
    if tiling is None:
        from detectron2.engine import DefaultTrainer

        trainer = DefaultTrainer(cfg)
    else:
        from .tiling import build_tiling_trainer, check_training_batch

        trainer = build_tiling_trainer(cfg, tiling)
    trainer.resume_or_load(resume=resume)
    if preflight:
        if tiling is None:
            raise ValueError("preflight is only required for tiled training")
        check_training_batch(trainer)
    trainer.train()
