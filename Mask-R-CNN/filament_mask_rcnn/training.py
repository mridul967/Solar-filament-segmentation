"""Detectron2 training lifecycle."""

from __future__ import annotations


def train(cfg, *, resume: bool = False) -> None:
    """Run the unmodified Detectron2 DefaultTrainer."""
    from detectron2.engine import DefaultTrainer

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=resume)
    trainer.train()
