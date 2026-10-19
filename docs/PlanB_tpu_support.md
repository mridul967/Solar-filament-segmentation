# Plan A — Quick validation run

For your quick test, change only these three lines in Cell 2:

```python
N_TRIALS = 2
TRIAL_EPOCHS = 2
EPOCHS = 2
```

That will perform two Optuna trials, each capped at two epochs.

One subtle but important point: the final call resumes from the winning trial checkpoint:

```python
fit_model(..., epochs=Config.EPOCHS, ..., resume=trial_checkpoint)
```

So with `TRIAL_EPOCHS = 2` and `EPOCHS = 2`, the winning trial has already completed epochs 0 and 1; resumed training starts at epoch 2 and does no additional epochs. That is probably exactly right for a fast pipeline test: it trains twice, picks the best, then runs calibration, audit, and submission generation. It does not do another final two-epoch train after Optuna.

Also, Optuna pruning will not activate in this tiny test because its warm-up is four epochs:

```python
MedianPruner(n_startup_trials=2, n_warmup_steps=4, ...)
```

That is fine—your goal is validating that the full pipeline works.

# Plan B — TPU (`tpu_max.py`)

This is feasible, but it is a real adaptation rather than changing `cuda` to `xla`.

Kaggle exposes TPU v3-8 through the notebook Accelerator setting; it has eight TPU cores. PyTorch models can run there through `torch_xla`, but all cores need a multiprocessing/distributed training structure for proper utilization. [Kaggle TPU documentation](https://www.kaggle.com/docs/tpu) and the [official PyTorch/XLA migration guide](https://docs.pytorch.org/xla/master/learn/migration-to-xla-on-tpus.html) describe this setup.

The essential changes would be:

- Install/import `torch_xla`; use `torch.device("xla")`, not CUDA.
- Replace CUDA AMP + `GradScaler`. TPUs use `bfloat16`; gradient scaling is normally unnecessary. [PyTorch/XLA AMP guidance](https://docs.pytorch.org/xla/master/perf/amp.html)
- Replace `optimizer.step()` with `xm.optimizer_step(optimizer)` for synchronized multi-core updates.
- Launch a worker process per TPU core and use `DistributedSampler` / `MpDeviceLoader`, so each core trains on a distinct data shard.
- Remove `nn.DataParallel`; it is a CUDA multi-GPU mechanism, not TPU distribution.
- Make checkpoints, CSV histories, Optuna database/results, calibration, audit output, and submission writing run only on the master process—otherwise eight processes attempt to overwrite identical files.
- Make validation metrics reduce across all TPU workers before deciding the best checkpoint or Optuna score.
- Avoid CUDA-only calls such as `torch.cuda.empty_cache()`, CUDA peak-memory queries, and `pin_memory=True`.
- Keep image shapes static—the current fixed `512×512` input is TPU-friendly.

The model itself is mostly portable: convolutions, GroupNorm, interpolation, AdamW, and the loss are normal PyTorch operations. The costly/awkward parts are the Python/OpenCV/COCO instance-decoding evaluation path and coordinating eight workers.

My recommendation: start with Plan A first. A 2-trial × 2-epoch run verifies the data, labels, model, loss, checkpointing, validation, calibration, and submission end-to-end. TPU compilation/startup overhead can make a two-epoch smoke test slower rather than faster. Once that passes, create `tpu_max.py` as a separate TPU-oriented implementation, preserving `ver3-7.py` unchanged.
