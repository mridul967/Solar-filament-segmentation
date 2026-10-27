# Running the updated V5 on Kaggle

The model architecture is unchanged: **21,133,419 parameters**, the same four-stage visual bridge, RSU blocks, attention gates and FiLM refiner. Input resolution, native crops, training loss terms and tiled/global inference are preserved.

The CUDA workaround is disabling native MHA inference fastpath. It is a candidate fix awaiting your two-T4 run, not a confirmed root cause. The bridge stays FP32, SDPA stays on math, cuDNN remains enabled with benchmarking disabled, and training/evaluation keep DataParallel by default. Faster alternatives are isolated controls, not simultaneous default changes.

## First Kaggle test

Use a fresh kernel with the two-T4 accelerator. If needed, install the data dependencies in a setup cell:

```python
%pip install 'albumentations>=2.0' pycocotools -q
```

The script no longer contains notebook-only `!pip` syntax; you can paste it into a code cell or run the uploaded `.py` file. Set environment variables **before executing the script**, since its Config reads them once at definition time.

```python
import os
os.environ['V5_MODE'] = 'smoke'
```

Then execute V5. This uses the real 1024-resolution pipeline on four training records and three validation records for two epochs. It checks the first warmup transition, evaluation singleton, checkpoint reload, native tiles, TTA, COCO RLE and a visualization. Artifacts go into `/kaggle/working/smoke/`, separately from full training.

Look for `PREFLIGHT PASS`, two finite epoch reports and `SMOKE PASS`. Any CUDA memory-access fault requires a kernel restart before another attempt. Do not rerun only the failing cell or call `empty_cache()` as recovery.

## Full run

After smoke passes, restart the kernel and run the script with:

```python
import os
os.environ['V5_MODE'] = 'train'  # also the default
```

It performs preflight, 25 training epochs with full validation, native-resolution holdout evaluation using the submission decoder, and the 180-image submission. `V5_NATIVE_EVAL=0` can skip the final holdout evaluation if you explicitly need to save runtime; it does not affect epoch validation or model architecture.

Use `V5_OUT_DIR` to choose a new output directory for each independent experiment. Training does not infer resume from old filenames. It deliberately ignores legacy `best_vlm.pth` files unless explicitly requested as a compatible transfer.

## Checkpoint recovery

`last_training.pth` is written atomically immediately after training an epoch, **before validation**, with phase `trained`. After validation and the scheduler step, it is replaced with phase `validated`. It includes the full network, optimizer, scaler, scheduler, best-so-far network, epoch, RNG state and experiment/split identity.

After a failed run, save/attach this file to the next Kaggle session and set:

```python
os.environ['V5_MODE'] = 'train'
os.environ['V5_RESUME'] = '/kaggle/input/YOUR-CHECKPOINT-DATASET/last_training.pth'
```

A `trained` checkpoint resumes at validation without training that epoch again. A `validated` checkpoint begins the next epoch. The scheduler advances once per completed epoch. Best-so-far weights are carried inside the resume file, so they survive moving to a fresh output directory.

The script checks architecture, training config, annotation content and the physical-file split. Smoke/benchmark checkpoints cannot silently resume as full training. Epoch/index seeding makes crop/augmentation selection independent of worker scheduling; exact CUDA bitwise reproducibility is not promised across backend/device changes.

`best_model.pth` is now an authoritative **complete model checkpoint**, including the bridge and metadata. `best_vlm.pth` is a separate bridge export. Normal inference uses the complete model, so an interrupted export cannot mix two epochs. Calling the loader with both paths explicitly also verifies their identity and contents. Legacy raw/split checkpoints are not accepted for resume or automatic inference.

For inference without retraining:

```python
os.environ['V5_MODE'] = 'submit'  # or 'evaluate' for the native holdout only
os.environ['V5_CHECKPOINT'] = '/kaggle/input/YOUR-CHECKPOINT-DATASET/best_model.pth'
```

## If smoke still crashes

Restart the kernel for **each** experiment. Start with a synchronized diagnostic control:

```python
os.environ['V5_MODE'] = 'diagnose'
os.environ['V5_LAUNCH_BLOCKING'] = '1'
os.environ['V5_DEBUG_SYNC'] = '1'
# Optional: reproduce against saved trained weights rather than initialization.
os.environ['V5_CHECKPOINT'] = '/kaggle/input/YOUR-CHECKPOINT-DATASET/last_training.pth'
```

`diagnose` runs full validation, prints record IDs, captures the first two CPU input batches before GPU execution, and synchronizes selected bridge/decoder module boundaries. The last `DEBUG enter` / first failed boundary is more useful than an asynchronous traceback. Artifacts go under `diagnose/`. A changed PyTorch/backend setting is allowed for diagnosis; training config and split must still match the checkpoint.

Captured inputs can be replayed with `V5_REPLAY_BATCH=/kaggle/input/.../diagnostic_batch_1.pth`. Use the same checkpoint with the same captured batch for meaningful comparisons. Capture files contain images/masks/spines/IDs, not model weights.

Change only one variable from the control:

| Experiment | Setting | Purpose |
|---|---|---|
| Original MHA route | `V5_MHA_FASTPATH=1` | Test whether re-enabling fastpath reproduces the failure. Default is 0. |
| Single GPU | `V5_USE_DP=0`, `V5_GPU=0` or `1` | Test each physical device independently. |
| One-image serial evaluation | `V5_EVAL_SERIAL=1` | Fit single-GPU/FP32 diagnosis; reassembles logits before computing the same full-batch criterion. Also works when DataParallel is disabled. |
| FP32 evaluation | `V5_EVAL_AMP=0` | Isolate the mixed-precision evaluation path. Use serial evaluation if memory requires it and compare with the matching serial control. |
| cuDNN disabled | `V5_CUDNN=0` | Only if the synchronized trace points to convolution/BatchNorm. Slower diagnostic variant. |

Do not enable every workaround together: that prevents identifying the relevant change. No automatic CUDA-error catch/retry is implemented because the failed process is not safe to reuse. Do not switch to `model.train()` for validation.

## Benchmarking and speed controls

Use a separate fresh run:

```python
os.environ['V5_MODE'] = 'benchmark'
# Optional trained weights; otherwise benchmarking uses initialization.
os.environ['V5_CHECKPOINT'] = '/kaggle/input/YOUR-CHECKPOINT-DATASET/best_model.pth'
```

After three warmup batches, it records median/p95 data wait, transfer, forward, criterion, backward, optimizer and combined batch times. It also records per-device peak allocated/reserved memory. It trains one short epoch and validates, saving under `benchmark/`; it does not generate a submission. Synchronization makes stage timings interpretable but changes overlap: compare normal full-epoch wall times before promoting an optimization. Set `V5_PROFILE_EPOCH=2` or `5` to measure the added Lovasz training cost at those objective weights. This changes the benchmark objective, not the number of training epochs. The default is epoch-1 weights; validation always uses the full fixed objective.

Try candidates individually after stability:

| Control | Default | Experiment |
|---|---|---|
| `V5_WORKERS` | `2` | Compare 2 and 4 against actual data-wait time. |
| `V5_PERSISTENT_WORKERS` | `0` | Set to 1 to retain worker processes/caches across epochs. Ignored for zero workers. |
| `V5_VAL_CACHE_ITEMS` | `8` | Per-worker LRU cache of downscaled image features keyed by physical filename. Targets remain annotation-specific. Set to 0 for no cache. |
| `V5_CUDNN_BENCHMARK` | `0` | Set to 1 to test convolution autotuning, then repeat singleton/train/eval checks. |
| `V5_BRIDGE_AMP` | `0` | Set to 1 only as a separate bridge precision experiment. Keep loss/reductions FP32 and measure numerical/metric differences. |
| `V5_SDP` | `math` | `auto`, `efficient`, `flash`, `cudnn` are optional support probes. Unsupported T4/dtype/head combinations can fail; no claimed universal speedup. |
| `V5_TTA` | `1` | Global TTA ablation only; compare native holdout metrics before accepting it. Tiles are unaugmented in both cases. |
| `V5_PROFILE_BATCHES` | `12` | Number of measured training batches after warmup. |
| `V5_PROFILE_EPOCH` | `1` | Training-objective epoch used only by benchmark; 2 activates warmed Lovasz, 5 uses its full weight. |

The inference policy remains four global predictions plus nine tile predictions (13 image-view equivalents). Validation uses a fixed full objective across epochs and logs Dice, focal, Lovasz, skeleton and auxiliary terms separately. Training retains the original Lovasz warmup.

DDP migration, `torch.compile`, channels-last, global AMP changes and new attention libraries remain deferred as in the plan. No profiling evidence yet justifies introducing them. No resolution/batch-size reduction has been made.

## Pretrained initialization is now explicit

The historical `VLMSemanticBridge` name is retained for checkpoint compatibility, but it is a custom visual-query bridge with no text encoder. The previous shape/suffix-based “60 SAM tensors loaded” operation was unsafe and has been removed.

By default the bridge is intentionally initialized and trained with the rest of V5. This changes initialization relative to that unsafe preload and can affect score/convergence; the previous claimed pretraining cannot be treated as a trustworthy baseline.

`V5_PRETRAINED` may point to a complete compatible native bridge/model checkpoint. Native keys/shapes must match exactly. For external SAM/HF weights, supply `V5_PRETRAINED_MAP` pointing to a reviewed JSON dictionary of exact destination-to-source keys. The script validates the complete mapping before any copy, rejects shape/nonfinite errors, reports parameter coverage/missing/unused keys and records provenance. It does not invent a mapping between different architectures. Optional HF/safetensors imports occur only when explicitly requested; install their packages in that case.

## Output and acceptance evidence

- `run_manifest.json`: source hash, package/CUDA/cuDNN/driver/GPU details, backend flags, split/annotation hashes and initialization provenance.
- `history.jsonl`: run ID, epoch, record-weighted losses/terms, timing and memory reports.
- `last_training.pth`: recoverable training state; `best_model.pth`: complete inference model; `best_vlm.pth`: bridge export.
- `native_validation.json`: one inference per physical file, each annotation record scored independently, then physical-file macro Dice/local PQ. Also logs split/merge diagnostics. This is not a verified implementation of the competition's evaluator and does not predict the public 0.31 directly.
- `submission.csv` and `submission.csv.audit.json`: exact two-column schema, unique IDs, output dimensions, per-instance RLE round trips and explicit accounting for images with no components. The default test-count check is 180; `V5_EXPECTED_TEST_IMAGES=0` disables it for a deliberately different dataset.

Local verification runs on CPU with PyTorch 2.11.0 and synthetic 64-pixel images / 32-pixel views to keep checks small. It covers finite gradients, model size, loss warmup, strict transfer/loading, RNG replay, training recovery, real data preprocessing, tiles/TTA, RLE, caching and image failure. It cannot establish two-T4 stability or throughput at 1024 resolution. Smoke on Kaggle, repeated full validation, then the full 25-epoch/180-image run are the remaining runtime acceptance steps.
