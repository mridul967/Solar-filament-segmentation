I confirmed [`HRMIP-colab.py`](/Users/vinodpandey/solar-filament-segmentation/V7/HRMIP-colab.py) is currently identical to the audited V7 script. I’ll modify only this Colab copy.

Your speed direction is sound, but I would make two safeguards: batch size 4 must be explicitly preflight-tested rather than assumed, and skipped PQ epochs must be recorded as “not evaluated,” not as real zero scores—otherwise checkpoint selection and analysis become misleading.

## Implementation plan

1. Colab paths and strict hardware gate

- Change `BASE_DIR` to `/content/dataset` and `OUT_DIR` to `/content/working`.
- Derive the existing train/test/annotation paths from the new base directory.
- Make `DEVICE = 'cuda'`; remove the current CUDA-or-CPU selection.
- At the first lines of `main()`:
  - require CUDA;
  - require GPU name containing `A100`;
  - require approximately 80 GB available GPU memory (so a partitioned/MIG A100 cannot pass);
  - confirm bfloat16 support;
  - create `/content/working`.
- Print the device model, total VRAM, and free VRAM before data loading.

Why: this prevents Colab from silently starting an unusable CPU or partial-A100 run. The A100 80 GB has 80 GB HBM2e and native BF16 Tensor Core support, so BF16 is the appropriate precision target. [NVIDIA A100 specifications](https://www.nvidia.com/en-us/data-center/a100/)

2. Make the 15-epoch target explicit

- Change `EPOCHS` from 25 to 15.
- Set `BATCH_SIZE = 4`, `ACCUM_STEPS = 1`, and `NUM_WORKERS = 8`.
- Keep the effective batch size at 4, but remove the accumulation overhead.
- Preserve pinned-memory loaders and add persistent workers only when worker count is nonzero, so Colab does not repeatedly respawn image-processing workers each epoch.

Why: the proposed “15 epoch” runtime estimate cannot apply while the file is configured for 25 epochs.

3. Add a fail-fast full-resolution batch-size preflight

Before training begins, the script will run one real forward/backward/optimizer cleanup at 2048² using one batch of four, BF16, and the actual model/loss. It will synchronize CUDA and report peak allocated/reserved memory.

- If it fits, training continues at batch 4.
- If it does not fit, it raises a clear OOM error with the measured memory values.
- It will not silently reduce the batch size or alter the architecture.

Why: an 80 GB A100 makes batch 4 plausible, but the true footprint includes the HRNet, five-head intelligence outputs, optimizer state, full-resolution targets, and compiled graph workspaces. “A100 80 GB” alone cannot prove it fits.

4. Change compilation from `max-autotune` to `default`

- Set `COMPILE_MODE = 'default'`.
- Keep `torch.compile`; do not disable it.

Why: PyTorch documents `default` as the balanced performance/overhead mode, while `max-autotune` profiles kernel choices—precisely the startup work you want to avoid for a short 15-epoch Colab run. [PyTorch `torch.compile` modes](https://docs.pytorch.org/docs/2.15/generated/torch.compile.html)

5. Throttle expensive PQ/DenseCRF validation without corrupting selection

- Retain normal tensor validation loss every epoch.
- Run full grouped PQ evaluation only at epochs 1, 5, 10, 15 (and always on the final epoch).
- On intervening epochs:
  - do not run DenseCRF or instance resolution;
  - write `pq_evaluated=False`;
  - store PQ-related fields as empty/NaN, not zero.
- Save best checkpoints only on epochs with an actual PQ result.
- Print a clear “PQ skipped this epoch” message.

Why: zero is a valid-looking but false metric. Including epoch 1 gives an initial PQ baseline; every fifth epoch preserves PQ-based selection while removing most of the costly full-development DenseCRF passes.

6. Replace calibration search with explicitly fixed decoder settings

- Do not call `build_decoder_cache`, `load_decoder_cache`, or `calibrate_decoder_cache`.
- Instantiate the requested fixed settings:

  - foreground threshold: 0.40
  - minimum area: 200
  - boundary threshold: 0.50
  - spine threshold: 0.50
  - seed threshold: 0.50
  - uncertainty threshold: 0.60
  - minimum seed area: 9

- Still write `v7_decoder_settings.json`, but label it `selection_mode: "fixed_defaults"` and record that no decoder search was run.
- Run the held-out audit once using these same frozen settings.
- Generate the submission with those exact settings.

Why: this removes the cache creation and repeated CPU watershed/PQ search while preserving an auditable record of what decoder configuration produced the submission. The tradeoff is explicit: these are fixed defaults, not validation-tuned values.

7. Preserve architecture and evaluation correctness

No changes to:

- HRNet-W18 and its native multi-resolution branches;
- the five intelligence decision maps;
- full 2048×2048 preprocessing/training;
- strict MAGFiLO-only data loading;
- grouped observation splits;
- official-style local PQ logic;
- strict DenseCRF dependency;
- final held-out audit and submission pipeline.

8. Verification after edits

I will run:

- syntax compilation;
- a static check that no Kaggle path or CPU fallback remains;
- a static check that calibration code is no longer reachable from `main()`;
- a one-batch CUDA smoke test only when run in Colab, since this local workspace does not expose your A100.

One correction to the proposed runtime claim: these changes should remove the dominant avoidable overhead, but I would not promise 45–55 minutes before measuring the first epoch. Full-resolution CPU feature construction, data loading, and the four scheduled DenseCRF/PQ passes can still materially affect elapsed time. The first completed epoch will give a reliable extrapolation; after compilation warm-up, record median training-step time and the full epoch-5 validation time separately.
