## Objective

Replace V7’s fixed decoder constants with a frozen `DecoderSettings` object selected by grouped-development PQ from cached full-resolution arrays. The HRNet, Intelligence Refiner, losses, CUDA path, DenseCRF, and model-training architecture remain unchanged.

## 1. Add decoder settings as a data object

Add a `DecoderSettings` dataclass near `Config`:

```python
@dataclass(frozen=True)
class DecoderSettings:
    foreground_threshold: float
    min_area: int
    boundary_threshold: float
    spine_threshold: float
    seed_threshold: float
    uncertainty_threshold: float
    min_seed_area: int
```

- No merge-distance field; DBSCAN/proximity merging remains absent.
- Keep current values only as initial baseline defaults.
- Add JSON save/load helpers for `/kaggle/working/v7_decoder_settings.json`.
- Persist metadata with settings: checkpoint path/hash, split seed, grouped development filenames, cache version, PQ/SQ/RQ/Dice, and all error rates.
- Never mutate `Config` during search.

## 2. Separate DenseCRF from instance resolution

Refactor the existing post-processing flow into two layers:

```text
raw foreground probability + limb image
    -> mandatory DenseCRF
    -> densecrf_foreground probability

densecrf_foreground + boundary/spine/seed/uncertainty probabilities
    -> deterministic instance resolver
    -> non-overlapping instance masks
```

Functions:

- `apply_densecrf_2k(...)`: unchanged and mandatory.
- `resolve_instances(...)`: accepts cached DenseCRF foreground plus the four learned maps and a `DecoderSettings`; performs thresholding, filtering, seed validation, and learned watershed.
- `postprocess_probability_v7(...)`: live-inference wrapper that runs DenseCRF once, then delegates to `resolve_instances(...)`.

This makes cached tuning fully CPU-only and ensures the final submission uses the exact same resolver.

## 3. Cache development decision maps once

After the best checkpoint has been loaded:

1. Iterate once over grouped development filenames.
2. Run model inference once per physical observation.
3. Produce and save six full-resolution float32 arrays per image:

   - `foreground.npy`
   - `boundary.npy`
   - `spine.npy`
   - `seed.npy`
   - `uncertainty.npy`
   - `densecrf_foreground.npy`

4. Write a `manifest.json` mapping each physical filename to safe cache file paths and recording checkpoint/split/cache metadata.

Suggested location:

```text
/kaggle/working/v7_decoder_cache/
  manifest.json
  0000_foreground.npy
  0000_boundary.npy
  ...
```

Use float32, not float16, so fine threshold differences are not altered by cache quantization.

The cache builder is the only calibration step allowed to run GPU inference or DenseCRF. Tuning reads only `.npy` files and COCO targets.

## 4. Evaluate cached maps against grouped annotation passes

Add `evaluate_cached_decoder(...)`.

For every cached physical observation:

1. Load the six saved arrays.
2. Resolve instances with one candidate `DecoderSettings`.
3. Evaluate the same predicted instances against every annotation pass associated with that filename.
4. Aggregate globally:

   - PQ, SQ, RQ, Dice
   - TP, FP, FN
   - predicted/ground-truth instance counts
   - split/merge counts
   - fragmentation, over-merge, FP, and FN rates

No model, CUDA tensor, DenseCRF, or image preprocessing is used inside this evaluator.

## 5. Use a staged calibration search

Do not construct a full Cartesian grid. Even cached 2048² watershed/PQ evaluation becomes too large across approximately 150,000 combinations.

### Stage A — foreground extraction

Search:

- foreground threshold: `0.25` to `0.55`
- minimum area: `{50, 100, 200, 400}`

Use connected components only. Keep the top few settings by grouped PQ.

### Stage B — intelligence-map thresholds

For each retained foreground candidate, tune:

- boundary threshold: `0.35` to `0.70`
- spine threshold: `0.35` to `0.70`
- seed threshold: `0.35` to `0.70`
- uncertainty threshold: `0.40` to `0.80`
- minimum seed area: `{5, 9, 16}`

Use coordinate search or small local grids: adjust one decision-map threshold at a time, retain improvements, then repeat until no parameter improves PQ.

### Stage C — fine search

Center a tighter grid around the best values from Stages A/B. If this remains too large, use Optuna TPE exclusively over cached arrays.

Every candidate is judged by grouped development PQ. No training and no re-inference occurs.

## 6. Deterministic winner selection

Rank candidates lexicographically:

1. highest grouped PQ;
2. highest Dice;
3. lowest fragmentation rate;
4. lowest over-merge rate;
5. lowest FP rate;
6. lowest FN rate;
7. stable deterministic parameter ordering if still tied.

Write only the winning settings and full diagnostic summary to `v7_decoder_settings.json`.

## 7. Freeze, audit, then submit

After calibration:

1. Load the frozen `DecoderSettings`.
2. Run the held-out audit fold exactly once using those settings.
3. Append one final `decoder_audit` row to `v7_training_metrics.csv`.
4. Report final PQ/SQ/RQ/Dice, all rates, and selected thresholds.
5. Use the same frozen JSON settings for test submission.
6. Do not tune after audit or inspect test-derived metrics.

## 8. Main-pipeline order

```text
train with baseline decoder
→ retain best checkpoint by existing development PQ rule
→ load best checkpoint
→ build grouped development cache once
→ calibrate DecoderSettings offline from .npy arrays
→ save frozen JSON
→ run audit once
→ generate test submission using frozen settings
```

This preserves V7’s architecture while making decoder parameters evidence-based, reproducible, and selected for the actual instance metric.
