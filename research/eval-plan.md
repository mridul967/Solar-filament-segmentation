The Kaggle self-eval notebook is the metric contract to mirror for V7 calibration and audit. The most important finding: V7’s current split/merge labels are reversed.

What the official evaluator does:

- Uses strict `IoU > 0.5` matching.
- Computes PQ as `sum(matched IoU) / (TP + 0.5 FP + 0.5 FN)`.
- Evaluates each annotator-image pass separately against the same submission prediction for that physical image.
- Defines:
  - one GT overlapping multiple predictions (`1:n`) as fragmentation/split;
  - one prediction overlapping multiple GT masks (`n:1`) as over-merge.
- Produces pairwise IoU/Dice distributions and `1:n` / `n:1` diagnostic counts.

Safe changes to adopt in V7:

1. Replace V7’s local PQ counting with an exact in-memory equivalent of the notebook’s `get_overlap_matrices`, `fp_count_hit`, `fn_count_hit`, and `get_pq_score`.

   This should be used for cached decoder calibration, checkpoint diagnostics, and the frozen audit. It ensures local PQ agrees with Kaggle’s handling of FP/FN and duplicate annotation passes.

2. Correct the current split/merge labels.

   Current V7 computes:

   - `nonzero.sum(axis=0) - 1` and calls it `splits`;
   - `nonzero.sum(axis=1) - 1` and calls it `merges`.

   Those are backwards. Columns are predictions, so multiple GT matches per prediction are over-merges. Rows are GT masks, so multiple predicted matches per GT are splits/fragmentation.

3. Add official-style diagnostics to `v7_training_metrics.csv`.

   - PQ from the official matching logic.
   - Pairwise IoU/Dice summary statistics: mean, median, and matched-pair mean.
   - `1:0`, `0:1`, `1:n`, and `n:1` counts/rates.
   - Keep global foreground Dice as an additional semantic diagnostic, but do not confuse it with the notebook’s pairwise Dice distribution.

4. Add an RLE parity check before final submission.

   - Encode V7 predicted masks.
   - Decode them using the same pycocotools convention as the notebook.
   - Assert exact mask equality.
   - Validate that generated `filament_id`s follow the evaluator’s parsing contract.

5. Keep plotting out of training.

   The notebook’s IoU/Dice histograms and many-to-one chart are valuable, but should be generated only for frozen development/audit diagnostics—not every epoch or every calibration candidate.

One naming caution: the notebook derives the prediction image key using text before the first underscore in `filament_id`. V7 currently emits `"{base_id}_{index}"`. This is safe only if `base_id` itself contains no underscore. Add an assertion before submission generation, or adopt the exact identifier format from the official sample submission.
