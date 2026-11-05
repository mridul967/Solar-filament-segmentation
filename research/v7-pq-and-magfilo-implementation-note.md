# V7 PQ and MAGFiLO implementation note

**Decision.** Treat this as *filament instance* segmentation, not a binary-mask
problem with a heuristic appended at submission time. First establish a
reproducible, organizer-parity PQ baseline; only then change the V7 model or
decoder. This is a research/audit note—no V7 model code was changed.

## What PQ measures

For one class, Panoptic Quality is

\[
PQ = \frac{\sum_{(p,g)\in TP} IoU(p,g)}{|TP| + 0.5|FP| + 0.5|FN|}
    = SQ \times RQ.
\]

`SQ` is mean IoU over matched masks; `RQ` is
`TP / (TP + 0.5 FP + 0.5 FN)`. A prediction and ground-truth instance match
only when their same-class IoU is **strictly greater than 0.5**. With
non-overlapping instances this matching is unique; unmatched predictions and
targets are FP and FN. Thus PQ gives equal weight to every filament, including
small/faint ones, and penalizes both shape error and split/merge/count error.
[Kirillov et al., 2019, Sec. 4](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf)
([IEEE record 8953237 / DOI](https://doi.org/10.1109/CVPR.2019.00963)).

**Evaluator consequence.** The current V4 helper is a strong reusable base,
but its condition `iou < 0.50` accepts exactly-0.50 pairs; canonical PQ requires
`iou <= 0.50` to reject them. Verify the competition/organizer evaluator before
choosing either convention; use its convention consistently. V4's greedy
ordering is otherwise sound when both GT and predictions are non-overlapping,
because the strict threshold makes matching unique.

## MAGFiLO annotation implications

MAGFiLO records a **separate mask, bounding box, spine polyline, and chirality
per filament** in COCO-style JSON. The source annotation policy expects a tight,
single-piece, hole-free mask that includes prominent barbs, plus a spine that
stays inside the filament and runs end-to-end. It also deliberately retains
natural ambiguity: one annotator may call pieces one large filament while
another calls them several. [Ahmadzadeh et al., 2024, annotation and
post-processing](https://www.nature.com/articles/s41597-024-03876-y.pdf).

This makes PQ appropriate but label-noise-sensitive: grouping differences become
FP/FN even for visually plausible output. The paper's double-blind duplicate
annotation design and retained disagreement support evaluating one prediction
against each available annotation pass rather than unioning passes into a new
target. Its final post-processing also pairs each segmentation with one spine;
the spine is therefore valid supervision/diagnostics, not proof that automatic
spatial merging is correct. [Data records and annotation process](https://www.nature.com/articles/s41597-024-03876-y.pdf).

## Repository audit and reusable assets

* `V4-heavy-model-plan.md` already documents duplicate physical JPEG records,
  filename-safe 80/10/10 splitting, per-pass evaluation, and selection by
  PQ/SQ/RQ with split/merge counts ([plan](../V4-heavy-model-plan.md)).
* `heavy-models/only-inference-code-V4.py` already provides reusable
  `build_physical_records`, split validation, COCO mask/spine rasterization,
  a non-overlapping connected-components decoder, PQ diagnostics, cached
  development logits, bounded decoder calibration, and RLE round-trip checks
  ([implementation](../heavy-models/only-inference-code-V4.py)). Do not copy
  its equality-at-0.5 behavior without organizer-parity confirmation.
* V7 currently builds one **union** foreground target per COCO image record,
  uses a shuffled record-level 80/20 split, selects by validation loss, and
  applies mandatory DenseCRF/hole filling plus optional distance-based DBSCAN
  merging ([dataset](../V7/Morden2048-IntelligenceNet.py),
  [training](../V7/Morden2048-IntelligenceNet.py),
  [decoder](../V7/Morden2048-IntelligenceNet.py)). This can leak repeated
  physical images across partitions and does not measure the submission metric.
  Geometry-only DBSCAN can turn nearby distinct filaments into one prediction;
  under one-to-one PQ that creates merges/FN. Conversely, its clustered mask may
  remain disconnected, contrary to the source annotation convention.

## Evidence-based staged V7 recommendation

1. **Metric/data foundation (before retraining).** Build a local evaluator with
   the organizer's exact empty-image, class, overlap, and threshold rules;
   unit-test exact match, `IoU=0.5`, FP, FN, split, merge, and overlap cases.
   Group all records by `file_name`, persist disjoint train/dev/audit filename
   sets, train against one selected annotation pass, and score each dev/audit
   prediction against every held-out pass. Log mean *and* median pass PQ plus
   SQ/RQ/TP/FP/FN/splits/merges.
2. **Freeze V7 architecture; measure decoders.** Cache native-2K V7 logits for
   dev and compare connected components against the existing CRF/hole/DBSCAN
   pipeline over a small dev-only grid of threshold/minimum area (and DBSCAN
   settings only if retained). Select checkpoint and decoder by PQ, freeze both,
   then run the audit set once. Retain the simplest decoder unless it improves
   frozen-audit PQ and its error breakdown.
3. **Only target diagnosed errors.** If RQ is limited by splits/FN, evaluate
   V4's conservative spine-seeded watershed (keep a component unsplit without
   reliable multiple markers). If merges dominate, disable/limit DBSCAN first;
   then test a boundary or instance-separation head. If SQ is low with normal
   RQ, investigate mask/preprocessing loss choices. Native 2048 training is
   already in V7, so tiling is not the first intervention.

This ordering isolates the metric-alignment and annotation-pass risks from a
model change, and uses the PQ decomposition to decide whether a subsequent
instance head is justified.

## Sources

1. A. Kirillov et al., “Panoptic Segmentation,” CVPR 2019, IEEE document
   8953237. [Official CVF open-access paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.html);
   [DOI](https://doi.org/10.1109/CVPR.2019.00963).
2. A. Ahmadzadeh et al., “A dataset of manually annotated filaments from
   H-alpha observations,” *Scientific Data* 11, 1031 (2024).
   [Official article](https://www.nature.com/articles/s41597-024-03876-y);
   [DOI](https://doi.org/10.1038/s41597-024-03876-y).
