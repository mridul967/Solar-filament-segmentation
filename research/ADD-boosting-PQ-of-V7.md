I completed the research and audit. The supporting note is at [v7-pq-and-magfilo-implementation-note.md](/Users/vinodpandey/solar-filament-segmentation/research/v7-pq-and-magfilo-implementation-note.md).

PQ is the right optimization target: instances match only when IoU is strictly greater than 0.5; then `PQ = SQ × RQ`, where SQ is mean IoU of matched instances and RQ penalizes unmatched predictions/ground truths. Merges and fragmentation primarily hurt RQ, not merely pixel IoU. [Panoptic Segmentation paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf)

Important audit findings:

- V7 currently trains a union foreground mask and has only a union spine target; it cannot learn separation or seed decisions.
- Its random split is by COCO image record, not physical filename. MAGFiLO contains independently annotated duplicate observations, so this can leak the same JPEG across train/validation.
- V7 selects checkpoints by validation loss, not PQ.
- Existing DBSCAN only merges nearby fragments; it cannot make evidence-based splits.
- The repo already has reusable, audited V4 implementations for boundary/spine targets, conservative watershed splitting, cached decoder search, and PQ split/merge diagnostics. Its PQ matcher needs one correction: it currently accepts IoU `== 0.5`, whereas canonical PQ requires `> 0.5`.
- MAGFiLO contains segmentation and spine supervision per filament, and its duplicated annotation passes can disagree on whether nearby pieces are one or multiple filaments—exactly the ambiguity the uncertainty output should represent. [MAGFiLO dataset paper](https://www.nature.com/articles/s41597-024-03876-y.pdf)

## V7 intelligence-layer plan

1. Fix evaluation before changing the architecture.

   - Group train/development/audit splits by physical filename, never COCO record.
   - Evaluate each prediction against every annotation pass for that filename.
   - Add canonical PQ, SQ, RQ, TP, FP, FN, split, and merge diagnostics.
   - Select checkpoints and decoder thresholds by development PQ, not loss.
   - Reuse the V4 evaluator logic, correcting its strict IoU threshold.

2. Replace the union-only dataset output with instance-aware targets.

   For one selected annotation pass per training sample, return:

   - `foreground`: union of its instance masks.
   - `boundary`: dilated inner contour of each individual instance; overlap/ambiguous pixels ignored.
   - `spine`: existing annotated/morphological centerline target.
   - `seed`: a compact midpoint segment from each instance’s spine, one isolated marker region per filament.
   - `uncertainty`: disagreement among duplicate annotation passes for the same physical image; ignored where only one annotation pass exists.
   - `instance_id`: integer map for diagnostics and decoder validation, not a direct regression target.

   All targets remain native 2048² and receive the same geometric augmentation.

3. Refactor `CognitiveRefinerV6` into `IntelligenceRefiner`.

   - Keep its dilated context branches, FiLM conditioning, magnetic gate, and 64-channel refined feature output.
   - Remove its single final-logit convolution.
   - Add five explicit 1×1 prediction heads from the shared refined features:

   ```text
   refined features
   ├─ foreground logits
   ├─ boundary logits
   ├─ spine logits
   ├─ instance-seed logits
   └─ uncertainty logits
   ```

   - Keep the existing stride-8 auxiliary output, renamed as `aux_foreground_logits`.
   - Return a named dictionary rather than positional tuples, preventing head-order mistakes.

4. Use a measurable multi-task objective.

   - Retain foreground Dice + Focal + warmed Lovász loss.
   - Retain skeleton recall on the foreground probability.
   - Add masked BCE+Dice losses for boundary, spine, and seed maps.
   - Add masked BCE for annotation-disagreement uncertainty only where duplicate annotation passes exist.
   - Keep auxiliary foreground supervision.
   - Log every loss separately; use initial conservative auxiliary weights and tune only through held-out PQ ablations.

5. Replace DBSCAN-only instance construction with evidence-based decoding.

   - Keep mandatory DenseCRF first.
   - Threshold foreground.
   - Use low-boundary, high-spine seed regions as watershed markers.
   - Split a connected foreground component only if it has at least two valid, separated seed markers and a supporting boundary ridge.
   - Otherwise retain the component intact to avoid creating false fragmentation.
   - Replace unconditional proximity-based DBSCAN joining with conservative topology-aware merging: bridge only fragments connected by high spine/foreground evidence and low boundary/uncertainty.
   - Use uncertainty only to reject weak, ambiguous predictions—not to cut a confident spine.

6. Calibrate the final decoder against PQ.

   Cache the five development probability maps once, then search a bounded set of foreground, boundary, spine, seed, uncertainty, and minimum-area thresholds. Compare:

   - connected components baseline;
   - seed-and-boundary watershed;
   - watershed plus conservative topology-aware merge.

   Freeze the winning settings, then run the audit split once.

7. Defer unsafe complexity.

   - No embedding/clustering head initially.
   - No TTA until the base decoder improves held-out PQ; if later added, transform and average all five maps identically.
   - No deletion of DenseCRF, HRNet, full-resolution training, or current physics channels.

This keeps V7 coherent: HRNet provides full-resolution features, MORDEN decodes them, the Intelligence Refiner makes explicit pixel decisions, and the post-processor converts those decisions into non-overlapping filament instances optimized for PQ.
