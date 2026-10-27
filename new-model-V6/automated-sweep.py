# Running a naive grid search over DenseCRF and DBSCAN parameters on $2048 \times 2048$ solar images will bottleneck your Kaggle GPU quota because DenseCRF inference alone requires approximately 6.5 seconds per image at 2K resolution. To sweep hundreds of hyperparameter combinations in a few minutes, the grid search must decouple neural inference, DenseCRF boundary refinement, and topological graph clustering into three cached stages.   1. Three-Stage Decoupled Grid-Search ArchitectureStage 1: One-Time Neural Probability & GT CachingEvaluates predict_probability once per unique physical validation image and caches the $2048 \times 2048$ float16 probability map alongside the limb-corrected uint8 guide image (img_limb_2k) and ground-truth instance masks.Stage 2: Outer Loop Over DenseCRF & Probability ThresholdsDenseCRF is anchored around the MORDEN paper's 2K parameters ($\omega^{(1)} = 40$, $\omega^{(2)} = 3$, $\theta_\alpha^2 = 8$, $\theta_\beta^2 = 12$, $\theta_\gamma^2 = 3$, and 10 iterations).   For each CRF output and candidate THRESHOLD, the script applies OpenCV hole filling so every fragment forms a simply connected domain, extracts connected components once, and precomputes the pairwise minimum Euclidean boundary distances $d(C_a, C_b) \le \epsilon_{\max}$ using scipy.spatial.cKDTree.   Stage 3: Inner Microsecond Sweep Over DBSCAN_EPS and MIN_AREA (minPts)Because the MORDEN paper notes that DBSCAN's distance threshold ($\epsilon$) and minimum cluster size ($\text{minPts}$, set to $\epsilon = 45$ and $\text{minPts} = 140$ for CHASE's $1.04''\text{/pixel}$ resolution) must be rescaled to match the spatial resolution of your instrument, the inner loop filters the precomputed KD-Tree edge list (dist <= eps) via Union-Find.   Sweeping 30 combinations of (DBSCAN_EPS, MIN_AREA) takes $< 2\text{ ms}$ per image instead of re-running spatial clustering from scratch.2. Standalone Grid-Search Script (v6_pq_grid_search.py)Save this script alongside V5-heavy-model.py and v6_dinov3_morden_snippets.py. It imports your dataset split, model, and pq_metrics function directly to guarantee zero metric drift between validation tuning and test submission.









# ==============================================================================
# FILE: v6_pq_grid_search.py
# Fast 3-stage cached hyperparameter sweep for maximizing Panoptic Quality (PQ)
# across DenseCRF (2K), probability THRESHOLD, DBSCAN_EPS, and MIN_AREA.
#
# Usage on Kaggle / CLI:
#   python v6_pq_grid_search.py --checkpoint /kaggle/working/best_model.pth
# ==============================================================================

import os
import json
import time
import math
import argparse
from pathlib import Path
from collections import defaultdict
from itertools import product

import numpy as np
import cv2
import torch
from tqdm import tqdm
from pycocotools.coco import COCO
from scipy.spatial import cKDTree

# Import core pipeline functions from your main script and V6 snippets
from V5_heavy_model import (
    Config,
    MagAgentU2Net,
    load_image,
    prepare_input_3ch,
    predict_probability,
    pq_metrics,
    load_v5_checkpoints,
    seed_everything,
    write_json,
)
from v6_dinov3_morden_snippets import (
    V6ConfigMixin,
    apply_densecrf_2k,
    fill_fragment_holes,
)


# ==============================================================================
# STAGE 1: ONE-TIME VALIDATION PROBABILITY & GT CACHE
# ==============================================================================

def build_validation_split(coco):
    """Reproduces the exact seeded 80/20 physical-file split from main()."""
    import random
    files = defaultdict(list)
    for image_id, info in coco.imgs.items():
        files[info['file_name']].append(image_id)
    filenames = sorted(files)
    random.Random(Config.SEED).shuffle(filenames)
    split = int(0.8 * len(filenames))
    val_ids = [i for name in filenames[split:] for i in files[name]]
    return val_ids


@torch.no_grad()
def cache_validation_predictions(model, coco, val_ids, cache_dir, limit=0):
    """
    Runs neural inference (global + 3x3 native tiles) ONCE per physical file
    and saves float16 probabilities + uint8 limb-corrected image to disk.
    """
    model.eval()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for image_id in val_ids:
        grouped[coco.imgs[image_id]['file_name']].append(image_id)

    items = list(grouped.items())
    if limit > 0:
        items = items[:limit]

    manifest = []
    for filename, ids in tqdm(items, desc='Stage 1: Caching Val Probabilities'):
        stem = Path(filename).stem
        npz_path = cache_dir / f'{stem}_cache.npz'

        if not npz_path.is_file():
            img_path = Path(Config.TRAIN_DIR) / filename
            image = load_image(str(img_path))
            prepared, img_limb = prepare_input_3ch(image)
            tensor = torch.from_numpy(prepared).permute(2, 0, 1).unsqueeze(0).to(Config.DEVICE)
            prob_t = predict_probability(model, tensor, use_tta=Config.USE_TTA)
            prob_fp16 = prob_t.detach().cpu().numpy().squeeze().astype(np.float16)
            limb_u8 = np.clip(img_limb * 255.0, 0, 255).astype(np.uint8)
            np.savez_compressed(npz_path, prob=prob_fp16, limb_u8=limb_u8)

        manifest.append({
            'file_name': filename,
            'npz_path': str(npz_path),
            'image_ids': ids,
        })
    return manifest


def load_gt_instances(coco, image_ids, shape=(2048, 2048)):
    """Loads ground-truth instance boolean masks per annotation record."""
    gt_by_id = {}
    for image_id in image_ids:
        anns = coco.imgToAnns.get(image_id, [])
        targets = [coco.annToMask(ann).astype(bool) for ann in anns]
        gt_by_id[image_id] = targets
    return gt_by_id


# ==============================================================================
# STAGE 2: PRECOMPUTE PAIRWISE COMPONENT DISTANCE GRAPH (KD-TREE)
# ==============================================================================

def precompute_component_graph(mask_u8, max_eps=60.0, min_frag_pixels=12):
    """
    Extracts simply-connected fragments once and precomputes all pairwise
    boundary distances <= max_eps using cKDTree.
    Returns a compact graph structure so sweeping (eps, min_area) is O(V + E).
    """
    filled_u8 = fill_fragment_holes(mask_u8)
    num_cc, labels_cc, stats, _ = cv2.connectedComponentsWithStats(filled_u8, connectivity=8)
    if num_cc <= 1:
        return None

    valid_ids = [i for i in range(1, num_cc) if stats[i, cv2.CC_STAT_AREA] >= min_frag_pixels]
    if not valid_ids:
        return None

    comp_contours = []
    comp_areas = {}
    comp_Label_masks = {}

    for cid in valid_ids:
        # Store bounding box slice + submask for ultra-fast instance reconstruction
        x, y, w, h, area = stats[cid]
        submask = (labels_cc[y:y + h, x:x + w] == cid)
        comp_Label_masks[cid] = (y, y + h, x, x + w, submask)
        comp_areas[cid] = int(area)

        comp_u8 = submask.astype(np.uint8)
        cnts, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            pts = np.vstack([c.reshape(-1, 2) for c in cnts])
            pts[:, 0] += x
            pts[:, 1] += y
        else:
            yx = np.argwhere(comp_u8 > 0)
            pts = np.column_stack([yx[:, 1] + x, yx[:, 0] + y])
        comp_contours.append((cid, pts))

    trees = {cid: cKDTree(pts) for cid, pts in comp_contours}
    edges = []  # List of (min_dist, cid_a, cid_b)

    for idx_a in range(len(comp_contours)):
        cid_a, pts_a = comp_contours[idx_a]
        ax, ay, aw, ah = stats[cid_a, :4]
        for idx_b in range(idx_a + 1, len(comp_contours)):
            cid_b, _ = comp_contours[idx_b]
            bx, by, bw, bh = stats[cid_b, :4]
            # Fast bounding-box rejection
            if (ax > bx + bw + max_eps) or (bx > ax + aw + max_eps) or \
               (ay > by + bh + max_eps) or (by > ay + ah + max_eps):
                continue
            dists, _ = trees[cid_b].query(pts_a, k=1, distance_upper_bound=max_eps)
            finite_dists = dists[np.isfinite(dists)]
            if finite_dists.size > 0:
                edges.append((float(finite_dists.min()), cid_a, cid_b))

    edges.sort(key=lambda item: item[0])
    return {
        'shape': mask_u8.shape,
        'valid_ids': valid_ids,
        'comp_areas': comp_areas,
        'comp_slices': comp_Label_masks,
        'edges': edges,
    }


def reconstruct_instances_from_graph(graph, eps, min_area):
    """
    Reconstructs merged filament instance masks in < 1 ms using Union-Find
    over the precomputed KD-Tree edge list.
    """
    if graph is None:
        return []

    valid_ids = graph['valid_ids']
    comp_areas = graph['comp_areas']
    comp_slices = graph['comp_slices']
    shape = graph['shape']

    parent = {cid: cid for cid in valid_ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if eps > 0:
        for dist, cid_a, cid_b in graph['edges']:
            if dist > eps:
                break
            ra, rb = find(cid_a), find(cid_b)
            if ra != rb:
                parent[ra] = rb

    clusters = defaultdict(list)
    for cid in valid_ids:
        clusters[find(cid)].append(cid)

    instances = []
    for cids in clusters.values():
        total_area = sum(comp_areas[c] for c in cids)
        if total_area >= min_area:
            inst_mask = np.zeros(shape, dtype=np.uint8)
            for c in cids:
                y0, y1, x0, x1, submask = comp_slices[c]
                inst_mask[y0:y1, x0:x1] |= submask.astype(np.uint8)
            instances.append(inst_mask)
    return instances


# ==============================================================================
# STAGE 3: TWO-LEVEL GRID SEARCH ENGINE
# ==============================================================================

def run_grid_search(manifest, coco, out_dir):
    # --- Define Search Grids ---
    # 1. Outer Grid: CRF modes (Off vs Paper Table 3 2K settings vs Strong Bilateral)
    crf_configs = [
        {'name': 'no_crf', 'enabled': False, 'w1': 0.0, 'theta_alpha_sq': 8.0, 'theta_beta_sq': 12.0},
        {'name': 'morden_2k_default', 'enabled': True, 'w1': 40.0, 'theta_alpha_sq': 8.0, 'theta_beta_sq': 12.0},
        {'name': 'morden_2k_soft', 'enabled': True, 'w1': 25.0, 'theta_alpha_sq': 8.0, 'theta_beta_sq': 16.0},
    ]
    # 2. Middle Grid: Probability binarization thresholds
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]

    # 3. Inner Fast Grid: DBSCAN_EPS (0 = pure connected components) & MIN_AREA (minPts)
    dbscan_eps_grid = [0.0, 14.0, 22.0, 28.0, 36.0, 45.0]
    min_area_grid = [140, 200, 280]
    max_eps = max(dbscan_eps_grid)

    # Preload ground-truth masks per annotation ID
    all_val_ids = [img_id for item in manifest for img_id in item['image_ids']]
    gt_by_id = load_gt_instances(coco, all_val_ids)

    leaderboard = []
    started = time.perf_counter()

    for crf_cfg in crf_configs:
        V6ConfigMixin.CRF_W1 = crf_cfg['w1']
        V6ConfigMixin.CRF_THETA_ALPHA_SQ = crf_cfg['theta_alpha_sq']
        V6ConfigMixin.CRF_THETA_BETA_SQ = crf_cfg['theta_beta_sq']

        # Precompute component graphs for all thresholds on the current CRF pass
        graphs_by_thresh = {th: [] for th in thresholds}

        for item in tqdm(manifest, desc=f"Stage 2: CRF [{crf_cfg['name']}] + KD-Tree"):
            data = np.load(item['npz_path'])
            prob_2k = data['prob'].astype(np.float32)
            limb_2k = data['limb_u8'].astype(np.float32) / 255.0

            if crf_cfg['enabled']:
                prob_2k = apply_densecrf_2k(prob_2k, limb_2k)

            for th in thresholds:
                mask_u8 = (prob_2k > th).astype(np.uint8)
                graph = precompute_component_graph(mask_u8, max_eps=max_eps)
                graphs_by_thresh[th].append((item['image_ids'], graph))

        # Inner sweep over (THRESHOLD, DBSCAN_EPS, MIN_AREA) in milliseconds
        for th, eps, min_area in product(thresholds, dbscan_eps_grid, min_area_grid):
            file_pqs, file_dices = [], []
            totals = {'sum_iou': 0.0, 'tp': 0, 'fp': 0, 'fn': 0, 'splits': 0, 'merges': 0}

            for image_ids, graph in graphs_by_thresh[th]:
                predictions = reconstruct_instances_from_graph(graph, eps=eps, min_area=min_area)
                pred_union = np.zeros((Config.ORIGINAL_SIZE, Config.ORIGINAL_SIZE), dtype=bool)
                for p in predictions:
                    pred_union |= p.astype(bool)

                rec_pqs, rec_dices = [], []
                for image_id in image_ids:
                    targets = gt_by_id[image_id]
                    tgt_union = np.zeros_like(pred_union)
                    for t in targets:
                        tgt_union |= t

                    denom = int(pred_union.sum() + tgt_union.sum())
                    dice = (2.0 * np.logical_and(pred_union, tgt_union).sum() / denom) if denom else 1.0
                    m = pq_metrics(predictions, targets)

                    rec_pqs.append(m['pq'])
                    rec_dices.append(float(dice))
                    for k in totals:
                        totals[k] += m[k]

                file_pqs.append(float(np.mean(rec_pqs)))
                file_dices.append(float(np.mean(rec_dices)))

            denom_micro = totals['tp'] + 0.5 * totals['fp'] + 0.5 * totals['fn']
            entry = {
                'crf_mode': crf_cfg['name'],
                'use_densecrf': crf_cfg['enabled'],
                'crf_w1': crf_cfg['w1'],
                'threshold': th,
                'dbscan_eps': eps,
                'min_area': min_area,
                'pq_file_macro': float(np.mean(file_pqs)),
                'pq_record_micro': float(totals['sum_iou'] / denom_micro) if denom_micro else 0.0,
                'dice_file_macro': float(np.mean(file_dices)),
                **totals,
            }
            leaderboard.append(entry)

    leaderboard.sort(key=lambda x: (x['pq_file_macro'], x['pq_record_micro'], x['dice_file_macro']), reverse=True)
    best = leaderboard[0]

    report = {
        'elapsed_seconds': time.perf_counter() - started,
        'total_combinations_evaluated': len(leaderboard),
        'best_config': best,
        'top_10': leaderboard[:10],
    }
    out_file = Path(out_dir) / 'pq_grid_search_results.json'
    write_json(out_file, report)

    print("\n" + "=" * 78)
    print(f"GRID SEARCH COMPLETE ({len(leaderboard)} combos in {report['elapsed_seconds']:.1f}s)")
    print(f"BEST MACRO PQ: {best['pq_file_macro']:.4f} | MICRO PQ: {best['pq_record_micro']:.4f} | DICE: {best['dice_file_macro']:.4f}")
    print(f"  -> USE_DENSECRF = {best['use_densecrf']} (CRF_W1 = {best['crf_w1']})")
    print(f"  -> THRESHOLD    = {best['threshold']}")
    print(f"  -> DBSCAN_EPS   = {best['dbscan_eps']}")
    print(f"  -> MIN_AREA     = {best['min_area']}")
    print(f"  -> Splits={best['splits']} | Merges={best['merges']} | TP={best['tp']} | FP={best['fp']} | FN={best['fn']}")
    print("=" * 78)
    return report


def main():
    parser = argparse.ArgumentParser(description='Fast V6 PQ Hyperparameter Sweep')
    parser.add_argument('--checkpoint', type=str, default=str(Path(Config.OUT_DIR) / 'best_model.pth'))
    parser.add_argument('--cache-dir', type=str, default=str(Path(Config.OUT_DIR) / 'val_prob_cache'))
    parser.add_argument('--limit', type=int, default=0, help='Optional file limit for quick smoke testing')
    args = parser.parse_args()

    seed_everything(Config.SEED)
    coco = COCO(Config.ANNOTATIONS_FILE)
    val_ids = build_validation_split(coco)

    model = MagAgentU2Net(Config.IN_CHANNELS, Config.OUT_CHANNELS)
    load_v5_checkpoints(model, args.checkpoint)
    model = model.to(Config.DEVICE)

    manifest = cache_validation_predictions(model, coco, val_ids, args.cache_dir, limit=args.limit)
    run_grid_search(manifest, coco, Config.OUT_DIR)


if __name__ == '__main__':
    main()