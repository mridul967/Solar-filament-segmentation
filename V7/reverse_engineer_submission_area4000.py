"""Create a compact diagnostic submission from an existing V7 submission CSV.

This is intentionally an offline operation: it never loads the neural network,
runs inference, changes a pixel in a retained mask, or uses test labels. It only
removes whole predicted instances that are below an area threshold and re-numbers
the remaining instance IDs within each image.

Why 4,000 pixels?
    The existing V7 submission has 30,387 disconnected predictions (median 167
    per image), while MAGFiLO training annotations have a median of 7 filaments
    per observation. At 4,000 pixels the existing output has median 7 predictions
    per test image and only one image with no retained prediction. This is a
    conservative runtime-safe diagnostic candidate, not tuned PQ post-processing.
"""

from pathlib import Path

import pandas as pd
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm

# Change INPUT_CSV only if the original V7 file has a different Colab location.
INPUT_CSV = Path('/content/working/submission-V7-HRMIP-colab.csv')
OUTPUT_CSV = Path('/content/working/submission-V7-HRMIP-area4000.csv')
IMAGE_HEIGHT = 2048
IMAGE_WIDTH = 2048
MIN_AREA_PX = 4000


def parse_image_id(filament_id: str) -> str:
    """Return the competition image ID and reject malformed instance identifiers."""
    image_id, separator, suffix = filament_id.rpartition('_')
    if not separator or not image_id or not suffix.isdigit():
        raise ValueError(f'Invalid filament_id: {filament_id!r}')
    return image_id


def rle_area(rle_string: str) -> int:
    """Measure one existing COCO RLE without decoding it to a full image array."""
    rle = {
        'size': [IMAGE_HEIGHT, IMAGE_WIDTH],
        'counts': rle_string.encode('utf-8'),
    }
    return int(mask_utils.area(rle))


if not INPUT_CSV.is_file():
    raise FileNotFoundError(f'Input submission is missing: {INPUT_CSV}')

submission = pd.read_csv(INPUT_CSV)
required_columns = ['filament_id', 'segmentation_rle']
if list(submission.columns) != required_columns:
    raise ValueError(
        f'Expected exactly {required_columns}, but found {submission.columns.tolist()}'
    )
if submission['filament_id'].duplicated().any():
    raise ValueError('Input CSV contains duplicate filament_id values.')

# Preserve source order as a deterministic tie-breaker after IDs are re-numbered.
submission['_source_order'] = range(len(submission))
submission['image_id'] = submission['filament_id'].map(parse_image_id)
submission['mask_area_px'] = [
    rle_area(rle_string)
    for rle_string in tqdm(submission['segmentation_rle'], desc='Measuring existing RLE areas')
]

# No morphology, unioning, resizing, or mask modification occurs here. A retained
# RLE is byte-for-byte the same segmentation mask produced by the original model.
filtered = submission[submission['mask_area_px'] >= MIN_AREA_PX].copy()
filtered = filtered.sort_values(['image_id', '_source_order']).reset_index(drop=True)
filtered['instance_number'] = filtered.groupby('image_id').cumcount() + 1
filtered['filament_id'] = (
    filtered['image_id'] + '_' + filtered['instance_number'].astype(str)
)

result = filtered[['filament_id', 'segmentation_rle']]
if result['filament_id'].duplicated().any():
    raise RuntimeError('Internal error: re-numbering produced duplicate filament IDs.')

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_CSV, index=False)

counts = filtered.groupby('image_id').size()
all_image_ids = submission['image_id'].unique()
zero_prediction_images = len(set(all_image_ids) - set(counts.index))

print('=' * 60)
print('OFFLINE V7 SUBMISSION REPAIR COMPLETE')
print('=' * 60)
print(f'Input rows: {len(submission):,}')
print(f'Minimum retained area: {MIN_AREA_PX:,} pixels')
print(f'Retained rows: {len(result):,}')
print(f'Removed rows: {len(submission) - len(result):,}')
print(f'Median retained instances/image: {counts.median():.1f}')
print(f'Images with zero retained instances: {zero_prediction_images}')
print(f'Output size: {OUTPUT_CSV.stat().st_size / 1024**2:.2f} MB')
print(f'Output CSV: {OUTPUT_CSV}')
