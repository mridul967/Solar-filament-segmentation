"""Optional solar-specific mask cleanup used for the postprocessed submission."""

from __future__ import annotations

import cv2
import numpy as np


def _solar_disk_mask(image: np.ndarray, margin: int = 18) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones(gray.shape, dtype=bool)
    center, radius = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    disk = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(disk, (int(center[0]), int(center[1])), max(0, int(radius - margin)), 255, -1)
    return disk.astype(bool)


def solar_postprocess(
    image: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    min_area: int = 200,
    max_disjoint_blobs: int = 3,
    disk_margin: int = 18,
) -> list[np.ndarray]:
    """Apply the agreed disk, component, and confidence-priority overlap policy."""
    if len(masks) != len(scores):
        raise ValueError("masks and scores must have the same length")
    solar_disk = _solar_disk_mask(image, margin=disk_margin)
    occupied = np.zeros(solar_disk.shape, dtype=bool)
    accepted: list[np.ndarray] = []

    for index in np.argsort(-scores):
        mask = masks[index].astype(bool) & solar_disk
        labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        blob_count = labels_count - 1
        if blob_count == 0 or blob_count > max_disjoint_blobs:
            continue
        primary = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        exclusive = (labels == primary) & ~occupied
        if int(exclusive.sum()) < min_area:
            continue
        occupied |= exclusive
        accepted.append(exclusive)
    return accepted
