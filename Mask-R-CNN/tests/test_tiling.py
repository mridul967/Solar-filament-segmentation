import unittest

import numpy as np

from filament_mask_rcnn.tiling import TilingSettings, merge_tile_candidates, tile_starts


class TilingTests(unittest.TestCase):
    def test_2048_grid_has_nine_1024_tiles_at_25_percent_overlap(self):
        self.assertEqual(tile_starts(2048, 1024, 0.25), [0, 768, 1024])

    def test_grid_covers_the_last_pixel(self):
        starts = tile_starts(2051, 1024, 0.25)
        self.assertEqual(starts[-1] + 1024, 2051)

    def test_cross_view_duplicates_merge_but_same_view_masks_do_not(self):
        first = np.zeros((8, 8), dtype=bool)
        second = np.zeros((8, 8), dtype=bool)
        first[1:5, 1:5] = True
        second[2:6, 1:5] = True
        settings = TilingSettings(tile_size=4, merge_iou=0.5, merge_ios=0.5)
        candidates = [
            {"mask": first, "score": 0.9, "source": "full", "cut": False},
            {"mask": second, "score": 0.8, "source": (0, 0), "cut": False},
            {"mask": second, "score": 0.7, "source": "full", "cut": False},
        ]
        self.assertEqual(len(merge_tile_candidates(candidates, settings)), 2)


if __name__ == "__main__":
    unittest.main()
