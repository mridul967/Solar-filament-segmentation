import unittest

import numpy as np

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    solar_postprocess = None
else:
    from filament_mask_rcnn.postprocess import solar_postprocess


@unittest.skipIf(solar_postprocess is None, "OpenCV is installed by requirements-colab.txt")
class SolarPostprocessTests(unittest.TestCase):
    def test_masks_are_exclusive_after_confidence_ordering(self):
        image = np.full((32, 32, 3), 255, dtype=np.uint8)
        first = np.zeros((32, 32), dtype=bool)
        second = np.zeros((32, 32), dtype=bool)
        first[8:20, 8:20] = True
        second[14:26, 14:26] = True

        masks = solar_postprocess(
            image,
            np.stack([second, first]),
            np.array([0.5, 0.9]),
            min_area=1,
            disk_margin=0,
        )

        self.assertEqual(len(masks), 2)
        self.assertEqual(int(np.logical_and(masks[0], masks[1]).sum()), 0)


if __name__ == "__main__":
    unittest.main()
