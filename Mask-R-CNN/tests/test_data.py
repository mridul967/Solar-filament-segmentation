import json
import tempfile
import unittest
from pathlib import Path

from filament_mask_rcnn.data import prepare_grouped_splits


class GroupedSplitTests(unittest.TestCase):
    def test_duplicate_physical_files_never_cross_the_split(self):
        coco = {
            "images": [
                {"id": 1, "file_name": "a.png"},
                {"id": 2, "file_name": "a.png"},
                {"id": 3, "file_name": "b.png"},
                {"id": 4, "file_name": "c.png"},
            ],
            "annotations": [
                {"id": "uuid-a", "image_id": 1, "category_id": 1},
                {"id": "uuid-b", "image_id": 2, "category_id": 1},
                {"id": "uuid-c", "image_id": 3, "category_id": 1},
            ],
            "categories": [{"id": 1, "name": "Left"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps(coco), encoding="utf-8")
            splits = prepare_grouped_splits(source, root / "splits", val_fraction=0.34, seed=42)
            train = json.loads(splits.train_json.read_text(encoding="utf-8"))
            val = json.loads(splits.val_json.read_text(encoding="utf-8"))

        train_names = {image["file_name"] for image in train["images"]}
        val_names = {image["file_name"] for image in val["images"]}
        self.assertFalse(train_names & val_names)
        self.assertTrue(all(isinstance(item["id"], int) for item in train["annotations"] + val["annotations"]))


if __name__ == "__main__":
    unittest.main()
