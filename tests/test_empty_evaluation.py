"""Empty detector outputs retain normal COCO undefined-metric semantics."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bddcv.constants import DET_CLASSES
from bddcv.evaluation import evaluate


class EmptyEvaluationTests(unittest.TestCase):
    def test_empty_results_keep_absent_sizes_and_classes_undefined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ground_truth = {
                "images": [{"id": 1, "file_name": "one.jpg", "width": 100, "height": 100}],
                "categories": [{"id": i + 1, "name": name} for i, name in enumerate(DET_CLASSES)],
                "annotations": [{"id": i + 1, "image_id": 1, "category_id": 1,
                                 "bbox": [i, i, 2, 2], "area": 4, "iscrowd": 0}
                                for i in range(10)],
            }
            (root / "gt.json").write_text(json.dumps(ground_truth))
            (root / "pred.json").write_text("[]")
            result = evaluate(root / "gt.json", root / "pred.json")
            self.assertEqual(result["overall"]["mAP50_95"], 0)
            self.assertEqual(result["overall"]["mAP_small"], 0)
            self.assertEqual(result["overall"]["mAP_large"], -1)
            self.assertEqual(result["overall"]["mAP50_95_reliable_classes"], 0)
            self.assertEqual(result["overall"]["n_reliable_classes"], 1)
            self.assertTrue(math.isnan(result["per_class"][DET_CLASSES[1]]["AP50_95"]))


if __name__ == "__main__":
    unittest.main()
