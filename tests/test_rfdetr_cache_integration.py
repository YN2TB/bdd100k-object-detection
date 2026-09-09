"""Exercise the pinned RF-DETR decoder cache without GPU/model construction."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


@unittest.skipUnless(importlib.util.find_spec('rfdetr'), 'run with .venv-rfdetr/bin/python')
class RFDETRCacheIntegrationTests(unittest.TestCase):
    def test_cached_native_decode_preserves_pixels_boxes_and_draft_scale(self):
        import torch
        from PIL import Image
        from rfdetr.datasets.coco import CocoDetection
        from bddcv.cache import DecodedImageCache
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'one.jpg'
            Image.new('RGB', (128, 64), color=(12, 34, 56)).save(image)
            annotation = root / 'annotations.json'
            annotation.write_text(json.dumps({
                'images': [{'id': 1, 'file_name': 'one.jpg', 'width': 128, 'height': 64}],
                'categories': [{'id': 1, 'name': 'pedestrian'}],
                'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1,
                                 'bbox': [16, 8, 32, 24], 'area': 768, 'iscrowd': 0}],
            }))
            dataset = CocoDetection(root, annotation, transforms=None, draft_size=32)
            before, target_before = dataset[0]
            original_scale = dataset._decode_image(1)[1]
            dataset._decode_image = DecodedImageCache(dataset.ids, dataset._decode_image)
            image.unlink()
            after, target_after = dataset[0]
            self.assertEqual(before.tobytes(), after.tobytes())
            self.assertEqual(original_scale, dataset._decode_image(1)[1])
            for key in target_before:
                if isinstance(target_before[key], torch.Tensor):
                    self.assertTrue(torch.equal(target_before[key], target_after[key]), key)


if __name__ == '__main__':
    unittest.main()
