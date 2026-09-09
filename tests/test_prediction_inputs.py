"""Prediction must reject incomplete validation inputs and resolution drift."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import predict
from scripts import predict_rfdetr
from bddcv.prediction import validated_prediction_images
from bddcv.registry import get_model_spec


class PredictionInputTests(unittest.TestCase):
    def test_exact_image_coverage_accepts_all_images_and_rejects_extras(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'one.jpg').touch()
            self.assertEqual(validated_prediction_images(root, {'one.jpg': 7}), [root / 'one.jpg'])
            (root / 'extra.jpg').touch()
            with self.assertRaisesRegex(ValueError, 'coverage'):
                validated_prediction_images(root, {'one.jpg': 7})

    def test_standalone_rfdetr_rejects_missing_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = root / 'gt.json'
            gt.write_text(json.dumps({'images': [{'id': 1, 'file_name': 'missing.jpg'}]}))
            with patch.object(predict_rfdetr, 'GT', gt), patch('sys.argv', ['predict_rfdetr', 'best.pth', '--adapter', str(root)]):
                with self.assertRaisesRegex(ValueError, 'coverage'):
                    predict_rfdetr.main()

    def test_missing_rfdetr_image_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'valid').mkdir()
            gt = root / 'gt.json'
            gt.write_text(json.dumps({'images': [{'id': 1, 'file_name': 'missing.jpg'}]}))
            output = root / 'predictions.json'
            args = SimpleNamespace(adapter=root, conf=0.001, max_det=300)
            with patch.object(predict, 'GT', gt), patch('bddcv.rfdetr.RFDETRAdapter.construct', return_value=object()), patch.object(predict, 'prepare_rfdetr_runtime'):
                with self.assertRaisesRegex(ValueError, 'coverage'):
                    predict.predict_rfdetr(get_model_spec('rfdetr-small'), root / 'weights.pth', output, args)
            self.assertFalse(output.exists())

    def test_unmatched_resolution_rejected_before_backend(self):
        with patch('sys.argv', ['predict', 'best.pt', '--model-id', 'yolo11s', '--imgsz', '1280']), patch.object(predict, 'predict_ultralytics', side_effect=AssertionError('backend should not run')):
            with self.assertRaises(SystemExit) as error:
                predict.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
