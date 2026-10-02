"""HTTP tests when application dependencies and the test client are installed."""

import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

API_DEPENDENCIES_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ["cv2", "fastapi", "httpx"]
)


@unittest.skipUnless(API_DEPENDENCIES_AVAILABLE, "Install requirements.txt and requirements-test.txt for HTTP tests.")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from apps import api

        cls.api = api
        cls.client_type = TestClient

    def test_missing_model_reports_503_for_health_and_detection(self):
        with TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"AERIAL_MODEL_PATH": str(Path(directory) / "missing.pt")}):
            with self.client_type(self.api.app) as client:
                self.assertEqual(client.get("/health").status_code, 503)
                result = client.post("/detect/image", files={"file": ("image.jpg", b"data", "image/jpeg")})
                self.assertEqual(result.status_code, 503)

    def test_form_validation_rejects_thresholds_outside_the_range(self):
        with TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"AERIAL_MODEL_PATH": str(Path(directory) / "missing.pt")}):
            with self.client_type(self.api.app) as client:
                result = client.post("/detect/image", data={"conf": "1.1", "iou": "-0.1"},
                                     files={"file": ("image.jpg", b"data", "image/jpeg")})
                self.assertEqual(result.status_code, 422)

    def test_ready_model_returns_detection_metadata_and_rejects_empty_uploads(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "weights.pt"
            checkpoint.write_bytes(b"model is mocked")
            model = MagicMock()
            model.detect_image.return_value = {"num_detections": 0, "detections": []}
            with patch.dict(os.environ, {"AERIAL_MODEL_PATH": str(checkpoint)}), \
                 patch.object(self.api, "Detector", return_value=model), \
                 patch.object(self.api.cv2, "imdecode", return_value=np.zeros((8, 8, 3), dtype=np.uint8)):
                with self.client_type(self.api.app) as client:
                    self.assertEqual(client.get("/health").status_code, 200)
                    empty = client.post("/detect/image", files={"file": ("empty.jpg", b"", "image/jpeg")})
                    self.assertEqual(empty.status_code, 400)
                    result = client.post("/detect/image", data={"conf": "0.4", "iou": "0.6"},
                                         files={"file": ("image.jpg", b"image data", "image/jpeg")})
                    self.assertEqual(result.status_code, 200)
                    self.assertEqual(result.json()["num_detections"], 0)
                    self.assertEqual(model.detect_image.call_args.kwargs, {"conf": 0.4, "iou": 0.6})


if __name__ == "__main__":
    unittest.main()
