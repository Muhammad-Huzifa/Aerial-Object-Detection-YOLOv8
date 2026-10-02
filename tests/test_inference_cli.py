"""Validate command-line arguments and JSON output without model weights."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from aerial_detection import inference, training


class InferenceCommandTests(unittest.TestCase):
    def test_image_and_json_outputs_are_serializable(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "image.jpg"
            source.write_bytes(b"input handled by the mocked model")
            output = Path(directory) / "annotated.jpg"
            report = Path(directory) / "detections.json"
            mock_detector = MagicMock()
            mock_detector.return_value.detect_image.return_value = {
                "detections": [], "num_detections": 0, "annotated_image": object(),
            }
            cv2 = MagicMock()
            cv2.imwrite.return_value = True
            with patch.object(inference, "Detector", mock_detector), \
                 patch.dict("sys.modules", {"cv2": cv2}), redirect_stdout(io.StringIO()):
                status = inference.main(["--source", str(source), "--output", str(output),
                                         "--json", str(report)])
            self.assertEqual(status, 0)
            self.assertEqual(json.loads(report.read_text()), {"detections": [], "num_detections": 0})

    def test_output_cannot_overwrite_input(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "image.jpg"
            source.write_bytes(b"original")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                inference.main(["--source", str(source), "--output", str(source)])
            self.assertEqual(stopped.exception.code, 1)
            self.assertEqual(source.read_bytes(), b"original")

    def test_json_cannot_overwrite_input(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "image.jpg"
            source.write_bytes(b"original")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                inference.main(["--source", str(source), "--json", str(source)])
            self.assertEqual(stopped.exception.code, 1)
            self.assertEqual(source.read_bytes(), b"original")

    def test_missing_input_fails_before_model_loading(self):
        with TemporaryDirectory() as directory:
            with patch.object(inference, "Detector") as model, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    inference.main(["--source", str(Path(directory) / "missing.mp4")])
            self.assertEqual(stopped.exception.code, 1)
            model.assert_not_called()

    def test_nonpositive_training_epochs_are_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            training.build_parser().parse_args(["--data", "data.yaml", "--epochs", "0"])
        self.assertEqual(stopped.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
