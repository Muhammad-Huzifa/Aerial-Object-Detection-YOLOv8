"""Detector regression checks with synthetic model results and mocked video I/O."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from aerial_detection.detector import Detector


class TensorValues:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def prediction():
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=TensorValues([[10, 20, 30, 40]]),
            conf=TensorValues([0.8]), cls=TensorValues([0]),
        ),
        plot=MagicMock(return_value=object()),
    )


def detector():
    service = Detector.__new__(Detector)
    service.device = "cpu"
    service.class_names = {0: "car"}
    service.model = MagicMock()
    service.model.predict.return_value = [prediction()]
    return service


def video_backend():
    cv2 = MagicMock()
    capture = cv2.VideoCapture.return_value
    capture.isOpened.return_value = True
    capture.get.side_effect = lambda field: {
        cv2.CAP_PROP_FPS: 29.97,
        cv2.CAP_PROP_FRAME_WIDTH: 640,
        cv2.CAP_PROP_FRAME_HEIGHT: 480,
    }[field]
    capture.read.side_effect = [(True, object()), (True, object()), (False, None)]
    cv2.VideoWriter.return_value.isOpened.return_value = True
    return cv2


class DetectorTests(unittest.TestCase):
    def test_missing_checkpoint_fails_before_loading_a_model(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Model not found"):
                Detector(Path(directory) / "missing.pt")

    def test_checkpoint_constructor_selects_cpu_without_cuda(self):
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.write_bytes(b"test checkpoint")
            torch = MagicMock()
            torch.cuda.is_available.return_value = False
            ultralytics = MagicMock()
            ultralytics.YOLO.return_value.names = {0: "car"}
            with patch.dict("sys.modules", {"torch": torch, "ultralytics": ultralytics}):
                service = Detector(checkpoint)
            self.assertEqual(service.device, "cpu")
            ultralytics.YOLO.assert_called_once_with(str(checkpoint))

    def test_image_results_include_pixel_bounding_boxes_and_forward_thresholds(self):
        service = detector()
        source = object()
        result = service.detect_image(source, conf=0.4, iou=0.6)
        self.assertEqual(result["num_detections"], 1)
        self.assertEqual(result["detections"][0]["class_name"], "car")
        self.assertEqual(result["detections"][0]["bbox"],
                         {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 40.0})
        service.model.predict.assert_called_once_with(
            source=source, conf=0.4, iou=0.6, device="cpu", verbose=False,
        )

    def test_empty_prediction_returns_no_detections(self):
        service = detector()
        self.assertEqual(service._parse_results(SimpleNamespace(boxes=None)), [])

    def test_invalid_thresholds_are_rejected_before_prediction(self):
        service = detector()
        for conf, iou in [(-0.1, 0.5), (0.5, 1.1), (float("nan"), 0.5)]:
            with self.subTest(conf=conf, iou=iou):
                with self.assertRaises(ValueError):
                    service.detect_image(object(), conf=conf, iou=iou)
        service.model.predict.assert_not_called()

    def test_video_keeps_fractional_fps_and_the_image_detection_schema(self):
        cv2 = video_backend()
        service = detector()
        with TemporaryDirectory() as directory, patch.dict("sys.modules", {"cv2": cv2}):
            result = service.detect_video("input.mp4", Path(directory) / "output.mp4", iou=0.7)
        self.assertEqual(result["total_frames"], 2)
        self.assertEqual([frame["frame"] for frame in result["detections"]], [0, 1])
        self.assertIn("bbox", result["detections"][0]["detections"][0])
        self.assertEqual(cv2.VideoWriter.call_args.args[2], 29.97)
        self.assertEqual(service.model.predict.call_args.kwargs["iou"], 0.7)
        cv2.VideoCapture.return_value.release.assert_called_once()
        cv2.VideoWriter.return_value.release.assert_called_once()

    def test_prediction_failure_releases_capture_and_writer(self):
        cv2 = video_backend()
        service = detector()
        service.model.predict.side_effect = RuntimeError("Prediction failed")
        with TemporaryDirectory() as directory, patch.dict("sys.modules", {"cv2": cv2}):
            with self.assertRaisesRegex(RuntimeError, "Prediction failed"):
                service.detect_video("input.mp4", Path(directory) / "output.mp4")
        cv2.VideoCapture.return_value.release.assert_called_once()
        cv2.VideoWriter.return_value.release.assert_called_once()

    def test_unavailable_capture_is_released(self):
        cv2 = video_backend()
        cv2.VideoCapture.return_value.isOpened.return_value = False
        with patch.dict("sys.modules", {"cv2": cv2}):
            with self.assertRaisesRegex(ValueError, "Cannot open video"):
                detector().detect_video("missing.mp4")
        cv2.VideoCapture.return_value.release.assert_called_once()

    def test_unavailable_writer_is_released(self):
        cv2 = video_backend()
        cv2.VideoWriter.return_value.isOpened.return_value = False
        with TemporaryDirectory() as directory, patch.dict("sys.modules", {"cv2": cv2}):
            with self.assertRaisesRegex(ValueError, "Cannot create output"):
                detector().detect_video("input.mp4", Path(directory) / "output.mp4")
        cv2.VideoCapture.return_value.release.assert_called_once()
        cv2.VideoWriter.return_value.release.assert_called_once()

    def test_video_output_cannot_overwrite_input(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / "video.mp4"
            source.write_bytes(b"original")
            with self.assertRaisesRegex(ValueError, "different from the input"):
                detector().detect_video(source, source)
            self.assertEqual(source.read_bytes(), b"original")

    def test_video_without_readable_frames_is_rejected_after_cleanup(self):
        cv2 = video_backend()
        cv2.VideoCapture.return_value.read.side_effect = [(False, None)]
        with patch.dict("sys.modules", {"cv2": cv2}):
            with self.assertRaisesRegex(ValueError, "no readable frames"):
                detector().detect_video("input.mp4")
        cv2.VideoCapture.return_value.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
