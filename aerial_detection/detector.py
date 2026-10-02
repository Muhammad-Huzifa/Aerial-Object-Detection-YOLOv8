"""Shared image and video inference with structured detection results."""

import math
from pathlib import Path

from aerial_detection.paths import model_path as resolve_model_path


class Detector:
    def __init__(self, model_path=None):
        self.model_path = resolve_model_path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. Provide a trained checkpoint or run train.py."
            )

        import torch
        from ultralytics import YOLO

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        self.class_names = self.model.names

    @staticmethod
    def _validate_thresholds(conf, iou):
        for name, value in [("Confidence", conf), ("IoU", iou)]:
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")

    def _parse_results(self, result):
        """Use the same bounding-box schema for image and video predictions."""
        if result.boxes is None:
            return []

        boxes = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        detections = []
        for box, score, class_id in zip(boxes, scores, classes):
            class_id = int(class_id)
            x1, y1, x2, y2 = box
            detections.append({
                "class_id": class_id,
                "class_name": self.class_names[class_id],
                "confidence": float(score),
                "bbox": {
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                },
            })
        return detections

    def detect_image(self, image, conf=0.25, iou=0.45):
        self._validate_thresholds(conf, iou)
        result = self.model.predict(
            source=image, conf=conf, iou=iou, device=self.device, verbose=False
        )[0]
        detections = self._parse_results(result)
        return {
            "detections": detections,
            "num_detections": len(detections),
            "annotated_image": result.plot(),
        }

    def detect_video(self, video_path, output_path=None, conf=0.25, iou=0.45, preview=False):
        """Process a local video and release capture/writer handles on every exit."""
        if output_path is not None and Path(output_path).resolve() == Path(video_path).resolve():
            raise ValueError("Choose an output path different from the input video.")
        import cv2

        self._validate_thresholds(conf, iou)
        cap = cv2.VideoCapture(str(video_path))
        writer = None
        frames = []
        try:
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            if output_path is not None:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
                    raise ValueError("Video has invalid frame rate or dimensions.")
                writer = cv2.VideoWriter(
                    str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise ValueError(f"Cannot create output video: {output_path}")

            while True:
                success, frame = cap.read()
                if not success:
                    break
                result = self.model.predict(
                    source=frame, conf=conf, iou=iou, device=self.device, verbose=False
                )[0]
                frames.append({"frame": len(frames), "detections": self._parse_results(result)})
                if writer is not None or preview:
                    annotated = result.plot()
                    if writer is not None:
                        writer.write(annotated)
                    if preview:
                        cv2.imshow("Inference preview (Q to stop)", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if preview:
                cv2.destroyAllWindows()

        if not frames:
            raise ValueError("Video contains no readable frames.")
        return {"total_frames": len(frames), "detections": frames}
