"""Command-line image and video inference."""

import argparse
import json
from pathlib import Path

from aerial_detection.detector import Detector
from aerial_detection.paths import OUTPUT_DIR

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def build_parser():
    parser = argparse.ArgumentParser(description="Detect objects in a local image or video.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--weights", help="Checkpoint path; defaults to models/best.pt or AERIAL_MODEL_PATH.")
    parser.add_argument("--output", type=Path, help="Annotated image or MP4 video path.")
    parser.add_argument("--json", type=Path, help="Save detection results as JSON.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--preview", action="store_true", help="Show a local video preview; Q stops it.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not args.source.is_file():
            raise FileNotFoundError(f"Input file not found: {args.source}")
        suffix = args.source.suffix.lower()
        if suffix not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
            raise ValueError(f"Unsupported input file type: {suffix}")
        output = args.output or OUTPUT_DIR / (
            args.source.stem + ("_detected.jpg" if suffix in IMAGE_SUFFIXES else "_detected.mp4")
        )
        if output.resolve() == args.source.resolve():
            raise ValueError("Choose an output path different from the input file.")
        if args.json is not None and args.json.resolve() in {args.source.resolve(), output.resolve()}:
            raise ValueError("Choose a JSON path different from the input and annotated output.")
        if suffix in VIDEO_SUFFIXES and output.suffix.lower() != ".mp4":
            raise ValueError("Use an .mp4 output path for annotated videos.")
        Detector._validate_thresholds(args.conf, args.iou)
        detector = Detector(args.weights)
        output.parent.mkdir(parents=True, exist_ok=True)
        if suffix in IMAGE_SUFFIXES:
            import cv2

            result = detector.detect_image(str(args.source), conf=args.conf, iou=args.iou)
            if not cv2.imwrite(str(output), result["annotated_image"]):
                raise OSError(f"Cannot save image: {output}")
            result = {key: value for key, value in result.items() if key != "annotated_image"}
        else:
            result = detector.detect_video(
                args.source, output_path=output, conf=args.conf, iou=args.iou, preview=args.preview
            )
        if args.json is not None:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Annotated output: {output}")
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"Inference failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
