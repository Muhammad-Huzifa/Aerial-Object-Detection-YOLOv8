"""Train YOLOv8 from local data or an explicitly requested Roboflow download."""

import argparse
import os
from pathlib import Path
import shutil

from aerial_detection.paths import DATA_DIR, TRAINING_DIR, model_path


def resolve_data_yaml(value):
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "data.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {path}")
    return path


def download_dataset(workspace, project, version=1):
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("Set ROBOFLOW_API_KEY before using --download.")

    try:
        from roboflow import Roboflow
    except ImportError as error:
        raise ImportError("Install the optional SDK: python -m pip install -r requirements-roboflow.txt") from error

    destination = DATA_DIR / f"{project}-{version}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = (
        Roboflow(api_key=api_key).workspace(workspace).project(project)
        .version(version).download("yolov8", location=str(destination))
    )
    return resolve_data_yaml(dataset.location)


def train_model(data, *, epochs=100, batch=16, imgsz=640, device=None,
                model="yolov8n.pt", project=TRAINING_DIR, name="run1",
                workers=0, seed=0, amp=False):
    data_yaml = resolve_data_yaml(data)
    import torch
    from ultralytics import YOLO

    os.environ.setdefault("WANDB_DISABLED", "true")
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    yolo = YOLO(model)
    yolo.train(
        data=str(data_yaml), epochs=epochs, batch=batch, imgsz=imgsz,
        device=device, patience=50, save=True, save_period=10,
        project=str(project), name=name, exist_ok=False,
        verbose=True, plots=True, workers=workers, cache=False,
        amp=amp, seed=seed,
    )
    return yolo


def export_best_checkpoint(model, destination=None):
    """Copy the checkpoint from the trainer's actual output directory."""
    checkpoint = Path(model.trainer.best)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Training did not produce a best checkpoint: {checkpoint}")
    destination = model_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.resolve() != destination.resolve():
        shutil.copy2(checkpoint, destination)
    return destination


def validate_model(checkpoint, data):
    from ultralytics import YOLO

    metrics = YOLO(str(checkpoint)).val(data=str(resolve_data_yaml(data)))
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    return metrics


def positive_integer(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="Train a YOLOv8 aerial detection model.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="Path to a local data.yaml file or dataset directory.")
    source.add_argument("--download", action="store_true", help="Download the configured Roboflow dataset.")
    parser.add_argument("--workspace", default="digital-image-proecessing")
    parser.add_argument("--dataset-project", default="visdrone-1as21-gd5bz")
    parser.add_argument("--version", type=positive_integer, default=1)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=positive_integer, default=100)
    parser.add_argument("--batch", type=positive_integer, default=16)
    parser.add_argument("--imgsz", type=positive_integer, default=640)
    parser.add_argument("--device", help="Ultralytics device, such as cpu or 0.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--project", type=Path, default=TRAINING_DIR)
    parser.add_argument("--name", default="run1")
    parser.add_argument("--weights-out", type=Path, default=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 0:
        parser.error("--workers must be zero or greater.")
    try:
        data = (
            download_dataset(args.workspace, args.dataset_project, args.version)
            if args.download else resolve_data_yaml(args.data)
        )
        model = train_model(
            data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
            device=args.device, model=args.model, project=args.project,
            name=args.name, workers=args.workers, seed=args.seed, amp=args.amp,
        )
        checkpoint = export_best_checkpoint(model, args.weights_out)
        print(f"Best checkpoint saved to: {checkpoint}")
        validate_model(checkpoint, data)
    except (FileNotFoundError, ValueError, ImportError) as error:
        parser.exit(1, f"Training failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
