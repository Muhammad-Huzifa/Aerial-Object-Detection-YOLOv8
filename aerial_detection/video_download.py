"""Optional URL download followed by the same local video inference pipeline."""

import argparse
from pathlib import Path

from aerial_detection.inference import main as predict
from aerial_detection.paths import DATA_DIR, model_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download a video URL and run local inference.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "downloaded_video.mp4")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weights")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(argv)

    if not model_path(args.weights).is_file():
        parser.exit(1, "Model weights are missing. Provide --weights before downloading a video.\n")

    try:
        import yt_dlp
    except ImportError:
        parser.exit(1, "Install the optional downloader: python -m pip install -r requirements-download.txt\n")
    args.input.parent.mkdir(parents=True, exist_ok=True)
    options = {"format": "best[ext=mp4]", "outtmpl": str(args.input)}
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            downloader.download([args.url])
    except yt_dlp.utils.DownloadError:
        parser.exit(1, "Video download failed. Check the URL and the downloader output.\n")

    command = ["--source", str(args.input), "--conf", str(args.conf), "--iou", str(args.iou)]
    if args.output is not None:
        command.extend(["--output", str(args.output)])
    if args.weights:
        command.extend(["--weights", args.weights])
    if args.preview:
        command.append("--preview")
    return predict(command)


if __name__ == "__main__":
    raise SystemExit(main())
