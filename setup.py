"""Environment bootstrap utility; this file is not a packaging installer."""

import argparse
import subprocess
import sys

from aerial_detection.paths import DATA_DIR, MODELS_DIR, OUTPUT_DIR, PROJECT_ROOT, TRAINING_DIR


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare project folders and install dependencies.")
    parser.add_argument("--skip-install", action="store_true", help="Create folders without installing packages.")
    args = parser.parse_args(argv)
    for directory in (DATA_DIR, MODELS_DIR, OUTPUT_DIR, TRAINING_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    if args.skip_install:
        print("Project directories are ready. Dependencies were not installed.")
        return 0

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")],
            check=True,
        )
        import torch
    except (subprocess.CalledProcessError, ImportError) as error:
        print(f"Setup failed: {error}", file=sys.stderr)
        return 1

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("Setup complete. See README.md for checkpoint setup and training commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
