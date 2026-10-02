# Training and model weights

## Local dataset

Use a dataset exported in YOLO detection format, with images, labels, and a `data.yaml` describing the splits and class names. The training command accepts either that YAML file or a directory containing `data.yaml`.

```bash
python train.py --data data/visdrone/data.yaml
```

Replace the example dataset path with the location of your export. Check the paths inside `data.yaml` against the downloaded dataset before starting a long run. The repository does not include the dataset.

## Roboflow download

Install the optional SDK in your active environment:

```bash
python -m pip install -r requirements-roboflow.txt
```

The default workspace is `digital-image-proecessing`, the project is `visdrone-1as21-gd5bz`, and the version is `1`, preserving the original project configuration. The SDK downloads a `yolov8` export under `data/`. Access depends on your Roboflow account and the selected project.

Set your own API key in the terminal. Replace the example value with your rotated key.

Windows Command Prompt:

```bat
set ROBOFLOW_API_KEY=your_rotated_key
python train.py --download
```

Git Bash, Linux, or macOS:

```bash
export ROBOFLOW_API_KEY="your_rotated_key"
python train.py --download
```

A different dataset can be selected explicitly:

```bash
python train.py --download --workspace my-workspace --dataset-project my-project --version 1
```

The previously committed key must be rotated in Roboflow. Removing it from current code does not remove it from earlier Git commits. The scripts read environment variables and do not automatically load a `.env` file.

## Training settings

| Argument | Default | Purpose |
| --- | --- | --- |
| `--model` | `yolov8n.pt` | Pretrained initializer or another starting checkpoint |
| `--epochs` | `100` | Maximum epochs |
| `--batch` | `16` | Batch size |
| `--imgsz` | `640` | Training image size |
| `--device` | Automatic | CUDA device 0 when available; otherwise CPU |
| `--workers` | `0` | Data-loading workers; a simple cross-platform default |
| `--seed` | `0` | Ultralytics training seed |
| `--amp` | Off | Enable automatic mixed precision |
| `--project` | `visdrone_training/` | Training output parent |
| `--name` | `run1` | Run name; Ultralytics allocates a new directory if it exists |
| `--weights-out` | `models/best.pt` or `AERIAL_MODEL_PATH` | Exported best checkpoint |

Patience remains 50 epochs, checkpoints are saved every 10 epochs, and image caching remains disabled. Training duration depends on the dataset and hardware.

```bash
python train.py --data data/visdrone/data.yaml --device 0 --batch 8 --workers 4
```

Use `--device cpu` if you want to choose CPU explicitly. For an NVIDIA GPU, install a compatible PyTorch CUDA build using the [official version-specific instructions](https://pytorch.org/get-started/previous-versions/#v241).

## Checkpoints and evaluation

Training copies `model.trainer.best` from the actual run directory to the configured destination, then validates that exported checkpoint. This avoids assuming that the run is stored under a particular `runs/train/` path.

The validation command prints mAP50 and mAP50-95. No numerical results are supplied in this repository. To make a result reproducible, preserve the dataset version, run arguments, checkpoint, and the generated Ultralytics logs together.

To use a checkpoint stored elsewhere, pass `--weights` to `predict.py` or set `AERIAL_MODEL_PATH` for the API and Streamlit app. Relative model paths are resolved from the repository root.

References: [Ultralytics training](https://docs.ultralytics.com/modes/train/), [Roboflow Python SDK](https://github.com/roboflow/roboflow-python).
