# Development and troubleshooting

Keep the virtual environment active and run commands from the repository root.

## Validate changes

```bash
python -m unittest discover -s tests -v
python -m compileall -q aerial_detection apps tests train.py predict.py app.py api.py
python -m pip check
```

Core tests use synthetic prediction results and mocked models or video handles. API tests require the application dependencies and `requirements-test.txt`; install both to run them:

```bash
python -m pip install -r requirements.txt -r requirements-test.txt
python -m unittest discover -s tests -v
```

Checks with mocked models do not measure detection accuracy, GPU behavior, or video codec support. With a real checkpoint, also run the sample image command, a short video, the Streamlit upload flow, and an API image request before relying on the application.

## Git workflow

```bash
git switch main
git pull --ff-only
git switch -c docs/update-project-guide
```

After editing:

```bash
git status
git diff
git add README.md docs
git commit -m "Update project documentation"
git push -u origin docs/update-project-guide
```

Open a pull request from that branch to `main`. Keep datasets, credentials, weights, and generated outputs out of commits; their paths are covered by `.gitignore`.

## Optional URL download

Install the optional downloader only if you use `shortvideo.py`:

```bash
python -m pip install -r requirements-download.txt
python shortvideo.py --url "YOUR_VIDEO_URL" --weights models/best.pt
```

Replace the example URL with the video you want to process. The helper checks that weights exist before downloading. It selects an available MP4 stream and then uses the same local video inference code. URL downloading is separate from dataset downloading.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Model weights are missing | Provide a trained checkpoint at `models/best.pt`, use `--weights`, or set `AERIAL_MODEL_PATH`. |
| Roboflow download fails | Check `ROBOFLOW_API_KEY`, project access, workspace, project name, and version. The local `--data` option does not require Roboflow access. |
| Dataset files cannot be found | Inspect `data.yaml` and the image/label directories in your export. |
| CUDA is unavailable | Check the PyTorch build and NVIDIA driver. CPU mode is supported. |
| Image display raises a Streamlit argument error | Use the pinned dependencies. This interface uses `use_column_width` for Streamlit 1.31.0. |
| Video cannot be opened or written | Check the input file, output folder, and local OpenCV codec support. Annotated video output uses MP4 with `mp4v`. |
| Video preview fails on a server | Omit `--preview`. Preview requires a local display-capable OpenCV installation. |
| Counts or class names differ from expectations | The loaded checkpoint supplies the class labels; check that it was trained on the intended dataset. |
| API health returns 503 | The service started without model weights. Set up the checkpoint and restart the service. |

The optional Roboflow SDK installs a headless OpenCV distribution alongside the desktop package. If desktop preview is unavailable after installing that SDK, use file output or the Streamlit upload interface. Video results include per-frame detections in memory, so use shorter clips when memory is limited.

References: [PyTorch versions](https://pytorch.org/get-started/previous-versions/), [yt-dlp installation](https://github.com/yt-dlp/yt-dlp#installation).
