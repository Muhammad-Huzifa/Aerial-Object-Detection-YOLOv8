# API requests and responses

Start the service from the repository root:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

At startup, the API loads `models/best.pt` or the checkpoint selected by `AERIAL_MODEL_PATH`. If weights are absent, the service still starts and detection requests return HTTP 503.

| Method | Route | Behavior |
| --- | --- | --- |
| GET | `/` | Service name and endpoint links |
| GET | `/health` | HTTP 200 when the model is loaded; HTTP 503 when it is missing |
| POST | `/detect/image` | Detect objects in an uploaded image |
| GET | `/docs` | Interactive Swagger documentation |

## Image request

In Git Bash, Linux, or macOS:

```bash
curl -X POST "http://localhost:8000/detect/image" -F "file=@examples/test.jpg" -F "conf=0.25" -F "iou=0.45"
```

On Windows Command Prompt, use `curl.exe` with the same arguments.

The `file` field contains the image. Confidence and IoU are optional numeric form fields between 0 and 1. They default to 0.25 and 0.45.

## Response

The response contains `success`, `filename`, `num_detections`, and `detections`. Each detection uses this schema:

| Field | Meaning |
| --- | --- |
| `class_id` | Numeric class index from the checkpoint |
| `class_name` | Label from the checkpoint |
| `confidence` | Prediction confidence |
| `bbox.x1`, `bbox.y1` | Top-left coordinates in image pixels |
| `bbox.x2`, `bbox.y2` | Bottom-right coordinates in image pixels |

Image API responses contain metadata, not an annotated image. Use the command line or Streamlit interface for annotated output. Video detection is available through those interfaces; the API has no video route.

Empty or undecodable images return HTTP 400. Invalid threshold values return HTTP 422. The API reads an upload in memory and performs detection on the loaded model.

Reference: [FastAPI forms and files](https://fastapi.tiangolo.com/tutorial/request-forms-and-files/).
