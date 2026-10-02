"""FastAPI image detection service."""

from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from aerial_detection.detector import Detector
from aerial_detection.paths import model_path


@asynccontextmanager
async def lifespan(app):
    checkpoint = model_path()
    app.state.detector = Detector(checkpoint) if checkpoint.is_file() else None
    yield
    app.state.detector = None


app = FastAPI(title="VisDrone Detection API", lifespan=lifespan)
app.state.detector = None
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["GET", "POST"], allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "name": "VisDrone Detection API",
        "endpoints": {"detect_image": "/detect/image", "health": "/health", "docs": "/docs"},
    }


@app.get("/health")
def health():
    ready = app.state.detector is not None
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "model_missing", "model_loaded": ready},
    )


@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.25, ge=0.0, le=1.0),
    iou: float = Form(0.45, ge=0.0, le=1.0),
):
    detector = app.state.detector
    if detector is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded"})

    contents = await file.read()
    if not contents:
        return JSONResponse(status_code=400, content={"error": "Empty image upload"})
    try:
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        image = None
    if image is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    result = detector.detect_image(image, conf=conf, iou=iou)
    return {
        "success": True, "filename": file.filename,
        "num_detections": result["num_detections"], "detections": result["detections"],
    }
