"""
FastAPI Application
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from pathlib import Path
from detector import Detector

app = FastAPI(title="VisDrone Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = None

@app.on_event("startup")
def load_model():
    global detector
    model_path = "models/best.pt"
    if Path(model_path).exists():
        detector = Detector(model_path)
        print("✅ Model loaded")
    else:
        print("⚠️ Model not found. Train first!")

@app.get("/")
def root():
    return {
        "name": "VisDrone Detection API",
        "status": "online",
        "endpoints": {
            "detect_image": "/detect/image",
            "health": "/health",
        }
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
    }

@app.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
):
    if detector is None:
        return JSONResponse(
            status_code=503,
            content={"error": "Model not loaded"}
        )
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image"}
        )
    
    results = detector.detect_image(image, conf=conf)
    
    return {
        "success": True,
        "filename": file.filename,
        "num_detections": results['num_detections'],
        "detections": results['detections'],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
