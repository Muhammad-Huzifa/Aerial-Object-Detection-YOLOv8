# VisDrone Object Detection (YOLOv8)

YOLOv8-based object detection system trained on the VisDrone dataset, designed for VS Code with GPU support.

## Quick Start

python setup.py
python train.py
python test_inference.py
python api.py
streamlit run app.py

## Project Structure

├── train.py
├── detector.py
├── api.py
├── app.py
├── test_inference.py
├── setup.py
├── requirements.txt
└── models/
    └── best.pt

## Requirements

Python 3.8+
CUDA-enabled GPU (recommended)
8GB+ RAM

## Training

Dataset: VisDrone (Roboflow)
Model: YOLOv8n
Epochs: 100
Output: models/best.pt

## Usage

Python:
from detector import Detector
detector = Detector("models/best.pt")
results = detector.detect_image(image)

API:
python api.py
POST http://localhost:8000/detect/image

Web App:
streamlit run app.py
http://localhost:8501

## Classes

1. pedestrian
2. people
3. bicycle
4. car
5. van
6. truck
7. tricycle
8. awning-tricycle
9. bus
10. motor

## Demo

https://github.com/user-attachments/assets/c4399ebd-9891-4bf0-9a91-435409b02493
