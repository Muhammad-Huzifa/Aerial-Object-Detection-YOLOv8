# VisDrone YOLOv8 Detection System

Complete object detection system for VSCode with GPU.

## Quick Start

```bash
# 1. Setup
python setup.py

# 2. Train (2-3 hours with GPU)
python train.py

# 3. Test
python test_inference.py

# 4. Run API
python api.py

# 5. Run Web App
streamlit run app.py
```

## Files

```
├── train.py              # Training script
├── detector.py           # Detection service
├── api.py               # FastAPI server
├── app.py               # Streamlit app
├── test_inference.py    # Test script
├── setup.py             # Setup script
├── requirements.txt     # Dependencies
└── models/
    └── best.pt          # Trained model (after training)
```

## Requirements

- Python 3.8+
- CUDA GPU (recommended)
- 8GB+ RAM

## Training

Training downloads VisDrone dataset from Roboflow and trains YOLOv8n for 100 epochs.

Output: `models/best.pt`

## Usage

### Python
```python
from detector import Detector

detector = Detector('models/best.pt')
results = detector.detect_image(image)
print(results['detections'])
```

### API
```bash
python api.py
# POST http://localhost:8000/detect/image
```

### Web App
```bash
streamlit run app.py
# Open http://localhost:8501
```

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
