# 🚁 VisDrone Object Detection (YOLOv8)

A high-performance object detection system based on **YOLOv8**, trained on the **VisDrone dataset** for aerial and surveillance scenarios.

## 🎥 Demo

https://github.com/user-attachments/assets/c4399ebd-9891-4bf0-9a91-435409b02493
## ⚡ Quick Start

python setup.py
python train.py
python test_inference.py

## 📁 Project Structure

├── train.py
├── detector.py
├── test_inference.py
├── setup.py
├── requirements.txt
└── models/
    └── best.pt

## 🧠 Model Details

• Architecture: YOLOv8n  
• Dataset: VisDrone (Roboflow)  
• Training Epochs: 100  
• Output Model: models/best.pt  

## 🖥️ System Requirements

• Python 3.8+  
• CUDA-enabled GPU (recommended)  
• 8GB+ RAM  

## 🎯 Supported Classes

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
