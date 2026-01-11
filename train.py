"""
Training Script for VisDrone Dataset
Run this first to train your model
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
from roboflow import Roboflow

def download_dataset():
    """Download dataset from Roboflow"""
    print("📥 Downloading dataset from Roboflow...")
    
    rf = Roboflow(api_key="K6OqYkRdqTDZskcGXQ8r")
    project = rf.workspace("digital-image-proecessing").project("visdrone-1as21-gd5bz")
    version = project.version(1)
    dataset = version.download("yolov11")
    
    dataset_path = Path(dataset.location)
    print(f"✅ Dataset downloaded to: {dataset_path}")
    
    return dataset_path

def train_model(dataset_path):
    """Train YOLOv8 model"""
    print("\n" + "="*60)
    print("🚀 Starting Training")
    print("="*60)
    
    # 1. Disable WandB explicitly to prevent future errors
    import os
    os.environ["WANDB_DISABLED"] = "true"
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("⚠️ No GPU detected, using CPU")
        device = 'cpu'
    
    # Data config
    data_yaml = dataset_path / 'data.yaml'
    
    # Load model
    print("\n📦 Loading YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    
    # Train
    print("\n🎯 Starting training...")
    results = model.train(
        data=str(data_yaml),
        epochs=100,
        batch=16,
        imgsz=640,
        device=device,
        patience=50,
        save=True,
        save_period=10,
        
        # --- THE FIX IS HERE ---
        project='visdrone_training',  # <--- Changed from 'runs/train' to 'visdrone_training'
        name='run1',                  # <--- Kept name simple
        # -----------------------
        
        exist_ok=True,
        verbose=True,
        plots=True,
        workers=8,
        cache=False,
        amp=False,
    )
    
    print("\n✅ Training Complete!")
    return model

def validate_model(model, dataset_path):
    """Validate trained model"""
    print("\n📊 Validating model...")
    
    data_yaml = dataset_path / 'data.yaml'
    metrics = model.val(data=str(data_yaml))
    
    print(f"\n📈 Results:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    
    return metrics

def main():
    print("="*60)
    print("VisDrone YOLOv8 Training")
    print("="*60)
    
    # Download dataset
    dataset_path = download_dataset()
    
    # Train
    model = train_model(dataset_path)
    
    # Validate
    metrics = validate_model(model, dataset_path)
    
    # Copy best model to models directory
    source = Path('runs/train/visdrone/weights/best.pt')
    dest = Path('models/best.pt')
    dest.parent.mkdir(exist_ok=True)
    
    import shutil
    shutil.copy(source, dest)
    
    print(f"\n✅ Model saved to: {dest}")
    print("\nNext: Run the inference/API/Streamlit scripts")

if __name__ == "__main__":
    main()
