"""
Setup Script
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run shell command"""
    print(f"\n{description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    print("✅ Done")
    return True

def main():
    print("="*60)
    print("VisDrone YOLOv8 Setup")
    print("="*60)
    
    # Create directories
    print("\n📁 Creating directories...")
    Path('models').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    Path('output').mkdir(exist_ok=True)
    Path('runs').mkdir(exist_ok=True)
    print("✅ Directories created")
    
    # Install requirements
    print("\n📦 Installing requirements...")
    print("This may take a few minutes...")
    
    if not run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing packages"
    ):
        return
    
    # Check PyTorch
    print("\n🔍 Checking PyTorch installation...")
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA not available (CPU mode)")
    except ImportError:
        print("❌ PyTorch not installed properly")
        return
    
    print("\n" + "="*60)
    print("✅ Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. python train.py          # Train model (2-3 hours)")
    print("2. python test_inference.py # Test detection")
    print("3. python api.py            # Start API")
    print("4. streamlit run app.py     # Start web app")

if __name__ == "__main__":
    main()
