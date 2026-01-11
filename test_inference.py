import cv2
import os
from pathlib import Path
from detector import Detector

# --- CONFIGURATION ---
INPUT_FILENAME = "london_drone_1.mp4"       # Name of your local video file
OUTPUT_FILENAME = "output_london_drone_1.mp4"
MODEL_PATH = r'visdrone_training/run1/weights/best.pt' 
# ---------------------

def process_video():
    """Runs inference on the local video file"""
    
    # 0. Check if Input File Exists
    if not os.path.exists(INPUT_FILENAME):
        print(f"❌ Error: Input video '{INPUT_FILENAME}' not found!")
        print("   Please make sure the file is in the same folder as this script.")
        return

    # 1. Load Model
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found at {MODEL_PATH}")
        return
    
    print(f"🔄 Loading model from {MODEL_PATH}...")
    detector = Detector(MODEL_PATH)

    # 2. Open Video
    cap = cv2.VideoCapture(INPUT_FILENAME)
    if not cap.isOpened():
        print(f"❌ Could not open video: {INPUT_FILENAME}")
        return

    # 3. Setup Video Writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Use 'mp4v' (or 'avc1' if mp4v fails)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(OUTPUT_FILENAME, fourcc, fps, (width, height))

    print(f"▶ Processing {INPUT_FILENAME} (Press 'Q' to stop early)...")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run Detection
        results = detector.detect_image(frame, conf=0.25)
        
        # Write frame
        out.write(results['annotated_image'])
        
        # Show preview
        cv2.imshow('Inference Preview', results['annotated_image'])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"   Processed {frame_count} frames...")

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Done! Output saved to: {OUTPUT_FILENAME}")

if __name__ == "__main__":
    process_video()