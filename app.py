"""
Streamlit Application
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from detector import Detector
import time

st.set_page_config(page_title="VisDrone Detection", page_icon="🚁", layout="wide")

@st.cache_resource
def load_detector():
    return Detector('models/best.pt')

def main():
    st.title("🚁 VisDrone Object Detection")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        conf_threshold = st.slider("Confidence", 0.0, 1.0, 0.25, 0.05)
        iou_threshold = st.slider("IoU", 0.0, 1.0, 0.45, 0.05)
        
        st.markdown("---")
        st.info("Upload images to detect objects")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📸 Image", "🎥 Video", "📹 Webcam"])
    
    with tab1:
        image_detection(conf_threshold, iou_threshold)
    
    with tab2:
        video_detection(conf_threshold)
    
    with tab3:
        webcam_detection(conf_threshold)

def image_detection(conf, iou):
    st.header("Image Detection")
    
    uploaded_files = st.file_uploader(
        "Upload Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        detector = load_detector()
        
        for uploaded_file in uploaded_files:
            st.subheader(uploaded_file.name)
            
            col1, col2 = st.columns(2)
            
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            with col1:
                st.image(image, caption="Original", use_container_width=True)
            
            with col2:
                with st.spinner("Detecting..."):
                    start = time.time()
                    results = detector.detect_image(image_cv, conf=conf, iou=iou)
                    elapsed = time.time() - start
                
                annotated_rgb = cv2.cvtColor(results['annotated_image'], cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption="Detections", use_container_width=True)
                
                st.metric("Objects", results['num_detections'])
                st.metric("Time", f"{elapsed:.2f}s")
            
            if results['detections']:
                with st.expander(f"Details ({len(results['detections'])} objects)"):
                    for i, det in enumerate(results['detections']):
                        st.write(f"**{i+1}. {det['class_name']}** - {det['confidence']:.2%}")
            
            st.markdown("---")

def video_detection(conf):
    st.header("Video Detection")
    
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if uploaded_file:
        st.info("Video detection takes time. Processing...")
        
        # Save uploaded file
        video_path = Path("temp_video.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        
        detector = load_detector()
        
        output_path = "output_video.mp4"
        with st.spinner("Processing video..."):
            results = detector.detect_video(
                str(video_path),
                output_path=output_path,
                conf=conf
            )
        
        st.success(f"Processed {results['total_frames']} frames")
        
        if Path(output_path).exists():
            with open(output_path, "rb") as f:
                st.download_button(
                    "Download Result",
                    f,
                    file_name="detected_video.mp4"
                )

def webcam_detection(conf):
    st.header("Webcam Detection")
    st.info("Webcam detection coming soon!")
    st.write("For now, use Image Detection with screenshots")

if __name__ == "__main__":
    model_path = Path("models/best.pt")
    if not model_path.exists():
        st.error("❌ Model not found! Train first: python train.py")
        st.stop()
    
    main()
