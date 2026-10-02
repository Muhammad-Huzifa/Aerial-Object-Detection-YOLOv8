"""Image and video upload interface using the shared detector."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

from aerial_detection.detector import Detector
from aerial_detection.paths import model_path


@st.cache_resource
def load_detector(checkpoint):
    return Detector(checkpoint)


def image_detection(detector, conf, iou):
    uploaded_files = st.file_uploader(
        "Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    for uploaded in uploaded_files or []:
        st.subheader(uploaded.name)
        try:
            with Image.open(uploaded) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            original_column, detected_column = st.columns(2)
            with original_column:
                st.image(image, caption="Original", use_column_width=True)
            with detected_column:
                with st.spinner("Detecting objects..."):
                    start = time.perf_counter()
                    result = detector.detect_image(
                        cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), conf=conf, iou=iou
                    )
                    elapsed = time.perf_counter() - start
                st.image(result["annotated_image"], channels="BGR",
                         caption="Detections", use_column_width=True)
                st.metric("Objects", result["num_detections"])
                st.metric("Inference time", f"{elapsed:.2f}s")
            if result["detections"]:
                with st.expander("Detection details"):
                    st.dataframe(result["detections"], use_container_width=True)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            st.error(f"Cannot process this image: {error}")


def video_detection(detector, conf, iou):
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded is None:
        return
    contents = uploaded.getvalue()
    signature = (hashlib.sha256(contents).hexdigest(), conf, iou)
    if st.button("Process video"):
        try:
            with TemporaryDirectory(prefix="aerial-video-") as directory:
                source = Path(directory) / ("input" + Path(uploaded.name).suffix.lower())
                output = Path(directory) / "detected.mp4"
                source.write_bytes(contents)
                with st.spinner("Processing video..."):
                    result = detector.detect_video(source, output_path=output, conf=conf, iou=iou)
                st.session_state["video_result"] = {
                    "signature": signature, "total_frames": result["total_frames"],
                    "data": output.read_bytes(),
                }
        except (OSError, ValueError) as error:
            st.session_state.pop("video_result", None)
            st.error(f"Cannot process this video: {error}")

    result = st.session_state.get("video_result")
    if result and result["signature"] == signature:
        st.success(f'Processed {result["total_frames"]} frames.')
        st.download_button(
            "Download annotated video", data=result["data"],
            file_name="detected_video.mp4", mime="video/mp4",
        )


def main():
    st.set_page_config(page_title="Aerial Object Detection", page_icon="🚁", layout="wide")
    st.title("Aerial Object Detection")
    st.caption("Upload images or videos and run inference with your trained YOLOv8 checkpoint.")
    checkpoint = model_path()
    if not checkpoint.is_file():
        st.error("Model weights are missing. Add models/best.pt or set AERIAL_MODEL_PATH.")
        st.info("See the README for training and checkpoint setup.")
        return

    with st.sidebar:
        st.header("Detection settings")
        conf = st.slider("Confidence", 0.0, 1.0, 0.25, 0.05)
        iou = st.slider("IoU", 0.0, 1.0, 0.45, 0.05)
        st.caption(f"Checkpoint: {checkpoint.name}")
    detector = load_detector(str(checkpoint))
    image_tab, video_tab = st.tabs(["Images", "Video"])
    with image_tab:
        image_detection(detector, conf, iou)
    with video_tab:
        video_detection(detector, conf, iou)
