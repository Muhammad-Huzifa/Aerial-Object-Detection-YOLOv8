"""
Detection Service
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path='models/best.pt'):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        self.class_names = self.model.names
        
        print(f"✅ Detector loaded on {self.device}")
    
    def detect_image(self, image, conf=0.25, iou=0.45):
        """Detect objects in image"""
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )[0]
        
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                detections.append({
                    'class_id': int(cls_id),
                    'class_name': self.class_names[cls_id],
                    'confidence': float(conf),
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                    }
                })
        
        annotated = results.plot()
        
        return {
            'detections': detections,
            'num_detections': len(detections),
            'annotated_image': annotated,
        }
    
    def detect_video(self, video_path, output_path=None, conf=0.25):
        """Detect objects in video"""
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        all_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = self.model.predict(
                source=frame,
                conf=conf,
                device=self.device,
                verbose=False,
            )[0]
            
            detections = self._parse_results(results)
            all_detections.append({
                'frame': frame_count,
                'detections': detections,
            })
            
            if writer:
                annotated = results.plot()
                writer.write(annotated)
            
            frame_count += 1
        
        cap.release()
        if writer:
            writer.release()
        
        return {
            'total_frames': frame_count,
            'detections': all_detections,
        }
    
    def _parse_results(self, results):
        """Parse YOLO results"""
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confidences = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                detections.append({
                    'class_id': int(cls_id),
                    'class_name': self.class_names[cls_id],
                    'confidence': float(conf),
                })
        return detections
