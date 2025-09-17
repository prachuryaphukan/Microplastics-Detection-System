# train_microplastics_model.py
import os
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class MicroplasticDetector:
    def __init__(self):
        self.model = None
        self.class_names = ['microplastic', 'fiber', 'fragment', 'sphere']
        
    def train_model(self, dataset_path):
        """Train YOLOv8 model on microplastics dataset"""
        # Initialize YOLOv8 model
        model = YOLO('yolov8n.pt')  # nano version for faster inference
        
        # Train the model
        results = model.train(
            data=f'{dataset_path}/data.yaml',
            epochs=100,
            imgsz=640,
            batch=16,
            name='microplastics_detector',
            patience=10,
            save=True,
            device=0 if torch.cuda.is_available() else 'cpu'
        )
        
        self.model = model
        return results
    
    def load_trained_model(self, model_path):
        """Load pre-trained model"""
        self.model = YOLO(model_path)
        
    def detect_microplastics(self, image_path, confidence=0.5):
        """Detect microplastics in image"""
        if not self.model:
            raise ValueError("Model not loaded. Train or load a model first.")
            
        results = self.model(image_path, conf=confidence)
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class': self.class_names[class_id],
                        'class_id': class_id
                    })
        
        return detections, results[0].plot()

# Initialize detector
detector = MicroplasticDetector()
