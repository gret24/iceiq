#!/usr/bin/env python3
"""
Debug detection issues - check what YOLO model actually detects
"""

import cv2
import torch
from ultralytics import YOLO
import numpy as np

def debug_detection():
    """Debug YOLO detection on sample frame"""
    
    # Load model
    model = YOLO('yolov8n.pt')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)
    
    # Load video
    cap = cv2.VideoCapture('data/videos/game1.mp4')
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read frame")
        cap.release()
        return
    
    print(f"Frame shape: {frame.shape}")
    
    # Test different confidence thresholds
    thresholds = [0.1, 0.25, 0.5, 0.7]
    
    for conf_thresh in thresholds:
        print(f"\n=== Testing confidence threshold: {conf_thresh} ===")
        
        # Run detection
        results = model(frame, conf=conf_thresh, device=device)
        
        if len(results) == 0:
            print("No results returned")
            continue
            
        result = results[0]
        
        if result.boxes is None:
            print("No boxes detected")
            continue
            
        boxes = result.boxes.cpu()
        print(f"Total detections: {len(boxes)}")
        
        # Group by class
        class_counts = {}
        for box in boxes:
            cls_id = int(box.cls)
            cls_name = model.names[cls_id]
            conf = float(box.conf)
            
            if cls_name not in class_counts:
                class_counts[cls_name] = []
            class_counts[cls_name].append(conf)
        
        # Print class summary
        for cls_name, confs in class_counts.items():
            print(f"  {cls_name}: {len(confs)} detections, conf range: {min(confs):.3f}-{max(confs):.3f}")
    
    cap.release()
    
    # Print all available classes
    print(f"\n=== Available YOLO classes ===")
    for i, name in model.names.items():
        print(f"  {i}: {name}")

if __name__ == "__main__":
    debug_detection()
