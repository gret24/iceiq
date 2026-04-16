#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.abspath('.'))

import cv2
import yaml
from pipeline.detection_module import DetectionModule

def test_detection_module():
    # Load config
    with open('configs/pipeline_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create detection module
    detector = DetectionModule(config)
    
    # Load test video
    cap = cv2.VideoCapture('data/videos/game1.mp4')
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Failed to read video frame")
        return
    
    print(f"Frame shape: {frame.shape}")
    
    # Run detection
    try:
        detections = detector.detect(frame)
        print(f"Detections: {len(detections)}")
        for i, det in enumerate(detections[:5]):  # Show first 5
            print(f"  Detection {i}: bbox={det['bbox']}, conf={det['confidence']:.3f}, team={det.get('team', 'unknown')}, jersey={det.get('jersey_number', 'unknown')}")
        
        print("Detection module test passed!")
        
    except Exception as e:
        print(f"Detection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_detection_module()
