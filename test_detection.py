#!/usr/bin/env python3
"""
Test detection module functionality
"""
import cv2
import torch
import numpy as np
from pipeline.detection_module import DetectionModule

def test_detection():
    """Test detection module on sample video"""
    
    # Initialize detection module
    detector = DetectionModule()
    
    # Load video
    cap = cv2.VideoCapture('data/videos/game1.mp4')
    if not cap.isOpened():
        print("ERROR: Cannot open video")
        return
    
    print(f"Video: {cap.get(cv2.CAP_PROP_FRAME_COUNT)} frames")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    frame_count = 0
    detection_count = 0
    
    # Process first 100 frames for testing
    while frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Run detection
        results = detector.detect_frame(frame)
        
        if results:
            detection_count += 1
            print(f"Frame {frame_count}: {len(results)} detections")
            
            # Show first few detections
            if frame_count <= 5:
                for i, det in enumerate(results[:3]):
                    print(f"  Det {i}: conf={det['confidence']:.3f} bbox={det['bbox']}")
        
        if frame_count % 20 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    
    # Calculate detection rate
    detection_rate = detection_count / frame_count if frame_count > 0 else 0
    
    print(f"\nRESULTS:")
    print(f"Frames processed: {frame_count}")
    print(f"Frames with detections: {detection_count}")
    print(f"Detection rate: {detection_rate:.3f}")
    print(f"Target detection rate: 0.98")
    print(f"PASS: {detection_rate >= 0.98}")
    
    return detection_rate >= 0.98

if __name__ == "__main__":
    success = test_detection()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
