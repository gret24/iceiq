#!/usr/bin/env python3
"""
Test script for Pipeline C - Full Hockey Analytics Pipeline
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))

import cv2
import numpy as np
import torch
from full_pipeline import HockeyPipeline

def test_pipeline_c():
    """Test the complete Pipeline C functionality"""
    
    # Initialize pipeline
    print("Initializing Hockey Pipeline...")
    pipeline = HockeyPipeline()
    
    # Test video path
    video_path = "data/videos/game1.mp4"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    print(f"Video opened successfully")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Frame count: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # Process first few frames
    frame_count = 0
    max_frames = 30  # Test with 30 frames
    
    print(f"\nProcessing {max_frames} frames...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame through pipeline
        try:
            results = pipeline.process_frame(frame, frame_count)
            
            # Print results summary every 10 frames
            if frame_count % 10 == 0:
                print(f"Frame {frame_count}:")
                if 'players' in results:
                    print(f"  - Players detected: {len(results['players'])}")
                if 'puck' in results:
                    print(f"  - Puck detected: {results['puck'] is not None}")
                if 'events' in results:
                    print(f"  - Events: {len(results['events'])}")
                if 'heatmaps' in results:
                    print(f"  - Heatmaps generated: {len(results['heatmaps'])}")
                    
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            cap.release()
            return False
            
        frame_count += 1
    
    cap.release()
    
    # Generate final reports
    print("\nGenerating final analytics...")
    try:
        final_analytics = pipeline.get_final_analytics()
        
        print("Final Analytics Summary:")
        for category, data in final_analytics.items():
            if isinstance(data, dict):
                print(f"  {category}: {len(data)} items")
            elif isinstance(data, list):
                print(f"  {category}: {len(data)} items")
            else:
                print(f"  {category}: {type(data)}")
                
    except Exception as e:
        print(f"Error generating final analytics: {e}")
        return False
    
    print(f"\nPipeline C test completed successfully!")
    print(f"Processed {frame_count} frames")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("IceIQ Pipeline C Test")
    print("=" * 60)
    
    success = test_pipeline_c()
    
    if success:
        print("\n✅ Pipeline C test PASSED")
        sys.exit(0)
    else:
        print("\n❌ Pipeline C test FAILED")
        sys.exit(1)
