#!/usr/bin/env python3
"""
Test detection module performance against targets
"""

import cv2
import numpy as np
import time
from collections import defaultdict
from pipeline.detection import PlayerDetector

def evaluate_detection_performance(video_path, num_frames=200):
    """
    Evaluate detection performance metrics
    """
    detector = PlayerDetector()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Metrics tracking
    frame_detections = []
    processing_times = []
    detection_counts = []
    team_assignments = []
    jersey_readings = []
    confidence_scores = []
    
    print("Evaluating detection performance...")
    
    for frame_idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        detections = detector.detect_players(frame)
        processing_time = time.time() - start_time
        
        processing_times.append(processing_time)
        detection_counts.append(len(detections))
        
        frame_data = {
            'frame_idx': frame_idx,
            'detections': detections,
            'processing_time': processing_time
        }
        frame_detections.append(frame_data)
        
        # Collect team and jersey data
        for det in detections:
            team_assignments.append(det['team'])
            if det['jersey_number'] is not None:
                jersey_readings.append(det['jersey_number'])
            confidence_scores.append(det['confidence'])
        
        if frame_idx % 20 == 0:
            print(f"Frame {frame_idx}: {len(detections)} detections, {processing_time:.3f}s")
    
    cap.release()
    
    # Calculate metrics
    metrics = calculate_metrics(frame_detections, processing_times, detection_counts, 
                              team_assignments, jersey_readings, confidence_scores)
    
    return metrics

def calculate_metrics(frame_detections, processing_times, detection_counts, 
                     team_assignments, jersey_readings, confidence_scores):
    """
    Calculate performance metrics
    """
    total_frames = len(frame_detections)
    total_detections = sum(detection_counts)
    
    # Processing performance
    avg_processing_time = np.mean(processing_times)
    avg_fps = 1 / avg_processing_time if avg_processing_time > 0 else 0
    
    # Detection rate (assuming 10-12 players visible on average)
    expected_players_per_frame = 11  # Conservative estimate
    total_expected = expected_players_per_frame * total_frames
    detection_rate = total_detections / total_expected if total_expected > 0 else 0
    
    # Confidence statistics
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    high_conf_detections = sum(1 for conf in confidence_scores if conf > 0.7)
    high_conf_rate = high_conf_detections / len(confidence_scores) if confidence_scores else 0
    
    # Team assignment rate (assuming all players should have team assignment)
    team_assignment_rate = sum(1 for team in team_assignments if team is not None) / len(team_assignments) if team_assignments else 0
    
    # Jersey number reading rate
    jersey_reading_rate = len(jersey_readings) / len(team_assignments) if team_assignments else 0
    
    # Consistency metrics
    detections_per_frame = np.array(detection_counts)
    detection_variance = np.var(detections_per_frame)
    detection_std = np.std(detections_per_frame)
    
    metrics = {
        'total_frames': total_frames,
        'total_detections': total_detections,
        'avg_detections_per_frame': np.mean(detection_counts),
        'detection_rate': detection_rate,
        'avg_processing_time': avg_processing_time,
        'avg_fps': avg_fps,
        'avg_confidence': avg_confidence,
        'high_confidence_rate': high_conf_rate,
        'team_assignment_rate': team_assignment_rate,
        'jersey_reading_rate': jersey_reading_rate,
        'detection_variance': detection_variance,
        'detection_std': detection_std,
        'target_detection_rate': 0.98,
        'target_team_accuracy': 0.98,
        'target_jersey_accuracy': 0.98
    }
    
    return metrics

def print_metrics_report(metrics):
    """
    Print formatted metrics report
    """
    print("\n" + "="*60)
    print("DETECTION PERFORMANCE METRICS")
    print("="*60)
    
    print(f"Frames processed: {metrics['total_frames']}")
    print(f"Total detections: {metrics['total_detections']}")
    print(f"Avg detections/frame: {metrics['avg_detections_per_frame']:.1f}")
    print(f"Detection variance: {metrics['detection_variance']:.2f}")
    print(f"Detection std dev: {metrics['detection_std']:.2f}")
    
    print("\nPERFORMANCE METRICS:")
    print(f"Processing time: {metrics['avg_processing_time']:.3f}s/frame")
    print(f"Processing FPS: {metrics['avg_fps']:.1f}")
    print(f"Average confidence: {metrics['avg_confidence']:.3f}")
    print(f"High confidence rate: {metrics['high_confidence_rate']:.3f}")
    
    print("\nACCURACY METRICS:")
    print(f"Detection rate: {metrics['detection_rate']:.3f} (target: {metrics['target_detection_rate']})")
    print(f"Team assignment rate: {metrics['team_assignment_rate']:.3f} (target: {metrics['target_team_accuracy']})")
    print(f"Jersey reading rate: {metrics['jersey_reading_rate']:.3f} (target: {metrics['target_jersey_accuracy']})")
    
    print("\nTARGET ANALYSIS:")
    detection_pass = metrics['detection_rate'] >= metrics['target_detection_rate']
    team_pass = metrics['team_assignment_rate'] >= metrics['target_team_accuracy']
    jersey_pass = metrics['jersey_reading_rate'] >= metrics['target_jersey_accuracy']
    
    print(f"Detection rate target: {'PASS' if detection_pass else 'FAIL'}")
    print(f"Team accuracy target: {'PASS' if team_pass else 'FAIL'}")
    print(f"Jersey accuracy target: {'PASS' if jersey_pass else 'FAIL'}")
    
    overall_pass = detection_pass and team_pass and jersey_pass
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")
    
    return overall_pass

def main():
    """
    Main evaluation function
    """
    video_path = "data/videos/game1.mp4"
    
    print("Starting detection performance evaluation...")
    
    metrics = evaluate_detection_performance(video_path, num_frames=200)
    
    if metrics is None:
        print("Failed to evaluate performance")
        return
    
    overall_pass = print_metrics_report(metrics)
    
    if not overall_pass:
        print("\nRECOMMENDATIONS:")
        if metrics['detection_rate'] < metrics['target_detection_rate']:
            print("- Lower YOLO confidence threshold to increase detection rate")
            print("- Verify YOLO model is detecting person class properly")
        
        if metrics['team_assignment_rate'] < metrics['target_team_accuracy']:
            print("- Improve team classification model or feature extraction")
            print("- Check jersey color detection algorithm")
        
        if metrics['jersey_reading_rate'] < metrics['target_jersey_accuracy']:
            print("- Improve OCR preprocessing (contrast, denoising)")
            print("- Adjust jersey number region extraction")
            print("- Consider ensemble OCR approach")

if __name__ == "__main__":
    main()
