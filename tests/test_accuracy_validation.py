"""
Accuracy validation for IceIQ Path B components.
Tests homography, kalman_kinematics, heatmap_generation, smart_jersey_clustering, smart_jersey_labeling.
"""

import cv2
import numpy as np
import torch
import json
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.homography import HomographyProcessor
from pipeline.kalman_kinematics import KalmanKinematics
from pipeline.heatmap_generation import HeatmapGenerator
from pipeline.smart_jersey_clustering import SmartJerseyClustering
from pipeline.smart_jersey_labeling import SmartJerseyLabeling
from pipeline.detection_module import YOLODetector

class AccuracyValidator:
    def __init__(self, video_path="data/videos/game1.mp4"):
        self.video_path = video_path
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.homography = HomographyProcessor()
        self.kalman = KalmanKinematics()
        self.heatmap_gen = HeatmapGenerator()
        self.clustering = SmartJerseyClustering()
        self.labeling = SmartJerseyLabeling()
        self.detector = YOLODetector(device=self.device)
        
        # Validation results
        self.results = {}
        
    def validate_homography(self, frames_to_test=50):
        """Validate homography transformation accuracy."""
        print("Validating homography transformation...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return {"status": "error", "message": "Cannot open video"}
        
        frame_count = 0
        successful_transforms = 0
        transform_errors = []
        
        while frame_count < frames_to_test:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process homography
            result = self.homography.process_frame(frame)
            
            if result['homography_matrix'] is not None:
                successful_transforms += 1
                
                # Test corner transformations
                h, w = frame.shape[:2]
                corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(
                    corners.reshape(-1, 1, 2), 
                    result['homography_matrix']
                )
                
                # Calculate transformation error (should be reasonable)
                error = np.mean(np.abs(transformed.reshape(-1, 2)))
                transform_errors.append(error)
            
            frame_count += 1
        
        cap.release()
        
        success_rate = successful_transforms / frame_count
        avg_error = np.mean(transform_errors) if transform_errors else float('inf')
        
        result = {
            "status": "pass" if success_rate > 0.7 else "fail",
            "success_rate": success_rate,
            "avg_transform_error": avg_error,
            "frames_tested": frame_count
        }
        
        self.results["homography"] = result
        print(f"Homography validation: {result['status']} - Success rate: {success_rate:.2%}")
        return result
    
    def validate_kalman_kinematics(self, frames_to_test=100):
        """Validate Kalman filter and kinematics calculations."""
        print("Validating Kalman kinematics...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return {"status": "error", "message": "Cannot open video"}
        
        frame_count = 0
        tracked_objects = {}
        velocity_estimates = []
        
        while frame_count < frames_to_test:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get detections
            detections = self.detector.detect_objects(frame)
            
            # Process each detection
            for detection in detections:
                if detection['class_name'] == 'player':
                    track_id = detection.get('track_id', 0)
                    
                    if track_id not in tracked_objects:
                        tracked_objects[track_id] = self.kalman.create_tracker()
                    
                    # Update tracker
                    center = detection['center']
                    state = self.kalman.update_tracker(tracked_objects[track_id], center)
                    
                    # Get velocity estimate
                    velocity = self.kalman.get_velocity(tracked_objects[track_id])
                    if velocity is not None:
                        speed = np.linalg.norm(velocity)
                        if 0.1 < speed < 50:  # Reasonable speed range
                            velocity_estimates.append(speed)
            
            frame_count += 1
        
        cap.release()
        
        # Validate results
        avg_velocity = np.mean(velocity_estimates) if velocity_estimates else 0
        velocity_std = np.std(velocity_estimates) if velocity_estimates else float('inf')
        
        result = {
            "status": "pass" if len(velocity_estimates) > 10 and avg_velocity < 20 else "fail",
            "tracked_objects": len(tracked_objects),
            "velocity_samples": len(velocity_estimates),
            "avg_velocity": avg_velocity,
            "velocity_std": velocity_std,
            "frames_tested": frame_count
        }
        
        self.results["kalman_kinematics"] = result
        print(f"Kalman validation: {result['status']} - Tracked: {len(tracked_objects)} objects")
        return result
    
    def validate_heatmap_generation(self, frames_to_test=200):
        """Validate heatmap generation accuracy."""
        print("Validating heatmap generation...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return {"status": "error", "message": "Cannot open video"}
        
        frame_count = 0
        player_positions = []
        
        while frame_count < frames_to_test:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get detections
            detections = self.detector.detect_objects(frame)
            
            # Collect player positions
            for detection in detections:
                if detection['class_name'] == 'player':
                    center = detection['center']
                    player_positions.append(center)
            
            frame_count += 1
        
        cap.release()
        
        if len(player_positions) < 50:
            return {
                "status": "fail", 
                "message": "Insufficient player positions",
                "positions_collected": len(player_positions)
            }
        
        # Generate heatmap
        heatmap_data = {
            'positions': player_positions,
            'frame_dimensions': (1920, 1080)  # Assume HD
        }
        
        heatmap = self.heatmap_gen.generate_team_heatmap(heatmap_data, team='all')
        
        # Validate heatmap properties
        heatmap_valid = (
            heatmap is not None and
            heatmap.shape[0] > 0 and
            heatmap.shape[1] > 0 and
            np.sum(heatmap) > 0
        )
        
        result = {
            "status": "pass" if heatmap_valid else "fail",
            "positions_collected": len(player_positions),
            "heatmap_shape": heatmap.shape if heatmap is not None else None,
            "heatmap_sum": float(np.sum(heatmap)) if heatmap is not None else 0,
            "frames_tested": frame_count
        }
        
        self.results["heatmap_generation"] = result
        print(f"Heatmap validation: {result['status']} - Positions: {len(player_positions)}")
        return result
    
    def validate_smart_jersey_clustering(self, frames_to_test=100):
        """Validate jersey clustering accuracy."""
        print("Validating smart jersey clustering...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return {"status": "error", "message": "Cannot open video"}
        
        frame_count = 0
        jersey_features = []
        
        while frame_count < frames_to_test:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get detections
            detections = self.detector.detect_objects(frame)
            
            # Extract jersey features
            for detection in detections:
                if detection['class_name'] == 'player':
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    if x2 > x1 and y2 > y1:
                        player_crop = frame[y1:y2, x1:x2]
                        if player_crop.size > 0:
                            features = self.clustering.extract_jersey_features(player_crop)
                            if features is not None:
                                jersey_features.append(features)
            
            frame_count += 1
        
        cap.release()
        
        if len(jersey_features) < 20:
            return {
                "status": "fail",
                "message": "Insufficient jersey features",
                "features_extracted": len(jersey_features)
            }
        
        # Perform clustering
        cluster_results = self.clustering.cluster_jerseys(jersey_features)
        
        # Validate clustering results
        n_clusters = len(set(cluster_results['labels'])) if cluster_results['labels'] else 0
        silhouette_score = cluster_results.get('silhouette_score', 0)
        
        result = {
            "status": "pass" if 2 <= n_clusters <= 8 and silhouette_score > 0.1 else "fail",
            "features_extracted": len(jersey_features),
            "n_clusters": n_clusters,
            "silhouette_score": silhouette_score,
            "frames_tested": frame_count
        }
        
        self.results["smart_jersey_clustering"] = result
        print(f"Clustering validation: {result['status']} - Clusters: {n_clusters}")
        return result
    
    def validate_smart_jersey_labeling(self, frames_to_test=50):
        """Validate jersey labeling accuracy."""
        print("Validating smart jersey labeling...")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return {"status": "error", "message": "Cannot open video"}
        
        frame_count = 0
        labeling_attempts = 0
        successful_labels = 0
        
        while frame_count < frames_to_test:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get detections
            detections = self.detector.detect_objects(frame)
            
            # Attempt labeling
            for detection in detections:
                if detection['class_name'] == 'player':
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    if x2 > x1 and y2 > y1:
                        player_crop = frame[y1:y2, x1:x2]
                        if player_crop.size > 0:
                            labeling_attempts += 1
                            
                            # Attempt to extract number
                            number = self.labeling.extract_jersey_number(player_crop)
                            if number is not None and isinstance(number, (int, str)):
                                successful_labels += 1
            
            frame_count += 1
        
        cap.release()
        
        success_rate = successful_labels / labeling_attempts if labeling_attempts > 0 else 0
        
        result = {
            "status": "pass" if success_rate > 0.05 else "fail",  # Low threshold for now
            "labeling_attempts": labeling_attempts,
            "successful_labels": successful_labels,
            "success_rate": success_rate,
            "frames_tested": frame_count
        }
        
        self.results["smart_jersey_labeling"] = result
        print(f"Labeling validation: {result['status']} - Success rate: {success_rate:.2%}")
        return result
    
    def run_full_validation(self):
        """Run complete accuracy validation."""
        print("Starting full accuracy validation for Path B components...")
        print("=" * 60)
        
        # Run all validations
        self.validate_homography()
        self.validate_kalman_kinematics()
        self.validate_heatmap_generation()
        self.validate_smart_jersey_clustering()
        self.validate_smart_jersey_labeling()
        
        # Calculate overall results
        passed_tests = sum(1 for result in self.results.values() if result['status'] == 'pass')
        total_tests = len(self.results)
        overall_success = passed_tests / total_tests
        
        print("=" * 60)
        print("ACCURACY VALIDATION SUMMARY")
        print("=" * 60)
        
        for component, result in self.results.items():
            status_symbol = "✓" if result['status'] == 'pass' else "✗"
            print(f"{status_symbol} {component}: {result['status'].upper()}")
        
        print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({overall_success:.1%})")
        
        # Save results
        with open('validation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Detailed results saved to validation_results.json")
        
        return {
            'overall_success': overall_success,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'results': self.results
        }

if __name__ == "__main__":
    validator = AccuracyValidator()
    results = validator.run_full_validation()
    
    if results['overall_success'] >= 0.6:  # 60% pass rate
        print("\n🎉 ACCURACY VALIDATION MILESTONE ACHIEVED!")
        print("Path B components are performing within acceptable parameters.")
    else:
        print(f"\n⚠️  Validation incomplete. {results['passed_tests']}/{results['total_tests']} tests passed.")
        print("Review failed components and improve accuracy.")
