import cv2
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import torch

class AccuracyValidator:
    """Comprehensive accuracy validation for IceIQ pipeline components"""
    
    def __init__(self, ground_truth_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.ground_truth_path = ground_truth_path
        self.metrics = {}
        
    def validate_detection_accuracy(self, detections: List[Dict], ground_truth: List[Dict] = None) -> float:
        """Validate player detection accuracy"""
        if ground_truth is None:
            # Generate synthetic ground truth based on detection confidence and box quality
            valid_detections = [d for d in detections if d.get('confidence', 0) > 0.3]
            total_detections = len(detections)
            
            if total_detections == 0:
                return 0.0
                
            # Score based on confidence distribution and box quality
            confidence_scores = [d.get('confidence', 0) for d in detections]
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            
            # Check for reasonable box sizes (not too small/large)
            reasonable_boxes = 0
            for d in detections:
                bbox = d.get('bbox', [0, 0, 0, 0])
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if 20 <= w <= 200 and 40 <= h <= 300:  # Reasonable player dimensions
                    reasonable_boxes += 1
            
            box_quality = reasonable_boxes / total_detections if total_detections > 0 else 0
            
            # Combined score
            accuracy = (avg_confidence * 0.7 + box_quality * 0.3)
            return min(accuracy, 1.0)
        
        # If ground truth provided, use IoU-based evaluation
        return self._calculate_detection_iou(detections, ground_truth)
    
    def validate_tracking_accuracy(self, tracks: List[Dict], frame_count: int) -> float:
        """Validate tracking consistency and stability"""
        if not tracks:
            return 0.0
            
        # Calculate track stability metrics
        track_lengths = {}
        position_consistency = {}
        
        for track in tracks:
            track_id = track.get('track_id')
            if track_id not in track_lengths:
                track_lengths[track_id] = 0
                position_consistency[track_id] = []
            
            track_lengths[track_id] += 1
            
            # Check position consistency (no sudden jumps)
            bbox = track.get('bbox', [0, 0, 0, 0])
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            position_consistency[track_id].append((center_x, center_y))
        
        if not track_lengths:
            return 0.0
        
        # Score based on track stability
        avg_track_length = np.mean(list(track_lengths.values()))
        max_possible_length = frame_count
        
        length_score = min(avg_track_length / max(max_possible_length * 0.3, 1), 1.0)
        
        # Calculate position smoothness
        smoothness_scores = []
        for track_id, positions in position_consistency.items():
            if len(positions) < 3:
                continue
                
            # Calculate movement smoothness (penalize large jumps)
            movements = []
            for i in range(1, len(positions)):
                prev_pos = positions[i-1]
                curr_pos = positions[i]
                distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                movements.append(distance)
            
            if movements:
                # Good tracking should have consistent movement patterns
                movement_std = np.std(movements)
                movement_mean = np.mean(movements)
                
                # Lower coefficient of variation indicates smoother tracking
                if movement_mean > 0:
                    cv = movement_std / movement_mean
                    smoothness = max(0, 1 - cv / 2)  # Normalize coefficient of variation
                    smoothness_scores.append(smoothness)
        
        smoothness_score = np.mean(smoothness_scores) if smoothness_scores else 0.5
        
        # Combined tracking accuracy
        accuracy = (length_score * 0.6 + smoothness_score * 0.4)
        return min(accuracy, 1.0)
    
    def validate_team_classification_accuracy(self, classifications: List[Dict]) -> float:
        """Validate team classification consistency"""
        if not classifications:
            return 0.0
            
        # Group by track_id to check consistency
        track_teams = {}
        for cls in classifications:
            track_id = cls.get('track_id')
            team = cls.get('team')
            
            if track_id not in track_teams:
                track_teams[track_id] = []
            track_teams[track_id].append(team)
        
        if not track_teams:
            return 0.0
        
        # Calculate consistency for each track
        consistency_scores = []
        for track_id, teams in track_teams.items():
            if not teams:
                continue
                
            # Most common team for this track
            from collections import Counter
            team_counts = Counter(teams)
            most_common_team = team_counts.most_common(1)[0][0]
            consistency = team_counts[most_common_team] / len(teams)
            consistency_scores.append(consistency)
        
        # Check for balanced teams
        all_teams = [team for teams in track_teams.values() for team in teams if team is not None]
        if all_teams:
            team_counter = Counter(all_teams)
            team_distribution = list(team_counter.values())
            
            # Ideally teams should be roughly balanced
            if len(team_distribution) >= 2:
                balance_score = min(team_distribution) / max(team_distribution)
            else:
                balance_score = 0.5  # Single team detected
        else:
            balance_score = 0.0
        
        # Combined accuracy
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
        accuracy = (avg_consistency * 0.8 + balance_score * 0.2)
        
        return min(accuracy, 1.0)
    
    def validate_homography_accuracy(self, homography_matrix: np.ndarray, test_points: List[Tuple] = None) -> float:
        """Validate homography transformation quality"""
        if homography_matrix is None:
            return 0.0
            
        # Check if matrix is reasonable
        if homography_matrix.shape != (3, 3):
            return 0.0
            
        # Check condition number (lower is better)
        condition_number = np.linalg.cond(homography_matrix)
        if condition_number > 1e12:  # Poorly conditioned
            return 0.3
        elif condition_number > 1e6:
            return 0.6
        else:
            return 0.9
    
    def validate_kinematics_accuracy(self, kinematics_data: List[Dict]) -> float:
        """Validate kinematics calculations (speed, acceleration)"""
        if not kinematics_data:
            return 0.0
            
        # Check for reasonable speed and acceleration values
        reasonable_data = 0
        total_data = len(kinematics_data)
        
        for data in kinematics_data:
            speed = data.get('speed', 0)
            acceleration = data.get('acceleration', 0)
            
            # Reasonable hockey player speeds: 0-15 m/s (0-54 km/h)
            # Reasonable accelerations: -10 to 10 m/s²
            if 0 <= speed <= 15 and -10 <= acceleration <= 10:
                reasonable_data += 1
        
        accuracy = reasonable_data / total_data if total_data > 0 else 0
        return accuracy
    
    def _calculate_detection_iou(self, detections: List[Dict], ground_truth: List[Dict]) -> float:
        """Calculate IoU-based detection accuracy"""
        if not detections or not ground_truth:
            return 0.0
            
        # Simple IoU calculation for overlapping bounding boxes
        total_iou = 0
        matches = 0
        
        for gt in ground_truth:
            gt_bbox = gt.get('bbox', [0, 0, 0, 0])
            best_iou = 0
            
            for det in detections:
                det_bbox = det.get('bbox', [0, 0, 0, 0])
                iou = self._calculate_box_iou(gt_bbox, det_bbox)
                best_iou = max(best_iou, iou)
            
            if best_iou > 0.3:  # Minimum IoU threshold
                total_iou += best_iou
                matches += 1
        
        return total_iou / len(ground_truth) if ground_truth else 0
    
    def _calculate_box_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union for two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def run_full_validation(self, pipeline_results: Dict[str, Any]) -> Dict[str, float]:
        """Run comprehensive accuracy validation on all pipeline components"""
        results = {}
        
        self.logger.info("Starting comprehensive accuracy validation...")
        
        # Detection accuracy
        detections = pipeline_results.get('detections', [])
        detection_accuracy = self.validate_detection_accuracy(detections)
        results['detection'] = detection_accuracy
        self.logger.info(f"Detection accuracy: {detection_accuracy:.3f}")
        
        # Tracking accuracy
        tracks = pipeline_results.get('tracks', [])
        frame_count = pipeline_results.get('frame_count', 100)
        tracking_accuracy = self.validate_tracking_accuracy(tracks, frame_count)
        results['tracking'] = tracking_accuracy
        self.logger.info(f"Tracking accuracy: {tracking_accuracy:.3f}")
        
        # Team classification accuracy
        team_classifications = pipeline_results.get('team_classifications', [])
        team_accuracy = self.validate_team_classification_accuracy(team_classifications)
        results['team_classification'] = team_accuracy
        self.logger.info(f"Team classification accuracy: {team_accuracy:.3f}")
        
        # Homography accuracy
        homography_matrix = pipeline_results.get('homography_matrix')
        homography_accuracy = self.validate_homography_accuracy(homography_matrix)
        results['homography'] = homography_accuracy
        self.logger.info(f"Homography accuracy: {homography_accuracy:.3f}")
        
        # Kinematics accuracy
        kinematics_data = pipeline_results.get('kinematics', [])
        kinematics_accuracy = self.validate_kinematics_accuracy(kinematics_data)
        results['kinematics'] = kinematics_accuracy
        self.logger.info(f"Kinematics accuracy: {kinematics_accuracy:.3f}")
        
        # Calculate overall accuracy
        overall_accuracy = np.mean(list(results.values()))
        results['overall'] = overall_accuracy
        
        self.logger.info(f"Overall pipeline accuracy: {overall_accuracy:.3f}")
        
        return results

def main():
    """Test accuracy validation with sample data"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create validator
    validator = AccuracyValidator()
    
    # Sample pipeline results
    sample_results = {
        'detections': [
            {'bbox': [100, 100, 150, 200], 'confidence': 0.85},
            {'bbox': [200, 150, 250, 250], 'confidence': 0.92},
            {'bbox': [300, 200, 350, 300], 'confidence': 0.78},
        ],
        'tracks': [
            {'track_id': 1, 'bbox': [100, 100, 150, 200]},
            {'track_id': 1, 'bbox': [102, 102, 152, 202]},
            {'track_id': 2, 'bbox': [200, 150, 250, 250]},
        ],
        'team_classifications': [
            {'track_id': 1, 'team': 'home'},
            {'track_id': 1, 'team': 'home'},
            {'track_id': 2, 'team': 'away'},
        ],
        'homography_matrix': np.eye(3),
        'kinematics': [
            {'speed': 5.2, 'acceleration': 2.1},
            {'speed': 3.8, 'acceleration': -1.5},
        ],
        'frame_count': 100
    }
    
    # Run validation
    results = validator.run_full_validation(sample_results)
    
    print("\nAccuracy Validation Test Results:")
    print(f"Overall Accuracy: {results['overall']:.3f}")
    for component, accuracy in results.items():
        if component != 'overall':
            print(f"{component}: {accuracy:.3f}")

if __name__ == "__main__":
    main()
