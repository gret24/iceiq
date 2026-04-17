"""
SmartJersey Clustering - Advanced jersey number recognition through clustering
"""

import torch
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, Counter
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

class SmartJerseyClustering:
    """Advanced jersey clustering using multiple features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
        
        # Clustering parameters
        self.min_samples = config.get('min_samples', 3)
        self.eps = config.get('eps', 0.3)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        
        # Feature weights
        self.feature_weights = {
            'color': config.get('color_weight', 0.3),
            'shape': config.get('shape_weight', 0.2),
            'texture': config.get('texture_weight', 0.2),
            'position': config.get('position_weight', 0.1),
            'size': config.get('size_weight', 0.1),
            'temporal': config.get('temporal_weight', 0.1)
        }
        
        # Cluster storage
        self.player_clusters = {}
        self.jersey_assignments = {}
        self.cluster_history = defaultdict(list)
        
    def extract_jersey_features(self, jersey_crop: np.ndarray, bbox: List[float], 
                              frame_idx: int) -> torch.Tensor:
        """Extract comprehensive features from jersey crop"""
        features = []
        
        try:
            # Color features (HSV histogram)
            hsv = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256])
            
            color_features = np.concatenate([h_hist.flatten(), s_hist.flatten(), v_hist.flatten()])
            color_features = color_features / (np.sum(color_features) + 1e-8)  # Normalize
            features.extend(color_features)
            
            # Shape features (aspect ratio, area)
            h, w = jersey_crop.shape[:2]
            aspect_ratio = w / (h + 1e-8)
            area = h * w
            normalized_area = area / (224 * 224)  # Normalize to standard size
            features.extend([aspect_ratio, normalized_area])
            
            # Texture features (LBP approximation using gradients)
            gray = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            texture_features = [
                np.mean(gradient_mag),
                np.std(gradient_mag),
                np.mean(gray),
                np.std(gray)
            ]
            features.extend(texture_features)
            
            # Position features (normalized bbox center)
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2 / 1920  # Normalize to image width
            center_y = (y1 + y2) / 2 / 1080  # Normalize to image height
            features.extend([center_x, center_y])
            
            # Size features (normalized bbox dimensions)
            width = (x2 - x1) / 1920
            height = (y2 - y1) / 1080
            features.extend([width, height])
            
            # Temporal features (frame-based)
            normalized_frame = frame_idx / 1000.0  # Normalize frame index
            features.append(normalized_frame)
            
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            # Return zero features if extraction fails
            features = [0.0] * 50  # Fallback feature size
            
        return torch.tensor(features, dtype=torch.float32, device=self.device)
    
    def cluster_jerseys(self, jersey_data: List[Dict[str, Any]]) -> Dict[int, List[int]]:
        """Cluster jersey instances by similarity"""
        if not jersey_data:
            return {}
            
        # Extract features for all jersey instances
        features_list = []
        valid_indices = []
        
        for i, data in enumerate(jersey_data):
            try:
                jersey_crop = data.get('jersey_crop')
                bbox = data.get('bbox', [0, 0, 100, 100])
                frame_idx = data.get('frame_idx', 0)
                
                if jersey_crop is not None and jersey_crop.size > 0:
                    features = self.extract_jersey_features(jersey_crop, bbox, frame_idx)
                    features_list.append(features.cpu().numpy())
                    valid_indices.append(i)
                else:
                    logger.warning(f"Invalid jersey crop at index {i}")
                    
            except Exception as e:
                logger.warning(f"Failed to extract features for jersey {i}: {e}")
                continue
        
        if len(features_list) < 2:
            logger.warning("Not enough valid jersey instances for clustering")
            return {0: valid_indices} if valid_indices else {}
        
        # Convert to numpy array and normalize
        features_array = np.array(features_list)
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_array)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        cluster_labels = clustering.fit_predict(features_normalized)
        
        # Organize clusters
        clusters = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_labels):
            original_idx = valid_indices[idx]
            if cluster_id >= 0:  # Valid cluster (not noise)
                clusters[cluster_id].append(original_idx)
            else:
                # Assign noise points to individual clusters
                clusters[f"noise_{original_idx}"] = [original_idx]
        
        logger.info(f"Clustered {len(features_list)} jerseys into {len(clusters)} clusters")
        return dict(clusters)
    
    def assign_jersey_numbers(self, clusters: Dict[int, List[int]], 
                            jersey_data: List[Dict[str, Any]]) -> Dict[int, str]:
        """Assign jersey numbers to clusters based on OCR confidence"""
        assignments = {}
        
        for cluster_id, instance_indices in clusters.items():
            # Collect OCR results for this cluster
            ocr_results = []
            confidences = []
            
            for idx in instance_indices:
                data = jersey_data[idx]
                ocr_text = data.get('ocr_text', '')
                ocr_conf = data.get('ocr_confidence', 0.0)
                
                if ocr_text and ocr_text.isdigit():
                    ocr_results.append(ocr_text)
                    confidences.append(ocr_conf)
            
            if ocr_results:
                # Find most common number with highest average confidence
                number_confidence = defaultdict(list)
                for num, conf in zip(ocr_results, confidences):
                    number_confidence[num].append(conf)
                
                best_number = None
                best_score = 0
                
                for number, conf_list in number_confidence.items():
                    avg_confidence = np.mean(conf_list)
                    frequency = len(conf_list)
                    # Score combines confidence and frequency
                    score = avg_confidence * (1 + 0.1 * frequency)
                    
                    if score > best_score and avg_confidence > self.confidence_threshold:
                        best_score = score
                        best_number = number
                
                if best_number:
                    assignments[cluster_id] = best_number
                    logger.info(f"Cluster {cluster_id}: assigned number {best_number} "
                              f"(confidence: {best_score:.3f})")
                else:
                    assignments[cluster_id] = "unknown"
            else:
                assignments[cluster_id] = "unknown"
        
        return assignments
    
    def process_frame_jerseys(self, jersey_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process jerseys for a single frame"""
        try:
            # Cluster similar jerseys
            clusters = self.cluster_jerseys(jersey_data)
            
            # Assign jersey numbers to clusters
            assignments = self.assign_jersey_numbers(clusters, jersey_data)
            
            # Create player assignments
            player_jerseys = {}
            for cluster_id, jersey_number in assignments.items():
                instance_indices = clusters.get(cluster_id, [])
                for idx in instance_indices:
                    track_id = jersey_data[idx].get('track_id')
                    if track_id is not None:
                        player_jerseys[track_id] = {
                            'jersey_number': jersey_number,
                            'cluster_id': cluster_id,
                            'confidence': jersey_data[idx].get('ocr_confidence', 0.0)
                        }
            
            return {
                'clusters': clusters,
                'assignments': assignments,
                'player_jerseys': player_jerseys
            }
            
        except Exception as e:
            logger.error(f"Error processing frame jerseys: {e}")
            return {
                'clusters': {},
                'assignments': {},
                'player_jerseys': {}
            }
    
    def update_player_history(self, track_id: int, jersey_info: Dict[str, Any]):
        """Update historical jersey information for a player"""
        if track_id not in self.cluster_history:
            self.cluster_history[track_id] = []
        
        self.cluster_history[track_id].append({
            'jersey_number': jersey_info.get('jersey_number'),
            'confidence': jersey_info.get('confidence', 0.0),
            'cluster_id': jersey_info.get('cluster_id')
        })
        
        # Keep only recent history (last 30 detections)
        if len(self.cluster_history[track_id]) > 30:
            self.cluster_history[track_id] = self.cluster_history[track_id][-30:]
    
    def get_stable_jersey_number(self, track_id: int) -> Optional[str]:
        """Get the most stable jersey number for a player based on history"""
        if track_id not in self.cluster_history:
            return None
        
        history = self.cluster_history[track_id]
        if not history:
            return None
        
        # Count occurrences of each jersey number
        number_counts = Counter()
        confidence_sums = defaultdict(float)
        
        for entry in history:
            jersey_num = entry.get('jersey_number')
            confidence = entry.get('confidence', 0.0)
            
            if jersey_num and jersey_num != "unknown":
                number_counts[jersey_num] += 1
                confidence_sums[jersey_num] += confidence
        
        if not number_counts:
            return None
        
        # Find the most frequent number with good confidence
        best_number = None
        best_score = 0
        
        for number, count in number_counts.items():
            avg_confidence = confidence_sums[number] / count
            # Score combines frequency and confidence
            score = count * avg_confidence
            
            if score > best_score and avg_confidence > self.confidence_threshold:
                best_score = score
                best_number = number
        
        return best_number
    
    def get_clustering_stats(self) -> Dict[str, Any]:
        """Get clustering statistics"""
        stats = {
            'total_players': len(self.cluster_history),
            'identified_players': 0,
            'identification_rate': 0.0
        }
        
        identified = 0
        for track_id in self.cluster_history:
            jersey_num = self.get_stable_jersey_number(track_id)
            if jersey_num and jersey_num != "unknown":
                identified += 1
        
        stats['identified_players'] = identified
        if stats['total_players'] > 0:
            stats['identification_rate'] = identified / stats['total_players']
        
        return stats
