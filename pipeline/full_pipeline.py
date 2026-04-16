"""
Full Pipeline for IceIQ Hockey Analysis
Integrates all components for complete video analysis
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple, Any
import time

from .detection import PlayerDetector
from .tracking import PlayerTracker
from .team_classification import TeamClassifier
from .roi import ROIProcessor
from .homography import HomographyProcessor
from .kinematics import KinematicsProcessor
from .smart_jersey_clustering import SmartJerseyClustering
from .smart_jersey_labeling import SmartJerseyLabeling
from .puck_detection import PuckDetector
from .puck_possession import PuckPossessionAnalyzer
from .event_detection import EventDetector
from .scoreboard_ocr import ScoreboardOCR
from .team_analytics import TeamAnalytics
from .heatmap_generation import HeatmapGenerator
from .fatigue_analysis import FatigueAnalyzer

logger = logging.getLogger(__name__)

class FullPipeline:
    """Complete pipeline for hockey video analysis"""
    
    def __init__(self, video_path: str = None, device: str = 'mps'):
        """
        Initialize the full pipeline
        
        Args:
            video_path: Path to video file (optional, can be set later)
            device: Device for computation
        """
        self.video_path = video_path
        self.device = device
        
        # Initialize components
        self.detector = PlayerDetector(device=device)
        self.tracker = PlayerTracker()
        self.team_classifier = TeamClassifier(device=device)
        self.roi_processor = ROIProcessor()
        self.homography = HomographyProcessor()
        self.kinematics = KinematicsProcessor()
        self.jersey_clustering = SmartJerseyClustering()
        self.jersey_labeling = SmartJerseyLabeling()
        self.puck_detector = PuckDetector(device=device)
        self.puck_possession = PuckPossessionAnalyzer()
        self.event_detector = EventDetector()
        self.scoreboard_ocr = ScoreboardOCR()
        self.team_analytics = TeamAnalytics()
        self.heatmap_generator = HeatmapGenerator()
        self.fatigue_analyzer = FatigueAnalyzer()
        
        # Analysis state
        self.frame_data = []
        self.player_tracks = {}
        self.puck_tracks = []
        self.events = []
        self.homography_matrix = None
        
        logger.info(f"FullPipeline initialized with device: {device}")
    
    def process_video(self, video_path: str = None, output_dir: str = "output") -> Dict[str, Any]:
        """
        Process a complete video through the pipeline
        
        Args:
            video_path: Path to video file
            output_dir: Directory for outputs
            
        Returns:
            Complete analysis results
        """
        if video_path:
            self.video_path = video_path
        
        if not self.video_path:
            raise ValueError("No video path provided")
        
        logger.info(f"Starting full pipeline processing: {self.video_path}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Process frames
        frame_count = 0
        start_time = time.time()
        
        # Setup homography on first frame
        ret, first_frame = cap.read()
        if ret:
            try:
                self.homography_matrix = self.homography.detect_and_setup(first_frame)
                logger.info("Homography matrix computed successfully")
            except Exception as e:
                logger.warning(f"Homography setup failed: {e}")
                self.homography_matrix = None
            
            # Reset to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Process single frame
                frame_result = self.process_frame(frame, frame_count)
                self.frame_data.append(frame_result)
                
                frame_count += 1
                
                # Progress logging
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps_processed = frame_count / elapsed if elapsed > 0 else 0
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) - {fps_processed:.1f} fps")
                
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                continue
        
        cap.release()
        
        # Post-process results
        results = self.post_process_analysis()
        
        # Save results
        self.save_results(results, output_dir)
        
        processing_time = time.time() - start_time
        logger.info(f"Pipeline completed in {processing_time:.2f}s - {frame_count/processing_time:.1f} fps average")
        
        return results
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Process a single frame through all components"""
        
        # Player detection
        detections = self.detector.detect(frame)
        
        # ROI processing
        roi_data = self.roi_processor.process_frame(frame, detections)
        
        # Team classification
        team_data = self.team_classifier.classify_teams(frame, detections)
        
        # Tracking
        tracks = self.tracker.update(detections)
        
        # Update player tracks
        for track in tracks:
            track_id = track['track_id']
            if track_id not in self.player_tracks:
                self.player_tracks[track_id] = []
            
            self.player_tracks[track_id].append({
                'frame': frame_idx,
                'bbox': track['bbox'],
                'team': team_data.get(track_id, {}).get('team', 'unknown'),
                'confidence': track.get('confidence', 0.0)
            })
        
        # Puck detection
        puck_detections = self.puck_detector.detect(frame)
        if puck_detections:
            self.puck_tracks.extend([{
                'frame': frame_idx,
                'bbox': det['bbox'],
                'confidence': det['confidence']
            } for det in puck_detections])
        
        # Kinematics (if homography available)
        kinematics_data = {}
        if self.homography_matrix is not None:
            try:
                kinematics_data = self.kinematics.analyze_frame(
                    tracks, frame_idx, self.homography_matrix
                )
            except Exception as e:
                logger.debug(f"Kinematics analysis failed for frame {frame_idx}: {e}")
        
        # Event detection
        events = self.event_detector.detect_events(frame, tracks, puck_detections)
        self.events.extend([{**event, 'frame': frame_idx} for event in events])
        
        return {
            'frame': frame_idx,
            'detections': detections,
            'tracks': tracks,
            'team_data': team_data,
            'roi_data': roi_data,
            'puck_detections': puck_detections,
            'kinematics': kinematics_data,
            'events': events
        }
    
    def post_process_analysis(self) -> Dict[str, Any]:
        """Post-process all collected data"""
        
        logger.info("Starting post-processing analysis...")
        
        # Jersey clustering and labeling
        cluster_results = {}
        label_results = {}
        
        if self.player_tracks:
            try:
                cluster_results = self.jersey_clustering.cluster_players(self.player_tracks)
                label_results = self.jersey_labeling.label_players(
                    self.player_tracks, cluster_results
                )
                logger.info("Jersey analysis completed")
            except Exception as e:
                logger.error(f"Jersey analysis failed: {e}")
        
        # Puck possession analysis
        possession_results = {}
        if self.puck_tracks and self.player_tracks:
            try:
                possession_results = self.puck_possession.analyze_possession(
                    self.puck_tracks, self.player_tracks
                )
                logger.info("Puck possession analysis completed")
            except Exception as e:
                logger.error(f"Puck possession analysis failed: {e}")
        
        # Team analytics
        team_analytics = {}
        try:
            team_analytics = self.team_analytics.generate_analytics(
                self.player_tracks, possession_results, self.events
            )
            logger.info("Team analytics completed")
        except Exception as e:
            logger.error(f"Team analytics failed: {e}")
        
        # Generate heatmaps
        heatmaps = {}
        if self.player_tracks and self.homography_matrix is not None:
            try:
                heatmaps = self.heatmap_generator.generate_team_heatmaps(
                    self.player_tracks, self.homography_matrix
                )
                logger.info("Heatmap generation completed")
            except Exception as e:
                logger.error(f"Heatmap generation failed: {e}")
        
        # Fatigue analysis
        fatigue_results = {}
        if self.player_tracks:
            try:
                fatigue_results = self.fatigue_analyzer.analyze_fatigue(
                    self.player_tracks, self.events
                )
                logger.info("Fatigue analysis completed")
            except Exception as e:
                logger.error(f"Fatigue analysis failed: {e}")
        
        return {
            'summary': {
                'total_frames': len(self.frame_data),
                'total_players': len(self.player_tracks),
                'total_events': len(self.events),
                'puck_detections': len(self.puck_tracks)
            },
            'player_tracks': self.player_tracks,
            'puck_tracks': self.puck_tracks,
            'events': self.events,
            'jersey_clusters': cluster_results,
            'jersey_labels': label_results,
            'possession': possession_results,
            'team_analytics': team_analytics,
            'heatmaps': heatmaps,
            'fatigue': fatigue_results,
            'frame_data': self.frame_data
        }
    
    def save_results(self, results: Dict[str, Any], output_dir: str):
        """Save analysis results"""
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save main results (excluding frame data for size)
        main_results = {k: v for k, v in results.items() if k != 'frame_data'}
        
        with open(output_path / 'analysis_results.json', 'w') as f:
            json.dump(main_results, f, indent=2, default=str)
        
        # Save heatmaps if generated
        if 'heatmaps' in results and results['heatmaps']:
            heatmap_dir = output_path / 'heatmaps'
            heatmap_dir.mkdir(exist_ok=True)
            
            for team, heatmap in results['heatmaps'].items():
                if isinstance(heatmap, np.ndarray):
                    cv2.imwrite(str(heatmap_dir / f'{team}_heatmap.png'), heatmap)
        
        logger.info(f"Results saved to {output_dir}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics"""
        if not self.frame_data:
            return {}
        
        return {
            'total_frames': len(self.frame_data),
            'total_detections': sum(len(f.get('detections', [])) for f in self.frame_data),
            'total_tracks': len(self.player_tracks),
            'total_events': len(self.events),
            'avg_detections_per_frame': sum(len(f.get('detections', [])) for f in self.frame_data) / len(self.frame_data),
            'frames_with_puck': sum(1 for f in self.frame_data if f.get('puck_detections')),
        }
