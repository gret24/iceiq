"""
Kalman Filter for Player Kinematics
Tracks position, velocity, and acceleration with noise filtering
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class KinematicState:
    """Player kinematic state"""
    position: Tuple[float, float]
    velocity: Tuple[float, float] 
    acceleration: Tuple[float, float]
    speed: float
    direction: float
    timestamp: float

class KalmanKinematics:
    """Kalman filter for smooth player kinematics tracking"""
    
    def __init__(self, dt: float = 1/30.0):
        """
        Initialize Kalman filter for 2D motion tracking
        
        Args:
            dt: Time step between frames (1/fps)
        """
        self.dt = dt
        self.filters = {}  # track_id -> kalman filter
        
        # State vector: [x, y, vx, vy, ax, ay]
        # 6D state: position, velocity, acceleration
        
    def _create_kalman_filter(self) -> cv2.KalmanFilter:
        """Create a new Kalman filter for player tracking"""
        kf = cv2.KalmanFilter(6, 2)  # 6 state vars, 2 measurements
        
        dt = self.dt
        dt2 = dt * dt / 2
        
        # Transition matrix (constant acceleration model)
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0,  dt2, 0],
            [0, 1, 0,  dt, 0,  dt2],
            [0, 0, 1,  0,  dt,  0],
            [0, 0, 0,  1,  0,  dt],
            [0, 0, 0,  0,  1,   0],
            [0, 0, 0,  0,  0,   1]
        ], dtype=np.float32)
        
        # Measurement matrix (we observe position only)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        process_noise = 0.1
        kf.processNoiseCov = process_noise * np.eye(6, dtype=np.float32)
        
        # Measurement noise covariance
        measurement_noise = 5.0
        kf.measurementNoiseCov = measurement_noise * np.eye(2, dtype=np.float32)
        
        # Error covariance matrix
        kf.errorCovPost = 100.0 * np.eye(6, dtype=np.float32)
        
        return kf
        
    def update(self, detections: List[Dict], frame_time: float) -> Dict[int, KinematicState]:
        """
        Update Kalman filters with new detections
        
        Args:
            detections: List of detection dicts with track_id, bbox
            frame_time: Current frame timestamp
            
        Returns:
            Dictionary of track_id -> KinematicState
        """
        states = {}
        
        for det in detections:
            track_id = det['track_id']
            bbox = det['bbox']
            
            # Get center position
            x = bbox[0] + bbox[2] / 2
            y = bbox[1] + bbox[3] / 2
            position = np.array([x, y], dtype=np.float32)
            
            # Initialize filter if new track
            if track_id not in self.filters:
                kf = self._create_kalman_filter()
                # Initialize state with position, zero velocity/acceleration
                kf.statePre = np.array([x, y, 0, 0, 0, 0], dtype=np.float32)
                kf.statePost = kf.statePre.copy()
                self.filters[track_id] = kf
            
            kf = self.filters[track_id]
            
            # Predict step
            prediction = kf.predict()
            
            # Update step
            kf.correct(position.reshape(-1, 1))
            
            # Extract kinematic state
            state_vec = kf.statePost.flatten()
            pos = (float(state_vec[0]), float(state_vec[1]))
            vel = (float(state_vec[2]), float(state_vec[3]))
            acc = (float(state_vec[4]), float(state_vec[5]))
            
            # Calculate derived metrics
            speed = np.sqrt(vel[0]**2 + vel[1]**2)
            direction = np.arctan2(vel[1], vel[0]) if speed > 0.1 else 0.0
            
            states[track_id] = KinematicState(
                position=pos,
                velocity=vel,
                acceleration=acc,
                speed=speed,
                direction=direction,
                timestamp=frame_time
            )
            
        return states
        
    def predict(self, track_id: int) -> Optional[KinematicState]:
        """
        Predict next state for a track without measurement
        
        Args:
            track_id: ID of track to predict
            
        Returns:
            Predicted kinematic state or None if track not found
        """
        if track_id not in self.filters:
            return None
            
        kf = self.filters[track_id]
        prediction = kf.predict()
        state_vec = prediction.flatten()
        
        pos = (float(state_vec[0]), float(state_vec[1]))
        vel = (float(state_vec[2]), float(state_vec[3]))
        acc = (float(state_vec[4]), float(state_vec[5]))
        
        speed = np.sqrt(vel[0]**2 + vel[1]**2)
        direction = np.arctan2(vel[1], vel[0]) if speed > 0.1 else 0.0
        
        return KinematicState(
            position=pos,
            velocity=vel,
            acceleration=acc,
            speed=speed,
            direction=direction,
            timestamp=0.0  # No timestamp for prediction
        )
        
    def get_trajectory(self, track_id: int, frames: int = 10) -> List[Tuple[float, float]]:
        """
        Predict future trajectory points
        
        Args:
            track_id: ID of track to predict
            frames: Number of future frames to predict
            
        Returns:
            List of predicted (x, y) positions
        """
        if track_id not in self.filters:
            return []
            
        kf = self.filters[track_id]
        current_state = kf.statePost.copy()
        
        trajectory = []
        temp_kf = cv2.KalmanFilter(6, 2)
        temp_kf.transitionMatrix = kf.transitionMatrix.copy()
        temp_kf.statePost = current_state.copy()
        temp_kf.statePre = current_state.copy()
        
        for _ in range(frames):
            prediction = temp_kf.predict()
            pos = (float(prediction[0]), float(prediction[1]))
            trajectory.append(pos)
            temp_kf.statePost = prediction.copy()
            
        return trajectory
        
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """
        Remove filters for tracks that are no longer active
        
        Args:
            active_track_ids: List of currently active track IDs
        """
        inactive_ids = set(self.filters.keys()) - set(active_track_ids)
        for track_id in inactive_ids:
            del self.filters[track_id]
            
    def get_smoothed_metrics(self, states_history: List[Dict[int, KinematicState]], 
                           window_size: int = 5) -> Dict[int, Dict[str, float]]:
        """
        Calculate smoothed kinematic metrics over time window
        
        Args:
            states_history: List of state dictionaries over time
            window_size: Size of smoothing window
            
        Returns:
            Dictionary of track_id -> metrics
        """
        metrics = {}
        
        if len(states_history) < 2:
            return metrics
            
        # Get all track IDs
        all_tracks = set()
        for states in states_history[-window_size:]:
            all_tracks.update(states.keys())
            
        for track_id in all_tracks:
            # Collect recent states for this track
            recent_states = []
            for states in states_history[-window_size:]:
                if track_id in states:
                    recent_states.append(states[track_id])
                    
            if len(recent_states) < 2:
                continue
                
            # Calculate smoothed metrics
            speeds = [s.speed for s in recent_states]
            accelerations = [np.sqrt(s.acceleration[0]**2 + s.acceleration[1]**2) 
                           for s in recent_states]
            
            # Calculate distance traveled
            total_distance = 0
            for i in range(1, len(recent_states)):
                prev_pos = recent_states[i-1].position
                curr_pos = recent_states[i].position
                dist = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                             (curr_pos[1] - prev_pos[1])**2)
                total_distance += dist
            
            # Calculate direction changes (agility indicator)
            direction_changes = 0
            for i in range(1, len(recent_states)):
                prev_dir = recent_states[i-1].direction
                curr_dir = recent_states[i].direction
                angle_diff = abs(curr_dir - prev_dir)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                if angle_diff > np.pi / 4:  # 45 degree threshold
                    direction_changes += 1
            
            metrics[track_id] = {
                'avg_speed': np.mean(speeds),
                'max_speed': np.max(speeds),
                'avg_acceleration': np.mean(accelerations),
                'max_acceleration': np.max(accelerations),
                'distance_traveled': total_distance,
                'direction_changes': direction_changes,
                'agility_score': direction_changes * np.mean(speeds)  # Combined metric
            }
            
        return metrics

def test_kalman_kinematics():
    """Test the KalmanKinematics implementation"""
    print("Testing KalmanKinematics...")
    
    # Initialize
    kk = KalmanKinematics(dt=1/30.0)
    
    # Simulate player movement
    detections_sequence = []
    for frame in range(10):
        # Simulate a player moving diagonally
        x = 100 + frame * 5 + np.random.normal(0, 2)  # Add noise
        y = 200 + frame * 3 + np.random.normal(0, 2)
        
        detections = [{
            'track_id': 1,
            'bbox': [x-10, y-10, 20, 20]  # Small bbox around center
        }]
        detections_sequence.append(detections)
    
    # Process detections
    states_history = []
    for frame, detections in enumerate(detections_sequence):
        states = kk.update(detections, frame / 30.0)
        states_history.append(states)
        
        if 1 in states:
            state = states[1]
            print(f"Frame {frame}: pos=({state.position[0]:.1f},{state.position[1]:.1f}), "
                  f"speed={state.speed:.2f}, direction={state.direction:.2f}")
    
    # Test prediction
    prediction = kk.predict(1)
    if prediction:
        print(f"Next prediction: pos=({prediction.position[0]:.1f},{prediction.position[1]:.1f})")
    
    # Test trajectory
    trajectory = kk.get_trajectory(1, frames=5)
    print(f"Future trajectory: {trajectory[:2]}...")
    
    # Test metrics
    metrics = kk.get_smoothed_metrics(states_history)
    if 1 in metrics:
        print(f"Metrics: {metrics[1]}")
    
    print("KalmanKinematics test completed!")

if __name__ == "__main__":
    test_kalman_kinematics()
