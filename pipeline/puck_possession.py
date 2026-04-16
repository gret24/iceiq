import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import torch
from collections import defaultdict, deque

@dataclass
class PossessionEvent:
    player_id: int
    team_id: int
    start_frame: int
    end_frame: Optional[int] = None
    confidence: float = 0.0
    event_type: str = "possession"  # possession, pass, shot, turnover

class PuckPossessionTracker:
    def __init__(self, 
                 proximity_threshold: float = 50.0,
                 possession_frames: int = 10,
                 confidence_threshold: float = 0.7):
        self.proximity_threshold = proximity_threshold
        self.possession_frames = possession_frames
        self.confidence_threshold = confidence_threshold
        
        # State tracking
        self.current_possessor = None
        self.possession_start_frame = None
        self.possession_history = deque(maxlen=possession_frames)
        self.events = []
        
        # Analysis
        self.possession_stats = defaultdict(lambda: {
            'total_time': 0,
            'possessions': 0,
            'passes_completed': 0,
            'passes_attempted': 0,
            'turnovers': 0
        })
    
    def calculate_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two positions."""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def find_closest_player(self, puck_pos: Tuple[float, float], 
                          players: List[Dict]) -> Tuple[Optional[Dict], float]:
        """Find the player closest to the puck."""
        if not players or puck_pos is None:
            return None, float('inf')
        
        closest_player = None
        min_distance = float('inf')
        
        for player in players:
            if 'bbox' not in player:
                continue
            
            # Get player center position
            x1, y1, x2, y2 = player['bbox']
            player_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            distance = self.calculate_distance(puck_pos, player_center)
            
            if distance < min_distance:
                min_distance = distance
                closest_player = player
        
        return closest_player, min_distance
    
    def analyze_velocity_patterns(self, puck_pos: Tuple[float, float], 
                                players: List[Dict]) -> Dict:
        """Analyze velocity patterns to determine possession likelihood."""
        analysis = {
            'puck_velocity': 0.0,
            'player_velocities': {},
            'direction_alignment': {}
        }
        
        # Simple velocity estimation (would need frame history for accurate calculation)
        for player in players:
            if 'bbox' in player and 'id' in player:
                player_id = player['id']
                analysis['player_velocities'][player_id] = 0.0
                analysis['direction_alignment'][player_id] = 0.0
        
        return analysis
    
    def update(self, frame_idx: int, puck_detection: Optional[Dict], 
               players: List[Dict]) -> List[PossessionEvent]:
        """Update possession tracking for current frame."""
        events_this_frame = []
        
        if puck_detection is None or 'bbox' not in puck_detection:
            # No puck detected, maintain current state but with lower confidence
            if self.current_possessor is not None:
                self.possession_history.append({
                    'frame': frame_idx,
                    'possessor_id': self.current_possessor,
                    'confidence': 0.3
                })
            return events_this_frame
        
        # Get puck position
        px1, py1, px2, py2 = puck_detection['bbox']
        puck_pos = ((px1 + px2) / 2, (py1 + py2) / 2)
        
        # Find closest player
        closest_player, distance = self.find_closest_player(puck_pos, players)
        
        # Determine possession
        possession_confidence = 0.0
        possessor_id = None
        
        if closest_player and distance < self.proximity_threshold:
            possessor_id = closest_player.get('id')
            # Calculate confidence based on distance and other factors
            proximity_score = max(0, 1 - distance / self.proximity_threshold)
            
            # Additional factors for confidence
            velocity_analysis = self.analyze_velocity_patterns(puck_pos, players)
            
            possession_confidence = proximity_score * 0.8  # Base confidence from proximity
            
            # Boost confidence if player has consistent possession history
            recent_possessions = [h for h in list(self.possession_history)[-5:] 
                                if h.get('possessor_id') == possessor_id]
            if len(recent_possessions) >= 3:
                possession_confidence = min(1.0, possession_confidence * 1.2)
        
        # Update possession history
        self.possession_history.append({
            'frame': frame_idx,
            'possessor_id': possessor_id,
            'confidence': possession_confidence,
            'puck_pos': puck_pos
        })
        
        # Determine if possession changed
        if possession_confidence > self.confidence_threshold:
            if self.current_possessor != possessor_id:
                # Possession change detected
                
                # End previous possession
                if self.current_possessor is not None and self.possession_start_frame is not None:
                    end_event = PossessionEvent(
                        player_id=self.current_possessor,
                        team_id=self._get_player_team(self.current_possessor, players),
                        start_frame=self.possession_start_frame,
                        end_frame=frame_idx,
                        confidence=0.8,
                        event_type="possession"
                    )
                    self.events.append(end_event)
                    events_this_frame.append(end_event)
                    
                    # Update stats
                    duration = frame_idx - self.possession_start_frame
                    team_id = self._get_player_team(self.current_possessor, players)
                    if team_id is not None:
                        self.possession_stats[team_id]['total_time'] += duration
                        self.possession_stats[team_id]['possessions'] += 1
                
                # Start new possession
                self.current_possessor = possessor_id
                self.possession_start_frame = frame_idx
                
                if possessor_id is not None:
                    start_event = PossessionEvent(
                        player_id=possessor_id,
                        team_id=self._get_player_team(possessor_id, players),
                        start_frame=frame_idx,
                        confidence=possession_confidence,
                        event_type="possession"
                    )
                    events_this_frame.append(start_event)
        
        return events_this_frame
    
    def _get_player_team(self, player_id: int, players: List[Dict]) -> Optional[int]:
        """Get team ID for a player."""
        for player in players:
            if player.get('id') == player_id:
                return player.get('team_id', 0)
        return None
    
    def get_possession_stats(self) -> Dict:
        """Get comprehensive possession statistics."""
        total_frames = sum(stats['total_time'] for stats in self.possession_stats.values())
        
        if total_frames == 0:
            return {'team_0': {'percentage': 0}, 'team_1': {'percentage': 0}}
        
        result = {}
        for team_id, stats in self.possession_stats.items():
            percentage = (stats['total_time'] / total_frames) * 100
            result[f'team_{team_id}'] = {
                'percentage': percentage,
                'total_possessions': stats['possessions'],
                'avg_possession_time': stats['total_time'] / max(1, stats['possessions']),
                'passes_completed': stats['passes_completed'],
                'passes_attempted': stats['passes_attempted'],
                'pass_accuracy': stats['passes_completed'] / max(1, stats['passes_attempted']) * 100,
                'turnovers': stats['turnovers']
            }
        
        return result
    
    def detect_events(self, frame_idx: int, puck_detection: Optional[Dict], 
                     players: List[Dict]) -> List[PossessionEvent]:
        """Detect specific hockey events like passes, shots, turnovers."""
        events = []
        
        if len(self.possession_history) < 3:
            return events
        
        recent = list(self.possession_history)[-3:]
        
        # Detect potential pass (possession change between teammates)
        if (len(recent) >= 2 and 
            recent[-2]['possessor_id'] != recent[-1]['possessor_id'] and
            recent[-2]['possessor_id'] is not None and 
            recent[-1]['possessor_id'] is not None):
            
            prev_team = self._get_player_team(recent[-2]['possessor_id'], players)
            curr_team = self._get_player_team(recent[-1]['possessor_id'], players)
            
            if prev_team == curr_team and prev_team is not None:
                # Successful pass
                pass_event = PossessionEvent(
                    player_id=recent[-2]['possessor_id'],
                    team_id=prev_team,
                    start_frame=frame_idx,
                    confidence=0.7,
                    event_type="pass"
                )
                events.append(pass_event)
                self.possession_stats[prev_team]['passes_completed'] += 1
                self.possession_stats[prev_team]['passes_attempted'] += 1
            
            elif prev_team != curr_team and prev_team is not None and curr_team is not None:
                # Turnover
                turnover_event = PossessionEvent(
                    player_id=recent[-2]['possessor_id'],
                    team_id=prev_team,
                    start_frame=frame_idx,
                    confidence=0.8,
                    event_type="turnover"
                )
                events.append(turnover_event)
                self.possession_stats[prev_team]['turnovers'] += 1
        
        return events
    
    def visualize_possession(self, frame: np.ndarray, 
                           current_possessor: Optional[int],
                           players: List[Dict]) -> np.ndarray:
        """Visualize current possession on frame."""
        vis_frame = frame.copy()
        
        if current_possessor is None:
            return vis_frame
        
        # Find and highlight current possessor
        for player in players:
            if player.get('id') == current_possessor and 'bbox' in player:
                x1, y1, x2, y2 = player['bbox']
                
                # Draw thick border around possessor
                cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), 
                            (0, 255, 0), 4)
                
                # Add possession indicator
                cv2.putText(vis_frame, f"PUCK CARRIER #{current_possessor}", 
                          (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.8, (0, 255, 0), 2)
                break
        
        return vis_frame

def test_puck_possession():
    """Test puck possession tracking."""
    print("Testing puck possession tracker...")
    
    tracker = PuckPossessionTracker()
    
    # Mock data
    mock_players = [
        {'id': 1, 'bbox': [100, 100, 150, 200], 'team_id': 0},
        {'id': 2, 'bbox': [300, 150, 350, 250], 'team_id': 1},
        {'id': 3, 'bbox': [200, 200, 250, 300], 'team_id': 0}
    ]
    
    mock_puck = {'bbox': [110, 120, 130, 140]}
    
    # Test possession detection
    events = tracker.update(1, mock_puck, mock_players)
    print(f"Frame 1 events: {len(events)}")
    
    # Move puck to different player
    mock_puck = {'bbox': [310, 170, 330, 190]}
    events = tracker.update(2, mock_puck, mock_players)
    print(f"Frame 2 events: {len(events)}")
    
    # Get stats
    stats = tracker.get_possession_stats()
    print(f"Possession stats: {stats}")
    
    print("Puck possession test completed!")

if __name__ == "__main__":
    test_puck_possession()
