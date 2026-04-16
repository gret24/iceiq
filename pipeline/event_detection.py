import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2

class EventDetector:
    """Advanced event detection for hockey analysis"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.events = []
        self.shot_threshold = 15.0  # m/s
        self.pass_threshold = 8.0   # m/s
        self.hit_distance = 2.0     # meters
        self.possession_min_frames = 5
        
    def detect_events(self, 
                     frame: np.ndarray,
                     frame_num: int,
                     players: List[Dict],
                     puck_data: Optional[Dict] = None) -> List[Dict]:
        """Detect hockey events in current frame"""
        events = []
        
        if puck_data and len(players) > 0:
            # Shot detection
            shots = self._detect_shots(frame_num, players, puck_data)
            events.extend(shots)
            
            # Pass detection
            passes = self._detect_passes(frame_num, players, puck_data)
            events.extend(passes)
            
            # Hit detection
            hits = self._detect_hits(frame_num, players)
            events.extend(hits)
            
            # Zone entry/exit detection
            zone_events = self._detect_zone_changes(frame_num, players)
            events.extend(zone_events)
            
        return events
    
    def _detect_shots(self, frame_num: int, players: List[Dict], puck_data: Dict) -> List[Dict]:
        """Detect shot attempts"""
        shots = []
        
        if 'velocity' not in puck_data:
            return shots
            
        puck_speed = np.linalg.norm(puck_data['velocity'])
        
        if puck_speed > self.shot_threshold:
            # Find closest player to puck
            puck_pos = np.array(puck_data['position'])
            closest_player = None
            min_distance = float('inf')
            
            for player in players:
                if 'position' in player:
                    player_pos = np.array(player['position'])
                    distance = np.linalg.norm(puck_pos - player_pos)
                    if distance < min_distance:
                        min_distance = distance
                        closest_player = player
            
            if closest_player and min_distance < 3.0:  # Within 3 meters
                shots.append({
                    'type': 'shot',
                    'frame': frame_num,
                    'player_id': closest_player.get('id', -1),
                    'team': closest_player.get('team', 'unknown'),
                    'position': puck_data['position'],
                    'velocity': puck_data['velocity'],
                    'speed': puck_speed,
                    'confidence': min(1.0, puck_speed / 25.0)
                })
        
        return shots
    
    def _detect_passes(self, frame_num: int, players: List[Dict], puck_data: Dict) -> List[Dict]:
        """Detect pass attempts"""
        passes = []
        
        if 'velocity' not in puck_data or 'possession_history' not in puck_data:
            return passes
            
        puck_speed = np.linalg.norm(puck_data['velocity'])
        
        # Check for possession change
        if (puck_speed > self.pass_threshold and 
            len(puck_data['possession_history']) >= 2):
            
            recent_possession = puck_data['possession_history'][-2:]
            if len(set(recent_possession)) > 1:  # Possession changed
                
                passer_id = recent_possession[0]
                receiver_id = recent_possession[1]
                
                # Find passer and receiver
                passer = next((p for p in players if p.get('id') == passer_id), None)
                receiver = next((p for p in players if p.get('id') == receiver_id), None)
                
                if passer and receiver:
                    # Check if same team
                    same_team = passer.get('team') == receiver.get('team')
                    
                    passes.append({
                        'type': 'pass' if same_team else 'turnover',
                        'frame': frame_num,
                        'passer_id': passer_id,
                        'receiver_id': receiver_id,
                        'passer_team': passer.get('team', 'unknown'),
                        'receiver_team': receiver.get('team', 'unknown'),
                        'position': puck_data['position'],
                        'velocity': puck_data['velocity'],
                        'speed': puck_speed,
                        'successful': same_team,
                        'confidence': min(1.0, puck_speed / 15.0)
                    })
        
        return passes
    
    def _detect_hits(self, frame_num: int, players: List[Dict]) -> List[Dict]:
        """Detect body checks and collisions"""
        hits = []
        
        for i, player1 in enumerate(players):
            for j, player2 in enumerate(players[i+1:], i+1):
                if ('position' in player1 and 'position' in player2 and
                    'velocity' in player1 and 'velocity' in player2):
                    
                    pos1 = np.array(player1['position'])
                    pos2 = np.array(player2['position'])
                    vel1 = np.array(player1['velocity'])
                    vel2 = np.array(player2['velocity'])
                    
                    distance = np.linalg.norm(pos1 - pos2)
                    
                    # Check if players are close and moving towards each other
                    if distance < self.hit_distance:
                        relative_velocity = vel1 - vel2
                        approach_direction = pos2 - pos1
                        
                        # Normalize vectors
                        if np.linalg.norm(approach_direction) > 0:
                            approach_direction = approach_direction / np.linalg.norm(approach_direction)
                            
                            # Check if velocities indicate collision
                            approach_speed = np.dot(relative_velocity, approach_direction)
                            
                            if approach_speed > 3.0:  # Significant approach speed
                                hits.append({
                                    'type': 'hit',
                                    'frame': frame_num,
                                    'player1_id': player1.get('id', -1),
                                    'player2_id': player2.get('id', -1),
                                    'player1_team': player1.get('team', 'unknown'),
                                    'player2_team': player2.get('team', 'unknown'),
                                    'position': ((pos1 + pos2) / 2).tolist(),
                                    'impact_speed': approach_speed,
                                    'confidence': min(1.0, approach_speed / 8.0)
                                })
        
        return hits
    
    def _detect_zone_changes(self, frame_num: int, players: List[Dict]) -> List[Dict]:
        """Detect zone entries and exits"""
        zone_events = []
        
        # Define zone boundaries (assuming standard rink)
        # These would be set based on rink dimensions
        defensive_zone = 61.0  # meters from center
        neutral_zone = 30.5    # meters from center
        
        for player in players:
            if 'position' in player and 'zone_history' in player:
                current_x = player['position'][0]
                
                # Determine current zone
                if current_x < -neutral_zone:
                    current_zone = 'defensive'
                elif current_x > neutral_zone:
                    current_zone = 'offensive'
                else:
                    current_zone = 'neutral'
                
                # Check for zone change
                if len(player['zone_history']) > 0:
                    previous_zone = player['zone_history'][-1]
                    
                    if current_zone != previous_zone:
                        zone_events.append({
                            'type': 'zone_entry' if current_zone == 'offensive' else 'zone_exit',
                            'frame': frame_num,
                            'player_id': player.get('id', -1),
                            'team': player.get('team', 'unknown'),
                            'from_zone': previous_zone,
                            'to_zone': current_zone,
                            'position': player['position'],
                            'confidence': 0.9
                        })
                
                # Update zone history
                if 'zone_history' not in player:
                    player['zone_history'] = []
                player['zone_history'].append(current_zone)
                
                # Keep only recent history
                if len(player['zone_history']) > 10:
                    player['zone_history'] = player['zone_history'][-10:]
        
        return zone_events
    
    def get_event_summary(self, events: List[Dict]) -> Dict:
        """Generate summary statistics for detected events"""
        summary = {
            'total_events': len(events),
            'shots': 0,
            'passes': 0,
            'turnovers': 0,
            'hits': 0,
            'zone_entries': 0,
            'zone_exits': 0
        }
        
        for event in events:
            event_type = event.get('type', 'unknown')
            if event_type in summary:
                summary[event_type] += 1
        
        return summary
    
    def filter_events_by_confidence(self, events: List[Dict], min_confidence: float = 0.5) -> List[Dict]:
        """Filter events by confidence threshold"""
        return [event for event in events if event.get('confidence', 0) >= min_confidence]
    
    def export_events(self, events: List[Dict], output_path: str):
        """Export events to JSON file"""
        import json
        
        with open(output_path, 'w') as f:
            json.dump(events, f, indent=2, default=str)
