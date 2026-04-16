"""
Team Analytics Module for IceIQ
Analyzes team performance, formations, and tactical patterns.
"""

import numpy as np
import cv2
import torch
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict
import json
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


@dataclass
class TeamFormation:
    """Team formation data structure"""
    formation_type: str
    players: List[int]
    center: Tuple[float, float]
    spread: float
    timestamp: float
    zone: str


@dataclass
class TeamMetrics:
    """Team performance metrics"""
    possession_time: float
    pass_completion_rate: float
    shot_percentage: float
    zone_time: Dict[str, float]
    formation_stability: float
    transition_speed: float
    pressure_rating: float


class TeamAnalytics:
    """Analyzes team performance and tactical patterns"""
    
    def __init__(self, rink_bounds: Optional[Dict] = None):
        self.rink_bounds = rink_bounds or {
            'left_x': 50, 'right_x': 1750, 
            'top_y': 100, 'bottom_y': 900,
            'center_x': 900, 'center_y': 500
        }
        
        # Zone definitions
        self.zones = {
            'defensive': (self.rink_bounds['left_x'], self.rink_bounds['center_x']),
            'neutral': (self.rink_bounds['center_x'] - 100, self.rink_bounds['center_x'] + 100),
            'offensive': (self.rink_bounds['center_x'], self.rink_bounds['right_x'])
        }
        
        self.team_data = defaultdict(list)
        self.formation_history = defaultdict(list)
        self.possession_history = []
    
    def get_zone(self, x: float) -> str:
        """Determine which zone a position is in"""
        if x < self.rink_bounds['center_x'] - 100:
            return 'defensive'
        elif x > self.rink_bounds['center_x'] + 100:
            return 'offensive'
        else:
            return 'neutral'
    
    def analyze_formation(self, players: List[Dict], team_id: int, timestamp: float) -> TeamFormation:
        """Analyze team formation at a given time"""
        team_players = [p for p in players if p.get('team_id') == team_id]
        
        if len(team_players) < 3:
            return TeamFormation('unknown', [], (0, 0), 0, timestamp, 'unknown')
        
        positions = np.array([[p['position'][0], p['position'][1]] for p in team_players])
        player_ids = [p['player_id'] for p in team_players]
        
        # Calculate formation metrics
        center = np.mean(positions, axis=0)
        distances = pdist(positions)
        spread = np.mean(distances)
        
        # Determine formation type
        formation_type = self._classify_formation(positions, team_players)
        
        # Determine zone
        zone = self.get_zone(center[0])
        
        formation = TeamFormation(
            formation_type=formation_type,
            players=player_ids,
            center=tuple(center),
            spread=spread,
            timestamp=timestamp,
            zone=zone
        )
        
        self.formation_history[team_id].append(formation)
        return formation
    
    def _classify_formation(self, positions: np.ndarray, players: List[Dict]) -> str:
        """Classify formation type based on player positions"""
        if len(positions) < 3:
            return 'unknown'
        
        # Simple formation classification based on spread and clustering
        clustering = DBSCAN(eps=100, min_samples=2).fit(positions)
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        
        spread = np.std(positions, axis=0)
        
        if n_clusters >= 3:
            return 'spread'
        elif spread[1] > spread[0]:  # More vertical spread
            return 'stack'
        elif spread[0] > spread[1]:  # More horizontal spread
            return 'line'
        else:
            return 'cluster'
    
    def analyze_possession_sequences(self, possession_data: List[Dict]) -> Dict[str, Any]:
        """Analyze possession sequences and patterns"""
        sequences = []
        current_sequence = []
        
        for event in possession_data:
            if event['event_type'] == 'possession_change':
                if current_sequence:
                    sequences.append(current_sequence)
                current_sequence = [event]
            else:
                current_sequence.append(event)
        
        if current_sequence:
            sequences.append(current_sequence)
        
        # Analyze sequence patterns
        sequence_analysis = {
            'avg_sequence_length': np.mean([len(seq) for seq in sequences]) if sequences else 0,
            'possession_changes': len(sequences),
            'longest_sequence': max([len(seq) for seq in sequences]) if sequences else 0,
            'zone_transitions': self._count_zone_transitions(sequences)
        }
        
        return sequence_analysis
    
    def _count_zone_transitions(self, sequences: List[List[Dict]]) -> int:
        """Count zone transitions in possession sequences"""
        transitions = 0
        for sequence in sequences:
            prev_zone = None
            for event in sequence:
                if 'position' in event:
                    current_zone = self.get_zone(event['position'][0])
                    if prev_zone and prev_zone != current_zone:
                        transitions += 1
                    prev_zone = current_zone
        return transitions
    
    def calculate_team_metrics(self, team_id: int, timeframe: Tuple[float, float]) -> TeamMetrics:
        """Calculate comprehensive team metrics"""
        start_time, end_time = timeframe
        
        # Filter data for timeframe
        formations = [f for f in self.formation_history[team_id] 
                     if start_time <= f.timestamp <= end_time]
        
        if not formations:
            return TeamMetrics(0, 0, 0, {}, 0, 0, 0)
        
        # Calculate metrics
        zone_times = defaultdict(float)
        for i in range(len(formations) - 1):
            duration = formations[i + 1].timestamp - formations[i].timestamp
            zone_times[formations[i].zone] += duration
        
        total_time = end_time - start_time
        zone_percentages = {zone: time / total_time for zone, time in zone_times.items()}
        
        # Formation stability (consistency of formations)
        formation_types = [f.formation_type for f in formations]
        most_common = max(set(formation_types), key=formation_types.count) if formation_types else 'unknown'
        stability = formation_types.count(most_common) / len(formation_types) if formation_types else 0
        
        # Transition speed (average time between formation changes)
        formation_changes = len(set(formation_types))
        transition_speed = formation_changes / total_time if total_time > 0 else 0
        
        return TeamMetrics(
            possession_time=zone_percentages.get('offensive', 0) * 100,
            pass_completion_rate=0.75,  # Placeholder - would need pass data
            shot_percentage=0.12,       # Placeholder - would need shot data
            zone_time=zone_percentages,
            formation_stability=stability,
            transition_speed=transition_speed,
            pressure_rating=zone_percentages.get('defensive', 0) * 100
        )
    
    def analyze_team_patterns(self, players_data: List[Dict], team_id: int) -> Dict[str, Any]:
        """Analyze team tactical patterns"""
        team_players = [p for p in players_data if p.get('team_id') == team_id]
        
        if not team_players:
            return {}
        
        patterns = {
            'avg_formation_spread': 0,
            'zone_preference': 'neutral',
            'formation_types': defaultdict(int),
            'coordination_score': 0,
            'pressure_patterns': []
        }
        
        # Analyze formations
        formations = self.formation_history[team_id]
        if formations:
            patterns['avg_formation_spread'] = np.mean([f.spread for f in formations])
            
            zone_counts = defaultdict(int)
            for f in formations:
                zone_counts[f.zone] += 1
                patterns['formation_types'][f.formation_type] += 1
            
            patterns['zone_preference'] = max(zone_counts, key=zone_counts.get)
        
        # Calculate coordination score based on formation consistency
        if len(formations) > 1:
            spreads = [f.spread for f in formations]
            coordination = 1.0 - (np.std(spreads) / np.mean(spreads)) if np.mean(spreads) > 0 else 0
            patterns['coordination_score'] = max(0, min(1, coordination))
        
        return patterns
    
    def generate_heatmap_data(self, team_id: int, timeframe: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Generate team positioning heatmap data"""
        formations = self.formation_history[team_id]
        
        if timeframe:
            start_time, end_time = timeframe
            formations = [f for f in formations if start_time <= f.timestamp <= end_time]
        
        # Create heatmap grid
        grid_size = (50, 30)
        heatmap = np.zeros(grid_size)
        
        x_bins = np.linspace(self.rink_bounds['left_x'], self.rink_bounds['right_x'], grid_size[0])
        y_bins = np.linspace(self.rink_bounds['top_y'], self.rink_bounds['bottom_y'], grid_size[1])
        
        for formation in formations:
            x, y = formation.center
            x_idx = np.digitize(x, x_bins) - 1
            y_idx = np.digitize(y, y_bins) - 1
            
            if 0 <= x_idx < grid_size[0] and 0 <= y_idx < grid_size[1]:
                heatmap[y_idx, x_idx] += 1
        
        # Normalize
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        return heatmap
    
    def compare_teams(self, team1_id: int, team2_id: int, timeframe: Tuple[float, float]) -> Dict[str, Any]:
        """Compare performance between two teams"""
        team1_metrics = self.calculate_team_metrics(team1_id, timeframe)
        team2_metrics = self.calculate_team_metrics(team2_id, timeframe)
        
        comparison = {
            'possession_advantage': team1_metrics.possession_time - team2_metrics.possession_time,
            'formation_stability_diff': team1_metrics.formation_stability - team2_metrics.formation_stability,
            'pressure_differential': team1_metrics.pressure_rating - team2_metrics.pressure_rating,
            'transition_speed_diff': team1_metrics.transition_speed - team2_metrics.transition_speed,
            'zone_control': {
                'team1': team1_metrics.zone_time,
                'team2': team2_metrics.zone_time
            }
        }
        
        return comparison
    
    def export_team_report(self, team_id: int, output_path: str, timeframe: Optional[Tuple[float, float]] = None):
        """Export comprehensive team analysis report"""
        metrics = self.calculate_team_metrics(team_id, timeframe or (0, float('inf')))
        patterns = self.analyze_team_patterns([], team_id)
        
        report = {
            'team_id': team_id,
            'analysis_timeframe': timeframe,
            'metrics': {
                'possession_time': metrics.possession_time,
                'formation_stability': metrics.formation_stability,
                'zone_time_distribution': metrics.zone_time,
                'transition_speed': metrics.transition_speed,
                'pressure_rating': metrics.pressure_rating
            },
            'patterns': patterns,
            'formations_analyzed': len(self.formation_history[team_id])
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def reset(self):
        """Reset all analytics data"""
        self.team_data.clear()
        self.formation_history.clear()
        self.possession_history.clear()


def create_team_analytics(rink_bounds: Optional[Dict] = None) -> TeamAnalytics:
    """Factory function to create team analytics instance"""
    return TeamAnalytics(rink_bounds)
