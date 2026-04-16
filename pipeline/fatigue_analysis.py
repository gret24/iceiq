import cv2
import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FatigueAnalyzer:
    def __init__(self, config_path: str = "configs/player_metrics.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Fatigue analysis parameters
        self.window_size = 300  # 10 seconds at 30fps
        self.shift_detection_threshold = 5  # seconds
        self.intensity_drop_threshold = 0.15  # 15% drop
        
        # Per-player tracking
        self.player_metrics = defaultdict(lambda: {
            'velocities': deque(maxlen=self.window_size),
            'accelerations': deque(maxlen=self.window_size),
            'shift_times': [],
            'total_ice_time': 0,
            'last_seen_frame': 0,
            'fatigue_score': 0.0,
            'performance_trend': deque(maxlen=10)  # Last 10 performance windows
        })
        
        # Team metrics
        self.team_shift_patterns = {'team_1': [], 'team_2': []}
        self.current_frame = 0
        
    def update_player_metrics(self, tracking_data: List[Dict], kinematics: Dict[int, Dict]) -> Dict:
        """Update fatigue metrics for all players"""
        self.current_frame += 1
        
        # Update metrics for each tracked player
        for player in tracking_data:
            player_id = player['player_id']
            
            if player_id in kinematics:
                velocity = kinematics[player_id].get('velocity', 0)
                acceleration = kinematics[player_id].get('acceleration', 0)
                
                # Update velocity and acceleration history
                self.player_metrics[player_id]['velocities'].append(velocity)
                self.player_metrics[player_id]['accelerations'].append(acceleration)
                self.player_metrics[player_id]['last_seen_frame'] = self.current_frame
                
                # Update ice time
                self.player_metrics[player_id]['total_ice_time'] += 1/30.0  # Assuming 30fps
        
        # Detect shifts and calculate fatigue
        self._detect_shifts()
        self._calculate_fatigue_scores()
        
        return self._get_current_fatigue_analysis()
    
    def _detect_shifts(self):
        """Detect line changes and shift patterns"""
        current_time = self.current_frame / 30.0  # Convert to seconds
        
        for player_id, metrics in self.player_metrics.items():
            # Check for shift end (player not seen for threshold time)
            if (self.current_frame - metrics['last_seen_frame']) > (self.shift_detection_threshold * 30):
                # End of shift detected
                if len(metrics['shift_times']) == 0 or len(metrics['shift_times']) % 2 == 1:
                    # Add shift end time
                    metrics['shift_times'].append(current_time)
    
    def _calculate_fatigue_scores(self):
        """Calculate fatigue scores based on performance degradation"""
        for player_id, metrics in self.player_metrics.items():
            if len(metrics['velocities']) < self.window_size // 2:
                continue  # Not enough data
            
            velocities = list(metrics['velocities'])
            
            # Split into early and late portions of the window
            mid_point = len(velocities) // 2
            early_performance = np.mean(velocities[:mid_point])
            late_performance = np.mean(velocities[mid_point:])
            
            # Calculate performance drop
            if early_performance > 0:
                performance_drop = (early_performance - late_performance) / early_performance
                
                # Update fatigue score (exponential moving average)
                alpha = 0.1
                current_fatigue = max(0, performance_drop)
                metrics['fatigue_score'] = (alpha * current_fatigue + 
                                           (1 - alpha) * metrics['fatigue_score'])
                
                # Store performance trend
                metrics['performance_trend'].append(late_performance)
    
    def _get_current_fatigue_analysis(self) -> Dict:
        """Get current fatigue analysis results"""
        analysis = {
            'frame': self.current_frame,
            'timestamp': self.current_frame / 30.0,
            'players': {},
            'team_analysis': self._analyze_team_fatigue()
        }
        
        for player_id, metrics in self.player_metrics.items():
            if len(metrics['velocities']) > 10:  # Minimum data requirement
                analysis['players'][player_id] = {
                    'fatigue_score': round(metrics['fatigue_score'], 3),
                    'ice_time': round(metrics['total_ice_time'], 1),
                    'shift_count': len(metrics['shift_times']) // 2,
                    'current_velocity': list(metrics['velocities'])[-1] if metrics['velocities'] else 0,
                    'avg_velocity': round(np.mean(list(metrics['velocities'])), 2),
                    'performance_trend': self._calculate_trend(metrics['performance_trend']),
                    'needs_rest': metrics['fatigue_score'] > 0.3  # High fatigue threshold
                }
        
        return analysis
    
    def _calculate_trend(self, performance_data: deque) -> str:
        """Calculate performance trend (improving/declining/stable)"""
        if len(performance_data) < 3:
            return "insufficient_data"
        
        recent = list(performance_data)
        first_half = np.mean(recent[:len(recent)//2])
        second_half = np.mean(recent[len(recent)//2:])
        
        if second_half > first_half * 1.05:
            return "improving"
        elif second_half < first_half * 0.95:
            return "declining"
        else:
            return "stable"
    
    def _analyze_team_fatigue(self) -> Dict:
        """Analyze team-level fatigue patterns"""
        team_fatigue = {'team_1': [], 'team_2': []}
        
        for player_id, metrics in self.player_metrics.items():
            # Assign to team based on player_id (simple heuristic)
            team = 'team_1' if player_id % 2 == 0 else 'team_2'
            
            if metrics['fatigue_score'] > 0:
                team_fatigue[team].append(metrics['fatigue_score'])
        
        analysis = {}
        for team, fatigue_scores in team_fatigue.items():
            if fatigue_scores:
                analysis[team] = {
                    'avg_fatigue': round(np.mean(fatigue_scores), 3),
                    'max_fatigue': round(max(fatigue_scores), 3),
                    'players_high_fatigue': sum(1 for f in fatigue_scores if f > 0.3),
                    'total_players': len(fatigue_scores)
                }
            else:
                analysis[team] = {
                    'avg_fatigue': 0,
                    'max_fatigue': 0,
                    'players_high_fatigue': 0,
                    'total_players': 0
                }
        
        return analysis
    
    def get_shift_analysis(self) -> Dict:
        """Get detailed shift analysis"""
        shift_analysis = {}
        
        for player_id, metrics in self.player_metrics.items():
            shift_times = metrics['shift_times']
            
            if len(shift_times) >= 2:
                # Calculate shift durations (pairs of start/end times)
                shift_durations = []
                for i in range(0, len(shift_times) - 1, 2):
                    duration = shift_times[i + 1] - shift_times[i]
                    shift_durations.append(duration)
                
                shift_analysis[player_id] = {
                    'total_shifts': len(shift_durations),
                    'avg_shift_duration': round(np.mean(shift_durations), 1) if shift_durations else 0,
                    'max_shift_duration': round(max(shift_durations), 1) if shift_durations else 0,
                    'total_ice_time': round(metrics['total_ice_time'], 1),
                    'last_shift_duration': round(shift_durations[-1], 1) if shift_durations else 0
                }
        
        return shift_analysis
    
    def generate_fatigue_alerts(self) -> List[Dict]:
        """Generate fatigue-based alerts for coaching staff"""
        alerts = []
        
        for player_id, metrics in self.player_metrics.items():
            # High fatigue alert
            if metrics['fatigue_score'] > 0.4:
                alerts.append({
                    'type': 'high_fatigue',
                    'player_id': player_id,
                    'severity': 'high',
                    'message': f"Player {player_id} showing high fatigue (score: {metrics['fatigue_score']:.2f})",
                    'recommendation': 'Consider line change'
                })
            
            # Long shift alert
            current_time = self.current_frame / 30.0
            shift_times = metrics['shift_times']
            if len(shift_times) % 2 == 1:  # Currently on ice
                current_shift_duration = current_time - shift_times[-1]
                if current_shift_duration > 45:  # 45 second shift
                    alerts.append({
                        'type': 'long_shift',
                        'player_id': player_id,
                        'severity': 'medium',
                        'message': f"Player {player_id} on long shift ({current_shift_duration:.1f}s)",
                        'recommendation': 'Consider line change soon'
                    })
            
            # Performance decline alert
            if len(metrics['performance_trend']) >= 5:
                recent_trend = list(metrics['performance_trend'])[-3:]
                if all(recent_trend[i] < recent_trend[i-1] for i in range(1, len(recent_trend))):
                    alerts.append({
                        'type': 'performance_decline',
                        'player_id': player_id,
                        'severity': 'medium',
                        'message': f"Player {player_id} showing declining performance",
                        'recommendation': 'Monitor closely'
                    })
        
        return alerts
    
    def save_fatigue_report(self, output_path: str):
        """Save comprehensive fatigue analysis report"""
        report = {
            'analysis_summary': self._get_current_fatigue_analysis(),
            'shift_analysis': self.get_shift_analysis(),
            'fatigue_alerts': self.generate_fatigue_alerts(),
            'metadata': {
                'total_frames_analyzed': self.current_frame,
                'analysis_duration': self.current_frame / 30.0,
                'players_tracked': len(self.player_metrics)
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Fatigue analysis report saved to {output_path}")


def test_fatigue_analyzer():
    """Test the fatigue analyzer with sample data"""
    analyzer = FatigueAnalyzer()
    
    # Simulate tracking data over time
    for frame in range(1000):  # ~33 seconds at 30fps
        # Create sample tracking data
        tracking_data = []
        kinematics = {}
        
        for player_id in range(1, 13):  # 12 players
            # Simulate player positions and movement
            tracking_data.append({
                'player_id': player_id,
                'bbox': [100 + player_id * 50, 100, 150 + player_id * 50, 200],
                'team': 'team_1' if player_id <= 6 else 'team_2'
            })
            
            # Simulate kinematics with fatigue effects
            base_velocity = 5.0
            fatigue_factor = max(0.5, 1.0 - (frame / 1000) * 0.3)  # Gradual fatigue
            velocity = base_velocity * fatigue_factor + np.random.normal(0, 0.5)
            
            kinematics[player_id] = {
                'velocity': max(0, velocity),
                'acceleration': np.random.normal(0, 1.0)
            }
        
        # Update analyzer
        analysis = analyzer.update_player_metrics(tracking_data, kinematics)
        
        # Print analysis every 10 seconds
        if frame % 300 == 0 and frame > 0:
            print(f"\nFrame {frame} ({frame/30:.1f}s):")
            print(f"Players analyzed: {len(analysis['players'])}")
            
            # Show top 3 most fatigued players
            players_by_fatigue = sorted(
                analysis['players'].items(),
                key=lambda x: x[1]['fatigue_score'],
                reverse=True
            )
            
            print("Most fatigued players:")
            for i, (pid, data) in enumerate(players_by_fatigue[:3]):
                print(f"  {i+1}. Player {pid}: fatigue={data['fatigue_score']:.3f}, "
                      f"trend={data['performance_trend']}")
            
            # Show alerts
            alerts = analyzer.generate_fatigue_alerts()
            if alerts:
                print(f"Active alerts: {len(alerts)}")
                for alert in alerts[:3]:  # Show first 3
                    print(f"  - {alert['type']}: {alert['message']}")
    
    # Save final report
    analyzer.save_fatigue_report("test_fatigue_report.json")
    print("\nTest completed! Report saved to test_fatigue_report.json")


if __name__ == "__main__":
    test_fatigue_analyzer()
