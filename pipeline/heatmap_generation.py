import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class HeatmapGenerator:
    """Generate heatmaps for player movement and team analysis"""
    
    def __init__(self, blur_kernel_size: int = 51, blur_sigma: float = 20.0):
        """
        Initialize heatmap generator
        
        Args:
            blur_kernel_size: Size of Gaussian blur kernel (must be odd)
            blur_sigma: Sigma value for Gaussian blur
        """
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
    
    def generate_player_heatmap(self, tracks: List[Dict], frame_shape: Tuple[int, int], 
                              player_id: Optional[int] = None, team: Optional[str] = None) -> np.ndarray:
        """
        Generate heatmap for player movement
        
        Args:
            tracks: List of track dictionaries with 'id', 'positions', 'team'
            frame_shape: (width, height) of the frame
            player_id: Filter for specific player ID
            team: Filter for specific team ('home' or 'away')
            
        Returns:
            Heatmap as numpy array
        """
        height, width = frame_shape[1], frame_shape[0]
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        for track in tracks:
            # Apply filters
            if player_id is not None and track.get('id') != player_id:
                continue
            if team is not None and track.get('team') != team:
                continue
                
            positions = track.get('positions', [])
            
            for pos in positions:
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    x, y = int(pos[0]), int(pos[1])
                    
                    # Check bounds
                    if 0 <= x < width and 0 <= y < height:
                        heatmap[y, x] += 1.0
        
        # Apply Gaussian blur for smooth heatmap
        if self.blur_kernel_size > 1:
            heatmap = cv2.GaussianBlur(heatmap, 
                                     (self.blur_kernel_size, self.blur_kernel_size), 
                                     self.blur_sigma)
        
        return heatmap
    
    def generate_team_heatmap(self, tracks: List[Dict], frame_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        Generate separate heatmaps for home and away teams
        
        Args:
            tracks: List of track dictionaries
            frame_shape: (width, height) of the frame
            
        Returns:
            Dictionary with 'home' and 'away' heatmaps
        """
        result = {}
        
        for team_name in ['home', 'away']:
            result[team_name] = self.generate_player_heatmap(tracks, frame_shape, team=team_name)
            
        return result
    
    def generate_zone_heatmap(self, tracks: List[Dict], frame_shape: Tuple[int, int], 
                            zones: Dict[str, List[Tuple[int, int]]]) -> Dict[str, float]:
        """
        Generate heatmap data for predefined zones (e.g., defensive, neutral, offensive)
        
        Args:
            tracks: List of track dictionaries
            frame_shape: (width, height) of the frame
            zones: Dictionary mapping zone names to list of polygon points
            
        Returns:
            Dictionary with zone names and density values
        """
        height, width = frame_shape[1], frame_shape[0]
        zone_counts = {zone: 0.0 for zone in zones.keys()}
        total_points = 0
        
        for track in tracks:
            positions = track.get('positions', [])
            
            for pos in positions:
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    x, y = int(pos[0]), int(pos[1])
                    
                    if 0 <= x < width and 0 <= y < height:
                        total_points += 1
                        
                        # Check which zone this point belongs to
                        point = (x, y)
                        for zone_name, zone_polygon in zones.items():
                            if len(zone_polygon) >= 3:
                                # Use cv2.pointPolygonTest for point-in-polygon
                                polygon = np.array(zone_polygon, dtype=np.int32)
                                if cv2.pointPolygonTest(polygon, point, False) >= 0:
                                    zone_counts[zone_name] += 1.0
                                    break
        
        # Normalize by total points
        if total_points > 0:
            for zone in zone_counts:
                zone_counts[zone] /= total_points
                
        return zone_counts
    
    def visualize_heatmap(self, heatmap: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Convert heatmap to colored visualization
        
        Args:
            heatmap: Input heatmap array
            colormap: OpenCV colormap constant
            
        Returns:
            Colored heatmap as BGR image
        """
        # Normalize to 0-255 range
        if heatmap.max() > 0:
            normalized = ((heatmap / heatmap.max()) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(heatmap, dtype=np.uint8)
            
        # Apply colormap
        colored = cv2.applyColorMap(normalized, colormap)
        
        return colored
    
    def overlay_heatmap(self, background: np.ndarray, heatmap: np.ndarray, 
                       alpha: float = 0.6, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay heatmap on background image
        
        Args:
            background: Background image (BGR)
            heatmap: Heatmap array
            alpha: Transparency factor (0.0 = transparent, 1.0 = opaque)
            colormap: OpenCV colormap constant
            
        Returns:
            Combined image
        """
        colored_heatmap = self.visualize_heatmap(heatmap, colormap)
        
        # Resize heatmap if needed
        if colored_heatmap.shape[:2] != background.shape[:2]:
            colored_heatmap = cv2.resize(colored_heatmap, 
                                       (background.shape[1], background.shape[0]))
        
        # Blend images
        result = cv2.addWeighted(background, 1-alpha, colored_heatmap, alpha, 0)
        
        return result
    
    def generate_time_series_heatmap(self, tracks: List[Dict], frame_shape: Tuple[int, int],
                                   time_window: int = 300) -> List[np.ndarray]:
        """
        Generate heatmaps over time windows
        
        Args:
            tracks: List of track dictionaries with timestamps
            frame_shape: (width, height) of the frame
            time_window: Duration of each time window in frames
            
        Returns:
            List of heatmaps for each time window
        """
        if not tracks:
            return []
        
        # Find time range
        all_timestamps = []
        for track in tracks:
            positions = track.get('positions', [])
            timestamps = track.get('timestamps', [])
            if len(timestamps) == len(positions):
                all_timestamps.extend(timestamps)
        
        if not all_timestamps:
            logger.warning("No timestamps found in tracks")
            return [self.generate_player_heatmap(tracks, frame_shape)]
        
        min_time = min(all_timestamps)
        max_time = max(all_timestamps)
        
        # Generate heatmaps for each time window
        heatmaps = []
        current_time = min_time
        
        while current_time < max_time:
            window_end = current_time + time_window
            
            # Filter tracks for this time window
            window_tracks = []
            for track in tracks:
                positions = track.get('positions', [])
                timestamps = track.get('timestamps', [])
                
                if len(timestamps) == len(positions):
                    window_positions = []
                    for i, ts in enumerate(timestamps):
                        if current_time <= ts < window_end:
                            window_positions.append(positions[i])
                    
                    if window_positions:
                        window_track = track.copy()
                        window_track['positions'] = window_positions
                        window_tracks.append(window_track)
            
            # Generate heatmap for this window
            window_heatmap = self.generate_player_heatmap(window_tracks, frame_shape)
            heatmaps.append(window_heatmap)
            
            current_time = window_end
        
        return heatmaps
