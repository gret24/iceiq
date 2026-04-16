import cv2
import numpy as np
import torch

class RinkHomography:
    def __init__(self):
        # Standard ice hockey rink dimensions (in meters)
        self.rink_width = 61.0  # 200 feet
        self.rink_height = 30.5  # 100 feet
        
        # Standard rink points (in rink coordinates)
        self.rink_points = np.array([
            [0, 0],                    # Top-left corner
            [self.rink_width, 0],      # Top-right corner
            [self.rink_width, self.rink_height],  # Bottom-right corner
            [0, self.rink_height]      # Bottom-left corner
        ], dtype=np.float32)
        
        self.homography_matrix = None
        self.inverse_homography = None
        
    def detect_rink_corners(self, frame):
        """Detect rink corners using edge detection and line fitting"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return None
            
        # For now, use frame corners as approximation
        h, w = frame.shape[:2]
        margin = 50
        
        image_points = np.array([
            [margin, margin],          # Top-left
            [w-margin, margin],        # Top-right
            [w-margin, h-margin],      # Bottom-right
            [margin, h-margin]         # Bottom-left
        ], dtype=np.float32)
        
        return image_points
        
    def compute_homography(self, frame, image_points=None):
        """Compute homography matrix from image points to rink coordinates"""
        if image_points is None:
            image_points = self.detect_rink_corners(frame)
            
        if image_points is None:
            return False
            
        # Compute homography
        self.homography_matrix, _ = cv2.findHomography(
            image_points, 
            self.rink_points,
            cv2.RANSAC
        )
        
        if self.homography_matrix is not None:
            self.inverse_homography = np.linalg.inv(self.homography_matrix)
            return True
            
        return False
        
    def image_to_rink(self, image_point):
        """Transform image coordinates to rink coordinates"""
        if self.homography_matrix is None:
            return None
            
        if isinstance(image_point, (list, tuple)):
            image_point = np.array(image_point, dtype=np.float32)
            
        # Add homogeneous coordinate
        if len(image_point.shape) == 1:
            point_h = np.array([image_point[0], image_point[1], 1.0])
        else:
            ones = np.ones((image_point.shape[0], 1))
            point_h = np.hstack([image_point, ones])
            
        # Transform
        if len(point_h.shape) == 1:
            rink_h = self.homography_matrix @ point_h
            return rink_h[:2] / rink_h[2] if rink_h[2] != 0 else None
        else:
            rink_h = (self.homography_matrix @ point_h.T).T
            rink_points = rink_h[:, :2] / rink_h[:, 2:3]
            return rink_points
            
    def rink_to_image(self, rink_point):
        """Transform rink coordinates to image coordinates"""
        if self.inverse_homography is None:
            return None
            
        if isinstance(rink_point, (list, tuple)):
            rink_point = np.array(rink_point, dtype=np.float32)
            
        # Add homogeneous coordinate
        if len(rink_point.shape) == 1:
            point_h = np.array([rink_point[0], rink_point[1], 1.0])
        else:
            ones = np.ones((rink_point.shape[0], 1))
            point_h = np.hstack([rink_point, ones])
            
        # Transform
        if len(point_h.shape) == 1:
            image_h = self.inverse_homography @ point_h
            return image_h[:2] / image_h[2] if image_h[2] != 0 else None
        else:
            image_h = (self.inverse_homography @ point_h.T).T
            image_points = image_h[:, :2] / image_h[:, 2:3]
            return image_points
            
    def draw_rink_overlay(self, frame):
        """Draw rink markings on the frame"""
        if self.inverse_homography is None:
            return frame
            
        overlay = frame.copy()
        
        # Draw rink outline
        rink_corners = np.array([
            [0, 0],
            [self.rink_width, 0],
            [self.rink_width, self.rink_height],
            [0, self.rink_height],
            [0, 0]
        ], dtype=np.float32)
        
        image_corners = self.rink_to_image(rink_corners)
        if image_corners is not None:
            image_corners = image_corners.astype(np.int32)
            cv2.polylines(overlay, [image_corners], False, (0, 255, 0), 2)
            
        # Draw center line
        center_line = np.array([
            [self.rink_width/2, 0],
            [self.rink_width/2, self.rink_height]
        ], dtype=np.float32)
        
        image_center = self.rink_to_image(center_line)
        if image_center is not None:
            image_center = image_center.astype(np.int32)
            cv2.line(overlay, tuple(image_center[0]), tuple(image_center[1]), (255, 0, 0), 2)
            
        return overlay
        
    def get_zone(self, rink_point):
        """Get the zone (defensive, neutral, offensive) for a rink position"""
        if rink_point is None:
            return "unknown"
            
        x = rink_point[0]
        
        # Zone boundaries (approximate)
        if x < self.rink_width * 0.33:
            return "defensive"
        elif x < self.rink_width * 0.67:
            return "neutral"
        else:
            return "offensive"
            
    def get_distance(self, point1, point2):
        """Get distance between two rink points in meters"""
        if point1 is None or point2 is None:
            return None
            
        return np.linalg.norm(np.array(point2) - np.array(point1))
        
    def is_initialized(self):
        """Check if homography is initialized"""
        return self.homography_matrix is not None
