import cv2
import numpy as np
import torch
from ultralytics import YOLO
import logging

class DetectionModule:
    def __init__(self, model_path="yolov8n.pt"):
        """Initialize detection module with YOLO model."""
        self.logger = logging.getLogger(__name__)
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
        
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self.logger.info(f"Detection model loaded on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load detection model: {e}")
            raise
    
    def detect_players(self, frame):
        """Detect players in frame and return bounding boxes."""
        try:
            results = self.model(frame, classes=[0], verbose=False)  # class 0 = person
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        if conf > 0.3:  # confidence threshold
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf)
                            })
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []
    
    def detect_puck(self, frame):
        """Detect puck in frame using specialized detection."""
        try:
            # Convert to grayscale for better puck detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use HoughCircles to detect circular objects (puck)
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=30,
                param1=50,
                param2=30,
                minRadius=3,
                maxRadius=15
            )
            
            detections = []
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    # Basic validation - puck should be on ice (lower part of frame)
                    if y > frame.shape[0] * 0.3:  # below 30% of frame height
                        detections.append({
                            'bbox': [x-r, y-r, x+r, y+r],
                            'confidence': 0.8,
                            'center': (x, y),
                            'radius': r
                        })
            
            return detections
            
        except Exception as e:
            self.logger.error(f"Puck detection failed: {e}")
            return []
    
    def process_frame(self, frame):
        """Process frame and return all detections."""
        try:
            player_detections = self.detect_players(frame)
            puck_detections = self.detect_puck(frame)
            
            return {
                'players': player_detections,
                'puck': puck_detections,
                'frame_shape': frame.shape
            }
            
        except Exception as e:
            self.logger.error(f"Frame processing failed: {e}")
            return {'players': [], 'puck': [], 'frame_shape': frame.shape}
