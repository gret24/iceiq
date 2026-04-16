import cv2
import numpy as np
import easyocr
from typing import Dict, List, Tuple, Optional
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScoreboardOCR:
    def __init__(self):
        """Initialize scoreboard OCR with EasyOCR reader."""
        self.reader = easyocr.Reader(['en'])
        self.scoreboard_roi = None
        self.time_pattern = re.compile(r'(\d{1,2}):(\d{2})')
        self.score_pattern = re.compile(r'(\d+)\s*[-–]\s*(\d+)')
        self.period_pattern = re.compile(r'(1st|2nd|3rd|OT|SO)', re.IGNORECASE)
        
    def detect_scoreboard_roi(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect scoreboard region in frame.
        Usually appears in top center area.
        
        Args:
            frame: Input frame
            
        Returns:
            (x, y, w, h) of scoreboard ROI or None
        """
        h, w = frame.shape[:2]
        
        # Common scoreboard positions
        search_regions = [
            (w//4, 0, w//2, h//8),  # Top center
            (w//3, 0, w//3, h//10), # Top center narrow
            (0, 0, w, h//12)        # Top full width
        ]
        
        for x, y, roi_w, roi_h in search_regions:
            roi = frame[y:y+roi_h, x:x+roi_w]
            
            # Look for rectangular overlay-like regions
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Find high contrast regions (typical of scoreboards)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum scoreboard size
                    x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
                    
                    # Check aspect ratio (scoreboards are typically wider)
                    aspect_ratio = w_c / h_c
                    if 2.0 < aspect_ratio < 8.0:
                        return (x + x_c, y + y_c, w_c, h_c)
        
        # Fallback: assume top center region
        return (w//4, 0, w//2, h//8)
    
    def preprocess_scoreboard(self, roi: np.ndarray) -> np.ndarray:
        """
        Preprocess scoreboard ROI for better OCR.
        
        Args:
            roi: Scoreboard region
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Scale up for better OCR
        scale_factor = 3
        height, width = denoised.shape
        upscaled = cv2.resize(denoised, (width * scale_factor, height * scale_factor), 
                             interpolation=cv2.INTER_CUBIC)
        
        return upscaled
    
    def extract_text(self, roi: np.ndarray) -> List[Tuple[str, float]]:
        """
        Extract text from scoreboard ROI.
        
        Args:
            roi: Preprocessed scoreboard region
            
        Returns:
            List of (text, confidence) tuples
        """
        try:
            results = self.reader.readtext(roi)
            text_results = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter low confidence
                    # Clean text
                    clean_text = text.strip()
                    if clean_text:
                        text_results.append((clean_text, confidence))
            
            return text_results
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return []
    
    def parse_game_time(self, texts: List[str]) -> Optional[Dict]:
        """
        Parse game time from OCR text.
        
        Args:
            texts: List of OCR text strings
            
        Returns:
            Dict with time info or None
        """
        for text in texts:
            # Look for time pattern MM:SS
            time_match = self.time_pattern.search(text)
            if time_match:
                minutes = int(time_match.group(1))
                seconds = int(time_match.group(2))
                
                # Validate reasonable hockey time
                if 0 <= minutes <= 20 and 0 <= seconds <= 59:
                    total_seconds = minutes * 60 + seconds
                    return {
                        'minutes': minutes,
                        'seconds': seconds,
                        'total_seconds': total_seconds,
                        'display': f"{minutes:02d}:{seconds:02d}"
                    }
        
        return None
    
    def parse_score(self, texts: List[str]) -> Optional[Dict]:
        """
        Parse score from OCR text.
        
        Args:
            texts: List of OCR text strings
            
        Returns:
            Dict with score info or None
        """
        for text in texts:
            # Look for score pattern N-N or N - N
            score_match = self.score_pattern.search(text)
            if score_match:
                home_score = int(score_match.group(1))
                away_score = int(score_match.group(2))
                
                # Validate reasonable hockey scores
                if 0 <= home_score <= 20 and 0 <= away_score <= 20:
                    return {
                        'home': home_score,
                        'away': away_score,
                        'total': home_score + away_score,
                        'display': f"{home_score}-{away_score}"
                    }
        
        return None
    
    def parse_period(self, texts: List[str]) -> Optional[str]:
        """
        Parse period/quarter from OCR text.
        
        Args:
            texts: List of OCR text strings
            
        Returns:
            Period string or None
        """
        for text in texts:
            period_match = self.period_pattern.search(text)
            if period_match:
                return period_match.group(1).upper()
        
        return None
    
    def extract_scoreboard_info(self, frame: np.ndarray) -> Dict:
        """
        Extract all scoreboard information from frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Dict with scoreboard info
        """
        result = {
            'time': None,
            'score': None,
            'period': None,
            'confidence': 0.0,
            'raw_texts': []
        }
        
        # Detect or use cached scoreboard ROI
        if self.scoreboard_roi is None:
            self.scoreboard_roi = self.detect_scoreboard_roi(frame)
        
        if self.scoreboard_roi is None:
            return result
        
        x, y, w, h = self.scoreboard_roi
        roi = frame[y:y+h, x:x+w]
        
        # Preprocess for OCR
        processed_roi = self.preprocess_scoreboard(roi)
        
        # Extract text
        text_results = self.extract_text(processed_roi)
        if not text_results:
            return result
        
        # Parse components
        texts = [text for text, conf in text_results]
        result['raw_texts'] = texts
        
        # Extract time
        result['time'] = self.parse_game_time(texts)
        
        # Extract score
        result['score'] = self.parse_score(texts)
        
        # Extract period
        result['period'] = self.parse_period(texts)
        
        # Calculate overall confidence
        if text_results:
            avg_confidence = np.mean([conf for _, conf in text_results])
            result['confidence'] = float(avg_confidence)
        
        return result
    
    def process_video_scoreboard(self, video_path: str, 
                                sample_interval: int = 30) -> List[Dict]:
        """
        Process video and extract scoreboard info at intervals.
        
        Args:
            video_path: Path to video file
            sample_interval: Sample every N frames
            
        Returns:
            List of scoreboard info dicts with frame numbers
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []
        
        scoreboard_data = []
        frame_count = 0
        
        logger.info("Processing video for scoreboard OCR...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_interval == 0:
                info = self.extract_scoreboard_info(frame)
                info['frame'] = frame_count
                
                # Calculate timestamp
                fps = cap.get(cv2.CAP_PROP_FPS)
                info['timestamp'] = frame_count / fps if fps > 0 else 0
                
                scoreboard_data.append(info)
                
                # Log progress
                if frame_count % (sample_interval * 10) == 0:
                    logger.info(f"Processed frame {frame_count}")
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"Extracted scoreboard info from {len(scoreboard_data)} samples")
        return scoreboard_data
    
    def visualize_scoreboard_detection(self, frame: np.ndarray, 
                                     info: Dict) -> np.ndarray:
        """
        Visualize scoreboard detection and extracted info.
        
        Args:
            frame: Input frame
            info: Scoreboard info dict
            
        Returns:
            Frame with visualization
        """
        vis_frame = frame.copy()
        
        # Draw scoreboard ROI
        if self.scoreboard_roi:
            x, y, w, h = self.scoreboard_roi
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(vis_frame, "Scoreboard ROI", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw extracted info
        y_offset = 30
        
        if info['time']:
            time_text = f"Time: {info['time']['display']}"
            cv2.putText(vis_frame, time_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_offset += 30
        
        if info['score']:
            score_text = f"Score: {info['score']['display']}"
            cv2.putText(vis_frame, score_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_offset += 30
        
        if info['period']:
            period_text = f"Period: {info['period']}"
            cv2.putText(vis_frame, period_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_offset += 30
        
        # Show confidence
        conf_text = f"Confidence: {info['confidence']:.2f}"
        cv2.putText(vis_frame, conf_text, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        return vis_frame

def test_scoreboard_ocr():
    """Test scoreboard OCR on sample video."""
    video_path = "data/videos/game1.mp4"
    
    ocr = ScoreboardOCR()
    
    # Test on first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        info = ocr.extract_scoreboard_info(frame)
        print(f"Scoreboard info: {info}")
        
        # Visualize
        vis_frame = ocr.visualize_scoreboard_detection(frame, info)
        cv2.imwrite("test_scoreboard_detection.jpg", vis_frame)
        print("Saved test_scoreboard_detection.jpg")
    else:
        print("Failed to read video frame")

if __name__ == "__main__":
    test_scoreboard_ocr()
