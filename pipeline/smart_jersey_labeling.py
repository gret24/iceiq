import cv2
import numpy as np
import easyocr
import logging
import difflib
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class PlayerLabeler:
    """
    PlayerLabeler identifies jersey numbers from player tracks using EasyOCR.
    """

    def __init__(self, roster: List[int] = [4, 11, 14, 25, 28, 44, 47]):
        """
        Initialize the PlayerLabeler with a roster and EasyOCR reader.
        """
        self.roster = [str(r) for r in roster]
        # Initialize EasyOCR reader for English digits.
        # gpu=False is used for broader compatibility; set to True if CUDA is available.
        try:
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.reader = None

    def find_best_frame(self, track_history: List[Tuple[np.ndarray, Tuple[float, float, float, float]]]) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float, float]]]:
        """
        Identify the highest quality frame for OCR based on bounding box area and sharpness.
        
        Args:
            track_history: List of (frame, bbox) where bbox is (x1, y1, x2, y2).
            
        Returns:
            A tuple of (best_frame, best_bbox).
        """
        best_frame = None
        best_bbox = None
        max_score = -1.0

        for frame, bbox in track_history:
            x1, y1, x2, y2 = bbox
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            area = w * h
            
            if area == 0:
                continue

            try:
                # Extract player crop for sharpness evaluation
                px1, py1, px2, py2 = map(int, [max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)])
                player_crop = frame[py1:py2, px1:px2]
                
                if player_crop.size == 0:
                    continue
                    
                gray = cv2.cvtColor(player_crop, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            except Exception as e:
                logger.debug(f"Sharpness calculation failed: {e}")
                sharpness = 0.0

            # Combined score heuristic
            score = area * sharpness
            if score > max_score:
                max_score = score
                best_frame = frame
                best_bbox = bbox

        return best_frame, best_bbox

    def crop_jersey_number(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Crop the likely jersey number region from the player's bounding box.
        
        Args:
            frame: The source video frame.
            bbox: The player's bounding box (x1, y1, x2, y2).
            
        Returns:
            The cropped image of the jersey number area.
        """
        x1, y1, x2, y2 = map(int, bbox)
        h = y2 - y1
        w = x2 - x1
        
        # Region of interest: upper-middle torso where jersey numbers are typically located.
        cy1 = max(0, y1 + int(h * 0.30))
        cy2 = min(frame.shape[0], y1 + int(h * 0.65))
        cx1 = max(0, x1 + int(w * 0.20))
        cx2 = min(frame.shape[1], x2 - int(w * 0.20))
        
        return frame[cy1:cy2, cx1:cx2]

    def snap_to_roster(self, raw_text: str, roster: List[int] = [4, 11, 14, 25, 28, 44, 47]) -> Optional[int]:
        """
        Map raw OCR text to the closest valid jersey number in the roster.
        
        Args:
            raw_text: The string returned by OCR.
            roster: The list of valid jersey numbers.
            
        Returns:
            The matched jersey number as an integer, or None if no match is found.
        """
        target_roster = [str(r) for r in roster]

        # Extract only numeric characters
        digits = "".join([c for c in raw_text if c.isdigit()])
        if not digits:
            return None

        if digits in target_roster:
            return int(digits)

        # Fuzzy matching for OCR errors
        matches = difflib.get_close_matches(digits, target_roster, n=1, cutoff=0.5)
        if matches:
            return int(matches[0])
            
        return None

    def label_all(self, track_histories: Dict[int, List[Tuple[np.ndarray, Tuple[float, float, float, float]]]]) -> Dict[int, Optional[int]]:
        """
        Process multiple track histories to assign jersey numbers.
        
        Args:
            track_histories: A dictionary mapping track IDs to lists of (frame, bbox).
            
        Returns:
            A dictionary mapping track IDs to identified jersey numbers.
        """
        results = {}
        if self.reader is None:
            logger.error("OCR reader is not available.")
            return {tid: None for tid in track_histories}

        for track_id, history in track_histories.items():
            best_frame, best_bbox = self.find_best_frame(history)
            
            if best_frame is None or best_bbox is None:
                results[track_id] = None
                continue
                
            crop = self.crop_jersey_number(best_frame, best_bbox)
            if crop.size == 0:
                results[track_id] = None
                continue
                
            try:
                # OCR inference
                ocr_results = self.reader.readtext(crop)
                raw_text = " ".join([res[1] for res in ocr_results])
                results[track_id] = self.snap_to_roster(raw_text)
            except Exception as e:
                logger.error(f"OCR failed for track {track_id}: {e}")
                results[track_id] = None
                
        return results
