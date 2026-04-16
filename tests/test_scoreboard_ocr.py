"""
Test scoreboard OCR functionality.
"""

import cv2
import numpy as np
import sys
import os

# Add pipeline to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipeline'))

from scoreboard_ocr import ScoreboardOCR

def test_scoreboard_ocr():
    """Test scoreboard OCR on sample frame."""
    
    # Initialize scoreboard OCR
    ocr = ScoreboardOCR()
    
    # Create synthetic scoreboard image for testing
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    # Draw scoreboard background
    cv2.rectangle(img, (400, 50), (880, 150), (50, 50, 50), -1)
    cv2.rectangle(img, (400, 50), (880, 150), (255, 255, 255), 2)
    
    # Add text elements
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "HOME", (420, 80), font, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "3", (480, 120), font, 1.5, (255, 255, 255), 3)
    
    cv2.putText(img, "AWAY", (750, 80), font, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "2", (810, 120), font, 1.5, (255, 255, 255), 3)
    
    cv2.putText(img, "12:34", (620, 100), font, 1.0, (255, 255, 255), 2)
    cv2.putText(img, "2ND", (630, 130), font, 0.6, (255, 255, 255), 2)
    
    # Test OCR
    result = ocr.extract_score_info(img)
    
    print("Scoreboard OCR Test Results:")
    print(f"Home Score: {result.get('home_score', 'Not detected')}")
    print(f"Away Score: {result.get('away_score', 'Not detected')}")
    print(f"Time: {result.get('time', 'Not detected')}")
    print(f"Period: {result.get('period', 'Not detected')}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    
    # Test region detection
    regions = ocr.detect_scoreboard_regions(img)
    print(f"\nDetected {len(regions)} scoreboard regions")
    
    for i, region in enumerate(regions):
        x, y, w, h = region
        print(f"Region {i}: ({x}, {y}) - {w}x{h}")
    
    return result

if __name__ == "__main__":
    test_scoreboard_ocr()
