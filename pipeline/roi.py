"""
Rink ROI masking module for IceIQ hockey analysis pipeline.

Detects or defines the playable ice surface polygon and filters player
detections to those whose foot-position falls inside the rink.

This eliminates false detections from benches, crowd, and boards that
caused the "24 players detected" problem.

Detection strategy
------------------
1. Load from a previously saved JSON file (fastest path on re-runs).
2. Auto-detect from video frames via HSV ice-colour segmentation.
3. Fall back to a full-frame rectangle if auto-detect fails.
"""

import cv2
import json
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_ROI_PATH = Path(__file__).parent.parent / "data" / "roi_polygon.json"


class RinkROI:
    """
    Defines and applies the playable ice surface ROI.

    Usage
    -----
    roi = create_roi(video_path="data/videos/game1.mp4")
    filtered_tracks = roi.filter_tracks(tracks)
    annotated = roi.draw(frame)
    """

    def __init__(self, roi_path: Optional[str] = None):
        self.polygon: Optional[np.ndarray] = None   # (N, 2) int32 [x, y]
        self._roi_path = Path(roi_path) if roi_path else _DEFAULT_ROI_PATH

        if self._roi_path.exists():
            self.load(str(self._roi_path))
            logger.info("Loaded ROI from %s  (%d vertices)", self._roi_path, len(self.polygon))

    # ------------------------------------------------------------------
    # Build / detect
    # ------------------------------------------------------------------

    def build_from_frame(self, frame: np.ndarray, save: bool = True) -> np.ndarray:
        """
        Auto-detect the ice surface polygon from a single BGR frame.
        Stores result in self.polygon and optionally saves to disk.
        """
        poly = self._detect_ice_polygon(frame)
        self.polygon = poly
        if save:
            self._save_safe(str(self._roi_path))
        return poly

    def build_from_video(
        self,
        video_path: str,
        n_frames: int = 5,
        save: bool = True,
    ) -> np.ndarray:
        """
        Build ROI by sampling n_frames from the beginning of the video.
        Takes the convex hull of all per-frame polygons for robustness.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        all_polys: List[np.ndarray] = []
        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            all_polys.append(self._detect_ice_polygon(frame))
        cap.release()

        if not all_polys:
            raise RuntimeError("No frames could be read from video.")

        all_pts = np.vstack(all_polys).reshape(-1, 1, 2).astype(np.int32)
        hull = cv2.convexHull(all_pts)
        self.polygon = hull.reshape(-1, 2)

        if save:
            self._save_safe(str(self._roi_path))
        return self.polygon

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def is_inside(self, x: float, y: float) -> bool:
        """Return True if point (x, y) is inside (or on) the ROI polygon."""
        if self.polygon is None:
            return True
        pt = (float(x), float(y))
        res = cv2.pointPolygonTest(
            self.polygon.reshape(-1, 1, 2).astype(np.float32), pt, False
        )
        return res >= 0.0

    def filter_tracks(self, tracks: List[Dict]) -> List[Dict]:
        """
        Keep tracks whose foot-position (bottom-centre of bbox) is inside ROI.
        Foot position = (cx, y2) of the bounding box.
        """
        if self.polygon is None:
            return tracks
        out = []
        for t in tracks:
            bbox = t.get("bbox") or t.get("box")
            if bbox is None:
                out.append(t)
                continue
            x1, y1, x2, y2 = bbox[:4]
            if self.is_inside((x1 + x2) / 2.0, float(y2)):
                out.append(t)
        return out

    def filter_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Filter raw YOLO detections by foot position.
        Accepts dicts with either 'bbox' or 'box' key.
        """
        if self.polygon is None:
            return detections
        out = []
        for det in detections:
            bbox = det.get("bbox") or det.get("box")
            if bbox is None:
                out.append(det)
                continue
            x1, y1, x2, y2 = bbox[:4]
            if self.is_inside((x1 + x2) / 2.0, float(y2)):
                out.append(det)
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save polygon to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"polygon": self.polygon.tolist()}, f)
        logger.info("ROI saved → %s", path)

    def load(self, path: str) -> None:
        """Load polygon from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        self.polygon = np.array(data["polygon"], dtype=np.int32)

    def set_polygon(self, points: List[Tuple[int, int]]) -> None:
        """Manually set polygon from a list of (x, y) points."""
        self.polygon = np.array(points, dtype=np.int32)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw the ROI polygon on a copy of frame and return it."""
        vis = frame.copy()
        if self.polygon is not None:
            pts = self.polygon.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)
        return vis

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_ice_polygon(self, frame: np.ndarray) -> np.ndarray:
        """
        Segment the ice surface using HSV colour thresholding.

        Ice characteristics:
          - High brightness (V > 140)
          - Low saturation (S < 70) — distinguishes ice from coloured jerseys
        The top 15 % of the frame is masked out to exclude crowd / scoreboard.
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Bright, low-saturation region → ice
        mask = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 70, 255]))

        # Exclude top 15 % (crowd / boards overhead)
        mask[: int(h * 0.15), :] = 0

        # Morphological cleanup: close small gaps, remove speckles
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.warning("Ice detection: no contours found — using full-frame fallback.")
            return self._full_frame_poly(w, h)

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < w * h * 0.10:
            logger.warning("Ice detection: largest contour < 10 % of frame — fallback.")
            return self._full_frame_poly(w, h)

        # Approximate to a clean polygon
        eps = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, eps, True)
        if len(approx) < 4:
            approx = cv2.convexHull(largest)

        logger.info("Ice polygon detected: %d vertices, area=%.0f px²",
                    len(approx), cv2.contourArea(largest))
        return approx.reshape(-1, 2)

    @staticmethod
    def _full_frame_poly(w: int, h: int) -> np.ndarray:
        return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)

    def _save_safe(self, path: str) -> None:
        try:
            self.save(path)
        except Exception as e:
            logger.warning("Could not save ROI: %s", e)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_roi(
    video_path: Optional[str] = None,
    roi_path: Optional[str] = None,
    force_rebuild: bool = False,
) -> RinkROI:
    """
    Create a RinkROI.

    * If a saved polygon exists and force_rebuild=False, loads it directly.
    * Otherwise auto-builds from the first 5 frames of video_path.
    * If no video is provided and no saved polygon exists, ROI passes everything.
    """
    roi = RinkROI(roi_path=roi_path)
    if (roi.polygon is None or force_rebuild) and video_path is not None:
        logger.info("Building ROI from video: %s", video_path)
        roi.build_from_video(video_path, n_frames=5, save=True)
    elif roi.polygon is None:
        logger.warning("No saved ROI and no video_path provided — pass-through mode.")
    return roi
