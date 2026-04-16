"""
pipeline/puck_detection.py — Puck Detection & Possession for IceIQ Hockey Analysis

3-stage hybrid pipeline
-----------------------
  Stage 1 : MOG2 background subtraction + blob filter
             - No training needed, works immediately
             - Blob constraints: area 2–350 px², circularity > 0.35, local contrast filter
  Stage 2 : Kalman filter for occlusion bridging
             - 6-state: x, y, vx, vy, ax, ay
             - Max extrapolation: 15 frames (~0.5 s at 30 fps)
  Stage 3 : YOLOv8n stub (placeholder — will be trained on labelled puck data)

Classes
-------
  PuckState       — dataclass returned by PuckTracker.update()
  PuckTracker     — main tracking class
  PuckPossession  — maps puck position to player/team possession
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── filterpy ──────────────────────────────────────────────────────────────────
try:
    from filterpy.kalman import KalmanFilter as _KF
    _FILTERPY_OK = True
except ImportError:  # pragma: no cover
    _FILTERPY_OK = False
    logger.warning("filterpy not installed — Kalman filter disabled. pip install filterpy")

# ── homography (optional — graceful degradation) ───────────────────────────────
try:
    from pipeline.homography import RinkHomography
    _HOMOGRAPHY_OK = True
except ImportError:
    _HOMOGRAPHY_OK = False
    logger.warning("pipeline.homography unavailable — rink_pos will be None")

# ── ROI masking ───────────────────────────────────────────────────────────────
try:
    from pipeline.roi import RinkROI, create_roi
    _ROI_OK = True
except ImportError:
    _ROI_OK = False
    logger.warning("pipeline.roi unavailable — ROI masking disabled")

# ── constants ─────────────────────────────────────────────────────────────────
MAX_OCCLUDED_FRAMES      = 15    # frames before giving up on Kalman extrapolation
BLOB_MIN_AREA_PX2        = 2    # minimum blob area (pixels²)
BLOB_MAX_AREA_PX2        = 350  # maximum blob area (pixels²)
BLOB_MIN_ROUNDNESS       = 0.35 # circularity threshold (1.0 = perfect circle)
BLOB_MAX_BRIGHTNESS      = 130  # ignore blobs brighter than this (ice is white)
LOCAL_CONTRAST_ANNULUS   = 20   # outer annulus radius for local contrast check (px)
LOCAL_CONTRAST_MIN_DIFF  = 0.12 # blob must be this fraction darker than annulus
MAX_JUMP_PX              = 100  # reject candidate if > this far from last known position
ROI_TOP_EXPAND_PX        = 60   # expand ROI top edge upward to include far boards
MOG2_HISTORY         = 200      # MOG2 history length (frames)
MOG2_VAR_THRESHOLD   = 40       # MOG2 pixel variance threshold
POSSESSION_RADIUS_M  = 1.5      # stick-reach radius in rink metres
IN_FLIGHT_SPEED_MS   = 8.0      # m/s above which puck is considered "in flight"

# ── goal area exclusion mask ──────────────────────────────────────────────────
GOAL_HALF_W_M        = 1.5      # half-width of excluded goal zone in rink metres
GOAL_HALF_H_M        = 1.5      # half-height of excluded goal zone in rink metres
FAR_GOAL_RY          = -27.13   # far  goal line Y in rink metres
NEAR_GOAL_RY         = +27.13   # near goal line Y in rink metres

# ── static blob rejection ─────────────────────────────────────────────────────
STATIC_GRID_PX       = 10       # grid cell size (pixels) for static-blob detection
STATIC_HISTORY_LEN   = 8        # rolling window length (frames)
STATIC_MIN_COUNT     = 7        # reject if blob in same cell for this many frames


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class PuckState:
    """Output from PuckTracker.update() for one frame."""
    pixel_pos: Optional[Tuple[float, float]]        # (cx, cy) in image pixels, or None
    rink_pos:  Optional[Tuple[float, float]]        # (rx, ry) in rink metres, or None
    confidence: float                               # 0.0 – 1.0
    state: Literal["detected", "occluded", "searching"]
    velocity_ms: Optional[Tuple[float, float]] = None   # rink velocity (vx, vy) m/s
    speed_ms: float = 0.0                           # scalar speed (m/s)
    frame_idx: int = 0


@dataclass
class _KalmanState:
    """Internal Kalman filter wrapper for puck tracking."""
    kf: object                          # filterpy KalmanFilter
    occluded_frames: int = 0
    last_pixel_pos: Optional[Tuple[float, float]] = None
    last_rink_pos:  Optional[Tuple[float, float]] = None


# ── helper: build 6-state Kalman filter ───────────────────────────────────────

def _build_kalman(fps: float = 30.0) -> "_KF":
    """
    Constant-acceleration Kalman filter in rink coordinates.
    State: [x, y, vx, vy, ax, ay]
    Measurement: [x, y]
    """
    dt = 1.0 / fps
    kf = _KF(dim_x=6, dim_z=2)

    # State transition: constant-acceleration model
    kf.F = np.array([
        [1, 0, dt,  0, 0.5*dt**2, 0         ],
        [0, 1,  0, dt, 0,         0.5*dt**2  ],
        [0, 0,  1,  0, dt,        0          ],
        [0, 0,  0,  1, 0,         dt         ],
        [0, 0,  0,  0, 1,         0          ],
        [0, 0,  0,  0, 0,         1          ],
    ], dtype=float)

    # Measurement matrix: observe x and y only
    kf.H = np.zeros((2, 6), dtype=float)
    kf.H[0, 0] = 1.0
    kf.H[1, 1] = 1.0

    # Measurement noise (rink metres — puck detection ≈ 0.1 m accuracy)
    kf.R = np.eye(2, dtype=float) * 0.05

    # Process noise
    q = 2.0   # puck can accelerate hard (shots, passes)
    kf.Q = np.eye(6, dtype=float) * q
    kf.Q[4, 4] = q * 0.1   # acceleration noise is smaller
    kf.Q[5, 5] = q * 0.1

    # Initial state covariance
    kf.P = np.eye(6, dtype=float) * 10.0

    kf.x = np.zeros((6, 1), dtype=float)
    return kf


# ── YOLOv8n stub ──────────────────────────────────────────────────────────────

class _YOLOv8nStub:
    """
    Placeholder for a future YOLOv8n puck detector.
    Returns no detections until a real model is loaded.
    """
    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float]]:
        """Returns list of (cx, cy, confidence) tuples — always empty for now."""
        return []


# ── PuckTracker ───────────────────────────────────────────────────────────────

class PuckTracker:
    """
    Hybrid 3-stage puck tracker.

    Usage
    -----
    >>> tracker = PuckTracker()
    >>> state = tracker.update(frame, player_bboxes, timestamp)
    """

    def __init__(
        self,
        fps: float = 30.0,
        homography: Optional["RinkHomography"] = None,
        cal_path: Optional[Path] = None,
        roi: Optional["RinkROI"] = None,
    ) -> None:
        self.fps = fps
        self._roi = roi  # RinkROI for masking detections to ice surface

        # ── Stage 1: MOG2 background subtractor ───────────────────────────────
        self._mog2 = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=False,
        )

        # ── Stage 2: Kalman filter (initialised on first detection) ───────────
        self._ks: Optional[_KalmanState] = None

        # ── Stage 3: YOLOv8n stub ─────────────────────────────────────────────
        self._yolo = _YOLOv8nStub()

        # ── Homography ────────────────────────────────────────────────────────
        self._hom: Optional["RinkHomography"] = homography
        if self._hom is None and _HOMOGRAPHY_OK:
            try:
                p = cal_path or (Path(__file__).parent.parent / "data" / "homography_calibration.json")
                self._hom = RinkHomography(cal_path=p)
            except Exception as exc:
                logger.warning("Could not load homography: %s", exc)

        self._frame_idx = 0
        self._last_blob_count = 0   # blobs found in most recent _stage1_mog2 call
        self._last_state: PuckState = PuckState(
            pixel_pos=None, rink_pos=None, confidence=0.0, state="searching"
        )

        # Goal area exclusion mask (built lazily on first frame)
        self._goal_mask: Optional[np.ndarray] = None

        # Static blob rejection: rolling window of per-frame occupied grid-cell sets
        self._static_cells: deque = deque(maxlen=STATIC_HISTORY_LEN)

    # ── public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        player_bboxes: List[Tuple[int, int, int, int]],
        timestamp: float,
    ) -> PuckState:
        """
        Process one frame and return PuckState.

        Parameters
        ----------
        frame         : BGR image (H×W×3)
        player_bboxes : list of (x1, y1, x2, y2) player bounding boxes in pixels
        timestamp     : seconds since video start (used for logging only)
        """
        self._frame_idx += 1

        # ── Stage 1: MOG2 candidate blobs ─────────────────────────────────────
        candidates = self._stage1_mog2(frame, player_bboxes)

        # ── Stage 3: YOLO candidates (stub — always empty) ────────────────────
        yolo_hits = self._yolo.detect(frame)

        # ── Merge candidates ──────────────────────────────────────────────────
        best_pixel = self._pick_best_candidate(candidates, yolo_hits, frame)

        # ── Stage 2: Kalman predict + update ──────────────────────────────────
        state = self._stage2_kalman(best_pixel, frame)

        state.frame_idx = self._frame_idx
        self._last_state = state
        return state

    # ── Stage 1 ───────────────────────────────────────────────────────────────

    def _stage1_mog2(
        self,
        frame: np.ndarray,
        player_bboxes: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[float, float, float]]:
        """
        Returns list of (cx, cy, score) blob candidates in pixel space.
        score ∈ [0, 1] based on size + roundness + darkness.
        """
        # Build player mask to suppress false positives
        player_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for x1, y1, x2, y2 in player_bboxes:
            # Expand by 5 px to cover skates / sticks near body
            px1 = max(0, x1 - 5)
            py1 = max(0, y1 - 5)
            px2 = min(frame.shape[1] - 1, x2 + 5)
            py2 = min(frame.shape[0] - 1, y2 + 5)
            player_mask[py1:py2, px1:px2] = 255

        # Apply MOG2
        fg_mask = self._mog2.apply(frame)

        # Suppress player regions and shadows
        fg_mask[player_mask > 0] = 0
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleanup — close small gaps, remove noise
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel3, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel3, iterations=1)

        # ── ROI mask: restrict detection to ice surface only ──────────────────
        if self._roi is not None and self._roi.polygon is not None:
            poly = self._roi.polygon.reshape(-1, 2).astype(np.float32)
            # Expand top edge upward by ROI_TOP_EXPAND_PX to include far boards
            min_y = poly[:, 1].min()
            top_mask = poly[:, 1] <= (min_y + 2)   # points within 2px of top edge
            poly[top_mask, 1] -= ROI_TOP_EXPAND_PX
            poly[:, 1] = np.clip(poly[:, 1], 0, frame.shape[0] - 1)
            roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(roi_mask, [poly.reshape(-1, 1, 2).astype(np.int32)], 255)
            fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=roi_mask)

        # ── Goal area exclusion mask ───────────────────────────────────────────
        h_img, w_img = frame.shape[:2]
        if self._goal_mask is None:
            self._goal_mask = self._build_goal_mask(h_img, w_img)
        fg_mask = cv2.bitwise_and(fg_mask, self._goal_mask)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Tuple[float, float, float]] = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (BLOB_MIN_AREA_PX2 <= area <= BLOB_MAX_AREA_PX2):
                continue

            # Equivalent radius (for spatial sampling)
            r = max(1.0, np.sqrt(area / np.pi))

            # Roundness (circularity) = 4π·A / P²
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1.0:
                continue
            circularity = 4.0 * np.pi * area / (perimeter ** 2)
            if circularity < BLOB_MIN_ROUNDNESS:
                continue

            # Centroid
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            icx, icy = int(cx), int(cy)
            ir = int(r)

            # Darkness check — puck is black rubber, not ice-white
            blob_patch = gray[
                max(0, icy - ir - 1): icy + ir + 2,
                max(0, icx - ir - 1): icx + ir + 2,
            ]
            if blob_patch.size == 0:
                continue
            mean_brightness = float(blob_patch.mean())
            if mean_brightness > BLOB_MAX_BRIGHTNESS:
                continue

            # Local contrast: blob must be ≥ LOCAL_CONTRAST_MIN_DIFF darker than
            # surrounding annulus (inner radius r, outer radius r + annulus width)
            ann_r = ir + LOCAL_CONTRAST_ANNULUS
            h, w = gray.shape
            ann_patch = gray[
                max(0, icy - ann_r): min(h, icy + ann_r + 1),
                max(0, icx - ann_r): min(w, icx + ann_r + 1),
            ]
            if ann_patch.size > 0:
                ann_mean = float(ann_patch.mean())
                # Only count surrounding annulus pixels (exclude blob core)
                # Approximate: annulus mean ≈ (total_sum - blob_sum) / annulus_count
                blob_sum   = mean_brightness * blob_patch.size
                total_sum  = ann_mean * ann_patch.size
                annulus_px = ann_patch.size - blob_patch.size
                if annulus_px > 0:
                    surround_mean = (total_sum - blob_sum) / annulus_px
                else:
                    surround_mean = ann_mean
                if surround_mean > 0 and (surround_mean - mean_brightness) / surround_mean < LOCAL_CONTRAST_MIN_DIFF:
                    continue

            # Score: prefer rounder & darker blobs in valid size range
            ideal_area  = 50.0   # ~r=4px target area
            size_score  = 1.0 - abs(area - ideal_area) / BLOB_MAX_AREA_PX2
            round_score = circularity
            dark_score  = 1.0 - mean_brightness / BLOB_MAX_BRIGHTNESS
            score = 0.35 * size_score + 0.35 * round_score + 0.30 * dark_score

            candidates.append((cx, cy, score))

        # ── Static blob rejection ──────────────────────────────────────────────
        # Compute occupied grid cells for this frame
        current_cells: Set[Tuple[int, int]] = {
            (int(cx) // STATIC_GRID_PX, int(cy) // STATIC_GRID_PX)
            for cx, cy, _ in candidates
        }

        # Count how many past frames each cell was occupied
        if len(self._static_cells) >= STATIC_MIN_COUNT:
            cell_counts: Dict[Tuple[int, int], int] = {}
            for past_cells in self._static_cells:
                for cell in past_cells:
                    cell_counts[cell] = cell_counts.get(cell, 0) + 1
            static_cells = {
                cell for cell, count in cell_counts.items()
                if count >= STATIC_MIN_COUNT
            }
            candidates = [
                (cx, cy, sc) for cx, cy, sc in candidates
                if (int(cx) // STATIC_GRID_PX, int(cy) // STATIC_GRID_PX) not in static_cells
            ]

        self._static_cells.append(current_cells)

        self._last_blob_count = len(candidates)
        return candidates

    # ── Goal area exclusion mask ───────────────────────────────────────────────

    def _build_goal_mask(self, h: int, w: int) -> np.ndarray:
        """
        Build a mask where goal areas are zeroed out (255 = valid, 0 = excluded).
        Projects goal zone corners from rink coords to pixels via homography.
        Falls back to all-255 if homography is unavailable.
        """
        mask = np.full((h, w), 255, dtype=np.uint8)
        if self._hom is None:
            return mask

        for goal_ry in (FAR_GOAL_RY, NEAR_GOAL_RY):
            corners_rink = [
                (-GOAL_HALF_W_M, goal_ry - GOAL_HALF_H_M),
                (+GOAL_HALF_W_M, goal_ry - GOAL_HALF_H_M),
                (+GOAL_HALF_W_M, goal_ry + GOAL_HALF_H_M),
                (-GOAL_HALF_W_M, goal_ry + GOAL_HALF_H_M),
            ]
            try:
                corners_px = [self._hom.rink_to_pixel(rx, ry) for rx, ry in corners_rink]
                pts = np.array(corners_px, dtype=np.float32).reshape(-1, 1, 2)
                pts[:, :, 0] = np.clip(pts[:, :, 0], 0, w - 1)
                pts[:, :, 1] = np.clip(pts[:, :, 1], 0, h - 1)
                cv2.fillPoly(mask, [pts.astype(np.int32)], 0)
                logger.debug("Goal exclusion zone projected: goal_ry=%.2f -> %s", goal_ry, corners_px)
            except Exception as exc:
                logger.warning("Could not project goal zone (goal_ry=%.2f): %s", goal_ry, exc)

        return mask

    # ── Candidate selection ────────────────────────────────────────────────────

    def _pick_best_candidate(
        self,
        mog2_candidates: List[Tuple[float, float, float]],
        yolo_hits: List[Tuple[float, float, float]],
        frame: np.ndarray,
    ) -> Optional[Tuple[float, float, float]]:
        """
        Merge MOG2 and YOLO candidates and return the highest-scoring one.
        YOLO detections get a confidence boost of +0.2.

        Returns (cx, cy, score) or None.
        """
        all_cands: List[Tuple[float, float, float]] = list(mog2_candidates)
        for cx, cy, conf in yolo_hits:
            all_cands.append((cx, cy, min(1.0, conf + 0.20)))

        if not all_cands:
            return None

        # ── Trajectory continuity: reject candidates that jump > MAX_JUMP_PX ──
        if self._ks is not None and self._ks.last_pixel_pos is not None:
            lpx, lpy = self._ks.last_pixel_pos
            all_cands = [
                c for c in all_cands
                if np.hypot(c[0] - lpx, c[1] - lpy) <= MAX_JUMP_PX
            ]
            if not all_cands:
                return None

        # If Kalman is active, prefer candidates closest to predicted position
        if self._ks is not None and self._ks.last_pixel_pos is not None:
            px, py = self._ks.last_pixel_pos
            def _priority(c: Tuple[float, float, float]) -> float:
                dist = np.hypot(c[0] - px, c[1] - py)
                dist_penalty = min(1.0, dist / 200.0)  # normalise over 200 px
                return c[2] - 0.5 * dist_penalty

            all_cands.sort(key=_priority, reverse=True)
        else:
            all_cands.sort(key=lambda c: c[2], reverse=True)

        return all_cands[0]

    # ── Stage 2: Kalman ───────────────────────────────────────────────────────

    def _stage2_kalman(
        self,
        best_pixel: Optional[Tuple[float, float, float]],
        frame: np.ndarray,
    ) -> PuckState:
        """Predict, optionally update, and emit PuckState."""
        if not _FILTERPY_OK:
            # Graceful fallback — no Kalman, return raw detection or searching
            if best_pixel is not None:
                cx, cy, score = best_pixel
                rp = self._to_rink(cx, cy)
                return PuckState(
                    pixel_pos=(cx, cy), rink_pos=rp,
                    confidence=score, state="detected"
                )
            return PuckState(pixel_pos=None, rink_pos=None, confidence=0.0, state="searching")

        # ── Initialise Kalman on first detection ──────────────────────────────
        if self._ks is None:
            if best_pixel is None:
                return PuckState(pixel_pos=None, rink_pos=None, confidence=0.0, state="searching")
            cx, cy, score = best_pixel
            rp = self._to_rink(cx, cy)
            kf = _build_kalman(self.fps)
            rx, ry = rp if rp else (cx, cy)
            kf.x[:2, 0] = [rx, ry]
            self._ks = _KalmanState(kf=kf, last_pixel_pos=(cx, cy), last_rink_pos=rp)
            return PuckState(
                pixel_pos=(cx, cy), rink_pos=rp,
                confidence=score, state="detected",
                velocity_ms=(0.0, 0.0), speed_ms=0.0,
            )

        ks = self._ks

        # ── Predict ───────────────────────────────────────────────────────────
        ks.kf.predict()
        pred_rink = (float(ks.kf.x[0, 0]), float(ks.kf.x[1, 0]))
        pred_pixel = self._from_rink(*pred_rink) or ks.last_pixel_pos

        if best_pixel is not None:
            # ── Update with observation ───────────────────────────────────────
            cx, cy, score = best_pixel
            rp = self._to_rink(cx, cy)
            if rp is None:
                rp = pred_rink  # fallback to predicted if homography fails

            ks.kf.update(np.array([[rp[0]], [rp[1]]], dtype=float))
            ks.occluded_frames = 0
            ks.last_pixel_pos = (cx, cy)
            ks.last_rink_pos  = rp

            vx = float(ks.kf.x[2, 0])
            vy = float(ks.kf.x[3, 0])
            speed = float(np.hypot(vx, vy))

            return PuckState(
                pixel_pos=(cx, cy), rink_pos=rp,
                confidence=min(1.0, score),
                state="detected",
                velocity_ms=(vx, vy), speed_ms=speed,
            )
        else:
            # ── No observation: extrapolate via Kalman ─────────────────────────
            ks.occluded_frames += 1

            if ks.occluded_frames > MAX_OCCLUDED_FRAMES:
                # Lost track entirely
                self._ks = None
                return PuckState(
                    pixel_pos=None, rink_pos=None,
                    confidence=0.0, state="searching"
                )

            # Confidence decays linearly over occluded period
            conf = max(0.0, 1.0 - ks.occluded_frames / MAX_OCCLUDED_FRAMES)
            vx = float(ks.kf.x[2, 0])
            vy = float(ks.kf.x[3, 0])
            speed = float(np.hypot(vx, vy))

            return PuckState(
                pixel_pos=pred_pixel,
                rink_pos=pred_rink,
                confidence=conf * 0.7,   # occluded confidence ceiling = 0.7
                state="occluded",
                velocity_ms=(vx, vy), speed_ms=speed,
            )

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _to_rink(self, px: float, py: float) -> Optional[Tuple[float, float]]:
        if self._hom is None:
            return None
        try:
            return self._hom.pixel_to_rink(px, py)
        except Exception:
            return None

    def _from_rink(self, rx: float, ry: float) -> Optional[Tuple[float, float]]:
        if self._hom is None:
            return None
        try:
            return self._hom.rink_to_pixel(rx, ry)
        except Exception:
            return None

    # ── debug / visualisation helper ─────────────────────────────────────────

    def draw(self, frame: np.ndarray, state: PuckState) -> np.ndarray:
        """Overlay puck state onto frame (in-place). Returns frame."""
        vis = frame.copy()
        if state.pixel_pos is None:
            cv2.putText(vis, f"Puck: {state.state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis

        cx, cy = int(state.pixel_pos[0]), int(state.pixel_pos[1])
        colour = {
            "detected":  (0, 255, 0),
            "occluded":  (0, 165, 255),
            "searching": (0, 0, 255),
        }[state.state]

        cv2.circle(vis, (cx, cy), 12, colour, 2)
        cv2.circle(vis, (cx, cy), 3, colour, -1)

        label = f"{state.state} {state.confidence:.2f} | {state.speed_ms:.1f}m/s"
        cv2.putText(vis, label, (cx + 14, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

        if state.rink_pos:
            rx, ry = state.rink_pos
            pos_str = f"rink ({rx:.1f}, {ry:.1f})m"
            cv2.putText(vis, pos_str, (cx + 14, cy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
        return vis


# ── PuckPossession ────────────────────────────────────────────────────────────

class PuckPossession:
    """
    Maps puck position + velocity to player/team possession.

    Usage
    -----
    >>> possession = PuckPossession()
    >>> result = possession.update(puck_state, player_positions, teams)
    >>> # result: {"team": "zenith", "player_id": 7, "confidence": 0.82}
    >>>           OR {"team": "in_flight", "player_id": None, "confidence": 1.0}
    >>>           OR {"team": None, "player_id": None, "confidence": 0.0}
    """

    def __init__(
        self,
        possession_radius_m: float = POSSESSION_RADIUS_M,
        in_flight_speed_ms: float = IN_FLIGHT_SPEED_MS,
    ) -> None:
        self.possession_radius_m = possession_radius_m
        self.in_flight_speed_ms  = in_flight_speed_ms

        # Temporal smoothing: last N possession results
        self._history: List[Dict] = []
        self._history_len = 5

    def update(
        self,
        puck_state: PuckState,
        player_positions: List[Dict],
        teams: Optional[Dict[int, str]] = None,
    ) -> Dict:
        """
        Determine possession for this frame.

        Parameters
        ----------
        puck_state       : PuckState from PuckTracker.update()
        player_positions : list of dicts with keys:
                             track_id (int)
                             rink_pos (Tuple[float, float]) metres
        teams            : dict mapping track_id → team label string (optional)

        Returns
        -------
        dict with keys:
            team        : str | None       team label or "in_flight"
            player_id   : int | None       track_id of possessing player
            confidence  : float            0.0 – 1.0
        """
        no_possession = {"team": None, "player_id": None, "confidence": 0.0}

        if puck_state.state == "searching" or puck_state.rink_pos is None:
            return no_possession

        # In-flight check: puck moving too fast for possession
        if puck_state.speed_ms >= self.in_flight_speed_ms:
            result = {"team": "in_flight", "player_id": None, "confidence": 1.0}
            self._push_history(result)
            return result

        puck_rx, puck_ry = puck_state.rink_pos

        # Find player closest to puck in rink space
        best_id:   Optional[int]   = None
        best_dist: float           = float("inf")

        for pinfo in player_positions:
            pid = pinfo.get("track_id")
            rpos = pinfo.get("rink_pos")
            if pid is None or rpos is None:
                continue
            dist = float(np.hypot(rpos[0] - puck_rx, rpos[1] - puck_ry))
            if dist < best_dist:
                best_dist = dist
                best_id = pid

        if best_id is None or best_dist > self.possession_radius_m:
            return no_possession

        # Confidence scales with proximity (1.0 at 0 m, 0.0 at possession_radius_m)
        prox_conf = max(0.0, 1.0 - best_dist / self.possession_radius_m)
        # Blend with puck detection confidence
        conf = prox_conf * puck_state.confidence

        team_label = (teams or {}).get(best_id)

        result = {
            "team":       team_label,
            "player_id":  best_id,
            "confidence": round(conf, 3),
        }
        self._push_history(result)
        return result

    def _push_history(self, result: Dict) -> None:
        self._history.append(result)
        if len(self._history) > self._history_len:
            self._history.pop(0)

    def smooth_result(self) -> Dict:
        """
        Return temporally smoothed possession result via majority vote
        over the last N frames (useful for display / downstream logic).
        """
        if not self._history:
            return {"team": None, "player_id": None, "confidence": 0.0}

        from collections import Counter
        pid_counter: Counter = Counter()
        for r in self._history:
            if r["player_id"] is not None:
                pid_counter[r["player_id"]] += 1

        if not pid_counter:
            return {"team": None, "player_id": None, "confidence": 0.0}

        best_pid, _ = pid_counter.most_common(1)[0]
        # Return the latest result for that player
        for r in reversed(self._history):
            if r["player_id"] == best_pid:
                return r
        return self._history[-1]


# ── Standalone test script ────────────────────────────────────────────────────

def _run_test(video_path: str, start_frame: int = 20000, n_frames: int = 200) -> None:
    """
    Test on game1.mp4: process n_frames starting at start_frame.
    Shows detections with confidence and possession assignments.
    Reports detection rate and blobs/frame.
    Saves annotated sample frame to /tmp/puck_final.jpg.
    """
    import sys

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}", file=sys.stderr)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video: {video_path}  |  {total_frames} frames  |  {fps:.1f} fps")

    # Build ROI from video so detections are restricted to ice surface
    roi = None
    if _ROI_OK:
        print("Building rink ROI from video...")
        try:
            roi = create_roi(video_path=video_path)
            print(f"ROI ready: {len(roi.polygon)} vertices" if roi.polygon is not None else "ROI: pass-through")
        except Exception as exc:
            print(f"[WARN] ROI build failed: {exc} — running without ROI mask")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    tracker    = PuckTracker(fps=fps, roi=roi)
    possession = PuckPossession()

    detected_count   = 0
    occluded_count   = 0
    searching_count  = 0
    total_blobs      = 0
    sample_vis       = None   # annotated frame saved to /tmp/puck_tuned.jpg
    sample_frame_idx = n_frames // 2  # capture mid-run sample

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] Stream ended at frame {start_frame + i}")
            break

        timestamp = (start_frame + i) / fps

        # No player bboxes in standalone test — pass empty list
        state = tracker.update(frame, player_bboxes=[], timestamp=timestamp)
        total_blobs += tracker._last_blob_count

        # Possession with no player data
        poss = possession.update(state, player_positions=[], teams={})

        # Count states
        if state.state == "detected":
            detected_count += 1
        elif state.state == "occluded":
            occluded_count += 1
        else:
            searching_count += 1

        # Annotate frame
        vis = tracker.draw(frame, state)

        # Draw ROI polygon if available
        if roi is not None:
            vis = roi.draw(vis, color=(0, 200, 255), thickness=1)

        # Possession overlay
        poss_str = "no possession"
        if poss["team"] == "in_flight":
            poss_str = "IN FLIGHT"
        elif poss["player_id"] is not None:
            poss_str = f"team={poss['team']} id={poss['player_id']} conf={poss['confidence']:.2f}"
        cv2.putText(vis, poss_str, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(vis, f"frame {start_frame + i}  blobs={tracker._last_blob_count}", (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if i == sample_frame_idx:
            sample_vis = vis.copy()

        cv2.imshow("IceIQ — Puck Detection", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quit by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save sample frame
    if sample_vis is not None:
        cv2.imwrite("/tmp/puck_final.jpg", sample_vis)
        print("Sample frame saved → /tmp/puck_final.jpg")

    processed = detected_count + occluded_count + searching_count
    if processed > 0:
        det_rate = (detected_count + occluded_count) / processed * 100
        avg_blobs = total_blobs / processed
    else:
        det_rate = 0.0
        avg_blobs = 0.0

    print("\n── Puck Detection Report ──────────────────────────")
    print(f"  Frames processed : {processed}")
    print(f"  Avg blobs/frame  : {avg_blobs:.1f}  (target: 1–5)")
    print(f"  Detected         : {detected_count}  ({100*detected_count/max(1,processed):.1f}%)")
    print(f"  Occluded (KF)    : {occluded_count}  ({100*occluded_count/max(1,processed):.1f}%)")
    print(f"  Searching        : {searching_count}  ({100*searching_count/max(1,processed):.1f}%)")
    print(f"  Overall track rate: {det_rate:.1f}%")
    print("───────────────────────────────────────────────────")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="IceIQ puck detection test")
    p.add_argument("--video",  default="data/videos/game1.mp4")
    p.add_argument("--start",  type=int, default=20000)
    p.add_argument("--frames", type=int, default=200)
    args = p.parse_args()
    _run_test(args.video, args.start, args.frames)
