"""
pipeline/kinematics.py — Per-player kinematics for IceIQ Hockey Analysis

Accumulates rink-space positions per track ID and provides:
  - Kalman-smoothed velocity / speed (filterpy KalmanFilter)
  - Cumulative distance skated (metres)
  - Sprint event detection  (speed > sprint_threshold_kmh)
  - Zone heatmap            (time-on-ice per grid cell, seconds)
  - Summary statistics dict

Coordinate system (from homography.py):
  Origin : centre ice
  X-axis : across width  [-12.95, +12.95] m
  Y-axis : along length  [-30.48, +30.48] m  (far end negative)

Processing cadence: 4 fps (configured in configs/pipeline_config.yaml)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ── filterpy availability ────────────────────────────────────────────────────
try:
    from filterpy.kalman import KalmanFilter as _KF
    _FILTERPY_AVAILABLE = True
except ImportError:
    _FILTERPY_AVAILABLE = False
    logger.warning(
        "filterpy not installed — falling back to finite-difference velocity. "
        "Install with: pip install filterpy"
    )

# ── Load pipeline config ─────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "pipeline_config.yaml"


def _load_cfg() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as fh:
            return yaml.safe_load(fh) or {}
    return {}


_cfg = _load_cfg()
_video_cfg    = _cfg.get("video",      {})
_rink_cfg     = _video_cfg.get("rink", {})
_analysis_cfg = _cfg.get("analysis",   {})
_kalman_cfg   = _cfg.get("kalman",     {})
_kinem_cfg    = _cfg.get("kinematics", {})

# Module-level defaults (also used as constructor defaults)
FPS:                    float = float(_video_cfg.get("fps",                    4))
RINK_LENGTH:            float = float(_rink_cfg.get("length",                 60.96))
RINK_WIDTH:             float = float(_rink_cfg.get("width",                  25.9))
SPRINT_THRESHOLD_KMH:   float = float(_analysis_cfg.get("sprint_threshold_kmh",  16.0))
STATIONARY_THRESHOLD_KMH: float = float(_analysis_cfg.get("stationary_threshold_kmh", 2.0))
SPEED_WINDOW:           int   = int(_analysis_cfg.get("speed_window_frames",  5))
PROCESS_NOISE:          float = float(_kalman_cfg.get("process_noise",        0.01))
MEASUREMENT_NOISE:      float = float(_kalman_cfg.get("measurement_noise",    0.06))

# Physical sanity cap: U-12 max ≈ 6.1 m/s (22 km/h)
_MAX_SPEED_MS: float = float(_kinem_cfg.get("max_speed_ms", 6.1))

# Teleport guard: skip position jumps > 5 m between consecutive 4-fps frames
_TELEPORT_THRESHOLD_M: float = 5.0


# ── Per-track state ──────────────────────────────────────────────────────────
@dataclass
class _TrackState:
    """Mutable per-track accumulator (internal)."""
    positions:      List[Tuple[float, float]] = field(default_factory=list)
    timestamps:     List[float]               = field(default_factory=list)
    velocities:     List[Tuple[float, float]] = field(default_factory=list)  # m/s
    speeds_kmh:     List[float]               = field(default_factory=list)
    total_distance: float                     = 0.0
    kf:             Optional[object]          = None   # _KF instance


def _build_kalman(dt: float, process_noise: float, measurement_noise: float) -> "_KF":
    """
    Constant-velocity 2-D Kalman filter.
    State:        [x, y, vx, vy]  (position m, velocity m/s)
    Measurement:  [x, y]
    """
    kf = _KF(dim_x=4, dim_z=2)
    kf.F = np.array(
        [[1, 0, dt, 0],
         [0, 1,  0, dt],
         [0, 0,  1,  0],
         [0, 0,  0,  1]],
        dtype=float,
    )
    kf.H = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]],
        dtype=float,
    )
    kf.Q = np.eye(4) * process_noise
    kf.R = np.eye(2) * measurement_noise
    kf.P = np.eye(4) * 10.0   # broad initial uncertainty
    return kf


# ── Main class ───────────────────────────────────────────────────────────────
class PlayerKinematics:
    """
    Accumulates rink-space positions per track ID (from HockeyTracker) and
    exposes smoothed velocity, distance, sprint events, heatmap, and stats.

    Typical usage::

        kinematics = PlayerKinematics()
        for frame_idx, frame in enumerate(frames):
            tracks = tracker.update(detections, frame)
            for tr in tracks:
                px, py = bbox_centre(tr["bbox"])
                rx, ry = homography.pixel_to_rink(px, py)
                t = frame_idx / source_fps
                kinematics.update(tr["track_id"], (rx, ry), t)

        for tid in kinematics.track_ids:
            print(tid, kinematics.get_stats(tid))
    """

    def __init__(
        self,
        fps:                      float = FPS,
        rink_length:              float = RINK_LENGTH,
        rink_width:               float = RINK_WIDTH,
        sprint_threshold_kmh:     float = SPRINT_THRESHOLD_KMH,
        stationary_threshold_kmh: float = STATIONARY_THRESHOLD_KMH,
        process_noise:            float = PROCESS_NOISE,
        measurement_noise:        float = MEASUREMENT_NOISE,
    ) -> None:
        self.fps                      = fps
        self.dt                       = 1.0 / fps
        self.rink_length              = rink_length
        self.rink_width               = rink_width
        self.sprint_threshold_kmh     = sprint_threshold_kmh
        self.stationary_threshold_kmh = stationary_threshold_kmh
        self.process_noise            = process_noise
        self.measurement_noise        = measurement_noise

        self._tracks: Dict[int, _TrackState] = defaultdict(_TrackState)

    # ── Core accumulation ────────────────────────────────────────────────────

    def update(
        self,
        track_id: int,
        rink_pos: Tuple[float, float],
        timestamp: float,
    ) -> None:
        """
        Record a new observation.

        Parameters
        ----------
        track_id  : stable ID from HockeyTracker.update()
        rink_pos  : (rx, ry) in metres, origin = centre ice
        timestamp : seconds since clip start (e.g. frame_index / source_fps)
        """
        state = self._tracks[track_id]
        rx, ry = float(rink_pos[0]), float(rink_pos[1])

        # ── Teleport guard ────────────────────────────────────────────────────
        # Track discontinuities (re-ID swaps, occluded re-entry) cause instant
        # position jumps that blow up the velocity estimate. Skip the update
        # entirely when the jump is physically impossible at 4 fps.
        if state.positions:
            prev_rx, prev_ry = state.positions[-1]
            step_dist = float(np.hypot(rx - prev_rx, ry - prev_ry))
            if step_dist > _TELEPORT_THRESHOLD_M:
                logger.debug(
                    "Track %d: teleport %.2f m detected — skipping update", track_id, step_dist
                )
                return

        # ── Kalman predict + update ───────────────────────────────────────────
        if _FILTERPY_AVAILABLE:
            if state.kf is None:
                state.kf = _build_kalman(self.dt, self.process_noise, self.measurement_noise)
                state.kf.x = np.array([[rx], [ry], [0.0], [0.0]])
            state.kf.predict()
            state.kf.update(np.array([[rx], [ry]]))
            vx = float(state.kf.x[2])
            vy = float(state.kf.x[3])
        else:
            # Finite-difference fallback
            if state.positions:
                prev_rx, prev_ry = state.positions[-1]
                prev_t = state.timestamps[-1]
                elapsed = (timestamp - prev_t) if (timestamp - prev_t) > 1e-6 else self.dt
                vx = (rx - prev_rx) / elapsed
                vy = (ry - prev_ry) / elapsed
            else:
                vx, vy = 0.0, 0.0

        # Clamp to physical limit
        speed_ms = float(np.hypot(vx, vy))
        if speed_ms > _MAX_SPEED_MS:
            scale = _MAX_SPEED_MS / speed_ms
            vx *= scale
            vy *= scale
            speed_ms = _MAX_SPEED_MS

        speed_kmh = speed_ms * 3.6

        # ── Distance accumulation ─────────────────────────────────────────────
        if state.positions:
            prev_rx, prev_ry = state.positions[-1]
            step_dist = float(np.hypot(rx - prev_rx, ry - prev_ry))
            # Discard teleport artefacts (> max possible in one tick)
            if step_dist <= _MAX_SPEED_MS * self.dt:
                state.total_distance += step_dist

        # ── Append to history ─────────────────────────────────────────────────
        state.positions.append((rx, ry))
        state.timestamps.append(float(timestamp))
        state.velocities.append((vx, vy))
        state.speeds_kmh.append(speed_kmh)

    # ── Public getters ───────────────────────────────────────────────────────

    def get_velocity(self, track_id: int) -> Tuple[float, float, float]:
        """
        Latest Kalman-smoothed velocity for *track_id*.

        Returns
        -------
        (vx_ms, vy_ms, speed_kmh)
            Velocity components in m/s and scalar speed in km/h.
            The speed is the rolling mean of the last SPEED_WINDOW observations
            to reduce single-frame noise.
            Returns (0.0, 0.0, 0.0) for unknown or empty tracks.
        """
        state = self._tracks.get(track_id)
        if not state or not state.velocities:
            return 0.0, 0.0, 0.0
        vx, vy = state.velocities[-1]
        # Smooth speed over a rolling window (matches speed_window_frames config)
        window = state.speeds_kmh[-SPEED_WINDOW:]
        speed_kmh = float(np.mean(window))
        return vx, vy, speed_kmh

    def get_distance(self, track_id: int) -> float:
        """Total distance skated in metres."""
        state = self._tracks.get(track_id)
        return state.total_distance if state else 0.0

    def get_sprint_events(
        self, track_id: int
    ) -> List[Tuple[float, float, float]]:
        """
        Sprint episodes where speed exceeds sprint_threshold_kmh.

        Returns
        -------
        List of (start_time_s, end_time_s, max_speed_kmh).
        Empty list if track unknown or no sprints detected.
        """
        state = self._tracks.get(track_id)
        if not state or not state.speeds_kmh:
            return []

        events: List[Tuple[float, float, float]] = []
        in_sprint   = False
        sprint_start = 0.0
        sprint_max   = 0.0

        for t, spd in zip(state.timestamps, state.speeds_kmh):
            if spd >= self.sprint_threshold_kmh:
                if not in_sprint:
                    in_sprint    = True
                    sprint_start = t
                    sprint_max   = spd
                else:
                    sprint_max = max(sprint_max, spd)
            else:
                if in_sprint:
                    events.append((sprint_start, t, sprint_max))
                    in_sprint  = False
                    sprint_max = 0.0

        # Close any sprint that extends to the final observation
        if in_sprint and state.timestamps:
            events.append((sprint_start, state.timestamps[-1], sprint_max))

        return events

    def get_heatmap(
        self,
        track_id: int,
        grid_size: Tuple[int, int] = (30, 13),
    ) -> np.ndarray:
        """
        2-D array of time-on-ice per zone (seconds).

        Parameters
        ----------
        track_id  : track to query
        grid_size : (n_along_length, n_along_width)
                    default (30, 13) → ~2.0 m × ~2.0 m cells

        Returns
        -------
        np.ndarray of shape *grid_size*, dtype float64.
        grid[i, j] — row i along rink length (Y axis, far-to-near),
                      col j along rink width  (X axis, left-to-right).
        """
        n_len, n_wid = grid_size
        grid = np.zeros((n_len, n_wid), dtype=float)

        state = self._tracks.get(track_id)
        if not state or not state.positions:
            return grid

        half_len = self.rink_length / 2.0   # 30.48 m
        half_wid = self.rink_width  / 2.0   # 12.95 m
        cell_len = self.rink_length / n_len
        cell_wid = self.rink_width  / n_wid

        for rx, ry in state.positions:
            # Clamp to rink boundaries
            ry_c = max(-half_len, min(half_len - 1e-9, ry))
            rx_c = max(-half_wid, min(half_wid - 1e-9, rx))

            # Map [-half_len, +half_len) → [0, n_len)
            i = int((ry_c + half_len) / cell_len)
            j = int((rx_c + half_wid) / cell_wid)

            i = min(i, n_len - 1)
            j = min(j, n_wid - 1)

            grid[i, j] += self.dt   # seconds per processed frame

        return grid

    def get_stats(self, track_id: int) -> dict:
        """
        Summary statistics for *track_id*.

        Returns
        -------
        dict with keys:
          avg_speed      (km/h, moving periods only)
          max_speed      (km/h)
          total_distance (m)
          sprint_count   (int)
          time_in_zones  dict{defensive, neutral, offensive} (seconds)
        """
        state = self._tracks.get(track_id)
        empty = {
            "avg_speed":       0.0,
            "max_speed":       0.0,
            "total_distance":  0.0,
            "sprint_count":    0,
            "time_in_zones":   {"defensive": 0.0, "neutral": 0.0, "offensive": 0.0},
        }
        if not state or not state.speeds_kmh:
            return empty

        speeds  = np.array(state.speeds_kmh)
        heatmap = self.get_heatmap(track_id)            # (30, 13)
        sprints = self.get_sprint_events(track_id)

        # Zone thirds along rink length axis (i-axis of heatmap)
        n_len = heatmap.shape[0]
        t1, t2 = n_len // 3, 2 * (n_len // 3)
        time_in_zones = {
            "defensive": float(heatmap[:t1].sum()),
            "neutral":   float(heatmap[t1:t2].sum()),
            "offensive": float(heatmap[t2:].sum()),
        }

        moving_mask = speeds >= self.stationary_threshold_kmh
        avg_speed = float(speeds[moving_mask].mean()) if moving_mask.any() else 0.0

        return {
            "avg_speed":       avg_speed,
            "max_speed":       float(speeds.max()),
            "total_distance":  self.get_distance(track_id),
            "sprint_count":    len(sprints),
            "time_in_zones":   time_in_zones,
        }

    # ── Utility ──────────────────────────────────────────────────────────────

    def reset(self, track_id: Optional[int] = None) -> None:
        """Clear state for *track_id*, or all tracks if None."""
        if track_id is None:
            self._tracks.clear()
        else:
            self._tracks.pop(track_id, None)

    @property
    def track_ids(self) -> List[int]:
        """All track IDs that have at least one observation."""
        return list(self._tracks.keys())

    def observation_count(self, track_id: int) -> int:
        """Number of update() calls recorded for *track_id*."""
        state = self._tracks.get(track_id)
        return len(state.positions) if state else 0
