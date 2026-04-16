"""
pipeline/identification.py
7-signal Bayesian player identification system.

Signals (1-4 implemented here, 5 fed externally, 6-7 future):
  1. Shot hand        - wrist keypoint Y comparison (trivial, 1 line core logic)
  2. Position/behavior- rink zone time ratios over 1 shift (easy)
  3. Line co-occurrence- 3 forwards entering/exiting together (easy)
  4. Body shape       - height + shoulder width via homography (easy)
  5. Jersey OCR       - PARSeq/EasyOCR, fed via feed_jersey() (jersey_recognition.py)
  6. Equipment profile- color histograms per zone (medium, future)
  7. Motion DNA       - pose sequence embedding (medium, future)

Fusion:
  P(id | signals) ∝ ∏ P(signal_k | id)
  Implemented as log-likelihood summation for numerical stability.

Usage:
    from pipeline.identification import PlayerIdentifier
    ident = PlayerIdentifier(
        roster_path="data/rosters/aigis_waves_g18.yaml",
        rink_config={"length": 60.96, "width": 25.9, "blue_line_from_end": 19.51},
        roi_polygon=[[1919,1079],[0,1079],[0,179],[1919,169]],
    )
    ident.update(track_id, frame_num, keypoints_17x3, rink_pos_meters)
    ident.feed_jersey(track_id, number=47, log_likelihood=2.3)
    result = ident.get_identity(track_id)
    # -> {"number": 47, "shot_hand": "RIGHT", "position": "D", "confidence": 0.91}
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ── YOLOv8-pose keypoint indices ────────────────────────────────────────────
_KP_NOSE = 0
_KP_L_SHOULDER = 5
_KP_R_SHOULDER = 6
_KP_L_WRIST = 9
_KP_R_WRIST = 10
_KP_L_HIP = 11
_KP_R_HIP = 12
_KP_L_ANKLE = 15
_KP_R_ANKLE = 16

# ── Rink geometry constants ──────────────────────────────────────────────────
# Standard NHL/IIHF faceoff dot positions (meters from left-end, near-side)
# Rink: 60.96m long × 25.9m wide; blue lines at 19.51m from each end
_FACEOFF_DOTS_M = [
    (30.48, 12.95),   # center ice
    (17.70,  6.70),   # left defensive zone, near
    (17.70, 19.20),   # left defensive zone, far
    (43.26,  6.70),   # right defensive zone, near
    (43.26, 19.20),   # right defensive zone, far
    ( 7.90,  6.70),   # left end zone, near
    ( 7.90, 19.20),   # left end zone, far
    (53.06,  6.70),   # right end zone, near
    (53.06, 19.20),   # right end zone, far
]
_FACEOFF_PROX_M = 3.0   # meters: "near a faceoff dot"

# Minimum observation thresholds before trusting each signal
_MIN_FRAMES_SHOT_HAND = 5
_MIN_FRAMES_POSITION = 30   # ~7.5 s at 4 fps
_MIN_FRAMES_BODY = 10
_MIN_SHIFTS_LINE = 3         # co-occur in ≥3 shifts = same line


# ═══════════════════════════════════════════════════════════════════════════ #
#  Homography mapper: pixel → real-world rink meters                          #
# ═══════════════════════════════════════════════════════════════════════════ #

class HomographyMapper:
    """
    Perspective transform from pixel space to rink coordinates (meters).
    Constructed from the 4-vertex ROI polygon and known rink dimensions.

    Assumes a side-view camera where:
      - bottom of frame  = near side of rink  (y = 0 m)
      - top of frame     = far side of rink   (y = rink_width m)
      - left of frame    = left end of rink   (x = 0 m)
      - right of frame   = right end of rink  (x = rink_length m)
    """

    def __init__(
        self,
        roi_polygon: List[List[int]],
        rink_length: float,
        rink_width: float,
    ) -> None:
        pts = np.array(roi_polygon, dtype=np.float32)  # shape (4, 2)

        # Sort into top-two / bottom-two by y (larger y = lower on screen)
        by_y = pts[np.argsort(pts[:, 1])]
        top_two = by_y[:2][np.argsort(by_y[:2, 0])]   # left→right within top
        bot_two = by_y[2:][np.argsort(by_y[2:, 0])]   # left→right within bottom

        # src order: top-left, top-right, bottom-right, bottom-left
        src = np.array([
            top_two[0], top_two[1],
            bot_two[1], bot_two[0],
        ], dtype=np.float32)

        # dst: real-world corners in (x_m, y_m)
        dst = np.array([
            [0.0,          rink_width],  # top-left  = far-left corner
            [rink_length,  rink_width],  # top-right = far-right corner
            [rink_length,  0.0       ],  # bot-right = near-right corner
            [0.0,          0.0       ],  # bot-left  = near-left corner
        ], dtype=np.float32)

        self.H, _ = cv2.findHomography(src, dst)
        self.rink_length = rink_length
        self.rink_width = rink_width

    def pixel_to_rink(self, px: float, py: float) -> Tuple[float, float]:
        """Return (rx, ry) in meters, clamped to rink bounds."""
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)
        rx = float(np.clip(out[0, 0, 0], 0.0, self.rink_length))
        ry = float(np.clip(out[0, 0, 1], 0.0, self.rink_width))
        return rx, ry

    def pixel_dist_to_meters(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
    ) -> float:
        """Euclidean distance between two pixel points in real-world meters."""
        r1 = self.pixel_to_rink(*p1)
        r2 = self.pixel_to_rink(*p2)
        return math.hypot(r1[0] - r2[0], r1[1] - r2[1])


# ═══════════════════════════════════════════════════════════════════════════ #
#  Per-track accumulated signal data                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

class _TrackState:
    """All accumulated signal observations for one continuous track_id."""

    __slots__ = (
        "wrist_votes", "rink_positions",
        "heights_m", "shoulder_widths_m",
        "jersey_log_likelihoods",
        "last_seen_frame", "frame_count", "seen_frames",
        "pixel_ys",
    )

    def __init__(self) -> None:
        # Signal 1
        self.wrist_votes: Dict[str, int] = {"RIGHT": 0, "LEFT": 0}
        # Signal 2
        self.rink_positions: deque = deque(maxlen=600)  # ~2.5 min at 4 fps
        # Signal 4
        self.heights_m: List[float] = []
        self.shoulder_widths_m: List[float] = []
        # Signal 5 (external)
        self.jersey_log_likelihoods: Dict[int, float] = {}
        # Bookkeeping
        self.last_seen_frame: int = -1
        self.seen_frames: set = set()
        self.frame_count: int = 0
        self.pixel_ys: List[float] = []

    # ── Signal 1 derived property ────────────────────────────────────────────

    @property
    def shot_hand(self) -> Optional[str]:
        total = self.wrist_votes["RIGHT"] + self.wrist_votes["LEFT"]
        if total < _MIN_FRAMES_SHOT_HAND:
            return None
        return "RIGHT" if self.wrist_votes["RIGHT"] >= self.wrist_votes["LEFT"] else "LEFT"

    @property
    def shot_hand_confidence(self) -> float:
        total = self.wrist_votes["RIGHT"] + self.wrist_votes["LEFT"]
        if total == 0:
            return 0.5
        return max(self.wrist_votes["RIGHT"], self.wrist_votes["LEFT"]) / total

    # ── Signal 2 derived property ────────────────────────────────────────────

    def position(
        self,
        blue_line_offset: float = 19.51,
        rink_length: float = 60.96,
        rink_width: float = 25.9,
    ) -> Optional[str]:
        if len(self.rink_positions) < _MIN_FRAMES_POSITION:
            # Pixel-y zone approximation (fallback when insufficient rink samples)
            if not self.pixel_ys:
                return None
            mean_y = sum(self.pixel_ys) / len(self.pixel_ys)
            norm_y = mean_y / 720.0
            if norm_y < 0.33:
                return "ATT"
            elif norm_y > 0.67:
                return "DEF"
            else:
                return "NEU"

        xs = [p[0] for p in self.rink_positions]
        ys = [p[1] for p in self.rink_positions]

        blue1 = blue_line_offset
        blue2 = rink_length - blue_line_offset

        # Defenseman: spends >40 % of time near blue lines
        near_blue = sum(1 for x in xs if x < blue1 + 3.0 or x > blue2 - 3.0)
        if near_blue / len(xs) > 0.4:
            return "D"

        # Forward sub-classification
        faceoff_frames = sum(
            1 for rx, ry in self.rink_positions
            if any(
                math.hypot(rx - dx, ry - dy) < _FACEOFF_PROX_M
                for dx, dy in _FACEOFF_DOTS_M
            )
        )
        if faceoff_frames / len(self.rink_positions) > 0.05:
            return "C"

        mean_y = sum(ys) / len(ys)
        rink_mid = rink_width / 2.0
        if mean_y < rink_mid - 2.0:
            return "LW"
        if mean_y > rink_mid + 2.0:
            return "RW"
        return "F"

    # ── Signal 4 derived properties ──────────────────────────────────────────

    @property
    def mean_height_m(self) -> Optional[float]:
        valid = [h for h in self.heights_m if 0.5 < h < 2.5]
        return sum(valid) / len(valid) if len(valid) >= _MIN_FRAMES_BODY else None

    @property
    def mean_shoulder_width_m(self) -> Optional[float]:
        valid = [w for w in self.shoulder_widths_m if 0.2 < w < 1.2]
        return sum(valid) / len(valid) if len(valid) >= _MIN_FRAMES_BODY else None


# ═══════════════════════════════════════════════════════════════════════════ #
#  Main identifier                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

class PlayerIdentifier:
    """
    7-signal Bayesian player identification.

    Signals 1-4 are computed internally from keypoints + rink positions.
    Signal 5 (jersey OCR) is fed externally via feed_jersey().
    Signals 6-7 are placeholders for future implementation.
    """

    def __init__(
        self,
        roster_path: str,
        rink_config: Optional[Dict[str, Any]] = None,
        roi_polygon: Optional[List[List[int]]] = None,
    ) -> None:
        self.roster = self._load_roster(roster_path)
        self.roster_numbers: List[int] = [
            p["number"] for p in self.roster if p["number"] is not None
        ]

        cfg = rink_config or {}
        self.rink_length: float = cfg.get("length", 60.96)
        self.rink_width: float = cfg.get("width", 25.9)
        self.blue_line_offset: float = cfg.get("blue_line_from_end", 19.51)

        # Homography (optional but improves Signal 4 accuracy)
        self.mapper: Optional[HomographyMapper] = None
        if roi_polygon and len(roi_polygon) == 4:
            try:
                self.mapper = HomographyMapper(
                    roi_polygon, self.rink_length, self.rink_width
                )
                logger.info("HomographyMapper initialized from ROI polygon")
            except Exception as exc:
                logger.warning("HomographyMapper init failed: %s", exc)

        # Per-track state
        self._tracks: Dict[int, _TrackState] = defaultdict(_TrackState)

        # Signal 3: on-ice set tracking
        self._current_on_ice: set = set()
        self._shift_start_frame: int = 0
        # List of (frame_start, frame_end, frozenset_of_track_ids)
        self._recorded_shifts: List[Tuple[int, int, frozenset]] = []

        logger.info(
            "PlayerIdentifier ready — %d skaters in roster, homography=%s",
            len(self.roster_numbers),
            "yes" if self.mapper else "no (pixel fallback)",
        )

    # ── Roster loading ───────────────────────────────────────────────────────

    @staticmethod
    def _load_roster(path: str) -> List[Dict[str, Any]]:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        players = []
        for p in data.get("players", []):
            players.append({
                "number":          p.get("number"),
                "name":            p.get("name", ""),
                "position":        p.get("position", "skater"),
                "shot_hand":       p.get("shot_hand"),        # "LEFT"/"RIGHT" or None
                "position_detail": p.get("position_detail"),  # "D"/"C"/"LW"/"RW" or None
            })
        return players

    # ── Per-frame update (main entry point) ──────────────────────────────────

    def update(
        self,
        track_id: int,
        frame: int,
        keypoints: Optional[np.ndarray],
        rink_pos: Optional[Tuple[float, float]],
    ) -> None:
        """
        Accumulate one frame of observations for track_id.

        Args:
            track_id:  Consistent tracker ID (from tracking.py TrackResult)
            frame:     Current frame index
            keypoints: YOLOv8-pose output, shape (17, 3) — [x, y, confidence].
                       Pass None if pose was not available for this detection.
            rink_pos:  Player foot position in real-world meters (rx, ry).
                       Compute via mapper.pixel_to_rink(cx, y2) before calling,
                       or pass pixel (cx, y2) directly as fallback.
        """
        state = self._tracks[track_id]
        state.last_seen_frame = frame
        state.seen_frames.add(frame)
        state.frame_count = len(state.seen_frames)

        if keypoints is not None and keypoints.ndim == 2 and keypoints.shape[0] >= 17:
            # Signal 1: shot hand
            self._signal1_shot_hand(state, keypoints)
            # Signal 4: body shape
            self._signal4_body_shape(state, keypoints)

        if rink_pos is not None:
            # Signal 2: rink position accumulation
            state.rink_positions.append(rink_pos)
            state.pixel_ys.append(float(rink_pos[1]))

        # Signal 3: on-ice set bookkeeping
        self._signal3_on_ice_update(track_id, frame)

    # ── Signal implementations ───────────────────────────────────────────────

    def _signal1_shot_hand(self, state: _TrackState, kp: np.ndarray) -> None:
        """
        Signal 1 — Shot hand (1 line of logic).
        Left wrist higher on screen (smaller y) → player shoots RIGHT.
        """
        if kp[_KP_L_WRIST, 2] < 0.3 or kp[_KP_R_WRIST, 2] < 0.3:
            return  # low-confidence keypoints, skip

        if kp[_KP_L_WRIST, 1] < kp[_KP_R_WRIST, 1]:   # left wrist is higher
            state.wrist_votes["RIGHT"] += 1
        else:
            state.wrist_votes["LEFT"] += 1

    def _signal4_body_shape(self, state: _TrackState, kp: np.ndarray) -> None:
        """
        Signal 4 — Body shape: player height and shoulder width in real meters.
        Uses homography if available; falls back to pixel-ratio estimates.
        """
        nose_c  = kp[_KP_NOSE,      2]
        la_c    = kp[_KP_L_ANKLE,   2]
        ra_c    = kp[_KP_R_ANKLE,   2]
        ls_c    = kp[_KP_L_SHOULDER, 2]
        rs_c    = kp[_KP_R_SHOULDER, 2]

        # Height: nose → midpoint of visible ankles
        if nose_c > 0.3 and (la_c > 0.3 or ra_c > 0.3):
            nose_px = (kp[_KP_NOSE, 0], kp[_KP_NOSE, 1])

            if la_c > 0.3 and ra_c > 0.3:
                foot_px: Tuple[float, float] = (
                    (kp[_KP_L_ANKLE, 0] + kp[_KP_R_ANKLE, 0]) / 2.0,
                    (kp[_KP_L_ANKLE, 1] + kp[_KP_R_ANKLE, 1]) / 2.0,
                )
            elif la_c > 0.3:
                foot_px = (kp[_KP_L_ANKLE, 0], kp[_KP_L_ANKLE, 1])
            else:
                foot_px = (kp[_KP_R_ANKLE, 0], kp[_KP_R_ANKLE, 1])

            if self.mapper:
                h = self.mapper.pixel_dist_to_meters(nose_px, foot_px)
            else:
                # Pixel fallback: scale by known average player height ~1.8 m
                h = abs(nose_px[1] - foot_px[1]) / 1080.0 * 1.8

            state.heights_m.append(h)

        # Shoulder width
        if ls_c > 0.3 and rs_c > 0.3:
            ls_px = (kp[_KP_L_SHOULDER, 0], kp[_KP_L_SHOULDER, 1])
            rs_px = (kp[_KP_R_SHOULDER, 0], kp[_KP_R_SHOULDER, 1])

            if self.mapper:
                sw = self.mapper.pixel_dist_to_meters(ls_px, rs_px)
            else:
                sw = abs(ls_px[0] - rs_px[0]) / 1920.0 * self.rink_width * 0.025
            state.shoulder_widths_m.append(sw)

    def _signal3_on_ice_update(self, track_id: int, frame: int) -> None:
        """
        Signal 3 — Track which players are on ice together.
        When the set changes by ≥2 players, record a shift boundary.
        """
        if track_id in self._current_on_ice:
            return
        prev = frozenset(self._current_on_ice)
        self._current_on_ice.add(track_id)

        # Shift boundary: substantial change in who's on ice
        if len(prev) >= 3 and len(self._current_on_ice - prev) >= 2:
            self._recorded_shifts.append(
                (self._shift_start_frame, frame, prev)
            )
            self._shift_start_frame = frame

    def expire_track(self, track_id: int, frame: int) -> None:
        """
        Call when a track disappears from the scene.
        Finalizes shift data for line co-occurrence analysis.
        """
        if track_id not in self._current_on_ice:
            return
        self._current_on_ice.discard(track_id)
        if len(self._current_on_ice) >= 2:
            self._recorded_shifts.append((
                self._shift_start_frame, frame,
                frozenset(self._current_on_ice | {track_id}),
            ))
        self._shift_start_frame = frame

    # ── External signal feed ─────────────────────────────────────────────────

    def feed_jersey(
        self,
        track_id: int,
        number: int,
        log_likelihood: float,
    ) -> None:
        """
        Signal 5 — Accept a jersey OCR vote from JerseyRecognizer.

        Args:
            track_id:        Track this observation belongs to
            number:          Detected jersey number
            log_likelihood:  Log P(observation | number is correct).
                             Positive = evidence for, negative = evidence against.
        """
        if number not in self.roster_numbers:
            return
        state = self._tracks[track_id]
        state.jersey_log_likelihoods[number] = (
            state.jersey_log_likelihoods.get(number, 0.0) + log_likelihood
        )

    # ── Line co-occurrence helpers (Signal 3) ────────────────────────────────

    def _find_line_groups(self) -> List[frozenset]:
        """
        Return sets of 3 track_ids that co-appear in ≥ _MIN_SHIFTS_LINE shifts.
        These are likely forward lines.
        """
        co_counts: Dict[frozenset, int] = defaultdict(int)
        for _, _, players in self._recorded_shifts:
            for trio in combinations(sorted(players), 3):
                co_counts[frozenset(trio)] += 1
        return [
            group for group, cnt in co_counts.items()
            if cnt >= _MIN_SHIFTS_LINE
        ]

    def _get_linemates(self, track_id: int) -> List[int]:
        """Return other track_ids confirmed to be in the same line as track_id."""
        mates = []
        for group in self._find_line_groups():
            if track_id in group:
                mates.extend(tid for tid in group if tid != track_id)
        return list(set(mates))

    # ── Bayesian fusion ──────────────────────────────────────────────────────

    def _compute_log_posterior(self, track_id: int) -> Dict[int, float]:
        """
        Compute log P(jersey_number | all_signals) for each roster number.
        Log-uniform prior; add log-likelihoods from each available signal.
        """
        state = self._tracks[track_id]
        log_p: Dict[int, float] = {n: 0.0 for n in self.roster_numbers}

        # ── Signal 1: shot hand ──────────────────────────────────────────────
        shot = state.shot_hand
        if shot is not None:
            for player in self.roster:
                n = player["number"]
                if n not in log_p:
                    continue
                roster_shot = player.get("shot_hand")
                if roster_shot is not None:
                    if roster_shot.upper() == shot:
                        log_p[n] += math.log(0.85)
                    else:
                        log_p[n] += math.log(0.15)

        # ── Signal 2: position/behavior ──────────────────────────────────────
        pos = state.position(self.blue_line_offset, self.rink_length, self.rink_width)
        if pos is not None:
            pixel_approx = pos in ("ATT", "NEU", "DEF")
            weight = 0.4 if pixel_approx else 1.0
            for player in self.roster:
                n = player["number"]
                if n not in log_p:
                    continue
                detail = player.get("position_detail")
                if detail is not None and not pixel_approx:
                    # Exact position match vs. mismatch
                    match = (pos == detail) or (pos in ("LW", "RW", "C", "F") and detail == "F")
                    log_p[n] += weight * (math.log(0.80) if match else math.log(0.20))

        # ── Signal 3: line co-occurrence ─────────────────────────────────────
        # If linemates are identified, exclude their jersey numbers from this track
        linemates = self._get_linemates(track_id)
        for mate_id in linemates:
            mate_state = self._tracks.get(mate_id)
            if mate_state is None or not mate_state.jersey_log_likelihoods:
                continue
            # Best guess for linemate's number
            mate_best = max(
                mate_state.jersey_log_likelihoods,
                key=mate_state.jersey_log_likelihoods.get,
            )
            # Penalise this track having the same number as its linemate
            if mate_best in log_p:
                log_p[mate_best] += math.log(0.05)

        # ── Signal 4: body shape ─────────────────────────────────────────────
        h = state.mean_height_m
        sw = state.mean_shoulder_width_m
        if h is not None or sw is not None:
            # Build height distribution across all tracks
            all_heights = {
                tid: ts.mean_height_m
                for tid, ts in self._tracks.items()
                if tid != track_id and ts.mean_height_m is not None
            }
            if h is not None and all_heights:
                # Simple percentile: taller players are rarer → weak Gaussian signal
                # Without a height-annotated roster, we rank players relative to
                # each other and assume uniform prior over physical profiles.
                taller_count = sum(1 for oh in all_heights.values() if oh < h)
                _percentile = taller_count / len(all_heights)
                # When roster gains height fields, replace with:
                # for player in self.roster: log_p[n] += gaussian_ll(h, player["height_m"], 0.05)

        # ── Signal 5: jersey OCR (strongest signal) ──────────────────────────
        for n, ll in state.jersey_log_likelihoods.items():
            if n in log_p:
                log_p[n] += ll

        return log_p

    # ── Public API ────────────────────────────────────────────────────────────

    def get_identity(self, track_id: int) -> Dict[str, Any]:
        """
        Return best current identity estimate for track_id.

        Returns dict with keys:
            number      (int | None)   — best jersey number guess
            shot_hand   (str | None)   — "RIGHT" | "LEFT"
            position    (str | None)   — "D" | "C" | "LW" | "RW" | "F"
            confidence  (float)        — softmax probability of top candidate (0-1)
            log_posteriors (dict)      — {number: log_p} for all roster numbers
        """
        state = self._tracks.get(track_id)
        if state is None:
            return {
                "number": None, "shot_hand": None,
                "position": None, "confidence": 0.0,
                "log_posteriors": {},
            }

        log_p = self._compute_log_posterior(track_id)
        best_n = max(log_p, key=log_p.get) if log_p else None

        # Softmax confidence
        if log_p:
            max_lp = max(log_p.values())
            exp_vals = {n: math.exp(lp - max_lp) for n, lp in log_p.items()}
            total = sum(exp_vals.values())
            confidence = exp_vals[best_n] / total if total > 0 else 0.0
        else:
            confidence = 0.0

        # Return None when confidence is at or near the uniform prior (no real signal)
        n_candidates = len(self.roster_numbers)
        if n_candidates > 0 and confidence < (1.0 / n_candidates) + 0.05:
            best_n = None

        return {
            "number":        best_n,
            "shot_hand":     state.shot_hand,
            "position":      state.position(self.blue_line_offset, self.rink_length, self.rink_width),
            "confidence":    round(confidence, 4),
            "frame_count":   state.frame_count,
            "log_posteriors": log_p,
        }

    def resolve_by_elimination(self, on_ice_track_ids: List[int]) -> Dict[int, Optional[int]]:
        """
        On-ice elimination: if 4 of 5 skaters are identified, the 5th is automatic.
        Only applies when exactly 5 skaters are on ice.
        """
        skaters = [
            tid for tid in on_ice_track_ids
            # exclude goalie (tracked separately) if distinguishable
        ]
        if len(skaters) != 5:
            return {}

        identities = {tid: self.get_identity(tid) for tid in skaters}
        identified = {
            tid: idn["number"]
            for tid, idn in identities.items()
            if idn["number"] is not None and idn["confidence"] > 0.7
        }

        if len(identified) == 4:
            unknown = next(tid for tid in skaters if tid not in identified)
            used = set(identified.values())
            remaining = [n for n in self.roster_numbers if n not in used]
            if len(remaining) == 1:
                identified[unknown] = remaining[0]
                logger.info(
                    "Elimination: track %d → #%d (last remaining)", unknown, remaining[0]
                )

        return identified

    def summary(self) -> Dict[int, Dict[str, Any]]:
        """Return identity estimates for all known tracks."""
        return {tid: self.get_identity(tid) for tid in self._tracks}
