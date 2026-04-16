"""
pipeline/smart_jersey.py
SmartJersey v2 — Phase 1 + Phase 2

Phase 1: Unsupervised Line Grouping  (NO OCR)
  ShiftTracker   — record on-ice sets per shift
  detect_shift_change — detect >2-player simultaneous swaps
  build_cooccurrence_matrix — NxN pair-frequency matrix
  cluster_lines  — spectral clustering → 3 F-lines + D-pairs
  assign_roles   — C / LW / RW from faceoff proximity & lateral bias
                   D from blue-line time fraction

Phase 2: Individual Profile Building (runs in parallel with Phase 1)
  PlayerProfile  — shot_hand, body_shape, equipment HSV, skate_profile
  ProfileBuilder — update(), get_profile(), find_best_ocr_frame()

Test entry point:
  if __name__ == "__main__":  game1.mp4 frames 5000-50000
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── sklearn spectral clustering (soft-fail) ──────────────────────────────────
try:
    from sklearn.cluster import SpectralClustering
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    logger.warning("scikit-learn not found — cluster_lines will use greedy fallback")

# ── YOLOv8-pose keypoint indices ─────────────────────────────────────────────
_KP_NOSE        = 0
_KP_L_EYE       = 1
_KP_R_EYE       = 2
_KP_L_SHOULDER  = 5
_KP_R_SHOULDER  = 6
_KP_L_WRIST     = 9
_KP_R_WRIST     = 10
_KP_L_HIP       = 11
_KP_R_HIP       = 12
_KP_L_ANKLE     = 15
_KP_R_ANKLE     = 16

# ── Rink geometry (identification.py convention: origin = left end boards) ───
# X = 0..60.96 m (left→right end), Y = 0..25.9 m (near→far side)
_RINK_LENGTH       = 60.96
_RINK_WIDTH        = 25.90
_BLUE_FROM_END     = 19.51   # m — both ends symmetric
_NEAR_BLUE_X       = _BLUE_FROM_END                        # 19.51
_FAR_BLUE_X        = _RINK_LENGTH - _BLUE_FROM_END         # 41.45
_RINK_MID_X        = _RINK_LENGTH / 2                      # 30.48
_RINK_MID_Y        = _RINK_WIDTH  / 2                      # 12.95

_FACEOFF_DOTS_M: List[Tuple[float, float]] = [
    (30.48, 12.95),   # center ice
    (17.70,  6.70),   # left DZ near
    (17.70, 19.20),   # left DZ far
    (43.26,  6.70),   # right DZ near
    (43.26, 19.20),   # right DZ far
    ( 7.90,  6.70),   # left EZ near
    ( 7.90, 19.20),   # left EZ far
    (53.06,  6.70),   # right EZ near
    (53.06, 19.20),   # right EZ far
]
_FACEOFF_PROX_M = 3.0   # m — "near a dot"
_BLUE_BAND_M    = 4.0   # m — "near the blue line"
_MIN_SHIFT_FRAMES = 15  # frames — shorter shifts are noise


# ═══════════════════════════════════════════════════════════════════════════ #
#  PHASE 1 — Unsupervised Line Grouping                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

class ShiftTracker:
    """
    Records on-ice sets per shift by watching for simultaneous >2-player swaps.

    Usage
    -----
    tracker = ShiftTracker()
    for each_frame:
        tracker.update(frame_num, current_on_ice_set, rink_positions_dict)
    groups = tracker.get_line_groups()
    """

    def __init__(self, swap_threshold: int = 2) -> None:
        """
        Args:
            swap_threshold: number of simultaneous player swaps that signals
                            a line change (default 2 → "more than 2").
        """
        self.swap_threshold = swap_threshold

        # Accumulated shift records: list of frozenset(track_ids)
        self.shifts: List[FrozenSet[int]] = []

        # Per-player rink position history: track_id → list[(rx, ry)]
        self.rink_positions: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

        # Internal state
        self._prev_set: FrozenSet[int] = frozenset()
        self._shift_start_frame: int = 0
        self._current_set: FrozenSet[int] = frozenset()
        self._frame_count_since_change: int = 0

    # ── Per-frame update ─────────────────────────────────────────────────────

    def update(
        self,
        frame_num: int,
        on_ice_set: Set[int],
        rink_positions: Optional[Dict[int, Tuple[float, float]]] = None,
    ) -> bool:
        """
        Update with the current frame's on-ice set.

        Args:
            frame_num:     Current frame index.
            on_ice_set:    Set of track_ids currently visible on ice.
            rink_positions: {track_id: (rx_m, ry_m)} for this frame.

        Returns:
            True if a shift change was detected (shift boundary recorded).
        """
        frozen = frozenset(on_ice_set)

        # Accumulate rink positions
        if rink_positions:
            for tid, pos in rink_positions.items():
                self.rink_positions[tid].append(pos)

        changed = self.detect_shift_change(frozen)

        if changed:
            # Record the completed shift if it was long enough to be real
            duration = frame_num - self._shift_start_frame
            if duration >= _MIN_SHIFT_FRAMES and len(self._prev_set) >= 3:
                self.shifts.append(self._prev_set)
                logger.debug(
                    "Shift recorded: %d players, %d frames  %s",
                    len(self._prev_set), duration, sorted(self._prev_set),
                )
            self._shift_start_frame = frame_num

        self._prev_set = frozen
        self._current_set = frozen
        return changed

    def detect_shift_change(self, on_ice_set: FrozenSet[int]) -> bool:
        """
        Return True when >swap_threshold players swap simultaneously.

        A swap means players leaving AND new players arriving in the same
        observation window.  We check both arrivals and departures so that
        a partial-frame capture (camera pan) doesn't falsely trigger.
        """
        if not self._prev_set:
            return False

        new_players   = on_ice_set - self._prev_set
        left_players  = self._prev_set - on_ice_set

        # Either direction exceeding threshold signals a line change
        return len(new_players) > self.swap_threshold or len(left_players) > self.swap_threshold

    # ── Co-occurrence matrix ─────────────────────────────────────────────────

    def build_cooccurrence_matrix(
        self,
        shifts: Optional[List[FrozenSet[int]]] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Build NxN symmetric co-occurrence matrix.

        Args:
            shifts: Override shift list (uses self.shifts if None).

        Returns:
            (matrix, track_id_index) where matrix[i,j] = number of shifts
            in which track_id_index[i] and track_id_index[j] appeared together.
        """
        shifts = shifts if shifts is not None else self.shifts
        if not shifts:
            return np.zeros((0, 0)), []

        all_ids: List[int] = sorted({tid for s in shifts for tid in s})
        idx = {tid: i for i, tid in enumerate(all_ids)}
        n = len(all_ids)
        mat = np.zeros((n, n), dtype=np.float32)

        for shift_set in shifts:
            ids = sorted(shift_set)
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    r, c = idx[a], idx[b]
                    mat[r, c] += 1
                    mat[c, r] += 1

        return mat, all_ids

    # ── Spectral clustering → line groups ───────────────────────────────────

    def cluster_lines(
        self,
        cooccurrence: Optional[np.ndarray] = None,
        track_ids: Optional[List[int]] = None,
        n_forward_lines: int = 3,
        n_d_pairs: int = 2,
    ) -> Dict[str, List[List[int]]]:
        """
        Cluster the co-occurrence matrix into forward lines and D-pairs.

        Strategy:
          1. Identify likely defensemen by blue-line-ratio > 0.35.
          2. Run SpectralClustering on the D sub-matrix (n_d_pairs groups).
          3. Run SpectralClustering on the F sub-matrix (n_forward_lines groups).

        Returns:
            {
              "forward_lines": [[tid, tid, tid], ...],  # n_forward_lines groups
              "d_pairs":       [[tid, tid], ...],        # n_d_pairs groups
            }
        """
        if cooccurrence is None:
            cooccurrence, track_ids = self.build_cooccurrence_matrix()

        if cooccurrence.size == 0 or not track_ids:
            return {"forward_lines": [], "d_pairs": []}

        # ── Classify each player as likely D or F via rink position ──────────
        d_ids, f_ids = self._split_d_f(track_ids)

        forward_lines = self._spectral_groups(
            cooccurrence, track_ids, f_ids, n_forward_lines
        )
        d_pairs = self._spectral_groups(
            cooccurrence, track_ids, d_ids, n_d_pairs
        )

        return {"forward_lines": forward_lines, "d_pairs": d_pairs}

    def _split_d_f(
        self, track_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Return (d_ids, f_ids) based on blue-line-ratio from recorded positions."""
        d_ids, f_ids = [], []
        for tid in track_ids:
            positions = self.rink_positions.get(tid, [])
            if not positions:
                f_ids.append(tid)  # unknown → assume forward
                continue
            xs = [p[0] for p in positions]
            near_blue = sum(
                1 for x in xs
                if x < _NEAR_BLUE_X + _BLUE_BAND_M or x > _FAR_BLUE_X - _BLUE_BAND_M
            )
            blue_ratio = near_blue / len(xs)
            if blue_ratio > 0.35:
                d_ids.append(tid)
            else:
                f_ids.append(tid)
        return d_ids, f_ids

    def _spectral_groups(
        self,
        mat: np.ndarray,
        all_ids: List[int],
        subset_ids: List[int],
        n_clusters: int,
    ) -> List[List[int]]:
        """Run spectral clustering on a subset of the co-occurrence matrix."""
        if len(subset_ids) < n_clusters:
            # Too few players — return each player as its own group
            return [[tid] for tid in subset_ids]

        idx_map = {tid: i for i, tid in enumerate(all_ids)}
        rows = [idx_map[tid] for tid in subset_ids if tid in idx_map]
        sub_mat = mat[np.ix_(rows, rows)]

        # Normalise to [0, 1] for affinity
        max_val = sub_mat.max()
        affinity = sub_mat / max_val if max_val > 0 else sub_mat + np.eye(len(rows))

        if _SKLEARN_OK and len(subset_ids) > n_clusters:
            try:
                sc = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity="precomputed",
                    random_state=42,
                    n_init=10,
                )
                labels = sc.fit_predict(affinity)
            except Exception as exc:
                logger.warning("SpectralClustering failed (%s) — greedy fallback", exc)
                labels = self._greedy_cluster(sub_mat, n_clusters)
        else:
            labels = self._greedy_cluster(sub_mat, n_clusters)

        groups: Dict[int, List[int]] = defaultdict(list)
        for i, tid in enumerate(subset_ids):
            groups[int(labels[i])].append(tid)
        return list(groups.values())

    @staticmethod
    def _greedy_cluster(mat: np.ndarray, n: int) -> np.ndarray:
        """Greedy fallback: assign each player to the group with highest co-occurrence."""
        k = mat.shape[0]
        labels = np.zeros(k, dtype=int)
        if k == 0:
            return labels
        # Initialise cluster centres greedily (spread them out)
        centers = [0]
        for _ in range(n - 1):
            min_scores = [min(mat[i, c] for c in centers) for i in range(k)]
            centers.append(int(np.argmin(min_scores)))
        for i in range(k):
            labels[i] = min(range(n), key=lambda c: -mat[i, centers[c]] if i != centers[c] else 0)
        return labels

    # ── Role assignment within a group ──────────────────────────────────────

    def assign_roles(
        self,
        group: List[int],
        is_d_pair: bool = False,
    ) -> Dict[int, str]:
        """
        Assign positional roles within a group using rink position history.

        For forwards: C = most faceoff-dot time, LW/RW = lateral bias.
        For D-pairs:  LD = mean Y < rink_mid,  RD = mean Y >= rink_mid.

        Args:
            group:     List of track_ids in this line or D-pair.
            is_d_pair: True if this is a D-pair (else forward line).

        Returns:
            {track_id: role_string}
        """
        roles: Dict[int, str] = {}

        if is_d_pair:
            # Assign LD / RD by lateral (Y) position
            scored = []
            for tid in group:
                positions = self.rink_positions.get(tid, [])
                mean_y = (
                    sum(p[1] for p in positions) / len(positions)
                    if positions else _RINK_MID_Y
                )
                scored.append((tid, mean_y))
            scored.sort(key=lambda x: x[1])
            role_names = ["LD", "RD"] if len(scored) >= 2 else ["D"]
            for i, (tid, _) in enumerate(scored):
                roles[tid] = role_names[min(i, len(role_names) - 1)]
            return roles

        # Forward line — compute scores per player
        faceoff_scores: Dict[int, float] = {}
        mean_ys: Dict[int, float] = {}

        for tid in group:
            positions = self.rink_positions.get(tid, [])
            if not positions:
                faceoff_scores[tid] = 0.0
                mean_ys[tid] = _RINK_MID_Y
                continue

            faceoff_count = sum(
                1 for rx, ry in positions
                if any(
                    math.hypot(rx - dx, ry - dy) < _FACEOFF_PROX_M
                    for dx, dy in _FACEOFF_DOTS_M
                )
            )
            faceoff_scores[tid] = faceoff_count / len(positions)
            mean_ys[tid] = sum(p[1] for p in positions) / len(positions)

        # Center: highest faceoff participation
        center_id = max(faceoff_scores, key=faceoff_scores.get)
        roles[center_id] = "C"

        # Remaining: LW / RW by lateral Y position
        wings = [tid for tid in group if tid != center_id]
        if len(wings) >= 2:
            wings.sort(key=lambda t: mean_ys[t])
            roles[wings[0]] = "LW"   # lower Y = near side = left wing
            roles[wings[1]] = "RW"
            for tid in wings[2:]:
                roles[tid] = "F"     # extra players → generic forward
        elif len(wings) == 1:
            tid = wings[0]
            roles[tid] = "LW" if mean_ys[tid] < _RINK_MID_Y else "RW"

        return roles

    # ── Convenience: full Phase 1 result ────────────────────────────────────

    def get_line_groups(self) -> Dict[str, Any]:
        """
        Run the full Phase 1 pipeline and return a structured result.

        Returns:
            {
              "n_shifts": int,
              "forward_lines": [
                  {"players": [tid,...], "roles": {tid: "C"|"LW"|"RW"}},
                  ...
              ],
              "d_pairs": [
                  {"players": [tid,...], "roles": {tid: "LD"|"RD"}},
                  ...
              ],
            }
        """
        mat, ids = self.build_cooccurrence_matrix()
        clusters = self.cluster_lines(mat, ids)

        forward_lines_out = []
        for line in clusters["forward_lines"]:
            roles = self.assign_roles(line, is_d_pair=False)
            forward_lines_out.append({"players": sorted(line), "roles": roles})

        d_pairs_out = []
        for pair in clusters["d_pairs"]:
            roles = self.assign_roles(pair, is_d_pair=True)
            d_pairs_out.append({"players": sorted(pair), "roles": roles})

        return {
            "n_shifts": len(self.shifts),
            "forward_lines": forward_lines_out,
            "d_pairs": d_pairs_out,
        }


# ═══════════════════════════════════════════════════════════════════════════ #
#  PHASE 2 — Individual Profile Building                                       #
# ═══════════════════════════════════════════════════════════════════════════ #

# Equipment zone definitions (fraction of bbox height)
_ZONES = {
    "helmet":    (0.00, 0.15),
    "shoulders": (0.15, 0.30),
    "torso":     (0.30, 0.55),
    "pants":     (0.55, 0.80),
    "skates":    (0.80, 1.00),
}
# HSV histogram bins per channel
_H_BINS, _S_BINS, _V_BINS = 18, 16, 16


@dataclass
class PlayerProfile:
    """
    All non-OCR signals for one track_id.

    shot_hand:       "RIGHT" | "LEFT" | None
    body_shape:      {"height_m": float, "shoulder_width_m": float} or {}
    equipment:       {"helmet": np.ndarray, ...}  — HSV histograms, one per zone
    skate_profile:   {"black_ratio": float, "accent_hsv": (H,S,V)} or {}
    frame_count:     total frames accumulated
    """
    track_id:     int
    shot_hand:    Optional[str]             = None
    body_shape:   Dict[str, float]          = field(default_factory=dict)
    equipment:    Dict[str, np.ndarray]     = field(default_factory=dict)
    skate_profile: Dict[str, Any]          = field(default_factory=dict)
    frame_count:  int                       = 0

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary."""
        dominant = {}
        for zone, hist in self.equipment.items():
            if hist is not None and hist.size > 0:
                # Find the most frequent hue bucket
                h_hist = hist[: _H_BINS]
                peak_h_bin = int(np.argmax(h_hist))
                dominant[zone] = f"hue_bin_{peak_h_bin}"
        return {
            "track_id":   self.track_id,
            "shot_hand":  self.shot_hand,
            "height_m":   round(self.body_shape.get("height_m", 0), 2) or None,
            "shoulder_m": round(self.body_shape.get("shoulder_width_m", 0), 2) or None,
            "equipment_dominant_hue": dominant,
            "skate_black_ratio": round(self.skate_profile.get("black_ratio", 0.0), 3),
            "frame_count": self.frame_count,
        }


# ── Internal per-track accumulator ──────────────────────────────────────────

@dataclass
class _ProfileState:
    wrist_right: int = 0
    wrist_left:  int = 0

    heights_m:         List[float] = field(default_factory=list)
    shoulder_widths_m: List[float] = field(default_factory=list)

    # zone_name → running sum of HSV hist, and count
    zone_hist_sum: Dict[str, np.ndarray] = field(default_factory=dict)
    zone_hist_cnt: Dict[str, int]        = field(default_factory=lambda: defaultdict(int))

    skate_black_sum:   float = 0.0
    skate_accent_sum:  np.ndarray = field(default_factory=lambda: np.zeros(3))
    skate_frame_count: int   = 0

    # OCR candidate selection
    # Each entry: (bbox_area, back_score, blur_score, frame_num, frame_img, bbox)
    ocr_candidates: deque = field(
        default_factory=lambda: deque(maxlen=50)
    )

    frame_count: int = 0


def _hsv_hist(patch: np.ndarray) -> np.ndarray:
    """Compute a compact HSV histogram for a BGR image patch."""
    if patch is None or patch.size == 0:
        return np.zeros(_H_BINS + _S_BINS + _V_BINS, dtype=np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [_H_BINS],  [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [_S_BINS],  [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [_V_BINS],  [0, 256]).flatten()
    combined = np.concatenate([h_hist, s_hist, v_hist])
    total = combined.sum()
    return combined / total if total > 0 else combined


def _laplacian_blur_score(patch: np.ndarray) -> float:
    """Higher = sharper. Variance of Laplacian."""
    if patch is None or patch.size == 0:
        return 0.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _back_facing_score(keypoints: Optional[np.ndarray]) -> float:
    """
    Score in [0, 1] for how likely the player is facing away from camera.
    Back = nose/eyes not visible but shoulders are.
    """
    if keypoints is None or keypoints.ndim < 2 or keypoints.shape[0] < 11:
        return 0.0
    nose_conf = float(keypoints[_KP_NOSE, 2])
    l_eye_conf = float(keypoints[_KP_L_EYE,  2]) if keypoints.shape[0] > _KP_L_EYE  else 0.0
    r_eye_conf = float(keypoints[_KP_R_EYE,  2]) if keypoints.shape[0] > _KP_R_EYE  else 0.0
    ls_conf  = float(keypoints[_KP_L_SHOULDER, 2])
    rs_conf  = float(keypoints[_KP_R_SHOULDER, 2])

    face_invisible = (nose_conf < 0.3) and (l_eye_conf < 0.3) and (r_eye_conf < 0.3)
    shoulders_visible = (ls_conf > 0.4) and (rs_conf > 0.4)

    if face_invisible and shoulders_visible:
        return 1.0 - (nose_conf + l_eye_conf + r_eye_conf) / 0.9   # softer score
    return 0.0


class ProfileBuilder:
    """
    Builds and maintains one PlayerProfile per track_id.

    Usage
    -----
    builder = ProfileBuilder(mapper=rink_homography_object)
    for each detected player in each frame:
        builder.update(track_id, frame_bgr, bbox, keypoints, rink_pos)

    profile = builder.get_profile(track_id)
    best_ocr = builder.find_best_ocr_frame(track_id)
    """

    def __init__(self, mapper=None) -> None:
        """
        Args:
            mapper: Optional HomographyMapper / RinkHomography instance that
                    provides pixel_dist_to_meters(p1, p2).  Used for body_shape.
        """
        self.mapper = mapper
        self._states: Dict[int, _ProfileState] = defaultdict(_ProfileState)

    # ── Main update ─────────────────────────────────────────────────────────

    def update(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: List[float],
        keypoints: Optional[np.ndarray] = None,
        rink_pos: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Accumulate one frame's observations for track_id.

        Args:
            track_id:  Consistent tracker ID.
            frame:     Full BGR video frame (H×W×3).
            bbox:      [x1, y1, x2, y2] in pixels.
            keypoints: YOLOv8-pose output, shape (17, 3) — [x, y, conf].
            rink_pos:  Player foot position in metres (rx, ry); unused here
                       but kept for API symmetry with PlayerIdentifier.
        """
        st = self._states[track_id]
        st.frame_count += 1

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1] - 1, x2); y2 = min(frame.shape[0] - 1, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            return

        crop = frame[y1:y2, x1:x2]

        # Signal: shot hand (keypoints)
        if keypoints is not None:
            self._update_shot_hand(st, keypoints)
            self._update_body_shape(st, keypoints)

        # Equipment HSV histograms
        self._update_equipment(st, crop)

        # Skate profile (bottom 20%)
        self._update_skates(st, crop)

        # OCR candidate bookkeeping
        bbox_area = bw * bh
        back_score = _back_facing_score(keypoints)
        blur_score = _laplacian_blur_score(crop)
        st.ocr_candidates.append((bbox_area, back_score, blur_score, st.frame_count, bbox))

    # ── Signal sub-routines ──────────────────────────────────────────────────

    def _update_shot_hand(self, st: _ProfileState, kp: np.ndarray) -> None:
        """
        Shot hand from wrist keypoints.
        left_wrist.y < right_wrist.y → shoots RIGHT (top hand is left).
        One strong frame is enough; accumulate votes for robustness.
        """
        if kp.shape[0] <= max(_KP_L_WRIST, _KP_R_WRIST):
            return
        lw_conf = float(kp[_KP_L_WRIST, 2])
        rw_conf = float(kp[_KP_R_WRIST, 2])
        if lw_conf < 0.3 or rw_conf < 0.3:
            return
        if kp[_KP_L_WRIST, 1] < kp[_KP_R_WRIST, 1]:
            st.wrist_right += 1   # left wrist higher → top hand left → RIGHT shot
        else:
            st.wrist_left += 1

    def _update_body_shape(self, st: _ProfileState, kp: np.ndarray) -> None:
        """Height and shoulder width via homography (or pixel fallback)."""
        n_kp = kp.shape[0]
        nose_c = float(kp[_KP_NOSE, 2]) if n_kp > _KP_NOSE else 0.0
        la_c   = float(kp[_KP_L_ANKLE, 2]) if n_kp > _KP_L_ANKLE else 0.0
        ra_c   = float(kp[_KP_R_ANKLE, 2]) if n_kp > _KP_R_ANKLE else 0.0
        ls_c   = float(kp[_KP_L_SHOULDER, 2]) if n_kp > _KP_L_SHOULDER else 0.0
        rs_c   = float(kp[_KP_R_SHOULDER, 2]) if n_kp > _KP_R_SHOULDER else 0.0

        if nose_c > 0.3 and (la_c > 0.3 or ra_c > 0.3):
            nose_px = (float(kp[_KP_NOSE, 0]), float(kp[_KP_NOSE, 1]))
            if la_c > 0.3 and ra_c > 0.3:
                foot_px: Tuple[float, float] = (
                    (float(kp[_KP_L_ANKLE, 0]) + float(kp[_KP_R_ANKLE, 0])) / 2,
                    (float(kp[_KP_L_ANKLE, 1]) + float(kp[_KP_R_ANKLE, 1])) / 2,
                )
            elif la_c > 0.3:
                foot_px = (float(kp[_KP_L_ANKLE, 0]), float(kp[_KP_L_ANKLE, 1]))
            else:
                foot_px = (float(kp[_KP_R_ANKLE, 0]), float(kp[_KP_R_ANKLE, 1]))

            if self.mapper is not None:
                try:
                    h = self.mapper.pixel_dist_to_meters(nose_px, foot_px)
                except Exception:
                    h = abs(nose_px[1] - foot_px[1]) / 1080.0 * 1.85
            else:
                h = abs(nose_px[1] - foot_px[1]) / 1080.0 * 1.85   # pixel fallback

            if 0.5 < h < 2.5:
                st.heights_m.append(h)

        if ls_c > 0.3 and rs_c > 0.3:
            ls_px = (float(kp[_KP_L_SHOULDER, 0]), float(kp[_KP_L_SHOULDER, 1]))
            rs_px = (float(kp[_KP_R_SHOULDER, 0]), float(kp[_KP_R_SHOULDER, 1]))
            if self.mapper is not None:
                try:
                    sw = self.mapper.pixel_dist_to_meters(ls_px, rs_px)
                except Exception:
                    sw = abs(ls_px[0] - rs_px[0]) / 1920.0 * _RINK_WIDTH * 0.025
            else:
                sw = abs(ls_px[0] - rs_px[0]) / 1920.0 * _RINK_WIDTH * 0.025
            if 0.1 < sw < 1.5:
                st.shoulder_widths_m.append(sw)

    def _update_equipment(self, st: _ProfileState, crop: np.ndarray) -> None:
        """HSV histogram accumulation per equipment zone."""
        h = crop.shape[0]
        for zone, (frac_top, frac_bot) in _ZONES.items():
            y_top = int(frac_top * h)
            y_bot = int(frac_bot * h)
            patch = crop[y_top:y_bot, :]
            if patch.size == 0:
                continue
            hist = _hsv_hist(patch)
            if zone not in st.zone_hist_sum:
                st.zone_hist_sum[zone] = np.zeros_like(hist)
            st.zone_hist_sum[zone] += hist
            st.zone_hist_cnt[zone] += 1

    def _update_skates(self, st: _ProfileState, crop: np.ndarray) -> None:
        """
        Skate profile: bottom 20% of bbox.
        black_ratio = fraction of pixels where V < 50.
        accent_color = mean HSV of non-black pixels.
        """
        h = crop.shape[0]
        skate_patch = crop[int(0.80 * h):, :]
        if skate_patch.size == 0:
            return
        hsv = cv2.cvtColor(skate_patch, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        black_mask = v_channel < 50
        black_ratio = float(black_mask.mean())
        st.skate_black_sum += black_ratio
        st.skate_frame_count += 1

        non_black = hsv[~black_mask]
        if non_black.shape[0] > 0:
            st.skate_accent_sum += non_black.mean(axis=0)

    # ── Public API ───────────────────────────────────────────────────────────

    def get_profile(self, track_id: int) -> PlayerProfile:
        """Return the current PlayerProfile for track_id."""
        st = self._states[track_id]

        # Shot hand
        total_wrist = st.wrist_right + st.wrist_left
        if total_wrist >= 5:
            shot_hand: Optional[str] = "RIGHT" if st.wrist_right >= st.wrist_left else "LEFT"
        else:
            shot_hand = None

        # Body shape
        valid_h = [h for h in st.heights_m if 0.5 < h < 2.5]
        valid_sw = [w for w in st.shoulder_widths_m if 0.1 < w < 1.5]
        body_shape: Dict[str, float] = {}
        if len(valid_h) >= 5:
            body_shape["height_m"] = float(np.median(valid_h))
        if len(valid_sw) >= 5:
            body_shape["shoulder_width_m"] = float(np.median(valid_sw))

        # Equipment histograms (normalised running mean)
        equipment: Dict[str, np.ndarray] = {}
        for zone, hist_sum in st.zone_hist_sum.items():
            cnt = st.zone_hist_cnt[zone]
            equipment[zone] = hist_sum / cnt if cnt > 0 else hist_sum

        # Skate profile
        skate_profile: Dict[str, Any] = {}
        if st.skate_frame_count > 0:
            skate_profile["black_ratio"] = st.skate_black_sum / st.skate_frame_count
            # accent HSV as tuple
            mean_accent = st.skate_accent_sum / st.skate_frame_count
            skate_profile["accent_hsv"] = tuple(float(v) for v in mean_accent)

        return PlayerProfile(
            track_id=track_id,
            shot_hand=shot_hand,
            body_shape=body_shape,
            equipment=equipment,
            skate_profile=skate_profile,
            frame_count=st.frame_count,
        )

    def find_best_ocr_frame(self, track_id: int) -> Optional[Dict[str, Any]]:
        """
        Return metadata for the single best frame to run OCR on.

        Selection criteria (weighted score):
          1. Largest bounding box area    (player close to camera)
          2. Back-facing                  (jersey number visible)
          3. Least motion blur            (sharpest crop)

        Returns:
            {"frame_num": int, "bbox": [x1,y1,x2,y2], "score": float}
            or None if no candidates available.
        """
        st = self._states.get(track_id)
        if not st or not st.ocr_candidates:
            return None

        candidates = list(st.ocr_candidates)
        if not candidates:
            return None

        # Normalise each dimension to [0, 1]
        areas       = np.array([c[0] for c in candidates], dtype=np.float32)
        back_scores = np.array([c[1] for c in candidates], dtype=np.float32)
        blur_scores = np.array([c[2] for c in candidates], dtype=np.float32)

        def _norm(arr: np.ndarray) -> np.ndarray:
            rng = arr.max() - arr.min()
            return (arr - arr.min()) / rng if rng > 0 else np.ones_like(arr) * 0.5

        combined = (
            0.40 * _norm(areas)
            + 0.35 * _norm(back_scores)
            + 0.25 * _norm(blur_scores)
        )

        best_i = int(np.argmax(combined))
        bbox_area, back_s, blur_s, frame_num, bbox = candidates[best_i]
        return {
            "frame_num": frame_num,
            "bbox":      bbox,
            "score":     float(combined[best_i]),
            "back_facing": back_s > 0.5,
            "blur_score":  float(blur_s),
            "bbox_area":   int(bbox_area),
        }

    def all_profiles(self) -> Dict[int, PlayerProfile]:
        """Return profiles for every tracked player."""
        return {tid: self.get_profile(tid) for tid in self._states}


# ═══════════════════════════════════════════════════════════════════════════ #
#  Test: game1.mp4 frames 5000-50000                                           #
# ═══════════════════════════════════════════════════════════════════════════ #

def _run_test(
    video_path: str = "data/videos/game1.mp4",
    start_frame: int = 5_000,
    end_frame: int = 50_000,
    step: int = 3,        # process every Nth frame (speed vs. accuracy)
    conf: float = 0.25,
) -> None:
    """
    End-to-end Phase 1 + Phase 2 test on a video segment.

    Attempts to use YOLOv8-pose for detection; falls back to basic
    background-subtraction blob detection when ultralytics is unavailable.
    """
    import os
    import subprocess

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("smart_jersey.test")

    # ── Open video ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Cannot open video: %s", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Video: %s  |  FPS=%.1f  |  Frames=%d", video_path, fps, total_frames)
    log.info("Processing frames %d → %d  (step=%d)", start_frame, end_frame, step)

    # ── Try to load YOLOv8-pose ──────────────────────────────────────────────
    yolo_model = None
    try:
        from ultralytics import YOLO as _YOLO
        yolo_model = _YOLO("yolov8m-pose.pt")
        log.info("YOLOv8-pose loaded")
    except Exception as exc:
        log.warning("YOLOv8-pose unavailable (%s) — using MOG2 blob fallback", exc)

    # ── Try to load homography ───────────────────────────────────────────────
    mapper = None
    try:
        from pipeline.homography import RinkHomography
        mapper = RinkHomography()
        cal_path = "data/homography_calibration.json"
        if os.path.exists(cal_path):
            mapper.load(cal_path)
            log.info("RinkHomography loaded from %s", cal_path)
    except Exception as exc:
        log.warning("Homography unavailable (%s) — pixel fallback", exc)

    # ── Phase 1 + Phase 2 objects ────────────────────────────────────────────
    shift_tracker   = ShiftTracker(swap_threshold=2)
    profile_builder = ProfileBuilder(mapper=mapper)

    # ── Background subtractor (MOG2 fallback) ────────────────────────────────
    mog2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50) if yolo_model is None else None

    # Seek to start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    track_id_counter = 0
    active_tracks: Dict[int, List[float]] = {}   # bbox fallback: track_id → last_bbox
    frames_processed = 0
    next_log_frame = start_frame + 1000
    end_frame_clamped = min(end_frame, total_frames)
    frame_num = start_frame - 1

    while True:
        frame_num += 1
        if frame_num >= end_frame_clamped:
            break
        ret, frame = cap.read()
        if not ret:
            break
        # Only process every step-th frame; still read sequentially (fast)
        if (frame_num - start_frame) % step != 0:
            continue

        on_ice: Set[int] = set()
        rink_positions: Dict[int, Tuple[float, float]] = {}

        if yolo_model is not None:
            # ── YOLOv8-pose detections ────────────────────────────────────────
            try:
                results = yolo_model(frame, conf=conf, verbose=False)
                for result in results:
                    boxes    = result.boxes
                    keypoints_all = result.keypoints

                    for i, box in enumerate(boxes):
                        cls_id = int(box.cls[0]) if box.cls is not None else -1
                        if cls_id != 0:   # person class only
                            continue

                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        x1, y1, x2, y2 = xyxy

                        # Assign a simple track_id based on bbox centroid proximity
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        tid = _assign_track_id(
                            cx, cy, active_tracks, track_id_counter
                        )
                        if tid == track_id_counter:
                            track_id_counter += 1
                        active_tracks[tid] = [x1, y1, x2, y2]
                        on_ice.add(tid)

                        # Keypoints: shape (17, 3)
                        kp: Optional[np.ndarray] = None
                        if keypoints_all is not None and i < len(keypoints_all.data):
                            kp = keypoints_all.data[i].cpu().numpy()  # (17, 3)

                        # Rink position from foot midpoint
                        if mapper is not None:
                            foot_x = (x1 + x2) / 2
                            foot_y = y2
                            try:
                                rx, ry = mapper.pixel_to_rink(foot_x, foot_y)
                                rink_positions[tid] = (rx, ry)
                            except Exception:
                                pass

                        profile_builder.update(tid, frame, [x1, y1, x2, y2], kp)

            except Exception as exc:
                log.debug("YOLO frame %d error: %s", frame_num, exc)
        else:
            # ── MOG2 blob fallback ────────────────────────────────────────────
            fg_mask = mog2.apply(frame)
            fg_mask = cv2.morphologyEx(
                fg_mask,
                cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 800:   # too small for a player
                    continue
                bx, by, bw, bh = cv2.boundingRect(cnt)
                # Aspect ratio filter: players are taller than wide
                if bh < bw or bh < 40:
                    continue
                cx = bx + bw / 2
                cy = by + bh / 2
                tid = _assign_track_id(
                    cx, cy, active_tracks, track_id_counter
                )
                if tid == track_id_counter:
                    track_id_counter += 1
                active_tracks[tid] = [bx, by, bx + bw, by + bh]
                on_ice.add(tid)
                profile_builder.update(
                    tid, frame, [float(bx), float(by), float(bx + bw), float(by + bh)]
                )

        # ── Update ShiftTracker ───────────────────────────────────────────────
        shift_tracker.update(frame_num, on_ice, rink_positions)

        frames_processed += 1
        if frame_num >= next_log_frame:
            log.info(
                "Frame %d | on_ice=%d | shifts=%d | tracks=%d",
                frame_num, len(on_ice), len(shift_tracker.shifts), len(active_tracks),
            )
            next_log_frame += 1000

    cap.release()
    log.info("Processed %d frames total", frames_processed)

    # ═══════════════════════════════════════════════════════════════════════ #
    #  Results                                                                 #
    # ═══════════════════════════════════════════════════════════════════════ #

    print("\n" + "=" * 70)
    print("SmartJersey v2 — Phase 1 + Phase 2 Results")
    print("=" * 70)

    # ── Phase 1: Line groups ─────────────────────────────────────────────────
    groups = shift_tracker.get_line_groups()
    print(f"\nPHASE 1 — {groups['n_shifts']} shifts recorded")

    if groups["forward_lines"]:
        print("\n  Forward Lines:")
        for i, line in enumerate(groups["forward_lines"], 1):
            role_str = "  ".join(
                f"track_{p}:{line['roles'].get(p, '?')}"
                for p in line["players"]
            )
            print(f"    Line {i}: {role_str}")
    else:
        print("  (No forward lines identified yet — need more shifts)")

    if groups["d_pairs"]:
        print("\n  D-Pairs:")
        for i, pair in enumerate(groups["d_pairs"], 1):
            role_str = "  ".join(
                f"track_{p}:{pair['roles'].get(p, '?')}"
                for p in pair["players"]
            )
            print(f"    Pair {i}: {role_str}")
    else:
        print("  (No D-pairs identified yet)")

    # ── Phase 2: Player profiles ─────────────────────────────────────────────
    all_profiles = profile_builder.all_profiles()
    print(f"\nPHASE 2 — {len(all_profiles)} player profiles built")

    # Sort by frame count (most-seen players first)
    top_tracks = sorted(
        all_profiles.values(),
        key=lambda p: p.frame_count,
        reverse=True,
    )[:20]   # show top 20

    print(f"\n  {'Track':>7}  {'Frames':>7}  {'ShotHand':>9}  {'Height(m)':>10}  {'SkateBlk':>9}")
    print("  " + "-" * 55)
    for prof in top_tracks:
        s = prof.summary()
        hand   = s["shot_hand"] or "?"
        height = f"{s['height_m']:.2f}" if s["height_m"] else "  ?"
        skblk  = f"{s['skate_black_ratio']:.3f}"
        print(
            f"  {s['track_id']:>7}  {s['frame_count']:>7}  "
            f"{hand:>9}  {height:>10}  {skblk:>9}"
        )
        if s["equipment_dominant_hue"]:
            zones = ", ".join(
                f"{z}={v}" for z, v in s["equipment_dominant_hue"].items()
            )
            print(f"           equipment: {zones}")

    # ── Best OCR frames ──────────────────────────────────────────────────────
    print("\n  Best OCR frames (top 10 players):")
    for prof in top_tracks[:10]:
        ocr_info = profile_builder.find_best_ocr_frame(prof.track_id)
        if ocr_info:
            print(
                f"    track_{prof.track_id}: "
                f"frame={ocr_info['frame_num']}  "
                f"area={ocr_info['bbox_area']}  "
                f"back={ocr_info['back_facing']}  "
                f"blur={ocr_info['blur_score']:.0f}  "
                f"score={ocr_info['score']:.3f}"
            )

    print("\n" + "=" * 70)

    # ── Completion event ─────────────────────────────────────────────────────
    try:
        subprocess.run(
            [
                "openclaw", "system", "event",
                "--text", "SmartJersey v2 Phase 1+2 complete: line clustering + player profiles",
                "--mode", "now",
            ],
            check=False,
        )
    except FileNotFoundError:
        log.warning("openclaw not found — skipping system event")


def _assign_track_id(
    cx: float,
    cy: float,
    active_tracks: Dict[int, List[float]],
    next_id: int,
    max_dist: float = 80.0,
) -> int:
    """
    Minimal centroid tracker: match detection to closest existing track.
    Returns existing track_id if close enough, else next_id (new track).
    """
    best_tid = next_id
    best_dist = max_dist
    for tid, bbox in active_tracks.items():
        tx = (bbox[0] + bbox[2]) / 2
        ty = (bbox[1] + bbox[3]) / 2
        d = math.hypot(cx - tx, cy - ty)
        if d < best_dist:
            best_dist = d
            best_tid = tid
    return best_tid


if __name__ == "__main__":
    _run_test()
