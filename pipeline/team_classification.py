"""
Team classification module for the IceIQ hockey analysis pipeline.

Classifies each tracked player as one of two user-defined teams based on
jersey colour. The colour ranges are injected at runtime so any team can be
supported without code changes.

Four-layer strategy
-------------------
1. HSV colour ratio on the torso strip (y: 20–65 % of bbox height)
2. Helmet region colour (top 15 % of bbox) — secondary signal
3. 30-frame temporal voting per track_id
4. Global on-ice count enforcement (max 7 players per team)
"""

import cv2
import numpy as np
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Default team labels (overridden via config) ──────────────────────────────
TEAM_A_DEFAULT = "team_a"
TEAM_B_DEFAULT = "team_b"

# ── Tuning constants ──────────────────────────────────────────────────────────
VOTE_WINDOW  = 30
MAX_PER_TEAM = 7
_MIN_CROP_AREA = 300

# ── Predefined HSV ranges for common jersey colours ───────────────────────────
# Format: (lower, upper) as np.uint8 arrays in HSV
_COLOUR_RANGES: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    "green": (
        np.array([ 40,  40,  20], dtype=np.uint8),
        np.array([100, 255, 180], dtype=np.uint8),
    ),
    "red": (
        # Red wraps around 0° in HSV — combine two ranges
        np.array([  0,  80,  50], dtype=np.uint8),
        np.array([ 10, 255, 255], dtype=np.uint8),
    ),
    "red2": (  # upper hue wrap
        np.array([160,  80,  50], dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8),
    ),
    "blue": (
        np.array([100,  80,  50], dtype=np.uint8),
        np.array([130, 255, 255], dtype=np.uint8),
    ),
    "white": (
        np.array([  0,   0, 180], dtype=np.uint8),
        np.array([180,  40, 255], dtype=np.uint8),
    ),
    "black": (
        np.array([  0,   0,   0], dtype=np.uint8),
        np.array([180, 255,  60], dtype=np.uint8),
    ),
    "yellow": (
        np.array([ 20, 100,  80], dtype=np.uint8),
        np.array([ 35, 255, 255], dtype=np.uint8),
    ),
    "orange": (
        np.array([ 10,  80,  80], dtype=np.uint8),
        np.array([ 25, 255, 255], dtype=np.uint8),
    ),
    "purple": (
        np.array([130,  50,  50], dtype=np.uint8),
        np.array([160, 255, 255], dtype=np.uint8),
    ),
    "navy": (  # 짙은 네이비 (H=100~130, 높은 채도, 낮은 밝기)
        np.array([100,  60,  20], dtype=np.uint8),
        np.array([130, 255, 120], dtype=np.uint8),
    ),
}

# Colours that need dual-range detection (hue wraps around 0°)
_DUAL_RANGE_COLOURS = {"red"}


def _colour_ratio(crop: np.ndarray, colour: str) -> float:
    """Return the fraction of pixels in *crop* matching the given colour."""
    if crop is None or crop.size < _MIN_CROP_AREA:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    n = crop.shape[0] * crop.shape[1]
    if n == 0:
        return 0.0

    if colour in _DUAL_RANGE_COLOURS:
        lo1, hi1 = _COLOUR_RANGES[colour]
        lo2, hi2 = _COLOUR_RANGES[colour + "2"]
        mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)
    elif colour in _COLOUR_RANGES:
        lo, hi = _COLOUR_RANGES[colour]
        mask = cv2.inRange(hsv, lo, hi)
    else:
        logger.warning(f"Unknown colour '{colour}', returning 0")
        return 0.0

    return float(np.count_nonzero(mask)) / n


class TeamConfig:
    """
    Holds identity and colour info for one team.

    Parameters
    ----------
    name   : Display name (e.g. "아이기스")
    label  : Internal label used as dict key (e.g. "aigis")
    colour : Jersey primary colour — one of the keys in _COLOUR_RANGES
             e.g. "red", "white", "green", "blue", ...
    threshold      : Minimum colour ratio to classify as this team (default 0.05)
    high_conf_thr  : Ratio above which confidence is high (default 0.18)
    """
    def __init__(
        self,
        name: str,
        label: str,
        colour: str,
        threshold: float = 0.05,
        high_conf_thr: float = 0.18,
    ):
        self.name = name
        self.label = label
        self.colour = colour.lower()
        self.threshold = threshold
        self.high_conf_thr = high_conf_thr

    def __repr__(self):
        return f"TeamConfig(name={self.name!r}, label={self.label!r}, colour={self.colour!r})"


# ── Built-in presets ──────────────────────────────────────────────────────────
PRESET_AIGIS_LOPEZ = (
    TeamConfig(name="아이기스",  label="aigis",  colour="red",   threshold=0.10, high_conf_thr=0.20),
    TeamConfig(name="로페즈",    label="lopez",  colour="white", threshold=0.05, high_conf_thr=0.15),
)

PRESET_ZENITH_AIGIS = (
    TeamConfig(name="Zenith Phoenix", label="zenith_phoenix", colour="green"),
    TeamConfig(name="아이기스",        label="aigis",          colour="white"),
)

PRESET_HOCKEY_MACHINE_AIGIS = (
    TeamConfig(name="하키머신", label="hockey_machine", colour="navy",  threshold=0.06, high_conf_thr=0.14),
    TeamConfig(name="아이기스", label="aigis",          colour="white", threshold=0.05, high_conf_thr=0.15),
)

# "흰색이면 team_b, 아니면 team_a" 전략용 preset
# white_primary=True 플래그로 분류 로직을 반전시킴
PRESET_AIGIS_PHOENIX = (
    TeamConfig(name="아이기스",      label="aigis",   colour="red",   threshold=0.10, high_conf_thr=0.20),
    TeamConfig(name="피닉스",        label="phoenix", colour="green", threshold=0.03, high_conf_thr=0.08),
)

PRESET_GOYANG_AIGIS = (
    TeamConfig(name="고양이글스", label="goyang_eagles", colour="black", threshold=0.06, high_conf_thr=0.14),
    TeamConfig(name="아이기스",   label="aigis",           colour="white", threshold=0.05, high_conf_thr=0.15),
)

PRESET_HOCKEY_MACHINE_AIGIS_WHITE = (
    TeamConfig(name="하키머신", label="hockey_machine", colour="white", threshold=0.05, high_conf_thr=0.15),
    TeamConfig(name="아이기스", label="aigis",          colour="white", threshold=0.05, high_conf_thr=0.15),
)


class HockeyTeamClassifier:
    """
    Classifies tracked hockey players into two teams using jersey colour,
    temporal voting, and global count enforcement.

    Parameters
    ----------
    team_a, team_b : TeamConfig objects defining each team's colour & label.
                     Defaults to the generic team_a / team_b preset.
    vote_window    : Frames of voting history per track_id (default 30)
    max_per_team   : Maximum players per team on ice (default 7)

    Example
    -------
    from pipeline.team_classification import HockeyTeamClassifier, PRESET_AIGIS_LOPEZ

    clf = HockeyTeamClassifier(*PRESET_AIGIS_LOPEZ)
    label, conf = clf.classify(frame, bbox, track_id)
    """

    def __init__(
        self,
        team_a: Optional[TeamConfig] = None,
        team_b: Optional[TeamConfig] = None,
        vote_window:  int = VOTE_WINDOW,
        max_per_team: int = MAX_PER_TEAM,
    ):
        self.team_a = team_a or TeamConfig("Team A", TEAM_A_DEFAULT, "red")
        self.team_b = team_b or TeamConfig("Team B", TEAM_B_DEFAULT, "white")
        self._vote_window  = vote_window
        self._max_per_team = max_per_team

        self._vote_history:      Dict[int, deque] = {}
        self._colour_ratio_cache: Dict[int, float] = {}

        logger.info(
            f"TeamClassifier: {self.team_a.name}({self.team_a.colour}) vs "
            f"{self.team_b.name}({self.team_b.colour})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        frame:    np.ndarray,
        bbox:     Tuple[float, float, float, float],
        track_id: int,
    ) -> Tuple[str, float]:
        """
        Classify a single player.

        Returns (label, confidence)  — label is one of team_a.label / team_b.label
        """
        raw_label, raw_conf, ratio = self._classify_single(frame, bbox)
        self._colour_ratio_cache[track_id] = ratio

        if track_id not in self._vote_history:
            self._vote_history[track_id] = deque(maxlen=self._vote_window)
        self._vote_history[track_id].append(raw_label)

        return self._voted_label(track_id, raw_conf)

    def classify_batch(
        self,
        frame:  np.ndarray,
        tracks: List[Dict],
    ) -> List[Dict]:
        """
        Classify all tracks, then enforce global count limits.
        Each track dict must have 'track_id' and 'bbox' keys.
        Returns the same list with 'team' and 'team_confidence' added.
        """
        results = []
        for t in tracks:
            label, conf = self.classify(frame, t["bbox"], t["track_id"])
            results.append({**t, "team": label, "team_confidence": conf})
        return self._enforce_count(results)

    def reset_track(self, track_id: int) -> None:
        """Clear vote history for a track (e.g. after confirmed ID switch)."""
        self._vote_history.pop(track_id, None)
        self._colour_ratio_cache.pop(track_id, None)

    def available_colours(self) -> List[str]:
        """Return the list of supported colour names."""
        return list(_COLOUR_RANGES.keys())

    # ------------------------------------------------------------------
    # Strategy 1 + 2: single-frame colour classification
    # ------------------------------------------------------------------

    def _classify_single(
        self,
        frame: np.ndarray,
        bbox: Tuple,
    ) -> Tuple[str, float, float]:
        """Returns (label, confidence, colour_ratio_for_team_a)."""
        fh, fw = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(fw, int(bbox[2]))
        y2 = min(fh, int(bbox[3]))
        bh, bw = y2 - y1, x2 - x1

        if bh <= 0 or bw <= 0:
            return self.team_b.label, 0.50, 0.0

        # ── Torso strip (Strategy 1) ──────────────────────────────────
        ty1 = y1 + int(bh * 0.20)
        ty2 = y1 + int(bh * 0.65)
        tx1 = x1 + int(bw * 0.15)
        tx2 = x2 - int(bw * 0.15)
        torso = frame[ty1:ty2, tx1:tx2]

        ratio_a_torso = _colour_ratio(torso, self.team_a.colour)
        ratio_b_torso = _colour_ratio(torso, self.team_b.colour)

        # ── Helmet region (Strategy 2) ────────────────────────────────
        hy2 = y1 + int(bh * 0.15)
        helmet = frame[y1:hy2, x1:x2]
        ratio_a_helmet = _colour_ratio(helmet, self.team_a.colour)

        # Weighted combination
        ratio_a = ratio_a_torso * 0.90 + ratio_a_helmet * 0.10

        # ── Decision ──────────────────────────────────────────────────
        # Prefer team_a if its colour ratio passes threshold
        if ratio_a >= self.team_a.high_conf_thr:
            label = self.team_a.label
            conf  = min(0.97, 0.72 + ratio_a)
        elif ratio_a >= self.team_a.threshold:
            label = self.team_a.label
            conf  = 0.55 + ratio_a * 2.0
        elif ratio_b_torso >= self.team_b.high_conf_thr:
            label = self.team_b.label
            conf  = min(0.97, 0.72 + ratio_b_torso)
        elif ratio_b_torso >= self.team_b.threshold:
            label = self.team_b.label
            conf  = 0.55 + ratio_b_torso * 2.0
        else:
            # Fallback: compare raw ratios
            if ratio_a >= ratio_b_torso:
                label = self.team_a.label
                conf  = 0.55
            else:
                label = self.team_b.label
                conf  = min(0.97, 0.65 + (1.0 - ratio_a) * 0.30)

        return label, float(np.clip(conf, 0.0, 1.0)), float(ratio_a)

    # ------------------------------------------------------------------
    # Strategy 3: temporal voting
    # ------------------------------------------------------------------

    def _voted_label(self, track_id: int, raw_conf: float) -> Tuple[str, float]:
        history = self._vote_history.get(track_id)
        if not history:
            return self.team_b.label, 0.50

        votes = list(history)
        total = len(votes)
        n_a = votes.count(self.team_a.label)
        n_b = votes.count(self.team_b.label)

        if n_a >= n_b:
            label     = self.team_a.label
            vote_frac = n_a / total
        else:
            label     = self.team_b.label
            vote_frac = n_b / total

        blend = min(1.0, total / self._vote_window)
        conf  = raw_conf * (1.0 - blend) + vote_frac * blend
        return label, float(np.clip(conf, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Strategy 4: global count enforcement
    # ------------------------------------------------------------------

    def _enforce_count(self, results: List[Dict]) -> List[Dict]:
        team_a = [dict(r) for r in results if r["team"] == self.team_a.label]
        team_b = [dict(r) for r in results if r["team"] == self.team_b.label]

        if len(team_a) > self._max_per_team:
            team_a, team_b = self._trim_excess(team_a, team_b, self.team_b.label)
        if len(team_b) > self._max_per_team:
            team_b, team_a = self._trim_excess(team_b, team_a, self.team_a.label)

        return team_a + team_b

    @staticmethod
    def _trim_excess(
        over: List[Dict],
        other: List[Dict],
        new_label: str,
    ) -> Tuple[List[Dict], List[Dict]]:
        n_excess = len(over) - MAX_PER_TEAM
        if n_excess <= 0:
            return over, other
        over_sorted = sorted(over, key=lambda r: r["team_confidence"])
        reclassified, kept = [], []
        for i, r in enumerate(over_sorted):
            if i < n_excess:
                reclassified.append({**r, "team": new_label,
                                     "team_confidence": 1.0 - r["team_confidence"]})
            else:
                kept.append(r)
        return kept, other + reclassified


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_classifier(config: Optional[Dict] = None) -> HockeyTeamClassifier:
    """
    Create a HockeyTeamClassifier from a config dict.

    Config keys:
        team_a_name, team_a_label, team_a_colour
        team_b_name, team_b_label, team_b_colour
        vote_window, max_per_team

    Example
    -------
    clf = create_classifier({
        "team_a_name": "아이기스", "team_a_label": "aigis",  "team_a_colour": "red",
        "team_b_name": "로페즈",   "team_b_label": "lopez",  "team_b_colour": "white",
    })
    """
    cfg = config or {}
    team_a = TeamConfig(
        name=cfg.get("team_a_name",   "Team A"),
        label=cfg.get("team_a_label",  TEAM_A_DEFAULT),
        colour=cfg.get("team_a_colour", "red"),
        threshold=cfg.get("team_a_threshold", 0.05),
        high_conf_thr=cfg.get("team_a_high_conf", 0.18),
    )
    team_b = TeamConfig(
        name=cfg.get("team_b_name",   "Team B"),
        label=cfg.get("team_b_label",  TEAM_B_DEFAULT),
        colour=cfg.get("team_b_colour", "white"),
        threshold=cfg.get("team_b_threshold", 0.05),
        high_conf_thr=cfg.get("team_b_high_conf", 0.18),
    )
    return HockeyTeamClassifier(
        team_a=team_a,
        team_b=team_b,
        vote_window=cfg.get("vote_window", VOTE_WINDOW),
        max_per_team=cfg.get("max_per_team", MAX_PER_TEAM),
    )


# ---------------------------------------------------------------------------
# Backward compatibility aliases
# ---------------------------------------------------------------------------
TEAM_ZENITH = "zenith_phoenix"
TEAM_AIGIS  = "aigis"
