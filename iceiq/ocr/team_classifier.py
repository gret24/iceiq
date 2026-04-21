"""
iceiq/ocr/team_classifier.py
Lightweight HSV-based team classifier for hockey player crops.

Supports any two-team matchup via metadata.json ``uniform`` fields (v3)
or roster JSON (legacy v2).

Usage
-----
    # v3 — from metadata.json game context
    from iceiq.ocr.team_classifier import TeamClassifierV2, from_metadata

    clf = from_metadata("data/videos/metadata.json", file_id="ku_vs_mega")
    label, conf = clf.classify_crop(crop_bgr)   # "our" | "opponent" | "uncertain"

    # legacy v2 — from roster JSON
    from iceiq.ocr.team_classifier import load_from_roster
    clf = load_from_roster("benchmark/labels/aigis_roster.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ─── HSV colour definitions ───────────────────────────────────────────────────
# Each entry: (lower_hsv, upper_hsv) as uint8 arrays.
# Colours that wrap H=0° (red) have a paired "<name>2" entry.

_HSV_RANGES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "white": (
        np.array([  0,   0, 170], dtype=np.uint8),
        np.array([180,  60, 255], dtype=np.uint8),
    ),
    "green": (
        np.array([ 40,  60,  50], dtype=np.uint8),
        np.array([ 90, 255, 255], dtype=np.uint8),
    ),
    "red": (
        np.array([  0,  80,  50], dtype=np.uint8),
        np.array([ 10, 255, 255], dtype=np.uint8),
    ),
    "red2": (
        np.array([160,  80,  50], dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8),
    ),
    "blue": (
        np.array([100,  80,  50], dtype=np.uint8),
        np.array([130, 255, 255], dtype=np.uint8),
    ),
    # 하늘색 (아이스킹덤): S_min=60으로 white(S≤60) 겹침 방지
    "light_blue": (
        np.array([ 85,  60, 150], dtype=np.uint8),
        np.array([110, 180, 255], dtype=np.uint8),
    ),
    # 보라 (메가): H 130–160
    "purple": (
        np.array([125,  60,  50], dtype=np.uint8),
        np.array([160, 255, 255], dtype=np.uint8),
    ),
    "navy": (
        np.array([100,  60,  20], dtype=np.uint8),
        np.array([130, 255, 120], dtype=np.uint8),
    ),
    "yellow": (
        np.array([ 20, 100,  80], dtype=np.uint8),
        np.array([ 35, 255, 255], dtype=np.uint8),
    ),
    "black": (
        np.array([  0,   0,   0], dtype=np.uint8),
        np.array([180, 255,  60], dtype=np.uint8),
    ),
}

_DUAL_RANGE = {"red"}        # colours that wrap around H=0° in HSV

# Default thresholds per colour — tuned per-type
# opp_threshold: how much opponent colour needed to declare "opponent"
# white is the background of the ice + boards, so requires a stricter guard
_OPP_THRESHOLDS: dict[str, float] = {
    "white":      0.30,   # high — rink boards are white
    "green":      0.05,
    "red":        0.06,
    "blue":       0.06,
    "light_blue": 0.08,   # slightly higher — can bleed into white region
    "purple":     0.05,
    "navy":       0.05,
    "yellow":     0.04,
    "black":      0.05,
}

# Maximum allowed opponent-colour ratio in an "our" crop.
# When the opponent wears white, KU red jerseys have white design elements
# (stripes, numbers, logos) + ice background bleeding into the torso strip.
# The default 0.03 is too strict for these cases.
_OUR_COLOR_MAX: dict[str, float] = {
    "white":      0.299,  # must stay just below _OPP_THRESHOLDS["white"]=0.30
    "green":      0.03,
    "red":        0.03,
    "blue":       0.04,
    "light_blue": 0.05,
    "purple":     0.03,
    "navy":       0.03,
    "yellow":     0.03,
    "black":      0.03,
}


def _colour_ratio(crop_hsv: np.ndarray, colour: str) -> float:
    """Return fraction of pixels in an HSV crop matching the given colour name."""
    n = crop_hsv.shape[0] * crop_hsv.shape[1]
    if n == 0:
        return 0.0
    if colour in _DUAL_RANGE:
        lo1, hi1 = _HSV_RANGES[colour]
        lo2, hi2 = _HSV_RANGES[colour + "2"]
        mask = cv2.inRange(crop_hsv, lo1, hi1) | cv2.inRange(crop_hsv, lo2, hi2)
    elif colour in _HSV_RANGES:
        lo, hi = _HSV_RANGES[colour]
        mask = cv2.inRange(crop_hsv, lo, hi)
    else:
        return 0.0
    return float(np.count_nonzero(mask)) / n


# ─── Classifier ───────────────────────────────────────────────────────────────

@dataclass
class TeamClassifierV2:
    """
    Classifies a single bounding-box crop as "our" / "opponent" / "uncertain".

    Parameters
    ----------
    our_color      : Primary jersey colour of our team (e.g. "white").
    opponent_color : Primary jersey colour of the opponent (e.g. "green").
    our_threshold  : Minimum colour ratio to declare "our"  (default 0.25).
    opp_threshold  : Minimum colour ratio to declare "opponent" (default 0.05).
    our_green_max  : Maximum opponent-colour ratio still allowed for "our"
                     (guards against false positives when lighting shifts, default 0.03).
    torso_y_range  : (y_lo_frac, y_hi_frac) of bbox height for torso strip.
    torso_x_margin : Fraction of bbox width to trim from each side (reduces edge noise).
    min_crop_area  : Minimum pixel count in the torso strip to attempt classification.
    """

    our_color:      str   = "white"
    opponent_color: str   = "green"
    our_threshold:  float = 0.25
    opp_threshold:  float = 0.05
    our_color_max:  float = 0.03   # max opponent-colour pixels in an "our" crop
    torso_y_range:  tuple[float, float] = (0.20, 0.65)
    torso_x_margin: float               = 0.15
    min_crop_area:  int                 = 300

    def _torso(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = crop_bgr.shape[:2]
        ty1  = int(h * self.torso_y_range[0])
        ty2  = int(h * self.torso_y_range[1])
        tx1  = int(w * self.torso_x_margin)
        tx2  = w - int(w * self.torso_x_margin)
        strip = crop_bgr[ty1:ty2, tx1:tx2]
        return strip if strip.size >= self.min_crop_area else None

    def _auto_opp_threshold(self) -> float:
        """Return per-colour default threshold if opp_threshold was not set manually."""
        return _OPP_THRESHOLDS.get(self.opponent_color, self.opp_threshold)

    def _auto_our_color_max(self) -> float:
        """Return per-opponent-colour maximum allowed opponent pixels in an 'our' crop.

        When the opponent wears white, jersey design elements and ice background
        cause high white ratios even in 'our' crops, so we need a relaxed limit.
        """
        return _OUR_COLOR_MAX.get(self.opponent_color, self.our_color_max)

    def classify_crop(
        self,
        crop_bgr: np.ndarray,
        pose_keypoints: Optional[dict] = None,
    ) -> tuple[str, float]:
        """
        Classify a player crop.

        Parameters
        ----------
        crop_bgr        : BGR image of the detected player bounding box.
        pose_keypoints  : (optional) dict with 'torso_mask' key containing a
                          binary mask (same shape as crop_bgr) to restrict
                          classification to the torso region only.
                          If None, uses the fixed torso strip heuristic.

        Returns
        -------
        (label, confidence)
            label      : "our" | "opponent" | "uncertain"
            confidence : float in [0, 1]
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return "uncertain", 0.0

        # ── 1. Extract torso region ───────────────────────────────────────────
        if pose_keypoints is not None and "torso_mask" in pose_keypoints:
            mask = pose_keypoints["torso_mask"]
            torso_bgr = cv2.bitwise_and(crop_bgr, crop_bgr, mask=mask)
            if torso_bgr.size < self.min_crop_area:
                torso_bgr = self._torso(crop_bgr)
        else:
            torso_bgr = self._torso(crop_bgr)

        if torso_bgr is None:
            return "uncertain", 0.0

        # ── 2. Convert to HSV ─────────────────────────────────────────────────
        torso_hsv = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2HSV)

        # ── 3. Compute colour ratios ──────────────────────────────────────────
        our_ratio  = _colour_ratio(torso_hsv, self.our_color)
        opp_ratio  = _colour_ratio(torso_hsv, self.opponent_color)

        # ── 4. Decision tree ──────────────────────────────────────────────────
        # Priority: opponent signal > our signal > uncertain
        # Rationale: distinctive colours beat white because ice/boards are white.
        eff_opp_threshold = self._auto_opp_threshold()
        if opp_ratio >= eff_opp_threshold:
            confidence = min(0.97, 0.50 + opp_ratio * 3.0)
            return "opponent", float(confidence)

        eff_our_color_max = self._auto_our_color_max()
        if our_ratio >= self.our_threshold and opp_ratio < eff_our_color_max:
            confidence = min(0.97, 0.50 + our_ratio)
            return "our", float(confidence)

        # ── 5. Uncertain ──────────────────────────────────────────────────────
        return "uncertain", 0.40

    def classify_batch(
        self,
        crops: list[np.ndarray],
        track_ids: Optional[list[int]] = None,
    ) -> list[dict]:
        """
        Classify a list of crops.  Returns list of dicts:
        {"team": str, "confidence": float, "track_id": int | None}
        """
        results = []
        for i, crop in enumerate(crops):
            tid = track_ids[i] if track_ids else None
            label, conf = self.classify_crop(crop)
            results.append({"team": label, "confidence": conf, "track_id": tid})
        return results


# ─── Factories ────────────────────────────────────────────────────────────────

def from_metadata(
    metadata_path: str | Path,
    file_id: str,
    swap_teams: bool = False,
) -> TeamClassifierV2:
    """Build a TeamClassifierV2 from ``data/videos/metadata.json``.

    Reads the ``uniform`` block for the given ``file_id``:
        ``our_color``      → classifier ``our_color``
        ``opponent_color`` → classifier ``opponent_color``

    Parameters
    ----------
    metadata_path : Path to ``data/videos/metadata.json``.
    file_id       : ``file_id`` value in the JSON (e.g. ``"ku_vs_mega"``).
    swap_teams    : If True, swap our/opponent roles (useful when filming from
                    the other bench).

    Example
    -------
    >>> clf = from_metadata("data/videos/metadata.json", "ku_vs_hldaegu")
    >>> # KU셀렉트 = red (our), HL대구 = white (opponent)
    """
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"metadata.json not found: {path}")

    with open(path, encoding="utf-8") as f:
        meta = json.load(f)

    video = next(
        (v for v in meta.get("videos", []) if v.get("file_id") == file_id),
        None,
    )
    if video is None:
        available = [v["file_id"] for v in meta.get("videos", [])]
        raise ValueError(
            f"file_id {file_id!r} not found in {path}. "
            f"Available: {available}"
        )

    uniform = video.get("uniform", {})
    our_color  = uniform.get("our_color",      "white")
    opp_color  = uniform.get("opponent_color", "green")

    if swap_teams:
        our_color, opp_color = opp_color, our_color

    return TeamClassifierV2(our_color=our_color, opponent_color=opp_color)


def load_from_roster(roster_path: str | Path) -> TeamClassifierV2:
    """
    Build a TeamClassifierV2 from a roster JSON that contains
    ``uniforms.home.primary_color`` and ``opponent_color.primary_color``.

    Falls back to white/green defaults if the fields are missing.

    Example roster JSON schema::

        {
          "team": "아이기스 U12",
          "uniforms": {
            "home": { "primary_color": "white", "hex": "#FFFFFF" }
          },
          "opponent_color": { "primary_color": "green", "hex": "#00A651" },
          ...
        }
    """
    path = Path(roster_path)
    if not path.exists():
        raise FileNotFoundError(f"Roster JSON not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    our_color = (
        data.get("uniforms", {})
            .get("home", {})
            .get("primary_color", "white")
    )
    opp_color = (
        data.get("opponent_color", {})
            .get("primary_color", "green")
    )

    return TeamClassifierV2(our_color=our_color, opponent_color=opp_color)


# ─── CLI validation helper ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import csv
    from pathlib import Path

    REPO_ROOT  = Path(__file__).resolve().parents[2]
    CROPS_DIR  = REPO_ROOT / "benchmark" / "labels" / "gt_crops"
    GT_CSV     = REPO_ROOT / "benchmark" / "labels" / "gt_labels_filled.csv"
    ROSTER     = REPO_ROOT / "benchmark" / "labels" / "aigis_roster.json"

    clf = load_from_roster(ROSTER)
    print(f"Classifier: our={clf.our_color!r}, opponent={clf.opponent_color!r}")
    print(f"Thresholds: our≥{clf.our_threshold}, opp≥{clf.opp_threshold}, our_color_max<{clf.our_color_max}")

    with open(GT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    tp_our = tp_opp = fp_our = fp_opp = uncertain = 0
    for row in rows:
        img = cv2.imread(str(CROPS_DIR / row["crop_filename"]))
        if img is None:
            continue
        label, conf = clf.classify_crop(img)
        tj = row["true_jersey"].strip()
        is_aigis   = tj not in ("wrong_team","unreadable","occluded","심판","goalie","non_player","")
        is_phoenix = tj == "wrong_team"
        if label == "uncertain":
            uncertain += 1
        elif label == "our" and is_aigis:
            tp_our += 1
        elif label == "our" and is_phoenix:
            fp_our += 1
        elif label == "opponent" and is_phoenix:
            tp_opp += 1
        elif label == "opponent" and is_aigis:
            fp_opp += 1

    total_eval = tp_our + tp_opp + fp_our + fp_opp
    acc = (tp_our + tp_opp) / total_eval if total_eval else 0
    print(f"\nGT validation (n=70):")
    print(f"  TP(our/Aigis)={tp_our}  TP(opp/Phoenix)={tp_opp}")
    print(f"  FP(our/Phoenix)={fp_our}  FP(opp/Aigis)={fp_opp}")
    print(f"  uncertain={uncertain}")
    print(f"  accuracy={100*acc:.1f}% on {total_eval} classified crops")
