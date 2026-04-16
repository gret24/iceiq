"""
pipeline/jersey_ocr.py
SmartJersey v2 — Phase 3: Jersey OCR for Aigis player identification.

Design
------
1. FrameQualityScorer
     size_score    = min(area / (150*150), 1.0)        weight 0.35
     blur_score    = min(Laplacian_var / 500.0, 1.0)   weight 0.35
     motion_score  = max(0, 1 - displacement/30)        weight 0.20
     aspect_score  = 1.0 if 0.3 < w/h < 0.8 else 0.5  weight 0.10

2. Jersey crop: y = 30%–65% of bbox height, x = centre 60% width

3. Preprocessing: INTER_CUBIC upscale to 96 px height → CLAHE →
   adaptive threshold (blockSize=15, C=8) → morph close → white padding

4. OCR on both normal & inverted crop; take best confidence.

5. snap_to_roster(): Levenshtein distance + numeric proximity
   AIGIS_ROSTER = [4, 11, 14, 25, 28, 44, 47]

6. Top-3 frame voting per track.

Test entry point (frames 5000–80000, every 15th frame of game1.mp4):
  python -m pipeline.jersey_ocr
"""
from __future__ import annotations

import logging
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── EasyOCR (soft-fail) ───────────────────────────────────────────────────────
try:
    import easyocr as _easyocr_mod
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available — OCR will return no results")

# ── Roster ────────────────────────────────────────────────────────────────────
AIGIS_ROSTER: List[int] = [4, 11, 14, 25, 28, 44, 47]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  FrameQualityScorer
# ─────────────────────────────────────────────────────────────────────────────

class FrameQualityScorer:
    """
    Score each bounding-box observation for OCR suitability.

    Weights
    -------
    size_score   0.35   large bbox → easier OCR
    blur_score   0.35   sharp crop → correct digit pixels
    motion_score 0.20   low inter-frame displacement → crisp crop
    aspect_score 0.10   upright player silhouette preferred

    Usage
    -----
    scorer = FrameQualityScorer()
    q = scorer.score(bbox, frame, prev_bbox=prev_bbox)
    """

    W_SIZE   = 0.35
    W_BLUR   = 0.35
    W_MOTION = 0.20
    W_ASPECT = 0.10

    def score(
        self,
        bbox: Tuple[float, float, float, float],
        frame: np.ndarray,
        prev_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> float:
        """
        Return composite quality score in [0, 1].

        Parameters
        ----------
        bbox      : (x1, y1, x2, y2) in pixels
        frame     : BGR video frame (used for blur measurement)
        prev_bbox : previous frame's bbox for motion estimation (None → perfect motion score)
        """
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 0.0)
        h = max(y2 - y1, 0.0)
        area = w * h

        # ── size ──────────────────────────────────────────────────────────────
        size_score = min(area / (150.0 * 150.0), 1.0)

        # ── blur (Laplacian variance on the jersey crop) ───────────────────────
        fh, fw = frame.shape[:2]
        cx1 = int(max(0, x1 + w * 0.20))
        cx2 = int(min(fw, x1 + w * 0.80))
        cy1 = int(max(0, y1 + h * 0.30))
        cy2 = int(min(fh, y1 + h * 0.65))
        blur_score = 0.0
        if cx2 > cx1 and cy2 > cy1:
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
                lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                blur_score = min(lap_var / 500.0, 1.0)

        # ── motion (displacement of bbox centre between frames) ────────────────
        if prev_bbox is not None:
            px1, py1, px2, py2 = prev_bbox
            cx_curr = (x1 + x2) / 2.0
            cy_curr = (y1 + y2) / 2.0
            cx_prev = (px1 + px2) / 2.0
            cy_prev = (py1 + py2) / 2.0
            displacement = math.hypot(cx_curr - cx_prev, cy_curr - cy_prev)
            motion_score = max(0.0, 1.0 - displacement / 30.0)
        else:
            motion_score = 1.0

        # ── aspect ────────────────────────────────────────────────────────────
        ratio = (w / h) if h > 0 else 0.0
        aspect_score = 1.0 if (0.3 < ratio < 0.8) else 0.5

        return (
            self.W_SIZE   * size_score
            + self.W_BLUR   * blur_score
            + self.W_MOTION * motion_score
            + self.W_ASPECT * aspect_score
        )

    def score_batch(
        self,
        observations: List[Tuple[Tuple, np.ndarray, Optional[Tuple]]],
    ) -> List[float]:
        """Convenience: score a list of (bbox, frame, prev_bbox) tuples."""
        return [self.score(bbox, frame, prev_bbox) for bbox, frame, prev_bbox in observations]


# ─────────────────────────────────────────────────────────────────────────────
# Levenshtein helper
# ─────────────────────────────────────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    """Standard dynamic-programming Levenshtein distance."""
    m, n = len(a), len(b)
    if m < n:
        a, b, m, n = b, a, n, m
    row = list(range(n + 1))
    for i in range(1, m + 1):
        prev, row[0] = row[0], i
        for j in range(1, n + 1):
            prev, row[j] = row[j], min(
                row[j] + 1,
                row[j - 1] + 1,
                prev + (0 if a[i - 1] == b[j - 1] else 1),
            )
    return row[n]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  JerseyOCR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _FrameVote:
    """One per-frame OCR result accumulated for a track."""
    number: Optional[int]
    confidence: float
    quality: float


class JerseyOCR:
    """
    Phase-3 jersey number OCR for Aigis players.

    Workflow per track
    ------------------
    1. Accumulate (frame, bbox, quality_score) observations.
    2. Select the top-3 quality frames.
    3. For each frame: crop → preprocess → OCR (normal + inverted) → snap_to_roster.
    4. Vote among the 3 results; return (number, confidence).

    Usage
    -----
    ocr = JerseyOCR()
    ocr.add_observation(track_id, frame, bbox, prev_bbox)
    result = ocr.get_result(track_id)   # (number_int or None, confidence)
    """

    def __init__(
        self,
        roster: List[int] = AIGIS_ROSTER,
        top_k: int = 3,
    ):
        self.roster = roster
        self.top_k = top_k
        self._scorer = FrameQualityScorer()

        # track_id → list of (quality, frame, bbox)
        self._obs: Dict[int, List[Tuple[float, np.ndarray, Tuple]]] = defaultdict(list)

        # OCR reader (lazy-init to avoid slow startup when not needed)
        self._reader: Optional[object] = None

    # ── OCR reader ────────────────────────────────────────────────────────────

    def _get_reader(self):
        if self._reader is None and _EASYOCR_AVAILABLE:
            try:
                self._reader = _easyocr_mod.Reader(["en"], gpu=False, verbose=False)
                logger.info("EasyOCR reader initialised")
            except Exception as exc:
                logger.error("EasyOCR init failed: %s", exc)
        return self._reader

    # ── 1. Accumulate observations ────────────────────────────────────────────

    def add_observation(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        prev_bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> float:
        """
        Score and store one frame observation for *track_id*.

        Returns the quality score (useful for real-time filtering).
        """
        q = self._scorer.score(bbox, frame, prev_bbox)
        # Store a copy of the frame crop to limit memory — only the jersey strip
        jersey_crop = self._extract_jersey_crop(frame, bbox)
        if jersey_crop is not None:
            self._obs[track_id].append((q, jersey_crop, bbox))
        return q

    # ── 2. Jersey crop ────────────────────────────────────────────────────────

    def _extract_jersey_crop(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """
        y = 30%–65% of bbox height (number panel is mid-torso, not shoulder/waist).
        x = centre 60% width (skip arm edges).
        """
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return None

        cx1 = int(max(0, x1 + bw * 0.20))
        cx2 = int(min(fw, x2 - bw * 0.20))
        cy1 = int(max(0, y1 + bh * 0.30))
        cy2 = int(min(fh, y1 + bh * 0.65))

        if cx2 <= cx1 or cy2 <= cy1:
            return None
        return frame[cy1:cy2, cx1:cx2].copy()

    # ── 3. Preprocessing ──────────────────────────────────────────────────────

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """
        INTER_CUBIC upscale to 96 px height → CLAHE →
        adaptive threshold (blockSize=15, C=8) → morph close → white padding.

        Returns an 8-bit single-channel image ready for OCR.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()

        # ── upscale to 96 px height ───────────────────────────────────────────
        h, w = gray.shape
        if h == 0 or w == 0:
            return gray
        scale = 96.0 / h
        new_w = max(1, int(w * scale))
        up = cv2.resize(gray, (new_w, 96), interpolation=cv2.INTER_CUBIC)

        # ── CLAHE ─────────────────────────────────────────────────────────────
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(up)

        # ── Adaptive threshold ─────────────────────────────────────────────────
        thresh = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=8,
        )

        # ── Morph close (connects broken digit strokes) ────────────────────────
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # ── White padding ─────────────────────────────────────────────────────
        pad = 8
        padded = cv2.copyMakeBorder(
            closed, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT, value=255,
        )
        return padded

    # ── 4. Run OCR (normal + inverted) ───────────────────────────────────────

    def _ocr_single(self, img: np.ndarray) -> List[Tuple[int, float]]:
        """
        Run EasyOCR on *img* (single-channel).
        Returns [(number_int, confidence), ...] filtered to 1–99.
        """
        reader = self._get_reader()
        if reader is None:
            return []
        try:
            results = reader.readtext(
                img,
                allowlist="0123456789",
                paragraph=False,
                min_size=4,
                detail=1,
            )
        except Exception as exc:
            logger.debug("EasyOCR readtext error: %s", exc)
            return []

        out: List[Tuple[int, float]] = []
        for (_, text, conf) in results:
            digits = "".join(c for c in text if c.isdigit())
            if not digits:
                continue
            try:
                n = int(digits)
                if 1 <= n <= 99:
                    out.append((n, float(conf)))
            except ValueError:
                pass
        return out

    def _run_ocr_on_crop(self, crop: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Preprocess → OCR on normal image AND inverted image.
        Return the (number, confidence) with the highest confidence.
        """
        if crop is None or crop.size == 0:
            return None, 0.0

        proc = self._preprocess(crop)
        inv  = cv2.bitwise_not(proc)

        cands_normal   = self._ocr_single(proc)
        cands_inverted = self._ocr_single(inv)

        all_cands = cands_normal + cands_inverted
        if not all_cands:
            return None, 0.0

        best_num, best_conf = max(all_cands, key=lambda x: x[1])
        return best_num, best_conf

    # ── 5. Roster constraint ──────────────────────────────────────────────────

    def snap_to_roster(
        self,
        raw_number: Optional[int],
        raw_conf: float,
        lev_threshold: int = 1,
    ) -> Tuple[Optional[int], float]:
        """
        Snap *raw_number* to the nearest roster entry using:
          1. Exact match → keep as-is.
          2. Levenshtein distance on string representation ≤ lev_threshold → snap.
          3. Numeric proximity (|raw - roster_n| ≤ 2) as tiebreaker.
          4. No match within threshold → return (None, 0.0).

        Confidence is penalised for non-exact matches.
        """
        if raw_number is None:
            return None, 0.0

        # Exact match
        if raw_number in self.roster:
            return raw_number, raw_conf

        raw_str = str(raw_number)
        best_roster = None
        best_dist = lev_threshold + 1
        best_num_dist = float("inf")

        for candidate in self.roster:
            cand_str = str(candidate)
            ld = _levenshtein(raw_str, cand_str)
            nd = abs(raw_number - candidate)

            if ld < best_dist or (ld == best_dist and nd < best_num_dist):
                best_dist = ld
                best_num_dist = nd
                best_roster = candidate

        if best_dist <= lev_threshold and best_roster is not None:
            # Penalise confidence for the snap
            penalty = 0.15 * best_dist
            return best_roster, max(0.0, raw_conf - penalty)

        return None, 0.0

    # ── 6. Top-3 voting ───────────────────────────────────────────────────────

    def _vote_top3(self, votes: List[_FrameVote]) -> Tuple[Optional[int], float]:
        """
        Weighted majority vote among up to top_k frame results.

        Each vote's weight = quality_score × confidence.
        Returns (winning_number, normalised_confidence).
        """
        if not votes:
            return None, 0.0

        # Accumulate weighted scores per candidate number
        weighted: Dict[int, float] = defaultdict(float)
        total_weight = 0.0
        for v in votes:
            if v.number is None:
                continue
            w = v.quality * v.confidence
            weighted[v.number] += w
            total_weight += w

        if not weighted:
            return None, 0.0

        best_num = max(weighted, key=lambda k: weighted[k])
        conf = weighted[best_num] / total_weight if total_weight > 0 else 0.0
        return best_num, conf

    # ── Public: get result for a track ───────────────────────────────────────

    def get_result(
        self, track_id: int
    ) -> Tuple[Optional[int], float, int]:
        """
        Select top-k quality frames, run OCR on each, and vote.

        Returns
        -------
        (number_int, confidence, legible_frames)
        """
        obs = self._obs.get(track_id, [])
        if not obs:
            return None, 0.0, 0

        # Sort by quality descending, take top-k
        top = sorted(obs, key=lambda x: x[0], reverse=True)[: self.top_k]

        votes: List[_FrameVote] = []
        for quality, crop, _bbox in top:
            raw_num, raw_conf = self._run_ocr_on_crop(crop)
            snapped_num, snapped_conf = self.snap_to_roster(raw_num, raw_conf)
            votes.append(_FrameVote(
                number=snapped_num,
                confidence=snapped_conf,
                quality=quality,
            ))

        legible = sum(1 for v in votes if v.number is not None)
        final_num, final_conf = self._vote_top3(votes)
        return final_num, final_conf, legible

    def get_all_results(self) -> Dict[int, Tuple[Optional[int], float, int]]:
        """Return get_result() for every accumulated track."""
        return {tid: self.get_result(tid) for tid in self._obs}

    def track_ids(self) -> List[int]:
        return list(self._obs.keys())

    def observation_count(self, track_id: int) -> int:
        return len(self._obs.get(track_id, []))


# ─────────────────────────────────────────────────────────────────────────────
# Test harness
# ─────────────────────────────────────────────────────────────────────────────

def run_test(
    video_path: str = "data/videos/game1.mp4",
    start_frame: int = 5000,
    end_frame: int = 80000,
    sample_every: int = 15,
) -> None:
    """
    Test Phase 3 OCR on game1.mp4 frames 5000–80000 (every 15th frame).

    Strategy
    --------
    * HockeyPlayerDetector → detections per frame
    * HockeyTeamClassifier → keep only Aigis players
    * Spatial-grid pseudo-tracker (16 × 16 cells) → stable track IDs
    * JerseyOCR.add_observation() → accumulate quality observations
    * After processing: get_all_results() → print per-track jersey numbers

    Expected outcome: ≥ 4 of 7 Aigis roster numbers correctly identified.
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vpath = os.path.join(base, video_path)

    if not os.path.exists(vpath):
        print(f"Video not found: {vpath}")
        return

    sys.path.insert(0, base)

    # ── Load detector ──────────────────────────────────────────────────────
    try:
        from pipeline.detection import HockeyPlayerDetector
        mpath = os.path.join(base, "yolov8m.pt")
        if not os.path.exists(mpath):
            mpath = os.path.join(base, "yolov8s.pt")
        if not os.path.exists(mpath):
            mpath = os.path.join(base, "yolov8n.pt")
        detector = HockeyPlayerDetector(model_path=mpath, conf=0.15)
        print(f"Detector loaded: {mpath}")
    except Exception as exc:
        print(f"Detector load failed: {exc}")
        return

    # ── Load team classifier ───────────────────────────────────────────────
    try:
        from pipeline.team_classification import HockeyTeamClassifier
        classifier = HockeyTeamClassifier()
        print("Team classifier loaded")
    except Exception as exc:
        print(f"Team classifier load failed: {exc}")
        classifier = None

    # ── Jersey OCR engine ──────────────────────────────────────────────────
    ocr = JerseyOCR(roster=AIGIS_ROSTER, top_k=3)
    scorer = FrameQualityScorer()

    # ── Pseudo-tracker: 16×16 spatial grid ────────────────────────────────
    GRID = 16
    grid_to_tid: Dict[Tuple[int, int], int] = {}
    next_tid: List[int] = [0]

    # Previous bbox per pseudo-track (for motion score)
    prev_bboxes: Dict[int, Tuple] = {}

    def grid_cell(cx: float, cy: float, fw: int, fh: int) -> Tuple[int, int]:
        gx = int(cx / fw * GRID)
        gy = int(cy / fh * GRID)
        return (min(gx, GRID - 1), min(gy, GRID - 1))

    # ── Open video ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print(f"Cannot open video: {vpath}")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {total} frames @ {fps:.1f} fps")
    print(f"Sampling frames {start_frame}–{end_frame} every {sample_every}th frame")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx       = start_frame
    frames_sampled  = 0
    detections_kept = 0
    aigis_obs       = 0

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % sample_every == 0:
            frames_sampled += 1
            fh, fw = frame.shape[:2]

            det = detector.detect_frame(frame)
            detections_kept += det["count"]

            for box, score in zip(det["boxes"], det["scores"]):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                cell = grid_cell(cx, cy, fw, fh)

                if cell not in grid_to_tid:
                    grid_to_tid[cell] = next_tid[0]
                    next_tid[0] += 1
                tid = grid_to_tid[cell]

                # ── Team classification ────────────────────────────────────
                team = "aigis"
                if classifier is not None:
                    try:
                        team, _tc = classifier.classify(frame, (x1, y1, x2, y2), tid)
                    except Exception:
                        pass

                if team != "aigis":
                    prev_bboxes[tid] = (x1, y1, x2, y2)
                    continue

                # ── Accumulate observation ─────────────────────────────────
                prev_bb = prev_bboxes.get(tid)
                ocr.add_observation(tid, frame, (x1, y1, x2, y2), prev_bb)
                prev_bboxes[tid] = (x1, y1, x2, y2)
                aigis_obs += 1

            if frames_sampled % 100 == 0:
                print(f"  frame {frame_idx:>6} | sampled {frames_sampled} | "
                      f"aigis obs {aigis_obs}")

        frame_idx += 1

    cap.release()

    # ── Results ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SMARTJERSEY v2 PHASE 3 — OCR RESULTS")
    print("=" * 65)
    print(f"Frames sampled     : {frames_sampled}")
    print(f"Detections total   : {detections_kept}")
    print(f"Aigis observations : {aigis_obs}")
    print(f"Pseudo-tracks seen : {len(ocr.track_ids())}")
    print()

    all_results = ocr.get_all_results()

    # Filter: tracks with ≥ 3 observations (enough data to be interesting)
    significant = {
        tid: (num, conf, leg)
        for tid, (num, conf, leg) in all_results.items()
        if ocr.observation_count(tid) >= 3 and num is not None
    }

    # Deduplicate by jersey number (keep highest-confidence assignment per number)
    best_per_number: Dict[int, Tuple[int, float, int]] = {}
    for tid, (num, conf, leg) in significant.items():
        if num is None:
            continue
        if num not in best_per_number or conf > best_per_number[num][1]:
            best_per_number[num] = (tid, conf, leg)

    identified = sorted(best_per_number.keys())
    missed     = [n for n in AIGIS_ROSTER if n not in best_per_number]

    print(f"Roster   : {AIGIS_ROSTER}")
    print(f"Identified ({len(identified)}): {identified}")
    print(f"Missed     ({len(missed)}):  {missed}")
    print()

    print(f"{'#':>4}  {'Track':>6}  {'Conf':>6}  {'Legible/top-3':>14}  {'Obs':>5}")
    print("  " + "-" * 42)
    for num in identified:
        tid, conf, leg = best_per_number[num]
        obs = ocr.observation_count(tid)
        print(f"  {num:>2}  {tid:>6}  {conf:>6.3f}  {leg:>14}  {obs:>5}")

    print()
    if len(identified) >= 4:
        print(f"PASS — {len(identified)} Aigis players identified (target ≥ 4)")
    else:
        print(f"WARN — only {len(identified)} identified (target ≥ 4); "
              f"consider lowering lev_threshold or sampling more frames")

    print("=" * 65)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
    )
    run_test()
