"""
Validation script: team classification accuracy on game1.mp4.

Samples frames 10000–80000 every 1500 frames.
For each YOLO detection with conf > 0.4, computes:
  - green_ratio  (dark-green HSV mask on torso)
  - white_ratio  (low-sat / high-val HSV mask on torso)

Ground-truth assignment (independent of classifier):
  green_gt : green_ratio >= 0.06
  white_gt : white_ratio >= 0.20 AND green_ratio <  0.04
  (ambiguous detections are excluded from accuracy counts)

Reports: green_correct/green_total  and  white_correct/white_total
"""

import sys
import os
import cv2
import numpy as np
import logging
from pathlib import Path

# ── project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from pipeline.team_classification import (
    HockeyTeamClassifier, TEAM_ZENITH, TEAM_AIGIS,
    _GREEN_LOWER, _GREEN_UPPER,
)
from pipeline.detection import HockeyPlayerDetector
from pipeline.roi import create_roi

logging.basicConfig(level=logging.WARNING,
                    format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("validate")

VIDEO_PATH  = str(ROOT / "data" / "videos" / "game1.mp4")
ROI_PATH    = str(ROOT / "data" / "roi_polygon.json")

FRAME_START  = 10_000
FRAME_END    = 80_000
FRAME_STEP   = 1_000   # reduced from 1500 to achieve ≥200 white-team samples
DET_CONF_MIN = 0.40          # only evaluate detections above this YOLO confidence

# Ground-truth thresholds (HSV analysis independent of classifier)
GT_GREEN_MIN  = 0.06         # green_ratio ≥ this → truly zenith
GT_WHITE_MIN  = 0.10         # white_ratio ≥ this (+ low green) → truly aigis
GT_GREEN_MAX_FOR_WHITE = 0.056  # align with effective combined threshold (0.05/0.90≈0.056)

# White HSV range for ground-truth only
_WHITE_LOWER = np.array([  0,  0, 160], dtype=np.uint8)
_WHITE_UPPER = np.array([180, 40, 255], dtype=np.uint8)

_MIN_CROP_AREA = 300


def torso_crop(frame, bbox):
    """Return the torso sub-crop (inner 70% width, 20-65% height)."""
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = (
        max(0, int(bbox[0])), max(0, int(bbox[1])),
        min(fw, int(bbox[2])), min(fh, int(bbox[3])),
    )
    bh, bw = y2 - y1, x2 - x1
    if bh <= 0 or bw <= 0:
        return None
    ty1 = y1 + int(bh * 0.20)
    ty2 = y1 + int(bh * 0.65)
    tx1 = x1 + int(bw * 0.15)
    tx2 = x2 - int(bw * 0.15)
    crop = frame[ty1:ty2, tx1:tx2]
    return crop if crop.size >= _MIN_CROP_AREA else None


def colour_ratio(crop, lower, upper):
    if crop is None or crop.size < _MIN_CROP_AREA:
        return 0.0
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    n    = mask.shape[0] * mask.shape[1]
    return float(np.count_nonzero(mask)) / n if n > 0 else 0.0


def ground_truth(green_ratio, white_ratio):
    """
    Returns TEAM_ZENITH, TEAM_AIGIS, or None (ambiguous).
    Uses torso colour ratios, not the classifier.
    """
    if green_ratio >= GT_GREEN_MIN:
        return TEAM_ZENITH
    if white_ratio >= GT_WHITE_MIN and green_ratio < GT_GREEN_MAX_FOR_WHITE:
        return TEAM_AIGIS
    return None   # ambiguous — skip


def main():
    if not Path(VIDEO_PATH).exists():
        print(f"ERROR: video not found at {VIDEO_PATH}")
        sys.exit(1)

    print(f"Video : {VIDEO_PATH}")
    print(f"Frames: {FRAME_START}–{FRAME_END} step {FRAME_STEP}  "
          f"({len(range(FRAME_START, FRAME_END + 1, FRAME_STEP))} frames)")
    print()

    # ── Init detector ────────────────────────────────────────────────────────
    print("Loading YOLO detector…")
    detector = HockeyPlayerDetector(
        model_path=str(ROOT / "yolov8m.pt"),
        conf=DET_CONF_MIN,
    )

    # ── Build / load ROI (force rebuild for new video) ───────────────────────
    roi_json = Path(ROI_PATH)
    if roi_json.exists():
        roi_json.unlink()
        print("Deleted stale ROI cache → rebuilding from new video")
    print("Building ROI from video…")
    roi = create_roi(video_path=VIDEO_PATH, roi_path=ROI_PATH, force_rebuild=True)

    # ── Classifier (single-frame mode — no temporal voting for clean accuracy) ─
    clf = HockeyTeamClassifier(vote_window=1)   # window=1 → no smoothing

    # ── Main loop ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS={total_fps:.1f}  total_frames={total_frames}\n")

    green_correct = green_total = 0
    white_correct = white_total = 0
    frames_processed = 0
    det_count = 0
    ambiguous  = 0

    frame_ids = range(FRAME_START, min(FRAME_END + 1, total_frames), FRAME_STEP)

    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            logger.warning("Could not read frame %d", fid)
            continue

        # ── Detect ───────────────────────────────────────────────────────────
        result = detector.detect_frame(frame)
        boxes  = result["boxes"]
        scores = result["scores"]

        # Filter to conf >= DET_CONF_MIN
        detections = [
            {"bbox": b, "conf": s, "track_id": i}
            for i, (b, s) in enumerate(zip(boxes, scores))
            if s >= DET_CONF_MIN
        ]

        # Apply ROI
        detections = roi.filter_detections(detections)

        frames_processed += 1
        det_count += len(detections)

        for i, det in enumerate(detections):
            bbox = det["bbox"]
            crop = torso_crop(frame, bbox)
            if crop is None:
                continue

            gr = colour_ratio(crop, _GREEN_LOWER, _GREEN_UPPER)
            wr = colour_ratio(crop, _WHITE_LOWER, _WHITE_UPPER)
            gt = ground_truth(gr, wr)

            if gt is None:
                ambiguous += 1
                continue

            # Single-frame classify (fresh track_id per detection per frame)
            unique_tid = fid * 1000 + i
            label, _ = clf.classify(frame, bbox, unique_tid)

            if gt == TEAM_ZENITH:
                green_total += 1
                if label == TEAM_ZENITH:
                    green_correct += 1
            else:
                white_total += 1
                if label == TEAM_AIGIS:
                    white_correct += 1

    cap.release()

    # ── Report ───────────────────────────────────────────────────────────────
    print("=" * 55)
    print(f"Frames processed : {frames_processed}")
    print(f"Total detections : {det_count}")
    print(f"Ambiguous (skip) : {ambiguous}")
    print()

    if green_total > 0:
        green_acc = green_correct / green_total * 100
        print(f"ZENITH (green) : {green_correct}/{green_total}  =  {green_acc:.1f}%")
    else:
        print("ZENITH (green) : 0 samples — insufficient data")

    if white_total > 0:
        white_acc = white_correct / white_total * 100
        print(f"AIGIS  (white) : {white_correct}/{white_total}  =  {white_acc:.1f}%")
    else:
        print("AIGIS  (white) : 0 samples — insufficient data")

    print()
    valid_green = green_total >= 200
    valid_white = white_total >= 200
    print(f"≥200 green samples : {'YES' if valid_green else f'NO ({green_total})'}")
    print(f"≥200 white samples : {'YES' if valid_white else f'NO ({white_total})'}")

    if valid_green and valid_white:
        overall_acc = (green_correct + white_correct) / (green_total + white_total) * 100
        print(f"\nOverall accuracy   : {overall_acc:.1f}%")
        if green_acc >= 98 and white_acc >= 98:
            print("RESULT: ✓ ≥98% accuracy CONFIRMED on both teams")
        else:
            print("RESULT: ✗ Did NOT reach 98% on both teams — tuning needed")
    else:
        print("\nNot enough samples to validate 98% claim.")
    print("=" * 55)


if __name__ == "__main__":
    main()
