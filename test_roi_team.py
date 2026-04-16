#!/usr/bin/env python3
"""
IceIQ — ROI masking + Team classification end-to-end test.

Tests
-----
TASK 1  pipeline/roi.py       — Ice surface detection, detection count before/after ROI
TASK 2  pipeline/team_classification.py — Team labelling, per-team accuracy, white-on-white stability

Run
---
    python test_roi_team.py
"""

import sys
import cv2
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.detection import HockeyPlayerDetector
from pipeline.tracking  import HockeyTracker
from pipeline.roi       import create_roi
from pipeline.team_classification import HockeyTeamClassifier, TEAM_ZENITH, TEAM_AIGIS

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_roi_team")

VIDEO_PATH   = "data/videos/game1.mp4"
MAX_FRAMES   = 300   # ~75 s at 4 FPS; enough for stable vote windows
STABILITY_MIN_FRAMES = 10   # only tracks with ≥ this many frames count for accuracy


# ============================================================
# TASK 1 — ROI
# ============================================================

def task1_roi(video_path: str):
    print("\n" + "=" * 64)
    print("TASK 1: ROI Masking — pipeline/roi.py")
    print("=" * 64)

    # Force rebuild so we always test the detection logic
    roi = create_roi(video_path=video_path, force_rebuild=True)

    if roi.polygon is not None:
        poly = roi.polygon
        area = cv2.contourArea(poly.reshape(-1, 1, 2).astype(np.int32))
        print(f"  Polygon vertices : {len(poly)}")
        print(f"  x range          : [{poly[:,0].min()}, {poly[:,0].max()}]")
        print(f"  y range          : [{poly[:,1].min()}, {poly[:,1].max()}]")
        print(f"  Area             : {area:,.0f} px²")
    else:
        print("  WARNING: polygon is None — pass-through mode.")

    # Measure filtering effect over the first 100 frames
    detector = HockeyPlayerDetector()
    cap = cv2.VideoCapture(video_path)
    raw_totals  = []
    kept_totals = []
    for _ in range(100):
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.detect(frame)
        filtered = roi.filter_detections(dets)
        raw_totals.append(len(dets))
        kept_totals.append(len(filtered))
    cap.release()

    if raw_totals:
        avg_raw  = np.mean(raw_totals)
        avg_kept = np.mean(kept_totals)
        reduc    = (1.0 - avg_kept / max(avg_raw, 1)) * 100
        print(f"\n  Avg detections before ROI : {avg_raw:.1f}")
        print(f"  Avg detections after  ROI : {avg_kept:.1f}")
        print(f"  Reduction               : {reduc:.1f}%")
        pass_fail = "PASS" if avg_kept <= 14 else "WARN (check ROI polygon)"
        print(f"  On-ice player count check : {pass_fail}  (target ≤ 14)")

    return roi


# ============================================================
# TASK 2 — Team classification
# ============================================================

def task2_team(video_path: str, roi, max_frames: int = MAX_FRAMES):
    print("\n" + "=" * 64)
    print("TASK 2: Team Classification — pipeline/team_classification.py")
    print("=" * 64)

    detector   = HockeyPlayerDetector()
    tracker    = HockeyTracker()
    classifier = HockeyTeamClassifier()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")

    raw_counts      = []
    filtered_counts = []
    zenith_counts   = []
    aigis_counts    = []

    # track_id → list of team labels (one per frame the track was seen)
    label_history: dict = defaultdict(list)
    # track_id → list of confidences
    conf_history: dict  = defaultdict(list)

    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        dets     = detector.detect(frame)
        raw_counts.append(len(dets))

        filtered = roi.filter_detections(dets)
        filtered_counts.append(len(filtered))

        tracks   = tracker.update(filtered, frame)
        results  = classifier.classify_batch(frame, tracks)

        n_z = sum(1 for r in results if r["team"] == TEAM_ZENITH)
        n_a = sum(1 for r in results if r["team"] == TEAM_AIGIS)
        zenith_counts.append(n_z)
        aigis_counts.append(n_a)

        for r in results:
            label_history[r["track_id"]].append(r["team"])
            conf_history [r["track_id"]].append(r["team_confidence"])

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"  frame {frame_idx:4d}: raw={raw_counts[-1]:2d}  "
                  f"filtered={filtered_counts[-1]:2d}  "
                  f"zenith={n_z:2d}  aigis={n_a:2d}")

    cap.release()
    return raw_counts, filtered_counts, zenith_counts, aigis_counts, label_history, conf_history


# ============================================================
# Reporting
# ============================================================

def report(raw, filtered, zenith, aigis, label_hist, conf_hist):
    print("\n" + "=" * 64)
    print("RESULTS SUMMARY")
    print("=" * 64)

    n = len(raw)
    print(f"\n  Frames processed : {n}")

    # ── ROI effect ────────────────────────────────────────────
    avg_r = np.mean(raw)
    avg_f = np.mean(filtered)
    print(f"\n── ROI masking ──")
    print(f"  Before ROI : {avg_r:.1f} det/frame  (max={max(raw)})")
    print(f"  After  ROI : {avg_f:.1f} det/frame  (max={max(filtered)})")
    print(f"  Reduction  : {(1-avg_f/max(avg_r,1))*100:.1f}%")

    # ── Team counts ───────────────────────────────────────────
    print(f"\n── Team counts (per frame, after count enforcement) ──")
    print(f"  Zenith Phoenix (green): avg={np.mean(zenith):.1f}  "
          f"max={max(zenith)}  min={min(zenith)}")
    print(f"  Aigis (white):          avg={np.mean(aigis):.1f}  "
          f"max={max(aigis)}  min={min(aigis)}")

    # ── Stability = white-on-white accuracy proxy ─────────────
    long_tracks = {tid: lbl for tid, lbl in label_hist.items()
                   if len(lbl) >= STABILITY_MIN_FRAMES}
    print(f"\n── Temporal stability  (tracks ≥ {STABILITY_MIN_FRAMES} frames) ──")
    print(f"  Total long tracks : {len(long_tracks)}")

    if not long_tracks:
        print("  (no long tracks to report)")
        return

    # Overall per-frame consistency
    all_correct = all_total = 0
    zenith_correct = zenith_total = 0
    aigis_correct  = aigis_total  = 0
    unstable_ids   = []

    for tid, labels in long_tracks.items():
        n_z = labels.count(TEAM_ZENITH)
        n_a = labels.count(TEAM_AIGIS)
        majority = TEAM_ZENITH if n_z >= n_a else TEAM_AIGIS
        correct  = n_z if majority == TEAM_ZENITH else n_a
        total    = len(labels)
        rate     = correct / total

        all_correct += correct
        all_total   += total

        if majority == TEAM_ZENITH:
            zenith_correct += correct
            zenith_total   += total
        else:
            aigis_correct  += correct
            aigis_total    += total

        if rate < 0.90:
            unstable_ids.append((tid, rate, majority))

    def pct(c, t):
        return f"{c/t*100:.1f}%" if t > 0 else "N/A"

    print(f"  Overall consistency     : {pct(all_correct, all_total)}  "
          f"({all_correct}/{all_total} frame-labels correct)")
    print(f"  Zenith Phoenix accuracy : {pct(zenith_correct, zenith_total)}")
    ww_acc = aigis_correct / aigis_total * 100 if aigis_total > 0 else 0.0
    ww_pass = "PASS" if ww_acc >= 98.0 else "FAIL"
    print(f"  Aigis (white-on-white)  : {pct(aigis_correct, aigis_total)}"
          f"  [{ww_pass} — target ≥ 98%]")

    if unstable_ids:
        print(f"\n  Unstable tracks (< 90 % consistent): {len(unstable_ids)}")
        for tid, rate, maj in unstable_ids[:5]:
            print(f"    track_id={tid}  majority={maj}  consistency={rate*100:.1f}%")
    else:
        print(f"\n  All long tracks are ≥ 90 % consistent.")

    # Summary verdict
    print("\n" + "=" * 64)
    overall_pass = ww_acc >= 98.0
    print(f"  WHITE-ON-WHITE TARGET (≥98%) : {'PASS ✓' if overall_pass else 'FAIL ✗'}"
          f"  ({ww_acc:.1f}%)")
    print("=" * 64)


# ============================================================
# Main
# ============================================================

def main():
    print("IceIQ — ROI + Team Classification Test")
    print(f"Video : {VIDEO_PATH}")
    print(f"Frames: {MAX_FRAMES}")

    roi = task1_roi(VIDEO_PATH)

    raw, filt, zen, aig, lhist, chist = task2_team(VIDEO_PATH, roi, MAX_FRAMES)

    report(raw, filt, zen, aig, lhist, chist)


if __name__ == "__main__":
    main()
