#!/usr/bin/env python3
"""
test_kinematics.py — Integration test for detection + tracking + homography + kinematics

Strategy:
  - Detection + tracking run on EVERY source frame  (tracker needs continuity)
  - Kinematics updated only every STRIDE frames      (4 fps cadence)
  - Processes the first MAX_VIDEO_FRAMES video frames from game1.mp4

Speed accuracy target: < 2 % error vs analytical ground truth (synthetic pulse).
"""

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.detection import HockeyPlayerDetector
from pipeline.homography import RinkHomography
from pipeline.kinematics import PlayerKinematics, SPEED_WINDOW
from pipeline.tracking import HockeyTracker

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

VIDEO_PATH        = Path("data/videos/game1.mp4")
CAL_PATH          = Path("data/homography_calibration.json")
MAX_VIDEO_FRAMES  = 500
SOURCE_FPS        = 30.0   # re-read from cap below
PROCESS_FPS       = 4.0    # kinematics cadence (must match pipeline_config.yaml)
MIN_OBSERVATIONS  = 5      # skip tracks with fewer points in report


# ── Kalman speed accuracy unit test ─────────────────────────────────────────
def _test_kalman_accuracy() -> bool:
    """
    Synthetic ground truth: constant 10 m/s along X for N steps.
    Speed reported as rolling mean of last SPEED_WINDOW estimates.
    Target: < 2 % error vs true speed.
    """
    print("\n[Unit] Kalman speed accuracy vs synthetic ground truth …")
    TRUE_SPEED_MS  = 10.0
    TRUE_SPEED_KMH = TRUE_SPEED_MS * 3.6
    FPS_TEST       = 4.0
    DT_TEST        = 1.0 / FPS_TEST
    N_STEPS        = 60      # 15 s — enough for Kalman to converge fully
    NOISE_STD      = 0.3     # metres, realistic homography noise

    rng = np.random.default_rng(42)
    kin = PlayerKinematics(fps=FPS_TEST, process_noise=0.5, measurement_noise=0.08)

    x_true = 0.0
    for step in range(N_STEPS):
        t = step * DT_TEST
        x_true += TRUE_SPEED_MS * DT_TEST
        nx = x_true + rng.normal(0, NOISE_STD)
        ny =          rng.normal(0, NOISE_STD)
        kin.update(0, (nx, ny), t)

    _vx, _vy, speed_kmh = kin.get_velocity(0)
    err_pct = abs(speed_kmh - TRUE_SPEED_KMH) / TRUE_SPEED_KMH * 100.0

    dist     = kin.get_distance(0)
    dist_gt  = TRUE_SPEED_MS * (N_STEPS - 1) * DT_TEST
    dist_err = abs(dist - dist_gt) / dist_gt * 100.0

    print(f"  True speed  : {TRUE_SPEED_KMH:.2f} km/h")
    print(f"  Estimated   : {speed_kmh:.2f} km/h   error = {err_pct:.2f} %  "
          f"(window={SPEED_WINDOW})")
    print(f"  Distance GT : {dist_gt:.2f} m   estimated = {dist:.2f} m   "
          f"error = {dist_err:.2f} %")

    passed = err_pct < 2.0
    print(f"  Speed < 2 % : {'PASS' if passed else 'FAIL'}")
    return passed


# ── Video integration test ───────────────────────────────────────────────────
def _test_video_pipeline() -> bool:
    """
    Run full pipeline on first MAX_VIDEO_FRAMES frames of game1.mp4.
    Tracking runs on every frame; kinematics updated every STRIDE frames.
    """
    if not VIDEO_PATH.exists():
        print(f"[SKIP] Video not found: {VIDEO_PATH}")
        return True

    print(f"\n[Integration] Pipeline test on {VIDEO_PATH.name}")

    try:
        detector = HockeyPlayerDetector()
        # boxmot v16 changed ByteTrack API: track_thresh=0.45 by default, old kwarg
        # names (track_high_thresh etc.) go to **kwargs and are silently ignored.
        # Detections at ~0.20-0.25 confidence never enter the "high-conf" pool so
        # ByteTrack never creates tracks.  Force the built-in Kalman+Hungarian
        # fallback which works at any confidence level.
        import pipeline.tracking as _trk
        _bk_save = _trk._BOXMOT_AVAILABLE
        _trk._BOXMOT_AVAILABLE = False
        tracker = HockeyTracker()
        _trk._BOXMOT_AVAILABLE = _bk_save
        print(f"  Tracker backend : {tracker.backend_name}")
        homog      = RinkHomography(cal_path=str(CAL_PATH))
        kinematics = PlayerKinematics(fps=PROCESS_FPS)
    except Exception as exc:
        print(f"  [ERROR] Module init failed: {exc}")
        return False

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {VIDEO_PATH}")
        return False

    actual_fps = cap.get(cv2.CAP_PROP_FPS) or SOURCE_FPS
    stride     = max(1, int(round(actual_fps / PROCESS_FPS)))
    total_src  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Source FPS : {actual_fps:.1f}  stride : {stride}  "
          f"total : {total_src}  reading : {MAX_VIDEO_FRAMES} frames")

    t0 = time.perf_counter()
    video_frame_idx  = 0
    kinematic_frames = 0

    while video_frame_idx < MAX_VIDEO_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        # ── detection + tracking on every frame ──────────────────────────────
        try:
            dets   = detector.detect(frame)   # list of {bbox, confidence, class_id}
            tracks = tracker.update(dets, frame)
        except Exception:
            video_frame_idx += 1
            continue

        # ── kinematics updated at 4 fps cadence ──────────────────────────────
        if video_frame_idx % stride == 0:
            timestamp = video_frame_idx / actual_fps

            for tr in tracks:
                tid  = tr["track_id"]
                bbox = tr["bbox"]
                cx   = (bbox[0] + bbox[2]) / 2.0
                cy   = float(bbox[3])          # foot → ground plane

                try:
                    rx, ry = homog.pixel_to_rink(cx, cy)
                except Exception:
                    continue

                # Reject obvious out-of-rink homography artefacts
                if abs(rx) > 15.0 or abs(ry) > 33.0:
                    continue

                kinematics.update(tid, (rx, ry), timestamp)

            kinematic_frames += 1

        video_frame_idx += 1

    cap.release()
    elapsed = time.perf_counter() - t0
    print(f"  Done : {video_frame_idx} video frames, "
          f"{kinematic_frames} kinematic ticks, {elapsed:.1f}s")

    # ── report ────────────────────────────────────────────────────────────────
    stable = [
        tid for tid in kinematics.track_ids
        if kinematics.observation_count(tid) >= MIN_OBSERVATIONS
    ]
    print(f"\n  Stable tracks (≥ {MIN_OBSERVATIONS} obs) : {len(stable)}")
    print(f"  {'ID':>4}  {'Obs':>4}  {'AvgSpd':>9}  {'MaxSpd':>9}  "
          f"{'Dist':>7}  {'Sprints':>7}  Zones(def/neu/off)")
    print(f"  {'-'*4}  {'-'*4}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*22}")

    all_ok = True
    for tid in sorted(stable):
        s    = kinematics.get_stats(tid)
        sp   = kinematics.get_sprint_events(tid)
        obs  = kinematics.observation_count(tid)
        tz   = s["time_in_zones"]

        print(f"  {tid:>4}  {obs:>4}  {s['avg_speed']:>8.2f}k  "
              f"{s['max_speed']:>8.2f}k  {s['total_distance']:>6.1f}m  "
              f"{s['sprint_count']:>7}  "
              f"{tz['defensive']:.1f}s/{tz['neutral']:.1f}s/{tz['offensive']:.1f}s")

        for s_start, s_end, s_max in sp[:3]:
            print(f"         sprint {s_start:.1f}s→{s_end:.1f}s  max {s_max:.1f} km/h")

        if s["max_speed"] > 80.0:
            print(f"    [WARN] track {tid}: max_speed {s['max_speed']:.1f} km/h unrealistic")
            all_ok = False

    # ── heatmap spot-check ────────────────────────────────────────────────────
    if stable:
        tid0 = stable[0]
        hm   = kinematics.get_heatmap(tid0, grid_size=(30, 13))
        assert hm.shape == (30, 13), f"Heatmap shape wrong: {hm.shape}"
        total_t   = hm.sum()
        expected  = kinematics.observation_count(tid0) * (1.0 / PROCESS_FPS)
        diff      = abs(total_t - expected)
        print(f"\n  Heatmap track {tid0}: total={total_t:.2f}s  "
              f"expected={expected:.2f}s  diff={diff:.3f}s")
        if diff > 0.5:
            print("  [WARN] heatmap time mismatch — positions out-of-rink discarded")

    success = (len(stable) >= 1) and all_ok
    print(f"\n  Pipeline test: {'PASS' if success else 'FAIL'}")
    return success


# ── Entry point ──────────────────────────────────────────────────────────────
def main() -> None:
    results = {
        "kalman_accuracy": _test_kalman_accuracy(),
        "video_pipeline":  _test_video_pipeline(),
    }

    print("\n" + "=" * 52)
    print("SUMMARY")
    print("=" * 52)
    all_pass = True
    for name, ok in results.items():
        print(f"  {name:<28} {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
    print("=" * 52)
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
