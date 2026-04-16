"""
Debug script: green jersey (Zenith Phoenix) detection at frame 50000.

Steps:
  1. Extract frame 50000 from data/videos/game1.mp4
  2. Run HockeyPlayerDetector – print all bounding boxes
  3. Save each crop to /tmp/debug_player_N.jpg
  4. Compute HSV green ratio per crop (torso strip y:20-65%)
  5. Check ROI: show polygon + whether each detection foot passes is_inside()
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import json
from pathlib import Path

# ── project imports ────────────────────────────────────────────────────────────
from pipeline.detection import HockeyPlayerDetector
from pipeline.roi import RinkROI

# ── HSV green range (from team_classification.py) ─────────────────────────────
GREEN_LOWER = np.array([35,  50,  50], dtype=np.uint8)
GREEN_UPPER = np.array([85, 255, 255], dtype=np.uint8)
GREEN_THRESHOLD = 0.08   # same as _GREEN_THRESHOLD in team_classification.py

TARGET_FRAME = 50_000
VIDEO_PATH   = "data/videos/game1.mp4"
ROI_PATH     = "data/roi_polygon.json"


def green_ratio_torso(crop: np.ndarray) -> float:
    """Green pixel fraction in torso strip (y 20-65%, x 15-85%) of a bbox crop."""
    if crop is None or crop.size == 0:
        return 0.0
    h, w = crop.shape[:2]
    ty1 = int(h * 0.20);  ty2 = int(h * 0.65)
    tx1 = int(w * 0.15);  tx2 = w - int(w * 0.15)
    torso = crop[ty1:ty2, tx1:tx2]
    if torso.size < 300:
        return 0.0
    hsv  = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    return float(np.count_nonzero(mask)) / (mask.shape[0] * mask.shape[1])


def green_ratio_full(crop: np.ndarray) -> float:
    """Green pixel fraction over the full crop."""
    if crop is None or crop.size == 0:
        return 0.0
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    return float(np.count_nonzero(mask)) / (mask.shape[0] * mask.shape[1])


def hsv_stats(crop: np.ndarray) -> dict:
    """Mean H, S, V values for the torso strip of a crop."""
    if crop is None or crop.size == 0:
        return {}
    h, w = crop.shape[:2]
    ty1 = int(h * 0.20);  ty2 = int(h * 0.65)
    tx1 = int(w * 0.15);  tx2 = w - int(w * 0.15)
    torso = crop[ty1:ty2, tx1:tx2]
    if torso.size < 300:
        return {}
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
    return {
        "H_mean": round(hsv[:, 0].mean(), 1),
        "S_mean": round(hsv[:, 1].mean(), 1),
        "V_mean": round(hsv[:, 2].mean(), 1),
        "H_std":  round(hsv[:, 0].std(),  1),
        "S_std":  round(hsv[:, 1].std(),  1),
    }


def main():
    print("=" * 70)
    print(f"IceIQ Green-Detection Debug  |  frame {TARGET_FRAME}")
    print("=" * 70)

    # ── 1. Open video & seek to frame 50000 ───────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {VIDEO_PATH}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nVideo: {width}x{height}  fps={fps:.2f}  total_frames={total_frames}")
    print(f"Target frame {TARGET_FRAME} ≈ {TARGET_FRAME/fps/60:.1f} min into video")

    cap.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"[ERROR] Could not read frame {TARGET_FRAME}")
        sys.exit(1)

    # Save raw frame for reference
    cv2.imwrite("/tmp/debug_frame_50000.jpg", frame)
    print(f"\nFrame saved → /tmp/debug_frame_50000.jpg")

    # ── 2. Load ROI ────────────────────────────────────────────────────────────
    roi = RinkROI(roi_path=ROI_PATH)
    print(f"\n── ROI Polygon ({len(roi.polygon)} vertices) ──────────────────────")
    for i, pt in enumerate(roi.polygon):
        print(f"  vertex {i}: ({pt[0]}, {pt[1]})")

    # Compute polygon bounding box
    pts = np.array(roi.polygon)
    print(f"  bbox: x=[{pts[:,0].min()}, {pts[:,0].max()}]  "
          f"y=[{pts[:,1].min()}, {pts[:,1].max()}]")
    print(f"  Frame size: {width}x{height}")

    # Save frame with ROI drawn
    frame_with_roi = roi.draw(frame)
    cv2.imwrite("/tmp/debug_frame_with_roi.jpg", frame_with_roi)
    print(f"  Frame+ROI saved → /tmp/debug_frame_with_roi.jpg")

    # ── 3. Run detector ────────────────────────────────────────────────────────
    print(f"\n── Running HockeyPlayerDetector (yolov8m conf=0.20) ────────────────")
    detector = HockeyPlayerDetector(model_path="yolov8m.pt", conf=0.20, imgsz=1280)
    result   = detector.detect_frame(frame)

    n_raw = result["count"]
    print(f"  Raw YOLO detections (before ROI): {n_raw}")

    if n_raw == 0:
        print("\n[CRITICAL] YOLO found ZERO detections at frame 50000!")
        print("  → The problem is in detection, not classification.")
        sys.exit(0)

    # ── 4. Per-detection analysis ──────────────────────────────────────────────
    print(f"\n── Per-Detection Analysis ──────────────────────────────────────────")
    header = (f"{'N':>3}  {'bbox':^40}  {'conf':>5}  "
              f"{'foot_cx':>7} {'foot_y2':>7}  {'in_roi':>6}  "
              f"{'gr_torso':>8}  {'gr_full':>7}  {'label':>10}")
    print(header)
    print("-" * len(header))

    fh, fw = frame.shape[:2]
    n_in_roi = 0
    n_green_torso = 0
    crops_with_info = []

    for idx, (bbox, score) in enumerate(zip(result["boxes"], result["scores"])):
        x1, y1, x2, y2 = bbox
        foot_cx = (x1 + x2) / 2.0
        foot_y2 = float(y2)
        in_roi  = roi.is_inside(foot_cx, foot_y2)

        # Crop for color analysis
        ix1 = max(0, int(x1));  iy1 = max(0, int(y1))
        ix2 = min(fw, int(x2)); iy2 = min(fh, int(y2))
        crop = frame[iy1:iy2, ix1:ix2].copy()

        gr_torso = green_ratio_torso(crop)
        gr_full  = green_ratio_full(crop)
        label    = "GREEN" if gr_torso >= GREEN_THRESHOLD else "white"

        if in_roi:
            n_in_roi += 1
        if gr_torso >= GREEN_THRESHOLD:
            n_green_torso += 1

        crops_with_info.append({
            "idx": idx, "bbox": bbox, "score": score,
            "foot_cx": foot_cx, "foot_y2": foot_y2,
            "in_roi": in_roi, "gr_torso": gr_torso, "gr_full": gr_full,
            "label": label, "crop": crop,
        })

        row = (f"{idx:>3}  "
               f"[{x1:>5.0f},{y1:>5.0f},{x2:>5.0f},{y2:>5.0f}]  "
               f"{score:>5.2f}  "
               f"{foot_cx:>7.1f} {foot_y2:>7.1f}  "
               f"{'YES' if in_roi else 'NO':>6}  "
               f"{gr_torso:>8.4f}  {gr_full:>7.4f}  "
               f"{label:>10}")
        print(row)

        # Save crop
        crop_path = f"/tmp/debug_player_{idx}.jpg"
        cv2.imwrite(crop_path, crop)

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print(f"\n── Summary ─────────────────────────────────────────────────────────")
    print(f"  Total YOLO detections:        {n_raw}")
    print(f"  Passed ROI filter:            {n_in_roi}")
    print(f"  Rejected by ROI:              {n_raw - n_in_roi}")
    print(f"  Green (torso ratio ≥ 0.08):   {n_green_torso}")
    print(f"  White (torso ratio < 0.08):   {n_raw - n_green_torso}")

    # ── 6. HSV stats for each crop ────────────────────────────────────────────
    print(f"\n── Torso HSV Stats (mean H/S/V in torso strip) ─────────────────────")
    print(f"{'N':>3}  {'H_mean':>7} {'S_mean':>7} {'V_mean':>7}  "
          f"{'H_std':>6}  {'in_roi':>6}  {'gr_torso':>8}  {'label':>10}")
    print("-" * 70)
    for info in crops_with_info:
        stats = hsv_stats(info["crop"])
        if not stats:
            continue
        print(f"{info['idx']:>3}  "
              f"{stats['H_mean']:>7.1f} {stats['S_mean']:>7.1f} {stats['V_mean']:>7.1f}  "
              f"{stats['H_std']:>6.1f}  "
              f"{'YES' if info['in_roi'] else 'NO':>6}  "
              f"{info['gr_torso']:>8.4f}  "
              f"{info['label']:>10}")

    # ── 7. ROI deep-dive: detections OUTSIDE roi ──────────────────────────────
    outside = [c for c in crops_with_info if not c["in_roi"]]
    if outside:
        print(f"\n── Detections OUTSIDE ROI ({len(outside)}) ──────────────────────────")
        for c in outside:
            print(f"  N={c['idx']}  foot=({c['foot_cx']:.0f},{c['foot_y2']:.0f})  "
                  f"bbox=[{c['bbox'][0]:.0f},{c['bbox'][1]:.0f},"
                  f"{c['bbox'][2]:.0f},{c['bbox'][3]:.0f}]  "
                  f"gr_torso={c['gr_torso']:.4f}  {c['label']}")
    else:
        print(f"\n  All {n_raw} detections are INSIDE the ROI.")

    # ── 8. Annotated frame ────────────────────────────────────────────────────
    vis = frame.copy()
    if roi.polygon is not None:
        pts_draw = roi.polygon.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(vis, [pts_draw], True, (0, 255, 255), 2)  # cyan ROI

    for c in crops_with_info:
        x1, y1, x2, y2 = [int(v) for v in c["bbox"]]
        color = (0, 255, 0) if c["in_roi"] else (0, 0, 255)   # green=in, red=out
        label_color = (0, 200, 0) if c["label"] == "GREEN" else (200, 200, 200)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        # foot dot
        fx, fy = int(c["foot_cx"]), int(c["foot_y2"])
        cv2.circle(vis, (fx, fy), 5, color, -1)
        cv2.putText(vis, f"#{c['idx']} {c['gr_torso']:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)

    cv2.imwrite("/tmp/debug_annotated.jpg", vis)
    print(f"\n  Annotated frame saved → /tmp/debug_annotated.jpg")
    print(f"  (green box = inside ROI, red box = outside ROI)")
    print(f"  Crops saved → /tmp/debug_player_0.jpg … /tmp/debug_player_{n_raw-1}.jpg")
    print("=" * 70)


if __name__ == "__main__":
    main()
