#!/usr/bin/env python3
"""
test_identification.py
Run 7-signal PlayerIdentifier on game1.mp4 and print results for Aigis players.

Usage:
    python test_identification.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(name)s — %(message)s",
)
logger = logging.getLogger("test_identification")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO = Path(__file__).parent
VIDEO_PATH   = REPO / "data/videos/game1.mp4"
ROSTER_PATH  = REPO / "data/rosters/aigis.yaml"
ROI_PATH     = REPO / "data/roi_polygon.json"
CONFIG_PATH  = REPO / "configs/pipeline_config.yaml"
POSE_MODEL   = "yolov8m-pose.pt"   # downloaded automatically if absent

# ── Load helpers ──────────────────────────────────────────────────────────────

def load_roi_polygon():
    if ROI_PATH.exists():
        with open(ROI_PATH) as f:
            return json.load(f)["polygon"]
    logger.warning("ROI polygon not found — homography disabled")
    return None


def load_rink_config():
    try:
        import yaml
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("video", {}).get("rink", {})
    except Exception as e:
        logger.warning("Could not load pipeline_config.yaml: %s", e)
        return {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Imports ───────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    try:
        from pipeline.identification import PlayerIdentifier, HomographyMapper
    except ImportError as e:
        print(f"ERROR: Cannot import identification module: {e}")
        sys.exit(1)

    # ── Video ─────────────────────────────────────────────────────────────────
    if not VIDEO_PATH.exists():
        print(f"ERROR: Video not found at {VIDEO_PATH}")
        sys.exit(1)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info("Video: %d frames @ %.1f fps", total_frames, fps)

    # ── YOLO pose model ───────────────────────────────────────────────────────
    logger.info("Loading YOLO pose model: %s", POSE_MODEL)
    model = YOLO(POSE_MODEL)

    # ── Setup identifier ──────────────────────────────────────────────────────
    roi_polygon = load_roi_polygon()
    rink_config = load_rink_config()

    ident = PlayerIdentifier(
        roster_path=str(ROSTER_PATH),
        rink_config=rink_config,
        roi_polygon=roi_polygon,
    )

    # Optional homography mapper for pixel→rink conversion
    mapper = None
    if roi_polygon:
        try:
            mapper = HomographyMapper(
                roi_polygon,
                rink_config.get("length", 60.96),
                rink_config.get("width", 25.9),
            )
        except Exception as e:
            logger.warning("HomographyMapper unavailable: %s", e)

    # ── Simple IoU tracker (avoids lap/ByteTrack dependency) ─────────────────
    # Assign stable track IDs using greedy IoU matching across frames.

    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        """IoU between two bboxes [x1,y1,x2,y2]."""
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        return inter / (area_a + area_b - inter)

    next_id: int = 1
    # {track_id: last_bbox}
    live_tracks: Dict[int, np.ndarray] = {}
    IOU_THRESH = 0.3
    MAX_MISSING = 10  # frames before a track is dropped
    track_last_seen: Dict[int, int] = {}

    MAX_FRAMES = min(total_frames, 720)
    frame_idx = 0
    detection_count = 0

    logger.info("Processing %d frames (simple IoU tracker)...", MAX_FRAMES)

    while frame_idx < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        # Pose inference (no tracker, just predict)
        try:
            results = model.predict(
                frame,
                conf=0.25,
                iou=0.45,
                classes=[0],
                verbose=False,
            )
        except Exception as e:
            logger.debug("Frame %d inference error: %s", frame_idx, e)
            frame_idx += 1
            continue

        if not results or results[0] is None:
            frame_idx += 1
            continue

        r = results[0]
        if r.boxes is None or r.keypoints is None:
            frame_idx += 1
            continue

        bboxes = r.boxes.xyxy.cpu().numpy()        # (N, 4)
        kps    = r.keypoints.data.cpu().numpy()    # (N, 17, 3)
        N      = len(bboxes)

        # Greedy IoU matching: new detections → existing tracks
        assigned = {}   # det_idx → track_id
        used_tracks = set()

        for di in range(N):
            best_tid, best_iou = -1, IOU_THRESH
            for tid, prev_bbox in live_tracks.items():
                if tid in used_tracks:
                    continue
                iou_val = _iou(bboxes[di], prev_bbox)
                if iou_val > best_iou:
                    best_iou, best_tid = iou_val, tid
            if best_tid >= 0:
                assigned[di] = best_tid
                used_tracks.add(best_tid)
            else:
                assigned[di] = next_id
                next_id += 1

        current_frame_tracks = set()
        new_live: Dict[int, np.ndarray] = {}

        for di in range(N):
            tid  = assigned[di]
            kp   = kps[di]
            bbox = bboxes[di]

            cx  = (bbox[0] + bbox[2]) / 2.0
            cy2 = bbox[3]

            if mapper is not None:
                rink_pos = mapper.pixel_to_rink(cx, cy2)
            else:
                rink_pos = (cx, cy2)

            ident.update(tid, frame_idx, kp, rink_pos)
            current_frame_tracks.add(tid)
            new_live[tid] = bbox
            track_last_seen[tid] = frame_idx
            detection_count += 1

        # Keep recently-seen tracks alive (handle occlusion gaps)
        for tid, last_f in list(track_last_seen.items()):
            if frame_idx - last_f > MAX_MISSING:
                ident.expire_track(tid, frame_idx)
                track_last_seen.pop(tid, None)
            elif tid not in current_frame_tracks and tid in live_tracks:
                new_live[tid] = live_tracks[tid]  # carry forward bbox

        live_tracks = new_live
        frame_idx += 1

        if frame_idx % 60 == 0:
            logger.info("  frame %d / %d — %d live tracks", frame_idx, MAX_FRAMES, len(live_tracks))

    cap.release()
    logger.info("Done. %d total detections across %d frames.", detection_count, frame_idx)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  AIGIS PLAYER IDENTIFICATION RESULTS")
    print("  Bayesian fusion of Signals 1-4 (5 external via OCR)")
    print("=" * 60)

    summary = ident.summary()
    if not summary:
        print("  No tracks detected. Check video path and model availability.")
        return

    # Sort by confidence descending
    sorted_tracks = sorted(summary.items(), key=lambda kv: kv[1]["confidence"], reverse=True)

    print(f"\n{'Track':>6}  {'#':>4}  {'Hand':>5}  {'Pos':>4}  {'Conf':>6}  {'Frames':>7}")
    print("-" * 50)

    for tid, info in sorted_tracks:
        state = ident._tracks[tid]
        frames = state.frame_count
        number   = info["number"]   or "?"
        hand     = info["shot_hand"] or "?"
        position = info["position"]  or "?"
        conf     = info["confidence"]

        print(
            f"{tid:>6}  {str(number):>4}  {hand:>5}  {position:>4}  {conf:>6.3f}  {frames:>7}"
        )

    # Per-track detail for high-confidence identifications
    print("\n── High-confidence identifications (conf > 0.6) ────────────────")
    identified_count = 0
    for tid, info in sorted_tracks:
        if info["number"] is None or info["confidence"] <= 0.6:
            continue
        identified_count += 1
        state = ident._tracks[tid]
        total_wrist = sum(state.wrist_votes.values())
        sh_detail = (
            f"  (R:{state.wrist_votes['RIGHT']} L:{state.wrist_votes['LEFT']}"
            f" from {total_wrist} frames)"
        )
        print(f"\n  Track {tid}:")
        print(f"    Jersey #:      {info['number']}")
        print(f"    Shot hand:     {info['shot_hand']}{sh_detail}")
        print(f"    Position:      {info['position']}")
        print(f"    Confidence:    {info['confidence']:.3f}")
        if state.mean_height_m is not None:
            print(f"    Height:        {state.mean_height_m:.2f} m (Signal 4)")
        if state.mean_shoulder_width_m is not None:
            print(f"    Shoulder W:    {state.mean_shoulder_width_m:.2f} m (Signal 4)")
        n_positions = len(state.rink_positions)
        print(f"    Rink samples:  {n_positions} (Signal 2)")
        print(f"    OCR votes:     {dict(state.jersey_log_likelihoods)} (Signal 5)")

    if identified_count == 0:
        print("  None yet — more frames needed, or feed jersey OCR via feed_jersey().")

    # Shot hand summary
    print("\n── Shot hand distribution (Signal 1) ──────────────────────────")
    for tid, info in sorted_tracks:
        state = ident._tracks[tid]
        total = sum(state.wrist_votes.values())
        if total >= 5:
            print(
                f"  Track {tid}: {info['shot_hand']:>5} "
                f"(R={state.wrist_votes['RIGHT']} L={state.wrist_votes['LEFT']})"
            )

    # Position summary
    print("\n── Position classification (Signal 2) ─────────────────────────")
    for tid, info in sorted_tracks:
        state = ident._tracks[tid]
        if len(state.rink_positions) >= 30:
            print(f"  Track {tid}: {info['position'] or '?':>4}  ({len(state.rink_positions)} rink samples)")

    # Body shape summary
    print("\n── Body shape profiles (Signal 4) ──────────────────────────────")
    for tid, info in sorted_tracks:
        state = ident._tracks[tid]
        h  = state.mean_height_m
        sw = state.mean_shoulder_width_m
        if h is not None or sw is not None:
            print(
                f"  Track {tid}:  height={h:.2f}m" if h else f"  Track {tid}:  height=?",
                f" shoulder={sw:.2f}m" if sw else " shoulder=?",
            )

    print("\n" + "=" * 60)
    print(f"  Total tracks observed: {len(summary)}")
    print(f"  Roster: {ident.roster_numbers}")
    print("=" * 60)


if __name__ == "__main__":
    main()
