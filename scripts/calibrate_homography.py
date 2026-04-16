#!/usr/bin/env python3
"""
scripts/calibrate_homography.py — One-time homography calibration tool.

Modes
-----
  Manual (default):
    python scripts/calibrate_homography.py --video data/videos/game1.mp4
    Opens a window; click known rink points, type their rink coordinates.
    Saves calibration to data/homography_calibration.json.

  Auto-calibration test:
    python scripts/calibrate_homography.py --auto --frame 15000
    Runs automatic line detection on frame 15000, shows detected points
    and reprojection error, saves annotated image.

  Load & validate only:
    python scripts/calibrate_homography.py --validate --frame 15000
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.homography import (
    RinkHomography,
    KNOWN_RINK_POINTS,
    FAR_BLUE_Y, FAR_GOAL_Y, FAR_BOARDS_Y,
    NEAR_BLUE_Y, HALF_WIDTH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VIDEO_PATH = Path(__file__).parent.parent / "data" / "videos" / "game1.mp4"
CAL_PATH   = Path(__file__).parent.parent / "data" / "homography_calibration.json"
OUT_DIR    = Path(__file__).parent.parent / "data"


# ── Helpers ────────────────────────────────────────────────────────────────

def read_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")
    return frame


def print_known_points() -> None:
    print("\nKnown rink points (origin = centre ice, metres):")
    print(f"  {'Name':<25}  {'(rx, ry)'}")
    print(f"  {'-'*25}  {'-'*20}")
    for i, (name, (rx, ry)) in enumerate(KNOWN_RINK_POINTS.items(), 1):
        print(f"  {i:2d}. {name:<22}  ({rx:+.2f}, {ry:+.2f})")


# ── Manual calibration mode ────────────────────────────────────────────────

class ManualCalibrator:
    """Click-based calibration: user clicks pixel points on a displayed frame."""

    def __init__(self, frame: np.ndarray, hom: RinkHomography):
        self.orig_frame  = frame.copy()
        self.disp_frame  = frame.copy()
        self.hom         = hom
        self.image_pts   = []
        self.rink_pts    = []
        self.pending_px  = None   # pixel coord waiting for rink assignment
        self.win_name    = "Homography Calibration  [click rink point, then enter coords]"

    def run(self) -> bool:
        """Returns True if calibration was saved successfully."""
        print_known_points()
        print("\nInstructions:")
        print("  • Click a point on the ice that you know the rink coordinates of.")
        print("  • Then type the rink coordinate in the terminal (or pick by number).")
        print("  • Press 'c' to compute homography (needs ≥4 points).")
        print("  • Press 'r' to reset all points.")
        print("  • Press 'q' to quit without saving.\n")

        cv2.namedWindow(self.win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.win_name, self._on_click)

        known_list = list(KNOWN_RINK_POINTS.items())

        while True:
            cv2.imshow(self.win_name, self.disp_frame)
            key = cv2.waitKey(50) & 0xFF

            if key == ord('q'):
                print("Quit — no calibration saved.")
                cv2.destroyAllWindows()
                return False

            elif key == ord('r'):
                self.image_pts.clear()
                self.rink_pts.clear()
                self.pending_px = None
                self.disp_frame = self.orig_frame.copy()
                print("Reset — all points cleared.\n")

            elif key == ord('c'):
                if len(self.image_pts) < 4:
                    print(f"Need ≥4 points, have {len(self.image_pts)}. Keep clicking.")
                    continue
                try:
                    self.hom.calibrate_from_points(self.image_pts, self.rink_pts)
                    result = self.hom.validate(self.orig_frame, draw=True)
                    print(f"\nCalibration saved to {self.hom._cal_path}")
                    if result["mean_error_px"] is not None:
                        print(f"Reprojection error: {result['mean_error_px']:.2f} px  "
                              f"/  {result['mean_error_cm']:.1f} cm")
                    out_path = OUT_DIR / "homography_calibration_overlay.jpg"
                    cv2.imwrite(str(out_path), result["annotated_frame"])
                    print(f"Overlay saved → {out_path}")
                    cv2.destroyAllWindows()
                    return True
                except Exception as exc:
                    print(f"Calibration failed: {exc}")

            elif self.pending_px is not None:
                # Accept numeric key to pick known point
                if ord('1') <= key <= ord('9'):
                    idx = key - ord('1')
                    if idx < len(known_list):
                        name, rk = known_list[idx]
                        self._accept_rink_coord(self.pending_px, rk, label=name)
                elif key == ord('t'):
                    # Type custom coordinates in terminal
                    try:
                        raw = input("  Enter rink coords as 'rx,ry' (metres): ").strip()
                        rx, ry = map(float, raw.split(","))
                        self._accept_rink_coord(self.pending_px, (rx, ry))
                    except (ValueError, EOFError) as exc:
                        print(f"  Invalid input: {exc}")

        cv2.destroyAllWindows()
        return False

    def _on_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self.pending_px = (float(x), float(y))
        self.disp_frame = self.orig_frame.copy()

        # Redraw accepted points
        for px, py in self.image_pts:
            cv2.circle(self.disp_frame, (int(px), int(py)), 6, (0, 255, 0), -1)

        # Highlight new pending point
        cv2.circle(self.disp_frame, (x, y), 8, (0, 80, 255), -1)
        cv2.putText(self.disp_frame, f"({x}, {y})", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)

        known_list = list(KNOWN_RINK_POINTS.items())
        print(f"\nClicked pixel ({x}, {y}).  Assign rink coord:")
        for i, (name, (rx, ry)) in enumerate(known_list, 1):
            print(f"  Press {i}: {name:<25} ({rx:+.2f}, {ry:+.2f})")
        print("  Press t: type custom coordinates")

    def _accept_rink_coord(self, px_pt, rk_pt, label=""):
        self.image_pts.append(px_pt)
        self.rink_pts.append(rk_pt)
        self.pending_px = None
        px, py = px_pt
        rx, ry = rk_pt
        tag = f" [{label}]" if label else ""
        print(f"  Accepted: pixel ({px:.0f},{py:.0f}) → rink ({rx:+.2f},{ry:+.2f}){tag}")
        print(f"  Total pairs: {len(self.image_pts)}"
              f"{'  — press c to compute' if len(self.image_pts) >= 4 else ''}")
        cv2.circle(self.disp_frame, (int(px), int(py)), 6, (0, 255, 0), -1)
        cv2.putText(self.disp_frame, label or f"({rx:.1f},{ry:.1f})",
                    (int(px) + 8, int(py) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)


# ── Auto-calibration test ─────────────────────────────────────────────────

def run_auto_test(video_path: str, frame_idx: int, camera_side: str) -> None:
    print(f"\n{'='*60}")
    print(f" Auto-calibration test  —  frame {frame_idx}")
    print(f"{'='*60}\n")

    frame = read_frame(video_path, frame_idx)
    h_px, w_px = frame.shape[:2]
    print(f"Frame resolution: {w_px}×{h_px}")

    hom = RinkHomography(camera_side=camera_side)
    img_pts, rk_pts = hom.detect_calibration_points(frame)

    print(f"\nDetected {len(img_pts)} calibration pairs:")
    for i, (ip, rp) in enumerate(zip(img_pts, rk_pts)):
        print(f"  {i+1:2d}.  pixel ({ip[0]:6.1f}, {ip[1]:6.1f})  →  "
              f"rink ({rp[0]:+6.2f}, {rp[1]:+6.2f}) m")

    if len(img_pts) < 4:
        print(f"\n[WARN] Only {len(img_pts)} points — need ≥4 for homography.")
        print("       Run with --manual for interactive calibration.")
        _save_detection_vis(frame, img_pts, frame_idx)
        return

    try:
        hom.calibrate_from_points(img_pts, rk_pts, save=True)
    except Exception as exc:
        print(f"\n[ERROR] calibrate_from_points failed: {exc}")
        _save_detection_vis(frame, img_pts, frame_idx)
        return

    result = hom.validate(frame, draw=True)
    print(f"\nReprojection error:")
    if result["mean_error_px"] is not None:
        print(f"  Mean pixel error : {result['mean_error_px']:.2f} px")
        print(f"  Mean rink error  : {result['mean_error_cm']:.1f} cm")
        per = result["per_point_px"]
        if per:
            print(f"  Min / Max        : {min(per):.2f} / {max(per):.2f} px")
    else:
        print("  (no calibration points stored for reprojection)")

    # Sample transforms
    print("\nSample transforms:")
    test_rink = [(0.0, 0.0), (0.0, FAR_BLUE_Y), (0.0, FAR_GOAL_Y),
                 (-HALF_WIDTH, FAR_BOARDS_Y), (HALF_WIDTH, FAR_BOARDS_Y)]
    for rx, ry in test_rink:
        try:
            px, py = hom.rink_to_pixel(rx, ry)
            print(f"  rink ({rx:+.2f}, {ry:+.2f}) → pixel ({px:.0f}, {py:.0f})")
        except Exception as exc:
            print(f"  rink ({rx:+.2f}, {ry:+.2f}) → ERROR: {exc}")

    out_path = OUT_DIR / f"homography_test_frame{frame_idx}.jpg"
    cv2.imwrite(str(out_path), result["annotated_frame"])
    print(f"\nAnnotated frame saved → {out_path}")


def _save_detection_vis(frame, img_pts, frame_idx):
    vis = frame.copy()
    for px, py in img_pts:
        cv2.circle(vis, (int(px), int(py)), 6, (0, 255, 0), -1)
    out = OUT_DIR / f"homography_detection_frame{frame_idx}.jpg"
    cv2.imwrite(str(out), vis)
    print(f"Detection frame saved → {out}")


# ── Validate-only mode ────────────────────────────────────────────────────

def run_validate(video_path: str, frame_idx: int) -> None:
    frame = read_frame(video_path, frame_idx)
    hom = RinkHomography()
    if hom.H is None:
        print("[ERROR] No saved calibration found.  Run calibration first.")
        sys.exit(1)
    result = hom.validate(frame, draw=True)
    print(f"Mean pixel error : {result['mean_error_px']}")
    print(f"Mean rink error  : {result['mean_error_cm']} cm")
    out_path = OUT_DIR / f"homography_validate_frame{frame_idx}.jpg"
    cv2.imwrite(str(out_path), result["annotated_frame"])
    print(f"Overlay saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IceIQ Homography Calibration")
    parser.add_argument("--video",  default=str(VIDEO_PATH), help="Path to game video")
    parser.add_argument("--frame",  type=int, default=15000,  help="Frame index for auto/validate modes")
    parser.add_argument("--side",   default="left",           help="Camera corner: left or right")
    parser.add_argument("--auto",   action="store_true",      help="Run auto-calibration test")
    parser.add_argument("--validate", action="store_true",    help="Validate existing calibration")
    args = parser.parse_args()

    if args.validate:
        run_validate(args.video, args.frame)
    elif args.auto:
        run_auto_test(args.video, args.frame, args.side)
    else:
        # Manual calibration
        frame = read_frame(args.video, args.frame)
        hom   = RinkHomography(camera_side=args.side)
        cal   = ManualCalibrator(frame, hom)
        cal.run()


if __name__ == "__main__":
    main()
