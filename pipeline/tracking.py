"""
pipeline/tracking.py
Multi-object tracker for hockey players.

Primary:  BoT-SORT with ReID via boxmot (boxmot >= 10.x)
Fallback: Kalman filter (6-state: cx,cy,w,h,vx,vy) + Hungarian / IoU matching

Input:   detections from HockeyPlayerDetector.detect_frame() or .detect()
Output:  List[TrackResult] with consistent track_id across frames
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── boxmot availability ─────────────────────────────────────────────────────
try:
    import boxmot as _boxmot_mod  # noqa: F401
    _BOXMOT_AVAILABLE = True
except ImportError:
    _BOXMOT_AVAILABLE = False
    logger.info("boxmot not found — will use Kalman+Hungarian fallback")


# ── output type ─────────────────────────────────────────────────────────────
class TrackResult(dict):
    """
    Typed dict for a single tracked player.
    Keys: track_id (int), bbox ([x1,y1,x2,y2]), confidence (float),
          class_id (int), velocity ([vx,vy] px/frame),
          keypoints (optional): [[x,y,conf]×17] COCO pose landmarks
    """


def _iou_1d(a: List[float], b: List[float]) -> float:
    """두 bbox [x1,y1,x2,y2] 간 IoU"""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _match_keypoints(track_bbox: List[float], detections: List[Dict]) -> List | None:
    """트래커 출력 bbox → IoU 최대 detection의 keypoints 반환"""
    best_iou, best_kps = 0.0, None
    for d in detections:
        det_bbox = d.get("bbox", [])
        if len(det_bbox) < 4:
            continue
        iou = _iou_1d(track_bbox, det_bbox)
        if iou > best_iou:
            best_iou = iou
            best_kps = d.get("keypoints")
    return best_kps if best_iou > 0.3 else None


def _make_result(
    track_id: int,
    bbox: List[float],
    confidence: float,
    class_id: int,
    velocity: List[float],
    keypoints: List | None = None,
) -> TrackResult:
    r = TrackResult(
        track_id=track_id,
        bbox=bbox,
        confidence=confidence,
        class_id=class_id,
        velocity=velocity,
    )
    if keypoints is not None:
        r["keypoints"] = keypoints
    return r


# ── device helper ───────────────────────────────────────────────────────────
def _auto_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ═══════════════════════════════════════════════════════════════════════════ #
#  boxmot back-end                                                            #
# ═══════════════════════════════════════════════════════════════════════════ #

class _BoxmotBackend:
    """Thin wrapper around boxmot ByteTrack / BotSort."""

    def __init__(
        self,
        method: str,
        device: str,
        reid_weights: Optional[Path],
        track_high_thresh: float,
        track_low_thresh: float,
        new_track_thresh: float,
        match_thresh: float,
        track_buffer: int,
        frame_rate: int,
        proximity_thresh: float = 0.25,
        appearance_thresh: float = 0.40,
    ) -> None:
        self.device = device
        self._prev_pos: Dict[int, np.ndarray] = {}
        self._tracker = self._build(
            method, reid_weights,
            track_high_thresh, track_low_thresh, new_track_thresh,
            match_thresh, track_buffer, frame_rate,
            proximity_thresh, appearance_thresh,
        )
        logger.info("_BoxmotBackend: %s on device=%s", method, device)

    # ------------------------------------------------------------------ #
    def _build(self, method, reid_weights,
               tht, tlt, ntt, mt, tb, fr,
               proximity_thresh=0.25, appearance_thresh=0.40):
        method = method.lower().replace("-", "").replace("_", "")

        if method == "botsort":
            return self._build_botsort(
                reid_weights, tht, tlt, ntt, mt, tb, fr,
                proximity_thresh, appearance_thresh,
            )

        # Default: ByteTrack
        return self._build_bytetrack(tht, tlt, ntt, mt, tb, fr)

    def _build_bytetrack(self, tht, tlt, ntt, mt, tb, fr):
        from boxmot import ByteTrack  # type: ignore
        # Try newer API first, fall back to older keyword set
        try:
            return ByteTrack(
                track_high_thresh=tht,
                track_low_thresh=tlt,
                new_track_thresh=ntt,
                match_thresh=mt,
                track_buffer=tb,
                frame_rate=fr,
            )
        except TypeError:
            # Some releases use positional / different keyword names
            try:
                return ByteTrack(
                    track_thresh=tht,
                    match_thresh=mt,
                    track_buffer=tb,
                    frame_rate=fr,
                )
            except TypeError:
                return ByteTrack()

    def _build_botsort(self, reid_weights, tht, tlt, ntt, mt, tb, fr,
                       proximity_thresh, appearance_thresh):
        # boxmot >= 10.x exports BotSort (some releases also have BOTSORT alias)
        try:
            from boxmot import BOTSORT as _BotSortCls  # type: ignore
        except ImportError:
            from boxmot import BotSort as _BotSortCls  # type: ignore
        weights = reid_weights or Path("osnet_x0_25_msmt17.pt")
        try:
            return _BotSortCls(
                reid_weights=weights,
                device=self.device,
                half=False,           # MPS does not support fp16
                track_high_thresh=tht,
                track_low_thresh=tlt,
                new_track_thresh=ntt,
                match_thresh=mt,
                track_buffer=tb,
                frame_rate=fr,
                proximity_thresh=proximity_thresh,
                appearance_thresh=appearance_thresh,
            )
        except TypeError as e:
            # Older boxmot releases may not have proximity_thresh / appearance_thresh
            logger.warning("BotSort full init failed (%s) — retrying without appearance params", e)
            try:
                return _BotSortCls(
                    reid_weights=weights,
                    device=self.device,
                    half=False,
                    track_high_thresh=tht,
                    track_low_thresh=tlt,
                    new_track_thresh=ntt,
                    match_thresh=mt,
                    track_buffer=tb,
                    frame_rate=fr,
                )
            except Exception as e2:
                logger.warning("BotSort init failed (%s) — falling back to ByteTrack", e2)
                return self._build_bytetrack(tht, tlt, ntt, mt, tb, fr)

    # ------------------------------------------------------------------ #
    def update(
        self, detections: List[Dict], frame: np.ndarray
    ) -> List[TrackResult]:
        if not detections:
            return []

        # [N, 6]: x1 y1 x2 y2 conf cls
        dets = np.array(
            [[*d["bbox"], d.get("confidence", 1.0), d.get("class_id", 0)]
             for d in detections],
            dtype=np.float32,
        )

        try:
            raw = self._tracker.update(dets, frame)
        except Exception as e:
            logger.warning("boxmot tracker.update error: %s", e)
            return []

        if raw is None or len(raw) == 0:
            return []

        raw = np.atleast_2d(raw)
        results: List[TrackResult] = []

        for row in raw:
            if len(row) < 5:
                continue
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            tid   = int(row[4])
            conf  = float(row[5]) if len(row) > 5 else 1.0
            cls   = int(row[6])   if len(row) > 6 else 0

            cur = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            vel = (cur - self._prev_pos[tid]).tolist() if tid in self._prev_pos else [0.0, 0.0]
            self._prev_pos[tid] = cur

            # keypoints: IoU 매칭으로 원본 detection에서 복원
            kps = _match_keypoints([x1, y1, x2, y2], detections)
            results.append(_make_result(tid, [x1, y1, x2, y2], conf, cls, vel, kps))

        return results

    def reset(self) -> None:
        self._prev_pos.clear()
        if hasattr(self._tracker, "reset"):
            self._tracker.reset()


# ═══════════════════════════════════════════════════════════════════════════ #
#  Kalman + Hungarian fallback                                                #
# ═══════════════════════════════════════════════════════════════════════════ #

class _KTrack:
    """6-state constant-velocity Kalman filter: [cx, cy, w, h, vx, vy]."""

    def __init__(self, det: Dict, tid: int) -> None:
        from filterpy.kalman import KalmanFilter  # type: ignore
        self.id = tid
        self.disappeared = 0
        self.hits = 1
        self.confidence = float(det.get("confidence", 1.0))
        self.class_id   = int(det.get("class_id", 0))

        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w,  h  = float(x2 - x1), float(y2 - y1)

        kf = KalmanFilter(dim_x=6, dim_z=4)
        kf.x = np.array([cx, cy, w, h, 0.0, 0.0])

        dt = 1.0
        kf.F = np.eye(6)
        kf.F[0, 4] = dt
        kf.F[1, 5] = dt

        kf.H = np.eye(4, 6)

        # Tuned for hockey: fast skaters → higher process noise on velocity
        kf.R = np.diag([4.0, 4.0, 16.0, 16.0])
        kf.Q = np.diag([1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
        kf.P = np.diag([50.0, 50.0, 200.0, 200.0, 500.0, 500.0])

        self.kf = kf

    def predict(self) -> None:
        self.kf.predict()

    def update(self, det: Dict) -> None:
        x1, y1, x2, y2 = det["bbox"]
        meas = np.array([
            (x1 + x2) / 2.0, (y1 + y2) / 2.0,
            float(x2 - x1), float(y2 - y1),
        ])
        self.kf.update(meas)
        self.disappeared = 0
        self.hits += 1
        self.confidence = float(det.get("confidence", self.confidence))

    @property
    def state(self) -> np.ndarray:
        return self.kf.x.copy()

    @property
    def bbox(self) -> List[float]:
        cx, cy, w, h = self.state[:4]
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    @property
    def velocity(self) -> List[float]:
        return [float(self.state[4]), float(self.state[5])]


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised pairwise IoU. a, b: [N,4] / [M,4] in x1y1x2y2 format."""
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


class _KalmanHungarianBackend:
    """IoU-based multi-stage Kalman+Hungarian tracker (ByteTrack-style cascade)."""

    # Stage 1: high-confidence dets ↔ all tracks  (IoU ≥ iou_high)
    # Stage 2: unmatched low-conf dets ↔ lost tracks (IoU ≥ iou_low)

    def __init__(
        self,
        max_disappeared: int = 45,
        iou_high: float = 0.20,
        iou_low: float = 0.10,
        min_hits: int = 1,
        high_conf_thresh: float = 0.40,
    ) -> None:
        self.max_disappeared  = max_disappeared
        self.iou_high         = iou_high
        self.iou_low          = iou_low
        self.min_hits         = min_hits
        self.high_conf_thresh = high_conf_thresh
        self._tracks: Dict[int, _KTrack] = {}
        self._next_id = 1

    # ------------------------------------------------------------------ #
    def update(
        self, detections: List[Dict], frame: Optional[np.ndarray] = None
    ) -> List[TrackResult]:
        from scipy.optimize import linear_sum_assignment  # type: ignore

        # Predict all tracks
        for t in self._tracks.values():
            t.predict()

        if not detections:
            for t in list(self._tracks.values()):
                t.disappeared += 1
                if t.disappeared > self.max_disappeared:
                    del self._tracks[t.id]
            return self._emit()

        # Split detections by confidence
        hi_dets = [d for d in detections if d.get("confidence", 1.0) >= self.high_conf_thresh]
        lo_dets = [d for d in detections if d.get("confidence", 1.0) <  self.high_conf_thresh]

        active_tracks = [t for t in self._tracks.values() if t.disappeared == 0]
        lost_tracks   = [t for t in self._tracks.values() if t.disappeared > 0]

        matched_det: set = set()
        matched_trk: set = set()

        # ── stage 1: high-conf dets ↔ active tracks ── #
        if hi_dets and active_tracks:
            d_boxes = np.array([d["bbox"] for d in hi_dets], dtype=np.float32)
            t_boxes = np.array([t.bbox    for t in active_tracks], dtype=np.float32)
            iou = _iou_matrix(d_boxes, t_boxes)
            di, ti = linear_sum_assignment(1.0 - iou)
            for d_i, t_i in zip(di, ti):
                if iou[d_i, t_i] >= self.iou_high:
                    active_tracks[t_i].update(hi_dets[d_i])
                    matched_det.add(("hi", d_i))
                    matched_trk.add(active_tracks[t_i].id)

        # ── stage 2: unmatched lo-conf dets ↔ lost tracks ── #
        unmatched_lo = [d for d in lo_dets]
        if unmatched_lo and lost_tracks:
            d_boxes = np.array([d["bbox"] for d in unmatched_lo], dtype=np.float32)
            t_boxes = np.array([t.bbox    for t in lost_tracks],  dtype=np.float32)
            iou = _iou_matrix(d_boxes, t_boxes)
            di, ti = linear_sum_assignment(1.0 - iou)
            for d_i, t_i in zip(di, ti):
                if iou[d_i, t_i] >= self.iou_low:
                    lost_tracks[t_i].update(unmatched_lo[d_i])
                    matched_trk.add(lost_tracks[t_i].id)

        # Also try matching remaining hi-conf dets against lost tracks
        unmatched_hi = [
            d for i, d in enumerate(hi_dets) if ("hi", i) not in matched_det
        ]
        if unmatched_hi and lost_tracks:
            remaining_lost = [t for t in lost_tracks if t.id not in matched_trk]
            if remaining_lost:
                d_boxes = np.array([d["bbox"] for d in unmatched_hi], dtype=np.float32)
                t_boxes = np.array([t.bbox    for t in remaining_lost], dtype=np.float32)
                iou = _iou_matrix(d_boxes, t_boxes)
                di, ti = linear_sum_assignment(1.0 - iou)
                for d_i, t_i in zip(di, ti):
                    if iou[d_i, t_i] >= self.iou_low:
                        remaining_lost[t_i].update(unmatched_hi[d_i])
                        matched_det.add(("hi2", d_i))
                        matched_trk.add(remaining_lost[t_i].id)
                unmatched_hi = [
                    d for i, d in enumerate(unmatched_hi)
                    if ("hi2", i) not in matched_det
                ]

        # New tracks for unmatched high-confidence detections
        for det in unmatched_hi:
            t = _KTrack(det, self._next_id)
            self._tracks[self._next_id] = t
            self._next_id += 1

        # Increment disappeared for unmatched active tracks
        for t in active_tracks:
            if t.id not in matched_trk:
                t.disappeared += 1
                if t.disappeared > self.max_disappeared:
                    del self._tracks[t.id]

        return self._emit()

    def _emit(self) -> List[TrackResult]:
        out = []
        for t in self._tracks.values():
            if t.hits >= self.min_hits and t.disappeared == 0:
                out.append(_make_result(t.id, t.bbox, t.confidence, t.class_id, t.velocity))
        return out

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1


# ═══════════════════════════════════════════════════════════════════════════ #
#  Public API                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

class HockeyTracker:
    """
    Multi-object tracker for hockey players.

    Preference chain:
      BoT-SORT + ReID (boxmot/BOTSORT) → ByteTrack (boxmot) → Kalman+Hungarian (built-in)

    Usage
    -----
    tracker = HockeyTracker()
    for frame_bgr in video:
        det_result = detector.detect_frame(frame_bgr)   # DetectionResult dict
        tracks = tracker.update(det_result, frame_bgr)  # List[TrackResult]
    """

    def __init__(
        self,
        method: str = "botsort",
        device: Optional[str] = None,
        reid_weights: Optional[Path] = None,
        track_high_thresh: float = 0.35,
        track_low_thresh: float = 0.10,
        new_track_thresh: float = 0.45,
        match_thresh: float = 0.75,
        track_buffer: int = 90,        # 3 s @ 30 fps — survives occlusion
        frame_rate: int = 30,
        proximity_thresh: float = 0.25,
        appearance_thresh: float = 0.40,
    ) -> None:
        self._method = method
        dev = device or _auto_device()
        self._backend: Any

        if _BOXMOT_AVAILABLE:
            try:
                self._backend = _BoxmotBackend(
                    method=method,
                    device=dev,
                    reid_weights=reid_weights,
                    track_high_thresh=track_high_thresh,
                    track_low_thresh=track_low_thresh,
                    new_track_thresh=new_track_thresh,
                    match_thresh=match_thresh,
                    track_buffer=track_buffer,
                    frame_rate=frame_rate,
                    proximity_thresh=proximity_thresh,
                    appearance_thresh=appearance_thresh,
                )
                logger.info("HockeyTracker: boxmot/%s on %s", method, dev)
                return
            except Exception as e:
                logger.warning("boxmot backend failed (%s) — using Kalman+Hungarian", e)

        self._backend = _KalmanHungarianBackend(
            max_disappeared=track_buffer,
            iou_high=0.20,
            iou_low=0.10,
            min_hits=1,
            high_conf_thresh=track_high_thresh,
        )
        logger.info("HockeyTracker: Kalman+Hungarian backend")

    # ------------------------------------------------------------------ #
    def update(
        self,
        detections: Any,
        frame: Optional[np.ndarray] = None,
    ) -> List[TrackResult]:
        """
        Update tracker for one frame.

        Parameters
        ----------
        detections:
            • DetectionResult dict from HockeyPlayerDetector.detect_frame()
              {"boxes": [[x1,y1,x2,y2], ...], "scores": [...], "count": N}
            • List[Dict] from HockeyPlayerDetector.detect()
              [{"bbox": [x1,y1,x2,y2], "confidence": f, "class_id": 0}, ...]
        frame:
            BGR numpy array (needed for BotSort ReID; can be None for ByteTrack)

        Returns
        -------
        List[TrackResult] — active tracks with consistent integer track_id.
        """
        dets = _normalise_detections(detections)

        if frame is None:
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        return self._backend.update(dets, frame)

    def reset(self) -> None:
        """Reset tracker state (use between video clips)."""
        self._backend.reset()

    @property
    def backend_name(self) -> str:
        return type(self._backend).__name__


# ── detection normalisation ─────────────────────────────────────────────── #

def _normalise_detections(raw: Any) -> List[Dict]:
    """Accept either DetectionResult dict or List[Dict] from .detect()."""
    if isinstance(raw, dict):
        boxes  = raw.get("boxes",  [])
        scores = raw.get("scores", [])
        return [
            {"bbox": list(b), "confidence": float(s), "class_id": 0}
            for b, s in zip(boxes, scores)
        ]
    return [dict(d) for d in raw]


# ── factory ─────────────────────────────────────────────────────────────── #

def create_tracker(config: Optional[Dict] = None) -> HockeyTracker:
    """Create a HockeyTracker from an optional config dict."""
    if not config:
        return HockeyTracker()
    return HockeyTracker(
        method             = config.get("method",              "botsort"),
        device             = config.get("device",              None),
        track_high_thresh  = config.get("track_high_thresh",   0.35),
        track_low_thresh   = config.get("track_low_thresh",    0.10),
        new_track_thresh   = config.get("new_track_thresh",    0.45),
        match_thresh       = config.get("match_thresh",        0.75),
        track_buffer       = config.get("track_buffer",        90),
        frame_rate         = config.get("frame_rate",          30),
        proximity_thresh   = config.get("proximity_thresh",    0.25),
        appearance_thresh  = config.get("appearance_thresh",   0.40),
    )


# ── backwards-compat alias ──────────────────────────────────────────────── #
PlayerTracker = HockeyTracker


# ═══════════════════════════════════════════════════════════════════════════ #
#  Smoke-test / sampling test                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

def _sample_test(
    video_path: str = "data/videos/game1.mp4",
    n_frames: int = 600,
    start_frame: int = 5000,
    verbose: bool = True,
) -> Dict:
    """
    Read n_frames *consecutive* frames from video_path starting at start_frame,
    run detection + tracking, and report consistency metrics.

    Consecutive frames are required to evaluate tracking quality — sparse
    sampling would reset all tracks between samples.
    """
    import cv2
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from pipeline.detection import HockeyPlayerDetector

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Seek to start_frame and read consecutive indices
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    indices      = list(range(start_frame, min(start_frame + n_frames, total_frames)))

    detector = HockeyPlayerDetector()
    tracker  = HockeyTracker(frame_rate=int(fps))

    track_history: Dict[int, List[int]] = {}   # tid → list of frame indices seen
    id_switches = 0
    total_detections = 0
    frame_track_counts: List[int] = []

    if verbose:
        print(f"Video: {video_path}  ({total_frames} frames @ {fps:.1f} fps)")
        print(f"Backend: {tracker.backend_name}")
        print(f"Reading {n_frames} consecutive frames from frame {start_frame} …")

    prev_active: set = set()
    for fi, frame_idx in enumerate(indices):
        ret, frame = cap.read()
        if not ret:
            break

        det_result = detector.detect_frame(frame)
        tracks     = tracker.update(det_result, frame)

        total_detections += det_result["count"]
        frame_track_counts.append(len(tracks))

        cur_active = {t["track_id"] for t in tracks}
        # ID switches: tracks that disappeared last frame but appear with same
        # box position as a new ID are hard to count without ground truth.
        # Proxy: count tracks that vanish and new IDs that appear simultaneously.
        vanished = prev_active - cur_active
        appeared = cur_active - prev_active
        # Each (vanish + appear) pair within 2 frames is a potential ID switch.
        id_switches += min(len(vanished), len(appeared))
        prev_active = cur_active

        for t in tracks:
            tid = t["track_id"]
            track_history.setdefault(tid, []).append(fi)

        if verbose and fi % 20 == 0:
            print(f"  frame {fi+1:3d}/{n_frames}  "
                  f"idx={frame_idx:5d}  "
                  f"dets={det_result['count']:2d}  "
                  f"tracks={len(tracks):2d}  "
                  f"active_ids={sorted(cur_active)}")

    cap.release()

    total_tracks    = len(track_history)
    track_lengths   = [len(v) for v in track_history.values()]
    avg_track_len   = float(np.mean(track_lengths)) if track_lengths else 0.0
    long_tracks     = sum(1 for l in track_lengths if l >= 5)
    avg_det_per_frm = total_detections / max(n_frames, 1)
    avg_trk_per_frm = float(np.mean(frame_track_counts)) if frame_track_counts else 0.0

    # Fragmentation ratio: many short tracks = fragmented tracking
    fragmentation = sum(1 for l in track_lengths if l <= 2) / max(total_tracks, 1)

    metrics = {
        "backend":            tracker.backend_name,
        "frames_sampled":     n_frames,
        "total_unique_ids":   total_tracks,
        "avg_detections_per_frame": round(avg_det_per_frm, 2),
        "avg_tracks_per_frame":     round(avg_trk_per_frm, 2),
        "avg_track_length_frames":  round(avg_track_len, 2),
        "long_tracks_ge5":    long_tracks,
        "fragmentation_ratio": round(fragmentation, 3),
        "proxy_id_switches":  id_switches,
    }

    if verbose:
        print("\n── Tracking Metrics ──────────────────────────────────")
        for k, v in metrics.items():
            print(f"  {k:<38s}: {v}")
        print()
        if fragmentation < 0.30 and avg_track_len >= 4:
            print("  ✓ Tracking looks consistent")
        else:
            print("  ⚠ High fragmentation — check thresholds or detection quality")

    return metrics


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _sample_test(verbose=True)
