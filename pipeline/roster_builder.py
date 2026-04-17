"""
Roster Builder for IceIQ.

Extracts player still-cuts from the first few minutes of a game video,
deduplicates by position grid, and returns them for user labeling in the app.

App UI flow:
  1. User uploads video
  2. RosterBuilder.extract_player_crops() → list of PlayerCrop
  3. App shows each crop as a card: [image | team selector | name field | jersey # field]
  4. User fills in details → submit
  5. RosterBuilder.build_roster(labeled_crops) → Roster object
  6. Roster is injected into HockeyTeamClassifier + CrossShiftIdentityTracker
"""

import cv2
import numpy as np
import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PlayerCrop:
    """A single player still-cut with metadata."""
    crop_id: str                    # unique ID, e.g. "grid_5_3"
    image_path: str                 # path to saved crop JPEG
    image_b64: Optional[str] = None # base64 for direct API/app delivery
    frame_number: int = 0
    timestamp_sec: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1,y1,x2,y2 in original frame

    # Filled by user in app:
    team_label: Optional[str] = None    # e.g. "aigis" or "lopez"
    player_name: Optional[str] = None   # e.g. "김민준"
    jersey_number: Optional[int] = None

    # Auto-detected hints (shown to user as suggestions):
    hint_colour: Optional[str] = None   # "red" / "white" / "unknown"
    hint_jersey: Optional[str] = None   # OCR result if available


@dataclass
class Roster:
    """Complete labeled roster for one game."""
    team_a_label: str
    team_a_name: str
    team_a_colour: str
    team_b_label: str
    team_b_name: str
    team_b_colour: str
    players: List[PlayerCrop] = field(default_factory=list)

    def players_for_team(self, team_label: str) -> List[PlayerCrop]:
        return [p for p in self.players if p.team_label == team_label]

    def jersey_to_player(self) -> Dict[Tuple[str, int], PlayerCrop]:
        """Map (team_label, jersey_number) → PlayerCrop."""
        return {
            (p.team_label, p.jersey_number): p
            for p in self.players
            if p.team_label and p.jersey_number is not None
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Roster":
        players = [PlayerCrop(**p) for p in d.pop("players", [])]
        return cls(**d, players=players)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Roster saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Roster":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


class RosterBuilder:
    """
    Extracts player crops from a video for the app's roster-labeling UI.

    Parameters
    ----------
    model_path  : YOLO model path
    device      : 'mps', 'cuda', or 'cpu'
    output_dir  : directory to save crop images
    sample_mins : list of minutes to sample for crop extraction
    grid_size   : deduplication grid cell size in pixels (default 80)
    min_height  : minimum bbox height to keep (default 50px)
    max_crops   : maximum number of crops to return (default 30)
    """

    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        device: str = None,
        output_dir: str = "/tmp/iceiq_roster",
        sample_mins: Optional[List[float]] = None,
        grid_size: int = 80,
        min_height: int = 50,
        max_crops: int = 30,
    ):
        from pipeline.detection import YOLODetector
        self.detector = YOLODetector(model_path=model_path, device=device)
        self.output_dir = output_dir
        self.sample_mins = sample_mins or [20.0, 20.5, 21.0, 21.5, 22.0]
        self.grid_size = grid_size
        self.min_height = min_height
        self.max_crops = max_crops
        os.makedirs(output_dir, exist_ok=True)

    def extract_player_crops(
        self,
        video_path: str,
        include_b64: bool = False,
    ) -> List[PlayerCrop]:
        """
        Sample frames from video, detect players, deduplicate, save crops.

        Parameters
        ----------
        video_path  : path to video file
        include_b64 : if True, embed base64 image in each PlayerCrop

        Returns
        -------
        List of PlayerCrop objects ready for app UI display.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_min = total_frames / fps / 60

        # Auto-adjust sample_mins if video is shorter
        sample_mins = [m for m in self.sample_mins if m < duration_min]
        if not sample_mins:
            sample_mins = [duration_min * 0.3]

        seen_grids: Dict[str, PlayerCrop] = {}

        for minute in sample_mins:
            fi = int(fps * 60 * minute)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if not ret:
                continue

            dets = self.detector.detect(frame)
            for i, d in enumerate(dets):
                x1, y1, x2, y2 = d["bbox"]
                bh, bw = y2 - y1, x2 - x1
                if bh < self.min_height:
                    continue
                if bw > 0 and bw / bh > 1.5:
                    continue  # skip wide/horizontal detections

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                grid_key = f"{cx // self.grid_size}_{cy // self.grid_size}"

                if grid_key in seen_grids:
                    continue  # already have a crop from this position

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_id = f"grid_{grid_key}"
                img_path = os.path.join(self.output_dir, f"player_{grid_key}.jpg")
                cv2.imwrite(img_path, crop)

                hint_colour = self._detect_colour_hint(crop)

                pc = PlayerCrop(
                    crop_id=crop_id,
                    image_path=img_path,
                    frame_number=fi,
                    timestamp_sec=round(minute * 60, 1),
                    bbox=(x1, y1, x2, y2),
                    hint_colour=hint_colour,
                )

                if include_b64:
                    import base64
                    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    pc.image_b64 = base64.b64encode(buf).decode()

                seen_grids[grid_key] = pc

            if len(seen_grids) >= self.max_crops:
                break

        cap.release()
        crops = list(seen_grids.values())[: self.max_crops]
        logger.info(f"Extracted {len(crops)} player crops from {video_path}")
        return crops

    def build_roster(
        self,
        labeled_crops: List[PlayerCrop],
        team_a_name: str,
        team_a_label: str,
        team_a_colour: str,
        team_b_name: str,
        team_b_label: str,
        team_b_colour: str,
    ) -> Roster:
        """
        Build a Roster from user-labeled crops.
        Only includes crops where team_label is set.
        """
        labeled = [c for c in labeled_crops if c.team_label]
        return Roster(
            team_a_label=team_a_label,
            team_a_name=team_a_name,
            team_a_colour=team_a_colour,
            team_b_label=team_b_label,
            team_b_name=team_b_name,
            team_b_colour=team_b_colour,
            players=labeled,
        )

    @staticmethod
    def _detect_colour_hint(crop: np.ndarray) -> str:
        """
        Quick colour hint for the app UI (shown as a suggestion chip).
        Returns 'red', 'white', 'black', or 'unknown'.
        """
        if crop is None or crop.size == 0:
            return "unknown"

        # Use torso region only
        h, w = crop.shape[:2]
        y1 = int(h * 0.20)
        y2 = int(h * 0.65)
        torso = crop[y1:y2, int(w * 0.15):w - int(w * 0.15)]
        if torso.size == 0:
            torso = crop

        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)

        # Red detection using LAB a* channel
        a_channel = lab[:, :, 1].astype(float)
        red_ratio = float(np.mean(a_channel > 140))  # a* > 140 in uint8 = positive red

        # White: high V, low S
        white_mask = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 170)
        white_ratio = float(np.mean(white_mask))

        # Black: low V
        black_mask = hsv[:, :, 2] < 60
        black_ratio = float(np.mean(black_mask))

        best = max(
            [("red", red_ratio), ("white", white_ratio), ("black", black_ratio)],
            key=lambda x: x[1],
        )
        return best[0] if best[1] > 0.15 else "unknown"


# ---------------------------------------------------------------------------
# Convenience function for API / testing
# ---------------------------------------------------------------------------

def extract_roster_crops(
    video_path: str,
    output_dir: str = "/tmp/iceiq_roster",
    model_path: str = "yolov8m.pt",
    device: str = None,
    include_b64: bool = False,
) -> List[dict]:
    """
    One-shot helper: extract crops and return as list of dicts (JSON-serializable).
    Use this from a REST API endpoint or test script.
    """
    builder = RosterBuilder(
        model_path=model_path,
        device=device,
        output_dir=output_dir,
    )
    crops = builder.extract_player_crops(video_path, include_b64=include_b64)
    return [asdict(c) for c in crops]


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else "data/videos/waves_g18.mp4"
    crops = extract_roster_crops(video, include_b64=False)
    print(f"\n추출된 선수 crop: {len(crops)}개\n")
    for c in crops:
        hint = c.get("hint_colour", "?")
        ts = c.get("timestamp_sec", 0)
        path = c.get("image_path", "")
        print(f"  [{hint:>7}] {os.path.basename(path)}  ({ts:.0f}s)")
