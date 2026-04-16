"""
IceIQ Puck Estimator (PLUCC-style)
YOLOv8m-pose 키포인트 → 선수 신체 방향 → 퍽 위치 추론

핵심 원리:
  선수들은 퍽을 향해 몸과 시선을 돌린다.
  10명의 신체 방향 벡터 교차점 = 퍽 추정 위치

YOLOv8 Pose 키포인트 (COCO 17):
  0:nose  1:l_eye  2:r_eye  3:l_ear  4:r_ear
  5:l_shoulder  6:r_shoulder  7:l_elbow  8:r_elbow
  9:l_wrist  10:r_wrist  11:l_hip  12:r_hip
  13:l_knee  14:r_knee  15:l_ankle  16:r_ankle

사용:
  python puck_estimator.py <pose_data.json> [--output puck_timeline.json]
"""

import json
import math
import sys
from typing import Optional

# ─── 상수 ──────────────────────────────────────────────────────────────

# 키포인트 인덱스
NOSE        = 0
L_EYE       = 1
R_EYE       = 2
L_SHOULDER  = 5
R_SHOULDER  = 6
L_WRIST     = 9
R_WRIST     = 10
L_HIP       = 11
R_HIP       = 12

# 퍽 추정 파라미터
GAZE_RAY_LENGTH         = 2000  # 시선 벡터 최대 연장 (px)
MIN_CONFIDENCE          = 0.3   # 키포인트 최소 신뢰도
MIN_PLAYERS_FOR_ESTIMATE = 2    # 최소 N명의 벡터 필요
TEMPORAL_SMOOTH_ALPHA   = 0.15  # 시간축 스무딩 (0=이전값 유지, 1=새값만)
SHOT_SPEED_THRESHOLD    = 2000  # px/frame — 이 이상이면 슈팅으로 판정


# ─── 유틸 ──────────────────────────────────────────────────────────────

def vec_sub(a: tuple, b: tuple) -> tuple:
    return (a[0] - b[0], a[1] - b[1])

def vec_len(v: tuple) -> float:
    return math.sqrt(v[0]**2 + v[1]**2)

def vec_normalize(v: tuple) -> tuple:
    l = vec_len(v)
    if l < 1e-6:
        return (0.0, 0.0)
    return (v[0] / l, v[1] / l)

def dist2d(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def midpoint(a: tuple, b: tuple) -> tuple:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


# ─── 1. 신체 방향 벡터 추출 ───────────────────────────────────────────

def get_body_orientation(keypoints: list, confidences: list = None) -> Optional[dict]:
    """
    포즈 키포인트 → 신체 방향 벡터 + 시선 원점 계산

    keypoints: [[x,y], ...] 17개 (COCO format)
    confidences: [conf, ...] 17개 (선택)

    반환: {
        "origin": (x, y),       # 시선 시작점 (어깨 중심)
        "direction": (dx, dy),  # 정규화된 방향 벡터
        "confidence": float,    # 방향 신뢰도
        "facing_angle": float,  # 각도 (degrees, 0=오른쪽, 반시계)
    }
    """
    if not keypoints or len(keypoints) < 17:
        return None

    conf = confidences if confidences else [1.0] * 17

    def kp(idx):
        """키포인트 좌표 반환 (신뢰도 체크)"""
        if idx >= len(keypoints):
            return None
        pt = keypoints[idx]
        if not pt or (isinstance(pt, (list, tuple)) and len(pt) < 2):
            return None
        x, y = float(pt[0]), float(pt[1])
        if x == 0 and y == 0:
            return None
        if conf[idx] < MIN_CONFIDENCE:
            return None
        return (x, y)

    # 방법 1: 어깨 중심 → 코 방향 (가장 정확)
    l_sh  = kp(L_SHOULDER)
    r_sh  = kp(R_SHOULDER)
    nose  = kp(NOSE)

    if l_sh and r_sh and nose:
        shoulder_mid = midpoint(l_sh, r_sh)
        # 어깨 법선 벡터 (어깨 선에 수직 = 몸이 향하는 방향)
        sh_vec  = vec_sub(r_sh, l_sh)
        # 수직 벡터 (90도 회전): (dx, dy) → (-dy, dx) 또는 (dy, -dx)
        # 코 방향으로 선택
        normal1 = (-sh_vec[1],  sh_vec[0])
        normal2 = ( sh_vec[1], -sh_vec[0])
        nose_dir = vec_sub(nose, shoulder_mid)
        dot1 = normal1[0] * nose_dir[0] + normal1[1] * nose_dir[1]
        dot2 = normal2[0] * nose_dir[0] + normal2[1] * nose_dir[1]
        facing = vec_normalize(normal1 if dot1 > dot2 else normal2)
        angle  = math.degrees(math.atan2(-facing[1], facing[0]))

        return {
            "origin":       shoulder_mid,
            "direction":    facing,
            "confidence":   min(conf[L_SHOULDER], conf[R_SHOULDER], conf[NOSE]),
            "facing_angle": angle,
        }

    # 방법 2: 엉덩이 중심 → 어깨 중심 (어깨 하나만 보일 때)
    l_hip = kp(L_HIP)
    r_hip = kp(R_HIP)
    any_shoulder = l_sh or r_sh

    if l_hip and r_hip and any_shoulder:
        hip_mid   = midpoint(l_hip, r_hip)
        sh        = l_sh or r_sh
        direction = vec_normalize(vec_sub(sh, hip_mid))
        return {
            "origin":       sh,
            "direction":    direction,
            "confidence":   0.5,
            "facing_angle": math.degrees(math.atan2(-direction[1], direction[0])),
        }

    # 방법 3: 코 → 눈 중심 방향 (머리만 보일 때)
    l_eye = kp(L_EYE)
    r_eye = kp(R_EYE)

    if nose and (l_eye or r_eye):
        eye_mid   = midpoint(l_eye, r_eye) if (l_eye and r_eye) else (l_eye or r_eye)
        direction = vec_normalize(vec_sub(nose, eye_mid))
        return {
            "origin":       nose,
            "direction":    direction,
            "confidence":   0.3,
            "facing_angle": math.degrees(math.atan2(-direction[1], direction[0])),
        }

    return None


# ─── 2. 퍽 위치 추론 (시선 교차) ──────────────────────────────────────

def estimate_puck_position(
    player_orientations: list,
    weights: list = None,
) -> Optional[dict]:
    """
    여러 선수의 시선 벡터 → 최적 교차점 = 퍽 추정 위치

    가중 최소제곱법 (Weighted Least Squares)으로
    모든 시선 레이에 가장 가까운 점을 찾음

    player_orientations: [{"origin": (x,y), "direction": (dx,dy), "confidence": float}, ...]
    weights: 각 선수의 가중치 (None이면 confidence 사용)
    """
    valid = [o for o in player_orientations if o is not None]

    if len(valid) < MIN_PLAYERS_FOR_ESTIMATE:
        return None

    # 가중 최소제곱 — 모든 레이에 가장 가까운 점 찾기
    # 각 레이: P + t*D (P=origin, D=direction)
    # 점 X가 레이에서의 거리 = ||(X-P) - ((X-P)·D)D||²
    # 미분해서 0으로 놓으면 선형 시스템 AX = b

    A = [[0.0, 0.0], [0.0, 0.0]]
    b = [0.0, 0.0]

    for i, ori in enumerate(valid):
        px, py = ori["origin"]
        dx, dy = ori["direction"]
        w = weights[i] if weights else ori.get("confidence", 1.0)

        # I - D*D^T (투영 행렬의 보수)
        m00 = (1 - dx * dx) * w
        m01 = (-dx * dy)    * w
        m11 = (1 - dy * dy) * w

        A[0][0] += m00
        A[0][1] += m01
        A[1][0] += m01
        A[1][1] += m11

        b[0] += m00 * px + m01 * py
        b[1] += m01 * px + m11 * py

    # 2×2 선형 시스템 풀기
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if abs(det) < 1e-10:
        # 모든 선수가 같은 방향을 보고 있음 → centroid fallback
        cx = sum(o["origin"][0] for o in valid) / len(valid)
        cy = sum(o["origin"][1] for o in valid) / len(valid)
        return {"x": cx, "y": cy, "confidence": 0.2, "method": "centroid_fallback"}

    x = (A[1][1] * b[0] - A[0][1] * b[1]) / det
    y = (A[0][0] * b[1] - A[1][0] * b[0]) / det

    # 신뢰도: 각 레이에서 추정점까지의 평균 거리 (작을수록 높음)
    total_dist = 0.0
    for ori in valid:
        px, py = ori["origin"]
        dx, dy = ori["direction"]
        to_point = (x - px, y - py)
        proj     = to_point[0] * dx + to_point[1] * dy
        closest  = (px + proj * dx, py + proj * dy)
        total_dist += dist2d((x, y), closest)

    avg_dist = total_dist / len(valid)
    conf     = max(0.0, min(1.0, 1 - avg_dist / 300))

    # outlier rejection: 신뢰도 낮으면 None
    if conf < 0.3:
        return None

    return {
        "x":               round(x, 1),
        "y":               round(y, 1),
        "confidence":      round(conf, 3),
        "method":          "gaze_intersection",
        "rays_used":       len(valid),
        "avg_ray_distance": round(avg_dist, 1),
    }


# ─── 3. 소유 판정 ─────────────────────────────────────────────────────

def determine_possession(
    puck_pos: dict,
    player_positions: list,
    player_teams: list,
    possession_radius: float = 50.0,
) -> dict:
    """
    퍽 위치 + 선수 위치 → 퍽 소유자 판정

    player_positions: [(x,y), ...]
    player_teams: ["HOME", "AWAY", ...]
    possession_radius: 퍽과 이 거리 이내면 소유로 판정 (px)
    """
    if not puck_pos or puck_pos.get("confidence", 0) < 0.2:
        return {"owner_idx": None, "team": None, "distance": None}

    px, py    = puck_pos["x"], puck_pos["y"]
    min_dist  = float("inf")
    owner_idx = -1

    for i, pos in enumerate(player_positions):
        d = dist2d((px, py), pos)
        if d < min_dist:
            min_dist  = d
            owner_idx = i

    if min_dist <= possession_radius and owner_idx >= 0:
        team = player_teams[owner_idx] if owner_idx < len(player_teams) else None
        if team in ("Aigis", "aigis"):   team = "HOME"
        elif team in ("Lopez", "lopez"): team = "AWAY"
        return {
            "owner_idx": owner_idx,
            "team":      team,
            "distance":  round(min_dist, 1),
        }

    return {"owner_idx": None, "team": None, "distance": round(min_dist, 1)}


# ─── 4. 이벤트 감지 ───────────────────────────────────────────────────

class EventDetector:
    """프레임별 퍽 위치 시계열 → 이벤트 감지"""

    def __init__(self, fps: float = 4.0):
        self.fps       = fps
        self.history   = []   # [{frame, x, y, confidence, team}, ...]
        self.events    = []
        self.prev_team = None

    def add_frame(self, frame: int, puck: dict, possession: dict):
        """프레임 데이터 추가 + 이벤트 감지"""
        if not puck:
            return

        team = possession.get("team")
        if team in ("Aigis", "aigis"):   team = "HOME"
        elif team in ("Lopez", "lopez"): team = "AWAY"
        entry = {
            "frame":      frame,
            "time_sec":   round(frame / self.fps, 2),
            "x":          puck.get("x", 0),
            "y":          puck.get("y", 0),
            "confidence": puck.get("confidence", 0),
            "team":       team,
        }
        self.history.append(entry)

        if len(self.history) >= 2:
            prev  = self.history[-2]
            curr  = self.history[-1]
            dx    = curr["x"] - prev["x"]
            dy    = curr["y"] - prev["y"]
            dt    = (curr["frame"] - prev["frame"]) or 1
            speed = math.sqrt(dx * dx + dy * dy) / dt

            # 비현실적 속도 = 위치 튀기 → 무시
            if speed > 5000:
                return

            # 슈팅 감지: 퍽 속도 급증
            if speed > SHOT_SPEED_THRESHOLD:
                self.events.append({
                    "type":     "SHOT",
                    "frame":    frame,
                    "time_sec": entry["time_sec"],
                    "position": (curr["x"], curr["y"]),
                    "speed":    round(speed, 1),
                    "team":     curr["team"],
                })

            # 턴오버 감지: 소유팀 변경
            if (self.prev_team and curr["team"] and
                    self.prev_team != curr["team"]):
                self.events.append({
                    "type":      "TURNOVER",
                    "frame":     frame,
                    "time_sec":  entry["time_sec"],
                    "position":  (curr["x"], curr["y"]),
                    "from_team": self.prev_team,
                    "to_team":   curr["team"],
                })

        if entry["team"]:
            self.prev_team = entry["team"]

    def get_possession_stats(self) -> dict:
        home_frames = sum(1 for h in self.history if h["team"] == "HOME")
        away_frames = sum(1 for h in self.history if h["team"] == "AWAY")
        total = home_frames + away_frames or 1
        return {
            "home_pct":      round(home_frames / total * 100, 1),
            "away_pct":      round(away_frames / total * 100, 1),
            "home_time_sec": round(home_frames / self.fps, 1),
            "away_time_sec": round(away_frames / self.fps, 1),
        }

    def get_shot_count(self) -> dict:
        shots = [e for e in self.events if e["type"] == "SHOT"]
        home  = sum(1 for s in shots if s["team"] == "HOME")
        away  = sum(1 for s in shots if s["team"] == "AWAY")
        return {"home": home, "away": away, "total": len(shots)}

    def get_turnover_count(self) -> dict:
        tos = [e for e in self.events if e["type"] == "TURNOVER"]
        return {"total": len(tos), "events": tos}


# ─── 5. xG (기대 골) 계산 ─────────────────────────────────────────────

def calculate_xg(
    shot_x: float, shot_y: float,
    rink_w: float = 1920, rink_h: float = 1080,
) -> float:
    """슈팅 위치 → 간단한 xG (기대 골) 계산"""
    goal_x, goal_y = rink_w * 0.95, rink_h / 2
    distance   = dist2d((shot_x, shot_y), (goal_x, goal_y))
    goal_width = rink_h * 0.06
    angle      = math.atan2(goal_width / 2, distance) * 2

    dist_factor  = math.exp(-distance / (rink_w * 0.15))
    angle_factor = min(angle / math.radians(20), 1.0)

    in_slot    = (shot_x > rink_w * 0.7 and rink_h * 0.3 < shot_y < rink_h * 0.7)
    slot_bonus = 1.5 if in_slot else 1.0

    xg = min(dist_factor * angle_factor * slot_bonus * 0.3, 0.95)
    return round(xg, 3)


# ─── 6. 시간축 스무딩 ─────────────────────────────────────────────────

class PuckSmoother:
    """퍽 위치 시간축 스무딩 (EMA)"""

    def __init__(self, alpha: float = TEMPORAL_SMOOTH_ALPHA):
        self.alpha  = alpha
        self.prev_x = None
        self.prev_y = None

    def smooth(self, puck: dict) -> dict:
        if not puck:
            return puck
        x, y = puck["x"], puck["y"]
        if self.prev_x is not None:
            x = self.alpha * x + (1 - self.alpha) * self.prev_x
            y = self.alpha * y + (1 - self.alpha) * self.prev_y
        self.prev_x, self.prev_y = x, y
        return {**puck, "x": round(x, 1), "y": round(y, 1), "smoothed": True}


# ─── 7. 전체 파이프라인 ───────────────────────────────────────────────

def process_game(
    pose_data: list,
    fps: float = 4.0,
    rink_w: float = 1920,
    rink_h: float = 1080,
) -> dict:
    """
    전체 경기 포즈 데이터 → 퍽 추적 + 이벤트 감지

    pose_data: [
        {
            "frame": int,
            "players": [
                {
                    "track_id": int,
                    "team": "HOME"|"AWAY",
                    "keypoints": [[x,y], ...],   # 17 COCO keypoints
                    "confidences": [float, ...],  # 17 confidence scores
                    "bbox": [x1,y1,x2,y2],
                }
            ]
        }
    ]
    """
    smoother = PuckSmoother()
    detector = EventDetector(fps)
    timeline = []

    for frame_data in pose_data:
        frame   = frame_data.get("frame", 0)
        players = frame_data.get("players", [])

        orientations = []
        positions    = []
        teams        = []

        for p in players:
            kps   = p.get("keypoints", [])
            confs = p.get("confidences", [1.0] * 17)
            raw_ori = p.get("ori")  # [ox, oy, dx, dy] 형태

            # ori가 있으면 바로 사용, 없으면 keypoints에서 계산
            if raw_ori and len(raw_ori) == 4:
                ori = {
                    "origin":     (raw_ori[0], raw_ori[1]),
                    "direction":  (raw_ori[2], raw_ori[3]),
                    "confidence": 0.8,
                }
            elif kps and len(kps) >= 7:
                ori = get_body_orientation(kps, confs)
            else:
                ori = None

            if ori:
                orientations.append(ori)

            bbox = p.get("bbox", [0, 0, 0, 0])
            cx   = (bbox[0] + bbox[2]) / 2
            cy   = (bbox[1] + bbox[3]) / 2
            positions.append((cx, cy))
            teams.append(p.get("team", "UNKNOWN"))

        puck_raw  = estimate_puck_position(orientations)
        puck      = smoother.smooth(puck_raw) if puck_raw else None
        possession = determine_possession(puck, positions, teams) if puck else {}

        detector.add_frame(frame, puck, possession)

        xg = None
        if detector.events and detector.events[-1]["frame"] == frame:
            last = detector.events[-1]
            if last["type"] == "SHOT":
                xg = calculate_xg(last["position"][0], last["position"][1], rink_w, rink_h)
                last["xg"] = xg

        timeline.append({
            "frame":      frame,
            "time_sec":   round(frame / fps, 2),
            "puck":       puck,
            "possession": possession,
            "xg":         xg,
        })

    return {
        "fps":          fps,
        "total_frames": len(pose_data),
        "duration_sec": round(len(pose_data) / fps, 1) if fps > 0 else 0,
        "possession":   detector.get_possession_stats(),
        "shots":        detector.get_shot_count(),
        "turnovers":    detector.get_turnover_count(),
        "events":       detector.events,
        "timeline":     timeline,
    }


# ─── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IceIQ 퍽 추정기 (PLUCC-style)")
    parser.add_argument("input",   help="포즈 데이터 JSON")
    parser.add_argument("--output", "-o", help="출력 JSON")
    parser.add_argument("--fps",   type=float, default=4.0)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if isinstance(data, list):
        pose_data = data
    elif "frames" in data:
        pose_data = data["frames"]
    else:
        print("⚠️ 입력 형식 미지원. 'frames' 키 또는 리스트 형태 필요.")
        sys.exit(1)

    result = process_game(pose_data, fps=args.fps)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ 저장: {args.output}")
    else:
        poss  = result["possession"]
        shots = result["shots"]
        tos   = result["turnovers"]

        print(f"\n{'='*50}")
        print(f"🏒 IceIQ 퍽 추정 결과")
        print(f"{'='*50}")
        print(f"\n📊 퍽 소유:")
        print(f"   HOME: {poss['home_pct']}% ({poss['home_time_sec']}초)")
        print(f"   AWAY: {poss['away_pct']}% ({poss['away_time_sec']}초)")
        print(f"\n🏒 슈팅:")
        print(f"   HOME: {shots['home']}  AWAY: {shots['away']}  합계: {shots['total']}")
        print(f"\n🔄 턴오버: {tos['total']}회")

        if result["events"]:
            print(f"\n📋 이벤트 타임라인:")
            for e in result["events"][:20]:
                t = e["time_sec"]
                if e["type"] == "SHOT":
                    xg_str = f" (xG: {e['xg']})" if e.get("xg") else ""
                    print(f"   [{t:>7.1f}s] 🏒 슈팅 by {e['team']} — 속도 {e['speed']}{xg_str}")
                elif e["type"] == "TURNOVER":
                    print(f"   [{t:>7.1f}s] 🔄 턴오버 {e['from_team']} → {e['to_team']}")

        confs = [t["puck"]["confidence"] for t in result["timeline"] if t.get("puck")]
        if confs:
            print(f"\n📈 퍽 추정 신뢰도:")
            print(f"   평균: {sum(confs)/len(confs):.2f}")
            print(f"   감지율: {len(confs)}/{len(result['timeline'])} 프레임 "
                  f"({len(confs)/len(result['timeline'])*100:.0f}%)")
        print()
