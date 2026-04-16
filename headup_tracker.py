"""
IceIQ Head-Up Tracker
포즈 키포인트 → 헤드업 빈도 + 시선 방향 분석

헤드업 = 머리를 들고 주변을 살피는 상태
헤드다운 = 퍽/발밑을 보고 있는 상태

하키 IQ의 핵심 지표:
  - 헤드업 비율 높을수록 시야 확보 → 패스 옵션 ↑
  - 헤드다운이 길수록 턴오버 위험 ↑
  - 퍼 소유 시 vs 미소유 시 헤드업 차이 = 게임센스 지표

키포인트 (COCO 17):
  0:nose  1:l_eye  2:r_eye  3:l_ear  4:r_ear
  5:l_shoulder  6:r_shoulder

사용:
  python headup_tracker.py <pose_data.json> --output headup_report.json
  python headup_tracker.py <all_tracks_with_ori.json>
"""

import json
import math
import sys
from collections import defaultdict
from typing import Optional

# ─── 키포인트 인덱스 ──────────────────────────────────────────────────

NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6

MIN_CONF = 0.3


# ─── 헤드업 판정 ──────────────────────────────────────────────────────

def classify_head_state(keypoints: list, confidences: list = None) -> dict:
    """
    포즈 키포인트 → 헤드업/다운 판정 + 시선 방향

    판정 기준:
      1. 코-어깨 수직 각도: 코가 어깨보다 위(또는 수평) = 헤드업
      2. 코 y < 어깨중심 y = 머리 들고 있음 (이미지 좌표: 위가 작음)
      3. 시선 방향: 코 → 눈 중심 벡터의 수평 각도

    반환: {
      "state": "HEAD_UP" | "HEAD_DOWN" | "NEUTRAL",
      "gaze_angle": float (degrees, 0=오른쪽, 반시계),
      "gaze_direction": "LEFT" | "RIGHT" | "FORWARD" | "BACKWARD",
      "head_tilt": float (degrees, 양수=머리 들기, 음수=숙이기),
      "confidence": float,
    }
    """
    if not keypoints or len(keypoints) < 7:
        return None

    conf = confidences if confidences else [1.0] * len(keypoints)

    def kp(idx):
        if idx >= len(keypoints):
            return None
        pt = keypoints[idx]
        if not pt or len(pt) < 2:
            return None
        x, y = float(pt[0]), float(pt[1])
        if x == 0 and y == 0:
            return None
        if idx < len(conf) and conf[idx] < MIN_CONF:
            return None
        return (x, y)

    nose = kp(NOSE)
    l_sh = kp(L_SHOULDER)
    r_sh = kp(R_SHOULDER)
    l_eye = kp(L_EYE)
    r_eye = kp(R_EYE)
    l_ear = kp(L_EAR)
    r_ear = kp(R_EAR)

    if not nose or not (l_sh or r_sh):
        return None

    # 어깨 중심
    if l_sh and r_sh:
        sh_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
    else:
        sh_mid = l_sh or r_sh

    # ── 1. 헤드업/다운 판정 ──────────────────────────────────
    # 이미지 좌표에서 y가 작을수록 위쪽
    # 코가 어깨중심보다 위에 있으면 = 머리 들고 있음
    head_offset_y = nose[1] - sh_mid[1]  # 음수 = 코가 위 = 헤드업

    # 어깨 너비로 정규화 (체형 보정)
    shoulder_width = abs(l_sh[0] - r_sh[0]) if (l_sh and r_sh) else 100
    if shoulder_width < 10:
        shoulder_width = 100

    # 정규화된 수직 오프셋 (-1 = 코가 어깨 너비만큼 위)
    normalized_offset = head_offset_y / shoulder_width

    # 헤드 틸트 각도 (degrees)
    head_tilt = math.degrees(math.atan2(-head_offset_y, shoulder_width))

    # 판정 기준
    # 헬멧 쓴 하키선수: 스케이팅 자세에서 약간 숙여도 정상
    if normalized_offset < -0.3:
        state = "HEAD_UP"        # 코가 어깨보다 확실히 위
    elif normalized_offset > 0.1:
        state = "HEAD_DOWN"      # 코가 어깨 아래로 (퍽/발밑 보기)
    else:
        state = "NEUTRAL"        # 일반 스케이팅 자세

    # ── 2. 시선 방향 계산 ──────────────────────────────────
    gaze_angle = None
    gaze_dir = "UNKNOWN"

    # 방법 A: 코 → 눈 중심 (가장 정확)
    if l_eye and r_eye:
        eye_mid = ((l_eye[0] + r_eye[0]) / 2, (l_eye[1] + r_eye[1]) / 2)
        gaze_vec = (nose[0] - eye_mid[0], nose[1] - eye_mid[1])
        gaze_angle = math.degrees(math.atan2(-gaze_vec[1], gaze_vec[0]))

    # 방법 B: 귀 비대칭 (고개 돌림 감지)
    elif l_ear and r_ear:
        # 왼쪽 귀가 보이면 오른쪽을 보고 있고, 반대도 마찬가지
        ear_diff = abs(l_ear[0] - nose[0]) - abs(r_ear[0] - nose[0])
        if ear_diff > shoulder_width * 0.1:
            gaze_angle = 0    # 오른쪽 보기
        elif ear_diff < -shoulder_width * 0.1:
            gaze_angle = 180  # 왼쪽 보기

    # 방법 C: 어깨 법선 (fallback)
    elif l_sh and r_sh:
        sh_vec = (r_sh[0] - l_sh[0], r_sh[1] - l_sh[1])
        normal = (-sh_vec[1], sh_vec[0])
        nose_dir = (nose[0] - sh_mid[0], nose[1] - sh_mid[1])
        dot = normal[0]*nose_dir[0] + normal[1]*nose_dir[1]
        facing = normal if dot > 0 else (sh_vec[1], -sh_vec[0])
        gaze_angle = math.degrees(math.atan2(-facing[1], facing[0]))

    # 각도 → 방향 텍스트
    if gaze_angle is not None:
        a = gaze_angle % 360
        if 315 <= a or a < 45:
            gaze_dir = "RIGHT"
        elif 45 <= a < 135:
            gaze_dir = "FORWARD"    # 위쪽 = 빙판에서 앞
        elif 135 <= a < 225:
            gaze_dir = "LEFT"
        else:
            gaze_dir = "BACKWARD"   # 뒤돌아보기

    return {
        "state": state,
        "gaze_angle": round(gaze_angle, 1) if gaze_angle is not None else None,
        "gaze_direction": gaze_dir,
        "head_tilt": round(head_tilt, 1),
        "normalized_offset": round(normalized_offset, 3),
        "confidence": min(conf[NOSE], conf[L_SHOULDER] if l_sh else 1, conf[R_SHOULDER] if r_sh else 1),
    }


# ─── ori 데이터에서 판정 (키포인트 없을 때) ─────────────────────────────

def classify_from_ori(ori: list, bbox: list) -> dict:
    """
    all_tracks의 ori [ox, oy, dx, dy] + bbox → 간소화된 시선 방향 분석

    키포인트가 없어도 방향 벡터만으로 기본 분석 가능
    (헤드업/다운은 ori만으로는 정확하지 않으므로 시선 방향만)
    """
    if not ori or len(ori) < 4:
        return None

    ox, oy, dx, dy = ori
    gaze_angle = math.degrees(math.atan2(-dy, dx))

    a = gaze_angle % 360
    if 315 <= a or a < 45:
        gaze_dir = "RIGHT"
    elif 45 <= a < 135:
        gaze_dir = "FORWARD"
    elif 135 <= a < 225:
        gaze_dir = "LEFT"
    else:
        gaze_dir = "BACKWARD"

    return {
        "state": "UNKNOWN",  # ori만으로는 헤드업/다운 판정 불가
        "gaze_angle": round(gaze_angle, 1),
        "gaze_direction": gaze_dir,
        "head_tilt": None,
        "normalized_offset": None,
        "confidence": 0.5,
    }


# ─── 선수별 집계 ──────────────────────────────────────────────────────

class HeadUpAnalyzer:
    """프레임별 헤드 상태 누적 → 선수별 헤드업 리포트"""

    def __init__(self, fps: float = 4.0):
        self.fps = fps
        self.player_data = defaultdict(lambda: {
            "frames": [],
            "head_up_count": 0,
            "head_down_count": 0,
            "neutral_count": 0,
            "gaze_directions": defaultdict(int),
            "head_up_streaks": [],      # 연속 헤드업 구간
            "head_down_streaks": [],    # 연속 헤드다운 구간
        })
        self._current_streak = {}  # {player_id: {"state": str, "start": int, "count": int}}

    def add_frame(self, player_id: str, frame: int, head_state: dict):
        """프레임 데이터 추가"""
        if not head_state:
            return

        pd = self.player_data[player_id]
        state = head_state["state"]

        pd["frames"].append({
            "frame": frame,
            "state": state,
            "gaze_angle": head_state.get("gaze_angle"),
            "gaze_direction": head_state.get("gaze_direction"),
        })

        if state == "HEAD_UP":
            pd["head_up_count"] += 1
        elif state == "HEAD_DOWN":
            pd["head_down_count"] += 1
        else:
            pd["neutral_count"] += 1

        gd = head_state.get("gaze_direction", "UNKNOWN")
        pd["gaze_directions"][gd] += 1

        # 연속 구간 추적
        streak = self._current_streak.get(player_id)
        if streak and streak["state"] == state:
            streak["count"] += 1
        else:
            # 이전 스트릭 저장
            if streak and streak["count"] >= 2:
                key = "head_up_streaks" if streak["state"] == "HEAD_UP" else "head_down_streaks"
                if streak["state"] in ("HEAD_UP", "HEAD_DOWN"):
                    pd[key].append({
                        "start_frame": streak["start"],
                        "end_frame": frame - 1,
                        "duration_sec": round(streak["count"] / self.fps, 1),
                        "frames": streak["count"],
                    })
            self._current_streak[player_id] = {"state": state, "start": frame, "count": 1}

    def get_report(self, player_id: str) -> dict:
        """선수별 헤드업 리포트 생성"""
        pd = self.player_data[player_id]
        total = pd["head_up_count"] + pd["head_down_count"] + pd["neutral_count"]
        if total == 0:
            return {"player_id": player_id, "total_frames": 0}

        head_up_pct = round(pd["head_up_count"] / total * 100, 1)
        head_down_pct = round(pd["head_down_count"] / total * 100, 1)

        # 최장 헤드업/다운 구간
        longest_up = max(pd["head_up_streaks"], key=lambda s: s["frames"], default=None)
        longest_down = max(pd["head_down_streaks"], key=lambda s: s["frames"], default=None)

        # 시선 방향 분포
        gaze_total = sum(pd["gaze_directions"].values()) or 1
        gaze_pct = {k: round(v / gaze_total * 100, 1) for k, v in pd["gaze_directions"].items()}

        # 헤드업 등급
        if head_up_pct >= 60:
            grade = "A"
            grade_label = "엘리트 시야"
        elif head_up_pct >= 45:
            grade = "B"
            grade_label = "좋은 시야"
        elif head_up_pct >= 30:
            grade = "C"
            grade_label = "보통"
        else:
            grade = "D"
            grade_label = "개선 필요"

        return {
            "player_id": player_id,
            "total_frames": total,
            "total_time_sec": round(total / self.fps, 1),

            "head_up_pct": head_up_pct,
            "head_down_pct": head_down_pct,
            "neutral_pct": round(pd["neutral_count"] / total * 100, 1),

            "head_up_count": pd["head_up_count"],
            "head_down_count": pd["head_down_count"],

            "grade": grade,
            "grade_label": grade_label,

            "gaze_distribution": gaze_pct,
            "primary_gaze": max(gaze_pct, key=gaze_pct.get) if gaze_pct else "UNKNOWN",

            "longest_head_up": longest_up,
            "longest_head_down": longest_down,

            "avg_head_up_streak_sec": round(
                sum(s["duration_sec"] for s in pd["head_up_streaks"]) / len(pd["head_up_streaks"]), 1
            ) if pd["head_up_streaks"] else 0,
            "avg_head_down_streak_sec": round(
                sum(s["duration_sec"] for s in pd["head_down_streaks"]) / len(pd["head_down_streaks"]), 1
            ) if pd["head_down_streaks"] else 0,
        }

    def get_all_reports(self) -> list:
        """모든 선수 리포트"""
        reports = []
        for pid in sorted(self.player_data.keys()):
            reports.append(self.get_report(pid))
        return reports


# ─── 전체 처리 ────────────────────────────────────────────────────────

def analyze_headup(
    pose_data: list,
    fps: float = 4.0,
) -> dict:
    """
    포즈 데이터 → 헤드업 분석

    pose_data: [
      {
        "frame": int,
        "players": [
          {"track_id"/"jersey": str, "team": str,
           "keypoints": [[x,y],...], "confidences": [...],
           "ori": [ox,oy,dx,dy]}  # ori만 있어도 방향 분석 가능
        ]
      }
    ]
    """
    analyzer = HeadUpAnalyzer(fps)

    for fd in pose_data:
        frame = fd.get("frame", 0)
        players = fd.get("players", [])

        for p in players:
            _jersey = p.get("jersey")
            pid = str(p.get("track_id", "?")) if (_jersey in (None, "", "?")) else str(_jersey)
            kps = p.get("keypoints", [])
            confs = p.get("confidences", [])
            ori = p.get("ori")

            # 키포인트가 있으면 정밀 분석
            if kps and len(kps) >= 7:
                state = classify_head_state(kps, confs)
            # ori만 있으면 방향만
            elif ori:
                state = classify_from_ori(ori, p.get("bbox", []))
            else:
                continue

            if state:
                analyzer.add_frame(pid, frame, state)

    reports = analyzer.get_all_reports()

    return {
        "fps": fps,
        "total_players": len(reports),
        "players": reports,
    }


# ─── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IceIQ 헤드업 분석기")
    parser.add_argument("input", help="포즈 데이터 JSON")
    parser.add_argument("--output", "-o", help="출력 JSON")
    parser.add_argument("--fps", type=float, default=4.0)
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if isinstance(data, list):
        pose_data = data
    elif "frames" in data:
        pose_data = data["frames"]
    else:
        print("⚠️ 'frames' 키 또는 리스트 형태 필요")
        sys.exit(1)

    result = analyze_headup(pose_data, fps=args.fps)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ 저장: {args.output}")
    else:
        print(f"\n{'='*50}")
        print(f"👀 IceIQ 헤드업 분석")
        print(f"{'='*50}")

        for r in result["players"]:
            pid = r["player_id"]
            print(f"\n  #{pid}")
            print(f"    ⏱  분석 시간: {r['total_time_sec']}초")
            print(f"    👆 헤드업: {r['head_up_pct']}% ({r['head_up_count']}프레임)")
            print(f"    👇 헤드다운: {r['head_down_pct']}% ({r['head_down_count']}프레임)")
            print(f"    📊 등급: {r['grade']} — {r['grade_label']}")
            print(f"    👀 주 시선: {r['primary_gaze']}")
            print(f"    🔄 시선 분포: {dict(r['gaze_distribution'])}")

            if r.get("longest_head_up"):
                print(f"    ⬆️  최장 헤드업: {r['longest_head_up']['duration_sec']}초")
            if r.get("longest_head_down"):
                print(f"    ⬇️  최장 헤드다운: {r['longest_head_down']['duration_sec']}초")
            if r.get("avg_head_up_streak_sec"):
                print(f"    📏 평균 헤드업 유지: {r['avg_head_up_streak_sec']}초")

        print()
