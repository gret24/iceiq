"""
IceIQ Tactics Classifier
10명 좌표 → 프레임별 전술 문맥 자동 분류

입력: all_tracks JSON
출력: tactics_timeline.json (프레임별 상황 분류)

분류 카테고리:
1. 존 상태 — OZ_ATTACK / NZ_TRANSIT / DZ_DEFEND / FACEOFF
2. 팀 포메이션 — 1-2-2 / 2-1-2 / 1-3-1 / 2-3 / SPREAD / SCRAMBLE
3. 플레이 타입 — BREAKOUT / FORECHECK / CYCLE / RUSH / DUMP / REGROUP / NEUTRAL
4. 수적 상황 — 5v5 / 4v5 / 5v4 / 3v3 / etc
"""

import json
import math
import sys
from collections import Counter
from typing import Any

# ─── 상수 ──────────────────────────────────────────────────────────────

# 링크 좌표 (픽셀 기준 — 영상 해상도에 따라 조정 필요)
# 호모그래피 적용 시 1040x520 기준
RINK_W = 1920  # 기본 영상 해상도 (조정 가능)
RINK_H = 1080

# 존 경계 (x좌표 비율)
ZONE_DZ_END = 0.31      # 수비존 끝 = 왼쪽 블루라인
ZONE_OZ_START = 0.69     # 공격존 시작 = 오른쪽 블루라인
CENTER_X = 0.50


# ─── 유틸 ──────────────────────────────────────────────────────────────

def bbox_center(bbox: list) -> tuple:
    """bbox → 중심점. [x1,y1,x2,y2] 또는 [cx,cy,w,h] 자동 감지."""
    if len(bbox) != 4:
        return (0, 0)
    # [cx,cy,w,h] 감지: w,h가 cx,cy보다 작으면
    if bbox[2] < bbox[0] and bbox[3] < bbox[1]:
        return (bbox[0], bbox[1])
    # [x1,y1,x2,y2]
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def dist(p1: tuple, p2: tuple) -> float:
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def mean_x(positions: list) -> float:
    if not positions:
        return 0
    return sum(p[0] for p in positions) / len(positions)


def mean_y(positions: list) -> float:
    if not positions:
        return 0
    return sum(p[1] for p in positions) / len(positions)


# ─── 1. 존 상태 분류 ──────────────────────────────────────────────────

def classify_zone_state(
    home_positions: list,
    away_positions: list,
    rink_w: float,
) -> dict:
    """
    양 팀 위치 → 존 상태 판정
    퍽 위치 추정: 밀집 중심
    """
    all_pos = home_positions + away_positions
    if not all_pos:
        return {"zone": "UNKNOWN", "puck_x_pct": 0.5}

    # 퍽 위치 추정: 모든 선수 중 가장 밀집된 영역의 중심
    puck_x = estimate_puck_x(all_pos)
    puck_x_pct = puck_x / rink_w if rink_w > 0 else 0.5

    if puck_x_pct < ZONE_DZ_END:
        zone = "DZ_DEFEND"
    elif puck_x_pct > ZONE_OZ_START:
        zone = "OZ_ATTACK"
    else:
        zone = "NZ_TRANSIT"

    return {"zone": zone, "puck_x_pct": round(puck_x_pct, 3)}


def estimate_puck_x(positions: list) -> float:
    """
    퍽 위치 추정: 선수 밀집 영역의 중심
    (실제 퍽 추적 없이 근사)
    """
    if len(positions) <= 2:
        return mean_x(positions)

    # 가장 가까운 3명의 x 중심
    xs = sorted(p[0] for p in positions)
    min_spread = float('inf')
    best_center = xs[len(xs) // 2]

    for i in range(len(xs) - 2):
        spread = xs[i + 2] - xs[i]
        if spread < min_spread:
            min_spread = spread
            best_center = (xs[i] + xs[i + 1] + xs[i + 2]) / 3

    return best_center


# ─── 2. 포메이션 분류 ─────────────────────────────────────────────────

def classify_formation(
    positions: list,
    rink_w: float,
    rink_h: float,
) -> str:
    """
    5명(수비팀) 위치 → 포메이션 분류
    x축 기준 라인 수 + y축 분포로 판단
    """
    if len(positions) < 3:
        return "UNKNOWN"

    # x좌표 기준 라인 분리 (클러스터링)
    xs = sorted(p[0] for p in positions)
    lines = cluster_1d(xs, threshold=rink_w * 0.08)
    n_lines = len(lines)

    # y좌표 분포 (넓게/좁게)
    ys = [p[1] for p in positions]
    y_spread = max(ys) - min(ys) if ys else 0
    y_spread_pct = y_spread / rink_h if rink_h > 0 else 0

    # 포메이션 판정
    if n_lines == 1:
        if y_spread_pct > 0.5:
            return "1-4"       # 한 줄에 넓게
        return "SCRAMBLE"       # 뭉침

    if n_lines == 2:
        sizes = sorted([len(l) for l in lines])
        if sizes == [1, 4]:
            return "1-4"
        if sizes == [2, 3]:
            return "2-3"
        return "2-LINE"

    if n_lines == 3:
        sizes = sorted([len(l) for l in lines])
        if sizes == [1, 1, 3]:
            return "1-1-3"
        if sizes == [1, 2, 2]:
            return "1-2-2"
        if sizes == [1, 1, 2]:  # 4명일 때
            return "1-2-1"
        return "3-LINE"

    if n_lines >= 4:
        return "SPREAD"

    return "UNKNOWN"


def cluster_1d(values: list, threshold: float) -> list:
    """1차원 값 리스트 → 클러스터 분리"""
    if not values:
        return []

    sorted_vals = sorted(values)
    clusters = [[sorted_vals[0]]]

    for v in sorted_vals[1:]:
        if v - clusters[-1][-1] <= threshold:
            clusters[-1].append(v)
        else:
            clusters.append([v])

    return clusters


# ─── 3. 플레이 타입 분류 ──────────────────────────────────────────────

def classify_play_type(
    zone: str,
    prev_zone: str,
    home_positions: list,
    away_positions: list,
    home_speed: float,
    rink_w: float,
) -> str:
    """
    존 상태 + 전환 방향 + 속도 → 플레이 타입 분류

    Returns: BREAKOUT / FORECHECK / CYCLE / RUSH / DUMP / REGROUP / NEUTRAL
    """
    # 존 전환 감지
    transitioning = zone != prev_zone

    if zone == "DZ_DEFEND":
        if transitioning and prev_zone == "NZ_TRANSIT":
            return "REGROUP"     # 뉴트럴→수비 = 재정비
        # 수비존에서 퍽 보유 → 브레이크아웃
        home_mean_x = mean_x(home_positions)
        if home_mean_x < rink_w * ZONE_DZ_END:
            return "BREAKOUT"
        return "DZ_DEFEND"

    elif zone == "OZ_ATTACK":
        if transitioning:
            if home_speed > 5:
                return "RUSH"    # 빠른 속도로 공격존 진입
            return "DUMP"        # 느린 진입 = 덤프
        # 공격존 체류 중
        # 선수들이 원형/순환 배치면 CYCLE
        home_in_oz = [p for p in home_positions if p[0] > rink_w * ZONE_OZ_START]
        if len(home_in_oz) >= 3:
            return "CYCLE"
        return "OZ_ATTACK"

    else:  # NZ_TRANSIT
        if transitioning and prev_zone == "DZ_DEFEND":
            return "BREAKOUT"    # 수비→뉴트럴 = 브레이크아웃 진행 중
        # 뉴트럴존에서 압박 여부
        away_in_nz = [p for p in away_positions
                      if rink_w * ZONE_DZ_END < p[0] < rink_w * ZONE_OZ_START]
        if len(away_in_nz) >= 3:
            return "FORECHECK"   # 상대가 뉴트럴존에 많으면 포체킹 당하는 중
        return "NEUTRAL"


# ─── 4. 수적 상황 ─────────────────────────────────────────────────────

def classify_manpower(home_count: int, away_count: int) -> str:
    """빙판 위 선수 수 → 수적 상황"""
    if home_count == 5 and away_count == 5:
        return "5v5"
    elif home_count == 5 and away_count == 4:
        return "PP"    # 파워플레이
    elif home_count == 4 and away_count == 5:
        return "PK"    # 페널티킬
    elif home_count == 4 and away_count == 4:
        return "4v4"
    elif home_count == 3 and away_count == 3:
        return "3v3"
    else:
        return f"{home_count}v{away_count}"


# ─── 프레임 처리 ──────────────────────────────────────────────────────

def dedupe_tracks_by_jersey(tracks: list) -> list:
    """같은 jersey의 중복 트랙 제거 — jersey당 가장 긴 트랙만 유지"""
    best: dict = {}  # key: "{team}_{jersey}" → track
    no_jersey = []
    for t in tracks:
        jersey = t.get("jersey", "")
        team = t.get("team", "")
        if not jersey or jersey == "?":
            no_jersey.append(t)
            continue
        key = f"{team}_{jersey}"
        if key not in best or len(t["points"]) > len(best[key]["points"]):
            best[key] = t
    return list(best.values()) + no_jersey


def get_frame_data(tracks: list, jersey_map: dict, frame: int, fps: float) -> dict:
    """특정 프레임의 모든 선수 위치 추출"""
    home_positions = []
    away_positions = []
    home_speeds = []

    for track in tracks:
        tid = str(track.get("track_id", ""))
        meta = jersey_map.get(tid, {})
        team = meta.get("team", track.get("team"))
        points = track.get("points", [])

        # 해당 프레임 찾기 (이진 탐색)
        pos = find_frame(points, frame)
        if pos is None:
            continue

        center = bbox_center(pos["bbox"])

        if team == "HOME":
            home_positions.append(center)
            # 속도 계산 (이전 프레임과 비교)
            prev_pos = find_frame(points, frame - 1)
            if prev_pos:
                prev_center = bbox_center(prev_pos["bbox"])
                speed = dist(center, prev_center) * fps
                home_speeds.append(speed)
        else:
            away_positions.append(center)

    return {
        "home_positions": home_positions,
        "away_positions": away_positions,
        "home_count": min(len(home_positions), 6),  # 골리 포함 최대 6
        "away_count": min(len(away_positions), 6),
        "avg_home_speed": sum(home_speeds) / len(home_speeds) if home_speeds else 0,
    }


def find_frame(points: list, target_frame: int) -> dict | None:
    """이진 탐색으로 프레임 찾기 (±1 프레임 허용)"""
    if not points:
        return None

    lo, hi = 0, len(points) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        f = points[mid].get("frame", 0)
        if f == target_frame:
            return points[mid]
        elif f < target_frame:
            lo = mid + 1
        else:
            hi = mid - 1

    # 정확한 프레임이 없으면 가장 가까운 프레임 반환 (±1 범위)
    if lo < len(points):
        frame_lo = points[lo].get("frame", 0)
        if abs(frame_lo - target_frame) <= 1:
            return points[lo]
    
    if hi >= 0:
        frame_hi = points[hi].get("frame", 0)
        if abs(frame_hi - target_frame) <= 1:
            return points[hi]
    
    return None


# ─── 전체 타임라인 생성 ───────────────────────────────────────────────

def generate_tactics_timeline(
    all_tracks: dict,
    sample_interval: int = 4,   # 매 N프레임마다 분석 (기본: 1초 간격)
) -> dict:
    """
    all_tracks → 프레임별 전술 분류 타임라인

    sample_interval: 분석 간격 (프레임 수). fps=4일 때 4 = 1초 간격
    """
    video_stem = all_tracks.get("video_stem", "unknown")
    fps = all_tracks.get("fps", 4)
    frame_count = all_tracks.get("frame_count", 0)
    tracks = all_tracks.get("tracks", [])
    jersey_map = all_tracks.get("jersey_map", {})

    # 팀 이름 → HOME/AWAY 정규화
    # tracks의 team 필드: "Aigis"→HOME, "Lopez"→AWAY (또는 team_a/team_b)
    team_names = set(t.get("team", "") for t in tracks if t.get("team"))
    team_list = sorted([n for n in team_names if n and n not in ("unknown", "")])
    team_home = team_list[0] if len(team_list) >= 1 else "HOME"
    team_away = team_list[1] if len(team_list) >= 2 else "AWAY"
    TEAM_MAP = {
        team_home: "HOME", team_away: "AWAY",
        "team_b": "HOME", "team_a": "AWAY",
        "HOME": "HOME", "AWAY": "AWAY",
    }
    # tracks의 team 필드 정규화
    for t in tracks:
        t["team"] = TEAM_MAP.get(t.get("team", ""), t.get("team", ""))
    # jersey_map도 정규화
    for v in jersey_map.values():
        v["team"] = TEAM_MAP.get(v.get("team", ""), v.get("team", ""))

    # 영상 해상도 추정 (첫 번째 bbox에서)
    rink_w = RINK_W
    rink_h = RINK_H
    for t in tracks:
        pts = t.get("points", [])
        if pts:
            bbox = pts[0].get("bbox", [0, 0, 1920, 1080])
            # bbox에서 영상 크기 추정
            max_x = max(bbox[0], bbox[2]) if len(bbox) == 4 else 1920
            if max_x > 100:
                rink_w = max(rink_w, max_x * 1.2)  # 여유 20%
            break

    # jersey 중복 트랙 제거 — jersey당 가장 긴 트랙만 사용
    tracks = dedupe_tracks_by_jersey(tracks)

    timeline = []
    prev_zone = "NZ_TRANSIT"

    for frame in range(0, frame_count, sample_interval):
        fd = get_frame_data(tracks, jersey_map, frame, fps)

        # 존 상태
        zone_info = classify_zone_state(
            fd["home_positions"], fd["away_positions"], rink_w)
        zone = zone_info["zone"]

        # 포메이션 (수비팀 기준)
        away_formation = classify_formation(
            fd["away_positions"], rink_w, rink_h)
        home_formation = classify_formation(
            fd["home_positions"], rink_w, rink_h)

        # 플레이 타입
        play_type = classify_play_type(
            zone, prev_zone,
            fd["home_positions"], fd["away_positions"],
            fd["avg_home_speed"], rink_w)

        # 수적 상황
        manpower = classify_manpower(fd["home_count"], fd["away_count"])

        timeline.append({
            "frame": frame,
            "time_sec": round(frame / fps, 1),
            "zone": zone,
            "play_type": play_type,
            "home_formation": home_formation,
            "away_formation": away_formation,
            "manpower": manpower,
            "home_count": fd["home_count"],
            "away_count": fd["away_count"],
            "puck_x_pct": zone_info["puck_x_pct"],
        })

        prev_zone = zone

    # ── 요약 통계 ──────────────────────────────────────────────
    total = len(timeline) or 1

    zone_counts = Counter(e["zone"] for e in timeline)
    play_counts = Counter(e["play_type"] for e in timeline)
    home_form_counts = Counter(e["home_formation"] for e in timeline)
    away_form_counts = Counter(e["away_formation"] for e in timeline)
    manpower_counts = Counter(e["manpower"] for e in timeline)

    summary = {
        "zone_pct": {k: round(v / total * 100, 1) for k, v in zone_counts.items()},
        "play_type_pct": {k: round(v / total * 100, 1) for k, v in play_counts.items()},
        "home_formation_pct": {k: round(v / total * 100, 1) for k, v in home_form_counts.items()},
        "away_formation_pct": {k: round(v / total * 100, 1) for k, v in away_form_counts.items()},
        "manpower_pct": {k: round(v / total * 100, 1) for k, v in manpower_counts.items()},
    }

    return {
        "video_stem": video_stem,
        "fps": fps,
        "frame_count": frame_count,
        "sample_interval": sample_interval,
        "total_samples": len(timeline),
        "duration_sec": round(frame_count / fps, 1),
        "summary": summary,
        "timeline": timeline,
    }


# ─── 상대팀 패턴 추출 (스카우팅용) ────────────────────────────────────

def extract_opponent_patterns(tactics_result: dict) -> dict:
    """
    전술 타임라인 → 상대팀 패턴 요약 (스카우팅 리포트용)
    """
    summary = tactics_result.get("summary", {})
    timeline = tactics_result.get("timeline", [])

    # 가장 빈번한 상대 포메이션 (UNKNOWN 제외)
    away_forms = summary.get("away_formation_pct", {})
    # UNKNOWN을 제외한 형성 중 가장 높은 것 선택
    known_forms = {k: v for k, v in away_forms.items() if k != "UNKNOWN"}
    primary_formation = max(known_forms, key=known_forms.get) if known_forms else "UNKNOWN"

    # 존 점유 패턴
    zone_pct = summary.get("zone_pct", {})

    # 포체킹 강도 (NZ에서 FORECHECK 비율)
    play_types = summary.get("play_type_pct", {})
    forecheck_pct = play_types.get("FORECHECK", 0)

    # 브레이크아웃 빈도
    breakout_pct = play_types.get("BREAKOUT", 0)

    # 전환 속도 추정 (존 변경 간격)
    zone_changes = 0
    for i in range(1, len(timeline)):
        if timeline[i]["zone"] != timeline[i-1]["zone"]:
            zone_changes += 1

    transition_rate = round(zone_changes / (len(timeline) or 1) * 100, 1)

    # 약점 감지
    weaknesses = []
    if forecheck_pct < 15:
        weaknesses.append({
            "type": "passive_forecheck",
            "description": "포체킹이 소극적 — 뉴트럴존에서 빠른 전환 공략 가능",
            "severity": "high",
        })
    if zone_pct.get("DZ_DEFEND", 0) > 40:
        weaknesses.append({
            "type": "defensive_heavy",
            "description": "수비존 체류가 많음 — 공격적 포체킹으로 압박 가능",
            "severity": "medium",
        })
    if breakout_pct > 25:
        weaknesses.append({
            "type": "breakout_dependent",
            "description": "브레이크아웃 의존도 높음 — 포체킹으로 턴오버 유도",
            "severity": "high",
        })
    if away_forms.get("SCRAMBLE", 0) > 20:
        weaknesses.append({
            "type": "disorganized",
            "description": "포메이션 유지 못함 — 빠른 패싱 플레이로 혼란 가중",
            "severity": "high",
        })

    return {
        "primary_formation": primary_formation,
        "formation_distribution": away_forms,
        "zone_tendency": zone_pct,
        "forecheck_intensity_pct": forecheck_pct,
        "breakout_frequency_pct": breakout_pct,
        "transition_rate_pct": transition_rate,
        "weaknesses": weaknesses,
        "play_type_distribution": play_types,
    }


# ─── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IceIQ Tactics Classifier")
    parser.add_argument("input", help="all_tracks JSON 경로")
    parser.add_argument("output", nargs="?", help="출력 JSON 경로 (생략 시 콘솔 출력)")
    parser.add_argument("--home", help="홈 팀 jersey 번호 (쉼표 구분, 예: 4,14,25)")
    parser.add_argument("--away", help="어웨이 팀 jersey 번호 (쉼표 구분, 예: 3,8,11)")
    parser.add_argument("--sample-interval", type=int, default=300,
                        help="샘플링 간격 (프레임 수, 기본: 300)")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    with open(input_path) as f:
        all_tracks = json.load(f)

    # --home / --away 인자로 팀 레이블 강제 지정
    if args.home or args.away:
        home_jerseys = set(args.home.split(",")) if args.home else set()
        away_jerseys = set(args.away.split(",")) if args.away else set()
        for track in all_tracks.get("tracks", []):
            jersey = str(track.get("jersey", ""))
            if jersey in home_jerseys:
                track["team"] = "home"
            elif jersey in away_jerseys:
                track["team"] = "away"

    result = generate_tactics_timeline(all_tracks, sample_interval=args.sample_interval)
    patterns = extract_opponent_patterns(result)

    if output_path:
        output = {**result, "opponent_patterns": patterns}
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"✅ 전술 분석 저장: {output_path}")
    else:
        s = result["summary"]
        print(f"\n{'='*60}")
        print(f"🏒 {result['video_stem']} — 전술 분석")
        print(f"   {result['duration_sec']}초 / {result['total_samples']}샘플")
        print(f"{'='*60}")

        print(f"\n📍 존 점유:")
        for z, pct in s.get("zone_pct", {}).items():
            bar = "█" * int(pct / 2)
            print(f"   {z:12s} {pct:5.1f}% {bar}")

        print(f"\n🎯 플레이 타입:")
        for pt, pct in sorted(s.get("play_type_pct", {}).items(),
                              key=lambda x: -x[1]):
            bar = "█" * int(pct / 2)
            print(f"   {pt:15s} {pct:5.1f}% {bar}")

        print(f"\n🏠 홈 포메이션:")
        for fm, pct in sorted(s.get("home_formation_pct", {}).items(),
                              key=lambda x: -x[1])[:3]:
            print(f"   {fm:12s} {pct:5.1f}%")

        print(f"\n✈️  어웨이 포메이션:")
        for fm, pct in sorted(s.get("away_formation_pct", {}).items(),
                              key=lambda x: -x[1])[:3]:
            print(f"   {fm:12s} {pct:5.1f}%")

        print(f"\n⚡ 수적 상황:")
        for mp, pct in s.get("manpower_pct", {}).items():
            print(f"   {mp:8s} {pct:5.1f}%")

        if patterns["weaknesses"]:
            print(f"\n🎯 상대 약점:")
            for w in patterns["weaknesses"]:
                sev = "🔴" if w["severity"] == "high" else "🟡"
                print(f"   {sev} {w['description']}")

        print()
