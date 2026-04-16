"""
IceIQ Player Metrics Engine
추적 좌표 데이터에서 파생 지표를 자동 계산

입력: all_tracks JSON (tracking/{video_stem}/all_tracks)
출력: player_profile.json (선수별 파생 지표)

지표 카테고리:
1. 속도 (Speed) — 평균/최고/스프린트
2. 거리 (Distance) — 총 이동/시프트당
3. 전환 (Transition) — 공수 전환 속도
4. 포지셔닝 (Positioning) — 존 체류/간격
5. 체력 (Stamina) — 시프트 후반 속도 감소
"""

import json
import math
import os
from typing import Any

# ─── 좌표 유틸 ─────────────────────────────────────────────────────────

def dist(p1: tuple, p2: tuple) -> float:
    """두 점 사이 유클리드 거리 (px)"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def detect_bbox_format(bbox: list) -> str:
    """
    bbox 형식 자동 감지
    [x1,y1,x2,y2]: x2 > x1, y2 > y1 (좌상단→우하단)
    [cx,cy,w,h]:   w,h가 cx,cy보다 훨씬 작음
    """
    if len(bbox) != 4:
        return "unknown"
    # x2 > x1 이고 y2 > y1 이면 [x1,y1,x2,y2]
    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
        # 추가 확인: x2-x1이 합리적인 선수 크기 (10~300px)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        if 5 < w < 500 and 5 < h < 500:
            return "xyxy"
    # w,h가 작으면 [cx,cy,w,h]
    if bbox[2] < bbox[0] and bbox[3] < bbox[1]:
        return "cxcywh"
    # w,h가 절대적으로 작으면 [cx,cy,w,h]
    if bbox[2] < 300 and bbox[3] < 300 and bbox[0] > 50:
        return "cxcywh"
    return "xyxy"  # 기본값


# 전역 캐시: 한번 감지하면 재사용
_bbox_format_cache = None


def _normalize_bbox(bbox: list) -> tuple:
    """bbox를 항상 (x1, y1, x2, y2) 형식으로 변환"""
    global _bbox_format_cache
    if _bbox_format_cache is None:
        _bbox_format_cache = detect_bbox_format(bbox)

    if _bbox_format_cache == "cxcywh":
        cx, cy, w, h = bbox
        return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
    else:
        return tuple(bbox)


def bbox_center(bbox: list) -> tuple:
    """bbox → 중심점 (cx, cy). 형식 자동 감지."""
    x1, y1, x2, y2 = _normalize_bbox(bbox)
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_bottom_center(bbox: list) -> tuple:
    """bbox → 발 위치 (하단 중심). 형식 자동 감지."""
    x1, y1, x2, y2 = _normalize_bbox(bbox)
    return ((x1 + x2) / 2, y2)


# ─── 존 판정 ──────────────────────────────────────────────────────────

# 호모그래피 적용된 링크 좌표 기준 (1040 x 520)
RINK_W = 1040
RINK_H = 520
BLUE_LINE_LEFT = RINK_W * 0.31    # ~322
BLUE_LINE_RIGHT = RINK_W * 0.69   # ~718
HIGH_SLOT_Y_MIN = RINK_H * 0.30   # ~156
HIGH_SLOT_Y_MAX = RINK_H * 0.70   # ~364
HIGH_SLOT_X_MIN = RINK_W * 0.75   # ~780 (공격존 기준)


def classify_zone(x: float, rink_w: float = RINK_W) -> str:
    """x좌표 → 존 분류 (공격 방향 = 오른쪽 기준)"""
    bl_left = rink_w * 0.31
    bl_right = rink_w * 0.69
    if x < bl_left:
        return "DZ"   # 수비존
    elif x > bl_right:
        return "OZ"   # 공격존
    else:
        return "NZ"   # 뉴트럴존


def is_high_slot(x: float, y: float) -> bool:
    """하이슬롯 영역 여부 (공격존 골대 앞)"""
    return (x > HIGH_SLOT_X_MIN and
            HIGH_SLOT_Y_MIN < y < HIGH_SLOT_Y_MAX)


# ─── 시프트 분리 ──────────────────────────────────────────────────────

def extract_shifts(frames: list, fps: float = 4.0, gap_sec: float = 10.0) -> list:
    """
    프레임 번호 리스트 → 시프트 리스트
    gap_sec 이상 빈 구간이 있으면 새 시프트로 분리
    """
    if not frames:
        return []

    gap_frames = int(gap_sec * fps)
    sorted_frames = sorted(frames)

    shifts = []
    shift_start = sorted_frames[0]
    prev = sorted_frames[0]

    for f in sorted_frames[1:]:
        if f - prev > gap_frames:
            shifts.append({
                "start_frame": shift_start,
                "end_frame": prev,
                "start_time": round(shift_start / fps, 1),
                "end_time": round(prev / fps, 1),
                "duration": round((prev - shift_start) / fps, 1),
            })
            shift_start = f
        prev = f

    # 마지막 시프트
    shifts.append({
        "start_frame": shift_start,
        "end_frame": prev,
        "start_time": round(shift_start / fps, 1),
        "end_time": round(prev / fps, 1),
        "duration": round((prev - shift_start) / fps, 1),
    })

    # shift_number 부여
    for i, s in enumerate(shifts):
        s["shift_number"] = i + 1

    return shifts


# ─── 핵심: 선수 지표 계산 ──────────────────────────────────────────────

def compute_player_metrics(
    track: dict,
    fps: float = 4.0,
    rink_w: float = RINK_W,
    rink_h: float = RINK_H,
    use_homography: bool = False,
) -> dict:
    """
    단일 선수의 추적 데이터 → 파생 지표 계산

    track: {
        "track_id": int,
        "jersey": str,
        "team": str,
        "points": [{"frame": int, "bbox": [x1,y1,x2,y2]}, ...]
    }
    """
    points = track.get("points", [])
    if len(points) < 2:
        return _empty_metrics(track)

    track_id = track.get("track_id")
    jersey = track.get("jersey", "?")
    team = track.get("team", "HOME")

    # 정렬
    points = sorted(points, key=lambda p: p["frame"])

    # 프레임 번호 리스트
    frames = [p["frame"] for p in points]

    # 위치 추출 (발 위치 사용)
    positions = [bbox_bottom_center(p["bbox"]) for p in points]

    # ── 1. 속도 지표 ─────────────────────────────────────────────
    speeds = []
    for i in range(1, len(positions)):
        d = dist(positions[i-1], positions[i])
        dt_frames = frames[i] - frames[i-1]
        if dt_frames > 0:
            speed_px_per_frame = d / dt_frames
            speeds.append(speed_px_per_frame)

    # 이상치 필터: 약 40km/h 상한 (6.5 px/frame @ 60fps ≈ 40km/h)
    max_reasonable_speed = 6.5
    speeds = [s for s in speeds if s < max_reasonable_speed]

    avg_speed = sum(speeds) / len(speeds) if speeds else 0
    max_speed = max(speeds) if speeds else 0
    # 스프린트: 상위 20% 속도 이상인 구간
    if speeds:
        sprint_threshold = sorted(speeds)[int(len(speeds) * 0.80)]
        sprint_count = sum(1 for s in speeds if s >= sprint_threshold)
    else:
        sprint_threshold = 0
        sprint_count = 0

    # ── 2. 거리 지표 ─────────────────────────────────────────────
    total_distance = sum(
        dist(positions[i-1], positions[i])
        for i in range(1, len(positions))
    )

    # 시프트 분리
    shifts = extract_shifts(frames, fps)

    # 시프트별 거리
    shift_distances = []
    for shift in shifts:
        sf, ef = shift["start_frame"], shift["end_frame"]
        shift_pts = [(positions[i], frames[i]) for i in range(len(frames))
                     if sf <= frames[i] <= ef]
        if len(shift_pts) >= 2:
            sd = sum(dist(shift_pts[j-1][0], shift_pts[j][0])
                     for j in range(1, len(shift_pts)))
            shift_distances.append(sd)
        else:
            shift_distances.append(0)

    avg_shift_distance = (sum(shift_distances) / len(shift_distances)
                          if shift_distances else 0)

    # ── 3. 전환 지표 ─────────────────────────────────────────────
    # 존 전환 감지: 프레임별 존 분류 → 전환 이벤트 추출
    zone_sequence = [classify_zone(p[0], rink_w) for p in positions]

    oz_to_dz_times = []   # 공격→수비 복귀
    dz_to_oz_times = []   # 수비→공격 전환

    last_zone = zone_sequence[0]
    last_change_frame = frames[0]

    for i in range(1, len(zone_sequence)):
        if zone_sequence[i] != last_zone:
            transition_frames = frames[i] - last_change_frame
            transition_sec = transition_frames / fps

            if last_zone == "OZ" and zone_sequence[i] == "DZ":
                oz_to_dz_times.append(transition_sec)
            elif last_zone == "DZ" and zone_sequence[i] == "OZ":
                dz_to_oz_times.append(transition_sec)

            last_zone = zone_sequence[i]
            last_change_frame = frames[i]

    avg_backcheck_sec = (sum(oz_to_dz_times) / len(oz_to_dz_times)
                         if oz_to_dz_times else 0)
    avg_rush_sec = (sum(dz_to_oz_times) / len(dz_to_oz_times)
                    if dz_to_oz_times else 0)

    # ── 4. 포지셔닝 지표 ──────────────────────────────────────────
    zone_counts = {"OZ": 0, "NZ": 0, "DZ": 0}
    high_slot_frames = 0

    for i, z in enumerate(zone_sequence):
        zone_counts[z] += 1
        if is_high_slot(positions[i][0], positions[i][1]):
            high_slot_frames += 1

    total_frames = len(zone_sequence)
    zone_pct = {k: round(v / total_frames * 100, 1) for k, v in zone_counts.items()}
    high_slot_pct = round(high_slot_frames / total_frames * 100, 1)

    # 뉴트럴존 체류 시간 (초)
    nz_time_sec = round(zone_counts["NZ"] / fps, 1)

    # ── 5. 체력 지표 (시프트 전/후반 속도 비교) ──────────────────
    stamina_scores = []
    for shift in shifts:
        sf, ef = shift["start_frame"], shift["end_frame"]
        shift_speeds = [
            speeds[i-1] for i in range(1, len(frames))
            if sf <= frames[i] <= ef and i-1 < len(speeds)
        ]
        if len(shift_speeds) >= 4:
            mid = len(shift_speeds) // 2
            first_half_avg = sum(shift_speeds[:mid]) / mid
            second_half_avg = sum(shift_speeds[mid:]) / (len(shift_speeds) - mid)
            if first_half_avg > 0:
                decay = round((second_half_avg - first_half_avg) / first_half_avg * 100, 1)
                stamina_scores.append(decay)

    avg_stamina_decay = (sum(stamina_scores) / len(stamina_scores)
                         if stamina_scores else 0)

    # ── 프로필 조합 ────────────────────────────────────────────────
    # 자동 포지션 분류
    if zone_pct.get("DZ", 0) > 40:
        auto_position = "D"
    elif zone_pct.get("OZ", 0) > 35:
        auto_position = "F"
    else:
        auto_position = "F"  # 기본값

    # 플레이 스타일 태그
    style_tags = []
    if avg_speed > 0:
        if sprint_count > len(speeds) * 0.25:
            style_tags.append("explosive")     # 폭발형
        if abs(avg_stamina_decay) < 10:
            style_tags.append("endurance")     # 지구력형
    if zone_pct.get("NZ", 0) > 35:
        style_tags.append("transition")        # 전환 플레이어
    if high_slot_pct > 15:
        style_tags.append("slot_presence")     # 슬롯 침투형
    if avg_backcheck_sec > 0 and avg_backcheck_sec < 5:
        style_tags.append("responsible_def")   # 책임감 있는 수비

    if not style_tags:
        style_tags.append("balanced")

    # 총 아이스타임
    total_ice_time_sec = sum(s["duration"] for s in shifts)

    return {
        "track_id": track_id,
        "jersey": jersey,
        "team": team,
        "total_frames": total_frames,
        "total_ice_time_sec": round(total_ice_time_sec, 1),
        "total_shifts": len(shifts),
        "auto_position": auto_position,
        "style_tags": style_tags,

        "speed": {
            "avg_px_per_frame": round(avg_speed, 2),
            "max_px_per_frame": round(max_speed, 2),
            "sprint_count": sprint_count,
            "sprint_threshold": round(sprint_threshold, 2),
        },

        "distance": {
            "total_px": round(total_distance, 1),
            "avg_per_shift_px": round(avg_shift_distance, 1),
            "shift_distances": [round(d, 1) for d in shift_distances],
        },

        "transition": {
            "avg_backcheck_sec": round(avg_backcheck_sec, 1),
            "avg_rush_sec": round(avg_rush_sec, 1),
            "backcheck_count": len(oz_to_dz_times),
            "rush_count": len(dz_to_oz_times),
        },

        "positioning": {
            "zone_pct": zone_pct,
            "high_slot_pct": high_slot_pct,
            "nz_time_sec": nz_time_sec,
        },

        "stamina": {
            "avg_decay_pct": round(avg_stamina_decay, 1),
            "per_shift_decay": [round(d, 1) for d in stamina_scores],
        },

        "shifts": shifts,
    }


def _empty_metrics(track: dict) -> dict:
    """데이터 부족 시 빈 지표"""
    return {
        "track_id": track.get("track_id"),
        "jersey": track.get("jersey", "?"),
        "team": track.get("team", "HOME"),
        "total_frames": 0,
        "total_ice_time_sec": 0,
        "total_shifts": 0,
        "auto_position": "F",
        "style_tags": ["unknown"],
        "speed": {"avg_px_per_frame": 0, "max_px_per_frame": 0,
                  "sprint_count": 0, "sprint_threshold": 0},
        "distance": {"total_px": 0, "avg_per_shift_px": 0, "shift_distances": []},
        "transition": {"avg_backcheck_sec": 0, "avg_rush_sec": 0,
                       "backcheck_count": 0, "rush_count": 0},
        "positioning": {"zone_pct": {"OZ": 0, "NZ": 0, "DZ": 0},
                        "high_slot_pct": 0, "nz_time_sec": 0},
        "stamina": {"avg_decay_pct": 0, "per_shift_decay": []},
        "shifts": [],
    }


# ─── 전체 경기 처리 ───────────────────────────────────────────────────

def compute_game_metrics(all_tracks: dict, fps: float = None) -> dict:
    """
    all_tracks 응답 전체 → 모든 선수 지표 계산

    all_tracks: {
        "video_stem": str,
        "fps": float,
        "frame_count": int,
        "tracks": [...],
        "jersey_map": {...}
    }
    """
    video_stem = all_tracks.get("video_stem", "unknown")
    fps = fps or all_tracks.get("fps", 4)
    frame_count = all_tracks.get("frame_count", 0)
    tracks = all_tracks.get("tracks", [])
    jersey_map = all_tracks.get("jersey_map", {})

    # bbox 형식 캐시 리셋
    global _bbox_format_cache
    _bbox_format_cache = None

    # 첫 번째 트랙의 bbox로 형식 감지 & 로그
    for t in tracks:
        pts = t.get("points", [])
        if pts:
            sample_bbox = pts[0].get("bbox", [])
            fmt = detect_bbox_format(sample_bbox)
            print(f"📐 bbox 형식 감지: {fmt} (샘플: {sample_bbox})")
            break

    players = []
    for track in tracks:
        # jersey_map에서 정보 보강
        tid = str(track.get("track_id", ""))
        if tid in jersey_map:
            track["jersey"] = jersey_map[tid].get("jersey", track.get("jersey"))
            track["team"] = jersey_map[tid].get("team", track.get("team"))

        # jersey가 없으면 스킵
        if not track.get("jersey"):
            continue

        metrics = compute_player_metrics(track, fps)
        players.append(metrics)

    # jersey 중복 시 프레임 수 기준 상위만 유지
    seen = {}
    for p in players:
        # jersey가 "?"이면 track_id로 구분 (jersey 없는 경우 중복 방지)
        jersey = p.get("jersey", "?")
        if jersey in ("?", "", None):
            key = f"track_{p.get('track_id', id(p))}_{p['team']}"
        else:
            key = f"{jersey}_{p['team']}"
        if key not in seen or p["total_frames"] > seen[key]["total_frames"]:
            seen[key] = p

    unique_players = sorted(seen.values(),
                            key=lambda x: x["total_ice_time_sec"],
                            reverse=True)

    # px/frame → km/h 변환 (homography 기반 스케일, 60fps 기준)
    # 1 px/frame @ 60fps ≈ 6.182 km/h (configs/homography_4point.json 기준)
    PX_PER_FRAME_TO_KMH = fps * 0.02862 * 3.6  # m/px * fps * 3.6
    PX_TO_KM = 0.02862 / 1000

    for p in unique_players:
        spd = p.get("speed", {})
        spd["avg_kmh"] = round(spd.get("avg_px_per_frame", 0) * PX_PER_FRAME_TO_KMH, 1)
        spd["max_kmh"] = round(spd.get("max_px_per_frame", 0) * PX_PER_FRAME_TO_KMH, 1)
        dist_d = p.get("distance", {})
        dist_d["total_km"] = round(dist_d.get("total_px", 0) * PX_TO_KM, 3)
        dist_d["avg_per_shift_km"] = round(dist_d.get("avg_per_shift_px", 0) * PX_TO_KM, 3)

    return {
        "video_stem": video_stem,
        "fps": fps,
        "frame_count": frame_count,
        "total_players": len(unique_players),
        "players": unique_players,
    }


# ─── 선수 프로필 요약 (앱 표시용) ─────────────────────────────────────

def summarize_profile(metrics: dict) -> dict:
    """상세 지표 → 앱에서 보여줄 간단한 프로필"""
    return {
        "jersey": metrics["jersey"],
        "team": metrics["team"],
        "position": metrics["auto_position"],
        "style": metrics["style_tags"],
        "ice_time_min": round(metrics["total_ice_time_sec"] / 60, 1),
        "shifts": metrics["total_shifts"],
        "avg_speed": metrics["speed"].get("avg_kmh", metrics["speed"]["avg_px_per_frame"]),
        "max_speed": metrics["speed"].get("max_kmh", metrics["speed"]["max_px_per_frame"]),
        "sprints": metrics["speed"]["sprint_count"],
        "total_distance": metrics["distance"].get("total_km", metrics["distance"]["total_px"]),
        "backcheck_sec": metrics["transition"]["avg_backcheck_sec"],
        "zone_pct": metrics["positioning"]["zone_pct"],
        "high_slot_pct": metrics["positioning"]["high_slot_pct"],
        "stamina_decay_pct": metrics["stamina"]["avg_decay_pct"],
    }


# ─── CLI 실행 ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python player_metrics.py <all_tracks.json> [output.json]")
        print("  all_tracks.json: GET /tracking/{stem}/all_tracks 응답")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    with open(input_path) as f:
        all_tracks = json.load(f)

    result = compute_game_metrics(all_tracks)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ {result['total_players']}명 지표 저장: {output_path}")
    else:
        # 요약 출력
        print(f"\n{'='*60}")
        print(f"📊 {result['video_stem']} — {result['total_players']}명")
        print(f"{'='*60}")
        for p in result["players"]:
            prof = summarize_profile(p)
            print(f"\n#{prof['jersey']} ({prof['team']}) — {prof['position']} "
                  f"[{', '.join(prof['style'])}]")
            print(f"  ⏱ {prof['ice_time_min']}분 · {prof['shifts']}시프트")
            print(f"  🏃 속도: avg {prof['avg_speed']:.1f}km/h / max {prof['max_speed']:.1f}km/h "
                  f"/ 스프린트 {prof['sprints']}회")
            print(f"  📏 거리: {prof['total_distance']:.3f}km")
            print(f"  🔄 복귀: {prof['backcheck_sec']}초")
            print(f"  🗺 존: OZ {prof['zone_pct']['OZ']}% / "
                  f"NZ {prof['zone_pct']['NZ']}% / "
                  f"DZ {prof['zone_pct']['DZ']}%")
            print(f"  🎯 하이슬롯: {prof['high_slot_pct']}%")
            print(f"  💪 체력감소: {prof['stamina_decay_pct']}%")


# ─── 연령별 성능 기준 ──────────────────────────────────────────────────────

def get_age_category(birthdate: str) -> tuple[str, int]:
    """
    생년월일 → (카테고리, 나이)
    
    Args:
        birthdate: "YYYY-MM-DD" 형식
    
    Returns:
        (category, age) tuple
        category: "U-10", "U-12", "U-14", "U-16", "U-18"
    """
    from datetime import date
    
    try:
        birth_date = date.fromisoformat(birthdate)
        age = (date.today() - birth_date).days // 365
        
        if age < 10:
            category = "U-10"
        elif age < 12:
            category = "U-12"
        elif age < 14:
            category = "U-14"
        elif age < 16:
            category = "U-16"
        else:
            category = "U-18"
        
        return category, age
    except (ValueError, TypeError):
        return "Unknown", None


def get_performance_benchmarks(category: str, position: str) -> dict:
    """
    연령대별, 포지션별 기대 성능 기준 (세분화)
    
    포지션:
    - C (Center): 페이스오프, 속도 높음, 거리 최대
    - LW (Left Wing): 좌측 슈팅, 속도 높음
    - RW (Right Wing): 우측 슈팅, 속도 높음
    - LD (Left Defense): 백체킹, 거리 중간, 커버리지
    - RD (Right Defense): 백체킹, 거리 중간, 커버리지
    - G (Goaltender): 세이브 반응속도, 횡이동 범위
    
    Args:
        category: "U-10", "U-12", "U-14", "U-16", "U-18"
        position: "C", "LW", "RW", "LD", "RD", "G"
    
    Returns:
        {
            "avg_speed_kmh": float,
            "max_speed_kmh": float,
            "distance_km": float,
            "sprint_count": int,
            "zone_tendency": str,  # OZ, NZ, DZ 등
            "role": str,
        }
    """
    benchmarks = {
        "U-10": {
            "C": {
                "avg_speed_kmh": 13.5,
                "max_speed_kmh": 23.0,
                "distance_km": 3.8,
                "sprint_count": 9,
                "zone_tendency": "OZ",
                "role": "페이스오프 & 중원 장악",
            },
            "LW": {
                "avg_speed_kmh": 13.2,
                "max_speed_kmh": 22.5,
                "distance_km": 3.6,
                "sprint_count": 8,
                "zone_tendency": "OZ",
                "role": "좌측 공격 & 슈팅",
            },
            "RW": {
                "avg_speed_kmh": 13.2,
                "max_speed_kmh": 22.5,
                "distance_km": 3.6,
                "sprint_count": 8,
                "zone_tendency": "OZ",
                "role": "우측 공격 & 슈팅",
            },
            "LD": {
                "avg_speed_kmh": 12.6,
                "max_speed_kmh": 21.0,
                "distance_km": 3.9,
                "sprint_count": 6,
                "zone_tendency": "DZ",
                "role": "좌측 수비 & 백체킹",
            },
            "RD": {
                "avg_speed_kmh": 12.6,
                "max_speed_kmh": 21.0,
                "distance_km": 3.9,
                "sprint_count": 6,
                "zone_tendency": "DZ",
                "role": "우측 수비 & 백체킹",
            },
            "G": {
                "avg_speed_kmh": 8.0,
                "max_speed_kmh": 12.0,
                "distance_km": 0.5,
                "sprint_count": 30,
                "zone_tendency": "DZ",
                "role": "세이브 & 요크아웃",
            },
        },
        "U-12": {
            "C": {
                "avg_speed_kmh": 15.0,
                "max_speed_kmh": 25.0,
                "distance_km": 4.5,
                "sprint_count": 11,
                "zone_tendency": "OZ",
                "role": "페이스오프 & 중원 장악",
            },
            "LW": {
                "avg_speed_kmh": 14.7,
                "max_speed_kmh": 24.5,
                "distance_km": 4.2,
                "sprint_count": 10,
                "zone_tendency": "OZ",
                "role": "좌측 공격 & 슈팅",
            },
            "RW": {
                "avg_speed_kmh": 14.7,
                "max_speed_kmh": 24.5,
                "distance_km": 4.2,
                "sprint_count": 10,
                "zone_tendency": "OZ",
                "role": "우측 공격 & 슈팅",
            },
            "LD": {
                "avg_speed_kmh": 14.0,
                "max_speed_kmh": 23.0,
                "distance_km": 4.6,
                "sprint_count": 8,
                "zone_tendency": "DZ",
                "role": "좌측 수비 & 백체킹",
            },
            "RD": {
                "avg_speed_kmh": 14.0,
                "max_speed_kmh": 23.0,
                "distance_km": 4.6,
                "sprint_count": 8,
                "zone_tendency": "DZ",
                "role": "우측 수비 & 백체킹",
            },
            "G": {
                "avg_speed_kmh": 8.5,
                "max_speed_kmh": 13.0,
                "distance_km": 0.6,
                "sprint_count": 35,
                "zone_tendency": "DZ",
                "role": "세이브 & 요크아웃",
            },
        },
        "U-14": {
            "C": {
                "avg_speed_kmh": 16.0,
                "max_speed_kmh": 27.0,
                "distance_km": 5.0,
                "sprint_count": 13,
                "zone_tendency": "OZ",
                "role": "페이스오프 & 중원 장악",
            },
            "LW": {
                "avg_speed_kmh": 15.7,
                "max_speed_kmh": 26.5,
                "distance_km": 4.8,
                "sprint_count": 12,
                "zone_tendency": "OZ",
                "role": "좌측 공격 & 슈팅",
            },
            "RW": {
                "avg_speed_kmh": 15.7,
                "max_speed_kmh": 26.5,
                "distance_km": 4.8,
                "sprint_count": 12,
                "zone_tendency": "OZ",
                "role": "우측 공격 & 슈팅",
            },
            "LD": {
                "avg_speed_kmh": 15.0,
                "max_speed_kmh": 25.0,
                "distance_km": 5.2,
                "sprint_count": 10,
                "zone_tendency": "DZ",
                "role": "좌측 수비 & 백체킹",
            },
            "RD": {
                "avg_speed_kmh": 15.0,
                "max_speed_kmh": 25.0,
                "distance_km": 5.2,
                "sprint_count": 10,
                "zone_tendency": "DZ",
                "role": "우측 수비 & 백체킹",
            },
            "G": {
                "avg_speed_kmh": 9.0,
                "max_speed_kmh": 14.0,
                "distance_km": 0.7,
                "sprint_count": 40,
                "zone_tendency": "DZ",
                "role": "세이브 & 요크아웃",
            },
        },
        "U-16": {
            "C": {
                "avg_speed_kmh": 17.0,
                "max_speed_kmh": 29.0,
                "distance_km": 5.5,
                "sprint_count": 15,
                "zone_tendency": "OZ",
                "role": "페이스오프 & 중원 장악",
            },
            "LW": {
                "avg_speed_kmh": 16.7,
                "max_speed_kmh": 28.5,
                "distance_km": 5.3,
                "sprint_count": 14,
                "zone_tendency": "OZ",
                "role": "좌측 공격 & 슈팅",
            },
            "RW": {
                "avg_speed_kmh": 16.7,
                "max_speed_kmh": 28.5,
                "distance_km": 5.3,
                "sprint_count": 14,
                "zone_tendency": "OZ",
                "role": "우측 공격 & 슈팅",
            },
            "LD": {
                "avg_speed_kmh": 16.0,
                "max_speed_kmh": 27.0,
                "distance_km": 5.7,
                "sprint_count": 12,
                "zone_tendency": "DZ",
                "role": "좌측 수비 & 백체킹",
            },
            "RD": {
                "avg_speed_kmh": 16.0,
                "max_speed_kmh": 27.0,
                "distance_km": 5.7,
                "sprint_count": 12,
                "zone_tendency": "DZ",
                "role": "우측 수비 & 백체킹",
            },
            "G": {
                "avg_speed_kmh": 9.5,
                "max_speed_kmh": 15.0,
                "distance_km": 0.8,
                "sprint_count": 45,
                "zone_tendency": "DZ",
                "role": "세이브 & 요크아웃",
            },
        },
        "U-18": {
            "C": {
                "avg_speed_kmh": 18.0,
                "max_speed_kmh": 31.0,
                "distance_km": 5.8,
                "sprint_count": 17,
                "zone_tendency": "OZ",
                "role": "페이스오프 & 중원 장악",
            },
            "LW": {
                "avg_speed_kmh": 17.7,
                "max_speed_kmh": 30.5,
                "distance_km": 5.6,
                "sprint_count": 16,
                "zone_tendency": "OZ",
                "role": "좌측 공격 & 슈팅",
            },
            "RW": {
                "avg_speed_kmh": 17.7,
                "max_speed_kmh": 30.5,
                "distance_km": 5.6,
                "sprint_count": 16,
                "zone_tendency": "OZ",
                "role": "우측 공격 & 슈팅",
            },
            "LD": {
                "avg_speed_kmh": 17.0,
                "max_speed_kmh": 29.0,
                "distance_km": 6.0,
                "sprint_count": 14,
                "zone_tendency": "DZ",
                "role": "좌측 수비 & 백체킹",
            },
            "RD": {
                "avg_speed_kmh": 17.0,
                "max_speed_kmh": 29.0,
                "distance_km": 6.0,
                "sprint_count": 14,
                "zone_tendency": "DZ",
                "role": "우측 수비 & 백체킹",
            },
            "G": {
                "avg_speed_kmh": 10.0,
                "max_speed_kmh": 16.0,
                "distance_km": 0.9,
                "sprint_count": 50,
                "zone_tendency": "DZ",
                "role": "세이브 & 요크아웃",
            },
        },
    }
    
    # 포지션 매핑 (기존 F/D → 새 포지션)
    position_map = {
        "F": "C",  # 기본값 (실제로는 C/LW/RW 중 하나)
        "D": "LD",  # 기본값 (실제로는 LD/RD 중 하나)
        "G": "G",
    }
    
    # 새 포지션이 없으면 매핑
    if position not in benchmarks.get(category, {}) and position in position_map:
        position = position_map[position]
    
    # 기본값
    if category not in benchmarks:
        category = "U-16"
    if position not in benchmarks[category]:
        position = "C"
    
    return benchmarks[category][position]


def evaluate_performance(profile: dict, benchmarks: dict) -> dict:
    """
    선수 프로필 vs 기대 기준 비교
    
    Args:
        profile: 선수 지표 딕셔너리
        benchmarks: get_performance_benchmarks() 반환값
    
    Returns:
        {
            "speed_rating": float,      # -2.0 ~ 2.0 (평균 vs 기대)
            "distance_rating": float,
            "intensity_rating": float,  # 스프린트 회수 vs 기대
            "overall": float,           # 종합 평가
        }
    """
    avg_speed = profile.get("avg_speed", 0)
    total_distance = profile.get("total_distance", 0)
    sprint_count = profile.get("sprints", 0)
    
    # 1. 속도 평가 (-2.0 ~ 2.0)
    bench_speed = benchmarks.get("avg_speed_kmh", 15.0)
    speed_rating = (avg_speed - bench_speed) / bench_speed * 2.0
    speed_rating = max(-2.0, min(2.0, speed_rating))
    
    # 2. 거리 평가
    bench_distance = benchmarks.get("distance_km", 4.5)
    distance_rating = (total_distance - bench_distance) / bench_distance * 2.0
    distance_rating = max(-2.0, min(2.0, distance_rating))
    
    # 3. 강도 평가 (스프린트)
    bench_sprint = benchmarks.get("sprint_count", 10)
    intensity_rating = (sprint_count - bench_sprint) / max(1, bench_sprint) * 2.0
    intensity_rating = max(-2.0, min(2.0, intensity_rating))
    
    # 4. 종합
    overall = (speed_rating + distance_rating + intensity_rating) / 3.0
    
    return {
        "speed_rating": round(speed_rating, 2),
        "distance_rating": round(distance_rating, 2),
        "intensity_rating": round(intensity_rating, 2),
        "overall": round(overall, 2),
    }
