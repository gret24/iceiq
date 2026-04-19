"""
아이스하키 링크 좌표 및 규격 프리셋.

좌표계 규약:
- 원점: 링크 정중앙
- X축: 링크 길이 방향 (보드에서 보드, -length/2 ~ +length/2)
- Y축: 링크 폭 방향 (사이드보드에서 사이드보드, -width/2 ~ +width/2)
- 단위: 미터(m)

IIHF 공식 규격 기반 (2021년 기준):
- 링크 크기: 60m × 30m
- 블루라인: 중앙에서 ±8.83m
- 골라인: 보드에서 4m (중앙에서 ±26m)
- 페이스오프 서클 반지름: 4.5m
- 엔드존 페이스오프: 골라인에서 6.7m 안쪽
- 뉴트럴존 페이스오프: 블루라인에서 1.5m 바깥
- 엔드존 페이스오프 Y축 위치: 중앙에서 ±7m
- 골 크리즈 반지름: 1.83m

호모그래피 계산의 기준점으로 사용.
"""

from typing import Final

# ============================================================================
# 링크 전체 치수
# ============================================================================

RINK_LENGTH_M: Final[float] = 60.0
RINK_WIDTH_M: Final[float] = 30.0
CORNER_RADIUS_M: Final[float] = 8.5  # IIHF 권장 7.0~8.5m 중 상한

# 경계 좌표
RINK_X_MIN: Final[float] = -RINK_LENGTH_M / 2  # -30.0
RINK_X_MAX: Final[float] = +RINK_LENGTH_M / 2  # +30.0
RINK_Y_MIN: Final[float] = -RINK_WIDTH_M / 2   # -15.0
RINK_Y_MAX: Final[float] = +RINK_WIDTH_M / 2   # +15.0


# ============================================================================
# 라인 위치 (X 좌표)
# ============================================================================

GOAL_LINE_FROM_BOARD_M: Final[float] = 4.0
BLUE_LINE_FROM_CENTER_M: Final[float] = 8.83

GOAL_LINE_X_LEFT: Final[float] = RINK_X_MIN + GOAL_LINE_FROM_BOARD_M   # -26.0
GOAL_LINE_X_RIGHT: Final[float] = RINK_X_MAX - GOAL_LINE_FROM_BOARD_M  # +26.0

BLUE_LINE_X_LEFT: Final[float] = -BLUE_LINE_FROM_CENTER_M   # -8.83
BLUE_LINE_X_RIGHT: Final[float] = +BLUE_LINE_FROM_CENTER_M  # +8.83

CENTER_LINE_X: Final[float] = 0.0


# ============================================================================
# 페이스오프 관련 치수
# ============================================================================

FACEOFF_CIRCLE_RADIUS_M: Final[float] = 4.5
FACEOFF_Y_FROM_CENTER_M: Final[float] = 7.0  # 엔드존·뉴트럴존 공통

# 엔드존 페이스오프 (골라인에서 6.7m 안쪽)
ENDZONE_FACEOFF_FROM_GOAL_LINE_M: Final[float] = 6.7
ENDZONE_FACEOFF_X_LEFT: Final[float] = GOAL_LINE_X_LEFT + ENDZONE_FACEOFF_FROM_GOAL_LINE_M   # -19.3
ENDZONE_FACEOFF_X_RIGHT: Final[float] = GOAL_LINE_X_RIGHT - ENDZONE_FACEOFF_FROM_GOAL_LINE_M  # +19.3

# 뉴트럴존 페이스오프 (블루라인에서 1.5m 바깥)
NEUTRAL_FACEOFF_FROM_BLUE_LINE_M: Final[float] = 1.5
NEUTRAL_FACEOFF_X_LEFT: Final[float] = BLUE_LINE_X_LEFT - NEUTRAL_FACEOFF_FROM_BLUE_LINE_M   # -10.33
NEUTRAL_FACEOFF_X_RIGHT: Final[float] = BLUE_LINE_X_RIGHT + NEUTRAL_FACEOFF_FROM_BLUE_LINE_M  # +10.33


# ============================================================================
# 골·크리즈
# ============================================================================

GOAL_WIDTH_M: Final[float] = 1.83  # 6피트
GOAL_CREASE_RADIUS_M: Final[float] = 1.83


# ============================================================================
# 표준 기준점 (호모그래피 계산용)
# ============================================================================
#
# 영상에서 이 중 4점 이상을 클릭해 태깅하면 호모그래피 행렬 계산 가능.
# 이름 규칙: <구역>_<세부위치>
# - 구역: center / blue / goal / corner / faceoff
# - 세부: top/bottom, left/right 등
#
# Y 양수 = 위쪽(한 사이드), Y 음수 = 아래쪽(반대 사이드)
# 편의상 Y+ 방향을 "top", Y- 방향을 "bottom"으로 통일.

STANDARD_POINTS: Final[dict[str, tuple[float, float]]] = {
    # --- 중앙 ---
    "center_faceoff": (0.0, 0.0),

    # --- 센터라인 ---
    "center_line_top": (0.0, RINK_Y_MAX),
    "center_line_bottom": (0.0, RINK_Y_MIN),

    # --- 블루라인 교차점 (보드 만남) ---
    "blue_line_left_top": (BLUE_LINE_X_LEFT, RINK_Y_MAX),
    "blue_line_left_bottom": (BLUE_LINE_X_LEFT, RINK_Y_MIN),
    "blue_line_right_top": (BLUE_LINE_X_RIGHT, RINK_Y_MAX),
    "blue_line_right_bottom": (BLUE_LINE_X_RIGHT, RINK_Y_MIN),

    # --- 골라인 (보드 만남 지점; 실제로는 코너 라운딩으로 살짝 안쪽) ---
    "goal_line_left_top": (GOAL_LINE_X_LEFT, RINK_Y_MAX),
    "goal_line_left_bottom": (GOAL_LINE_X_LEFT, RINK_Y_MIN),
    "goal_line_right_top": (GOAL_LINE_X_RIGHT, RINK_Y_MAX),
    "goal_line_right_bottom": (GOAL_LINE_X_RIGHT, RINK_Y_MIN),

    # --- 골 위치 (골라인 중앙) ---
    "goal_left": (GOAL_LINE_X_LEFT, 0.0),
    "goal_right": (GOAL_LINE_X_RIGHT, 0.0),

    # --- 엔드존 페이스오프 서클 중심 (4개) ---
    "faceoff_dz_left_top": (ENDZONE_FACEOFF_X_LEFT, +FACEOFF_Y_FROM_CENTER_M),
    "faceoff_dz_left_bottom": (ENDZONE_FACEOFF_X_LEFT, -FACEOFF_Y_FROM_CENTER_M),
    "faceoff_dz_right_top": (ENDZONE_FACEOFF_X_RIGHT, +FACEOFF_Y_FROM_CENTER_M),
    "faceoff_dz_right_bottom": (ENDZONE_FACEOFF_X_RIGHT, -FACEOFF_Y_FROM_CENTER_M),

    # --- 뉴트럴존 페이스오프 스팟 (4개, 서클 없음) ---
    "faceoff_nz_left_top": (NEUTRAL_FACEOFF_X_LEFT, +FACEOFF_Y_FROM_CENTER_M),
    "faceoff_nz_left_bottom": (NEUTRAL_FACEOFF_X_LEFT, -FACEOFF_Y_FROM_CENTER_M),
    "faceoff_nz_right_top": (NEUTRAL_FACEOFF_X_RIGHT, +FACEOFF_Y_FROM_CENTER_M),
    "faceoff_nz_right_bottom": (NEUTRAL_FACEOFF_X_RIGHT, -FACEOFF_Y_FROM_CENTER_M),
}


# ============================================================================
# 구역 정의 (이벤트 귀속에 활용)
# ============================================================================

def get_zone(x: float, y: float) -> str:
    """링크 좌표를 받아 해당 구역 이름 반환.

    Returns:
        'offensive_left'  : 왼쪽 엔드존 (상대 골 있는 쪽, x < -8.83)
        'neutral'         : 중립 지대 (-8.83 <= x <= +8.83)
        'offensive_right' : 오른쪽 엔드존 (우리 골 있는 쪽, x > +8.83)
        'out_of_bounds'   : 링크 밖
    """
    if not (RINK_X_MIN <= x <= RINK_X_MAX and RINK_Y_MIN <= y <= RINK_Y_MAX):
        return "out_of_bounds"
    if x < BLUE_LINE_X_LEFT:
        return "offensive_left"
    if x > BLUE_LINE_X_RIGHT:
        return "offensive_right"
    return "neutral"


def get_sub_zone(x: float, y: float) -> str:
    """보다 세밀한 구역 분류 (슬롯·코너·서클 등)."""
    main_zone = get_zone(x, y)
    if main_zone == "out_of_bounds":
        return "out_of_bounds"

    # 슬롯 영역: 양 골 앞 중앙 (대략 골라인 2~8m 앞, Y |±3|)
    in_slot_left = (GOAL_LINE_X_LEFT + 2 < x < GOAL_LINE_X_LEFT + 8) and abs(y) < 3
    in_slot_right = (GOAL_LINE_X_RIGHT - 8 < x < GOAL_LINE_X_RIGHT - 2) and abs(y) < 3
    if in_slot_left or in_slot_right:
        return "slot"

    # 코너: 엔드존 내 양 사이드 보드 근처
    if main_zone.startswith("offensive"):
        if abs(y) > 10:
            return "corner"
        if abs(y) > 5:
            return "faceoff_circle_area"

    return main_zone


# ============================================================================
# 링크 규격 프리셋
# ============================================================================
#
# 각 프리셋은 링크 치수만 담음. STANDARD_POINTS는 IIHF 기준으로 생성되며,
# 다른 프리셋을 쓸 경우 해당 치수에 맞춰 별도로 포인트를 재계산해야 함.
#
# KOREA_YOUTH_TBD: 국내 유소년 경기장 실측 전 잠정 수치.
#   목동, 태릉, 과천 등 국내 실내 링크는 대부분 60×26m 내외이나
#   유소년 구획(하프링크 포함) 실제 경계선은 현장 측량 필요.
#   실측 완료 후 이 항목을 업데이트하고 _TBD 접미사 제거 예정.

from typing import TypedDict


class RinkDimensions(TypedDict):
    length: float        # 링크 길이 (m), 골보드-골보드
    width: float         # 링크 폭 (m), 사이드보드-사이드보드
    corner_radius: float # 코너 라운딩 반지름 (m)


RINK_PRESETS: Final[dict[str, RinkDimensions]] = {
    # IIHF 국제 표준 (올림픽·세계선수권)
    "IIHF_STANDARD": {
        "length": 60.0,
        "width": 30.0,
        "corner_radius": 8.5,
    },
    # NHL 북미 표준
    "NHL_STANDARD": {
        "length": 61.0,
        "width": 26.0,
        "corner_radius": 8.5,
    },
    # 한국 유소년 링크 (잠정치 — 실측 후 확정)
    # 국내 실내 링크(목동·태릉·과천) 기준 추정값.
    # 실측값 확보 시 이 항목 업데이트 후 키 이름에서 _TBD 제거.
    "KOREA_YOUTH_TBD": {
        "length": 56.0,   # 추정: 국제 규격보다 짧은 유소년 구획
        "width": 26.0,    # 추정: 목동 등 실내 링크 폭
        "corner_radius": 7.0,  # 추정
    },
}


# ============================================================================
# 벤치·페널티 박스 (근사)
# ============================================================================
# 실제 링크마다 벤치 위치 다를 수 있으므로 기본값만 정의.
# 대개 한쪽 사이드 중앙에 두 팀 벤치가 인접.

BENCH_Y_SIGN: Final[int] = +1  # Y 양수 쪽에 벤치 있다고 가정
BENCH_X_LEFT_EDGE: Final[float] = -6.0   # 센터라인 기준 좌측 벤치 시작
BENCH_X_RIGHT_EDGE: Final[float] = +6.0  # 센터라인 기준 우측 벤치 끝

PENALTY_BOX_Y_SIGN: Final[int] = -1  # 반대쪽에 페널티 박스


def is_in_bench_area(x: float, y: float, threshold_m: float = 1.5) -> bool:
    """선수가 벤치 영역(경계 근처)에 있는지 판정.

    교체 이벤트 감지에 사용.
    벤치는 보드 바깥에 있으므로, 보드에서 threshold_m 내 + 벤치 쪽 Y면 간주.
    """
    if BENCH_X_LEFT_EDGE <= x <= BENCH_X_RIGHT_EDGE:
        distance_to_bench_board = RINK_Y_MAX - y if BENCH_Y_SIGN > 0 else y - RINK_Y_MIN
        return 0 <= distance_to_bench_board < threshold_m
    return False


# ============================================================================
# 유틸: 전체 구조 요약 출력
# ============================================================================

def print_rink_summary() -> None:
    """개발·디버깅용 링크 구조 요약 출력."""
    print("=" * 60)
    print("IIHF Standard Rink Specifications")
    print("=" * 60)
    print(f"Rink size       : {RINK_LENGTH_M}m × {RINK_WIDTH_M}m")
    print(f"Corner radius   : {CORNER_RADIUS_M}m")
    print(f"Goal line X     : {GOAL_LINE_X_LEFT} / {GOAL_LINE_X_RIGHT}")
    print(f"Blue line X     : {BLUE_LINE_X_LEFT} / {BLUE_LINE_X_RIGHT}")
    print(f"Faceoff (endzone): x={ENDZONE_FACEOFF_X_LEFT:.2f} / {ENDZONE_FACEOFF_X_RIGHT:.2f}, y=±{FACEOFF_Y_FROM_CENTER_M}")
    print(f"Faceoff (neutral): x={NEUTRAL_FACEOFF_X_LEFT:.2f} / {NEUTRAL_FACEOFF_X_RIGHT:.2f}, y=±{FACEOFF_Y_FROM_CENTER_M}")
    print(f"\nTotal standard points: {len(STANDARD_POINTS)}")
    for name, (x, y) in STANDARD_POINTS.items():
        print(f"  {name:35s} -> ({x:+.2f}, {y:+.2f})")


if __name__ == "__main__":
    print_rink_summary()
