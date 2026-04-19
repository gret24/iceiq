"""
호모그래피 코어 동작 검증.

가상의 카메라 투영을 만들어 pixel <-> rink 왕복 변환이 정확한지 확인.
월요일 실제 영상 투입 전 코드 자체가 작동하는지 테스트.
"""

import numpy as np

from homography import (
    compute_homography, pixel_to_rink, rink_to_pixel,
    validate_homography, build_rink_map, is_in_rink,
)
from schemas import TaggedPoint
from standard_rink import STANDARD_POINTS


def synthesize_camera_projection(
    rink_points: list[tuple[float, float]],
    camera_matrix_3x3: np.ndarray | None = None,
) -> list[tuple[float, float]]:
    """링크 실좌표를 가짜 카메라 행렬로 픽셀 좌표로 투영.

    실제 영상 없이 코드 동작 검증용.
    """
    if camera_matrix_3x3 is None:
        # 일반적인 사이드라인 카메라를 모사한 적당한 투영
        # (실제로는 카메라 내부·외부 파라미터 복잡함; 여기선 임의 호모그래피)
        camera_matrix_3x3 = np.array([
            [25.0,  0.0, 960.0],
            [ 0.0, 25.0, 540.0],
            [ 0.0,  0.0,   1.0],
        ], dtype=np.float64)
        # 약간의 원근 왜곡 추가
        camera_matrix_3x3[2, 0] = 0.002
        camera_matrix_3x3[2, 1] = 0.001

    pixels = []
    for rx, ry in rink_points:
        v = np.array([rx, ry, 1.0])
        uvw = camera_matrix_3x3 @ v
        pixels.append((uvw[0] / uvw[2], uvw[1] / uvw[2]))
    return pixels


def test_basic_4point_homography():
    """가장 단순한 4점으로 왕복 변환 검증."""
    print("\n[Test 1] Basic 4-point homography (synthetic camera)")

    # 링크 표준점 4개 선택 (가장 멀리 떨어진 코너 스타일)
    point_names = [
        "blue_line_left_top",
        "blue_line_right_top",
        "blue_line_left_bottom",
        "blue_line_right_bottom",
    ]
    rink_coords = [STANDARD_POINTS[n] for n in point_names]
    pixel_coords = synthesize_camera_projection(rink_coords)

    tagged = [
        TaggedPoint.from_name_and_pixel(name, px, py)
        for name, (px, py) in zip(point_names, pixel_coords)
    ]

    H = compute_homography(tagged, use_ransac=False)  # 4점 정확히면 RANSAC 불필요
    print(f"  H shape: {H.shape}")

    # 왕복 확인
    for p in tagged:
        rx_pred, ry_pred = pixel_to_rink(p.pixel_x, p.pixel_y, H)
        err = np.sqrt((rx_pred - p.rink_x_m) ** 2 + (ry_pred - p.rink_y_m) ** 2)
        print(f"  {p.name:25s} expected ({p.rink_x_m:+.2f}, {p.rink_y_m:+.2f}) "
              f"got ({rx_pred:+.3f}, {ry_pred:+.3f}) err={err*100:.2f}cm")
        assert err < 0.01, f"Round-trip error too large: {err}m"

    print("  OK — round-trip within 1cm")


def test_more_points_with_noise():
    """8점 + 픽셀 노이즈로 RANSAC 검증."""
    print("\n[Test 2] 8 points with random pixel noise (RANSAC)")

    point_names = [
        "center_faceoff",
        "blue_line_left_top", "blue_line_right_top",
        "blue_line_left_bottom", "blue_line_right_bottom",
        "faceoff_dz_left_top", "faceoff_dz_right_bottom",
        "center_line_top",
    ]
    rink_coords = [STANDARD_POINTS[n] for n in point_names]
    pixel_coords = synthesize_camera_projection(rink_coords)

    # 노이즈 추가 (±2픽셀)
    rng = np.random.default_rng(42)
    noisy_pixels = [
        (px + rng.normal(0, 2.0), py + rng.normal(0, 2.0))
        for px, py in pixel_coords
    ]

    tagged = [
        TaggedPoint.from_name_and_pixel(name, px, py)
        for name, (px, py) in zip(point_names, noisy_pixels)
    ]

    H = compute_homography(tagged, use_ransac=True)
    validation = validate_homography(tagged, H)

    print(f"  mean reproj error: {validation['mean_error_px']:.2f}px")
    print(f"  max reproj error : {validation['max_error_px']:.2f}px")
    print(f"  quality score    : {validation['quality_score']:.3f}")
    print(f"  quality grade    : {validation['quality_grade']}")

    # 노이즈 ±2px이면 평균 오차 5px 이내여야 정상
    assert validation['mean_error_px'] < 10.0


def test_rink_to_pixel_inverse():
    """역변환 일치 확인."""
    print("\n[Test 3] rink_to_pixel as inverse")

    point_names = [
        "center_faceoff",
        "goal_left", "goal_right",
        "blue_line_left_top", "blue_line_right_bottom",
    ]
    rink_coords = [STANDARD_POINTS[n] for n in point_names]
    pixel_coords = synthesize_camera_projection(rink_coords)

    tagged = [
        TaggedPoint.from_name_and_pixel(name, px, py)
        for name, (px, py) in zip(point_names, pixel_coords)
    ]

    H = compute_homography(tagged, use_ransac=False)

    # 새 링크 좌표 → 픽셀 → 다시 링크 변환이 일치?
    test_rink = (-5.0, 3.0)
    px, py = rink_to_pixel(*test_rink, H)
    rx_back, ry_back = pixel_to_rink(px, py, H)
    err = np.sqrt((rx_back - test_rink[0]) ** 2 + (ry_back - test_rink[1]) ** 2)
    print(f"  ({test_rink[0]:+.2f}, {test_rink[1]:+.2f}) -> "
          f"pixel ({px:.1f}, {py:.1f}) -> "
          f"back ({rx_back:+.3f}, {ry_back:+.3f}) err={err*100:.3f}cm")
    assert err < 0.001


def test_build_rink_map_full():
    """RinkMap 전체 생성 플로우."""
    print("\n[Test 4] Full RinkMap build")

    point_names = [
        "blue_line_left_top", "blue_line_right_top",
        "blue_line_left_bottom", "blue_line_right_bottom",
        "center_faceoff", "goal_left", "goal_right",
    ]
    rink_coords = [STANDARD_POINTS[n] for n in point_names]
    pixel_coords = synthesize_camera_projection(rink_coords)

    tagged = [
        TaggedPoint.from_name_and_pixel(name, px, py)
        for name, (px, py) in zip(point_names, pixel_coords)
    ]

    rink_map = build_rink_map(
        rink_id="test_rink_01",
        camera_position_id="sideline_center",
        frame_width_px=1920,
        frame_height_px=1080,
        tagged_points=tagged,
        note="synthetic test",
    )

    print(f"  rink_id        : {rink_map.rink_id}")
    print(f"  quality_score  : {rink_map.quality_score:.3f}")
    print(f"  mean_err_px    : {rink_map.reprojection_error_px_mean:.2f}")
    print(f"  tagged count   : {len(rink_map.tagged_points)}")

    # JSON 직렬화 확인
    json_str = rink_map.model_dump_json(indent=2)
    assert len(json_str) > 100
    print(f"  JSON ok ({len(json_str)} chars)")


def test_is_in_rink():
    """링크 경계 확인."""
    print("\n[Test 5] is_in_rink bounds check")
    assert is_in_rink(0.0, 0.0)
    assert is_in_rink(-25.0, 10.0)
    assert not is_in_rink(50.0, 0.0)      # 링크 밖
    assert not is_in_rink(0.0, -20.0)     # 링크 밖
    assert is_in_rink(-30.5, 0.0)         # 마진 1m 내
    print("  OK")


if __name__ == "__main__":
    test_basic_4point_homography()
    test_more_points_with_noise()
    test_rink_to_pixel_inverse()
    test_build_rink_map_full()
    test_is_in_rink()
    print("\nAll tests passed.")
