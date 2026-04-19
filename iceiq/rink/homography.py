"""
호모그래피(Homography) 계산·변환·검증.

영상의 픽셀 좌표와 링크 실좌표(미터) 사이를 매핑.
OpenCV의 findHomography + RANSAC 사용.

핵심 함수:
- compute_homography : 4점 이상 대응관계에서 3x3 변환행렬 계산
- pixel_to_rink      : 픽셀 → 링크 미터 좌표
- rink_to_pixel      : 링크 미터 → 픽셀
- validate_homography: 정렬 품질 평가 (재투영 오차)
"""

from __future__ import annotations

import numpy as np
import cv2

from schemas import TaggedPoint, RinkMap, grade_from_score
from standard_rink import RINK_X_MIN, RINK_X_MAX, RINK_Y_MIN, RINK_Y_MAX


# ============================================================================
# 핵심 계산
# ============================================================================

def compute_homography(
    tagged_points: list[TaggedPoint],
    use_ransac: bool = True,
    ransac_threshold_px: float = 3.0,
) -> np.ndarray:
    """4점 이상의 (픽셀, 링크미터) 대응관계로 3x3 호모그래피 행렬 계산.

    Args:
        tagged_points: 최소 4개 이상의 TaggedPoint.
        use_ransac: True면 RANSAC으로 이상치 배제. False면 DLT.
        ransac_threshold_px: RANSAC 이상치 판정 픽셀 임계값.

    Returns:
        3x3 numpy 배열. 픽셀 -> 링크 미터 변환.

    Raises:
        ValueError: 점이 4개 미만이거나 행렬 계산 실패.
    """
    if len(tagged_points) < 4:
        raise ValueError(f"Need at least 4 points, got {len(tagged_points)}")

    src = np.array(
        [[p.pixel_x, p.pixel_y] for p in tagged_points], dtype=np.float64
    )
    dst = np.array(
        [[p.rink_x_m, p.rink_y_m] for p in tagged_points], dtype=np.float64
    )

    method = cv2.RANSAC if use_ransac else 0
    H, mask = cv2.findHomography(src, dst, method=method, ransacReprojThreshold=ransac_threshold_px)

    if H is None:
        raise ValueError(
            "findHomography returned None. "
            "Points may be collinear or degenerate."
        )
    return H


def pixel_to_rink(px: float, py: float, H: np.ndarray) -> tuple[float, float]:
    """픽셀 좌표를 링크 미터 좌표로 변환."""
    src = np.array([[[px, py]]], dtype=np.float64)  # shape (1,1,2)
    dst = cv2.perspectiveTransform(src, H)
    return float(dst[0, 0, 0]), float(dst[0, 0, 1])


def pixel_to_rink_batch(
    points_px: np.ndarray, H: np.ndarray
) -> np.ndarray:
    """다수 픽셀 좌표를 한 번에 변환.

    Args:
        points_px: shape (N, 2) 픽셀 좌표 배열.
        H: 3x3 호모그래피.
    Returns:
        shape (N, 2) 링크 미터 좌표 배열.
    """
    if points_px.ndim != 2 or points_px.shape[1] != 2:
        raise ValueError(f"Expected shape (N, 2), got {points_px.shape}")
    src = points_px.reshape(-1, 1, 2).astype(np.float64)
    dst = cv2.perspectiveTransform(src, H)
    return dst.reshape(-1, 2)


def rink_to_pixel(rx: float, ry: float, H: np.ndarray) -> tuple[float, float]:
    """링크 미터 좌표를 픽셀 좌표로 변환 (역변환).

    내부적으로 H의 역행렬을 사용. 반복 호출이 많으면 inverse 캐싱 권장.
    """
    H_inv = np.linalg.inv(H)
    return pixel_to_rink(rx, ry, H_inv)


# ============================================================================
# 품질 검증
# ============================================================================

def reprojection_errors_px(
    tagged_points: list[TaggedPoint], H: np.ndarray
) -> np.ndarray:
    """각 태깅 점의 재투영 오차(픽셀) 배열 반환.

    '링크 좌표를 다시 픽셀로 변환했을 때, 원래 태깅한 픽셀과 얼마나 다른가'.
    오차가 작을수록 정렬 품질 우수.
    """
    H_inv = np.linalg.inv(H)
    errors = []
    for p in tagged_points:
        src_rink = np.array([[[p.rink_x_m, p.rink_y_m]]], dtype=np.float64)
        back = cv2.perspectiveTransform(src_rink, H_inv)
        px_pred, py_pred = back[0, 0, 0], back[0, 0, 1]
        err = np.sqrt((px_pred - p.pixel_x) ** 2 + (py_pred - p.pixel_y) ** 2)
        errors.append(err)
    return np.array(errors)


def quality_score_from_errors(
    errors_px: np.ndarray,
    good_threshold_px: float = 5.0,
    bad_threshold_px: float = 30.0,
) -> float:
    """재투영 오차로부터 0~1 품질 점수 계산.

    - 평균 오차 <= good_threshold_px → 1.0 근처
    - 평균 오차 >= bad_threshold_px → 0 근처
    - 그 사이는 선형 보간
    """
    mean_err = float(np.mean(errors_px))
    if mean_err <= good_threshold_px:
        return max(0.0, min(1.0, 1.0 - (mean_err / good_threshold_px) * 0.1))
    if mean_err >= bad_threshold_px:
        return 0.0
    # 선형 보간 (good=0.9, bad=0.0)
    ratio = (mean_err - good_threshold_px) / (bad_threshold_px - good_threshold_px)
    return max(0.0, 0.9 * (1.0 - ratio))


def validate_homography(
    tagged_points: list[TaggedPoint], H: np.ndarray
) -> dict:
    """호모그래피 품질 종합 검증.

    Returns:
        dict with keys:
          - errors_px: np.ndarray of per-point errors
          - mean_error_px, max_error_px
          - quality_score (0~1)
          - quality_grade ('excellent'/'good'/'fair'/'poor')
    """
    errors = reprojection_errors_px(tagged_points, H)
    score = quality_score_from_errors(errors)
    return {
        "errors_px": errors,
        "mean_error_px": float(np.mean(errors)),
        "max_error_px": float(np.max(errors)),
        "quality_score": score,
        "quality_grade": grade_from_score(score),
    }


# ============================================================================
# 편의: RinkMap 빌드
# ============================================================================

def build_rink_map(
    rink_id: str,
    camera_position_id: str,
    frame_width_px: int,
    frame_height_px: int,
    tagged_points: list[TaggedPoint],
    sample_frame_path: str | None = None,
    created_by: str = "system",
    note: str = "",
) -> RinkMap:
    """태깅 결과로부터 완전한 RinkMap 생성.

    호모그래피 계산 + 품질 평가까지 한 번에 수행.
    """
    H = compute_homography(tagged_points)
    validation = validate_homography(tagged_points, H)

    return RinkMap(
        rink_id=rink_id,
        camera_position_id=camera_position_id,
        sample_frame_path=sample_frame_path,
        frame_width_px=frame_width_px,
        frame_height_px=frame_height_px,
        tagged_points=tagged_points,
        homography_matrix=H.tolist(),
        quality_score=validation["quality_score"],
        reprojection_error_px_mean=validation["mean_error_px"],
        reprojection_error_px_max=validation["max_error_px"],
        created_by=created_by,
        note=note,
    )


def load_homography(rink_map: RinkMap) -> np.ndarray:
    """RinkMap에서 numpy 3x3 호모그래피 복원."""
    return np.array(rink_map.homography_matrix, dtype=np.float64)


# ============================================================================
# 안전 체크
# ============================================================================

def is_in_rink(rx: float, ry: float, margin_m: float = 1.0) -> bool:
    """링크 범위 내 좌표인지 확인.

    호모그래피 결과가 비현실적이면(보드 밖으로 멀리 벗어남) 이상 신호.
    """
    return (
        (RINK_X_MIN - margin_m) <= rx <= (RINK_X_MAX + margin_m)
        and (RINK_Y_MIN - margin_m) <= ry <= (RINK_Y_MAX + margin_m)
    )
