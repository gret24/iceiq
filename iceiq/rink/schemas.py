"""
링크 좌표계 관련 데이터 스키마.

Pydantic 기반. FastAPI 엔드포인트·JSON 직렬화에 그대로 사용 가능.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from standard_rink import STANDARD_POINTS, RINK_PRESETS


# ============================================================================
# 태깅 점
# ============================================================================

class TaggedPoint(BaseModel):
    """영상 프레임에서 코치가 클릭한 한 점.

    name은 STANDARD_POINTS의 키 중 하나여야 함.
    pixel_* 는 영상 내 픽셀 좌표, rink_* 는 자동으로 STANDARD_POINTS에서 채워짐.
    """

    name: str = Field(..., description="STANDARD_POINTS의 키")
    pixel_x: float = Field(..., description="영상 내 픽셀 X 좌표")
    pixel_y: float = Field(..., description="영상 내 픽셀 Y 좌표")
    rink_x_m: float = Field(..., description="링크 좌표 X (미터)")
    rink_y_m: float = Field(..., description="링크 좌표 Y (미터)")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if v not in STANDARD_POINTS:
            valid = ", ".join(sorted(STANDARD_POINTS.keys()))
            raise ValueError(f"Unknown point name: '{v}'. Valid: {valid}")
        return v

    @classmethod
    def from_name_and_pixel(cls, name: str, pixel_x: float, pixel_y: float) -> "TaggedPoint":
        """이름과 픽셀 좌표만 주면 rink 좌표는 STANDARD_POINTS에서 자동 조회."""
        if name not in STANDARD_POINTS:
            raise ValueError(f"Unknown point name: '{name}'")
        rx, ry = STANDARD_POINTS[name]
        return cls(
            name=name,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            rink_x_m=rx,
            rink_y_m=ry,
        )


# ============================================================================
# 링크 맵 (영상 하나에 대한 정렬 결과)
# ============================================================================

class RinkMap(BaseModel):
    """한 영상(또는 카메라 위치)에 대한 호모그래피 정렬 결과.

    homography_matrix는 3x3 중첩 리스트로 저장 (JSON 친화).
    같은 rink_id + camera_position_id 조합이면 재사용 가능.
    """

    rink_id: str = Field(..., description="링크 식별자 (예: mokdong_01)")
    camera_position_id: str = Field(..., description="카메라 위치 (예: sideline_center)")
    preset_name: str | None = Field(
        None,
        description=(
            "RINK_PRESETS 키 (예: 'IIHF_STANDARD', 'NHL_STANDARD', 'KOREA_YOUTH_TBD'). "
            "None이면 비표준 또는 미지정 링크."
        ),
    )

    sample_frame_path: str | None = Field(None, description="태깅에 사용된 대표 프레임 경로")
    frame_width_px: int = Field(..., gt=0)
    frame_height_px: int = Field(..., gt=0)

    tagged_points: list[TaggedPoint] = Field(..., min_length=4)

    homography_matrix: list[list[float]] = Field(
        ..., description="3x3 호모그래피 행렬 (row-major)"
    )

    quality_score: float = Field(..., ge=0.0, le=1.0, description="0~1, 1이 최고")
    reprojection_error_px_mean: float = Field(..., ge=0.0)
    reprojection_error_px_max: float = Field(..., ge=0.0)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str = Field(default="system")
    note: str = Field(default="")

    @field_validator("homography_matrix")
    @classmethod
    def _validate_h_shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(row) != 3 for row in v):
            raise ValueError("homography_matrix must be 3x3")
        return v

    @field_validator("preset_name")
    @classmethod
    def _validate_preset_name(cls, v: str | None) -> str | None:
        if v is not None and v not in RINK_PRESETS:
            valid = ", ".join(sorted(RINK_PRESETS.keys()))
            raise ValueError(f"Unknown preset_name: '{v}'. Valid: {valid}")
        return v


# ============================================================================
# 좌표 변환 요청·응답 (API용)
# ============================================================================

class PixelToRinkRequest(BaseModel):
    rink_map_id: str
    points_px: list[tuple[float, float]]


class PixelToRinkResponse(BaseModel):
    points_m: list[tuple[float, float]]
    zones: list[str]


# ============================================================================
# 품질 등급 (사용자에게 노출)
# ============================================================================

QualityGrade = Literal["excellent", "good", "fair", "poor"]


def grade_from_score(score: float) -> QualityGrade:
    """0~1 점수를 사람이 이해할 수 있는 등급으로 변환.

    > 0.9: excellent — 본격 사용 OK
    0.8~0.9: good — 대부분의 분석 신뢰 가능
    0.7~0.8: fair — 히트맵 정도는 OK, 정밀 측정은 주의
    < 0.7: poor — 재태깅 권장
    """
    if score >= 0.9:
        return "excellent"
    if score >= 0.8:
        return "good"
    if score >= 0.7:
        return "fair"
    return "poor"
