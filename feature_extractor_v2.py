#!/usr/bin/env python3
"""
선수 특징 추출기 v2 (Claude Vision + CV 분석)
추가된 특징:
  1. 스케이팅 스타일 (연속 프레임 다리 움직임)
  2. 스틱 핸들링 패턴 (왼손/오른손, 스틱 각도)
  3. 등번호 폰트/크기/색상
  4. 보호대 색상 (어깨, 팔꿈치)
  5. 헬멧 바이저/케이지 유무
"""
import base64, io, json, os, re, math
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import anthropic

client = anthropic.Anthropic()

# ── Claude Vision 프롬프트 (확장) ─────────────────────────
FEATURE_PROMPT_V2 = """This is a cropped image of an ice hockey player.
Extract ALL visible features and respond in JSON format ONLY:

{
  "jersey_number": "number on jersey or null",
  "jersey_color": "main jersey color (e.g. white, red, black, blue)",
  "jersey_number_font": "number font style: block/outline/thin/bold or null",
  "jersey_number_size": "relative size of number on jersey: small/medium/large or null",
  "jersey_number_colors": "colors of the number text and outline (e.g. black on white, red outline) or null",
  "helmet_color": "helmet color",
  "helmet_visor": "visor type: full_visor/half_visor/cage/none or null",
  "helmet_design": "solid/striped/logo or null",
  "glove_color": "glove color",
  "stick_handedness": "left if stick on left side, right if on right side, or null",
  "stick_color": "stick shaft color",
  "stick_angle": "stick angle relative to body: upright/diagonal/horizontal or null",
  "shoulder_pad_color": "visible shoulder pad color or null",
  "elbow_pad_color": "visible elbow pad color or null",
  "skate_color": "skate boot color",
  "body_build": "shoulder width: narrow/medium/wide",
  "team": "HOME if light/white jersey, AWAY if dark jersey"
}

If not visible, use null. JSON only."""


def extract_features(crop_img: Image.Image, track_id: int,
                      prev_crops: list = None) -> dict:
    """
    선수 크롭 이미지에서 특징 추출
    prev_crops: 이전 프레임 크롭 목록 (스케이팅 스타일 분석용)
    """
    w, h = crop_img.size
    if w < 30 or h < 30:
        return {}

    # 전처리
    big = crop_img.resize((w*2, h*2), Image.LANCZOS)
    big = ImageEnhance.Contrast(big).enhance(1.5)
    buf = io.BytesIO()
    big.save(buf, format='JPEG', quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()

    features = {"track_id": track_id}

    # ── Claude Vision 특징 추출 ─────────────────────────
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": FEATURE_PROMPT_V2}
                ]
            }]
        )
        text = resp.content[0].text.strip()
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            features.update(json.loads(m.group()))
    except Exception:
        pass

    # ── CV 기반 특징 추출 ───────────────────────────────
    bgr = cv2.cvtColor(np.array(crop_img), cv2.COLOR_RGB2BGR)
    h_img, w_img = bgr.shape[:2]

    # [1] 보호대 색상 (어깨: 상단 20~40%, 팔꿈치: 중간 40~65%)
    if features.get("shoulder_pad_color") is None:
        shoulder = bgr[int(h_img*0.20):int(h_img*0.40), :]
        features["shoulder_pad_color_cv"] = _dominant_color_name(shoulder)

    if features.get("elbow_pad_color") is None:
        elbow_l = bgr[int(h_img*0.40):int(h_img*0.65), :int(w_img*0.25)]
        elbow_r = bgr[int(h_img*0.40):int(h_img*0.65), int(w_img*0.75):]
        # 양쪽 팔꿈치 개별 분석 후 평균
        if elbow_l.size == 0 and elbow_r.size == 0:
            pad = np.zeros((1,1,3), dtype=np.uint8)
        elif elbow_l.size == 0:
            pad = elbow_r
        elif elbow_r.size == 0:
            pad = elbow_l
        else:
            pad = elbow_l  # 왼쪽 기준
        features["elbow_pad_color_cv"] = _dominant_color_name(pad)

    # [2] 스틱 방향 (하단 30%에서 어두운 수직/대각선 감지)
    if features.get("stick_handedness") is None:
        lower = bgr[int(h_img*0.70):, :]
        gray_l = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_l, 60, 255, cv2.THRESH_BINARY_INV)
        left_mass = np.sum(thresh[:, :w_img//2])
        right_mass = np.sum(thresh[:, w_img//2:])
        if left_mass + right_mass > 0:
            features["stick_handedness_cv"] = "left" if left_mass > right_mass * 1.3 \
                else ("right" if right_mass > left_mass * 1.3 else "center")

    # [3] 등번호 영역 밝기/대비 (폰트 굵기 추정)
    num_region = bgr[int(h_img*0.15):int(h_img*0.65), int(w_img*0.10):int(w_img*0.90)]
    if num_region.size > 0:
        gray_num = cv2.cvtColor(num_region, cv2.COLOR_BGR2GRAY)
        std = float(np.std(gray_num))
        features["jersey_number_contrast_cv"] = round(std, 1)
        # 대비 높으면 bold 폰트 가능성
        if features.get("jersey_number_font") is None:
            features["jersey_number_font_cv"] = "bold" if std > 60 else "thin"

    # [4] 헬멧 바이저 (상단 25% 중앙에 밝은 직선 존재 여부)
    if features.get("helmet_visor") is None:
        helmet_region = bgr[:int(h_img*0.25), int(w_img*0.15):int(w_img*0.85)]
        if helmet_region.size > 0:
            gray_h = cv2.cvtColor(helmet_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_h, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15,
                                     minLineLength=10, maxLineGap=5)
            features["helmet_visor_cv"] = "detected" if lines is not None else "not_detected"

    # [5] 스케이팅 스타일 (연속 프레임 제공 시)
    if prev_crops and len(prev_crops) >= 2:
        features["skating_style_cv"] = _skating_style(crop_img, prev_crops)

    features["feature_version"] = "v2"
    return features


def _dominant_color_name(bgr_region: np.ndarray) -> str:
    """BGR 영역에서 주요 색상 이름 반환"""
    if bgr_region.size == 0:
        return "unknown"
    pixels = bgr_region.reshape(-1, 3).astype(np.float32)
    # 평균 HSV
    hsv = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2HSV)
    h_mean = float(np.mean(hsv[:,:,0]))
    s_mean = float(np.mean(hsv[:,:,1]))
    v_mean = float(np.mean(hsv[:,:,2]))

    if v_mean < 50:   return "black"
    if v_mean > 200 and s_mean < 50: return "white"
    if s_mean < 40:   return "gray"
    if h_mean < 10 or h_mean > 160: return "red"
    if 10 <= h_mean < 25:  return "orange"
    if 25 <= h_mean < 35:  return "yellow"
    if 35 <= h_mean < 85:  return "green"
    if 85 <= h_mean < 130: return "blue"
    if 130 <= h_mean < 160: return "purple"
    return "mixed"


def _skating_style(current: Image.Image, prev_list: list) -> str:
    """연속 프레임에서 다리 움직임 패턴으로 스케이팅 스타일 추정"""
    try:
        scores = []
        curr_arr = np.array(current.convert("L"))
        h, w = curr_arr.shape
        lower_curr = curr_arr[int(h*0.65):, :]

        for prev in prev_list[-3:]:
            prev_arr = np.array(prev.convert("L"))
            if prev_arr.shape != curr_arr.shape:
                prev_arr = np.array(prev.resize(current.size).convert("L"))
            lower_prev = prev_arr[int(h*0.65):, :]
            diff = np.abs(lower_curr.astype(float) - lower_prev.astype(float))
            scores.append(float(np.mean(diff)))

        avg_motion = np.mean(scores) if scores else 0
        if avg_motion > 30:   return "active_stride"
        elif avg_motion > 15: return "moderate_glide"
        else:                 return "stationary_coast"
    except Exception:
        return "unknown"


# ── 유사도 계산 v2 ─────────────────────────────────────────
def feature_similarity(f1: dict, f2: dict) -> float:
    """두 특징 벡터 간 유사도 v2 (0~1) — 5가지 추가 포인트 포함"""
    weights = {
        # 기존 핵심 특징
        "jersey_number":         3.0,
        "jersey_color":          2.0,
        "team":                  2.0,
        "helmet_color":          1.5,
        "glove_color":           1.0,
        "stick_color":           0.8,
        "skate_color":           0.8,
        "body_build":            0.5,
        # v2 추가: 스틱 핸들링
        "stick_handedness":      1.5,
        "stick_handedness_cv":   1.2,
        "stick_angle":           0.8,
        # v2 추가: 등번호 폰트/크기/색상
        "jersey_number_font":    1.2,
        "jersey_number_font_cv": 0.8,
        "jersey_number_size":    1.0,
        "jersey_number_colors":  1.0,
        # v2 추가: 보호대 색상
        "shoulder_pad_color":    1.2,
        "shoulder_pad_color_cv": 0.8,
        "elbow_pad_color":       1.0,
        "elbow_pad_color_cv":    0.7,
        # v2 추가: 헬멧 바이저
        "helmet_visor":          1.5,
        "helmet_visor_cv":       0.8,
        "helmet_design":         0.8,
        # v2 추가: 스케이팅 스타일
        "skating_style_cv":      0.6,
    }

    color_keys = {
        "jersey_color", "helmet_color", "glove_color", "stick_color",
        "skate_color", "shoulder_pad_color", "shoulder_pad_color_cv",
        "elbow_pad_color", "elbow_pad_color_cv",
    }

    score = 0.0
    total_weight = 0.0

    for key, weight in weights.items():
        v1 = f1.get(key)
        v2 = f2.get(key)
        if v1 is None or v2 is None:
            continue
        total_weight += weight
        s1, s2 = str(v1).lower(), str(v2).lower()
        if s1 == s2:
            score += weight
        elif key in color_keys:
            if s1 in s2 or s2 in s1:
                score += weight * 0.7
        elif key == "jersey_number_colors":
            # 색상 조합 부분 매칭
            common = set(s1.split()) & set(s2.split())
            if common:
                score += weight * (len(common) / max(len(s1.split()), len(s2.split())))

    return score / total_weight if total_weight > 0 else 0.0


def feature_similarity_breakdown(f1: dict, f2: dict) -> dict:
    """유사도 항목별 상세 분석 (디버그용)"""
    groups = {
        "기본": ["jersey_number","jersey_color","team","helmet_color","glove_color","stick_color","skate_color","body_build"],
        "스틱핸들링": ["stick_handedness","stick_handedness_cv","stick_angle"],
        "등번호폰트": ["jersey_number_font","jersey_number_font_cv","jersey_number_size","jersey_number_colors"],
        "보호대": ["shoulder_pad_color","shoulder_pad_color_cv","elbow_pad_color","elbow_pad_color_cv"],
        "헬멧바이저": ["helmet_visor","helmet_visor_cv","helmet_design"],
        "스케이팅": ["skating_style_cv"],
    }
    result = {}
    for group, keys in groups.items():
        g1 = {k: f1.get(k) for k in keys}
        g2 = {k: f2.get(k) for k in keys}
        sim = feature_similarity(g1, g2)
        result[group] = round(sim, 3)
    result["전체"] = round(feature_similarity(f1, f2), 3)
    return result
