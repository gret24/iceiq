#!/usr/bin/env python3
"""
선수 특징 추출기 (Claude Vision 기반)
첫 감지 시 선수의 다양한 특징을 추출해서 저장
"""
import base64, io, json, os, re
import numpy as np
from PIL import Image, ImageEnhance
import anthropic

client = anthropic.Anthropic()

FEATURE_PROMPT = """This is a cropped image of an ice hockey player.
Extract the following features and respond in JSON format ONLY:

{
  "jersey_number": "number visible on jersey, or null",
  "jersey_color": "main color of jersey (e.g. white, red, black, blue)",
  "helmet_color": "helmet color (e.g. white, black, red)",
  "helmet_design": "any stripe/logo pattern visible (e.g. solid, striped, logo)",
  "glove_color": "glove color",
  "stick_color": "stick shaft color (e.g. black, white, carbon)",
  "skate_color": "skate boot color (e.g. black, white)",
  "body_build": "estimate shoulder width relative to height: narrow/medium/wide",
  "team": "HOME if jersey is mostly light/white, AWAY if mostly dark"
}

If a feature is not visible, use null. Respond with JSON only, no explanation."""


def extract_features(crop_img: Image.Image, track_id: int) -> dict:
    """선수 크롭 이미지에서 특징 추출"""
    buf = io.BytesIO()
    # 더 잘 보이게 전처리
    w, h = crop_img.size
    if w < 30 or h < 30:
        return {}
    big = crop_img.resize((w*2, h*2), Image.LANCZOS)
    big = ImageEnhance.Contrast(big).enhance(1.5)
    big.save(buf, format='JPEG', quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": FEATURE_PROMPT}
                ]
            }]
        )
        text = resp.content[0].text.strip()
        # JSON 추출
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            features = json.loads(json_match.group())
            features["track_id"] = track_id
            return features
    except Exception as e:
        pass
    return {"track_id": track_id}


def feature_similarity(f1: dict, f2: dict) -> float:
    """두 특징 벡터 간 유사도 (0~1)"""
    score = 0.0
    weights = {
        "jersey_number": 3.0,   # 등번호 가장 중요
        "jersey_color": 2.0,
        "helmet_color": 1.5,
        "glove_color": 1.0,
        "stick_color": 0.8,
        "skate_color": 0.8,
        "team": 2.0,
        "body_build": 0.5,
    }
    total_weight = 0.0

    for key, weight in weights.items():
        v1 = f1.get(key)
        v2 = f2.get(key)
        if v1 is None or v2 is None:
            continue
        total_weight += weight
        if str(v1).lower() == str(v2).lower():
            score += weight
        elif key in ("jersey_color", "helmet_color", "glove_color"):
            # 색상 부분 매칭 (예: "dark blue" vs "blue")
            if str(v1).lower() in str(v2).lower() or str(v2).lower() in str(v1).lower():
                score += weight * 0.7

    return score / total_weight if total_weight > 0 else 0.0
