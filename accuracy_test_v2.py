#!/usr/bin/env python3
"""
정확도 측정 v2 - 6가지 개선 적용
1. confidence 0.1
2. 전체 박스 크롭
3. 2배 확대 + 대비 강화
4. 모션 블러 처리 (선명한 프레임만 OCR)
5. ByteTrack 연동 (한 번 인식 후 추적으로 유지)
6. Claude Vision 보조 (EasyOCR 실패 시 재시도)
"""

import os, re, json, base64
import numpy as np
from PIL import Image, ImageEnhance
import cv2

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
TRACKS_JSON = os.path.join(BASE_DIR, "tracks.json")
JERSEY_JSON = os.path.join(BASE_DIR, "jersey_map.json")

# ── 샘플 선택 ────────────────────────────────────────────
all_frames = sorted(os.listdir(FRAMES_DIR))
step = max(1, len(all_frames) // 20)
samples = all_frames[::step][:20]
print(f"총 프레임: {len(all_frames)}장 | 샘플: {len(samples)}장\n")

# ── tracks.json 로드 ─────────────────────────────────────
with open(TRACKS_JSON) as f:
    tracks_db = json.load(f)

# ── jersey_map.json 로드 (기존 인식 결과) ────────────────
with open(JERSEY_JSON) as f:
    jersey_map = json.load(f)  # {track_id: {jersey, team}}

# ── [개선 4] 모션 블러 감지 함수 ──────────────────────────
def laplacian_variance(img_path):
    """라플라시안 분산 → 높을수록 선명, 낮을수록 블러"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    return cv2.Laplacian(img, cv2.CV_64F).var()

BLUR_THRESHOLD = 50  # 이 이하면 블러로 판단 (심한 블러만 스킵)

# ── YOLO 선수 감지 ────────────────────────────────────────
from ultralytics import YOLO
model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))
MIN_BOX_AREA = 5000

print("=== [1/2] 선수 감지 (YOLO) ===")
detected_frames = 0
total_detections = 0
frame_boxes = {}  # fname → list of valid boxes

for fname in samples:
    fpath = os.path.join(FRAMES_DIR, fname)
    results = model(fpath, classes=[0], verbose=False)[0]
    boxes = results.boxes
    valid_boxes = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area >= MIN_BOX_AREA:
                valid_boxes.append((x1, y1, x2, y2))
    frame_boxes[fname] = valid_boxes
    detected = len(valid_boxes) > 0
    if detected:
        detected_frames += 1
    total_detections += len(valid_boxes)
    print(f"  {fname}: {len(valid_boxes)}명 감지 {'✅' if detected else '❌'}")

detect_rate = detected_frames / len(samples) * 100
avg_per_frame = total_detections / len(samples)
print(f"\n선수 감지율: {detect_rate:.1f}% ({detected_frames}/{len(samples)} 프레임)")
print(f"평균 감지 인원: {avg_per_frame:.1f}명/프레임")

# ── EasyOCR 초기화 ───────────────────────────────────────
import easyocr
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

def ocr_crop(crop_img):
    """전처리 후 EasyOCR → 숫자 문자열 또는 None"""
    cw, ch = crop_img.size
    # 너무 작으면 스킵
    if cw < 10 or ch < 10:
        return None
    # 2배 확대 + 대비 강화
    big = crop_img.resize((cw * 2, ch * 2), Image.LANCZOS)
    big = ImageEnhance.Contrast(big).enhance(2.0)
    result = reader.readtext(np.array(big))
    for _, text, conf in result:
        clean = re.sub(r'[^0-9]', '', text)
        if clean and conf > 0.1:
            return clean
    return None

# ── Claude Vision 보조 ───────────────────────────────────
import anthropic

claude = anthropic.Anthropic()

def claude_ocr(crop_img):
    """EasyOCR 실패 시 Claude Vision으로 재시도"""
    import io
    buf = io.BytesIO()
    crop_img.save(buf, format='JPEG', quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        resp = claude.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}
                    },
                    {
                        "type": "text",
                        "text": "This is a cropped image of a hockey player. What jersey number do you see? Reply with ONLY the number (e.g. '23'). If you cannot see any number clearly, reply 'NONE'."
                    }
                ]
            }]
        )
        answer = resp.content[0].text.strip()
        clean = re.sub(r'[^0-9]', '', answer)
        return clean if clean else None
    except Exception as e:
        print(f"    Claude 오류: {e}")
        return None

# ── [개선 5] ByteTrack 연동 - 기존 jersey_map 활용 ────────
def get_jersey_from_tracks(fname):
    """tracks.json에서 해당 프레임의 track_id 가져와 jersey_map 조회"""
    frame_tracks = tracks_db.get(fname, [])
    for t in frame_tracks:
        tid = str(t['track_id'])
        if tid in jersey_map:
            return jersey_map[tid]['jersey']
    return None

# ── 등번호 인식 메인 루프 ─────────────────────────────────
print("\n=== [2/2] 등번호 인식 (개선판 v2) ===")
print("  적용: confidence 0.1 / 전체 박스 / 2배확대+대비 / 블러스킵 / ByteTrack / Claude Vision\n")

ocr_frames = 0
skip_blur = 0
bytetrack_hit = 0
easyocr_hit = 0
claude_hit = 0
claude_calls = 0

for fname in samples:
    fpath = os.path.join(FRAMES_DIR, fname)
    result_number = None
    method = ""

    # [개선 4] 모션 블러 체크
    sharpness = laplacian_variance(fpath)
    if sharpness < BLUR_THRESHOLD:
        skip_blur += 1
        print(f"  {fname}: 블러 스킵 (sharpness={sharpness:.1f}) ⏭️")
        continue

    # [개선 5] ByteTrack 연동 - 기존 jersey_map에서 먼저 확인
    jersey_from_track = get_jersey_from_tracks(fname)
    if jersey_from_track:
        result_number = jersey_from_track
        method = f"ByteTrack(#{result_number})"
        bytetrack_hit += 1

    # EasyOCR 시도
    if not result_number:
        img = Image.open(fpath)
        w, h = img.size
        for (x1, y1, x2, y2) in frame_boxes.get(fname, []):
            # [개선 2] 전체 박스 크롭
            cx1, cy1 = max(0, int(x1)), max(0, int(y1))
            cx2, cy2 = min(w, int(x2)), min(h, int(y2))
            crop = img.crop((cx1, cy1, cx2, cy2))
            num = ocr_crop(crop)
            if num:
                result_number = num
                method = f"EasyOCR(#{num})"
                easyocr_hit += 1
                break

    # [개선 6] Claude Vision 보조
    if not result_number and frame_boxes.get(fname):
        img = Image.open(fpath)
        w, h = img.size
        x1, y1, x2, y2 = frame_boxes[fname][0]
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(w, int(x2)), min(h, int(y2))
        crop = img.crop((cx1, cy1, cx2, cy2))
        cw, ch = crop.size
        big_crop = crop.resize((cw * 2, ch * 2), Image.LANCZOS)
        big_crop = ImageEnhance.Contrast(big_crop).enhance(2.0)
        claude_calls += 1
        print(f"    → Claude Vision 시도... ", end="", flush=True)
        num = claude_ocr(big_crop)
        if num:
            result_number = num
            method = f"Claude(#{num})"
            claude_hit += 1
            print(f"✅ #{num}")
        else:
            print("❌")

    if result_number:
        ocr_frames += 1
        print(f"  {fname}: ✅ {method} (sharpness={sharpness:.1f})")
    else:
        print(f"  {fname}: ❌ (sharpness={sharpness:.1f})")

# 분모: 블러 스킵 제외한 프레임
effective = len(samples) - skip_blur
ocr_rate_all = ocr_frames / len(samples) * 100
ocr_rate_eff = ocr_frames / max(1, effective) * 100

print(f"\n등번호 인식율 (전체): {ocr_rate_all:.1f}% ({ocr_frames}/{len(samples)})")
print(f"등번호 인식율 (선명한 프레임): {ocr_rate_eff:.1f}% ({ocr_frames}/{effective})")
print(f"  - 블러 스킵: {skip_blur}장")
print(f"  - ByteTrack 히트: {bytetrack_hit}건")
print(f"  - EasyOCR 히트: {easyocr_hit}건")
print(f"  - Claude Vision 히트: {claude_hit}/{claude_calls}건")

print("\n========== 최종 결과 ==========")
print(f"1. 선수 감지율:         {detect_rate:.1f}%")
print(f"2. 등번호 인식율 (전체): {ocr_rate_all:.1f}%")
print(f"   등번호 인식율 (유효): {ocr_rate_eff:.1f}%")
print(f"   목표 50%+ {'✅ 달성!' if ocr_rate_eff >= 50 else '❌ 미달'}")
print("================================")
