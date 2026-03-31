#!/usr/bin/env python3
"""
정확도 측정 스크립트
frames 폴더에서 샘플 20장을 골라:
1. 선수 감지율 (YOLO)
2. 등번호 인식율 (EasyOCR)
"""

import os
import random
import sys
import numpy as np
from PIL import Image

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")

# 샘플 20장 선택 (균등 분포)
all_frames = sorted(os.listdir(FRAMES_DIR))
step = max(1, len(all_frames) // 20)
samples = all_frames[::step][:20]
print(f"총 프레임: {len(all_frames)}장 | 샘플: {len(samples)}장\n")

# ── YOLO 선수 감지 ────────────────────────────────────────
from ultralytics import YOLO
model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))

MIN_BOX_AREA = 5000
detected_frames = 0
total_detections = 0

print("=== [1/2] 선수 감지 (YOLO) ===")
frame_detections = []
for fname in samples:
    fpath = os.path.join(FRAMES_DIR, fname)
    results = model(fpath, classes=[0], verbose=False)[0]  # class 0 = person
    boxes = results.boxes
    valid = 0
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area >= MIN_BOX_AREA:
                valid += 1
    frame_detections.append(valid)
    detected = valid > 0
    if detected:
        detected_frames += 1
    total_detections += valid
    print(f"  {fname}: {valid}명 감지 {'✅' if detected else '❌'}")

detect_rate = detected_frames / len(samples) * 100
avg_per_frame = total_detections / len(samples)
print(f"\n선수 감지율: {detect_rate:.1f}% ({detected_frames}/{len(samples)} 프레임)")
print(f"평균 감지 인원: {avg_per_frame:.1f}명/프레임")

# ── EasyOCR 등번호 인식 (개선판) ─────────────────────────
import easyocr
import re
from PIL import ImageEnhance

print("\n=== [2/2] 등번호 인식 (EasyOCR - 개선판) ===")
print("  적용: confidence 0.1 / 전체 박스 크롭 / 2배 확대 + 대비 강화")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

ocr_frames = 0

for i, fname in enumerate(samples):
    fpath = os.path.join(FRAMES_DIR, fname)
    results_yolo = model(fpath, classes=[0], verbose=False)[0]
    img = Image.open(fpath)
    w, h = img.size

    found_number = False
    boxes = results_yolo.boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area = (x2 - x1) * (y2 - y1)
            if area < MIN_BOX_AREA:
                continue

            # [개선 2] 전체 박스 크롭 (상반신 60% → 100%)
            cx1 = max(0, int(x1))
            cy1 = max(0, int(y1))
            cx2 = min(w, int(x2))
            cy2 = min(h, int(y2))
            crop = img.crop((cx1, cy1, cx2, cy2))

            # [개선 3] 2배 확대 + 대비 강화
            cw, ch = crop.size
            crop = crop.resize((cw * 2, ch * 2), Image.LANCZOS)
            crop = ImageEnhance.Contrast(crop).enhance(2.0)

            ocr_result = reader.readtext(np.array(crop))
            for _, text, conf in ocr_result:
                clean = re.sub(r'[^0-9]', '', text)
                # [개선 1] confidence 0.3 → 0.1
                if clean and conf > 0.1:
                    found_number = True
                    break
            if found_number:
                break

    if found_number:
        ocr_frames += 1
    print(f"  {fname}: 등번호 인식 {'✅' if found_number else '❌'}")

ocr_rate = ocr_frames / len(samples) * 100
print(f"\n등번호 인식율: {ocr_rate:.1f}% ({ocr_frames}/{len(samples)} 프레임)")

print("\n========== 최종 결과 ==========")
print(f"1. 선수 감지율:   {detect_rate:.1f}%")
print(f"2. 등번호 인식율: {ocr_rate:.1f}%")
print("================================")
