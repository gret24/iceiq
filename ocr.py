#!/usr/bin/env python3
from ultralytics import YOLO
import easyocr
from PIL import Image
import numpy as np
import os
import re
import time

input_dir = "detected"
output_file = "results.txt"
SAMPLE_EVERY = 10       # 10장당 1장 샘플링
MIN_BOX_AREA = 5000     # 최소 박스 면적 (px²)

model = YOLO("yolov8n.pt")
reader = easyocr.Reader(['en'], gpu=False, verbose=False)

def get_uniform_color(torso_array):
    brightness = np.mean(torso_array)
    return "HOME" if brightness > 128 else "AWAY"

def is_jersey_number(text):
    text = text.strip()
    if re.match(r'^\d{1,2}$', text):
        num = int(text)
        return 1 <= num <= 99
    return False

image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])
sampled_files = image_files[::SAMPLE_EVERY]
total = len(sampled_files)
result_count = 0
skipped_small = 0

print(f"전체 {len(image_files)}장 중 {total}장 처리 (10장당 1장 샘플링)")

start_time = time.time()

with open(output_file, "w", encoding="utf-8") as out:
    out.write("파일명 | 감지된번호 | 유니폼색상 | x좌표 | y좌표\n")
    out.write("-" * 60 + "\n")

    for i, filename in enumerate(sampled_files, 1):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert("RGB")
        img_array = np.array(img)

        detections = model(img_path, verbose=False)[0]

        for box in detections.boxes:
            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            if area <= MIN_BOX_AREA:
                skipped_small += 1
                continue

            h = y2 - y1
            torso_y1 = y1 + int(h * 0.20)
            torso_y2 = y1 + int(h * 0.65)
            torso = img_array[torso_y1:torso_y2, x1:x2]

            if torso.size == 0:
                continue

            uniform_color = get_uniform_color(torso)

            ocr_results = reader.readtext(torso, allowlist='0123456789', min_size=5)

            for (pts, text, conf) in ocr_results:
                if not is_jersey_number(text):
                    continue

                pts = np.array(pts)
                ocr_x = int(np.mean(pts[:, 0])) + x1
                ocr_y = int(np.mean(pts[:, 1])) + torso_y1

                out.write(f"{filename} | {text} | {uniform_color} | {ocr_x} | {ocr_y}\n")
                result_count += 1

        if i % 10 == 0 or i == total:
            elapsed = time.time() - start_time
            speed = i / elapsed
            remaining = (total - i) / speed if speed > 0 else 0
            print(f"[{i}/{total}] 처리 중... | 누적 감지: {result_count}건 | 남은 시간: {remaining/60:.1f}분")

elapsed_total = time.time() - start_time
print(f"\n완료: {total}장 처리 ({elapsed_total/60:.1f}분 소요)")
print(f"감지된 등번호: 총 {result_count}건")
print(f"작은 박스 건너뜀: {skipped_small}건")
print(f"결과 저장: {output_file}")
