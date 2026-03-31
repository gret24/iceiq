#!/usr/bin/env python3
from ultralytics import YOLO
from PIL import Image, ImageDraw
import os

input_dir = "frames"
output_dir = "detected"
os.makedirs(output_dir, exist_ok=True)

model = YOLO("yolov8n.pt")

image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg")])
total = len(image_files)
total_persons = 0

for i, filename in enumerate(image_files, 1):
    img_path = os.path.join(input_dir, filename)
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    results = model(img_path, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # class 0 = person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
            total_persons += 1

    out_path = os.path.join(output_dir, filename)
    img.save(out_path)

    if i % 100 == 0 or i == total:
        print(f"[{i}/{total}] 처리 중...")

print(f"\n완료: 총 {total}장 처리 → detected/ 폴더 저장")
print(f"감지된 사람 수: 총 {total_persons}명")
