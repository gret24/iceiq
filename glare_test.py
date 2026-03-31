#!/usr/bin/env python3
"""
선수 감지율 개선 테스트 - 다양한 방법 비교
목표: 85% → 88%+
"""
import os, cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
MIN_BOX_AREA = 5000

all_frames = sorted(os.listdir(FRAMES_DIR))
step = max(1, len(all_frames) // 20)
samples = all_frames[::step][:20]

model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))

def count_detections(img_source, conf=0.3, iou=0.45):
    results = model(img_source, classes=[0], verbose=False, conf=conf, iou=iou)[0]
    boxes = results.boxes
    count = 0
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if (x2-x1)*(y2-y1) >= MIN_BOX_AREA:
                count += 1
    return count

configs = [
    ("원본 (conf=0.3)", 0.3, 0.45),
    ("conf=0.25",       0.25, 0.45),
    ("conf=0.2",        0.20, 0.45),
    ("conf=0.2+iou=0.5",0.20, 0.50),
]

results_table = {name: {"detected": 0, "total": 0} for name, _, _ in configs}

print(f"{'프레임':<23}", end="")
for name, _, _ in configs:
    print(f"  {name:>15}", end="")
print()
print("-" * 90)

for fname in samples:
    fpath = os.path.join(FRAMES_DIR, fname)
    print(f"  {fname:<21}", end="")
    for name, conf, iou in configs:
        cnt = count_detections(fpath, conf, iou)
        if cnt > 0:
            results_table[name]["detected"] += 1
        results_table[name]["total"] += cnt
        print(f"  {cnt:>15}", end="")
    print()

print("\n========== 감지율 비교 ==========")
for name, _, _ in configs:
    rate = results_table[name]["detected"] / len(samples) * 100
    total = results_table[name]["total"]
    goal = "✅" if rate >= 88 else "❌"
    print(f"  {name:<20}: {rate:.1f}%  (총 {total}명)  {goal}")
print("==================================")
