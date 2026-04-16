"""
scripts/detect_roi_game2.py

game2.mp4의 ROI(링크 빙판 영역)를 자동 감지하여 data/roi_game2.json으로 저장.

방법:
1. 10분, 20분, 30분 프레임 3장 추출
2. 각 프레임에서 얼음판(밝고 채도 낮은 영역) HSV 마스크 생성
3. 3장의 교집합(bitwise AND)으로 안정적인 ROI 폴리곤 계산
4. data/roi_game2.json 저장 ({"game2": [[x,y], ...]})
5. 검증용 annotated 이미지 /tmp/roi_game2_check.jpg 저장
"""

import sys
import os
sys.path.insert(0, '/Users/hanhyeonseong/iceiq-dev')
os.chdir('/Users/hanhyeonseong/iceiq-dev')

import cv2
import json
import numpy as np
from pathlib import Path

VIDEO_PATH = 'data/videos/game2.mp4'
OUTPUT_JSON = 'data/roi_game2.json'
CHECK_IMAGE = '/tmp/roi_game2_check.jpg'

# 샘플 타임스탬프 (분)
SAMPLE_MINUTES = [10, 20, 30]


def detect_ice_mask(frame: np.ndarray) -> np.ndarray:
    """
    얼음판 영역 이진 마스크 생성.
    얼음 특성: 높은 밝기(V > 150), 낮은 채도(S < 60)
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 밝고 채도 낮은 영역 = 얼음
    mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 60, 255]))

    # 상단 15% 제거 (관중석/스코어보드)
    mask[:int(h * 0.15), :] = 0

    # 하단 5% 제거 (하단 광고판 등)
    mask[int(h * 0.95):, :] = 0

    # 모폴로지 정리: 작은 구멍 채우고 잡음 제거
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    return mask


def mask_to_polygon(mask: np.ndarray, frame_shape) -> np.ndarray:
    """마스크에서 가장 큰 컨투어의 근사 폴리곤 반환."""
    h, w = frame_shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("  경고: 컨투어 없음 → 전체 프레임 사용")
        return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.int32)

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    print(f"  최대 컨투어 면적: {area:.0f} px² ({area/(w*h)*100:.1f}% of frame)")

    if area < w * h * 0.10:
        print("  경고: 면적 < 10% → 전체 프레임 사용")
        return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.int32)

    eps = 0.015 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, eps, True)
    if len(approx) < 4:
        approx = cv2.convexHull(largest)

    return approx.reshape(-1, 2)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"오류: 비디오를 열 수 없습니다 — {VIDEO_PATH}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_min = total_frames / fps / 60
    print(f"비디오: {VIDEO_PATH}")
    print(f"FPS: {fps:.2f}, 총 프레임: {total_frames}, 길이: {total_min:.1f}분")

    # 1) 각 타임스탬프에서 프레임 추출 & 마스크 생성
    frames = []
    masks = []
    actual_minutes = []

    for minute in SAMPLE_MINUTES:
        frame_idx = int(minute * 60 * fps)
        if frame_idx >= total_frames:
            print(f"  {minute}분 프레임이 영상 길이 초과 — 스킵")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  {minute}분 프레임 읽기 실패 — 스킵")
            continue

        print(f"\n[{minute}분 프레임 ({frame_idx})]")
        mask = detect_ice_mask(frame)
        ice_pixels = np.count_nonzero(mask)
        print(f"  얼음 마스크 픽셀: {ice_pixels}")

        frames.append(frame)
        masks.append(mask)
        actual_minutes.append(minute)

    cap.release()

    if len(masks) == 0:
        print("오류: 처리할 프레임이 없습니다.")
        sys.exit(1)

    # 2) 3장 마스크의 교집합 (bitwise AND)
    print(f"\n교집합 마스크 계산 ({len(masks)}장)...")
    intersection_mask = masks[0].copy()
    for m in masks[1:]:
        intersection_mask = cv2.bitwise_and(intersection_mask, m)

    # 교집합이 너무 작으면 합집합(OR)으로 폴백
    h, w = frames[0].shape[:2]
    intersection_pixels = np.count_nonzero(intersection_mask)
    print(f"  교집합 픽셀: {intersection_pixels} ({intersection_pixels/(w*h)*100:.1f}%)")

    if intersection_pixels < w * h * 0.08:
        print("  교집합이 너무 작음 → 합집합(OR)으로 폴백")
        intersection_mask = masks[0].copy()
        for m in masks[1:]:
            intersection_mask = cv2.bitwise_or(intersection_mask, m)
        # 폴백 시 추가 erosion으로 노이즈 제거
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        intersection_mask = cv2.morphologyEx(intersection_mask, cv2.MORPH_ERODE, k2)

    # 3) 폴리곤 계산
    poly = mask_to_polygon(intersection_mask, frames[0].shape)
    print(f"\nROI 폴리곤: {len(poly)}개 꼭짓점")
    print(f"  x 범위: {poly[:, 0].min()} ~ {poly[:, 0].max()}")
    print(f"  y 범위: {poly[:, 1].min()} ~ {poly[:, 1].max()}")

    # 4) JSON 저장
    roi_data = {"game2": poly.tolist()}
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(roi_data, f, ensure_ascii=False, indent=2)
    print(f"\nROI 저장 → {OUTPUT_JSON}")

    # 5) 검증 이미지 생성
    # 3장의 프레임을 가로로 이어붙이고 각각 ROI 표시 + 교집합 마스크 표시
    vis_frames = []
    poly_cv = poly.reshape(-1, 1, 2).astype(np.int32)

    for i, (frame, minute) in enumerate(zip(frames, actual_minutes)):
        vis = frame.copy()
        # ROI 폴리곤 그리기
        cv2.polylines(vis, [poly_cv], isClosed=True, color=(0, 255, 0), thickness=3)
        # 개별 마스크 컨투어 (파란색)
        ctrs, _ = cv2.findContours(masks[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if ctrs:
            largest_c = max(ctrs, key=cv2.contourArea)
            cv2.drawContours(vis, [largest_c], -1, (255, 100, 0), 2)
        # 라벨
        cv2.putText(vis, f"{minute}min", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3)
        vis_frames.append(vis)

    # 교집합 마스크 시각화 (마지막 패널)
    if frames:
        mask_vis = cv2.cvtColor(intersection_mask, cv2.COLOR_GRAY2BGR)
        cv2.polylines(mask_vis, [poly_cv], isClosed=True, color=(0, 255, 0), thickness=3)
        cv2.putText(mask_vis, "Intersection", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        vis_frames.append(mask_vis)

    # 동일 크기로 리사이즈
    target_h = 360
    resized = []
    for vf in vis_frames:
        rh, rw = vf.shape[:2]
        scale = target_h / rh
        resized.append(cv2.resize(vf, (int(rw * scale), target_h)))

    combined = np.hstack(resized)
    cv2.imwrite(CHECK_IMAGE, combined)
    print(f"검증 이미지 저장 → {CHECK_IMAGE}")
    print("\n완료!")


if __name__ == '__main__':
    main()
