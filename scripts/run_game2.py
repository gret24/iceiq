"""
scripts/run_game2.py

game2.mp4 전체 경기 분석 파이프라인.
  - 아이기스 (흰색) vs 고양이글스 (검정)
  - CNN 팀 분류기 (models/team_classifier_cnn.pt)
  - ROI 필터 (data/roi_game2.json)
  - 아이기스 번호 인식 (pipeline/jersey_classifier.py)
  - 30초 간격 전체 경기 분석
  - 결과: data/results/game2/timeline.json
"""

import sys
import os
sys.path.insert(0, '/Users/hanhyeonseong/iceiq-dev')
os.chdir('/Users/hanhyeonseong/iceiq-dev')

import cv2
import json
import numpy as np
import time
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

from collections import Counter
from pathlib import Path
from torchvision import models, transforms
from pipeline.detection import YOLODetector
from pipeline.jersey_classifier import JerseyClassifier

# ── 설정 ─────────────────────────────────────────────────────────────────────
VIDEO_PATH      = 'data/videos/game2.mp4'
ROI_JSON        = 'data/roi_game2.json'
TEAM_CNN_PATH   = 'models/team_classifier_cnn.pt'
OUTPUT_DIR      = 'data/results/game2'
STRIDE_SEC      = 30          # 샘플링 간격 (초)
CONF_THRESHOLD  = 0.45        # YOLO 신뢰도 임계값
MIN_HEIGHT      = 35          # 최소 바운딩박스 높이 (px)
CNN_CONF_MIN    = 0.55        # CNN 팀 분류 최소 신뢰도

# ── 아이기스 로스터 (game2) ───────────────────────────────────────────────────
AIGIS_ROSTER = {4, 11, 12, 14, 25, 42, 47, 61, 94}
AIGIS_NAMES  = {
    4:  '윤지성', 11: '박리오', 12: '이준표', 14: '이봄',
    25: '김재원', 42: '남다름', 47: '한승원', 61: '김승후', 94: '김태윤'
}

# ── Device ───────────────────────────────────────────────────────────────────
_DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

# ── CNN 팀 분류기 ─────────────────────────────────────────────────────────────
_CNN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_team_cnn(path: str):
    """team_classifier_cnn.pt 로드. 반환: (model, idx_to_label)"""
    ckpt = torch.load(path, map_location=_DEVICE, weights_only=False)
    model = models.mobilenet_v3_small()
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, ckpt['n_classes'])
    model.load_state_dict(ckpt['model_state'])
    model = model.to(_DEVICE).eval()
    # game2: 아이기스=0(흰색), 고양이글스=1(검정)
    # 원본 label_map은 {0:'aigis', 1:'hockey_machine'} → 1을 'goyang_eagles'로 재매핑
    idx_to_label = {
        0: 'aigis',
        1: 'goyang_eagles'
    }
    print(f"팀 CNN 로드 완료: {path}")
    print(f"  클래스 매핑: {idx_to_label}")
    return model, idx_to_label


def classify_team_cnn(model, idx_to_label, crop_bgr: np.ndarray):
    """
    MobileNetV3-small CNN으로 팀 분류.
    Returns: (label: str, confidence: float)
    """
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 10:
        return 'unknown', 0.0

    # 상체 torso 영역 추출 (jersey_classifier.py와 동일한 전처리)
    y1, y2 = int(h * 0.20), int(h * 0.68)
    x1, x2 = int(w * 0.12), int(w * 0.88)
    torso = crop_bgr[y1:y2, x1:x2]
    if torso.size == 0:
        torso = crop_bgr

    try:
        resized = cv2.resize(torso, (96, 96), interpolation=cv2.INTER_CUBIC)
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = _CNN_TRANSFORM(img_rgb).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)[0]
            idx = probs.argmax().item()
            conf = probs[idx].item()
        label = idx_to_label.get(idx, 'unknown')
        return label, float(conf)
    except Exception:
        return 'unknown', 0.0


def load_roi(json_path: str) -> np.ndarray:
    """data/roi_game2.json 로드 → numpy polygon"""
    with open(json_path) as f:
        data = json.load(f)
    # {"game2": [[x,y], ...]} 형식
    pts = data.get('game2') or list(data.values())[0]
    return np.array(pts, dtype=np.int32)


def in_roi(roi_pts: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    """발 위치(bottom-center)가 ROI 안에 있는지 확인."""
    foot_x = (x1 + x2) / 2.0
    foot_y = float(y2)
    return cv2.pointPolygonTest(
        roi_pts.reshape(-1, 1, 2).astype(np.float32),
        (foot_x, foot_y), False
    ) >= 0


def main():
    t0 = time.time()

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    print("=== game2 분석 파이프라인 시작 ===")
    print(f"영상: {VIDEO_PATH}")
    print(f"팀: 아이기스(흰색) vs 고양이글스(검정)\n")

    detector  = YOLODetector(model_path='yolov8m.pt', device='mps')
    team_model, idx_to_label = load_team_cnn(TEAM_CNN_PATH)
    jersey_clf = JerseyClassifier()
    print(f"저지 분류기 로드 팀: {jersey_clf.available_teams}")

    # ── ROI 로드 ───────────────────────────────────────────────────────────────
    if not Path(ROI_JSON).exists():
        print(f"\n오류: ROI 파일이 없습니다 — {ROI_JSON}")
        print("먼저 scripts/detect_roi_game2.py를 실행하세요.")
        sys.exit(1)

    roi_pts = load_roi(ROI_JSON)
    print(f"\nROI 로드: {len(roi_pts)}개 꼭짓점 ({ROI_JSON})")

    # ── 비디오 열기 ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"오류: 비디오 열기 실패 — {VIDEO_PATH}")
        sys.exit(1)

    fps         = cap.get(cv2.CAP_PROP_FPS)
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_min   = total / fps / 60
    STRIDE      = int(fps * STRIDE_SEC)
    n_segments  = total // STRIDE

    print(f"FPS: {fps:.2f}, 총 프레임: {total}, 길이: {total_min:.1f}분")
    print(f"샘플링 간격: {STRIDE_SEC}초 ({n_segments}구간)\n")

    # ── 결과 저장소 ────────────────────────────────────────────────────────────
    team_stats = {
        'aigis':        Counter(),
        'goyang_eagles': Counter(),
    }
    timeline   = []
    low_conf_skipped = 0
    total_detections = 0
    roi_filtered     = 0

    # ── 메인 루프 ──────────────────────────────────────────────────────────────
    print(f"{'구간':>6}  {'시간':>6}  {'감지':>4}  {'ROI통과':>6}  {'아이기스':>7}  {'고양이글스':>8}")
    print("-" * 55)

    for frame_idx in range(0, total, STRIDE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        dets = detector.detect(frame)
        aigis_nums, goyang_nums = [], []
        seg_detections = 0
        seg_roi_pass   = 0

        for det in dets:
            if det['confidence'] < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, det['bbox'])
            if (y2 - y1) < MIN_HEIGHT:
                continue

            seg_detections += 1
            total_detections += 1

            # ROI 필터
            if not in_roi(roi_pts, x1, y1, x2, y2):
                roi_filtered += 1
                continue

            seg_roi_pass += 1
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue

            # CNN 팀 분류
            team, t_conf = classify_team_cnn(team_model, idx_to_label, crop)
            if t_conf < CNN_CONF_MIN:
                low_conf_skipped += 1
                continue

            # 저지 번호 인식 (아이기스만)
            num = None
            if team == 'aigis':
                num, n_conf = jersey_clf.predict(crop, 'aigis')
                if num and num not in AIGIS_ROSTER:
                    num = None  # 로스터에 없는 번호 제외

            if num is not None:
                team_stats[team][num] += 1
                if team == 'aigis':
                    aigis_nums.append(num)
                else:
                    goyang_nums.append(num)

        t_min = round(frame_idx / fps / 60, 1)
        timeline.append({
            't':      t_min,
            'aigis':  aigis_nums,
            'goyang': goyang_nums,
        })

        # 10구간마다 진행상황 출력
        seg_num = frame_idx // STRIDE
        if seg_num % 10 == 0 or seg_num < 5:
            print(f"{seg_num:>6}  {t_min:>5.1f}분  {seg_detections:>4}  {seg_roi_pass:>6}  "
                  f"{aigis_nums!s:>12}  {goyang_nums!s:>12}")

    cap.release()

    # ── 결과 저장 ──────────────────────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = f'{OUTPUT_DIR}/timeline.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(timeline, f, ensure_ascii=False, indent=2)
    print(f"\n타임라인 저장 → {out_path}")

    # ── 아이스타임 출력 ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    total_game_min = timeline[-1]['t'] if timeline else 0.0
    ice_time_factor = STRIDE_SEC / 60.0   # 30초 = 0.5분 단위

    print(f"\n{'='*55}")
    print(f"완료 (처리시간 {elapsed:.0f}초, 경기 {total_game_min:.0f}분)")
    print(f"총 감지: {total_detections}, ROI 필터: {roi_filtered}, "
          f"저신뢰도 스킵: {low_conf_skipped}")
    print(f"{'='*55}\n")

    print("=== 아이기스 아이스타임 ===")
    aigis_sorted = sorted(team_stats['aigis'].items(), key=lambda x: -x[1])
    if aigis_sorted:
        for num, cnt in aigis_sorted:
            minutes = cnt * ice_time_factor
            name = AIGIS_NAMES.get(num, '?')
            print(f"  #{num:>2} {name:8s}  {minutes:5.1f}분  (감지 {cnt}회)")
    else:
        print("  (감지된 선수 없음)")

    print("\n=== 고양이글스 아이스타임 ===")
    goyang_sorted = sorted(team_stats['goyang_eagles'].items(), key=lambda x: -x[1])
    if goyang_sorted:
        for num, cnt in goyang_sorted:
            minutes = cnt * ice_time_factor
            print(f"  #{num:>2}  {minutes:5.1f}분  (감지 {cnt}회)")
    else:
        print("  (감지된 선수 없음 — 고양이글스 CNN 모델 미적용)")

    # 팀별 통계 요약도 JSON에 추가 저장
    summary = {
        'video':        VIDEO_PATH,
        'total_minutes': total_game_min,
        'stride_sec':   STRIDE_SEC,
        'aigis_icetime':    {str(k): round(v * ice_time_factor, 1)
                             for k, v in team_stats['aigis'].items()},
        'goyang_icetime':   {str(k): round(v * ice_time_factor, 1)
                             for k, v in team_stats['goyang_eagles'].items()},
    }
    summary_path = f'{OUTPUT_DIR}/summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n요약 저장 → {summary_path}")


if __name__ == '__main__':
    main()
