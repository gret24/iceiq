"""
scripts/identify_by_signals.py

다중 신호 기반 아이기스 미확정 선수 식별
  대상: #28 임도준, #36 이민서, #47 한승원
  (jersey_classifier_aigis.pt 학습셋에 없어 OCR/CNN으로 미확정인 선수들)

Steps:
  1. 10초 간격 샘플링 — YOLOv8m + CNN 팀 분류 → 아이기스 감지 수집
     각 감지에서: bbox 크기(키), cx_norm(좌우 위치), jersey top-k(저신뢰도 포함)
  2. 확정(#4/#14/#61 high-conf) / 미확정 분리 → K-Means(k=3) 클러스터링
  3. 체형 분포: 클러스터별 평균 bbox height 비교
  4. 포지션 패턴: x좌표 분포 → 좌/중/우 경향
  5. 동시출현: 같은 프레임의 확정 선수 → 라인 패턴
  6. 소거법 + 신호 종합 → 최종 추천

출력: data/results/game2/signal_identification.json
"""

import sys
import os
sys.path.insert(0, '/Users/hanhyeonseong/iceiq-dev')
os.chdir('/Users/hanhyeonseong/iceiq-dev')

import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

from collections import Counter, defaultdict
from pathlib import Path
from torchvision import models, transforms

from pipeline.detection import YOLODetector
from pipeline.jersey_classifier import JerseyClassifier

# ── 경로 / 설정 ───────────────────────────────────────────────────────────────
VIDEO_PATH    = 'data/videos/game2.mp4'
ROI_JSON      = 'data/roi_game2.json'
TEAM_CNN_PATH = 'models/team_classifier_cnn.pt'
OUTPUT_DIR    = 'data/results/game2'
OUTPUT_FILE   = f'{OUTPUT_DIR}/signal_identification.json'

STRIDE_SEC       = 10     # 샘플링 간격 (초)
CONF_THRESHOLD   = 0.45   # YOLO 감지 신뢰도 최소값
MIN_HEIGHT_PX    = 35     # bbox 최소 높이
CNN_CONF_MIN     = 0.55   # 팀 분류 최소 신뢰도
JERSEY_CONF_HIGH = 0.60   # jersey 확정 임계값

# ── 아이기스 game2 로스터 ─────────────────────────────────────────────────────
ROSTER_NAMES = {4: '윤지성', 14: '이봄', 28: '임도준', 36: '이민서', 47: '한승원', 61: '김승후'}

# jersey_classifier_aigis.pt 가 알고 있는 번호 (학습 클래스)
CLASSIFIER_KNOWN = {4, 11, 12, 14, 25, 61, 94}
# game2 로스터와의 교집합 → 높은 신뢰도로 확정 가능한 번호
CONFIRMABLE      = CLASSIFIER_KNOWN & set(ROSTER_NAMES)   # {4, 14, 61}

# 식별 목표 선수
TARGET_NUMBERS = [28, 36, 47]
TARGET_NAMES   = {28: '임도준', 36: '이민서', 47: '한승원'}

_DEVICE = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

# ─────────────────────────────────────────────────────────────────────────────
# 팀 분류 CNN (run_game2.py 동일)
# ─────────────────────────────────────────────────────────────────────────────
_CNN_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_team_cnn(path: str):
    ckpt = torch.load(path, map_location=_DEVICE, weights_only=False)
    m = models.mobilenet_v3_small()
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, ckpt['n_classes'])
    m.load_state_dict(ckpt['model_state'])
    m = m.to(_DEVICE).eval()
    idx_to_label = {0: 'aigis', 1: 'goyang_eagles'}
    print(f"팀 CNN 로드: {path} (클래스: {idx_to_label})")
    return m, idx_to_label


def classify_team_cnn(model, idx_to_label, crop_bgr: np.ndarray):
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 10:
        return 'unknown', 0.0
    y1, y2 = int(h * 0.20), int(h * 0.68)
    x1, x2 = int(w * 0.12), int(w * 0.88)
    torso = crop_bgr[y1:y2, x1:x2]
    if torso.size == 0:
        torso = crop_bgr
    try:
        resized  = cv2.resize(torso, (96, 96), interpolation=cv2.INTER_CUBIC)
        img_rgb  = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = _CNN_TRANSFORM(img_rgb).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
            idx   = probs.argmax().item()
            conf  = probs[idx].item()
        return idx_to_label.get(idx, 'unknown'), float(conf)
    except Exception:
        return 'unknown', 0.0


def load_roi(json_path: str) -> np.ndarray:
    with open(json_path) as f:
        data = json.load(f)
    pts = data.get('game2') or list(data.values())[0]
    return np.array(pts, dtype=np.int32)


def in_roi(roi_pts: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
    foot_x = (x1 + x2) / 2.0
    foot_y = float(y2)
    return cv2.pointPolygonTest(
        roi_pts.reshape(-1, 1, 2).astype(np.float32),
        (foot_x, foot_y), False
    ) >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 전체 경기 데이터 수집
# ─────────────────────────────────────────────────────────────────────────────

def collect_detections(video_path, roi_pts, team_model, idx_to_label,
                       jersey_clf, stride_sec, frame_w, frame_h):
    """
    10초 간격 샘플링 → 아이기스 감지 레코드 리스트 반환.

    각 레코드:
        frame, time_min, bbox, cx, cy, h_px,
        cx_norm (0=왼쪽, 1=오른쪽),
        team_conf, jersey_topk [(num, conf)],
        identified_num, identified_conf, status
    """
    detector = YOLODetector(model_path='yolov8m.pt', device='mps')
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오 열기 실패: {video_path}")

    fps       = cap.get(cv2.CAP_PROP_FPS)
    n_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_min = n_frames / fps / 60
    stride    = max(1, int(fps * stride_sec))
    n_segs    = n_frames // stride

    print(f"FPS: {fps:.2f} | 총 {n_frames}프레임 ({total_min:.1f}분) | "
          f"{stride_sec}초 간격 → {n_segs}구간\n")

    all_dets = []

    for frame_idx in range(0, n_frames, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        t_min = frame_idx / fps / 60
        dets  = detector.detect(frame)
        frame_records = []

        for det in dets:
            if det['confidence'] < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, det['bbox'])
            h_px = y2 - y1
            if h_px < MIN_HEIGHT_PX:
                continue
            if not in_roi(roi_pts, x1, y1, x2, y2):
                continue

            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue

            # 팀 분류
            team, t_conf = classify_team_cnn(team_model, idx_to_label, crop)
            if t_conf < CNN_CONF_MIN or team != 'aigis':
                continue

            # jersey 분류 — top-3, 낮은 신뢰도 포함
            topk = jersey_clf.predict_top_k(crop, 'aigis', k=3)
            if topk is None:
                topk = []

            # 확정 여부: CONFIRMABLE 번호 + 높은 신뢰도
            identified_num  = None
            identified_conf = 0.0
            for num, conf in topk:
                if num in CONFIRMABLE and conf >= JERSEY_CONF_HIGH:
                    identified_num  = num
                    identified_conf = conf
                    break

            cx = (x1 + x2) / 2.0

            frame_records.append({
                'frame':           frame_idx,
                'time_min':        round(t_min, 2),
                'bbox':            [x1, y1, x2, y2],
                'cx':              round(cx, 1),
                'cy':              round((y1 + y2) / 2.0, 1),
                'h_px':            h_px,
                'cx_norm':         round(cx / max(frame_w, 1), 4),
                'team_conf':       round(t_conf, 3),
                'jersey_topk':     [(n, round(c, 3)) for n, c in topk],
                'identified_num':  identified_num,
                'identified_conf': round(identified_conf, 3),
                'status':          'identified' if identified_num is not None else 'unidentified',
            })

        all_dets.extend(frame_records)

        seg = frame_idx // stride
        if seg % 30 == 0 or seg < 3:
            ids  = [r['identified_num'] for r in frame_records if r['identified_num']]
            unid = len(frame_records) - len(ids)
            print(f"  {t_min:5.1f}분 | 아이기스 {len(frame_records):2d}명 "
                  f"| 확정 {ids} | 미확정 {unid}명")

    cap.release()
    return all_dets, total_min


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: K-Means 클러스터링 (미확정만)
# ─────────────────────────────────────────────────────────────────────────────

def cluster_unidentified(unid_dets, total_min, n_clusters=3):
    """
    미확정 감지를 K-Means(k=3)으로 분류.
    특징: [cx_norm, 표준화된 h_px, 시간 정규화]
    sklearn 없으면 cx_norm 분위수 폴백.
    """
    if len(unid_dets) < n_clusters:
        # 감지 수 부족: 순서대로 할당
        return {i: i % n_clusters for i in range(len(unid_dets))}

    heights = [d['h_px'] for d in unid_dets]
    h_mean  = float(np.mean(heights))
    h_std   = max(float(np.std(heights)), 1.0)

    # 특징 벡터 — x위치(주 신호), 키(약한 신호), 시간(동질성 보조)
    features = np.array([
        [
            d['cx_norm'],                                   # x 위치 (0~1)
            (d['h_px'] - h_mean) / h_std * 0.4,            # 표준화 키
            d['time_min'] / max(total_min, 1.0) * 0.3,     # 시간
        ]
        for d in unid_dets
    ], dtype=np.float32)

    try:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=n_clusters, n_init=30, random_state=42)
        labels = km.fit_predict(features)
        print("  K-Means(sklearn) 클러스터링 완료")
        return {i: int(labels[i]) for i in range(len(unid_dets))}

    except ImportError:
        print("  sklearn 없음 — cx_norm 분위수 폴백 클러스터링")
        sorted_idx = sorted(range(len(unid_dets)),
                            key=lambda i: unid_dets[i]['cx_norm'])
        n = len(sorted_idx)
        labels = {}
        for rank, idx in enumerate(sorted_idx):
            labels[idx] = min(int(rank / n * n_clusters), n_clusters - 1)
        return labels


# ─────────────────────────────────────────────────────────────────────────────
# Step 3-4: 클러스터 특성 분석 (체형 + 포지션 + 동시출현)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_clusters(all_dets, unid_dets, cluster_labels, total_min):
    """
    각 클러스터에 대해 분석:
      - 체형: 평균 bbox height
      - 포지션: cx_norm 분포 → 좌/중/우 경향
      - 동시출현: 같은 프레임의 확정 선수 번호 집계
      - CLF 힌트: 저신뢰도 jersey 예측 분포
    """
    n_clusters = max(cluster_labels.values(), default=0) + 1
    clusters   = {}

    for cid in range(n_clusters):
        idxs = [i for i, lbl in cluster_labels.items() if lbl == cid]
        dets = [unid_dets[i] for i in idxs]

        if not dets:
            continue

        heights  = [d['h_px']    for d in dets]
        cx_norms = [d['cx_norm'] for d in dets]
        times    = [d['time_min'] for d in dets]

        mean_h  = float(np.mean(heights))
        std_h   = float(np.std(heights))
        mean_cx = float(np.mean(cx_norms))

        # 포지션 경향
        left_pct  = sum(1 for x in cx_norms if x < 0.40) / len(cx_norms)
        right_pct = sum(1 for x in cx_norms if x > 0.60) / len(cx_norms)
        if left_pct > 0.50:
            pos_label = 'LEFT (수비적)'
        elif right_pct > 0.50:
            pos_label = 'RIGHT (공격적)'
        else:
            pos_label = 'CENTER (중앙)'

        # 동시출현: 이 클러스터 감지가 있는 프레임에서의 확정 선수
        frames_in_cluster = {d['frame'] for d in dets}
        cooc: Counter = Counter()
        for d in all_dets:
            if d['status'] == 'identified' and d['frame'] in frames_in_cluster:
                cooc[d['identified_num']] += 1

        # CLF 힌트 (conf >= 0.15 이상이면 집계)
        clf_hints: Counter = Counter()
        for d in dets:
            for num, conf in d['jersey_topk']:
                if conf >= 0.15:
                    clf_hints[num] += 1

        # 최빈 jersey 예측 (어떤 번호로 잘못 분류되는지)
        clf_hints_top5 = dict(clf_hints.most_common(5))

        clusters[cid] = {
            'detection_count':            len(dets),
            'mean_height_px':             round(mean_h, 1),
            'std_height_px':              round(std_h, 1),
            'mean_cx_norm':               round(mean_cx, 3),
            'position_tendency':          pos_label,
            'left_pct':                   round(left_pct, 3),
            'right_pct':                  round(right_pct, 3),
            'time_range_min':             [round(min(times), 1), round(max(times), 1)],
            'cooccurrence_with_identified': dict(cooc.most_common()),
            'clf_hints_low_conf':         clf_hints_top5,
        }

    return clusters


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: 신호 종합 → 점수 산정 → 소거법 할당
# ─────────────────────────────────────────────────────────────────────────────

def score_and_recommend(clusters, identified_stats):
    """
    각 클러스터에 대해 #28/#36/#47 점수 산정 후 소거법(greedy)으로 최적 할당.

    점수 기여 요소:
      A. CLF 힌트 비율   : 저신뢰도 예측에서 특정 번호가 많이 나오면 가산
      B. 체형 상대 순위  : 클러스터간 키 순위 (roster 키 정보 없으면 약신호)
      C. 포지션 패턴     : 설명용 (정량 가산 없음, 이유 기록용)
      D. 동시출현 라인   : 확정 선수와의 동시출현 패턴 (이유 기록용)
    """
    if not clusters:
        return {}, {}

    candidate_nums = TARGET_NUMBERS[:len(clusters)]
    n_c = len(clusters)

    # 체형 순위: 평균 키 큰 순 (0 = 가장 큼)
    height_rank = {
        cid: rank
        for rank, cid in enumerate(
            sorted(clusters, key=lambda c: -clusters[c]['mean_height_px'])
        )
    }

    scores  = {cid: {n: 0.0 for n in candidate_nums} for cid in clusters}
    reasons = {cid: {n: []  for n in candidate_nums} for cid in clusters}

    for cid, cinfo in clusters.items():
        total_clf_hints = sum(cinfo['clf_hints_low_conf'].values()) or 1

        for num in candidate_nums:
            sc   = 0.0
            rsns = []

            # A. CLF 힌트
            hint_cnt = cinfo['clf_hints_low_conf'].get(num, 0)
            hint_ratio = hint_cnt / total_clf_hints
            if hint_ratio >= 0.25:
                sc += 2.5 * hint_ratio
                rsns.append(
                    f"CLF 힌트 {hint_cnt}회/{total_clf_hints}회 "
                    f"({hint_ratio*100:.0f}%) → +{2.5*hint_ratio:.2f}"
                )
            elif hint_ratio > 0.0:
                sc += 0.5 * hint_ratio
                rsns.append(f"CLF 힌트 미약 {hint_cnt}회 ({hint_ratio*100:.0f}%)")

            # B. 체형 상대 순위 (약한 신호)
            rank = height_rank[cid]
            body_bonus = max(0, (n_c - 1 - rank)) * 0.25
            sc += body_bonus
            rsns.append(
                f"체형 순위 {rank+1}/{n_c} "
                f"(평균 {cinfo['mean_height_px']:.0f}px) → +{body_bonus:.2f}"
            )

            # C. 포지션 패턴 (기록용)
            rsns.append(f"포지션 분포: {cinfo['position_tendency']} "
                        f"(cx_norm={cinfo['mean_cx_norm']:.2f})")

            # D. 동시출현 (기록용)
            cooc = cinfo['cooccurrence_with_identified']
            if cooc:
                top2 = sorted(cooc.items(), key=lambda x: -x[1])[:2]
                cooc_str = ', '.join(f"#{n}×{c}" for n, c in top2)
                rsns.append(f"동시출현 확정선수: {cooc_str}")

            scores[cid][num]  = round(sc, 3)
            reasons[cid][num] = rsns

    # ── 소거법 (greedy) — 가장 높은 점수부터 확정 ──
    assignment       = {}   # cid → number
    remaining_cids   = set(clusters.keys())
    remaining_nums   = set(candidate_nums)

    all_pairs = sorted(
        [(scores[cid][n], cid, n)
         for cid in remaining_cids for n in remaining_nums],
        reverse=True,
    )
    for sc, cid, num in all_pairs:
        if cid in remaining_cids and num in remaining_nums:
            assignment[cid] = num
            remaining_cids.discard(cid)
            remaining_nums.discard(num)

    # 혹시 남은 쌍 처리
    for cid in list(remaining_cids):
        for num in list(remaining_nums):
            assignment[cid] = num
            remaining_cids.discard(cid)
            remaining_nums.discard(num)
            break

    # ── 추천 결과 ──
    recommendations = {}
    for cid, num in assignment.items():
        name    = TARGET_NAMES.get(num, '?')
        cinfo   = clusters[cid]
        sc      = scores[cid][num]
        conf_lv = '높음' if sc >= 1.5 else ('중간' if sc >= 0.5 else '낮음')

        ranked = sorted(candidate_nums, key=lambda n: -scores[cid][n])

        # 설명 문장 생성
        rsn_text = '; '.join(reasons[cid][num])
        rec_text = (
            f"클러스터 {cid} → #{num} {name}일 가능성 높음 "
            f"[신뢰도: {conf_lv}, score={sc:.2f}] "
            f"(이유: {rsn_text})"
        )

        recommendations[cid] = {
            'assigned_number':   num,
            'assigned_name':     name,
            'score':             sc,
            'confidence_level':  conf_lv,
            'candidates_ranked': [
                {
                    'number':  n,
                    'name':    TARGET_NAMES.get(n, '?'),
                    'score':   scores[cid][n],
                    'reasons': reasons[cid][n],
                }
                for n in ranked
            ],
            'recommendation_text': rec_text,
        }

    score_matrix = {
        str(cid): {str(n): scores[cid][n] for n in candidate_nums}
        for cid in clusters
    }
    return recommendations, score_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("IceIQ 다중신호 선수 식별: #28 임도준 / #36 이민서 / #47 한승원")
    print("=" * 65)

    # ── 모델 / ROI 로드 ──
    team_model, idx_to_label = load_team_cnn(TEAM_CNN_PATH)
    # min_conf=0.0 → predict_top_k는 conf 무관 반환, 확정 판단은 코드에서
    jersey_clf = JerseyClassifier(min_conf=0.0)
    roi_pts    = load_roi(ROI_JSON)
    print(f"ROI: {len(roi_pts)}개 꼭짓점")

    # 영상 크기
    cap         = cv2.VideoCapture(VIDEO_PATH)
    frame_w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"영상: {frame_w}×{frame_h}\n")

    # ─────────────────────────────────────────────
    print("─" * 40)
    print("Step 1: 데이터 수집 (10초 간격)")
    print("─" * 40)
    all_dets, total_min = collect_detections(
        VIDEO_PATH, roi_pts, team_model, idx_to_label,
        jersey_clf, STRIDE_SEC, frame_w, frame_h,
    )

    identified_dets = [d for d in all_dets if d['status'] == 'identified']
    unid_dets       = [d for d in all_dets if d['status'] == 'unidentified']

    print(f"\n총 아이기스 감지: {len(all_dets)}")
    print(f"  ├ 확정: {len(identified_dets)}")
    print(f"  └ 미확정: {len(unid_dets)}")

    # 확정 선수 통계
    identified_stats = {}
    for num in CONFIRMABLE:
        dets_n  = [d for d in identified_dets if d['identified_num'] == num]
        if not dets_n:
            continue
        heights = [d['h_px']    for d in dets_n]
        xs      = [d['cx_norm'] for d in dets_n]
        identified_stats[num] = {
            'name':            ROSTER_NAMES.get(num, '?'),
            'detection_count': len(dets_n),
            'mean_height_px':  round(float(np.mean(heights)), 1),
            'mean_cx_norm':    round(float(np.mean(xs)), 3),
        }

    print(f"\n[확정 선수 현황]")
    for num, st in sorted(identified_stats.items()):
        print(f"  #{num:>2} {st['name']:6s} | {st['detection_count']:4d}회 | "
              f"평균키 {st['mean_height_px']:.0f}px | cx_norm {st['mean_cx_norm']:.2f}")

    # ─────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("Step 2: 미확정 K-Means 클러스터링 (k=3)")
    print("─" * 40)
    cluster_labels = cluster_unidentified(unid_dets, total_min, n_clusters=3)

    for cid, cnt in sorted(Counter(cluster_labels.values()).items()):
        print(f"  클러스터 {cid}: {cnt}개 감지")

    # ─────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("Step 3-4: 체형 / 포지션 / 동시출현 분석")
    print("─" * 40)
    clusters = analyze_clusters(all_dets, unid_dets, cluster_labels, total_min)

    print(f"\n[확정 선수 평균 키 기준선]")
    for num, st in sorted(identified_stats.items()):
        print(f"  #{num} {st['name']:6s}: {st['mean_height_px']:.0f}px")

    print(f"\n[미확정 클러스터 분석]")
    for cid, cinfo in sorted(clusters.items()):
        print(f"\n  ┌─ 클러스터 {cid} ─────────────────────────────")
        print(f"  │ 감지수: {cinfo['detection_count']}  "
              f"평균키: {cinfo['mean_height_px']:.0f}px (±{cinfo['std_height_px']:.0f})")
        print(f"  │ 포지션: {cinfo['position_tendency']}  "
              f"cx_norm={cinfo['mean_cx_norm']:.2f}  "
              f"(좌:{cinfo['left_pct']*100:.0f}% 우:{cinfo['right_pct']*100:.0f}%)")
        print(f"  │ 활동시간: {cinfo['time_range_min']}")
        print(f"  │ 동시출현 확정선수: {cinfo['cooccurrence_with_identified']}")
        print(f"  └ CLF 힌트(저신뢰도): {cinfo['clf_hints_low_conf']}")

    # ─────────────────────────────────────────────
    print("\n" + "─" * 40)
    print("Step 5: 소거법 + 신호 종합 → 최종 추천")
    print("─" * 40)
    recommendations, score_matrix = score_and_recommend(clusters, identified_stats)

    print()
    for cid, rec in sorted(recommendations.items()):
        print(f"  [{rec['recommendation_text']}]")
        print(f"   후보 순위: " +
              ", ".join(f"#{c['number']} {c['name']}({c['score']:.2f})"
                        for c in rec['candidates_ranked']))
        print()

    # ─────────────────────────────────────────────
    # 결과 저장
    # ─────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    result = {
        'video':       VIDEO_PATH,
        'total_minutes': round(total_min, 1),
        'stride_sec':  STRIDE_SEC,
        'summary': {
            'total_aigis_detections':  len(all_dets),
            'identified_detections':   len(identified_dets),
            'unidentified_detections': len(unid_dets),
            'confirmable_numbers':     sorted(CONFIRMABLE),
            'target_numbers':          TARGET_NUMBERS,
        },
        'identified_players':   {str(k): v for k, v in identified_stats.items()},
        'clusters':             {str(k): v for k, v in clusters.items()},
        'score_matrix':         score_matrix,
        'recommendations':      {str(k): v for k, v in recommendations.items()},
        'final_assignment': {
            str(cid): {
                'number':     rec['assigned_number'],
                'name':       rec['assigned_name'],
                'confidence': rec['confidence_level'],
                'score':      rec['score'],
            }
            for cid, rec in recommendations.items()
        },
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 65)
    print(f"결과 저장 → {OUTPUT_FILE}")
    print("=" * 65)

    print("\n[최종 요약]")
    for cid, rec in sorted(recommendations.items()):
        print(f"  클러스터 {cid} → #{rec['assigned_number']} {rec['assigned_name']} "
              f"[{rec['confidence_level']}] (score={rec['score']:.2f})")


if __name__ == '__main__':
    main()
