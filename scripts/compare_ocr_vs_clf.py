"""
OCR vs Classifier 동시 비교 스크립트
Usage:
    python3 scripts/compare_ocr_vs_clf.py <video_path>
    또는 텔레그램: /compare <영상경로>
"""
import sys, os, cv2, numpy as np, json, time, easyocr, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.detection import YOLODetector
from pipeline.team_classification import HockeyTeamClassifier, PRESET_HOCKEY_MACHINE_AIGIS
from pipeline.jersey_classifier import JerseyClassifier
from torchvision import models, transforms
from collections import Counter, defaultdict
import warnings; warnings.filterwarnings('ignore')

# ── 설정 ──────────────────────────────────────────────────────────────────────
DEVICE    = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
WHITE_LO  = np.array([0, 0, 160], dtype=np.uint8)
WHITE_HI  = np.array([180, 50, 255], dtype=np.uint8)
WHITE_THR = 0.12   # 아이기스(흰색) 판별 threshold
STRIDE_SEC = 30    # 분석 간격 (초)
AIGIS_ROSTER = {4,11,12,14,25,42,47,61,94}
AIGIS_NAMES  = {4:'윤지성',11:'박리오',12:'이준표',14:'이봄',25:'김재원',
                42:'남다름',47:'한승원',61:'김승후',94:'김태윤'}

# ── 모듈 초기화 ───────────────────────────────────────────────────────────────
def init_modules(roi_path=None):
    detector   = YOLODetector(model_path='yolov8m.pt', device='mps')
    clf_jersey = JerseyClassifier(min_conf=0.55)
    ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    roi_pts = None
    if roi_path and os.path.exists(roi_path):
        roi_data = json.load(open(roi_path))
        key = list(roi_data.keys())[0]
        roi_pts = np.array(roi_data[key], dtype=np.int32)
    return detector, clf_jersey, ocr_reader, roi_pts

# ── 유틸 ──────────────────────────────────────────────────────────────────────
def in_roi(roi_pts, x1, y1, x2, y2):
    if roi_pts is None: return True
    return cv2.pointPolygonTest(roi_pts, ((x1+x2)/2, (y1+y2)/2), False) >= 0

def white_ratio(crop):
    h,w = crop.shape[:2]
    t = crop[int(h*0.20):int(h*0.65), int(w*0.15):int(w*0.85)]
    if t.size == 0: return 0
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, WHITE_LO, WHITE_HI).sum() / 255 / (t.shape[0]*t.shape[1])

def ocr_jersey(reader, crop):
    h,w = crop.shape[:2]
    if h<10 or w<10: return None, 0
    scale = max(128/h, 1.0)
    up = cv2.resize(crop, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    rh,rw = up.shape[:2]
    torso = up[int(rh*0.25):int(rh*0.70), int(rw*0.10):int(rw*0.90)]
    if torso.size == 0: return None, 0
    gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(3.0,(4,4)); enh = clahe.apply(gray)
    b = cv2.adaptiveThreshold(enh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,8)
    p = cv2.copyMakeBorder(b,12,12,12,12,cv2.BORDER_CONSTANT,value=255)
    best, bc = None, 0
    for img in [p, cv2.bitwise_not(p)]:
        for _,text,conf in reader.readtext(img, allowlist='0123456789', detail=1,
                                            paragraph=False, min_size=8):
            text = text.strip()
            if text.isdigit() and 1<=int(text)<=99 and conf>bc:
                best, bc = int(text), conf
    return best, bc

# ── 메인 분석 ─────────────────────────────────────────────────────────────────
def run(video_path, roi_path=None, output_dir=None):
    print(f'\n=== OCR vs Classifier 비교 ===')
    print(f'영상: {os.path.basename(video_path)}')

    detector, clf_jersey, ocr_reader, roi_pts = init_modules(roi_path)

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur   = total / fps / 60
    print(f'길이: {dur:.1f}분 | FPS: {fps:.0f} | 분석 간격: {STRIDE_SEC}초')

    STRIDE = int(fps * STRIDE_SEC)
    # 카운터
    clf_aigis   = Counter()
    clf_hm      = Counter()
    ocr_aigis   = Counter()
    ocr_hm      = Counter()
    clf_hits, ocr_hits = 0, 0
    total_dets  = 0

    t0 = time.time()
    for frame_idx in range(0, total, STRIDE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue

        dets = detector.detect(frame)
        for i, det in enumerate(dets):
            if det['confidence'] < 0.45: continue
            x1,y1,x2,y2 = map(int, det['bbox'])
            if (y2-y1) < 35: continue
            if not in_roi(roi_pts, x1,y1,x2,y2): continue
            total_dets += 1

            crop = frame[max(0,y1):y2, max(0,x1):x2]
            wr   = white_ratio(crop)
            team = 'aigis' if wr >= WHITE_THR else 'hockey_machine'

            # ── Classifier ──────────────────────────────────────────────
            clf_num, clf_conf = clf_jersey.predict(crop, team)
            if clf_num:
                clf_hits += 1
                if team == 'aigis': clf_aigis[clf_num] += 1
                else:               clf_hm[clf_num]    += 1

            # ── OCR ─────────────────────────────────────────────────────
            ocr_num, ocr_conf = ocr_jersey(ocr_reader, crop)
            if ocr_num:
                # roster constraint (아이기스만)
                if team == 'aigis' and ocr_num not in AIGIS_ROSTER:
                    ocr_num = None
            if ocr_num:
                ocr_hits += 1
                if team == 'aigis': ocr_aigis[ocr_num] += 1
                else:               ocr_hm[ocr_num]    += 1

    cap.release()
    elapsed = time.time() - t0

    # ── 리포트 ────────────────────────────────────────────────────────────
    segments = total // STRIDE
    print(f'\n처리: {segments}구간 | 감지: {total_dets}건 | 소요: {elapsed:.0f}초')
    print(f'\n{"":30} {"Classifier":>12} {"OCR":>12}')
    print(f'{"번호 인식률":30} {clf_hits/max(total_dets,1)*100:>11.1f}% {ocr_hits/max(total_dets,1)*100:>11.1f}%')

    print(f'\n=== 아이기스 아이스타임 ===')
    print(f'{"선수":20} {"Classifier":>12} {"OCR":>12}')
    all_nums = sorted(set(list(clf_aigis.keys()) + list(ocr_aigis.keys())))
    for num in all_nums:
        name = AIGIS_NAMES.get(num, f'#{num}')
        ct = clf_aigis.get(num, 0) * STRIDE_SEC / 60
        ot = ocr_aigis.get(num, 0) * STRIDE_SEC / 60
        match = '✅' if abs(ct-ot) < 5 else '⚠️'
        print(f'  #{num:>2} {name:15} {ct:>8.1f}분    {ot:>8.1f}분  {match}')

    print(f'\n=== 하키머신 아이스타임 ===')
    all_hm = sorted(set(list(clf_hm.keys()) + list(ocr_hm.keys())))
    for num in all_hm:
        ct = clf_hm.get(num, 0) * STRIDE_SEC / 60
        ot = ocr_hm.get(num, 0) * STRIDE_SEC / 60
        match = '✅' if abs(ct-ot) < 5 else '⚠️'
        print(f'  #{num:>2} {ct:>8.1f}분    {ot:>8.1f}분  {match}')

    print(f'\n=== 팀 합계 ===')
    clf_a_tot = sum(clf_aigis.values()) * STRIDE_SEC / 60
    clf_h_tot = sum(clf_hm.values())    * STRIDE_SEC / 60
    ocr_a_tot = sum(ocr_aigis.values()) * STRIDE_SEC / 60
    ocr_h_tot = sum(ocr_hm.values())    * STRIDE_SEC / 60
    print(f'  아이기스  CLF: {clf_a_tot:.0f}분  OCR: {ocr_a_tot:.0f}분')
    print(f'  하키머신  CLF: {clf_h_tot:.0f}분  OCR: {ocr_h_tot:.0f}분')
    print(f'  팀 비율   CLF: {clf_a_tot/max(clf_h_tot,1):.2f}x  OCR: {ocr_a_tot/max(ocr_h_tot,1):.2f}x')

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result = {
            'video': os.path.basename(video_path),
            'duration_min': round(dur, 1),
            'segments': segments,
            'total_detections': total_dets,
            'clf': {'aigis': dict(clf_aigis), 'hockey_machine': dict(clf_hm),
                    'hit_rate': round(clf_hits/max(total_dets,1), 3)},
            'ocr': {'aigis': dict(ocr_aigis), 'hockey_machine': dict(ocr_hm),
                    'hit_rate': round(ocr_hits/max(total_dets,1), 3)},
        }
        out_path = os.path.join(output_dir, 'compare_result.json')
        json.dump(result, open(out_path,'w'), ensure_ascii=False, indent=2)
        print(f'\n결과 저장: {out_path}')

    return result if output_dir else None

if __name__ == '__main__':
    video = sys.argv[1] if len(sys.argv) > 1 else 'data/videos/test_game.mp4'
    roi   = 'data/roi_test_game.json'
    run(video, roi_path=roi, output_dir='data/results/test_game')
