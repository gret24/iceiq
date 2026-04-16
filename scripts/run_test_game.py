import sys, cv2, numpy as np, os, json, time, easyocr
sys.path.insert(0, '/Users/hanhyeonseong/iceiq-dev')
from pipeline.detection import YOLODetector
from collections import Counter
import warnings; warnings.filterwarnings('ignore')

os.chdir('/Users/hanhyeonseong/iceiq-dev')

detector = YOLODetector(model_path='yolov8m.pt', device='mps')
reader   = easyocr.Reader(['en'], gpu=True, verbose=False)

cap   = cv2.VideoCapture('data/videos/test_game.mp4')
fps   = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

roi_pts = np.array(json.load(open('data/roi_test_game.json'))['test_game'], dtype=np.int32)

WHITE_LO = np.array([0, 0, 160], dtype=np.uint8)
WHITE_HI = np.array([180, 50, 255], dtype=np.uint8)
WHITE_THR = 0.15
AIGIS_ROSTER = {4, 11, 12, 14, 25, 42, 47, 61, 94}

def in_roi(x1, y1, x2, y2):
    cx, cy = (x1+x2)//2, (y1+y2)//2
    return cv2.pointPolygonTest(roi_pts, (float(cx), float(cy)), False) >= 0

def white_ratio(crop):
    h, w = crop.shape[:2]
    torso = crop[int(h*0.20):int(h*0.65), int(w*0.15):int(w*0.85)]
    if torso.size == 0: return 0
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WHITE_LO, WHITE_HI)
    return mask.sum() / 255 / (torso.shape[0] * torso.shape[1])

def classify_team(crop):
    wr = white_ratio(crop)
    return ('aigis', wr) if wr >= WHITE_THR else ('hockey_machine', 1-wr)

def ocr_number(crop):
    h, w = crop.shape[:2]
    if h < 10 or w < 10: return None, 0
    scale = max(128/h, 1.0)
    up = cv2.resize(crop, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    rh, rw = up.shape[:2]
    torso = up[int(rh*0.25):int(rh*0.70), int(rw*0.10):int(rw*0.90)]
    if torso.size == 0: return None, 0
    gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(3.0, (4,4)); enh = clahe.apply(gray)
    b = cv2.adaptiveThreshold(enh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,8)
    p = cv2.copyMakeBorder(b,12,12,12,12,cv2.BORDER_CONSTANT,value=255)
    best, bc = None, 0
    for img in [p, cv2.bitwise_not(p)]:
        for _, text, conf in reader.readtext(img, allowlist='0123456789', detail=1, paragraph=False, min_size=8):
            text = text.strip()
            if text.isdigit() and 1 <= int(text) <= 99 and conf > bc:
                best, bc = int(text), conf
    return best, bc

OUTPUT = 'data/results/test_game'
os.makedirs(OUTPUT, exist_ok=True)

STRIDE = int(fps * 30)
team_stats = {'aigis': Counter(), 'hockey_machine': Counter()}
timeline = []
t0 = time.time()

print(f'ROI 적용 end-to-end ({total//STRIDE}구간)...')
for frame_idx in range(0, total, STRIDE):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret: continue

    dets = detector.detect(frame)
    aigis_nums, hm_nums = [], []

    for i, det in enumerate(dets):
        if det['confidence'] < 0.45: continue
        x1,y1,x2,y2 = map(int, det['bbox'])
        if (y2-y1) < 35: continue
        if not in_roi(x1, y1, x2, y2): continue

        crop = frame[max(0,y1):y2, max(0,x1):x2]
        team, tc = classify_team(crop)
        num, nc = ocr_number(crop)

        if num and team == 'aigis' and num not in AIGIS_ROSTER:
            num = None

        if num:
            team_stats[team][num] += 1
            if team == 'aigis': aigis_nums.append(num)
            else: hm_nums.append(num)

    t_min = round(frame_idx/fps/60, 1)
    timeline.append({'t': t_min, 'aigis': aigis_nums, 'hm': hm_nums})

cap.release()
with open(f'{OUTPUT}/timeline_roi.json', 'w') as f:
    json.dump(timeline, f, ensure_ascii=False, indent=2)

total_min = timeline[-1]['t'] if timeline else 0
print(f'완료 ({time.time()-t0:.0f}초) — {total_min:.0f}분')
print()
aigis_names = {4:'윤지성',11:'박리오',12:'이준표',14:'이봄',25:'김재원',42:'남다름',47:'한승원',61:'김승후',94:'김태윤'}
print('=== 아이기스 ===')
for num,cnt in sorted(team_stats['aigis'].items(), key=lambda x:-x[1]):
    print(f'  #{num:>2} {aigis_names.get(num,"?"):8} {cnt*0.5:5.1f}분')
print()
print('=== 하키머신 ===')
for num,cnt in sorted(team_stats['hockey_machine'].items(), key=lambda x:-x[1]):
    print(f'  #{num:>2}  {cnt*0.5:5.1f}분')
