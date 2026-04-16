"""
compare_pipeline.py — OCR vs Classifier 비교 분석
Usage:
    python compare_pipeline.py <video_path> [--roster roster.yaml]
    Telegram: /compare <video_url>
"""
import sys, os, time, json, yaml, cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.detection import YOLODetector
from pipeline.team_classification import HockeyTeamClassifier, PRESET_HOCKEY_MACHINE_AIGIS
from pipeline.jersey_classifier import JerseyClassifier
from pipeline.jersey_recognition import JerseyRecognizer
import easyocr
import warnings; warnings.filterwarnings('ignore')

# ── 팀 분류 설정 ──────────────────────────────────────────────────────────────
WHITE_LO  = np.array([0, 0, 160], dtype=np.uint8)
WHITE_HI  = np.array([180, 50, 255], dtype=np.uint8)
WHITE_THR = 0.12

AIGIS_ROSTER = {4,11,12,14,25,42,47,61,94}
AIGIS_NAMES  = {4:'윤지성',11:'박리오',12:'이준표',14:'이봄',25:'김재원',
                42:'남다름',47:'한승원',61:'김승후',94:'김태윤'}


def _white_ratio(crop):
    h,w = crop.shape[:2]
    t = crop[int(h*0.20):int(h*0.65), int(w*0.15):int(w*0.85)]
    if t.size == 0: return 0
    hsv = cv2.cvtColor(t, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, WHITE_LO, WHITE_HI).sum() / 255 / (t.shape[0]*t.shape[1])


def _ocr_read(reader, crop):
    h,w = crop.shape[:2]
    if h<10 or w<10: return None, 0
    scale = max(128/h, 1.0)
    up = cv2.resize(crop,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_CUBIC)
    rh,rw = up.shape[:2]
    torso = up[int(rh*0.25):int(rh*0.70), int(rw*0.10):int(rw*0.90)]
    if torso.size == 0: return None, 0
    gray = cv2.cvtColor(torso, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(3.0,(4,4)); enh = clahe.apply(gray)
    b = cv2.adaptiveThreshold(enh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,8)
    p = cv2.copyMakeBorder(b,12,12,12,12,cv2.BORDER_CONSTANT,value=255)
    best, bc = None, 0
    for img in [p, cv2.bitwise_not(p)]:
        for _,text,conf in reader.readtext(img,allowlist='0123456789',detail=1,
                                            paragraph=False,min_size=8):
            text=text.strip()
            if text.isdigit() and 1<=int(text)<=99 and conf>bc:
                best, bc = int(text), conf
    return best, bc


class ComparisonPipeline:
    def __init__(self, video_path, roster_path=None, fps_sample=2, roi_path=None):
        self.video      = video_path
        self.fps_sample = fps_sample
        self.roster     = self._load_roster(roster_path) if roster_path else None

        # ROI
        self.roi_pts = None
        if roi_path and os.path.exists(roi_path):
            roi_data = json.load(open(roi_path))
            key = list(roi_data.keys())[0]
            self.roi_pts = np.array(roi_data[key], dtype=np.int32)

        # 결과 저장
        self.ocr_results = defaultdict(lambda: {"frames": 0})
        self.cls_results = defaultdict(lambda: {"frames": 0})

        # 모듈 초기화
        print('모듈 로드 중...')
        self.detector  = YOLODetector(model_path='yolov8m.pt', device='mps')
        self.clf       = JerseyClassifier(min_conf=0.55)
        self.ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        print('준비 완료')

    def _load_roster(self, path):
        with open(path) as f:
            return yaml.safe_load(f)

    def _in_roi(self, x1, y1, x2, y2):
        if self.roi_pts is None: return True
        return cv2.pointPolygonTest(self.roi_pts,((x1+x2)/2,(y1+y2)/2),False) >= 0

    def run(self):
        cap     = cv2.VideoCapture(self.video)
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        skip    = max(1, int(src_fps / self.fps_sample))
        spf     = skip / src_fps
        dur_min = total / src_fps / 60

        print(f'영상: {os.path.basename(self.video)} ({dur_min:.1f}분)')
        print(f'{src_fps:.0f}fps → {self.fps_sample}fps 샘플링 (skip={skip})')

        t_ocr = t_cls = 0
        frame_idx = processed = 0
        total_dets = 0

        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % skip != 0:
                frame_idx += 1
                continue

            dets = self.detector.detect(frame)

            for i, det in enumerate(dets):
                if det['confidence'] < 0.45: continue
                x1,y1,x2,y2 = map(int, det['bbox'])
                if (y2-y1) < 35: continue
                if not self._in_roi(x1,y1,x2,y2): continue
                total_dets += 1

                crop = frame[max(0,y1):y2, max(0,x1):x2]
                wr   = _white_ratio(crop)
                team = 'aigis' if wr >= WHITE_THR else 'hockey_machine'

                # ── Classifier ──────────────────────────────────────────
                t0 = time.time()
                clf_num, clf_conf = self.clf.predict(crop, team)
                t_cls += time.time() - t0
                if clf_num:
                    key = f'{team}_{clf_num}'
                    self.cls_results[key]['frames'] += 1

                # ── OCR ─────────────────────────────────────────────────
                t0 = time.time()
                ocr_num, ocr_conf = _ocr_read(self.ocr_reader, crop)
                t_ocr += time.time() - t0
                if ocr_num:
                    if team == 'aigis' and ocr_num not in AIGIS_ROSTER:
                        ocr_num = None
                if ocr_num:
                    key = f'{team}_{ocr_num}'
                    self.ocr_results[key]['frames'] += 1

            processed += 1
            frame_idx += 1

            if processed % 50 == 0:
                t_min = frame_idx / src_fps / 60
                print(f'  {t_min:.0f}분... clf히트:{sum(v["frames"] for v in self.cls_results.values())} ocr히트:{sum(v["frames"] for v in self.ocr_results.values())}')

        cap.release()
        return self._build_report(spf, t_ocr, t_cls, processed, total_dets, dur_min)

    def _calc_icetime(self, results, spf):
        times = {}
        for key, data in results.items():
            times[key] = round(data['frames'] * spf, 1)
        return dict(sorted(times.items(), key=lambda x: -x[1]))

    def _build_report(self, spf, t_ocr, t_cls, n_frames, total_dets, dur_min):
        ocr_times = self._calc_icetime(self.ocr_results, spf)
        cls_times = self._calc_icetime(self.cls_results, spf)
        roster_count = sum(len(t.get('players',[])) for t in self.roster.values()) \
                       if self.roster else '?'
        return {
            'video':            os.path.basename(self.video),
            'duration_min':     round(dur_min, 1),
            'frames_processed': n_frames,
            'total_detections': total_dets,
            'spf':              round(spf, 3),
            'ocr': {
                'players_found': len(ocr_times),
                'time_sec':      round(t_ocr, 2),
                'ice_times':     ocr_times,
                'hit_rate':      round(sum(v['frames'] for v in self.ocr_results.values()) / max(total_dets,1), 3),
            },
            'classifier': {
                'players_found': len(cls_times),
                'time_sec':      round(t_cls, 2),
                'ice_times':     cls_times,
                'hit_rate':      round(sum(v['frames'] for v in self.cls_results.values()) / max(total_dets,1), 3),
            },
            'roster_total': roster_count,
        }


def format_report(r):
    """텔레그램용 텍스트 리포트"""
    lines = [f"📊 비교 분석: {r['video']}",
             f"경기: {r['duration_min']}분 | 감지: {r['total_detections']}건", ""]

    lines.append("═══ OCR 파이프라인 ═══")
    lines.append(f"인식: {r['ocr']['players_found']}명 | 인식률: {r['ocr']['hit_rate']*100:.1f}% | {r['ocr']['time_sec']}s")
    for p, t in list(r['ocr']['ice_times'].items())[:10]:
        m,s = divmod(int(t), 60)
        team, num = p.split('_', 1)
        name = AIGIS_NAMES.get(int(num), f'#{num}') if team == 'aigis' else f'HM#{num}'
        lines.append(f"  {name}: {m}:{s:02d}")

    lines.append("")
    lines.append("═══ Classifier 파이프라인 ═══")
    lines.append(f"인식: {r['classifier']['players_found']}명 | 인식률: {r['classifier']['hit_rate']*100:.1f}% | {r['classifier']['time_sec']}s")
    for p, t in list(r['classifier']['ice_times'].items())[:10]:
        m,s = divmod(int(t), 60)
        team, num = p.split('_', 1)
        name = AIGIS_NAMES.get(int(num), f'#{num}') if team == 'aigis' else f'HM#{num}'
        lines.append(f"  {name}: {m}:{s:02d}")

    lines.append("")
    lines.append("═══ 비교 ═══")
    ocr_n = r['ocr']['players_found']
    cls_n = r['classifier']['players_found']
    lines.append(f"인식 선수: OCR {ocr_n}명 vs Classifier {cls_n}명 / 로스터 {r['roster_total']}명")
    if r['ocr']['time_sec'] > 0:
        speedup = r['ocr']['time_sec'] / max(r['classifier']['time_sec'], 0.01)
        lines.append(f"속도: Classifier {speedup:.1f}x 빠름")
    return "\n".join(lines)


def handle_compare(args):
    """OpenClaw /compare 커맨드 핸들러"""
    video = args[0] if args else None
    if not video:
        return "Usage: /compare <video_path_or_url>"

    if video.startswith("http"):
        import subprocess
        out = "/tmp/compare_video.mp4"
        subprocess.run(["yt-dlp", "-f", "300", "-o", out, video], check=True)
        video = out

    roi   = "data/roi_test_game.json" if os.path.exists("data/roi_test_game.json") else None
    roster = "data/rosters/aigis_test_game.yaml" if os.path.exists("data/rosters/aigis_test_game.yaml") else None

    pipe   = ComparisonPipeline(video, roster_path=roster, fps_sample=2, roi_path=roi)
    report = pipe.run()

    out_path = "/tmp/compare_report.json"
    json.dump(report, open(out_path,'w'), ensure_ascii=False, indent=2)

    return format_report(report)


if __name__ == "__main__":
    video  = sys.argv[1] if len(sys.argv) > 1 else "data/videos/test_game.mp4"
    roster = sys.argv[2] if len(sys.argv) > 2 else "data/rosters/aigis_test_game.yaml"
    roi    = "data/roi_test_game.json"

    pipe   = ComparisonPipeline(video, roster_path=roster, fps_sample=2, roi_path=roi)
    report = pipe.run()
    print(format_report(report))

    out = "compare_report.json"
    json.dump(report, open(out,'w'), ensure_ascii=False, indent=2)
    print(f"\n→ {out} 저장됨")
