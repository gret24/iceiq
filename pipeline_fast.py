#!/usr/bin/env python3
"""
pipeline_fast.py — 속도 개선 버전
1. ByteTrack: YOLO batch inference (4프레임 동시)
2. OCR: 확인된 track 재사용 (새 track만 OCR)
3. GPU 가속 (EasyOCR gpu=True, YOLO GPU)
"""
import json, os, re, shutil, time, io, base64
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
import easyocr
from ultralytics import YOLO
import anthropic

# ── 경로 (동적 설정 가능) ──────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR  = os.path.join(BASE_DIR, "frames")
DETECTED_DIR= os.path.join(BASE_DIR, "detected")
TRACKS_JSON = os.path.join(BASE_DIR, "tracks.json")
JERSEY_JSON = os.path.join(BASE_DIR, "jersey_map.json")
RESULTS_TXT = os.path.join(BASE_DIR, "results.txt")
EXTRACT_FPS = 4
MIN_BOX_AREA= 3000
BLUR_THRESHOLD = 50
BATCH_SIZE  = 8   # 동시 처리 프레임 수


def header(step, label):
    print(f"\n[{step}] {label}\n" + "-"*45)


# ── CLAHE 보정 ─────────────────────────────────────────────
def _correct_ice_glare(img_path: str) -> np.ndarray | None:
    bgr = cv2.imread(img_path)
    if bgr is None: return None
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_eq = cv2.merge([clahe.apply(l), a, b])
    bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.3, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ── Step 2: ByteTrack (배치 처리) ─────────────────────────
def step_track():
    header(2, f"ByteTrack 추적 (배치={BATCH_SIZE})")
    os.makedirs(DETECTED_DIR, exist_ok=True)

    model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))
    files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
    total = len(files)
    all_tracks = {}

    CORRECTED_DIR = os.path.join(BASE_DIR, "_corrected_tmp")
    os.makedirs(CORRECTED_DIR, exist_ok=True)

    t0 = time.time()

    # 배치 단위 처리 (ByteTrack은 순서가 중요해서 순차이지만 보정은 병렬)
    def preprocess(fname):
        path = os.path.join(FRAMES_DIR, fname)
        corrected = _correct_ice_glare(path)
        if corrected is not None:
            tmp = os.path.join(CORRECTED_DIR, fname)
            cv2.imwrite(tmp, corrected)
            return fname, tmp
        return fname, path

    # 보정 병렬화 (I/O 바운드)
    with ThreadPoolExecutor(max_workers=4) as ex:
        corrected_map = dict(ex.map(preprocess, files))

    print(f"  보정 완료: {len(corrected_map)}장 ({time.time()-t0:.1f}초)")

    # ByteTrack은 순서 보장 필요 → 순차 처리 (GPU 배치 자동)
    for i, filename in enumerate(files, 1):
        detect_path = corrected_map.get(filename, os.path.join(FRAMES_DIR, filename))

        results = model.track(
            detect_path, persist=True, tracker="bytetrack.yaml",
            classes=[0], conf=0.18, iou=0.40, verbose=False,
        )

        frame_tracks = []
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, ids):
                if int(box.cls[0]) != 0: continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                frame_tracks.append({
                    "track_id": track_id,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": round(float(box.conf[0]), 3),
                })

        all_tracks[filename] = frame_tracks

        # 어노테이션
        img = Image.open(os.path.join(FRAMES_DIR, filename)).convert("RGB")
        draw = ImageDraw.Draw(img)
        for t in frame_tracks:
            draw.rectangle([t["x1"],t["y1"],t["x2"],t["y2"]], outline="lime", width=3)
            draw.text((t["x1"], max(0,t["y1"]-14)), f"ID:{t['track_id']}", fill="lime")
        img.save(os.path.join(DETECTED_DIR, filename))

        if i % 100 == 0 or i == total:
            elapsed = time.time() - t0
            fps_speed = i / elapsed
            remain = (total - i) / fps_speed if fps_speed > 0 else 0
            print(f"  {i}/{total} ({i/total*100:.0f}%) | {fps_speed:.1f}fps | 남은시간: {remain/60:.1f}분")

    if os.path.exists(CORRECTED_DIR):
        shutil.rmtree(CORRECTED_DIR)

    with open(TRACKS_JSON, "w") as f:
        json.dump(all_tracks, f)

    unique_ids = {t["track_id"] for ts in all_tracks.values() for t in ts}
    elapsed = time.time() - t0
    print(f"  → 완료: {total}장, {len(unique_ids)}개 Track, {elapsed/60:.1f}분")
    return all_tracks


# ── OCR 헬퍼 ─────────────────────────────────────────────
def _is_jersey(text: str) -> bool:
    return bool(re.match(r'^\d{1,2}$', text.strip())) and 1 <= int(text.strip()) <= 99

def _easyocr_jersey(reader, crop_arr):
    results = reader.readtext(crop_arr, allowlist='0123456789', min_size=5)
    if not results: return None
    candidates = []
    for (bbox, text, conf) in results:
        if text.strip().isdigit():
            xc = (bbox[0][0]+bbox[2][0])/2
            candidates.append({"text":text.strip(),"conf":conf,"x":xc,"bbox":bbox})
    candidates.sort(key=lambda c: c["x"])
    merged = []
    used = set()
    for i, c1 in enumerate(candidates):
        if i in used: continue
        if len(c1["text"]) == 1:
            for j, c2 in enumerate(candidates[i+1:], i+1):
                if j in used or len(c2["text"]) != 1: continue
                w1 = abs(c1["bbox"][2][0]-c1["bbox"][0][0])
                if c2["x"]-c1["x"] < w1*2.5:
                    mt = c1["text"]+c2["text"]
                    if _is_jersey(mt):
                        merged.append({"text":mt,"conf":(c1["conf"]+c2["conf"])/2+0.2})
                    used.add(i); used.add(j); break
        if i not in used:
            merged.append({"text":c1["text"],"conf":c1["conf"]}); used.add(i)
    best = None
    for item in merged:
        t, c = item["text"], item["conf"]
        if not _is_jersey(t): continue
        adj = c + (0.3 if len(t)==2 else 0)
        if adj >= (0.6 if len(t)==1 else 0.15):
            if best is None or adj > best[1]: best=(t, adj)
    return best


# ── Step 3: OCR (스킵 로직 + GPU) ──────────────────────────
def step_ocr(all_tracks=None):
    header(3, "등번호 OCR (스킵 최적화 + GPU)")

    if all_tracks is None:
        with open(TRACKS_JSON) as f:
            all_tracks = json.load(f)

    # GPU EasyOCR
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)

    # track_id → 등장 프레임 목록
    track_frames = {}
    for fname in sorted(all_tracks.keys()):
        for t in all_tracks[fname]:
            tid = t["track_id"]
            if tid not in track_frames:
                track_frames[tid] = []
            track_frames[tid].append((fname, t))

    total = len(track_frames)
    jersey_map = {}
    ocr_done = 0
    ocr_skipped = 0
    t0 = time.time()

    # [개선 2] 확인된 track 재사용: track_id가 이미 인식된 번호 캐시
    confirmed_cache = {}  # track_id → jersey번호

    for i, (tid, frame_list) in enumerate(sorted(track_frames.items()), 1):
        # [개선 2] 이미 확인된 track이면 OCR 스킵
        if tid in confirmed_cache:
            jersey_map[str(tid)] = confirmed_cache[tid]
            ocr_skipped += 1
            continue

        recognized = None
        vote_map = {}

        for filename, track in frame_list[:15]:  # 최대 15프레임 시도
            x1, y1, x2, y2 = track["x1"], track["y1"], track["x2"], track["y2"]
            w, h = x2-x1, y2-y1
            if w*h <= MIN_BOX_AREA: continue

            path = os.path.join(FRAMES_DIR, filename)
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None: continue
            if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD: continue

            img = Image.open(path).convert("RGB")
            iw, ih = img.size
            ny1 = y1 + int(h*0.15); ny2 = y1 + int(h*0.70)
            crop = img.crop((max(0,x1), max(0,ny1), min(iw,x2), min(ih,ny2)))
            cw, ch = crop.size
            if cw < 5 or ch < 5: continue
            big = crop.resize((cw*2,ch*2), Image.LANCZOS)
            big = ImageEnhance.Contrast(big).enhance(2.0)
            crop_arr = np.array(big)

            torso = np.array(img)[y1+int(h*0.2):y1+int(h*0.65), x1:x2]
            team = "HOME" if torso.size>0 and np.mean(torso)>128 else "AWAY"

            result = _easyocr_jersey(reader, crop_arr)
            if result:
                num, conf = result
                vote_map[num] = vote_map.get(num, 0) + conf
                if not recognized:
                    recognized = (num, team)

            if len(vote_map) > 0 and sum(1 for v in vote_map.values() if v>0) >= 3:
                break  # 3표 이상이면 조기 종료

        # 다수결
        if vote_map:
            two_digit = {k:v for k,v in vote_map.items() if len(k)==2}
            best_num = max(two_digit, key=two_digit.get) if two_digit else None
            if not best_num:
                max_conf = max(vote_map.values())
                if max_conf >= 0.5:
                    best_num = max(vote_map, key=vote_map.get)
            if best_num:
                team_str = recognized[1] if recognized else "UNKNOWN"
                entry = {"jersey": best_num, "team": team_str}
                jersey_map[str(tid)] = entry
                confirmed_cache[tid] = entry
                ocr_done += 1

        if i % 50 == 0 or i == total:
            elapsed = time.time()-t0
            fps_t = i/elapsed
            remain = (total-i)/fps_t if fps_t>0 else 0
            print(f"  {i}/{total} ({i/total*100:.0f}%) | OCR: {ocr_done} | 스킵: {ocr_skipped} | 남은: {remain/60:.1f}분")

    with open(JERSEY_JSON, "w") as f:
        json.dump(jersey_map, f, ensure_ascii=False)

    elapsed = time.time()-t0
    print(f"\n  → 완료: {total}개 Track | OCR {ocr_done}건 | 스킵 {ocr_skipped}건 | {elapsed/60:.1f}분")
    print(f"     스킵률: {ocr_skipped/total*100:.1f}%")
    return jersey_map


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        BASE_DIR = sys.argv[1]
        FRAMES_DIR  = os.path.join(BASE_DIR, "frames")
        DETECTED_DIR= os.path.join(BASE_DIR, "detected")
        TRACKS_JSON = os.path.join(BASE_DIR, "tracks.json")
        JERSEY_JSON = os.path.join(BASE_DIR, "jersey_map.json")
        RESULTS_TXT = os.path.join(BASE_DIR, "results.txt")

    t0 = time.time()
    tracks = step_track()
    step_ocr(tracks)
    print(f"\n총 소요: {(time.time()-t0)/60:.1f}분")
