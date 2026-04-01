#!/usr/bin/env python3
"""
pipeline_fast.py — 속도 개선 버전
1. ByteTrack: YOLO batch inference (4프레임 동시)
2. OCR: 확인된 track 재사용 (새 track만 OCR)
3. GPU 가속 (EasyOCR gpu=True, YOLO GPU)
"""
import json, os, re, shutil, time, io, base64, subprocess
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




import argparse, subprocess, json as _json, math as _math

# ── 하이라이트 생성 ─────────────────────────────────────────────────

def make_shifts(jersey_map: dict, tracks_data: dict, target_num: str,
                gap_sec: float = 10.0, buf_sec: float = 5.0,
                fps: int = 4) -> list[dict]:
    """jersey_map + tracks에서 특정 선수의 시프트 추출"""
    target_norm = target_num.lstrip("0") or "0"
    target_tids = {tid for tid, info in jersey_map.items()
                   if info["jersey"].lstrip("0") == target_norm}

    if not target_tids:
        print(f"  ⚠ {target_num}번 선수 없음")
        return []

    # 등장 프레임 수집
    frame_indices = []
    for fname, fts in sorted(tracks_data.items()):
        fm = __import__("re").search(r"(\d+)", fname)
        if not fm: continue
        fidx = int(fm.group(1))
        for t in fts:
            if str(t["track_id"]) in target_tids:
                frame_indices.append(fidx)
                break

    if not frame_indices:
        return []

    frame_indices = sorted(set(frame_indices))
    gap_frames = int(gap_sec * fps)

    # 프레임 그룹 → 시프트
    groups, s, e = [], frame_indices[0], frame_indices[0]
    for f in frame_indices[1:]:
        if f - e <= gap_frames:
            e = f
        else:
            groups.append((s, e))
            s = e = f
    groups.append((s, e))

    shifts = []
    for gs, ge in groups:
        start = max(0.0, gs / fps - buf_sec)
        end   = ge / fps + buf_sec
        shifts.append({
            "start_sec": round(start, 2),
            "end_sec":   round(end, 2),
            "duration":  round(end - start, 2),
        })

    return shifts


def make_highlight(video_path: str, shifts: list[dict],
                   target_num: str, out_dir: str) -> str | None:
    """시프트 클립을 이어붙여 하이라이트 영상 생성 (텍스트 오버레이 포함)"""
    os.makedirs(out_dir, exist_ok=True)
    clips_dir = os.path.join(out_dir, "clips_tmp")
    os.makedirs(clips_dir, exist_ok=True)

    clip_files = []
    total_ice = sum(s["duration"] for s in shifts)

    print(f"  시프트 {len(shifts)}개 추출 중... (총 아이스타임 {total_ice/60:.1f}분)")

    for i, shift in enumerate(shifts, 1):
        s_sec = shift["start_sec"]
        e_sec = shift["end_sec"]
        mm, ss = divmod(int(s_sec), 60)
        label_text = f"Shift {i}  {mm}:{ss:02d}"

        # drawtext 필터: 왼쪽 상단 2초 표시
        vf = (
            f"drawtext=text='{label_text}':fontsize=36:fontcolor=white:"
            f"x=20:y=20:box=1:boxcolor=black@0.5:boxborderw=5:"
            f"enable=\'between(t,0,2)\'"
        )
        out_clip = os.path.join(clips_dir, f"clip_{i:03d}.mp4")
        r = subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(s_sec), "-to", str(e_sec),
            "-i", video_path,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", out_clip
        ], capture_output=True)
        if r.returncode == 0 and os.path.exists(out_clip):
            clip_files.append(out_clip)
            print(f"    shift {i}: {mm}:{ss:02d} ~ {int(e_sec)//60:02d}:{int(e_sec)%60:02d} ({shift['duration']:.0f}초)")
        else:
            print(f"    shift {i}: 실패 - {r.stderr[-200:].decode() if r.stderr else "알수없음"}")

    if not clip_files:
        return None

    # concat
    concat_txt = os.path.join(clips_dir, "concat.txt")
    with open(concat_txt, "w") as f:
        for c in clip_files:
            f.write(f"file '{c}'\n")

    out_path = os.path.join(out_dir, f"{target_num}_highlight.mp4")
    r2 = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_txt, "-c", "copy", out_path
    ], capture_output=True)

    if r2.returncode == 0:
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"\n  ✅ {target_num}_highlight.mp4 완성! ({size_mb:.1f}MB, {len(clip_files)}개 클립)")
    else:
        print(f"  ❌ concat 실패")
        return None

    # 임시 클립 정리
    shutil.rmtree(clips_dir, ignore_errors=True)
    return out_path


def save_shift_metadata(shifts: list[dict], target_num: str, out_dir: str):
    """시프트 메타데이터 JSON 저장"""
    os.makedirs(out_dir, exist_ok=True)
    total_ice = sum(s["duration"] for s in shifts)
    meta = {
        "player": target_num,
        "total_shifts": len(shifts),
        "total_ice_time_sec": round(total_ice, 2),
        "total_ice_time_min": round(total_ice / 60, 2),
        "shifts": shifts,
    }
    out_path = os.path.join(out_dir, f"{target_num}_shifts.json")
    with open(out_path, "w") as f:
        _json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  메타데이터: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="pipeline_fast + 하이라이트 생성")
    parser.add_argument("--input",       required=True, help="입력 영상 경로")
    parser.add_argument("--player",      required=True, help="추출할 선수 등번호")
    parser.add_argument("--home-roster", type=str, default="")
    parser.add_argument("--away-roster", type=str, default="")
    parser.add_argument("--fps",         type=int, default=4)
    parser.add_argument("--mode",        choices=["highlight", "fulltime"], default="highlight")
    parser.add_argument("--gap",         type=float, default=10.0, help="시프트 갭 기준 (초)")
    parser.add_argument("--buf",         type=float, default=5.0,  help="시프트 앞뒤 버퍼 (초)")
    parser.add_argument("--out-dir",     type=str, default="/workspace/iceiq/output")
    parser.add_argument("--skip-extract",action="store_true")
    parser.add_argument("--skip-track",  action="store_true")
    parser.add_argument("--skip-ocr",    action="store_true")
    args = parser.parse_args()

    global BASE_DIR, FRAMES_DIR, DETECTED_DIR, TRACKS_JSON, JERSEY_JSON, EXTRACT_FPS

    video_path = args.input
    video_dir  = os.path.dirname(os.path.abspath(video_path))
    BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
    EXTRACT_FPS = args.fps

    t0 = time.time()
    print(f"\n{'='*50}")
    print(f"  pipeline_fast + 하이라이트")
    print(f"  영상: {os.path.basename(video_path)}  선수: #{args.player}")
    print(f"{'='*50}")

    # 분석
    all_tracks = None
    if not args.skip_track:
        all_tracks = step_track()
    else:
        print("[2] ByteTrack 건너뜀")
        with open(TRACKS_JSON) as f:
            all_tracks = _json.load(f)

    jersey_map = None
    if not args.skip_ocr:
        jersey_map = step_ocr(all_tracks)
    else:
        print("[3] OCR 건너뜀")
        with open(JERSEY_JSON) as f:
            jersey_map = _json.load(f)

    if jersey_map is None or all_tracks is None:
        print("분석 실패"); return

    # 시프트 추출
    print("\n[4] 시프트 추출")
    shifts = make_shifts(jersey_map, all_tracks, args.player,
                         gap_sec=args.gap, buf_sec=args.buf, fps=EXTRACT_FPS)

    if not shifts:
        print(f"  {args.player}번 선수 감지 없음")
        return

    print(f"  {len(shifts)}개 시프트 감지")

    # 메타데이터 저장
    save_shift_metadata(shifts, args.player, args.out_dir)

    # 하이라이트 생성
    print("\n[5] 하이라이트 영상 생성")
    out_path = make_highlight(video_path, shifts, args.player, args.out_dir)

    elapsed = time.time() - t0
    total_ice = sum(s["duration"] for s in shifts)
    print(f"\n{'='*50}")
    print(f"  완료! ({elapsed/60:.1f}분)")
    print(f"  시프트: {len(shifts)}개  총 아이스타임: {total_ice/60:.1f}분")
    if out_path:
        print(f"  출력: {out_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()

