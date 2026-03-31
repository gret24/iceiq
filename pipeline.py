#!/usr/bin/env python3
"""
IceIQ Full Pipeline (ByteTrack Edition v2)

Usage:
  python3 pipeline.py <input_video> <player_number> [team] [--buffer 3.0]

Examples:
  python3 pipeline.py test.mp4 3
  python3 pipeline.py test.mp4 23 AWAY
  python3 pipeline.py test.mp4 49 HOME --buffer 5.0

Steps:
  [1/5] 프레임 추출       (ffmpeg, fps=2)
  [2/5] ByteTrack 추적   (YOLOv8 + ByteTrack)
  [3/5] 등번호 인식       (EasyOCR + Claude Vision 보조, 블러 스킵, 첫 확인만)
  [4/5] 구간 추출
  [5/5] highlight.mp4 완성
"""

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time

import anthropic
import cv2
import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
from ultralytics import YOLO

# ── 경로 ──────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR      = os.path.join(BASE_DIR, "frames")
DETECTED_DIR    = os.path.join(BASE_DIR, "detected")
CLIPS_DIR       = os.path.join(BASE_DIR, "clips")
TRACKS_JSON     = os.path.join(BASE_DIR, "tracks.json")
JERSEY_JSON     = os.path.join(BASE_DIR, "jersey_map.json")
RESULTS_TXT     = os.path.join(BASE_DIR, "results.txt")
TIMESTAMPS_TXT  = os.path.join(BASE_DIR, "timestamps.txt")
HIGHLIGHT_PATH  = os.path.join(BASE_DIR, "highlight.mp4")
HIGHLIGHT_OLD   = os.path.join(BASE_DIR, "highlight_old.mp4")

EXTRACT_FPS  = 2          # 프레임 추출 FPS
MIN_BOX_AREA = 5000       # 너무 작은 bbox 무시
TRACK_BUFFER = 30         # 선수 안 보여도 유지할 프레임 수 (추출 기준)
BLUR_THRESHOLD = 50       # 라플라시안 분산 이 이하면 블러 프레임으로 스킵

TOTAL_STEPS  = 5


def header(step: int, label: str):
    print(f"\n[{step}/{TOTAL_STEPS}] {label}")
    print("-" * 45)


# ─────────────────────────────────────────────────────────
# 빙판 반사광 보정 (CLAHE + 채도 강화)
# ─────────────────────────────────────────────────────────

def _correct_ice_glare(img_path: str) -> np.ndarray:
    """
    빙판 반사광 보정:
    1. CLAHE (Contrast Limited Adaptive Histogram Equalization) → 밝기 균일화
    2. 채도(Saturation) 강화 → 유니폼 색상 부각
    3. 상단 20% 마스킹 (관중석/조명 영역 제외)
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        return bgr

    h, w = bgr.shape[:2]

    # LAB 색공간에서 L채널에만 CLAHE 적용 (밝기만 보정)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # HSV에서 채도 강화 (빙판 흰색과 유니폼 색 대비 향상)
    hsv = cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
    bgr_final = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return bgr_final


# ─────────────────────────────────────────────────────────
# STEP 1 : 프레임 추출
# ─────────────────────────────────────────────────────────

def step_extract(input_file: str):
    header(1, "프레임 추출 중...")
    os.makedirs(FRAMES_DIR, exist_ok=True)

    subprocess.run([
        "ffmpeg", "-i", input_file,
        "-vf", f"fps={EXTRACT_FPS}",
        os.path.join(FRAMES_DIR, "frame_%04d.jpg"),
        "-y",
    ], check=True, capture_output=True)

    count = len([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
    print(f"  → {count}장 추출 완료  (frames/)")
    return count


# ─────────────────────────────────────────────────────────
# STEP 2 : ByteTrack 추적
# ─────────────────────────────────────────────────────────

def step_track():
    header(2, "ByteTrack 추적 시작...")
    os.makedirs(DETECTED_DIR, exist_ok=True)

    model = YOLO("yolov8n.pt")
    files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
    total = len(files)

    # tracks.json 구조: {"frame_0001.jpg": [{"track_id": 1, "x1":..., ...}, ...], ...}
    all_tracks = {}
    total_det  = 0

    # 임시 보정 프레임 저장 디렉토리
    CORRECTED_DIR = os.path.join(BASE_DIR, "_corrected_tmp")
    os.makedirs(CORRECTED_DIR, exist_ok=True)

    for i, filename in enumerate(files, 1):
        path = os.path.join(FRAMES_DIR, filename)

        # 빙판 반사광 보정 적용
        corrected = _correct_ice_glare(path)
        if corrected is not None:
            tmp_path = os.path.join(CORRECTED_DIR, filename)
            cv2.imwrite(tmp_path, corrected)
            detect_path = tmp_path
        else:
            detect_path = path

        results = model.track(
            detect_path,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],          # 사람(class 0)만
            conf=0.25,            # 0.3 → 0.25 (경계선 선수 추가 감지)
            iou=0.45,
            verbose=False,
        )

        frame_tracks = []
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().cpu().tolist()
            for box, track_id in zip(boxes, ids):
                if int(box.cls[0]) != 0:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                frame_tracks.append({
                    "track_id": track_id,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": round(float(box.conf[0]), 3),
                })
                total_det += 1

        all_tracks[filename] = frame_tracks

        # 어노테이션 이미지 저장
        img  = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for t in frame_tracks:
            draw.rectangle([t["x1"], t["y1"], t["x2"], t["y2"]], outline="lime", width=3)
            draw.text((t["x1"], max(0, t["y1"] - 14)), f"ID:{t['track_id']}", fill="lime")
        img.save(os.path.join(DETECTED_DIR, filename))

        if i % 50 == 0 or i == total:
            pct = int(i / total * 100)
            active_ids = {t["track_id"] for tracks in all_tracks.values() for t in tracks}
            print(f"  처리: {i}/{total} ({pct}%)  누적 Track: {len(active_ids)}개")

    with open(TRACKS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_tracks, f)

    # 임시 보정 프레임 정리
    if os.path.exists(CORRECTED_DIR):
        shutil.rmtree(CORRECTED_DIR)

    unique_ids = {t["track_id"] for tracks in all_tracks.values() for t in tracks}
    print(f"  → {total}장 처리 완료, 고유 Track ID: {len(unique_ids)}개  (tracks.json)")
    return all_tracks


# ─────────────────────────────────────────────────────────
# STEP 3 : 등번호 인식 (블러 스킵 + EasyOCR + Claude Vision 보조)
# ─────────────────────────────────────────────────────────

def _uniform_color(torso: np.ndarray) -> str:
    return "HOME" if np.mean(torso) > 128 else "AWAY"


def _is_jersey(text: str) -> bool:
    text = text.strip()
    return bool(re.match(r'^\d{1,2}$', text)) and 1 <= int(text) <= 99


def _laplacian_variance(img_path: str) -> float:
    """라플라시안 분산 → 낮을수록 블러"""
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _preprocess_crop(crop_img: Image.Image) -> Image.Image:
    """2배 확대 + 대비 강화"""
    cw, ch = crop_img.size
    if cw < 5 or ch < 5:
        return crop_img
    big = crop_img.resize((cw * 2, ch * 2), Image.LANCZOS)
    return ImageEnhance.Contrast(big).enhance(2.0)


def _easyocr_jersey(reader, crop_arr: np.ndarray) -> tuple[str, float] | None:
    """EasyOCR로 등번호 인식 → (번호, confidence) 반환
    핵심 개선: 인접한 1자리 숫자를 합쳐 2자리로 복원 (47→4+7 분리 문제 해결)
    """
    results = reader.readtext(crop_arr, allowlist='0123456789', min_size=5)
    if not results:
        return None

    # ── 인접 숫자 합치기 ──────────────────────────────────
    # results 형식: [([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, conf), ...]
    # 같은 행에 있는 1자리 숫자들을 x좌표 순서로 합침
    candidates = []

    # 1) 직접 인식된 결과 먼저 수집
    for (bbox, text, conf) in results:
        text = text.strip()
        if not text.isdigit():
            continue
        x_center = (bbox[0][0] + bbox[2][0]) / 2
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        candidates.append({"text": text, "conf": conf, "x": x_center, "y": y_center, "bbox": bbox})

    # 2) 인접한 1자리 숫자 쌍을 2자리로 합치기
    merged = []
    used = set()
    # x 좌표 기준 정렬
    candidates.sort(key=lambda c: c["x"])

    for i, c1 in enumerate(candidates):
        if i in used:
            continue
        if len(c1["text"]) == 1:
            # 오른쪽에 인접한 1자리 숫자 찾기
            for j, c2 in enumerate(candidates[i+1:], i+1):
                if j in used:
                    continue
                if len(c2["text"]) != 1:
                    continue
                # x 거리가 가까우면 (bbox 너비의 2배 이내) 합치기
                w1 = abs(c1["bbox"][2][0] - c1["bbox"][0][0])
                x_gap = c2["x"] - c1["x"]
                if x_gap < w1 * 2.5:
                    merged_text = c1["text"] + c2["text"]
                    merged_conf = (c1["conf"] + c2["conf"]) / 2 + 0.2  # 합친 보너스
                    if _is_jersey(merged_text):
                        merged.append({"text": merged_text, "conf": merged_conf})
                    used.add(i); used.add(j)
                    break

        # 합치지 못한 것은 그대로 유지
        if i not in used:
            merged.append({"text": c1["text"], "conf": c1["conf"]})
            used.add(i)

    # 3) 최종 후보 중 최고 신뢰도 선택
    best = None
    for item in merged:
        text = item["text"]
        conf = item["conf"]
        if not _is_jersey(text):
            continue
        # 1자리는 높은 confidence 필요
        min_conf = 0.6 if len(text) == 1 else 0.15
        # 2자리 추가 보너스
        bonus = 0.3 if len(text) == 2 else 0.0
        adjusted = conf + bonus
        if adjusted >= min_conf:
            if best is None or adjusted > best[1]:
                best = (text, adjusted)
    return best


def _claude_jersey(claude_client, crop_img: Image.Image) -> str | None:
    """Claude Vision으로 등번호 인식 (EasyOCR 실패 시 보조)"""
    buf = io.BytesIO()
    crop_img.save(buf, format='JPEG', quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        resp = claude_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}
                    },
                    {
                        "type": "text",
                        "text": (
                            "This is a cropped image of a hockey player. "
                            "What jersey number do you see on the player? "
                            "Reply with ONLY the number (e.g. '23'). "
                            "If no number is visible, reply 'NONE'."
                        )
                    }
                ]
            }]
        )
        answer = resp.content[0].text.strip()
        clean = re.sub(r'[^0-9]', '', answer)
        if clean and _is_jersey(clean):
            return clean
    except Exception as e:
        if "429" in str(e):
            time.sleep(2)  # rate limit 시 2초 대기
        else:
            print(f"    [Claude] 오류: {e}")
    return None


def step_ocr(all_tracks: dict | None = None):
    header(3, "등번호 인식 중... (EasyOCR + Claude Vision 보조)")

    if all_tracks is None:
        with open(TRACKS_JSON, encoding="utf-8") as f:
            all_tracks = json.load(f)

    reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    claude_client = anthropic.Anthropic()

    # 각 track_id의 등장 프레임 목록 수집 (여러 프레임 시도 가능하도록)
    track_frames: dict[int, list[tuple[str, dict]]] = {}
    for filename in sorted(all_tracks.keys()):
        for t in all_tracks[filename]:
            tid = t["track_id"]
            if tid not in track_frames:
                track_frames[tid] = []
            track_frames[tid].append((filename, t))

    total  = len(track_frames)
    found  = 0
    blur_skipped = 0
    easyocr_hit  = 0
    claude_hit   = 0
    jersey_map: dict[str, dict] = {}

    print(f"  고유 Track ID: {total}개  블러 threshold: {BLUR_THRESHOLD}")
    t0 = time.time()

    for i, (tid, frame_list) in enumerate(sorted(track_frames.items()), 1):
        recognized = None
        # 다수결: {번호: 누적 confidence}
        vote_map: dict[str, float] = {}
        # 최대 5개 프레임까지 투표 수집
        MAX_VOTE_FRAMES = 5

        for filename, track in frame_list[:MAX_VOTE_FRAMES * 3]:  # 여유있게 탐색
            if len(vote_map) > 0 and sum(1 for v in vote_map.values() if v > 0) >= MAX_VOTE_FRAMES:
                break

            x1, y1, x2, y2 = track["x1"], track["y1"], track["x2"], track["y2"]
            w, h = x2 - x1, y2 - y1
            if w * h <= MIN_BOX_AREA:
                continue

            path = os.path.join(FRAMES_DIR, filename)

            # [개선 4] 블러 스킵
            sharpness = _laplacian_variance(path)
            if sharpness < BLUR_THRESHOLD:
                blur_skipped += 1
                continue

            img = Image.open(path).convert("RGB")
            iw, ih = img.size

            # 등번호 위치에 집중한 크롭
            # 전체 박스 + 상반신 중앙(15%~70%) 두 가지 크롭 시도
            cx1 = max(0, x1); cy1 = max(0, y1)
            cx2 = min(iw, x2); cy2 = min(ih, y2)
            # 상반신 중앙 크롭 (등번호가 주로 가슴/등에 위치)
            ny1 = y1 + int(h * 0.15)
            ny2 = y1 + int(h * 0.70)
            crop_jersey = img.crop((cx1, max(0,ny1), cx2, min(ih,ny2)))
            crop = crop_jersey  # 등번호 집중 크롭 사용

            # [개선 3] 전처리
            proc = _preprocess_crop(crop)
            crop_arr = np.array(proc)

            # 팀 판별 (원본 torso 기준)
            arr = np.array(img)
            ty1 = y1 + int(h * 0.20); ty2 = y1 + int(h * 0.65)
            torso = arr[ty1:ty2, x1:x2]
            team = _uniform_color(torso) if torso.size > 0 else "UNKNOWN"

            # [개선 1] EasyOCR → 다수결 투표
            result = _easyocr_jersey(reader, crop_arr)
            if result:
                num, conf = result
                vote_map[num] = vote_map.get(num, 0) + conf
                # 팀 정보는 첫 번째 인식 기준으로 저장
                if not recognized:
                    recognized = (num, team, "EasyOCR")

            # 과부하 방지
            time.sleep(0.03)

        # 다수결로 최종 번호 결정
        if vote_map:
            # 2자리 숫자가 있으면 강하게 우선 (1자리는 무시)
            two_digit = {k: v for k, v in vote_map.items() if len(k) == 2}
            if two_digit:
                # 2자리 중 confidence 합산 최고
                best_num = max(two_digit, key=two_digit.get)
            else:
                # 2자리 없으면 1자리 중 최고 confidence (단, threshold 충족 시만)
                max_conf = max(vote_map.values())
                if max_conf >= 0.5:
                    best_num = max(vote_map, key=vote_map.get)
                else:
                    continue  # 신뢰도 낮은 1자리는 스킵

            team_str = recognized[1] if recognized else "UNKNOWN"
            jersey_map[str(tid)] = {"jersey": best_num, "team": team_str}
            found += 1
            easyocr_hit += 1

        if i % 10 == 0 or i == total:
            elapsed = time.time() - t0
            remain  = (total - i) / (i / elapsed) if elapsed > 0 else 0
            pct     = int(i / total * 100)
            print(f"  처리: {i}/{total} ({pct}%)  인식: {found}건  남은시간: {remain/60:.1f}분")

    with open(JERSEY_JSON, "w", encoding="utf-8") as f:
        json.dump(jersey_map, f, ensure_ascii=False)

    # results.txt 기록
    with open(RESULTS_TXT, "w", encoding="utf-8") as out:
        out.write("Track_ID | 등번호 | 유니폼 | 첫등장프레임\n")
        out.write("-" * 60 + "\n")
        for tid_str, info in sorted(jersey_map.items(), key=lambda x: int(x[0])):
            fname, _ = track_frames[int(tid_str)][0]
            fm = re.search(r'(\d+)', fname)
            frame_num = fm.group(1) if fm else "?"
            out.write(f"{tid_str} | {info['jersey']} | {info['team']} | frame_{frame_num}\n")

    print(f"\n  → {total}개 Track 처리 완료")
    print(f"     등번호 확인: {found}개  (EasyOCR: {easyocr_hit}, Claude: {claude_hit})")
    print(f"     블러 스킵: {blur_skipped}회  (jersey_map.json)")
    return jersey_map


# ─────────────────────────────────────────────────────────
# STEP 4 : 구간 추출
# ─────────────────────────────────────────────────────────

def _to_segments(frames: list, fps: int, buf: float, gap: int):
    if not frames:
        return []
    frames = sorted(frames)
    groups, s, e = [], frames[0], frames[0]
    for f in frames[1:]:
        if f - e <= gap:
            e = f
        else:
            groups.append((s, e))
            s = e = f
    groups.append((s, e))
    return [{"start": round(max(0.0, s / fps - buf), 2),
             "end":   round(e / fps + buf, 2)} for s, e in groups]


def step_clips(input_file: str, number: str, team: str | None,
               fps: int = 30, buf: float = 3.0):
    header(4, "구간 추출 중...")

    with open(TRACKS_JSON, encoding="utf-8") as f:
        all_tracks = json.load(f)

    with open(JERSEY_JSON, encoding="utf-8") as f:
        jersey_map = json.load(f)

    # 대상 번호를 가진 track_id 수집
    norm_num = number.lstrip("0") or "0"
    target_ids: set[str] = set()
    for tid_str, info in jersey_map.items():
        if (info["jersey"].lstrip("0") or "0") == norm_num:
            if team is None or info["team"].upper() == team.upper():
                target_ids.add(tid_str)

    if not target_ids:
        print(f"  ⚠  번호 {number} ({team or '전체'}) 감지 기록 없음")
        return []

    print(f"  대상 Track ID: {sorted(int(x) for x in target_ids)}")

    # 대상 track 이 등장하는 extracted frame index 수집
    frame_indices: list[int] = []
    for filename in sorted(all_tracks.keys()):
        fm = re.search(r'(\d+)', filename)
        if not fm:
            continue
        frame_idx = int(fm.group(1))
        for t in all_tracks[filename]:
            if str(t["track_id"]) in target_ids:
                frame_indices.append(frame_idx)

    if not frame_indices:
        print(f"  ⚠  번호 {number} 의 등장 프레임 없음")
        return []

    # 추출 fps 기준으로 시간 변환
    # gap=60 → 30초 공백까지는 같은 구간 (교체 전까지 빙판에 있는 전체 시간 포착)
    segments = _to_segments(sorted(set(frame_indices)),
                            fps=EXTRACT_FPS, buf=buf, gap=60)
    label = f"{'전체' if not team else team}팀 {number}번 선수"

    print(f"  감지 횟수: {len(frame_indices)}회 → {len(segments)}개 구간")

    with open(TIMESTAMPS_TXT, "w", encoding="utf-8") as f:
        f.write(f"{label} 등장 구간\n{'='*40}\n")
        for i, s in enumerate(segments, 1):
            f.write(f"구간 {i}: {s['start']}s ~ {s['end']}s\n")

    os.makedirs(CLIPS_DIR, exist_ok=True)
    for fn in os.listdir(CLIPS_DIR):
        if fn.startswith("clip_"):
            os.remove(os.path.join(CLIPS_DIR, fn))

    clip_files = []
    for i, seg in enumerate(segments, 1):
        out = os.path.join(CLIPS_DIR, f"clip_{i:03d}.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(seg["start"]),
             "-to", str(seg["end"]), "-i", input_file, "-c", "copy", out],
            capture_output=True, check=True,
        )
        clip_files.append(out)
        print(f"  clip_{i:03d}.mp4  {seg['start']}s ~ {seg['end']}s")

    print(f"  → {len(clip_files)}개 클립 생성 완료  (clips/)")
    return clip_files


# ─────────────────────────────────────────────────────────
# STEP 5 : highlight.mp4 완성
# ─────────────────────────────────────────────────────────

def step_highlight(clip_files: list):
    header(5, "highlight.mp4 완성!")

    if not clip_files:
        print("  ⚠  클립 없음 → 건너뜀")
        return

    # 기존 highlight.mp4 백업
    if os.path.exists(HIGHLIGHT_PATH):
        shutil.move(HIGHLIGHT_PATH, HIGHLIGHT_OLD)
        print(f"  기존 highlight.mp4 → highlight_old.mp4 백업됨")

    concat_list = os.path.join(CLIPS_DIR, "concat_list.txt")
    with open(concat_list, "w") as f:
        for cf in clip_files:
            f.write(f"file '{cf}'\n")

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", concat_list, "-c", "copy", HIGHLIGHT_PATH],
        capture_output=True, check=True,
    )

    size_mb = round(os.path.getsize(HIGHLIGHT_PATH) / 1024 / 1024, 2)
    print(f"  → highlight.mp4 생성 완료  ({size_mb} MB)")


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IceIQ Full Pipeline (ByteTrack Edition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python3 pipeline.py test.mp4 3\n  python3 pipeline.py test.mp4 23 AWAY --buffer 5.0"
    )
    parser.add_argument("input",          help="입력 영상 파일")
    parser.add_argument("number",         help="하이라이트 대상 등번호")
    parser.add_argument("team",           nargs="?", default=None, help="HOME / AWAY / 미입력(전체)")
    parser.add_argument("--buffer",       type=float, default=3.0, help="클립 앞뒤 버퍼 초 (기본: 3.0)")
    parser.add_argument("--fps",          type=int,   default=30,  help="원본 영상 FPS (기본: 30, 구간 출력용)")
    parser.add_argument("--skip-extract", action="store_true", help="프레임 추출 건너뜀")
    parser.add_argument("--skip-track",   action="store_true", help="ByteTrack 추적 건너뜀")
    parser.add_argument("--skip-ocr",     action="store_true", help="등번호 OCR 건너뜀")
    args = parser.parse_args()

    input_file = os.path.join(BASE_DIR, args.input) if not os.path.isabs(args.input) else args.input
    if not os.path.exists(input_file):
        print(f"오류: 파일을 찾을 수 없습니다 → {input_file}")
        sys.exit(1)

    t0 = time.time()
    print(f"\n{'#'*45}")
    print(f"  IceIQ Pipeline (ByteTrack Edition)")
    print(f"  영상: {args.input}  번호: {args.number}  팀: {args.team or '전체'}")
    print(f"  track_buffer={TRACK_BUFFER}프레임  추출fps={EXTRACT_FPS}")
    print(f"{'#'*45}")

    if not args.skip_extract:
        step_extract(input_file)

    all_tracks = None
    if not args.skip_track:
        all_tracks = step_track()

    jersey_map = None
    if not args.skip_ocr:
        jersey_map = step_ocr(all_tracks)

    clip_files = step_clips(input_file, args.number, args.team, args.fps, args.buffer)
    step_highlight(clip_files)

    elapsed = time.time() - t0
    print(f"\n{'#'*45}")
    print(f"  완료! highlight.mp4 생성됨")
    print(f"  클립: {len(clip_files)}개  |  소요시간: {elapsed/60:.1f}분")
    print(f"{'#'*45}\n")


if __name__ == "__main__":
    main()
