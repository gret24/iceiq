#!/usr/bin/env python3
"""
HockeyTV 채널 스트리밍 분석
- yt-dlp 스트리밍 (다운로드 없이)
- 영상당 프레임 20장 샘플 추출
- YOLOv8 선수 감지 + EasyOCR 등번호 인식
- 결과: channel_results.txt
"""

import json, os, re, subprocess, sys, tempfile, time
import numpy as np
import cv2
import easyocr
from PIL import Image, ImageEnhance
from ultralytics import YOLO

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE = os.path.join(BASE_DIR, "channel_results.txt")
CHANNEL_URL = "https://youtube.com/@hockeytv_"
NUM_VIDEOS  = 3
FRAMES_PER_VIDEO = 20
MIN_BOX_AREA = 5000
BLUR_THRESHOLD = 50

print("=" * 55)
print("  HockeyTV 채널 스트리밍 분석")
print("=" * 55)

# ── 1. 채널 영상 목록 수집 ─────────────────────────────
print(f"\n[1/3] 채널 영상 목록 수집 중...")
result = subprocess.run(
    ["yt-dlp", "--flat-playlist", "-j", "--playlist-end", str(NUM_VIDEOS), CHANNEL_URL],
    capture_output=True, text=True, timeout=30
)
videos = []
for line in result.stdout.strip().split("\n"):
    if not line.strip():
        continue
    try:
        info = json.loads(line)
        vid_id = info.get("id", "")
        title  = info.get("title", "")[:60]
        if vid_id:
            videos.append({"id": vid_id, "title": title, "url": f"https://youtube.com/watch?v={vid_id}"})
    except:
        pass

print(f"  → {len(videos)}개 영상 수집:")
for v in videos:
    print(f"    - [{v['id']}] {v['title']}")

# ── 모델 로드 ─────────────────────────────────────────
print("\n[2/3] 모델 로딩...")
model  = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
print("  → YOLO + EasyOCR 준비 완료")

# ── 분석 함수 ─────────────────────────────────────────
def get_stream_url(video_url):
    """yt-dlp로 스트리밍 URL 추출"""
    r = subprocess.run(
        ["yt-dlp", "-f", "best[height<=720]/best", "-g", video_url],
        capture_output=True, text=True, timeout=20
    )
    return r.stdout.strip().split("\n")[0] if r.returncode == 0 else None

def extract_frames_streaming(stream_url, n_frames=20):
    """ffmpeg 스트리밍으로 프레임 추출 (파일 저장 없이)"""
    # 영상 길이 확인
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", stream_url],
        capture_output=True, text=True, timeout=15
    )
    duration = 60.0
    try:
        fmt = json.loads(probe.stdout).get("format", {})
        duration = float(fmt.get("duration", 60))
    except:
        pass

    # 균등 간격으로 타임스탬프 선택 (앞뒤 10% 제외)
    start = duration * 0.1
    end   = duration * 0.9
    timestamps = [start + (end - start) * i / (n_frames - 1) for i in range(n_frames)]

    frames = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for ts in timestamps:
            out_path = os.path.join(tmpdir, "frame.jpg")
            r = subprocess.run(
                ["ffmpeg", "-y", "-ss", str(ts), "-i", stream_url,
                 "-frames:v", "1", "-q:v", "2", out_path],
                capture_output=True, timeout=15
            )
            if r.returncode == 0 and os.path.exists(out_path):
                img = cv2.imread(out_path)
                if img is not None:
                    frames.append((ts, img.copy()))
    return frames

def analyze_frame(bgr_img):
    """프레임 분석: 선수 감지 + 등번호 인식"""
    tmp = "/tmp/analysis_frame.jpg"
    cv2.imwrite(tmp, bgr_img)

    # 블러 체크
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # YOLO 감지
    results = model(tmp, classes=[0], verbose=False, conf=0.25)[0]
    boxes = results.boxes
    valid_boxes = []
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if (x2-x1)*(y2-y1) >= MIN_BOX_AREA:
                valid_boxes.append((int(x1), int(y1), int(x2), int(y2)))

    # 등번호 인식 (선명한 프레임만)
    found_number = None
    if valid_boxes and sharpness >= BLUR_THRESHOLD:
        h, w = bgr_img.shape[:2]
        img_pil = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        for (x1, y1, x2, y2) in valid_boxes:
            crop = img_pil.crop((x1, y1, x2, y2))
            cw, ch = crop.size
            if cw < 10 or ch < 10:
                continue
            big = crop.resize((cw*2, ch*2), Image.LANCZOS)
            big = ImageEnhance.Contrast(big).enhance(2.0)
            ocr_results = reader.readtext(np.array(big), allowlist='0123456789', min_size=5)
            for (_, text, conf) in ocr_results:
                clean = re.sub(r'[^0-9]', '', text)
                if clean and 1 <= int(clean) <= 99 and conf > 0.1:
                    found_number = clean
                    break
            if found_number:
                break

    return {
        "detected": len(valid_boxes) > 0,
        "count": len(valid_boxes),
        "jersey": found_number,
        "sharpness": round(sharpness, 1)
    }

# ── 3. 영상별 분석 ────────────────────────────────────
print("\n[3/3] 영상 분석 시작...\n")
all_results = []

for vi, video in enumerate(videos, 1):
    print(f"  [{vi}/{len(videos)}] {video['title'][:50]}")
    print(f"    URL: {video['url']}")

    t0 = time.time()

    # 스트리밍 URL 추출
    stream_url = get_stream_url(video['url'])
    if not stream_url:
        print(f"    ⚠️ 스트리밍 URL 추출 실패, 스킵")
        continue
    print(f"    스트리밍 URL 추출 완료")

    # 프레임 추출
    frames = extract_frames_streaming(stream_url, FRAMES_PER_VIDEO)
    print(f"    프레임 추출: {len(frames)}장")

    if not frames:
        print(f"    ⚠️ 프레임 추출 실패, 스킵")
        continue

    # 분석
    detected_count = 0
    jersey_count = 0
    frame_results = []

    for fi, (ts, bgr) in enumerate(frames):
        res = analyze_frame(bgr)
        frame_results.append(res)
        if res["detected"]: detected_count += 1
        if res["jersey"]:   jersey_count += 1
        print(f"    프레임 {fi+1:2d}: 선수 {res['count']}명 {'✅' if res['detected'] else '❌'}  "
              f"등번호 {'#'+res['jersey'] if res['jersey'] else '❌'}  "
              f"선명도:{res['sharpness']:.0f}")

    detect_rate = detected_count / len(frames) * 100
    jersey_rate = jersey_count / len(frames) * 100
    elapsed = time.time() - t0

    print(f"\n    ── 결과: 감지율 {detect_rate:.0f}%  등번호 {jersey_rate:.0f}%  ({elapsed:.0f}초)\n")

    all_results.append({
        "title": video["title"],
        "url": video["url"],
        "frames": len(frames),
        "detect_rate": detect_rate,
        "jersey_rate": jersey_rate,
        "frame_results": frame_results
    })

# ── 결과 저장 ─────────────────────────────────────────
with open(RESULT_FILE, "w", encoding="utf-8") as f:
    f.write("HockeyTV 채널 스트리밍 분석 결과\n")
    f.write("=" * 55 + "\n\n")
    total_detect = sum(r["detect_rate"] for r in all_results) / max(1, len(all_results))
    total_jersey = sum(r["jersey_rate"] for r in all_results) / max(1, len(all_results))

    for r in all_results:
        f.write(f"영상: {r['title']}\n")
        f.write(f"URL: {r['url']}\n")
        f.write(f"선수 감지율: {r['detect_rate']:.1f}%\n")
        f.write(f"등번호 인식율: {r['jersey_rate']:.1f}%\n")
        f.write("-" * 40 + "\n")

    f.write(f"\n평균 선수 감지율: {total_detect:.1f}%\n")
    f.write(f"평균 등번호 인식율: {total_jersey:.1f}%\n")
    f.write(f"목표 70%+ {'달성!' if total_jersey >= 70 else '미달'}\n")

print("=" * 55)
print(f"  평균 선수 감지율: {total_detect:.1f}%")
print(f"  평균 등번호 인식율: {total_jersey:.1f}%")
print(f"  목표 70%+ {'✅ 달성!' if total_jersey >= 70 else '❌ 미달'}")
print(f"  결과 저장: channel_results.txt")
print("=" * 55)
