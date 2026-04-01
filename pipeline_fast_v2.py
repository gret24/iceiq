#!/usr/bin/env python3
"""
pipeline_fast_v2.py — PARSeq OCR + 3단계 선수 식별 + OCR 최소화
- EasyOCR 대신 PARSeq (GPU 배치 추론)
- confirmed_tracks 캐시로 OCR 최소화
- RosterManager 3단계 식별 통합
- 4프레임마다 OCR, bbox 조건 필터
"""
import json, os, re, shutil, sys, time, io, argparse
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw
from ultralytics import YOLO
import torch
from torchvision import transforms

# ── 경로 (동적 설정) ────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR   = os.path.join(BASE_DIR, "frames")
DETECTED_DIR = os.path.join(BASE_DIR, "detected")
TRACKS_JSON  = os.path.join(BASE_DIR, "tracks.json")
JERSEY_JSON  = os.path.join(BASE_DIR, "jersey_map.json")
RESULTS_TXT  = os.path.join(BASE_DIR, "results.txt")

EXTRACT_FPS      = 4
MIN_BOX_AREA     = 3000
BLUR_THRESHOLD   = 50
BATCH_SIZE       = 8
MIN_BBOX_HEIGHT  = 50   # px
EDGE_MARGIN      = 0.10 # 10% 경계 무시
OCR_FRAME_STRIDE = 4    # 4프레임마다 OCR
OCR_CONFIRM_THRESH = 2  # 2번 이상 같은 번호 → 확정


def header(step, label):
    print(f"\n[{step}] {label}\n" + "-"*45)


# ── PARSeq 모델 로드 ─────────────────────────────────────────
_parseq_model = None
_parseq_transform = None

def get_parseq():
    global _parseq_model, _parseq_transform
    if _parseq_model is None:
        print("  PARSeq 모델 로딩 중...")
        _parseq_model = torch.hub.load(
            "baudm/parseq", "parseq",
            pretrained=True, trust_repo=True
        )
        _parseq_model = _parseq_model.eval().cuda()
        _parseq_transform = transforms.Compose([
            transforms.Resize(
                (32, 128),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])
        print("  PARSeq 로딩 완료")
    return _parseq_model, _parseq_transform


def _parseq_batch_ocr(pil_images: list) -> list[tuple[str, float]]:
    """
    PIL 이미지 배치를 PARSeq로 추론.
    Returns list of (text, confidence) per image.
    """
    if not pil_images:
        return []
    model, transform = get_parseq()
    tensors = [transform(img.convert("RGB")) for img in pil_images]
    batch = torch.stack(tensors).cuda()
    with torch.no_grad():
        logits = model(batch)
    # decode: logits → text
    probs = logits.softmax(-1)
    preds, pred_confs = model.tokenizer.decode(probs)

    results = []
    for pred, confs in zip(preds, pred_confs):
        # 숫자만 필터
        digits_only = re.sub(r'[^0-9]', '', pred)
        if digits_only:
            conf = float(confs.mean().item()) if hasattr(confs, 'mean') else 0.5
            results.append((digits_only, conf))
        else:
            results.append(("", 0.0))
    return results


def _is_jersey(text: str) -> bool:
    return bool(re.match(r'^\d{1,2}$', text.strip())) and 1 <= int(text.strip()) <= 99


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


# ── 색상 특징 추출 ─────────────────────────────────────────
def extract_color_features(img_rgb: np.ndarray, x1, y1, x2, y2) -> dict:
    """토르소 영역에서 HSV 평균 색상 특징 추출"""
    h = y2 - y1
    ty1 = max(0, y1 + int(h * 0.20))
    ty2 = min(img_rgb.shape[0], y1 + int(h * 0.65))
    torso = img_rgb[ty1:ty2, x1:x2]
    if torso.size == 0:
        return {}
    torso_hsv = cv2.cvtColor(torso, cv2.COLOR_RGB2HSV).astype(np.float32)
    return {
        "h_mean": float(np.mean(torso_hsv[:, :, 0])),
        "s_mean": float(np.mean(torso_hsv[:, :, 1])),
        "v_mean": float(np.mean(torso_hsv[:, :, 2])),
        "h_std":  float(np.std(torso_hsv[:, :, 0])),
    }


# ── Step 2: ByteTrack (배치 처리) ─────────────────────────
def step_track_fast():
    header(2, f"ByteTrack 추적 (배치={BATCH_SIZE})")
    os.makedirs(DETECTED_DIR, exist_ok=True)

    model = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))
    files = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
    total = len(files)
    all_tracks = {}

    CORRECTED_DIR = os.path.join(BASE_DIR, "_corrected_tmp")
    os.makedirs(CORRECTED_DIR, exist_ok=True)

    t0 = time.time()

    def preprocess(fname):
        path = os.path.join(FRAMES_DIR, fname)
        corrected = _correct_ice_glare(path)
        if corrected is not None:
            tmp = os.path.join(CORRECTED_DIR, fname)
            cv2.imwrite(tmp, corrected)
            return fname, tmp
        return fname, path

    with ThreadPoolExecutor(max_workers=4) as ex:
        corrected_map = dict(ex.map(preprocess, files))

    print(f"  보정 완료: {len(corrected_map)}장 ({time.time()-t0:.1f}초)")

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


# ── Step 3: OCR (PARSeq + 최소화 + RosterManager) ─────────
def step_ocr_fast(all_tracks=None, roster_mgr=None):
    header(3, "등번호 OCR (PARSeq + 최소화 + 3단계 식별)")

    if all_tracks is None:
        with open(TRACKS_JSON) as f:
            all_tracks = json.load(f)

    # PARSeq 초기화
    get_parseq()

    # 프레임 크기 감지 (첫 번째 프레임으로)
    frame_w, frame_h = 1920, 1080  # 기본값
    first_files = sorted(all_tracks.keys())
    for fn in first_files[:5]:
        fp = os.path.join(FRAMES_DIR, fn)
        if os.path.exists(fp):
            img = cv2.imread(fp)
            if img is not None:
                frame_h, frame_w = img.shape[:2]
                break

    # 통계
    confirmed_tracks = {}           # track_id → jersey_number (확정)
    ocr_vote = defaultdict(Counter) # track_id → {num: count}
    ocr_call_count = 0
    cache_hit_count = 0
    t0 = time.time()

    # 로스터에서 확인 대상 번호 수집
    roster_numbers = set()
    if roster_mgr:
        for n in roster_mgr._home_roster + roster_mgr._away_roster:
            roster_numbers.add(str(n))

    # 프레임 목록 (정렬)
    sorted_frames = sorted(all_tracks.keys())
    total_frames = len(sorted_frames)
    total_tracks_all = len({
        t["track_id"]
        for ts in all_tracks.values()
        for t in ts
    })

    jersey_map = {}
    last_pct = -1

    for frame_idx, filename in enumerate(sorted_frames):
        frame_tracks = all_tracks[filename]
        if not frame_tracks:
            continue

        # 진행률 (10% 단위)
        pct = int(frame_idx / total_frames * 100)
        if pct // 10 > last_pct // 10 and pct > 0:
            elapsed = time.time() - t0
            total_calls = ocr_call_count + cache_hit_count
            hit_rate = cache_hit_count / total_calls * 100 if total_calls > 0 else 0.0
            confirmed_count = len(confirmed_tracks)
            roster_target = len(roster_numbers) if roster_numbers else total_tracks_all
            print(
                f"처리: {frame_idx}/{total_frames} ({pct}%) | "
                f"OCR호출: {ocr_call_count} | "
                f"캐시히트: {hit_rate:.1f}% | "
                f"확정: {confirmed_count}/{roster_target}"
            )
            last_pct = pct

        # 이미지 로드
        img_path = os.path.join(FRAMES_DIR, filename)
        if not os.path.exists(img_path):
            continue

        img_pil = None  # lazy load
        img_bgr = None

        # 프레임 내 처리할 크롭 배치 수집
        batch_crops = []   # (track_id, pil_crop)
        batch_meta  = []   # track metadata

        for track in frame_tracks:
            tid = track["track_id"]
            x1, y1, x2, y2 = track["x1"], track["y1"], track["x2"], track["y2"]
            w = x2 - x1
            h = y2 - y1

            # RosterManager bbox 버퍼링
            if roster_mgr:
                roster_mgr.buffer_bbox(tid, (x1, y1, x2, y2), frame_idx)

            # ── OCR 스킵 조건들 ──────────────────────────────

            # 1) 이미 확정된 track
            if tid in confirmed_tracks:
                cache_hit_count += 1
                jersey_map[str(tid)] = confirmed_tracks[tid]
                continue

            # 2) bbox 높이 너무 작음
            if h < MIN_BBOX_HEIGHT:
                continue

            # 3) 화면 경계 영역
            if (y1 < frame_h * EDGE_MARGIN or
                y2 > frame_h * (1 - EDGE_MARGIN) or
                x1 < frame_w * EDGE_MARGIN or
                x2 > frame_w * (1 - EDGE_MARGIN)):
                continue

            # 4) 4프레임마다만 OCR
            if frame_idx % OCR_FRAME_STRIDE != 0:
                continue

            # 5) 조기 종료: 로스터가 있고 모든 번호 확정
            if roster_mgr and roster_numbers:
                confirmed_nums = {v.get("jersey") if isinstance(v, dict) else v
                                  for v in confirmed_tracks.values()}
                if roster_numbers.issubset(confirmed_nums):
                    cache_hit_count += 1
                    continue

            # 이미지 lazy load
            if img_pil is None:
                img_pil = Image.open(img_path).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            iw, ih = img_pil.size
            # 상체 크롭 (10%~75% 높이)
            expand_left = int(w * 0.20)
            cx1 = max(0, x1 - expand_left)
            cy1 = max(0, y1 + int(h * 0.10))
            cy2 = min(ih, y1 + int(h * 0.75))
            if cy2 - cy1 < 5 or (x2 - cx1) < 5:
                continue

            crop = img_pil.crop((cx1, cy1, min(iw, x2), cy2))
            # 2x 확대 + 대비 강화
            cw, ch_c = crop.size
            if cw < 5 or ch_c < 5:
                continue
            crop = crop.resize((cw*2, ch_c*2), Image.LANCZOS)
            crop = ImageEnhance.Contrast(crop).enhance(2.0)

            batch_crops.append(crop)
            batch_meta.append((tid, x1, y1, x2, y2))

        # ── PARSeq 배치 추론 ─────────────────────────────────
        if batch_crops:
            ocr_call_count += len(batch_crops)
            ocr_results = _parseq_batch_ocr(batch_crops)

            img_rgb_arr = None

            for (tid, x1, y1, x2, y2), (text, conf) in zip(batch_meta, ocr_results):
                if not text:
                    continue

                # 숫자 1~2자리만
                digits = re.sub(r'[^0-9]', '', text)
                if not digits:
                    continue

                # 1~2자리 번호 추출
                num = digits[:2] if len(digits) >= 2 else digits
                if not _is_jersey(num):
                    # 두 자리씩 시도
                    for i in range(len(digits) - 1):
                        candidate = digits[i:i+2]
                        if _is_jersey(candidate):
                            num = candidate
                            break
                    else:
                        if digits[:1] and _is_jersey(digits[:1]):
                            num = digits[:1]
                        else:
                            continue

                # 로스터 매칭 (있으면)
                if roster_mgr and roster_numbers:
                    matched = roster_mgr.roster_match(num)
                    if matched:
                        num = matched

                ocr_vote[tid][num] += 1

                # 확정 체크 (같은 번호 2회 이상)
                if ocr_vote[tid][num] >= OCR_CONFIRM_THRESH:
                    # 색상 특징 추출
                    if img_bgr is not None:
                        if img_rgb_arr is None:
                            img_rgb_arr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        color_feats = extract_color_features(img_rgb_arr, x1, y1, x2, y2)
                    else:
                        color_feats = {}

                    # 팀 추정
                    team = "HOME" if color_feats.get("v_mean", 128) > 128 else "AWAY"

                    entry = {"jersey": num, "team": team}
                    confirmed_tracks[tid] = entry
                    jersey_map[str(tid)] = entry

                    # RosterManager에 색상 특징 등록
                    if roster_mgr:
                        roster_mgr.register_features(num, team, color_feats)

        # ── OCR 없이 지나간 track → RosterManager로 보조 식별 ──
        if roster_mgr and img_pil is not None:
            if img_bgr is None:
                img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            img_rgb_arr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            for track in frame_tracks:
                tid = track["track_id"]
                if str(tid) in jersey_map:
                    continue
                if tid in confirmed_tracks:
                    continue

                x1, y1, x2, y2 = track["x1"], track["y1"], track["x2"], track["y2"]
                color_feats = extract_color_features(img_rgb_arr, x1, y1, x2, y2)
                if not color_feats:
                    continue

                result = roster_mgr.identify_player(
                    track_id=tid,
                    frame_idx=frame_idx,
                    ocr_text="",
                    ocr_confidence=0.0,
                    color_features=color_feats,
                )
                if result:
                    team = "HOME" if color_feats.get("v_mean", 128) > 128 else "AWAY"
                    jersey_map[str(tid)] = {"jersey": result, "team": team}

    # 최종 통계 출력
    elapsed = time.time() - t0
    total_calls = ocr_call_count + cache_hit_count
    hit_rate = cache_hit_count / total_calls * 100 if total_calls > 0 else 0.0
    baseline = total_frames  # 기존 EasyOCR는 매 프레임 track마다

    print(f"\n처리: {total_frames}/{total_frames} (100%) | "
          f"OCR호출: {ocr_call_count} | "
          f"캐시히트: {hit_rate:.1f}% | "
          f"확정: {len(confirmed_tracks)}/{len(roster_numbers) if roster_numbers else total_tracks_all}")

    reduction = (1 - ocr_call_count / max(baseline, 1)) * 100
    print(f"\n=== pipeline_fast_v2 통계 ===")
    print(f"PARSeq OCR 호출: {ocr_call_count}건 (기존 EasyOCR: ~{baseline}건 대비 {reduction:.1f}% 감소)")
    print(f"캐시 히트율: {hit_rate:.1f}%")

    if roster_mgr:
        stats = roster_mgr.get_stats()
        total_id = stats.get("_total", 0)
        print(f"method별 식별:")
        for method in ["direct_ocr", "correction_map", "color_match",
                        "behavior_match", "combined_match", "history", "failed"]:
            info = stats.get(method, {"count": 0, "pct": 0.0})
            if isinstance(info, dict):
                cnt = info.get("count", 0)
                pct = info.get("pct", 0.0)
            else:
                cnt, pct = info, 0.0
            if cnt > 0:
                print(f"  {method:<16}: {cnt}건 ({pct:.1f}%)")

    print(f"총 처리시간: {elapsed/60:.1f}분")

    with open(JERSEY_JSON, "w") as f:
        json.dump(jersey_map, f, ensure_ascii=False)

    unique_ids = len(jersey_map)
    print(f"\n  → 완료: {unique_ids}개 Track 식별 | {elapsed/60:.1f}분")
    return jersey_map


# ── Step 4: 타겟 번호 결과 출력 ──────────────────────────
def step_find_target(jersey_map, target_num, all_tracks=None):
    header(4, f"#{target_num} 검색")

    norm_target = target_num.lstrip("0") or "0"

    if all_tracks is None:
        with open(TRACKS_JSON) as f:
            all_tracks = json.load(f)

    matched_tracks = []
    for tid_str, info in jersey_map.items():
        jersey = info.get("jersey") if isinstance(info, dict) else str(info)
        jersey_norm = (jersey or "").lstrip("0") or "0"
        if jersey_norm == norm_target:
            matched_tracks.append(int(tid_str))

    if not matched_tracks:
        print(f"  #{target_num} 번호를 가진 Track 없음")
        return []

    print(f"  #{target_num} Track ID: {sorted(matched_tracks)}")

    # 각 트랙의 프레임 수 통계
    track_frame_counts = defaultdict(int)
    for fname, fts in all_tracks.items():
        for t in fts:
            if t["track_id"] in matched_tracks:
                track_frame_counts[t["track_id"]] += 1

    for tid in sorted(matched_tracks):
        team = jersey_map.get(str(tid), {})
        team_str = team.get("team", "?") if isinstance(team, dict) else "?"
        frames = track_frame_counts.get(tid, 0)
        print(f"    Track {tid}: {frames}프레임 | 팀: {team_str}")

    print(f"  총 {len(matched_tracks)}개 Track에서 #{target_num} 발견")

    with open(RESULTS_TXT, "w") as f:
        f.write(f"#{target_num} 탐지 결과\n")
        f.write(f"Track IDs: {sorted(matched_tracks)}\n")
        for tid in sorted(matched_tracks):
            frames = track_frame_counts.get(tid, 0)
            f.write(f"  Track {tid}: {frames}프레임\n")

    return matched_tracks


# ── main ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="pipeline_fast_v2: PARSeq OCR + 3단계 식별")
    parser.add_argument("video", help="비디오 파일 경로")
    parser.add_argument("number", nargs="?", help="추적할 선수 번호")
    parser.add_argument("--player", dest="player", help="추적할 선수 번호 (--player 형태)")
    parser.add_argument("--home-roster", type=str, default="", help="홈팀 로스터 (쉼표 구분)")
    parser.add_argument("--away-roster", type=str, default="", help="어웨이팀 로스터 (쉼표 구분)")
    parser.add_argument("--skip-extract", action="store_true", help="프레임 추출 건너뜀")
    parser.add_argument("--skip-track", action="store_true", help="ByteTrack 건너뜀")
    parser.add_argument("--base-dir", type=str, default="", help="작업 디렉토리")

    args = parser.parse_args()

    # 타겟 번호
    target_num = args.player or args.number or ""

    # base dir 설정
    global BASE_DIR, FRAMES_DIR, DETECTED_DIR, TRACKS_JSON, JERSEY_JSON, RESULTS_TXT
    if args.base_dir:
        BASE_DIR = args.base_dir
    elif args.video and not args.video.endswith(".mp4"):
        BASE_DIR = args.video
    else:
        BASE_DIR = os.path.dirname(os.path.abspath(args.video)) if args.video else BASE_DIR

    FRAMES_DIR   = os.path.join(BASE_DIR, "frames")
    DETECTED_DIR = os.path.join(BASE_DIR, "detected")
    TRACKS_JSON  = os.path.join(BASE_DIR, "tracks.json")
    JERSEY_JSON  = os.path.join(BASE_DIR, "jersey_map.json")
    RESULTS_TXT  = os.path.join(BASE_DIR, "results.txt")

    print(f"\n=== pipeline_fast_v2 시작 ===")
    print(f"비디오: {args.video}")
    print(f"타겟 번호: #{target_num}")
    print(f"작업 디렉토리: {BASE_DIR}")

    t0 = time.time()

    # ── Step 1: 프레임 추출 ────────────────────────────────
    if not args.skip_extract:
        header(1, "프레임 추출")
        os.makedirs(FRAMES_DIR, exist_ok=True)
        fps_arg = str(EXTRACT_FPS)
        video_path = args.video
        if not os.path.isabs(video_path):
            video_path = os.path.join(BASE_DIR, video_path)
        ret = os.system(
            f'ffmpeg -i "{video_path}" -vf fps={fps_arg} '
            f'"{FRAMES_DIR}/frame_%06d.jpg" -y -loglevel error'
        )
        frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg")])
        print(f"  추출 완료: {len(frames)}프레임")
    else:
        print("[1] 프레임 추출 건너뜀")

    # ── Step 2: ByteTrack ──────────────────────────────────
    if not args.skip_track:
        all_tracks = step_track_fast()
    else:
        print("[2] ByteTrack 건너뜀")
        with open(TRACKS_JSON) as f:
            all_tracks = json.load(f)

    # ── RosterManager 초기화 ───────────────────────────────
    try:
        sys.path.insert(0, BASE_DIR)
        from roster_manager import RosterManager
        roster_mgr = RosterManager()

        home_list = [n.strip() for n in args.home_roster.split(",") if n.strip()]
        away_list = [n.strip() for n in args.away_roster.split(",") if n.strip()]

        if home_list or away_list:
            home_ints = [int(n) for n in home_list if n.isdigit()]
            away_ints = [int(n) for n in away_list if n.isdigit()]
            roster_mgr.set_roster(home=home_ints, away=away_ints)
            print(f"  RosterManager 초기화: HOME={home_list} AWAY={away_list}")
        else:
            roster_mgr = None
            print("  로스터 미설정 → RosterManager 비활성화")
    except ImportError as e:
        print(f"  ⚠ RosterManager 임포트 실패: {e}")
        roster_mgr = None

    # ── Step 3: OCR ───────────────────────────────────────
    jersey_map = step_ocr_fast(all_tracks, roster_mgr=roster_mgr)

    # ── Step 4: 타겟 번호 검색 ────────────────────────────
    if target_num:
        matched = step_find_target(jersey_map, target_num, all_tracks)

    total_elapsed = time.time() - t0
    print(f"\n총 소요: {total_elapsed/60:.1f}분")


if __name__ == "__main__":
    main()
