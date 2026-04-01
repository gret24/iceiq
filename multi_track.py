#!/usr/bin/env python3
"""
멀티 특징 기반 선수 추적 파이프라인 v2
- OCR 크롭 왼쪽 여백 확장 (앞자리 잘림 방지)
- 번호 후보 목록 사전 수집
- 컨텍스트 검증 (같은 팀 다른 번호와 구분)
- 10초 갭 규칙

사용법:
  python3 multi_track.py <video.mp4> <번호> [팀] [--gap 20]
"""
import argparse, json, os, re, subprocess, sys, time
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import easyocr
from ultralytics import YOLO

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
FRAMES_DIR     = os.path.join(BASE_DIR, "frames")
TRACKS_JSON    = os.path.join(BASE_DIR, "tracks.json")
JERSEY_JSON    = os.path.join(BASE_DIR, "jersey_map.json")
FEATURES_JSON  = os.path.join(BASE_DIR, "features_map.json")
# _highlight_path / _clips_dir are set dynamically inside run_pipeline
EXTRACT_FPS    = 4
MIN_BOX_AREA   = 2500
BLUR_THRESHOLD = 40

sys.path.insert(0, BASE_DIR)
try:
    from feature_extractor_v2 import extract_features, feature_similarity, feature_similarity_breakdown
except ImportError:
    from feature_extractor import extract_features, feature_similarity


def header(label):
    print(f"\n{'─'*50}\n  {label}\n{'─'*50}")


def get_team_color(bgr, x1, y1, x2, y2):
    h = y2 - y1
    ty1 = max(0, y1 + int(h*0.2))
    ty2 = min(bgr.shape[0], y1 + int(h*0.65))
    torso = bgr[ty1:ty2, x1:x2]
    if torso.size == 0: return "UNKNOWN"
    return "HOME" if np.mean(torso) > 128 else "AWAY"


def ocr_all_numbers(reader, crop_img) -> list[tuple[str, float]]:
    """이미지에서 인식 가능한 모든 번호 반환 [(번호, conf), ...]"""
    cw, ch = crop_img.size
    if cw < 8 or ch < 8: return []
    big = crop_img.resize((cw*2, ch*2), Image.LANCZOS)
    big = ImageEnhance.Contrast(big).enhance(2.0)
    results = reader.readtext(np.array(big), allowlist='0123456789', min_size=3)

    items = []
    for (bbox, text, conf) in results:
        if text.strip().isdigit():
            x_c = (bbox[0][0] + bbox[2][0]) / 2
            items.append((x_c, text.strip(), conf))
    items.sort(key=lambda x: x[0])

    candidates = []

    # 직접 인식된 숫자
    for (_, text, conf) in results:
        t = text.strip()
        if t.isdigit() and conf > 0.1:
            norm = t.lstrip("0") or "0"
            candidates.append((norm, conf))

    # 인접 숫자 합치기 (앞자리 잘림 복원)
    for i in range(len(items)-1):
        merged = items[i][1] + items[i+1][1]
        gap = items[i+1][0] - items[i][0]
        if gap < 80:  # 여백 기준 확대 (기존 60→80)
            norm = merged.lstrip("0") or "0"
            avg_conf = (items[i][2]+items[i+1][2])/2
            candidates.append((norm, avg_conf))

    return candidates


def ocr_jersey_v2(reader, crop_img, target_num, candidate_pool: set = None) -> tuple[bool, float]:
    """
    개선된 OCR:
    1. 왼쪽 여백 확장 크롭
    2. 후보 목록으로 검증
    3. 유사 번호 필터 (4/14/94 혼동 방지)
    """
    norm_target = target_num.lstrip("0") or "0"
    candidates = ocr_all_numbers(reader, crop_img)

    for (num, conf) in candidates:
        if num == norm_target:
            # 후보 목록 검증: 이 번호가 실제 존재하는 번호인지
            if candidate_pool and norm_target not in candidate_pool:
                continue  # 후보 목록에 없으면 스킵
            return True, conf

    return False, None


def build_candidate_pool(jersey_map: dict, team_filter: str = None) -> set:
    """jersey_map에서 이 팀에 실제 존재하는 번호 목록 수집"""
    pool = set()
    for info in jersey_map.values():
        if team_filter is None or info.get("team","").upper() == team_filter.upper():
            num = info.get("jersey","").lstrip("0") or "0"
            if num.isdigit():
                pool.add(num)
    return pool


def make_ocr_crop(img_rgb, x1, y1, x2, y2, iw, ih):
    """
    개선된 OCR 크롭:
    - 왼쪽 여백을 박스 너비의 20% 더 확장 (앞자리 잘림 방지)
    - 상하는 상반신 중앙 집중
    """
    w = x2 - x1
    h = y2 - y1
    # [개선 1] 왼쪽 여백 확장
    expand_left = int(w * 0.20)
    cx1 = max(0, x1 - expand_left)
    cx2 = min(iw, x2)
    cy1 = max(0, y1 + int(h * 0.10))
    cy2 = min(ih, y1 + int(h * 0.75))
    return img_rgb.crop((cx1, cy1, cx2, cy2))


def run_pipeline(video_path, target_num, team_filter=None, gap_frames=20, buf=3.0, out_suffix=None):
    sfx = out_suffix or target_num
    _highlight_path = os.path.join(BASE_DIR, f'highlight_{sfx}.mp4')
    _clips_dir      = os.path.join(BASE_DIR, f'clips_{sfx}')

    header(f"멀티 특징 추적 v2: {target_num}번 {team_filter or '전체'}")

    model  = YOLO(os.path.join(BASE_DIR, "yolov8n.pt"))
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)

    with open(TRACKS_JSON) as f:
        all_tracks = json.load(f)

    jersey_map = {}
    if os.path.exists(JERSEY_JSON):
        with open(JERSEY_JSON) as f:
            jersey_map = json.load(f)

    features_map = {}
    if os.path.exists(FEATURES_JSON):
        with open(FEATURES_JSON) as f:
            features_map = json.load(f)

    norm_target = target_num.lstrip("0") or "0"

    # ── [개선 3] 번호 후보 목록 수집
    candidate_pool = build_candidate_pool(jersey_map, team_filter)
    print(f"  번호 후보 목록 ({team_filter or '전체'}): {sorted(candidate_pool, key=lambda x: int(x) if x.isdigit() else 0)}")

    # 혼동 가능 번호 감지
    confusable = set()
    for num in candidate_pool:
        if num != norm_target and (
            norm_target in num or num in norm_target or
            num.endswith(norm_target) or norm_target.endswith(num)
        ):
            confusable.add(num)
    if confusable:
        print(f"  ⚠ 혼동 가능 번호: {sorted(confusable)} → 특징 검증 강화")

    # ── jersey_map에서 확인된 Track ID 수집
    # jersey_map에서 확인된 track 수집 + 최소 등장 프레임 필터
    confirmed_ids: set[str] = set()
    for tid_str, info in jersey_map.items():
        if (info["jersey"].lstrip("0") or "0") == norm_target:
            if team_filter is None or info["team"].upper() == team_filter.upper():
                # 해당 track이 최소 3프레임 이상 등장해야 신뢰
                tid_int = int(tid_str)
                frame_count = sum(
                    1 for fts in all_tracks.values()
                    if any(t["track_id"] == tid_int for t in fts)
                )
                if frame_count >= 10:  # 10프레임 이상만 신뢰
                    confirmed_ids.add(tid_str)

    # 최대 15개 track만 사용 (frame_count 상위)
    if len(confirmed_ids) > 15:
        with_counts = []
        for tid_str in confirmed_ids:
            tid_int = int(tid_str)
            fc = sum(1 for fts in all_tracks.values()
                     if any(t["track_id"] == tid_int for t in fts))
            with_counts.append((tid_str, fc))
        with_counts.sort(key=lambda x: x[1], reverse=True)
        confirmed_ids = set(t for t, _ in with_counts[:15])

    print(f"  jersey_map 확인 Track: {sorted(int(x) for x in confirmed_ids)}")

    # ── 특징 벡터 수집
    target_features = []
    if confirmed_ids:
        print(f"\n  특징 추출 중 (Claude Vision)...")
        processed = set()
        for fname in sorted(all_tracks.keys()):
            for t in all_tracks[fname]:
                tid = str(t["track_id"])
                if tid not in confirmed_ids or tid in processed: continue
                path = os.path.join(FRAMES_DIR, fname)
                if not os.path.exists(path): continue

                bgr = cv2.imread(path)
                if bgr is None: continue
                x1,y1,x2,y2 = t["x1"],t["y1"],t["x2"],t["y2"]
                if (x2-x1)*(y2-y1) < MIN_BOX_AREA: continue

                img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                iw, ih = img.size
                crop = img.crop((max(0,x1), max(0,y1), min(iw,x2), min(ih,y2)))

                tid_key = f"{tid}_{fname}"
                if tid_key in features_map:
                    feat = features_map[tid_key]
                else:
                    feat = extract_features(crop, int(tid))
                    features_map[tid_key] = feat
                    print(f"    Track#{tid}: 번호={feat.get('jersey_number','?')} 유니폼={feat.get('jersey_color','?')} 헬멧={feat.get('helmet_color','?')}")

                if feat:
                    target_features.append(feat)
                    processed.add(tid)
                if len(processed) >= 5: break
            if len(processed) >= 5: break

        with open(FEATURES_JSON, "w") as f:
            json.dump(features_map, f, ensure_ascii=False, indent=2)

    print(f"  대상 특징 벡터: {len(target_features)}개")

    # ── 전체 프레임 스캔
    header("멀티 특징 매칭 스캔 (v2)")
    matched_frames: list[int] = []
    checked = 0
    new_confirmed = set(confirmed_ids)
    confused_blocked = 0  # 혼동 번호로 차단된 횟수

    for fname in sorted(all_tracks.keys()):
        frame_tracks = all_tracks[fname]
        if not frame_tracks: continue

        path = os.path.join(FRAMES_DIR, fname)
        if not os.path.exists(path): continue

        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None: continue
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < BLUR_THRESHOLD: continue

        bgr = cv2.imread(path)
        img_rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        iw, ih = img_rgb.size

        fm = re.search(r'(\d+)', fname)
        if not fm: continue
        frame_idx = int(fm.group(1))

        for t in frame_tracks:
            tid = str(t["track_id"])
            x1,y1,x2,y2 = t["x1"],t["y1"],t["x2"],t["y2"]
            w,h = x2-x1, y2-y1
            if w*h < MIN_BOX_AREA: continue

            # 이미 확인된 Track
            if tid in new_confirmed:
                matched_frames.append(frame_idx)
                break

            # confirmed_ids에 없는 track은 OCR/특징 매칭 스킵
            if tid not in confirmed_ids:
                continue

            # 팀 필터
            if team_filter:
                team = get_team_color(bgr, x1, y1, x2, y2)
                if team != team_filter: continue

            # [개선 1] 왼쪽 여백 확장 크롭
            ocr_crop = make_ocr_crop(img_rgb, x1, y1, x2, y2, iw, ih)

            # 이 박스에서 인식된 모든 번호
            all_nums = ocr_all_numbers(reader, ocr_crop)

            # [개선 3] 후보 목록으로 검증
            matched_num = None
            for (num, conf) in all_nums:
                if num == norm_target and conf > 0.1:
                    matched_num = (num, conf)
                    break

            if matched_num:
                # [개선 2] 컨텍스트 검증: 혼동 번호가 같은 프레임에 있으면 특징으로 구분
                if confusable and target_features:
                    full_crop = img_rgb.crop((max(0,x1), max(0,y1), min(iw,x2), min(ih,y2)))
                    tid_key = f"{tid}_{fname}"
                    if tid_key not in features_map:
                        feat = extract_features(full_crop, int(tid))
                        features_map[tid_key] = feat if feat else {}
                    else:
                        feat = features_map[tid_key]

                    if feat:
                        sim = max(feature_similarity(feat, tf) for tf in target_features)
                        if sim < 0.55:
                            confused_blocked += 1
                            continue

                # new_confirmed 확장 완전 차단 - confirmed_ids 기반 프레임만 기록
                matched_frames.append(frame_idx)
                break

            # 특징 매칭 (OCR 실패 시)
            if target_features:
                full_crop = img_rgb.crop((max(0,x1), max(0,y1), min(iw,x2), min(ih,y2)))
                tid_key = f"{tid}_{fname}"
                if tid_key not in features_map:
                    feat = extract_features(full_crop, int(tid))
                    features_map[tid_key] = feat if feat else {}
                else:
                    feat = features_map[tid_key]

                if feat:
                    best_sim = max(feature_similarity(feat, tf) for tf in target_features)
                    if best_sim >= 0.78:
                        matched_frames.append(frame_idx)
                        break

        checked += 1
        if checked % 500 == 0:
            pct = checked / len(all_tracks) * 100
            print(f"  {checked}/{len(all_tracks)} ({pct:.0f}%)  매칭: {len(matched_frames)}  혼동차단: {confused_blocked}")

    with open(FEATURES_JSON, "w") as f:
        json.dump(features_map, f, ensure_ascii=False, indent=2)

    # ── 구간 생성 (10초 갭 규칙)
    header("구간 생성")
    if not matched_frames:
        print(f"  ⚠ {target_num}번 선수 프레임 없음")
        return

    matched_frames = sorted(set(matched_frames))
    print(f"  총 매칭 프레임: {len(matched_frames)}  혼동 차단: {confused_blocked}건")

    groups, s, e = [], matched_frames[0], matched_frames[0]
    for f in matched_frames[1:]:
        if f - e <= gap_frames:
            e = f
        else:
            groups.append((s, e))
            s = e = f
    groups.append((s, e))

    raw_segments = []
    for (gs, ge) in groups:
        start = round(max(0.0, gs / EXTRACT_FPS - buf), 2)
        end   = round(ge / EXTRACT_FPS + buf, 2)
        raw_segments.append({"start": start, "end": end})

    # 후처리: 10초 이내 갭 합치기
    GAP_SEC = 10.0
    segments = []
    for seg in raw_segments:
        if segments and seg["start"] - segments[-1]["end"] <= GAP_SEC:
            segments[-1]["end"] = max(segments[-1]["end"], seg["end"])
        else:
            segments.append(dict(seg))

    total_dur = sum(s["end"]-s["start"] for s in segments)
    print(f"  {len(raw_segments)}개 원본 → {len(segments)}개 출전 구간 (합산 {total_dur/60:.1f}분)")
    for i, seg in enumerate(segments, 1):
        dur = seg["end"] - seg["start"]
        mm, ss = divmod(int(seg["start"]), 60)
        print(f"    출전{i:02d}: {mm:02d}:{ss:02d} ~ {int(seg['end'])//60:02d}:{int(seg['end'])%60:02d}  ({dur:.0f}초)")

    # ── 클립 추출 & 하이라이트 생성
    os.makedirs(_clips_dir, exist_ok=True)
    for fn in os.listdir(_clips_dir):
        if fn.startswith("clip_"): os.remove(os.path.join(_clips_dir, fn))

    clip_files = []
    for i, seg in enumerate(segments, 1):
        out = os.path.join(_clips_dir, f"clip_{i:03d}.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(seg["start"]),
             "-to", str(seg["end"]), "-i", video_path, "-c", "copy", out],
            capture_output=True
        )
        if os.path.exists(out):
            clip_files.append(out)

    concat_file = os.path.join(_clips_dir, "concat.txt")
    with open(concat_file, "w") as f:
        for cf in clip_files:
            f.write(f"file '{cf}'\n")

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", concat_file, "-c", "copy", _highlight_path],
        capture_output=True
    )

    size_mb = os.path.getsize(_highlight_path) / 1024 / 1024
    print(f"\n  ✅ highlight_multi.mp4 완성! ({size_mb:.1f}MB, {len(clip_files)}개 클립)")


def main():
    parser = argparse.ArgumentParser(
        description="멀티 특징 기반 선수 추적 v2 - 팀 미지정(전체 스캔) 권장",
        epilog="예시: python3 multi_track.py input_video.mp4 4\n       python3 multi_track.py input_video.mp4 94 --out test_94")
    parser.add_argument("video")
    parser.add_argument("number")
    parser.add_argument("team", nargs="?", default=None, help="팀 필터: HOME / AWAY (미입력 시 전체 스캔 - 기본값 권장)")
    parser.add_argument("--gap",  type=int,   default=10)
    parser.add_argument("--buf",  type=float, default=3.0)
    parser.add_argument("--out",  type=str,   default=None, help="출력 파일 suffix (기본: 번호)")
    args = parser.parse_args()

    video_path = os.path.join(BASE_DIR, args.video) if not os.path.isabs(args.video) else args.video
    run_pipeline(video_path, args.number, args.team, args.gap, args.buf, args.out)


if __name__ == "__main__":
    main()
