#!/usr/bin/env python3
"""
build_highlight.py — 후처리 전용 하이라이트 생성기
pipeline_fast.py 분석 결과(tracks.json + jersey_map.json)를 읽어서
특정 선수의 시프트만 이어붙인 하이라이트 영상 생성.

분석을 다시 돌리지 않고 결과 파일만 사용.

사용법:
  python build_highlight.py \\
    --input /path/to/original.mp4 \\
    --data  /path/to/analysis_dir  \\   # tracks.json + jersey_map.json 위치
    --player 47 \\
    --output /workspace/iceiq/output/47_fulltime.mp4 \\
    --gap 10

  (--output 생략 시 자동으로 /workspace/iceiq/output/{player}_fulltime.mp4)
"""
import argparse, json, math, os, re, shutil, subprocess, sys, time


# ── 시프트 추출 ──────────────────────────────────────────────
def extract_shifts(jersey_map: dict, tracks_data: dict,
                   player: str,
                   gap_sec: float = 10.0,
                   buf_sec: float = 5.0,
                   fps: int = 4) -> list[dict]:
    norm = player.lstrip("0") or "0"
    target_tids = {tid for tid, info in jersey_map.items()
                   if info["jersey"].lstrip("0") == norm}

    print(f"  #{player} 해당 track: {len(target_tids)}개")
    if not target_tids:
        return []

    # 등장 프레임 인덱스 수집
    frame_indices = []
    for fname, fts in sorted(tracks_data.items()):
        m = re.search(r"(\d+)", fname)
        if not m:
            continue
        fidx = int(m.group(1))
        for t in fts:
            if str(t["track_id"]) in target_tids:
                frame_indices.append(fidx)
                break

    if not frame_indices:
        return []

    frame_indices = sorted(set(frame_indices))
    gap_frames = int(gap_sec * fps)

    # 연속 프레임 → 그룹
    groups, s, e = [], frame_indices[0], frame_indices[0]
    for f in frame_indices[1:]:
        if f - e <= gap_frames:
            e = f
        else:
            groups.append((s, e))
            s = e = f
    groups.append((s, e))

    # 그룹 → 시프트 (버퍼 추가)
    shifts = []
    for i, (gs, ge) in enumerate(groups, 1):
        start = round(max(0.0, gs / fps - buf_sec), 2)
        end   = round(ge / fps + buf_sec, 2)
        shifts.append({
            "shift_number": i,
            "start_time":   start,
            "end_time":     end,
            "duration":     round(end - start, 2),
        })
    return shifts


# ── 하이라이트 영상 생성 ─────────────────────────────────────
def build_video(video: str, shifts: list[dict],
                player: str, output: str) -> bool:
    out_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, f"_tmp_{player}")
    os.makedirs(tmp_dir, exist_ok=True)

    clip_files = []
    for sh in shifts:
        i   = sh["shift_number"]
        mm  = int(sh["start_time"]) // 60
        ss  = int(sh["start_time"]) % 60
        label = f"Shift {i} - {mm}:{ss:02d}"

        vf = (
            f"drawtext=text='{label}':"
            f"fontsize=36:fontcolor=white:"
            f"x=20:y=20:"
            f"box=1:boxcolor=black@0.5:boxborderw=5:"
            f"enable='between(t,0,2)'"
        )
        clip = os.path.join(tmp_dir, f"clip_{i:03d}.mp4")
        r = subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(sh["start_time"]),
            "-to", str(sh["end_time"]),
            "-i", video,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac",
            clip
        ], capture_output=True)

        if r.returncode == 0 and os.path.exists(clip):
            size = os.path.getsize(clip) / 1024 / 1024
            mm2, ss2 = divmod(int(sh["end_time"]), 60)
            print(f"  shift {i:2d}: {mm:02d}:{ss:02d} ~ "
                  f"{mm2:02d}:{ss2:02d}  ({sh['duration']:.0f}초, {size:.1f}MB)")
            clip_files.append(clip)
        else:
            err = r.stderr[-200:].decode(errors="ignore") if r.stderr else "unknown"
            print(f"  shift {i}: 실패 — {err}")

    if not clip_files:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return False

    # concat
    concat_txt = os.path.join(tmp_dir, "concat.txt")
    with open(concat_txt, "w") as f:
        for c in clip_files:
            f.write(f"file '{c}'\n")

    r2 = subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_txt,
        "-c", "copy",
        output
    ], capture_output=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if r2.returncode == 0 and os.path.exists(output):
        size = os.path.getsize(output) / 1024 / 1024
        # 영상 길이
        r3 = subprocess.run([
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", output
        ], capture_output=True, text=True)
        dur = float(r3.stdout.strip()) if r3.stdout.strip() else 0
        print(f"\n  ✅ {os.path.basename(output)}")
        print(f"     영상 길이: {dur/60:.1f}분 ({dur:.0f}초)")
        print(f"     파일 크기: {size:.1f}MB")
        return True
    else:
        print(f"  ❌ concat 실패")
        return False


# ── 메타데이터 저장 ──────────────────────────────────────────
def save_shifts_json(shifts: list[dict], player: str, out_dir: str) -> str:
    total_ice = sum(s["duration"] for s in shifts)
    avg_shift = total_ice / len(shifts) if shifts else 0
    meta = {
        "player":           player,
        "total_shifts":     len(shifts),
        "total_ice_time_sec": round(total_ice, 2),
        "average_shift_sec":  round(avg_shift, 2),
        "shifts": shifts,
    }
    path = os.path.join(out_dir, f"{player}_shifts.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  메타데이터: {path}")
    return path


# ── main ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="pipeline_fast.py 분석 결과에서 선수 하이라이트 생성 (후처리 전용)"
    )
    parser.add_argument("--input",   required=True,
                        help="원본 영상 경로")
    parser.add_argument("--data",    required=True,
                        help="분석 결과 디렉토리 (tracks.json + jersey_map.json 위치)")
    parser.add_argument("--player",  required=True,
                        help="선수 등번호 (예: 47)")
    parser.add_argument("--output",  default=None,
                        help="출력 mp4 경로 (기본: /workspace/iceiq/output/{player}_fulltime.mp4)")
    parser.add_argument("--gap",     type=float, default=10.0,
                        help="시프트 구분 기준 초 (기본 10)")
    parser.add_argument("--buf",     type=float, default=5.0,
                        help="시프트 앞뒤 버퍼 초 (기본 5)")
    parser.add_argument("--fps",     type=int,   default=4,
                        help="분석 FPS (기본 4)")
    args = parser.parse_args()

    # 출력 경로 기본값
    if args.output is None:
        args.output = f"/workspace/iceiq/output/{args.player}_fulltime.mp4"
    out_dir = os.path.dirname(os.path.abspath(args.output))

    # 분석 결과 로드
    tracks_path = os.path.join(args.data, "tracks.json")
    jersey_path = os.path.join(args.data, "jersey_map.json")

    for p, name in [(tracks_path, "tracks.json"), (jersey_path, "jersey_map.json")]:
        if not os.path.exists(p):
            print(f"❌ {name} 없음: {p}")
            sys.exit(1)

    with open(tracks_path) as f:
        tracks = json.load(f)
    with open(jersey_path) as f:
        jmap = json.load(f)

    t0 = time.time()
    print(f"\n{'='*52}")
    print(f"  build_highlight.py — #{args.player}")
    print(f"  영상: {os.path.basename(args.input)}")
    print(f"  데이터: {args.data}")
    print(f"  갭 기준: {args.gap}초  버퍼: {args.buf}초")
    print(f"{'='*52}\n")

    # 시프트 추출
    print("[1] 시프트 추출")
    shifts = extract_shifts(jmap, tracks, args.player,
                             args.gap, args.buf, args.fps)
    if not shifts:
        print(f"  #{args.player}번 감지 데이터 없음 — jersey_map에 등록된 선수인지 확인하세요")
        sys.exit(1)

    total_ice = sum(s["duration"] for s in shifts)
    avg_shift = total_ice / len(shifts)
    print(f"  시프트: {len(shifts)}개  총 아이스타임: {total_ice/60:.1f}분 ({total_ice:.0f}초)  평균: {avg_shift:.0f}초\n")

    # 메타데이터 저장
    save_shifts_json(shifts, args.player, out_dir)

    # 하이라이트 영상 생성
    print("\n[2] 하이라이트 영상 생성")
    ok = build_video(args.input, shifts, args.player, args.output)

    # 최종 요약
    elapsed = time.time() - t0
    print(f"\n{'='*52}")
    print(f"  완료! ({elapsed/60:.1f}분)")
    print(f"  시프트 수:     {len(shifts)}개")
    print(f"  총 아이스타임: {total_ice/60:.1f}분 ({total_ice:.0f}초)")
    print(f"  평균 시프트:   {avg_shift:.0f}초")
    if ok:
        print(f"  출력 영상:     {args.output}")
        print(f"  메타데이터:    {out_dir}/{args.player}_shifts.json")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
