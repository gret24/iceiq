#!/usr/bin/env python3
"""
build_highlight.py — 분석 결과에서 선수 하이라이트 영상 생성
분석(pipeline_fast.py)은 1번만 돌리고, 여기서는 후처리만 수행

사용법:
  python3 build_highlight.py \
    --input /root/iceiq/aigis_g18.mp4 \
    --jersey-map /root/iceiq/jersey_map.json \
    --tracks /root/iceiq/tracks.json \
    --player 47 \
    --output /workspace/iceiq/output/47_fulltime.mp4
"""
import argparse, json, os, re, shutil, subprocess, sys, time


def extract_shifts(jersey_map: dict, tracks_data: dict,
                   player: str, gap_sec: float = 10.0,
                   buf_sec: float = 5.0, fps: int = 4) -> list[dict]:
    """tracks + jersey_map에서 특정 선수 시프트 추출"""
    norm = player.lstrip("0") or "0"
    target_tids = {tid for tid, info in jersey_map.items()
                   if info["jersey"].lstrip("0") == norm}
    print(f"  #{player} 해당 track: {len(target_tids)}개")

    frame_indices = []
    for fname, fts in sorted(tracks_data.items()):
        m = re.search(r"(\d+)", fname)
        if not m: continue
        fidx = int(m.group(1))
        for t in fts:
            if str(t["track_id"]) in target_tids:
                frame_indices.append(fidx)
                break

    if not frame_indices:
        return []

    frame_indices = sorted(set(frame_indices))
    gap_frames = int(gap_sec * fps)

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
        start = round(max(0.0, gs / fps - buf_sec), 2)
        end   = round(ge / fps + buf_sec, 2)
        shifts.append({"start_sec": start, "end_sec": end,
                        "duration": round(end - start, 2)})
    return shifts


def build_highlight(video: str, shifts: list[dict],
                    player: str, output: str) -> bool:
    out_dir = os.path.dirname(output)
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, f"_tmp_{player}")
    os.makedirs(tmp_dir, exist_ok=True)

    clip_files = []
    for i, sh in enumerate(shifts, 1):
        mm, ss = divmod(int(sh["start_sec"]), 60)
        label = f"Shift {i} - {mm}:{ss:02d}"
        vf = (f"drawtext=text='{label}':fontsize=36:fontcolor=white:"
              f"x=20:y=20:box=1:boxcolor=black@0.5:boxborderw=5:"
              f"enable='between(t,0,2)'")
        clip = os.path.join(tmp_dir, f"clip_{i:03d}.mp4")
        r = subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(sh["start_sec"]), "-to", str(sh["end_sec"]),
            "-i", video,
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", clip
        ], capture_output=True)
        if r.returncode == 0 and os.path.exists(clip):
            size = os.path.getsize(clip) / 1024 / 1024
            print(f"  shift {i:2d}: {mm:02d}:{ss:02d} ~ "
                  f"{int(sh['end_sec'])//60:02d}:{int(sh['end_sec'])%60:02d}"
                  f"  ({sh['duration']:.0f}초, {size:.1f}MB)")
            clip_files.append(clip)
        else:
            print(f"  shift {i}: 실패")

    if not clip_files:
        return False

    concat_txt = os.path.join(tmp_dir, "concat.txt")
    with open(concat_txt, "w") as f:
        for c in clip_files:
            f.write(f"file '{c}'\n")

    r2 = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_txt, "-c", "copy", output
    ], capture_output=True)

    shutil.rmtree(tmp_dir, ignore_errors=True)

    if r2.returncode == 0:
        size = os.path.getsize(output) / 1024 / 1024
        print(f"\n  ✅ {os.path.basename(output)} ({size:.1f}MB)")
        return True
    return False


def save_metadata(shifts: list[dict], player: str, output: str):
    total = sum(s["duration"] for s in shifts)
    avg   = total / len(shifts) if shifts else 0
    meta  = {
        "player": player,
        "total_shifts": len(shifts),
        "total_ice_time": round(total, 2),
        "average_shift":  round(avg, 2),
        "shifts": shifts,
    }
    json_path = output.replace(".mp4", ".json").replace("_fulltime", "_shifts")
    # 항상 shifts.json으로 저장
    base = os.path.join(os.path.dirname(output), f"{player}_shifts.json")
    with open(base, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  메타데이터: {base}")
    return meta


def main():
    parser = argparse.ArgumentParser(description="선수 하이라이트 생성 (후처리 전용)")
    parser.add_argument("--input",      required=True, help="원본 영상 경로")
    parser.add_argument("--jersey-map", default="/root/iceiq/jersey_map.json")
    parser.add_argument("--tracks",     default="/root/iceiq/tracks.json")
    parser.add_argument("--player",     required=True, help="선수 등번호")
    parser.add_argument("--output",     required=True, help="출력 mp4 경로")
    parser.add_argument("--gap",        type=float, default=10.0, help="시프트 갭 기준(초)")
    parser.add_argument("--buf",        type=float, default=5.0,  help="시프트 앞뒤 버퍼(초)")
    parser.add_argument("--fps",        type=int,   default=4,    help="분석 FPS")
    args = parser.parse_args()

    t0 = time.time()
    print(f"\n{'='*50}")
    print(f"  build_highlight.py — #{args.player}")
    print(f"  영상: {os.path.basename(args.input)}")
    print(f"{'='*50}\n")

    with open(args.jersey_map) as f:
        jmap = json.load(f)
    with open(args.tracks) as f:
        tracks = json.load(f)

    print("[1] 시프트 추출")
    shifts = extract_shifts(jmap, tracks, args.player,
                             args.gap, args.buf, args.fps)
    if not shifts:
        print(f"  #{args.player}번 감지 데이터 없음"); return

    total = sum(s["duration"] for s in shifts)
    avg   = total / len(shifts)
    print(f"  시프트: {len(shifts)}개  총 아이스타임: {total/60:.1f}분  평균: {avg:.0f}초\n")

    meta = save_metadata(shifts, args.player, args.output)

    print("[2] 하이라이트 영상 생성")
    ok = build_highlight(args.input, shifts, args.player, args.output)

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"  완료! ({elapsed/60:.1f}분)")
    print(f"  시프트 수:    {meta['total_shifts']}개")
    print(f"  총 아이스타임: {meta['total_ice_time']/60:.1f}분 ({meta['total_ice_time']:.0f}초)")
    print(f"  평균 시프트:  {meta['average_shift']:.0f}초")
    if ok:
        size = os.path.getsize(args.output) / 1024 / 1024
        # 영상 길이 확인
        r = subprocess.run(["ffprobe","-v","quiet","-show_entries",
                            "format=duration","-of","csv=p=0", args.output],
                           capture_output=True, text=True)
        dur = float(r.stdout.strip()) if r.stdout.strip() else 0
        print(f"  영상 길이:    {dur/60:.1f}분 ({dur:.0f}초)")
        print(f"  파일 크기:    {size:.1f}MB")
        print(f"  출력:         {args.output}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
