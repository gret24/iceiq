#!/usr/bin/env python3
"""
IceIQ 멀티 플레이어 하이라이트
여러 선수를 동시에 추적해서 각각 하이라이트 생성

사용법:
  python3 multi_player.py <YouTube URL> <번호1> <번호2> ... [--team HOME/AWAY]

예시:
  python3 multi_player.py https://youtu.be/xxx 23 47 11
  python3 multi_player.py https://youtu.be/xxx 23 47 11 --team HOME

흐름:
  1. YouTube 영상 다운로드
  2. RunPod으로 전송
  3. 추적 + OCR 1회 실행 (공유)
  4. 각 선수별 클립 추출 & 하이라이트 생성
  5. 전체 다운로드 → iCloud 저장
"""

import argparse, os, subprocess, sys, time

RUNPOD_HOST    = "216.81.151.44"
RUNPOD_PORT    = 19415
RUNPOD_USER    = "root"
RUNPOD_DIR     = "/root/iceiq"
SSH_KEY        = os.path.expanduser("~/.ssh/id_ed25519")
LOCAL_ICEIQ    = os.path.expanduser("~/iceiq")
ICLOUD_DIR     = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs")
TMP_VIDEO      = "/tmp/iceiq_input.mp4"

def ssh_cmd(command, capture=False, timeout=30):
    """SSH 명령어 실행 (PTY 없이)"""
    full = ["ssh", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY,
            "-p", str(RUNPOD_PORT), f"{RUNPOD_USER}@{RUNPOD_HOST}", command]
    if capture:
        return subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    return subprocess.run(full, timeout=timeout)

def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd)

def step(n, total, label):
    print(f"\n{'─'*50}")
    print(f"  [{n}/{total}] {label}")
    print(f"{'─'*50}")

def wait_for_completion(log_file, timeout_min=30):
    """RunPod 작업 완료 대기"""
    start = time.time()
    while True:
        time.sleep(30)
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY,
             "-p", str(RUNPOD_PORT), f"{RUNPOD_USER}@{RUNPOD_HOST}",
             f"tail -3 {log_file}"],
            capture_output=True, text=True, timeout=20
        )
        log = r.stdout.strip()
        last_line = log.split('\n')[-1][:80] if log else "확인 중..."
        print(f"  → {last_line}")
        if "완료!" in log or "highlight.mp4 생성됨" in log:
            return True
        if "오류" in log or "Error" in log:
            print(f"  ❌ 오류!\n{log}")
            return False
        if (time.time() - start) > timeout_min * 60:
            print(f"  ⏰ 타임아웃 ({timeout_min}분)")
            return False

def main():
    parser = argparse.ArgumentParser(description="IceIQ 멀티 플레이어 하이라이트")
    parser.add_argument("url",    help="YouTube URL")
    parser.add_argument("numbers", nargs="+", help="등번호 목록 (예: 23 47 11)")
    parser.add_argument("--team", default=None, help="HOME / AWAY")
    parser.add_argument("--buffer", type=float, default=3.0)
    args = parser.parse_args()

    numbers = args.numbers
    team_str = args.team or "전체"
    t_start = time.time()

    print(f"\n{'='*55}")
    print(f"  IceIQ 멀티 플레이어 하이라이트")
    print(f"  URL: {args.url}")
    print(f"  추적 선수: {', '.join(numbers)}번  팀: {team_str}")
    print(f"{'='*55}")

    # ── 1. YouTube 다운로드 ──────────────────────────────
    step(1, 5, "YouTube 영상 다운로드")
    r = run(["yt-dlp", "-f", "best[height<=720]/best", "--no-part", "-o", TMP_VIDEO, args.url])
    if r.returncode != 0:
        print("❌ 다운로드 실패!")
        sys.exit(1)
    size = os.path.getsize(TMP_VIDEO) / 1024 / 1024
    print(f"  ✅ 완료 ({size:.0f}MB)")

    # ── 2. RunPod 전송 ───────────────────────────────────
    step(2, 5, f"RunPod 전송 ({size:.0f}MB)")
    r = run(["scp", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY, "-P", str(RUNPOD_PORT),
             TMP_VIDEO, f"{RUNPOD_USER}@{RUNPOD_HOST}:{RUNPOD_DIR}/input_video.mp4"])
    if r.returncode != 0:
        print("❌ 전송 실패!")
        sys.exit(1)
    print(f"  ✅ 전송 완료")

    # ── 3. 프레임 추출 + ByteTrack + OCR (1회만 실행) ────
    step(3, 5, "GPU 분석 (1회 공통 처리)")
    first_num = numbers[0]
    team_arg = args.team if args.team else ""

    # 첫 번째 선수로 전체 파이프라인 실행 (프레임+추적+OCR)
    cmd = (
        f"cd {RUNPOD_DIR} && "
        f"rm -rf frames detected clips tracks.json jersey_map.json 2>/dev/null; "
        f"tmux kill-session -t multi 2>/dev/null; "
        f"tmux new-session -d -s multi "
        f"'python3 -u pipeline.py input_video.mp4 {first_num} {team_arg} "
        f"--buffer {args.buffer} > /workspace/multi_result.log 2>&1'"
    )
    subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY,
         "-p", str(RUNPOD_PORT), f"{RUNPOD_USER}@{RUNPOD_HOST}", cmd],
        timeout=30
    )
    print(f"  ✅ 분석 시작 (30초마다 진행 확인)")
    wait_for_completion("/workspace/multi_result.log", timeout_min=60)

    # ── 4. 나머지 선수들 클립 추출 (OCR 재사용) ──────────
    step(4, 5, f"각 선수별 클립 추출 ({len(numbers)}명)")
    
    result_files = []
    for num in numbers:
        print(f"\n  📋 {num}번 선수 클립 추출 중...")
        highlight_remote = f"{RUNPOD_DIR}/highlight_{num}.mp4"
        extract_cmd = (
            f"cd {RUNPOD_DIR} && "
            f"python3 -c \""
            f"import sys; sys.path.insert(0, '.'); "
            f"from pipeline import step_clips, step_highlight; "
            f"clips = step_clips('input_video.mp4', '{num}', {repr(args.team)}, fps=30, buf={args.buffer}); "
            f"import shutil, os; "
            f"shutil.move('highlight.mp4', 'highlight_{num}.mp4') if os.path.exists('highlight.mp4') else None; "
            f"print('DONE_{num}')\" >> /workspace/multi_result.log 2>&1"
        )
        r = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY,
             "-p", str(RUNPOD_PORT), f"{RUNPOD_USER}@{RUNPOD_HOST}", extract_cmd],
            capture_output=True, text=True, timeout=120
        )
        
        # 파일 존재 확인
        check = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY,
             "-p", str(RUNPOD_PORT), f"{RUNPOD_USER}@{RUNPOD_HOST}",
             f"ls -lh {highlight_remote} 2>/dev/null && echo EXISTS"],
            capture_output=True, text=True, timeout=15
        )
        if "EXISTS" in check.stdout:
            size_mb = [l for l in check.stdout.split('\n') if 'highlight' in l]
            print(f"  ✅ {num}번 완료")
            result_files.append((num, highlight_remote))
        else:
            print(f"  ⚠️ {num}번 클립 없음 (영상에 미등장)")

    # 첫 번째 선수 하이라이트도 복사
    r_check = subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY,
         "-p", str(RUNPOD_PORT), f"{RUNPOD_USER}@{RUNPOD_HOST}",
         f"test -f {RUNPOD_DIR}/highlight_{first_num}.mp4 && echo EXISTS || echo NO"],
        capture_output=True, text=True, timeout=15
    )
    if "NO" in r_check.stdout:
        # 이미 pipeline.py가 만든 highlight.mp4를 이름 변경
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY,
             "-p", str(RUNPOD_PORT), f"{RUNPOD_USER}@{RUNPOD_HOST}",
             f"cp {RUNPOD_DIR}/highlight_old.mp4 {RUNPOD_DIR}/highlight_{first_num}.mp4 2>/dev/null || true"],
            timeout=15
        )

    # ── 5. 다운로드 & iCloud 저장 ────────────────────────
    step(5, 5, f"다운로드 & iCloud 저장")
    import shutil

    for num, remote_path in result_files:
        local_path = os.path.join(LOCAL_ICEIQ, f"highlight_{num}_{team_str}.mp4")
        icloud_path = os.path.join(ICLOUD_DIR, f"highlight_{num}_{team_str}.mp4")

        r = subprocess.run(
            ["scp", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY, "-P", str(RUNPOD_PORT),
             f"{RUNPOD_USER}@{RUNPOD_HOST}:{remote_path}", local_path],
            timeout=120
        )
        if r.returncode == 0:
            size_mb = os.path.getsize(local_path) / 1024 / 1024
            try:
                shutil.copy2(local_path, icloud_path)
                print(f"  ✅ {num}번: {size_mb:.1f}MB → iCloud 저장")
            except Exception as e:
                print(f"  ✅ {num}번: {size_mb:.1f}MB (iCloud 실패: {e})")
        else:
            print(f"  ❌ {num}번 다운로드 실패")

    elapsed = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"  ✅ 완료! 총 소요시간: {elapsed/60:.1f}분")
    print(f"  파일: highlight_<번호>_{team_str}.mp4")
    print(f"  로컬: http://192.168.68.50:8899/")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()
