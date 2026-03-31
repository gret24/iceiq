#!/usr/bin/env python3
"""
IceIQ 자동화 파이프라인
사용법:
  python3 auto_highlight.py <YouTube URL> <등번호> [팀(HOME/AWAY)]

예시:
  python3 auto_highlight.py https://youtu.be/xxxxx 47 HOME
  python3 auto_highlight.py https://youtu.be/xxxxx 3

흐름:
  1. YouTube 영상 다운로드 (yt-dlp)
  2. RunPod GPU 서버로 전송 (scp)
  3. pipeline.py 실행 (GPU 분석)
  4. highlight.mp4 다운로드
  5. iCloud Drive 저장
  6. 완료 알림
"""

import argparse, os, subprocess, sys, time

# ── 설정 ──────────────────────────────────────────────────
RUNPOD_HOST    = "216.81.151.44"
RUNPOD_PORT    = 19415
RUNPOD_USER    = "root"
RUNPOD_DIR     = "/root/iceiq"
SSH_KEY        = os.path.expanduser("~/.ssh/id_ed25519")
LOCAL_ICEIQ    = os.path.expanduser("~/iceiq")
ICLOUD_DIR     = os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs")
TMP_VIDEO      = "/tmp/iceiq_input.mp4"

SSH_BASE = ["ssh", "-o", "StrictHostKeyChecking=no", "-tt",
            "-i", SSH_KEY, "-p", str(RUNPOD_PORT),
            f"{RUNPOD_USER}@{RUNPOD_HOST}"]

def run(cmd, **kwargs):
    print(f"  $ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    return subprocess.run(cmd, **kwargs)

def ssh(command, capture=False):
    full = SSH_BASE + [command]
    if capture:
        return subprocess.run(full, capture_output=True, text=True, timeout=30)
    return subprocess.run(full, timeout=300)

def step(n, total, label):
    print(f"\n{'─'*50}")
    print(f"  [{n}/{total}] {label}")
    print(f"{'─'*50}")

def main():
    parser = argparse.ArgumentParser(description="IceIQ 자동화 파이프라인")
    parser.add_argument("url",    help="YouTube URL")
    parser.add_argument("number", help="등번호 (예: 47)")
    parser.add_argument("team",   nargs="?", default=None, help="HOME / AWAY (선택)")
    parser.add_argument("--buffer", type=float, default=3.0, help="클립 앞뒤 버퍼 초")
    args = parser.parse_args()

    t_start = time.time()
    team_str = args.team or "전체"
    output_name = f"highlight_{args.number}_{team_str}.mp4"
    local_output = os.path.join(LOCAL_ICEIQ, output_name)
    icloud_output = os.path.join(ICLOUD_DIR, output_name)

    print(f"\n{'='*50}")
    print(f"  IceIQ 자동화 파이프라인")
    print(f"  URL: {args.url}")
    print(f"  선수: {args.number}번 ({team_str})")
    print(f"{'='*50}")

    # ── 1. YouTube 다운로드 ──────────────────────────────
    step(1, 5, "YouTube 영상 다운로드")
    r = run(["yt-dlp", "-f", "best[height<=720]/best",
             "--no-part", "-o", TMP_VIDEO, args.url])
    if r.returncode != 0:
        print("❌ 다운로드 실패!")
        sys.exit(1)

    size = os.path.getsize(TMP_VIDEO) / 1024 / 1024
    print(f"  ✅ 다운로드 완료 ({size:.0f}MB)")

    # ── 2. RunPod 전송 ───────────────────────────────────
    step(2, 5, f"RunPod 서버로 전송 ({size:.0f}MB)")
    remote_video = f"{RUNPOD_DIR}/input_video.mp4"
    r = run(["scp", "-o", "StrictHostKeyChecking=no",
             "-i", SSH_KEY, "-P", str(RUNPOD_PORT),
             TMP_VIDEO, f"{RUNPOD_USER}@{RUNPOD_HOST}:{remote_video}"])
    if r.returncode != 0:
        print("❌ 전송 실패!")
        sys.exit(1)
    print(f"  ✅ 전송 완료")

    # ── 3. RunPod에서 pipeline.py 실행 ───────────────────
    step(3, 5, "GPU 분석 시작 (RunPod)")
    team_arg = args.team if args.team else ""
    pipeline_cmd = (
        f"cd {RUNPOD_DIR} && "
        f"rm -rf frames detected clips tracks.json jersey_map.json 2>/dev/null; "
        f"tmux kill-session -t iceiq 2>/dev/null; "
        f"tmux new-session -d -s iceiq "
        f"'python3 -u pipeline.py input_video.mp4 {args.number} {team_arg} "
        f"--buffer {args.buffer} > /workspace/auto_result.log 2>&1'"
    )
    ssh(pipeline_cmd)
    print(f"  ✅ 분석 시작됨")

    # ── 4. 완료 대기 ─────────────────────────────────────
    step(4, 5, "분석 완료 대기...")
    print("  (30초마다 진행상황 확인)")
    while True:
        time.sleep(30)
        # PTY 없이 직접 SSH 명령 실행
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-i", SSH_KEY,
             "-p", str(RUNPOD_PORT), f"{RUNPOD_USER}@{RUNPOD_HOST}",
             "tail -3 /workspace/auto_result.log"],
            capture_output=True, text=True, timeout=20
        )
        log = result.stdout.strip()
        last_line = log.split('\n')[-1][:80] if log else "확인 중..."
        print(f"  → {last_line}")
        if "highlight.mp4 생성됨" in log or "완료!" in log:
            break
        if "오류" in log or "Error" in log:
            print(f"  ❌ 오류 발생!\n{log}")
            sys.exit(1)

    # 결과 로그 출력
    r = ssh("tail -8 /workspace/auto_result.log", capture=True)
    print(f"\n{r.stdout}")

    # ── 5. highlight.mp4 다운로드 ────────────────────────
    step(5, 5, "highlight.mp4 다운로드")
    r = run(["scp", "-o", "StrictHostKeyChecking=no",
             "-i", SSH_KEY, "-P", str(RUNPOD_PORT),
             f"{RUNPOD_USER}@{RUNPOD_HOST}:{RUNPOD_DIR}/highlight.mp4",
             local_output])
    if r.returncode != 0:
        print("❌ 다운로드 실패!")
        sys.exit(1)

    size_out = os.path.getsize(local_output) / 1024 / 1024
    print(f"  ✅ 저장: {local_output} ({size_out:.1f}MB)")

    # iCloud Drive 복사
    try:
        import shutil
        shutil.copy2(local_output, icloud_output)
        print(f"  ✅ iCloud Drive 저장: {output_name}")
    except Exception as e:
        print(f"  ⚠️ iCloud 저장 실패: {e}")

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"  ✅ 완료! 총 소요시간: {elapsed/60:.1f}분")
    print(f"  파일: {output_name}")
    print(f"  로컬: http://192.168.68.50:8899/{output_name}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
