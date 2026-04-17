# Day 9 TODO (2026-04-18)

## 블로커 (먼저 해결)
- [ ] RunPod Pod SSH 복구
  - 새 IP/port 확인 → ~/.ssh/config 업데이트
  - openssh-server 재설치 + authorized_keys 재등록
  - sshd 수동 시작
- [ ] 근본 해결: Docker 이미지 리빌드
  - openssh-server, git, filterpy, anthropic, numpy<2 베이크
  - PUBLIC_KEY env var 자동 등록
  - startup script로 sshd 자동 시작

## 배포 대기 (SSH 복구 후 즉시)
- [ ] server-fixes의 히트맵 수정 RunPod 반영
  - server.py (output_dir 전달)
  - heatmap_homo.py (절대경로 인자 수용)

## 기능 연결 (Day 8의 5개 메뉴 중 나머지)
- [ ] Heatmap 표시 (배포 후 테스트)
- [ ] heatmap_homo.py 로스터 하드코딩 제거 (line 85)
- [ ] Ice Time Shifts 연동
- [ ] 하이라이트 생성 연동
- [ ] 스카우팅 연동

## UX 개선
- [ ] 분석 progress 실시간 표시 (현재 20%에서 멈춤)
- [ ] "Software caused connection abort" 재현 조사 (큰 영상 업로드)
- [ ] 분석 시간 단축 (현재 23분/영상)

## 인프라
- [ ] OpenClaw 봇 /runpod LLM 해석 제거, 직접 SSH 실행
- [ ] iceiq vs iceiq-dev repo 관계 정리
- [ ] server-fixes 브랜치 → main 머지 전략

## 누적 해결한 버그 목록 (Day 8)
1. R2 presigned "Network request failed" → FileSystem.uploadAsync
2. NoneType .read() → communicate() + stderr=PIPE
3. /opt/homebrew/bin/conda macOS 경로 → sys.executable
4. RunPod git 부재 → apt install git
5. Private repo 인증 → Public 전환
6. Branch 꼬임 (master vs main) → server-fixes
7. OpenSSH Server 미설치 → apt install + sshd
8. Docker-init 로그 경로 → SSH 모니터링 우회
9. anthropic 모듈 누락 → pip install
10. NumPy 2.x ↔ PyTorch 호환 → numpy<2
11. filterpy 누락 → pip install
12. pipeline not a package → __init__.py
13-19. device='mps' 하드코딩 7개 파일 → auto-detect
20. analyze_game.py 경로 불일치 → --output 인자
21. heatmap 호출 경로 누락 → server.py output_dir 전달
    (Day 9에 배포 검증)
