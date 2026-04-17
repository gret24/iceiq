#!/usr/bin/env bash
# IceIQ Server — RunPod 배포 스크립트
# 사용: ./deploy.sh
set -e

echo "=== IceIQ Deploy ==="
cd "$(dirname "$0")"

echo "[1/4] git pull..."
git pull origin main
git log -1 --oneline

echo "[2/4] uvicorn 종료..."
pkill -9 -f uvicorn 2>/dev/null || true
sleep 2

echo "[3/4] uvicorn 시작..."
nohup python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 \
  > /tmp/server.log 2>&1 &

echo "[4/4] 헬스체크..."
sleep 3
curl -s http://localhost:8000/api/health && echo ""

echo "=== 배포 완료 ==="
echo "로그: tail -f /tmp/server.log"
