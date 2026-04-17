#!/usr/bin/env bash
# IceIQ Server — RunPod 배포 스크립트
# 사용: ./deploy.sh
set -e

echo "=== IceIQ Deploy ==="
cd "$(dirname "$0")"

# ─── [1/5] git pull ───────────────────────────────────────────────────────────
echo ""
echo "[1/5] git pull..."
git pull origin main
git log -1 --oneline

# ─── [2/5] ML 의존성이 있는 Python 자동 감지 ─────────────────────────────────
echo ""
echo "[2/5] Python 환경 탐색..."

ICEIQ_PYTHON=""
CANDIDATES=(
    "/opt/conda/envs/iceiq/bin/python3"
    "/root/miniconda3/envs/iceiq/bin/python3"
    "/opt/conda/bin/python3"
    "/root/miniconda3/bin/python3"
    "/usr/bin/python3"
    "$(which python3 2>/dev/null || true)"
)

for pybin in "${CANDIDATES[@]}"; do
    [ -z "$pybin" ] && continue
    [ -x "$pybin" ] || continue
    if "$pybin" -c "import ultralytics, cv2, easyocr" 2>/dev/null; then
        ICEIQ_PYTHON="$pybin"
        echo "  ✅ ML 패키지 확인: $ICEIQ_PYTHON"
        break
    else
        echo "  ⬜ ML 없음: $pybin"
    fi
done

if [ -z "$ICEIQ_PYTHON" ]; then
    echo "  ⚠️  ML 패키지 있는 Python을 못 찾음 — sys.executable 사용"
    ICEIQ_PYTHON="$(which python3)"
fi

export ICEIQ_PYTHON
echo "  → ICEIQ_PYTHON=$ICEIQ_PYTHON"

# ─── [3/5] sanity check ───────────────────────────────────────────────────────
echo ""
echo "[3/5] 의존성 sanity check..."
"$ICEIQ_PYTHON" -c "
import sys
print(f'  Python: {sys.executable}')
mods = ['ultralytics', 'cv2', 'easyocr', 'numpy', 'scipy', 'boto3']
ok, fail = [], []
for m in mods:
    try:
        __import__(m)
        ok.append(m)
    except ImportError:
        fail.append(m)
print(f'  OK   : {ok}')
if fail:
    print(f'  FAIL : {fail}')
    sys.exit(1)
"

# ─── [4/5] uvicorn 재시작 ─────────────────────────────────────────────────────
echo ""
echo "[4/5] uvicorn 재시작..."
pkill -9 -f uvicorn 2>/dev/null || true
sleep 2

nohup "$ICEIQ_PYTHON" -m uvicorn server:app \
    --host 0.0.0.0 --port 8000 \
    > /tmp/server.log 2>&1 &

UVICORN_PID=$!
echo "  PID: $UVICORN_PID"

# ─── [5/5] 헬스체크 ───────────────────────────────────────────────────────────
echo ""
echo "[5/5] 헬스체크..."
sleep 4
HEALTH=$(curl -s http://localhost:8000/api/health 2>/dev/null || echo "FAIL")
echo "  $HEALTH"

if echo "$HEALTH" | grep -q '"ok"'; then
    echo ""
    echo "=== ✅ 배포 완료 ==="
    echo "  Python : $ICEIQ_PYTHON"
    echo "  PID    : $UVICORN_PID"
    echo "  로그   : tail -f /tmp/server.log"
else
    echo ""
    echo "=== ❌ 서버 시작 실패 ==="
    echo "--- 최근 로그 ---"
    tail -20 /tmp/server.log
    exit 1
fi
