# RunPod 배포 가이드

## 1. 계정 & 이미지 준비

1. [runpod.io](https://runpod.io) 가입 후 크레딧 충전
2. GitHub Actions 빌드 완료 확인  
   → `https://github.com/gret24/iceiq-dev/actions`  
   → 이미지 주소: `ghcr.io/gret24/iceiq-dev:latest`

---

## 2. Template 생성

**My Templates → New Template**

| 항목 | 값 |
|---|---|
| Container Image | `ghcr.io/gret24/iceiq-dev:latest` |
| Container Disk | `20 GB` |
| Volume Disk | `50 GB` (data/ 영구 저장) |
| Volume Mount Path | `/app/data` |
| Expose HTTP Ports | `8000` |

**Environment Variables 추가:**
```
ICEIQ_API_KEY = iceiq-dev-key-2026
```

---

## 3. GPU Pod 생성

**Deploy → GPU Pod → 템플릿 선택**

| GPU | 가격 | 용도 |
|---|---|---|
| RTX 4090 | $0.34/hr | 개발/테스트 |
| A100 80GB | $0.89/hr | 프로덕션 |
| Serverless | 요청 시만 과금 | 저빈도 사용 |

> 쓸 때만 켜고 끄면 RTX 4090으로 충분.

---

## 4. 배포 후 테스트

Pod 실행 후 **Connect → HTTP Service [8000]** 에서 URL 확인.

```bash
# 헬스 체크
curl https://{pod-id}-8000.proxy.runpod.net/

# 분석 API 테스트
curl -X POST https://{pod-id}-8000.proxy.runpod.net/api/analyze \
  -F "file=@sample.mp4" \
  -F "team_name=test" \
  -F "roster_file=aigis.json"
```

---

## 5. 앱 URL 연결

`~/gret24-app/api/config.ts` 의 `API_BASE_URL` 변경:

```ts
// 로컬 개발
API_BASE_URL = 'http://localhost:8000'

// RunPod 배포
API_BASE_URL = 'https://{pod-id}-8000.proxy.runpod.net'
```

---

## 6. 업데이트 배포

```bash
# 코드 수정 후
git push origin main
# → GitHub Actions가 자동 빌드 & GHCR 푸시 (~10분)

# RunPod Pod에서
# My Pods → Pod 메뉴 → Reset Pod (최신 이미지 pull)
```

---

## 비용 참고

| 항목 | 비용 |
|---|---|
| RTX 4090 × 1hr/일 | ~$10/월 |
| Cloudflare R2 10GB | 무료 |
| GitHub Actions | 무료 (public repo) |
| GHCR | 무료 (public repo) |
