# IceIQ 하이라이트 공유 & 워터마크 시스템

## 🎯 개요

선수 하이라이트 영상에 IceIQ 로고를 워터마크로 삽입하고, 공유 가능한 URL을 생성하여 카톡, 인스타, 문자 등으로 공유할 수 있는 시스템입니다.

---

## 📋 API 엔드포인트

### 1. 하이라이트 생성 (공유 URL 포함)
```bash
POST /highlight
Content-Type: application/x-www-form-urlencoded

video_path=/path/to/video.mp4
video_stem=game1
player=4
gap=30
buf=5
```

**응답:**
```json
{
  "player": "4",
  "shifts": 12,
  "total_ice_time_min": 15.45,
  "file_path": "data/highlights/highlights_game1_4_1744873200.mp4",
  "stream_url": "/video/highlights/highlights_game1_4_1744873200.mp4",
  "share_id": "abc123",
  "share_url": "https://iceiq.app/s/abc123",
  "shift_detail": [
    {
      "shift_number": 1,
      "start_time": 123.4,
      "end_time": 145.2,
      "duration": 21.8
    }
  ]
}
```

### 2. 공유된 하이라이트 보기 (공개 페이지)
```bash
GET https://iceiq.app/s/{share_id}
```

**반환:** HTML 페이지 (로그인 불필요)
- 영상 재생
- 선수 정보
- "IceIQ로 분석하기" CTA 버튼
- "공유하기" 버튼 (OS 공유 시트)

### 3. 공유 메타데이터 조회 (API)
```bash
GET /api/highlights/{share_id}
```

**응답:**
```json
{
  "share_id": "abc123",
  "video_stem": "game1",
  "player": "4",
  "created_at": "2026-04-15T21:53:00",
  "file_path": "data/highlights/highlights_game1_4_1744873200.mp4",
  "stream_url": "/video/highlights/highlights_game1_4_1744873200.mp4"
}
```

---

## 🎨 워터마크 구현

### FFmpeg 필터 (현재 구현)
```bash
drawtext=\
  text='IceIQ':\
  fontfile=/System/Library/Fonts/Helvetica.ttc:\
  fontsize=24:\
  fontcolor=white:\
  x=main_w-140:\
  y=main_h-45:\
  box=1:\
  boxcolor=black@0.5:\
  boxborderw=2
```

**위치:** 우하단 (bottom-right)
**스타일:** 흰색 텍스트, 검은색 반투명 배경

### 향후 개선 (로고 이미지)
```bash
# IceIQ 로고 이미지 오버레이
overlay=iceiq_logo.png:x=W-150:y=H-100:enable='between(t,0,duration)'
```

---

## 📱 앱 구현 (React Native / Flutter)

### 공유 버튼 구현
```javascript
// iOS/Android OS 공유 시트
import Share from 'react-native-share';

const shareHighlight = (shareUrl, playerName) => {
  Share.open({
    message: `${playerName}의 하이라이트 영상을 확인하세요!`,
    url: shareUrl,  // https://iceiq.app/s/abc123
    title: 'IceIQ 하이라이트 공유',
    failOnCancel: false,
  });
};
```

### 공유 페이지 (웹)
```javascript
// 클라이언트 스크립트
function openIceIQ() {
  window.location.href = 'https://iceiq.app/dashboard?' +
    `video=game1&player=4`;
}

function shareVideo() {
  const shareData = {
    title: 'IceIQ 하이라이트',
    text: '선수 #4의 하이라이트를 확인하세요!',
    url: window.location.href,
  };
  
  // Web Share API (모바일 브라우저)
  if (navigator.share) {
    navigator.share(shareData);
  } else {
    // Fallback: URL 복사
    navigator.clipboard.writeText(window.location.href);
    alert('링크가 복사되었습니다!');
  }
}
```

---

## 🔄 플로우 다이어그램

```
┌─────────────────────────────────────────────┐
│ 1. 하이라이트 생성 (POST /highlight)       │
│    video_stem=game1, player=4               │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 2. FFmpeg 실행                              │
│    - 시프트 구간 추출                       │
│    - 사운드 포함 인코딩                     │
│    - 워터마크 삽입 (우하단 "IceIQ")        │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 3. 공유 ID 생성 & 메타데이터 저장         │
│    share_id = "abc123"                      │
│    share_url = "https://iceiq.app/s/abc123"│
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 4. 앱에서 "공유" 버튼 클릭                │
│    OS 공유 시트 열기 (카톡, 인스타, 문자) │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 5. 사용자 공유 (카톡/인스타/문자)         │
│    "선수 #4 하이라이트 보기"               │
│    + share_url 링크                         │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 6. 공유 링크 접속 (GET /s/abc123)          │
│    로그인 불필요                           │
│    - 영상 자동 재생                        │
│    - "IceIQ로 분석하기" CTA                │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 7. "IceIQ로 분석하기" 클릭                 │
│    dashboard?video=game1&player=4로 이동   │
│    → 전체 게임 분석 화면으로 진입          │
└─────────────────────────────────────────────┘
```

---

## 🛠️ 공유 페이지 UI/UX

### 레이아웃
```
┌────────────────────────────────┐
│      IceIQ 하이라이트          │
├────────────────────────────────┤
│                                │
│                                │
│     [영상 플레이어]            │
│     (16:9 비율)                │
│                                │
│                                │
├────────────────────────────────┤
│ 선수 #4 하이라이트            │
│ 경기: game1                    │
│ 공유일: 2026-04-15            │
├────────────────────────────────┤
│ [IceIQ로 분석하기 →]  [공유]  │
└────────────────────────────────┘
```

### CTA 버튼
- **"IceIQ로 분석하기 →"** (파란색)
  - 클릭 시: `dashboard?video=game1&player=4`로 이동
  - 목적: 전체 분석으로 유도 (유저 획득)

- **"공유하기"** (회색)
  - 클릭 시: OS 공유 시트 표시
  - 플랫폼: 카톡, 인스타, 문자, 이메일 등

---

## 📊 공유 추적 (분석용 - 향후)

```json
{
  "share_id": "abc123",
  "created_at": "2026-04-15T21:53:00",
  "clicks": 5,
  "conversions": 1,
  "shared_via": ["kakao", "instagram"],
  "user_agent": [
    "iOS Safari",
    "Android Chrome"
  ]
}
```

---

## 🔐 보안

### 공개 공유 (현재)
- share_id 예측 어려움 (6자리 무작위)
- 인증 필요 없음
- CDN으로 영상 제공 권장

### 향후 개선
- `shared_by`: 공유자 정보 (선택)
- `expires_at`: 공유 만료 시간 (기본: 영구)
- `password`: 비밀번호 보호 (선택)

---

## 💡 사용 예시

### 1. 하이라이트 생성
```bash
curl -X POST http://localhost:8000/highlight \
  -d "video_path=/path/to/game1.mp4&video_stem=game1&player=4&gap=30&buf=5"
```

응답:
```json
{
  "share_id": "abc123",
  "share_url": "https://iceiq.app/s/abc123",
  "stream_url": "/video/highlights/highlights_game1_4_1744873200.mp4"
}
```

### 2. 앱에서 공유
```javascript
const response = await fetch('/highlight', {
  method: 'POST',
  body: formData,
});

const data = await response.json();

// 공유 버튼 표시
showShareButton(data.share_url, 'Player #4 Highlight');
```

### 3. 공유 링크 접속
```
https://iceiq.app/s/abc123
→ HTML 페이지 (영상 + CTA)
```

---

## 🚀 배포 체크리스트

- [ ] FFmpeg 설치 (서버)
- [ ] 워터마크 폰트 경로 확인
- [ ] HIGHLIGHTS_DIR 권한 설정
- [ ] 공유 URL 도메인 설정 (`iceiq.app`)
- [ ] CORS 설정 (공개 공유)
- [ ] CDN 연동 (영상 배포)
- [ ] 공유 분석 로깅 (선택)

---

## 📝 워터마크 커스터마이징

### 로고 추가 (향후)
```python
def add_watermark_with_logo(input_video, output_video, logo_path):
    """로고 이미지를 오버레이로 추가"""
    cmd = [
        "ffmpeg", "-i", input_video,
        "-i", logo_path,
        "-filter_complex", 
        "overlay=x=main_w-150:y=main_h-100",
        "-c:a", "aac",
        output_video
    ]
```

### 텍스트 커스터마이징
- `fontsize`: 폰트 크기
- `fontcolor`: 폰트 색상
- `boxcolor`: 배경색
- `x`, `y`: 위치

---

## 🎬 FFmpeg 명령어 참고

### 현재 구현
```bash
ffmpeg -i input.mp4 \
  -vf "drawtext=text='IceIQ':fontsize=24:..." \
  -c:a aac output.mp4
```

### 로고 + 텍스트
```bash
ffmpeg -i input.mp4 -i logo.png \
  -filter_complex "[0]drawtext=...[v1];[v1][1]overlay=..." \
  -c:a aac output.mp4
```

### 배치 처리 (큰 파일)
```bash
ffmpeg -i input.mp4 \
  -vf "... watermark ..." \
  -c:v libx264 -crf 23 \
  -c:a aac -b:a 128k \
  output.mp4
```
