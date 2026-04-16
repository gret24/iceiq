# IceIQ 앱 통합 가이드

## 🚀 하이라이트 공유 기능 (완전 구현)

### 시스템 아키텍처

```
┌─────────────────────────────────────────────────┐
│ 모바일 앱 (React Native / Expo)                │
├─────────────────────────────────────────────────┤
│ 1. GameAnalysisScreen                          │
│    - 경기별 하이라이트 목록                    │
│    - 선수별 카드 (썸네일 + 통계)              │
│                                                 │
│ 2. HighlightShareScreen                        │
│    - 영상 재생 (expo-av)                      │
│    - "공유 준비" 버튼                         │
│    - OS 공유 시트 ("공유하기")                │
│                                                 │
│ 3. Share API 통합                              │
│    - POST /share (highlight → 공유 URL)      │
│    - GET /s/{share_id} (공개 웹 페이지)      │
└─────────────────────────────────────────────────┘
          ↓ (HTTP/REST)
┌─────────────────────────────────────────────────┐
│ IceIQ API Server (FastAPI)                     │
├─────────────────────────────────────────────────┤
│ POST /share                                     │
│ - 파일 업로드                                  │
│ - FFmpeg 워터마크 적용                        │
│ - 공유 ID 생성 (8자 UUID)                    │
│ - 메타데이터 저장 (JSON)                      │
│                                                 │
│ GET /s/{share_id}                              │
│ - 공개 웹 페이지 (HTML)                       │
│ - 영상 재생 + CTA 버튼                       │
│                                                 │
│ GET /video/share/{filename}                    │
│ - 영상 스트리밍                               │
└─────────────────────────────────────────────────┘
```

---

## 📱 앱 구현 (React Native)

### 1. 설치 & 의존성

```bash
# Expo 프로젝트 생성
npx create-expo-app iceiq-app
cd iceiq-app

# 필수 패키지
npx expo install expo-av              # 영상 재생
npx expo install expo-file-system     # 파일 접근
npx expo install @react-native-clipboard/clipboard  # 클립보드

# 네비게이션
npm install @react-navigation/native @react-navigation/bottom-tabs @react-navigation/native-stack
npx expo install react-native-screens react-native-safe-area-context
```

### 2. 화면 구조

#### GameAnalysisScreen (경기 분석)
```typescript
📊 경기 분석
├─ 경기 선택 탭 (경기1, 경기2, 경기3)
├─ 하이라이트 카드 목록
│  ├─ 썸네일 (영상 플레이스홀더)
│  ├─ 선수 번호 배지 (#4)
│  ├─ 빠른 공유 버튼 (우상단)
│  ├─ 선수명 & 팀
│  ├─ 통계 (아이스타임, 시프트)
│  └─ "재생 & 공유" 버튼
└─ 비어있으면 안내 메시지
```

#### HighlightShareScreen (재생 & 공유)
```typescript
🎬 하이라이트 공유
├─ 영상 플레이어 (16:9)
├─ 선수 정보
│  ├─ 선수명 & 번호
│  ├─ 팀
│  └─ 메타데이터 (경기, 날짜)
├─ 공유 URL (복사 가능)
├─ "공유 준비" 버튼
│  └─ → FFmpeg 워터마크 적용
│  └─ → 서버 업로드
│  └─ → 공유 URL 생성
└─ "공유하기" 버튼
   └─ → OS 공유 시트
   └─ → 카톡, 인스타, 문자, 이메일
```

### 3. 핵심 코드

#### 업로드 & 공유 URL 생성
```typescript
const uploadAndShare = async () => {
  setIsLoading(true);

  const formData = new FormData();
  formData.append('file', {
    uri: video_path,
    type: 'video/mp4',
    name: `highlight_${player_number}.mp4`,
  });
  formData.append('video_stem', video_stem);
  formData.append('player', player_number);

  const response = await fetch('https://api.iceiq.app/share', {
    method: 'POST',
    body: formData,
  });

  const data = await response.json();
  // {
  //   "share_id": "ABC12345",
  //   "share_url": "https://iceiq.app/s/ABC12345",
  //   "watermarked": true
  // }
  
  setShareUrl(data.share_url);
};
```

#### OS 공유 시트 열기
```typescript
const onShare = async (url: string, playerName: string) => {
  await Share.share({
    message: `${playerName} 선수의 하이라이트를 확인하세요!`,
    url: url,  // iOS만
    title: 'IceIQ 하이라이트',
  });
};
```

---

## 🌐 서버 엔드포인트

### POST /share - 공유 URL 생성

```bash
curl -X POST https://api.iceiq.app/share \
  -F "file=@highlight.mp4" \
  -F "video_stem=game1" \
  -F "player=4"
```

**응답:**
```json
{
  "share_id": "ABC12345",
  "share_url": "https://iceiq.app/s/ABC12345",
  "stream_url": "/video/share/ABC12345_watermarked.mp4",
  "video_stem": "game1",
  "player": "4",
  "watermarked": true,
  "file_size_mb": 45.2
}
```

### GET /s/{share_id} - 공개 웹 페이지

```bash
https://iceiq.app/s/ABC12345
```

**응답:** HTML 페이지
- 영상 재생 (HTML5 video)
- 선수 정보
- "IceIQ로 분석하기" CTA
- "공유하기" 버튼 (웹 Share API)

### GET /video/share/{filename} - 영상 스트리밍

```bash
https://iceiq.app/video/share/ABC12345_watermarked.mp4
```

**응답:** MP4 영상 (스트리밍)

---

## 🎨 UI/UX 플로우

### 1️⃣ 경기 분석 화면
```
┌─────────────────────────┐
│ 경기 분석                │
├─────────────────────────┤
│ [경기1] [경기2] [경기3] │
├─────────────────────────┤
│ ┌─────────────────────┐ │
│ │ [썸네일]  #4        │ │
│ │           윤지성    │ │
│ │ [공유]    Aigis     │ │
│ │          15:30 / 12회
│ │  [재생 & 공유]      │ │
│ └─────────────────────┘ │
│                         │
│ ┌─────────────────────┐ │
│ │ [썸네일]  #11       │ │
│ │           박리오    │ │
│ │ [공유]    Aigis     │ │
│ │          18:15 / 14회
│ │  [재생 & 공유]      │ │
│ └─────────────────────┘ │
└─────────────────────────┘
```

### 2️⃣ 재생 & 공유 화면
```
┌─────────────────────────┐
│        [영상]           │
│  (16:9 비율)            │
│      [컨트롤]           │
├─────────────────────────┤
│ 윤지성                  │
│ #4 • Aigis              │
│ 15:30 • game1           │
│                         │
│ 공유 링크               │
│ https://iceiq.app/...   │
│  [복사]                 │
├─────────────────────────┤
│ [공유 준비]             │
│ (로딩 중... 1-2분)      │
│ [공유하기]  (비활성)    │
└─────────────────────────┘
```

### 3️⃣ 공유 후
```
┌─────────────────────────┐
│        [영상]           │
│  (16:9 비율)            │
│      [컨트롤]           │
├─────────────────────────┤
│ 윤지성                  │
│ #4 • Aigis              │
│ 15:30 • game1           │
│                         │
│ 공유 링크               │
│ https://iceiq.app/s/... │
│  [복사] ✅              │
├─────────────────────────┤
│ [공유 준비] ✅ 완료     │
│ [공유하기]  (활성)      │
│  ↓                      │
│  카톡, 인스타,          │
│  문자, 이메일 선택      │
└─────────────────────────┘
```

---

## 🔐 파일 구조

```
iceiq-app/
├── app/
│   ├── screens/
│   │   ├── HomeScreen.tsx
│   │   ├── GameAnalysisScreen.tsx       ← 경기 분석
│   │   ├── HighlightShareScreen.tsx     ← 재생 & 공유
│   │   ├── RosterScreen.tsx
│   │   ├── PlayerProfileScreen.tsx
│   │   └── SettingsScreen.tsx
│   ├── navigation/
│   │   └── AppNavigator.tsx             ← 네비게이션
│   └── App.tsx                          ← 진입점
├── package.json
└── app.json
```

---

## 💾 로컬 저장소

### 영상 캐시
```
Documents/iceiq/
├── highlights/
│   ├── highlight_game1_4.mp4
│   ├── highlight_game1_11.mp4
│   └── ...
```

### 메타데이터 (AsyncStorage)
```typescript
{
  "highlight_game1_4": {
    "video_stem": "game1",
    "player_number": "4",
    "player_name": "윤지성",
    "duration": 930,
    "created_at": "2026-04-15T21:53:00"
  }
}
```

---

## 🧪 테스트 시나리오

### 시나리오 1: 기본 공유 플로우
```
1. GameAnalysisScreen 진입
2. "재생 & 공유" 클릭
3. HighlightShareScreen 진입 (영상 재생)
4. "공유 준비" 클릭
   → FormData로 파일 업로드 (POST /share)
   → 서버: FFmpeg 워터마크 적용
   → 서버: 공유 ID 생성 (ABC12345)
   → 앱: share_url 설정 (https://iceiq.app/s/ABC12345)
5. "공유하기" 클릭 (활성화됨)
   → OS 공유 시트 열기
6. 사용자가 "카톡"/"인스타" 선택
   → 메시지 + URL 전송
```

### 시나리오 2: 공개 웹 페이지 접속
```
1. 친구가 카톡에서 링크 클릭
   https://iceiq.app/s/ABC12345
2. 웹 브라우저 열림 (로그인 불필요)
3. 영상 자동 재생
4. "IceIQ로 분석하기" 클릭
   → App Store / Google Play로 이동
   → 또는 웹 대시보드로 이동
5. 앱 설치 / 경기 분석 진입
```

---

## 🚀 배포 체크리스트

### 앱 배포
- [ ] API 엔드포인트 URL 설정 (development/production)
- [ ] 앱 아이콘 & 스플래시 화면
- [ ] 권한 설정 (카메라, 파일, 네트워크)
- [ ] TestFlight / Google Play 베타 배포
- [ ] 리뷰 가이드 준수 (공유 기능)

### 서버 배포
- [ ] FFmpeg 설치 (Linux 서버)
- [ ] 워터마크 폰트 경로 확인
- [ ] /video/share 스트리밍 최적화
- [ ] CORS 설정 (앱 도메인)
- [ ] CDN 연동 (영상 배포)
- [ ] 디스크 공간 모니터링

### 웹 페이지
- [ ] iceiq.app/s/{share_id} 라우팅
- [ ] 메타 태그 (OG, Twitter Card)
- [ ] 모바일 반응형 레이아웃
- [ ] Share API 지원 브라우저 확인
- [ ] SSL/HTTPS 설정

---

## 📊 분석 & 모니터링

### 공유 추적 (선택)
```json
POST /analytics/share
{
  "share_id": "ABC12345",
  "event": "created|clicked|shared",
  "platform": "ios|android|web",
  "user_agent": "..."
}
```

### 메트릭
- 공유 생성 수
- 공유 링크 클릭 수
- 플랫폼별 분포 (iOS/Android/Web)
- 전환율 (공유 → 앱 설치)

---

## 🔗 참고 자료

### React Native
- [Expo AV 문서](https://docs.expo.dev/versions/latest/sdk/av/)
- [Share API](https://reactnative.dev/docs/share)
- [File System](https://docs.expo.dev/versions/latest/sdk/filesystem/)

### FFmpeg
- [FFmpeg 공식 문서](https://ffmpeg.org/documentation.html)
- [drawtext 필터](https://ffmpeg.org/ffmpeg-filters.html#drawtext-1)

### FastAPI
- [File 업로드](https://fastapi.tiangolo.com/tutorial/request-files/)
- [Static Files](https://fastapi.tiangolo.com/tutorial/static-files/)

---

## 📝 다음 단계

1. **로컬 테스트**
   - Expo Go에서 앱 테스트
   - FFmpeg 워터마크 동작 확인

2. **베타 배포**
   - TestFlight (iOS) / Internal Testing (Android)
   - 사용자 피드백 수집

3. **공유 분석**
   - 공유 이벤트 로깅
   - 전환율 모니터링

4. **마케팅 연동**
   - SNS 미리보기 (OG 태그)
   - 공유 인센티브 (선택)

---

## 💬 FAQ

### Q: 왜 "공유 준비"가 필요한가?
A: 서버에서 FFmpeg로 워터마크를 추가하는 데 1-2분이 소요되기 때문입니다.

### Q: 공유 URL은 영구적인가?
A: 기본적으로 영구적입니다. 향후 만료 시간 설정 가능.

### Q: 워터마크 없이 공유할 수 있는가?
A: 서버의 `add_watermark_to_video` 함수를 스킵하면 가능합니다.

### Q: 공유 URL은 몇 개까지 만들 수 있는가?
A: 제한 없음 (서버 스토리지 용량에 따라 다름).

---

✅ **완전한 앱 통합 가이드 완료!** 🎉
