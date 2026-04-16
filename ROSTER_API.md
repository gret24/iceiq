# 로스터 & 선수 정보 API 가이드

## 📋 API 엔드포인트

### 1. 로스터 목록 조회 (연령대 분포 포함)
```bash
GET /api/rosters
```

**응답:**
```json
[
  {
    "filename": "aigis.json",
    "team": "Aigis",
    "total_players": 9,
    "age_distribution": {
      "U-10": 4,
      "U-12": 3,
      "U-14": 1,
      "U-16": 1
    }
  }
]
```

---

### 2. 로스터 상세 조회 (선수 정보 + 성능 기준)
```bash
GET /api/rosters/{filename}
```

**응답:**
```json
{
  "team": "Aigis",
  "season": "2025-2026",
  "organization": "Aigis Youth Hockey Club",
  "coach": "Coach Name",
  "contact": "contact@aigis.example.com",
  "players": [
    {
      "number": 4,
      "name": "윤지성",
      "position": "D",
      "shot_hand": "R",
      "birthdate": "2016-03-15",
      "height_cm": 172,
      "weight_kg": 68,
      "jersey": "4",
      "age": 10,
      "category": "U-12",
      "benchmarks": {
        "avg_speed_kmh": 14.0,
        "max_speed_kmh": 23.0,
        "distance_km": 4.5,
        "sprint_count": 8,
        "dz_pct": 56.0
      }
    }
  ]
}
```

---

### 3. 선수 추가
```bash
POST /api/rosters/{filename}/players
Content-Type: application/json

{
  "number": 99,
  "name": "신입선수",
  "position": "F",
  "shot_hand": "R",
  "birthdate": "2017-05-20",
  "height_cm": 170,
  "weight_kg": 65
}
```

**응답:**
```json
{
  "message": "Player added",
  "player": {
    "number": 99,
    "name": "신입선수",
    "position": "F",
    "shot_hand": "R",
    "birthdate": "2017-05-20",
    "height_cm": 170,
    "weight_kg": 65,
    "jersey": "99"
  }
}
```

---

### 4. 선수 정보 수정
```bash
PUT /api/rosters/{filename}/players/{number}
Content-Type: application/json

{
  "height_cm": 172,
  "weight_kg": 68
}
```

---

### 5. 선수 삭제
```bash
DELETE /api/rosters/{filename}/players/{number}
```

---

### 6. 선수 성능 평가
```bash
POST /api/evaluate-performance
Content-Type: application/json

{
  "video_stem": "game1",
  "jersey": "4",
  "roster_file": "aigis.json"
}
```

**응답:**
```json
{
  "video_stem": "game1",
  "player": {
    "number": 4,
    "name": "윤지성",
    "position": "D",
    "age": 10,
    "category": "U-12"
  },
  "profile": {
    "avg_speed": 14.5,
    "max_speed": 25.0,
    "total_distance": 4.8,
    "sprints": 10,
    "zone_pct": {
      "DZ": 60,
      "NZ": 30,
      "OZ": 10
    }
  },
  "benchmarks": {
    "avg_speed_kmh": 14.0,
    "max_speed_kmh": 23.0,
    "distance_km": 4.5,
    "sprint_count": 8,
    "dz_pct": 56.0
  },
  "evaluation": {
    "speed_rating": 0.07,
    "distance_rating": 0.07,
    "intensity_rating": 0.25,
    "overall": 0.13
  }
}
```

---

## 📊 연령대 분류

| 카테고리 | 나이 | 평균 속도 (F) | 거리 (F) | 스프린트 |
|---------|------|-------------|---------|---------|
| U-10 | < 10 | 13.0 km/h | 3.5 km | 8회 |
| U-12 | 10-11 | 14.5 km/h | 4.2 km | 10회 |
| U-14 | 12-13 | 15.5 km/h | 4.8 km | 12회 |
| U-16 | 14-15 | 16.5 km/h | 5.2 km | 14회 |
| U-18 | 16+ | 17.5 km/h | 5.5 km | 16회 |

---

## 🎯 성능 평가 등급

| 등급 | 범위 | 의미 |
|------|------|------|
| 🟢 Excellent | +1.5 ~ +2.0 | 기대치 훨씬 초과 |
| 🟢 Good | +0.5 ~ +1.5 | 기대치 초과 |
| 🟡 Average | -0.5 ~ +0.5 | 기대치 수준 |
| 🔴 Below Average | -1.5 ~ -0.5 | 기대치 미만 |
| 🔴 Poor | -2.0 ~ -1.5 | 기대치 훨씬 미만 |

---

## 🛠️ 로스터 JSON 형식

```json
{
  "team": "팀명",
  "season": "2025-2026",
  "organization": "조직명",
  "coach": "코치명",
  "contact": "연락처",
  "players": [
    {
      "number": 선수번호,
      "name": "선수명",
      "position": "F|D|G",
      "shot_hand": "L|R|A",
      "birthdate": "YYYY-MM-DD",
      "height_cm": 신장(cm),
      "weight_kg": 체중(kg),
      "jersey": "유니폼번호"
    }
  ]
}
```

### 필수 필드
- `number`: 선수 번호
- `name`: 선수명
- `position`: F(포워드), D(디펜더), G(골리)
- `shot_hand`: L(좌), R(우), A(자동)

### 선택 필드
- `birthdate`: 생년월일 (age + category 자동 계산)
- `height_cm`: 신장 (cm)
- `weight_kg`: 체중 (kg)
- `jersey`: 유니폼 번호 (미지정 시 number 사용)

---

## 💡 사용 예시

### 1. 팀 로스터 생성
```bash
curl -X POST http://localhost:8000/api/rosters \
  -H "Content-Type: application/json" \
  -d '{
    "team": "Dragons",
    "season": "2025-2026",
    "organization": "Dragons Youth Club",
    "players": [
      {
        "number": 1,
        "name": "홍길동",
        "position": "F",
        "shot_hand": "R",
        "birthdate": "2016-05-15",
        "height_cm": 175,
        "weight_kg": 72
      }
    ]
  }'
```

### 2. 로스터에 선수 추가
```bash
curl -X POST http://localhost:8000/api/rosters/aigis.json/players \
  -H "Content-Type: application/json" \
  -d '{
    "number": 88,
    "name": "이순신",
    "position": "D",
    "shot_hand": "L",
    "birthdate": "2015-10-20",
    "height_cm": 180,
    "weight_kg": 78
  }'
```

### 3. 선수 성능 평가
```bash
curl -X POST http://localhost:8000/api/evaluate-performance \
  -H "Content-Type: application/json" \
  -d '{
    "video_stem": "game1",
    "jersey": "4",
    "roster_file": "aigis.json"
  }'
```

---

## 🚀 앱 구현 팁

### 선수 등록 화면
```
┌────────────────────────┐
│ 선수 등록 (U-12 리그)   │
├────────────────────────┤
│ 번호: [  ]             │
│ 이름: [        ]       │
│ 포지션: [F][D][G]      │
│ 슈팅손: [L][R]        │
│ 생년월일: [2016-03-15]│ ← age자동계산
│ 신장(cm): [  ]        │
│ 체중(kg): [  ]        │
│                        │
│ 기대 성과             │
│ - 평균속도: 14.0 km/h│
│ - 거리: 4.5 km       │
│ - 스프린트: 8회      │
│                        │
│ [추가] [취소]         │
└────────────────────────┘
```

### 성능 분석 화면
```
윤지성 선수 분석 (경기 1)
━━━━━━━━━━━━━━━━━━━━━
실제: 14.5 km/h → 기대: 14.0 km/h
🟢 Good (+0.07)

거리: 4.8 km → 기대: 4.5 km  
🟢 Good (+0.07)

강도(스프린트): 10회 → 기대: 8회
🟢 Good (+0.25)

종합 평가: 🟡 Average (+0.13)
```
