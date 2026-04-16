# Game Intelligence Module - AI 코칭 리포트 생성 가이드

## 🤖 개요

`game_intelligence.py`는 IceIQ의 모든 분석 결과를 종합하여 AI 기반 한국어 코칭 리포트를 생성합니다.

**포함된 6개 분석 모듈:**
1. **GameStoryGenerator** - 5분 구간별 경기 흐름 내러티브
2. **CausalAnalyzer** - 턴오버/슈팅 원인 분석 (역추적 -15초)
3. **DecisionScorer** - 퍽 소유 시 판단력 평가 (0~100점)
4. **PositioningGrader** - 포지션별 기대 존 vs 실제 존 비교
5. **TeamChemistryAnalyzer** - 라인 조합별 성과 비교
6. **CoachReportGenerator** - Claude API로 한국어 코칭 리포트 생성

---

## 🚀 설치 & 사용

### 기본 사용법

```bash
python3 game_intelligence.py --game waves_g18 --roster aigis.json
```

### Mock 모드 (LLM 없이 테스트)

```bash
python3 game_intelligence.py --game waves_g18 --roster aigis.json --mock
```

### 옵션

```
--game       : 경기 ID (required, 예: game1, waves_g18)
--roster     : 로스터 파일 (기본값: aigis.json)
--mock       : LLM 없이 테스트 (플래그)
--base-dir   : 베이스 디렉토리 (기본값: ~/iceiq-dev)
```

---

## 📊 분석 모듈 상세

### 1️⃣ GameStoryGenerator

**목적:** 5분 구간별 경기 흐름을 자연어 내러티브로 변환

**출력:**
```json
{
  "period": 1,
  "start_min": 0.0,
  "end_min": 5.0,
  "narrative": "00분-05분: 공격존에서 주도적 플레이, NEUTRAL를 통한 슈팅 기회 창출",
  "key_events": ["NEUTRAL - Frame 1920", "RUSH - Frame 2100"],
  "momentum": "우리팀"
}
```

**분석 항목:**
- 존 분포 (OZ/NZ/DZ)
- 주요 플레이타입 (NEUTRAL, BREAKOUT, FORECHECK 등)
- 모멘텀 판정 (우리팀 / 상대팀 / 균형)

---

### 2️⃣ CausalAnalyzer

**목적:** 턴오버와 슈팅이 발생한 원인을 역추적

**기법:** 이벤트 발생 시점에서 -15초 전의 플레이 분석

**출력:**
```json
{
  "total_turnovers": 12,
  "turnovers": [
    {
      "frame": 5000,
      "timestamp": 83.33,
      "cause": "포체킹 압박에서 실수로 인한 턴오버",
      "zone": "DZ"
    }
  ],
  "summary": {
    "most_common_turnover": "중원 패싱 오류",
    "most_common_shot_setup": "빠른 브레이크아웃 후 슈팅 기회",
    "turnover_rate_pct": 45.2
  }
}
```

**분석 항목:**
- 턴오버 원인 분류
- 슈팅 세팅 분석
- 턴오버율 계산

---

### 3️⃣ DecisionScorer

**목적:** 퍽 소유 시 판단력을 0~100점으로 평가

**점수 기준:**
- 기본값: 50점
- 플레이타입: OZ_ATTACK +15, BREAKOUT +10
- 존: OZ +10, DZ -5
- 매워파워: 우리팀 우위 +5

**등급:**
- A (80점 이상): 우수
- B (70~79점): 양호
- C (60~69점): 중간
- D (50~59점): 미흡
- F (50점 미만): 부족

**출력:**
```json
{
  "total_decisions": 250,
  "average_score": 68.5,
  "grade": "C (중간)",
  "top_decisions": [
    {
      "frame": 10000,
      "timestamp": 166.67,
      "play_type": "OZ_ATTACK",
      "zone": "OZ",
      "manpower": "2v1",
      "score": 95
    }
  ]
}
```

---

### 4️⃣ PositioningGrader

**목적:** 포지션별 기대 존(expected zone) vs 실제 존(actual zone) 비교

**포지션별 기대 존:**
| 포지션 | 기대 존 |
|--------|--------|
| C, LW, RW | OZ (공격존) |
| LD, RD | DZ (수비존) |

**심각도 판정:**
- LOW: 기대 존과 일치
- MEDIUM: 기대 존과 불일치
- HIGH: 위험한 불일치

**출력:**
```json
{
  "total_positioning_issues": 15,
  "issues": [
    {
      "period": 1,
      "player": "LD",
      "position": "LD",
      "expected_zone": "DZ",
      "actual_zone": "OZ",
      "severity": "high"
    }
  ],
  "recommendations": [
    "🔴 방어 포지셔닝 강화 필요",
    "🔵 수비수의 존 커버리지 재정의 필요"
  ]
}
```

---

### 5️⃣ TeamChemistryAnalyzer

**목적:** 라인 조합별 성과를 비교하여 최적 라인 추천

**케미스트리 점수 계산:**
- 기본값: 5.0
- 골 영향: +1.0 per goal_for, -1.0 per goal_against
- 슈팅 차이: +0.2 per shot_differential
- 점유율: 55% 이상 +0.5

**출력:**
```json
{
  "total_line_combinations": 5,
  "best_combinations": [
    {
      "players": ["임도준", "박리오", "김재원"],
      "duration_sec": 300,
      "goals_for": 2,
      "goals_against": 0,
      "shot_differential": 8,
      "possession_pct": 58.0,
      "chemistry_score": 8.5
    }
  ],
  "worst_combinations": [...],
  "recommendations": [
    "✅ 임도준 - 박리오 - 김재원 라인 계속 유지",
    "🔄 하위 라인과 순환 로테이션으로 피로도 관리"
  ]
}
```

---

### 6️⃣ CoachReportGenerator

**목적:** 모든 분석 결과를 통합하여 한국어 코칭 리포트 생성

**동작 모드:**
1. **Live 모드** (API 키 설정 시)
   - Claude API 호출
   - 고급 자연어 처리
   - 맥락 기반 분석

2. **Mock 모드** (--mock 플래그)
   - 템플릿 기반 리포트
   - LLM 없이 즉시 실행
   - 테스트 및 프로토타이핑용

**Live 모드 설정:**
```bash
export CLAUDE_API_KEY="sk-ant-..."
python3 game_intelligence.py --game waves_g18 --roster aigis.json
```

**리포트 구성:**
1. 경기 요약 (3문장)
2. 강점 (3-4개 항목)
3. 개선 필요 사항 (3-4개 항목)
4. 주요 선수 성과 (상위 3명)
5. 전술 분석
6. 다음 경기 준비 방안 (3-4개)
7. 종합 평가 (A~F 등급)

**예시 리포트:**
```markdown
🏒 IceIQ 코칭 리포트 - waves_g18
══════════════════════════════════════════════

📊 경기 요약
이번 경기는 전반적으로 우리팀이 중원 장악을 통해 경기를 진행했습니다.
판단력 점수 68.5점으로 준수한 수준의 의사결정을 보여주었으며,
포지셔닝과 라인 조합에서 개선의 여지가 있습니다.

💪 강점
1. 높은 판단력: 공격존에서의 의사결정 수준이 우수함
2. 라인 조합: 케미스트리 점수 8.5/10
3. 빠른 전개: 브레이크아웃에서 슈팅까지의 시간 단축 성공

⚠️  개선 사항
1. 포지셔닝: 수비수의 존 커버리지 재정의 필요
2. 턴오버 감소: 중원에서의 패싱 오류 개선 중점
3. 포체킹 강도: 전반 3분 이후 포체킹 강도 증가 필요
```

---

## 💾 입력 데이터 포맷

### Cache (cache.pkl)
```python
{
  "fps": 60,
  "frame_count": 232800,
  "teams": {track_id: "team_a" | "team_b"},
  "tracks": {frame: [{track_id, cx, cy, w, h, keypoints}]},
  "jersey_map": {track_id: {jersey, name, team}},
}
```

### Player Stats (player_stats.json)
```json
{
  "4": {
    "jersey": "4",
    "name": "윤지성",
    "position": "RD",
    "team": "Aigis",
    "track_ids": [1, 45],
    "total_frames": 10000
  }
}
```

### Roster (aigis.json)
```json
{
  "team": "Aigis",
  "season": "2025-2026",
  "players": [
    {
      "number": 4,
      "name": "윤지성",
      "position": "RD",
      "shot_hand": "R",
      "birthdate": "2016-03-15"
    }
  ]
}
```

### Tactics Timeline (tactics_classifier.py 출력)
```json
{
  "frame": 1920,
  "timestamp": 32.0,
  "zone": "OZ",
  "play_type": "NEUTRAL",
  "home_formation": "3-LINE",
  "away_formation": "SPREAD",
  "manpower": "0v0"
}
```

---

## 📤 출력 파일

### 리포트 저장 위치
```
~/iceiq-dev/data/reports/{game_id}_coaching_report.md
```

### 예시
```
waves_g18_coaching_report.md
game1_coaching_report.md
```

---

## 🧪 테스트 시나리오

### 시나리오 1: Mock 모드 빠른 테스트
```bash
python3 game_intelligence.py --game waves_g18 --roster aigis.json --mock
```
**예상 시간:** <5초
**출력:** 템플릿 기반 리포트

### 시나리오 2: 실제 분석 (API 필요)
```bash
export CLAUDE_API_KEY="sk-ant-..."
python3 game_intelligence.py --game game1 --roster aigis.json
```
**예상 시간:** 10-30초 (API 응답 시간)
**출력:** Claude 생성 고급 리포트

### 시나리오 3: 배치 처리
```bash
for game in game1 game2 game3 waves_g18; do
  python3 game_intelligence.py --game $game --roster aigis.json --mock
done
```

---

## 🔍 분석 결과 해석

### GameStoryGenerator 해석
- **momentum "우리팀"**: 공격존 장악 중
- **momentum "상대팀"**: 수비 압박 받는 상황
- **momentum "균형"**: 중원 장악 싸움

### DecisionScorer 해석
```
90-100점: 매우 우수한 판단 (즉시 슈팅 또는 정확한 패싱)
70-89점: 좋은 판단 (플레이 계속 진행)
50-69점: 보통 판단 (더 나은 선택지 있었음)
0-49점: 부족한 판단 (턴오버 위험)
```

### CausalAnalyzer 해석
- **"포체킹 압박에서 실수"**: 방어 압박 증강 필요
- **"중원 패싱 오류"**: 패싱 드릴 집중
- **"빠른 브레이크아웃"**: 좋은 신호, 계속 유지

### PositioningGrader 해석
- **HIGH 심각도**: 즉시 포지셔닝 훈련 필요
- **MEDIUM 심각도**: 주의 필요, 모니터링
- **LOW 심각도**: 안정적 포지셔닝

### TeamChemistryAnalyzer 해석
- **8.0 이상**: 최고의 라인 조합, 우선 배치
- **6.0~8.0**: 좋은 라인, 로테이션 활용
- **4.0~6.0**: 대체 라인, 개선 필요

---

## 🛠️ 커스터마이징

### 새로운 분석 모듈 추가
```python
class CustomAnalyzer:
    def analyze(self, data: Dict) -> Dict:
        # 분석 로직
        return {"results": [...]}

# main() 함수에서 호출
custom = CustomAnalyzer()
custom_results = custom.analyze(cache)
analysis_results["custom"] = custom_results
```

### 리포트 템플릿 수정
```python
def _generate_mock_report(self, game_id: str, analysis: Dict) -> str:
    # 템플릿 수정
    report = f"""
    # 커스텀 리포트 시작
    ...
    """
    return report
```

---

## 📈 성능 최적화

### 대용량 경기 분석 시
```bash
# Parallel processing (배치)
for game in game1 game2 game3; do
  python3 game_intelligence.py --game $game --roster aigis.json --mock &
done
wait
```

### 메모리 최적화
- Timeline 샘플링 활용 (모든 프레임 분석 X)
- 필요한 분석만 선택 실행

---

## 🔐 보안 & API

### Claude API 키 관리
```bash
# 환경 변수 설정
export CLAUDE_API_KEY="sk-ant-..."

# 또는 .env 파일
echo "CLAUDE_API_KEY=sk-ant-..." > .env
```

### API 요청 제한
- 기본 타임아웃: 30초
- Retry: 자동 3회
- Rate limit: API 정책 준수

---

## 📊 대시보드 통합 (향후)

### 웹 대시보드 API
```python
@app.post("/api/generate-coaching-report")
async def generate_report(game_id: str, roster: str):
    gen = GameIntelligence(game_id, roster)
    report = gen.run()
    return {"report": report}
```

---

## 💡 FAQ

### Q: Mock 모드와 Live 모드의 차이는?
**A:** Mock 모드는 LLM 없이 템플릿으로 빠르게 생성, Live 모드는 Claude API로 고급 분석.

### Q: 리포트 생성에 얼마나 걸리나?
**A:** Mock은 <5초, Live는 10-30초 (API 응답 포함).

### Q: 여러 경기를 동시에 분석할 수 있나?
**A:** 네, 배시 스크립트로 병렬 처리 가능.

### Q: 리포트를 커스터마이징할 수 있나?
**A:** 네, `_generate_mock_report()` 메서드 수정하거나 Claude 프롬프트 변경.

### Q: API 키가 없으면?
**A:** `--mock` 플래그로 LLM 없이 실행.

---

✅ **Game Intelligence Module 완전 구현!** 🤖🏒
