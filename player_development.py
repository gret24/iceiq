#!/usr/bin/env python3
"""
IceIQ Player Development System
NHL 트레이너 수준의 장기 선수 육성 시스템

7개 엔진:
1. PhysicalDevelopmentEngine - 신체 발달 추적
2. SkillProgressionEngine - 기술 발달 단계
3. WeaknessDetector - 약점 감지
4. TrainingPlanGenerator - 훈련 프로그램
5. MilestoneTracker - 성장 이정표
6. RecruitingProfileBuilder - 리크루팅 프로필
7. ParentReportGenerator - 학부모 리포트
"""

import json
import pickle
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import argparse
import statistics


# ─── 데이터 모델 ───────────────────────────────────────────────────────────

@dataclass
class PhysicalMetrics:
    """신체 발달 지표"""
    game: str
    date: str
    avg_speed_kmh: float
    max_speed_kmh: float
    total_distance_km: float
    stamina_decay_pct: float  # 3P 후반 속도 저하율


@dataclass
class SkillStage:
    """기술 단계"""
    skill: str  # skating, puck_handling, passing, shooting, positioning, game_sense
    stage: str  # Foundation, Developing, Competent, Advanced, Elite
    score: float  # 0-100
    trend: str  # "up", "stable", "down"
    consistency: float  # 일관성 (3경기 기준)


@dataclass
class Weakness:
    """약점"""
    category: str  # technical, tactical, physical, psychological
    description: str
    severity: str  # Critical, Major, Minor
    frequency: int  # 나타난 경기 수
    improvement_weeks: int  # 예상 개선 주수


@dataclass
class TrainingSession:
    """훈련 세션"""
    day: str
    focus: str
    duration_min: int
    intensity: str  # light, moderate, high
    drills: List[str]
    goal: str


# ─── 1. PhysicalDevelopmentEngine ───────────────────────────────────────────

class PhysicalDevelopmentEngine:
    """신체 발달 추적"""
    
    def __init__(self, player_jersey: str, age: int):
        self.player_jersey = player_jersey
        self.age = age
        self.metrics_history: List[PhysicalMetrics] = []
    
    def add_game_metrics(self, game_id: str, metrics: Dict):
        """경기 메트릭 추가"""
        physical = PhysicalMetrics(
            game=game_id,
            date=datetime.now().isoformat(),
            avg_speed_kmh=metrics.get("avg_speed", 0),
            max_speed_kmh=metrics.get("max_speed", 0),
            total_distance_km=metrics.get("total_distance", 0),
            stamina_decay_pct=metrics.get("stamina_decay_pct", 0)
        )
        self.metrics_history.append(physical)
    
    def get_growth_curves(self) -> Dict:
        """성장 곡선 계산"""
        if not self.metrics_history:
            return {}
        
        avg_speeds = [m.avg_speed_kmh for m in self.metrics_history]
        max_speeds = [m.max_speed_kmh for m in self.metrics_history]
        distances = [m.total_distance_km for m in self.metrics_history]
        
        return {
            "avg_speed_curve": avg_speeds,
            "max_speed_curve": max_speeds,
            "distance_curve": distances,
            "avg_speed_trend": self._calculate_trend(avg_speeds),
            "max_speed_trend": self._calculate_trend(max_speeds),
            "distance_trend": self._calculate_trend(distances),
        }
    
    def get_percentile(self, metric: str) -> Dict:
        """표준 곡선 대비 퍼센타일"""
        # 표준 곡선 (U-12 평균)
        standards = {
            10: {"avg_speed": 13.5, "max_speed": 23.0, "distance": 3.8},
            12: {"avg_speed": 14.5, "max_speed": 24.0, "distance": 4.2},
            14: {"avg_speed": 15.5, "max_speed": 26.0, "distance": 4.8},
            16: {"avg_speed": 16.5, "max_speed": 28.0, "distance": 5.2},
        }
        
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        age_group = ((self.age // 2) * 2)  # 가까운 짝수 연령대로
        age_group = min(max(age_group, 10), 16)
        
        standard = standards.get(age_group, standards[12])
        
        if metric == "avg_speed":
            actual = latest.avg_speed_kmh
            std = standard["avg_speed"]
        elif metric == "max_speed":
            actual = latest.max_speed_kmh
            std = standard["max_speed"]
        else:
            actual = latest.total_distance_km
            std = standard["distance"]
        
        percentile = (actual / std) * 100 if std > 0 else 0
        
        return {
            "metric": metric,
            "actual": actual,
            "standard": std,
            "percentile": round(percentile, 1),
            "assessment": self._percentile_assessment(percentile)
        }
    
    def detect_growth_spurt(self) -> Optional[str]:
        """성장 스퍼트 감지"""
        if len(self.metrics_history) < 3:
            return None
        
        recent_speeds = [m.max_speed_kmh for m in self.metrics_history[-3:]]
        growth_rate = (recent_speeds[-1] - recent_speeds[0]) / recent_speeds[0] * 100
        
        if growth_rate > 15:
            return f"최고속도 +{growth_rate:.0f}% 성장 감지 — 성장기 진입 추정"
        
        return None
    
    def detect_overtraining(self) -> Optional[str]:
        """과훈련 경고"""
        if len(self.metrics_history) < 2:
            return None
        
        recent_stamina = [m.stamina_decay_pct for m in self.metrics_history[-2:]]
        
        if recent_stamina[-1] > recent_stamina[-2] + 10:
            return f"스태미나 급감({recent_stamina[-1]:.0f}%) + 속도 저하 → 과훈련 경고"
        
        return None
    
    def _calculate_trend(self, values: List[float]) -> str:
        """추세 계산"""
        if len(values) < 2:
            return "insufficient_data"
        
        recent_avg = statistics.mean(values[-2:])
        overall_avg = statistics.mean(values)
        
        if recent_avg > overall_avg * 1.05:
            return "상승중"
        elif recent_avg < overall_avg * 0.95:
            return "하강중"
        else:
            return "안정적"
    
    def _percentile_assessment(self, percentile: float) -> str:
        """퍼센타일 평가"""
        if percentile > 110:
            return "상위 15%"
        elif percentile > 100:
            return "평균 이상"
        elif percentile > 90:
            return "평균"
        else:
            return "하위 40%"


# ─── 2. SkillProgressionEngine ──────────────────────────────────────────────

class SkillProgressionEngine:
    """기술 발달 단계"""
    
    SKILL_CATEGORIES = [
        "skating", "puck_handling", "passing", 
        "shooting", "positioning", "game_sense"
    ]
    
    STAGES = ["Foundation", "Developing", "Competent", "Advanced", "Elite"]
    
    def __init__(self, player_jersey: str):
        self.player_jersey = player_jersey
        self.skill_scores: Dict[str, List[float]] = {
            skill: [] for skill in self.SKILL_CATEGORIES
        }
    
    def add_game_skills(self, skills: Dict[str, float]):
        """경기별 기술 점수 추가"""
        for skill, score in skills.items():
            if skill in self.skill_scores:
                self.skill_scores[skill].append(score)
    
    def get_skill_stages(self) -> List[SkillStage]:
        """기술별 단계 판정"""
        stages = []
        
        for skill in self.SKILL_CATEGORIES:
            scores = self.skill_scores.get(skill, [])
            
            if not scores:
                continue
            
            avg_score = statistics.mean(scores)
            consistency = self._calculate_consistency(scores)
            trend = self._calculate_trend(scores)
            
            # 점수 → 단계 매핑
            if avg_score < 40:
                stage = "Foundation"
            elif avg_score < 60:
                stage = "Developing"
            elif avg_score < 75:
                stage = "Competent"
            elif avg_score < 90:
                stage = "Advanced"
            else:
                stage = "Elite"
            
            stages.append(SkillStage(
                skill=skill,
                stage=stage,
                score=round(avg_score, 1),
                trend=trend,
                consistency=round(consistency, 1)
            ))
        
        return stages
    
    def _calculate_consistency(self, scores: List[float]) -> float:
        """일관성 (표준편차의 역수)"""
        if len(scores) < 2:
            return 100.0
        
        std_dev = statistics.stdev(scores)
        consistency = 100 - min(std_dev, 50)  # 0~100
        
        return consistency
    
    def _calculate_trend(self, scores: List[float]) -> str:
        """추세"""
        if len(scores) < 2:
            return "new"
        
        recent_avg = statistics.mean(scores[-2:])
        overall_avg = statistics.mean(scores)
        
        if recent_avg > overall_avg * 1.05:
            return "up"
        elif recent_avg < overall_avg * 0.95:
            return "down"
        else:
            return "stable"


# ─── 3. WeaknessDetector ───────────────────────────────────────────────────

class WeaknessDetector:
    """약점 감지"""
    
    def __init__(self, player_jersey: str):
        self.player_jersey = player_jersey
        self.game_issues: Dict[str, List[Dict]] = defaultdict(list)
    
    def add_game_issues(self, game_id: str, issues: List[Dict]):
        """경기별 이슈 추가"""
        for issue in issues:
            self.game_issues[issue.get("category", "unknown")].append({
                "game": game_id,
                "description": issue.get("description"),
                "metric": issue.get("metric"),
                "value": issue.get("value")
            })
    
    def detect_weaknesses(self, num_games: int = 3) -> List[Weakness]:
        """약점 감지 (3경기 이상 반복만)"""
        weaknesses = []
        
        for category, issues in self.game_issues.items():
            # 같은 이슈가 3경기 이상 반복되었는지 확인
            issue_counter = Counter(i["description"] for i in issues)
            
            for description, frequency in issue_counter.items():
                if frequency >= num_games:
                    severity = self._assess_severity(frequency, category)
                    improvement_weeks = self._estimate_improvement(category)
                    
                    weaknesses.append(Weakness(
                        category=category,
                        description=description,
                        severity=severity,
                        frequency=frequency,
                        improvement_weeks=improvement_weeks
                    ))
        
        return sorted(weaknesses, key=lambda w: w.frequency, reverse=True)
    
    def _assess_severity(self, frequency: int, category: str) -> str:
        """심각도 평가"""
        if frequency >= 5:
            return "Critical"
        elif frequency >= 4:
            return "Major"
        else:
            return "Minor"
    
    def _estimate_improvement(self, category: str) -> int:
        """개선 소요 주수"""
        estimates = {
            "technical": 4,
            "tactical": 3,
            "physical": 2,
            "psychological": 6
        }
        return estimates.get(category, 4)


# ─── 4. TrainingPlanGenerator ──────────────────────────────────────────────

class TrainingPlanGenerator:
    """훈련 계획 생성"""
    
    def __init__(self, age: int):
        self.age = age
    
    def generate_weekly_plan(self, weaknesses: List[Weakness]) -> List[TrainingSession]:
        """주간 훈련 계획 생성"""
        plan = []
        
        # 약점 카테고리별 드릴
        focus_drills = self._get_focus_drills(weaknesses)
        
        # 월: 스케이팅
        plan.append(TrainingSession(
            day="월",
            focus="스케이팅 파워",
            duration_min=self._get_duration(45),
            intensity="high",
            drills=["크로스오버", "파워스타트", "아일랜드 피벗"],
            goal="가속력 향상"
        ))
        
        # 화: 퍽 핸들링
        plan.append(TrainingSession(
            day="화",
            focus="퍽 핸들링 + 헤드업",
            duration_min=self._get_duration(40),
            intensity="moderate",
            drills=["콘 드리블", "시야 훈련", "한손 핸들링"],
            goal="헤드업 비율 60% 달성"
        ))
        
        # 수: 전술
        plan.append(TrainingSession(
            day="수",
            focus="전술 포지셔닝",
            duration_min=self._get_duration(50),
            intensity="high",
            drills=["3v2 러시", "백체킹 시뮬레이션", "포메이션 유지"],
            goal="존 적절성 향상"
        ))
        
        # 목: 체력
        plan.append(TrainingSession(
            day="목",
            focus="체력",
            duration_min=self._get_duration(35),
            intensity="high",
            drills=["인터벌 스케이팅", "코어 강화", "무산소 훈련"],
            goal="스태미나 향상"
        ))
        
        # 금: 스크리미지
        plan.append(TrainingSession(
            day="금",
            focus="스크리미지",
            duration_min=self._get_duration(60),
            intensity="moderate",
            drills=["5v5 게임", "약점 포커스"],
            goal="이번 주 학습 적용"
        ))
        
        return plan
    
    def _get_duration(self, base_min: int) -> int:
        """연령대별 훈련 시간 조절"""
        if self.age < 12:
            return int(base_min * 0.7)  # 30% 단축
        elif self.age < 14:
            return base_min
        else:
            return int(base_min * 1.2)  # 20% 연장
    
    def _get_focus_drills(self, weaknesses: List[Weakness]) -> List[str]:
        """약점별 드릴 선택"""
        drills = []
        
        for weakness in weaknesses[:2]:  # 상위 2개 약점
            if "백체킹" in weakness.description:
                drills.extend(["백체킹 시뮬레이션", "3v2 러시"])
            elif "헤드업" in weakness.description:
                drills.extend(["시야 훈련", "콘 드리블"])
            elif "스태미나" in weakness.description:
                drills.extend(["인터벌", "코어 강화"])
        
        return drills if drills else ["기본 기술 훈련"]


# ─── 5. MilestoneTracker ───────────────────────────────────────────────────

class MilestoneTracker:
    """성장 이정표"""
    
    def __init__(self, player_jersey: str, age: int):
        self.player_jersey = player_jersey
        self.age = age
        self.milestones: List[Dict] = []
    
    def check_milestones(self, metrics: Dict) -> List[str]:
        """마일스톤 자동 감지"""
        milestones = []
        
        # 속도 마일스톤
        if metrics.get("max_speed", 0) >= 30:
            milestones.append("🔥 최고속도 30km/h 돌파!")
        if metrics.get("max_speed", 0) >= 25:
            milestones.append("⚡ 최고속도 25km/h 달성!")
        
        # 헤드업 마일스톤
        if metrics.get("headup_pct", 0) >= 60:
            milestones.append("👀 헤드업 60% 달성!")
        
        # Hockey IQ 마일스톤
        if metrics.get("hockey_iq", 0) >= 75:
            milestones.append("🧠 Hockey IQ 75점 달성!")
        
        # 거리 마일스톤
        if metrics.get("total_distance", 0) >= 5.0:
            milestones.append("🏃 경기당 5km 돌파!")
        
        return milestones
    
    def project_future(self, growth_rates: List[float]) -> str:
        """장기 프로젝션"""
        future_age = self.age + 2
        
        if not growth_rates:
            return f"데이터 부족으로 프로젝션 불가"
        
        avg_growth = statistics.mean(growth_rates)
        
        if avg_growth > 0.05:  # 5% 이상 성장
            return f"현재 성장률({avg_growth*100:.1f}%) 유지 시 {future_age}세에 평균 속도 상위 10% 진입 예상"
        elif avg_growth > 0.02:
            return f"현재 성장률({avg_growth*100:.1f}%) 유지 시 {future_age}세에 평균 수준 유지 예상"
        else:
            return f"현재 성장률({avg_growth*100:.1f}%)은 저조 — 훈련 강화 필요"


# ─── 6. RecruitingProfileBuilder ────────────────────────────────────────────

class RecruitingProfileBuilder:
    """리크루팅 프로필 (미국 고교/대학 기준)"""
    
    def __init__(self, player_data: Dict):
        self.name = player_data.get("name", "Unknown")
        self.jersey = player_data.get("jersey", "?")
        self.position = player_data.get("position", "F")
        self.age = player_data.get("age", 12)
        self.birthdate = player_data.get("birthdate", "2014-01-01")
        self.height_cm = player_data.get("height_cm", 170)
        self.weight_kg = player_data.get("weight_kg", 65)
    
    def build_profile(self, skill_stages: List[SkillStage], 
                      hockey_iq: float, growth_rates: List[float]) -> Dict:
        """리크루팅 프로필 생성"""
        
        # 기술 레이더 차트 (6개 축)
        skills_radar = {}
        for stage in skill_stages:
            skills_radar[stage.skill] = stage.score
        
        # 미국 표준 등급 판정
        tier = self._assess_tier(skill_stages, hockey_iq)
        
        # 성장 곡선 평가
        growth_trajectory = "ascending" if statistics.mean(growth_rates) > 0.03 else "stable"
        
        profile = {
            "name": self.name,
            "jersey": self.jersey,
            "position": self.position,
            "age": self.age,
            "birthdate": self.birthdate,
            "physical": {
                "height_cm": self.height_cm,
                "weight_kg": self.weight_kg,
                "bmi": round(self.weight_kg / ((self.height_cm/100) ** 2), 1)
            },
            "skills_radar": skills_radar,
            "hockey_iq": round(hockey_iq, 1),
            "tier": tier,
            "growth_trajectory": growth_trajectory,
            "strengths": self._identify_strengths(skill_stages),
            "development_areas": self._identify_development_areas(skill_stages),
        }
        
        return profile
    
    def _assess_tier(self, skills: List[SkillStage], hockey_iq: float) -> str:
        """미국 고교/대학 기준 등급 판정"""
        avg_skill = statistics.mean([s.score for s in skills])
        
        # 나이와 기술을 종합 평가
        age_factor = 1.0 + (self.age - 12) * 0.05  # 나이가 높을수록 가산
        combined_score = (avg_skill * 0.6 + hockey_iq * 0.4) * age_factor
        
        if combined_score >= 80:
            return "U-14 Tier 1 (Elite) — 미국 U-14 최상위 수준"
        elif combined_score >= 70:
            return "U-14 Tier 2 (AAA) — 미국 AAA 수준"
        elif combined_score >= 60:
            return "U-14 Tier 3 (AA) — 미국 AA 수준"
        else:
            return "Development — 기초 단계"
    
    def _identify_strengths(self, skills: List[SkillStage]) -> List[str]:
        """강점 식별 (상위 3개)"""
        sorted_skills = sorted(skills, key=lambda s: s.score, reverse=True)
        return [f"{s.skill}: {s.stage}" for s in sorted_skills[:3]]
    
    def _identify_development_areas(self, skills: List[SkillStage]) -> List[str]:
        """개발 영역 (하위 2개)"""
        sorted_skills = sorted(skills, key=lambda s: s.score)
        return [f"{s.skill}: {s.stage}" for s in sorted_skills[:2]]
    
    def generate_pdf_summary(self) -> str:
        """PDF 요약 텍스트"""
        return f"""
RECRUITING PROFILE
{self.name} (#{self.jersey}) | {self.position}
Age: {self.age} | DOB: {self.birthdate}
Height: {self.height_cm}cm | Weight: {self.weight_kg}kg

Physical Profile
- Well-proportioned frame for age
- Good mobility and coordination

Technical Skills
- Strong skating fundamentals
- Developing puck handling
- Good decision-making

Projection: {self.position} prospect with solid foundation
Recommended: Advanced skill development program
"""


# ─── 7. ParentReportGenerator ──────────────────────────────────────────────

class ParentReportGenerator:
    """학부모 친화적 리포트"""
    
    def __init__(self, mock: bool = True):
        self.mock = mock
    
    def generate_report(self, player_info: Dict, analysis: Dict) -> str:
        """학부모 리포트 생성"""
        
        if self.mock:
            return self._generate_mock_report(player_info, analysis)
        
        # Claude API 버전 (실제로는 Claude 호출)
        return self._generate_with_claude(player_info, analysis)
    
    def _generate_mock_report(self, player_info: Dict, analysis: Dict) -> str:
        """Mock 리포트"""
        
        name = player_info.get("name", "선수명")
        position = player_info.get("position", "포지션")
        
        report = f"""
👨‍👧‍👦 {name} 선수 육성 리포트
{'='*60}

🎯 이번 달 종합 평가
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{name} 선수는 이번 달 전반적으로 우수한 성장을 보였습니다.
특히 스케이팅 기본기와 결정력이 돋보이며, 앞으로의 발전이 기대됩니다.

💪 잘한 점 (TOP 3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 스케이팅 파워: 경기마다 일관된 높은 속도 유지
2. 포지셔닝 감각: {position} 포지션에서 올바른 위치 선택
3. 팀 의식: 팀 플레이를 우선으로 하는 자세

🌱 더 성장할 점 (TOP 2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 슈팅 자신감: 기회 때 과감하게 슈팅할 용기 필요
   → 가정에서: "슈팅 연습"이 중요합니다. 거실에서 할 수 있는 
     핸들링 드릴 영상을 첨부했으니 참고해주세요.

2. 3기간 후반 스태미나: 마지막 시간에 집중력 유지 필요
   → 가정에서: 규칙적인 수면과 영양 관리가 도움됩니다.

📅 이번 주 훈련 포커스
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
월: 스케이팅 파워 강화 (크로스오버 + 파워스타트)
화: 퍽 핸들링 + 시야 훈련
수: 포지셔닝 드릴 (3v2 러시 연습)
목: 체력 운동 (인터벌 스케이팅)
금: 실제 경기 상황 연습 (스크리미지)

👀 다음 경기 관전 포인트
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
다음 경기에서 주목해야 할 점:

1. {name} 선수의 **헤드업 습관**을 지켜봐주세요.
   - 패스를 받기 전에 주변을 살피는 습관
   - 하키는 "눈이 빠른 선수"가 이깁니다!

2. **3기간 후반 속도 유지**를 관찰해주세요.
   - 1~2기간에 비해 3기간의 움직임이 얼마나 빠른지 비교
   - "처음처럼 스케이팅했네!" 라고 말씀해주면 큰 격려가 됩니다.

3. **슈팅 시도** 횟수를 세어보세요.
   - 기회가 오면 슈팅하는 용기를 응원해주세요!

🏆 성장 마일스톤
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ 최고속도 24km/h 달성
⏳ 목표: 다음달 최고속도 26km/h (추가 성장)
⏳ 목표: 헤드업 비율 60% 달성

📞 코치와 소통
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{name} 선수의 발전을 응원합니다!
혹시 질문이 있으시면 언제든 연락주세요.

생성일: {datetime.now().strftime('%Y년 %m월 %d일')}
"""
        
        return report
    
    def _generate_with_claude(self, player_info: Dict, analysis: Dict) -> str:
        """Claude API를 사용한 리포트"""
        # 실제 구현에서는 Claude API 호출
        # 여기서는 mock 반환
        return self._generate_mock_report(player_info, analysis)


# ─── Main ──────────────────────────────────────────────────────────────────

def load_player_games(player_jersey: str, games: List[str], 
                      base_dir: Path) -> Dict:
    """여러 경기의 선수 데이터 로드"""
    player_data = {
        "avg_speeds": [],
        "max_speeds": [],
        "distances": [],
        "hockey_iqs": [],
    }
    
    for game in games:
        results_dir = base_dir / "data" / "results" / game
        
        # Player stats 로드
        stats_file = results_dir / "player_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
            
            if player_jersey in stats:
                p = stats[player_jersey]
                player_data["avg_speeds"].append(p.get("avg_speed", 0))
                player_data["max_speeds"].append(p.get("max_speed", 0))
                player_data["distances"].append(p.get("total_distance", 0))
    
    return player_data


def main():
    parser = argparse.ArgumentParser(
        description="IceIQ Player Development - 선수 육성 시스템"
    )
    parser.add_argument("--player", required=True, help="선수 번호")
    parser.add_argument("--games", nargs="+", required=True, help="경기 ID 목록")
    parser.add_argument("--roster", default="aigis.json", help="로스터 파일")
    parser.add_argument("--section", help="개별 섹션 (physical/skills/weaknesses/training)")
    parser.add_argument("--mock", action="store_true", help="LLM 없이 테스트")
    parser.add_argument("--base-dir", default="~/iceiq-dev", help="베이스 디렉토리")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).expanduser()
    roster_dir = base_dir / "data" / "rosters"
    
    print(f"\n🏒 IceIQ Player Development System")
    print(f"{'='*60}")
    print(f"선수: #{args.player}")
    print(f"경기: {', '.join(args.games)}")
    print(f"{'='*60}\n")
    
    try:
        # 로스터에서 선수 정보 로드
        roster_file = roster_dir / args.roster
        with open(roster_file) as f:
            roster = json.load(f)
        
        player = None
        for p in roster.get("players", []):
            if str(p.get("number")) == str(args.player):
                player = p
                break
        
        if not player:
            print(f"❌ 선수 #{args.player}를 찾을 수 없습니다.\n")
            return
        
        player_name = player.get("name", "Unknown")
        position = player.get("position", "Unknown")
        age = player.get("age", 12)
        
        print(f"이름: {player_name}")
        print(f"포지션: {position}")
        print(f"나이: {age}세\n")
        
        # 다경기 데이터 로드
        print("📂 경기 데이터 로딩...")
        player_data = load_player_games(args.player, args.games, base_dir)
        print(f"✅ {len(args.games)}경기 데이터 로드 완료\n")
        
        # 1. PhysicalDevelopmentEngine
        if not args.section or args.section == "physical":
            print("1️⃣  Physical Development")
            physical = PhysicalDevelopmentEngine(args.player, age)
            
            for i, game in enumerate(args.games):
                if i < len(player_data["avg_speeds"]):
                    metrics = {
                        "avg_speed": player_data["avg_speeds"][i],
                        "max_speed": player_data["max_speeds"][i],
                        "total_distance": player_data["distances"][i],
                        "stamina_decay_pct": 15 + i * 3  # 더미
                    }
                    physical.add_game_metrics(game, metrics)
            
            growth = physical.get_growth_curves()
            percentile = physical.get_percentile("max_speed")
            growth_spurt = physical.detect_growth_spurt()
            overtraining = physical.detect_overtraining()
            
            print(f"  최고속도: {percentile.get('actual', 0):.1f} km/h")
            print(f"  평가: {percentile.get('assessment', 'N/A')}")
            
            if growth_spurt:
                print(f"  📈 {growth_spurt}")
            if overtraining:
                print(f"  ⚠️  {overtraining}")
            print()
        
        # 2. SkillProgressionEngine
        stages = None  # 나중에 사용하기 위해 먼저 선언
        if not args.section or args.section == "skills":
            print("2️⃣  Skill Progression")
            skills_engine = SkillProgressionEngine(args.player)
            
            # 더미 데이터
            for i in range(len(args.games)):
                game_skills = {
                    "skating": 65 + i*3,
                    "puck_handling": 60 + i*2,
                    "passing": 70 + i*2,
                    "shooting": 55 + i*3,
                    "positioning": 65 + i*2,
                    "game_sense": 70 + i*3,
                }
                skills_engine.add_game_skills(game_skills)
            
            stages = skills_engine.get_skill_stages()
            for stage in stages:
                print(f"  {stage.skill:20s}: {stage.stage:12s} ({stage.score:5.1f}점) [{stage.trend}]")
            print()
        else:
            # 다른 섹션에서도 stages가 필요하면 미리 계산
            skills_engine = SkillProgressionEngine(args.player)
            for i in range(len(args.games)):
                game_skills = {
                    "skating": 65 + i*3,
                    "puck_handling": 60 + i*2,
                    "passing": 70 + i*2,
                    "shooting": 55 + i*3,
                    "positioning": 65 + i*2,
                    "game_sense": 70 + i*3,
                }
                skills_engine.add_game_skills(game_skills)
            stages = skills_engine.get_skill_stages()
        
        # 3. WeaknessDetector
        if not args.section or args.section == "weaknesses":
            print("3️⃣  Weakness Detection")
            weakness = WeaknessDetector(args.player)
            
            # 더미 이슈
            for game in args.games:
                issues = [
                    {"category": "physical", "description": "3P 후반 백체킹 반응 느려짐", "metric": "backcheck_time", "value": 1.2},
                    {"category": "tactical", "description": "우측 시야 제한", "metric": "headup_pct", "value": 0.12},
                ]
                weakness.add_game_issues(game, issues)
            
            weaknesses = weakness.detect_weaknesses(num_games=2)
            for w in weaknesses:
                print(f"  [{w.severity}] {w.description} ({w.frequency}/3 경기)")
                print(f"        → {w.improvement_weeks}주 훈련으로 개선 가능")
            print()
        
        # 4. TrainingPlanGenerator
        if not args.section or args.section == "training":
            print("4️⃣  Training Plan")
            training = TrainingPlanGenerator(age)
            weaknesses = []  # 실제로는 위의 약점 데이터 사용
            
            plan = training.generate_weekly_plan(weaknesses)
            for session in plan:
                print(f"  {session.day}: {session.focus} ({session.duration_min}분)")
            print()
        
        # 5. MilestoneTracker
        if not args.section or args.section == "milestones":
            print("5️⃣  Milestones")
            milestone = MilestoneTracker(args.player, age)
            
            # 최신 지표로 마일스톤 확인
            latest_metrics = {
                "max_speed": player_data["max_speeds"][-1] if player_data["max_speeds"] else 24,
                "total_distance": player_data["distances"][-1] if player_data["distances"] else 4.5,
                "headup_pct": 45,
                "hockey_iq": 70,
            }
            
            milestones = milestone.check_milestones(latest_metrics)
            for m in milestones:
                print(f"  {m}")
            
            growth_rates = []
            if len(player_data["max_speeds"]) > 1:
                growth_rates = [(player_data["max_speeds"][i+1] - player_data["max_speeds"][i]) / player_data["max_speeds"][i] 
                               for i in range(len(player_data["max_speeds"])-1)]
            
            projection = milestone.project_future(growth_rates)
            print(f"  📊 {projection}")
            print()
        
        # 6. RecruitingProfileBuilder
        if not args.section or args.section == "recruiting":
            print("6️⃣  Recruiting Profile")
            recruiter = RecruitingProfileBuilder({
                "name": player_name,
                "jersey": args.player,
                "position": position,
                "age": age,
                "birthdate": player.get("birthdate", "2014-01-01"),
                "height_cm": player.get("height_cm", 170),
                "weight_kg": player.get("weight_kg", 65),
            })
            
            profile = recruiter.build_profile(stages, 70, [0.03, 0.02])
            print(f"  등급: {profile['tier']}")
            print(f"  성장 추세: {profile['growth_trajectory']}")
            print(f"  강점: {', '.join(profile['strengths'][:2])}")
            print(f"  개발영역: {', '.join(profile['development_areas'])}")
            print()
        
        # 7. ParentReportGenerator
        if not args.section or args.section == "parent":
            print("7️⃣  Parent Report")
            parent_gen = ParentReportGenerator(mock=True)
            
            report = parent_gen.generate_report(
                {
                    "name": player_name,
                    "position": position,
                    "age": age,
                },
                {}
            )
            
            print(report)
            print()
        
        print(f"{'='*60}")
        print(f"✅ 선수 분석 완료\n")
        
    except FileNotFoundError as e:
        print(f"❌ 오류: {e}\n")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}\n")


if __name__ == "__main__":
    main()
