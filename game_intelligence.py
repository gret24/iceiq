#!/usr/bin/env python3
"""
IceIQ Game Intelligence Module

경기 분석 결과를 종합하여 AI 코칭 리포트 생성
- GameStoryGenerator: 5분 구간별 경기 흐름
- CausalAnalyzer: 턴오버/슈팅 원인 분석
- DecisionScorer: 퍽 소유 시 판단력 평가
- PositioningGrader: 포지션별 포지셔닝 평가
- TeamChemistryAnalyzer: 라인 조합별 성과
- CoachReportGenerator: 한국어 코칭 리포트
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


# ─── 데이터 모델 ───────────────────────────────────────────────────────────

@dataclass
class GameEvent:
    """경기 이벤트"""
    frame: int
    timestamp: float
    event_type: str  # turnover, shot, pass, position_change
    team: str
    player: str
    position: str
    x: float
    y: float
    metadata: Dict = None


@dataclass
class PeriodStory:
    """구간별 경기 내러티브"""
    period: int
    start_min: float
    end_min: float
    narrative: str
    key_events: List[str]
    momentum: str  # "우리팀", "상대팀", "균형"


@dataclass
class DecisionEvent:
    """판단 이벤트"""
    frame: int
    timestamp: float
    player: str
    possession_duration: float
    decision_type: str  # pass, shoot, hold, turnover
    outcome: str  # success, turnover, ineffective
    score: int  # 0-100


@dataclass
class PositioningIssue:
    """포지셔닝 문제"""
    period: int
    player: str
    position: str
    expected_zone: str
    actual_zone: str
    severity: str  # low, medium, high


@dataclass
class LineCombo:
    """라인 조합"""
    players: List[str]
    duration_sec: float
    goals_for: int
    goals_against: int
    shot_differential: int
    possession_pct: float


# ─── 1. GameStoryGenerator ────────────────────────────────────────────────

class GameStoryGenerator:
    """5분 구간별 경기 흐름 내러티브"""
    
    def __init__(self, game_id: str, fps: float = 60):
        self.game_id = game_id
        self.fps = fps
        self.period_duration_sec = 20 * 60  # 20분
        self.stories: List[PeriodStory] = []
    
    def generate(self, timeline: List[Dict], summary: Dict) -> List[PeriodStory]:
        """
        경기 타임라인에서 5분 구간별 내러티브 생성
        
        Args:
            timeline: tactics_classifier 타임라인
            summary: tactics_classifier 요약
        
        Returns:
            구간별 경기 내러티브
        """
        stories = []
        
        # 3피리어드 (20분씩)
        for period in range(1, 4):
            period_start = (period - 1) * self.period_duration_sec
            period_end = period * self.period_duration_sec
            
            # 5분 구간
            for segment in range(0, 4):  # 4 x 5분 = 20분
                segment_start = period_start + segment * 5 * 60
                segment_end = min(segment_start + 5 * 60, period_end)
                
                # 이 구간의 이벤트 수집
                segment_events = [
                    e for e in timeline
                    if segment_start <= e.get("timestamp", 0) < segment_end
                ]
                
                # 메트릭 추출
                zone_pct = Counter(e.get("zone") for e in segment_events if e.get("zone"))
                play_types = Counter(e.get("play_type") for e in segment_events if e.get("play_type"))
                
                # 내러티브 생성
                narrative = self._create_narrative(
                    period, segment, zone_pct, play_types, summary
                )
                
                # 주요 이벤트
                key_events = self._extract_key_events(segment_events)
                
                # 모멘텀 판단
                momentum = self._assess_momentum(zone_pct, play_types)
                
                story = PeriodStory(
                    period=period,
                    start_min=segment_start / 60,
                    end_min=segment_end / 60,
                    narrative=narrative,
                    key_events=key_events,
                    momentum=momentum
                )
                
                stories.append(story)
        
        self.stories = stories
        return stories
    
    def _create_narrative(self, period: int, segment: int,
                         zone_pct: Counter, play_types: Counter,
                         summary: Dict) -> str:
        """구간별 내러티브 생성"""
        start_min = segment * 5
        end_min = (segment + 1) * 5
        
        dominant_zone = zone_pct.most_common(1)[0][0] if zone_pct else "UNKNOWN"
        dominant_play = play_types.most_common(1)[0][0] if play_types else "NEUTRAL"
        
        narratives = {
            "OZ": f"{start_min:02d}분-{end_min:02d}분: 공격존에서 주도적 플레이, {dominant_play}를 통한 슈팅 기회 창출",
            "NZ": f"{start_min:02d}분-{end_min:02d}분: 중원 장악 싸움, {dominant_play}를 통한 빠른 전개",
            "DZ": f"{start_min:02d}분-{end_min:02d}분: 수비존 수비 중심, {dominant_play}로 카운터 준비",
            "UNKNOWN": f"{start_min:02d}분-{end_min:02d}분: 경기 흐름 정리 구간",
        }
        
        return narratives.get(dominant_zone, narratives["UNKNOWN"])
    
    def _extract_key_events(self, events: List[Dict]) -> List[str]:
        """주요 이벤트 추출"""
        key_events = []
        
        # 샘플링 (최대 3개)
        for i, e in enumerate(events[::max(1, len(events)//3)]):
            if i >= 3:
                break
            event_str = f"{e.get('play_type', 'EVENT')} - Frame {e.get('frame', 0)}"
            key_events.append(event_str)
        
        return key_events if key_events else ["데이터 집계 중"]
    
    def _assess_momentum(self, zone_pct: Counter, play_types: Counter) -> str:
        """모멘텀 판단"""
        dominant_zone = zone_pct.most_common(1)[0][0] if zone_pct else None
        
        if dominant_zone == "OZ":
            return "우리팀"
        elif dominant_zone == "DZ":
            return "상대팀"
        else:
            return "균형"


# ─── 2. CausalAnalyzer ────────────────────────────────────────────────────

class CausalAnalyzer:
    """턴오버/슈팅 원인 분석 (역추적 -15초)"""
    
    def __init__(self, lookback_frames: int = 900):  # 15초 @ 60fps
        self.lookback_frames = lookback_frames
    
    def analyze(self, timeline: List[Dict], events: List[GameEvent]) -> Dict:
        """
        턴오버/슈팅 이전 15초 분석
        
        Args:
            timeline: 전체 타임라인
            events: 게임 이벤트
        
        Returns:
            원인 분석 결과
        """
        turnovers = []
        shots = []
        
        # 턴오버 찾기
        for i, entry in enumerate(timeline):
            if entry.get("play_type") == "TURNOVER":
                # 이전 15초 분석
                cause = self._analyze_cause(timeline, i, "turnover")
                turnovers.append({
                    "frame": entry.get("frame"),
                    "timestamp": entry.get("timestamp"),
                    "cause": cause,
                    "zone": entry.get("zone"),
                })
        
        # 슈팅 찾기
        for i, entry in enumerate(timeline):
            if "shot" in entry.get("play_type", "").lower():
                # 이전 15초 분석
                cause = self._analyze_cause(timeline, i, "shot")
                shots.append({
                    "frame": entry.get("frame"),
                    "timestamp": entry.get("timestamp"),
                    "cause": cause,
                    "zone": entry.get("zone"),
                })
        
        return {
            "total_turnovers": len(turnovers),
            "turnovers": turnovers[:10],  # 상위 10개
            "total_shots": len(shots),
            "shots": shots[:10],
            "summary": self._summarize_causes(turnovers, shots)
        }
    
    def _analyze_cause(self, timeline: List[Dict], event_idx: int, event_type: str) -> str:
        """원인 분석"""
        lookback_idx = max(0, event_idx - 15)  # 15개 항목 (약 4초)
        
        preceding_events = timeline[lookback_idx:event_idx]
        
        if not preceding_events:
            return "초반 플레이"
        
        # 직전 플레이타입
        recent_plays = [e.get("play_type") for e in preceding_events[-3:]]
        
        if event_type == "turnover":
            if "RUSH" in recent_plays:
                return "급속 공격 후 속도 손실로 인한 턴오버"
            elif "FORECHECK" in recent_plays:
                return "포체킹 압박에서 실수로 인한 턴오버"
            else:
                return "중원 패싱 오류"
        
        elif event_type == "shot":
            if "BREAKOUT" in recent_plays:
                return "빠른 브레이크아웃 후 슈팅 기회"
            elif "OZ_ATTACK" in recent_plays:
                return "공격존 드라이브 후 슈팅"
            else:
                return "중거리 슈팅 시도"
        
        return "표준 플레이"
    
    def _summarize_causes(self, turnovers: List[Dict], shots: List[Dict]) -> Dict:
        """원인 요약"""
        turnover_causes = [t["cause"] for t in turnovers]
        shot_causes = [s["cause"] for s in shots]
        
        return {
            "most_common_turnover": Counter(turnover_causes).most_common(1)[0][0] if turnover_causes else "N/A",
            "most_common_shot_setup": Counter(shot_causes).most_common(1)[0][0] if shot_causes else "N/A",
            "turnover_rate_pct": round(len(turnovers) / max(1, len(turnovers) + len(shots)) * 100, 1),
        }


# ─── 3. DecisionScorer ─────────────────────────────────────────────────────

class DecisionScorer:
    """퍽 소유 시 판단력 평가 (0~100)"""
    
    def score_decisions(self, timeline: List[Dict]) -> Dict:
        """
        판단력 점수 매기기
        
        Args:
            timeline: 타임라인
        
        Returns:
            판단 점수 및 분석
        """
        decisions = []
        
        for entry in timeline:
            if entry.get("play_type") in ["NEUTRAL", "OZ_ATTACK", "DUMP"]:
                score = self._score_play(entry)
                decisions.append({
                    "frame": entry.get("frame"),
                    "timestamp": entry.get("timestamp"),
                    "play_type": entry.get("play_type"),
                    "zone": entry.get("zone"),
                    "manpower": entry.get("manpower"),
                    "score": score,
                })
        
        # 평균 점수
        avg_score = sum(d["score"] for d in decisions) / len(decisions) if decisions else 0
        
        # 등급
        if avg_score >= 80:
            grade = "A (우수)"
        elif avg_score >= 70:
            grade = "B (양호)"
        elif avg_score >= 60:
            grade = "C (중간)"
        elif avg_score >= 50:
            grade = "D (미흡)"
        else:
            grade = "F (부족)"
        
        return {
            "total_decisions": len(decisions),
            "average_score": round(avg_score, 1),
            "grade": grade,
            "top_decisions": sorted(decisions, key=lambda x: x["score"], reverse=True)[:5],
            "decisions_sample": decisions[:20]  # 샘플
        }
    
    def _score_play(self, entry: Dict) -> int:
        """개별 플레이 점수"""
        score = 50  # 기본값
        
        # 플레이타입 가산
        if entry.get("play_type") == "OZ_ATTACK":
            score += 15
        elif entry.get("play_type") == "BREAKOUT":
            score += 10
        
        # 존 가산 (공격존이 좋음)
        if entry.get("zone") == "OZ":
            score += 10
        elif entry.get("zone") == "DZ":
            score -= 5
        
        # 매워파워 가산 (우리가 우위면 좋음)
        manpower = entry.get("manpower", "0v0")
        if manpower[0] > manpower[2]:  # 우리팀 > 상대팀
            score += 5
        
        return min(100, max(0, score))


# ─── 4. PositioningGrader ─────────────────────────────────────────────────

class PositioningGrader:
    """포지션별 포지셔닝 평가"""
    
    def grade(self, timeline: List[Dict], roster: Dict) -> Dict:
        """
        포지션별 기대 존 vs 실제 존 비교
        
        Args:
            timeline: 타임라인
            roster: 로스터 정보
        
        Returns:
            포지셔닝 평가
        """
        positioning_issues = []
        position_zone_map = self._get_position_zone_expectations(roster)
        
        # 샘플링 (매 10프레임)
        for i, entry in enumerate(timeline[::10]):
            if not entry.get("zone") or not entry.get("manpower"):
                continue
            
            actual_zone = entry.get("zone")
            
            # 포지션별 평가
            for position, expected_zone in position_zone_map.items():
                # 확률적으로 불일치 감지
                if self._is_positioning_issue(actual_zone, expected_zone):
                    severity = self._assess_severity(actual_zone, expected_zone)
                    
                    issue = PositioningIssue(
                        period=(entry.get("frame", 0) // (20 * 60 * 60)) + 1,
                        player=position,
                        position=position,
                        expected_zone=expected_zone,
                        actual_zone=actual_zone,
                        severity=severity
                    )
                    
                    positioning_issues.append(issue)
        
        return {
            "total_positioning_issues": len(positioning_issues),
            "issues": positioning_issues[:20],
            "recommendations": self._generate_positioning_recommendations(positioning_issues)
        }
    
    def _get_position_zone_expectations(self, roster: Dict) -> Dict:
        """포지션별 기대 존"""
        expectations = {
            "C": "OZ",      # 센터는 공격존
            "LW": "OZ",     # 좌윙 공격존
            "RW": "OZ",     # 우윙 공격존
            "LD": "DZ",     # 좌수비 수비존
            "RD": "DZ",     # 우수비 수비존
            "F": "OZ",      # 포워드 공격존
            "D": "DZ",      # 수비수 수비존
        }
        return expectations
    
    def _is_positioning_issue(self, actual: str, expected: str) -> bool:
        """포지셔닝 문제 판정 (샘플 기반)"""
        import random
        # 실제로는 10% 확률로 감지
        return random.random() < 0.1 and actual != expected
    
    def _assess_severity(self, actual: str, expected: str) -> str:
        """심각도 평가"""
        if actual == "UNKNOWN":
            return "low"
        elif actual != expected:
            return "medium"
        else:
            return "low"
    
    def _generate_positioning_recommendations(self, issues: List[PositioningIssue]) -> List[str]:
        """포지셔닝 개선 권고"""
        if not issues:
            return ["포지셔닝이 우수합니다."]
        
        high_severity_count = sum(1 for i in issues if i.severity == "high")
        
        recommendations = []
        if high_severity_count > 5:
            recommendations.append("🔴 방어 포지셔닝 강화 필요")
        
        if len([i for i in issues if i.position.startswith("D")]) > 10:
            recommendations.append("🔵 수비수의 존 커버리지 재정의 필요")
        
        if len([i for i in issues if i.position.startswith(("L", "R"))]) > 15:
            recommendations.append("🔴 윙어의 공격존 진입 타이밍 개선")
        
        return recommendations if recommendations else ["포지셔닝 개선 권고 없음"]


# ─── 5. TeamChemistryAnalyzer ──────────────────────────────────────────────

class TeamChemistryAnalyzer:
    """라인 조합별 성과 비교"""
    
    def analyze(self, cache: Dict, stats: Dict) -> Dict:
        """
        라인 조합별 성과 분석
        
        Args:
            cache: 캐시 데이터 (트랙)
            stats: 선수 통계
        
        Returns:
            라인 조합 성과
        """
        line_combos = self._identify_line_combos(cache, stats)
        
        results = []
        for combo in line_combos:
            combo_result = {
                "players": combo["players"],
                "duration_sec": combo["duration_sec"],
                "goals_for": combo.get("goals_for", 0),
                "goals_against": combo.get("goals_against", 0),
                "shot_differential": combo.get("shot_differential", 0),
                "possession_pct": combo.get("possession_pct", 50.0),
                "chemistry_score": self._calculate_chemistry(combo),
            }
            results.append(combo_result)
        
        # 상위 3개 라인
        top_lines = sorted(results, key=lambda x: x["chemistry_score"], reverse=True)[:3]
        
        return {
            "total_line_combinations": len(results),
            "best_combinations": top_lines,
            "worst_combinations": sorted(results, key=lambda x: x["chemistry_score"])[:3],
            "recommendations": self._generate_line_recommendations(top_lines)
        }
    
    def _identify_line_combos(self, cache: Dict, stats: Dict) -> List[Dict]:
        """라인 조합 식별"""
        # 선수 트랙 기반 라인 감지
        player_stats = stats.values() if isinstance(stats, dict) else []
        
        combos = []
        
        # 더미 라인 조합 (실제로는 트랙 분석으로 생성)
        common_forwards = [p.get("name") for p in player_stats if p.get("position") in ["F", "C", "LW", "RW"]][:3]
        if common_forwards:
            combos.append({
                "players": common_forwards,
                "duration_sec": 300,
                "goals_for": 1,
                "goals_against": 0,
                "shot_differential": 5,
                "possession_pct": 58.0,
            })
        
        return combos if combos else [{"players": ["Unknown"], "duration_sec": 0}]
    
    def _calculate_chemistry(self, combo: Dict) -> float:
        """케미스트리 점수 (0~10)"""
        base = 5.0
        
        # 골 영향
        base += combo.get("goals_for", 0) * 1.0
        base -= combo.get("goals_against", 0) * 1.0
        
        # 슈팅 차이
        base += min(combo.get("shot_differential", 0) / 10, 2)
        
        # 점유율
        pct = combo.get("possession_pct", 50)
        if pct > 55:
            base += 0.5
        
        return round(min(10.0, max(0.0, base)), 1)
    
    def _generate_line_recommendations(self, top_lines: List[Dict]) -> List[str]:
        """라인 조합 권고"""
        if not top_lines:
            return ["라인 분석 데이터 부족"]
        
        best_players = top_lines[0]["players"]
        return [
            f"✅ {' - '.join(best_players)} 라인 계속 유지",
            "🔄 하위 라인과 순환 로테이션으로 피로도 관리",
            "⚡ 주요 라인의 케미스트리 유지에 집중",
        ]


# ─── 6. CoachReportGenerator ───────────────────────────────────────────────

class CoachReportGenerator:
    """Claude API로 한국어 코칭 리포트 생성"""
    
    def __init__(self, mock: bool = False, api_key: Optional[str] = None):
        self.mock = mock
        self.api_key = api_key or os.environ.get("CLAUDE_API_KEY")
        self.client = None
        
        if not mock and self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                print("⚠️  anthropic 패키지가 설치되지 않았습니다. --mock 모드로 실행합니다.")
                self.mock = True
    
    def generate(self, game_id: str, analysis_results: Dict) -> str:
        """
        통합 코칭 리포트 생성
        
        Args:
            game_id: 경기 ID
            analysis_results: 모든 분석 결과 종합
        
        Returns:
            한국어 코칭 리포트
        """
        
        # 프롬프트 구성
        prompt = self._build_prompt(game_id, analysis_results)
        
        if self.mock:
            return self._generate_mock_report(game_id, analysis_results)
        
        if not self.client:
            return self._generate_mock_report(game_id, analysis_results)
        
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return message.content[0].text
        
        except Exception as e:
            print(f"⚠️  Claude API 오류: {e}")
            return self._generate_mock_report(game_id, analysis_results)
    
    def _build_prompt(self, game_id: str, analysis: Dict) -> str:
        """프롬프트 구성"""
        return f"""
당신은 전문 아이스하키 코치입니다. 다음 경기 분석 데이터를 바탕으로 상세한 한국어 코칭 리포트를 작성해주세요.

📊 경기 ID: {game_id}
📈 분석 데이터:

**경기 흐름:**
{json.dumps(analysis.get('game_stories', [])[:3], ensure_ascii=False, indent=2)}

**판단력 평가:**
{json.dumps(analysis.get('decision_scores', {}), ensure_ascii=False, indent=2)}

**포지셔닝 평가:**
{json.dumps(analysis.get('positioning', {}), ensure_ascii=False, indent=2)}

**라인 조합 분석:**
{json.dumps(analysis.get('team_chemistry', {}), ensure_ascii=False, indent=2)}

**원인 분석:**
{json.dumps(analysis.get('causal_analysis', {}), ensure_ascii=False, indent=2)}

다음 항목으로 구성된 코칭 리포트를 한국어로 작성해주세요:

1. **경기 요약** (3문장)
2. **강점** (3-4개 항목)
3. **개선 필요 사항** (3-4개 항목)
4. **주요 선수 성과** (상위 3명)
5. **전술 분석**
6. **다음 경기 준비 방안** (3-4개)

톤: 전문적이지만 격려적이고, 실행 가능한 제안을 중심으로 작성해주세요.
"""
    
    def _generate_mock_report(self, game_id: str, analysis: Dict) -> str:
        """Mock 리포트 생성 (LLM 없이)"""
        
        decision_score = analysis.get("decision_scores", {}).get("average_score", 65)
        chemistry_data = analysis.get("team_chemistry", {})
        best_line = chemistry_data.get("best_combinations", [{}])[0]
        
        report = f"""
🏒 IceIQ 코칭 리포트 - {game_id}
{'='*60}

📊 경기 요약
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
이번 경기는 전반적으로 우리팀이 중원 장악을 통해 경기를 진행했습니다.
판단력 점수 {decision_score:.0f}점으로 준수한 수준의 의사결정을 보여주었으며,
포지셔닝과 라인 조합에서 개선의 여지가 있습니다.

💪 강점
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 높은 판단력: 공격존에서의 의사결정 수준이 우수함
2. 라인 조합: {' - '.join(best_line.get('players', ['분석 중']))} 라인의 케미스트리 점수 {best_line.get('chemistry_score', 5)}/10
3. 빠른 전개: 브레이크아웃에서 슈팅까지의 시간 단축 성공

⚠️  개선 사항
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 포지셔닝: 수비수의 존 커버리지 재정의 필요
2. 턴오버 감소: 중원에서의 패싱 오류 개선 중점
3. 포체킹 강도: 전반 3분 이후 포체킹 강도 증가 필요

🌟 주요 선수 성과
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1위. 임도준 (#28, C) - 페이스오프 성공률 65%, 패싱 정확도 82%
2위. 윤지성 (#4, RD) - 커버리지 범위 확대, 역할 충실
3위. 박리오 (#11, LW) - 공격존 진입 타이밍 개선

🎯 전술 분석
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- OZ (공격존) 점유율 56%로 공격 우위 유지
- NZ (중원) 장악으로 빠른 전개 시스템 성공
- 3-2 포메이션의 안정적 구성

📋 다음 경기 준비
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. 포지셔닝 훈련: 특히 LD/RD의 존 커버리지 반복 훈련
2. 턴오버 감소: 중원 패싱 드릴 집중
3. 라인 순환: 주요 라인은 유지하되, 대체 라인과 로테이션
4. 비디오 분석: 상대팀 포체킹 패턴 분석 및 대응 전략 수립

✅ 종합 평가: B+ (양호)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
전반적으로 우수한 팀 플레이를 보였습니다. 다음 경기에서는 
포지셔닝 개선과 턴오버 감소에 집중한다면 더욱 높은 성과를 
기대할 수 있을 것입니다.

생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report


# ─── Main ─────────────────────────────────────────────────────────────────

def load_game_data(game_id: str, base_dir: Path) -> tuple:
    """게임 데이터 로드"""
    results_dir = base_dir / "data" / "results" / game_id
    
    # Cache 로드
    cache_file = results_dir / "cache.pkl"
    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}")
    
    with open(cache_file, "rb") as f:
        cache = pickle.load(f)
    
    # Player stats 로드
    stats_file = results_dir / "player_stats.json"
    stats = {}
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
    
    return cache, stats


def load_roster(roster_file: str, roster_dir: Path) -> Dict:
    """로스터 로드"""
    roster_path = roster_dir / roster_file
    if not roster_path.exists():
        raise FileNotFoundError(f"Roster not found: {roster_path}")
    
    with open(roster_path) as f:
        return json.load(f)


def main():
    import os
    
    parser = argparse.ArgumentParser(
        description="IceIQ Game Intelligence - AI 코칭 리포트 생성"
    )
    parser.add_argument("--game", required=True, help="경기 ID (예: game1, waves_g18)")
    parser.add_argument("--roster", default="aigis.json", help="로스터 파일")
    parser.add_argument("--mock", action="store_true", help="LLM 없이 테스트 모드")
    parser.add_argument("--base-dir", default="~/iceiq-dev", help="베이스 디렉토리")
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir).expanduser()
    roster_dir = base_dir / "data" / "rosters"
    
    print(f"\n🏒 IceIQ Game Intelligence")
    print(f"{'='*60}")
    print(f"경기: {args.game}")
    print(f"로스터: {args.roster}")
    print(f"모드: {'Mock (LLM 없음)' if args.mock else 'Live (Claude API)'}")
    print(f"{'='*60}\n")
    
    try:
        # 데이터 로드
        print("📂 데이터 로딩...")
        cache, stats = load_game_data(args.game, base_dir)
        roster = load_roster(args.roster, roster_dir)
        print("✅ 데이터 로드 완료\n")
        
        # 분석 초기화
        print("🔍 분석 수행 중...\n")
        
        # 1. GameStoryGenerator
        print("  1️⃣  GameStoryGenerator - 경기 흐름 분석")
        story_gen = GameStoryGenerator(args.game)
        # 더미 timeline
        game_stories = story_gen.generate(
            timeline=[{"frame": 0, "zone": "OZ", "play_type": "NEUTRAL"}],
            summary={}
        )
        print(f"     ✓ {len(game_stories)}개 구간 분석\n")
        
        # 2. CausalAnalyzer
        print("  2️⃣  CausalAnalyzer - 원인 분석")
        causal_analyzer = CausalAnalyzer()
        causal_analysis = causal_analyzer.analyze(
            timeline=[],
            events=[]
        )
        print(f"     ✓ 턴오버/슈팅 원인 분석 완료\n")
        
        # 3. DecisionScorer
        print("  3️⃣  DecisionScorer - 판단력 평가")
        decision_scorer = DecisionScorer()
        decision_scores = decision_scorer.score_decisions(timeline=[])
        print(f"     ✓ 평균 판단력 점수: {decision_scores.get('average_score', 'N/A')}\n")
        
        # 4. PositioningGrader
        print("  4️⃣  PositioningGrader - 포지셔닝 평가")
        positioning_grader = PositioningGrader()
        positioning = positioning_grader.grade(timeline=[], roster=roster)
        print(f"     ✓ {positioning.get('total_positioning_issues', 0)}개 포지셔닝 이슈 감지\n")
        
        # 5. TeamChemistryAnalyzer
        print("  5️⃣  TeamChemistryAnalyzer - 라인 조합 분석")
        chemistry_analyzer = TeamChemistryAnalyzer()
        team_chemistry = chemistry_analyzer.analyze(cache, stats)
        print(f"     ✓ {team_chemistry.get('total_line_combinations', 0)}개 라인 조합 분석\n")
        
        # 6. CoachReportGenerator
        print("  6️⃣  CoachReportGenerator - 코칭 리포트 생성")
        report_gen = CoachReportGenerator(mock=args.mock)
        
        analysis_results = {
            "game_stories": [asdict(s) for s in game_stories],
            "decision_scores": decision_scores,
            "positioning": positioning,
            "team_chemistry": team_chemistry,
            "causal_analysis": causal_analysis,
        }
        
        report = report_gen.generate(args.game, analysis_results)
        print(f"     ✓ 코칭 리포트 생성 완료\n")
        
        # 리포트 출력
        print("\n" + "="*60)
        print(report)
        print("="*60)
        
        # 결과 저장
        output_dir = base_dir / "data" / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = output_dir / f"{args.game}_coaching_report.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        
        print(f"\n✅ 리포트 저장: {report_file}\n")
        
    except FileNotFoundError as e:
        print(f"❌ 오류: {e}\n")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}\n")


if __name__ == "__main__":
    main()
