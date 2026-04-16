"""
IceIQ Scouting Engine
상대팀 전술 분석 + 우리 팀 프로필 → AI 스카우팅 리포트 생성

입력:
  opponent.json  — tactics_classifier.py 출력 (opponent_patterns 포함)
  our_team.json  — player_metrics.py 출력

출력:
  scouting_report.json — 전략 권고 + AI 코멘터리

사용법:
  # 목업 테스트 (API 없이)
  python3 scouting_engine.py opponent.json our_team.json --opponent-name "Lopez" --mock

  # 실제 AI 분석 (OPENAI_API_KEY 필요)
  python3 scouting_engine.py opponent.json our_team.json --opponent-name "Lopez"
"""

import json
import os
import sys
from datetime import datetime
from typing import Any


# ─── 규칙 기반 전략 생성 ───────────────────────────────────────────────

def build_strategy(opponent_patterns: dict, our_players: list) -> dict:
    """
    상대 약점 + 우리 팀 강점 → 구체적 전략 권고
    """
    weaknesses = opponent_patterns.get("weaknesses", [])
    forecheck_pct = opponent_patterns.get("forecheck_intensity_pct", 0)
    breakout_pct = opponent_patterns.get("breakout_frequency_pct", 0)
    primary_formation = opponent_patterns.get("primary_formation", "UNKNOWN")
    zone_tendency = opponent_patterns.get("zone_tendency", {})
    play_types = opponent_patterns.get("play_type_distribution", {})

    # 우리 팀 분석
    our_fast_players = sorted(
        [p for p in our_players if p.get("speed", {}).get("avg_kmh", 0) > 0],
        key=lambda p: p["speed"].get("avg_kmh", 0),
        reverse=True
    )[:3]

    our_stamina_players = sorted(
        [p for p in our_players if p.get("stamina", {}).get("avg_decay_pct", 100) < 20],
        key=lambda p: p.get("stamina", {}).get("avg_decay_pct", 100)
    )[:3]

    recommendations = []

    # 포체킹 관련 전략
    if forecheck_pct < 15:
        recommendations.append({
            "priority": 1,
            "phase": "공격",
            "tactic": "적극적 포체킹",
            "description": "상대 포체킹이 소극적입니다. 1-2-2 포체킹으로 상대 브레이크아웃을 차단하세요.",
            "key_players": [f"#{p['jersey']}" for p in our_fast_players],
        })

    if breakout_pct > 25:
        recommendations.append({
            "priority": 1,
            "phase": "수비",
            "tactic": "브레이크아웃 차단",
            "description": f"상대가 브레이크아웃을 자주 사용합니다({breakout_pct:.0f}%). "
                           f"뉴트럴존 트랩으로 대응하세요.",
            "key_players": [f"#{p['jersey']}" for p in our_stamina_players],
        })

    # 포메이션별 대응
    if primary_formation in ("2-1-2", "2-3"):
        recommendations.append({
            "priority": 2,
            "phase": "공격",
            "tactic": "위크사이드 활용",
            "description": f"상대 주 포메이션({primary_formation})의 위크사이드 공간을 공략하세요. "
                           f"역방향 패싱으로 디펜더를 당기세요.",
            "key_players": [],
        })
    elif primary_formation == "1-3-1":
        recommendations.append({
            "priority": 2,
            "phase": "공격",
            "tactic": "와이드 플레이",
            "description": "1-3-1 포메이션은 와이드 공간에 취약합니다. 사이드 공략 후 크로스를 노리세요.",
            "key_players": [],
        })
    elif primary_formation == "SCRAMBLE":
        recommendations.append({
            "priority": 1,
            "phase": "공격",
            "tactic": "빠른 전환 압박",
            "description": "상대 포메이션이 불안정합니다. 빠른 패싱과 스크린 플레이로 혼란을 가중하세요.",
            "key_players": [f"#{p['jersey']}" for p in our_fast_players],
        })

    # 존 점유 기반 전략
    oz_pct = zone_tendency.get("OZ_ATTACK", 0)
    dz_pct = zone_tendency.get("DZ_DEFEND", 0)
    if dz_pct > 40:
        recommendations.append({
            "priority": 2,
            "phase": "공격",
            "tactic": "롱 시프트 압박",
            "description": f"상대가 수비존에 많이 머뭅니다({dz_pct:.0f}%). "
                           f"지속적 공격압박으로 체력 소모를 유도하세요.",
            "key_players": [f"#{p['jersey']}" for p in our_stamina_players],
        })

    # CYCLE 플레이 대응
    cycle_pct = play_types.get("CYCLE", 0)
    if cycle_pct > 20:
        recommendations.append({
            "priority": 2,
            "phase": "수비",
            "tactic": "싸이클 차단",
            "description": f"상대가 싸이클 플레이를 즐깁니다({cycle_pct:.0f}%). "
                           f"저점 커버리지를 강화하고 포인트 슈터를 마크하세요.",
            "key_players": [],
        })

    # 기본 권고 (없으면 추가)
    if not recommendations:
        recommendations.append({
            "priority": 3,
            "phase": "전술",
            "tactic": "균형 잡힌 플레이",
            "description": "뚜렷한 약점이 없습니다. 기본 시스템 유지와 실수 최소화에 집중하세요.",
            "key_players": [],
        })

    # 우선순위 정렬
    recommendations.sort(key=lambda x: x["priority"])

    return {
        "recommendations": recommendations,
        "matchup_advantage": _assess_matchup(opponent_patterns, our_players),
    }


def _assess_matchup(opponent_patterns: dict, our_players: list) -> dict:
    """우리 팀 vs 상대팀 매치업 평가"""
    weaknesses = opponent_patterns.get("weaknesses", [])
    high_severity = sum(1 for w in weaknesses if w.get("severity") == "high")
    mid_severity = sum(1 for w in weaknesses if w.get("severity") == "medium")

    # 우리 팀 평균 속도
    speeds = [p.get("speed", {}).get("avg_kmh", 0) for p in our_players if p.get("speed", {}).get("avg_kmh", 0) > 0]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0

    score = high_severity * 2 + mid_severity
    if score >= 4:
        advantage = "높음"
    elif score >= 2:
        advantage = "보통"
    else:
        advantage = "낮음"

    return {
        "overall": advantage,
        "exploitable_weaknesses": high_severity,
        "our_avg_speed_kmh": round(avg_speed, 1),
        "notes": f"상대 고위험 약점 {high_severity}개, 중위험 {mid_severity}개 발견",
    }


# ─── 목업 리포트 ──────────────────────────────────────────────────────

def generate_mock_report(opponent_name: str, opponent_patterns: dict,
                          our_players: list, strategy: dict) -> dict:
    """AI 없이 규칙 기반으로 리포트 생성 (테스트/오프라인 용)"""
    weaknesses = opponent_patterns.get("weaknesses", [])
    primary_formation = opponent_patterns.get("primary_formation", "UNKNOWN")
    forecheck_pct = opponent_patterns.get("forecheck_intensity_pct", 0)
    breakout_pct = opponent_patterns.get("breakout_frequency_pct", 0)
    transition_rate = opponent_patterns.get("transition_rate_pct", 0)

    recs = strategy.get("recommendations", [])
    top_rec = recs[0] if recs else {}
    advantage = strategy.get("matchup_advantage", {})

    summary = (
        f"{opponent_name} 팀은 {primary_formation} 포메이션을 주로 사용합니다. "
        f"포체킹 강도 {forecheck_pct:.0f}%, 브레이크아웃 빈도 {breakout_pct:.0f}%, "
        f"전환률 {transition_rate:.0f}%입니다. "
    )
    if weaknesses:
        w_desc = weaknesses[0]["description"]
        summary += f"주요 약점: {w_desc}. "
    summary += f"전체 매치업 우위: {advantage.get('overall', 'N/A')}."

    coaching_notes = []
    for rec in recs[:3]:
        coaching_notes.append(
            f"[{rec['phase']}] {rec['tactic']}: {rec['description']}"
        )

    player_matchups = []
    for p in our_players[:5]:
        jersey = p.get("jersey", "?")
        position = p.get("auto_position", "FWD")
        tags = p.get("style_tags", [])
        player_matchups.append({
            "jersey": jersey,
            "position": position,
            "role": f"{'공격 첨병' if 'SPEED' in tags else '균형 플레이'}",
            "note": f"스타일: {', '.join(tags) if tags else 'N/A'}",
        })

    return {
        "generated_by": "mock",
        "summary": summary,
        "coaching_notes": coaching_notes,
        "player_matchups": player_matchups,
    }


# ─── AI 리포트 (OpenAI) ──────────────────────────────────────────────

def generate_ai_report(opponent_name: str, opponent_patterns: dict,
                        our_players: list, strategy: dict) -> dict:
    """Claude API로 자연어 스카우팅 리포트 생성"""
    try:
        import anthropic
    except ImportError:
        print("⚠️  anthropic 패키지 없음. pip install anthropic")
        return generate_mock_report(opponent_name, opponent_patterns, our_players, strategy)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY 환경변수 없음. --mock 모드로 대체합니다.")
        return generate_mock_report(opponent_name, opponent_patterns, our_players, strategy)

    client = anthropic.Anthropic(api_key=api_key)

    # 프롬프트용 데이터 요약
    weaknesses_text = "\n".join(
        f"  - [{w['severity'].upper()}] {w['description']}" for w in opponent_patterns.get("weaknesses", [])
    ) or "  - (발견된 약점 없음)"

    recs_text = "\n".join(
        f"  {i+1}. [{r['phase']}] {r['tactic']}: {r['description']}"
        for i, r in enumerate(strategy.get("recommendations", []))
    )

    our_players_summary = "\n".join(
        f"  - #{p.get('jersey','?')} {p.get('auto_position','?')} "
        f"| 평균속도 {p.get('speed',{}).get('avg_kmh',0):.1f}km/h "
        f"| 스타일: {', '.join(p.get('style_tags',[]) or ['없음'])}"
        for p in our_players[:8]
    )

    prompt = f"""당신은 아이스하키 전문 스카우팅 코치입니다.
다음 분석 데이터를 바탕으로 실전에서 바로 쓸 수 있는 스카우팅 리포트를 작성하세요.

## 상대팀: {opponent_name}
- 주 포메이션: {opponent_patterns.get('primary_formation', 'UNKNOWN')}
- 포체킹 강도: {opponent_patterns.get('forecheck_intensity_pct', 0):.0f}%
- 브레이크아웃 빈도: {opponent_patterns.get('breakout_frequency_pct', 0):.0f}%
- 전환률: {opponent_patterns.get('transition_rate_pct', 0):.0f}%
- 존 점유: {json.dumps(opponent_patterns.get('zone_tendency', {}), ensure_ascii=False)}

## 상대 약점
{weaknesses_text}

## 전략 권고
{recs_text}

## 우리 팀 선수
{our_players_summary}

## 요청
1. 2~3문장으로 상대팀 전반 평가 요약
2. 핵심 전략 3가지 (코치용 지시 어조, 간결하게)
3. 주의할 점 1~2가지

한국어로 작성. 기술적이고 실용적으로."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        ai_text = response.content[0].text.strip()

        return {
            "generated_by": "claude-haiku-4-5",
            "ai_analysis": ai_text,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
        }
    except Exception as e:
        print(f"⚠️  Claude API 오류: {e}")
        return generate_mock_report(opponent_name, opponent_patterns, our_players, strategy)


# ─── 메인 리포트 빌더 ────────────────────────────────────────────────

def build_scouting_report(opponent_data: dict, our_team_data: dict,
                           opponent_name: str = "상대팀",
                           mock: bool = False) -> dict:
    """
    전체 스카우팅 리포트 조립
    """
    # 상대팀 패턴 추출
    opponent_patterns = opponent_data.get("opponent_patterns")
    if not opponent_patterns:
        # opponent_patterns가 없으면 직접 summary에서 빌드
        from tactics_classifier import extract_opponent_patterns
        opponent_patterns = extract_opponent_patterns(opponent_data)

    # 우리 팀 선수 목록
    our_players = our_team_data.get("players", [])

    # 전략 생성
    strategy = build_strategy(opponent_patterns, our_players)

    # AI or 목업 리포트
    if mock:
        ai_section = generate_mock_report(opponent_name, opponent_patterns, our_players, strategy)
    else:
        ai_section = generate_ai_report(opponent_name, opponent_patterns, our_players, strategy)

    return {
        "generated_at": datetime.now().isoformat(),
        "opponent_name": opponent_name,
        "opponent_summary": {
            "primary_formation": opponent_patterns.get("primary_formation"),
            "forecheck_intensity_pct": opponent_patterns.get("forecheck_intensity_pct"),
            "breakout_frequency_pct": opponent_patterns.get("breakout_frequency_pct"),
            "transition_rate_pct": opponent_patterns.get("transition_rate_pct"),
            "weaknesses_count": len(opponent_patterns.get("weaknesses", [])),
            "weaknesses": opponent_patterns.get("weaknesses", []),
        },
        "our_team_summary": {
            "player_count": our_team_data.get("total_players", len(our_players)),
            "video_stem": our_team_data.get("video_stem", ""),
        },
        "strategy": strategy,
        "report": ai_section,
    }


# ─── CLI ──────────────────────────────────────────────────────────────

def print_report(report: dict):
    """터미널 출력 포맷"""
    opp = report["opponent_summary"]
    strat = report["strategy"]
    rep = report["report"]
    advantage = strat.get("matchup_advantage", {})

    print(f"\n{'='*65}")
    print(f"🏒 IceIQ 스카우팅 리포트 — {report['opponent_name']}")
    print(f"   생성: {report['generated_at'][:19]}")
    print(f"{'='*65}")

    print(f"\n📋 상대팀 분석")
    print(f"   주 포메이션:    {opp['primary_formation']}")
    print(f"   포체킹 강도:   {opp['forecheck_intensity_pct']:.0f}%")
    print(f"   브레이크아웃: {opp['breakout_frequency_pct']:.0f}%")
    print(f"   전환률:         {opp['transition_rate_pct']:.0f}%")

    if opp["weaknesses"]:
        print(f"\n⚠️  약점 ({opp['weaknesses_count']}개)")
        for w in opp["weaknesses"]:
            sev = "🔴" if w["severity"] == "high" else "🟡"
            print(f"   {sev} {w['description']}")

    print(f"\n🎯 매치업 우위: {advantage.get('overall', 'N/A')}")
    print(f"   {advantage.get('notes', '')}")
    if advantage.get("our_avg_speed_kmh", 0) > 0:
        print(f"   우리 팀 평균속도: {advantage['our_avg_speed_kmh']} km/h")

    print(f"\n📌 전략 권고")
    for rec in strat.get("recommendations", []):
        players = f" (핵심: {', '.join(rec['key_players'])})" if rec.get("key_players") else ""
        print(f"   [{rec['phase']}] {rec['tactic']}{players}")
        print(f"          {rec['description']}")

    print(f"\n💬 AI 분석 ({rep.get('generated_by', 'unknown')})")
    print(f"   {'-'*55}")
    if rep.get("ai_analysis"):
        for line in rep["ai_analysis"].split("\n"):
            print(f"   {line}")
    elif rep.get("summary"):
        print(f"   {rep['summary']}")
    if rep.get("coaching_notes"):
        print()
        for note in rep["coaching_notes"]:
            print(f"   📍 {note}")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IceIQ Scouting Engine")
    parser.add_argument("opponent", help="상대팀 전술 분석 JSON (tactics_classifier 출력)")
    parser.add_argument("our_team", help="우리 팀 지표 JSON (player_metrics 출력)")
    parser.add_argument("--opponent-name", default="상대팀", help="상대팀 이름 (기본: 상대팀)")
    parser.add_argument("--mock", action="store_true", help="API 없이 목업 리포트 생성")
    parser.add_argument("--output", help="출력 JSON 경로 (생략 시 콘솔만)")
    args = parser.parse_args()

    with open(args.opponent) as f:
        opponent_data = json.load(f)
    with open(args.our_team) as f:
        our_team_data = json.load(f)

    report = build_scouting_report(
        opponent_data=opponent_data,
        our_team_data=our_team_data,
        opponent_name=args.opponent_name,
        mock=args.mock,
    )

    print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"✅ 리포트 저장: {args.output}")
