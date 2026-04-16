"""
IceIQ Growth System
여러 게임의 player_metrics → 선수 성장 추적 + AI 피드백

입력:
  game1_metrics.json, game2_metrics.json, ... (player_metrics.py 출력)

출력:
  성장 트렌드 + AI 코칭 피드백

사용법:
  # 목업 테스트
  python3 growth_system.py --games game1.json game2.json --jersey 47 --mock

  # 실제 AI 분석 (ANTHROPIC_API_KEY 필요)
  python3 growth_system.py --games game1.json game2.json --jersey 47
"""

import json
import os
import sys
from datetime import datetime
from typing import Any


# ─── 선수 데이터 추출 ─────────────────────────────────────────────────

def find_player(metrics: dict, jersey: str) -> dict | None:
    """metrics JSON에서 jersey 번호로 선수 찾기"""
    for p in metrics.get("players", []):
        if str(p.get("jersey", "")) == str(jersey):
            return p
    return None


def extract_snapshot(player: dict, game_index: int, game_label: str) -> dict:
    """선수 데이터 → 성장 추적용 스냅샷"""
    spd = player.get("speed", {})
    dist = player.get("distance", {})
    trans = player.get("transition", {})
    pos = player.get("positioning", {})
    stam = player.get("stamina", {})

    return {
        "game": game_index + 1,
        "label": game_label,
        # 속도
        "avg_speed_kmh": spd.get("avg_kmh", 0),
        "max_speed_kmh": spd.get("max_kmh", 0),
        "sprint_count": spd.get("sprint_count", 0),
        # 거리
        "total_km": dist.get("total_km", 0),
        "avg_shift_km": dist.get("avg_per_shift_km", 0),
        # 체력
        "ice_time_sec": player.get("total_ice_time_sec", 0),
        "total_shifts": player.get("total_shifts", 0),
        "stamina_decay_pct": stam.get("avg_decay_pct", 0),
        # 전환
        "backcheck_sec": trans.get("avg_backcheck_sec", 0),
        "rush_sec": trans.get("avg_rush_sec", 0),
        # 포지셔닝
        "oz_pct": pos.get("zone_pct", {}).get("OZ", 0),
        "nz_pct": pos.get("zone_pct", {}).get("NZ", 0),
        "dz_pct": pos.get("zone_pct", {}).get("DZ", 0),
        "high_slot_pct": pos.get("high_slot_pct", 0),
        # 포지션/스타일
        "position": player.get("auto_position", "?"),
        "style_tags": player.get("style_tags", []),
    }


# ─── 성장 분석 ────────────────────────────────────────────────────────

def compute_growth(snapshots: list[dict]) -> dict:
    """스냅샷 리스트 → 게임별 변화 & 트렌드"""
    if len(snapshots) < 2:
        return {"trend": "데이터 부족 (게임 2개 이상 필요)", "deltas": []}

    metrics_to_track = [
        ("avg_speed_kmh", "평균속도(km/h)", "higher"),
        ("max_speed_kmh", "최고속도(km/h)", "higher"),
        ("sprint_count", "스프린트 횟수", "higher"),
        ("total_km", "총 이동거리(km)", "higher"),
        ("stamina_decay_pct", "체력 감소율(%)", "lower"),
        ("backcheck_sec", "백체크 시간(초)", "lower"),
        ("high_slot_pct", "하이슬롯 점유(%)", "higher"),
        ("oz_pct", "공격존 체류(%)", "higher"),
    ]

    deltas = []
    for key, label, direction in metrics_to_track:
        values = [s[key] for s in snapshots]
        if len(values) >= 2:
            first = values[0]
            last = values[-1]
            change = last - first
            change_pct = (change / first * 100) if first != 0 else 0

            if direction == "higher":
                improved = change > 0
            else:
                improved = change < 0

            deltas.append({
                "metric": key,
                "label": label,
                "direction": direction,
                "values": values,
                "change": round(change, 2),
                "change_pct": round(change_pct, 1),
                "improved": improved,
                "trend": "📈" if improved else ("📉" if not improved and change != 0 else "➡️"),
            })

    # 전체 트렌드 평가
    improved_count = sum(1 for d in deltas if d["improved"])
    total = len(deltas)
    if improved_count >= total * 0.7:
        overall = "상승세 🟢"
    elif improved_count >= total * 0.4:
        overall = "혼조세 🟡"
    else:
        overall = "하락세 🔴"

    return {
        "overall_trend": overall,
        "improved_count": improved_count,
        "total_metrics": total,
        "deltas": deltas,
        "snapshots": snapshots,
    }


# ─── 목업 피드백 ──────────────────────────────────────────────────────

def generate_mock_feedback(jersey: str, growth: dict, snapshots: list) -> dict:
    """AI 없이 규칙 기반 피드백 생성"""
    deltas = growth.get("deltas", [])
    overall = growth.get("overall_trend", "N/A")

    improved = [d for d in deltas if d["improved"]]
    regressed = [d for d in deltas if not d["improved"] and d["change"] != 0]

    strengths = [d["label"] for d in improved[:3]]
    concerns = [d["label"] for d in regressed[:2]]

    coaching = []
    for d in regressed[:3]:
        if d["metric"] == "stamina_decay_pct":
            coaching.append("체력 감소 심화 — 인터벌 트레이닝 추가 권장")
        elif d["metric"] == "avg_speed_kmh":
            coaching.append("평균속도 감소 — 스케이팅 드릴 집중")
        elif d["metric"] == "sprint_count":
              coaching.append("스프린트 감소 — 단거리 폭발력 훈련 필요")
        elif d["metric"] == "high_slot_pct":
            coaching.append("슬롯 점유 감소 — 공격 포지셔닝 재교육")
        elif d["metric"] == "backcheck_sec":
            coaching.append("백체크 시간 증가 — 수비 복귀 의식 강화")
        elif d["metric"] == "oz_pct":
            coaching.append("공격존 체류 감소 — 공격 가담 독려")

    summary = (
        f"#{jersey} 선수 {len(snapshots)}경기 분석 결과 전체 트렌드: {overall}. "
        f"개선 항목 {growth['improved_count']}/{growth['total_metrics']}개. "
    )
    if strengths:
        summary += f"강점: {', '.join(strengths)}. "
    if concerns:
        summary += f"주의: {', '.join(concerns)}."

    return {
        "generated_by": "mock",
        "summary": summary,
        "strengths": strengths,
        "concerns": concerns,
        "coaching_points": coaching,
    }


# ─── Claude AI 피드백 ────────────────────────────────────────────────

def generate_ai_feedback(jersey: str, growth: dict, snapshots: list) -> dict:
    """Claude API로 성장 피드백 생성"""
    try:
        import anthropic
    except ImportError:
        print("⚠️  anthropic 패키지 없음. pip install anthropic")
        return generate_mock_feedback(jersey, growth, snapshots)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY 환경변수 없음. --mock 모드로 대체합니다.")
        return generate_mock_feedback(jersey, growth, snapshots)

    client = anthropic.Anthropic(api_key=api_key)

    # 스냅샷 요약
    snap_text = ""
    for s in snapshots:
        snap_text += (
            f"\n  게임 {s['game']} ({s['label']}):\n"
            f"    평균속도 {s['avg_speed_kmh']}km/h | 최고속도 {s['max_speed_kmh']}km/h | "
            f"스프린트 {s['sprint_count']}회\n"
            f"    이동거리 {s['total_km']}km | 아이스타임 {round(s['ice_time_sec']/60,1)}분 | "
            f"시프트 {s['total_shifts']}회\n"
            f"    체력감소 {s['stamina_decay_pct']}% | 백체크 {s['backcheck_sec']}초\n"
            f"    존 점유 OZ {s['oz_pct']}% / NZ {s['nz_pct']}% / DZ {s['dz_pct']}% | "
            f"하이슬롯 {s['high_slot_pct']}%\n"
            f"    스타일: {', '.join(s['style_tags']) or '없음'}\n"
        )

    # 변화 요약
    delta_text = ""
    for d in growth.get("deltas", []):
        vals = " → ".join(str(v) for v in d["values"])
        delta_text += f"  {d['trend']} {d['label']}: {vals} (변화: {d['change']:+.2f})\n"

    overall = growth.get("overall_trend", "N/A")

    prompt = f"""당신은 아이스하키 선수 개발 전문 코치입니다.
다음 데이터를 바탕으로 #{jersey} 선수의 성장 분석 리포트를 작성하세요.

## 선수: #{jersey}
## 전체 트렌드: {overall} ({growth['improved_count']}/{growth['total_metrics']} 항목 개선)

## 게임별 스탯
{snap_text}

## 주요 변화
{delta_text}

## 요청
1. 선수 성장/퇴보 전반 평가 (2~3문장)
2. 잘하고 있는 점 2가지 (구체적 수치 언급)
3. 개선이 필요한 점 2가지 + 각각 구체적 훈련 방법 1줄
4. 다음 경기 집중 목표 1가지

한국어로. 코치가 선수에게 직접 말하는 어조로. 따뜻하지만 날카롭게."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        ai_text = response.content[0].text.strip()

        return {
            "generated_by": "claude-haiku-4-5",
            "ai_feedback": ai_text,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
        }
    except Exception as e:
        print(f"⚠️  Claude API 오류: {e}")
        return generate_mock_feedback(jersey, growth, snapshots)


# ─── 출력 포맷 ────────────────────────────────────────────────────────

def print_growth_report(jersey: str, growth: dict, feedback: dict, snapshots: list):
    print(f"\n{'='*65}")
    print(f"📈 IceIQ 성장 리포트 — #{jersey}")
    print(f"   분석: {len(snapshots)}경기 | 트렌드: {growth['overall_trend']}")
    print(f"   개선: {growth['improved_count']}/{growth['total_metrics']} 항목")
    print(f"{'='*65}")

    print(f"\n📊 게임별 핵심 지표")
    header = f"  {'항목':<20}" + "".join(f"  G{s['game']:<5}" for s in snapshots)
    print(header)
    print(f"  {'-'*60}")
    for d in growth.get("deltas", []):
        vals = "".join(f"  {v:<7}" for v in d["values"])
        print(f"  {d['trend']} {d['label']:<18}{vals}")

    print(f"\n💬 AI 피드백 ({feedback.get('generated_by', 'unknown')})")
    print(f"   {'-'*55}")
    if feedback.get("ai_feedback"):
        for line in feedback["ai_feedback"].split("\n"):
            print(f"   {line}")
    else:
        print(f"   {feedback.get('summary', '')}")
        if feedback.get("strengths"):
            print(f"\n   ✅ 강점: {', '.join(feedback['strengths'])}")
        if feedback.get("concerns"):
            print(f"   ⚠️  주의: {', '.join(feedback['concerns'])}")
        if feedback.get("coaching_points"):
            print(f"\n   📌 코칭 포인트:")
            for cp in feedback["coaching_points"]:
                print(f"      - {cp}")
    print()


# ─── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IceIQ Growth System")
    parser.add_argument("--games", nargs="+", required=True,
                        help="게임별 metrics JSON 경로 (시간 순서대로)")
    parser.add_argument("--jersey", required=True,
                        help="분석할 선수 jersey 번호")
    parser.add_argument("--mock", action="store_true",
                        help="API 없이 목업 피드백 생성")
    parser.add_argument("--output", help="출력 JSON 경로 (생략 시 콘솔만)")
    parser.add_argument("--labels", nargs="+",
                        help="게임 레이블 (예: 'vs팀A' 'vs팀B')")
    args = parser.parse_args()

    # 게임 데이터 로드
    snapshots = []
    not_found_games = []
    for i, game_path in enumerate(args.games):
        label = args.labels[i] if args.labels and i < len(args.labels) else os.path.basename(game_path)
        with open(game_path) as f:
            metrics = json.load(f)

        player = find_player(metrics, args.jersey)
        if player is None:
            print(f"⚠️  게임 {i+1} ({game_path}): #{args.jersey} 선수 없음 — 스킵")
            not_found_games.append(label)
            continue

        snap = extract_snapshot(player, i, label)
        snapshots.append(snap)
        print(f"✅ 게임 {i+1} ({label}): #{args.jersey} 데이터 로드")

    if len(snapshots) < 1:
        print(f"❌ #{args.jersey} 선수 데이터가 없습니다.")
        sys.exit(1)

    if len(snapshots) < 2:
        print(f"⚠️  게임 1개만 있어 트렌드 분석 생략 (단일 게임 스냅샷만 표시)")

    # 성장 분석
    growth = compute_growth(snapshots)

    # 피드백 생성
    if args.mock or len(snapshots) < 2:
        feedback = generate_mock_feedback(args.jersey, growth, snapshots)
    else:
        feedback = generate_ai_feedback(args.jersey, growth, snapshots)

    # 출력
    print_growth_report(args.jersey, growth, feedback, snapshots)

    # JSON 저장
    if args.output:
        result = {
            "generated_at": datetime.now().isoformat(),
            "jersey": args.jersey,
            "games_analyzed": len(snapshots),
            "growth": growth,
            "feedback": feedback,
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"✅ 리포트 저장: {args.output}")
