#!/usr/bin/env python3
"""
roster.py — 로스터 사전 등록 시스템
- 경기 전 팀별 번호 목록 등록
- OCR 결과를 로스터와 매칭하여 보정
- 로스터에 없는 번호 필터링
- "4+7" → "47" 같은 분리 감지 복원

사용법:
  from roster import Roster
  r = Roster(home=[2,4,14,26,47,94], away=[3,7,9,19,40,89])
  corrected = r.correct("4")   # → "47" (가장 가까운 로스터 번호)
  r.save("aigis_g18_roster.json")
  r2 = Roster.load("aigis_g18_roster.json")
"""

import json, os
from difflib import SequenceMatcher


class Roster:
    def __init__(self, home: list = None, away: list = None, both: list = None):
        """
        home: HOME팀 번호 목록
        away: AWAY팀 번호 목록
        both: 팀 구분 없이 전체 번호 (home/away 미입력 시 사용)
        """
        self.home = [str(n).lstrip("0") or "0" for n in (home or [])]
        self.away = [str(n).lstrip("0") or "0" for n in (away or [])]
        self.both = [str(n).lstrip("0") or "0" for n in (both or [])]
        self._all = list(set(self.home + self.away + self.both))

    def all_numbers(self, team: str = None) -> list:
        """팀별 또는 전체 번호 목록"""
        t = (team or "").upper()
        if t == "HOME" and self.home:
            return self.home
        if t == "AWAY" and self.away:
            return self.away
        return self._all

    def is_valid(self, num: str, team: str = None) -> bool:
        """번호가 로스터에 있는지 확인"""
        n = str(num).lstrip("0") or "0"
        pool = self.all_numbers(team)
        if not pool:
            return True   # 로스터 없으면 통과
        return n in pool

    def correct(self, num: str, team: str = None, max_dist: int = 2) -> str | None:
        """
        OCR 번호를 로스터에서 가장 가까운 번호로 보정
        max_dist: 허용 편집 거리 (기본 2)
        반환: 보정된 번호 or None (너무 다를 경우)
        """
        n = str(num).lstrip("0") or "0"
        pool = self.all_numbers(team)
        if not pool:
            return n
        if n in pool:
            return n

        # 편집 거리 기반 최근접 매칭
        best, best_score = None, -1
        for candidate in pool:
            # 숫자 포함 관계 체크 (4 ⊂ 47, 7 ⊂ 47)
            if n in candidate or candidate in n:
                score = len(n) / len(candidate) if len(candidate) >= len(n) else len(candidate) / len(n)
                if score > best_score:
                    best, best_score = candidate, score
            # 시퀀스 매칭
            ratio = SequenceMatcher(None, n, candidate).ratio()
            if ratio > best_score:
                best, best_score = candidate, ratio

        # 편집거리 계산
        if best:
            dist = _edit_distance(n, best)
            if dist <= max_dist:
                return best
        return None

    def correct_merged(self, parts: list[str], team: str = None) -> str | None:
        """
        분리 감지된 숫자들을 합쳐서 로스터 매칭
        예) ["4", "7"] → "47" (로스터에 있으면)
        예) ["1", "4"] → "14"
        """
        pool = self.all_numbers(team)
        if not pool:
            merged = "".join(parts)
            return merged if merged.isdigit() and 1 <= int(merged) <= 99 else None

        # 가능한 조합 생성 (순서 포함)
        from itertools import permutations
        candidates = []
        for r in range(1, len(parts)+1):
            for perm in permutations(parts, r):
                merged = "".join(perm)
                if merged.isdigit() and 1 <= int(merged) <= 99:
                    candidates.append(merged)

        # 로스터에 있는 것 우선
        for c in candidates:
            n = c.lstrip("0") or "0"
            if n in pool:
                return n
        return None

    def filter_ocr_results(self, ocr_results: list, team: str = None) -> list:
        """
        OCR 결과 리스트 필터링 + 보정
        ocr_results: [(번호, confidence), ...]
        반환: [(보정된_번호, confidence, 원본번호), ...]
        """
        filtered = []
        for (num, conf) in ocr_results:
            corrected = self.correct(num, team)
            if corrected:
                filtered.append((corrected, conf, num))
        return filtered

    def save(self, path: str):
        data = {"home": self.home, "away": self.away, "both": self.both}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"로스터 저장: {path}")

    @classmethod
    def load(cls, path: str) -> "Roster":
        with open(path) as f:
            d = json.load(f)
        return cls(home=d.get("home",[]), away=d.get("away",[]), both=d.get("both",[]))

    def __repr__(self):
        return f"Roster(home={self.home}, away={self.away}, both={self.both})"


def _edit_distance(a: str, b: str) -> int:
    """레벤슈타인 편집 거리"""
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i][j] = dp[i-1][j-1] if a[i-1]==b[j-1] else \
                        1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


# ── 사전 등록된 로스터 (JSON 없으면 빈 상태) ─────────────────
_DEFAULT_ROSTER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roster.json")

def get_roster(path: str = None) -> Roster:
    """저장된 로스터 로드 (없으면 빈 Roster)"""
    p = path or _DEFAULT_ROSTER_PATH
    if os.path.exists(p):
        return Roster.load(p)
    return Roster()


if __name__ == "__main__":
    # 테스트
    r = Roster(home=[2,4,14,26,47,94], away=[3,7,9,19,40,89])
    print(r)
    print("'4' 보정:", r.correct("4", "HOME"))         # → 4 (로스터에 있음)
    print("'7' 보정:", r.correct("7", "HOME"))         # → None (HOME에 없음) or 47
    print("['4','7'] 합치기:", r.correct_merged(["4","7"], "HOME"))  # → 47
    print("'1' HOME 보정:", r.correct("1", "HOME"))    # → 14 (가장 가까운)
    print("'9' HOME 보정:", r.correct("9", "HOME"))    # → 94
    print("유효성 '47' HOME:", r.is_valid("47", "HOME"))  # → True
    print("유효성 '99' HOME:", r.is_valid("99", "HOME"))  # → False
