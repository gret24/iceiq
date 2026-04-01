#!/usr/bin/env python3
"""
roster_manager.py — 3단계 선수 식별 시스템
번호 우선 → 외형 특징 → 행동 패턴
"""
from __future__ import annotations
import math, re
from collections import deque
from dataclasses import dataclass, field
from itertools import permutations
from typing import Optional


# ── 데이터 클래스 ──────────────────────────────────────────

@dataclass
class PlayerProfile:
    jersey_number: str
    team: str
    color_signature: dict = field(default_factory=dict)   # HSV 평균 특징벡터
    skating_signature: dict = field(default_factory=dict) # 스케이팅 패턴
    stick_signature: dict = field(default_factory=dict)   # 스틱 패턴
    _color_count: int = field(default=0, repr=False)      # 누적 평균용

    def update_color(self, new_features: dict):
        """색상 특징 누적 평균 업데이트"""
        if not new_features:
            return
        n = self._color_count
        for k, v in new_features.items():
            if isinstance(v, (int, float)):
                old = self.color_signature.get(k, 0.0)
                self.color_signature[k] = (old * n + v) / (n + 1)
        self._color_count += 1

    def update_behavior(self, skating: dict, stick: dict):
        if skating:
            self.skating_signature.update(skating)
        if stick:
            self.stick_signature.update(stick)


# ── 행동 특징 추출기 ───────────────────────────────────────

class BehaviorExtractor:

    def extract_skating_features(self, bbox_sequence: list[tuple], fps: int = 4) -> dict:
        """최소 5프레임 bbox 시퀀스에서 스케이팅 특징 추출"""
        if len(bbox_sequence) < 5:
            return {}

        cx = [(x1+x2)/2 for x1,y1,x2,y2 in bbox_sequence]
        cy = [(y1+y2)/2 for x1,y1,x2,y2 in bbox_sequence]
        widths  = [x2-x1 for x1,y1,x2,y2 in bbox_sequence]
        heights = [y2-y1 for x1,y1,x2,y2 in bbox_sequence]

        # 보폭 (중심점 이동 거리 평균)
        displacements = [math.hypot(cx[i]-cx[i-1], cy[i]-cy[i-1])
                         for i in range(1, len(bbox_sequence))]
        stride_length = float(sum(displacements) / len(displacements)) if displacements else 0.0

        # 속도 프로필
        speed_profile = [d * fps for d in displacements]

        # 자세 비율 (높이/너비 평균)
        posture_ratio = float(sum(h/w for h, w in zip(heights, widths) if w > 0) / len(heights))

        # 크로스오버 빈도 (너비 변화의 부호 반전 수)
        w_changes = [widths[i] - widths[i-1] for i in range(1, len(widths))]
        crossovers = sum(1 for i in range(1, len(w_changes))
                         if w_changes[i] * w_changes[i-1] < 0)
        duration = len(bbox_sequence) / fps
        crossover_frequency = crossovers / duration if duration > 0 else 0.0

        # 보폭 빈도 (속도 zero-crossing)
        speed_signs = [1 if s > stride_length else -1 for s in speed_profile]
        zc = sum(1 for i in range(1, len(speed_signs)) if speed_signs[i] != speed_signs[i-1])
        stride_frequency = zc / duration if duration > 0 else 0.0

        # 기울기 (x축 편차 표준편차)
        mean_cx = sum(cx) / len(cx)
        body_lean_angle = float(math.sqrt(sum((x - mean_cx)**2 for x in cx) / len(cx)))

        return {
            "stride_length": round(stride_length, 3),
            "stride_frequency": round(stride_frequency, 3),
            "posture_ratio": round(posture_ratio, 3),
            "crossover_frequency": round(crossover_frequency, 3),
            "speed_profile": [round(s, 3) for s in speed_profile[:8]],
            "body_lean_angle": round(body_lean_angle, 3),
        }

    def extract_stick_features(self, bbox_sequence: list[tuple],
                                stick_regions: list[tuple] = None) -> dict:
        """스틱 특징 추출"""
        if not bbox_sequence:
            return {}

        heights = [y2-y1 for x1,y1,x2,y2 in bbox_sequence]
        mean_h = sum(heights) / len(heights) if heights else 1.0

        if stick_regions and len(stick_regions) == len(bbox_sequence):
            # 스틱 영역 기준 특징
            stick_ys = [(sy1+sy2)/2 for _,sy1,_,sy2 in stick_regions]
            bbox_y1s = [y1 for _,y1,_,_ in bbox_sequence]
            rel_positions = [(sy - by) / h for sy, by, h in zip(stick_ys, bbox_y1s, heights) if h > 0]
            stick_position = sum(rel_positions) / len(rel_positions) if rel_positions else 0.5
            pos_changes = [abs(rel_positions[i] - rel_positions[i-1])
                           for i in range(1, len(rel_positions))]
            handling_rhythm = float(sum(pos_changes)/len(pos_changes)) if pos_changes else 0.0
            carry_height = stick_position
            stick_angle = 0.0
        else:
            # bbox만으로 추정
            stick_position = 0.7  # 일반적으로 하단 30% 위치
            handling_rhythm = 0.0
            carry_height = 0.7
            stick_angle = 0.0

        return {
            "stick_position": round(stick_position, 3),
            "stick_angle": round(stick_angle, 3),
            "handling_rhythm": round(handling_rhythm, 3),
            "carry_height": round(carry_height, 3),
        }

    def compute_behavior_similarity(self, sig1: dict, sig2: dict) -> float:
        """가중 행동 유사도 (0~1)"""
        if not sig1 or not sig2:
            return 0.0

        weights = {
            "stride_length":       0.25,
            "posture_ratio":       0.20,
            "speed_profile":       0.20,
            "crossover_frequency": 0.15,
            "stick_position":      0.10,
            "handling_rhythm":     0.10,
        }

        score = 0.0
        total_w = 0.0

        for key, w in weights.items():
            v1 = sig1.get(key)
            v2 = sig2.get(key)
            if v1 is None or v2 is None:
                continue
            total_w += w

            if key == "speed_profile" and isinstance(v1, list) and isinstance(v2, list):
                # 코사인 유사도
                sim = _cosine_similarity(v1, v2)
            elif isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                # 정규화된 절대 차이
                max_val = max(abs(v1), abs(v2), 1e-9)
                sim = 1.0 - min(abs(v1 - v2) / max_val, 1.0)
            else:
                continue
            score += w * sim

        return score / total_w if total_w > 0 else 0.0


# ── 로스터 매니저 ──────────────────────────────────────────

class RosterManager:

    def __init__(self):
        self._profiles: dict[str, PlayerProfile] = {}   # "47_HOME" -> PlayerProfile
        self._track_history: dict[int, str] = {}         # track_id -> jersey_number
        self._bbox_buffer: dict[int, deque] = {}         # track_id -> deque(maxlen=10)
        self._home_roster: list[str] = []
        self._away_roster: list[str] = []
        self._correction_map: dict[tuple, str] = {}      # (4,7) -> "47"
        self._behavior = BehaviorExtractor()

    # ── 로스터 설정 ─────────────────────────────────────────
    def set_roster(self, home: list = None, away: list = None):
        self._home_roster = [str(n).lstrip("0") or "0" for n in (home or [])]
        self._away_roster = [str(n).lstrip("0") or "0" for n in (away or [])]
        self._correction_map = {}
        self._build_correction_map(self._home_roster)
        self._build_correction_map(self._away_roster)

    def _build_correction_map(self, roster: list):
        """번호 분리 인식 보정맵 자동 생성"""
        for num in roster:
            if len(num) < 2:
                continue
            # 모든 분할 방식 생성
            for i in range(1, len(num)):
                parts = (num[:i], num[i:])
                for perm in permutations(parts):
                    key = tuple(int(p) for p in perm if p.isdigit())
                    if key not in self._correction_map:
                        self._correction_map[key] = num

    def roster_match(self, num: str, team: str = None) -> Optional[str]:
        """OCR 번호를 로스터에서 확인"""
        n = num.lstrip("0") or "0"
        roster = self._get_roster(team)
        if n in roster:
            return n
        # 편집거리 1 이내 보정
        for r in roster:
            if _edit_distance(n, r) <= 1:
                return r
        return None

    def _get_roster(self, team: str = None) -> list:
        t = (team or "").upper()
        if t == "HOME" and self._home_roster:
            return self._home_roster
        if t == "AWAY" and self._away_roster:
            return self._away_roster
        return list(set(self._home_roster + self._away_roster))

    def correction_lookup(self, digits: list[int]) -> Optional[str]:
        """분리 감지된 숫자 쌍으로 번호 복원"""
        key = tuple(digits)
        return self._correction_map.get(key)

    # ── 특징 등록 ────────────────────────────────────────────
    def register_features(self, jersey_number: str, team: str,
                           color_features: dict, behavior_features: dict = None):
        key = f"{jersey_number}_{team.upper()}"
        if key not in self._profiles:
            self._profiles[key] = PlayerProfile(jersey_number=jersey_number, team=team)
        p = self._profiles[key]
        p.update_color(color_features)
        if behavior_features:
            sk = {k: v for k, v in behavior_features.items()
                  if k in ("stride_length","stride_frequency","posture_ratio",
                            "crossover_frequency","speed_profile","body_lean_angle")}
            st = {k: v for k, v in behavior_features.items()
                  if k in ("stick_position","stick_angle","handling_rhythm","carry_height")}
            p.update_behavior(sk, st)

    # ── 특징 기반 식별 ───────────────────────────────────────
    def identify_by_features(self, color_features: dict,
                              behavior_features: dict,
                              team: str = None) -> tuple[Optional[str], float]:
        best_num, best_score = None, 0.0
        for key, profile in self._profiles.items():
            if team and not key.endswith(team.upper()):
                continue
            c_sim = _color_similarity(color_features, profile.color_signature)
            b_sig = {**profile.skating_signature, **profile.stick_signature}
            b_sim = self._behavior.compute_behavior_similarity(behavior_features, b_sig) \
                    if behavior_features and b_sig else 0.0
            combined = c_sim * 0.5 + b_sim * 0.5
            if combined > best_score:
                best_score, best_num = combined, profile.jersey_number
        return (best_num, best_score) if best_score > 0.70 else (None, best_score)

    # ── bbox 버퍼 ────────────────────────────────────────────
    def buffer_bbox(self, track_id: int, bbox: tuple, frame_idx: int):
        if track_id not in self._bbox_buffer:
            self._bbox_buffer[track_id] = deque(maxlen=10)
        self._bbox_buffer[track_id].append(bbox)

    # ── 통합 식별 ────────────────────────────────────────────
    def identify_player(self, track_id: int, frame_idx: int,
                         ocr_text: str, ocr_confidence: float,
                         color_features: dict,
                         stick_regions: list = None,
                         team: str = None) -> Optional[str]:
        """
        3단계 식별:
        ① OCR > 0.3 → 로스터 매칭
        ② 보정맵 (분리 인식)
        ③ 색상 유사도 > 0.75
        ④ 행동 유사도 > 0.70 (bbox_buffer >= 5)
        ⑤ 색상+행동 결합 > 0.70
        ⑥ track_id 히스토리
        ⑦ None
        """
        # ① OCR 직접 매칭
        if ocr_confidence > 0.3 and ocr_text:
            matched = self.roster_match(ocr_text, team)
            if matched:
                self.register_features(matched, team or "UNKNOWN", color_features)
                self._track_history[track_id] = matched
                return matched

        # ② 분리 인식 보정 (ocr_text에서 숫자 파싱)
        if ocr_text:
            digits = [int(c) for c in re.findall(r'\d', ocr_text)]
            corrected = self.correction_lookup(digits)
            if corrected:
                self.register_features(corrected, team or "UNKNOWN", color_features)
                self._track_history[track_id] = corrected
                return corrected

        # ③ 색상 유사도
        best_color, best_color_score = None, 0.0
        for key, profile in self._profiles.items():
            if team and not key.endswith(team.upper()):
                continue
            sim = _color_similarity(color_features, profile.color_signature)
            if sim > best_color_score:
                best_color_score, best_color = sim, profile.jersey_number
        if best_color_score > 0.75:
            self._track_history[track_id] = best_color
            return best_color

        # ④⑤ 행동 기반 (bbox_buffer >= 5)
        bboxes = list(self._bbox_buffer.get(track_id, []))
        if len(bboxes) >= 5:
            skating = self._behavior.extract_skating_features(bboxes)
            stick = self._behavior.extract_stick_features(bboxes, stick_regions)
            behavior = {**skating, **stick}

            # ④ 행동만
            best_beh, best_beh_score = None, 0.0
            for key, profile in self._profiles.items():
                if team and not key.endswith(team.upper()):
                    continue
                b_sig = {**profile.skating_signature, **profile.stick_signature}
                sim = self._behavior.compute_behavior_similarity(behavior, b_sig)
                if sim > best_beh_score:
                    best_beh_score, best_beh = sim, profile.jersey_number
            if best_beh_score > 0.70:
                self._track_history[track_id] = best_beh
                return best_beh

            # ⑤ 색상 + 행동 결합
            num, score = self.identify_by_features(color_features, behavior, team)
            if num:
                self._track_history[track_id] = num
                return num

        # ⑥ track_id 히스토리
        if track_id in self._track_history:
            return self._track_history[track_id]

        return None


# ── 유틸리티 ──────────────────────────────────────────────

def _cosine_similarity(a: list, b: list) -> float:
    if not a or not b:
        return 0.0
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb) if na * nb > 0 else 0.0

def _color_similarity(f1: dict, f2: dict) -> float:
    if not f1 or not f2:
        return 0.0
    keys = set(f1) & set(f2)
    if not keys:
        return 0.0
    sims = []
    for k in keys:
        v1, v2 = f1[k], f2[k]
        if isinstance(v1, (int,float)) and isinstance(v2, (int,float)):
            mx = max(abs(v1), abs(v2), 1e-9)
            sims.append(1.0 - min(abs(v1-v2)/mx, 1.0))
    return sum(sims)/len(sims) if sims else 0.0

def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n+1))
    for i in range(1, m+1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n+1):
            dp[j] = prev[j-1] if a[i-1]==b[j-1] else 1+min(prev[j], dp[j-1], prev[j-1])
    return dp[n]


# ── 테스트 ────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    passed = 0
    failed = 0

    def check(name, result, expected):
        global passed, failed
        ok = result == expected
        status = "PASS" if ok else "FAIL"
        if not ok:
            print(f"  [{status}] {name}: got {result!r}, expected {expected!r}")
            failed += 1
        else:
            print(f"  [{status}] {name}")
            passed += 1

    print("=" * 50)
    print("  roster_manager.py 테스트")
    print("=" * 50)

    rm = RosterManager()
    rm.set_roster(home=[2, 4, 14, 26, 47, 94], away=[3, 8, 11, 19, 40, 89])

    # ── Test 1: OCR 직접 매칭 ──────────────────────────────
    print("\n[1] OCR 직접 매칭")
    color = {"h_jersey": 10.0, "s_jersey": 200.0, "v_jersey": 180.0}
    result = rm.identify_player(
        track_id=1, frame_idx=100,
        ocr_text="47", ocr_confidence=0.8,
        color_features=color, team="HOME"
    )
    check("OCR 47번 HOME 매칭", result, "47")

    # ── Test 2: 분리 인식 보정 ─────────────────────────────
    print("\n[2] 분리 인식 보정 [4,7] → 47")
    rm2 = RosterManager()
    rm2.set_roster(home=[2, 4, 14, 26, 47, 94])
    corrected = rm2.correction_lookup([4, 7])
    check("correction_map [4,7] → '47'", corrected, "47")
    corrected2 = rm2.correction_lookup([1, 4])
    check("correction_map [1,4] → '14'", corrected2, "14")

    # ── Test 3: 색상 등록 후 매칭 ─────────────────────────
    print("\n[3] 색상 등록 후 매칭")
    rm3 = RosterManager()
    rm3.set_roster(home=[47])
    ref_color = {"h_j": 120.0, "s_j": 180.0, "v_j": 160.0, "h_h": 30.0}
    rm3.register_features("47", "HOME", ref_color)
    # 약간 다른 색상으로 매칭 시도
    query_color = {"h_j": 118.0, "s_j": 182.0, "v_j": 158.0, "h_h": 31.0}
    result3 = rm3.identify_player(
        track_id=2, frame_idx=200,
        ocr_text="", ocr_confidence=0.0,
        color_features=query_color, team="HOME"
    )
    check("색상 유사도 매칭 → '47'", result3, "47")

    # ── Test 4: 스케이팅 폼 추출 + 행동 유사도 ────────────
    print("\n[4] bbox 시퀀스 스케이팅 특징 추출")
    be = BehaviorExtractor()
    bboxes = [
        (100+i*3, 200+i*2, 140+i*3, 280+i*2)
        for i in range(10)
    ]
    sk_feat = be.extract_skating_features(bboxes, fps=4)
    check("stride_length 존재", "stride_length" in sk_feat, True)
    check("posture_ratio 존재", "posture_ratio" in sk_feat, True)
    check("speed_profile 비어있지 않음", len(sk_feat.get("speed_profile", [])) > 0, True)

    # 동일 시퀀스 유사도 1.0에 가까운지
    sim = be.compute_behavior_similarity(sk_feat, sk_feat)
    check("동일 시그니처 유사도 >= 0.95", sim >= 0.95, True)

    # ── Test 5: identify_player 통합 흐름 (OCR) ───────────
    print("\n[5] identify_player 통합 흐름 (OCR)")
    rm5 = RosterManager()
    rm5.set_roster(home=[4, 14, 47, 94])
    c5 = {"h": 50.0, "s": 150.0, "v": 200.0}
    r5 = rm5.identify_player(
        track_id=10, frame_idx=500,
        ocr_text="94", ocr_confidence=0.6,
        color_features=c5, team="HOME"
    )
    check("OCR '94' HOME 매칭", r5, "94")
    # track_id 히스토리 확인
    r5b = rm5.identify_player(
        track_id=10, frame_idx=501,
        ocr_text="", ocr_confidence=0.0,
        color_features={}, team="HOME"
    )
    check("track_id 히스토리 재사용", r5b, "94")

    # ── Test 6: 색상+행동 결합 매칭 ───────────────────────
    print("\n[6] 색상+행동 결합 매칭")
    rm6 = RosterManager()
    rm6.set_roster(home=[47])
    ref_c6 = {"h_j": 80.0, "s_j": 160.0, "v_j": 140.0}
    # 행동 특징도 등록
    bboxes6 = [(200+i*2, 300, 250+i*2, 400) for i in range(10)]
    sk6 = be.extract_skating_features(bboxes6)
    st6 = be.extract_stick_features(bboxes6)
    rm6.register_features("47", "HOME", ref_c6, {**sk6, **st6})

    # bbox 버퍼에 유사한 시퀀스 추가
    for i, box in enumerate(bboxes6):
        rm6.buffer_bbox(track_id=20, bbox=box, frame_idx=i)

    query_c6 = {"h_j": 82.0, "s_j": 158.0, "v_j": 142.0}
    r6 = rm6.identify_player(
        track_id=20, frame_idx=10,
        ocr_text="", ocr_confidence=0.0,
        color_features=query_c6, team="HOME"
    )
    check("색상+행동 결합 매칭 → '47'", r6, "47")

    # ── 결과 ──────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  결과: {passed}개 통과 / {passed+failed}개 테스트")
    if failed == 0:
        print("  ✅ 모든 테스트 통과!")
    else:
        print(f"  ❌ {failed}개 실패")
    print("="*50)
    sys.exit(0 if failed == 0 else 1)
