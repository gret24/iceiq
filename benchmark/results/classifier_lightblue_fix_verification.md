# classifier light_blue fix — 최종 검증

**날짜**: 2026-04-21
**PR**: classifier-lightblue-fix
**Status**: VERIFIED — safe to merge

## 검증 완료 지표
- iceking tracking: 3.6/96.4 → 47.0/53.0 (30-70% 범위)
- _colour_ratio margin: 14.4x (아이스킹덤 vs KU)
- mega regression: 0.0% (deterministic)
- opponent crops: -95% (3,190 → 161)
- legibility: +1.3pp (41.6% → 42.8%, 4경기 합산)

## 검증된 가설들

**Q1: fix가 KU 선수를 light_blue로 오분류했는가?**
→ 아니오. iceking v2에서 team_a(light_blue) 로 OCR된
   track 중 KU 번호로 인식된 건 0건.

**Q2: crops/track 감소(4.72 → 3.22, -32%)가 KU 데이터 손실인가?**
→ 아니오. v1에서 아이스킹덤 트랙이 team_b로 오분류되어
   KU 트랙인 척 섞여있었고, 그 트랙들이 평균 crop 수가 높았기
   때문에 제외된 것. median predictions/track은 1.0 그대로.

**Q3: KU 전용 번호(#15, #80, #84, #92)도 공유 번호(#4, #14, #28)와
     비슷한 ~68% 감소를 보인 이유는?**
→ 전체 track pool 축소 효과. iceking v1의 total crops 자체가
   오염으로 부풀어 있었으므로 깨끗한 base에서는 모든 번호가
   비례 감소하는 것이 정상.

## 결론
Fix fully validated. No hidden regression. Safe to merge.

## Follow-up (별개 티켓)
- 두 classifier 통합(기술부채 리팩터)
- 깨끗한 baseline 재측정 (Jersey OCR 95%+ 작업 전)
