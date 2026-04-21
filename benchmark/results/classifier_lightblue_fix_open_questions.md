# Open Questions (post-merge investigation)

## Crops/track 감소 불일치
- tracks: -54%
- our crops: -68%
- 예상: 비례해야 정상. 14%p 차이는 KU 선수 일부가
  light_blue로 재분류됐을 가능성 시사.

## KU 전용 번호도 공유 번호와 동일 감소율
- #15 최연 -68%, #80 -76%, #84 -70%, #92 -74%
- 공유 번호(#4, #14, #28, #47)와 거의 같은 감소율
- 가설 A: 전체 track pool 축소 효과 (Claude Code 주장)
- 가설 B: KU 선수가 light_blue team_a로 일부 오분류

## 검증 방법
- iceking v2 cache에서 team_a로 분류된 track 중
  KU 로스터 번호로 OCR된 track 수 집계
- 0이면 가설 A, 상당수면 가설 B
- 가설 B면 S_min=60이 너무 헐거운 것, S_min=70 시도

## 우선순위
MED — merge는 진행하되, Jersey OCR 95% 작업 전 재검토
