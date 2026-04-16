"""
팀 분류기 검증 스크립트
라벨링된 roster_waves_g18.json의 crop 이미지로 현재 팀 분류기 정확도 측정
"""
import json, cv2, sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.team_classification import HockeyTeamClassifier, PRESET_AIGIS_LOPEZ

def run():
    roster = json.load(open('data/roster_waves_g18.json'))
    clf = HockeyTeamClassifier(*PRESET_AIGIS_LOPEZ, vote_window=1)

    results = {'aigis': {'correct': 0, 'total': 0},
               'lopez': {'correct': 0, 'total': 0}}
    errors = []

    for idx, p in enumerate(roster['players']):
        team = p.get('team_label')
        if team not in ('aigis', 'lopez'):
            continue
        path = p.get('image_path', '')
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        bbox = [0, 0, w, h]
        # 단일 프레임용: track_id를 고유하게
        track_id = idx * 1000
        pred_label, conf = clf.classify(img, bbox, track_id)

        results[team]['total'] += 1
        if pred_label == team:
            results[team]['correct'] += 1
        else:
            errors.append({
                'crop_id': p['crop_id'],
                'true': team,
                'pred': pred_label,
                'conf': round(conf, 3)
            })

    print('=== 팀 분류기 검증 결과 (단일 프레임, 투표 없음) ===')
    for team, r in results.items():
        if r['total'] == 0:
            continue
        acc = r['correct'] / r['total'] * 100
        print(f'  {team:8}: {r["correct"]:>3}/{r["total"]:>3} = {acc:.1f}%')

    total_c = sum(r['correct'] for r in results.values())
    total_t = sum(r['total'] for r in results.values())
    if total_t:
        print(f'  {"전체":8}: {total_c:>3}/{total_t:>3} = {total_c/total_t*100:.1f}%')

    if errors:
        print(f'\n오분류 {len(errors)}개 (상위 10개):')
        for e in errors[:10]:
            print(f'  {e["crop_id"]:20} true={e["true"]:8} pred={e["pred"]:8} conf={e["conf"]}')

if __name__ == '__main__':
    run()
