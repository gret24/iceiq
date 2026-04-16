"""
IceIQ Command Handler
Usage:
    python iceiq_commands.py /compare <video_path>
    python iceiq_commands.py /retrain_team
    python iceiq_commands.py /retrain_jersey <team> <number> <crop_folder>
    python iceiq_commands.py /retest
"""
import sys, os, json, cv2, numpy as np, torch, torch.nn as nn, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import warnings; warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR   = os.path.join(BASE, 'models')
DATA_DIR     = os.path.join(BASE, 'data')
RESULTS_DIR  = os.path.join(DATA_DIR, 'results')
ROSTER_PATH  = os.path.join(DATA_DIR, 'roster_waves_g18.json')
DEVICE       = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
TF = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

AIGIS_NAMES = {4:'윤지성',11:'박리오',12:'이준표',14:'이봄',25:'김재원',
               42:'남다름',47:'한승원',61:'김승후',94:'김태윤'}


# ═══════════════════════════════════════════════════════════════════════════════
# 공통 유틸
# ═══════════════════════════════════════════════════════════════════════════════

def _prep(img, size=96):
    h,w = img.shape[:2]
    t = img[int(h*0.15):int(h*0.70), int(w*0.10):int(w*0.90)]
    if t.size==0: t=img
    return cv2.resize(t,(size,size),interpolation=cv2.INTER_CUBIC)

def _augment(img, n=30):
    imgs=[]; h,w=img.shape[:2]
    for _ in range(n):
        a=np.clip(img.astype(float)*random.uniform(0.65,1.35)+random.randint(-30,30),0,255).astype(np.uint8)
        if random.random()<0.5: a=cv2.GaussianBlur(a,(3,3),0)
        if random.random()<0.5: a=cv2.flip(a,1)
        M=cv2.getRotationMatrix2D((w//2,h//2),random.uniform(-12,12),1.0)
        a=cv2.warpAffine(a,M,(w,h)); imgs.append(a)
    return imgs

def _load_model(path, n_classes):
    m = models.mobilenet_v3_small()
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt['model_state'])
    return m.to(DEVICE).eval(), ckpt

def _train(samples, n_classes, epochs=80, label=None):
    """samples: [(img_bgr, label_int), ...]"""
    class DS(Dataset):
        def __init__(self, s):
            self.s = s
        def __len__(self): return len(self.s)
        def __getitem__(self,i):
            img,lbl = self.s[i]
            return TF(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)), lbl

    loader = DataLoader(DS(samples), batch_size=32, shuffle=True)
    model  = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, n_classes)
    model  = model.to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit   = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        ls,correct,total=0,0,0
        for imgs,labels in loader:
            imgs,labels=imgs.to(DEVICE),labels.to(DEVICE)
            opt.zero_grad(); out=model(imgs)
            loss=crit(out,labels); loss.backward(); opt.step()
            ls+=loss.item(); correct+=(out.argmax(1)==labels).sum().item(); total+=len(labels)
        sched.step()
        if (ep+1)%(epochs//4)==0 and label:
            print(f'  [{label}] ep{ep+1} acc={correct/total*100:.1f}%')
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# /retrain_team — HSV → CNN 팀 분류기 교체
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_retrain_team():
    print('=== /retrain_team: 팀 분류 CNN 재학습 ===')
    roster = json.load(open(ROSTER_PATH))
    team_data = defaultdict(list)

    # 라벨된 crop 로드
    for p in roster['players']:
        team = p.get('team_label')
        path = p.get('image_path','')
        if team not in ('aigis','hockey_machine') or not os.path.exists(path): continue
        img = cv2.imread(path)
        if img is None: continue
        team_data[team].append(img)

    # diag crop 추가 (아이기스)
    diag_dir = '/tmp/diag_47'
    if os.path.exists(diag_dir):
        for f in os.listdir(diag_dir):
            img = cv2.imread(os.path.join(diag_dir,f))
            if img is not None: team_data['aigis'].append(img)

    label_map = {'aigis':0, 'hockey_machine':1}
    aug_n     = {'aigis':5, 'hockey_machine':25}

    samples = []
    for team, imgs in team_data.items():
        lbl = label_map[team]
        for img in imgs:
            p = _prep(img)
            samples.append((p, lbl))
            for a in _augment(p, aug_n[team]):
                samples.append((a, lbl))

    n0 = sum(1 for _,l in samples if l==0)
    n1 = sum(1 for _,l in samples if l==1)
    print(f'샘플: 아이기스 {n0} | 하키머신 {n1}')

    model = _train(samples, 2, epochs=60, label='team')
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, 'team_classifier_cnn.pt')
    torch.save({'model_state':model.state_dict(),
                'label_map':label_map,
                'idx_to_label':{0:'aigis',1:'hockey_machine'},
                'n_classes':2}, save_path)
    print(f'저장 → {save_path}')
    return f'✅ 팀 분류 CNN 재학습 완료 (아이기스 {n0} | 하키머신 {n1})'


# ═══════════════════════════════════════════════════════════════════════════════
# /retrain_jersey <team> <number> <crop_folder> — 번호 classifier 업데이트
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_retrain_jersey(team, number, crop_folder):
    number = int(number)
    print(f'=== /retrain_jersey {team} #{number} {crop_folder} ===')

    # 기존 모델 로드
    model_path = os.path.join(MODELS_DIR, f'jersey_classifier_{team}.pt')
    if not os.path.exists(model_path):
        return f'❌ 모델 없음: {model_path}'

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    idx_to_num = ckpt['idx_to_num']
    existing_nums = sorted(idx_to_num.values())

    # 새 번호 추가
    if number not in existing_nums:
        existing_nums = sorted(existing_nums + [number])
        print(f'새 클래스 추가: #{number} → 총 {len(existing_nums)}클래스')
    else:
        print(f'기존 클래스 #{number} 데이터 추가')

    new_label_map  = {n:i for i,n in enumerate(existing_nums)}
    new_idx_to_num = {i:n for n,i in new_label_map.items()}

    # 기존 roster crop 로드
    roster = json.load(open(ROSTER_PATH))
    data_by_num = defaultdict(list)
    for p in roster['players']:
        if p.get('team_label') != team: continue
        num = p.get('jersey_number')
        path = p.get('image_path','')
        if num not in existing_nums or not os.path.exists(path): continue
        img = cv2.imread(path)
        if img is not None: data_by_num[num].append(img)

    # 새 crop 추가
    new_crops = []
    if os.path.exists(crop_folder):
        for f in os.listdir(crop_folder):
            img = cv2.imread(os.path.join(crop_folder,f))
            if img is not None:
                data_by_num[number].append(img)
                new_crops.append(f)

    print(f'새 crop: {len(new_crops)}개')
    for n in sorted(data_by_num): print(f'  #{n}: {len(data_by_num[n])}개')

    # 학습 샘플 구성
    samples = []
    for num, imgs in data_by_num.items():
        lbl = new_label_map[num]
        aug_n = max(10, 60//max(len(imgs),1))
        for img in imgs:
            p = _prep(img)
            samples.append((p, lbl))
            for a in _augment(p, aug_n): samples.append((a, lbl))

    model = _train(samples, len(existing_nums), epochs=80, label=f'{team}_jersey')

    torch.save({'model_state':model.state_dict(),
                'label_map':new_label_map,
                'idx_to_num':new_idx_to_num,
                'n_classes':len(existing_nums)}, model_path)
    print(f'저장 → {model_path}')

    # roster에 새 crop 기록
    for f in new_crops:
        path = os.path.join(crop_folder, f)
        roster['players'].append({
            'crop_id': f'retrain_{team}_{number}_{f}',
            'image_path': path,
            'team_label': team,
            'player_name': AIGIS_NAMES.get(number) if team=='aigis' else None,
            'jersey_number': number,
            'role': 'player',
        })
    json.dump(roster, open(ROSTER_PATH,'w'), ensure_ascii=False, indent=2)

    return f'✅ {team} #{number} classifier 재학습 완료 (클래스: {existing_nums})'


# ═══════════════════════════════════════════════════════════════════════════════
# /compare <video_path> — OCR vs Classifier 비교
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_compare(video_path):
    from scripts.compare_pipeline import ComparisonPipeline, format_report
    roi = os.path.join(DATA_DIR, 'roi_test_game.json')
    roi = roi if os.path.exists(roi) else None
    roster = os.path.join(DATA_DIR, 'rosters', 'aigis_test_game.yaml')
    roster = roster if os.path.exists(roster) else None
    pipe = ComparisonPipeline(video_path, roster_path=roster, fps_sample=2, roi_path=roi)
    report = pipe.run()
    out = os.path.join(RESULTS_DIR, 'last_compare.json')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(report, open(out,'w'), ensure_ascii=False, indent=2)
    return format_report(report)


# ═══════════════════════════════════════════════════════════════════════════════
# /retest — 전체 파이프라인 재실행 + 이전 결과 비교
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_retest(video_path=None):
    if not video_path:
        video_path = os.path.join(DATA_DIR, 'videos', 'test_game.mp4')

    print(f'=== /retest: {os.path.basename(video_path)} ===')

    # 이전 결과 로드
    prev_path = os.path.join(RESULTS_DIR, 'test_game', 'timeline_clf.json')
    prev_stats = {}
    if os.path.exists(prev_path):
        prev = json.load(open(prev_path))
        for seg in prev:
            for n in seg.get('aigis',[]): prev_stats[f'aigis_{n}'] = prev_stats.get(f'aigis_{n}',0)+1
            for n in seg.get('hm',[]):    prev_stats[f'hm_{n}']    = prev_stats.get(f'hm_{n}',0)+1

    # 현재 분석
    from pipeline.detection import YOLODetector
    from pipeline.jersey_classifier import JerseyClassifier

    detector  = YOLODetector(model_path='yolov8m.pt', device='mps')
    jersey_clf = JerseyClassifier(min_conf=0.55)

    # CNN 팀 분류기
    team_model_path = os.path.join(MODELS_DIR, 'team_classifier_cnn.pt')
    team_ckpt = torch.load(team_model_path, map_location=DEVICE, weights_only=False)
    tm = models.mobilenet_v3_small()
    tm.classifier[3] = nn.Linear(tm.classifier[3].in_features, 2)
    tm.load_state_dict(team_ckpt['model_state'])
    tm = tm.to(DEVICE).eval()
    IDX_TO_TEAM = {0:'aigis',1:'hockey_machine'}

    def classify_team(crop):
        h,w=crop.shape[:2]
        t=crop[int(h*0.15):int(h*0.70),int(w*0.10):int(w*0.90)]
        if t.size==0: t=crop
        t=cv2.resize(t,(96,96),interpolation=cv2.INTER_CUBIC)
        x=TF(cv2.cvtColor(t,cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs=torch.softmax(tm(x),1)[0]
            idx=probs.argmax().item()
        return IDX_TO_TEAM[idx], probs[idx].item()

    roi_path = os.path.join(DATA_DIR, 'roi_test_game.json')
    roi_pts = None
    if os.path.exists(roi_path):
        rd = json.load(open(roi_path))
        roi_pts = np.array(list(rd.values())[0], dtype=np.int32)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    STRIDE = int(fps*30)
    cur_stats = Counter()
    timeline  = []
    t0 = time.time()

    for frame_idx in range(0,total,STRIDE):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_idx); ret,frame=cap.read()
        if not ret: continue
        aigis_n,hm_n=[],[]
        for i,det in enumerate(detector.detect(frame)):
            if det['confidence']<0.45: continue
            x1,y1,x2,y2=map(int,det['bbox'])
            if (y2-y1)<35: continue
            if roi_pts is not None:
                if cv2.pointPolygonTest(roi_pts,((x1+x2)/2,(y1+y2)/2),False)<0: continue
            crop=frame[max(0,y1):y2,max(0,x1):x2]
            team,tc=classify_team(crop)
            num,nc=jersey_clf.predict(crop,team)
            if num:
                cur_stats[f'{team}_{num}']+=1
                if team=='aigis': aigis_n.append(num)
                else: hm_n.append(num)
        timeline.append({'t':round(frame_idx/fps/60,1),'aigis':aigis_n,'hm':hm_n})

    cap.release()
    os.makedirs(os.path.join(RESULTS_DIR,'test_game'),exist_ok=True)
    json.dump(timeline,open(os.path.join(RESULTS_DIR,'test_game','timeline_retest.json'),'w'),
              ensure_ascii=False,indent=2)

    # 비교 리포트
    total_min = timeline[-1]['t'] if timeline else 0
    lines=[f'=== /retest 결과 ({os.path.basename(video_path)}, {total_min:.0f}분) ===',
           f'소요: {time.time()-t0:.0f}초', '']

    lines.append(f'{"선수":20} {"이전":>8} {"이번":>8} {"변화":>8}')
    lines.append('─'*50)
    all_keys = sorted(set(list(prev_stats.keys())+list(cur_stats.keys())))
    for key in all_keys:
        team,num_s = key.split('_',1)
        num = int(num_s)
        name = AIGIS_NAMES.get(num,f'#{num}') if team=='aigis' else f'HM#{num}'
        prev_t = prev_stats.get(key,0)*0.5
        cur_t  = cur_stats.get(key,0)*0.5
        delta  = cur_t - prev_t
        arrow  = '↑' if delta>2 else ('↓' if delta<-2 else '→')
        lines.append(f'{name:20} {prev_t:>7.1f}분 {cur_t:>7.1f}분 {arrow}{abs(delta):>5.1f}분')

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    os.chdir(BASE)

    if cmd == '/compare':
        print(cmd_compare(args[0] if args else 'data/videos/test_game.mp4'))
    elif cmd == '/retrain_team':
        print(cmd_retrain_team())
    elif cmd == '/retrain_jersey':
        if len(args) < 3:
            print('Usage: /retrain_jersey <team> <number> <crop_folder>')
        else:
            print(cmd_retrain_jersey(args[0], args[1], args[2]))
    elif cmd == '/retest':
        print(cmd_retest(args[0] if args else None))
    else:
        print(f'Unknown command: {cmd}')
        print(__doc__)
