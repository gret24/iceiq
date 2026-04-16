import pickle, numpy as np, os, cv2, json, sys
from collections import defaultdict

os.chdir(os.path.expanduser('~/iceiq-dev'))

# Allow passing game name as argument
game_name = sys.argv[1] if len(sys.argv) > 1 else 'game2'
cache_path = f'data/results/{game_name}/cache.pkl'
out_dir = f'data/results/{game_name}/heatmaps'

print(f'[heatmap_homo] Game: {game_name}')

with open(cache_path, 'rb') as f:
    data = pickle.load(f)

teams = {k: {'team_a':'Lopez','team_b':'Aigis'}.get(v,v) for k,v in data['teams'].items()}

# Load homography
with open('configs/homography_4point.json') as f:
    cfg = json.load(f)
src = np.float32(cfg['video_points'])
dst = np.float32(cfg['rink_points'])
H_mat, _ = cv2.findHomography(src, dst)

def to_rink(px, py):
    pt = np.array([px, py, 1.0])
    r = H_mat @ pt
    return r[0]/r[2], r[1]/r[2]

track_pos = defaultdict(list)
track_frames = defaultdict(list)

for fn, tl in data['tracks'].items():
    for t in tl:
        tid = t.get('track_id', 0)
        cx, cy = t.get('cx', 0), t.get('cy', 0)
        if cx > 0 and teams.get(tid) == 'Aigis':
            rx, ry = to_rink(cx, cy)
            if 0 <= rx <= 52 and 0 <= ry <= 26:
                fn_int = int(fn)
                # 2P (frame 28000-52000): flip x for end change
                if 28000 <= fn_int <= 52000:
                    rx = 52.0 - rx  # flip rink x
                track_pos[tid].append((rx, ry))
                track_frames[tid].append(fn_int)

sorted_tids = sorted(track_pos.keys(), key=lambda t: -len(track_pos[t]))
merged = []
for tid in sorted_tids:
    frames = set(track_frames[tid])
    matched = False
    for g in merged:
        if len(g['frames'].intersection(frames)) < 5:
            g['frames'].update(frames)
            g['positions'].extend(track_pos[tid])
            matched = True
            break
    if not matched:
        merged.append({'frames': set(frames), 'positions': list(track_pos[tid])})
merged.sort(key=lambda g: -len(g['positions']))

W2, H2 = 1040, 520
RW_M, RH_M = 52.0, 26.0

def draw_rink():
    img = np.ones((H2, W2, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (0, 0), (W2-1, H2-1), (0, 0, 0), 2)
    cx, cy = W2//2, H2//2
    cv2.line(img, (cx, 0), (cx, H2), (0, 0, 200), 3)
    cv2.circle(img, (cx, cy), int(45*W2/1200), (0, 0, 200), 2)
    bl, br = int(0.35*W2), int(0.65*W2)
    cv2.line(img, (bl, 0), (bl, H2), (200, 50, 0), 3)
    cv2.line(img, (br, 0), (br, H2), (200, 50, 0), 3)
    gl, gr = int(0.08*W2), int(0.92*W2)
    cv2.line(img, (gl, 0), (gl, H2), (0, 0, 200), 2)
    cv2.line(img, (gr, 0), (gr, H2), (0, 0, 200), 2)
    cv2.rectangle(img, (gl-12, cy-25), (gl, cy+25), (0, 0, 180), 2)
    cv2.rectangle(img, (gr, cy-25), (gr+12, cy+25), (0, 0, 180), 2)
    for fx in [0.22, 0.78]:
        for fy in [0.3, 0.7]:
            cv2.circle(img, (int(fx*W2), int(fy*H2)), 30, (200, 0, 0), 2)
            cv2.circle(img, (int(fx*W2), int(fy*H2)), 3, (200, 0, 0), -1)
    return img

roster = [('4','Yoon'),('14','Lee'),('11','Park'),('28','Lim'),('25','Kim_J'),('36','Nam'),('47','Han'),('61','Kim_S')]
os.makedirs(out_dir, exist_ok=True)

for i in range(min(8, len(merged))):
    positions = merged[i]['positions']
    jersey, name = roster[i]
    xs = [p[0]/RW_M*W2 for p in positions]
    ys = [p[1]/RH_M*H2 for p in positions]
    hmap, _, _ = np.histogram2d(ys, xs, bins=[H2, W2], range=[[0, H2], [0, W2]])
    hmap = cv2.GaussianBlur(hmap.astype(np.float32), (31, 31), 8)
    if hmap.max() > 0: hmap = hmap / hmap.max()
    rink = draw_rink()
    heat = cv2.applyColorMap((hmap*255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = hmap > 0.05
    for c in range(3):
        rink[:,:,c] = np.where(mask, (rink[:,:,c]*(1-hmap*0.7)+heat[:,:,c]*hmap*0.7).astype(np.uint8), rink[:,:,c])
    cv2.putText(rink, f'#{jersey} {name} [2P-flip]', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(rink, f'{len(positions)}pts', (W2-140, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    out_path = f'{out_dir}/homo_{jersey}_{name}.png'
    cv2.imwrite(out_path, rink)
    print(f'Saved: homo_{jersey}_{name}.png ({len(positions)} pts)')

print('Done!')
