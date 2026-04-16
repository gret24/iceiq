import pickle, numpy as np, os, cv2
from collections import defaultdict

os.chdir(os.path.expanduser('~/iceiq-dev'))

with open('data/results/game1/cache.pkl','rb') as f:
    data = pickle.load(f)
teams = {k: {'team_a':'Lopez','team_b':'Aigis'}.get(v,v) for k,v in data['teams'].items()}

track_pos = defaultdict(list)
track_frames = defaultdict(list)
for fn, tl in data['tracks'].items():
    for t in tl:
        tid = t.get('track_id',0)
        cx, cy = t.get('cx',0), t.get('cy',0)
        if cx > 0 and teams.get(tid) == 'Aigis':
            track_pos[tid].append((cx, cy))
            track_frames[tid].append(fn)

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

W, H = 1200, 520
PAD, VW, VH = 40, 1920, 1080
RW, RH = W-2*PAD, H-2*PAD

def draw_rink():
    img = np.ones((H,W,3), dtype=np.uint8)*240
    x1,y1,x2,y2 = PAD,PAD,W-PAD,H-PAD
    r=80
    cv2.rectangle(img,(x1+r,y1),(x2-r,y2),(255,255,255),-1)
    cv2.rectangle(img,(x1,y1+r),(x2,y2-r),(255,255,255),-1)
    for cx,cy,sa,ea in [(x1+r,y1+r,180,270),(x2-r,y1+r,270,360),(x1+r,y2-r,90,180),(x2-r,y2-r,0,90)]:
        cv2.ellipse(img,(cx,cy),(r,r),0,sa,ea,(255,255,255),-1)
        cv2.ellipse(img,(cx,cy),(r,r),0,sa,ea,(0,0,0),2)
    cv2.rectangle(img,(x1+r,y1),(x2-r,y1),(0,0,0),2)
    cv2.rectangle(img,(x1+r,y2),(x2-r,y2),(0,0,0),2)
    cv2.rectangle(img,(x1,y1+r),(x1,y2-r),(0,0,0),2)
    cv2.rectangle(img,(x2,y1+r),(x2,y2-r),(0,0,0),2)
    cx2,cy2=W//2,H//2
    cv2.line(img,(cx2,y1),(cx2,y2),(0,0,200),3)
    cv2.circle(img,(cx2,cy2),45,(0,0,200),2)
    bl,br=int(x1+RW*0.35),int(x1+RW*0.65)
    cv2.line(img,(bl,y1),(bl,y2),(200,50,0),4)
    cv2.line(img,(br,y1),(br,y2),(200,50,0),4)
    gl,gr=int(x1+RW*0.08),int(x1+RW*0.92)
    cv2.line(img,(gl,y1+r),(gl,y2-r),(0,0,200),2)
    cv2.line(img,(gr,y1+r),(gr,y2-r),(0,0,200),2)
    cv2.rectangle(img,(gl-15,cy2-30),(gl,cy2+30),(0,0,180),2)
    cv2.rectangle(img,(gr,cy2-30),(gr+15,cy2+30),(0,0,180),2)
    cv2.ellipse(img,(gl,cy2),(35,35),0,-90,90,(255,200,200),-1)
    cv2.ellipse(img,(gr,cy2),(35,35),0,90,270,(255,200,200),-1)
    for fx in [int(x1+RW*0.22),int(x1+RW*0.78)]:
        for fy in [int(y1+RH*0.3),int(y1+RH*0.7)]:
            cv2.circle(img,(fx,fy),40,(200,0,0),2)
            cv2.circle(img,(fx,fy),4,(200,0,0),-1)
    return img

roster=[('4','Yoon'),('14','Lee'),('11','Park'),('28','Lim'),('25','Kim_J'),('36','Nam'),('47','Han'),('61','Kim_S')]
os.makedirs('data/results/game1/heatmaps',exist_ok=True)

for i in range(8):
    positions=merged[i]['positions']
    jersey,name=roster[i]
    xs=[PAD+(vx/VW)*RW for vx,vy in positions]
    ys=[PAD+(vy/VH)*RH for vx,vy in positions]
    hmap,_,_=np.histogram2d(ys,xs,bins=[H,W],range=[[0,H],[0,W]])
    hmap=cv2.GaussianBlur(hmap.astype(np.float32),(31,31),8)
    if hmap.max()>0: hmap=hmap/hmap.max()
    rink=draw_rink()
    heat_rgb=cv2.applyColorMap((hmap*255).astype(np.uint8),cv2.COLORMAP_JET)
    mask=hmap>0.05
    for c in range(3):
        rink[:,:,c]=np.where(mask,(rink[:,:,c]*(1-hmap*0.7)+heat_rgb[:,:,c]*hmap*0.7).astype(np.uint8),rink[:,:,c])
    cv2.putText(rink,f'#{jersey} {name}',(15,25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)
    cv2.putText(rink,f'{len(positions)}pts',(W-150,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,100,100),1)
    cv2.imwrite(f'data/results/game1/heatmaps/full_{jersey}_{name}.png',rink)
    print(f'Saved: full_{jersey}_{name}.png ({len(positions)} pts)')
print('Done!')
