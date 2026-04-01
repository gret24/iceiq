import json, re
from collections import defaultdict

TRACKS  = "/root/iceiq/videos/game1_val/tracks.json"
JMAP    = "/root/iceiq/videos/game1_val/jersey_map.json"
FPS     = 4
GAP_SEC = 30

with open(TRACKS) as f: tracks = json.load(f)
with open(JMAP) as f:   jmap   = json.load(f)

tid_info = {tid: (v["jersey"].lstrip("0") or "0", v.get("team","UNKNOWN"))
            for tid, v in jmap.items()}

player_frames = defaultdict(list)
for fname, fts in sorted(tracks.items()):
    m = re.search(r"(\d+)", fname)
    if not m: continue
    fidx = int(m.group(1))
    for t in fts:
        tid = str(t["track_id"])
        if tid in tid_info:
            num, team = tid_info[tid]
            player_frames[(num, team)].append(fidx)

def calc_shifts(frame_list, fps=FPS, gap=GAP_SEC):
    if not frame_list: return []
    fl = sorted(set(frame_list))
    gap_f = int(gap * fps)
    segs, s, e = [], fl[0], fl[0]
    for f in fl[1:]:
        if f - e <= gap_f: e = f
        else:
            segs.append((round(s/fps,1), round(e/fps,1)))
            s = e = f
    segs.append((round(s/fps,1), round(e/fps,1)))
    return segs

players = []
for (num, team), frames in sorted(player_frames.items(), key=lambda x: -len(x[1])):
    shifts = calc_shifts(frames)
    total_ice = sum(e-s for s,e in shifts)
    players.append({
        "jersey": num, "team": team,
        "total_frames": len(frames),
        "total_shifts": len(shifts),
        "total_ice_time_sec": round(total_ice, 1),
        "total_ice_time_min": round(total_ice/60, 2),
        "shifts": [{"start": s, "end": e, "duration": round(e-s,1)} for s,e in shifts]
    })

home_p = [p for p in players if p["team"] == "HOME"]
away_p = [p for p in players if p["team"] == "AWAY"]

report = {
    "summary": {
        "total_players": len(players),
        "home_players": len(home_p),
        "away_players": len(away_p),
    },
    "players": players,
}

with open("/workspace/iceiq/output/game1_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("=== game1_report.json 생성 완료 ===")
print(f"총 선수: {len(players)}명 | HOME: {len(home_p)} | AWAY: {len(away_p)}")
print("\nTOP10 아이스타임:")
for p in players[:10]:
    print(f"  #{p['jersey']:>3} ({p['team']:7}) | {p['total_ice_time_min']:5.1f}분 | {p['total_shifts']}시프트")
