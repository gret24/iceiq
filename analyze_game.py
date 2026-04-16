"""
IceIQ Unified Pipeline - analyze_game.py (v2)
Matches actual module APIs.
"""

import argparse
import json
import os
import pickle
import time
import sys
import cv2
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/iceiq-dev"))


def phase1_video(video_path, cache_path, skip_video=False):
    if os.path.exists(cache_path) and skip_video:
        print("[Phase 1] Loading cache...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    if skip_video:
        print("[Phase 1] ERROR: no cache")
        sys.exit(1)
    
    from pipeline.detection import YOLODetector
    from pipeline.tracking import HockeyTracker
    from pipeline.team_classification import HockeyTeamClassifier
    
    t0 = time.time()
    
    # pose 모델 우선, 없으면 일반 모델 fallback
    _base = os.path.dirname(os.path.abspath(__file__))
    _pose_model = os.path.join(_base, "yolov8n-pose.pt")
    _bbox_model = os.path.join(_base, "yolov8n.pt")
    _model_path = _pose_model if os.path.exists(_pose_model) else _bbox_model
    detector = YOLODetector(model_path=_model_path)
    tracker = HockeyTracker()
    classifier = HockeyTeamClassifier()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Phase 1] Video: {total_frames} frames, {fps:.0f} fps")
    
    all_tracks = {}
    all_teams = {}
    frame_num = 0
    det_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections = detector.detect(frame)
        det_count += len(detections)
        
        track_results = tracker.update(detections, frame=frame)
        
        frame_tracks = []
        for tr in track_results:
            td = dict(tr)

            tid = td.get("track_id", 0)

            bbox = td.get("bbox", td.get("tlbr", None))
            if bbox is not None and len(bbox) >= 4:
                td["cx"] = (bbox[0] + bbox[2]) / 2
                td["cy"] = (bbox[1] + bbox[3]) / 2
                td["h"] = bbox[3] - bbox[1]
                td["w"] = bbox[2] - bbox[0]

            # ── 방향 벡터 계산 (pose 모델 전용) ──────────────────────
            # keypoints: [[x,y,conf]×17] COCO 순서
            # 어깨(5,6) + 코(0) → 어깨 중점 기준 정면 방향
            kps = td.get("keypoints")
            ori = None
            if kps and len(kps) >= 7:
                try:
                    nose, l_sh, r_sh = kps[0], kps[5], kps[6]
                    if l_sh[2] > 0.3 and r_sh[2] > 0.3 and nose[2] > 0.3:
                        # 어깨 중점 (origin)
                        ox = (l_sh[0] + r_sh[0]) / 2
                        oy = (l_sh[1] + r_sh[1]) / 2
                        # 어깨 벡터 → 법선 (수직)
                        sv = [r_sh[0] - l_sh[0], r_sh[1] - l_sh[1]]
                        n  = [-sv[1], sv[0]]
                        # 코 방향으로 법선 부호 선택
                        nd = [nose[0] - ox, nose[1] - oy]
                        if n[0] * nd[0] + n[1] * nd[1] < 0:
                            n = [sv[1], -sv[0]]
                        # 정규화
                        ln = (n[0] ** 2 + n[1] ** 2) ** 0.5
                        if ln > 1e-6:
                            # .item()은 numpy scalar용 — 일반 float도 안전하게 처리
                            def _f(v): return float(v)
                            ori = [round(_f(ox), 1), round(_f(oy), 1),
                                   round(_f(n[0] / ln), 4), round(_f(n[1] / ln), 4)]
                except Exception:
                    pass
            if ori:
                td["ori"] = ori
            # ──────────────────────────────────────────────────────────

            frame_tracks.append(td)
            
            if tid not in all_teams:
                try:
                    bbox = td.get('bbox')
                    if bbox and len(bbox) >= 4:
                        label, conf = classifier.classify(frame, tuple(bbox), tid)
                        if label and conf > 0.3:
                            all_teams[tid] = label
                except:
                    pass
        

        
        all_tracks[frame_num] = frame_tracks
        frame_num += 1
        for _ in range(4): cap.read()  # skip 4 frames (5x faster)
        frame_num += 4

        if frame_num % 300 == 0 or frame_num >= total_frames:
            elapsed = time.time() - t0
            pct = frame_num / total_frames
            bar_len = 30
            filled = int(bar_len * pct)
            bar = "█" * filled + "░" * (bar_len - filled)
            eta = (elapsed / pct * (1 - pct)) if pct > 0 else 0
            print(f"\r  [{bar}] {pct*100:.0f}% {frame_num}/{total_frames} | {elapsed:.0f}s | ETA {eta:.0f}s", end="", flush=True)
    
    cap.release()
    
    track_ids = set()
    for f, tl in all_tracks.items():
        for t in tl:
            track_ids.add(t.get("track_id", 0))
    
    for tid in track_ids:
        if tid not in all_teams:
            all_teams[tid] = "unknown"
    
    team_counts = Counter(all_teams.values())
    
    data = {
        "tracks": all_tracks,
        "teams": all_teams,
        "track_ids": track_ids,
        "frame_count": frame_num,
        "det_count": det_count,
        "team_counts": dict(team_counts),
        "fps": fps,
        "video_path": str(video_path),
    }
    
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    
    elapsed = time.time() - t0
    print(f"  {frame_num} frames, {det_count} dets, {len(track_ids)} tracks")
    print(f"  Teams: {dict(team_counts)}")
    print(f"  Cached. Phase 1: {elapsed:.0f}s")
    return data


def phase2_identify(data, roster=None):
    print("[Phase 2] Merging tracks...")
    t0 = time.time()
    
    tracks = data["tracks"]
    teams = data["teams"]
    teams = {k: {"team_a":"Lopez","team_b":"Aigis"}.get(v,v) for k,v in teams.items()}
    
    profiles = {}
    for frame_num, frame_tracks in tracks.items():
        for t in frame_tracks:
            tid = t.get("track_id", 0)
            if tid not in profiles:
                profiles[tid] = {"track_id": tid, "team": teams.get(tid, "unknown"),
                                 "frames": [], "positions": [], "heights": []}
            profiles[tid]["frames"].append(frame_num)
            cx = t.get("cx", 0)
            cy = t.get("cy", 0)
            if cx > 0:
                profiles[tid]["positions"].append((cx, cy))
            h = t.get("h", 0)
            if h > 0:
                profiles[tid]["heights"].append(h)
    
    for tid, p in profiles.items():
        p["frame_count"] = len(p["frames"])
        p["frame_start"] = min(p["frames"]) if p["frames"] else 0
        p["frame_end"] = max(p["frames"]) if p["frames"] else 0
        p["avg_height"] = float(np.mean(p["heights"])) if p["heights"] else 0
        if p["positions"]:
            p["avg_x"] = float(np.mean([x[0] for x in p["positions"]]))
            p["avg_y"] = float(np.mean([x[1] for x in p["positions"]]))
            p["last_pos"] = p["positions"][-1]
            p["first_pos"] = p["positions"][0]
            ys = [x[1] for x in p["positions"]]
            p["y_range"] = max(ys) - min(ys) if len(ys) > 1 else 0
        else:
            p["avg_x"] = p["avg_y"] = 0
            p["last_pos"] = p["first_pos"] = (0, 0)
            p["y_range"] = 0
    
    valid = {tid: p for tid, p in profiles.items() if p["frame_count"] >= 10}
    print(f"  Valid tracks: {len(valid)}")
    
    team_groups = defaultdict(list)
    for tid, p in valid.items():
        team_groups[p["team"]].append(p)
    
    players = {}
    pid_n = 0
    
    for team_name, profs in team_groups.items():
        profs.sort(key=lambda p: p["frame_start"])
        merged = []
        
        for prof in profs:
            matched = False
            for g in merged:
                if _can_merge(prof, g):
                    g["tl"].append(prof)
                    g["fe"] = max(g["fe"], prof["frame_end"])
                    g["tf"] += prof["frame_count"]
                    g["ap"].extend(prof["positions"])
                    g["ah"].extend(prof["heights"])
                    g["lp"] = prof["last_pos"]
                    matched = True
                    break
            if not matched:
                merged.append({"tl": [prof], "tm": team_name,
                               "fs": prof["frame_start"], "fe": prof["frame_end"],
                               "tf": prof["frame_count"],
                               "ap": list(prof["positions"]),
                               "ah": list(prof["heights"]),
                               "lp": prof["last_pos"]})
        
        for g in merged:
            if g["tf"] < 20:
                continue
            pid = f"P{pid_n:02d}"
            pid_n += 1
            ah = float(np.mean(g["ah"])) if g["ah"] else 0
            yrs = [gg["tl"][0]["y_range"] for gg in merged if gg["tl"]]
            myr = float(np.median(yrs)) if yrs else 100
            oyr = 0
            if g["ap"]:
                ys = [p[1] for p in g["ap"]]
                oyr = max(ys) - min(ys) if len(ys) > 1 else 0
            pos = "D" if oyr < myr * 0.7 else "F"
            
            players[pid] = {"player_id": pid, "team": team_name,
                           "track_ids": [t["track_id"] for t in g["tl"]],
                           "track_count": len(g["tl"]), "total_frames": g["tf"],
                           "avg_bbox_height": round(ah, 1), "position": pos,
                           "all_positions": g["ap"]}
    
    if roster:
        _match_roster(players, roster)
    
    # Filter: keep only roster-matched + top unmatched per team
    matched = {pid: p for pid, p in players.items() if p.get("jersey") != "?"}
    unmatched = {pid: p for pid, p in players.items() if p.get("jersey") == "?"}
    
    # Keep top 12 unmatched per team (opponents)
    for team in set(p["team"] for p in unmatched.values()):
        team_um = sorted(
            [(pid,p) for pid,p in unmatched.items() if p["team"]==team],
            key=lambda x: -x[1]["total_frames"]
        )
        for pid, p in team_um[12:]:
            del players[pid]
    
    print(f"  {len(players)} players ({time.time()-t0:.1f}s)")
    for pid, p in sorted(players.items()):
        j = p.get("jersey", "?")
        n = p.get("name", "")
        print(f"    {pid}: #{j} {n} {p['team']} {p['position']} "
              f"trk={p['track_count']} frm={p['total_frames']}")
    return players


def _can_merge(new, group):
    if new["team"] != group["tm"]:
        return False
    if new["frame_start"] <= group["fe"]:
        ef = set()
        for t in group["tl"]:
            ef.update(t["frames"])
        if len(ef.intersection(set(new["frames"]))) > 5:
            return False
    last = group["lp"]
    first = new["first_pos"]
    dist = np.sqrt((last[0]-first[0])**2 + (last[1]-first[1])**2)
    gap = new["frame_start"] - group["fe"]
    score = 0
    if dist < 200 and gap < 100:
        score += 2
    elif dist < 400 and gap < 200:
        score += 1
    if new["avg_height"] > 0 and group["ah"]:
        if abs(new["avg_height"] - np.mean(group["ah"])) < np.mean(group["ah"]) * 0.15:
            score += 1
    return score >= 1


def _match_roster(players, roster):
    rp = roster.get("players", [])
    if not rp:
        return
    # Match by ice time order within team
    team_name = roster.get("team", "")
    team_players = sorted(
        [p for p in players.values() if p["team"] == "Aigis"],
        key=lambda p: -p["total_frames"]
    )
    roster_sorted = sorted(rp, key=lambda r: r["number"])
    
    for i, tp in enumerate(team_players):
        if i < len(roster_sorted):
            tp["jersey"] = roster_sorted[i]["number"]
            tp["name"] = roster_sorted[i].get("name", "")
            tp["position"] = roster_sorted[i].get("position", "F")
    
    for p in players.values():
        if "jersey" not in p:
            p["jersey"] = "?"
            p["name"] = ""


def phase3_analyze(players, data, config):
    print("[Phase 3] Stats...")
    t0 = time.time()
    fps = config.get("fps", data.get("fps", 30))
    mpx = config.get("meters_per_pixel_x", 0.0271)
    mpy = config.get("meters_per_pixel_y", 0.0325)
    
    stats = {}
    for pid, player in players.items():
        pos = player.get("all_positions", [])
        tf = player.get("total_frames", 0)
        if len(pos) < 2:
            continue
        
        it_s = tf / fps
        it_m = it_s / 60
        td = 0
        spds = []
        for i in range(1, len(pos)):
            dx = (pos[i][0]-pos[i-1][0])*mpx
            dy = (pos[i][1]-pos[i-1][1])*mpy
            d = np.sqrt(dx**2+dy**2)
            if d < 2.0:
                td += d
                s = d*fps*3.6
                if s < 50:
                    if s < 40: spds.append(s)
        
        avg_s = float(np.mean(spds)) if spds else 0
        max_s = float(np.max(spds)) if spds else 0
        t95 = float(np.percentile(spds, 95)) if len(spds) > 10 else max_s
        
        sc = 0; ins = False; sl = 0
        for s in spds:
            if s > 20:
                sl += 1
                if sl >= 2 and not ins:
                    sc += 1; ins = True
            else:
                ins = False; sl = 0
        
        zn = {"O":0,"N":0,"D":0}
        for p in pos:
            yr = p[1]/1080
            if yr < 0.33: zn["O"] += 1
            elif yr < 0.67: zn["N"] += 1
            else: zn["D"] += 1
        zt = sum(zn.values()) or 1
        
        stats[pid] = {
            "player_id": pid, "jersey": player.get("jersey","?"),
            "name": player.get("name",""), "team": player.get("team","?"),
            "position": player.get("position","?"),
            "track_count": player.get("track_count",0),
            "track_ids": player.get("track_ids", []),
            "fps": fps,
            "ice_time_min": round(it_m,1),
            "total_frames": tf,
            "total_distance_km": round(td/1000,2),
            "dist_per_min": round(td/it_m,1) if it_m > 0 else 0,
            "avg_speed_kmh": round(avg_s,1),
            "max_speed_kmh": round(max_s,1),
            "top95_speed_kmh": round(t95,1),
            "sprint_count": sc,
            "zone_O": round(zn["O"]/zt*100,1),
            "zone_N": round(zn["N"]/zt*100,1),
            "zone_D": round(zn["D"]/zt*100,1),
        }
    
    ts = {}
    for tn in set(p["team"] for p in players.values()):
        tp = [s for s in stats.values() if s["team"] == tn]
        if tp:
            ts[tn] = {"players": len(tp),
                      "avg_dist": round(np.mean([p["total_distance_km"] for p in tp]),2),
                      "avg_spd": round(np.mean([p["avg_speed_kmh"] for p in tp]),1),
                      "sprints": sum(p["sprint_count"] for p in tp)}
    
    print(f"  {len(stats)} players ({time.time()-t0:.1f}s)")
    return stats, ts


def phase4_output(stats, ts, players, output_dir):
    print("[Phase 4] Saving...")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "player_stats.json"), "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "team_stats.json"), "w") as f:
        json.dump(ts, f, indent=2, ensure_ascii=False)
    
    lines = ["="*65, "  IceIQ GAME ANALYSIS REPORT", "="*65]
    name_map = {'team_a': 'team_a', 'team_b': 'team_b'}
    for tn, t in ts.items():
        tn = name_map.get(tn, tn)
        lines.append(f"\n  TEAM: {tn} ({t['players']}p) dist:{t['avg_dist']}km spd:{t['avg_spd']}km/h spr:{t['sprints']}")
    
    lines.append("\n" + "-"*65)
    lines.append(f"  {'#':>4} {'Name':>8} {'Team':>8} {'Pos':>3} {'Time':>6} {'Dist':>6} {'Avg':>5} {'Max':>5} {'Spr':>4}")
    lines.append("-"*65)
    
    for p in sorted(stats.values(), key=lambda x: -x["ice_time_min"]):
        lines.append(f"  {str(p.get('jersey','?')):>4} {p.get('name','')[:6]:>8} {p['team']:>8} {p['position']:>3} "
                     f"{p['ice_time_min']:>5.1f}m {p['total_distance_km']:>5.2f}k {p['avg_speed_kmh']:>4.1f} "
                     f"{p['top95_speed_kmh']:>4.1f} {p['sprint_count']:>4}")
    
    lines.append("-"*65)
    summary = "\n".join(lines)
    
    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)
    print(summary)
    print(f"\n  Saved: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--roster", default=None)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--rink-length", type=float, default=52.0)
    args = parser.parse_args()
    
    vn = Path(args.video).stem
    od = os.path.expanduser(f"~/iceiq-dev/data/results/{vn}")
    cp = os.path.join(od, "cache.pkl")
    
    cfp = os.path.expanduser("~/iceiq-dev/configs/simple_homography.json")
    if os.path.exists(cfp):
        with open(cfp) as f: config = json.load(f)
    else:
        config = {"meters_per_pixel_x": args.rink_length/1920, "meters_per_pixel_y": args.rink_length/1920*1.2}
    
    roster = None
    if args.roster and os.path.exists(args.roster):
        with open(args.roster) as f: roster = json.load(f)
        print(f"Roster: {len(roster.get('players',[]))}p")
    
    print(f"\nIceIQ Analyzer\nVideo: {args.video}\nOutput: {od}\nRink: {args.rink_length}m\n")
    
    t0 = time.time()
    data = phase1_video(args.video, cp, args.skip_video)
    players = phase2_identify(data, roster)
    stats, ts = phase3_analyze(players, data, config)
    phase4_output(stats, ts, players, od)
    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
