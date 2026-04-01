#!/usr/bin/env python3
"""
IceIQ API v2 — pipeline_fast + build_highlight + gen_report 연결
"""
import json, os, re, sys, uuid, threading
from collections import Counter, defaultdict
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, "/root/iceiq")
sys.path.insert(0, "/workspace/iceiq")

app = FastAPI(title="IceIQ API v2", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

BASE_DIR   = "/root/iceiq"
VIDEOS_DIR = f"{BASE_DIR}/videos"
OUTPUT_DIR = "/workspace/iceiq/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 잡 트래킹 ─────────────────────────────────────────────
_jobs: dict = {}
_lock = threading.Lock()

def _new_job(video_stem: str) -> str:
    jid = str(uuid.uuid4())[:8]
    with _lock:
        _jobs[jid] = {"status": "pending", "message": "대기 중", "progress": 0, "video_stem": video_stem}
    return jid

def _set_job(jid, **kw):
    with _lock:
        _jobs[jid].update(kw)


# ── 유틸: shifts 계산 ────────────────────────────────────
def _calc_shifts(frame_list, fps=4, gap_sec=30, buf_sec=5):
    if not frame_list: return []
    fl = sorted(set(frame_list))
    gap_f = int(gap_sec * fps)
    segs, s, e = [], fl[0], fl[0]
    for f in fl[1:]:
        if f - e <= gap_f: e = f
        else:
            segs.append((s, e)); s = e = f
    segs.append((s, e))
    return [{"start": round(max(0, s/fps - buf_sec), 1),
             "end":   round(e/fps + buf_sec, 1),
             "duration": round(e/fps + buf_sec - max(0, s/fps - buf_sec), 1)}
            for s, e in segs]


# ── 백그라운드: 분석 파이프라인 ──────────────────────────
def _run_analysis(jid: str, video_path: str, fps: int):
    import subprocess
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    vdir = f"{VIDEOS_DIR}/{video_stem}"
    os.makedirs(f"{vdir}/frames", exist_ok=True)

    try:
        # 1. 프레임 추출
        _set_job(jid, status="running", message="프레임 추출 중", progress=5)
        frame_count = len([f for f in os.listdir(f"{vdir}/frames") if f.endswith(".jpg")])
        if frame_count < 100:
            subprocess.run(["ffmpeg", "-y", "-i", video_path,
                            "-vf", f"fps={fps}", f"{vdir}/frames/frame_%05d.jpg"],
                           capture_output=True, check=True)

        # 2. ByteTrack
        _set_job(jid, message="ByteTrack 추적 중", progress=20)
        import pipeline_fast as pf
        pf.BASE_DIR = vdir; pf.FRAMES_DIR = f"{vdir}/frames"
        pf.DETECTED_DIR = f"/workspace/iceiq/detected_{video_stem}"
        pf.TRACKS_JSON = f"{vdir}/tracks.json"
        pf.JERSEY_JSON = f"{vdir}/jersey_map.json"
        pf.EXTRACT_FPS = fps
        os.makedirs(pf.DETECTED_DIR, exist_ok=True)
        tracks = pf.step_track()

        # 3. OCR
        _set_job(jid, message="OCR 등번호 인식 중", progress=60)
        pf.step_ocr(tracks,
            color_profiles=pf.PlayerColorProfiles(),
            behavior_profiles=pf.PlayerBehaviorProfiles(),
            team_calibrator=pf.TeamCalibrator(3000),
            home_roster_set=None)

        _set_job(jid, status="done", message="완료", progress=100)
    except Exception as e:
        _set_job(jid, status="error", message=str(e), progress=0)


# ══════════════════════════════════════════════════════════
# 엔드포인트
# ══════════════════════════════════════════════════════════

class AnalyzeRequest(BaseModel):
    video_path: str
    fps: int = 4

class HighlightRequest(BaseModel):
    video_path: str
    video_stem: str
    player: str
    gap: float = 30
    buf: float = 5


@app.get("/")
def root():
    return {
        "status": "ok", "service": "IceIQ API v2",
        "endpoints": [
            "POST /analyze", "GET /status/{job_id}",
            "GET /players/{video_stem}", "POST /highlight",
            "GET /report/{video_stem}", "GET /video/{filepath:path}",
            "GET /docs"
        ]
    }


@app.post("/analyze")
def analyze(req: AnalyzeRequest, bg: BackgroundTasks):
    if not os.path.exists(req.video_path):
        raise HTTPException(404, f"영상 없음: {req.video_path}")
    stem = os.path.splitext(os.path.basename(req.video_path))[0]
    jid = _new_job(stem)
    bg.add_task(_run_analysis, jid, req.video_path, req.fps)
    return {"job_id": jid, "video_stem": stem, "track": f"/status/{jid}"}


@app.get("/status/{job_id}")
def status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job 없음")
    return job


@app.get("/players/{video_stem}")
def players(video_stem: str):
    jpath = f"{VIDEOS_DIR}/{video_stem}/jersey_map.json"
    if not os.path.exists(jpath):
        raise HTTPException(404, f"jersey_map 없음: {jpath}")
    with open(jpath) as f:
        jmap = json.load(f)
    cnt = Counter(v["jersey"].lstrip("0") or "0" for v in jmap.values())
    team_cnt: dict = defaultdict(lambda: defaultdict(int))
    for v in jmap.values():
        num = v["jersey"].lstrip("0") or "0"
        team_cnt[num][v.get("team","UNKNOWN")] += 1
    result = []
    for num, count in cnt.most_common():
        teams = dict(team_cnt[num])
        result.append({"jersey": num, "detections": count, "teams": teams})
    return {"video_stem": video_stem, "total_tracks": len(jmap), "players": result}


@app.post("/highlight")
def highlight(req: HighlightRequest):
    jpath = f"{VIDEOS_DIR}/{req.video_stem}/jersey_map.json"
    tpath = f"{VIDEOS_DIR}/{req.video_stem}/tracks.json"
    for p, n in [(jpath,"jersey_map"), (tpath,"tracks"), (req.video_path,"video")]:
        if not os.path.exists(p):
            raise HTTPException(404, f"{n} 없음")

    with open(jpath) as f: jmap = json.load(f)
    with open(tpath) as f: tracks = json.load(f)

    norm = req.player.lstrip("0") or "0"
    tids = {tid for tid, v in jmap.items() if v["jersey"].lstrip("0") == norm}
    if not tids:
        raise HTTPException(404, f"#{req.player} 없음")

    frames = []
    for fname, fts in sorted(tracks.items()):
        m = re.search(r"(\d+)", fname)
        if not m: continue
        fidx = int(m.group(1))
        for t in fts:
            if str(t["track_id"]) in tids:
                frames.append(fidx); break

    shifts = _calc_shifts(frames, fps=4, gap_sec=req.gap, buf_sec=req.buf)
    if not shifts:
        raise HTTPException(404, "감지 구간 없음")

    total_ice = sum(s["duration"] for s in shifts)
    out = f"{OUTPUT_DIR}/{req.video_stem}_{norm}_highlight.mp4"

    import subprocess, shutil, tempfile
    tmpdir = tempfile.mkdtemp()
    clips = []
    for i, sh in enumerate(shifts, 1):
        mm, ss = divmod(int(sh["start"]), 60)
        label = f"Shift {i} - {mm}:{ss:02d}"
        vf = f"drawtext=text='{label}':fontsize=36:fontcolor=white:x=20:y=20:box=1:boxcolor=black@0.5:boxborderw=5:enable='between(t,0,2)'"
        clip = f"{tmpdir}/clip_{i:03d}.mp4"
        subprocess.run(["ffmpeg","-y","-ss",str(sh["start"]),"-to",str(sh["end"]),
                        "-i",req.video_path,"-vf",vf,"-c:v","libx264","-preset","fast",
                        "-crf","23","-c:a","aac",clip], capture_output=True)
        if os.path.exists(clip):
            clips.append(clip)

    concat = f"{tmpdir}/concat.txt"
    with open(concat,"w") as f:
        for c in clips: f.write(f"file '{c}'\n")
    subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",concat,"-c","copy",out],
                   capture_output=True)
    shutil.rmtree(tmpdir, ignore_errors=True)

    size = os.path.getsize(out)/1024/1024 if os.path.exists(out) else 0
    return {
        "player": norm, "shifts": len(shifts), "total_ice_time_min": round(total_ice/60, 2),
        "file_path": out, "file_size_mb": round(size, 1),
        "stream_url": f"/video/{out.lstrip('/')}",
        "shift_detail": shifts
    }


@app.get("/report/{video_stem}")
def report(video_stem: str):
    jpath = f"{VIDEOS_DIR}/{video_stem}/jersey_map.json"
    tpath = f"{VIDEOS_DIR}/{video_stem}/tracks.json"
    for p, n in [(jpath,"jersey_map"), (tpath,"tracks")]:
        if not os.path.exists(p):
            raise HTTPException(404, f"{n} 없음")

    with open(jpath) as f: jmap = json.load(f)
    with open(tpath) as f: tracks = json.load(f)

    tid_info = {tid: (v["jersey"].lstrip("0") or "0", v.get("team","UNKNOWN")) for tid,v in jmap.items()}
    player_frames: dict = defaultdict(list)
    for fname, fts in sorted(tracks.items()):
        m = re.search(r"(\d+)", fname)
        if not m: continue
        fidx = int(m.group(1))
        for t in fts:
            tid = str(t["track_id"])
            if tid in tid_info:
                num, team = tid_info[tid]
                player_frames[(num, team)].append(fidx)

    players = []
    for (num, team), frames in sorted(player_frames.items(), key=lambda x: -len(x[1])):
        shifts = _calc_shifts(frames)
        total = sum(s["duration"] for s in shifts)
        players.append({
            "jersey": num, "team": team,
            "total_frames": len(frames),
            "total_shifts": len(shifts),
            "total_ice_time_sec": round(total, 1),
            "total_ice_time_min": round(total/60, 2),
        })

    return {
        "video_stem": video_stem,
        "total_players": len(players),
        "home_players": sum(1 for p in players if p["team"]=="HOME"),
        "away_players": sum(1 for p in players if p["team"]=="AWAY"),
        "players": players
    }


@app.get("/video/{filepath:path}")
def stream_video(filepath: str):
    path = f"/{filepath}" if not filepath.startswith("/") else filepath
    if not os.path.exists(path):
        raise HTTPException(404, f"파일 없음: {path}")
    def _iter():
        with open(path, "rb") as f:
            while chunk := f.read(1024*1024):
                yield chunk
    return StreamingResponse(_iter(), media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_v2:app", host="0.0.0.0", port=8000, reload=False)
