"""
IceIQ API Server
FastAPI wrapper for analyze_game.py pipeline
"""
import os, re, json, uuid, time, asyncio, shutil, subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
import math

# ─── 포즈 키포인트 → 방향 벡터 ──────────────────────────────────────────────

def calc_ori(kps) -> list | None:
    """
    keypoints ([[x,y,conf]×17] 또는 JSON 문자열) → [ox, oy, dx, dy]
    어깨(5,6) + 코(0) 기반 정면 방향 벡터
    """
    try:
        if kps is None:
            return None
        kps = json.loads(kps) if isinstance(kps, str) else kps
        if not kps or len(kps) < 7:
            return None
        l_sh, r_sh, nose = kps[5], kps[6], kps[0]
        if l_sh[2] < 0.3 or r_sh[2] < 0.3 or nose[2] < 0.3:
            return None
        mx = (l_sh[0] + r_sh[0]) / 2
        my = (l_sh[1] + r_sh[1]) / 2
        sx = r_sh[0] - l_sh[0]
        sy = r_sh[1] - l_sh[1]
        nx, ny = -sy, sx
        if nx * (nose[0] - mx) + ny * (nose[1] - my) < 0:
            nx, ny = sy, -sx
        ln = math.sqrt(nx * nx + ny * ny)
        if ln < 1e-6:
            return None
        return [round(mx, 1), round(my, 1), round(nx / ln, 4), round(ny / ln, 4)]
    except Exception:
        return None


# === Config ===
# Docker: /app  |  로컬: ~/iceiq-dev
BASE_DIR = Path(os.environ.get("ICEIQ_BASE_DIR", Path(__file__).parent))
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
RESULTS_DIR = BASE_DIR / "data" / "results"
ROSTER_DIR = BASE_DIR / "data" / "rosters"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = 10 * 1024 * 1024 * 1024  # 10 GB


class LimitUploadSize(BaseHTTPMiddleware):
    """업로드 파일 크기를 MAX_UPLOAD_SIZE로 제한 (413 반환)."""
    async def dispatch(self, request: Request, call_next) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_UPLOAD_SIZE:
            return Response(
                content=f"Request too large. Max {MAX_UPLOAD_SIZE // (1024**2)} MB.",
                status_code=413,
            )
        return await call_next(request)


app = FastAPI(title="IceIQ API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(LimitUploadSize)

# In-memory job tracker (replace with Redis/Firestore later)
jobs = {}


# ─── Pose orientation helper ──────────────────────────────────────────────────

def _kp_orientation(kps: list | None) -> list | None:
    """
    YOLO pose keypoints → 방향 벡터 [ox, oy, dx, dy]

    kps: [[x, y, conf], ...] × 17  (COCO 순서)
         혹은 None

    COCO 인덱스:
      5=왼어깨  6=오른어깨  11=왼엉덩이  12=오른엉덩이

    origin  = 어깨 중점 (상체 기준점)
    direction = 어깨 중점 → 엉덩이 중점 벡터 (정규화, 스케이팅 방향 근사)
    반환: [ox, oy, dx, dy] (float, 소수점 2자리) 또는 None
    """
    if kps is None or len(kps) < 13:
        return None

    # 신뢰도 임계값
    CONF_THRESH = 0.3

    def valid(idx):
        return len(kps[idx]) >= 3 and kps[idx][2] >= CONF_THRESH

    # 어깨 중점
    if valid(5) and valid(6):
        sx = (kps[5][0] + kps[6][0]) / 2
        sy = (kps[5][1] + kps[6][1]) / 2
    elif valid(5):
        sx, sy = kps[5][0], kps[5][1]
    elif valid(6):
        sx, sy = kps[6][0], kps[6][1]
    else:
        return None

    # 엉덩이 중점
    if valid(11) and valid(12):
        hx = (kps[11][0] + kps[12][0]) / 2
        hy = (kps[11][1] + kps[12][1]) / 2
    elif valid(11):
        hx, hy = kps[11][0], kps[11][1]
    elif valid(12):
        hx, hy = kps[12][0], kps[12][1]
    else:
        return None

    # 방향 벡터 (어깨→엉덩이) 정규화
    dx, dy = hx - sx, hy - sy
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return None
    dx, dy = dx / length, dy / length

    return [round(sx, 2), round(sy, 2), round(dx, 4), round(dy, 4)]

# === Models ===
class AnalyzeRequest(BaseModel):
    team_name: str = "Aigis"
    roster_file: str = "aigis.json"
    homography_points: Optional[list] = None  # [[x,y], [x,y], [x,y], [x,y]]
    jersey_color: Optional[str] = None

class JobStatus(BaseModel):
    job_id: str
    status: str  # queued, processing, phase1, phase2, phase3, phase4, done, error
    progress: int  # 0-100
    message: str
    created_at: str
    completed_at: Optional[str] = None
    result_path: Optional[str] = None

class PlayerStats(BaseModel):
    jersey: int
    name: str
    position: str
    team: str
    ice_time_min: float
    distance_km: float
    avg_speed_kmh: float
    top_speed_kmh: float
    sprints: int

class GameResult(BaseModel):
    game_id: str
    video_name: str
    teams: dict
    players: list
    heatmap_urls: list
    analyzed_at: str

# === Background Analysis ===
async def run_analysis(job_id: str, video_path: str, roster_path: str, homo_points: list = None):
    """Run analyze_game.py in background"""
    try:
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["message"] = "Starting analysis..."

        game_name = Path(video_path).stem
        output_dir = RESULTS_DIR / game_name

        # Build command
        cmd = f'cd {BASE_DIR} && eval "$(/opt/homebrew/bin/conda shell.bash hook)" && conda activate iceiq && '
        cmd += f'python3 analyze_game.py --video {video_path} --roster {roster_path}'

        # Update status phases
        jobs[job_id]["status"] = "phase1"
        jobs[job_id]["message"] = "Detecting & tracking players..."
        jobs[job_id]["progress"] = 10

        # Run the actual pipeline
        process = await asyncio.create_subprocess_shell(
            cmd,
            cwd=str(BASE_DIR)
        )

        # Wait for completion (log to file instead of pipe)
        await process.wait()

        if process.returncode == 0:
            # Generate heatmaps
            heatmap_cmd = f'cd {BASE_DIR} && eval "$(/opt/homebrew/bin/conda shell.bash hook)" && conda activate iceiq && '
            heatmap_cmd += f'python3 heatmap_homo.py'
            heatmap_proc = await asyncio.create_subprocess_shell(heatmap_cmd, cwd=str(BASE_DIR))
            await heatmap_proc.wait()

            jobs[job_id]["status"] = "done"
            jobs[job_id]["message"] = "Analysis complete!"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
            jobs[job_id]["result_path"] = str(output_dir)
        else:
            stderr = await process.stderr.read()
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = f"Error: {stderr.decode()[:200]}"

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(e)[:200]

# === API Endpoints ===
@app.get("/")
async def root():
    return {"service": "IceIQ API", "version": "1.0.0", "status": "running"}

@app.get("/api/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# --- Upload & Analyze ---
@app.post("/api/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    team_name: str = Form("Aigis"),
    roster_file: str = Form("aigis.json"),
):
    """Upload video and start analysis"""
    # Save uploaded video
    job_id = str(uuid.uuid4())[:8]
    video_filename = f"{job_id}_{file.filename}"
    video_path = UPLOAD_DIR / video_filename

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    roster_path = ROSTER_DIR / roster_file
    if not roster_path.exists():
        raise HTTPException(status_code=400, detail=f"Roster not found: {roster_file}")

    video_stem = Path(file.filename).stem

    # Create job
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued for analysis",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result_path": None,
        "video_name": file.filename,
        "video_stem": video_stem,
        "team_name": team_name,
    }

    # Start background analysis
    background_tasks.add_task(
        run_analysis, job_id, str(video_path), str(roster_path)
    )

    return {"job_id": job_id, "video_stem": video_stem, "game_id": f"{job_id}_{video_stem}", "status": "queued", "message": "Analysis started"}

@app.post("/api/analyze/youtube")
async def analyze_youtube(
    background_tasks: BackgroundTasks,
    youtube_url: str = Form(...),
    team_name: str = Form("Aigis"),
    roster_file: str = Form("aigis.json"),
):
    """YouTube URL에서 영상 동기 다운로드 후 분석 큐 등록"""
    import yt_dlp as _yt_dlp

    job_id = str(uuid.uuid4())[:8]

    ydl_opts = {
        'outtmpl': str(UPLOAD_DIR / f'{job_id}_%(title).60s.%(ext)s'),
        'format': 'best[height<=1080]/best',
        'quiet': True,
        'no_warnings': True,
    }

    try:
        with _yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            video_path = Path(ydl.prepare_filename(info))
            # ffmpeg 변환 후 실제 파일 탐색
            if not video_path.exists():
                for ext in ('mp4', 'mkv', 'webm', 'mov'):
                    candidate = video_path.with_suffix(f'.{ext}')
                    if candidate.exists():
                        video_path = candidate
                        break
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"YouTube download failed: {e}")

    roster_path = ROSTER_DIR / roster_file
    if not roster_path.exists():
        raise HTTPException(status_code=400, detail=f"Roster not found: {roster_file}")

    # 파일명 sanitize: 영문/숫자/한글/언더스코어/하이픈만 남기고 나머지는 _로 치환
    raw_stem = video_path.stem
    safe_stem = re.sub(r'[^\w\-가-힣]', '_', raw_stem)
    safe_stem = re.sub(r'_+', '_', safe_stem)   # 연속 _ 제거
    safe_stem = safe_stem.strip('_')             # 앞뒤 _ 제거
    safe_stem = safe_stem[:50]                   # 50자 제한
    video_stem = safe_stem or job_id             # 공백 방지용 fallback

    safe_path = video_path.parent / f"{video_stem}{video_path.suffix}"
    if video_path != safe_path:
        video_path.rename(safe_path)
        video_path = safe_path

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Downloaded from YouTube, queued for analysis",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result_path": None,
        "video_name": video_path.name,
        "video_stem": video_stem,
        "team_name": team_name,
    }

    background_tasks.add_task(run_analysis, job_id, str(video_path), str(roster_path))

    return {
        "job_id": job_id,
        "video_stem": video_stem,
        "game_id": f"{job_id}_{video_stem}",
        "status": "queued",
        "message": "YouTube video downloaded and queued",
    }

@app.post("/api/analyze/local")
async def analyze_local(
    background_tasks: BackgroundTasks,
    video_path: str = Form(...),
    roster_file: str = Form("aigis.json"),
):
    """Analyze a video already on the server (for development)"""
    full_video = BASE_DIR / video_path
    if not full_video.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    roster_path = ROSTER_DIR / roster_file
    if not roster_path.exists():
        raise HTTPException(status_code=400, detail=f"Roster not found: {roster_file}")

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued for analysis",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "result_path": None,
        "video_name": Path(video_path).name,
        "team_name": "Aigis",
    }

    background_tasks.add_task(
        run_analysis, job_id, str(full_video), str(roster_path)
    )

    return {"job_id": job_id, "status": "queued"}

# --- Job Status ---
@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get analysis job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/api/jobs")
async def list_jobs():
    """List all jobs"""
    return list(jobs.values())

# --- Game Results ---
@app.get("/api/games")
async def list_games():
    """List all analyzed games"""
    games = []
    if RESULTS_DIR.exists():
        for d in sorted(RESULTS_DIR.iterdir()):
            stats_file = d / "player_stats.json"
            if stats_file.exists():
                with open(stats_file) as f:
                    stats = json.load(f)
                named = [s for s in stats.values() if s.get("name")]
                games.append({
                    "game_id": d.name,
                    "path": str(d),
                    "players_identified": len(named),
                    "total_players": len(stats),
                })
    return games

@app.get("/api/games/{game_id}")
async def get_game(game_id: str):
    """Get full game results"""
    game_dir = RESULTS_DIR / game_id
    if not game_dir.exists():
        matched = sorted(
            [d for d in RESULTS_DIR.iterdir() if d.name.endswith(f"_{game_id}")],
            key=lambda d: d.stat().st_mtime,
        )
        if matched:
            game_dir = matched[-1]
    stats_file = game_dir / "player_stats.json"
    if not stats_file.exists():
        raise HTTPException(status_code=404, detail="Game not found")

    with open(stats_file) as f:
        stats = json.load(f)

    # --- Coverage correction from cache.pkl ---
    import pickle
    coverage_info = None
    cf_home = 1.0   # team_b (Aigis)
    cf_away = 1.0   # team_a (opponent)

    cache_file = game_dir / "cache.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)

        team_labels = cache_data.get("teams", {})   # track_id -> "team_a"|"team_b"
        tracks_by_frame = cache_data.get("tracks", {})

        counts_a, counts_b = [], []
        for fn, track_list in tracks_by_frame.items():
            n_a = sum(1 for t in track_list if team_labels.get(t.get("track_id", 0)) == "team_a")
            n_b = sum(1 for t in track_list if team_labels.get(t.get("track_id", 0)) == "team_b")
            # 각 팀별로 2명 이상 감지된 프레임만 포함
            if n_a >= 2:
                counts_a.append(n_a)
            if n_b >= 2:
                counts_b.append(n_b)

        if counts_a or counts_b:
            avg_a = sum(counts_a) / len(counts_a) if counts_a else 2.0
            avg_b = sum(counts_b) / len(counts_b) if counts_b else 2.0
            cf_away = round(min(5.0 / avg_a, 3.0), 3) if avg_a > 0 else 1.0
            cf_home = round(min(5.0 / avg_b, 3.0), 3) if avg_b > 0 else 1.0
            coverage_info = {
                "avg_detected_home": round(avg_b, 2),
                "avg_detected_away": round(avg_a, 2),
                "correction_factor_home": cf_home,
                "correction_factor_away": cf_away,
                "camera_coverage_pct": round((avg_a + avg_b) / 2 / 5 * 100),
                "note": "5인 제약 기반 보정. 실제 아이스타임은 추정값입니다.",
            }

    # Build player list
    players = []
    for pid, s in stats.items():
        ice_min = s.get("ice_time_min", 0)
        tracked_sec = round(ice_min * 60, 1)
        # team_b = Aigis (home), team_a = away — same convention as shifts endpoint
        cf = cf_home if s.get("team") == "Aigis" else cf_away
        players.append({
            "player_id": pid,
            "jersey": s.get("jersey", "?"),
            "name": s.get("name", ""),
            "team": s.get("team", ""),
            "position": s.get("position", "F"),
            "ice_time_min": ice_min,
            "distance_km": s.get("total_distance_km", 0),
            "avg_speed_kmh": s.get("avg_speed_kmh", 0),
            "top_speed_kmh": s.get("top95_speed_kmh", s.get("top_speed_kmh", 0)),
            "sprints": s.get("sprint_count", 0),
            "tracked_time_sec": tracked_sec,
            "correction_factor": cf,
            "estimated_ice_time_sec": round(tracked_sec * cf, 1),
        })

    # Sort by estimated ice time
    players.sort(key=lambda p: -p["estimated_ice_time_sec"])

    # Heatmap files
    heatmap_dir = game_dir / "heatmaps"
    heatmaps = []
    if heatmap_dir.exists():
        for f in sorted(heatmap_dir.glob("homo_*.png")):
            heatmaps.append(f"/api/games/{game_id}/heatmaps/{f.name}")

    return {
        "game_id": game_id,
        "players": players,
        "heatmaps": heatmaps,
        "analyzed_at": datetime.fromtimestamp(stats_file.stat().st_mtime).isoformat(),
        "coverage_info": coverage_info,
    }

@app.get("/api/games/{game_id}/players/{player_id}")
async def get_player(game_id: str, player_id: str):
    """Get individual player stats"""
    stats_file = RESULTS_DIR / game_id / "player_stats.json"
    if not stats_file.exists():
        raise HTTPException(status_code=404, detail="Game not found")

    with open(stats_file) as f:
        stats = json.load(f)

    if player_id not in stats:
        raise HTTPException(status_code=404, detail="Player not found")

    return stats[player_id]

@app.get("/api/games/{game_id}/heatmaps/{filename}")
async def get_heatmap(game_id: str, filename: str):
    """Serve heatmap image"""
    path = RESULTS_DIR / game_id / "heatmaps" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Heatmap not found")
    return FileResponse(path, media_type="image/png")

# --- Tracking ---
@app.get("/tracking/{video_stem}/all_tracks")
async def get_all_tracks(video_stem: str):
    """Return all track data from cache — player_metrics.py compatible format"""
    import pickle
    from collections import defaultdict as _dd

    cache_file = RESULTS_DIR / video_stem / "cache.pkl"
    stats_file = RESULTS_DIR / video_stem / "player_stats.json"
    if not cache_file.exists():
        raise HTTPException(status_code=404, detail=f"Cache not found: {video_stem}")

    with open(cache_file, "rb") as f:
        data = pickle.load(f)

    fps = data.get("fps", 30)
    team_map = {k: ("Aigis" if v == "team_b" else "Lopez") for k, v in data["teams"].items()}

    # Load jersey map from stats
    jersey_map = {}
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
        for p in stats.values():
            for tid in p.get("track_ids", []):
                jersey_map[tid] = {
                    "jersey": str(p.get("jersey", "?")),
                    "name": p.get("name", ""),
                    "team": p.get("team", ""),
                }

    # Aggregate: track_id → list of {frame, bbox}
    track_points = _dd(list)
    for fn, tl in data["tracks"].items():
        for t in tl:
            tid = t.get("track_id", 0)
            cx, cy = t.get("cx", 0), t.get("cy", 0)
            if cx <= 0:
                continue
            w = t.get("w", 40)
            h = t.get("h", 80)
            pt = {
                "frame": int(fn),
                "bbox": [cx - w/2, cy - h/2, cx + w/2, cy + h/2],
            }
            # ori 우선순위: 1) 파이프라인이 직접 계산한 ori  2) keypoints에서 calc_ori
            ori = t.get("ori") or calc_ori(t.get("keypoints"))
            if ori:
                pt["ori"] = ori
            track_points[tid].append(pt)

    # Build tracks list in player_metrics format
    tracks = []
    for tid, points in track_points.items():
        jinfo = jersey_map.get(tid, {})
        tracks.append({
            "track_id": tid,
            "jersey": jinfo.get("jersey", ""),
            "name": jinfo.get("name", ""),
            "team": jinfo.get("team", "") or team_map.get(tid, "unknown"),
            "points": sorted(points, key=lambda p: p["frame"]),
        })

    return {
        "video_stem": video_stem,
        "fps": fps,
        "frame_count": data.get("frame_count", 0),
        "total_players": len(tracks),
        "tracks": tracks,
        "jersey_map": {str(k): v for k, v in jersey_map.items()},
    }

# --- Tactics ---
def _build_tactics_result(video_stem: str, home_jerseys: list, away_jerseys: list):
    """tactics_classifier.py로 전술 분석 실행 후 결과 반환"""
    import pickle
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from tactics_classifier import generate_tactics_timeline, extract_opponent_patterns, dedupe_tracks_by_jersey

    cache_file = RESULTS_DIR / video_stem / "cache.pkl"
    stats_file = RESULTS_DIR / video_stem / "player_stats.json"

    if not cache_file.exists():
        raise HTTPException(status_code=404, detail=f"Cache not found: {video_stem}")

    with open(cache_file, "rb") as f:
        data = pickle.load(f)

    fps = data.get("fps", 30)
    team_map_raw = {k: ("Aigis" if v == "team_b" else "Lopez") for k, v in data["teams"].items()}

    # jersey_map from stats
    jersey_map = {}
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)
        for p in stats.values():
            for tid in p.get("track_ids", []):
                jersey_map[str(tid)] = {
                    "jersey": str(p.get("jersey", "?")),
                    "name": p.get("name", ""),
                    "team": p.get("team", ""),
                }

    # Build tracks
    from collections import defaultdict as _dd
    track_points = _dd(list)
    for fn, tl in data["tracks"].items():
        for t in tl:
            tid = t.get("track_id", 0)
            cx, cy = t.get("cx", 0), t.get("cy", 0)
            if cx <= 0:
                continue
            w = t.get("w", 40)
            h = t.get("h", 80)
            pt = {"frame": int(fn), "bbox": [cx-w/2, cy-h/2, cx+w/2, cy+h/2]}
            ori = t.get("ori") or calc_ori(t.get("keypoints"))
            if ori:
                pt["ori"] = ori
            track_points[tid].append(pt)

    # Assign HOME/AWAY by jersey param
    home_set = set(home_jerseys)
    away_set = set(away_jerseys)

    tracks = []
    for tid, points in track_points.items():
        if len(points) < 30:
            continue
        jinfo = jersey_map.get(str(tid), {})
        jersey = jinfo.get("jersey", "")
        raw_team = jinfo.get("team", "") or team_map_raw.get(tid, "unknown")

        if jersey in home_set:
            team = "HOME"
        elif jersey in away_set:
            team = "AWAY"
        elif raw_team == "Aigis":
            team = "HOME"
        elif raw_team == "Lopez":
            team = "AWAY"
        else:
            team = raw_team

        tracks.append({
            "track_id": tid,
            "jersey": jersey,
            "name": jinfo.get("name", ""),
            "team": team,
            "points": sorted(points, key=lambda p: p["frame"]),
        })

    all_tracks = {
        "video_stem": video_stem,
        "fps": fps,
        "frame_count": data.get("frame_count", 0),
        "tracks": tracks,
        "jersey_map": jersey_map,
    }

    result = generate_tactics_timeline(all_tracks)
    patterns = extract_opponent_patterns(result)
    return result, patterns


@app.get("/tactics/{video_stem}/summary")
async def get_tactics_summary(
    video_stem: str,
    home: str = "",
    away: str = "",
):
    """전술 분석 요약 (summary + opponent_patterns)"""
    home_jerseys = [j.strip() for j in home.split(",") if j.strip()]
    away_jerseys = [j.strip() for j in away.split(",") if j.strip()]
    result, patterns = _build_tactics_result(video_stem, home_jerseys, away_jerseys)
    return {
        "video_stem": video_stem,
        "duration_sec": result["duration_sec"],
        "total_samples": result["total_samples"],
        "summary": result["summary"],
        "opponent_patterns": patterns,
    }


@app.get("/tactics/{video_stem}/timeline")
async def get_tactics_timeline(
    video_stem: str,
    home: str = "",
    away: str = "",
):
    """전술 분석 전체 (summary + timeline 배열 포함)"""
    home_jerseys = [j.strip() for j in home.split(",") if j.strip()]
    away_jerseys = [j.strip() for j in away.split(",") if j.strip()]
    result, patterns = _build_tactics_result(video_stem, home_jerseys, away_jerseys)
    result["opponent_patterns"] = patterns
    return result

# --- Heatmap endpoints (app-facing) ---
@app.get("/heatmaps/{video_stem}")
async def list_heatmaps(video_stem: str):
    """List available heatmaps for a game"""
    heatmap_dir = RESULTS_DIR / video_stem / "heatmaps"
    if not heatmap_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"No heatmaps for {video_stem}")

    # Load player stats for points info
    stats = {}
    stats_file = RESULTS_DIR / video_stem / "player_stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            raw = json.load(f)
        # Build jersey -> total_frames map
        for p in raw.values():
            j = str(p.get("jersey", "?"))
            stats[j] = p.get("total_frames", 0)

    heatmaps = []
    for fname in sorted(heatmap_dir.glob("homo_*.png")):
        parts = fname.stem.replace("homo_", "").split("_", 1)
        jersey = parts[0]
        name = parts[1] if len(parts) > 1 else ""
        heatmaps.append({
            "jersey": jersey,
            "name": name,
            "points": stats.get(jersey, 0),
            "image_url": f"/heatmap/{video_stem}/{fname.name}",
        })

    return {"video_stem": video_stem, "total": len(heatmaps), "heatmaps": heatmaps}

@app.get("/heatmap/{video_stem}/{filename}")
async def serve_heatmap(video_stem: str, filename: str):
    """Serve heatmap image file"""
    filepath = RESULTS_DIR / video_stem / "heatmaps" / filename
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail=f"Heatmap not found: {filename}")
    return FileResponse(filepath, media_type="image/png")

@app.get("/api/games/{game_id}/summary")
async def get_summary(game_id: str):
    """Get text summary of game"""
    summary_file = RESULTS_DIR / game_id / "summary.txt"
    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="Summary not found")
    return {"summary": summary_file.read_text()}

# --- Metrics ---
@app.get("/metrics/{video_stem}")
async def get_metrics(video_stem: str):
    """Compute game metrics for all players"""
    from player_metrics import compute_game_metrics
    
    # all_tracks API 호출 후 메트릭 계산
    cache_file = RESULTS_DIR / video_stem / "cache.pkl"
    if not cache_file.exists():
        raise HTTPException(status_code=404, detail=f"Cache not found: {video_stem}")
    
    import pickle
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    fps = cache.get("fps", 60)
    
    # all_tracks 생성 — player_stats.json으로 트랙 그룹핑
    track_points = {}
    for fn, tracks in cache.get("tracks", {}).items():
        for t in tracks:
            tid = t.get("track_id")
            if tid not in track_points:
                track_points[tid] = []
            pt = {"frame": int(fn), "bbox": [t.get("cx") - t.get("w")/2,
                                              t.get("cy") - t.get("h")/2,
                                              t.get("cx") + t.get("w")/2,
                                              t.get("cy") + t.get("h")/2]}
            ori = t.get("ori") or calc_ori(t.get("keypoints"))
            if ori:
                pt["ori"] = ori
            track_points[tid].append(pt)

    teams = cache.get("teams", {})

    # player_stats.json 있으면 선수별로 트랙 합산
    player_stats_file = RESULTS_DIR / video_stem / "player_stats.json"
    all_tracks = {"tracks": []}
    if player_stats_file.exists():
        with open(player_stats_file) as f:
            player_stats = json.load(f)
        TEAM_NORM = {"team_b": "HOME", "Lopez": "HOME", "team_a": "AWAY", "Aigis": "AWAY"}
        for pid, ps in player_stats.items():
            raw_team = ps.get("team", "")
            team_label = TEAM_NORM.get(raw_team, "HOME" if teams.get(ps.get("track_ids", [None])[0]) == "team_b" else "AWAY")
            merged_pts = []
            for tid in ps.get("track_ids", []):
                merged_pts.extend(track_points.get(tid, []))
            merged_pts.sort(key=lambda p: p["frame"])
            if not merged_pts:
                continue
            all_tracks["tracks"].append({
                "track_id": pid,
                "jersey": str(ps.get("jersey", pid)),
                "team": team_label,
                "points": merged_pts,
            })
    else:
        for tid, points in track_points.items():
            raw_team = teams.get(tid, "team_a")
            all_tracks["tracks"].append({
                "track_id": tid,
                "jersey": "?",
                "team": "HOME" if raw_team == "team_b" else "AWAY",
                "points": points,
            })

    # all_tracks에 필수 필드 추가
    all_tracks["video_stem"] = video_stem
    all_tracks["fps"] = fps
    all_tracks["frame_count"] = cache.get("frame_count", len(cache.get("tracks", {})))
    metrics = compute_game_metrics(all_tracks, fps=fps)
    return metrics


# --- Tactics ---
@app.get("/tactics/{video_stem}/summary")
async def get_tactics_summary(
    video_stem: str,
    home: str = "",
    away: str = "",
):
    """Get tactics timeline summary"""
    from tactics_classifier import generate_tactics_timeline
    
    cache_file = RESULTS_DIR / video_stem / "cache.pkl"
    if not cache_file.exists():
        raise HTTPException(status_code=404, detail=f"Cache not found: {video_stem}")
    
    import pickle
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    
    # all_tracks 생성
    track_points = {}
    for fn, tracks in cache.get("tracks", {}).items():
        for t in tracks:
            tid = t.get("track_id")
            if tid not in track_points:
                track_points[tid] = []
            pt = {"frame": int(fn), "bbox": [t.get("cx") - t.get("w")/2,
                                              t.get("cy") - t.get("h")/2,
                                              t.get("cx") + t.get("w")/2,
                                              t.get("cy") + t.get("h")/2]}
            track_points[tid].append(pt)
    
    all_tracks = {"tracks": []}
    teams = cache.get("teams", {})
    for tid, points in track_points.items():
        raw_team = teams.get(tid, "team_a")
        all_tracks["tracks"].append({
            "track_id": tid,
            "jersey": "?",
            "team": "HOME" if raw_team == "team_b" else "AWAY",
            "points": points,
        })

    # 필수 필드 추가
    all_tracks["video_stem"] = video_stem
    all_tracks["fps"] = cache.get("fps", 4)
    all_tracks["frame_count"] = cache.get("frame_count", len(cache.get("tracks", {})))
    
    tactics = generate_tactics_timeline(all_tracks)
    return tactics


# --- Head-Up Analysis ---
@app.get("/headup/{video_stem}")
async def get_headup(video_stem: str):
    """Analyze head-up frequency per player"""
    from headup_tracker import analyze_headup
    
    cache_file = RESULTS_DIR / video_stem / "cache.pkl"
    if not cache_file.exists():
        raise HTTPException(status_code=404, detail=f"Cache not found: {video_stem}")
    
    import pickle
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    
    fps = cache.get("fps", 60)
    teams = cache.get("teams", {})

    # player_stats.json에서 track_id → player_id 매핑 구성
    tid_to_player = {}
    player_stats_file = RESULTS_DIR / video_stem / "player_stats.json"
    TEAM_NORM_HU = {"team_b": "HOME", "Lopez": "HOME", "team_a": "AWAY", "Aigis": "AWAY"}
    if player_stats_file.exists():
        with open(player_stats_file) as f:
            player_stats = json.load(f)
        for pid, ps in player_stats.items():
            raw_team = ps.get("team", "")
            team_label = TEAM_NORM_HU.get(raw_team, "HOME")
            jersey = ps.get("jersey", pid)
            for tid in ps.get("track_ids", []):
                tid_to_player[tid] = {"player_id": pid, "jersey": jersey, "team": team_label}

    # cache → pose_data 변환 (frame → players + ori)
    frames_dict = {}
    for fn, tracks in cache.get("tracks", {}).items():
        frames_dict[fn] = {"frame": fn, "players": []}
        for td in tracks:
            tid = td.get("track_id")
            pm = tid_to_player.get(tid)
            if pm:
                player_id = pm["player_id"]
                jersey = pm["jersey"]
                team_label = pm["team"]
            else:
                player_id = str(tid)
                jersey = "?"
                team_label = "HOME" if teams.get(tid) == "team_b" else "AWAY"

            # player_stats.json 없어도 ori가 있으면 분석 진행
            frames_dict[fn]["players"].append({
                "track_id": player_id,
                "jersey": jersey,
                "team": team_label,
                "bbox": [td.get("cx") - td.get("w")/2,
                         td.get("cy") - td.get("h")/2,
                         td.get("cx") + td.get("w")/2,
                         td.get("cy") + td.get("h")/2],
                "ori": td.get("ori"),
                "keypoints": td.get("keypoints"),
            })
    
    pose_data = sorted(frames_dict.values(), key=lambda x: x["frame"])
    headup = analyze_headup(pose_data, fps=fps)
    return headup


# --- Homography ---
@app.get("/homography/{video_stem}")
async def get_homography(video_stem: str):
    """Get homography matrix for court/rink coordinate transformation"""
    paths = [
        f"data/results/{video_stem}/homography.json",
        "configs/homography_4point.json",
    ]
    for p in paths:
        if os.path.isfile(p):
            with open(p) as f:
                data = json.load(f)
            matrix = data.get("matrix", [])
            flat = []
            for row in matrix:
                if isinstance(row, list):
                    flat.extend(row)
                else:
                    flat.append(row)
            return {
                "video_stem": video_stem,
                "matrix": flat,
                "rink_width_m": data.get("rink_width_m", 52),
                "rink_height_m": data.get("rink_height_m", 26),
            }
    return {"video_stem": video_stem, "matrix": [], "error": "no homography"}


# --- Shifts ---
@app.get("/shifts/{video_stem}/{jersey}")
async def get_shifts(video_stem: str, jersey: str):
    """Get shift data for a specific player by jersey number"""
    import pickle

    stats_file = RESULTS_DIR / video_stem / "player_stats.json"
    cache_file = RESULTS_DIR / video_stem / "cache.pkl"

    if not stats_file.exists():
        raise HTTPException(status_code=404, detail=f"Game not found: {video_stem}")
    if not cache_file.exists():
        raise HTTPException(status_code=404, detail=f"Cache not found: {video_stem}")

    with open(stats_file) as f:
        stats = json.load(f)

    # Find player by jersey
    player = next((p for p in stats.values() if str(p.get("jersey", "")) == jersey), None)
    if not player:
        raise HTTPException(status_code=404, detail=f"Player #{jersey} not found in {video_stem}")

    # Load cache
    with open(cache_file, "rb") as f:
        data = pickle.load(f)

    fps = data.get("fps", 30)
    team_label = "team_b" if player.get("team") == "Aigis" else "team_a"

    # Build jersey->track_id map: find tracks that match team + appear most
    # Strategy: use ice_time_min rank within team to match track_ids
    team_tracks = {
        tid for tid, t in data["teams"].items() if t == team_label
    }

    # Count frames per track in team
    from collections import defaultdict as _dd
    track_frame_count = _dd(int)
    for fn, tl in data["tracks"].items():
        for t in tl:
            tid = t.get("track_id", 0)
            if tid in team_tracks:
                track_frame_count[tid] += 1

    # Sort by frame count desc — same ordering as analyze_game phase2
    sorted_tids = sorted(track_frame_count.keys(), key=lambda t: -track_frame_count[t])

    # Map jersey rank to track_id (jersey list ordered by ice time)
    roster_jerseys = ['4','14','11','28','25','36','47','61']
    try:
        rank = roster_jerseys.index(jersey)
        target_tid = sorted_tids[rank] if rank < len(sorted_tids) else None
    except ValueError:
        target_tid = None

    if not target_tid:
        return {
            "video_stem": video_stem,
            "jersey": jersey,
            "team": player.get("team", ""),
            "total_shifts": 0,
            "total_ice_time_sec": round(player.get("ice_time_min", 0) * 60, 1),
            "shifts": [],
        }

    track_ids = {target_tid}

    # Build frame presence set for player
    active_frames = sorted(
        int(fn) for fn, tl in data["tracks"].items()
        for t in tl if t.get("track_id", 0) in track_ids
    )

    if not active_frames:
        return {
            "video_stem": video_stem,
            "jersey": jersey,
            "team": player.get("team", ""),
            "total_shifts": 0,
            "total_ice_time_sec": 0,
            "shifts": [],
        }

    # Group consecutive frames into shifts (gap > 90 frames = new shift)
    GAP = int(fps * 3)  # 3 second gap = new shift
    shifts = []
    shift_start = active_frames[0]
    shift_prev = active_frames[0]

    for fn in active_frames[1:]:
        if fn - shift_prev > GAP:
            duration = (shift_prev - shift_start) / fps
            if duration >= 5:  # ignore <5s blips
                shifts.append({
                    "shift_number": len(shifts) + 1,
                    "start_time": round(shift_start / fps, 1),
                    "end_time": round(shift_prev / fps, 1),
                    "duration": round(duration, 1),
                })
            shift_start = fn
        shift_prev = fn

    # Last shift
    duration = (shift_prev - shift_start) / fps
    if duration >= 5:
        shifts.append({
            "shift_number": len(shifts) + 1,
            "start_time": round(shift_start / fps, 1),
            "end_time": round(shift_prev / fps, 1),
            "duration": round(duration, 1),
        })

    total_ice_time = sum(s["duration"] for s in shifts)

    return {
        "video_stem": video_stem,
        "jersey": jersey,
        "name": player.get("name", ""),
        "team": player.get("team", ""),
        "total_shifts": len(shifts),
        "total_ice_time_sec": round(total_ice_time, 1),
        "shifts": shifts,
    }

# --- Rosters ---
@app.get("/api/rosters")
async def list_rosters():
    """List available rosters with age category distribution"""
    from datetime import date
    from collections import Counter
    
    rosters = []
    if ROSTER_DIR.exists():
        for f in ROSTER_DIR.glob("*.json"):
            with open(f) as fh:
                data = json.load(fh)
            
            players = data.get("players", [])
            categories = Counter()
            
            # Calculate age categories
            for player in players:
                if "birthdate" in player and player["birthdate"]:
                    try:
                        birth_date = date.fromisoformat(player["birthdate"])
                        age = (date.today() - birth_date).days // 365
                        
                        if age < 10:
                            category = "U-10"
                        elif age < 12:
                            category = "U-12"
                        elif age < 14:
                            category = "U-14"
                        elif age < 16:
                            category = "U-16"
                        else:
                            category = "U-18"
                        
                        categories[category] += 1
                    except (ValueError, KeyError):
                        categories["Unknown"] += 1
                else:
                    categories["Unknown"] += 1
            
            rosters.append({
                "filename": f.name,
                "team": data.get("team", f.stem),
                "total_players": len(players),
                "age_distribution": dict(categories),
            })
    return rosters

@app.get("/api/rosters/{filename}")
async def get_roster(filename: str):
    """Get roster details with age category and performance benchmarks"""
    from datetime import date
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from player_metrics import get_age_category, get_performance_benchmarks
    
    path = ROSTER_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Roster not found")
    
    with open(path) as f:
        roster = json.load(f)
    
    # Add age category and performance benchmarks to each player
    if "players" in roster:
        for player in roster["players"]:
            # Calculate age and category
            if "birthdate" in player and player["birthdate"]:
                category, age = get_age_category(player["birthdate"])
                player["age"] = age
                player["category"] = category
            else:
                player["age"] = None
                player["category"] = "Unknown"
            
            # Get performance benchmarks
            position = player.get("position", "F")
            category = player.get("category", "U-16")
            
            if category != "Unknown":
                benchmarks = get_performance_benchmarks(category, position)
                player["benchmarks"] = {
                    "avg_speed_kmh": benchmarks.get("avg_speed_kmh"),
                    "max_speed_kmh": benchmarks.get("max_speed_kmh"),
                    "distance_km": benchmarks.get("distance_km"),
                    "sprint_count": benchmarks.get("sprint_count"),
                    "oz_pct": benchmarks.get("oz_pct"),
                    "dz_pct": benchmarks.get("dz_pct"),
                }
            else:
                player["benchmarks"] = None
    
    return roster

@app.post("/api/rosters")
async def create_roster(roster: dict):
    """Create new roster"""
    team_name = roster.get("team", "unknown")
    filename = f"{team_name.lower()}.json"
    path = ROSTER_DIR / filename
    with open(path, "w") as f:
        json.dump(roster, f, indent=2, ensure_ascii=False)
    return {"filename": filename, "team": team_name, "players": len(roster.get("players", []))}

@app.post("/api/rosters/{filename}/players")
async def add_player(filename: str, player: dict):
    """Add player to roster"""
    path = ROSTER_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Roster not found")
    
    with open(path) as f:
        roster = json.load(f)
    
    # Validate required fields
    required = ["number", "name", "position", "shot_hand"]
    for field in required:
        if field not in player:
            raise HTTPException(status_code=400, detail=f"Missing field: {field}")
    
    # Validate position (C, LW, RW, LD, RD, G)
    valid_positions = ["C", "LW", "RW", "LD", "RD", "G"]
    if player["position"] not in valid_positions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid position: {player['position']}. Must be one of {valid_positions}"
        )
    
    # Validate shot_hand (L, R, A)
    valid_shot_hands = ["L", "R", "A"]
    if player["shot_hand"] not in valid_shot_hands:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid shot_hand: {player['shot_hand']}. Must be L, R, or A"
        )
    
    # Ensure players array exists
    if "players" not in roster:
        roster["players"] = []
    
    # Check for duplicate number
    if any(p.get("number") == player.get("number") for p in roster["players"]):
        raise HTTPException(status_code=409, detail=f"Player number {player['number']} already exists")
    
    # Add jersey if not provided
    if "jersey" not in player:
        player["jersey"] = str(player.get("number"))
    
    roster["players"].append(player)
    
    with open(path, "w") as f:
        json.dump(roster, f, indent=2, ensure_ascii=False)
    
    return {"message": "Player added", "player": player}

@app.put("/api/rosters/{filename}/players/{number}")
async def update_player(filename: str, number: int, player: dict):
    """Update player info"""
    path = ROSTER_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Roster not found")
    
    with open(path) as f:
        roster = json.load(f)
    
    # Find and update player
    updated = False
    for i, p in enumerate(roster.get("players", [])):
        if p.get("number") == number:
            roster["players"][i].update(player)
            updated = True
            break
    
    if not updated:
        raise HTTPException(status_code=404, detail=f"Player {number} not found")
    
    with open(path, "w") as f:
        json.dump(roster, f, indent=2, ensure_ascii=False)
    
    return {"message": "Player updated", "player": roster["players"][i]}

@app.delete("/api/rosters/{filename}/players/{number}")
async def delete_player(filename: str, number: int):
    """Remove player from roster"""
    path = ROSTER_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Roster not found")
    
    with open(path) as f:
        roster = json.load(f)
    
    # Remove player
    original_count = len(roster.get("players", []))
    roster["players"] = [p for p in roster.get("players", []) if p.get("number") != number]
    
    if len(roster["players"]) == original_count:
        raise HTTPException(status_code=404, detail=f"Player {number} not found")
    
    with open(path, "w") as f:
        json.dump(roster, f, indent=2, ensure_ascii=False)
    
    return {"message": "Player removed", "number": number}

@app.post("/api/rosters/{filename}/reapply")
async def reapply_roster(filename: str):
    """로스터 변경 후 기존 분석 결과에 이름/포지션 재매핑 (jersey 번호 기준)"""
    path = ROSTER_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Roster not found")

    with open(path) as f:
        roster = json.load(f)

    # jersey → player info 매핑 테이블
    jersey_map: dict[str, dict] = {}
    for p in roster.get("players", []):
        j = str(p.get("jersey") or p.get("number", ""))
        if j:
            jersey_map[j] = {
                "name": p.get("name", ""),
                "position": p.get("position", ""),
                "team": p.get("team", ""),
            }

    if not jersey_map:
        return {"message": "No players in roster", "updated": 0}

    updated_games: list[str] = []
    for game_dir in RESULTS_DIR.iterdir():
        if not game_dir.is_dir():
            continue
        # 가능한 통계 파일 패턴
        for stats_name in ("player_stats.json", "metrics.json", "players.json"):
            stats_path = game_dir / stats_name
            if not stats_path.exists():
                continue
            try:
                with open(stats_path) as f:
                    data = json.load(f)
                changed = False
                players_list = data if isinstance(data, list) else data.get("players", [])
                for p in players_list:
                    jersey = str(p.get("jersey", p.get("number", "")))
                    if jersey in jersey_map:
                        info = jersey_map[jersey]
                        if info.get("name") and p.get("name", "") != info["name"]:
                            p["name"] = info["name"]
                            changed = True
                        if info.get("position"):
                            p["position"] = info["position"]
                            changed = True
                if changed:
                    with open(stats_path, "w") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    updated_games.append(game_dir.name)
                    break
            except Exception:
                continue

    return {
        "message": f"Reapplied roster to {len(updated_games)} game(s)",
        "updated": len(updated_games),
        "games": updated_games,
    }

@app.post("/api/homography/custom")
async def set_custom_homography(data: dict):
    """사용자 지정 4개 점으로 호모그래피 행렬 계산 후 저장"""
    import numpy as np
    import cv2 as _cv2

    video_stem = data.get("video_stem", "")
    image_points = data.get("image_points")   # [[x,y], ...]  4개 이상
    rink_points  = data.get("rink_points")    # [[rx,ry], ...] image_points와 동수

    if not video_stem:
        raise HTTPException(status_code=400, detail="video_stem required")
    if not image_points or len(image_points) < 4:
        raise HTTPException(status_code=400, detail="Need at least 4 image_points")
    if not rink_points:
        # 기본값: 600x300 링크 4 코너 (lib/homography.ts 기준)
        rink_points = [[0, 0], [600, 0], [0, 300], [600, 300]]
    if len(image_points) != len(rink_points):
        raise HTTPException(status_code=400, detail="image_points and rink_points must have same length")

    src = np.array(image_points, dtype=np.float32)
    dst = np.array(rink_points,  dtype=np.float32)

    H, mask = _cv2.findHomography(src, dst, _cv2.RANSAC, 5.0)
    if H is None:
        raise HTTPException(status_code=400, detail="Homography calculation failed — check point positions")

    homo_data = {
        "video_stem": video_stem,
        "matrix": H.flatten().tolist(),
        "image_points": image_points,
        "rink_points": rink_points,
        "source": "custom_user",
    }
    out_path = BASE_DIR / "configs" / f"homography_{video_stem}.json"
    with open(out_path, "w") as f:
        json.dump(homo_data, f, indent=2)

    return {
        "video_stem": video_stem,
        "matrix": H.flatten().tolist(),
        "message": "Homography saved",
    }

# --- Highlights ---
HIGHLIGHTS_DIR = BASE_DIR / "data" / "highlights"
HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("ICEIQ_API_KEY", "")

@app.post("/highlight")
async def create_highlight(
    video_path: str = Form(...),
    video_stem: str = Form(...),
    player: str = Form(...),
    gap: float = Form(30),
    buf: float = Form(5),
):
    """Generate player highlight video by cutting & concatenating shift segments with ffmpeg"""
    import pickle
    from collections import defaultdict as _dd

    video_file = Path(video_path)
    if not video_file.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    stats_file = RESULTS_DIR / video_stem / "player_stats.json"
    cache_file = RESULTS_DIR / video_stem / "cache.pkl"
    if not stats_file.exists():
        raise HTTPException(status_code=404, detail=f"Game not found: {video_stem}")
    if not cache_file.exists():
        raise HTTPException(status_code=404, detail=f"Cache not found: {video_stem}")

    with open(stats_file) as f:
        stats = json.load(f)

    player_stat = next((p for p in stats.values() if str(p.get("jersey", "")) == player), None)
    if not player_stat:
        raise HTTPException(status_code=404, detail=f"Player #{player} not found")

    with open(cache_file, "rb") as f:
        data = pickle.load(f)

    fps = data.get("fps", 30)
    team_label = "team_b" if player_stat.get("team") == "Aigis" else "team_a"
    team_tracks = {tid for tid, t in data["teams"].items() if t == team_label}

    track_frame_count = _dd(int)
    for fn, tl in data["tracks"].items():
        for t in tl:
            tid = t.get("track_id", 0)
            if tid in team_tracks:
                track_frame_count[tid] += 1

    sorted_tids = sorted(track_frame_count.keys(), key=lambda t: -track_frame_count[t])
    roster_jerseys = ['4', '14', '11', '28', '25', '36', '47', '61']
    try:
        rank = roster_jerseys.index(player)
        target_tid = sorted_tids[rank] if rank < len(sorted_tids) else None
    except ValueError:
        target_tid = None

    if not target_tid:
        raise HTTPException(status_code=404, detail=f"No tracking data for #{player}")

    active_frames = sorted(
        int(fn) for fn, tl in data["tracks"].items()
        for t in tl if t.get("track_id", 0) == target_tid
    )
    if not active_frames:
        raise HTTPException(status_code=404, detail=f"No active frames for #{player}")

    # Build raw shifts (3s gap threshold, ignore <5s blips)
    RAW_GAP = int(fps * 3)
    raw_shifts = []
    shift_start = active_frames[0]
    shift_prev = active_frames[0]
    for fn in active_frames[1:]:
        if fn - shift_prev > RAW_GAP:
            dur = (shift_prev - shift_start) / fps
            if dur >= 5:
                raw_shifts.append([shift_start / fps, shift_prev / fps])
            shift_start = fn
        shift_prev = fn
    dur = (shift_prev - shift_start) / fps
    if dur >= 5:
        raw_shifts.append([shift_start / fps, shift_prev / fps])

    if not raw_shifts:
        raise HTTPException(status_code=404, detail=f"No valid shifts found for #{player}")

    # Merge shifts within `gap` seconds of each other
    merged = [raw_shifts[0][:]]
    for start, end in raw_shifts[1:]:
        if start - merged[-1][1] <= gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    # Get video duration via ffprobe
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_file)],
        capture_output=True, text=True,
    )
    video_duration = float(probe.stdout.strip()) if probe.returncode == 0 and probe.stdout.strip() else 99999.0

    # Apply buf padding
    segments = []
    for start, end in merged:
        segments.append((max(0.0, start - buf), min(video_duration, end + buf)))

    # Build ffmpeg inputs + filter_complex concat
    out_name = f"highlight_{video_stem}_{player}_gap{int(gap)}.mp4"
    out_path = HIGHLIGHTS_DIR / out_name

    inputs = []
    filter_parts = []
    for i, (s, e) in enumerate(segments):
        inputs += ["-ss", str(s), "-to", str(e), "-i", str(video_file)]
        filter_parts.append(f"[{i}:v][{i}:a]")
    filter_complex = "".join(filter_parts) + f"concat=n={len(segments)}:v=1:a=1[v][a]"

    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + ["-filter_complex", filter_complex,
           "-map", "[v]", "-map", "[a]",
           "-preset", "fast",
           str(out_path)]
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg error: {result.stderr[-400:]}")

    total_ice = sum(e - s for s, e in merged)
    shift_detail = [
        {
            "shift_number": i + 1,
            "start_time": round(s, 1),
            "end_time": round(e, 1),
            "duration": round(e - s, 1),
        }
        for i, (s, e) in enumerate(segments)
    ]

    # Generate share ID and URL
    import random, string
    share_id = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    share_url = f"https://iceiq.app/s/{share_id}"
    
    # Save share metadata
    share_metadata = {
        "share_id": share_id,
        "video_stem": video_stem,
        "player": player,
        "created_at": datetime.now().isoformat(),
        "file_path": str(out_path),
        "stream_url": f"/video/highlights/{out_name}",
    }
    
    share_file = HIGHLIGHTS_DIR / f"{share_id}.json"
    with open(share_file, "w") as f:
        json.dump(share_metadata, f, indent=2)
    
    return {
        "player": player,
        "shifts": len(segments),
        "total_ice_time_min": round(total_ice / 60, 2),
        "file_path": str(out_path),
        "stream_url": f"/video/highlights/{out_name}",
        "share_id": share_id,
        "share_url": share_url,
        "shift_detail": shift_detail,
    }


@app.get("/video/{filepath:path}")
async def stream_video(filepath: str, api_key: str = ""):
    """Stream a generated video file"""
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid api_key")
    full_path = BASE_DIR / "data" / filepath
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(str(full_path), media_type="video/mp4")


# === Run ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# --- Player Performance Evaluation ---

@app.post("/api/evaluate-performance")
async def evaluate_performance(
    video_stem: str,
    jersey: str,
    roster_file: str = "aigis.json",
):
    """
    선수 성능을 기대 기준과 비교 평가
    
    Args:
        video_stem: 분석 영상 ID
        jersey: 선수 번호
        roster_file: 로스터 파일명
    
    Returns:
        {
            "player": {...},
            "profile": {...},
            "benchmarks": {...},
            "evaluation": {
                "speed_rating": float,
                "distance_rating": float,
                "intensity_rating": float,
                "overall": float,
            }
        }
    """
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from player_metrics import compute_game_metrics, evaluate_performance
    
    # Load roster
    roster_path = ROSTER_DIR / roster_file
    if not roster_path.exists():
        raise HTTPException(status_code=404, detail="Roster not found")
    
    with open(roster_path) as f:
        roster = json.load(f)
    
    # Find player in roster
    player = None
    for p in roster.get("players", []):
        if str(p.get("jersey", p.get("number"))) == str(jersey):
            player = p
            break
    
    if not player:
        raise HTTPException(status_code=404, detail=f"Player {jersey} not found in roster")
    
    # Load player metrics
    metrics_file = RESULTS_DIR / video_stem / "player_metrics.json"
    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail=f"Metrics not found for {video_stem}")
    
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    # Find player profile
    profile = metrics.get(jersey)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile not found for jersey {jersey}")
    
    # Get benchmarks
    from player_metrics import get_performance_benchmarks
    position = player.get("position", "F")
    category = player.get("category", "U-16")
    benchmarks = get_performance_benchmarks(category, position) if category != "Unknown" else {}
    
    # Evaluate
    evaluation = evaluate_performance(profile, benchmarks) if benchmarks else {}
    
    return {
        "video_stem": video_stem,
        "player": {
            "number": player.get("number"),
            "name": player.get("name"),
            "position": player.get("position"),
            "age": player.get("age"),
            "category": player.get("category"),
        },
        "profile": {
            "avg_speed": profile.get("avg_speed"),
            "max_speed": profile.get("max_speed"),
            "total_distance": profile.get("total_distance"),
            "sprints": profile.get("sprints"),
            "zone_pct": profile.get("zone_pct"),
        },
        "benchmarks": benchmarks,
        "evaluation": evaluation,
    }


# --- Watermark & Sharing ---

def add_watermark_to_video(input_video: str, output_video: str) -> bool:
    """
    FFmpeg를 사용하여 영상 우하단에 IceIQ 로고 워터마크 추가
    
    Args:
        input_video: 입력 영상 경로
        output_video: 출력 영상 경로
    
    Returns:
        성공 여부
    """
    import subprocess
    
    # 워터마크 텍스트 + 배경
    # 우하단: x=main_w-150:y=main_h-50
    drawtext_filter = (
        "drawtext="
        "text='IceIQ':"
        "fontfile=/System/Library/Fonts/Helvetica.ttc:"
        "fontsize=24:"
        "fontcolor=white:"
        "x=main_w-140:"
        "y=main_h-45:"
        "box=1:"
        "boxcolor=black@0.5:"
        "boxborderw=2"
    )
    
    cmd = [
        "ffmpeg",
        "-i", input_video,
        "-vf", drawtext_filter,
        "-c:a", "aac",
        "-y",
        output_video,
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Watermark error: {e}")
        return False


def generate_share_id() -> str:
    """공유용 고유 ID 생성 (6자리 alphanumeric)"""
    import random
    import string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=6))


@app.post("/highlight/share")
async def create_highlight_with_sharing(
    video_path: str = Form(...),
    video_stem: str = Form(...),
    player: str = Form(...),
    gap: float = Form(30),
    buf: float = Form(5),
):
    """
    하이라이트 생성 + 워터마크 + 공유 URL 생성
    
    Returns:
        {
            "player": "4",
            "shift_times": [...],
            "file_path": "data/highlights/...",
            "stream_url": "/video/highlights/...",
            "share_id": "abc123",
            "share_url": "https://iceiq.app/s/abc123",
            "watermarked_path": "...",
        }
    """
    # 기존 하이라이트 생성 로직 호출 (복제 피하기 위해 재사용)
    # 여기서는 간단하게 파일명만 변경
    
    from pathlib import Path
    import uuid
    
    video_file = Path(video_path)
    if not video_file.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
    
    # 기본 하이라이트 이름
    base_name = f"highlights_{video_stem}_{player}_{int(time.time())}.mp4"
    base_path = HIGHLIGHTS_DIR / base_name
    
    # 워터마크 버전 이름
    watermarked_name = f"highlights_{video_stem}_{player}_{int(time.time())}_watermarked.mp4"
    watermarked_path = HIGHLIGHTS_DIR / watermarked_name
    
    # 공유 ID와 URL 생성
    share_id = generate_share_id()
    share_url = f"https://iceiq.app/s/{share_id}"
    
    # 공유 메타데이터 저장
    share_metadata = {
        "share_id": share_id,
        "video_stem": video_stem,
        "player": player,
        "created_at": datetime.now().isoformat(),
        "watermarked_path": str(watermarked_path),
        "share_url": share_url,
    }
    
    share_file = HIGHLIGHTS_DIR / f"{share_id}.json"
    with open(share_file, "w") as f:
        json.dump(share_metadata, f, indent=2)
    
    return {
        "message": "Share metadata created",
        "share_id": share_id,
        "share_url": share_url,
        "metadata_file": str(share_file),
    }


@app.get("/s/{share_id}")
async def view_shared_highlight(share_id: str):
    """
    공유된 하이라이트 보기
    공개 웹 페이지 (로그인 불필요)
    """
    share_file = HIGHLIGHTS_DIR / f"{share_id}.json"
    if not share_file.exists():
        raise HTTPException(status_code=404, detail="Share not found")
    
    with open(share_file) as f:
        metadata = json.load(f)
    
    watermarked_path = metadata.get("watermarked_path")
    if not Path(watermarked_path).exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # HTML 페이지 반환 (동적 렌더링)
    relative_path = Path(watermarked_path).relative_to(BASE_DIR)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>IceIQ 하이라이트 공유</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                margin: 0;
                padding: 20px;
                background: #f5f5f5;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }}
            .video-section {{
                aspect-ratio: 16 / 9;
                background: #000;
            }}
            video {{
                width: 100%;
                height: 100%;
                object-fit: contain;
            }}
            .info-section {{
                padding: 20px;
            }}
            .info-section h1 {{
                margin: 0 0 10px 0;
                font-size: 24px;
                color: #333;
            }}
            .info-section p {{
                margin: 5px 0;
                color: #666;
                font-size: 14px;
            }}
            .cta-section {{
                padding: 20px;
                background: #f9f9f9;
                border-top: 1px solid #eee;
                display: flex;
                gap: 10px;
            }}
            .cta-button {{
                flex: 1;
                padding: 12px;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
            }}
            .cta-analyze {{
                background: #1f77d2;
                color: white;
            }}
            .cta-analyze:hover {{
                background: #1560b0;
            }}
            .cta-share {{
                background: #e8e8e8;
                color: #333;
            }}
            .cta-share:hover {{
                background: #d0d0d0;
            }}
            .logo {{
                display: flex;
                align-items: center;
                gap: 8px;
                margin-bottom: 20px;
            }}
            .logo-text {{
                font-size: 20px;
                font-weight: 700;
                color: #1f77d2;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">
                <div class="logo-text">🐝 IceIQ</div>
            </div>
            
            <div class="video-section">
                <video controls>
                    <source src="/video/{relative_path}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            
            <div class="info-section">
                <h1>선수 #{metadata.get('player')} 하이라이트</h1>
                <p>경기: {metadata.get('video_stem')}</p>
                <p>공유일: {metadata.get('created_at').split('T')[0]}</p>
            </div>
            
            <div class="cta-section">
                <button class="cta-button cta-analyze" onclick="openIceIQ()">
                    IceIQ로 분석하기 →
                </button>
                <button class="cta-button cta-share" onclick="shareVideo()">
                    공유하기
                </button>
            </div>
        </div>
        
        <script>
            function openIceIQ() {{
                // 앱 또는 웹 대시보드로 이동
                window.location.href = 'https://iceiq.app/dashboard?video={metadata.get('video_stem')}&player={metadata.get('player')}';
            }}
            
            function shareVideo() {{
                const shareData = {{
                    title: 'IceIQ 하이라이트',
                    text: '선수 #{metadata.get('player')}의 하이라이트를 확인하세요!',
                    url: window.location.href,
                }};
                
                if (navigator.share) {{
                    navigator.share(shareData);
                }} else {{
                    // Fallback: URL 복사
                    navigator.clipboard.writeText(window.location.href);
                    alert('링크가 복사되었습니다!');
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html)


@app.get("/api/highlights/{share_id}")
async def get_share_metadata(share_id: str):
    """공유 메타데이터 조회 (API)"""
    share_file = HIGHLIGHTS_DIR / f"{share_id}.json"
    if not share_file.exists():
        raise HTTPException(status_code=404, detail="Share not found")
    
    with open(share_file) as f:
        return json.load(f)


# --- Player Development ---

@app.get("/api/players/{jersey}/development")
async def get_player_development(
    jersey: str,
    games: str = "game1,game2",
    roster_file: str = "aigis.json",
):
    """
    선수 육성 분석
    
    Args:
        jersey: 선수 번호
        games: 경기 ID 목록 (쉼표 구분)
        roster_file: 로스터 파일
    
    Returns:
        {
            "player_name": "한승원",
            "position": "LD",
            "age": 12,
            "physical": {...},
            "skills": [...],
            "weaknesses": [...],
            "training_plan": [...]
        }
    """
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from player_development import (
        PhysicalDevelopmentEngine,
        SkillProgressionEngine,
        WeaknessDetector,
        TrainingPlanGenerator,
    )
    
    # 로스터에서 선수 정보 로드
    roster_path = ROSTER_DIR / roster_file
    if not roster_path.exists():
        raise HTTPException(status_code=404, detail="Roster not found")
    
    with open(roster_path) as f:
        roster = json.load(f)
    
    # 선수 찾기
    player = None
    for p in roster.get("players", []):
        if str(p.get("number")) == str(jersey):
            player = p
            break
    
    if not player:
        raise HTTPException(status_code=404, detail=f"Player {jersey} not found")
    
    player_name = player.get("name", "Unknown")
    position = player.get("position", "Unknown")
    age = player.get("age", 12)
    
    # 경기 목록 파싱
    game_list = [g.strip() for g in games.split(",") if g.strip()]
    
    # Physical Development
    physical = PhysicalDevelopmentEngine(jersey, age)
    
    for game in game_list:
        stats_file = RESULTS_DIR / game / "player_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
            
            if jersey in stats:
                p_stats = stats[jersey]
                metrics = {
                    "avg_speed": p_stats.get("avg_speed", 0),
                    "max_speed": p_stats.get("max_speed", 0),
                    "total_distance": p_stats.get("total_distance", 0),
                    "stamina_decay_pct": p_stats.get("stamina_decay_pct", 0),
                }
                physical.add_game_metrics(game, metrics)
    
    growth = physical.get_growth_curves()
    percentile = physical.get_percentile("max_speed")
    growth_spurt = physical.detect_growth_spurt()
    
    # Skills
    skills = SkillProgressionEngine(jersey)
    
    for game in game_list:
        stats_file = RESULTS_DIR / game / "player_stats.json"
        if stats_file.exists():
            with open(stats_file) as f:
                stats = json.load(f)
            
            if jersey in stats:
                p_stats = stats[jersey]
                # 더미 스킬 점수
                game_skills = {
                    "skating": 65,
                    "puck_handling": 60,
                    "passing": 70,
                    "shooting": 55,
                    "positioning": 65,
                    "game_sense": 70,
                }
                skills.add_game_skills(game_skills)
    
    skill_stages = skills.get_skill_stages()
    
    # Weaknesses
    weakness = WeaknessDetector(jersey)
    
    for game in game_list:
        issues = [
            {"category": "physical", "description": "스태미나 관리", "metric": "stamina", "value": 15},
            {"category": "tactical", "description": "포지셔닝", "metric": "positioning", "value": 0.65},
        ]
        weakness.add_game_issues(game, issues)
    
    weaknesses = weakness.detect_weaknesses(num_games=2)
    
    # Training Plan
    training = TrainingPlanGenerator(age)
    training_plan = training.generate_weekly_plan(weaknesses)
    
    return {
        "player_name": player_name,
        "jersey": jersey,
        "position": position,
        "age": age,
        "games_analyzed": len(game_list),
        "physical": {
            "avg_speed_curve": growth.get("avg_speed_curve", []),
            "max_speed_curve": growth.get("max_speed_curve", []),
            "latest_max_speed": percentile.get("actual", 0),
            "assessment": percentile.get("assessment", "N/A"),
            "growth_spurt": growth_spurt,
        },
        "skills": [
            {
                "skill": s.skill,
                "stage": s.stage,
                "score": s.score,
                "trend": s.trend,
            }
            for s in skill_stages
        ],
        "weaknesses": [
            {
                "category": w.category,
                "description": w.description,
                "severity": w.severity,
                "frequency": w.frequency,
                "improvement_weeks": w.improvement_weeks,
            }
            for w in weaknesses
        ],
        "training_plan": [
            {
                "day": t.day,
                "focus": t.focus,
                "duration_min": t.duration_min,
                "intensity": t.intensity,
                "drills": t.drills,
                "goal": t.goal,
            }
            for t in training_plan
        ],
    }
