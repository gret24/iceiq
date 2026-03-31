#!/usr/bin/env python3
"""
IceIQ Video Analysis API
Pipeline: upload → extract.py → detect.py → ocr.py → highlight
"""

import os
import re
import shutil
import subprocess
import threading
from collections import Counter
from typing import Optional

import aiofiles
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="IceIQ Video Analysis API", version="2.0.0")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
VENV_PY   = os.path.join(BASE_DIR, "venv", "bin", "python")
UPLOADS   = os.path.join(BASE_DIR, "uploads")
CLIPS_DIR = os.path.join(BASE_DIR, "clips")
VIDEO_PATH      = os.path.join(BASE_DIR, "test.mp4")
HIGHLIGHT_PATH  = os.path.join(BASE_DIR, "highlight.mp4")
RESULTS_TXT     = os.path.join(BASE_DIR, "results.txt")
TIMESTAMPS_TXT  = os.path.join(BASE_DIR, "timestamps.txt")

FPS           = 30
BUFFER        = 3.0
GAP_THRESHOLD = 90

os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

# ── 전역 작업 상태 ────────────────────────────────────────
_job: dict = {
    "status":        "idle",   # idle | uploading | extracting | detecting | ocr | highlighting | done | error
    "progress":      0,        # 0 ~ 100
    "message":       "",
    "player_number": None,
    "team":          None,
    "error":         None,
    "highlight_url": None,
}
_lock = threading.Lock()


def _set(status=None, progress=None, message=None, error=None, highlight_url=None):
    with _lock:
        if status        is not None: _job["status"]        = status
        if progress      is not None: _job["progress"]      = progress
        if message       is not None: _job["message"]       = message
        if error         is not None: _job["error"]         = error
        if highlight_url is not None: _job["highlight_url"] = highlight_url


# ── 파이프라인 유틸 ───────────────────────────────────────

def _run_script(script: str, step_label: str,
                progress_start: int, progress_end: int,
                parse_pattern: Optional[str] = None):
    """
    스크립트를 subprocess로 실행하며 stdout 파싱으로 진행률 업데이트.
    parse_pattern: r'\\[(\\d+)/(\\d+)\\]' 형태의 패턴
    """
    _set(message=step_label, progress=progress_start)
    proc = subprocess.Popen(
        [VENV_PY, script],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout:
        line = line.strip()
        if parse_pattern and (m := re.search(parse_pattern, line)):
            cur, total = int(m.group(1)), int(m.group(2))
            pct = progress_start + int((cur / total) * (progress_end - progress_start))
            _set(progress=pct)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{step_label} 실패 (exit {proc.returncode})")
    _set(progress=progress_end)


def _generate_highlight(player_number: str, team: Optional[str]):
    """results.txt → 클립 생성 → highlight.mp4"""
    _set(status="highlighting", message="하이라이트 생성 중", progress=90)

    records = _parse_results(team_filter=team, number_filter=player_number)
    if not records:
        raise RuntimeError(f"번호 {player_number} 감지 기록 없음")

    frames   = sorted({r["frame"] for r in records})
    segments = _frames_to_segments(frames)
    label    = f"{'전체' if not team else team}팀 {player_number}번 선수"

    # timestamps.txt
    with open(TIMESTAMPS_TXT, "w") as f:
        f.write(f"{label} 등장 구간\n{'='*40}\n")
        for i, s in enumerate(segments, 1):
            f.write(f"구간 {i}: {s['start']}s ~ {s['end']}s\n")

    # 기존 클립 제거
    for fn in os.listdir(CLIPS_DIR):
        if fn.startswith("clip_"):
            os.remove(os.path.join(CLIPS_DIR, fn))

    # 클립 생성
    clip_files = []
    for i, seg in enumerate(segments, 1):
        out = os.path.join(CLIPS_DIR, f"clip_{i:03d}.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(seg["start"]),
             "-to", str(seg["end"]), "-i", VIDEO_PATH, "-c", "copy", out],
            capture_output=True, check=True,
        )
        clip_files.append(out)
        pct = 90 + int((i / len(segments)) * 8)
        _set(progress=pct)

    # concat → highlight.mp4
    concat_list = os.path.join(CLIPS_DIR, "concat_list.txt")
    with open(concat_list, "w") as f:
        for cf in clip_files:
            f.write(f"file '{cf}'\n")

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", concat_list, "-c", "copy", HIGHLIGHT_PATH],
        capture_output=True, check=True,
    )
    _set(progress=100)
    return len(clip_files)


def _pipeline(player_number: str, team: Optional[str]):
    """백그라운드 파이프라인 전체 실행"""
    try:
        # 1. 프레임 추출
        _set(status="extracting")
        _run_script("extract.py", "프레임 추출 중", 5, 20)

        # 2. 인물 감지 (YOLO)
        _set(status="detecting")
        _run_script("detect.py", "선수 감지 중 (YOLO)", 20, 60,
                    parse_pattern=r'\[(\d+)/(\d+)\]')

        # 3. OCR (등번호 인식)
        _set(status="ocr")
        _run_script("ocr.py", "등번호 인식 중 (OCR)", 60, 90,
                    parse_pattern=r'\[(\d+)/(\d+)\]')

        # 4. 하이라이트 생성
        n_clips = _generate_highlight(player_number, team)

        _set(status="done", progress=100,
             message=f"완료! 클립 {n_clips}개 생성",
             highlight_url="/result/download")

    except Exception as e:
        _set(status="error", message="오류 발생", error=str(e))


# ── results.txt 파싱 유틸 ────────────────────────────────

def _parse_results(team_filter=None, number_filter=None):
    if not os.path.exists(RESULTS_TXT):
        return []
    records = []
    with open(RESULTS_TXT, "r") as f:
        for line in f:
            m = re.match(r'frame_(\d+)\.jpg \| (\S+) \| (\S+) \| (\d+) \| (\d+)', line.strip())
            if not m:
                continue
            frame  = int(m.group(1))
            number = m.group(2).lstrip("0") or "0"
            team   = m.group(3)
            x, y   = int(m.group(4)), int(m.group(5))
            if team_filter   and team.upper() != team_filter.upper(): continue
            if number_filter and number != (number_filter.lstrip("0") or "0"): continue
            records.append({"frame": frame, "number": number, "team": team, "x": x, "y": y})
    return records


def _frames_to_segments(frames: list[int]):
    if not frames:
        return []
    groups, start, end = [], frames[0], frames[0]
    for f in frames[1:]:
        if f - end <= GAP_THRESHOLD:
            end = f
        else:
            groups.append((start, end)); start = end = f
    groups.append((start, end))
    return [{"start": round(max(0.0, s/FPS - BUFFER), 2),
             "end":   round(e/FPS + BUFFER, 2)} for s, e in groups]


def _stream_video(path: str):
    def _iter():
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk
    return StreamingResponse(_iter(), media_type="video/mp4")


# ═══════════════════════════════════════════════════════════
# 1. POST /upload
# ═══════════════════════════════════════════════════════════

@app.post("/upload", summary="영상 업로드 + 선수 번호 등록")
async def upload(
    file: UploadFile = File(...),
    player_number: str = Form(...),
    team: Optional[str] = Form(None),
):
    """
    - **file**: 분석할 영상 파일 (mp4 권장)
    - **player_number**: 하이라이트 대상 등번호
    - **team**: HOME / AWAY / 미입력(전체)
    """
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(400, "지원하지 않는 파일 형식입니다 (mp4/avi/mov/mkv)")

    # uploads/ 에 원본 저장
    save_path = os.path.join(UPLOADS, file.filename)
    async with aiofiles.open(save_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            await out.write(chunk)

    # 파이프라인용 test.mp4 로 복사
    shutil.copy2(save_path, VIDEO_PATH)

    with _lock:
        _job["player_number"] = player_number
        _job["team"]          = team
        _job["status"]        = "idle"
        _job["progress"]      = 0
        _job["error"]         = None
        _job["highlight_url"] = None

    return {
        "message":       "업로드 완료",
        "filename":      file.filename,
        "player_number": player_number,
        "team":          team or "ALL",
        "next":          "POST /analyze 로 분석 시작",
    }


# ═══════════════════════════════════════════════════════════
# 2. POST /analyze
# ═══════════════════════════════════════════════════════════

@app.post("/analyze", summary="파이프라인 백그라운드 실행")
def analyze(
    player_number: Optional[str] = Form(None),
    team: Optional[str] = Form(None),
):
    """
    - **player_number**: 미입력 시 /upload 에서 등록한 번호 사용
    - **team**: HOME / AWAY / 미입력(전체)

    분석은 백그라운드로 실행됩니다. GET /status 로 진행률 확인.
    """
    with _lock:
        if _job["status"] not in ("idle", "done", "error"):
            raise HTTPException(409, f"이미 분석 중입니다 (status: {_job['status']})")

        number = player_number or _job.get("player_number")
        if not number:
            raise HTTPException(400, "player_number 를 입력하거나 먼저 /upload 를 호출하세요")

        _job["player_number"] = number
        _job["team"]          = team or _job.get("team")
        _job["status"]        = "extracting"
        _job["progress"]      = 0
        _job["error"]         = None
        _job["highlight_url"] = None

    t = threading.Thread(target=_pipeline, args=(number, _job["team"]), daemon=True)
    t.start()

    return {
        "message":       "분석 시작",
        "player_number": number,
        "team":          _job["team"] or "ALL",
        "track":         "GET /status",
    }


# ═══════════════════════════════════════════════════════════
# 3. GET /status
# ═══════════════════════════════════════════════════════════

@app.get("/status", summary="분석 진행 상태 조회")
def status():
    """
    Returns: status, progress(0~100), message, error
    """
    with _lock:
        return {
            "status":        _job["status"],
            "progress":      _job["progress"],
            "message":       _job["message"],
            "player_number": _job["player_number"],
            "team":          _job["team"] or "ALL",
            "error":         _job["error"],
        }


# ═══════════════════════════════════════════════════════════
# 4. GET /result
# ═══════════════════════════════════════════════════════════

@app.get("/result", summary="분석 결과 요약 + 다운로드 링크")
def result():
    with _lock:
        st = _job["status"]

    if st == "error":
        raise HTTPException(500, f"분석 실패: {_job['error']}")
    if st != "done":
        raise HTTPException(202, f"아직 분석 중입니다 (status: {st}, progress: {_job['progress']}%)")

    if not os.path.exists(HIGHLIGHT_PATH):
        raise HTTPException(404, "highlight.mp4 가 없습니다")

    clips = sorted([f for f in os.listdir(CLIPS_DIR) if f.startswith("clip_") and f.endswith(".mp4")])
    size_mb = round(os.path.getsize(HIGHLIGHT_PATH) / 1024 / 1024, 2)

    return {
        "status":         "done",
        "player_number":  _job["player_number"],
        "team":           _job["team"] or "ALL",
        "clips_count":    len(clips),
        "highlight_size": f"{size_mb} MB",
        "download":       "/result/download",
        "stream":         "/video/highlight",
    }


@app.get("/result/download", summary="highlight.mp4 다운로드")
def download_highlight():
    if not os.path.exists(HIGHLIGHT_PATH):
        raise HTTPException(404, "highlight.mp4 없음")
    return FileResponse(HIGHLIGHT_PATH, media_type="video/mp4",
                        filename="highlight.mp4")


# ── 기존 조회 / 스트리밍 API ─────────────────────────────

@app.get("/players", summary="선수 감지 기록 조회")
def get_players(team: Optional[str] = None, number: Optional[str] = None):
    records = _parse_results(team_filter=team, number_filter=number)
    return {"count": len(records), "records": records}


@app.get("/players/top", summary="등장 횟수 TOP N")
def get_top_players(n: int = 5, team: Optional[str] = None):
    records = _parse_results(team_filter=team)
    counter = Counter(r["number"] for r in records)
    return {"top": [{"number": num, "count": cnt} for num, cnt in counter.most_common(n)]}


@app.get("/players/{number}/segments", summary="특정 번호 등장 구간")
def get_segments(number: str, team: Optional[str] = None):
    records = _parse_results(team_filter=team, number_filter=number)
    if not records:
        raise HTTPException(404, f"번호 {number} 감지 기록 없음")
    segments = _frames_to_segments(sorted({r["frame"] for r in records}))
    return {"number": number, "team": team or "ALL",
            "detections": len(records), "segments": segments}


@app.get("/timestamps", summary="현재 timestamps.txt 내용")
def get_timestamps():
    if not os.path.exists(TIMESTAMPS_TXT):
        raise HTTPException(404, "timestamps.txt 없음")
    return {"content": open(TIMESTAMPS_TXT).read()}


@app.get("/video/highlight", summary="highlight.mp4 스트리밍")
def stream_highlight():
    if not os.path.exists(HIGHLIGHT_PATH):
        raise HTTPException(404, "highlight.mp4 없음")
    return _stream_video(HIGHLIGHT_PATH)


@app.get("/video/clips/{filename}", summary="개별 클립 스트리밍")
def stream_clip(filename: str):
    path = os.path.join(CLIPS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"{filename} 없음")
    return _stream_video(path)


@app.get("/video/original", summary="원본 영상 스트리밍")
def stream_original():
    if not os.path.exists(VIDEO_PATH):
        raise HTTPException(404, "test.mp4 없음")
    return _stream_video(VIDEO_PATH)


@app.get("/clips", summary="생성된 클립 목록")
def list_clips():
    if not os.path.exists(CLIPS_DIR):
        return {"clips": []}
    clips = sorted([f for f in os.listdir(CLIPS_DIR) if f.startswith("clip_") and f.endswith(".mp4")])
    return {"count": len(clips), "clips": clips}


@app.get("/", summary="헬스체크")
def root():
    return {"status": "ok", "service": "IceIQ Video Analysis API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
