#!/usr/bin/env python3
"""
ice_time_estimator.py
5-slot constraint-based ice time estimation for hockey video analysis.

Shift detection approach:
  - In game1-quality footage, each track_id persists ~35-60s ≈ one shift.
  - Track_ids are sorted by first-appearance frame within each team.
  - Concurrent track_ids (overlapping time) belong to different players.
  - Sequential track_ids (non-overlapping, same player slot) form a shift timeline.
  - Players are ranked by ice_time_min from player_stats.json and assigned
    to track_id slots by rank (most ice time → most frames tracked).

Usage:
    python3 ice_time_estimator.py --game game1
    python3 ice_time_estimator.py --game game1 --json
"""

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "data" / "results"
HOMOGRAPHY_PATH = BASE_DIR / "configs" / "homography_4point.json"

SLOTS_PER_TEAM = 5
BENCH_ZONE_LOW = 2.0
BENCH_ZONE_HIGH = 24.0
CENTER_Y_LOW = 8.0
CENTER_Y_HIGH = 18.0
BLUELINE_NEAR = 17.5
BLUELINE_FAR = 34.5

# Typical adult hockey shift (used for statistical fallback)
TYPICAL_SHIFT_SEC = 60.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_homography(path: Path) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    return np.array(data["matrix"], dtype=np.float64)


def pixel_to_rink(cx: float, cy: float, H: np.ndarray) -> tuple:
    pt = H @ np.array([cx, cy, 1.0])
    return float(pt[0] / pt[2]), float(pt[1] / pt[2])


def zone_label(rx: float) -> str:
    if rx < BLUELINE_NEAR:
        return "low"
    elif rx < BLUELINE_FAR:
        return "nz"
    else:
        return "high"


def build_track_timeline(tracks_by_frame: dict, team_label: str,
                         team_labels: dict) -> dict:
    """
    For each track_id in the given team, compute:
      first_frame, last_frame, frame_count, sorted active_frames
    Returns dict: track_id → {first, last, count, frames}
    """
    info: dict = defaultdict(lambda: {"first": None, "last": None,
                                      "count": 0, "frames": []})
    for fn, tl in tracks_by_frame.items():
        for t in tl:
            tid = t.get("track_id", 0)
            if team_labels.get(tid) != team_label:
                continue
            entry = info[tid]
            entry["count"] += 1
            entry["frames"].append(fn)
            if entry["first"] is None or fn < entry["first"]:
                entry["first"] = fn
            if entry["last"] is None or fn > entry["last"]:
                entry["last"] = fn
    return dict(info)


def assign_shifts_by_rank(timeline: dict, player_rank: int,
                          n_players: int, fps: float,
                          min_shift_sec: float = 5.0,
                          merge_gap_sec: float = 8.0,
                          max_shift_sec: float = 120.0) -> list:
    """
    Assign shifts to a player at `player_rank` (0 = most ice time).

    Strategy:
      1. Sort all track_ids by first_frame.
      2. At each moment in time, at most SLOTS_PER_TEAM tracks are concurrent.
         Within a concurrent group, assign slot 0 = longest active, slot 1 = next, …
      3. Collect all track_ids where this player's rank slot is dominant.
      4. Merge consecutive track_ids with gap < merge_gap_sec into one shift.

    Returns list of {"start_sec", "end_sec", "duration_sec"}.
    """
    if not timeline:
        return []

    merge_gap = int(fps * merge_gap_sec)
    min_shift_frames = int(fps * min_shift_sec)
    max_shift_frames = int(fps * max_shift_sec)

    # Sort tracks by first appearance
    sorted_tids = sorted(timeline.keys(), key=lambda t: timeline[t]["first"])

    # For each track_id, compute its "concurrent rank" at its midpoint
    # Rank = how many other tracks are also active at its midpoint, by frame count desc
    assigned_frames: list = []  # (frame, ) for this player's rank

    # Sliding: for each track_id, find who else is active at the same time
    # and assign rank by frame_count desc
    for tid in sorted_tids:
        t_info = timeline[tid]
        t_first, t_last, t_count = t_info["first"], t_info["last"], t_info["count"]
        t_mid = (t_first + t_last) // 2

        # Find all tracks active at t_mid
        concurrent = [
            (timeline[other]["count"], other)
            for other in sorted_tids
            if timeline[other]["first"] <= t_mid <= timeline[other]["last"]
        ]
        concurrent.sort(reverse=True)  # most frames = rank 0

        # What rank is this track_id among concurrent ones?
        rank_here = next(
            (i for i, (_, oid) in enumerate(concurrent) if oid == tid),
            len(concurrent)
        )

        if rank_here == player_rank:
            assigned_frames.extend(t_info["frames"])

    if not assigned_frames:
        return []

    assigned_frames.sort()

    # Group into shifts with merge_gap, split on max_shift_sec
    def flush_shift(start_f, end_f):
        """Append shift(s), splitting if duration exceeds max_shift_frames."""
        dur = end_f - start_f
        if dur < min_shift_frames:
            return
        while dur > max_shift_frames:
            shifts.append({
                "start_sec": round(start_f / fps, 1),
                "end_sec": round((start_f + max_shift_frames) / fps, 1),
                "duration_sec": round(max_shift_frames / fps, 1),
            })
            start_f += max_shift_frames
            dur -= max_shift_frames
        if dur >= min_shift_frames:
            shifts.append({
                "start_sec": round(start_f / fps, 1),
                "end_sec": round(end_f / fps, 1),
                "duration_sec": round(dur / fps, 1),
            })

    shifts = []
    s = assigned_frames[0]
    prev = assigned_frames[0]
    for fn in assigned_frames[1:]:
        if fn - prev > merge_gap:
            flush_shift(s, prev)
            s = fn
        prev = fn
    flush_shift(s, prev)

    return shifts


def zone_pcts(tracks_by_frame: dict, timeline: dict,
              player_rank: int, team_label: str,
              fps: float, H: np.ndarray) -> dict:
    """Compute OZ/NZ/DZ percentages for this player's rank slot."""
    sorted_tids = sorted(timeline.keys(), key=lambda t: timeline[t]["first"])
    player_tids: set = set()

    for tid in sorted_tids:
        t_info = timeline[tid]
        t_mid = (t_info["first"] + t_info["last"]) // 2
        concurrent = [
            (timeline[other]["count"], other)
            for other in sorted_tids
            if timeline[other]["first"] <= t_mid <= timeline[other]["last"]
        ]
        concurrent.sort(reverse=True)
        rank_here = next(
            (i for i, (_, oid) in enumerate(concurrent) if oid == tid),
            len(concurrent)
        )
        if rank_here == player_rank:
            player_tids.add(tid)

    zc: dict = defaultdict(int)
    for fn, tl in tracks_by_frame.items():
        for t in tl:
            if t.get("track_id", 0) in player_tids:
                cx, cy = t.get("cx", 0.0), t.get("cy", 0.0)
                rx, _ = pixel_to_rink(cx, cy, H)
                zc[zone_label(rx)] += 1

    total = sum(zc.values()) or 1
    if team_label == "team_a":
        oz = round(zc["low"] / total * 100)
        dz = round(zc["high"] / total * 100)
    else:
        oz = round(zc["high"] / total * 100)
        dz = round(zc["low"] / total * 100)
    return {"oz_pct": oz, "nz_pct": 100 - oz - dz, "dz_pct": dz}


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

def compute_icetime(game: str) -> dict:
    game_dir = RESULTS_DIR / game
    cache_file = game_dir / "cache.pkl"
    stats_file = game_dir / "player_stats.json"

    if not cache_file.exists():
        raise FileNotFoundError(f"Cache not found: {cache_file}")

    with open(cache_file, "rb") as f:
        cache = pickle.load(f)

    fps = float(cache.get("fps", 60.0))
    team_labels: dict = cache.get("teams", {})
    tracks_by_frame: dict = cache.get("tracks", {})
    total_frames = int(cache.get("frame_count", 0) or
                       (max(tracks_by_frame.keys()) + 1 if tracks_by_frame else 0))
    video_duration_sec = total_frames / fps

    H = load_homography(HOMOGRAPHY_PATH)

    # Team name mapping
    team_name_map = {"team_a": "team_a", "team_b": "team_b"}
    raw_stats: dict = {}
    if stats_file.exists():
        with open(stats_file) as f:
            raw_stats = json.load(f)
        for p in raw_stats.values():
            t_name = p.get("team", "")
            label = "team_b" if t_name == "Aigis" else "team_a"
            team_name_map[label] = t_name

    # -----------------------------------------------------------------------
    # Correction factor (cap at 5, floor at 1.0)
    # -----------------------------------------------------------------------
    counts_a, counts_b = [], []
    for tl in tracks_by_frame.values():
        n_a = min(sum(1 for t in tl if team_labels.get(t.get("track_id", 0)) == "team_a"),
                  SLOTS_PER_TEAM)
        n_b = min(sum(1 for t in tl if team_labels.get(t.get("track_id", 0)) == "team_b"),
                  SLOTS_PER_TEAM)
        if n_a >= 2:
            counts_a.append(n_a)
        if n_b >= 2:
            counts_b.append(n_b)

    avg_a = sum(counts_a) / len(counts_a) if counts_a else 0.0
    avg_b = sum(counts_b) / len(counts_b) if counts_b else 0.0
    cf_a = round(max(1.0, min(SLOTS_PER_TEAM / avg_a, 3.0)), 3) if avg_a > 0 else 1.0
    cf_b = round(max(1.0, min(SLOTS_PER_TEAM / avg_b, 3.0)), 3) if avg_b > 0 else 1.0
    camera_coverage_pct = round((avg_a + avg_b) / 2 / SLOTS_PER_TEAM * 100) if (avg_a or avg_b) else 0

    # -----------------------------------------------------------------------
    # Build track timelines per team
    # -----------------------------------------------------------------------
    timeline_a = build_track_timeline(tracks_by_frame, "team_a", team_labels)
    timeline_b = build_track_timeline(tracks_by_frame, "team_b", team_labels)

    # -----------------------------------------------------------------------
    # Per-player output
    # -----------------------------------------------------------------------
    # Sort players per team by ice_time_min desc → rank 0 = most ice time
    players_a = sorted(
        [(pid, p) for pid, p in raw_stats.items() if team_name_map.get("team_a") == p.get("team") or
         ("Aigis" != p.get("team") and team_name_map.get("team_a") != "team_a")],
        key=lambda x: -x[1].get("ice_time_min", 0)
    )
    players_b = sorted(
        [(pid, p) for pid, p in raw_stats.items() if p.get("team") == "Aigis"],
        key=lambda x: -x[1].get("ice_time_min", 0)
    )

    # Fallback: if team name mapping is ambiguous, split by team field
    def split_by_team():
        a, b = [], []
        for pid, p in raw_stats.items():
            if p.get("team") == "Aigis":
                b.append((pid, p))
            else:
                a.append((pid, p))
        return (sorted(a, key=lambda x: -x[1].get("ice_time_min", 0)),
                sorted(b, key=lambda x: -x[1].get("ice_time_min", 0)))

    players_a, players_b = split_by_team()

    def build_player_entry(pid, p, rank, team_label, cf, timeline):
        tracked_sec = round(p.get("ice_time_min", 0) * 60, 1)
        estimated_sec = round(tracked_sec * cf, 1)
        confidence = round(1.0 / cf, 2) if cf > 0 else 1.0

        # Shift detection from track timeline
        shifts = assign_shifts_by_rank(timeline, rank, SLOTS_PER_TEAM, fps)

        if shifts:
            detected_sec = sum(sh["duration_sec"] for sh in shifts)
            num_shifts = len(shifts)
            avg_shift_sec = round(detected_sec / num_shifts, 1)
        else:
            # Statistical fallback
            num_shifts = max(1, round(estimated_sec / TYPICAL_SHIFT_SEC))
            avg_shift_sec = round(estimated_sec / num_shifts, 1)

        zt = zone_pcts(tracks_by_frame, timeline, rank, team_label, fps, H)

        return {
            "player_id": pid,
            "jersey": p.get("jersey", "?"),
            "name": p.get("name", ""),
            "team": p.get("team", ""),
            "team_label": team_label,
            "tracked_time_sec": tracked_sec,
            "correction_factor": cf,
            "estimated_ice_time_sec": estimated_sec,
            "shifts": shifts,
            "num_shifts": num_shifts,
            "avg_shift_sec": avg_shift_sec,
            "zone_time": zt,
            "confidence": confidence,
        }

    players = []
    for rank, (pid, p) in enumerate(players_a):
        players.append(build_player_entry(pid, p, rank, "team_a", cf_a, timeline_a))
    for rank, (pid, p) in enumerate(players_b):
        players.append(build_player_entry(pid, p, rank, "team_b", cf_b, timeline_b))

    players.sort(key=lambda p: -p["estimated_ice_time_sec"])

    total_est_a = sum(p["estimated_ice_time_sec"] for p in players if p["team_label"] == "team_a")
    total_est_b = sum(p["estimated_ice_time_sec"] for p in players if p["team_label"] == "team_b")

    game_info = {
        "game": game,
        "video_duration_sec": round(video_duration_sec, 1),
        "fps": fps,
        "team_a_name": team_name_map.get("team_a", "team_a"),
        "team_b_name": team_name_map.get("team_b", "team_b"),
        "avg_detected_a": round(avg_a, 2),
        "avg_detected_b": round(avg_b, 2),
        "correction_factor_a": cf_a,
        "correction_factor_b": cf_b,
        "team_a_estimated_min": round(total_est_a / 60, 1),
        "team_b_estimated_min": round(total_est_b / 60, 1),
        "team_total_estimated_min": round((total_est_a + total_est_b) / 60, 1),
        "expected_total_min": round(SLOTS_PER_TEAM * video_duration_sec / 60, 1),
        "camera_coverage_pct": camera_coverage_pct,
    }

    return {"game_info": game_info, "players": players}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hockey ice time estimator (5-slot)")
    parser.add_argument("--game", required=True, help="Game name (e.g. game1)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    result = compute_icetime(args.game)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    gi = result["game_info"]
    print(f"\n=== {gi['game']} — Ice Time Estimation (5-slot) ===")
    print(f"Duration   : {gi['video_duration_sec']:.0f}s ({gi['video_duration_sec']/60:.1f}min)")
    print(f"Coverage   : {gi['camera_coverage_pct']}%")
    print(f"Correction : {gi['team_a_name']} ×{gi['correction_factor_a']}  |  "
          f"{gi['team_b_name']} ×{gi['correction_factor_b']}")
    print(f"Estimated  : {gi['team_a_name']} {gi['team_a_estimated_min']:.1f}min  |  "
          f"{gi['team_b_name']} {gi['team_b_estimated_min']:.1f}min  "
          f"(expected ~{gi['expected_total_min']:.0f}min total)")
    print()
    fmt = "{:>4}  {:<12}  {:<8}  {:>8}  {:>4}  {:>9}  {:>6}  {:>7}  {:>4}{:>4}{:>4}  {:>5}"
    print(fmt.format("#", "Name", "Team", "Tracked", "×CF", "Estimated",
                     "Shifts", "AvgShft", "OZ", "NZ", "DZ", "Conf"))
    print("-" * 85)
    for p in result["players"]:
        zt = p["zone_time"]
        print(fmt.format(
            str(p.get("jersey", "?")),
            p.get("name", "")[:12],
            p["team"][:8],
            f"{p['tracked_time_sec']/60:.1f}m",
            f"×{p['correction_factor']:.2f}",
            f"{p['estimated_ice_time_sec']/60:.1f}m",
            str(p["num_shifts"]),
            f"{p['avg_shift_sec']:.0f}s",
            f"{zt['oz_pct']}%",
            f"{zt['nz_pct']}%",
            f"{zt['dz_pct']}%",
            f"{p['confidence']:.0%}",
        ))


if __name__ == "__main__":
    main()
