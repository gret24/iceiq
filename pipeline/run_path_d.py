"""
Path D end-to-end test runner for IceIQ.
Tests: HighlightExtraction, EventDetection, RecruitingPDF, CrossShiftIdentity
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = {}

# ── 1. Highlight Extraction ─────────────────────────────────────────────────
print("\n" + "="*60)
print("1. HIGHLIGHT EXTRACTION")
print("="*60)

try:
    from pipeline.highlight_extraction import HighlightExtractor

    # Build config
    config_dir = os.path.expanduser("~/iceiq-dev/configs/")
    os.makedirs(config_dir, exist_ok=True)
    cfg_path = os.path.join(config_dir, "highlight_extraction_config.json")
    cfg = {
        "event_scores": {
            "goal": 100, "shot_on_goal": 50, "big_hit": 70,
            "power_play": 60, "man_advantage": 55, "zone_entry": 25,
            "turnover": 30
        },
        "min_score_threshold": 25,
        "default_highlight_duration": 150,
        "pre_event_duration_frames": 90,
        "post_event_duration_frames": 150
    }
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # Load timeline.json → derive events
    timeline_path = os.path.expanduser("~/iceiq-dev/data/results/game2/timeline.json")
    with open(timeline_path) as f:
        timeline = json.load(f)

    # Build synthetic events from timeline icetime transitions
    events = []
    prev = {"aigis": set(), "goyang": set()}
    FPS = 30

    for entry in timeline:
        t = entry["t"]
        frame = int(t * FPS)
        aigis_now = set(entry.get("aigis", []))
        goyang_now = set(entry.get("goyang", []))

        # Man-advantage: aigis has more players visible
        if len(aigis_now) > len(goyang_now) and len(aigis_now) >= 3:
            events.append({
                "type": "man_advantage",
                "frame_number": frame,
                "description": f"Aigis man-advantage ({len(aigis_now)} vs {len(goyang_now)}) @ t={t:.1f}s",
                "team": "aigis"
            })

        # Zone entries / player appearance (jersey first appears)
        for jnum in aigis_now - prev["aigis"]:
            events.append({
                "type": "zone_entry",
                "frame_number": frame,
                "description": f"Aigis #{jnum} enters zone @ t={t:.1f}s",
                "player": jnum, "team": "aigis"
            })

        prev["aigis"] = aigis_now
        prev["goyang"] = goyang_now

    # Add simulated shot/goal events at high-activity timestamps
    # Use timestamps where aigis has many players on ice
    active = [(e["t"], set(e.get("aigis", []))) for e in timeline if len(e.get("aigis", [])) >= 2]
    if active:
        for t, players in active[:5]:
            frame = int(t * FPS)
            events.append({
                "type": "shot_on_goal",
                "frame_number": frame,
                "description": f"Shot by Aigis (players: {sorted(players)}) @ t={t:.1f}s",
                "team": "aigis"
            })

    print(f"  Timeline entries: {len(timeline)}")
    print(f"  Synthetic events built: {len(events)}")

    extractor = HighlightExtractor(config_path=cfg_path)
    highlights = extractor.extract_highlights({"events": events})

    # Save results
    out_dir = os.path.expanduser("~/iceiq-dev/data/results/game2/highlights/")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "highlights.json")
    with open(out_path, "w") as f:
        json.dump(highlights, f, indent=2)

    print(f"  Highlights extracted: {len(highlights)}")
    for h in highlights[:5]:
        print(f"    [{h['start_frame']}→{h['end_frame']}] {h['description']} (score={h['score']})")
    if len(highlights) > 5:
        print(f"    ... and {len(highlights)-5} more")
    print(f"  Saved → {out_path}")
    RESULTS["highlight_extraction"] = {"status": "OK", "highlights": len(highlights), "output": out_path}

except Exception as e:
    import traceback
    print(f"  ERROR: {e}")
    traceback.print_exc()
    RESULTS["highlight_extraction"] = {"status": "ERROR", "error": str(e)}


# ── 2. Event Detection ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("2. EVENT DETECTION")
print("="*60)

try:
    from pipeline.event_detection import EventDetector
    import numpy as np

    detector = EventDetector(device="cpu")

    # Simulate 10 frames with players + puck
    all_events = []
    rng = np.random.default_rng(42)

    # Aigis jersey numbers from summary
    aigis_jerseys = [4, 11, 12, 14, 25, 61, 94]

    for frame_num in range(0, 300, 30):
        # 4 Aigis + 3 Goyang players, positions on rink
        players = []
        for i, jnum in enumerate(aigis_jerseys[:4]):
            x = rng.uniform(-20, 20)
            y = rng.uniform(-10, 10)
            vx = rng.uniform(-3, 3)
            vy = rng.uniform(-3, 3)
            players.append({
                "id": jnum, "team": "aigis", "jersey": jnum,
                "position": [x, y], "velocity": [vx, vy],
                "zone_history": []
            })
        for i in range(3):
            x = rng.uniform(-20, 20)
            y = rng.uniform(-10, 10)
            players.append({
                "id": 200+i, "team": "goyang",
                "position": [x, y], "velocity": [rng.uniform(-2,2), rng.uniform(-2,2)],
                "zone_history": []
            })

        puck_data = {
            "position": [rng.uniform(-5, 5), rng.uniform(-3, 3)],
            "velocity": [rng.uniform(-20, 20), rng.uniform(-20, 20)],
            "possession_history": [aigis_jerseys[frame_num % 4], aigis_jerseys[(frame_num+1) % 4]]
        }

        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_events = detector.detect_events(dummy_frame, frame_num, players, puck_data)
        all_events.extend(frame_events)

    summary = detector.get_event_summary(all_events)
    high_conf = detector.filter_events_by_confidence(all_events, min_confidence=0.5)

    # Timeline-based: count man-advantage windows from timeline
    with open(os.path.expanduser("~/iceiq-dev/data/results/game2/timeline.json")) as f:
        tl = json.load(f)

    man_adv_windows = []
    in_window = False
    window_start = None
    for entry in tl:
        aigis_cnt = len(entry.get("aigis", []))
        goyang_cnt = len(entry.get("goyang", []))
        if aigis_cnt > goyang_cnt and aigis_cnt >= 2:
            if not in_window:
                in_window = True
                window_start = entry["t"]
        else:
            if in_window:
                man_adv_windows.append({"start": window_start, "end": entry["t"],
                                        "duration": entry["t"] - window_start})
                in_window = False
    if in_window:
        man_adv_windows.append({"start": window_start, "end": tl[-1]["t"],
                                "duration": tl[-1]["t"] - window_start})

    out_dir = os.path.expanduser("~/iceiq-dev/data/results/game2/highlights/")
    events_path = os.path.join(out_dir, "events.json")
    with open(events_path, "w") as f:
        json.dump({
            "detector_events": all_events,
            "summary": summary,
            "man_advantage_windows": man_adv_windows
        }, f, indent=2, default=str)

    print(f"  Detector events (10 frames): {len(all_events)}")
    print(f"  Summary: {summary}")
    print(f"  High-confidence (≥0.5): {len(high_conf)}")
    print(f"  Man-advantage windows from timeline: {len(man_adv_windows)}")
    for w in man_adv_windows[:3]:
        print(f"    t={w['start']:.1f}s → {w['end']:.1f}s ({w['duration']:.1f}s)")
    print(f"  Saved → {events_path}")
    RESULTS["event_detection"] = {"status": "OK", "events": len(all_events),
                                   "man_adv_windows": len(man_adv_windows), "output": events_path}

except Exception as e:
    import traceback
    print(f"  ERROR: {e}")
    traceback.print_exc()
    RESULTS["event_detection"] = {"status": "ERROR", "error": str(e)}


# ── 3. Recruiting PDF ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("3. RECRUITING PDF")
print("="*60)

try:
    from pipeline.recruiting_pdf import generate_recruiting_pdf, batch_generate_reports

    # Load ice time from summary
    summary_path = os.path.expanduser("~/iceiq-dev/data/results/game2/summary.json")
    with open(summary_path) as f:
        game_summary = json.load(f)

    aigis_icetime = game_summary.get("aigis_icetime", {})

    # Build player data for each Aigis player
    jersey_positions = {
        "4": "defense", "11": "forward", "12": "forward",
        "14": "forward", "25": "defense", "61": "forward", "94": "goalie"
    }
    players_data = []
    for jersey, icetime in aigis_icetime.items():
        minutes = icetime / 60.0
        # Fabricate reasonable metrics based on icetime
        shots = max(1, int(minutes * 0.8))
        players_data.append({
            "name": f"Aigis #{jersey}",
            "jersey_number": int(jersey),
            "team": "Aigis Ice Hockey",
            "position": jersey_positions.get(jersey, "forward"),
            "shot_hand": "left",
            "metrics": {
                "ice_time_efficiency": min(1.0, icetime / 300.0),
                "skating_speed": round(0.6 + (icetime % 60) / 300, 3),
                "positioning": round(0.5 + (int(jersey) % 10) * 0.04, 3),
            },
            "game_stats": {
                "ice_time_seconds": icetime,
                "ice_time_minutes": round(minutes, 1),
                "estimated_shots": shots,
            },
            "highlights": [
                {"timestamp": int(icetime * 0.3), "description": f"#{jersey} active play", "score": 0.7},
                {"timestamp": int(icetime * 0.7), "description": f"#{jersey} key moment", "score": 0.65},
            ]
        })

    out_dir = os.path.expanduser("/tmp/recruiting_aigis/")
    os.makedirs(out_dir, exist_ok=True)
    paths = batch_generate_reports(players_data, output_dir=out_dir)

    # Also generate combined sample
    sample_path = "/tmp/recruiting_sample.pdf"
    sample_player = max(players_data, key=lambda p: p["game_stats"]["ice_time_seconds"])
    out = generate_recruiting_pdf(sample_player, sample_path)

    print(f"  Players processed: {len(players_data)}")
    for p in players_data:
        print(f"    #{p['jersey_number']:2d} {p['position']:8s}  ice={p['game_stats']['ice_time_seconds']:.1f}s")
    print(f"  Batch reports → {out_dir}")
    for p in paths:
        exists = "✓" if os.path.exists(p) else "✗"
        print(f"    {exists} {os.path.basename(p)}")
    print(f"  Sample PDF → {out}")
    RESULTS["recruiting_pdf"] = {"status": "OK", "players": len(players_data),
                                  "sample": out, "batch_dir": out_dir}

except Exception as e:
    import traceback
    print(f"  ERROR: {e}")
    traceback.print_exc()
    RESULTS["recruiting_pdf"] = {"status": "ERROR", "error": str(e)}


# ── 4. Cross-Shift Identity ─────────────────────────────────────────────────
print("\n" + "="*60)
print("4. CROSS-SHIFT IDENTITY")
print("="*60)

try:
    from pipeline.cross_shift_identity import CrossShiftIdentityTracker
    import numpy as np

    tracker = CrossShiftIdentityTracker(similarity_threshold=0.7)
    rng = np.random.default_rng(0)

    # Simulate timeline: 7 Aigis players appearing/disappearing across shifts
    aigis_jerseys = [4, 11, 12, 14, 25, 61, 94]
    # Each player has a stable embedding
    player_embeddings = {j: rng.random(32).tolist() for j in aigis_jerseys}

    assigned = {}  # jersey → global_id
    re_id_success = 0
    re_id_attempts = 0

    # Shift 1: players 14, 25, 61 on ice (track IDs 1,2,3)
    shift1 = [(1, 14), (2, 25), (3, 61)]
    for tid, jersey in shift1:
        gid = tracker.register_player(tid, {
            "jersey_number": jersey, "team": "aigis",
            "embedding": player_embeddings[jersey]
        })
        assigned[jersey] = gid
        print(f"  Shift1 tid={tid} jersey=#{jersey} → global_id={gid}")

    # Shift 2: same players, new track IDs (re-ID test)
    shift2 = [(10, 14), (11, 25), (12, 61)]
    for tid, jersey in shift2:
        re_id_attempts += 1
        gid = tracker.register_player(tid, {
            "jersey_number": jersey, "team": "aigis",
            "embedding": player_embeddings[jersey]
        })
        if gid == assigned[jersey]:
            re_id_success += 1
            status = "✓ re-ID"
        else:
            status = f"✗ new_id={gid} (expected {assigned[jersey]})"
        print(f"  Shift2 tid={tid} jersey=#{jersey} → global_id={gid}  {status}")

    # Shift 3: new players (4, 11, 12) + one returning (14)
    shift3 = [(20, 4), (21, 11), (22, 12), (23, 14)]
    for tid, jersey in shift3:
        gid = tracker.register_player(tid, {
            "jersey_number": jersey, "team": "aigis",
            "embedding": player_embeddings[jersey]
        })
        if jersey not in assigned:
            assigned[jersey] = gid
        print(f"  Shift3 tid={tid} jersey=#{jersey} → global_id={gid}")

    stats = tracker.get_stats()
    accuracy = re_id_success / re_id_attempts if re_id_attempts else 0

    print(f"\n  Registry stats: {stats}")
    print(f"  Re-ID accuracy: {re_id_success}/{re_id_attempts} = {accuracy:.1%}")

    RESULTS["cross_shift_identity"] = {
        "status": "OK",
        "re_id_accuracy": f"{re_id_success}/{re_id_attempts}",
        "registry": stats
    }

except Exception as e:
    import traceback
    print(f"  ERROR: {e}")
    traceback.print_exc()
    RESULTS["cross_shift_identity"] = {"status": "ERROR", "error": str(e)}


# ── 5. Unit Tests ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("5. UNIT TESTS (pipeline_d_test.py)")
print("="*60)

import subprocess
result = subprocess.run(
    [sys.executable, "-m", "pytest", "pipeline/pipeline_d_test.py", "-v", "--tb=short"],
    cwd=os.path.expanduser("~/iceiq-dev"),
    capture_output=True, text=True
)
print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
if result.stderr:
    print(result.stderr[-1000:])
RESULTS["unit_tests"] = {"returncode": result.returncode}


# ── Final Summary ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("PATH D TEST SUMMARY")
print("="*60)
for component, res in RESULTS.items():
    status = res.get("status", "UNKNOWN")
    icon = "✓" if status == "OK" else ("✗" if status == "ERROR" else "?")
    print(f"  {icon} {component:30s} {status}")
    for k, v in res.items():
        if k != "status":
            print(f"      {k}: {v}")

# Save summary
out_path = os.path.expanduser("~/iceiq-dev/data/results/game2/path_d_results.json")
with open(out_path, "w") as f:
    json.dump(RESULTS, f, indent=2)
print(f"\n  Full results → {out_path}")
