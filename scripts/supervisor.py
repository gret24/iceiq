#!/usr/bin/env python3
"""
Supervisor: monitors autodev, detects stuck patterns, asks Claude for fixes.
"""
import anthropic
import subprocess
import json
import os
import time
import re

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
PROJECT_DIR = os.path.expanduser("~/iceiq-dev")
STATE_FILE = os.path.join(PROJECT_DIR, "autodev_state.json")
LOG_FILE = os.path.join(PROJECT_DIR, "logs/autodev_run2.log")
SUPERVISOR_LOG = os.path.join(PROJECT_DIR, "logs/supervisor.log")
FAIL_THRESHOLD = 8  # consecutive fails before intervening
CHECK_INTERVAL = 60  # seconds between checks

def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(SUPERVISOR_LOG, "a") as f:
        f.write(line + "\n")

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}

def get_recent_fails():
    state = load_state()
    hist = state.get("hist", [])
    if not hist:
        return 0, []
    fails = 0
    errors = []
    for h in reversed(hist):
        if not h.get("ok"):
            fails += 1
            if h.get("err"):
                errors.append(h["err"][:300])
        else:
            break
    return fails, errors

def get_current_code(filename="pipeline/detection.py"):
    fp = os.path.join(PROJECT_DIR, filename)
    if os.path.exists(fp):
        with open(fp) as f:
            return f.read()
    return ""

def ask_claude_for_fix(milestone, errors, current_code):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""You are debugging an IceIQ ice hockey pipeline on Mac Mini M4 (Apple Silicon, MPS).

STUCK MILESTONE: {milestone}
RECENT ERRORS:
{chr(10).join(set(errors[:5]))}

CURRENT CODE:
```python
{current_code[:3000]}
```

ENV: conda iceiq, Python 3.10, ultralytics, torch 2.11 MPS, opencv, filterpy, boxmot
VIDEO: ~/iceiq-dev/data/videos/game1.mp4 (youth hockey, wide-angle, full rink)

Provide ONE fix:
1. ACTION: WRITE_FILE + PATH + complete fixed code, OR
2. ACTION: RUN_COMMAND + CMD to fix the issue

Rules:
- Complete working code only, no placeholders
- Use device='mps' for torch
- Fix the root cause, not symptoms

Respond exactly:
ACTION: WRITE_FILE
PATH: pipeline/xxx.py
CONTENT:
```python
[complete code]
```
OR
ACTION: RUN_COMMAND
CMD: [shell command]"""

    r = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.content[0].text

def parse_and_apply(resp, milestone):
    lines = resp.split("\n")
    action = None
    path = None
    cmd = None
    for line in lines:
        l = line.strip()
        if l == "ACTION: WRITE_FILE":
            action = "write"
        elif l == "ACTION: RUN_COMMAND":
            action = "run"
        elif l.startswith("PATH: "):
            path = l[6:].strip()
        elif l.startswith("CMD: "):
            cmd = l[5:].strip()

    if action == "write" and path:
        s = resp.find("```python\n")
        e = resp.rfind("```")
        if s >= 0 and e > s:
            content = resp[s + 10:e]
            fp = os.path.join(PROJECT_DIR, path)
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, "w") as f:
                f.write(content)
            log(f"FIXED: wrote {path}")
            return True
    elif action == "run" and cmd:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120, cwd=PROJECT_DIR)
        log(f"FIX CMD: {'OK' if r.returncode == 0 else 'FAIL'} - {cmd[:80]}")
        if r.stdout:
            log(f"  OUT: {r.stdout[:200]}")
        if r.stderr:
            log(f"  ERR: {r.stderr[:200]}")
        return r.returncode == 0
    return False

def is_autodev_running():
    r = subprocess.run("ps aux | grep 'autodev.py' | grep -v grep", shell=True, capture_output=True, text=True)
    return bool(r.stdout.strip())

def restart_autodev():
    log("Restarting autodev...")
    cmd = (
        'export PATH="/opt/homebrew/bin:$PATH" && '
        f'export ANTHROPIC_API_KEY="{ANTHROPIC_API_KEY}" && '
        'export TELEGRAM_BOT_TOKEN="AAEnqp7lEAnJdVV4Zrx51MylritPn4nLmAg" && '
        'export TELEGRAM_CHAT_ID="8740736142" && '
        'source $(conda info --base)/etc/profile.d/conda.sh && '
        'conda activate iceiq && '
        'cd ~/iceiq-dev && '
        'PYTHONUNBUFFERED=1 python3 -u scripts/autodev.py >> logs/autodev_run2.log 2>&1 &'
    )
    subprocess.Popen(cmd, shell=True, cwd=PROJECT_DIR)
    time.sleep(5)
    log(f"Autodev running: {is_autodev_running()}")

def main():
    log("Supervisor started")
    last_milestone = None
    intervention_count = 0

    while True:
        try:
            state = load_state()
            goals = ["path_b", "path_c", "path_d"]
            goal_idx = state.get("goal_idx", 0)
            ms_idx = state.get("ms_idx", 0)
            done = state.get("done", [])
            iteration = state.get("iter", 0)

            GOALS = [
                {"id":"path_b","milestones":["detection_module","tracking_module","team_classification","jersey_recognition","homography","kalman_kinematics","heatmap_generation","full_pipeline_test","accuracy_validation"]},
                {"id":"path_c","milestones":["puck_detection","puck_possession","team_analytics","sprint_detection","contact_detection","faceoff_detection","powerplay_detection","scoreboard_ocr","zone_entry_exit","fatigue_analysis","full_pipeline_c_test","accuracy_validation_c"]},
                {"id":"path_d","milestones":["highlight_scoring","highlight_extraction","cross_shift_identity","season_trend_analysis","matchup_analysis","recruiting_pdf","full_pipeline_d_test","accuracy_validation_d"]},
            ]
            if goal_idx < len(GOALS):
                goal = GOALS[goal_idx]
                if ms_idx < len(goal["milestones"]):
                    milestone = goal["milestones"][ms_idx]
                else:
                    milestone = "done"
            else:
                milestone = "ALL_DONE"

            fails, errors = get_recent_fails()
            log(f"Iter={iteration} | {milestone} | fails={fails} | done={len(done)}/29")

            if milestone == "ALL_DONE":
                log("ALL GOALS COMPLETE!")
                break

            if fails >= FAIL_THRESHOLD:
                log(f"Stuck on {milestone} ({fails} fails). Asking Claude...")
                intervention_count += 1

                # Get the files relevant to this milestone
                code_file = f"pipeline/{milestone.replace('_module','')}.py"
                current_code = get_current_code(code_file)
                if not current_code:
                    current_code = get_current_code("pipeline/detection.py")

                try:
                    resp = ask_claude_for_fix(milestone, errors, current_code)
                    applied = parse_and_apply(resp, milestone)
                    log(f"Fix applied: {applied} (intervention #{intervention_count})")
                except Exception as e:
                    log(f"Claude fix failed: {e}")

            # Make sure autodev is still running
            if not is_autodev_running():
                log("Autodev not running! Restarting...")
                restart_autodev()

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            log("Supervisor stopped")
            break
        except Exception as e:
            log(f"Supervisor error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    if not ANTHROPIC_API_KEY:
        print("Set ANTHROPIC_API_KEY")
        exit(1)
    main()
