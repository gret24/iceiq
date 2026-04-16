import subprocess
import json
import os
import time
import requests
from datetime import datetime

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
PROJECT_DIR = os.path.expanduser("~/iceiq-dev")
MAX_ITERATIONS = 2000
MAX_ERRORS = 5
STATE_FILE = os.path.join(PROJECT_DIR, "autodev_state.json")

SYSTEM_PROMPT = """You are building IceIQ, an ice hockey video analysis platform.
Mac Mini M4, MPS GPU, conda iceiq, Python 3.10.
Video: data/videos/game1.mp4 (youth hockey, fixed wide-angle, full rink).
COMPLETED: detection, tracking(ByteTrack), team_classification(99.3%), ROI, basic identification(shot_hand, position), clustering.
IN PROGRESS: homography, SmartJersey v2.
STRATEGIES in ~/iceiq-dev/configs/:
- identification_v2.json: Cluster first, label later. 7 signals.
- player_metrics.json: 30+ ability metrics.
- rink_detection.json: Auto rink detection with board+goal+lines.

Respond with exactly ONE action:

WRITE_FILE
PATH: pipeline/some/file.py
CONTENT:
```python
[complete code]
```
or
RUN_TEST
CMD: cd ~/iceiq-dev && python3 -c "..."
or
DONE
MILESTONE: milestone_name
NEXT: what to do next

Rules: Complete files only. No TODO. Use device=mps. Test after every write."""

GOALS = [
    {"id": "path_b", "name": "Path B", "milestones": ["homography", "kalman_kinematics", "heatmap_generation", "smart_jersey_labeling", "full_pipeline_test", "accuracy_validation"], "targets": {"all": 0.98}},
    {"id": "path_c", "name": "Path C", "milestones": ["puck_detection", "puck_possession", "team_analytics", "event_detection", "scoreboard_ocr", "fatigue_analysis", "pipeline_c_test"], "targets": {"all": 0.98}},
    {"id": "path_d", "name": "Path D", "milestones": ["highlight_scoring", "highlight_extraction", "cross_shift_identity", "recruiting_pdf", "pipeline_d_test"], "targets": {"all": 0.98}},
]

def notify(msg):
    text = "IceIQ: " + msg
    print(text)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(
                "https://api.telegram.org/bot" + TELEGRAM_BOT_TOKEN + "/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": text[:4000]},
                timeout=10)
        except:
            pass

def run_cmd(cmd, timeout=600):
    import tempfile, re
    try:
        # python3 -c "..." 또는 멀티라인 python3 코드를 임시 .py 파일로 실행
        py_match = re.search(r'python3\s+-c\s+"(.*)', cmd, re.DOTALL)
        if not py_match:
            py_match = re.search(r"python3\s+-c\s+'(.*)", cmd, re.DOTALL)
        if py_match:
            # 코드 부분 추출 (첫/마지막 따옴표 제거)
            py_code = py_match.group(1).strip()
            if py_code.endswith('"') or py_code.endswith("'"):
                py_code = py_code[:-1]
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='/tmp') as f:
                f.write(py_code)
                tmp_py = f.name
            try:
                r = subprocess.run(['python3', tmp_py], capture_output=True, text=True,
                                   timeout=timeout, cwd=PROJECT_DIR,
                                   env=dict(os.environ, PYTHONPATH=PROJECT_DIR, KMP_DUPLICATE_LIB_OK="TRUE"))
            finally:
                os.unlink(tmp_py)
        else:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                f.write('#!/bin/sh\n' + cmd + '\n')
                tmp_sh = f.name
            os.chmod(tmp_sh, 0o755)
            try:
                r = subprocess.run(['/bin/sh', tmp_sh], capture_output=True, text=True,
                                   timeout=timeout, cwd=PROJECT_DIR,
                                   env=dict(os.environ, PYTHONPATH=PROJECT_DIR, KMP_DUPLICATE_LIB_OK="TRUE"))
            finally:
                os.unlink(tmp_sh)
        return {"ok": r.returncode == 0, "out": r.stdout[-2000:], "err": r.stderr[-1000:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "out": "", "err": "TIMEOUT"}
    except Exception as e:
        return {"ok": False, "out": "", "err": str(e)}

def write_file(path, content):
    fp = os.path.join(PROJECT_DIR, path)
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp, "w") as f:
        f.write(content)
    return fp

def ask_gemini(prompt):
    # Try Gemini first, fallback to Claude Sonnet
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=" + gemini_key
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 8192}
        }
        try:
            r = requests.post(url, json=payload, timeout=120)
            if r.status_code == 200:
                data = r.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
            elif r.status_code == 429:
                print("  [Gemini quota exceeded, using Claude]")
        except Exception as e:
            print(f"  [Gemini error: {e}, using Claude]")

    # Claude fallback
    import anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        return "ERROR: no API key available"
    client = anthropic.Anthropic(api_key=anthropic_key)
    r = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}])
    return r.content[0].text

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"goal_idx": 0, "ms_idx": 0, "done": [], "iter": 0, "last_result": None}

def save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2, default=str)

def parse_action(resp):
    a = {"type": None}
    lines = resp.split("\n")
    i = 0
    while i < len(lines):
        l = lines[i].strip()
        if l == "WRITE_FILE":
            a["type"] = "write"
        elif l == "RUN_TEST":
            a["type"] = "run"
        elif l == "DONE":
            a["type"] = "done"
        elif l.startswith("PATH: "):
            a["path"] = l[6:].strip()
        elif l.startswith("CMD: "):
            cmd_lines = [l[5:].strip()]
            j = i + 1
            while j < len(lines):
                nl = lines[j].strip()
                if nl in ("WRITE_FILE", "RUN_TEST", "DONE", "") or nl.startswith("MILESTONE: ") or nl.startswith("NEXT: "):
                    break
                cmd_lines.append(lines[j])
                j += 1
            a["cmd"] = "\n".join(cmd_lines).strip()
            i = j - 1
        elif l.startswith("MILESTONE: "):
            a["milestone"] = l[11:].strip()
        elif l.startswith("NEXT: "):
            a["next"] = l[6:].strip()
        i += 1
    if a["type"] == "write":
        import re
        # Find opening code fence (```python or ```)
        m = re.search(r'```(?:python)?\s*\n', resp)
        if m:
            code_start = m.end()
            # Find closing fence after the opening
            close = resp.find("```", code_start)
            if close > code_start:
                a["content"] = resp[code_start:close]
            else:
                # truncated — take everything after opening
                a["content"] = resp[code_start:]
        else:
            # Try CONTENT: marker as fallback
            cm = resp.find("CONTENT:\n")
            if cm >= 0:
                a["content"] = resp[cm + 9:]
    return a

def get_files():
    files = []
    for root, _, fs in os.walk(os.path.join(PROJECT_DIR, "pipeline")):
        for f in fs:
            if f.endswith(".py"):
                files.append(os.path.relpath(os.path.join(root, f), PROJECT_DIR))
    return files

def main():
    state = load_state()
    errors = 0
    total_ms = sum(len(g["milestones"]) for g in GOALS)
    notify("Gemini AutoDev started. Done: " + str(len(state["done"])) + "/" + str(total_ms))

    while state["iter"] < MAX_ITERATIONS:
        try:
            if state["goal_idx"] >= len(GOALS):
                notify("ALL GOALS COMPLETE!")
                break

            goal = GOALS[state["goal_idx"]]
            if state["ms_idx"] >= len(goal["milestones"]):
                notify("Goal done: " + goal["name"])
                state["goal_idx"] += 1
                state["ms_idx"] = 0
                save_state(state)
                continue

            ms = goal["milestones"][state["ms_idx"]]
            prompt = "GOAL: " + goal["name"] + "\n"
            prompt += "MILESTONE: " + ms + "\n"
            prompt += "DONE: " + str(state["done"]) + "\n"
            prompt += "FILES: " + str(get_files()) + "\n"
            if state.get("last_result"):
                lr = state["last_result"]
                status = "SUCCESS" if lr.get("ok") else "FAILED"
                prompt += "LAST: " + str(lr.get("act", "")) + " [" + status + "]\n"
                if lr.get("out"):
                    prompt += "OUTPUT: " + lr["out"][-800:] + "\n"
                if lr.get("err"):
                    prompt += "ERROR: " + lr["err"][-500:] + "\n"
                if lr.get("ok") and lr.get("act", "").startswith("write "):
                    prompt += "NOTE: File written successfully. If milestone is complete, respond with DONE.\n"
            prompt += "Next action?"

            state["iter"] += 1
            resp = ask_gemini(prompt)

            if resp.startswith("ERROR:"):
                print("[" + str(state["iter"]) + "] API ERROR: " + resp[:200])
                state["last_result"] = {"act": "api_error", "ok": False, "err": resp[:300]}
                save_state(state)
                time.sleep(120)
                errors += 1
                if errors >= MAX_ERRORS:
                    notify("API errors. Pausing 10min.")
                    errors = 0
                    time.sleep(600)
                continue

            act = parse_action(resp)
            if act["type"] is None or (act["type"] == "write" and (not act.get("path") or not act.get("content"))):
                print("[DEBUG ACT] type=" + str(act["type"]) + " path=" + str(act.get("path")) + " has_content=" + str(bool(act.get("content"))))
                print("[DEBUG RESP] " + repr(resp[:500]))

            if act["type"] == "write" and act.get("path") and act.get("content"):
                write_file(act["path"], act["content"])
                state["last_result"] = {"act": "write " + act["path"], "ok": True}
                print("[" + str(state["iter"]) + "] WRITE " + act["path"])
                save_state(state)
                check = run_cmd("python3 -c \"import py_compile; py_compile.compile('" + os.path.join(PROJECT_DIR, act["path"]) + "', doraise=True)\"")
                if not check["ok"]:
                    state["last_result"]["ok"] = False
                    state["last_result"]["err"] = check["err"][-500:]
                    print("  SYNTAX ERROR")
                    save_state(state)
                time.sleep(30)

            elif act["type"] == "run" and act.get("cmd"):
                print("[" + str(state["iter"]) + "] TEST: " + act["cmd"][:80])
                r = run_cmd(act["cmd"], timeout=300)
                state["last_result"] = {
                    "act": "test", "ok": r["ok"],
                    "out": r["out"][-1000:], "err": r["err"][-500:]
                }
                print("  " + ("PASS" if r["ok"] else "FAIL"))
                save_state(state)
                time.sleep(10)

            elif act["type"] == "done":
                state["done"].append(ms)
                state["ms_idx"] += 1
                state["last_result"] = {"act": "milestone " + ms, "ok": True}
                done = len(state["done"])
                notify("Milestone: " + ms + " (" + str(done) + "/" + str(total_ms) + ")")
                save_state(state)
                time.sleep(5)

            else:
                state["last_result"] = {"act": "parse_error", "ok": False, "err": resp[:300]}
                print("[" + str(state["iter"]) + "] PARSE ERROR")
                save_state(state)
                time.sleep(60)

            if state["last_result"].get("ok"):
                errors = 0
            else:
                errors += 1
                if errors >= MAX_ERRORS:
                    notify(str(MAX_ERRORS) + " errors. Pausing 5min.")
                    errors = 0
                    time.sleep(300)

        except KeyboardInterrupt:
            notify("Stopped")
            save_state(state)
            break
        except Exception as e:
            print("EXCEPTION: " + str(e))
            notify("Error: " + str(e)[:300])
            save_state(state)
            time.sleep(60)

    save_state(state)
    notify("Ended. " + str(len(state["done"])) + "/" + str(total_ms))

if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("  [No GEMINI_API_KEY, running Claude-only mode]")
    main()
