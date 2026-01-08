import os
import json
import subprocess
import base64
import requests
from datetime import datetime, date
from flask import Flask, jsonify, send_file, request

app = Flask(__name__)

# ------------------------------------------------------
# Project root (εκεί που είναι το app.py)
# ------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def abs_path(rel_path: str) -> str:
    return os.path.join(PROJECT_ROOT, rel_path)

# ------------------------------------------------------
# Basic admin protection (optional but recommended)
# Set ADMIN_KEY env var on Render
# ------------------------------------------------------
ADMIN_KEY = os.environ.get("ADMIN_KEY", "").strip()

def require_admin():
    if not ADMIN_KEY:
        return None  # protection disabled
    if request.headers.get("X-ADMIN-KEY") != ADMIN_KEY:
        return jsonify({
            "status": "error",
            "message": "unauthorized",
            "timestamp": datetime.utcnow().isoformat()
        }), 401
    return None

# ------------------------------------------------------
# Run script with correct cwd + absolute path
# ------------------------------------------------------
SCRIPT_TIMEOUT_SEC = int(os.environ.get("SCRIPT_TIMEOUT_SEC", "240"))
MAX_LOG_CHARS = int(os.environ.get("MAX_LOG_CHARS", "8000"))

def run_script(script_rel_path: str, extra_env: dict | None = None):
    script_full = abs_path(script_rel_path)
    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    try:
        print(f"▶️ Running script: {script_rel_path}", flush=True)

        result = subprocess.run(
            ["python3", script_full],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT_SEC,
            env=env,
        )

        stdout = (result.stdout or "")[:MAX_LOG_CHARS]
        stderr = (result.stderr or "")[:MAX_LOG_CHARS]

        print("----- SCRIPT OUTPUT START -----", flush=True)
        print(stdout, flush=True)
        print("----- SCRIPT OUTPUT END -----", flush=True)

        if stderr:
            print("⚠️ SCRIPT ERRORS:", flush=True)
            print(stderr, flush=True)

        return {
            "ok": (result.returncode == 0),
            "return_code": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "script": script_rel_path,
        }

    except subprocess.TimeoutExpired:
        msg = f"Timeout: script exceeded {SCRIPT_TIMEOUT_SEC}s"
        print(f"⏳ {msg} ({script_rel_path})", flush=True)
        return {
            "ok": False,
            "return_code": -2,
            "stdout": "",
            "stderr": msg,
            "script": script_rel_path,
        }

    except Exception as e:
        print(f"❌ Error running {script_rel_path}: {e}", flush=True)
        return {
            "ok": False,
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
            "script": script_rel_path,
        }

# ------------------------------------------------------
# Load JSON report from logs/
# ------------------------------------------------------
def load_json_report(report_rel_path: str):
    full_path = abs_path(report_rel_path)

    if not os.path.exists(full_path):
        msg = f"Report file not found: {full_path}"
        print(f"⚠️ {msg}", flush=True)
        return None, msg

    try:
        with open(full_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, None
    except Exception as e:
        msg = f"Failed to load report file {full_path}: {e}"
        print(f"⚠️ {msg}", flush=True)
        return None, msg

def save_json_report(report_rel_path: str, payload: dict):
    full_path = abs_path(report_rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------
# WEEK ID (ISO Week) for naming
# ------------------------------------------------------
def iso_week_id_from_report(report: dict | None):
    # Prefer report.window.from if exists, else today
    d = None
    try:
        w = (report or {}).get("window") or {}
        frm = w.get("from") or w.get("start") or None
        if frm:
            d = date.fromisoformat(frm)
    except Exception:
        d = None

    if d is None:
        d = datetime.utcnow().date()

    y, w, _ = d.isocalendar()
    return f"{y}-W{int(w):02d}"

# ------------------------------------------------------
# OPTIONAL: GitHub archive (free "storage")
# Env:
#   GITHUB_TOKEN, GITHUB_OWNER, GITHUB_REPO
#   optional: GITHUB_BRANCH=main, GITHUB_ARCHIVE_ENABLED=true
# ------------------------------------------------------
GITHUB_ARCHIVE_ENABLED = os.getenv("GITHUB_ARCHIVE_ENABLED", "false").lower() == "true"
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_OWNER  = os.getenv("GITHUB_OWNER", "").strip()
GITHUB_REPO   = os.getenv("GITHUB_REPO", "").strip()
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main").strip()

def _gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

def _gh_get_file_sha(path_in_repo: str):
    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{path_in_repo}"
    r = requests.get(url, headers=_gh_headers(), params={"ref": GITHUB_BRANCH}, timeout=20)
    if r.status_code == 200:
        js = r.json()
        return js.get("sha")
    return None

def archive_json_to_github(report_rel_path: str, report_obj: dict):
    if not GITHUB_ARCHIVE_ENABLED:
        return {"archived": False, "reason": "disabled"}

    if not (GITHUB_TOKEN and GITHUB_OWNER and GITHUB_REPO):
        return {"archived": False, "reason": "missing_github_env"}

    week_id = iso_week_id_from_report(report_obj)
    filename = os.path.basename(report_rel_path)
    path_in_repo = f"logs/archive/{week_id}/{filename}"

    content_bytes = json.dumps(report_obj, ensure_ascii=False, indent=2).encode("utf-8")
    content_b64 = base64.b64encode(content_bytes).decode("utf-8")

    sha = _gh_get_file_sha(path_in_repo)

    url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{path_in_repo}"
    body = {
        "message": f"archive: {filename} ({week_id})",
        "content": content_b64,
        "branch": GITHUB_BRANCH,
    }
    if sha:
        body["sha"] = sha

    try:
        r = requests.put(url, headers=_gh_headers(), json=body, timeout=25)
        if r.status_code in (200, 201):
            return {"archived": True, "path": path_in_repo, "week_id": week_id}
        return {"archived": False, "reason": f"github_status_{r.status_code}", "detail": r.text[:500]}
    except Exception as e:
        return {"archived": False, "reason": "github_exception", "detail": str(e)}

# ------------------------------------------------------
# HEALTHCHECK
# ------------------------------------------------------
@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({
        "status": "ok",
        "message": "Bombay Engine alive",
        "timestamp": datetime.utcnow().isoformat()
    })

# ------------------------------------------------------
# MANUAL RUN ENDPOINTS (admin only)
# These run scripts and overwrite logs/ (expected).
# ------------------------------------------------------
@app.route("/run/thursday-v3", methods=["GET"])
def run_thursday_v3():
    guard = require_admin()
    if guard:
        return guard

    r = run_script("src/analysis/thursday_engine_full_v3.py")
    if not r["ok"]:
        return jsonify({**r, "status": "error", "timestamp": datetime.utcnow().isoformat()}), 500

    report, _ = load_json_report("logs/thursday_report_v3.json")
    arch = archive_json_to_github("logs/thursday_report_v3.json", report) if report else {"archived": False}
    return jsonify({**r, "status": "ok", "timestamp": datetime.utcnow().isoformat(), "archive": arch})

@app.route("/run/friday-shortlist-v3", methods=["GET"])
def run_friday_v3():
    guard = require_admin()
    if guard:
        return guard

    r = run_script("src/analysis/friday_shortlist_v3.py")
    if not r["ok"]:
        return jsonify({**r, "status": "error", "timestamp": datetime.utcnow().isoformat()}), 500

    report, _ = load_json_report("logs/friday_shortlist_v3.json")
    arch = archive_json_to_github("logs/friday_shortlist_v3.json", report) if report else {"archived": False}
    return jsonify({**r, "status": "ok", "timestamp": datetime.utcnow().isoformat(), "archive": arch})

# NEW: Tuesday recap v3 run
@app.route("/run/tuesday-recap-v3", methods=["GET"])
def run_tuesday_v3():
    guard = require_admin()
    if guard:
        return guard

    # optional: choose which Friday file to use (for old weekend recap)
    # example: /run/tuesday-recap-v3?friday_path=logs/archive/2026-W02/friday_shortlist_v3.json
    friday_path = request.args.get("friday_path")
    extra_env = {}
    if friday_path:
        extra_env["FRIDAY_REPORT_PATH"] = abs_path(friday_path) if not friday_path.startswith("/") else friday_path

    r = run_script("src/analysis/tuesday_recap_v3.py", extra_env=extra_env)
    if not r["ok"]:
        return jsonify({**r, "status": "error", "timestamp": datetime.utcnow().isoformat()}), 500

    report, _ = load_json_report("logs/tuesday_recap_v3.json")
    arch = archive_json_to_github("logs/tuesday_recap_v3.json", report) if report else {"archived": False}
    return jsonify({**r, "status": "ok", "timestamp": datetime.utcnow().isoformat(), "archive": arch})

# Backwards alias (what you already hit)
@app.route("/run/tuesday-recap", methods=["GET"])
def run_tuesday_alias():
    return run_tuesday_v3()

# ------------------------------------------------------
# REPORT ENDPOINTS (PUBLIC) — DO NOT run scripts
# These are the ones for Custom GPT actions.
# Returns RAW JSON ONLY (no wrappers).
# ------------------------------------------------------
@app.route("/thursday-analysis-v3", methods=["GET"])
def thursday_analysis_v3():
    report, error = load_json_report("logs/thursday_report_v3.json")
    if report is None:
        return jsonify({"status": "error", "message": "thursday_report_v3_missing", "error": error}), 404
    return jsonify(report)

@app.route("/friday-shortlist-v3", methods=["GET"])
def friday_shortlist_v3():
    report, error = load_json_report("logs/friday_shortlist_v3.json")
    if report is None:
        return jsonify({"status": "error", "message": "friday_shortlist_v3_missing", "error": error}), 404
    return jsonify(report)

# Tuesday recap public (v3)
@app.route("/tuesday-recap", methods=["GET"])
def tuesday_recap():
    report, error = load_json_report("logs/tuesday_recap_v3.json")
    if report is None:
        return jsonify({"status": "error", "message": "tuesday_recap_v3_missing", "error": error}), 404
    return jsonify(report)

# Optional explicit v3 path
@app.route("/tuesday-recap-v3", methods=["GET"])
def tuesday_recap_v3():
    return tuesday_recap()

# ------------------------------------------------------
# DOWNLOAD ENDPOINTS (admin)
# ------------------------------------------------------
@app.route("/download/thursday-report-v3", methods=["GET"])
def download_thursday_report_v3():
    guard = require_admin()
    if guard:
        return guard
    full_path = abs_path("logs/thursday_report_v3.json")
    if not os.path.exists(full_path):
        return jsonify({"status": "error", "message": "file_not_found", "path": full_path}), 404
    return send_file(full_path, mimetype="application/json", as_attachment=True)

@app.route("/download/friday-shortlist-v3", methods=["GET"])
def download_friday_shortlist_v3():
    guard = require_admin()
    if guard:
        return guard
    full_path = abs_path("logs/friday_shortlist_v3.json")
    if not os.path.exists(full_path):
        return jsonify({"status": "error", "message": "file_not_found", "path": full_path}), 404
    return send_file(full_path, mimetype="application/json", as_attachment=True)

@app.route("/download/tuesday-recap-v3", methods=["GET"])
def download_tuesday_recap_v3():
    guard = require_admin()
    if guard:
        return guard
    full_path = abs_path("logs/tuesday_recap_v3.json")
    if not os.path.exists(full_path):
        return jsonify({"status": "error", "message": "file_not_found", "path": full_path}), 404
    return send_file(full_path, mimetype="application/json", as_attachment=True)

# ------------------------------------------------------
# OPTIONAL: SEED upload (admin)
# Use when you have an OLD Friday JSON and want Tuesday recap for that,
# WITHOUT running Friday again.
# POST JSON body to overwrite logs/friday_shortlist_v3.json
# ------------------------------------------------------
@app.route("/admin/seed/friday-shortlist-v3", methods=["POST"])
def seed_friday():
    guard = require_admin()
    if guard:
        return guard

    try:
        payload = request.get_json(force=True)
        if not isinstance(payload, dict):
            return jsonify({"status": "error", "message": "invalid_json"}), 400
        save_json_report("logs/friday_shortlist_v3.json", payload)
        return jsonify({"status": "ok", "message": "seeded_friday_shortlist_v3"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": "seed_failed", "detail": str(e)}), 500

# ------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting Bombay Engine Flask Server on port {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, use_reloader=False)
