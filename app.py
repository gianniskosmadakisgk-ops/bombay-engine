import os
import json
import subprocess
from datetime import datetime
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
    # If ADMIN_KEY is not set, protection is disabled (your call)
    if not ADMIN_KEY:
        return None
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
SCRIPT_TIMEOUT_SEC = int(os.environ.get("SCRIPT_TIMEOUT_SEC", "120"))
MAX_LOG_CHARS = int(os.environ.get("MAX_LOG_CHARS", "8000"))

def run_script(script_rel_path: str):
    script_full = abs_path(script_rel_path)

    try:
        print(f"▶️ Running script: {script_rel_path}", flush=True)

        result = subprocess.run(
            ["python3", script_full],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=SCRIPT_TIMEOUT_SEC,
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
# MANUAL RUN ENDPOINTS (protected)
# ------------------------------------------------------
@app.route("/run/thursday-v3", methods=["GET"])
def manual_run_thursday_v3():
    guard = require_admin()
    if guard:
        return guard
    r = run_script("src/analysis/thursday_engine_full_v3.py")
    return jsonify({**r, "status": "ok" if r["ok"] else "error", "timestamp": datetime.utcnow().isoformat()})

@app.route("/run/friday-shortlist-v3", methods=["GET"])
def manual_run_friday_shortlist_v3():
    guard = require_admin()
    if guard:
        return guard
    r = run_script("src/analysis/friday_shortlist_v3.py")
    return jsonify({**r, "status": "ok" if r["ok"] else "error", "timestamp": datetime.utcnow().isoformat()})

# ------------------------------------------------------
# DOWNLOAD ENDPOINTS (protected)
# ------------------------------------------------------
@app.route("/download/thursday-report-v3", methods=["GET"])
def download_thursday_report_v3():
    guard = require_admin()
    if guard:
        return guard
    full_path = abs_path("logs/thursday_report_v3.json")
    if not os.path.exists(full_path):
        return jsonify({
            "status": "error",
            "message": "Thursday report file not found",
            "path": full_path,
            "timestamp": datetime.utcnow().isoformat()
        }), 404
    return send_file(full_path, mimetype="application/json", as_attachment=True)

@app.route("/download/friday-shortlist-v3", methods=["GET"])
def download_friday_shortlist_v3():
    guard = require_admin()
    if guard:
        return guard
    full_path = abs_path("logs/friday_shortlist_v3.json")
    if not os.path.exists(full_path):
        return jsonify({
            "status": "error",
            "message": "Friday shortlist v3 file not found",
            "path": full_path,
            "timestamp": datetime.utcnow().isoformat()
        }), 404
    return send_file(full_path, mimetype="application/json", as_attachment=True)

@app.route("/download/tuesday-recap-v2", methods=["GET"])
def download_tuesday_recap_v2():
    guard = require_admin()
    if guard:
        return guard
    full_path = abs_path("logs/tuesday_recap_v2.json")
    if not os.path.exists(full_path):
        return jsonify({
            "status": "error",
            "message": "Tuesday recap v2 file not found",
            "path": full_path,
            "timestamp": datetime.utcnow().isoformat()
        }), 404
    return send_file(full_path, mimetype="application/json", as_attachment=True)

# ------------------------------------------------------
# GPT ENDPOINTS (public - run + return report)
# ------------------------------------------------------
@app.route("/thursday-analysis-v3", methods=["GET"])
def api_thursday_analysis_v3():
    r = run_script("src/analysis/thursday_engine_full_v3.py")
    if not r["ok"]:
        return jsonify({
            "status": "error",
            "message": "Thursday engine failed",
            "run": r,
            "timestamp": datetime.utcnow().isoformat(),
            "report": None
        }), 500

    full_report, error = load_json_report("logs/thursday_report_v3.json")
    if full_report is None:
        return jsonify({
            "status": "error",
            "message": "Thursday report not available",
            "error": error,
            "run": r,
            "timestamp": datetime.utcnow().isoformat(),
            "report": None
        }), 500

    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "run": r,
        "report": full_report
    })

@app.route("/friday-shortlist-v3", methods=["GET"])
def api_friday_shortlist_v3():
    r = run_script("src/analysis/friday_shortlist_v3.py")
    if not r["ok"]:
        return jsonify({
            "status": "error",
            "message": "Friday shortlist failed",
            "run": r,
            "timestamp": datetime.utcnow().isoformat(),
            "report": None
        }), 500

    report, error = load_json_report("logs/friday_shortlist_v3.json")
    if report is None:
        return jsonify({
            "status": "error",
            "message": "Friday shortlist v3 not available",
            "error": error,
            "run": r,
            "timestamp": datetime.utcnow().isoformat(),
            "report": None
        }), 500

    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "run": r,
        "report": report
    })

@app.route("/tuesday-recap", methods=["GET"])
def api_tuesday_recap():
    report, error = load_json_report("logs/tuesday_recap_v2.json")
    if report is None:
        return jsonify({
            "status": "error",
            "message": "Tuesday recap v2 not available",
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
            "report": None
        }), 500
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "report": report
    })

# ------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting Bombay Engine Flask Server on port {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, use_reloader=False)
