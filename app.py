import os
import json
import subprocess
from datetime import datetime
from flask import Flask, jsonify, send_file, request, Response

app = Flask(__name__)

# ------------------------------------------------------
# Project root (εκεί που είναι το app.py)
# ------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def abs_path(rel_path: str) -> str:
    return os.path.join(PROJECT_ROOT, rel_path)

# ------------------------------------------------------
# Basic admin protection (optional)
# Set ADMIN_KEY env var on Render (optional)
# ------------------------------------------------------
ADMIN_KEY = os.environ.get("ADMIN_KEY", "").strip()

def require_admin():
    # If ADMIN_KEY is not set, protection is disabled
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
SCRIPT_TIMEOUT_SEC = int(os.environ.get("SCRIPT_TIMEOUT_SEC", "180"))
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

        if stdout:
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
# Load / Save JSON report from logs/
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

def save_json_report(report_rel_path: str, obj: dict):
    full_path = abs_path(report_rel_path)
    os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return full_path

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
# UPLOAD (for free Render: re-inject Friday JSON before Tuesday)
# ------------------------------------------------------
UPLOAD_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Bombay Upload</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; }
    .box { max-width: 760px; padding: 16px; border: 1px solid #ddd; border-radius: 10px; }
    label { display:block; margin-top: 12px; }
    button { margin-top: 16px; padding: 10px 14px; }
    code { background:#f6f6f6; padding:2px 6px; border-radius:6px; }
  </style>
</head>
<body>
  <div class="box">
    <h2>Upload JSON into server logs</h2>
    <p>Ανεβάζεις JSON ώστε να υπάρχει στο <code>logs/</code> του server (free Render μπορεί να τα “χάνει”).</p>

    <form method="post" action="/upload" enctype="multipart/form-data">
      <label>Τύπος report</label>
      <select name="kind">
        <option value="thursday">Thursday Report V3 (logs/thursday_report_v3.json)</option>
        <option value="friday">Friday Shortlist V3 (logs/friday_shortlist_v3.json)</option>
        <option value="tuesday">Tuesday Recap V3 (logs/tuesday_recap_v3.json)</option>
      </select>

      <label>JSON αρχείο</label>
      <input type="file" name="file" accept=".json,application/json" required />

      <button type="submit">Upload</button>
    </form>

    <hr/>
    <p><b>Manual σειρά:</b></p>
    <ol>
      <li><code>/run/thursday-v3</code></li>
      <li><code>/run/friday-shortlist-v3</code></li>
      <li><code>/run/tuesday-recap</code></li>
    </ol>
  </div>
</body>
</html>
"""

@app.route("/upload", methods=["GET"])
def upload_page():
    return Response(UPLOAD_HTML, mimetype="text/html")

@app.route("/upload", methods=["POST"])
def upload_post():
    guard = require_admin()
    if guard:
        return guard

    kind = (request.form.get("kind") or "").strip().lower()
    f = request.files.get("file")
    if not f:
        return jsonify({"status": "error", "message": "missing file", "timestamp": datetime.utcnow().isoformat()}), 400

    try:
        payload = json.load(f)
    except Exception as e:
        return jsonify({"status": "error", "message": f"invalid json: {e}", "timestamp": datetime.utcnow().isoformat()}), 400

    if kind == "thursday":
        rel = "logs/thursday_report_v3.json"
    elif kind == "friday":
        rel = "logs/friday_shortlist_v3.json"
    elif kind == "tuesday":
        rel = "logs/tuesday_recap_v3.json"
    else:
        return jsonify({"status": "error", "message": "invalid kind", "timestamp": datetime.utcnow().isoformat()}), 400

    full = save_json_report(rel, payload)
    return jsonify({
        "status": "ok",
        "saved_to": rel,
        "full_path": full,
        "timestamp": datetime.utcnow().isoformat()
    })

# ------------------------------------------------------
# MANUAL RUN ENDPOINTS (protected if ADMIN_KEY set)
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

@app.route("/run/tuesday-recap-v3", methods=["GET"])
def manual_run_tuesday_recap_v3():
    guard = require_admin()
    if guard:
        return guard
    r = run_script("src/analysis/tuesday_recap_v3.py")
    return jsonify({**r, "status": "ok" if r["ok"] else "error", "timestamp": datetime.utcnow().isoformat()})

@app.route("/run/tuesday-recap", methods=["GET"])
def manual_run_tuesday_recap_alias():
    return manual_run_tuesday_recap_v3()

# ------------------------------------------------------
# DOWNLOAD ENDPOINTS (protected if ADMIN_KEY set)
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

@app.route("/download/tuesday-recap-v3", methods=["GET"])
def download_tuesday_recap_v3():
    guard = require_admin()
    if guard:
        return guard
    full_path = abs_path("logs/tuesday_recap_v3.json")
    if not os.path.exists(full_path):
        return jsonify({
            "status": "error",
            "message": "Tuesday recap v3 file not found",
            "path": full_path,
            "timestamp": datetime.utcnow().isoformat()
        }), 404
    return send_file(full_path, mimetype="application/json", as_attachment=True)

# ------------------------------------------------------
# GPT ENDPOINTS (public - report-only)
# ------------------------------------------------------
@app.route("/thursday-analysis-v3", methods=["GET"])
def api_thursday_analysis_v3():
    report, error = load_json_report("logs/thursday_report_v3.json")
    if report is None:
        return jsonify({
            "status": "error",
            "message": "Thursday report not available",
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
            "report": None
        }), 404
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "report": report
    })

@app.route("/friday-shortlist-v3", methods=["GET"])
def api_friday_shortlist_v3():
    report, error = load_json_report("logs/friday_shortlist_v3.json")
    if report is None:
        return jsonify({
            "status": "error",
            "message": "Friday shortlist v3 not available",
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
            "report": None
        }), 404
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "report": report
    })

@app.route("/tuesday-recap", methods=["GET"])
def api_tuesday_recap():
    report, error = load_json_report("logs/tuesday_recap_v3.json")
    if report is None:
        return jsonify({
            "status": "error",
            "message": "Tuesday recap v3 not available",
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
            "report": None
        }), 404
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "report": report
    })

@app.route("/tuesday-recap-v3", methods=["GET"])
def api_tuesday_recap_v3():
    return api_tuesday_recap()

# ------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting Bombay Engine Flask Server on port {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, use_reloader=False)
