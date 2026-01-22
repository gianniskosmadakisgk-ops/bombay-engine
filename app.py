import os
import json
import subprocess
from datetime import datetime
from flask import Flask, jsonify, send_file, request, Response

app = Flask(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def abs_path(rel_path: str) -> str:
    return os.path.join(PROJECT_ROOT, rel_path)

LOGS_DIR = abs_path("logs")

# Optional admin protection
ADMIN_KEY = os.environ.get("ADMIN_KEY", "").strip()

def require_admin():
    if not ADMIN_KEY:
        return None
    if request.headers.get("X-ADMIN-KEY") != ADMIN_KEY:
        return jsonify({
            "status": "error",
            "message": "unauthorized",
            "timestamp": datetime.utcnow().isoformat()
        }), 401
    return None

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

def atomic_write_json(full_path: str, obj: dict):
    os.makedirs(os.path.dirname(full_path) or ".", exist_ok=True)
    tmp_path = full_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, full_path)

def load_json_report(report_rel_path: str):
    full_path = abs_path(report_rel_path)
    if not os.path.exists(full_path):
        return None, f"Report file not found: {full_path}"
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, f"Failed to load report file {full_path}: {e}"

def save_json_report(report_rel_path: str, obj: dict):
    full_path = abs_path(report_rel_path)
    atomic_write_json(full_path, obj)
    return full_path

def list_logs_dir():
    try:
        if not os.path.exists(LOGS_DIR):
            return {"exists": False, "path": LOGS_DIR, "files": []}
        files = []
        for name in sorted(os.listdir(LOGS_DIR)):
            p = os.path.join(LOGS_DIR, name)
            try:
                files.append({
                    "name": name,
                    "size": os.path.getsize(p),
                    "mtime": datetime.utcfromtimestamp(os.path.getmtime(p)).isoformat() + "Z"
                })
            except Exception:
                files.append({"name": name})
        return {"exists": True, "path": LOGS_DIR, "files": files}
    except Exception as e:
        return {"exists": None, "path": LOGS_DIR, "error": str(e), "files": []}

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({
        "status": "ok",
        "message": "Bombay Engine alive",
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/debug/logs", methods=["GET"])
def debug_logs():
    guard = require_admin()
    if guard:
        return guard
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "project_root": PROJECT_ROOT,
        "logs": list_logs_dir(),
    })

# ---------------- Upload UI ----------------
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
    <form method="post" action="/upload" enctype="multipart/form-data">
      <label>Τύπος</label>
      <select name="kind">
        <option value="thursday">Thursday (logs/thursday_report_v3.json)</option>
        <option value="friday">Friday (logs/friday_shortlist_v3.json)</option>
        <option value="tuesday">Tuesday (logs/tuesday_recap_v3.json)</option>
        <option value="history">History (logs/tuesday_history_v3.json)</option>
      </select>
      <label>JSON αρχείο</label>
      <input type="file" name="file" accept=".json,application/json" required />
      <button type="submit">Upload</button>
    </form>
    <hr/>
    <p>Debug: <code>/debug/logs</code></p>
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
        return jsonify({"status": "error", "message": "missing file"}), 400

    try:
        payload = json.load(f)
    except Exception as e:
        return jsonify({"status": "error", "message": f"invalid json: {e}"}), 400

    if kind == "thursday":
        rel = "logs/thursday_report_v3.json"
    elif kind == "friday":
        rel = "logs/friday_shortlist_v3.json"
    elif kind == "tuesday":
        rel = "logs/tuesday_recap_v3.json"
    elif kind == "history":
        rel = "logs/tuesday_history_v3.json"
    else:
        return jsonify({"status": "error", "message": "invalid kind"}), 400

    full = save_json_report(rel, payload)

    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "saved_to": rel,
        "full_path": full,
        "exists_after_write": os.path.exists(full),
        "size_bytes": os.path.getsize(full) if os.path.exists(full) else None,
        "logs_dir": list_logs_dir(),
    })

# ---------------- RUN endpoints (manual) ----------------
@app.route("/run/thursday-v3", methods=["GET"])
def run_thursday():
    guard = require_admin()
    if guard:
        return guard
    r = run_script("src/analysis/thursday_engine_full_v3.py")
    return jsonify({**r, "status": "ok" if r["ok"] else "error", "timestamp": datetime.utcnow().isoformat()})

@app.route("/run/friday-shortlist-v3", methods=["GET"])
def run_friday():
    guard = require_admin()
    if guard:
        return guard
    r = run_script("src/analysis/friday_shortlist_v3.py")
    return jsonify({**r, "status": "ok" if r["ok"] else "error", "timestamp": datetime.utcnow().isoformat()})

@app.route("/run/tuesday-recap-v3", methods=["GET"])
def run_tuesday():
    guard = require_admin()
    if guard:
        return guard
    r = run_script("src/analysis/tuesday_recap_v3.py")
    return jsonify({**r, "status": "ok" if r["ok"] else "error", "timestamp": datetime.utcnow().isoformat()})

@app.route("/run/tuesday-recap", methods=["GET"])
def run_tuesday_alias():
    return run_tuesday()

# ---------------- DOWNLOAD endpoints ----------------
@app.route("/download/thursday-report-v3", methods=["GET"])
def download_thursday():
    guard = require_admin()
    if guard:
        return guard
    p = abs_path("logs/thursday_report_v3.json")
    if not os.path.exists(p):
        return jsonify({"status":"error","message":"missing","path":p,"logs":list_logs_dir()}), 404
    return send_file(p, mimetype="application/json", as_attachment=True)

@app.route("/download/friday-shortlist-v3", methods=["GET"])
def download_friday():
    guard = require_admin()
    if guard:
        return guard
    p = abs_path("logs/friday_shortlist_v3.json")
    if not os.path.exists(p):
        return jsonify({"status":"error","message":"missing","path":p,"logs":list_logs_dir()}), 404
    return send_file(p, mimetype="application/json", as_attachment=True)

@app.route("/download/tuesday-recap-v3", methods=["GET"])
def download_tuesday_recap():
    guard = require_admin()
    if guard:
        return guard
    p = abs_path("logs/tuesday_recap_v3.json")
    if not os.path.exists(p):
        return jsonify({"status":"error","message":"missing","path":p,"logs":list_logs_dir()}), 404
    return send_file(p, mimetype="application/json", as_attachment=True)

@app.route("/download/tuesday-history-v3", methods=["GET"])
def download_tuesday_history():
    guard = require_admin()
    if guard:
        return guard
    p = abs_path("logs/tuesday_history_v3.json")
    if not os.path.exists(p):
        return jsonify({"status":"error","message":"missing","path":p,"logs":list_logs_dir()}), 404
    return send_file(p, mimetype="application/json", as_attachment=True)

# ---------------- GPT read endpoints (report-only) ----------------
def _group_leagues(report: dict) -> list:
    # Prefer engine_leagues ordering if present
    eng = report.get("engine_leagues")
    if isinstance(eng, list) and eng:
        return [str(x) for x in eng]
    # fallback: derive from fixtures
    leagues = []
    seen = set()
    for fx in report.get("fixtures", []) or []:
        lg = fx.get("league")
        if lg and lg not in seen:
            seen.add(lg)
            leagues.append(lg)
    return leagues

def _apply_lite_fixture(fx: dict) -> dict:
    # Keep only what Presenter needs (drop big fields)
    drop = {
        "league_baseline",
        "snap_home_prob","snap_draw_prob","snap_away_prob","snap_over_2_5_prob","snap_under_2_5_prob",
    }
    out = {}
    for k, v in fx.items():
        if k in drop:
            continue
        out[k] = v
    return out

@app.route("/thursday-analysis-v3", methods=["GET"])
def gpt_thursday():
    report, error = load_json_report("logs/thursday_report_v3.json")
    if report is None:
        return jsonify({
            "status":"error",
            "message":"Thursday report not available",
            "error":error,
            "timestamp":datetime.utcnow().isoformat(),
            "report":None
        }), 404

    # Optional chunking to avoid ResponseTooLargeError in GPT connector
    cursor = request.args.get("cursor", default=None, type=int)
    per_page = request.args.get("per_page", default=3, type=int)
    lite = request.args.get("lite", default="false").lower() in ("1","true","yes","y")

    leagues = _group_leagues(report)
    total_leagues = len(leagues)

    # If cursor is provided OR lite requested, we attach chunk info.
    if cursor is not None:
        if cursor < 0:
            cursor = 0
        if per_page < 1:
            per_page = 3
        if per_page > 10:
            per_page = 10

        start = cursor
        end = min(cursor + per_page, total_leagues)
        page_leagues = leagues[start:end]
        next_cursor = end if end < total_leagues else None

        fixtures = report.get("fixtures", []) or []
        filtered = [fx for fx in fixtures if fx.get("league") in set(page_leagues)]

        if lite:
            filtered = [_apply_lite_fixture(fx) for fx in filtered]

        out_report = dict(report)
        out_report["fixtures_total_full"] = int(report.get("fixtures_total") or len(fixtures))
        out_report["leagues_total"] = total_leagues
        out_report["fixtures"] = filtered
        out_report["fixtures_total"] = len(filtered)
        out_report["chunk"] = {
            "cursor": start,
            "per_page": per_page,
            "leagues": page_leagues,
            "next_cursor": next_cursor,
            "total_leagues": total_leagues,
        }

        return jsonify({
            "status":"ok",
            "timestamp": datetime.utcnow().isoformat(),
            "report": out_report
        })

    # No cursor: return full (optionally lite)
    if lite:
        fixtures = report.get("fixtures", []) or []
        out_report = dict(report)
        out_report["fixtures"] = [_apply_lite_fixture(fx) for fx in fixtures]
        out_report["fixtures_total"] = len(out_report["fixtures"])
        out_report["chunk"] = None
        return jsonify({
            "status":"ok",
            "timestamp": datetime.utcnow().isoformat(),
            "report": out_report
        })

    return jsonify({
        "status":"ok",
        "timestamp": datetime.utcnow().isoformat(),
        "report": report
    })

@app.route("/friday-shortlist-v3", methods=["GET"])
def gpt_friday():
    report, error = load_json_report("logs/friday_shortlist_v3.json")
    if report is None:
        return jsonify({
            "status":"error",
            "message":"Friday shortlist v3 not available",
            "error":error,
            "timestamp":datetime.utcnow().isoformat(),
            "report":None
        }), 404
    return jsonify({
        "status":"ok",
        "timestamp":datetime.utcnow().isoformat(),
        "report":report
    })

@app.route("/tuesday-recap", methods=["GET"])
def gpt_tuesday():
    report, error = load_json_report("logs/tuesday_recap_v3.json")
    if report is None:
        return jsonify({
            "status":"error",
            "message":"Tuesday recap v3 not available",
            "error":error,
            "timestamp":datetime.utcnow().isoformat(),
            "report":None
        }), 404
    return jsonify({
        "status":"ok",
        "timestamp":datetime.utcnow().isoformat(),
        "report":report
    })

@app.route("/tuesday-recap-v3", methods=["GET"])
def gpt_tuesday_v3():
    return gpt_tuesday()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting Bombay Engine Flask Server on port {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, use_reloader=False)
