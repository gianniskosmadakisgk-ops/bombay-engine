import os
import json
import subprocess
import importlib.util
from datetime import datetime
from flask import Flask, jsonify, send_file, request, Response

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

app = Flask(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

def abs_path(rel_path: str) -> str:
    return os.path.join(PROJECT_ROOT, rel_path)

LOGS_DIR = abs_path("logs")
DATA_DIR = abs_path("data")

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

PERSIST_TARGETS = {
    "thursday-v3": "logs/thursday_report_v3.json",
    "friday-v3": "logs/friday_shortlist_v3.json",
    "friday-v10-1": "logs/friday_shortlist_v10_1.json",
    "tuesday-recap-v3": "logs/tuesday_recap_v3.json",
    "tuesday-history-v3": "logs/tuesday_history_v3.json",
    "style-metrics": "data/team_style_metrics.json",
    "domestic-history": "data/team_domestic_history_last5.json",
    "market-values": "data/team_market_values.json",
}

def load_github_storage():
    p = abs_path("src/tools/github_storage_v1.py")
    if not os.path.exists(p):
        return None, f"missing github_storage_v1.py at {p}"

    try:
        spec = importlib.util.spec_from_file_location("github_storage_v1", p)
        if spec is None or spec.loader is None:
            return None, "failed to create import spec"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, None
    except Exception as e:
        return None, str(e)

def github_status():
    mod, error = load_github_storage()
    if mod is None:
        return {
            "available": False,
            "error": error,
            "repo": os.environ.get("GITHUB_REPO", ""),
            "branch": os.environ.get("GITHUB_BRANCH", "main"),
            "storage_prefix": os.environ.get("GITHUB_STORAGE_PREFIX", "persistent"),
            "token_present": bool(os.environ.get("GITHUB_TOKEN", "").strip()),
        }

    try:
        ok, msg = mod.ok_config()
        return {
            "available": bool(ok),
            "message": msg,
            "repo": mod.github_repo(),
            "branch": mod.github_branch(),
            "storage_prefix": mod.storage_prefix(),
            "token_present": bool(os.environ.get("GITHUB_TOKEN", "").strip()),
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "token_present": bool(os.environ.get("GITHUB_TOKEN", "").strip()),
        }

def persist_file(target_key: str):
    rel = PERSIST_TARGETS.get(target_key)
    if not rel:
        return {"status": "error", "message": "invalid target", "target_key": target_key}

    mod, error = load_github_storage()
    if mod is None:
        return {"status": "error", "message": error, "target_key": target_key}

    local_path = abs_path(rel)
    return mod.github_upload_json(
        local_path=local_path,
        target_relative_path=rel,
        message=f"Bombay persistent save: {target_key}"
    )

def restore_file(target_key: str):
    rel = PERSIST_TARGETS.get(target_key)
    if not rel:
        return {"status": "error", "message": "invalid target", "target_key": target_key}

    mod, error = load_github_storage()
    if mod is None:
        return {"status": "error", "message": error, "target_key": target_key}

    local_path = abs_path(rel)
    return mod.github_download_json(
        source_relative_path=rel,
        local_path=local_path
    )

def auto_persist_by_rel(rel: str):
    for key, value in PERSIST_TARGETS.items():
        if value == rel:
            return persist_file(key)
    return {"status": "skipped", "message": "no persist target for rel", "rel": rel}

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
            "ok": result.returncode == 0,
            "return_code": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "script": script_rel_path,
        }

    except subprocess.TimeoutExpired:
        msg = f"Timeout: script exceeded {SCRIPT_TIMEOUT_SEC}s"
        return {"ok": False, "return_code": -2, "stdout": "", "stderr": msg, "script": script_rel_path}

    except Exception as e:
        return {"ok": False, "return_code": -1, "stdout": "", "stderr": str(e), "script": script_rel_path}

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

def load_json_report_with_restore(report_rel_path: str, target_key: str):
    report, error = load_json_report(report_rel_path)
    if report is not None:
        return report, None, {"restore_attempted": False}

    restored = restore_file(target_key)
    if restored.get("status") == "ok":
        report2, error2 = load_json_report(report_rel_path)
        return report2, error2, {"restore_attempted": True, "restore": restored}

    return None, error, {"restore_attempted": True, "restore": restored}

def save_json_report(report_rel_path: str, obj: dict):
    full_path = abs_path(report_rel_path)
    atomic_write_json(full_path, obj)
    return full_path

def list_dir(p: str):
    try:
        if not os.path.exists(p):
            return {"exists": False, "path": p, "files": []}

        files = []
        for name in sorted(os.listdir(p)):
            fp = os.path.join(p, name)
            try:
                files.append({
                    "name": name,
                    "size": os.path.getsize(fp),
                    "mtime": datetime.utcfromtimestamp(os.path.getmtime(fp)).isoformat() + "Z"
                })
            except Exception:
                files.append({"name": name})

        return {"exists": True, "path": p, "files": files}

    except Exception as e:
        return {"exists": None, "path": p, "error": str(e), "files": []}

def list_logs_dir():
    return list_dir(LOGS_DIR)

def list_data_dir():
    return list_dir(DATA_DIR)

def _tz_convert_date_time(date_str: str, time_str: str, tz_name: str):
    if not ZoneInfo:
        return None, None
    if not date_str or not time_str:
        return None, None
    try:
        dt_utc = datetime.fromisoformat(f"{date_str}T{time_str}:00").replace(tzinfo=ZoneInfo("UTC"))
        dt_tz = dt_utc.astimezone(ZoneInfo(tz_name))
        return dt_tz.date().isoformat(), dt_tz.strftime("%H:%M")
    except Exception:
        return None, None

def _slim_fixture_for_display(fx: dict):
    keep = {
        "fixture_id", "date", "time", "league", "league_id", "home", "away",
        "home_prob", "draw_prob", "away_prob", "over_2_5_prob", "under_2_5_prob",
        "fair_1", "fair_x", "fair_2", "fair_over_2_5", "fair_under_2_5",
        "offered_1", "offered_x", "offered_2", "offered_over_2_5", "offered_under_2_5",
        "value_pct_1", "value_pct_x", "value_pct_2", "value_pct_over", "value_pct_under",
        "ev_1", "ev_x", "ev_2", "ev_over", "ev_under",
        "odds_match", "flags"
    }

    out = {k: fx.get(k) for k in keep if k in fx}

    flags = fx.get("flags") or {}
    out["flags"] = {
        "confidence": flags.get("confidence"),
        "confidence_band": flags.get("confidence_band"),
        "tight_game": flags.get("tight_game"),
        "home_shape": flags.get("home_shape"),
        "away_shape": flags.get("away_shape"),
        "draw_shape": flags.get("draw_shape"),
        "under_elite": flags.get("under_elite"),
        "over_good_shape": flags.get("over_good_shape"),
        "over_friendly_league": flags.get("over_friendly_league"),
        "draw_friendly_league": flags.get("draw_friendly_league"),
        "low_tempo_league": flags.get("low_tempo_league"),
        "odds_strict_ok": flags.get("odds_strict_ok"),
        "prob_instability": flags.get("prob_instability"),
        "snap_gap_max": flags.get("snap_gap_max"),
        "value_missing": flags.get("value_missing"),
        "history_missing": flags.get("history_missing"),
        "style_missing": flags.get("style_missing"),
    }

    om = fx.get("odds_match") or {}
    out["odds_match"] = {
        "matched": om.get("matched"),
        "score": om.get("score"),
        "grade": om.get("grade"),
        "reason": om.get("reason"),
    }

    return out

def _chunk_thursday_report(report: dict, cursor: int, per_page: int, lite: bool, tz: str):
    fixtures = report.get("fixtures") or []
    league_order = report.get("engine_leagues")

    if not isinstance(league_order, list) or not league_order:
        league_order = sorted({(f.get("league") or "") for f in fixtures if f.get("league")})

    total_leagues = len(league_order)
    cursor = max(0, int(cursor))
    per_page = max(1, min(3, int(per_page)))

    start = cursor * per_page
    end = start + per_page
    leagues_slice = league_order[start:end]

    fx_slice = [f for f in fixtures if f.get("league") in leagues_slice]

    if lite:
        fx_slice = [_slim_fixture_for_display(f) for f in fx_slice]

    tz_name = "Europe/Athens" if tz == "Europe/Athens" else "UTC"
    if tz_name != "UTC":
        for f in fx_slice:
            dgr, tgr = _tz_convert_date_time(f.get("date"), f.get("time"), tz_name)
            f["date_gr"] = dgr
            f["time_gr"] = tgr

    next_cursor = cursor + 1 if end < total_leagues else None

    return {
        "generated_at": report.get("generated_at"),
        "season_used": report.get("season_used"),
        "window": report.get("window"),
        "engine_leagues": league_order,
        "fixtures_total": report.get("fixtures_total", len(fixtures)),
        "fixtures": fx_slice,
        "chunk": {
            "cursor": cursor,
            "per_page": per_page,
            "leagues": leagues_slice,
            "next_cursor": next_cursor,
            "total_leagues": total_leagues,
            "lite": bool(lite),
            "tz": tz_name,
        }
    }

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

    friday_v10_1_full = abs_path("logs/friday_shortlist_v10_1.json")
    friday_v10_1_resolved = friday_v10_1_full if os.path.exists(friday_v10_1_full) else None

    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "project_root": PROJECT_ROOT,
        "logs": list_logs_dir(),
        "data": list_data_dir(),
        "github_storage": github_status(),
        "friday_v10_1_full_path": friday_v10_1_full,
        "friday_v10_1_resolved": friday_v10_1_resolved,
        "persist_targets": PERSIST_TARGETS,
    })

UPLOAD_HTML = """<!doctype html>
<html>
<head><meta charset="utf-8"/><title>Bombay Upload</title></head>
<body style="font-family:Arial;margin:24px">
  <h2>Upload JSON into server</h2>
  <p><b>Important:</b> Uploads now also try to persist to GitHub.</p>
  <form method="post" action="/upload" enctype="multipart/form-data">
    <label>Τύπος</label>
    <select name="kind">
      <option value="thursday">Thursday report V3 (logs/thursday_report_v3.json)</option>
      <option value="friday">Friday shortlist V3 (logs/friday_shortlist_v3.json)</option>
      <option value="friday_v10_1">Friday shortlist V10.1 (logs/friday_shortlist_v10_1.json)</option>
      <option value="tuesday">Tuesday recap V3 (logs/tuesday_recap_v3.json)</option>
      <option value="history">Tuesday history V3 (logs/tuesday_history_v3.json)</option>
      <option value="style">Style metrics (data/team_style_metrics.json)</option>
      <option value="domestic_history">Domestic history (data/team_domestic_history_last5.json)</option>
      <option value="market_values">Market values (data/team_market_values.json)</option>
    </select>
    <br/><br/>
    <label>JSON αρχείο</label>
    <input type="file" name="file" accept=".json,application/json" required />
    <br/><br/>
    <button type="submit">Upload</button>
  </form>
  <hr/>
  <p>Debug: <code>/debug/logs</code></p>
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
    elif kind == "friday_v10_1":
        rel = "logs/friday_shortlist_v10_1.json"
    elif kind == "tuesday":
        rel = "logs/tuesday_recap_v3.json"
    elif kind == "history":
        rel = "logs/tuesday_history_v3.json"
    elif kind == "style":
        rel = "data/team_style_metrics.json"
    elif kind == "domestic_history":
        rel = "data/team_domestic_history_last5.json"
    elif kind == "market_values":
        rel = "data/team_market_values.json"
    else:
        return jsonify({"status": "error", "message": "invalid kind"}), 400

    full = save_json_report(rel, payload)
    persist_result = auto_persist_by_rel(rel)

    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "saved_to": rel,
        "full_path": full,
        "exists_after_write": os.path.exists(full),
        "size_bytes": os.path.getsize(full) if os.path.exists(full) else None,
        "github_persist": persist_result,
        "logs_dir": list_logs_dir(),
        "data_dir": list_data_dir(),
    })

# ---------------- PERSIST / RESTORE endpoints ----------------

@app.route("/persist/<target_key>", methods=["GET"])
def persist_target(target_key):
    guard = require_admin()
    if guard:
        return guard

    result = persist_file(target_key)
    status_code = 200 if result.get("status") == "ok" else 400
    return jsonify({
        "status": result.get("status"),
        "timestamp": datetime.utcnow().isoformat(),
        "target_key": target_key,
        "result": result,
    }), status_code

@app.route("/restore/<target_key>", methods=["GET"])
def restore_target(target_key):
    guard = require_admin()
    if guard:
        return guard

    result = restore_file(target_key)
    status_code = 200 if result.get("status") == "ok" else 400
    return jsonify({
        "status": result.get("status"),
        "timestamp": datetime.utcnow().isoformat(),
        "target_key": target_key,
        "result": result,
        "logs": list_logs_dir(),
        "data": list_data_dir(),
    }), status_code

@app.route("/github/check/<target_key>", methods=["GET"])
def github_check_target(target_key):
    guard = require_admin()
    if guard:
        return guard

    rel = PERSIST_TARGETS.get(target_key)
    if not rel:
        return jsonify({"status": "error", "message": "invalid target", "target_key": target_key}), 400

    mod, error = load_github_storage()
    if mod is None:
        return jsonify({"status": "error", "message": error}), 400

    result = mod.github_file_exists(rel)
    return jsonify({
        "status": result.get("status"),
        "timestamp": datetime.utcnow().isoformat(),
        "target_key": target_key,
        "result": result,
    })

# ---------------- RUN endpoints ----------------

@app.route("/run/thursday-v3", methods=["GET"])
def run_thursday():
    guard = require_admin()
    if guard:
        return guard

    r = run_script("src/analysis/thursday_engine_full_v3.py")
    persist_result = persist_file("thursday-v3") if r["ok"] else {"status": "skipped", "reason": "script_failed"}

    return jsonify({
        **r,
        "status": "ok" if r["ok"] else "error",
        "github_persist": persist_result,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/run/friday-shortlist-v3", methods=["GET"])
def run_friday():
    guard = require_admin()
    if guard:
        return guard

    r = run_script("src/analysis/friday_shortlist_v3.py")
    persist_result = persist_file("friday-v3") if r["ok"] else {"status": "skipped", "reason": "script_failed"}

    return jsonify({
        **r,
        "status": "ok" if r["ok"] else "error",
        "github_persist": persist_result,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/run/friday-v10-1", methods=["GET"])
def run_friday_v10_1():
    guard = require_admin()
    if guard:
        return guard

    r = run_script("src/analysis/friday_shortlist_v10_1.py")
    persist_result = persist_file("friday-v10-1") if r["ok"] else {"status": "skipped", "reason": "script_failed"}

    return jsonify({
        **r,
        "status": "ok" if r["ok"] else "error",
        "github_persist": persist_result,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/run/friday-shortlist-v10-1", methods=["GET"])
def run_friday_shortlist_v10_1_alias():
    return run_friday_v10_1()

@app.route("/run/tuesday-recap-v3", methods=["GET"])
def run_tuesday():
    guard = require_admin()
    if guard:
        return guard

    r = run_script("src/analysis/tuesday_recap_v3.py")
    persist_result = persist_file("tuesday-recap-v3") if r["ok"] else {"status": "skipped", "reason": "script_failed"}

    return jsonify({
        **r,
        "status": "ok" if r["ok"] else "error",
        "github_persist": persist_result,
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/run/tuesday-recap", methods=["GET"])
def run_tuesday_alias():
    return run_tuesday()

@app.route("/run/style-builder-v1", methods=["GET"])
def run_style_builder():
    guard = require_admin()
    if guard:
        return guard

    r = run_script("src/tools/style_builder_v1.py")
    persist_result = persist_file("style-metrics") if r["ok"] else {"status": "skipped", "reason": "script_failed"}

    return jsonify({
        **r,
        "status": "ok" if r["ok"] else "error",
        "github_persist": persist_result,
        "timestamp": datetime.utcnow().isoformat()
    })

# ---------------- DOWNLOAD endpoints ----------------

@app.route("/download/thursday-report-v3", methods=["GET"])
def download_thursday():
    guard = require_admin()
    if guard:
        return guard

    p = abs_path("logs/thursday_report_v3.json")
    if not os.path.exists(p):
        restore_file("thursday-v3")

    if not os.path.exists(p):
        return jsonify({"status": "error", "message": "missing", "path": p, "logs": list_logs_dir()}), 404

    return send_file(p, mimetype="application/json", as_attachment=True)

@app.route("/download/friday-shortlist-v3", methods=["GET"])
def download_friday():
    guard = require_admin()
    if guard:
        return guard

    p = abs_path("logs/friday_shortlist_v3.json")
    if not os.path.exists(p):
        restore_file("friday-v3")

    if not os.path.exists(p):
        return jsonify({"status": "error", "message": "missing", "path": p, "logs": list_logs_dir()}), 404

    return send_file(p, mimetype="application/json", as_attachment=True)

@app.route("/download/friday-v10-1", methods=["GET"])
def download_friday_v10_1():
    guard = require_admin()
    if guard:
        return guard

    p = abs_path("logs/friday_shortlist_v10_1.json")
    if not os.path.exists(p):
        restore_file("friday-v10-1")

    if not os.path.exists(p):
        return jsonify({"status": "error", "message": "missing", "path": p, "logs": list_logs_dir()}), 404

    return send_file(p, mimetype="application/json", as_attachment=True)

@app.route("/download/friday-shortlist-v10-1", methods=["GET"])
def download_friday_shortlist_v10_1_alias():
    return download_friday_v10_1()

@app.route("/download/tuesday-recap-v3", methods=["GET"])
def download_tuesday_recap():
    guard = require_admin()
    if guard:
        return guard

    p = abs_path("logs/tuesday_recap_v3.json")
    if not os.path.exists(p):
        restore_file("tuesday-recap-v3")

    if not os.path.exists(p):
        return jsonify({"status": "error", "message": "missing", "path": p, "logs": list_logs_dir()}), 404

    return send_file(p, mimetype="application/json", as_attachment=True)

@app.route("/download/tuesday-history-v3", methods=["GET"])
def download_tuesday_history():
    guard = require_admin()
    if guard:
        return guard

    p = abs_path("logs/tuesday_history_v3.json")
    if not os.path.exists(p):
        restore_file("tuesday-history-v3")

    if not os.path.exists(p):
        return jsonify({"status": "error", "message": "missing", "path": p, "logs": list_logs_dir()}), 404

    return send_file(p, mimetype="application/json", as_attachment=True)

@app.route("/download/style-metrics", methods=["GET"])
def download_style():
    guard = require_admin()
    if guard:
        return guard

    p = abs_path("data/team_style_metrics.json")
    if not os.path.exists(p):
        restore_file("style-metrics")

    if not os.path.exists(p):
        return jsonify({"status": "error", "message": "missing", "path": p, "data": list_data_dir()}), 404

    return send_file(p, mimetype="application/json", as_attachment=True)

# ---------------- GPT read endpoints ----------------

@app.route("/thursday-analysis-v3", methods=["GET"])
def gpt_thursday():
    report, error, restore_meta = load_json_report_with_restore(
        "logs/thursday_report_v3.json",
        "thursday-v3"
    )

    if report is None:
        return jsonify({
            "status": "error",
            "message": "Thursday report not available",
            "error": error,
            "restore": restore_meta,
            "timestamp": datetime.utcnow().isoformat(),
            "report": None
        }), 404

    cursor = request.args.get("cursor", "0")
    per_page = request.args.get("per_page", "2")
    lite = request.args.get("lite", "true").lower() == "true"
    tz = (request.args.get("tz", "Europe/Athens") or "Europe/Athens").strip()

    try:
        chunked = _chunk_thursday_report(report, int(cursor), int(per_page), bool(lite), tz)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Thursday report chunking failed",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
            "report": None
        }), 500

    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "restore": restore_meta,
        "report": chunked
    })

@app.route("/friday-shortlist-v3", methods=["GET"])
def gpt_friday():
    report, error, restore_meta = load_json_report_with_restore(
        "logs/friday_shortlist_v3.json",
        "friday-v3"
    )

    if report is None:
        return jsonify({
            "status": "error",
            "message": "Friday shortlist v3 not available",
            "error": error,
            "restore": restore_meta,
            "timestamp": datetime.utcnow().isoformat(),
            "report": None
        }), 404

    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "restore": restore_meta,
        "report": report
    })

@app.route("/friday-v10-1", methods=["GET"])
def gpt_friday_v10_1():
    report, error, restore_meta = load_json_report_with_restore(
        "logs/friday_shortlist_v10_1.json",
        "friday-v10-1"
    )

    if report is None:
        return jsonify({
            "status": "error",
            "message": "Friday shortlist V10.1 not available",
            "error": error,
            "restore": restore_meta,
            "timestamp": datetime.utcnow().isoformat(),
            "report": None
        }), 404

    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "restore": restore_meta,
        "resolved_file": "logs/friday_shortlist_v10_1.json",
        "report": report
    })

@app.route("/friday-shortlist-v10-1", methods=["GET"])
def gpt_friday_shortlist_v10_1_alias():
    return gpt_friday_v10_1()

@app.route("/tuesday-recap", methods=["GET"])
def gpt_tuesday():
    report, error, restore_meta = load_json_report_with_restore(
        "logs/tuesday_recap_v3.json",
        "tuesday-recap-v3"
    )

    if report is None:
        return jsonify({
            "status": "error",
            "message": "Tuesday recap v3 not available",
            "error": error,
            "restore": restore_meta,
            "timestamp": datetime.utcnow().isoformat(),
            "report": None
        }), 404

    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "restore": restore_meta,
        "report": report
    })

@app.route("/tuesday-recap-v3", methods=["GET"])
def gpt_tuesday_v3():
    return gpt_tuesday()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting Bombay Engine Flask Server on port {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, use_reloader=False)
