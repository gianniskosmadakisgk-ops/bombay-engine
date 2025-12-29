import os
import json
import subprocess
from datetime import datetime

from flask import Flask, jsonify, send_file

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name: str):
    try:
        print(f"▶️ Running script: {script_name}", flush=True)

        result = subprocess.run(
            ["python3", script_name],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
        )

        print("----- SCRIPT OUTPUT START -----", flush=True)
        print(result.stdout, flush=True)
        print("----- SCRIPT OUTPUT END -----", flush=True)

        if result.stderr:
            print("⚠️ SCRIPT ERRORS:", flush=True)
            print(result.stderr, flush=True)

        return {
            "ok": (result.returncode == 0),
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}", flush=True)
        return {
            "ok": False,
            "return_code": -1,
            "stdout": "",
            "stderr": str(e),
        }

def load_json_report(relative_path: str):
    full_path = os.path.join(BASE_DIR, relative_path)

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

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "Bombay Engine alive"})

# ------------------- MANUAL RUN ENDPOINTS -------------------
@app.route("/run/thursday-v3", methods=["GET"])
def manual_run_thursday_v3():
    r = run_script("src/analysis/thursday_engine_full_v3.py")
    return jsonify(
        {
            "status": "ok" if r["ok"] else "error",
            "script": "src/analysis/thursday_engine_full_v3.py",
            "return_code": r["return_code"],
            "stdout": r["stdout"],
            "stderr": r["stderr"],
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

@app.route("/run/friday-shortlist-v3", methods=["GET"])
def manual_run_friday_shortlist_v3():
    r = run_script("src/analysis/friday_shortlist_v3.py")
    return jsonify(
        {
            "status": "ok" if r["ok"] else "error",
            "script": "src/analysis/friday_shortlist_v3.py",
            "return_code": r["return_code"],
            "stdout": r["stdout"],
            "stderr": r["stderr"],
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

@app.route("/run/tuesday-recap-v3", methods=["GET"])
def manual_run_tuesday_recap_v3():
    r = run_script("src/analysis/tuesday_recap_v3.py")
    return jsonify(
        {
            "status": "ok" if r["ok"] else "error",
            "script": "src/analysis/tuesday_recap_v3.py",
            "return_code": r["return_code"],
            "stdout": r["stdout"],
            "stderr": r["stderr"],
            "timestamp": datetime.utcnow().isoformat(),
        }
    )

# ------------------- DOWNLOAD ENDPOINTS -------------------
@app.route("/download/thursday-report-v3", methods=["GET"])
def download_thursday_report_v3():
    full_path = os.path.join(BASE_DIR, "logs", "thursday_report_v3.json")
    if not os.path.exists(full_path):
        return jsonify(
            {
                "status": "error",
                "message": "Thursday report file not found",
                "path": full_path,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    return send_file(full_path, mimetype="application/json", as_attachment=True)

@app.route("/download/friday-shortlist-v3", methods=["GET"])
def download_friday_shortlist_v3():
    full_path = os.path.join(BASE_DIR, "logs", "friday_shortlist_v3.json")
    if not os.path.exists(full_path):
        return jsonify(
            {
                "status": "error",
                "message": "Friday shortlist v3 file not found",
                "path": full_path,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    return send_file(full_path, mimetype="application/json", as_attachment=True)

@app.route("/download/tuesday-recap-v3", methods=["GET"])
def download_tuesday_recap_v3():
    full_path = os.path.join(BASE_DIR, "logs", "tuesday_recap_v3.json")
    if not os.path.exists(full_path):
        return jsonify(
            {
                "status": "error",
                "message": "Tuesday recap v3 file not found",
                "path": full_path,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    return send_file(full_path, mimetype="application/json", as_attachment=True)

# ------------------- GPT ENDPOINTS -------------------
@app.route("/thursday-analysis-v3", methods=["GET"])
def api_thursday_analysis_v3():
    """
    Auto-run Thursday engine then serve a LIGHT report to GPT.
    IMPORTANT: scores are taken FROM JSON (backend), no recalculation here.
    """
    try:
        run_script("src/analysis/thursday_engine_full_v3.py")
    except Exception as e:
        print(f"⚠️ Error while auto-running Thursday engine: {e}", flush=True)

    full_report, error = load_json_report("logs/thursday_report_v3.json")
    if full_report is None:
        return jsonify(
            {
                "status": "error",
                "message": "Thursday report not available",
                "error": error,
                "report": None,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    fixtures = full_report.get("fixtures", []) or []
    light_fixtures = []
    for fx in fixtures:
        light_fixtures.append(
            {
                "fixture_id": fx.get("fixture_id"),
                "date": fx.get("date"),
                "time": fx.get("time"),
                "league_id": fx.get("league_id"),
                "league": fx.get("league"),
                "home": fx.get("home"),
                "away": fx.get("away"),
                "model": fx.get("model"),

                # tempo info (optional in presenter)
                "lambda_total": fx.get("lambda_total"),
                "tempo": fx.get("tempo"),

                # FAIR odds
                "fair_1": fx.get("fair_1"),
                "fair_x": fx.get("fair_x"),
                "fair_2": fx.get("fair_2"),
                "fair_over_2_5": fx.get("fair_over_2_5"),
                "fair_under_2_5": fx.get("fair_under_2_5"),

                # Offered odds
                "offered_1": fx.get("offered_1"),
                "offered_x": fx.get("offered_x"),
                "offered_2": fx.get("offered_2"),
                "offered_over_2_5": fx.get("offered_over_2_5"),
                "offered_under_2_5": fx.get("offered_under_2_5"),

                # Value % (already computed by engine; can be None)
                "value_pct_1": fx.get("value_pct_1"),
                "value_pct_x": fx.get("value_pct_x"),
                "value_pct_2": fx.get("value_pct_2"),
                "value_pct_over": fx.get("value_pct_over"),
                "value_pct_under": fx.get("value_pct_under"),

                # Scores from backend
                "score_draw": fx.get("score_draw"),
                "score_over": fx.get("score_over"),
            }
        )

    light_report = {
        "generated_at": full_report.get("generated_at"),
        "window": full_report.get("window", {}),
        "fixtures_analyzed": len(light_fixtures),
        "leagues_selected": full_report.get("leagues_selected"),
        "fixtures": light_fixtures,
    }

    return jsonify(
        {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "report": light_report,
        }
    )

@app.route("/friday-shortlist-v3", methods=["GET"])
def api_friday_shortlist_v3():
    try:
        run_script("src/analysis/friday_shortlist_v3.py")
    except Exception as e:
        print(f"⚠️ Error while auto-running Friday shortlist: {e}", flush=True)

    report, error = load_json_report("logs/friday_shortlist_v3.json")
    if report is None:
        return jsonify(
            {
                "status": "error",
                "message": "Friday shortlist v3 not available",
                "error": error,
                "report": None,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat(), "report": report})

@app.route("/tuesday-recap", methods=["GET"])
def api_tuesday_recap():
    report, error = load_json_report("logs/tuesday_recap_v3.json")
    if report is None:
        return jsonify(
            {
                "status": "error",
                "message": "Tuesday recap v3 not available",
                "error": error,
                "report": None,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat(), "report": report})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting Bombay Engine Flask Server on port {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, use_reloader=False)
