import os
import json
import subprocess
from datetime import datetime

from flask import Flask, jsonify, send_file

app = Flask(__name__)

# ------------------------------------------------------
# Project root (εκεί που είναι το app.py)
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------
# Helper: run script (cwd=BASE_DIR)
# ------------------------------------------------------
def run_script(script_rel_path: str):
    """
    script_rel_path: π.χ. 'src/analysis/thursday_engine_full_v3.py'
    Τρέχει με cwd=BASE_DIR ώστε όλα τα relative paths (logs/...) να γράφουν σωστά.
    """
    try:
        print(f"▶️ Running script: {script_rel_path}", flush=True)

        result = subprocess.run(
            ["python3", script_rel_path],
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
# Helper: load JSON report from logs/
# ------------------------------------------------------
def load_json_report(rel_path: str):
    full_path = os.path.join(BASE_DIR, rel_path)

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
    return jsonify({"status": "ok", "message": "Bombay Engine alive", "timestamp": datetime.utcnow().isoformat()})

# ------------------------------------------------------
# MANUAL RUN ENDPOINTS
# ------------------------------------------------------
@app.route("/run/thursday-v3", methods=["GET"])
def manual_run_thursday_v3():
    r = run_script("src/analysis/thursday_engine_full_v3.py")
    return jsonify({**r, "status": "ok" if r["ok"] else "error", "timestamp": datetime.utcnow().isoformat()})

@app.route("/run/friday-shortlist-v3", methods=["GET"])
def manual_run_friday_shortlist_v3():
    r = run_script("src/analysis/friday_shortlist_v3.py")
    return jsonify({**r, "status": "ok" if r["ok"] else "error", "timestamp": datetime.utcnow().isoformat()})

# ------------------------------------------------------
# DOWNLOAD ENDPOINTS
# ------------------------------------------------------
@app.route("/download/thursday-report-v3", methods=["GET"])
def download_thursday_report_v3():
    full_path = os.path.join(BASE_DIR, "logs", "thursday_report_v3.json")
    if not os.path.exists(full_path):
        return jsonify({"status": "error", "message": "Thursday report file not found", "path": full_path, "timestamp": datetime.utcnow().isoformat()}), 404
    return send_file(full_path, mimetype="application/json", as_attachment=True)

@app.route("/download/friday-shortlist-v3", methods=["GET"])
def download_friday_shortlist_v3():
    full_path = os.path.join(BASE_DIR, "logs", "friday_shortlist_v3.json")
    if not os.path.exists(full_path):
        return jsonify({"status": "error", "message": "Friday shortlist v3 file not found", "path": full_path, "timestamp": datetime.utcnow().isoformat()}), 404
    return send_file(full_path, mimetype="application/json", as_attachment=True)

# ✅ Αν όντως έχεις ΜΟΝΟ v2, το κρατάμε ξεκάθαρα v2.
@app.route("/download/tuesday-recap-v2", methods=["GET"])
def download_tuesday_recap_v2():
    full_path = os.path.join(BASE_DIR, "logs", "tuesday_recap_v2.json")
    if not os.path.exists(full_path):
        return jsonify({"status": "error", "message": "Tuesday recap v2 file not found", "path": full_path, "timestamp": datetime.utcnow().isoformat()}), 404
    return send_file(full_path, mimetype="application/json", as_attachment=True)

# ------------------------------------------------------
# GPT ENDPOINTS (Thursday / Friday / Tuesday)
# ------------------------------------------------------
@app.route("/thursday-analysis-v3", methods=["GET"])
def api_thursday_analysis_v3():
    """
    Auto-run Thursday engine, μετά σερβίρει light report.
    """
    r = run_script("src/analysis/thursday_engine_full_v3.py")
    if not r["ok"]:
        return jsonify({"status": "error", "message": "Thursday engine failed", "run": r, "timestamp": datetime.utcnow().isoformat(), "report": None}), 500

    full_report, error = load_json_report("logs/thursday_report_v3.json")
    if full_report is None:
        return jsonify({"status": "error", "message": "Thursday report not available", "error": error, "run": r, "timestamp": datetime.utcnow().isoformat(), "report": None}), 500

    fixtures = full_report.get("fixtures", [])
    light_fixtures = []

    for fx in fixtures:
        draw_prob = fx.get("draw_prob")
        over_prob = fx.get("over_2_5_prob")

        # αυτά είναι scaled probabilities (όχι “model score”)
        def scaled_prob_score(p):
            if isinstance(p, (int, float)):
                s = round(p * 10, 1)
                return max(1, min(10, s))
            return None

        light_fixtures.append({
            "fixture_id": fx.get("fixture_id"),
            "date": fx.get("date"),
            "time": fx.get("time"),
            "league_id": fx.get("league_id"),
            "league": fx.get("league"),
            "home": fx.get("home"),
            "away": fx.get("away"),
            "model": fx.get("model"),

            "fair_1": fx.get("fair_1"),
            "fair_x": fx.get("fair_x"),
            "fair_2": fx.get("fair_2"),
            "fair_over_2_5": fx.get("fair_over_2_5"),
            "fair_under_2_5": fx.get("fair_under_2_5"),

            "home_prob": fx.get("home_prob"),
            "draw_prob": draw_prob,
            "away_prob": fx.get("away_prob"),
            "over_2_5_prob": over_prob,
            "under_2_5_prob": fx.get("under_2_5_prob"),

            "offered_1": fx.get("offered_1"),
            "offered_x": fx.get("offered_x"),
            "offered_2": fx.get("offered_2"),
            "offered_over_2_5": fx.get("offered_over_2_5"),
            "offered_under_2_5": fx.get("offered_under_2_5"),

            "value_pct_1": fx.get("value_pct_1"),
            "value_pct_x": fx.get("value_pct_x"),
            "value_pct_2": fx.get("value_pct_2"),
            "value_pct_over": fx.get("value_pct_over"),
            "value_pct_under": fx.get("value_pct_under"),

            "prob_score_draw": scaled_prob_score(draw_prob),
            "prob_score_over": scaled_prob_score(over_prob),

            # important για Friday gate
            "odds_match": fx.get("odds_match"),
        })

    light_report = {
        "generated_at": full_report.get("generated_at"),
        "window": full_report.get("window", {}),
        "fixtures_analyzed": len(light_fixtures),
        "fixtures": light_fixtures,
    }

    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "run": r,
        "report": light_report,
    })

@app.route("/friday-shortlist-v3", methods=["GET"])
def api_friday_shortlist_v3():
    """
    Auto-run Friday shortlist και μετά σερβίρει το json.
    """
    r = run_script("src/analysis/friday_shortlist_v3.py")
    if not r["ok"]:
        return jsonify({"status": "error", "message": "Friday shortlist failed", "run": r, "timestamp": datetime.utcnow().isoformat(), "report": None}), 500

    report, error = load_json_report("logs/friday_shortlist_v3.json")
    if report is None:
        return jsonify({"status": "error", "message": "Friday shortlist v3 not available", "error": error, "run": r, "timestamp": datetime.utcnow().isoformat(), "report": None}), 500

    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat(), "run": r, "report": report})

@app.route("/tuesday-recap", methods=["GET"])
def api_tuesday_recap():
    """
    ✅ ΞΕΚΑΘΑΡΟ: εδώ σερβίρουμε v2 (όπως είναι το file).
    Αν θες v3, φτιάχνεις logs/tuesday_recap_v3.json και ανοίγουμε νέο endpoint.
    """
    report, error = load_json_report("logs/tuesday_recap_v2.json")
    if report is None:
        return jsonify({"status": "error", "message": "Tuesday recap v2 not available", "error": error, "timestamp": datetime.utcnow().isoformat(), "report": None}), 500

    return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat(), "report": report})

# ------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting Bombay Engine Flask Server on port {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, use_reloader=False)
