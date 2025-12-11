import os
import json
import subprocess
from datetime import datetime

from flask import Flask, jsonify, send_file

app = Flask(__name__)

# ------------------------------------------------------
# Ριζικός φάκελος (εκεί που βρίσκεται το app.py)
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------
# Βοηθητικό: τρέξιμο script (χειροκίνητο, όχι GPT)
# ------------------------------------------------------
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

# ------------------------------------------------------
# Βοηθητικό: φόρτωση JSON report από logs/
# ------------------------------------------------------
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

# ------------------------------------------------------
# HEALTHCHECK
# ------------------------------------------------------
@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "Bombay Engine alive"})

# ------------------------------------------------------
# MANUAL RUN ENDPOINTS
# ------------------------------------------------------
@app.route("/run/thursday-v3", methods=["GET"])
def manual_run_thursday_v3():
    """
    Τρέχει ΜΟΝΟ χειροκίνητα από browser.
    Γράφει το logs/thursday_report_v3.json.
    """
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
    """
    Τρέχει το Friday shortlist v3 script.
    Γράφει logs/friday_shortlist_v3.json.
    """
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

# ------------------------------------------------------
# DOWNLOAD ENDPOINTS (για manual upload στο Custom GPT)
# ------------------------------------------------------
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

@app.route("/download/tuesday-recap-v2", methods=["GET"])
def download_tuesday_recap_v2():
    full_path = os.path.join(BASE_DIR, "logs", "tuesday_recap_v2.json")

    if not os.path.exists(full_path):
        return jsonify(
            {
                "status": "error",
                "message": "Tuesday recap file not found",
                "path": full_path,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    return send_file(full_path, mimetype="application/json", as_attachment=True)

# ------------------------------------------------------
# GPT ENDPOINTS (Thursday / Friday / Tuesday)
# ------------------------------------------------------
@app.route("/thursday-analysis-v3", methods=["GET"])
def api_thursday_analysis_v3():
    """
    Το GPT παίρνει μια "light" έκδοση του Thursday report,
    βασισμένη στο logs/thursday_report_v3.json.

    ΠΡΙΝ το διαβάσει, κάνει auto-run τον Thursday engine.
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

    fixtures = full_report.get("fixtures", [])
    light_fixtures = []

    for fx in fixtures:
        draw_prob = fx.get("draw_prob")
        over_prob = fx.get("over_2_5_prob")

        # Υπολογισμός SCORE DRAW (σύμφωνα με το spec)
        if isinstance(draw_prob, (int, float)):
            score_draw_raw = draw_prob * 10
            score_draw = round(score_draw_raw, 1)
            if score_draw < 1:
                score_draw = 1
            if score_draw > 10:
                score_draw = 10
        else:
            score_draw = None

        # Υπολογισμός SCORE OVER (σύμφωνα με το spec)
        if isinstance(over_prob, (int, float)):
            score_over_raw = over_prob * 10
            score_over = round(score_over_raw, 1)
            if score_over < 1:
                score_over = 1
            if score_over > 10:
                score_over = 10
        else:
            score_over = None

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

                # FAIR odds
                "fair_1": fx.get("fair_1"),
                "fair_x": fx.get("fair_x"),
                "fair_2": fx.get("fair_2"),
                "fair_over_2_5": fx.get("fair_over_2_5"),
                "fair_under_2_5": fx.get("fair_under_2_5"),

                # Probabilities
                "draw_prob": draw_prob,
                "over_2_5_prob": over_prob,
                "under_2_5_prob": fx.get("under_2_5_prob"),

                # Offered odds (για να μην τα υπολογίζει/μαντεύει το GPT)
                "offered_1": fx.get("offered_1"),
                "offered_x": fx.get("offered_x"),
                "offered_2": fx.get("offered_2"),
                "offered_over_2_5": fx.get("offered_over_2_5"),
                "offered_under_2_5": fx.get("offered_under_2_5"),

                # Scores ήδη υπολογισμένα από backend
                "score_draw": score_draw,
                "score_over": score_over,
            }
        )

    light_report = {
        "generated_at": full_report.get("generated_at"),
        "window": full_report.get("window", {}),
        "fixtures_analyzed": len(light_fixtures),
        "fixtures": light_fixtures,
    }

    return jsonify(
        {
            "status": "ok",
            "script": "src/analysis/thursday_engine_full_v3.py",
            "timestamp": datetime.utcnow().isoformat(),
            "report": light_report,
        }
    )

@app.route("/friday-shortlist-v3", methods=["GET"])
def api_friday_shortlist_v3():
    """
    Το GPT ζητάει το Friday shortlist.
    ΠΡΙΝ το σερβίρουμε, τρέχουμε (auto-run) το Friday engine,
    ώστε να υπάρχει ΠΑΝΤΑ φρέσκο JSON.
    """
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

    return jsonify(
        {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "report": report,
        }
    )

@app.route("/tuesday-recap", methods=["GET"])
def api_tuesday_recap():
    report, error = load_json_report("logs/tuesday_recap_v2.json")

    if report is None:
        return jsonify(
            {
                "status": "error",
                "message": "Tuesday recap not available",
                "error": error,
                "report": None,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    return jsonify(
        {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "report": report,
        }
    )

# ------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(
        f"🚀 Starting Bombay Engine Flask Server on port {port}...",
        flush=True,
    )
    app.run(host="0.0.0.0", port=port, use_reloader=False)
