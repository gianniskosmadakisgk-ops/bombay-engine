import os
import json
import subprocess
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)

# ---------------------------------------
#  Ριζικός φάκελος στο Render
# ---------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------
#  Βοηθητικό: τρέχει script και δείχνει log
# ---------------------------------------
def run_script(script_name: str):
    try:
        print(f"🚀 Running script: {script_name}", flush=True)

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

        return jsonify(
            {
                "status": "ok" if result.returncode == 0 else "error",
                "script": script_name,
                "return_code": result.returncode,
                "stderr": result.stderr,
                "stdout": result.stdout,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

    except Exception as e:
        print(f"❌ Error running {script_name}: {e}", flush=True)
        return jsonify({"status": "error", "script": script_name, "error": str(e)}), 500


# ---------------------------------------
#  Βοηθητικό: διαβάζει report JSON
# ---------------------------------------
def load_json_report(relative_path: str):
    """Διαβάζει JSON report από logs/*.json χωρίς να τρέχει engine."""
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


# ---------------------------------------
#  HEALTHCHECK
# ---------------------------------------
@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "Bombay Engine alive"})


# ---------------------------------------
#  MANUAL RUN ENDPOINT (ΜΟΝΟ ΓΙΑ ΣΕΝΑ)
# ---------------------------------------
@app.route("/run/thursday-v3", methods=["GET"])
def manual_run_thursday_v3():
    """
    Τρέχει το full Thursday engine και δημιουργεί logs/thursday_report_v3.json.
    """
    return run_script("src/analysis/thursday_engine_full_v3.py")


# ---------------------------------------
#  GPT ENDPOINT — Thursday Analysis
# ---------------------------------------
@app.route("/thursday-analysis-v3", methods=["GET"])
def api_thursday_analysis_v3():
    """
    Το endpoint που καλεί ο GPT μέσω runThursdayAnalysis.
    Δεν τρέχει engine. Διαβάζει ΜΟΝΟ το logs/thursday_report_v3.json.
    """
    report, error = load_json_report("logs/thursday_report_v3.json")

    if report is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Thursday report not available",
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            503,
        )

    return jsonify(
        {
            "status": "ok",
            "script": "src/analysis/thursday_engine_full_v3.py",
            "timestamp": datetime.utcnow().isoformat(),
            "report": report,
        }
    )


# ---------------------------------------
#  GPT ENDPOINT — Friday Shortlist
# ---------------------------------------
@app.route("/friday-shortlist-v3", methods=["GET"])
def api_friday_shortlist_v3():
    report, error = load_json_report("logs/friday_shortlist_v3.json")

    if report is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Friday shortlist not available",
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            503,
        )

    return jsonify(
        {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "report": report,
        }
    )


# ---------------------------------------
#  GPT ENDPOINT — Tuesday Recap
# ---------------------------------------
@app.route("/tuesday-recap", methods=["GET"])
def api_tuesday_recap():
    report, error = load_json_report("logs/tuesday_recap_v2.json")

    if report is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Tuesday recap not available",
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            503,
        )

    return jsonify(
        {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "report": report,
        }
    )


# ---------------------------------------
#  Entry point
# ---------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🟢 Starting Bombay Engine Flask Server on port {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, use_reloader=False)
