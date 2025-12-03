import os
import json
import subprocess
from datetime import datetime

from flask import Flask, jsonify

app = Flask(__name__)


# ======================================================
# Βοηθητικό: τρέχει script και γυρίζει stdout / stderr
# ======================================================
def run_script(script_name: str):
    try:
        print(f"🚀 Running script: {script_name}", flush=True)

        result = subprocess.run(
            ["python3", script_name],
            cwd="/opt/render/project/src",
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
            }
        )

    except Exception as e:
        print(f"❌ Error running {script_name}: {e}", flush=True)
        return jsonify({"status": "error", "script": script_name, "error": str(e)}), 500


# ======================================================
# Βοηθητικό: τρέχει script ΚΑΙ φορτώνει JSON report
# (για χρήση από τον Agent Bombay μέσω OpenAPI)
# ======================================================
def run_script_with_report(script_name: str, report_path: str):
    try:
        print(f"🚀 Running script with report: {script_name}", flush=True)

        result = subprocess.run(
            ["python3", script_name],
            cwd="/opt/render/project/src",
            capture_output=True,
            text=True,
        )

        print("----- SCRIPT OUTPUT START -----", flush=True)
        print(result.stdout, flush=True)
        print("----- SCRIPT OUTPUT END -----", flush=True)

        if result.stderr:
            print("⚠️ SCRIPT ERRORS:", flush=True)
            print(result.stderr, flush=True)

        report_data = None
        if os.path.exists(report_path):
            try:
                with open(report_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load report file {report_path}: {e}", flush=True)
        else:
            print(f"⚠️ Report file not found: {report_path}", flush=True)

        return jsonify(
            {
                "status": "ok" if result.returncode == 0 else "error",
                "script": script_name,
                "timestamp": datetime.utcnow().isoformat(),
                "return_code": result.returncode,
                "stderr": result.stderr,
                "stdout": result.stdout,
                "report": report_data,
            }
        )

    except Exception as e:
        print(f"❌ Error running {script_name} with report: {e}", flush=True)
        return (
            jsonify(
                {
                    "status": "error",
                    "script": script_name,
                    "error": str(e),
                }
            ),
            500,
        )


# ======================================================
#  ΠΑΛΙΑ MANUAL ENDPOINTS (για browser tests)
# ======================================================
@app.route("/run/thursday", methods=["GET"])
def run_thursday():
    # Πλέον τρέχουμε το νέο full engine script (v3)
    return run_script("analysis/thursday_engine_full_v3.py")


@app.route("/run/friday", methods=["GET"])
def run_friday():
    return run_script("friday_shortlist_v2.py")


@app.route("/run/tuesday", methods=["GET"])
def run_tuesday():
    return run_script("tuesday_recap_v2.py")


# ======================================================
#  ΝΕΑ ENDPOINTS ΓΙΑ ΤΟΝ AGENT (OpenAPI)
#  Εδώ γυρίζουμε και το JSON report
# ======================================================
@app.route("/thursday-analysis", methods=["GET"])
def api_thursday_analysis():
    # Το report παραμένει στο ίδιο path, το γράφει πλέον το v3 engine
    return run_script_with_report(
        "analysis/thursday_engine_full_v3.py", "logs/thursday_report_v1.json"
    )


@app.route("/friday-shortlist", methods=["GET"])
def api_friday_shortlist():
    return run_script_with_report(
        "friday_shortlist_v2.py", "logs/friday_shortlist_v2.json"
    )


@app.route("/tuesday-recap", methods=["GET"])
def api_tuesday_recap():
    return run_script_with_report(
        "tuesday_recap_v2.py", "logs/tuesday_recap_v2.json"
    )


# ======================================================
#  Healthcheck
# ======================================================
@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "Bombay Engine alive"})


# ======================================================
#  Entry point
# ======================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🟢 Starting Bombay Engine Flask Server on port {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, use_reloader=False)
