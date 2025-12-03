import os
import json
import subprocess
from datetime import datetime

from flask import Flask, jsonify

app = Flask(__name__)


# ======================================================
# Helper: run script and return stdout / stderr
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
# Helper: run script AND load JSON report
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
#  ΠΑΛΙΑ MANUAL ENDPOINTS (v1) – τα κρατάμε για backup
# ======================================================
@app.route("/run/thursday", methods=["GET"])
def run_thursday_v1():
    return run_script("thursday_analysis_v1.py")


@app.route("/thursday-analysis", methods=["GET"])
def api_thursday_analysis_v1():
    return run_script_with_report(
        "thursday_analysis_v1.py", "logs/thursday_report_v1.json"
    )


@app.route("/run/friday", methods=["GET"])
def run_friday_v2():
    return run_script("friday_shortlist_v2.py")


@app.route("/friday-shortlist", methods=["GET"])
def api_friday_shortlist_v2():
    return run_script_with_report(
        "friday_shortlist_v2.py", "logs/friday_shortlist_v2.json"
    )


@app.route("/run/tuesday", methods=["GET"])
def run_tuesday_v2():
    return run_script("tuesday_recap_v2.py")


@app.route("/tuesday-recap", methods=["GET"])
def api_tuesday_recap_v2():
    return run_script_with_report(
        "tuesday_recap_v2.py", "logs/tuesday_recap_v2.json"
    )


# ======================================================
#  ΝΕΑ ENDPOINTS ΓΙΑ ΤΟΝ FULL ENGINE v3
# ======================================================
@app.route("/run/thursday-v3", methods=["GET"])
def run_thursday_v3():
    # νέο full engine – βρίσκεται στο src/analysis
    return run_script("src/analysis/thursday_engine_full_v3.py")


@app.route("/thursday-analysis-v3", methods=["GET"])
def api_thursday_analysis_v3():
    return run_script_with_report(
        "src/analysis/thursday_engine_full_v3.py",
        "logs/thursday_report_v3.json",
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
