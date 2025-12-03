import os
import json
import subprocess
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)

# ======================================================
#  Βοηθητικό: τρέχει script και γυρίζει stdout / stderr
# ======================================================
def run_script(script_name: str):
    """
    Τρέχει ένα Python script μέσα στο /opt/render/project/src
    και γυρίζει μόνο stdout / stderr (για manual debug).
    """
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
#  Helper για scripts με JSON report (για GPT)
# ======================================================
def run_script_with_report(script_name: str, report_path: str):
    """
    Τρέχει ένα script και μετά προσπαθεί να φορτώσει JSON report
    από το report_path (relative στο /opt/render/project/src).
    """
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
        report_full_path = os.path.join("/opt/render/project/src", report_path)

        if os.path.exists(report_full_path):
            try:
                with open(report_full_path, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load report file {report_full_path}: {e}", flush=True)
        else:
            print(f"⚠️ Report file not found: {report_full_path}", flush=True)

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
#  MANUAL ENDPOINTS (browser tests)
# ======================================================

@app.route("/run/thursday-v3", methods=["GET"])
def run_thursday_v3():
    # Full Thursday engine (v3) με όλα τα μοντέλα
    return run_script("src/analysis/thursday_engine_full_v3.py")


@app.route("/run/friday-v3", methods=["GET"])
def run_friday_v3():
    # Friday shortlist (v2 script)
    return run_script("friday_shortlist_v2.py")


@app.route("/run/tuesday-v3", methods=["GET"])
def run_tuesday_v3():
    # Tuesday recap (v2 script)
    return run_script("tuesday_recap_v2.py")


# ======================================================
#  API ENDPOINTS ΓΙΑ GPT (OpenAPI)
# ======================================================

@app.route("/thursday-analysis-v3", methods=["GET"])
def api_thursday_analysis_v3():
    """
    Χρησιμοποιείται από το OpenAPI path /thursday-analysis-v3
    και γυρίζει:
      - status, script, timestamp, stdout/stderr
      - report: το περιεχόμενο του logs/thursday_report_v3.json
    """
    return run_script_with_report(
        "src/analysis/thursday_engine_full_v3.py",
        "logs/thursday_report_v3.json",
    )


@app.route("/thursday-analysis", methods=["GET"])
def api_thursday_analysis_alias():
    """
    Alias για συμβατότητα – ίδιο αποτέλεσμα με /thursday-analysis-v3
    """
    return api_thursday_analysis_v3()


@app.route("/friday-shortlist", methods=["GET"])
def api_friday_shortlist():
    """
    Χρησιμοποιείται από το OpenAPI path /friday-shortlist
    Διαβάζει το τελευταίο Thursday report και βγάζει shortlist
    σε logs/friday_shortlist_v2.json
    """
    return run_script_with_report(
        "friday_shortlist_v2.py",
        "logs/friday_shortlist_v2.json",
    )


@app.route("/tuesday-recap", methods=["GET"])
def api_tuesday_recap():
    """
    Χρησιμοποιείται από το OpenAPI path /tuesday-recap
    Γυρίζει weekly recap σε logs/tuesday_recap_v2.json
    """
    return run_script_with_report(
        "tuesday_recap_v2.py",
        "logs/tuesday_recap_v2.json",
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
