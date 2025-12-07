import os
import json
import subprocess
from datetime import datetime
from flask import Flask, jsonify, send_file  # + send_file

app = Flask(__name__)

# ------------------------------------------------------
#  Ριζικός φάκελος (εκεί που βρίσκεται το app.py)
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------
#  Βοηθητικό: τρέξιμο script (χειροκίνητο, όχι GPT)
# ------------------------------------------------------
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
        return jsonify(
            {
                "status": "error",
                "script": script_name,
                "error": str(e),
            }
        ), 500


# ------------------------------------------------------
#  Βοηθητικό: φόρτωση JSON report από logs/
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
#  HEALTHCHECK
# ------------------------------------------------------
@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "Bombay Engine alive"})


# ------------------------------------------------------
#  MANUAL RUN — Thursday Engine v3
# ------------------------------------------------------
@app.route("/run/thursday-v3", methods=["GET"])
def manual_run_thursday_v3():
    """
    Τρέχει ΜΟΝΟ χειροκίνητα από browser.
    Γράφει το logs/thursday_report_v3.json.
    """
    return run_script("src/analysis/thursday_engine_full_v3.py")


# ------------------------------------------------------
#  DOWNLOAD ENDPOINTS (για manual upload στο Custom GPT)
# ------------------------------------------------------
@app.route("/download/thursday-report-v3", methods=["GET"])
def download_thursday_report_v3():
    """
    Κατεβάζει το τελευταίο Thursday report σαν JSON αρχείο.
    """
    full_path = os.path.join(BASE_DIR, "logs", "thursday_report_v3.json")

    if not os.path.exists(full_path):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Thursday report file not found",
                    "path": full_path,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            404,
        )

    return send_file(
        full_path,
        mimetype="application/json",
        as_attachment=True,
    )


@app.route("/download/friday-shortlist-v3", methods=["GET"])
def download_friday_shortlist_v3():
    """
    Κατεβάζει το τελευταίο Friday shortlist σαν JSON αρχείο.
    """
    full_path = os.path.join(BASE_DIR, "logs", "friday_shortlist_v3.json")

    if not os.path.exists(full_path):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Friday shortlist file not found",
                    "path": full_path,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            404,
        )

    return send_file(
        full_path,
        mimetype="application/json",
        as_attachment=True,
    )


@app.route("/download/tuesday-recap-v2", methods=["GET"])
def download_tuesday_recap_v2():
    """
    Κατεβάζει το τελευταίο Tuesday recap σαν JSON αρχείο.
    """
    full_path = os.path.join(BASE_DIR, "logs", "tuesday_recap_v2.json")

    if not os.path.exists(full_path):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Tuesday recap file not found",
                    "path": full_path,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ),
            404,
        )

    return send_file(
        full_path,
        mimetype="application/json",
        as_attachment=True,
    )


# ------------------------------------------------------
#  GPT ENDPOINTS (READ-ONLY JSON REPORTS)
# ------------------------------------------------------
@app.route("/thursday-analysis-v3", methods=["GET"])
def api_thursday_analysis_v3():
    """
    Το GPT παίρνει το Thursday report από logs/thursday_report_v3.json.
    ΠΡΙΝ το διαβάσει, τρέχει τον Thursday engine για να φτιαχτεί/φρεσκαριστεί το report.
    """
    # 1) Τρέχουμε τον Thursday engine (ίδιο script με το /run/thursday-v3)
    #    Αγνοούμε την JSON απόκριση του run_script, το θέλουμε μόνο για το side-effect:
    #    να γραφτεί/ενημερωθεί το logs/thursday_report_v3.json.
    try:
        run_script("src/analysis/thursday_engine_full_v3.py")
    except Exception as e:
        print(f"⚠️ Error while auto-running Thursday engine: {e}", flush=True)

    # 2) Διαβάζουμε το report από logs/thursday_report_v3.json
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

    # 3) Επιστρέφουμε στο GPT το report όπως είναι, μέσα στο πεδίο "report"
    return jsonify(
        {
            "status": "ok",
            "script": "src/analysis/thursday_engine_full_v3.py",
            "timestamp": datetime.utcnow().isoformat(),
            "report": report,
        }
    )


@app.route("/friday-shortlist-v3", methods=["GET"])
def api_friday_shortlist_v3():
    """
    Το GPT παίρνει Friday shortlist από logs/friday_shortlist_v3.json.
    """
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


@app.route("/tuesday-recap", methods=["GET"])
def api_tuesday_recap():
    """
    Το GPT παίρνει Tuesday recap από logs/tuesday_recap_v2.json.
    """
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


# ------------------------------------------------------
#  ENTRY POINT
# ------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(
        f"🟢 Starting Bombay Engine Flask Server on port {port}...",
        flush=True,
    )
    app.run(host="0.0.0.0", port=port, use_reloader=False)
