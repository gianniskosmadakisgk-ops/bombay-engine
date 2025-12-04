import os
import json
import subprocess
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)

# ======================================================
#  Helper: τρέχει script και γυρίζει stdout / stderr
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
#  Helper: φορτώνει JSON report από δίσκο (ΧΩΡΙΣ να τρέχει script)
# ======================================================
def load_report_json(report_path: str):
    if not os.path.exists(report_path):
        print(f"⚠️ Report file not found: {report_path}", flush=True)
        return None, f"Report file not found: {report_path}"

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, None
    except Exception as e:
        print(f"⚠️ Failed to load report file {report_path}: {e}", flush=True)
        return None, str(e)


# ======================================================
#  MANUAL ENDPOINTS  (τα χρησιμοποιείς εσύ από browser)
# ======================================================
@app.route("/run/thursday-v3", methods=["GET"])
def run_thursday_v3():
    # Αυτό καλεί το μεγάλο script και γράφει το logs/thursday_report_v3.json
    return run_script("src/analysis/thursday_engine_full_v3.py")


# ======================================================
#  API ENDPOINTS ΓΙΑ GPT – γρήγορα, μόνο ανάγνωση report
# ======================================================
@app.route("/thursday-analysis-v3", methods=["GET"])
def api_thursday_analysis_v3():
    """
    Επιστρέφει το τελευταίο Thursday report από logs/thursday_report_v3.json
    ΔΕΝ ξανατρέχει το engine – υποθέτει ότι το /run/thursday-v3 έχει ήδη τρέξει.
    """
    report_path = "logs/thursday_report_v3.json"
    report_data, error = load_report_json(report_path)

    if report_data is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "script": "src/analysis/thursday_engine_full_v3.py",
                    "message": "Thursday report not available yet. Run /run/thursday-v3 first.",
                    "error": error,
                    "timestamp": datetime.utcnow().isoformat(),
                    "report": None,
                }
            ),
            503,
        )

    return jsonify(
        {
            "status": "ok",
            "script": "src/analysis/thursday_engine_full_v3.py",
            "timestamp": datetime.utcnow().isoformat(),
            "report": report_data,
        }
    )


# (placeholder – θα τα προσθέσουμε αργότερα αν θέλεις να συνδέσουμε Friday / Tuesday)
# @app.route("/friday-shortlist-v3", methods=["GET"])
# def api_friday_shortlist_v3():
#     ...

# @app.route("/tuesday-recap", methods=["GET"])
# def api_tuesday_recap():
#     ...


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
