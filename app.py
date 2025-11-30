import os
import subprocess
from flask import Flask, jsonify

app = Flask(__name__)

# ============================================
# Helper to run scripts safely
# ============================================
def run_script(script_name):
    try:
        print(f"=== Running script: {script_name} ===", flush=True)

        result = subprocess.run(
            ["python3", script_name],
            cwd="/opt/render/project/src",  # root του project στη Render
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
            "status": "ok",
            "script": script_name,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except Exception as e:
        print(f"❌ Error running {script_name}: {e}", flush=True)
        return {
            "status": "error",
            "script": script_name,
            "error": str(e),
        }


# ============================================
# Routes για manual runs
# ============================================
@app.route("/run/thursday", methods=["GET"])
def run_thursday():
    # Το αρχείο μας λέγεται thursday_analysis_v1.py
    return jsonify(run_script("thursday_analysis_v1.py"))


@app.route("/run/friday", methods=["GET"])
def run_friday():
    # Το κανονικό shortlist είναι το v2
    return jsonify(run_script("friday_shortlist_v2.py"))


@app.route("/run/tuesday", methods=["GET"])
def run_tuesday():
    # Το recap είναι το v2
    return jsonify(run_script("tuesday_recap_v2.py"))


@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🟢 Starting Bombay Engine Flask Server on port {port}...", flush=True)
    app.run(host="0.0.0.0", port=port, use_reloader=False)
