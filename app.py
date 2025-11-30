import os
import subprocess
from flask import Flask, jsonify

app = Flask(__name__)

# ----------------------------------------
# Helper to run scripts safely
# ----------------------------------------
def run_script(script_name):
    try:
        print(f"=== Running script: {script_name} ===", flush=True)

        result = subprocess.run(
            ["python3", script_name],
            cwd="/opt/render/project/src",
            capture_output=True,
            text=True
        )

        print(result.stdout)
        print("=== SCRIPT OUTPUT END ===", flush=True)

        if result.stderr:
            print("⚠ SCRIPT ERRORS:", flush=True)
            print(result.stderr)

        return jsonify({
            "status": "ok",
            "script": script_name,
            "stdout": result.stdout,
            "stderr": result.stderr
        })

    except Exception as e:
        print(f"❌ Error running {script_name}: {e}", flush=True)
        return jsonify({"status": "error", "message": str(e)}), 500


# ----------------------------------------
# ROUTES
# ----------------------------------------
@app.route("/")
def home():
    return jsonify({"message": "Bombay Engine API is running."})


@app.route("/run/thursday", methods=["GET"])
def run_thursday():
    return run_script("thursday_analysis_v2.py")


@app.route("/run/friday", methods=["GET"])
def run_friday():
    return run_script("friday_shortlist_v2.py")


@app.route("/run/tuesday", methods=["GET"])
def run_tuesday():
    return run_script("tuesday_recap_v2.py")


# ----------------------------------------
# START SERVER
# ----------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"=== Starting Bombay Engine Flask Server on port {port} ===", flush=True)
    app.run(host="0.0.0.0", port=port)
