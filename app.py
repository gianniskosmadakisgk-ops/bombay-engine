from flask import Flask, request, jsonify
import subprocess
import requests
import json
import os
import sys

# -----------------------------------------------------------
# 🔧 Render log flush (ώστε να βλέπεις τα print άμεσα)
# -----------------------------------------------------------
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# -----------------------------------------------------------
# Flask App setup
# -----------------------------------------------------------
app = Flask(__name__)

CHAT_FORWARD_URL = "https://bombay-engine.onrender.com/chat_forward"


# -----------------------------------------------------------
# /chat_command — Trigger engine commands
# -----------------------------------------------------------
@app.route("/chat_command", methods=["POST"])
def chat_command():
    try:
        print("📩 Received POST /chat_command")
        data = request.get_json(force=True)
        print(f"🧾 Raw data: {data}")

        command = (data.get("command", "") or "").lower().strip()
        print(f"🧭 Command detected: {command}")

        # Επιλογή script
        if "thursday" in command:
            script = "thursday_analysis_v1.py"
            label = "Thursday Analysis"
        elif "friday" in command:
            script = "friday_shortlist_v1.py"
            label = "Friday Shortlist"
        elif "tuesday" in command:
            script = "tuesday_recap.py"
            label = "Tuesday Recap"
        else:
            print("❌ Unknown command received.")
            return jsonify({"error": "❓ Unknown command"}), 400

        print(f"🚀 Εκτέλεση εντολής: {label} ({script})")

        # Εκτέλεση script στο Render environment
        env = os.environ.copy()
        print("⚙️ Starting subprocess...")

        result = subprocess.run(
            ["python3", script],
            cwd="/opt/render/project/src",
            env=env,
            capture_output=True,
            text=True,
            check=True
        )

        print("✅ Subprocess finished successfully.")
        print("----- SCRIPT OUTPUT START -----")
        print(result.stdout)
        print("----- SCRIPT OUTPUT END -----")

        if result.stderr:
            print("⚠️ SCRIPT ERRORS:")
            print(result.stderr)

        # Εύρεση του report file
        report_file = {
            "thursday_analysis_v1.py": "logs/thursday_report_v1.json",
            "friday_shortlist_v1.py": "logs/friday_shortlist_v1.json",
            "tuesday_recap.py": "logs/tuesday_recap_v1.json",
        }.get(script)

        report_data = {}
        if report_file and os.path.exists(report_file):
            with open(report_file, "r", encoding="utf-8") as f:
                report_data = json.load(f)
        else:
            print("⚠️ No report file found after script run.")

        # Αποστολή δεδομένων στο Chat
        message = {
            "message": f"✅ {label} ολοκληρώθηκε επιτυχώς.",
            "output": result.stdout or "No console output",
            "data": report_data or {"info": "No data"},
        }

        response = requests.post(CHAT_FORWARD_URL, json=message, timeout=15)
        print(f"📤 Report sent to chat, status: {response.status_code}")

        return jsonify({
            "response": f"{label} executed",
            "status": "ok",
            "http_status": response.status_code
        })

    except subprocess.CalledProcessError as e:
        print(f"❌ Subprocess failed: {e}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return jsonify({"error": "Subprocess failed", "details": e.stderr}), 500

    except Exception as e:
        print(f"⚠️ General error executing command: {e}")
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------
# /chat_forward — internal messages
# -----------------------------------------------------------
@app.route("/chat_forward", methods=["POST"])
def chat_forward():
    try:
        data = request.get_json()
        print("💬 Incoming message:", data)
        return jsonify({"status": "received"}), 200
    except Exception as e:
        print(f"⚠️ Error in chat_forward: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


# -----------------------------------------------------------
# /healthcheck — server heartbeat
# -----------------------------------------------------------
@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"message": "Server running", "status": "ok"})


# -----------------------------------------------------------
# MAIN ENTRYPOINT
# -----------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🟢 Starting Bombay Engine Flask Server on port {port}...")
    app.run(host="0.0.0.0", port=port, use_reloader=False)
