from flask import Flask, request, jsonify
import subprocess
import requests
import json
import os

app = Flask(__name__)

# -----------------------------------------------------------
# Chat Forward URL (όπου στέλνονται τα reports)
# -----------------------------------------------------------
CHAT_FORWARD_URL = "https://bombay-engine.onrender.com/chat_forward"


# -----------------------------------------------------------
# Chat Command Handler
# -----------------------------------------------------------
@app.route("/chat_command", methods=["POST"])
def chat_command():
    try:
        data = request.get_json()
        command = data.get("command", "").lower().strip()

        # Αναγνώριση εντολής
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
            return jsonify({"error": "❓ Unknown command"}), 400

        print(f"🚀 Running {label} using script: {script}")

        # Εκτέλεση του script
        result = subprocess.run(
            ["python3", script],
            capture_output=True, text=True
        )

        # -----------------------------------------------------------
        # Διαβάζει το παραγόμενο JSON report (αν υπάρχει)
        # -----------------------------------------------------------
        report_file = None
        if "thursday" in script:
            report_file = "thursday_report_v1.json"
        elif "friday" in script:
            report_file = "friday_shortlist_v1.json"
        elif "tuesday" in script:
            report_file = "tuesday_recap_v1.json"

        report_data = None
        if report_file and os.path.exists(report_file):
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    report_data = json.load(f)
            except Exception as e:
                report_data = {"error": f"⚠️ Error reading report file: {str(e)}"}
        else:
            report_data = {"info": "⚠️ No report file found."}

        # -----------------------------------------------------------
        # Προετοιμασία δεδομένων για αποστολή στο Chat
        # -----------------------------------------------------------
        message = {
            "message": f"✅ {label} executed successfully.",
            "output": result.stdout or "No console output",
            "data": report_data
        }

        # -----------------------------------------------------------
        # Αποστολή στο Chat Forward endpoint
        # -----------------------------------------------------------
        response = requests.post(CHAT_FORWARD_URL, json=message, timeout=20)
        print(f"📤 Report sent to chat, status: {response.status_code}")

        return jsonify({
            "response": f"{label} executed",
            "status": "ok",
            "http_status": response.status_code
        })

    except Exception as e:
        print(f"⚠️ Error executing command: {e}")
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------
# Chat Forward Endpoint (δέχεται reports)
# -----------------------------------------------------------
@app.route("/chat_forward", methods=["POST"])
def chat_forward():
    try:
        data = request.get_json()
        print("💬 Incoming message to chat:", data.get("message", "No message"))
        return jsonify({"status": "received", "message": data.get("message")}), 200
    except Exception as e:
        print(f"⚠️ Error in chat_forward: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


# -----------------------------------------------------------
# Healthcheck (έλεγχος λειτουργίας)
# -----------------------------------------------------------
@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"message": "Server running", "status": "ok"})


# -----------------------------------------------------------
# Main (εκκίνηση Flask server)
# -----------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
