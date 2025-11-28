from flask import Flask, request, jsonify
import subprocess
import requests
import json
import os

app = Flask(__name__)

CHAT_FORWARD_URL = "https://bombay-engine.onrender.com/chat_forward"

@app.route("/chat_command", methods=["POST"])
def chat_command():
    try:
        data = request.get_json()
        command = data.get("command", "").lower().strip()

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

        print(f"🚀 Εκτέλεση εντολής: {label} ({script})")

        # Εκτέλεση του script
        result = subprocess.run(
            ["python3", script],
            capture_output=True, text=True
        )

        print("----- SCRIPT OUTPUT START -----")
        print(result.stdout)
        print("----- SCRIPT OUTPUT END -----")
        if result.stderr:
            print("⚠️ SCRIPT ERRORS:")
            print(result.stderr)

        # Αναζήτηση του παραγόμενου JSON report
        report_file = {
            "thursday_analysis_v1.py": "thursday_report_v1.json",
            "friday_shortlist_v1.py": "friday_shortlist_v1.json",
            "tuesday_recap.py": "tuesday_recap_v1.json",
        }.get(script)

        report_data = {}
        if report_file and os.path.exists(report_file):
            with open(report_file, "r", encoding="utf-8") as f:
                report_data = json.load(f)

        # Στέλνει στο chat
        message = {
            "message": f"✅ {label} ολοκληρώθηκε.",
            "data": report_data or {"info": "No data"},
        }
        requests.post(CHAT_FORWARD_URL, json=message, timeout=15)

        return jsonify({"response": f"{label} executed", "status": "ok"})

    except Exception as e:
        print(f"⚠️ Error executing command: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/chat_forward", methods=["POST"])
def chat_forward():
    data = request.get_json()
    print("💬 Incoming message:", data)
    return jsonify({"status": "received"}), 200


@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"message": "Server running", "status": "ok"})


if __name__ == "__main__":
    print("🟢 Starting Bombay Engine Flask Server...")
    app.run(host="0.0.0.0", port=10000)
