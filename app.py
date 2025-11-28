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

        print(f"🚀 Εκτέλεση εντολής: {label} ({script})")

        # -----------------------------------------------------------
        # Εκτέλεση του script (με logs)
        # -----------------------------------------------------------
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

        # -----------------------------------------------------------
        # Αναζήτηση του παραγόμενου JSON report
        # -----------------------------------------------------------
        report_file = {
            "thursday_analysis_v1.py": "thursday_report_v1.json",
            "friday_shortlist_v1.py": "friday_shortlist_v1.json",
            "tuesday_recap.py": "tuesday_recap_v1.json",
        }.get(script)

        report_data = {}
        if report_file and os.path.exists(report_file):
            with open(report_file, "r", encoding="utf-8") as f:
                report_data = json.load(f)
        else:
            report_data = {"info": f"⚠️ Report file not found: {report_file}"}

        # -----------------------------------------------------------
        # Στέλνει στο chat (μέσω chat_forward)
        # -----------------------------------------------------------
        message = {
            "message": f"✅ {label} ολοκληρώθηκε επιτυχώς.",
            "data": report_data,
        }
        response = requests.post(CHAT_FORWARD_URL, json=message, timeout=15)
        print(f"📤 Report sent to chat, status: {response.status_code}")

        return jsonify({"response": f"{label} executed", "status": "ok"})

    except Exception as e:
        print(f"⚠️ Error executing command: {e}")
        return jsonify({"error": str(e)}), 500


# -----------------------------------------------------------
# Chat Forward Endpoint (δέχεται reports)
# -----------------------------------------------------------
@app.route("/chat_forward", methods=["GET", "POST"])
def chat_forward():
    try:
        if request.method == "GET":
            return jsonify({"status": "ok", "info": "Chat Forward endpoint is live."}), 200

        data = request.get_json()
        print("💬 Incoming message:", json.dumps(data, indent=2, ensure_ascii=False))
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
    print("🟢 Starting Bombay Engine Flask Server...")
    app.run(host="0.0.0.0", port=10000)
