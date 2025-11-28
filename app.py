from flask import Flask, request, jsonify
import subprocess
import requests
import json

app = Flask(__name__)

CHAT_FORWARD_URL = "https://bombay-engine.onrender.com/chat_forward"

@app.route("/chat_command", methods=["POST"])
def chat_command():
    try:
        data = request.get_json()
        command = data.get("command", "").lower()

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
            return jsonify({"error": "Unknown command"}), 400

        # Εκτέλεση του script
        result = subprocess.run(
            ["python3", script],
            capture_output=True, text=True
        )

        # Προετοιμασία δεδομένων για αποστολή στο chat
        message = {
            "message": f"✅ {label} executed and sent.",
            "output": result.stdout or "No output",
        }

        # Αποστολή στο chat
        requests.post(CHAT_FORWARD_URL, json=message, timeout=15)

        return jsonify({"response": f"{label} executed", "status": "ok"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"message": "Server running", "status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
