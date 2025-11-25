import os
import requests
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
import json
from waitress import serve

app = Flask(__name__)

# -----------------------------------------------------------
# Environment variables
# -----------------------------------------------------------
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
API_URL = "https://v3.football.api-sports.io/fixtures"

HEADERS = {
    "x-apisports-key": FOOTBALL_API_KEY,
    "x-rapidapi-host": "v3.football.api-sports.io"
}

# -----------------------------------------------------------
# Helper: Υπολογισμός Παρασκευής - Δευτέρας
# -----------------------------------------------------------
def next_weekend_dates():
    today = datetime.utcnow()
    days_ahead = (4 - today.weekday()) % 7  # Παρασκευή
    friday = today + timedelta(days=days_ahead)
    monday = today + timedelta(days=days_ahead + 3)
    return friday.strftime("%Y-%m-%d"), monday.strftime("%Y-%m-%d")

# -----------------------------------------------------------
# Health check
# -----------------------------------------------------------
@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"message": "Server running", "status": "ok"})

# -----------------------------------------------------------
# Thursday Analysis (με debug logging)
# -----------------------------------------------------------
@app.route("/run_thursday_analysis", methods=["GET"])
def run_thursday_analysis():
    friday, monday = next_weekend_dates()
    params = {
        "from": friday,
        "to": monday,
        "season": 2025
    }

    print(f"📡 Fetching fixtures from {friday} to {monday}...")
    print(f"🔑 Using API key: {FOOTBALL_API_KEY[:6]}***")

    try:
        response = requests.get(API_URL, headers=HEADERS, params=params, timeout=30)
        print(f"🌍 API URL called: {response.url}")
        print(f"📦 Status code: {response.status_code}")
        data = response.json()
        print(f"🧾 API Response: {json.dumps(data, indent=2)[:800]}")

        if not data.get("response"):
            print("⚠️ Empty API response!")
            return jsonify({"status": "empty", "message": "No fixtures returned"}), 200

        return jsonify({
            "count": len(data.get("response", [])),
            "range": {"from": friday, "to": monday},
            "data_sample": data.get("response", []),
            "status": "success"
        })

    except Exception as e:
        print(f"❌ Error fetching fixtures: {e}")
        return jsonify({"error": str(e), "status": "fail"}), 500

# -----------------------------------------------------------
# Chat forward helper
# -----------------------------------------------------------
def send_to_chat(message):
    try:
        response = requests.post(
            "https://bombay-engine.onrender.com/chat_forward",
            json={"message": message},
            timeout=10
        )
        print("Chat forward:", response.status_code)
    except Exception as e:
        print("Chat forward error:", e)

# -----------------------------------------------------------
# Chat command handler
# -----------------------------------------------------------
@app.route("/chat_command", methods=["POST"])
def chat_command():
    try:
        data = request.get_json()
        command = data.get("command", "").lower().strip()

        if "thursday" in command:
            os.system("python3 thursday_analysis_v1.py")
            send_to_chat("🧠 Thursday Analysis executed successfully.")
            return jsonify({"response": "🧠 Thursday Analysis executed", "status": "ok"})

        elif "friday" in command:
            os.system("python3 friday.py")
            send_to_chat("🎯 Friday Shortlist executed successfully.")
            return jsonify({"response": "🎯 Friday Shortlist executed", "status": "ok"})

        elif "tuesday" in command:
            os.system("python3 tuesday_recap.py")
            send_to_chat("📊 Tuesday Recap executed successfully.")
            return jsonify({"response": "📊 Tuesday Recap executed", "status": "ok"})

        else:
            send_to_chat("❓ Unknown command received.")
            return jsonify({"response": "❓ Unknown command", "status": "fail"}), 400

    except Exception as e:
        send_to_chat(f"⚠️ Error in chat_command: {str(e)}")
        return jsonify({"response": str(e), "status": "error"}), 500

# -----------------------------------------------------------
# API Key check route
# -----------------------------------------------------------
@app.route("/check_api_key", methods=["GET"])
def check_api_key():
    key = os.getenv("FOOTBALL_API_KEY")
    if key:
        return jsonify({"status": "ok", "key_length": len(key), "starts_with": key[:6]})
    else:
        return jsonify({"status": "missing", "key": None})

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=10000)
