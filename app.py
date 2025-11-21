import os
import requests
from flask import Flask, jsonify

app = Flask(__name__)

# ========== CONFIGURATION ==========
API_KEY_FOOTBALL = os.getenv("FOOTBALL_API_KEY")
API_KEY_ODDS = os.getenv("ODDS_API_KEY")
WEBHOOK_URL = os.getenv("CHATGPT_WEBHOOK_URL")

BASE_URL_FOOTBALL = "https://v3.football.api-sports.io"
HEADERS_FOOTBALL = {"x-apisports-key": API_KEY_FOOTBALL}

# ========== BASIC ROUTES ==========
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Bombay Engine is live", "timestamp": os.popen("date -u").read().strip()})

# ========== SEND TO CHAT FUNCTION ==========
def send_to_chat(message: str):
    """Send message back to ChatGPT Agent via webhook"""
    if not WEBHOOK_URL:
        print("⚠️ No ChatGPT Webhook URL set.")
        return {"error": "No webhook URL configured"}

    try:
        payload = {"content": message}
        response = requests.post(WEBHOOK_URL, json=payload)
        response.raise_for_status()
        print(f"✅ Message sent to ChatGPT: {message}")
        return {"status": "sent", "message": message}
    except Exception as e:
        print(f"❌ Error sending message to ChatGPT: {e}")
        return {"error": str(e)}

# ========== THURSDAY ANALYSIS ==========
@app.route("/thursday-analysis", methods=["GET"])
def thursday_analysis():
    try:
        url = f"{BASE_URL_FOOTBALL}/fixtures"
        params = {"date": "2025-11-20"}  # Σημερινή ημερομηνία για δοκιμή
        response = requests.get(url, headers=HEADERS_FOOTBALL, params=params)
        data = response.json()

        # Ανάλυση (μπορείς να το προσαρμόσεις)
        matches = data.get("response", [])
        result = f"Thursday Analysis complete. Found {len(matches)} fixtures."

        # Στείλε στο Chat
        send_to_chat(result)

        return jsonify({"status": "Thursday Analysis complete", "count": len(matches)})
    except Exception as e:
        return jsonify({"error": str(e)})

# ========== FRIDAY SHORTLIST ==========
@app.route("/friday-shortlist", methods=["GET"])
def friday_shortlist():
    try:
        url = f"{BASE_URL_FOOTBALL}/fixtures"
        params = {"date": "2025-11-21"}
        response = requests.get(url, headers=HEADERS_FOOTBALL, params=params)
        data = response.json()

        matches = data.get("response", [])
        result = f"Friday Shortlist ready. Found {len(matches)} fixtures."

        send_to_chat(result)

        return jsonify({"status": "Friday Shortlist ready", "count": len(matches)})
    except Exception as e:
        return jsonify({"error": str(e)})

# ========== TUESDAY RECAP ==========
@app.route("/tuesday-recap", methods=["GET"])
def tuesday_recap():
    try:
        url = f"{BASE_URL_FOOTBALL}/fixtures"
        params = {"date": "2025-11-18"}
        response = requests.get(url, headers=HEADERS_FOOTBALL, params=params)
        data = response.json()

        matches = data.get("response", [])
        result = f"Tuesday Recap complete. Found {len(matches)} fixtures."

        send_to_chat(result)

        return jsonify({"status": "Tuesday Recap complete", "count": len(matches)})
    except Exception as e:
        return jsonify({"error": str(e)})

# ========== MAIN ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
