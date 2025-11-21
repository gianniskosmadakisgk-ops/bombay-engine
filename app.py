import os
import requests
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# ===== CONFIGURATION =====
API_KEY_FOOTBALL = os.getenv("FOOTBALL_API_KEY")
API_KEY_ODDS = os.getenv("ODDS_API_KEY")
WEBHOOK_URL = os.getenv("CHATGPT_WEBHOOK_URL")

BASE_URL_FOOTBALL = "https://v3.football.api-sports.io"
HEADERS_FOOTBALL = {"x-apisports-key": API_KEY_FOOTBALL}

# ===== BASIC ROUTE =====
@app.route("/")
def home():
    return jsonify({
        "status": "Bombay Engine is live",
        "timestamp": datetime.utcnow().isoformat()
    })

# ===== CHAT RECEIVER =====
@app.route("/chat", methods=["POST"])
def chat_receiver():
    """Endpoint για να λαμβάνει και να εμφανίζει μηνύματα από το Render στο Chat"""
    data = request.get_json()
    print("📩 Ελήφθη μήνυμα στο Chat:", data)
    return jsonify({"status": "Message received by Chat", "data": data}), 200

# ===== SEND TO CHAT FUNCTION =====
def send_to_chat(message: str):
    """Αποστολή μηνύματος στο ChatGPT Webhook"""
    if WEBHOOK_URL:
        try:
            payload = {"content": message}
            requests.post(WEBHOOK_URL, json=payload, timeout=10)
            print("✅ Μήνυμα εστάλη στο Chat:", message)
        except Exception as e:
            print("⚠️ Webhook Error:", e)

# ===== ANALYSIS ROUTES =====
@app.route("/thursday-analysis", methods=["GET"])
def thursday_analysis():
    result = {
        "status": "Thursday Analysis complete",
        "timestamp": datetime.utcnow().isoformat()
    }
    send_to_chat(f"✅ Thursday Analysis Complete\n🕒 {result['timestamp']}")
    return jsonify(result)

@app.route("/friday-shortlist", methods=["GET"])
def friday_shortlist():
    result = {
        "status": "Friday Shortlist ready",
        "timestamp": datetime.utcnow().isoformat()
    }
    send_to_chat(f"📋 Friday Shortlist Ready\n🕒 {result['timestamp']}")
    return jsonify(result)

@app.route("/tuesday-recap", methods=["GET"])
def tuesday_recap():
    result = {
        "status": "Tuesday Recap completed",
        "timestamp": datetime.utcnow().isoformat()
    }
    send_to_chat(f"🔁 Tuesday Recap Completed\n🕒 {result['timestamp']}")
    return jsonify(result)

# ===== RUN APP =====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
