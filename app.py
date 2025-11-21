import os
import requests
from flask import Flask, jsonify
from datetime import datetime

app = Flask(__name__)

# ===== CONFIGURATION =====
API_KEY_FOOTBALL = os.getenv("FOOTBALL_API_KEY")
API_KEY_ODDS = os.getenv("ODDS_API_KEY")
WEBHOOK_URL = os.getenv("CHATGPT_WEBHOOK_URL")

BASE_URL_FOOTBALL = "https://v3.football.api-sports.io"
HEADERS_FOOTBALL = {"x-apisports-key": API_KEY_FOOTBALL}


# ===== HELPERS =====
def send_webhook(event, message):
    """Send JSON payload to ChatGPT Webhook"""
    if not WEBHOOK_URL:
        print("⚠️ No CHATGPT_WEBHOOK_URL found.")
        return
    payload = {
        "event": event,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        print(f"Webhook → {event}: {r.status_code}")
    except Exception as e:
        print(f"Webhook error: {e}")


# ===== BASIC ROUTES =====
@app.route("/")
def home():
    """Startup check & webhook handshake"""
    send_webhook("Startup", "Bombay Engine connected successfully ✅")
    return jsonify({"status": "Bombay Engine Active", "version": "6.0"})


# ===== THURSDAY ANALYSIS =====
@app.route("/thursday-analysis", methods=["GET"])
def thursday_analysis():
    try:
        url = f"{BASE_URL_FOOTBALL}/fixtures"
        params = {"date": datetime.utcnow().strftime("%Y-%m-%d")}
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params)
        data = r.json()
        result = {
            "event": "ThursdayAnalysis",
            "status": "complete",
            "fixture_count": len(data.get("response", [])),
            "timestamp": datetime.utcnow().isoformat()
        }
        send_webhook("ThursdayAnalysis", f"✅ Thursday report ready ({result['fixture_count']} fixtures)")
        return jsonify(result)
    except Exception as e:
        send_webhook("ThursdayAnalysisError", str(e))
        return jsonify({"error": str(e)})


# ===== FRIDAY SHORTLIST =====
@app.route("/friday-shortlist", methods=["GET"])
def friday_shortlist():
    result = {"status": "Friday Shortlist ready", "timestamp": datetime.utcnow().isoformat()}
    send_webhook("FridayShortlist", "🎯 Friday shortlist generated successfully")
    return jsonify(result)


# ===== TUESDAY RECAP =====
@app.route("/tuesday-recap", methods=["GET"])
def tuesday_recap():
    result = {"status": "Tuesday Recap complete", "timestamp": datetime.utcnow().isoformat()}
    send_webhook("TuesdayRecap", "📊 Tuesday recap completed successfully")
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
