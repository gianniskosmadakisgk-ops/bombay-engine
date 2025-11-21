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

# ===== STARTUP TEST =====
def send_webhook_test():
    if not WEBHOOK_URL:
        print("⚠️ No CHATGPT_WEBHOOK_URL found.")
        return
    payload = {
        "event": "Startup",
        "message": "Bombay Engine is live and connected ✅",
        "timestamp": datetime.utcnow().isoformat()
    }
    try:
        r = requests.post(WEBHOOK_URL, json=payload, timeout=10)
        print(f"Webhook test sent: {r.status_code} - {r.text[:100]}")
    except Exception as e:
        print(f"Webhook test failed: {e}")

# ===== BASIC ROUTES =====
@app.route("/")
def home():
    return jsonify({"status": "Bombay Engine Active", "version": "6.0"})

# ===== THURSDAY ANALYSIS =====
@app.route("/thursday-analysis", methods=["GET"])
def thursday_analysis():
    try:
        url = f"{BASE_URL_FOOTBALL}/fixtures"
        params = {"date": datetime.utcnow().strftime("%Y-%m-%d")}
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params)
        data = r.json()
        webhook_payload = {
            "event": "ThursdayAnalysis",
            "status": "complete",
            "fixture_count": len(data.get("response", [])),
            "timestamp": datetime.utcnow().isoformat()
        }
        requests.post(WEBHOOK_URL, json=webhook_payload)
        return jsonify(webhook_payload)
    except Exception as e:
        return jsonify({"error": str(e)})

# ===== FRIDAY SHORTLIST =====
@app.route("/friday-shortlist", methods=["GET"])
def friday_shortlist():
    payload = {"status": "Friday Shortlist ready", "timestamp": datetime.utcnow().isoformat()}
    requests.post(WEBHOOK_URL, json=payload)
    return jsonify(payload)

# ===== TUESDAY RECAP =====
@app.route("/tuesday-recap", methods=["GET"])
def tuesday_recap():
    payload = {"status": "Tuesday Recap complete", "timestamp": datetime.utcnow().isoformat()}
    requests.post(WEBHOOK_URL, json=payload)
    return jsonify(payload)

if __name__ == "__main__":
    send_webhook_test()
    app.run(host="0.0.0.0", port=10000)
