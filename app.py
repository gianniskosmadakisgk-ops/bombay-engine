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
@app.route("/")
def home():
    return jsonify({
        "status": "Bombay Engine is live",
        "timestamp": "active"
    })


# ========== THURSDAY ANALYSIS ==========
@app.route("/thursday-analysis", methods=["GET"])
def thursday_analysis():
    try:
        # Παίρνουμε fixtures από API
        url = f"{BASE_URL_FOOTBALL}/fixtures"
        params = {"date": "2025-11-21"}  # Ενδεικτική ημερομηνία (θα μπει dynamic αργότερα)
        response = requests.get(url, headers=HEADERS_FOOTBALL, params=params)
        data = response.json()

        # Ετοιμάζουμε απλό αποτέλεσμα
        analysis_result = {
            "status": "Thursday Analysis complete",
            "matches_found": len(data.get("response", [])),
            "timestamp": "2025-11-21T09:55:00Z"
        }

        # Στέλνουμε το αποτέλεσμα πίσω στο ChatGPT (webhook)
        if WEBHOOK_URL:
            try:
                requests.post(WEBHOOK_URL, json=analysis_result, timeout=10)
            except Exception as e:
                print("⚠️ Webhook failed:", e)

        return jsonify(analysis_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ========== FRIDAY SHORTLIST ==========
@app.route("/friday-shortlist", methods=["GET"])
def friday_shortlist():
    shortlist = {
        "status": "Friday Shortlist ready",
        "picks": ["Team A - Team B", "Team C - Team D"],
        "timestamp": "2025-11-21T09:55:00Z"
    }

    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json=shortlist, timeout=10)
        except Exception as e:
            print("⚠️ Webhook failed:", e)

    return jsonify(shortlist)


# ========== TUESDAY SUMMARY ==========
@app.route("/tuesday-recap", methods=["GET"])
def tuesday_recap():
    recap = {
        "status": "Tuesday Recap complete",
        "summary": "5 matches analyzed, 3 wins, 2 losses",
        "timestamp": "2025-11-21T09:55:00Z"
    }

    if WEBHOOK_URL:
        try:
            requests.post(WEBHOOK_URL, json=recap, timeout=10)
        except Exception as e:
            print("⚠️ Webhook failed:", e)

    return jsonify(recap)


# ========== MAIN ==========
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
