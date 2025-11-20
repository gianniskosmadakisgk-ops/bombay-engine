import os
import requests
from flask import Flask, jsonify

app = Flask(__name__)

# ============================================================
# 🔧 CONFIGURATION
# ============================================================
API_KEY_FOOTBALL = os.getenv("FOOTBALL_API_KEY")
API_KEY_ODDS = os.getenv("ODDS_API_KEY")
WEBHOOK_URL = os.getenv("CHATGPT_WEBHOOK_URL")

BASE_URL_FOOTBALL = "https://v3.football.api-sports.io"
HEADERS_FOOTBALL = {"x-apisports-key": API_KEY_FOOTBALL}

# ============================================================
# 🧠 BASIC ROUTES
# ============================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Bombay Engine Active", "version": "6.0"})

# ============================================================
# ⚽ THURSDAY ANALYSIS
# ============================================================
@app.route("/thursday-analysis", methods=["GET"])
def thursday_analysis():
    try:
        url = f"{BASE_URL_FOOTBALL}/fixtures"
        params = {"date": "2025-11-20"}  # 📅 Σημερινή ημερομηνία
        response = requests.get(url, headers=HEADERS_FOOTBALL, params=params)
        data = response.json()

        if "response" not in data:
            return jsonify({"status": "error", "message": "No data field returned", "data": data})

        matches = []
        for match in data["response"]:
            league = match["league"]["name"]
            home = match["teams"]["home"]["name"]
            away = match["teams"]["away"]["name"]
            date = match["fixture"]["date"]
            matches.append({"league": league, "home": home, "away": away, "date": date})

        return jsonify({
            "status": "Thursday Analysis complete",
            "count": len(matches),
            "matches": matches
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ============================================================
# 🧮 FRIDAY SHORTLIST
# ============================================================
@app.route("/friday-shortlist", methods=["GET"])
def friday_shortlist():
    try:
        # Placeholder - θα μπει λογική value picks
        return jsonify({
            "status": "Friday Shortlist complete",
            "matches": []
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ============================================================
# 📊 TUESDAY RECAP
# ============================================================
@app.route("/tuesday-recap", methods=["GET"])
def tuesday_recap():
    try:
        # Placeholder - θα μπει λογική ταμείων & ROIs
        return jsonify({
            "status": "Tuesday Recap complete",
            "summary": {}
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# ============================================================
# 🚀 START APP
# ============================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
