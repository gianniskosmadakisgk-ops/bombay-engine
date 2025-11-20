import os
import requests
from flask import Flask, jsonify

app = Flask(__name__)

# ====== CONFIGURATION ======
API_KEY_FOOTBALL = os.getenv("FOOTBALL_API_KEY")
API_KEY_ODDS = os.getenv("ODDS_API_KEY")
WEBHOOK_URL = os.getenv("CHATGPT_WEBHOOK_URL")

BASE_URL_FOOTBALL = "https://v3.football.api-sports.io"
HEADERS_FOOTBALL = {"x-apisports-key": API_KEY_FOOTBALL}

# ====== BASIC ROUTES ======

@app.route("/")
def home():
    return jsonify({"status": "Bombay Engine Active", "version": "6.0"})

# ---------------------------------------------------------------------------
# 🧠 THURSDAY ANALYSIS
# ---------------------------------------------------------------------------

@app.route("/thursday-analysis", methods=["GET"])
def thursday_analysis():
    try:
        url = f"{BASE_URL_FOOTBALL}/fixtures"
        params = {"date": "2025-11-20"}  # ✅ Σημερινή ημερομηνία για δοκιμή
        response = requests.get(url, headers=HEADERS_FOOTBALL, params=params)
        data = response.json()

        matches = []
        for f in data.get("response", []):
            home = f["teams"]["home"]["name"]
            away = f["teams"]["away"]["name"]
            league = f["league"]["name"]
            odds = f.get("odds", None)
            matches.append({
                "league": league,
                "home": home,
                "away": away,
                "fixture_id": f["fixture"]["id"],
                "status": f["fixture"]["status"]["short"]
            })

        result = {
            "status": "Thursday Analysis complete",
            "count": len(matches),
            "matches": matches[:10]  # δείχνουμε τα πρώτα 10 για αρχή
        }

        # Προαιρετικά: Στέλνει στο ChatGPT Webhook
        if WEBHOOK_URL:
            try:
                requests.post(WEBHOOK_URL, json=result)
            except Exception as e:
                print("Webhook send failed:", e)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------------------------
# 🎯 FRIDAY SHORTLIST
# ---------------------------------------------------------------------------

@app.route("/friday-shortlist", methods=["GET"])
def friday_shortlist():
    try:
        # Παίρνουμε fixtures του ΣΚ (Παρασκευή–Κυριακή)
        url = f"{BASE_URL_FOOTBALL}/fixtures"
        params = {"from": "2025-11-21", "to": "2025-11-23"}
        response = requests.get(url, headers=HEADERS_FOOTBALL, params=params)
        data = response.json()

        shortlist = []
        for f in data.get("response", []):
            home = f["teams"]["home"]["name"]
            away = f["teams"]["away"]["name"]
            league = f["league"]["name"]
            shortlist.append({
                "league": league,
                "home": home,
                "away": away,
                "fixture_id": f["fixture"]["id"]
            })

        result = {
            "status": "Friday Shortlist complete",
            "count": len(shortlist),
            "shortlist": shortlist[:10]
        }

        if WEBHOOK_URL:
            try:
                requests.post(WEBHOOK_URL, json=result)
            except Exception as e:
                print("Webhook send failed:", e)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------------------------------------------------------------------
# 📊 TUESDAY RECAP
# ---------------------------------------------------------------------------

@app.route("/tuesday-recap", methods=["GET"])
def tuesday_recap():
    try:
        # Εδώ θα μπορούσαμε να τραβήξουμε παλιές αναφορές ή να δείξουμε dummy data
        recap = {
            "status": "Tuesday Recap complete",
            "week": "Week 1",
            "summary": {
                "Draw Engine": {"bets": 10, "won": 4, "roi": "+6.2%"},
                "Over/Under": {"bets": 8, "won": 5, "roi": "+11.5%"},
                "FanBet Draws": {"bets": 15, "won": 6, "roi": "-3.1%"},
                "Fraction Kelly": {"bets": 10, "won": 5, "roi": "+9.4%"}
            }
        }

        if WEBHOOK_URL:
            try:
                requests.post(WEBHOOK_URL, json=recap)
            except Exception as e:
                print("Webhook send failed:", e)

        return jsonify(recap)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
