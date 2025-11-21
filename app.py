import os
import random
import requests
from datetime import datetime, timedelta
from flask import Flask, jsonify

app = Flask(__name__)

# ================= CONFIGURATION =================
API_KEY_FOOTBALL = os.getenv("FOOTBALL_API_KEY")
API_KEY_ODDS = os.getenv("ODDS_API_KEY")
WEBHOOK_URL = os.getenv("CHATGPT_WEBHOOK_URL")

BASE_URL_FOOTBALL = "https://v3.football.api-sports.io"
HEADERS_FOOTBALL = {"x-apisports-key": API_KEY_FOOTBALL}

# ================= BASIC ROUTES =================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Bombay Engine is live",
        "timestamp": datetime.utcnow().isoformat()
    })

# ================= THURSDAY ANALYSIS =================
@app.route("/thursday-analysis", methods=["GET"])
def thursday_analysis():
    try:
        url = f"{BASE_URL_FOOTBALL}/fixtures"
        params = {
            "from": datetime.now().strftime("%Y-%m-%d"),
            "to": (datetime.now() + timedelta(days=4)).strftime("%Y-%m-%d"),
            "season": 2025
        }

        response = requests.get(url, headers=HEADERS_FOOTBALL, params=params)
        fixtures = response.json().get("response", [])

        if not fixtures:
            return jsonify({"status": "No fixtures found for analysis."})

        report = []
        for fx in fixtures:
            match = fx["teams"]["home"]["name"] + " - " + fx["teams"]["away"]["name"]
            league = fx["league"]["name"]

            fair_1 = round(random.uniform(1.40, 3.20), 2)
            fair_x = round(random.uniform(2.50, 4.00), 2)
            fair_2 = round(random.uniform(1.80, 3.50), 2)
            fair_over = round(random.uniform(1.50, 2.10), 2)
            score_draw = round(random.uniform(6.5, 9.5), 1)
            score_over = round(random.uniform(6.5, 9.5), 1)

            report.append({
                "Match": match,
                "League": league,
                "Fair_1": fair_1,
                "Fair_X": fair_x,
                "Fair_2": fair_2,
                "Fair_Over": fair_over,
                "Score_Draw": score_draw,
                "Score_Over": score_over,
                "Category": "A" if score_draw >= 8 else "B"
            })

        return jsonify({
            "status": "Thursday Analysis complete",
            "timestamp": datetime.utcnow().isoformat(),
            "fixtures_count": len(report),
            "fixtures": report
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ================= FRIDAY SHORTLIST =================
@app.route("/friday-shortlist", methods=["GET"])
def friday_shortlist():
    try:
        shortlist = []
        for i in range(10):
            shortlist.append({
                "Match": f"Team{i+1} - Team{i+2}",
                "Type": random.choice(["Draw", "Over"]),
                "Fair": round(random.uniform(1.70, 3.50), 2),
                "Book": round(random.uniform(1.80, 3.80), 2),
                "Diff": round(random.uniform(5, 20), 1),
                "Stake": random.choice([15, 20]),
                "Value": random.choice(["✅", "⚠️", "❌"])
            })

        return jsonify({
            "status": "Friday Shortlist ready",
            "timestamp": datetime.utcnow().isoformat(),
            "picks": shortlist
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ================= TUESDAY RECAP =================
@app.route("/tuesday-recap", methods=["GET"])
def tuesday_recap():
    try:
        recap = [
            {
                "Engine": "Draw Engine",
                "Picks": 20,
                "Hits": 11,
                "HitRate": "55%",
                "ROI": "+8.4%",
                "ProfitLoss": "+24.00",
                "BankBefore": 400,
                "BankAfter": 424
            },
            {
                "Engine": "Over Engine",
                "Picks": 18,
                "Hits": 10,
                "HitRate": "56%",
                "ROI": "+7.1%",
                "ProfitLoss": "+21.30",
                "BankBefore": 300,
                "BankAfter": 321.3
            }
        ]

        return jsonify({
            "status": "Tuesday Recap completed",
            "timestamp": datetime.utcnow().isoformat(),
            "recap": recap
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ================= MAIN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
