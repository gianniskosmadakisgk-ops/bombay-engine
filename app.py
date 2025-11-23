import os
import requests
from datetime import datetime, timedelta
from flask import Flask, jsonify

app = Flask(__name__)

# Παίρνει το API key από το Render Environment
API_KEY = os.getenv("FOOTBALL_API_KEY")

# Βασική διαδρομή API
BASE_URL = "https://v3.football.api-sports.io/fixtures"

# Λίγκες που παρακολουθούμε
LEAGUES = [
    61,   # Ligue 1
    135,  # Serie A
    140,  # La Liga
    40,   # Championship
    138,  # Serie B
    62,   # Ligue 2
    94,   # Liga Portugal 2
    207,  # Swiss Super League
    78,   # Bundesliga
    88,   # Eredivisie
    144   # Jupiler Pro League
]

@app.route("/run_thursday_analysis", methods=["GET"])
def run_thursday_analysis():
    try:
        today = datetime.now()
        start_date = (today + timedelta(days=(4 - today.weekday()) % 7)).strftime("%Y-%m-%d")  # Παρασκευή
        end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=3)).strftime("%Y-%m-%d")  # Δευτέρα

        fixtures = []

        headers = {
            "x-apisports-key": API_KEY
        }

        for league_id in LEAGUES:
            params = {
                "league": league_id,
                "season": 2024,
                "from": start_date,
                "to": end_date
            }

            response = requests.get(BASE_URL, headers=headers, params=params)
            data = response.json()

            if "response" in data:
                fixtures.extend(data["response"])

        return jsonify({
            "count": len(fixtures),
            "data": fixtures,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
