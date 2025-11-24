import os
import requests
from datetime import datetime, timedelta
from flask import Flask, jsonify

app = Flask(__name__)

# Environment variables
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
API_URL = "https://v3.football.api-sports.io/fixtures"

HEADERS = {
    "x-apisports-key": FOOTBALL_API_KEY
}

# --- Utility: calculate the date range for next Fri–Mon ---
def next_weekend_dates():
    today = datetime.utcnow()
    # find next Friday
    days_ahead = (4 - today.weekday()) % 7  # 4 = Friday
    friday = today + timedelta(days=days_ahead)
    monday = friday + timedelta(days=3)
    return friday.strftime("%Y-%m-%d"), monday.strftime("%Y-%m-%d")

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({"message": "Server running", "status": "ok"})

# --- Thursday Analysis ---
@app.route('/run_thursday_analysis', methods=['GET'])
def run_thursday_analysis():
    try:
        friday, monday = next_weekend_dates()
        params = {
            "from": friday,
            "to": monday,
            "season": 2024
        }
        response = requests.get(API_URL, headers=HEADERS, params=params, timeout=30)
        data = response.json().get("response", [])
        return jsonify({"count": len(data), "data": data, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"})

# --- Friday Shortlist ---
@app.route('/run_friday_shortlist', methods=['GET'])
def run_friday_shortlist():
    try:
        friday, monday = next_weekend_dates()
        params = {
            "from": friday,
            "to": monday,
            "season": 2024
        }
        response = requests.get(API_URL, headers=HEADERS, params=params, timeout=30)
        data = response.json().get("response", [])
        shortlist = []
        for match in data[:10]:
            team_home = match["teams"]["home"]["name"]
            team_away = match["teams"]["away"]["name"]
            league = match["league"]["name"]
            shortlist.append({
                "match": f"{team_home} vs {team_away}",
                "league": league
            })
        return jsonify({"count": len(shortlist), "data": shortlist, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"})

# --- Tuesday Recap ---
@app.route('/run_tuesday_recap', methods=['GET'])
def run_tuesday_recap():
    try:
        friday, monday = next_weekend_dates()
        params = {
            "from": friday,
            "to": monday,
            "season": 2024
        }
        response = requests.get(API_URL, headers=HEADERS, params=params, timeout=30)
        data = response.json().get("response", [])
        recap = [{"match_id": m["fixture"]["id"], "status": m["fixture"]["status"]["short"]} for m in data]
        return jsonify({"count": len(recap), "data": recap, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
