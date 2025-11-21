from flask import Flask, jsonify
import requests
from datetime import datetime, timedelta

app = Flask(__name__)

API_KEY = "0e0464506d8f342bb0a2ee20ef6cad79"
BASE_URL = "https://v3.football.api-sports.io/fixtures"

LEAGUES = {
    "Premier League": 39,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61,
    "Ligue 2": 62,
    "Serie B": 136,
    "Championship": 40,
    "Eredivisie": 88,
    "Jupiler Pro League": 144,
    "Liga Portugal 2": 69,
    "Swiss Super League": 207
}

@app.route('/run_thursday_analysis', methods=['GET'])
def run_thursday_analysis():
    today = datetime.now()
    start_date = today.strftime("%Y-%m-%d")
    end_date = (today + timedelta(days=10)).strftime("%Y-%m-%d")

    fixtures_data = {}
    debug_log = []

    for league_name, league_id in LEAGUES.items():
        url = f"{BASE_URL}?league={league_id}&season=2025&from={start_date}&to={end_date}"
        headers = {
            "x-apisports-key": API_KEY,
            "Accept": "application/json"
        }

        response = requests.get(url, headers=headers)
        json_data = response.json()

        fixtures_data[league_name] = json_data.get("response", [])
        debug_log.append({
            "league": league_name,
            "url": url,
            "status_code": response.status_code,
            "results": len(json_data.get("response", [])),
            "error": json_data.get("errors")
        })

    return jsonify({
        "debug_log": debug_log,
        "fixtures": fixtures_data,
        "fixtures_count": sum(len(v) for v in fixtures_data.values()),
        "status": "Thursday Analysis complete",
        "timestamp": datetime.utcnow().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
