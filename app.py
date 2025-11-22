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

    all_fixtures = []
    debug_log = []

    for league_name, league_id in LEAGUES.items():
        url = f"{BASE_URL}?league={league_id}&season=2025&from={start_date}&to={end_date}"
        headers = {
            "x-apisports-key": API_KEY,
            "Accept": "application/json"
        }

        response = requests.get(url, headers=headers)
        data = response.json()

        fixtures = data.get("response", [])
        for fx in fixtures:
            fixture_info = fx.get("fixture", {})
            teams = fx.get("teams", {})
            home_team = teams.get("home", {}).get("name")
            away_team = teams.get("away", {}).get("name")
            date = fixture_info.get("date")

            # Dummy fair odds until we apply model
            fair_1 = round(1.6 + 1.0 * (hash(home_team) % 50) / 100, 2)
            fair_x = round(3.0 + 0.5 * (hash(away_team) % 50) / 100, 2)
            fair_2 = round(2.0 + 1.0 * (hash(home_team + away_team) % 50) / 100, 2)
            fair_over = round(1.7 + 0.4 * (hash(date) % 50) / 100, 2)

            all_fixtures.append({
                "League": league_name,
                "Date": date,
                "Match": f"{home_team} - {away_team}",
                "Fair_1": fair_1,
                "Fair_X": fair_x,
                "Fair_2": fair_2,
                "Fair_Over": fair_over
            })

        debug_log.append({
            "League": league_name,
            "status_code": response.status_code,
            "results": len(fixtures),
            "url": url
        })

    return jsonify({
        "fixtures": all_fixtures,
        "fixtures_count": len(all_fixtures),
        "status": "Thursday Analysis complete",
        "timestamp": datetime.utcnow().isoformat(),
        "debug_log": debug_log
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
