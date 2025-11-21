from flask import Flask, jsonify
import requests
from datetime import datetime, timedelta

app = Flask(__name__)

API_KEY = "0e0464506d8f342bb0a2ee20ef6cad79"

LEAGUES = {
    "🇫🇷 Ligue 1": 61,
    "🇫🇷 Ligue 2": 62,
    "🇮🇹 Serie A": 135,
    "🇮🇹 Serie B": 136,
    "🇪🇸 La Liga": 140,
    "🇵🇹 Liga Portugal 2": 94,
    "🇩🇪 Bundesliga": 78,
    "🇬🇧 Premier League": 39,
    "🇬🇧 Championship": 40,
    "🇳🇱 Eredivisie": 88,
    "🇧🇪 Jupiler Pro League": 144,
    "🇨🇭 Swiss Super League": 207
}

@app.route("/")
def home():
    return jsonify({"message": "Bombay Engine Active"})

@app.route("/run_thursday_analysis")
def run_thursday_analysis():
    today = datetime.utcnow()
    from_date = today.strftime("%Y-%m-%d")
    to_date = (today + timedelta(days=10)).strftime("%Y-%m-%d")

    results = {}
    total_fixtures = 0
    debug_log = []

    for league_name, league_id in LEAGUES.items():
        url = f"https://v3.football.api-sports.io/fixtures?league={league_id}&season=2024&from={from_date}&to={to_date}"
        headers = {"x-apisports-key": API_KEY}
        response = requests.get(url, headers=headers)
        data = response.json()

        league_matches = []
        for item in data.get("response", []):
            match_info = item.get("teams", {})
            home = match_info.get("home", {}).get("name")
            away = match_info.get("away", {}).get("name")
            if not home or not away:
                continue

            fair_1 = round(2.0 + hash(home) % 50 / 100, 2)
            fair_x = round(3.0 + hash(away) % 50 / 100, 2)
            fair_2 = round(2.2 + (hash(home + away) % 40) / 100, 2)
            fair_over = round(1.7 + (hash(away + home) % 20) / 100, 2)
            score_draw = round((hash(home) % 10) + 1, 1)
            score_over = round((hash(away) % 10) + 1, 1)

            if score_draw >= 7.5 or score_over >= 7.5:
                category = "A"
            elif score_draw >= 6.0 or score_over >= 6.0:
                category = "B"
            else:
                category = "C"

            league_matches.append({
                "Match": f"{home} - {away}",
                "Fair_1": fair_1,
                "Fair_X": fair_x,
                "Fair_2": fair_2,
                "Fair_Over": fair_over,
                "Score_Draw": score_draw,
                "Score_Over": score_over,
                "Category": category
            })

        results[league_name] = league_matches
        total_fixtures += len(league_matches)
        debug_log.append({league_name: len(league_matches)})

    return jsonify({
        "fixtures": results,
        "fixtures_count": total_fixtures,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "Thursday Analysis complete",
        "debug_log": debug_log
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
