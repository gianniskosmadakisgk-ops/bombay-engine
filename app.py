from flask import Flask, jsonify
import requests

app = Flask(__name__)

API_KEY = "0e0464506d8f342bb0a2ee20ef6cad79"

# Λίγκες που μας ενδιαφέρουν (μόνο οι κύριες για αρχή)
LEAGUES = {
    "Ligue 1": 61,
    "Serie A": 135,
    "La Liga": 140,
    "Premier League": 39,
    "Bundesliga": 78,
    "Eredivisie": 88,
    "Ligue 2": 62,
    "Serie B": 136,
    "Championship": 40
}

@app.route("/")
def home():
    return jsonify({"message": "Bombay Engine Active"})

@app.route("/run_thursday_analysis")
def run_thursday_analysis():
    results = {}
    for league_name, league_id in LEAGUES.items():
        url = f"https://v3.football.api-sports.io/fixtures?league={league_id}&season=2024&next=10"
        headers = {"x-apisports-key": API_KEY}
        response = requests.get(url, headers=headers)
        data = response.json()

        league_matches = []
        for item in data.get("response", []):
            match = item.get("teams", {})
            home = match.get("home", {}).get("name")
            away = match.get("away", {}).get("name")
            if not home or not away:
                continue

            # Fair odds (προσεγγιστικά για την επίδειξη)
            fair_1 = round(2.0 + hash(home) % 50 / 100, 2)
            fair_x = round(3.0 + hash(away) % 50 / 100, 2)
            fair_2 = round(2.2 + (hash(home + away) % 40) / 100, 2)
            fair_over = round(1.7 + (hash(away + home) % 20) / 100, 2)
            score_draw = round((hash(home) % 10) + 1, 1)
            score_over = round((hash(away) % 10) + 1, 1)
            category = "Top Tier" if score_draw >= 7.5 or score_over >= 7.5 else "Average"

            league_matches.append({
                "home": home,
                "away": away,
                "fair_1": fair_1,
                "fair_x": fair_x,
                "fair_2": fair_2,
                "fair_over": fair_over,
                "score_draw": score_draw,
                "score_over": score_over,
                "category": category
            })

        results[league_name] = league_matches

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
