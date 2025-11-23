import datetime
import requests
from flask import Flask, jsonify

app = Flask(__name__)

API_URL = "https://v3.football.api-sports.io/fixtures"
HEADERS = {
    "x-api-key": "0e0464506d8f342bb0a2ee20ef6cad79",
    "x-apisports-host": "v3.football.api-sports.io"
}

LEAGUE_IDS = {
    "Ligue 1": 61,
    "Serie A": 135,
    "La Liga": 140,
    "Championship": 40,
    "Serie B": 136,
    "Ligue 2": 62,
    "Liga Portugal 2": 88,
    "Swiss Super League": 207,
    "Bundesliga": 78,
    "Eredivisie": 88,
    "Jupiler Pro League": 144
}

def fetch_fixtures(league_id):
    today = datetime.date.today()
    next_friday = today + datetime.timedelta((4 - today.weekday()) % 7)
    monday_after = next_friday + datetime.timedelta(days=3)

    from_date = next_friday.strftime("%Y-%m-%d")
    to_date = monday_after.strftime("%Y-%m-%d")

    params = {"league": league_id, "season": 2024, "from": from_date, "to": to_date}
    response = requests.get(API_URL, headers=HEADERS, params=params)
    data = response.json()

    count = len(data.get("response", []))
    print(f"✅ League {league_id}: Found {count} fixtures ({from_date} → {to_date})")

    return data.get("response", [])

@app.route("/run_thursday_analysis")
def run_thursday_analysis():
    all_fixtures = []
    total_count = 0

    for league_name, league_id in LEAGUE_IDS.items():
        fixtures = fetch_fixtures(league_id)
        if fixtures:
            for f in fixtures:
                fixture_info = {
                    "league": league_name,
                    "match": f"{f['teams']['home']['name']} – {f['teams']['away']['name']}",
                    "date": f['fixture']['date'],
                    "status": f['fixture']['status']['short']
                }
                all_fixtures.append(fixture_info)
            total_count += len(fixtures)

    print(f"📊 Total Fixtures Collected: {total_count}")
    return jsonify({"count": total_count, "data": all_fixtures, "status": "success"})

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "routes": ["/run_thursday_analysis"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
