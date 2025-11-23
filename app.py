from flask import Flask, jsonify
import requests
import pandas as pd
import datetime

app = Flask(__name__)

API_KEY = "YOUR_API_KEY"  # βάλε εδώ το δικό σου API key
API_URL = "https://api-football-v1.p.rapidapi.com/v3/fixtures"
HEADERS = {"x-rapidapi-host": "api-football-v1.p.rapidapi.com", "x-rapidapi-key": API_KEY}

# 🔒 Locked leagues για Draw + Over analysis
LEAGUES = [39, 140, 135, 78, 61, 62, 94, 169, 88, 144]

def fetch_fixtures(league_id):
    today = datetime.date.today()
    from_date = today.strftime("%Y-%m-%d")
    to_date = (today + datetime.timedelta(days=5)).strftime("%Y-%m-%d")

    params = {"league": league_id, "season": 2024, "from": from_date, "to": to_date}
    response = requests.get(API_URL, headers=HEADERS, params=params)
    data = response.json()
    return data.get("response", [])

@app.route("/run_thursday_analysis", methods=["GET"])
def run_thursday_analysis():
    all_fixtures = []
    for league in LEAGUES:
        fixtures = fetch_fixtures(league)
        for match in fixtures:
            info = match["teams"]
            league_name = match["league"]["name"]
            all_fixtures.append({
                "league": league_name,
                "home": info["home"]["name"],
                "away": info["away"]["name"],
                "date": match["fixture"]["date"],
                "status": match["fixture"]["status"]["short"]
            })

    df = pd.DataFrame(all_fixtures)
    result = df.to_dict(orient="records")
    return jsonify({"status": "success", "count": len(result), "data": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
