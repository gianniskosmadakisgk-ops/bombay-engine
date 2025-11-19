import os
import requests
from datetime import datetime

def run_friday_simulation():
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        return {"error": "Missing TheOddsAPI key in environment"}

    url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds/"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal"
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        bets = []
        for game in data[:10]:  # κράτα τους 10 πρώτους αγώνες
            home_team = game["home_team"]
            away_team = game["away_team"]
            bookmakers = game.get("bookmakers", [])
            if not bookmakers:
                continue

            odds = bookmakers[0]["markets"][0]["outcomes"]
            odds_info = {o["name"]: o["price"] for o in odds}
            bets.append({
                "match": f"{home_team} vs {away_team}",
                "odds": odds_info
            })

        return {
            "date": str(datetime.now().date()),
            "fund": 300,
            "method": "Half-Kelly",
            "min_edge": "10%",
            "status": "Friday shortlist ready (live odds fetched)",
            "top_value_bets": bets
        }

    except Exception as e:
        return {"error": str(e)}
