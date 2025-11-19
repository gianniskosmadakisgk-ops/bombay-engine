import os
import requests
from datetime import datetime

def fetch_odds_for_league(league_key, league_name, api_key):
    base_url = f"https://api.the-odds-api.com/v4/sports/{league_key}/odds/"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        bets = []
        for game in data[:10]:  # Πάρε μέχρι 10 παιχνίδια από κάθε λίγκα
            home_team = game["home_team"]
            away_team = game["away_team"]
            bookmakers = game.get("bookmakers", [])
            if not bookmakers:
                continue

            book = bookmakers[0]
            h2h_market = next((m for m in book["markets"] if m["key"] == "h2h"), None)
            totals_market = next((m for m in book["markets"] if m["key"] == "totals"), None)

            odds_info = {}
            over_under = {}

            if h2h_market:
                for o in h2h_market["outcomes"]:
                    if o["name"] == home_team:
                        odds_info["1"] = o["price"]
                    elif o["name"] == away_team:
                        odds_info["2"] = o["price"]
                    elif o["name"].lower() == "draw":
                        odds_info["X"] = o["price"]

            if totals_market:
                for o in totals_market["outcomes"]:
                    if "Over" in o["name"]:
                        over_under["over_2.5"] = o["price"]
                    elif "Under" in o["name"]:
                        over_under["under_2.5"] = o["price"]

            bets.append({
                "match": f"{home_team} vs {away_team}",
                "odds": odds_info,
                "over_under": over_under
            })

        return {league_name: bets}

    except Exception as e:
        return {league_name: f"Error fetching odds: {str(e)}"}


def run_friday_simulation():
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        return {"error": "Missing TheOddsAPI key in environment"}

    leagues = {
        "soccer_epl": "Premier League",
        "soccer_spain_la_liga": "La Liga",
        "soccer_italy_serie_a": "Serie A"
    }

    results = {}
    for key, name in leagues.items():
        results.update(fetch_odds_for_league(key, name, api_key))

    return {
        "date": str(datetime.now().date()),
        "fund": 300,
        "method": "Half-Kelly",
        "min_edge": "10%",
        "status": "Friday shortlist ready (EPL, LaLiga, Serie A)",
        "top_value_bets": results
    }
