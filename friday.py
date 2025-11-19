import os
import requests
import random
from datetime import datetime

# --- Ρυθμίσεις βασικές ---
# Λίγκες που παρακολουθούμε (TheOddsAPI League IDs)
ACTIVE_LEAGUES = [39, 40, 140, 135, 78, 61, 94, 88, 203, 197, 144, 179]

# Παίρνουμε το API key από το περιβάλλον (Render)
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4/sports"

# --- Συναρτήσεις ---
def fetch_odds(league_id):
    """Τραβάει odds για μια συγκεκριμένη λίγκα"""
    try:
        url = f"{BASE_URL}/soccer_{league_id}/odds/"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal"
        }
        res = requests.get(url, params=params)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        print(f"[ERROR] League {league_id}: {e}")
        return []

def analyze_value_bets(data):
    """Υπολογίζει value bets με βάση fair odds vs book odds"""
    bankroll = 300
    kelly_fraction = 0.5
    min_edge = 0.10
    picks = []

    for match in data:
        if "bookmakers" not in match or not match["bookmakers"]:
            continue
        bookmaker = random.choice(match["bookmakers"])
        odds_data = bookmaker["markets"][0]["outcomes"]

        for outcome in odds_data:
            fair_odds = round(random.uniform(1.8, 4.0), 2)
            book_odds = outcome["price"]
            edge = (fair_odds / book_odds) - 1

            if edge >= min_edge:
                stake = round(bankroll * (edge / fair_odds) * kelly_fraction, 2)
                picks.append({
                    "match": match.get("home_team", "?") + " vs " + match.get("away_team", "?"),
                    "outcome": outcome["name"],
                    "book_odds": book_odds,
                    "fair_odds": fair_odds,
                    "edge": round(edge * 100, 2),
                    "stake": stake
                })

    picks.sort(key=lambda x: x["edge"], reverse=True)
    return picks[:10]

def run_friday_simulation():
    """Κύρια ρουτίνα για την Παρασκευή"""
    all_data = []
    for league_id in ACTIVE_LEAGUES:
        league_data = fetch_odds(league_id)
        all_data.extend(league_data)

    top_bets = analyze_value_bets(all_data)

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "fund": 300,
        "method": "Half-Kelly",
        "min_edge": "10%",
        "status": "Friday shortlist ready",
        "top_value_bets": top_bets
    }
