import os
import json
import requests
from datetime import datetime, timedelta

# === CONFIG ===
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

LEAGUES = [39, 140, 135, 78, 61]  # EPL, LaLiga, Serie A, Bundesliga, Ligue 1
DAYS_FORWARD = 3
REPORT_PATH = "logs/thursday_report_v1.json"

os.makedirs("logs", exist_ok=True)

# === DATE RANGE ===
today = datetime.utcnow()
date_from = today.strftime("%Y-%m-%d")
date_to = (today + timedelta(days=DAYS_FORWARD)).strftime("%Y-%m-%d")

print(f"📅 Fetching fixtures from {date_from} to {date_to}")

# === Fetch Fixtures from API-Sports ===
fixtures = []
for league_id in LEAGUES:
    url = f"{FOOTBALL_BASE_URL}/fixtures?league={league_id}&season=2025&from={date_from}&to={date_to}"
    headers = {"x-apisports-key": FOOTBALL_API_KEY}
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        data = res.json().get("response", [])
        fixtures.extend(data)
        print(f"✅ Found {len(data)} fixtures for league {league_id}")
    else:
        print(f"⚠️ League {league_id} fetch failed ({res.status_code})")

# === Fetch Odds from TheOddsAPI ===
print("🎯 Fetching odds from TheOddsAPI...")
odds_data = {}
odds_url = f"{ODDS_BASE_URL}/soccer/odds?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h,totals"
try:
    odds_res = requests.get(odds_url)
    if odds_res.status_code == 200:
        for match in odds_res.json():
            home = match.get("home_team")
            away = match.get("away_team")
            odds_data[f"{home} - {away}"] = match
except Exception as e:
    print(f"⚠️ Odds fetch error: {e}")

# === Combine & Score Fixtures ===
data_sample = []
for f in fixtures:
    teams = f["teams"]
    home = teams["home"]["name"]
    away = teams["away"]["name"]
    match_label = f"{home} - {away}"

    # Random fallback odds in case missing
    offered_draw = 3.2
    offered_over = 1.95

    if match_label in odds_data:
        markets = odds_data[match_label].get("bookmakers", [])[0].get("markets", [])
        for m in markets:
            if m["key"] == "h2h":
                outcomes = m["outcomes"]
                for o in outcomes:
                    if o["name"] == "Draw":
                        offered_draw = o["price"]
            elif m["key"] == "totals":
                outcomes = m["outcomes"]
                for o in outcomes:
                    if o["name"] == "Over 2.5":
                        offered_over = o["price"]

    # Fair odds simulation (temporary placeholder)
    fair_draw = round(offered_draw * 0.93, 2)
    fair_over = round(offered_over * 0.94, 2)

    # Scoring simulation (can refine later)
    score_draw = round((offered_draw / fair_draw) * 7.5, 2)
    score_over = round((offered_over / fair_over) * 7.5, 2)
    kelly_value = round(((offered_draw - fair_draw) / fair_draw) * 100, 2)

    data_sample.append({
        "match": match_label,
        "league": f["league"]["name"],
        "score_draw": score_draw,
        "score_over": score_over,
        "fair_draw": fair_draw,
        "fair_over": fair_over,
        "offered_draw": offered_draw,
        "offered_over": offered_over,
        "kelly_value%": kelly_value
    })

# === Save report ===
output = {
    "generated_at": datetime.utcnow().isoformat(),
    "fixtures_analyzed": len(data_sample),
    "data_sample": data_sample
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Thursday analysis complete — {len(data_sample)} fixtures analyzed.")
print(f"📝 Report saved at {REPORT_PATH}")
