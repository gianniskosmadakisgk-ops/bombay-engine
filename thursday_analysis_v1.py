import os
import json
import requests
import yaml
from datetime import datetime, timedelta
from pathlib import Path

# === CONFIG ===
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

LEAGUES = [39, 140, 135, 78, 61]  # EPL, LaLiga, Serie A, Bundesliga, Ligue 1
DAYS_FORWARD = 3
REPORT_PATH = "logs/thursday_report_v1.json"

os.makedirs("logs", exist_ok=True)

# === Load Core Config (optional) ===
core_path = Path(__file__).resolve().parent / "core"
engines_path = Path(__file__).resolve().parent / "engines"

try:
    with open(core_path / "bombay_rules_v4.yaml", "r", encoding="utf-8") as f:
        bombay_rules = yaml.safe_load(f)
    print("✅ Loaded Bombay Rules (v4)")

    with open(engines_path / "bookmaker_logic.yaml", "r", encoding="utf-8") as f:
        bookmaker_logic = yaml.safe_load(f)
    print("✅ Loaded Bookmaker Logic")

except Exception as e:
    print(f"⚠️ Skipped loading configs: {e}")

# === DATE RANGE ===
today = datetime.utcnow()
date_from = today.strftime("%Y-%m-%d")
date_to = (today + timedelta(days=DAYS_FORWARD)).strftime("%Y-%m-%d")

print(f"📅 Fetching fixtures from {date_from} to {date_to}")

# === Fetch Fixtures from API-Sports ===
fixtures = []
for league_id in LEAGUES:
    try:
        url = f"{FOOTBALL_BASE_URL}/fixtures?league={league_id}&season=2025&from={date_from}&to={date_to}"
        headers = {"x-apisports-key": FOOTBALL_API_KEY}
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json().get("response", [])
            fixtures.extend(data)
            print(f"✅ Found {len(data)} fixtures for league {league_id}")
        else:
            print(f"⚠️ League {league_id} fetch failed ({res.status_code})")
    except Exception as e:
        print(f"⚠️ Error fetching league {league_id}: {e}")

# === Fetch Odds from TheOddsAPI ===
print("🎯 Fetching odds from TheOddsAPI...")
odds_data = {}
try:
    odds_url = f"{ODDS_BASE_URL}/soccer/odds?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h,totals"
    odds_res = requests.get(odds_url, timeout=10)
    if odds_res.status_code == 200:
        for match in odds_res.json():
            home = match.get("home_team")
            away = match.get("away_team")
            odds_data[f"{home} - {away}"] = match
    else:
        print(f"⚠️ Odds API returned {odds_res.status_code}")
except Exception as e:
    print(f"⚠️ Odds fetch error: {e}")

# === Combine Fixtures + Odds and Compute Scores ===
processed = []
for f in fixtures:
    try:
        league = f["league"]["name"]
        home = f["teams"]["home"]["name"]
        away = f["teams"]["away"]["name"]
        match_label = f"{home} - {away}"

        # Default odds (σε περίπτωση που δεν βρεθεί το ματς)
        odds_x = 3.2
        odds_over = 1.95

        if match_label in odds_data:
            for b in odds_data[match_label].get("bookmakers", []):
                for m in b.get("markets", []):
                    if m["key"] == "h2h":
                        for o in m["outcomes"]:
                            if o["name"] == "Draw":
                                odds_x = o["price"]
                    elif m["key"] == "totals":
                        for o in m["outcomes"]:
                            if o["name"] == "Over 2.5":
                                odds_over = o["price"]

        fair_x = round(odds_x * 0.93, 2)
        fair_over = round(odds_over * 0.94, 2)
        score_draw = round((odds_x / fair_x) * 7.5, 2)
        score_over = round((odds_over / fair_over) * 7.5, 2)

        processed.append({
            "league": league,
            "match": match_label,
            "fair_x": fair_x,
            "odds_x": odds_x,
            "fair_over": fair_over,
            "odds_over": odds_over,
            "score_draw": score_draw,
            "score_over": score_over
        })
    except Exception as e:
        print(f"⚠️ Error processing fixture: {e}")

# === Save JSON report ===
output = {
    "timestamp": datetime.utcnow().isoformat(),
    "fixtures": processed,
    "meta": {
        "fixtures_analyzed": len(processed),
        "source": "Thursday Analysis v1"
    }
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Thursday analysis complete — {len(processed)} fixtures analyzed.")
print(f"📝 Report saved at {REPORT_PATH}")
