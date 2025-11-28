import os
import json
import requests
from datetime import datetime, timedelta

# === CONFIG ===
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
API_FIXTURES = "https://v3.football.api-sports.io/fixtures"
API_ODDS = "https://api.the-odds-api.com/v4/sports/soccer/odds"
HEADERS = {"x-apisports-key": FOOTBALL_API_KEY}

DIFF_THRESHOLD = 0.10  # 10% minimum value
BANKROLL = 200  # Euro fund

# === LEAGUES ===
LEAGUES = [39, 61, 140, 135, 41, 71, 62, 94, 197, 88, 144]

# === Date Range ===
today = datetime.utcnow()
friday = today + timedelta(days=(4 - today.weekday()) % 7)
monday = friday + timedelta(days=3)
date_from, date_to = friday.strftime("%Y-%m-%d"), monday.strftime("%Y-%m-%d")

# === Utility functions ===
def fair_odd_calc(base):
    return round(base * 0.97, 2)

def kelly_fraction(p, b):
    """Returns Kelly fraction"""
    return max(((p * (b + 1)) - 1) / b, 0)

def implied_prob(odd):
    return 1 / odd if odd > 0 else 0

# === Fetch fixtures ===
fixtures = []
for lid in LEAGUES:
    params = {"league": lid, "season": 2025, "from": date_from, "to": date_to}
    r = requests.get(API_FIXTURES, headers=HEADERS, params=params, timeout=15)
    data = r.json()
    if not data.get("response"):
        continue
    for f in data["response"]:
        home = f["teams"]["home"]["name"]
        away = f["teams"]["away"]["name"]
        match = f"{home} - {away}"
        fixtures.append({
            "match": match,
            "league": f["league"]["name"],
            "country": f["league"]["country"],
            "date": f["fixture"]["date"]
        })

# === Fetch real odds ===
real_odds = {}
for f in fixtures:
    q = requests.get(
        f"{API_ODDS}?apiKey={ODDS_API_KEY}&regions=eu&markets=h2h,totals&oddsFormat=decimal",
        timeout=20
    )
    data = q.json()
    for item in data:
        if "bookmakers" not in item:
            continue
        match_name = item["home_team"] + " - " + item["away_team"]
        try:
            outcomes = item["bookmakers"][0]["markets"][0]["outcomes"]
            real_odds[match_name] = {
                "Home": outcomes[0]["price"],
                "Draw": outcomes[1]["price"],
                "Away": outcomes[2]["price"]
            }
        except Exception:
            continue

# === Fair vs Book comparison & Kelly ===
fraction_kelly = []
for f in fixtures:
    match = f["match"]
    offered = real_odds.get(match, {})

    for market, odd in offered.items():
        fair = fair_odd_calc(odd * 0.9)
        diff = (odd - fair) / fair

        if diff < DIFF_THRESHOLD:
            continue  # Skip <10% diff

        prob = implied_prob(fair)
        kelly_f = kelly_fraction(prob, odd - 1)
        stake = round(kelly_f * BANKROLL, 2)

        fraction_kelly.append({
            "match": match,
            "market": market.lower(),
            "fair": fair,
            "offered": odd,
            "diff%": round(diff * 100, 2),
            "kelly_f": round(kelly_f, 3),
            "stake (€)": stake
        })

# === Sort by diff% descending ===
fraction_kelly.sort(key=lambda x: x["diff%"], reverse=True)

# === Final output ===
os.makedirs("logs", exist_ok=True)
output_path = "logs/friday_shortlist_v1.json"

report = {
    "status": "success",
    "count": len(fixtures),
    "fraction_kelly": fraction_kelly,
    "range": {"from": date_from, "to": date_to}
}

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"✅ Friday Shortlist complete — {len(fraction_kelly)} value picks saved to {output_path}")
