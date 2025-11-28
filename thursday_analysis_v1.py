import os
import requests
from datetime import datetime, timedelta
import json

# === Config ===
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
API_URL = "https://v3.football.api-sports.io/fixtures"
HEADERS = {"x-apisports-key": FOOTBALL_API_KEY}

# === Multi-league setup ===
LEAGUES = [
    39,   # Premier League
    61,   # Ligue 1
    140,  # La Liga
    135,  # Serie A
    41,   # Championship
    71,   # Serie B
    62,   # Ligue 2
    94,   # Liga Portugal 2
    197,  # Swiss Super League
    88,   # Eredivisie
    144   # Jupiler Pro League
]

# === Date range: Fri–Mon ===
today = datetime.utcnow()
days_ahead = (4 - today.weekday()) % 7  # Friday
friday = today + timedelta(days=days_ahead)
monday = friday + timedelta(days=3)
date_from, date_to = friday.strftime("%Y-%m-%d"), monday.strftime("%Y-%m-%d")

# === Utility functions ===
def fair_odd_calc(fixture, outcome):
    if outcome == "draw":
        base = 3.00
    elif outcome == "home":
        base = 2.00
    else:
        base = 3.50
    return round(base, 2)

def score_draw_calc(f):
    score = 5 + (1 - abs(1.4 - 1.3)) + 0.8
    return round(min(score, 10), 2)

def score_over_calc(f):
    score = 5 + ((2.8 - 2.5) + 0.3)
    return round(min(score, 10), 2)

def classify_fixture(f):
    return "balanced"

# === Fetch fixtures ===
fixtures = []
for league_id in LEAGUES:
    params = {"league": league_id, "season": 2025, "from": date_from, "to": date_to}
    try:
        res = requests.get(API_URL, headers=HEADERS, params=params, timeout=10)
        data = res.json()
        if data and data.get("response"):
            for f in data["response"]:
                fixtures.append({
                    "match": f["teams"]["home"]["name"] + " - " + f["teams"]["away"]["name"],
                    "league": f["league"]["name"],
                    "country": f["league"]["country"],
                    "fair_1": fair_odd_calc(f, "home"),
                    "fair_x": fair_odd_calc(f, "draw"),
                    "fair_2": fair_odd_calc(f, "away"),
                    "fair_over": fair_odd_calc(f, "over"),
                    "score_draw": score_draw_calc(f),
                    "score_over": score_over_calc(f),
                    "category": classify_fixture(f)
                })
    except Exception as e:
        print(f"⚠️ Error fetching league {league_id}: {e}")

# === Save output ===
os.makedirs("logs", exist_ok=True)
report_path = os.path.join("logs", "thursday_report_v1.json")

output = {
    "status": "success",
    "count": len(fixtures),
    "range": {"from": date_from, "to": date_to},
    "data_sample": fixtures[:10]  # δείγμα για πιο μικρό JSON
}

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Thursday Analysis complete — saved {len(fixtures)} fixtures to {report_path}")
