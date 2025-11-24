import os
import requests
from datetime import datetime, timedelta
import json

# === Config ===
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
API_URL = "https://v3.football.api-sports.io/fixtures"
HEADERS = {"x-apisports-key": FOOTBALL_API_KEY}

# === Multi-league setup (Paid API) ===
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

# === Scoring functions ===
def fair_odd_calc(fixture, outcome):
    """Δίκαιες αποδόσεις (προβλεπόμενες πιθανότητες)"""
    if outcome == "draw":
        base = 3.00
    elif outcome == "home":
        base = 2.00
    else:
        base = 3.50
    return round(base, 2)

def score_draw_calc(f):
    """Υπολογισμός Score Ισοπαλίας"""
    score = 0
    league_draw_rate = 30  # placeholder %
    xg_diff = abs(1.4 - 1.3)
    team_form = 0.8
    score = 5 + (1 - xg_diff) + team_form
    return round(min(score, 10), 2)

def score_over_calc(f):
    """Υπολογισμός Score Over"""
    avg_xg = 2.8
    form_boost = 0.3
    score = 5 + (avg_xg - 2.5) + form_boost
    return round(min(score, 10), 2)

def classify_fixture(f):
    """Κατηγορία αγώνα"""
    return "balanced"

# === Pull fixtures ===
fixtures = []
for league_id in LEAGUES:
    params = {"league": league_id, "season": 2025, "from": date_from, "to": date_to}
    res = requests.get(API_URL, headers=HEADERS, params=params)
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

# === Save output ===
output = {
    "count": len(fixtures),
    "range": {"from": date_from, "to": date_to},
    "data_sample": fixtures,
    "status": "success"
}

os.makedirs("logs", exist_ok=True)
with open("logs/thursday_output.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(json.dumps(output, indent=2, ensure_ascii=False))
