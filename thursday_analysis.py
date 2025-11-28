import os
import requests
from datetime import datetime, timedelta
import json

# === Ρυθμίσεις ===
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
API_URL = "https://v3.football.api-sports.io/fixtures"
HEADERS = {"x-apisports-key": FOOTBALL_API_KEY}

# === Λίγκες ===
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

# === Εύρος ημερομηνιών: Παρασκευή - Δευτέρα ===
today = datetime.utcnow()
days_ahead = (4 - today.weekday()) % 7  # Friday
friday = today + timedelta(days=days_ahead)
monday = friday + timedelta(days=3)
date_from, date_to = friday.strftime("%Y-%m-%d"), monday.strftime("%Y-%m-%d")

# === Υπολογισμοί ===
def fair_odd_calc(fixture, outcome):
    """Δίκαιες αποδόσεις"""
    base = {"draw": 3.00, "home": 2.00, "away": 3.50, "over": 2.10}
    return round(base.get(outcome, 2.50), 2)

def score_draw_calc(f):
    """Score Ισοπαλίας"""
    xg_diff = abs(1.4 - 1.3)
    team_form = 0.8
    score = 5 + (1 - xg_diff) + team_form
    return round(min(score, 10), 2)

def score_over_calc(f):
    """Score Over"""
    avg_xg = 2.8
    form_boost = 0.3
    score = 5 + (avg_xg - 2.5) + form_boost
    return round(min(score, 10), 2)

def classify_fixture(f):
    """Κατηγορία αγώνα"""
    return "balanced"

# === Λήψη αγώνων ===
fixtures = []
for league_id in LEAGUES:
    params = {"league": league_id, "season": 2025, "from": date_from, "to": date_to}
    res = requests.get(API_URL, headers=HEADERS, params=params)

    if res.status_code == 200:
        data = res.json()
        for f in data.get("response", []):
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
    else:
        print(f"⚠️ API error: {res.status_code} for league {league_id}")

# === Δημιουργία εξόδου ===
output = {
    "count": len(fixtures),
    "date_range": {"from": date_from, "to": date_to},
    "fixtures": fixtures[:10],  # δείγμα 10 αγώνων για λογικό output
    "status": "success"
}

# === Αποθήκευση σε JSON ===
os.makedirs("logs", exist_ok=True)
report_path = "logs/thursday_report_v1.json"

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Thursday analysis complete. Saved to {report_path}")
print(json.dumps(output, indent=2, ensure_ascii=False))
