import os
import requests
from datetime import datetime, timedelta
import json
import random

# === CONFIG ===
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
API_URL = "https://v3.football.api-sports.io/fixtures"
HEADERS = {"x-apisports-key": FOOTBALL_API_KEY}

# === LEAGUES ===
LEAGUES = [
    39,  # Premier League
    61,  # Ligue 1
    140, # La Liga
    135, # Serie A
    41,  # Championship
    71,  # Serie B
    62,  # Ligue 2
    94,  # Liga Portugal 2
    197, # Swiss Super League
    88,  # Eredivisie
    144  # Jupiler Pro League
]

# === DATE RANGE (Fri–Mon) ===
today = datetime.utcnow()
days_ahead = (4 - today.weekday()) % 7  # Friday
friday = today + timedelta(days=days_ahead)
monday = friday + timedelta(days=3)
date_from, date_to = friday.strftime("%Y-%m-%d"), monday.strftime("%Y-%m-%d")

print(f"📅 Fetching fixtures from {date_from} to {date_to}")
print(f"🔑 Using API key: {FOOTBALL_API_KEY[:5]}***")

# === ENGINE HELPERS ===
def draw_engine(fixture):
    """Υπολογισμός score πιθανότητας ισοπαλίας (0–10)"""
    base = 4.0

    # Dummy historical & form data (placeholder μέχρι να συνδεθεί DB)
    league_draw_rate = random.uniform(25, 35)
    h2h_draws = random.uniform(0, 40)
    avg_xg_diff = random.uniform(0.1, 0.6)
    form_delta = random.uniform(0, 1)
    weather_bad = random.choice([0, 0.5])
    motivation = random.choice([0, 1.0])
    offered_vs_fair = random.choice([0, 0.5])  # +0.5 boost if offered > fair +10%

    score = (
        base
        + (league_draw_rate - 25) * 0.05
        + (h2h_draws / 20)
        + (1 - avg_xg_diff) * 2
        + (1 - form_delta)
        + weather_bad
        + motivation
        + offered_vs_fair
    )

    return round(min(score, 10), 2)


def over_engine(fixture):
    """Υπολογισμός score πιθανότητας Over 2.5 (0–10)"""
    base = 4.5

    avg_xg_total = random.uniform(2.3, 3.2)
    avg_goals_last5 = random.uniform(2.2, 3.5)
    league_over_rate = random.uniform(45, 70)
    h2h_over_rate = random.uniform(40, 75)
    weather_good = random.choice([0, 0.3])
    team_form = random.choice([0, 0.5])
    offered_vs_fair = random.choice([0, 0.5])

    score = (
        base
        + (avg_xg_total - 2.5) * 3
        + (avg_goals_last5 - 2.5)
        + (league_over_rate - 50) * 0.05
        + (h2h_over_rate - 50) * 0.04
        + weather_good
        + team_form
        + offered_vs_fair
    )

    return round(min(score, 10), 2)


# === FETCH FIXTURES ===
fixtures = []
for league_id in LEAGUES:
    params = {"league": league_id, "season": 2025, "from": date_from, "to": date_to}
    try:
        res = requests.get(API_URL, headers=HEADERS, params=params, timeout=10)
        data = res.json()
        if data and data.get("response"):
            print(f"✅ Found {len(data['response'])} fixtures for league {league_id}")
            for f in data["response"]:
                fixture_info = {
                    "match": f["teams"]["home"]["name"] + " - " + f["teams"]["away"]["name"],
                    "league": f["league"]["name"],
                    "country": f["league"]["country"],
                }

                # Calculate engine scores
                fixture_info["score_draw"] = draw_engine(f)
                fixture_info["score_over"] = over_engine(f)

                # Fair odds (placeholders)
                fixture_info["fair_1"] = 2.00
                fixture_info["fair_x"] = 3.10
                fixture_info["fair_2"] = 3.50
                fixture_info["fair_over"] = 1.90

                fixtures.append(fixture_info)
        else:
            print(f"⚠️ No fixtures found for league {league_id}")
    except Exception as e:
        print(f"⚠️ Error fetching league {league_id}: {e}")


# === FILTER RESULTS ===
draw_candidates = [f for f in fixtures if f["score_draw"] >= 7.0]
over_candidates = [f for f in fixtures if f["score_over"] >= 7.0]

summary = {
    "total_fixtures": len(fixtures),
    "draw_candidates": len(draw_candidates),
    "over_candidates": len(over_candidates),
}

# === OUTPUT JSON ===
output = {
    "timestamp": datetime.utcnow().isoformat(),
    "range": {"from": date_from, "to": date_to},
    "summary": summary,
    "draw_candidates": draw_candidates,
    "over_candidates": over_candidates,
}

os.makedirs("logs", exist_ok=True)
report_path = os.path.join("logs", "thursday_report_v1.json")

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Thursday Analysis complete — saved {len(fixtures)} fixtures to {report_path}")
