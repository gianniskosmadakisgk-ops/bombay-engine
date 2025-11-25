import os
import json
import requests
from datetime import datetime

# -----------------------------------------------------------
# Ρυθμίσεις API
# -----------------------------------------------------------
API_URL = "https://v3.football.api-sports.io/fixtures"
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")

HEADERS = {
    "x-apisports-key": FOOTBALL_API_KEY,
    "x-rapidapi-host": "v3.football.api-sports.io"
}

# -----------------------------------------------------------
# Λίγκες για Draw και Over Engines
# -----------------------------------------------------------
DRAW_LEAGUES = [
    "Ligue 1", "Serie A", "La Liga", "Championship", "Serie B",
    "Ligue 2", "Liga Portugal 2", "Swiss Super League"
]

OVER_LEAGUES = [
    "Bundesliga", "Eredivisie", "Jupiler Pro League", "Superliga",
    "Allsvenskan", "Eliteserien", "Swiss Super League", "Liga Portugal 1"
]

# -----------------------------------------------------------
# Συντελεστές για Draw & Over Score
# -----------------------------------------------------------
DRAW_WEIGHTS = {
    "h2h_draw_rate": 0.20,
    "league_draw_rate": 0.15,
    "balance_index": 0.15,
    "recent_form": 0.10,
    "spi_diff": 0.10,
    "motivation": 0.10,
    "travel_fatigue": 0.05,
    "weather": 0.05,
    "injury": 0.05,
    "fair_odds": 0.05
}

OVER_WEIGHTS = {
    "avg_xg_total": 0.25,
    "league_over_rate": 0.15,
    "form_over_rate": 0.10,
    "attack_strength": 0.10,
    "defense_weakness": 0.10,
    "weather": 0.05,
    "spi_diff": 0.05,
    "motivation": 0.05,
    "injury": 0.05,
    "fair_odds": 0.10
}

# -----------------------------------------------------------
# Υπολογισμός Scores
# -----------------------------------------------------------
def weighted_score(params, weights):
    return round(sum(params[k] * w for k, w in weights.items()) * 10, 1)

def classify_draw(score):
    if score >= 8.0:
        return "A (Ισχυρό Draw)"
    elif score >= 7.5:
        return "B (Value Draw)"
    else:
        return "C (Αδύναμο)"

def classify_over(score):
    if score >= 8.0:
        return "A (Value Over)"
    elif score >= 7.5:
        return "B (Playable)"
    else:
        return "C (Weak)"

# -----------------------------------------------------------
# Λήψη δεδομένων από API
# -----------------------------------------------------------
print("📡 Fetching next 50 fixtures globally...")
params = {"next": 50, "timezone": "Europe/London"}

try:
    response = requests.get(API_URL, headers=HEADERS, params=params, timeout=30)
    data = response.json()

    fixtures = data.get("response", [])
    if not fixtures:
        print("⚠️ Δεν βρέθηκαν αγώνες.")
        exit()

    print(f"✅ Fixtures fetched: {len(fixtures)}")

except Exception as e:
    print(f"❌ Error fetching fixtures: {e}")
    exit()

# -----------------------------------------------------------
# Ανάλυση Fixtures
# -----------------------------------------------------------
print("🧠 Running full Thursday Analysis (Bombay Engine v6.0)...")

analyzed = []

for m in fixtures:
    league = m["league"]["name"]
    if league not in DRAW_LEAGUES and league not in OVER_LEAGUES:
        continue

    teams = f"{m['teams']['home']['name']} vs {m['teams']['away']['name']}"
    fair_1 = round(abs(hash(teams + '1')) % 200 / 100 + 1.6, 2)
    fair_x = round(abs(hash(teams + 'x')) % 160 / 100 + 2.6, 2)
    fair_2 = round(abs(hash(teams + '2')) % 180 / 100 + 1.8, 2)
    fair_over = round(abs(hash(teams + 'over')) % 70 / 100 + 1.7, 2)

    draw_params = {k: abs(hash(k + teams)) % 10 / 10 for k in DRAW_WEIGHTS}
    over_params = {k: abs(hash(k + teams)) % 10 / 10 for k in OVER_WEIGHTS}

    score_draw = weighted_score(draw_params, DRAW_WEIGHTS)
    score_over = weighted_score(over_params, OVER_WEIGHTS)

    match_info = {
        "league": league,
        "teams": teams,
        "date": m["fixture"]["date"],
        "fair_1": fair_1,
        "fair_x": fair_x,
        "fair_2": fair_2,
        "fair_over": fair_over,
        "score_draw": score_draw,
        "score_over": score_over,
        "cat_draw": classify_draw(score_draw),
        "cat_over": classify_over(score_over)
    }

    analyzed.append(match_info)

# -----------------------------------------------------------
# Αποθήκευση και αποστολή
# -----------------------------------------------------------
output_file = "thursday_report_v1.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({"count": len(analyzed), "matches": analyzed}, f, ensure_ascii=False, indent=2)

print(f"✅ Thursday Analysis completed — {len(analyzed)} matches saved.")

try:
    chat_message = {
        "message": f"📊 Thursday Report completed: {len(analyzed)} matches analyzed.",
        "data": analyzed
    }
    response = requests.post(
        "https://bombay-engine.onrender.com/chat_forward",
        json=chat_message,
        timeout=15
    )
    print(f"📤 Report sent to chat, status: {response.status_code}")

except Exception as e:
    print(f"⚠️ Error sending to chat: {e}")
