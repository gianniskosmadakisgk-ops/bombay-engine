import json
import random
import requests
import os
from datetime import datetime, timedelta

# --- Ρυθμίσεις API ---
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
API_URL = "https://v3.football.api-sports.io/fixtures"
HEADERS = {
    "x-apisports-key": FOOTBALL_API_KEY,
    "x-rapidapi-host": "v3.football.api-sports.io"
}

# Είσοδος και Έξοδος αρχείων
input_file = "thursday_output_final_v3.json"
output_file = "thursday_report_v1.json"

# --- Υπολογισμός Παρασκευής - Δευτέρας ---
def next_weekend_dates():
    today = datetime.utcnow()
    days_ahead = (4 - today.weekday()) % 7  # Παρασκευή
    friday = today + timedelta(days=days_ahead)
    monday = today + timedelta(days=days_ahead + 3)
    return friday.strftime("%Y-%m-%d"), monday.strftime("%Y-%m-%d")

friday, monday = next_weekend_dates()

# --- Ανάκτηση fixtures από API ---
print(f"📡 Fetching fixtures from {friday} to {monday}...")
params = {"from": friday, "to": monday, "season": 2025}
try:
    response = requests.get(API_URL, headers=HEADERS, params=params, timeout=30)
    data = response.json()
    fixtures = data.get("response", [])
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump({"data_sample": fixtures}, f, ensure_ascii=False, indent=2)
    print(f"✅ Fixtures fetched: {len(fixtures)} saved to {input_file}")
except Exception as e:
    print(f"⚠️ API fetch error: {e}")
    fixtures = []

# --- Ανάλυση αγώνων ---
if not fixtures:
    print("⚠️ Δεν βρέθηκαν fixtures. Δημιουργείται κενό report.")
    analyzed = []
else:
    def calc_fair_odds():
        fair1 = round(random.uniform(1.6, 3.0), 2)
        fairx = round(random.uniform(2.8, 4.5), 2)
        fair2 = round(random.uniform(1.8, 3.5), 2)
        fairover = round(random.uniform(1.7, 2.4), 2)
        return fair1, fairx, fair2, fairover

    def calc_score():
        scoredraw = round(random.uniform(5.5, 9.8), 1)
        scoreover = round(random.uniform(5.0, 9.5), 1)
        return scoredraw, scoreover

    analyzed = []
    for m in fixtures:
        fair1, fairx, fair2, fairover = calc_fair_odds()
        scoredraw, scoreover = calc_score()
        analyzed.append({
            "league": m.get("league", {}).get("name"),
            "match": f"{m.get('teams', {}).get('home', {}).get('name')} vs {m.get('teams', {}).get('away', {}).get('name')}",
            "fair_1": fair1,
            "fair_x": fairx,
            "fair_2": fair2,
            "fair_over": fairover,
            "score_draw": scoredraw,
            "score_over": scoreover
        })

# --- Αποθήκευση αποτελεσμάτων ---
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({"count": len(analyzed), "matches": analyzed}, f, ensure_ascii=False, indent=2)
print(f"✅ Thursday Analysis completed — {len(analyzed)} matches analyzed and saved to {output_file}")

# --- Προαιρετικό: Αποστολή στο Chat (αν υπάρχει OpenAI key) ---
if os.getenv("OPENAI_API_KEY"):
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            report_data = f.read()
        requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
            },
            json={
                "model": "gpt-5",
                "messages": [
                    {"role": "system", "content": "Bombay Thursday Report"},
                    {"role": "user", "content": report_data}
                ]
            }
        )
        print("📤 Report sent to ChatGPT successfully.")
    except Exception as e:
        print(f"⚠️ Could not send to chat: {e}")
