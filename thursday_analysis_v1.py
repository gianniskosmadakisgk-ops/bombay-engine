import os
import json
import random
import requests

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
# Παίρνει τους επόμενους 50 αγώνες (σίγουρη επιστροφή δεδομένων)
# -----------------------------------------------------------
print("📡 Fetching next 50 fixtures globally...")

params = {
    "next": 50,
    "timezone": "Europe/London"
}

try:
    response = requests.get(API_URL, headers=HEADERS, params=params, timeout=30)
    data = response.json()

    if not data.get("response"):
        print("⚠️ Δεν βρέθηκαν αγώνες από το API.")
        with open("thursday_output_final_v3.json", "w", encoding="utf-8") as f:
            json.dump({"response": []}, f, ensure_ascii=False, indent=2)
        exit()

    with open("thursday_output_final_v3.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Fixtures fetched: {len(data['response'])} saved to thursday_output_final_v3.json")

except Exception as e:
    print(f"❌ Error fetching fixtures: {e}")
    exit()

# -----------------------------------------------------------
# Ανάλυση αγώνων (τυχαία fair odds για δοκιμή)
# -----------------------------------------------------------
print("🧠 Running Thursday Analysis...")

fixtures = data.get("response", [])
if not fixtures:
    print("⚠️ Δεν υπάρχουν fixtures για ανάλυση.")
    exit()

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

    match_info = {
        "league": m["league"]["name"] if "league" in m else "Unknown",
        "teams": f"{m['teams']['home']['name']} vs {m['teams']['away']['name']}" if "teams" in m else "Unknown",
        "date": m["fixture"]["date"] if "fixture" in m else "N/A",
        "fair_1": fair1,
        "fair_x": fairx,
        "fair_2": fair2,
        "fair_over": fairover,
        "score_draw": scoredraw,
        "score_over": scoreover
    }

    analyzed.append(match_info)

# -----------------------------------------------------------
# Αποθήκευση αποτελεσμάτων
# -----------------------------------------------------------
output_file = "thursday_report_v1.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({"count": len(analyzed), "matches": analyzed}, f, ensure_ascii=False, indent=2)

print(f"✅ Thursday Analysis completed — {len(analyzed)} matches analyzed and saved to {output_file}")

# -----------------------------------------------------------
# Αποστολή του report στο Chat
# -----------------------------------------------------------
try:
    with open(output_file, "r", encoding="utf-8") as f:
        report_data = json.load(f)

    chat_message = {
        "message": f"📊 Thursday Report ({len(report_data.get('matches', []))} matches) sent successfully.",
        "data": report_data
    }

    response = requests.post(
        "https://bombay-engine.onrender.com/chat_forward",
        json=chat_message,
        timeout=15
    )

    print(f"📤 Report sent to chat, status: {response.status_code}")

except Exception as e:
    print(f"⚠️ Error sending report to chat: {e}")
