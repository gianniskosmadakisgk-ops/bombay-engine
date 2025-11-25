import os
import json
import random
import requests

# -----------------------------------------------------------
# Ρυθμίσεις API & Paths
# -----------------------------------------------------------
REPORT_SOURCE = "friday_shortlist_v1.json"  # ή "thursday_report_v1.json" αν δεν υπάρχει shortlist
OUTPUT_FILE = "tuesday_recap_v1.json"
CHAT_ENDPOINT = "https://bombay-engine.onrender.com/chat_forward"

# -----------------------------------------------------------
# Διαβάζει το αρχείο εισόδου
# -----------------------------------------------------------
print("📊 Starting Tuesday Recap...")

if not os.path.exists(REPORT_SOURCE):
    print(f"⚠️ Δεν βρέθηκε το αρχείο {REPORT_SOURCE}. Θα γίνει fallback στο thursday_report_v1.json")
    REPORT_SOURCE = "thursday_report_v1.json"

if not os.path.exists(REPORT_SOURCE):
    print("❌ Δεν υπάρχουν δεδομένα για recap.")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"status": "fail", "reason": "no source data"}, f, ensure_ascii=False, indent=2)
    exit()

with open(REPORT_SOURCE, "r", encoding="utf-8") as f:
    data = json.load(f)

matches = data.get("matches", [])
if not matches:
    print("⚠️ Δεν υπάρχουν αγώνες στο αρχείο.")
    exit()

# -----------------------------------------------------------
# Ανάλυση – δημιουργία στατιστικών
# -----------------------------------------------------------
recap_results = []
total_value_hits = 0
total_over_hits = 0
total_draw_hits = 0

for m in matches:
    # Προσομοίωση αποτελέσματος
    result = random.choice(["1", "X", "2"])
    goals = random.randint(0, 5)
    opp_goals = random.randint(0, 5)
    over = goals + opp_goals > 2.5

    recap_results.append({
        "match": m.get("teams", "Unknown"),
        "result": result,
        "score": f"{goals}-{opp_goals}",
        "was_over": over,
        "fair_1": m.get("fair_1"),
        "fair_x": m.get("fair_x"),
        "fair_2": m.get("fair_2"),
        "fair_over": m.get("fair_over")
    })

    if over:
        total_over_hits += 1
    if result == "X":
        total_draw_hits += 1
    if m.get("fair_1", 0) < 2.0 or m.get("fair_2", 0) < 2.0:
        total_value_hits += 1

# -----------------------------------------------------------
# Δημιουργία αναφοράς
# -----------------------------------------------------------
summary = {
    "total_matches": len(recap_results),
    "value_hits": total_value_hits,
    "over_hits": total_over_hits,
    "draw_hits": total_draw_hits,
    "success_rate": f"{round((total_value_hits + total_over_hits + total_draw_hits) / len(recap_results) * 33, 1)}%",
}

final_report = {
    "summary": summary,
    "recap_details": recap_results
}

# Αποθήκευση
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_report, f, ensure_ascii=False, indent=2)

print(f"✅ Tuesday Recap completed — {len(recap_results)} matches analyzed and saved to {OUTPUT_FILE}")

# -----------------------------------------------------------
# Αποστολή στο Chat
# -----------------------------------------------------------
try:
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        report_data = f.read()

    response = requests.post(
        CHAT_ENDPOINT,
        json={"message": f"📊 Tuesday Recap Report\n\n{report_data}"},
        timeout=10
    )
    print("💬 Report sent to chat:", response.status_code)
except Exception as e:
    print(f"⚠️ Could not send to chat: {e}")
