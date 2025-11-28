import os
import json
from datetime import datetime

# === PATHS ===
LOG_DIR = "logs"
REPORT_PATH = os.path.join(LOG_DIR, "tuesday_recap_v1.json")

# === FILE SOURCES ===
FILES = {
    "Draw": "friday_draw_shortlist.json",
    "Over": "friday_over_shortlist.json",
    "FunBet Draw": "friday_funbet_draw.json",
    "FunBet Over": "friday_funbet_over.json",
    "Kelly": "friday_kelly.json"
}

BANKS = {
    "Draw": 400,
    "Over": 300,
    "FunBet Draw": 200,
    "FunBet Over": 200,
    "Kelly": 300
}

# === Helper: calculate metrics ===
def calculate_recap(data, bankroll):
    if not data or "fixtures" not in data:
        return {"hits": 0, "bets": 0, "roi": 0, "profit": 0, "bank_after": bankroll}

    fixtures = data["fixtures"]
    bets = len(fixtures)
    hits = sum(1 for f in fixtures if f.get("won", False))

    avg_odds = sum(f.get("offered_draw", 0) or f.get("offered_over", 0) for f in fixtures) / max(bets, 1)
    stake = fixtures[0].get("stake", 10)
    total_staked = stake * bets
    winnings = hits * stake * avg_odds

    profit = winnings - total_staked
    roi = (profit / total_staked * 100) if total_staked else 0
    bank_after = bankroll + profit

    return {
        "hits": hits,
        "bets": bets,
        "roi": round(roi, 2),
        "profit": round(profit, 2),
        "bank_after": round(bank_after, 2)
    }

# === Process all funds ===
recap = {"generated_at": datetime.utcnow().isoformat(), "funds": {}}
os.makedirs(LOG_DIR, exist_ok=True)

for fund, filename in FILES.items():
    path = os.path.join(LOG_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            recap["funds"][fund] = calculate_recap(data, BANKS[fund])
    else:
        recap["funds"][fund] = {"error": "File not found", "bank_after": BANKS[fund]}

# === Totals ===
total_roi = sum(v.get("roi", 0) for v in recap["funds"].values()) / len(recap["funds"])
recap["total_roi"] = round(total_roi, 2)
recap["lifetime_bank"] = sum(v.get("bank_after", 0) for v in recap["funds"].values())
recap["status"] = "Tuesday recap complete"

# === Save ===
with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(recap, f, indent=2, ensure_ascii=False)

print("✅ Tuesday recap complete.")
print(json.dumps(recap, indent=2, ensure_ascii=False))
