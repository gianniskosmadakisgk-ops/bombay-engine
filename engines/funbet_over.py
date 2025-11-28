import os, json, itertools

BANKROLL = 200
STAKE_PER_COLUMN = 3

with open("logs/friday_over_shortlist.json", "r", encoding="utf-8") as f:
    data = json.load(f)["fixtures"]

top5 = [o for o in data if o.get("fair_over", 0) > 2.20][:5]
columns = []

# 2-3-4 σύστημα (όλες οι δυνατές δυάδες, τριάδες, τετράδες)
for n in [2, 3, 4]:
    for combo in itertools.combinations(top5, n):
        columns.append([m["match"] for m in combo])

out = {
    "count": len(columns),
    "stake_total": len(columns) * STAKE_PER_COLUMN,
    "columns": columns,
    "wallet": BANKROLL
}

os.makedirs("logs", exist_ok=True)
with open("logs/friday_funbet_over.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"🔥 FunBet Over system ready — {len(columns)} combos created.")
