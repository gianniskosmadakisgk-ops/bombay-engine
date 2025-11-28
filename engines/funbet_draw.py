import os, json, itertools

BANKROLL = 200
STAKE_PER_COLUMN = 3

with open("logs/friday_draw_shortlist.json", "r", encoding="utf-8") as f:
    data = json.load(f)["fixtures"]

# Top 5 για FunBet (ρίσκο)
top5 = sorted(data, key=lambda x: x["fair_x"], reverse=True)[:5]
columns = []

# 3-4-5 σύστημα (όλες οι δυνατές τριάδες, τετράδες, πεντάδες)
for n in [3, 4, 5]:
    for combo in itertools.combinations(top5, n):
        columns.append([m["match"] for m in combo])

out = {
    "count": len(columns),
    "stake_total": len(columns) * STAKE_PER_COLUMN,
    "columns": columns,
    "wallet": BANKROLL
}

os.makedirs("logs", exist_ok=True)
with open("logs/friday_funbet_draw.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"🎯 FunBet Draw system ready — {len(columns)} combos created.")
