import os, json

BANKROLL = 300
THRESHOLD = 7.5
FLAT_STAKE = 15

with open("logs/thursday_report_v1.json", "r", encoding="utf-8") as f:
    fixtures = json.load(f).get("data_sample", [])

overs = [
    f for f in fixtures if f.get("score_over", 0) >= THRESHOLD
]

overs = sorted(overs, key=lambda x: x["score_over"], reverse=True)[:10]

for o in overs:
    o["stake"] = FLAT_STAKE
    o["wallet"] = BANKROLL

os.makedirs("logs", exist_ok=True)
out = {"count": len(overs), "fixtures": overs}
with open("logs/friday_over_shortlist.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"✅ Over shortlist saved — {len(overs)} matches found.")
