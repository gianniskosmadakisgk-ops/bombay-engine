import os, json

BANKROLL = 400
THRESHOLD = 7.5
FLAT_STAKE = 15

with open("logs/thursday_report_v1.json", "r", encoding="utf-8") as f:
    fixtures = json.load(f).get("data_sample", [])

draws = [
    f for f in fixtures if f.get("score_draw", 0) >= THRESHOLD
]

draws = sorted(draws, key=lambda x: x["score_draw"], reverse=True)[:10]

for d in draws:
    d["stake"] = FLAT_STAKE
    d["wallet"] = BANKROLL

os.makedirs("logs", exist_ok=True)
out = {"count": len(draws), "fixtures": draws}
with open("logs/friday_draw_shortlist.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"✅ Draw shortlist saved — {len(draws)} matches found.")
