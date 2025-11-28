import os, json, requests

BANKROLL = 300
KELLY_FRACTION = 0.30
MIN_DIFF = 0.15
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

with open("logs/thursday_report_v1.json", "r", encoding="utf-8") as f:
    fixtures = json.load(f).get("data_sample", [])

value_picks = []

def calc_kelly(fair, book, bankroll):
    P = 1 / fair
    B = book - 1
    Q = 1 - P
    F = (B * P - Q) / B
    if F <= 0:
        return 0
    return round(0.30 * F * bankroll, 2)

for f in fixtures:
    for market, fair_key, label in [
        ("Draw", "fair_x", "draw"),
        ("Home", "fair_1", "home"),
        ("Away", "fair_2", "away"),
        ("Over", "fair_over", "over"),
    ]:
        fair = f.get(fair_key)
        if not fair:
            continue

        offered = fair * (1 + MIN_DIFF)
        diff = (offered - fair) / fair
        if diff >= MIN_DIFF:
            stake = calc_kelly(fair, offered, BANKROLL)
            if stake > 0:
                value_picks.append({
                    "match": f["match"],
                    "market": label,
                    "fair": fair,
                    "book": offered,
                    "diff%": f"{int(diff*100)}%",
                    "stake": stake
                })

value_picks = sorted(value_picks, key=lambda x: x["stake"], reverse=True)[:10]

os.makedirs("logs", exist_ok=True)
out = {"count": len(value_picks), "picks": value_picks}
with open("logs/friday_kelly.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(f"📈 Kelly Engine complete — {len(value_picks)} value picks found.")
