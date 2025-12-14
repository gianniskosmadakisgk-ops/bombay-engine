import os
import json
import math
from datetime import datetime
from itertools import combinations

THURSDAY_REPORT = "logs/thursday_report_v3.json"
FRIDAY_REPORT = "logs/friday_shortlist_v3.json"

BANKROLL_CORE = 1000
BANKROLL_FUN = 500

CORE_EXPOSURE = 0.18
FUN_EXPOSURE = 0.20

CORE_ALLOWED = {"Home", "Over 2.5", "Under 2.5"}

CORE_BANDS = [
    (1.50, 1.70, 35, 40),
    (1.70, 1.90, 28, 32),
    (1.90, 2.10, 18, 22),
    (2.10, 2.30, 10, 14),
]

def stake_for_band(odds, value):
    for lo, hi, s_lo, s_hi in CORE_BANDS:
        if lo <= odds <= hi:
            v = max(0, min(1, (value - 3) / 12))
            return round(s_lo + (s_hi - s_lo) * v, 1)
    return 0

def main():
    with open(THURSDAY_REPORT, "r", encoding="utf-8") as f:
        th = json.load(f)

    rows = []
    for fx in th["fixtures"]:
        for m, pk, fk, ok, vk in [
            ("Home", "home_prob", "fair_1", "offered_1", "value_pct_1"),
            ("Over 2.5", "over_2_5_prob", "fair_over_2_5", "offered_over_2_5", "value_pct_over"),
            ("Under 2.5", "under_2_5_prob", "fair_under_2_5", "offered_under_2_5", "value_pct_under"),
        ]:
            if fx.get(pk) and fx.get(ok) and fx.get(vk) is not None:
                rows.append({
                    "match": f"{fx['home']} – {fx['away']}",
                    "league": fx["league"],
                    "market": m,
                    "prob": fx[pk],
                    "fair": fx[fk],
                    "odds": fx[ok],
                    "value": fx[vk],
                })

    rows.sort(key=lambda x: (x["value"], x["prob"]), reverse=True)

    core = []
    used = set()
    for r in rows:
        if r["match"] in used:
            continue
        stake = stake_for_band(r["odds"], r["value"])
        if stake <= 0:
            continue
        core.append({**r, "stake": stake})
        used.add(r["match"])
        if len(core) == 6:
            break

    fun = [r for r in rows if r["odds"] >= 1.90][:7]
    n = len(fun)

    if n >= 7:
        system = "3-4-5/7"
        cols = math.comb(7,3) + math.comb(7,4) + math.comb(7,5)
    elif n == 6:
        system = "3/6"
        cols = math.comb(6,3)
    elif n == 5:
        system = "3/5"
        cols = math.comb(5,3)
    else:
        system, cols = None, 0

    unit = max(1, int((BANKROLL_FUN * FUN_EXPOSURE) // cols)) if cols else 0

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "core": {
            "bankroll": BANKROLL_CORE,
            "picks": core,
            "open": sum(p["stake"] for p in core),
        },
        "funbet": {
            "bankroll": BANKROLL_FUN,
            "system": system,
            "columns": cols,
            "unit": unit,
            "total_stake": unit * cols,
            "picks": fun,
        }
    }

    with open(FRIDAY_REPORT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Friday shortlist saved → {FRIDAY_REPORT}")

if __name__ == "__main__":
    main()
