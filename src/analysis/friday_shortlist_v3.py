# =========================
# FILE: src/analysis/friday_shortlist_v3.py
# =========================
import os
import json
import math
from datetime import datetime
from itertools import combinations

THURSDAY_REPORT_PATH = "logs/thursday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v3.json"

# ------------------------- BANKROLLS -------------------------
BANKROLL_CORE = float(os.getenv("BANKROLL_CORE", "700"))
BANKROLL_FUN = float(os.getenv("BANKROLL_FUN", "300"))

CORE_EXPOSURE_CAP = 0.18
FUN_EXPOSURE_CAP = 0.20

# ------------------------- CORE RULES -------------------------
CORE_MIN_ODDS = 1.50
CORE_MAX_ODDS = 2.20

CORE_ALLOWED_MARKETS = {"Home", "Away", "Over 2.5", "Under 2.5"}

CORE_STAKE_BY_ODDS = [
    (1.50, 1.65, 25.0),
    (1.65, 1.85, 23.0),
    (1.85, 2.20, 18.0),
]

# ------------------------- FUN RULES -------------------------
FUN_MIN_ODDS = 1.90
FUN_MAX_ODDS = 6.50
FUN_MIN_VALUE_PCT = 5.0
FUN_MAX_PICKS = 7

FUN_ALLOWED_MARKETS = {"Home", "Draw", "Away", "Over 2.5", "Under 2.5"}

MARKET_CODE = {
    "Home": "1",
    "Draw": "X",
    "Away": "2",
    "Over 2.5": "O25",
    "Under 2.5": "U25",
}

# ------------------------- HELPERS -------------------------
def log(msg):
    print(msg, flush=True)

def safe_float(v):
    try:
        return float(v)
    except:
        return None

def load_thursday():
    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["fixtures"], data

def make_pick_id(fid, code):
    return f"{fid}:{code}"

def stake_core_by_odds(odds):
    for lo, hi, stake in CORE_STAKE_BY_ODDS:
        if lo <= odds <= hi:
            return stake
    return 0.0

def fun_single_stake_by_odds(odds):
    if 1.90 <= odds <= 2.20:
        return 17.0
    if 2.20 < odds <= 3.00:
        return 13.0
    if odds > 3.00:
        return 10.0
    return 0.0

def norm(x, lo, hi):
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))

# ------------------------- MARKET ROWS -------------------------
def rows_from_fixture(fx):
    out = []
    fid = fx["fixture_id"]
    match = f'{fx["home"]} – {fx["away"]}'
    league = fx["league"]

    def add(market, prob, fair, odds, value):
        if odds and value is not None:
            out.append({
                "fixture_id": fid,
                "pick_id": make_pick_id(fid, MARKET_CODE[market]),
                "market_code": MARKET_CODE[market],
                "match": match,
                "league": league,
                "market": market,
                "prob": prob,
                "fair": fair,
                "odds": odds,
                "value_pct": value,
            })

    add("Home", fx["home_prob"], fx["fair_1"], fx["offered_1"], fx["value_pct_1"])
    add("Draw", fx["draw_prob"], fx["fair_x"], fx["offered_x"], fx["value_pct_x"])
    add("Away", fx["away_prob"], fx["fair_2"], fx["offered_2"], fx["value_pct_2"])
    add("Over 2.5", fx["over_2_5_prob"], fx["fair_over_2_5"], fx["offered_over_2_5"], fx["value_pct_over"])
    add("Under 2.5", fx["under_2_5_prob"], fx["fair_under_2_5"], fx["offered_under_2_5"], fx["value_pct_under"])

    return out

# ------------------------- CORE -------------------------
def choose_core(rows):
    picks = []
    used = set()

    for r in sorted(rows, key=lambda x: (x["value_pct"], x["prob"]), reverse=True):
        if r["market"] not in CORE_ALLOWED_MARKETS:
            continue
        if not (CORE_MIN_ODDS <= r["odds"] <= CORE_MAX_ODDS):
            continue
        if r["match"] in used:
            continue

        stake = stake_core_by_odds(r["odds"])
        if stake <= 0:
            continue

        picks.append({
            "pick_id": r["pick_id"],
            "fixture_id": r["fixture_id"],
            "market_code": r["market_code"],
            "match": r["match"],
            "league": r["league"],
            "market": r["market"],
            "odds": r["odds"],
            "stake": stake,
        })
        used.add(r["match"])

    return picks

# ------------------------- FUN -------------------------
def choose_fun(rows):
    cand = []
    for r in rows:
        if r["market"] in FUN_ALLOWED_MARKETS and FUN_MIN_ODDS <= r["odds"] <= FUN_MAX_ODDS and r["value_pct"] >= FUN_MIN_VALUE_PCT:
            cand.append(r)

    cand.sort(key=lambda x: (x["value_pct"], x["prob"]), reverse=True)

    picks = []
    used = set()
    for r in cand:
        if r["match"] in used:
            continue
        picks.append(r)
        used.add(r["match"])
        if len(picks) >= FUN_MAX_PICKS:
            break

    if len(picks) < 3:
        return {"picks": [], "singles": [], "system": None}

    vals = [p["value_pct"] for p in picks]
    probs = [p["prob"] for p in picks]

    for p in picks:
        p["fun_score"] = 0.55 * norm(p["value_pct"], min(vals), max(vals)) + \
                         0.45 * norm(p["prob"], min(probs), max(probs))

    picks.sort(key=lambda x: x["fun_score"], reverse=True)

    singles = []
    for p in picks[:3]:
        stake = fun_single_stake_by_odds(p["odds"])
        if stake > 0 and p["prob"] >= 0.34:
            singles.append({
                "pick_id": p["pick_id"],
                "fixture_id": p["fixture_id"],
                "market_code": p["market_code"],
                "match": p["match"],
                "league": p["league"],
                "market": p["market"],
                "odds": p["odds"],
                "stake": stake,
            })

    n = len(picks)
    system = "3/{}".format(n)

    cols = math.comb(n, 3)
    unit = max(1.0, min(3.0, int((BANKROLL_FUN * FUN_EXPOSURE_CAP) // cols)))
    total = round(unit * cols, 1)

    return {
        "picks": picks,
        "singles": singles,
        "system": system,
        "columns": cols,
        "unit": unit,
        "total_stake": total,
    }

# ------------------------- MAIN -------------------------
def main():
    log("🚀 Friday Shortlist v3 FINAL")

    fixtures, th = load_thursday()
    rows = []
    for fx in fixtures:
        rows.extend(rows_from_fixture(fx))

    core = choose_core(rows)
    fun = choose_fun(rows)

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "window": th.get("window"),
        "fixtures_total": len(fixtures),
        "core": {
            "bankroll": BANKROLL_CORE,
            "singles": core,
            "open": sum(p["stake"] for p in core),
            "after_open": BANKROLL_CORE - sum(p["stake"] for p in core),
        },
        "funbet": {
            "bankroll": BANKROLL_FUN,
            "singles": fun["singles"],
            "system": fun.get("system"),
            "columns": fun.get("columns"),
            "unit": fun.get("unit"),
            "total_stake": fun.get("total_stake"),
            "picks": fun.get("picks"),
        },
    }

    os.makedirs("logs", exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log("✅ Friday Shortlist v3 READY")

if __name__ == "__main__":
    main()
