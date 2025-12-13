import os
import json
import math
from datetime import datetime
from itertools import combinations

THURSDAY_REPORT_PATH = "logs/thursday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v4.json"

# ------------------------- BANKROLLS (units) -------------------------
BANKROLL_CORE = float(os.getenv("BANKROLL_CORE", "1000"))
BANKROLL_FUN = float(os.getenv("BANKROLL_FUN", "500"))

# Exposure caps
CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.18"))  # 18% of core bankroll
FUN_EXPOSURE_CAP = float(os.getenv("FUN_EXPOSURE_CAP", "0.20"))    # 20% of fun bankroll

# ------------------------- CORE RULES -------------------------
CORE_MIN_ODDS = float(os.getenv("CORE_MIN_ODDS", "1.30"))
CORE_MAX_ODDS = float(os.getenv("CORE_MAX_ODDS", "2.20"))
CORE_DOUBLE_MAX_LEG_ODDS = float(os.getenv("CORE_DOUBLE_MAX_LEG_ODDS", "1.50"))

CORE_MIN_VALUE_PCT = float(os.getenv("CORE_MIN_VALUE_PCT", "3.0"))  # require at least +3% value
CORE_MAX_SINGLES = int(os.getenv("CORE_MAX_SINGLES", "6"))
CORE_MIN_SINGLES = int(os.getenv("CORE_MIN_SINGLES", "4"))

# ------------------------- FUNBET RULES -------------------------
FUN_MIN_ODDS = float(os.getenv("FUN_MIN_ODDS", "1.90"))
FUN_MAX_ODDS = float(os.getenv("FUN_MAX_ODDS", "6.50"))   # safety (no 50s)
FUN_MIN_VALUE_PCT = float(os.getenv("FUN_MIN_VALUE_PCT", "5.0"))
FUN_MAX_PICKS = int(os.getenv("FUN_MAX_PICKS", "7"))

# ------------------------- STAKE SCALING (CORE) -------------------------
# Your intent: higher odds => lower stake, and overall scale by "confidence".
# Confidence proxy we are allowed to use from Thursday JSON:
#   - value_pct_* (already computed in Thursday JSON)
#   - prob (already computed in Thursday JSON)
#
# We do NOT invent probabilities. We just map existing numbers into stake.

CORE_STAKE_MAX = float(os.getenv("CORE_STAKE_MAX", "40"))
CORE_STAKE_MIN = float(os.getenv("CORE_STAKE_MIN", "18"))

def log(msg: str):
    print(msg, flush=True)

def safe_float(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default

def load_thursday():
    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("fixtures", []) or [], data

def market_rows_from_fixture(fx):
    """
    Return normalized market candidates using ONLY Thursday JSON fields.
    Markets: Home(1), Draw(X), Away(2), Over2.5, Under2.5
    """
    home = fx.get("home")
    away = fx.get("away")
    league = fx.get("league")
    match = f"{home} – {away}"

    rows = []

    def add(market, prob_key, fair_key, odds_key, value_key):
        prob = safe_float(fx.get(prob_key), None)
        fair = safe_float(fx.get(fair_key), None)
        odds = safe_float(fx.get(odds_key), None)
        val = fx.get(value_key, None)  # already percent or None
        if prob is None or fair is None:
            return
        if odds is None or odds <= 1.0:
            return
        if val is None:
            return
        rows.append({
            "match": match,
            "league": league,
            "market": market,
            "prob": round(prob, 3),
            "fair": fair,
            "odds": odds,
            "value_pct": float(val),
        })

    add("Home", "home_prob", "fair_1", "offered_1", "value_pct_1")
    add("Draw", "draw_prob", "fair_x", "offered_x", "value_pct_x")
    add("Away", "away_prob", "fair_2", "offered_2", "value_pct_2")
    add("Over 2.5", "over_2_5_prob", "fair_over_2_5", "offered_over_2_5", "value_pct_over")
    add("Under 2.5", "under_2_5_prob", "fair_under_2_5", "offered_under_2_5", "value_pct_under")

    return rows

def core_stake(odds, value_pct, prob):
    """
    Stake scaling:
      - base bigger when value_pct bigger
      - base smaller when odds bigger
      - bounded [CORE_STAKE_MIN, CORE_STAKE_MAX]
    """
    # value factor: 3% => ~1.0 ; 10% => ~1.5 ; 20% => ~2.0 (capped)
    vf = min(2.0, max(1.0, (value_pct / 7.0)))  # tuned simple
    # odds factor: 1.30 => ~1.0 ; 2.20 => ~0.65
    of = max(0.55, min(1.0, 1.0 - (odds - 1.30) * 0.40))
    # probability factor: 0.55 => ~1.0 ; 0.70 => ~1.15 ; 0.80 => ~1.25
    pf = max(0.90, min(1.25, 0.65 + prob))
    raw = CORE_STAKE_MIN * vf * of * pf
    return float(max(CORE_STAKE_MIN, min(CORE_STAKE_MAX, raw)))

def choose_core(rows):
    # Filter for core band
    cand = []
    for r in rows:
        if r["odds"] < CORE_MIN_ODDS or r["odds"] > CORE_MAX_ODDS:
            continue
        if r["value_pct"] < CORE_MIN_VALUE_PCT:
            continue
        cand.append(r)

    # Sort by value_pct then prob
    cand.sort(key=lambda x: (x["value_pct"], x["prob"]), reverse=True)

    # Build singles ensuring we don't take multiple markets of the same match unless necessary
    singles = []
    used_matches = set()
    for r in cand:
        if r["match"] in used_matches:
            continue
        stake = round(core_stake(r["odds"], r["value_pct"], r["prob"]), 1)
        singles.append({
            "match": r["match"],
            "league": r["league"],
            "market": r["market"],
            "prob": r["prob"],
            "fair": r["fair"],
            "odds": r["odds"],
            "value_pct": round(r["value_pct"], 1),
            "stake": stake,
        })
        used_matches.add(r["match"])
        if len(singles) >= CORE_MAX_SINGLES:
            break

    # Optional double from low-odds legs (<1.50), only if we have at least 2 such singles candidates
    low = [s for s in singles if s["odds"] < CORE_DOUBLE_MAX_LEG_ODDS]
    double = None
    if len(low) >= 2:
        # choose the best pair by product of value_pct
        best_pair = None
        best_score = -1
        for a, b in combinations(low, 2):
            sc = (a["value_pct"] * b["value_pct"]) + (a["prob"] + b["prob"])
            if sc > best_score:
                best_score = sc
                best_pair = (a, b)

        if best_pair:
            a, b = best_pair
            dbl_odds = round(a["odds"] * b["odds"], 2)
            # stake for double: smaller than singles, based on odds band
            dbl_stake = round(max(18.0, min(30.0, 28.0 * (1.8 / max(1.2, dbl_odds)))), 1)
            double = {
                "type": "Double",
                "legs": [
                    {"match": a["match"], "market": a["market"], "odds": a["odds"]},
                    {"match": b["match"], "market": b["market"], "odds": b["odds"]},
                ],
                "combo_odds": dbl_odds,
                "stake": dbl_stake,
            }

    return singles, double

def fun_system_for_n(n, avg_value, avg_odds):
    """
    Adaptive system decision.
    If n==7 and it's strong enough, use 3-4-5/7 (91 columns).
    Otherwise use simpler.
    """
    if n >= 7:
        if avg_value >= 8.0 and avg_odds >= 2.10:
            return "3-4-5/7"
        return "4/7"
    if n == 6:
        return "3-4/6"
    if n == 5:
        return "3/5"
    if n == 4:
        return "3/4"
    if n == 3:
        return "3/3"
    return None

def columns_for_system(system, n):
    if system is None:
        return 0
    if system == "3-4-5/7":
        return math.comb(7,3) + math.comb(7,4) + math.comb(7,5)  # 91
    if system == "4/7":
        return math.comb(7,4)  # 35
    if system == "3-4/6":
        return math.comb(6,3) + math.comb(6,4)  # 35
    if system == "3/5":
        return math.comb(5,3)  # 10
    if system == "3/4":
        return math.comb(4,3)  # 4
    if system == "3/3":
        return 1
    return 0

def choose_fun(rows):
    cand = []
    for r in rows:
        if r["odds"] < FUN_MIN_ODDS or r["odds"] > FUN_MAX_ODDS:
            continue
        if r["value_pct"] < FUN_MIN_VALUE_PCT:
            continue
        cand.append(r)

    cand.sort(key=lambda x: (x["value_pct"], x["prob"]), reverse=True)

    picks = []
    used_matches = set()
    for r in cand:
        if r["match"] in used_matches:
            continue
        picks.append({
            "match": r["match"],
            "league": r["league"],
            "market": r["market"],
            "prob": r["prob"],
            "fair": r["fair"],
            "odds": r["odds"],
            "value_pct": round(r["value_pct"], 1),
        })
        used_matches.add(r["match"])
        if len(picks) >= FUN_MAX_PICKS:
            break

    n = len(picks)
    if n < 3:
        return {"system": None, "columns": 0, "unit": 0.0, "total_stake": 0.0, "picks": []}

    avg_value = sum(p["value_pct"] for p in picks) / n
    avg_odds = sum(p["odds"] for p in picks) / n

    system = fun_system_for_n(n, avg_value, avg_odds)
    cols = columns_for_system(system, n)

    max_exposure = BANKROLL_FUN * FUN_EXPOSURE_CAP
    unit = 0.0
    total = 0.0
    if cols > 0:
        # keep it simple: unit is integer, min 1, max 5
        unit = max(1.0, min(5.0, float(int(max_exposure // cols) or 1)))
        total = unit * cols
        if total > max_exposure:
            unit = max(1.0, float(int(max_exposure // cols)))
            total = unit * cols

    return {"system": system, "columns": cols, "unit": unit, "total_stake": round(total, 1), "picks": picks}

def cap_core_exposure(core_singles, core_double):
    max_exposure = BANKROLL_CORE * CORE_EXPOSURE_CAP
    open_singles = sum(p["stake"] for p in core_singles)
    open_double = core_double["stake"] if core_double else 0.0
    total_open = open_singles + open_double

    if total_open <= max_exposure or total_open <= 0:
        return core_singles, core_double, 1.0

    scale = max_exposure / total_open
    for p in core_singles:
        p["stake"] = round(p["stake"] * scale, 1)
    if core_double:
        core_double["stake"] = round(core_double["stake"] * scale, 1)
    return core_singles, core_double, round(scale, 3)

def main():
    log("🚀 Running Friday Shortlist v4.0 (CORE + FUNBET only)")

    fixtures, th = load_thursday()
    log(f"Loaded {len(fixtures)} fixtures from {THURSDAY_REPORT_PATH}")

    all_rows = []
    for fx in fixtures:
        all_rows.extend(market_rows_from_fixture(fx))

    core_singles, core_double = choose_core(all_rows)
    fun = choose_fun(all_rows)

    # ensure core minimum singles (if not enough, we still output what we have—no invention)
    core_singles, core_double, scale = cap_core_exposure(core_singles, core_double)

    core_open = round(sum(p["stake"] for p in core_singles) + (core_double["stake"] if core_double else 0.0), 1)
    fun_open = round(fun["total_stake"], 1)

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "window": th.get("window", {}),
        "fixtures_total": len(fixtures),

        "core": {
            "bankroll": BANKROLL_CORE,
            "exposure_cap_pct": CORE_EXPOSURE_CAP,
            "exposure_scale_applied": scale,
            "singles": core_singles,
            "double": core_double,
            "open": core_open,
            "after_open": round(BANKROLL_CORE - core_open, 1),
        },

        "funbet": {
            "bankroll": BANKROLL_FUN,
            "exposure_cap_pct": FUN_EXPOSURE_CAP,
            "system": fun["system"],
            "columns": fun["columns"],
            "unit": fun["unit"],
            "total_stake": fun["total_stake"],
            "picks": fun["picks"],
            "open": fun_open,
            "after_open": round(BANKROLL_FUN - fun_open, 1),
        },
    }

    os.makedirs("logs", exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log(f"✅ Friday Shortlist v4.0 saved → {FRIDAY_REPORT_PATH}")

if __name__ == "__main__":
    main()
