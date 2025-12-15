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

# ------------------------- BANKROLLS (units) -------------------------
BANKROLL_CORE = float(os.getenv("BANKROLL_CORE", "1000"))
BANKROLL_FUN = float(os.getenv("BANKROLL_FUN", "500"))

CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.18"))  # 18%
FUN_EXPOSURE_CAP = float(os.getenv("FUN_EXPOSURE_CAP", "0.20"))    # 20%

# ------------------------- CORE RULES -------------------------
CORE_MIN_ODDS = float(os.getenv("CORE_MIN_ODDS", "1.30"))
CORE_MAX_ODDS = float(os.getenv("CORE_MAX_ODDS", "2.20"))
CORE_ELITE_MAX_ODDS = float(os.getenv("CORE_ELITE_MAX_ODDS", "2.30"))

CORE_DOUBLE_MAX_LEG_ODDS = float(os.getenv("CORE_DOUBLE_MAX_LEG_ODDS", "1.50"))
CORE_DOUBLE_TARGET_MIN = float(os.getenv("CORE_DOUBLE_TARGET_MIN", "1.70"))
CORE_DOUBLE_TARGET_MAX = float(os.getenv("CORE_DOUBLE_TARGET_MAX", "1.90"))

CORE_MIN_VALUE_PCT = float(os.getenv("CORE_MIN_VALUE_PCT", "3.0"))
CORE_ELITE_MIN_VALUE_PCT = float(os.getenv("CORE_ELITE_MIN_VALUE_PCT", "10.0"))

CORE_MAX_SINGLES = int(os.getenv("CORE_MAX_SINGLES", "6"))
CORE_MIN_SINGLES = int(os.getenv("CORE_MIN_SINGLES", "4"))

# Allowed markets in CORE only
CORE_ALLOWED_MARKETS = {"Home", "Over 2.5", "Under 2.5"}

# ------------------------- FUN RULES -------------------------
FUN_MIN_ODDS = float(os.getenv("FUN_MIN_ODDS", "1.90"))
FUN_MAX_ODDS = float(os.getenv("FUN_MAX_ODDS", "6.50"))
FUN_MIN_VALUE_PCT = float(os.getenv("FUN_MIN_VALUE_PCT", "5.0"))
FUN_MAX_PICKS = int(os.getenv("FUN_MAX_PICKS", "7"))

FUN_ALLOWED_MARKETS = {"Home", "Draw", "Away", "Over 2.5", "Under 2.5"}

# ------------------------- STAKE BANDS (CORE) -------------------------
CORE_STAKE_BANDS = [
    (1.50, 1.70, 35.0, 40.0),
    (1.70, 1.90, 28.0, 32.0),
    (1.90, 2.10, 18.0, 22.0),
    (2.10, 2.30, 10.0, 14.0),
]

MARKET_CODE = {
    "Home": "1",
    "Draw": "X",
    "Away": "2",
    "Over 2.5": "O25",
    "Under 2.5": "U25",
}

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
    if not os.path.exists(THURSDAY_REPORT_PATH):
        raise FileNotFoundError(f"Thursday report not found: {THURSDAY_REPORT_PATH}")
    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("fixtures", []) or [], data

def make_pick_id(fixture_id, market_code):
    return f"{fixture_id}:{market_code}"

def market_rows_from_fixture(fx):
    fixture_id = fx.get("fixture_id")
    home = fx.get("home")
    away = fx.get("away")
    league = fx.get("league")
    match = f"{home} – {away}"

    rows = []

    def add(market, prob_key, fair_key, odds_key, value_key):
        prob = safe_float(fx.get(prob_key), None)
        fair = safe_float(fx.get(fair_key), None)
        odds = safe_float(fx.get(odds_key), None)
        val = fx.get(value_key, None)
        if prob is None or fair is None:
            return
        if odds is None or odds <= 1.0:
            return
        if val is None:
            return

        mcode = MARKET_CODE.get(market)
        pid = make_pick_id(fixture_id, mcode) if fixture_id and mcode else None

        rows.append({
            "fixture_id": fixture_id,
            "pick_id": pid,
            "market_code": mcode,
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

def band_stake(odds, value_pct, elite=False):
    if odds is None:
        return 0.0

    lo = hi = None
    for a, b, s_lo, s_hi in CORE_STAKE_BANDS:
        if a <= odds <= b:
            lo, hi = s_lo, s_hi
            break

    if lo is None:
        return 0.0

    v = max(0.0, min(1.0, (value_pct - 3.0) / 12.0))
    stake = lo + (hi - lo) * v
    if elite:
        stake = min(stake, 14.0)
    return round(stake, 1)

def choose_core(rows):
    cand = []
    for r in rows:
        if r["market"] not in CORE_ALLOWED_MARKETS:
            continue
        if r["odds"] < CORE_MIN_ODDS:
            continue

        if r["odds"] <= CORE_MAX_ODDS:
            if r["value_pct"] < CORE_MIN_VALUE_PCT:
                continue
            cand.append((False, r))
            continue

        if CORE_MAX_ODDS < r["odds"] <= CORE_ELITE_MAX_ODDS:
            if r["value_pct"] < CORE_ELITE_MIN_VALUE_PCT:
                continue
            cand.append((True, r))

    cand.sort(key=lambda t: (t[1]["value_pct"], t[1]["prob"]), reverse=True)

    singles = []
    used_matches = set()

    for is_elite, r in cand:
        if r["match"] in used_matches:
            continue

        # <1.50 singles: ONLY for double
        if r["odds"] < CORE_DOUBLE_MAX_LEG_ODDS:
            continue

        stake = band_stake(r["odds"], r["value_pct"], elite=is_elite)
        if stake <= 0:
            continue

        singles.append({
            "pick_id": r.get("pick_id"),
            "fixture_id": r.get("fixture_id"),
            "market_code": r.get("market_code"),

            "match": r["match"],
            "league": r["league"],
            "market": r["market"],
            "prob": r["prob"],
            "fair": r["fair"],
            "odds": r["odds"],
            "value_pct": round(r["value_pct"], 1),
            "stake": stake,
            "tag": "elite" if is_elite else "core",
        })
        used_matches.add(r["match"])
        if len(singles) >= CORE_MAX_SINGLES:
            break

    # Optional Double from legs <1.50
    low_legs = []
    for r in rows:
        if r["market"] not in CORE_ALLOWED_MARKETS:
            continue
        if r["odds"] is None or r["odds"] >= CORE_DOUBLE_MAX_LEG_ODDS:
            continue
        if r["value_pct"] is None or r["value_pct"] < CORE_MIN_VALUE_PCT:
            continue
        low_legs.append(r)

    low_legs.sort(key=lambda x: (x["value_pct"], x["prob"]), reverse=True)

    core_double = None
    if len(low_legs) >= 2:
        best = None
        best_score = -1e9
        pool = low_legs[:10]
        for a, b in combinations(pool, 2):
            if a["match"] == b["match"]:
                continue
            combo_odds = a["odds"] * b["odds"]
            if not (CORE_DOUBLE_TARGET_MIN <= combo_odds <= CORE_DOUBLE_TARGET_MAX):
                continue
            score = (a["value_pct"] + b["value_pct"]) + (a["prob"] + b["prob"])
            if score > best_score:
                best_score = score
                best = (a, b, combo_odds)

        if best:
            a, b, combo_odds = best
            core_double = {
                "type": "Double",
                "combo_odds": round(combo_odds, 2),
                "stake": 30.0,
                "legs": [
                    {
                        "pick_id": a.get("pick_id"),
                        "fixture_id": a.get("fixture_id"),
                        "market_code": a.get("market_code"),
                        "match": a["match"],
                        "league": a["league"],
                        "market": a["market"],
                        "odds": a["odds"],
                    },
                    {
                        "pick_id": b.get("pick_id"),
                        "fixture_id": b.get("fixture_id"),
                        "market_code": b.get("market_code"),
                        "match": b["match"],
                        "league": b["league"],
                        "market": b["market"],
                        "odds": b["odds"],
                    },
                ],
            }

    return singles, core_double

# ---------- FUN SYSTEM ----------
def columns_for_system(system, n):
    if system is None:
        return 0
    if system == "3-4-5/7":
        return math.comb(7,3) + math.comb(7,4) + math.comb(7,5)  # 91
    if system == "4/7":
        return math.comb(7,4)  # 35
    if system == "3/7":
        return math.comb(7,3)  # 35
    if system == "4/6":
        return math.comb(6,4)  # 15
    if system == "3/6":
        return math.comb(6,3)  # 20
    if system == "3/5":
        return math.comb(5,3)  # 10
    if system == "3/4":
        return math.comb(4,3)  # 4
    if system == "3/3":
        return 1
    return 0

def min_success_return_estimate(picks, system, unit):
    if not picks or not system or unit <= 0:
        return 0.0
    n = len(picks)
    worst = min(p["odds"] for p in picks if p.get("odds"))
    if worst <= 1.0:
        return 0.0

    if system in ("3/3","3/4","3/5","3/6","3/7","3-4-5/7"):
        k = 3
    elif system in ("4/6","4/7"):
        k = 4
    else:
        k = 3

    lines_k = math.comb(n, k) if n >= k else 0
    return lines_k * unit * (worst ** k)

def choose_fun(rows):
    cand = []
    for r in rows:
        if r["market"] not in FUN_ALLOWED_MARKETS:
            continue
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
            "pick_id": r.get("pick_id"),
            "fixture_id": r.get("fixture_id"),
            "market_code": r.get("market_code"),

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
        return {"system": None, "columns": 0, "unit": 0.0, "total_stake": 0.0, "picks": [], "rule_ok": False}

    # ---- “ελεύθερο” σύστημα ανά confidence (proxy: avg value_pct & avg odds) ----
    avg_value = sum(p["value_pct"] for p in picks) / n
    avg_odds = sum(p["odds"] for p in picks) / n

    # starting system decision
    if n == 7:
        system = "3-4-5/7" if (avg_value >= 8.0 and avg_odds >= 2.10) else "4/7"
    elif n == 6:
        system = "3/6" if avg_value >= 6.5 else "4/6"
    elif n == 5:
        system = "3/5"
    elif n == 4:
        system = "3/4"
    else:
        system = "3/3"

    max_exposure = BANKROLL_FUN * FUN_EXPOSURE_CAP
    cols = columns_for_system(system, n)
    if cols <= 0:
        return {"system": None, "columns": 0, "unit": 0.0, "total_stake": 0.0, "picks": picks, "rule_ok": False}

    unit = float(int(max_exposure // cols) or 1)
    unit = max(1.0, min(5.0, unit))
    total = unit * cols

    # math rule: min success should return >= total stake (guardrail)
    est_return = min_success_return_estimate(picks, system, unit)
    rule_ok = est_return >= total

    downgrade_order = []
    if n == 7:
        downgrade_order = ["4/7", "3/7"]
    elif n == 6:
        downgrade_order = ["4/6", "3/6"]
    else:
        downgrade_order = [system]

    if not rule_ok:
        for sys2 in downgrade_order:
            cols2 = columns_for_system(sys2, n)
            if cols2 <= 0:
                continue
            unit2 = float(int(max_exposure // cols2) or 1)
            unit2 = max(1.0, min(5.0, unit2))
            total2 = unit2 * cols2
            if min_success_return_estimate(picks, sys2, unit2) >= total2:
                system, cols, unit, total = sys2, cols2, unit2, total2
                rule_ok = True
                break

    while not rule_ok and len(picks) > 3:
        picks = picks[:-1]
        n = len(picks)
        if n == 6:
            system = "3/6"
        elif n == 5:
            system = "3/5"
        elif n == 4:
            system = "3/4"
        else:
            system = "3/3"
        cols = columns_for_system(system, n)
        unit = float(int(max_exposure // cols) or 1)
        unit = max(1.0, min(5.0, unit))
        total = unit * cols
        rule_ok = min_success_return_estimate(picks, system, unit) >= total

    return {
        "system": system,
        "columns": cols,
        "unit": unit,
        "total_stake": round(total, 1),
        "picks": picks,
        "rule_ok": rule_ok,
        "avg_value": round(avg_value, 2),
        "avg_odds": round(avg_odds, 2),
    }

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
    log("🚀 Running Friday Shortlist v3 (CORE + FUN only)")

    fixtures, th = load_thursday()
    log(f"Loaded {len(fixtures)} fixtures from {THURSDAY_REPORT_PATH}")

    all_rows = []
    for fx in fixtures:
        all_rows.extend(market_rows_from_fixture(fx))

    core_singles, core_double = choose_core(all_rows)
    fun = choose_fun(all_rows)

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
            "rules": {
                "odds_range": [CORE_MIN_ODDS, CORE_MAX_ODDS],
                "elite_max_odds": CORE_ELITE_MAX_ODDS,
                "allowed_markets": sorted(list(CORE_ALLOWED_MARKETS)),
                "double_leg_max_odds": CORE_DOUBLE_MAX_LEG_ODDS,
                "double_target_combo_odds": [CORE_DOUBLE_TARGET_MIN, CORE_DOUBLE_TARGET_MAX],
            },
            "singles": core_singles,
            "double": core_double,
            "open": core_open,
            "after_open": round(BANKROLL_CORE - core_open, 1),
            "picks_count": len(core_singles) + (1 if core_double else 0),
        },

        "funbet": {
            "bankroll": BANKROLL_FUN,
            "exposure_cap_pct": FUN_EXPOSURE_CAP,
            "rules": {
                "odds_min": FUN_MIN_ODDS,
                "odds_max": FUN_MAX_ODDS,
                "min_value_pct": FUN_MIN_VALUE_PCT,
                "max_picks": FUN_MAX_PICKS,
                "allowed_markets": sorted(list(FUN_ALLOWED_MARKETS)),
                "min_success_rule_ok": fun.get("rule_ok", False),
            },
            "system": fun["system"],
            "columns": fun["columns"],
            "unit": fun["unit"],
            "total_stake": fun["total_stake"],
            "picks": fun["picks"],
            "open": fun_open,
            "after_open": round(BANKROLL_FUN - fun_open, 1),
            "picks_count": len(fun["picks"]),
            "avg_value": fun.get("avg_value"),
            "avg_odds": fun.get("avg_odds"),
        },
    }

    os.makedirs("logs", exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log(f"✅ Friday Shortlist v3 saved → {FRIDAY_REPORT_PATH}")

if __name__ == "__main__":
    main()
