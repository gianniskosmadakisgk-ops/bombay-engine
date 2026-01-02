import os
import json
import math
from datetime import datetime
from itertools import combinations

THURSDAY_REPORT_PATH = "logs/thursday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v3.json"

# ------------------------- BANKROLLS -------------------------
BANKROLL_CORE = float(os.getenv("BANKROLL_CORE", "700"))
BANKROLL_FUN = float(os.getenv("BANKROLL_FUN", "400"))

CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.18"))
FUN_EXPOSURE_CAP = float(os.getenv("FUN_EXPOSURE_CAP", "0.20"))

# ------------------------- CORE RULES -------------------------
CORE_MIN_ODDS = float(os.getenv("CORE_MIN_ODDS", "1.50"))
CORE_MAX_ODDS = float(os.getenv("CORE_MAX_ODDS", "2.20"))

CORE_MIN_VALUE_LOW = float(os.getenv("CORE_MIN_VALUE_LOW", "4.0"))    # 1.50-1.65
CORE_MIN_VALUE_MID = float(os.getenv("CORE_MIN_VALUE_MID", "3.0"))    # 1.65-2.00
CORE_MIN_VALUE_HIGH = float(os.getenv("CORE_MIN_VALUE_HIGH", "2.5"))  # 2.00-2.20

CORE_MAX_SINGLES = int(os.getenv("CORE_MAX_SINGLES", "10"))
CORE_MIN_SINGLES = int(os.getenv("CORE_MIN_SINGLES", "4"))

CORE_ALLOWED_MARKETS = {"Home", "Away", "Over 2.5", "Under 2.5"}

# Core doubles from legs <1.50
CORE_DOUBLE_MAX_LEG_ODDS = float(os.getenv("CORE_DOUBLE_MAX_LEG_ODDS", "1.50"))
CORE_DOUBLE_TARGET_MIN = float(os.getenv("CORE_DOUBLE_TARGET_MIN", "1.55"))
CORE_DOUBLE_TARGET_MAX = float(os.getenv("CORE_DOUBLE_TARGET_MAX", "2.20"))

# CORE stake bands (flat)
CORE_STAKE_LOW_ODDS = float(os.getenv("CORE_STAKE_LOW_ODDS", "25"))  # 1.50-1.65
CORE_STAKE_MID_ODDS = float(os.getenv("CORE_STAKE_MID_ODDS", "23"))  # 1.65-1.90/2.00
CORE_STAKE_HIGH_ODDS = float(os.getenv("CORE_STAKE_HIGH_ODDS", "18"))# 1.90/2.00-2.20

# ------------------------- FUN RULES -------------------------
FUN_MIN_ODDS = float(os.getenv("FUN_MIN_ODDS", "2.00"))
FUN_MAX_ODDS = float(os.getenv("FUN_MAX_ODDS", "3.00"))
FUN_MIN_VALUE_PCT = float(os.getenv("FUN_MIN_VALUE_PCT", "5.0"))

FUN_MAX_PICKS = int(os.getenv("FUN_MAX_PICKS", "8"))
FUN_MAX_SINGLES = int(os.getenv("FUN_MAX_SINGLES", "7"))

FUN_ALLOWED_MARKETS = {"Home", "Draw", "Away", "Over 2.5", "Under 2.5"}

# FUN singles stake bands (flat) - higher odds => lower stake
FUN_STAKE_LOW = float(os.getenv("FUN_STAKE_LOW", "17"))   # 2.00-2.30
FUN_STAKE_MID = float(os.getenv("FUN_STAKE_MID", "13"))   # 2.30-2.60
FUN_STAKE_HIGH = float(os.getenv("FUN_STAKE_HIGH", "10")) # 2.60-3.00

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

# ------------------------- CORE -------------------------
def core_min_value_for_odds(odds: float) -> float:
    if odds < 1.65:
        return CORE_MIN_VALUE_LOW
    if odds < 2.00:
        return CORE_MIN_VALUE_MID
    return CORE_MIN_VALUE_HIGH

def core_stake_for_odds(odds: float) -> float:
    if 1.50 <= odds < 1.65:
        return CORE_STAKE_LOW_ODDS
    if 1.65 <= odds < 2.00:
        return CORE_STAKE_MID_ODDS
    if 2.00 <= odds <= 2.20:
        return CORE_STAKE_HIGH_ODDS
    return 0.0

def choose_core(rows):
    cand = []
    for r in rows:
        if r["market"] not in CORE_ALLOWED_MARKETS:
            continue
        if r["odds"] < CORE_MIN_ODDS or r["odds"] > CORE_MAX_ODDS:
            continue
        if r["value_pct"] < core_min_value_for_odds(r["odds"]):
            continue
        stake = core_stake_for_odds(r["odds"])
        if stake <= 0:
            continue
        cand.append((r, stake))

    cand.sort(key=lambda t: (t[0]["value_pct"], t[0]["prob"]), reverse=True)

    singles = []
    used_matches = set()
    for r, stake in cand:
        if r["match"] in used_matches:
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
            "stake": round(stake, 1),
            "tag": "core",
        })
        used_matches.add(r["match"])
        if len(singles) >= CORE_MAX_SINGLES:
            break

    # Doubles pool: legs < 1.50
    low_legs = []
    for r in rows:
        if r["market"] not in CORE_ALLOWED_MARKETS:
            continue
        if r["odds"] is None or r["odds"] >= CORE_DOUBLE_MAX_LEG_ODDS:
            continue
        if r["value_pct"] < core_min_value_for_odds(1.60):  # same intent: require real value
            continue
        low_legs.append(r)

    low_legs.sort(key=lambda x: (x["value_pct"], x["prob"]), reverse=True)

    doubles = []
    used_in_doubles = set()

    # Greedy pairing: best available pairs within target odds
    pool = low_legs[:16]
    for a, b in combinations(pool, 2):
        if a["match"] == b["match"]:
            continue
        if a["match"] in used_in_doubles or b["match"] in used_in_doubles:
            continue
        combo_odds = a["odds"] * b["odds"]
        if not (CORE_DOUBLE_TARGET_MIN <= combo_odds <= CORE_DOUBLE_TARGET_MAX):
            continue

        stake = core_stake_for_odds(combo_odds if combo_odds <= 2.20 else 2.20)
        if stake <= 0:
            stake = CORE_STAKE_MID_ODDS

        doubles.append({
            "type": "Double",
            "combo_odds": round(combo_odds, 2),
            "stake": round(stake, 1),
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
        })
        used_in_doubles.add(a["match"])
        used_in_doubles.add(b["match"])
        if len(doubles) >= 6:
            break

    primary_double = doubles[0] if doubles else None
    return singles, primary_double, doubles

def cap_core_exposure(core_singles, core_double, core_doubles):
    max_exposure = BANKROLL_CORE * CORE_EXPOSURE_CAP
    open_singles = sum(p["stake"] for p in core_singles)
    open_double = (core_double["stake"] if core_double else 0.0)
    open_more = sum(d["stake"] for d in (core_doubles or []))
    total_open = open_singles + open_double + open_more

    if total_open <= max_exposure or total_open <= 0:
        return core_singles, core_double, core_doubles, 1.0

    scale = max_exposure / total_open
    for p in core_singles:
        p["stake"] = round(p["stake"] * scale, 1)
    if core_double:
        core_double["stake"] = round(core_double["stake"] * scale, 1)
    for d in (core_doubles or []):
        d["stake"] = round(d["stake"] * scale, 1)

    return core_singles, core_double, core_doubles, round(scale, 3)

# ------------------------- FUN -------------------------
def fun_stake_for_odds(odds: float) -> float:
    if 2.00 <= odds < 2.30:
        return FUN_STAKE_LOW
    if 2.30 <= odds < 2.60:
        return FUN_STAKE_MID
    if 2.60 <= odds <= 3.00:
        return FUN_STAKE_HIGH
    return 0.0

def columns_for_system(system, n):
    if system is None:
        return 0
    if system == "4/7":
        return math.comb(7,4)
    if system == "3/7":
        return math.comb(7,3)
    if system == "4/6":
        return math.comb(6,4)
    if system == "3/6":
        return math.comb(6,3)
    if system == "3/5":
        return math.comb(5,3)
    if system == "3/4":
        return math.comb(4,3)
    if system == "3/3":
        return 1
    if system == "3-4-5/7":
        return math.comb(7,3) + math.comb(7,4) + math.comb(7,5)
    return 0

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

    # singles selection: only if "worth it"
    singles = []
    for p in picks:
        st = fun_stake_for_odds(p["odds"])
        if st <= 0:
            continue
        # stronger threshold for singles (so we don't force singles)
        if p["value_pct"] >= 9.0 or p["prob"] >= 0.42:
            singles.append({
                "pick_id": p["pick_id"],
                "fixture_id": p["fixture_id"],
                "market_code": p["market_code"],
                "match": p["match"],
                "league": p["league"],
                "market": p["market"],
                "odds": p["odds"],
                "stake": round(st, 1),
            })
        if len(singles) >= FUN_MAX_SINGLES:
            break

    n = len(picks)
    if n < 3:
        return {
            "system": None, "columns": 0, "unit": 0.0, "total_stake": 0.0,
            "picks": [], "singles": singles, "rule_ok": False
        }

    avg_value = sum(p["value_pct"] for p in picks) / n
    avg_odds = sum(p["odds"] for p in picks) / n

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
        return {
            "system": None, "columns": 0, "unit": 0.0, "total_stake": 0.0,
            "picks": picks, "singles": singles, "rule_ok": False
        }

    # unit: 1-3 ευρώ
    unit = float(int(max_exposure // cols) or 1)
    unit = max(1.0, min(3.0, unit))
    total = unit * cols

    return {
        "system": system,
        "columns": cols,
        "unit": unit,
        "total_stake": round(total, 1),
        "picks": picks,
        "singles": singles,
        "rule_ok": True,
        "avg_value": round(avg_value, 2),
        "avg_odds": round(avg_odds, 2),
    }

def main():
    log("🚀 Running Friday Shortlist v3 (UPDATED)")

    fixtures, th = load_thursday()
    log(f"Loaded {len(fixtures)} fixtures from {THURSDAY_REPORT_PATH}")

    all_rows = []
    for fx in fixtures:
        all_rows.extend(market_rows_from_fixture(fx))

    core_singles, core_double, core_doubles = choose_core(all_rows)
    fun = choose_fun(all_rows)

    core_singles, core_double, core_doubles, scale = cap_core_exposure(core_singles, core_double, core_doubles)

    core_open = round(
        sum(p["stake"] for p in core_singles)
        + (core_double["stake"] if core_double else 0.0)
        + sum(d["stake"] for d in (core_doubles or [])),
        1
    )
    fun_open = round(fun["total_stake"] + sum(s["stake"] for s in fun.get("singles", []) or []), 1)

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
                "allowed_markets": sorted(list(CORE_ALLOWED_MARKETS)),
                "min_value_bands": {
                    "1.50-1.65": CORE_MIN_VALUE_LOW,
                    "1.65-2.00": CORE_MIN_VALUE_MID,
                    "2.00-2.20": CORE_MIN_VALUE_HIGH,
                },
                "stake_bands": {
                    "1.50-1.65": CORE_STAKE_LOW_ODDS,
                    "1.65-2.00": CORE_STAKE_MID_ODDS,
                    "2.00-2.20": CORE_STAKE_HIGH_ODDS,
                },
                "double_leg_max_odds": CORE_DOUBLE_MAX_LEG_ODDS,
                "double_target_combo_odds": [CORE_DOUBLE_TARGET_MIN, CORE_DOUBLE_TARGET_MAX],
            },
            "singles": core_singles,
            "double": core_double,          # legacy: best double
            "doubles": core_doubles,        # all doubles found
            "open": core_open,
            "after_open": round(BANKROLL_CORE - core_open, 1),
            "picks_count": len(core_singles) + (len(core_doubles) if core_doubles else 0),
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
                "single_stake_bands": {
                    "2.00-2.30": FUN_STAKE_LOW,
                    "2.30-2.60": FUN_STAKE_MID,
                    "2.60-3.00": FUN_STAKE_HIGH,
                },
                "system_unit_eur_range": [1, 3],
            },
            "system": fun["system"],
            "columns": fun["columns"],
            "unit": fun["unit"],
            "total_stake": fun["total_stake"],
            "picks": fun["picks"],          # system picks
            "singles": fun.get("singles", []),  # singles (can overlap with system)
            "open": fun_open,
            "after_open": round(BANKROLL_FUN - fun_open, 1),
            "picks_count": len(fun["picks"]),
            "singles_count": len(fun.get("singles", []) or []),
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
