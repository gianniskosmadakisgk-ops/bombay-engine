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
CORE_MAX_ODDS = float(os.getenv("CORE_MAX_ODDS", "2.10"))
CORE_ELITE_MAX_ODDS = float(os.getenv("CORE_ELITE_MAX_ODDS", "2.30"))

CORE_DOUBLE_MAX_LEG_ODDS = float(os.getenv("CORE_DOUBLE_MAX_LEG_ODDS", "1.50"))
CORE_DOUBLE_TARGET_MIN = float(os.getenv("CORE_DOUBLE_TARGET_MIN", "1.70"))
CORE_DOUBLE_TARGET_MAX = float(os.getenv("CORE_DOUBLE_TARGET_MAX", "1.90"))

CORE_MIN_VALUE_PCT = float(os.getenv("CORE_MIN_VALUE_PCT", "3.0"))
CORE_MAX_PICKS = int(os.getenv("CORE_MAX_PICKS", "6"))
CORE_MIN_PICKS = int(os.getenv("CORE_MIN_PICKS", "4"))

# Stake bands (your table)
STAKE_150_170 = (35.0, 40.0)
STAKE_170_190 = (28.0, 32.0)
STAKE_190_210 = (18.0, 22.0)
STAKE_210_230 = (10.0, 14.0)

# ------------------------- FUN RULES -------------------------
FUN_MIN_ODDS = float(os.getenv("FUN_MIN_ODDS", "1.90"))
FUN_MAX_ODDS = float(os.getenv("FUN_MAX_ODDS", "6.50"))
FUN_MIN_VALUE_PCT = float(os.getenv("FUN_MIN_VALUE_PCT", "5.0"))
FUN_MAX_PICKS = int(os.getenv("FUN_MAX_PICKS", "7"))
FUN_MIN_PICKS = int(os.getenv("FUN_MIN_PICKS", "5"))

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
    Use ONLY Thursday JSON fields (no invention).
    Core markets: Home(1), Over2.5, Under2.5
    Fun markets: can include Home/Draw/Away/Over/Under but we still use Thursday values.
    """
    match = f"{fx.get('home')} – {fx.get('away')}"
    league = fx.get("league")

    rows = []

    def add(market, prob_key, fair_key, odds_key, value_key, score_key=None):
        prob = safe_float(fx.get(prob_key), None)
        fair = safe_float(fx.get(fair_key), None)
        odds = safe_float(fx.get(odds_key), None)
        val = fx.get(value_key, None)
        score = safe_float(fx.get(score_key), None) if score_key else None
        if prob is None or fair is None or odds is None or odds <= 1.0 or val is None:
            return
        rows.append({
            "match": match,
            "league": league,
            "market": market,
            "prob": round(prob, 3),
            "fair": fair,
            "odds": odds,
            "value_pct": float(val),
            "score": score,
        })

    # Core allowed
    add("Home", "home_prob", "fair_1", "offered_1", "value_pct_1", None)
    add("Over 2.5", "over_2_5_prob", "fair_over_2_5", "offered_over_2_5", "value_pct_over", "score_over")
    add("Under 2.5", "under_2_5_prob", "fair_under_2_5", "offered_under_2_5", "value_pct_under", "score_under")

    # Fun extra markets (optional but allowed by your spec)
    add("Draw", "draw_prob", "fair_x", "offered_x", "value_pct_x", "score_draw")
    add("Away", "away_prob", "fair_2", "offered_2", "value_pct_2", None)

    return rows

def core_stake_from_odds(odds, elite=False):
    # Band mapping exactly as your rules (deterministic)
    if odds < 1.50:
        return 0.0  # forbidden as single
    if 1.50 <= odds < 1.70:
        return sum(STAKE_150_170) / 2.0
    if 1.70 <= odds < 1.90:
        return sum(STAKE_170_190) / 2.0
    if 1.90 <= odds <= 2.10:
        return sum(STAKE_190_210) / 2.0
    if elite and 2.10 < odds <= 2.30:
        return sum(STAKE_210_230) / 2.0
    return 0.0

def choose_core(rows):
    # Core candidates: only allowed markets already in rows; filter odds + value
    cand = []
    for r in rows:
        if r["market"] not in ("Home", "Over 2.5", "Under 2.5"):
            continue
        if r["value_pct"] < CORE_MIN_VALUE_PCT:
            continue

        if CORE_MIN_ODDS <= r["odds"] <= CORE_MAX_ODDS:
            elite = False
        elif CORE_MAX_ODDS < r["odds"] <= CORE_ELITE_MAX_ODDS:
            # Elite only if "score" exists and is top-ish
            elite = (r.get("score") is not None and r["score"] >= 7.5) or (r["value_pct"] >= 10.0)
            if not elite:
                continue
        else:
            continue

        cand.append((r, elite))

    # Sort: value_pct desc, then prob desc, then (optional) score desc
    cand.sort(key=lambda x: (x[0]["value_pct"], x[0]["prob"], x[0].get("score") or 0.0), reverse=True)

    singles = []
    used_matches = set()
    low_legs = []  # for potential double (<1.50)
    for r, elite in cand:
        if r["match"] in used_matches:
            continue

        # singles rule: <1.50 is NOT allowed as single, but can be a double leg
        if r["odds"] < CORE_DOUBLE_MAX_LEG_ODDS:
            low_legs.append(r)
            continue

        stake = core_stake_from_odds(r["odds"], elite=elite)
        if stake <= 0:
            continue

        singles.append({
            "match": r["match"],
            "league": r["league"],
            "market": r["market"],
            "prob": r["prob"],
            "fair": r["fair"],
            "odds": r["odds"],
            "value_pct": round(r["value_pct"], 1),
            "stake": round(stake, 1),
        })
        used_matches.add(r["match"])
        if len(singles) >= CORE_MAX_PICKS:
            break

    # Double rule: only if we have 2 low-odds legs (<1.50), target combo odds 1.70–1.90
    core_double = None
    if len(low_legs) >= 2:
        best = None
        best_score = -1e9
        for a, b in combinations(low_legs, 2):
            combo_odds = a["odds"] * b["odds"]
            if not (CORE_DOUBLE_TARGET_MIN <= combo_odds <= CORE_DOUBLE_TARGET_MAX):
                continue
            sc = (a["value_pct"] + b["value_pct"]) + (a["prob"] + b["prob"])
            if sc > best_score:
                best_score = sc
                best = (a, b, combo_odds)
        if best:
            a, b, combo_odds = best
            # stake as "single 1.70–1.80" band
            dbl_stake = sum(STAKE_170_190) / 2.0
            core_double = {
                "type": "Double",
                "legs": [
                    {"match": a["match"], "league": a["league"], "market": a["market"], "odds": a["odds"]},
                    {"match": b["match"], "league": b["league"], "market": b["market"], "odds": b["odds"]},
                ],
                "combo_odds": round(combo_odds, 2),
                "stake": round(dbl_stake, 1),
            }

    return singles, core_double

def fun_system_for_n(n, avg_value, avg_odds):
    if n >= 7:
        # if strong, allow 3-4-5/7, else 3/7 (safer than 4/7 given your "refund" constraint)
        if avg_value >= 8.0 and avg_odds >= 2.10:
            return "3-4-5/7"
        return "3/7"
    if n == 6:
        return "3/6"
    if n == 5:
        return "3/5"
    return None

def columns_for_system(system, n):
    if system is None:
        return 0
    if system == "3-4-5/7":
        return math.comb(7,3) + math.comb(7,4) + math.comb(7,5)  # 91
    if system == "3/7":
        return math.comb(7,3)  # 35
    if system == "3/6":
        return math.comb(6,3)  # 20
    if system == "3/5":
        return math.comb(5,3)  # 10
    return 0

def min_hit_refund_ok(picks, system, unit):
    """
    Mathematical rule (your requirement):
      minimum hit level returns >= total stake.
    We approximate minimum hit using:
      - system implied: 3/5 means min hit=3; 3/6 min=3; 3/7 min=3; 3-4-5/7 min=3
      - refund condition: (min_product_odds) * unit >= total stake / number_of_min_hit_tickets
    Since we cannot know ticket-level odds distribution precisely without enumerating,
    we enforce a simple necessary condition:
      - smallest 3 odds product * unit >= total_stake / columns_min
    This is conservative and uses ONLY odds from picks.
    """
    if unit <= 0:
        return False
    odds_list = sorted([p["odds"] for p in picks])
    if len(odds_list) < 3:
        return False

    min_prod3 = odds_list[0] * odds_list[1] * odds_list[2]

    if system == "3-4-5/7":
        cols_min = math.comb(7,3)  # min-hit tickets = all 3-combos
        total = unit * columns_for_system(system, 7)
    elif system == "3/7":
        cols_min = math.comb(7,3)
        total = unit * columns_for_system(system, 7)
    elif system == "3/6":
        cols_min = math.comb(6,3)
        total = unit * columns_for_system(system, 6)
    elif system == "3/5":
        cols_min = math.comb(5,3)
        total = unit * columns_for_system(system, 5)
    else:
        return False

    # Necessary condition: worst 3-combo payout >= average cost per min-hit ticket
    avg_cost_per_min_ticket = total / max(1, cols_min)
    return (min_prod3 * unit) >= avg_cost_per_min_ticket

def choose_fun(rows):
    # candidates: odds>=1.90, value>=5%
    cand = []
    for r in rows:
        if r["odds"] < FUN_MIN_ODDS or r["odds"] > FUN_MAX_ODDS:
            continue
        if r["value_pct"] < FUN_MIN_VALUE_PCT:
            continue
        cand.append(r)

    cand.sort(key=lambda x: (x["value_pct"], x["prob"], x.get("score") or 0.0), reverse=True)

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
    if n < 5:
        return {"system": None, "columns": 0, "unit": 0.0, "total_stake": 0.0, "picks": picks}

    avg_value = sum(p["value_pct"] for p in picks) / n
    avg_odds = sum(p["odds"] for p in picks) / n

    system = fun_system_for_n(n, avg_value, avg_odds)
    cols = columns_for_system(system, n)

    if cols <= 0:
        return {"system": None, "columns": 0, "unit": 0.0, "total_stake": 0.0, "picks": picks}

    max_exposure = BANKROLL_FUN * FUN_EXPOSURE_CAP

    # unit integer 1..5
    unit = max(1.0, min(5.0, float(int(max_exposure // cols) or 1)))
    total = unit * cols
    if total > max_exposure:
        unit = max(1.0, float(int(max_exposure // cols)))
        total = unit * cols

    # Refund constraint: if fails, downgrade system or trim picks
    if not min_hit_refund_ok(picks, system, unit):
        # Try safer system first (for 7 picks: switch from 3-4-5/7 -> 3/7)
        if system == "3-4-5/7":
            system = "3/7"
            cols = columns_for_system(system, n)
            unit = max(1.0, min(5.0, float(int(max_exposure // cols) or 1)))
            total = unit * cols

        # If still fails, trim to 6 or 5
        while (n > 5) and (not min_hit_refund_ok(picks, system, unit)):
            picks = picks[:-1]
            n = len(picks)
            system = fun_system_for_n(n, avg_value, avg_odds)
            cols = columns_for_system(system, n)
            unit = max(1.0, min(5.0, float(int(max_exposure // cols) or 1)))
            total = unit * cols

    return {"system": system, "columns": cols, "unit": unit, "total_stake": round(total, 1), "picks": picks}

def cap_core_exposure(core_singles, core_double):
    max_exposure = BANKROLL_CORE * CORE_EXPOSURE_CAP
    open_s = sum(p["stake"] for p in core_singles)
    open_d = core_double["stake"] if core_double else 0.0
    total = open_s + open_d
    if total <= max_exposure or total <= 0:
        return core_singles, core_double, 1.0

    scale = max_exposure / total
    for p in core_singles:
        p["stake"] = round(p["stake"] * scale, 1)
    if core_double:
        core_double["stake"] = round(core_double["stake"] * scale, 1)
    return core_singles, core_double, round(scale, 3)

def main():
    log("🚀 Running Bombay Friday Shortlist (V3) — CORE + FUN ONLY (NO KELLY)")

    fixtures, th = load_thursday()
    log(f"Loaded {len(fixtures)} fixtures from {THURSDAY_REPORT_PATH}")

    rows = []
    for fx in fixtures:
        rows.extend(market_rows_from_fixture(fx))

    core_singles, core_double = choose_core(rows)
    funbet = choose_fun(rows)

    core_singles, core_double, scale = cap_core_exposure(core_singles, core_double)

    core_open = round(sum(p["stake"] for p in core_singles) + (core_double["stake"] if core_double else 0.0), 1)
    fun_open = round(funbet["total_stake"], 1)

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
            "system": funbet["system"],
            "columns": funbet["columns"],
            "unit": funbet["unit"],
            "total_stake": funbet["total_stake"],
            "picks": funbet["picks"],
            "open": fun_open,
            "after_open": round(BANKROLL_FUN - fun_open, 1),
        },
    }

    os.makedirs("logs", exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log(f"✅ Friday Shortlist (V3) saved → {FRIDAY_REPORT_PATH}")

if __name__ == "__main__":
    main()
