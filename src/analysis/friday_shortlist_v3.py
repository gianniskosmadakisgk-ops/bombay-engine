# =========================
# FRIDAY SHORTLIST v3.3 — FUN 10 PICKS + AUTO SYSTEM (EV-BASED) + ODDS-BASED STAKES
# - CORE: same logic (value_adj bands + stake bands + exposure cap + optional double)
# - FUN:
#     * Select up to 10 best value picks (any market: 1/X/2/O25/U25)
#     * Choose FUN singles subset with stake decreasing as odds rise
#     * Build FUN system using up to 8 matches
#     * Choose system type (2/n,3/n,4/n,3-4/n,3-4-5/n,...) by maximizing EV per € (approx)
# - Fixes: system size always matches available picks (no "3-4-5/8" on 7 games)
# =========================

import os
import json
from datetime import datetime
from itertools import combinations
from math import comb

THURSDAY_REPORT_PATH = "logs/thursday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v3.json"

# ------------------------- BANKROLLS -------------------------
BANKROLL_CORE = float(os.getenv("BANKROLL_CORE", "700"))
BANKROLL_FUN  = float(os.getenv("BANKROLL_FUN", "400"))

CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.30"))
FUN_EXPOSURE_CAP  = float(os.getenv("FUN_EXPOSURE_CAP", "0.20"))

ODDS_MATCH_MIN_SCORE_CORE = float(os.getenv("ODDS_MATCH_MIN_SCORE_CORE", "0.75"))
ODDS_MATCH_MIN_SCORE_FUN  = float(os.getenv("ODDS_MATCH_MIN_SCORE_FUN",  "0.65"))  # ✅ πιο “fun” ανοχή

# ------------------------- CORE RULES -------------------------
CORE_MIN_ODDS = 1.50
CORE_MAX_ODDS = 2.20
CORE_ALLOWED_MARKETS = {"Home", "Away", "Over 2.5", "Under 2.5"}
CORE_MAX_SINGLES = 10

CORE_MIN_VALUE_BANDS = {
    (1.50, 1.65): 4.0,
    (1.65, 2.00): 3.0,
    (2.00, 2.20): 2.5,
}

CORE_STAKE_BANDS = {
    (1.50, 1.65): 25.0,
    (1.65, 2.00): 20.0,
    (2.00, 2.20): 15.0,
}

CORE_DOUBLE_PREF_MAX_LEG_ODDS = 1.55
CORE_DOUBLE_MAX_LEG_ODDS      = 1.70
CORE_DOUBLE_TARGET_MIN        = 1.55
CORE_DOUBLE_TARGET_MAX        = 2.20

# ------------------------- FUN RULES -------------------------
FUN_ALLOWED_MARKETS = {"Home", "Draw", "Away", "Over 2.5", "Under 2.5"}

# ✅ FUN can be anything value-positive, but we still avoid total chaos:
FUN_MIN_ODDS_DEFAULT = 1.70
FUN_MAX_ODDS_BY_MARKET = {
    "Home": 3.20,
    "Away": 3.20,
    "Over 2.5": 3.20,
    "Under 2.5": 3.20,
    "Draw": 4.60,  # ✅ επιτρέπει Χ μέχρι ~4.6 (όχι 5.60 “όνειρο θερινής”)
}

FUN_MIN_VALUE_PCT = float(os.getenv("FUN_MIN_VALUE_PCT", "4.0"))  # value_adj threshold
FUN_MAX_PICKS_TOTAL = 10
FUN_SYSTEM_MAX_MATCHES = 8

# Singles selection
FUN_MAX_SINGLES = 7  # όπως πριν, αλλά τώρα επιλέγονται από τα 10
FUN_AVOID_CORE_OVERLAP = True

# Exposure split (so singles don't get crushed by system)
FUN_CAP_SPLIT_SINGLES = float(os.getenv("FUN_CAP_SPLIT_SINGLES", "0.45"))
FUN_CAP_SPLIT_SINGLES = max(0.10, min(0.80, FUN_CAP_SPLIT_SINGLES))

FUN_SYSTEM_UNIT_BASE = float(os.getenv("FUN_SYSTEM_UNIT", "0.31"))

MARKET_CODE = {
    "Home": "1",
    "Draw": "X",
    "Away": "2",
    "Over 2.5": "O25",
    "Under 2.5": "U25",
}

def safe_float(v, d=None):
    try:
        return float(v)
    except:
        return d

def band_lookup(x, band_map, default=None):
    for (a, b), val in band_map.items():
        if a <= x <= b:
            return val
    return default

def load_thursday_fixtures():
    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["fixtures"], data

def odds_match_ok(fx, min_score):
    om = fx.get("odds_match") or {}
    if not om.get("matched"):
        return False
    score = safe_float(om.get("score"), 0.0)
    return score >= min_score

def _get_over_value_and_penalty(fx):
    v_raw = fx.get("selection_value_pct_over")
    if v_raw is None:
        v_raw = fx.get("value_pct_over")
    v_raw = safe_float(v_raw, None)

    pen = safe_float(fx.get("over_value_penalty_pts"), 0.0) or 0.0

    if v_raw is None:
        return None, 0.0, None

    v_adj = v_raw - pen
    return v_raw, pen, v_adj

def build_rows(fixtures):
    rows = []
    markets = [
        ("Home", "home_prob", "fair_1", "offered_1", "value_pct_1"),
        ("Draw", "draw_prob", "fair_x", "offered_x", "value_pct_x"),
        ("Away", "away_prob", "fair_2", "offered_2", "value_pct_2"),
        ("Over 2.5", "over_2_5_prob", "fair_over_2_5", "offered_over_2_5", "value_pct_over"),
        ("Under 2.5", "under_2_5_prob", "fair_under_2_5", "offered_under_2_5", "value_pct_under"),
    ]

    for fx in fixtures:
        # CORE/FUN will check odds_match with different thresholds later
        over_v_raw, over_pen_pts, over_v_adj = _get_over_value_and_penalty(fx)

        for market, pkey, fkey, okey, vkey in markets:
            odds = safe_float(fx.get(okey))
            if not odds or odds <= 1:
                continue

            if market == "Over 2.5":
                if over_v_raw is None:
                    continue
                val_raw = over_v_raw
                val_adj = over_v_adj if over_v_adj is not None else val_raw
                pen_pts = over_pen_pts
            else:
                val_raw = safe_float(fx.get(vkey), None)
                if val_raw is None:
                    continue
                val_adj = val_raw
                pen_pts = 0.0

            rows.append({
                "fixture_id": fx.get("fixture_id"),
                "date": fx.get("date"),
                "time": fx.get("time"),
                "league": fx.get("league"),
                "match": f'{fx.get("home")} – {fx.get("away")}',
                "market": market,
                "market_code": MARKET_CODE[market],
                "prob": safe_float(fx.get(pkey), 0.0),
                "fair": safe_float(fx.get(fkey)),
                "odds": odds,
                "value_pct": val_raw,
                "value_adj": val_adj,
                "penalty_pts": pen_pts,
                "flags": fx.get("flags") or {},
                "odds_match": fx.get("odds_match") or {},
            })
    return rows

def scale_stakes(items, cap_amount, stake_key="stake"):
    total = sum(safe_float(x.get(stake_key), 0.0) for x in items)
    if total <= 0:
        return items, 0.0, total
    if total <= cap_amount:
        return items, 1.0, total

    s = cap_amount / total
    for x in items:
        x[stake_key] = round(safe_float(x.get(stake_key), 0.0) * s, 1)
    return items, s, total

# ------------------------- CORE PICKER -------------------------
def pick_core(rows, fixtures_by_id):
    core_candidates = []
    for r in rows:
        fx = fixtures_by_id.get(r["fixture_id"])
        if not fx or not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_CORE):
            continue

        if r["market"] not in CORE_ALLOWED_MARKETS:
            continue
        if r["odds"] < CORE_MIN_ODDS or r["odds"] > CORE_MAX_ODDS:
            continue

        min_val = band_lookup(r["odds"], CORE_MIN_VALUE_BANDS, default=9999)
        if r["value_adj"] < min_val:
            continue

        stake = band_lookup(r["odds"], CORE_STAKE_BANDS, default=0.0)
        if stake <= 0:
            continue

        core_candidates.append({**r, "stake": float(stake), "tag": "core"})

    core_candidates.sort(key=lambda x: (x["value_adj"], x["prob"]), reverse=True)

    core_singles = []
    used_matches = set()
    for r in core_candidates:
        if r["match"] in used_matches:
            continue
        core_singles.append(r)
        used_matches.add(r["match"])
        if len(core_singles) >= CORE_MAX_SINGLES:
            break

    cap_amount = BANKROLL_CORE * CORE_EXPOSURE_CAP
    core_singles, core_scale, core_base_total = scale_stakes(core_singles, cap_amount, "stake")

    def best_double_from_pool(pool):
        best = None
        best_score = -1e18
        for a, b in combinations(pool, 2):
            combo_odds = a["odds"] * b["odds"]
            if not (CORE_DOUBLE_TARGET_MIN <= combo_odds <= CORE_DOUBLE_TARGET_MAX):
                continue
            score = (a["value_adj"] + b["value_adj"]) + (a["prob"] + b["prob"]) * 10
            if score > best_score:
                best_score = score
                best = {
                    "legs": [
                        {"pick_id": f'{a["fixture_id"]}:{a["market_code"]}', "match": a["match"], "market": a["market"], "odds": a["odds"]},
                        {"pick_id": f'{b["fixture_id"]}:{b["market_code"]}', "match": b["match"], "market": b["market"], "odds": b["odds"]},
                    ],
                    "combo_odds": round(combo_odds, 2),
                    "tag": "core_double",
                }
        return best

    eligible_pref = [x for x in core_singles if x["odds"] <= CORE_DOUBLE_PREF_MAX_LEG_ODDS]
    eligible_fallback = [x for x in core_singles if x["odds"] <= CORE_DOUBLE_MAX_LEG_ODDS]
    core_double = best_double_from_pool(eligible_pref) or best_double_from_pool(eligible_fallback)

    open_amount = round(sum(x["stake"] for x in core_singles), 1)
    core_meta = {
        "bankroll": BANKROLL_CORE,
        "exposure_cap_pct": CORE_EXPOSURE_CAP,
        "open": open_amount,
        "after_open": round(BANKROLL_CORE - open_amount, 1),
        "picks_count": len(core_singles),
        "scale_applied": round(core_scale, 3),
    }
    return core_singles, core_double, core_meta

# ------------------------- FUN: STAKES -------------------------
def fun_single_stake(odds: float) -> float:
    """
    Stake decreases as odds rise.
    Keep it simple and sane.
    """
    if odds < 2.00:
        return 14.0
    if odds <= 2.30:
        return 12.0
    if odds <= 2.60:
        return 10.0
    if odds <= 3.00:
        return 8.0
    if odds <= 3.60:
        return 6.0
    if odds <= 4.60:
        return 4.0
    return 0.0

# ------------------------- FUN: SYSTEM SELECTION (EV APPROX) -------------------------
def pick_best_system(picks, max_r=5):
    """
    Choose system type by maximizing EV per 1€ unit (approx):
    For each combination, expected return per unit ~ Π(p_i * odds_i)
    Total EV per unit across all columns = Σ combo_return - columns
    EV_per_euro = EV / columns  (profit per 1€ spent)
    We try r sets (single r or mixed like 3-4-5) depending on n.
    """
    n = len(picks)
    if n < 3:
        return None

    # helper: compute EV for a set of r sizes
    def ev_for_sizes(sizes):
        cols = 0
        ev = 0.0
        for r in sizes:
            if r > n or r < 2:
                continue
            cols_r = comb(n, r)
            cols += cols_r
            for idxs in combinations(range(n), r):
                prod = 1.0
                for i in idxs:
                    # p*odds = expected multiple for that leg
                    prod *= (max(0.0001, picks[i]["prob"]) * max(1.01, picks[i]["odds"]))
                ev += prod
        ev_profit = ev - cols
        ev_per_euro = ev_profit / cols if cols > 0 else -1e9
        return ev_per_euro, ev_profit, cols

    # candidate systems by n
    candidates = []
    if n >= 6:
        candidates += [([3,4,5], f"3-4-5/{n}"), ([3,4], f"3-4/{n}"), ([2,3], f"2-3/{n}")]
    if n == 5:
        candidates += [([3,4,5], "3-4-5/5"), ([2,3], "2-3/5"), ([3], "3/5")]
    if n == 4:
        candidates += [([2,3,4], "2-3-4/4"), ([2,3], "2-3/4"), ([2], "2/4")]
    if n == 3:
        candidates += [([2,3], "2-3/3"), ([2], "2/3")]

    best = None
    best_score = -1e18
    for sizes, label in candidates:
        score, ev_profit, cols = ev_for_sizes(sizes)
        # guardrail: prefer positive EV per euro; if all negative, still choose "least bad"
        if score > best_score:
            best_score = score
            best = {
                "system": label,
                "sizes": sizes,
                "columns": cols,
                "ev_profit_per_unit": round(ev_profit, 4),
                "ev_per_euro": round(score, 6),
            }

    return best

# ------------------------- FUN PICKER -------------------------
def pick_fun(rows, fixtures_by_id, core_singles):
    core_fixture_ids = {x["fixture_id"] for x in core_singles}

    fun_candidates = []
    for r in rows:
        fx = fixtures_by_id.get(r["fixture_id"])
        if not fx or not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_FUN):
            continue

        if r["market"] not in FUN_ALLOWED_MARKETS:
            continue

        if FUN_AVOID_CORE_OVERLAP and r["fixture_id"] in core_fixture_ids:
            continue

        max_odds = FUN_MAX_ODDS_BY_MARKET.get(r["market"], 3.00)
        if r["odds"] < FUN_MIN_ODDS_DEFAULT or r["odds"] > max_odds:
            continue

        if r["value_adj"] < FUN_MIN_VALUE_PCT:
            continue

        fun_candidates.append(r)

    # best by value_adj, then prob
    fun_candidates.sort(key=lambda x: (x["value_adj"], x["prob"]), reverse=True)

    # take up to 10 UNIQUE matches (one market per match)
    fun_picks = []
    used_matches = set()
    for r in fun_candidates:
        if r["match"] in used_matches:
            continue
        fun_picks.append(r)
        used_matches.add(r["match"])
        if len(fun_picks) >= FUN_MAX_PICKS_TOTAL:
            break

    # ---- singles subset (top 7 from these 10) ----
    fun_singles = []
    for r in fun_picks[:FUN_MAX_SINGLES]:
        st = fun_single_stake(r["odds"])
        if st <= 0:
            continue
        fun_singles.append({
            "pick_id": f'{r["fixture_id"]}:{r["market_code"]}',
            "fixture_id": r["fixture_id"],
            "market_code": r["market_code"],
            "match": r["match"],
            "league": r["league"],
            "market": r["market"],
            "odds": r["odds"],
            "prob": r["prob"],
            "value_adj": r["value_adj"],
            "stake": float(st),
        })

    # ---- system pool: up to 8 matches ----
    system_pool = fun_picks[:FUN_SYSTEM_MAX_MATCHES]
    n_sys = len(system_pool)

    system_choice = pick_best_system(system_pool) if n_sys >= 3 else None
    cols = system_choice["columns"] if system_choice else 0
    system_label = system_choice["system"] if system_choice else None
    system_ev_per_euro = system_choice["ev_per_euro"] if system_choice else None

    # ---- bankroll caps (split singles/system) ----
    cap_amount = BANKROLL_FUN * FUN_EXPOSURE_CAP
    cap_singles = cap_amount * FUN_CAP_SPLIT_SINGLES
    cap_system  = cap_amount * (1.0 - FUN_CAP_SPLIT_SINGLES)

    singles_total = sum(x["stake"] for x in fun_singles)
    singles_scale = 1.0
    if singles_total > 0 and singles_total > cap_singles:
        singles_scale = cap_singles / singles_total
        for x in fun_singles:
            x["stake"] = round(x["stake"] * singles_scale, 1)

    base_unit = FUN_SYSTEM_UNIT_BASE
    base_system_stake = base_unit * cols
    system_scale = 1.0
    if base_system_stake > 0 and base_system_stake > cap_system:
        system_scale = cap_system / base_system_stake

    unit = round(base_unit * system_scale, 2) if cols > 0 else 0.0
    system_stake = round(unit * cols, 1) if cols > 0 else 0.0

    open_amount = round(system_stake + sum(x["stake"] for x in fun_singles), 1)

    payload = {
        "bankroll": BANKROLL_FUN,
        "exposure_cap_pct": FUN_EXPOSURE_CAP,
        "cap_split": {"singles_pct": round(FUN_CAP_SPLIT_SINGLES, 2), "system_pct": round(1.0 - FUN_CAP_SPLIT_SINGLES, 2)},
        "scales": {"singles_scale": round(singles_scale, 3), "system_scale": round(system_scale, 3)},
        "rules": {
            "odds_match_min_score_fun": ODDS_MATCH_MIN_SCORE_FUN,
            "min_value_pct": FUN_MIN_VALUE_PCT,
            "max_picks_total": FUN_MAX_PICKS_TOTAL,
            "max_system_matches": FUN_SYSTEM_MAX_MATCHES,
            "max_singles": FUN_MAX_SINGLES,
            "max_odds_by_market": FUN_MAX_ODDS_BY_MARKET,
            "min_odds_default": FUN_MIN_ODDS_DEFAULT,
            "avoid_core_overlap": FUN_AVOID_CORE_OVERLAP,
        },

        "picks_total": [
            {
                "pick_id": f'{r["fixture_id"]}:{r["market_code"]}',
                "fixture_id": r["fixture_id"],
                "market_code": r["market_code"],
                "match": r["match"],
                "league": r["league"],
                "market": r["market"],
                "prob": r["prob"],
                "fair": r["fair"],
                "odds": r["odds"],
                "value_pct": r["value_pct"],
                "value_adj": r["value_adj"],
                "penalty_pts": r.get("penalty_pts", 0.0),
            }
            for r in fun_picks
        ],

        "singles": fun_singles,

        "system_pool": [
            {
                "pick_id": f'{r["fixture_id"]}:{r["market_code"]}',
                "fixture_id": r["fixture_id"],
                "market_code": r["market_code"],
                "match": r["match"],
                "league": r["league"],
                "market": r["market"],
                "prob": r["prob"],
                "odds": r["odds"],
                "value_adj": r["value_adj"],
            }
            for r in system_pool
        ],

        "system": {
            "label": system_label,
            "columns": cols,
            "unit": unit,
            "stake": system_stake,
            "ev_per_euro": system_ev_per_euro,
        },

        "open": open_amount,
        "after_open": round(BANKROLL_FUN - open_amount, 1),
        "counts": {
            "picks_total": len(fun_picks),
            "singles": len(fun_singles),
            "system_pool": len(system_pool),
        },
        "avg": {
            "value_adj": round(sum(x["value_adj"] for x in fun_picks) / len(fun_picks), 2) if fun_picks else 0.0,
            "odds": round(sum(x["odds"] for x in fun_picks) / len(fun_picks), 2) if fun_picks else 0.0,
        },
    }
    return payload

def main():
    fixtures, th_meta = load_thursday_fixtures()
    fixtures_by_id = {fx.get("fixture_id"): fx for fx in fixtures}

    rows = build_rows(fixtures)

    core_singles, core_double, core_meta = pick_core(rows, fixtures_by_id)
    fun_payload = pick_fun(rows, fixtures_by_id, core_singles)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "window": th_meta.get("window", {}),
        "fixtures_total": th_meta.get("fixtures_total", len(fixtures)),

        "core": {
            "bankroll": BANKROLL_CORE,
            "exposure_cap_pct": CORE_EXPOSURE_CAP,
            "rules": {
                "odds_range": [CORE_MIN_ODDS, CORE_MAX_ODDS],
                "allowed_markets": sorted(list(CORE_ALLOWED_MARKETS)),
                "odds_match_min_score_core": ODDS_MATCH_MIN_SCORE_CORE,
                "min_value_bands": {
                    "1.50-1.65": CORE_MIN_VALUE_BANDS[(1.50, 1.65)],
                    "1.65-2.00": CORE_MIN_VALUE_BANDS[(1.65, 2.00)],
                    "2.00-2.20": CORE_MIN_VALUE_BANDS[(2.00, 2.20)],
                },
                "stake_bands": {
                    "1.50-1.65": CORE_STAKE_BANDS[(1.50, 1.65)],
                    "1.65-2.00": CORE_STAKE_BANDS[(1.65, 2.00)],
                    "2.00-2.20": CORE_STAKE_BANDS[(2.00, 2.20)],
                },
                "double_leg_pref_max_odds": CORE_DOUBLE_PREF_MAX_LEG_ODDS,
                "double_leg_max_odds": CORE_DOUBLE_MAX_LEG_ODDS,
                "double_target_combo_odds": [CORE_DOUBLE_TARGET_MIN, CORE_DOUBLE_TARGET_MAX],
            },
            "singles": [
                {
                    "pick_id": f'{x["fixture_id"]}:{x["market_code"]}',
                    "fixture_id": x["fixture_id"],
                    "market_code": x["market_code"],
                    "match": x["match"],
                    "league": x["league"],
                    "market": x["market"],
                    "prob": x["prob"],
                    "fair": x["fair"],
                    "odds": x["odds"],
                    "value_pct": x["value_pct"],
                    "value_adj": x["value_adj"],
                    "penalty_pts": x.get("penalty_pts", 0.0),
                    "stake": x["stake"],
                    "tag": x.get("tag", "core"),
                }
                for x in core_singles
            ],
            "double": core_double,
            "doubles": [core_double] if core_double else [],
            "open": core_meta["open"],
            "after_open": core_meta["after_open"],
            "picks_count": core_meta["picks_count"],
            "scale_applied": core_meta["scale_applied"],
        },

        "funbet": fun_payload,
    }

    os.makedirs("logs", exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
