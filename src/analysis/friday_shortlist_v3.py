# =========================
# FRIDAY SHORTLIST v3 — FIXED
# - CORE singles stakes: 15/20/25
# - CORE exposure cap: 0.30 (so stakes don't get crushed)
# - CORE doubles: allow up to 1.70 per leg
# - FUN singles stakes remain boosted (8/7/6) + cap scaling
# - Outputs report compatible with your previous JSON structure
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
BANKROLL_FUN = float(os.getenv("BANKROLL_FUN", "400"))

# ✅ FIX: core cap raised so 15/20/25 doesn't get scaled down to 10-12
CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.30"))
FUN_EXPOSURE_CAP = float(os.getenv("FUN_EXPOSURE_CAP", "0.20"))

ODDS_MATCH_MIN_SCORE = float(os.getenv("ODDS_MATCH_MIN_SCORE", "0.75"))

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

# ✅ FIX: requested stake bands (final targets)
CORE_STAKE_BANDS = {
    (1.50, 1.65): 25.0,
    (1.65, 2.00): 20.0,
    (2.00, 2.20): 15.0,
}

# ✅ FIX: allow real doubles
CORE_DOUBLE_MAX_LEG_ODDS = 1.70
CORE_DOUBLE_TARGET_MIN = 1.55
CORE_DOUBLE_TARGET_MAX = 2.20

# ------------------------- FUN RULES -------------------------
FUN_MIN_ODDS = 2.00
FUN_MAX_ODDS = 3.00
FUN_MIN_VALUE_PCT = 5.0
FUN_MAX_PICKS = 8
FUN_MAX_SINGLES = 7

FUN_ALLOWED_MARKETS = {"Home", "Draw", "Away", "Over 2.5", "Under 2.5"}
FUN_AVOID_CORE_OVERLAP = True

# ✅ FIX: boosted fun single stakes (no more peanuts)
FUN_SINGLE_STAKE_BANDS = {
    (2.00, 2.30): 8.0,
    (2.30, 2.60): 7.0,
    (2.60, 3.00): 6.0,
}

# FUN system settings (keep your 3-4-5/8 structure)
FUN_SYSTEM = "3-4-5/8"
FUN_BASE_UNIT = float(os.getenv("FUN_SYSTEM_UNIT", "0.31"))  # was ~0.31 in your report

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

def odds_match_ok(fx):
    om = fx.get("odds_match") or {}
    if not om.get("matched"):
        return False
    score = safe_float(om.get("score"), 0.0)
    return score >= ODDS_MATCH_MIN_SCORE

def build_rows(fixtures):
    rows = []
    for fx in fixtures:
        # skip if odds matching is weak (keeps garbage out)
        if not odds_match_ok(fx):
            continue

        markets = [
            ("Home", "home_prob", "fair_1", "offered_1", "value_pct_1"),
            ("Draw", "draw_prob", "fair_x", "offered_x", "value_pct_x"),
            ("Away", "away_prob", "fair_2", "offered_2", "value_pct_2"),
            ("Over 2.5", "over_2_5_prob", "fair_over_2_5", "offered_over_2_5", "value_pct_over"),
            ("Under 2.5", "under_2_5_prob", "fair_under_2_5", "offered_under_2_5", "value_pct_under"),
        ]

        for market, pkey, fkey, okey, vkey in markets:
            val = fx.get(vkey)
            if val is None:
                continue

            odds = safe_float(fx.get(okey))
            if not odds or odds <= 1:
                continue

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
                "value_pct": safe_float(val, 0.0),
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

def pick_core(rows):
    # filter core candidates
    core_candidates = []
    for r in rows:
        if r["market"] not in CORE_ALLOWED_MARKETS:
            continue
        if r["odds"] < CORE_MIN_ODDS or r["odds"] > CORE_MAX_ODDS:
            continue

        min_val = band_lookup(r["odds"], CORE_MIN_VALUE_BANDS, default=9999)
        if r["value_pct"] < min_val:
            continue

        stake = band_lookup(r["odds"], CORE_STAKE_BANDS, default=0.0)
        if stake <= 0:
            continue

        core_candidates.append({**r, "stake": float(stake), "tag": "core"})

    # sort by value_pct desc, then prob desc
    core_candidates.sort(key=lambda x: (x["value_pct"], x["prob"]), reverse=True)

    # keep unique matches
    core_singles = []
    used_matches = set()
    for r in core_candidates:
        if r["match"] in used_matches:
            continue
        core_singles.append(r)
        used_matches.add(r["match"])
        if len(core_singles) >= CORE_MAX_SINGLES:
            break

    # scale to exposure cap
    cap_amount = BANKROLL_CORE * CORE_EXPOSURE_CAP
    core_singles, core_scale, core_base_total = scale_stakes(core_singles, cap_amount, "stake")

    # build a core double (best value pair) from eligible legs
    eligible_for_double = [x for x in core_singles if x["odds"] <= CORE_DOUBLE_MAX_LEG_ODDS]
    best_double = None
    best_score = -1e9

    for a, b in combinations(eligible_for_double, 2):
        combo_odds = a["odds"] * b["odds"]
        if not (CORE_DOUBLE_TARGET_MIN <= combo_odds <= CORE_DOUBLE_TARGET_MAX):
            continue
        score = (a["value_pct"] + b["value_pct"]) + (a["prob"] + b["prob"]) * 10
        if score > best_score:
            best_score = score
            best_double = {
                "legs": [
                    {"pick_id": f'{a["fixture_id"]}:{a["market_code"]}', "match": a["match"], "market": a["market"], "odds": a["odds"]},
                    {"pick_id": f'{b["fixture_id"]}:{b["market_code"]}', "match": b["match"], "market": b["market"], "odds": b["odds"]},
                ],
                "combo_odds": round(combo_odds, 2),
                "tag": "core_double",
            }

    open_amount = round(sum(x["stake"] for x in core_singles), 1)
    return core_singles, best_double, {
        "bankroll": BANKROLL_CORE,
        "exposure_cap_pct": CORE_EXPOSURE_CAP,
        "exposure_scale_applied": round(open_amount / (core_base_total if core_base_total else open_amount), 3) if open_amount else 0.0,
        "open": open_amount,
        "after_open": round(BANKROLL_CORE - open_amount, 1),
        "picks_count": len(core_singles),
    }

def pick_fun(rows, core_singles):
    core_fixture_ids = {x["fixture_id"] for x in core_singles}

    fun_candidates = []
    for r in rows:
        if r["market"] not in FUN_ALLOWED_MARKETS:
            continue
        if r["odds"] < FUN_MIN_ODDS or r["odds"] > FUN_MAX_ODDS:
            continue
        if r["value_pct"] < FUN_MIN_VALUE_PCT:
            continue
        if FUN_AVOID_CORE_OVERLAP and r["fixture_id"] in core_fixture_ids:
            continue

        fun_candidates.append(r)

    fun_candidates.sort(key=lambda x: (x["value_pct"], x["prob"]), reverse=True)

    fun_picks = []
    used_matches = set()
    for r in fun_candidates:
        if r["match"] in used_matches:
            continue
        fun_picks.append(r)
        used_matches.add(r["match"])
        if len(fun_picks) >= FUN_MAX_PICKS:
            break

    # fun singles (max 7)
    fun_singles = []
    for r in fun_picks[:FUN_MAX_SINGLES]:
        stake = band_lookup(r["odds"], FUN_SINGLE_STAKE_BANDS, default=0.0)
        if stake <= 0:
            continue
        fun_singles.append({
            "pick_id": f'{r["fixture_id"]}:{r["market_code"]}',
            "fixture_id": r["fixture_id"],
            "market_code": r["market_code"],
            "match": r["match"],
            "league": r["league"],
            "market": r["market"],
            "odds": r["odds"],
            "stake": float(stake),
        })

    # system columns for 3-4-5/8
    n = len(fun_picks)
    cols = 0
    if n >= 5:
        cols = comb(n, 3) + comb(n, 4) + comb(n, 5)

    base_unit = FUN_BASE_UNIT
    base_system_stake = base_unit * cols
    base_singles_stake = sum(x["stake"] for x in fun_singles)
    base_total = base_system_stake + base_singles_stake

    cap_amount = BANKROLL_FUN * FUN_EXPOSURE_CAP
    scale = 1.0
    if base_total > 0 and base_total > cap_amount:
        scale = cap_amount / base_total

    unit = round(base_unit * scale, 2)
    total_stake = round(unit * cols, 1)

    # scale fun singles too
    for x in fun_singles:
        x["stake"] = round(x["stake"] * scale, 1)

    open_amount = round(total_stake + sum(x["stake"] for x in fun_singles), 1)

    fun_payload = {
        "bankroll": BANKROLL_FUN,
        "exposure_cap_pct": FUN_EXPOSURE_CAP,
        "exposure_scale_applied": round(scale, 3),
        "rules": {
            "odds_min": FUN_MIN_ODDS,
            "odds_max": FUN_MAX_ODDS,
            "min_value_pct": FUN_MIN_VALUE_PCT,
            "max_picks": FUN_MAX_PICKS,
            "allowed_markets": sorted(list(FUN_ALLOWED_MARKETS)),
            "single_stake_bands": {
                "2.00-2.30": FUN_SINGLE_STAKE_BANDS[(2.00, 2.30)],
                "2.30-2.60": FUN_SINGLE_STAKE_BANDS[(2.30, 2.60)],
                "2.60-3.00": FUN_SINGLE_STAKE_BANDS[(2.60, 3.00)],
            },
            "system_unit_eur_range": [1, 3],
            "avoid_core_overlap": FUN_AVOID_CORE_OVERLAP,
            "odds_match_min_score": ODDS_MATCH_MIN_SCORE,
        },
        "system": FUN_SYSTEM,
        "columns": cols,
        "unit": unit,
        "total_stake": open_amount,  # overall fun exposure (system + singles)
        "picks": [
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
            }
            for r in fun_picks
        ],
        "singles": fun_singles,
        "open": open_amount,
        "after_open": round(BANKROLL_FUN - open_amount, 1),
        "picks_count": len(fun_picks),
        "singles_count": len(fun_singles),
        "avg_value": round(sum(x["value_pct"] for x in fun_picks) / len(fun_picks), 2) if fun_picks else 0.0,
        "avg_odds": round(sum(x["odds"] for x in fun_picks) / len(fun_picks), 2) if fun_picks else 0.0,
    }
    return fun_payload

def main():
    fixtures, th_meta = load_thursday_fixtures()
    rows = build_rows(fixtures)

    core_singles, core_double, core_meta = pick_core(rows)
    fun_payload = pick_fun(rows, core_singles)

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
                "double_leg_max_odds": CORE_DOUBLE_MAX_LEG_ODDS,
                "double_target_combo_odds": [CORE_DOUBLE_TARGET_MIN, CORE_DOUBLE_TARGET_MAX],
                "odds_match_min_score": ODDS_MATCH_MIN_SCORE,
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
        },
        "funbet": fun_payload,
    }

    os.makedirs("logs", exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
