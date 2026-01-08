# =========================
# FRIDAY SHORTLIST v3.6 — Core + Fun (singles + system) aligned with research thresholds
#
# Updates vs v3.5:
#  - Fun singles stakes increased (15 / 12 / 8 by odds)
#  - Fun system decision table: 3/5, 4/6, 5/7 (avoid 8; trim to 7)
#  - Under 2.5 strict gating: higher EV, higher p, tight_game + draw_prob + low total lambda
#  - Keeps EV per pick (prob*odds-1) in output (additive)
# =========================

import os
import json
from datetime import datetime
from itertools import combinations
from math import comb

THURSDAY_REPORT_PATH = os.getenv("THURSDAY_REPORT_PATH", "logs/thursday_report_v3.json")
FRIDAY_REPORT_PATH   = os.getenv("FRIDAY_REPORT_PATH",   "logs/friday_shortlist_v3.json")

# ------------------------- BANKROLLS -------------------------
BANKROLL_CORE = float(os.getenv("BANKROLL_CORE", "800"))
BANKROLL_FUN  = float(os.getenv("BANKROLL_FUN",  "400"))

CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.15"))  # ~110–120€
FUN_EXPOSURE_CAP  = float(os.getenv("FUN_EXPOSURE_CAP",  "0.20"))  # ~75–80€

# Odds-match thresholds
ODDS_MATCH_MIN_SCORE_CORE = float(os.getenv("ODDS_MATCH_MIN_SCORE_CORE", "0.75"))
ODDS_MATCH_MIN_SCORE_FUN  = float(os.getenv("ODDS_MATCH_MIN_SCORE_FUN",  "0.65"))

# Optional confidence gate (if Thursday sets flags.confidence)
CORE_MIN_CONFIDENCE = float(os.getenv("CORE_MIN_CONFIDENCE", "0.55"))
FUN_MIN_CONFIDENCE  = float(os.getenv("FUN_MIN_CONFIDENCE",  "0.45"))

# ------------------------- MARKETS -------------------------
CORE_ALLOWED_MARKETS = {"Home", "Away", "Over 2.5", "Under 2.5"}
FUN_ALLOWED_MARKETS  = {"Home", "Draw", "Away", "Over 2.5", "Under 2.5"}

MARKET_CODE = {
    "Home": "1",
    "Draw": "X",
    "Away": "2",
    "Over 2.5": "O25",
    "Under 2.5": "U25",
}

# ------------------------- CORE RULES -------------------------
CORE_SINGLES_MIN_ODDS = float(os.getenv("CORE_SINGLES_MIN_ODDS", "1.50"))
CORE_SINGLES_MAX_ODDS = float(os.getenv("CORE_SINGLES_MAX_ODDS", "1.75"))

# Low-odds legs for doubles
CORE_DOUBLE_LEG_MIN_ODDS = float(os.getenv("CORE_DOUBLE_LEG_MIN_ODDS", "1.20"))
CORE_DOUBLE_LEG_MAX_ODDS = float(os.getenv("CORE_DOUBLE_LEG_MAX_ODDS", "1.49"))

# Partner leg for doubles
CORE_DOUBLE_PARTNER_MIN_ODDS = float(os.getenv("CORE_DOUBLE_PARTNER_MIN_ODDS", "1.35"))
CORE_DOUBLE_PARTNER_MAX_ODDS = float(os.getenv("CORE_DOUBLE_PARTNER_MAX_ODDS", "1.70"))

# Double combo target
CORE_DOUBLE_TARGET_MIN = float(os.getenv("CORE_DOUBLE_TARGET_MIN", "1.75"))
CORE_DOUBLE_TARGET_MAX = float(os.getenv("CORE_DOUBLE_TARGET_MAX", "2.05"))

CORE_TARGET_SINGLES = int(os.getenv("CORE_TARGET_SINGLES", "4"))  # 3–5
CORE_MIN_SINGLES    = int(os.getenv("CORE_MIN_SINGLES",    "3"))
CORE_MAX_SINGLES    = int(os.getenv("CORE_MAX_SINGLES",    "5"))
CORE_MAX_DOUBLES    = int(os.getenv("CORE_MAX_DOUBLES",    "1"))

# Stakes: 30–50€ per single
CORE_STAKE_HIGH = float(os.getenv("CORE_STAKE_HIGH", "50"))
CORE_STAKE_MID  = float(os.getenv("CORE_STAKE_MID",  "40"))
CORE_STAKE_LOW  = float(os.getenv("CORE_STAKE_LOW",  "30"))

CORE_DOUBLE_STAKE = float(os.getenv("CORE_DOUBLE_STAKE", "25"))

# Value thresholds (bands)
CORE_MIN_VALUE_BANDS = {
    (1.20, 1.50): float(os.getenv("CORE_MIN_VALUE_120_150", "-5.0")),
    (1.50, 1.65): float(os.getenv("CORE_MIN_VALUE_150_165", "0.5")),
    (1.65, 1.75): float(os.getenv("CORE_MIN_VALUE_165_175", "0.5")),
    (1.75, 2.20): float(os.getenv("CORE_MIN_VALUE_175_220", "0.5")),
}

CORE_DOUBLE_LOWLEG_MIN_PROB = float(os.getenv("CORE_DOUBLE_LOWLEG_MIN_PROB", "0.70"))

# ------------------------- FUN THRESHOLDS (from research) -------------------------
EV_MIN_GENERAL = float(os.getenv("EV_MIN_GENERAL", "0.05"))      # +5%
EV_MIN_UNDER   = float(os.getenv("EV_MIN_UNDER",   "0.08"))      # +8%
P_MIN_HOME     = float(os.getenv("P_MIN_HOME",     "0.30"))
P_MIN_DRAW     = float(os.getenv("P_MIN_DRAW",     "0.25"))
P_MIN_AWAY     = float(os.getenv("P_MIN_AWAY",     "0.22"))
P_MIN_OVER     = float(os.getenv("P_MIN_OVER",     "0.40"))
P_MIN_UNDER    = float(os.getenv("P_MIN_UNDER",    "0.55"))

UNDER_DRAW_MIN = float(os.getenv("UNDER_DRAW_MIN", "0.30"))
UNDER_LTOTAL_MAX = float(os.getenv("UNDER_LTOTAL_MAX", "2.20"))

FUN_MIN_ODDS_DEFAULT = float(os.getenv("FUN_MIN_ODDS_DEFAULT", "1.85"))
FUN_MAX_ODDS_BY_MARKET = {
    "Home":  float(os.getenv("FUN_MAX_ODDS_HOME", "3.20")),
    "Away":  float(os.getenv("FUN_MAX_ODDS_AWAY", "3.20")),
    "Over 2.5":  float(os.getenv("FUN_MAX_ODDS_O25", "3.20")),
    "Under 2.5": float(os.getenv("FUN_MAX_ODDS_U25", "3.20")),
    "Draw":  float(os.getenv("FUN_MAX_ODDS_DRAW", "4.60")),
}

FUN_MAX_PICKS_TOTAL = int(os.getenv("FUN_MAX_PICKS_TOTAL", "8"))
FUN_MIN_PICKS_TOTAL = int(os.getenv("FUN_MIN_PICKS_TOTAL", "6"))

FUN_SINGLES_K = int(os.getenv("FUN_SINGLES_K", "4"))               # 3–5
FUN_SINGLES_MIN = int(os.getenv("FUN_SINGLES_MIN", "3"))
FUN_SINGLES_MAX = int(os.getenv("FUN_SINGLES_MAX", "5"))

# System pool (avoid 8)
FUN_SYSTEM_POOL_MAX = int(os.getenv("FUN_SYSTEM_POOL_MAX", "7"))
FUN_SYSTEM_POOL_MIN = int(os.getenv("FUN_SYSTEM_POOL_MIN", "5"))

# FUN stakes (UPDATED)
def fun_single_stake(odds: float) -> float:
    # 1.85–2.20 => 15, 2.20–3.50 => 12, 3.50–4.60 => 8
    if odds < 1.85:
        return 0.0
    if odds <= 2.20:
        return 15.0
    if odds <= 3.50:
        return 12.0
    if odds <= 4.60:
        return 8.0
    return 0.0

# More meaningful singles allocation
FUN_CAP_SPLIT_SINGLES = float(os.getenv("FUN_CAP_SPLIT_SINGLES", "0.70"))
FUN_CAP_SPLIT_SINGLES = max(0.10, min(0.90, FUN_CAP_SPLIT_SINGLES))

# System unit targeting
FUN_SYSTEM_UNIT_MIN  = float(os.getenv("FUN_SYSTEM_UNIT_MIN",  "0.50"))
FUN_SYSTEM_UNIT_MAX  = float(os.getenv("FUN_SYSTEM_UNIT_MAX",  "2.00"))
FUN_SYSTEM_TARGET_TOTAL = float(os.getenv("FUN_SYSTEM_TARGET_TOTAL", "15.0"))  # aim 10–15

FUN_AVOID_CORE_OVERLAP = os.getenv("FUN_AVOID_CORE_OVERLAP", "true").lower() == "true"

# ------------------------- HELPERS -------------------------
def safe_float(v, d=None):
    try:
        return float(v)
    except Exception:
        return d

def band_lookup(x, band_map, default=None):
    for (a, b), val in band_map.items():
        if a <= x <= b:
            return val
    return default

def odds_match_ok(fx, min_score):
    om = fx.get("odds_match") or {}
    if not om.get("matched"):
        return False
    score = safe_float(om.get("score"), 0.0)
    return score >= min_score

def confidence_ok(fx, min_conf):
    flags = fx.get("flags") or {}
    c = safe_float(flags.get("confidence"), None)
    if c is None:
        return True
    return c >= min_conf

def load_thursday():
    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["fixtures"], data

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

def ev(prob, odds):
    if prob is None or odds is None:
        return None
    p = safe_float(prob, None)
    o = safe_float(odds, None)
    if p is None or o is None:
        return None
    return round(p * o - 1.0, 4)

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
        over_v_raw, over_pen_pts, over_v_adj = _get_over_value_and_penalty(fx)

        for market, pkey, fkey, okey, vkey in markets:
            odds = safe_float(fx.get(okey))
            if not odds or odds <= 1.0:
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

            prob = safe_float(fx.get(pkey), None)

            rows.append({
                "fixture_id": fx.get("fixture_id"),
                "date": fx.get("date"),
                "time": fx.get("time"),
                "league": fx.get("league"),
                "match": f'{fx.get("home")} – {fx.get("away")}',
                "market": market,
                "market_code": MARKET_CODE[market],
                "prob": prob,
                "fair": safe_float(fx.get(fkey)),
                "odds": odds,
                "value_pct": val_raw,
                "value_adj": val_adj,
                "penalty_pts": pen_pts,
                "ev": ev(prob, odds),
                "flags": fx.get("flags") or {},
                "odds_match": fx.get("odds_match") or {},
            })
    return rows

def scale_stakes(items, cap_amount, key="stake"):
    total = sum(safe_float(x.get(key), 0.0) for x in items)
    if total <= 0:
        return items, 0.0, total
    if total <= cap_amount:
        return items, 1.0, total
    s = cap_amount / total
    for x in items:
        x[key] = round(safe_float(x.get(key), 0.0) * s, 1)
    return items, s, total

# ------------------------- CORE -------------------------
def core_single_stake(odds: float) -> float:
    if odds <= 1.55:
        return CORE_STAKE_HIGH
    if odds <= 1.65:
        return CORE_STAKE_MID
    return CORE_STAKE_LOW

def pick_core(rows, fixtures_by_id):
    candidates = []
    lowlegs = []

    for r in rows:
        fx = fixtures_by_id.get(r["fixture_id"])
        if not fx:
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_CORE):
            continue
        if not confidence_ok(fx, CORE_MIN_CONFIDENCE):
            continue
        if r["market"] not in CORE_ALLOWED_MARKETS:
            continue

        if CORE_DOUBLE_LEG_MIN_ODDS <= r["odds"] <= CORE_DOUBLE_LEG_MAX_ODDS:
            p = safe_float(r.get("prob"), 0.0) or 0.0
            if p < CORE_DOUBLE_LOWLEG_MIN_PROB:
                continue
            lowlegs.append(r)
            continue

        min_val = band_lookup(r["odds"], CORE_MIN_VALUE_BANDS, default=9999)
        if r["value_adj"] < min_val:
            continue

        if CORE_SINGLES_MIN_ODDS <= r["odds"] <= CORE_SINGLES_MAX_ODDS:
            candidates.append(r)

    candidates.sort(key=lambda x: (x["value_adj"], safe_float(x["prob"], 0.0)), reverse=True)
    lowlegs.sort(key=lambda x: (safe_float(x["prob"], 0.0), x["value_adj"]), reverse=True)

    singles = []
    used_matches = set()
    for r in candidates:
        if r["match"] in used_matches:
            continue
        singles.append({**r, "stake": float(core_single_stake(r["odds"])), "tag": "core"})
        used_matches.add(r["match"])
        if len(singles) >= min(CORE_TARGET_SINGLES, CORE_MAX_SINGLES):
            break

    if len(singles) < CORE_MIN_SINGLES:
        for r in candidates:
            if r["match"] in used_matches:
                continue
            singles.append({**r, "stake": float(core_single_stake(r["odds"])), "tag": "core"})
            used_matches.add(r["match"])
            if len(singles) >= CORE_MIN_SINGLES:
                break

    doubles = []
    if lowlegs and CORE_MAX_DOUBLES > 0:
        partner_pool = []
        for r in rows:
            fx = fixtures_by_id.get(r["fixture_id"])
            if not fx:
                continue
            if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_CORE):
                continue
            if not confidence_ok(fx, CORE_MIN_CONFIDENCE):
                continue
            if r["market"] not in CORE_ALLOWED_MARKETS:
                continue
            if r["odds"] < CORE_DOUBLE_PARTNER_MIN_ODDS or r["odds"] > CORE_DOUBLE_PARTNER_MAX_ODDS:
                continue
            min_val = band_lookup(r["odds"], CORE_MIN_VALUE_BANDS, default=9999)
            if r["value_adj"] < min_val:
                continue
            partner_pool.append(r)

        partner_pool.sort(key=lambda x: (x["value_adj"], safe_float(x["prob"], 0.0)), reverse=True)

        used = set()
        for leg1 in lowlegs:
            if len(doubles) >= CORE_MAX_DOUBLES:
                break
            for leg2 in partner_pool:
                if leg2["match"] == leg1["match"]:
                    continue
                if leg1["match"] in used or leg2["match"] in used:
                    continue
                combo_odds = leg1["odds"] * leg2["odds"]
                if not (CORE_DOUBLE_TARGET_MIN <= combo_odds <= CORE_DOUBLE_TARGET_MAX):
                    continue
                doubles.append({
                    "legs": [
                        {"pick_id": f'{leg1["fixture_id"]}:{leg1["market_code"]}', "match": leg1["match"], "market": leg1["market"], "odds": leg1["odds"]},
                        {"pick_id": f'{leg2["fixture_id"]}:{leg2["market_code"]}', "match": leg2["match"], "market": leg2["market"], "odds": leg2["odds"]},
                    ],
                    "combo_odds": round(combo_odds, 2),
                    "stake": round(float(CORE_DOUBLE_STAKE), 1),
                    "tag": "core_double",
                })
                used.add(leg1["match"])
                used.add(leg2["match"])
                break

    cap_amount = BANKROLL_CORE * CORE_EXPOSURE_CAP
    open_total = sum(x["stake"] for x in singles) + sum(d.get("stake", 0.0) for d in doubles)
    scale_applied = 1.0

    if open_total > cap_amount and singles:
        remaining = max(0.0, cap_amount - sum(d.get("stake", 0.0) for d in doubles))
        if remaining <= 0:
            remaining = cap_amount
        singles, s1, _ = scale_stakes(singles, remaining, "stake")
        scale_applied = s1 if s1 else 1.0

    open_total = sum(x["stake"] for x in singles) + sum(d.get("stake", 0.0) for d in doubles)
    if open_total > cap_amount and doubles:
        s = cap_amount / open_total if open_total > 0 else 1.0
        for x in singles:
            x["stake"] = round(x["stake"] * s, 1)
        for d in doubles:
            d["stake"] = round(float(d.get("stake", 0.0)) * s, 1)
        scale_applied = round(scale_applied * s, 3)
        open_total = sum(x["stake"] for x in singles) + sum(d.get("stake", 0.0) for d in doubles)

    meta = {
        "bankroll": BANKROLL_CORE,
        "exposure_cap_pct": CORE_EXPOSURE_CAP,
        "open": round(open_total, 1),
        "after_open": round(BANKROLL_CORE - open_total, 1),
        "picks_count": len(singles),
        "scale_applied": round(scale_applied, 3),
    }

    core_double = doubles[0] if doubles else None
    return singles, core_double, doubles, meta

# ------------------------- FUN SYSTEM (decision table) -------------------------
def _columns_for_sizes(n: int, sizes):
    if not sizes or n <= 0:
        return 0
    return sum(comb(n, r) for r in sizes if 1 <= r <= n)

def choose_system_type(n: int):
    if n >= 7:
        return [5], f"5/{n}"
    if n == 6:
        return [4], "4/6"
    if n == 5:
        return [3], "3/5"
    if n == 4:
        return [3], "3/4"
    if n == 3:
        return [2], "2/3"
    return None, None

def pick_fun(rows, fixtures_by_id, core_singles):
    core_fixture_ids = {x["fixture_id"] for x in core_singles}

    fun_candidates = []
    for r in rows:
        fx = fixtures_by_id.get(r["fixture_id"])
        if not fx:
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_FUN):
            continue
        if not confidence_ok(fx, FUN_MIN_CONFIDENCE):
            continue

        if r["market"] not in FUN_ALLOWED_MARKETS:
            continue
        if FUN_AVOID_CORE_OVERLAP and r["fixture_id"] in core_fixture_ids:
            continue

        max_odds = FUN_MAX_ODDS_BY_MARKET.get(r["market"], 3.00)
        if r["odds"] < FUN_MIN_ODDS_DEFAULT or r["odds"] > max_odds:
            continue

        p = safe_float(r.get("prob"), None)
        e = safe_float(r.get("ev"), None)
        if p is None or e is None:
            continue

        # Market-specific thresholds
        if r["market"] == "Home":
            if e < EV_MIN_GENERAL or p < P_MIN_HOME:
                continue
        elif r["market"] == "Draw":
            if e < EV_MIN_GENERAL or p < P_MIN_DRAW:
                continue
        elif r["market"] == "Away":
            if e < EV_MIN_GENERAL or p < P_MIN_AWAY:
                continue
        elif r["market"] == "Over 2.5":
            if e < EV_MIN_GENERAL or p < P_MIN_OVER:
                continue
        elif r["market"] == "Under 2.5":
            if e < EV_MIN_UNDER or p < P_MIN_UNDER:
                continue
            flags = fx.get("flags") or {}
            ltot = (safe_float(fx.get("lambda_home"), 0.0) or 0.0) + (safe_float(fx.get("lambda_away"), 0.0) or 0.0)
            if not flags.get("tight_game", False):
                continue
            if safe_float(fx.get("draw_prob"), 0.0) < UNDER_DRAW_MIN:
                continue
            if ltot > UNDER_LTOTAL_MAX:
                continue

        fun_candidates.append(r)

    fun_candidates.sort(key=lambda x: (x["value_adj"], safe_float(x["prob"], 0.0)), reverse=True)

    # take up to 8 unique matches
    fun_picks = []
    used_matches = set()
    for r in fun_candidates:
        if r["match"] in used_matches:
            continue
        fun_picks.append(r)
        used_matches.add(r["match"])
        if len(fun_picks) >= FUN_MAX_PICKS_TOTAL:
            break

    # Singles: top K (3–5)
    k = max(FUN_SINGLES_MIN, min(FUN_SINGLES_MAX, FUN_SINGLES_K))
    fun_singles = []
    for r in fun_picks[:k]:
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
            "ev": r.get("ev"),
            "stake": float(st),
        })

    # System pool: 5–7, trim to 7
    system_pool = fun_picks[:min(max(FUN_SYSTEM_POOL_MIN, len(fun_picks)), FUN_SYSTEM_POOL_MAX)]
    if len(system_pool) > 7:
        system_pool = system_pool[:7]

    sizes, system_label = choose_system_type(len(system_pool))
    cols = _columns_for_sizes(len(system_pool), sizes) if sizes else 0

    # bankroll caps split
    cap_amount = BANKROLL_FUN * FUN_EXPOSURE_CAP
    cap_singles = cap_amount * FUN_CAP_SPLIT_SINGLES
    cap_system  = cap_amount * (1.0 - FUN_CAP_SPLIT_SINGLES)

    # scale singles
    singles_total = sum(x["stake"] for x in fun_singles)
    singles_scale = 1.0
    if singles_total > 0 and singles_total > cap_singles:
        singles_scale = cap_singles / singles_total
        for x in fun_singles:
            x["stake"] = round(x["stake"] * singles_scale, 1)

    # system unit targeting total 10–15
    unit = 0.0
    system_stake = 0.0
    system_scale = 1.0
    if cols > 0:
        target_total = min(FUN_SYSTEM_TARGET_TOTAL, cap_system)
        base_unit = (target_total / cols) if cols > 0 else 0.0
        base_unit = max(FUN_SYSTEM_UNIT_MIN, min(FUN_SYSTEM_UNIT_MAX, base_unit))
        base_system_stake = base_unit * cols
        if base_system_stake > cap_system and base_system_stake > 0:
            system_scale = cap_system / base_system_stake
        unit = round(base_unit * system_scale, 2)
        system_stake = round(unit * cols, 1)

    open_amount = round(system_stake + sum(x["stake"] for x in fun_singles), 1)

    payload = {
        "bankroll": BANKROLL_FUN,
        "exposure_cap_pct": FUN_EXPOSURE_CAP,
        "cap_split": {"singles_pct": round(FUN_CAP_SPLIT_SINGLES, 2), "system_pct": round(1.0 - FUN_CAP_SPLIT_SINGLES, 2)},
        "scales": {"singles_scale": round(singles_scale, 3), "system_scale": round(system_scale, 3)},
        "rules": {
            "thresholds": {
                "ev_min_general": EV_MIN_GENERAL,
                "ev_min_under": EV_MIN_UNDER,
                "p_min_home": P_MIN_HOME,
                "p_min_draw": P_MIN_DRAW,
                "p_min_away": P_MIN_AWAY,
                "p_min_over": P_MIN_OVER,
                "p_min_under": P_MIN_UNDER,
                "under_draw_min": UNDER_DRAW_MIN,
                "under_ltotal_max": UNDER_LTOTAL_MAX,
            },
            "system_decision_table": True,
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
                "ev": r.get("ev"),
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
                "ev": r.get("ev"),
            }
            for r in system_pool
        ],

        "system": {
            "label": system_label,
            "columns": cols,
            "unit": unit,
            "stake": system_stake,
            "ev_per_euro": None,
        },

        "open": open_amount,
        "after_open": round(BANKROLL_FUN - open_amount, 1),
        "counts": {"picks_total": len(fun_picks), "singles": len(fun_singles), "system_pool": len(system_pool)},
    }
    return payload

# ------------------------- MAIN -------------------------
def main():
    fixtures, th_meta = load_thursday()
    fixtures_by_id = {fx.get("fixture_id"): fx for fx in fixtures}

    rows = build_rows(fixtures)

    core_singles, core_double, core_doubles, core_meta = pick_core(rows, fixtures_by_id)
    fun_payload = pick_fun(rows, fixtures_by_id, core_singles)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "window": th_meta.get("window", {}),
        "fixtures_total": th_meta.get("fixtures_total", len(fixtures)),

        "core": {
            "bankroll": BANKROLL_CORE,
            "exposure_cap_pct": CORE_EXPOSURE_CAP,
            "rules": {
                "singles_odds_range": [CORE_SINGLES_MIN_ODDS, CORE_SINGLES_MAX_ODDS],
                "double_low_leg_odds_range": [CORE_DOUBLE_LEG_MIN_ODDS, CORE_DOUBLE_LEG_MAX_ODDS],
                "double_partner_odds_range": [CORE_DOUBLE_PARTNER_MIN_ODDS, CORE_DOUBLE_PARTNER_MAX_ODDS],
                "double_target_combo_odds": [CORE_DOUBLE_TARGET_MIN, CORE_DOUBLE_TARGET_MAX],
                "allowed_markets": sorted(list(CORE_ALLOWED_MARKETS)),
                "odds_match_min_score_core": ODDS_MATCH_MIN_SCORE_CORE,
                "min_confidence": CORE_MIN_CONFIDENCE,
                "stake_plan": "Singles 30–50€ (odds-weighted) + doubles only to carry <1.50 legs",
                "double_stake": CORE_DOUBLE_STAKE,
                "min_value_bands": {f"{a:.2f}-{b:.2f}": v for (a,b),v in CORE_MIN_VALUE_BANDS.items()},
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
                    "ev": x.get("ev"),
                    "penalty_pts": x.get("penalty_pts", 0.0),
                    "stake": x["stake"],
                    "tag": x.get("tag", "core"),
                }
                for x in core_singles
            ],
            "double": core_double,
            "doubles": core_doubles,
            "open": core_meta["open"],
            "after_open": core_meta["after_open"],
            "picks_count": core_meta["picks_count"],
            "scale_applied": core_meta["scale_applied"],
        },

        "funbet": fun_payload,
    }

    os.makedirs("logs", exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
