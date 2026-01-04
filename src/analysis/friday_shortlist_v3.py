
# =========================
# FRIDAY SHORTLIST v3.5 — Core + Fun (singles + system) with sane odds bands
#
# Goals (per latest rules):
#  CORE:
#   - Singles odds: 1.50–1.75 (default)
#   - If a strong pick is <1.50, it is NOT played single by default; it is used as a DOUBLE leg
#   - Optional double is created ONLY if at least one leg is <1.50 (default behavior)
#   - Stakes: 30–50€ per single, double stake configurable
#
#  FUN:
#   - Picks are ranked by value_adj, then prob
#   - Singles: top K (default 4) BUT they can ALSO be included in the system pool (overlap allowed)
#   - System pool: 5–6 picks by default (will shrink if coverage is bad)
#   - System type chosen by "coverage" (conservative breakeven-style check using odds only)
#
# Output:
#   - Reads logs/thursday_report_v3.json
#   - Writes logs/friday_shortlist_v3.json
# Schema-safe:
#   - Keeps the same top-level keys/structure used by previous v3 outputs.
#   - Adds optional fields (like "ev") without breaking existing consumers.
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

# Exposure caps (target: core open ~110–120 on 800, fun open ~75–80 on 400)
CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.15"))
FUN_EXPOSURE_CAP  = float(os.getenv("FUN_EXPOSURE_CAP",  "0.20"))

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

# Low-odds legs for doubles (the "don't miss <1.50" bucket)
CORE_DOUBLE_LEG_MIN_ODDS = float(os.getenv("CORE_DOUBLE_LEG_MIN_ODDS", "1.20"))
CORE_DOUBLE_LEG_MAX_ODDS = float(os.getenv("CORE_DOUBLE_LEG_MAX_ODDS", "1.49"))

# Partner leg for doubles (can be low or normal single-range)
CORE_DOUBLE_PARTNER_MIN_ODDS = float(os.getenv("CORE_DOUBLE_PARTNER_MIN_ODDS", "1.35"))
CORE_DOUBLE_PARTNER_MAX_ODDS = float(os.getenv("CORE_DOUBLE_PARTNER_MAX_ODDS", "1.70"))

# Double combo target
CORE_DOUBLE_TARGET_MIN = float(os.getenv("CORE_DOUBLE_TARGET_MIN", "1.75"))
CORE_DOUBLE_TARGET_MAX = float(os.getenv("CORE_DOUBLE_TARGET_MAX", "2.05"))

# Counts
CORE_TARGET_SINGLES = int(os.getenv("CORE_TARGET_SINGLES", "4"))  # 3–4 typical
CORE_MIN_SINGLES    = int(os.getenv("CORE_MIN_SINGLES",    "3"))
CORE_MAX_SINGLES    = int(os.getenv("CORE_MAX_SINGLES",    "5"))

CORE_MAX_DOUBLES    = int(os.getenv("CORE_MAX_DOUBLES",    "1"))  # keep it tight

# Stakes: 30–50€ per single; double stake 20–35€ (scaled into cap)
CORE_STAKE_HIGH = float(os.getenv("CORE_STAKE_HIGH", "50"))  # lowest odds
CORE_STAKE_MID  = float(os.getenv("CORE_STAKE_MID",  "40"))
CORE_STAKE_LOW  = float(os.getenv("CORE_STAKE_LOW",  "30"))  # highest odds in core range

CORE_DOUBLE_STAKE = float(os.getenv("CORE_DOUBLE_STAKE", "25"))

# Value thresholds (kept as bands)
CORE_MIN_VALUE_BANDS = {
    # CORE is "tight on odds, loose on edge": keep value gates minimal.
    (1.20, 1.50): float(os.getenv("CORE_MIN_VALUE_120_150", "-5.0")),
    (1.50, 1.65): float(os.getenv("CORE_MIN_VALUE_150_165", "0.5")),
    (1.65, 1.75): float(os.getenv("CORE_MIN_VALUE_165_175", "0.5")),
    (1.75, 2.20): float(os.getenv("CORE_MIN_VALUE_175_220", "0.5")),
}

CORE_DOUBLE_LOWLEG_MIN_PROB = float(os.getenv("CORE_DOUBLE_LOWLEG_MIN_PROB", "0.70"))
CORE_DOUBLE_LOWLEG_MIN_CONF_BAND = os.getenv("CORE_DOUBLE_LOWLEG_MIN_CONF_BAND", "mid")  # low/mid/high

# Optional: market diversity cap (set to 99 to disable)
CORE_MAX_OVERS = int(os.getenv("CORE_MAX_OVERS", "99"))

# ------------------------- FUN RULES -------------------------
FUN_MIN_ODDS_DEFAULT = float(os.getenv("FUN_MIN_ODDS_DEFAULT", "1.85"))
FUN_MAX_ODDS_BY_MARKET = {
    "Home":  float(os.getenv("FUN_MAX_ODDS_HOME", "3.20")),
    "Away":  float(os.getenv("FUN_MAX_ODDS_AWAY", "3.20")),
    "Over 2.5":  float(os.getenv("FUN_MAX_ODDS_O25", "3.20")),
    "Under 2.5": float(os.getenv("FUN_MAX_ODDS_U25", "3.20")),
    "Draw":  float(os.getenv("FUN_MAX_ODDS_DRAW", "4.60")),
}

FUN_MIN_VALUE_PCT = float(os.getenv("FUN_MIN_VALUE_PCT", "4.0"))
FUN_MIN_PROB      = float(os.getenv("FUN_MIN_PROB", "0.22"))

FUN_MAX_PICKS_TOTAL = int(os.getenv("FUN_MAX_PICKS_TOTAL", "8"))   # target 7–8, not 10
FUN_MIN_PICKS_TOTAL = int(os.getenv("FUN_MIN_PICKS_TOTAL", "6"))

FUN_SINGLES_K = int(os.getenv("FUN_SINGLES_K", "4"))               # 3–5 typical
FUN_SINGLES_MIN = int(os.getenv("FUN_SINGLES_MIN", "3"))
FUN_SINGLES_MAX = int(os.getenv("FUN_SINGLES_MAX", "5"))

# System pool size (overlap with singles allowed)
FUN_SYSTEM_POOL_MAX = int(os.getenv("FUN_SYSTEM_POOL_MAX", "6"))   # keep breakeven sane
FUN_SYSTEM_POOL_MIN = int(os.getenv("FUN_SYSTEM_POOL_MIN", "5"))

# FUN stakes
def fun_single_stake(odds: float) -> float:
    # keep current behavior: 6–8€
    if odds <= 2.30:
        return 8.0
    if odds <= 3.20:
        return 7.0
    if odds <= 4.60:
        return 6.0
    return 0.0

FUN_CAP_SPLIT_SINGLES = float(os.getenv("FUN_CAP_SPLIT_SINGLES", "0.40"))
FUN_CAP_SPLIT_SINGLES = max(0.10, min(0.80, FUN_CAP_SPLIT_SINGLES))

FUN_SYSTEM_UNIT_BASE = float(os.getenv("FUN_SYSTEM_UNIT_BASE", "1.0"))
FUN_SYSTEM_UNIT_MIN  = float(os.getenv("FUN_SYSTEM_UNIT_MIN",  "0.50"))
FUN_SYSTEM_UNIT_MAX  = float(os.getenv("FUN_SYSTEM_UNIT_MAX",  "1.00"))

FUN_AVOID_CORE_OVERLAP = os.getenv("FUN_AVOID_CORE_OVERLAP", "true").lower() == "true"

# System coverage thresholds (conservative)
# We evaluate payout/cost ratios at k hits using the k LOWEST odds in pool.
SYS_TARGET_RATIO_AT_R    = float(os.getenv("SYS_TARGET_RATIO_AT_R",    "0.50"))
SYS_TARGET_RATIO_AT_RP1  = float(os.getenv("SYS_TARGET_RATIO_AT_RP1",  "0.90"))

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
    # 30–50, lower odds -> higher stake
    if odds <= 1.55:
        return CORE_STAKE_HIGH
    if odds <= 1.65:
        return CORE_STAKE_MID
    return CORE_STAKE_LOW

def pick_core(rows, fixtures_by_id):
    # candidate singles
    candidates = []
    lowlegs = []

    overs_count = 0

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

        min_val = band_lookup(r["odds"], CORE_MIN_VALUE_BANDS, default=9999)
        # For CORE singles we still apply a minimal edge gate.
        # For CORE low-odds legs (<1.50), we prioritize "high probability" over value.
        if CORE_DOUBLE_LEG_MIN_ODDS <= r["odds"] <= CORE_DOUBLE_LEG_MAX_ODDS:
            p = safe_float(r.get("prob"), 0.0) or 0.0
            if p < CORE_DOUBLE_LOWLEG_MIN_PROB:
                continue
            lowlegs.append(r)
            continue

        if r["value_adj"] < min_val:
            continue


        if CORE_SINGLES_MIN_ODDS <= r["odds"] <= CORE_SINGLES_MAX_ODDS:
            # optional overs cap
            if r["market"] == "Over 2.5" and overs_count >= CORE_MAX_OVERS:
                continue
            candidates.append(r)
            if r["market"] == "Over 2.5":
                overs_count += 1

    # sort by value_adj then prob
    def _conf(x):
        return safe_float((x.get("flags") or {}).get("confidence"), 0.0)
    candidates.sort(key=lambda x: (x["value_adj"], _conf(x), safe_float(x["prob"], 0.0)), reverse=True)
    lowlegs.sort(key=lambda x: (x["value_adj"], _conf(x), safe_float(x["prob"], 0.0)), reverse=True)

    # pick singles (unique matches)
    singles = []
    used_matches = set()
    for r in candidates:
        if r["match"] in used_matches:
            continue
        singles.append({**r, "stake": float(core_single_stake(r["odds"])), "tag": "core"})
        used_matches.add(r["match"])
        if len(singles) >= min(CORE_TARGET_SINGLES, CORE_MAX_SINGLES):
            break

    # ensure minimum singles if possible
    if len(singles) < CORE_MIN_SINGLES:
        for r in candidates:
            if r["match"] in used_matches:
                continue
            singles.append({**r, "stake": float(core_single_stake(r["odds"])), "tag": "core"})
            used_matches.add(r["match"])
            if len(singles) >= CORE_MIN_SINGLES:
                break

    # build up to N doubles, requiring at least one leg <1.50
    doubles = []
    if lowlegs and CORE_MAX_DOUBLES > 0:
        # partner pool can be from singles candidates + other non-low rows in a wider range
        partner_pool = [r for r in rows if r["market"] in CORE_ALLOWED_MARKETS]
        # filter partners
        tmp = []
        for r in partner_pool:
            fx = fixtures_by_id.get(r["fixture_id"])
            if not fx:
                continue
            if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_CORE):
                continue
            if not confidence_ok(fx, CORE_MIN_CONFIDENCE):
                continue
            if r["odds"] < CORE_DOUBLE_PARTNER_MIN_ODDS or r["odds"] > CORE_DOUBLE_PARTNER_MAX_ODDS:
                continue
            min_val = band_lookup(r["odds"], CORE_MIN_VALUE_BANDS, default=9999)
            if r["value_adj"] < min_val:
                continue
            tmp.append(r)

        tmp.sort(key=lambda x: (x["value_adj"], _conf(x), safe_float(x["prob"], 0.0)), reverse=True)

        used_dbl_matches = set()
        for leg1 in lowlegs:
            if len(doubles) >= CORE_MAX_DOUBLES:
                break
            # pick best partner not same match and not reusing double matches
            for leg2 in tmp:
                if leg2["match"] == leg1["match"]:
                    continue
                if leg1["match"] in used_dbl_matches or leg2["match"] in used_dbl_matches:
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
                used_dbl_matches.add(leg1["match"])
                used_dbl_matches.add(leg2["match"])
                break

    cap_amount = BANKROLL_CORE * CORE_EXPOSURE_CAP
    open_total = sum(x["stake"] for x in singles) + sum(d.get("stake", 0.0) for d in doubles)

    # scale singles first; if still above cap, scale doubles too
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

    # keep backward compatibility: "double" = first double (if any)
    core_double = doubles[0] if doubles else None
    return singles, core_double, doubles, meta

# ------------------------- FUN SYSTEM COVERAGE -------------------------
def _columns_for_sizes(n: int, sizes):
    return sum(comb(n, r) for r in sizes if 1 <= r <= n)

def _payout_for_exact_k_hits(k_odds, sizes, k: int):
    # k_odds length == k, represents the odds of the picks that HIT.
    # payout per 1€ unit: sum of products for all winning combos within the k hits.
    out = 0.0
    for r in sizes:
        if r > k or r < 1:
            continue
        for idxs in combinations(range(k), r):
            prod = 1.0
            for i in idxs:
                prod *= max(1.01, float(k_odds[i]))
            out += prod
    return out

def system_coverage(odds_list, sizes):
    """
    Conservative coverage ratios (payout/cost) at:
      - k = r_min (minimum hits)
      - k = r_min + 1
    using the k LOWEST odds in the pool (worst-case payout).
    """
    n = len(odds_list)
    if n <= 0:
        return None
    sizes = [int(x) for x in sizes if int(x) >= 1]
    if not sizes:
        return None
    r_min = min(sizes)
    cols = _columns_for_sizes(n, sizes)
    if cols <= 0:
        return None

    odds_sorted = sorted([float(x) for x in odds_list])
    k0 = r_min
    k1 = min(n, r_min + 1)

    pay0 = _payout_for_exact_k_hits(odds_sorted[:k0], sizes, k0)
    pay1 = _payout_for_exact_k_hits(odds_sorted[:k1], sizes, k1)

    ratio0 = pay0 / cols
    ratio1 = pay1 / cols
    return {
        "r_min": r_min,
        "columns": cols,
        "ratio_at_r": round(ratio0, 4),
        "ratio_at_rp1": round(ratio1, 4),
    }

def choose_fun_system(system_pool):
    """
    Pick system type AND optionally shrink pool to keep coverage sane.
    Returns: (chosen_pool, system_dict) or ([], None)
    """
    if len(system_pool) < FUN_SYSTEM_POOL_MIN:
        return system_pool, None

    # candidates to evaluate (sizes, label)
    def candidates_for_n(n):
        c = []
        if n >= 6:
            c += [([4], f"4/{n}"), ([3,4], f"3-4/{n}"), ([3], f"3/{n}")]
        if n == 5:
            c += [([3], f"3/5"), ([2,3], f"2-3/5"), ([2], f"2/5")]
        if n == 4:
            c += [([3], f"3/4"), ([2,3], f"2-3/4"), ([2], f"2/4")]
        return c

    # try shrinking n from max down to min
    for n in range(min(FUN_SYSTEM_POOL_MAX, len(system_pool)), FUN_SYSTEM_POOL_MIN - 1, -1):
        pool = system_pool[:n]
        odds_list = [p["odds"] for p in pool]

        best = None
        best_key = (-1e9, -1e9)  # (ratio_at_rp1, ratio_at_r)
        for sizes, label in candidates_for_n(n):
            cov = system_coverage(odds_list, sizes)
            if not cov:
                continue
            # coverage constraints
            if cov["ratio_at_r"] < SYS_TARGET_RATIO_AT_R:
                continue
            if cov["ratio_at_rp1"] < SYS_TARGET_RATIO_AT_RP1:
                continue

            key = (cov["ratio_at_rp1"], cov["ratio_at_r"])
            if key > best_key:
                best_key = key
                best = {"label": label, "sizes": sizes, "columns": cov["columns"], "coverage": cov}

        if best:
            return pool, best

    # fallback: choose a simple 3/5 on the first 5 picks if possible, even if coverage isn't perfect
    if len(system_pool) >= 5:
        pool = system_pool[:5]
        odds_list = [p["odds"] for p in pool]
        best = {"label": "3/5", "sizes": [3], "columns": comb(5,3), "coverage": system_coverage(odds_list, [3])}
        return pool, best

    return system_pool, None

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
        if r["value_adj"] < FUN_MIN_VALUE_PCT:
            continue
        if r["prob"] is not None and r["prob"] < FUN_MIN_PROB:
            continue

        fun_candidates.append(r)

    fun_candidates.sort(key=lambda x: (x["value_adj"], safe_float(x["prob"], 0.0)), reverse=True)

    # take up to 7–8 unique matches (one market per match)
    fun_picks = []
    used_matches = set()
    for r in fun_candidates:
        if r["match"] in used_matches:
            continue
        fun_picks.append(r)
        used_matches.add(r["match"])
        if len(fun_picks) >= FUN_MAX_PICKS_TOTAL:
            break

    # if too few, allow up to 10 as a fallback (still unique matches)
    if len(fun_picks) < FUN_MIN_PICKS_TOTAL:
        for r in fun_candidates:
            if r["match"] in used_matches:
                continue
            fun_picks.append(r)
            used_matches.add(r["match"])
            if len(fun_picks) >= 10:
                break

    # ---- singles: top K (3–5), stakes by odds ----
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

    # ---- system pool (overlap allowed): top 5–6 ----
    system_pool_raw = fun_picks[:max(FUN_SYSTEM_POOL_MIN, min(FUN_SYSTEM_POOL_MAX, len(fun_picks)))]
    system_pool, sys_choice = choose_fun_system(system_pool_raw)

    cols = sys_choice["columns"] if sys_choice else 0
    system_label = sys_choice["label"] if sys_choice else None

    # ---- bankroll caps (split singles/system) ----
    cap_amount = BANKROLL_FUN * FUN_EXPOSURE_CAP
    cap_singles = cap_amount * FUN_CAP_SPLIT_SINGLES
    cap_system  = cap_amount * (1.0 - FUN_CAP_SPLIT_SINGLES)

    # scale singles into cap_singles
    singles_total = sum(x["stake"] for x in fun_singles)
    singles_scale = 1.0
    if singles_total > 0 and singles_total > cap_singles:
        singles_scale = cap_singles / singles_total
        for x in fun_singles:
            x["stake"] = round(x["stake"] * singles_scale, 1)

    # system unit: 0.5–1.0 per combo, scaled into cap_system
    unit = 0.0
    system_stake = 0.0
    system_scale = 1.0
    if cols > 0:
        base_unit = max(FUN_SYSTEM_UNIT_MIN, min(FUN_SYSTEM_UNIT_MAX, FUN_SYSTEM_UNIT_BASE))
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
            "odds_match_min_score_fun": ODDS_MATCH_MIN_SCORE_FUN,
            "min_value_pct": FUN_MIN_VALUE_PCT,
            "min_prob": FUN_MIN_PROB,
            "min_odds_default": FUN_MIN_ODDS_DEFAULT,
            "max_picks_total": FUN_MAX_PICKS_TOTAL,
            "singles_k": k,
            "system_pool_range": [FUN_SYSTEM_POOL_MIN, FUN_SYSTEM_POOL_MAX],
            "avoid_core_overlap": FUN_AVOID_CORE_OVERLAP,
            "system_coverage_targets": {"ratio_at_r": SYS_TARGET_RATIO_AT_R, "ratio_at_rp1": SYS_TARGET_RATIO_AT_RP1},
            "system_overlap_with_singles": True,
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
            "coverage": (sys_choice.get("coverage") if sys_choice else None),
            "ev_per_euro": None,
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
        json.dump(report, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
