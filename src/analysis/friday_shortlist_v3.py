# =========================
# FRIDAY SHORTLIST v3.4 — CORE 800 (3–4 singles + optional double) / FUN 400 (1–3 singles + system)
# Drop-in replacement for: src/analysis/friday_shortlist_v3.py
#
# Output: logs/friday_shortlist_v3.json
# Schema-safe: keeps the same top-level report keys and structures.
# =========================

import os
import json
from datetime import datetime
from itertools import combinations
from math import comb

THURSDAY_REPORT_PATH = "logs/thursday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v3.json"

# ------------------------- BANKROLLS (defaults per new framework) -------------------------
BANKROLL_CORE = float(os.getenv("BANKROLL_CORE", "800"))
BANKROLL_FUN  = float(os.getenv("BANKROLL_FUN", "400"))

# Exposure caps
CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.25"))  # ~200€ on 800
FUN_EXPOSURE_CAP  = float(os.getenv("FUN_EXPOSURE_CAP",  "0.20"))  # ~80€ on 400

# Odds-match thresholds
ODDS_MATCH_MIN_SCORE_CORE = float(os.getenv("ODDS_MATCH_MIN_SCORE_CORE", "0.75"))
ODDS_MATCH_MIN_SCORE_FUN  = float(os.getenv("ODDS_MATCH_MIN_SCORE_FUN",  "0.65"))  # fun πιο ανεκτικό

# Optional confidence gate (Thursday adds flags.confidence)
CORE_MIN_CONFIDENCE = float(os.getenv("CORE_MIN_CONFIDENCE", "0.55"))
FUN_MIN_CONFIDENCE  = float(os.getenv("FUN_MIN_CONFIDENCE",  "0.45"))

# ------------------------- CORE RULES -------------------------
CORE_ALLOWED_MARKETS = {"Home", "Away", "Over 2.5", "Under 2.5"}

CORE_MIN_ODDS = float(os.getenv("CORE_MIN_ODDS", "1.50"))
CORE_MAX_ODDS = float(os.getenv("CORE_MAX_ODDS", "2.20"))

CORE_TARGET_SINGLES = int(os.getenv("CORE_TARGET_SINGLES", "4"))  # 3–4 by default logic below
CORE_MIN_SINGLES = int(os.getenv("CORE_MIN_SINGLES", "3"))

# Value thresholds (kept from previous version)
CORE_MIN_VALUE_BANDS = {
    (1.50, 1.65): float(os.getenv("CORE_MIN_VALUE_150_165", "4.0")),
    (1.65, 2.00): float(os.getenv("CORE_MIN_VALUE_165_200", "3.0")),
    (2.00, 2.20): float(os.getenv("CORE_MIN_VALUE_200_220", "2.5")),
}

# Stakes per new plan: 40–50€ singles, (slightly odds-weighted)
CORE_STAKE_LOW  = float(os.getenv("CORE_STAKE_LOW",  "40"))  # higher odds end
CORE_STAKE_MID  = float(os.getenv("CORE_STAKE_MID",  "45"))
CORE_STAKE_HIGH = float(os.getenv("CORE_STAKE_HIGH", "50"))  # low odds end

# Core double: optional, low-odds legs
CORE_DOUBLE_LEG_MAX_ODDS = float(os.getenv("CORE_DOUBLE_LEG_MAX_ODDS", "1.55"))
CORE_DOUBLE_TARGET_MIN   = float(os.getenv("CORE_DOUBLE_TARGET_MIN",   "1.70"))
CORE_DOUBLE_TARGET_MAX   = float(os.getenv("CORE_DOUBLE_TARGET_MAX",   "2.20"))
CORE_DOUBLE_STAKE_MIN    = float(os.getenv("CORE_DOUBLE_STAKE_MIN",    "30"))
CORE_DOUBLE_STAKE_MAX    = float(os.getenv("CORE_DOUBLE_STAKE_MAX",    "40"))

# ------------------------- FUN RULES -------------------------
FUN_ALLOWED_MARKETS = {"Home", "Draw", "Away", "Over 2.5", "Under 2.5"}

FUN_MIN_ODDS_DEFAULT = float(os.getenv("FUN_MIN_ODDS_DEFAULT", "1.70"))
FUN_MAX_ODDS_BY_MARKET = {
    "Home": float(os.getenv("FUN_MAX_ODDS_HOME", "3.20")),
    "Away": float(os.getenv("FUN_MAX_ODDS_AWAY", "3.20")),
    "Over 2.5": float(os.getenv("FUN_MAX_ODDS_O25", "3.20")),
    "Under 2.5": float(os.getenv("FUN_MAX_ODDS_U25", "3.20")),
    "Draw": float(os.getenv("FUN_MAX_ODDS_DRAW", "4.60")),
}

FUN_MIN_VALUE_PCT = float(os.getenv("FUN_MIN_VALUE_PCT", "4.0"))
FUN_MIN_PROB      = float(os.getenv("FUN_MIN_PROB", "0.22"))  # avoid pure lottery

FUN_MAX_PICKS_TOTAL = int(os.getenv("FUN_MAX_PICKS_TOTAL", "10"))

# Per new framework: 1–3 singles + system from remaining
FUN_MAX_SINGLES = int(os.getenv("FUN_MAX_SINGLES", "3"))
FUN_MIN_SINGLES = int(os.getenv("FUN_MIN_SINGLES", "1"))

# Per notes: system usually 3/5 or 4/6 (cap pool by default to 6)
FUN_SYSTEM_MAX_MATCHES = int(os.getenv("FUN_SYSTEM_MAX_MATCHES", "6"))

# Exposure split inside FUN cap (singles vs system)
FUN_CAP_SPLIT_SINGLES = float(os.getenv("FUN_CAP_SPLIT_SINGLES", "0.40"))
FUN_CAP_SPLIT_SINGLES = max(0.10, min(0.80, FUN_CAP_SPLIT_SINGLES))

# System unit base (0.5–1 per combo); will be scaled down if needed
FUN_SYSTEM_UNIT_BASE = float(os.getenv("FUN_SYSTEM_UNIT_BASE", "1.0"))
FUN_SYSTEM_UNIT_MIN  = float(os.getenv("FUN_SYSTEM_UNIT_MIN",  "0.50"))
FUN_SYSTEM_UNIT_MAX  = float(os.getenv("FUN_SYSTEM_UNIT_MAX",  "1.00"))

FUN_AVOID_CORE_OVERLAP = os.getenv("FUN_AVOID_CORE_OVERLAP", "true").lower() == "true"

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
    except Exception:
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

def confidence_ok(fx, min_conf):
    flags = fx.get("flags") or {}
    c = safe_float(flags.get("confidence"), None)
    if c is None:
        return True  # if missing, don't block
    return c >= min_conf

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
def core_single_stake(odds: float) -> float:
    # 40–50 zone, slightly odds-weighted
    if odds <= 1.65:
        return CORE_STAKE_HIGH
    if odds <= 2.00:
        return CORE_STAKE_MID
    return CORE_STAKE_LOW

def _core_flag_match(fx_flags, market: str) -> bool:
    """
    Thursday flags optionally mark "core_1/core_2/core_over/core_under".
    If present and True, we treat as strong hint. If missing, ignore.
    """
    if not fx_flags:
        return False
    if market == "Home":
        return bool(fx_flags.get("core_1"))
    if market == "Away":
        return bool(fx_flags.get("core_2"))
    if market == "Over 2.5":
        return bool(fx_flags.get("core_over"))
    if market == "Under 2.5":
        return bool(fx_flags.get("core_under"))
    return False

def pick_core(rows, fixtures_by_id):
    core_candidates = []
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
        if r["odds"] < CORE_MIN_ODDS or r["odds"] > CORE_MAX_ODDS:
            continue

        # Value gate (banded). If Thursday flags says core_* True, allow slightly softer gate.
        min_val = band_lookup(r["odds"], CORE_MIN_VALUE_BANDS, default=9999)
        hinted = _core_flag_match(r.get("flags") or {}, r["market"])
        if not hinted and r["value_adj"] < min_val:
            continue
        if hinted and r["value_adj"] < (min_val - 0.8):
            continue

        stake = core_single_stake(r["odds"])
        core_candidates.append({**r, "stake": float(stake), "tag": "core"})

    # Sort: prioritize value_adj, then confidence (if exists), then prob
    def _conf(x):
        return safe_float((x.get("flags") or {}).get("confidence"), 0.0)

    core_candidates.sort(key=lambda x: (x["value_adj"], _conf(x), x["prob"]), reverse=True)

    # Pick 3–4 singles, one market per match
    core_singles = []
    used_matches = set()
    for r in core_candidates:
        if r["match"] in used_matches:
            continue
        core_singles.append(r)
        used_matches.add(r["match"])
        if len(core_singles) >= CORE_TARGET_SINGLES:
            break

    # Ensure minimum 3 if possible
    if len(core_singles) < CORE_MIN_SINGLES:
        # try to fill up to min from remaining candidates (still unique match)
        for r in core_candidates:
            if r["match"] in used_matches:
                continue
            core_singles.append(r)
            used_matches.add(r["match"])
            if len(core_singles) >= CORE_MIN_SINGLES:
                break

    # Optional double from low-odds legs (<=1.55), prefer different matches
    def best_double(pool):
        best = None
        best_score = -1e18
        for a, b in combinations(pool, 2):
            if a["match"] == b["match"]:
                continue
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

    double_pool = [x for x in core_singles if x["odds"] <= CORE_DOUBLE_LEG_MAX_ODDS]
    core_double = best_double(double_pool) if len(double_pool) >= 2 else None

    # Stakes & cap management (include double stake in exposure)
    cap_amount = BANKROLL_CORE * CORE_EXPOSURE_CAP

    # set double stake (30–40) if exists
    double_stake = 0.0
    if core_double:
        base = 0.045 * BANKROLL_CORE
        double_stake = max(CORE_DOUBLE_STAKE_MIN, min(CORE_DOUBLE_STAKE_MAX, base))
        core_double["stake"] = round(double_stake, 1)

    singles_total = sum(x["stake"] for x in core_singles)
    open_total = singles_total + double_stake

    # If over cap, scale singles first, then double if still over
    core_scale = 1.0
    if open_total > cap_amount and singles_total > 0:
        # keep double as-is first, scale singles into remaining cap
        remaining_for_singles = max(0.0, cap_amount - double_stake)
        if remaining_for_singles <= 0:
            remaining_for_singles = cap_amount  # fallback: scale everything together later
        core_singles, core_scale, _ = scale_stakes(core_singles, remaining_for_singles, "stake")
        singles_total = sum(x["stake"] for x in core_singles)
        open_total = singles_total + double_stake

    # Still over cap? scale double too
    if open_total > cap_amount and core_double and double_stake > 0:
        s = cap_amount / open_total if open_total > 0 else 1.0
        for x in core_singles:
            x["stake"] = round(x["stake"] * s, 1)
        core_double["stake"] = round(core_double["stake"] * s, 1)
        core_scale = round(core_scale * s, 3)
        singles_total = sum(x["stake"] for x in core_singles)
        double_stake = core_double["stake"]
        open_total = singles_total + double_stake

    core_meta = {
        "bankroll": BANKROLL_CORE,
        "exposure_cap_pct": CORE_EXPOSURE_CAP,
        "open": round(open_total, 1),
        "after_open": round(BANKROLL_CORE - open_total, 1),
        "picks_count": len(core_singles),
        "scale_applied": round(core_scale, 3),
    }
    return core_singles, core_double, core_meta

# ------------------------- FUN: STAKES -------------------------
def fun_single_stake(odds: float) -> float:
    """
    New plan: 6–8€ each.
    """
    if odds <= 2.30:
        return 8.0
    if odds <= 3.20:
        return 7.0
    if odds <= 4.60:
        return 6.0
    return 0.0

def _fun_flag_match(fx_flags, market: str) -> bool:
    """
    Thursday flags optionally mark "fun_1/fun_x/fun_2/fun_over/fun_under".
    """
    if not fx_flags:
        return False
    if market == "Home":
        return bool(fx_flags.get("fun_1"))
    if market == "Draw":
        return bool(fx_flags.get("fun_x"))
    if market == "Away":
        return bool(fx_flags.get("fun_2"))
    if market == "Over 2.5":
        return bool(fx_flags.get("fun_over"))
    if market == "Under 2.5":
        return bool(fx_flags.get("fun_under"))
    return False

def _choose_fun_system(n: int):
    """
    Align with notes: prefer 3/n for ~5 picks, 4/n for ~6 picks.
    """
    if n >= 6:
        return 4, f"4/{n}"
    if n >= 5:
        return 3, f"3/{n}"
    if n == 4:
        return 3, f"3/{n}"
    if n == 3:
        return 2, f"2/{n}"
    return None, None

# ------------------------- FUN PICKER -------------------------
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
        hinted = _fun_flag_match(r.get("flags") or {}, r["market"])

        # Odds gate: if hinted, allow a bit looser; otherwise enforce.
        if not hinted:
            if r["odds"] < FUN_MIN_ODDS_DEFAULT or r["odds"] > max_odds:
                continue
        else:
            if r["odds"] < (FUN_MIN_ODDS_DEFAULT - 0.20) or r["odds"] > (max_odds + 0.40):
                continue

        # Value gate
        if r["value_adj"] < FUN_MIN_VALUE_PCT:
            continue

        # Prob floor (soft): if hinted allow slightly lower
        pmin = FUN_MIN_PROB - (0.03 if hinted else 0.0)
        if r["prob"] < pmin:
            continue

        fun_candidates.append(r)

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

    # ---- singles subset: top 1–3 ----
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

    # ensure at least 1 single if possible (fallback)
    if len(fun_singles) < FUN_MIN_SINGLES and fun_picks:
        r = fun_picks[0]
        st = fun_single_stake(r["odds"])
        if st > 0 and not fun_singles:
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

    # ---- system pool: remaining picks (exclude singles) ----
    single_ids = {x["pick_id"] for x in fun_singles}
    remaining = [r for r in fun_picks if f'{r["fixture_id"]}:{r["market_code"]}' not in single_ids]

    system_pool = remaining[:FUN_SYSTEM_MAX_MATCHES]
    n_sys = len(system_pool)

    r_size, system_label = _choose_fun_system(n_sys)
    cols = comb(n_sys, r_size) if (r_size and n_sys >= r_size) else 0

    # ---- bankroll caps (split singles/system) ----
    cap_amount = BANKROLL_FUN * FUN_EXPOSURE_CAP
    cap_singles = cap_amount * FUN_CAP_SPLIT_SINGLES
    cap_system  = cap_amount * (1.0 - FUN_CAP_SPLIT_SINGLES)

    # scale singles into cap_singles if needed
    singles_total = sum(x["stake"] for x in fun_singles)
    singles_scale = 1.0
    if singles_total > 0 and singles_total > cap_singles:
        singles_scale = cap_singles / singles_total
        for x in fun_singles:
            x["stake"] = round(x["stake"] * singles_scale, 1)

    # system unit: aim 0.5–1 per combo, scale down if cap_system tight
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
        "cap_split": {
            "singles_pct": round(FUN_CAP_SPLIT_SINGLES, 2),
            "system_pct": round(1.0 - FUN_CAP_SPLIT_SINGLES, 2)
        },
        "scales": {
            "singles_scale": round(singles_scale, 3),
            "system_scale": round(system_scale, 3)
        },
        "rules": {
            "odds_match_min_score_fun": ODDS_MATCH_MIN_SCORE_FUN,
            "min_value_pct": FUN_MIN_VALUE_PCT,
            "min_prob": FUN_MIN_PROB,
            "max_picks_total": FUN_MAX_PICKS_TOTAL,
            "max_system_matches": FUN_SYSTEM_MAX_MATCHES,
            "max_singles": FUN_MAX_SINGLES,
            "max_odds_by_market": FUN_MAX_ODDS_BY_MARKET,
            "min_odds_default": FUN_MIN_ODDS_DEFAULT,
            "avoid_core_overlap": FUN_AVOID_CORE_OVERLAP,
            "system_preference": "3/n for 5, 4/n for 6 (scaled unit 0.5–1)",
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
            "ev_per_euro": None,  # keep field but not used in this deterministic system mode
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
                "min_confidence": CORE_MIN_CONFIDENCE,
                "min_value_bands": {
                    "1.50-1.65": CORE_MIN_VALUE_BANDS[(1.50, 1.65)],
                    "1.65-2.00": CORE_MIN_VALUE_BANDS[(1.65, 2.00)],
                    "2.00-2.20": CORE_MIN_VALUE_BANDS[(2.00, 2.20)],
                },
                "stake_plan": "3–4 singles @ 40–50€ + optional double @ 30–40€",
                "double_leg_max_odds": CORE_DOUBLE_LEG_MAX_ODDS,
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
