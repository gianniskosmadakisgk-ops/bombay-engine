# =========================
# FRIDAY SHORTLIST v3.10 — PRODUCTION
# - Reads:  logs/thursday_report_v3.json
# - (Optional) Reads: logs/tuesday_history_v3.json  (bankroll continuity + week numbering)
# - Writes: logs/friday_shortlist_v3.json
#
# Key rules:
#  CORE
#   - Singles only in [1.50 .. 1.75] (default)
#   - Anything <1.50 is NEVER a single — only allowed as DOUBLE leg
#   - Doubles are created ONLY if we have at least one <1.50 leg
#
#  FUN
#   - Threshold filters by EV/prob, Under is stricter
#   - System chosen with refund ratio target 0.80, fallback 0.65
#   - Fun singles stake ladder: 15/12/8
#
# Additive fields:
#  - week_id, week_no, week_label
#  - bankroll_start + bankroll_source (history/default)
# =========================

import os
import json
from datetime import datetime, date
from math import comb

THURSDAY_REPORT_PATH = os.getenv("THURSDAY_REPORT_PATH", "logs/thursday_report_v3.json")
FRIDAY_REPORT_PATH   = os.getenv("FRIDAY_REPORT_PATH",   "logs/friday_shortlist_v3.json")
TUESDAY_HISTORY_PATH = os.getenv("TUESDAY_HISTORY_PATH", "logs/tuesday_history_v3.json")

# ------------------------- DEFAULT BANKROLLS -------------------------
DEFAULT_BANKROLL_CORE = float(os.getenv("BANKROLL_CORE", "800"))
DEFAULT_BANKROLL_FUN  = float(os.getenv("BANKROLL_FUN",  "400"))

CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.15"))
FUN_EXPOSURE_CAP  = float(os.getenv("FUN_EXPOSURE_CAP",  "0.20"))

ODDS_MATCH_MIN_SCORE_CORE = float(os.getenv("ODDS_MATCH_MIN_SCORE_CORE", "0.75"))
ODDS_MATCH_MIN_SCORE_FUN  = float(os.getenv("ODDS_MATCH_MIN_SCORE_FUN",  "0.65"))

CORE_MIN_CONFIDENCE = float(os.getenv("CORE_MIN_CONFIDENCE", "0.55"))
FUN_MIN_CONFIDENCE  = float(os.getenv("FUN_MIN_CONFIDENCE",  "0.45"))

# ✅ IMPORTANT: was missing in your deployed file
FUN_AVOID_CORE_OVERLAP = os.getenv("FUN_AVOID_CORE_OVERLAP", "true").lower() == "true"

# ------------------------- MARKETS -------------------------
CORE_ALLOWED_MARKETS = {"Home", "Away", "Over 2.5", "Under 2.5"}
FUN_ALLOWED_MARKETS  = {"Home", "Draw", "Away", "Over 2.5", "Under 2.5"}

MARKET_CODE = {"Home":"1","Draw":"X","Away":"2","Over 2.5":"O25","Under 2.5":"U25"}

# ------------------------- HELPERS -------------------------
def safe_float(v, d=None):
    try:
        return float(v)
    except Exception:
        return d

def iso_week_id_from_window(window: dict | None) -> str:
    d = None
    try:
        frm = (window or {}).get("from")
        if frm:
            d = date.fromisoformat(frm)
    except Exception:
        d = None
    if d is None:
        d = datetime.utcnow().date()
    y, w, _ = d.isocalendar()
    return f"{y}-W{int(w):02d}"

def load_history(path: str) -> dict:
    if not os.path.exists(path):
        return {"week_count": 0, "weeks": {}, "core": {"bankroll_current": None}, "funbet": {"bankroll_current": None}}
    try:
        h = json.load(open(path, "r", encoding="utf-8"))
        h.setdefault("week_count", 0)
        h.setdefault("weeks", {})
        h.setdefault("core", {}).setdefault("bankroll_current", None)
        h.setdefault("funbet", {}).setdefault("bankroll_current", None)
        return h
    except Exception:
        return {"week_count": 0, "weeks": {}, "core": {"bankroll_current": None}, "funbet": {"bankroll_current": None}}

def get_week_fields(window: dict, history: dict):
    window_from = (window or {}).get("from")
    week_id = iso_week_id_from_window(window)
    if window_from and window_from in (history.get("weeks") or {}):
        week_no = int(history["weeks"][window_from].get("week_no") or 1)
    else:
        week_no = int(history.get("week_count") or 0) + 1
    return {"week_id": week_id, "week_no": week_no, "week_label": f"Week {week_no}", "window_from": window_from}

def load_thursday():
    data = json.load(open(THURSDAY_REPORT_PATH, "r", encoding="utf-8"))
    if isinstance(data, dict) and "fixtures" in data:
        return data["fixtures"], data
    if isinstance(data, dict) and isinstance(data.get("report"), dict):
        rep = data["report"]
        if "fixtures" in rep:
            return rep["fixtures"], rep
        if isinstance(rep.get("report"), dict) and "fixtures" in rep["report"]:
            return rep["report"]["fixtures"], rep["report"]
    raise KeyError("fixtures")

def odds_match_ok(fx, min_score):
    om = fx.get("odds_match") or {}
    if not om.get("matched"):
        return False
    return (safe_float(om.get("score"), 0.0) or 0.0) >= min_score

def confidence_ok(fx, min_conf):
    flags = fx.get("flags") or {}
    c = safe_float(flags.get("confidence"), None)
    if c is None:
        return True
    return c >= min_conf

def ev(prob, odds):
    p = safe_float(prob, None)
    o = safe_float(odds, None)
    if p is None or o is None:
        return None
    return round(p * o - 1.0, 4)

def _get_over_value_and_penalty(fx):
    v_raw = fx.get("selection_value_pct_over")
    if v_raw is None:
        v_raw = fx.get("value_pct_over")
    v_raw = safe_float(v_raw, None)
    pen = safe_float(fx.get("over_value_penalty_pts"), 0.0) or 0.0
    if v_raw is None:
        return None, 0.0, None
    return v_raw, pen, v_raw - pen

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
                val_adj = over_v_adj
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
                "fx_ref": fx,
            })
    return rows

def scale_stakes(items, cap_amount, key="stake"):
    total = sum(safe_float(x.get(key), 0.0) for x in items)
    if total <= 0:
        return items, 1.0
    if total <= cap_amount:
        return items, 1.0
    s = cap_amount / total
    for x in items:
        x[key] = round(safe_float(x.get(key), 0.0) * s, 2)
    return items, round(s, 3)

# ============================================================
# CORE — locked rules
# ============================================================
CORE_SINGLES_MIN_ODDS = 1.50
CORE_SINGLES_MAX_ODDS = float(os.getenv("CORE_SINGLES_MAX_ODDS", "1.75"))

CORE_DOUBLE_LEG_MIN_ODDS = float(os.getenv("CORE_DOUBLE_LEG_MIN_ODDS", "1.20"))
CORE_DOUBLE_LEG_MAX_ODDS = 1.49

CORE_DOUBLE_PARTNER_MIN_ODDS = float(os.getenv("CORE_DOUBLE_PARTNER_MIN_ODDS", "1.35"))
CORE_DOUBLE_PARTNER_MAX_ODDS = float(os.getenv("CORE_DOUBLE_PARTNER_MAX_ODDS", "1.75"))

CORE_DOUBLE_TARGET_MIN = float(os.getenv("CORE_DOUBLE_TARGET_MIN", "1.75"))
CORE_DOUBLE_TARGET_MAX = float(os.getenv("CORE_DOUBLE_TARGET_MAX", "2.10"))

CORE_TARGET_SINGLES = int(os.getenv("CORE_TARGET_SINGLES", "4"))
CORE_MIN_SINGLES    = int(os.getenv("CORE_MIN_SINGLES", "3"))
CORE_MAX_SINGLES    = int(os.getenv("CORE_MAX_SINGLES", "6"))
CORE_MAX_DOUBLES    = int(os.getenv("CORE_MAX_DOUBLES", "1"))

CORE_STAKE_HIGH = float(os.getenv("CORE_STAKE_HIGH", "50"))
CORE_STAKE_MID  = float(os.getenv("CORE_STAKE_MID",  "40"))
CORE_STAKE_LOW  = float(os.getenv("CORE_STAKE_LOW",  "30"))
CORE_DOUBLE_STAKE = float(os.getenv("CORE_DOUBLE_STAKE", "25"))

CORE_MIN_VALUE_BANDS = {(1.50,1.65):0.5,(1.65,1.75):0.5}
CORE_RELAX_VALUE_PTS = float(os.getenv("CORE_RELAX_VALUE_PTS", "1.5"))
CORE_DOUBLE_LOWLEG_MIN_PROB = float(os.getenv("CORE_DOUBLE_LOWLEG_MIN_PROB", "0.70"))

def core_single_stake(odds: float) -> float:
    if odds <= 1.55: return CORE_STAKE_HIGH
    if odds <= 1.65: return CORE_STAKE_MID
    return CORE_STAKE_LOW

def band_min_val(odds: float) -> float:
    for (a,b),v in CORE_MIN_VALUE_BANDS.items():
        if a <= odds <= b:
            return v
    return 9999.0

def pick_core(rows, fixtures_by_id):
    strict = []
    relaxed = []
    lowlegs = []

    for r in rows:
        fx = r["fx_ref"]
        if r["market"] not in CORE_ALLOWED_MARKETS:
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_CORE):
            continue
        if not confidence_ok(fx, CORE_MIN_CONFIDENCE):
            continue

        o = safe_float(r["odds"], None)
        if o is None:
            continue

        # below 1.50 => ONLY double leg
        if CORE_DOUBLE_LEG_MIN_ODDS <= o <= CORE_DOUBLE_LEG_MAX_ODDS:
            if (safe_float(r.get("prob"), 0.0) or 0.0) >= CORE_DOUBLE_LOWLEG_MIN_PROB:
                lowlegs.append(r)
            continue

        # singles only >= 1.50
        if o < CORE_SINGLES_MIN_ODDS or o > CORE_SINGLES_MAX_ODDS:
            continue

        v_adj = safe_float(r.get("value_adj"), None)
        if v_adj is None:
            continue

        thr = band_min_val(o)
        if v_adj >= thr:
            strict.append(r)
        if v_adj >= (thr - CORE_RELAX_VALUE_PTS):
            relaxed.append(r)

    strict.sort(key=lambda x: (safe_float(x.get("value_adj"), -9999.0), safe_float(x.get("prob"), 0.0)), reverse=True)
    relaxed.sort(key=lambda x: (safe_float(x.get("value_adj"), -9999.0), safe_float(x.get("prob"), 0.0)), reverse=True)
    lowlegs.sort(key=lambda x: (safe_float(x.get("prob"), 0.0), safe_float(x.get("value_adj"), -9999.0)), reverse=True)

    singles = []
    used = set()

    def add(pool, limit):
        nonlocal singles, used
        for r in pool:
            if len(singles) >= limit:
                break
            if r["match"] in used:
                continue
            singles.append({
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
                "stake": float(core_single_stake(r["odds"])),
                "tag": "core",
            })
            used.add(r["match"])

    add(strict, CORE_TARGET_SINGLES)
    if len(singles) < CORE_MIN_SINGLES:
        add(relaxed, CORE_MIN_SINGLES)
    if len(singles) < CORE_MAX_SINGLES:
        add(relaxed, CORE_MAX_SINGLES)

    doubles = []
    if lowlegs and CORE_MAX_DOUBLES > 0:
        partner_pool = [r for r in strict + relaxed if CORE_DOUBLE_PARTNER_MIN_ODDS <= safe_float(r["odds"],0.0) <= CORE_DOUBLE_PARTNER_MAX_ODDS]
        partner_pool.sort(key=lambda x: (safe_float(x.get("value_adj"), -9999.0), safe_float(x.get("prob"), 0.0)), reverse=True)

        used_d = set()
        for leg1 in lowlegs:
            if len(doubles) >= CORE_MAX_DOUBLES:
                break
            for leg2 in partner_pool:
                if leg2["match"] == leg1["match"]:
                    continue
                if leg1["match"] in used_d or leg2["match"] in used_d:
                    continue
                combo = safe_float(leg1["odds"],1.0) * safe_float(leg2["odds"],1.0)
                if not (CORE_DOUBLE_TARGET_MIN <= combo <= CORE_DOUBLE_TARGET_MAX):
                    continue
                doubles.append({
                    "legs": [
                        {"pick_id": f'{leg1["fixture_id"]}:{leg1["market_code"]}', "match": leg1["match"], "market": leg1["market"], "odds": leg1["odds"]},
                        {"pick_id": f'{leg2["fixture_id"]}:{leg2["market_code"]}', "match": leg2["match"], "market": leg2["market"], "odds": leg2["odds"]},
                    ],
                    "combo_odds": round(combo, 2),
                    "stake": round(float(CORE_DOUBLE_STAKE), 2),
                    "tag": "core_double",
                })
                used_d.add(leg1["match"]); used_d.add(leg2["match"])
                break

    cap_amount = DEFAULT_BANKROLL_CORE * CORE_EXPOSURE_CAP
    open_total = sum(x["stake"] for x in singles) + sum(d.get("stake", 0.0) for d in doubles)
    scale = 1.0
    if open_total > cap_amount and open_total > 0:
        s = cap_amount / open_total
        for x in singles:
            x["stake"] = round(x["stake"] * s, 2)
        for d in doubles:
            d["stake"] = round(float(d.get("stake", 0.0)) * s, 2)
        scale = round(s, 3)
        open_total = sum(x["stake"] for x in singles) + sum(d.get("stake", 0.0) for d in doubles)

    meta = {"open": round(open_total, 2), "scale_applied": scale, "picks_count": len(singles)}
    return singles, (doubles[0] if doubles else None), doubles, meta

# ============================================================
# FUN — thresholds + system refund fallback
# ============================================================
EV_MIN_GENERAL = float(os.getenv("EV_MIN_GENERAL", "0.05"))
EV_MIN_UNDER   = float(os.getenv("EV_MIN_UNDER",   "0.08"))
P_MIN_HOME     = float(os.getenv("P_MIN_HOME",     "0.30"))
P_MIN_DRAW     = float(os.getenv("P_MIN_DRAW",     "0.25"))
P_MIN_AWAY     = float(os.getenv("P_MIN_AWAY",     "0.22"))
P_MIN_OVER     = float(os.getenv("P_MIN_OVER",     "0.40"))
P_MIN_UNDER    = float(os.getenv("P_MIN_UNDER",    "0.55"))

UNDER_DRAW_MIN   = float(os.getenv("UNDER_DRAW_MIN", "0.30"))
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
FUN_SINGLES_K = int(os.getenv("FUN_SINGLES_K", "4"))
FUN_SINGLES_MIN = int(os.getenv("FUN_SINGLES_MIN", "3"))
FUN_SINGLES_MAX = int(os.getenv("FUN_SINGLES_MAX", "5"))

FUN_CAP_SPLIT_SINGLES = float(os.getenv("FUN_CAP_SPLIT_SINGLES", "0.70"))
FUN_CAP_SPLIT_SINGLES = max(0.10, min(0.90, FUN_CAP_SPLIT_SINGLES))

FUN_SYSTEM_UNIT_MIN  = float(os.getenv("FUN_SYSTEM_UNIT_MIN",  "0.50"))
FUN_SYSTEM_UNIT_MAX  = float(os.getenv("FUN_SYSTEM_UNIT_MAX",  "2.00"))
FUN_SYSTEM_TARGET_MIN = float(os.getenv("FUN_SYSTEM_TARGET_MIN", "12.0"))
FUN_SYSTEM_TARGET_MAX = float(os.getenv("FUN_SYSTEM_TARGET_MAX", "20.0"))

SYS_REFUND_PRIMARY  = float(os.getenv("SYS_REFUND_PRIMARY", "0.80"))
SYS_REFUND_FALLBACK = float(os.getenv("SYS_REFUND_FALLBACK", "0.65"))

FUN_SYSTEM_POOL_MAX = int(os.getenv("FUN_SYSTEM_POOL_MAX", "7"))
FUN_SYSTEM_POOL_MIN = int(os.getenv("FUN_SYSTEM_POOL_MIN", "5"))

def fun_single_stake(odds: float) -> float:
    if odds < 1.85: return 0.0
    if odds <= 2.20: return 15.0
    if odds <= 3.50: return 12.0
    if odds <= 4.60: return 8.0
    return 0.0

def _columns_for_r(n: int, r: int) -> int:
    return comb(n, r) if (n > 0 and 0 < r <= n) else 0

def _refund_ratio_worst_case(odds_list, r_min: int, columns: int) -> float:
    if columns <= 0 or r_min <= 0:
        return 0.0
    o = sorted([float(x) for x in odds_list])[:r_min]
    prod = 1.0
    for v in o:
        prod *= max(1.01, float(v))
    return prod / float(columns)

def _try_system(pool, r_try: int, label: str):
    n = len(pool)
    cols = _columns_for_r(n, r_try)
    if cols <= 0:
        return None
    rr = _refund_ratio_worst_case([p["odds"] for p in pool], r_try, cols)
    return {"label": label, "n": n, "columns": cols, "min_hits": r_try, "refund_ratio_min_hits": round(rr, 4)}

def choose_fun_system(fun_picks):
    for threshold in [SYS_REFUND_PRIMARY, SYS_REFUND_FALLBACK]:
        breached = (threshold != SYS_REFUND_PRIMARY)
        for n in [7,6,5]:
            if len(fun_picks) < n:
                continue
            pool = fun_picks[:n]
            trials = []
            if n == 7:
                trials = [_try_system(pool, 4, "4/7"), _try_system(pool, 5, "5/7")]
            elif n == 6:
                trials = [_try_system(pool, 3, "3/6"), _try_system(pool, 4, "4/6")]
            else:
                trials = [_try_system(pool, 3, "3/5")]

            trials = [t for t in trials if t]
            passing = [t for t in trials if (t.get("refund_ratio_min_hits") or 0.0) >= threshold]
            if passing:
                passing.sort(key=lambda x: (x["columns"], -float(x["refund_ratio_min_hits"])))
                return pool, passing[0], threshold, breached, True

    return fun_picks[:min(len(fun_picks),7)], None, SYS_REFUND_PRIMARY, True, False

def _system_unit_for_target(cols: int, cap_system: float):
    if cols <= 0:
        return 0.0, 0.0, 1.0
    desired = min(FUN_SYSTEM_TARGET_MAX, cap_system)
    if desired < FUN_SYSTEM_TARGET_MIN:
        desired = cap_system
    if desired <= 0:
        return 0.0, 0.0, 1.0
    base_unit = desired / float(cols)
    base_unit = max(FUN_SYSTEM_UNIT_MIN, min(FUN_SYSTEM_UNIT_MAX, base_unit))
    base_stake = base_unit * cols
    scale = 1.0
    if base_stake > cap_system and base_stake > 0:
        scale = cap_system / base_stake
    unit = round(base_unit * scale, 2)
    stake = round(unit * cols, 2)
    return unit, stake, round(scale, 3)

def pick_fun(rows, fixtures_by_id, core_singles):
    core_fixture_ids = {x["fixture_id"] for x in core_singles}

    fun_candidates = []
    for r in rows:
        fx = r["fx_ref"]
        if r["market"] not in FUN_ALLOWED_MARKETS:
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_FUN):
            continue
        if not confidence_ok(fx, FUN_MIN_CONFIDENCE):
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

        if r["market"] == "Home" and (e < EV_MIN_GENERAL or p < P_MIN_HOME):
            continue
        if r["market"] == "Draw" and (e < EV_MIN_GENERAL or p < P_MIN_DRAW):
            continue
        if r["market"] == "Away" and (e < EV_MIN_GENERAL or p < P_MIN_AWAY):
            continue
        if r["market"] == "Over 2.5" and (e < EV_MIN_GENERAL or p < P_MIN_OVER):
            continue
        if r["market"] == "Under 2.5":
            if e < EV_MIN_UNDER or p < P_MIN_UNDER:
                continue
            flags = fx.get("flags") or {}
            ltot = (safe_float(fx.get("lambda_home"),0.0) or 0.0) + (safe_float(fx.get("lambda_away"),0.0) or 0.0)
            if not flags.get("tight_game", False):
                continue
            if safe_float(fx.get("draw_prob"), 0.0) < UNDER_DRAW_MIN:
                continue
            if ltot > UNDER_LTOTAL_MAX:
                continue

        fun_candidates.append(r)

    fun_candidates.sort(key=lambda x: (safe_float(x.get("value_adj"), -9999.0), safe_float(x.get("prob"), 0.0)), reverse=True)

    fun_picks = []
    used = set()
    for r in fun_candidates:
        if r["match"] in used:
            continue
        fun_picks.append(r)
        used.add(r["match"])
        if len(fun_picks) >= FUN_MAX_PICKS_TOTAL:
            break

    if len(fun_picks) < FUN_MIN_PICKS_TOTAL:
        for r in fun_candidates:
            if r["match"] in used:
                continue
            fun_picks.append(r)
            used.add(r["match"])
            if len(fun_picks) >= 10:
                break

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

    system_pool_raw = fun_picks[:min(max(FUN_SYSTEM_POOL_MIN, len(fun_picks)), FUN_SYSTEM_POOL_MAX)]
    if len(system_pool_raw) > 7:
        system_pool_raw = system_pool_raw[:7]

    system_pool, sys_choice, threshold_used, breached, has_system = choose_fun_system(system_pool_raw)
    cols = int(sys_choice["columns"]) if sys_choice else 0

    cap_amount = DEFAULT_BANKROLL_FUN * FUN_EXPOSURE_CAP
    cap_singles = cap_amount * FUN_CAP_SPLIT_SINGLES
    cap_system  = cap_amount * (1.0 - FUN_CAP_SPLIT_SINGLES)

    singles_total = sum(x["stake"] for x in fun_singles)
    singles_scale = 1.0
    if singles_total > 0 and singles_total > cap_singles:
        singles_scale = cap_singles / singles_total
        for x in fun_singles:
            x["stake"] = round(x["stake"] * singles_scale, 2)

    unit, system_stake, system_scale = (0.0, 0.0, 1.0)
    if cols > 0:
        unit, system_stake, system_scale = _system_unit_for_target(cols, cap_system)

    open_amount = round(system_stake + sum(x["stake"] for x in fun_singles), 2)

    payload = {
        "bankroll": DEFAULT_BANKROLL_FUN,
        "exposure_cap_pct": FUN_EXPOSURE_CAP,
        "cap_split": {"singles_pct": round(FUN_CAP_SPLIT_SINGLES, 2), "system_pct": round(1.0 - FUN_CAP_SPLIT_SINGLES, 2)},
        "scales": {"singles_scale": round(singles_scale, 3), "system_scale": round(system_scale, 3)},
        "rules": {"refund_primary": SYS_REFUND_PRIMARY, "refund_fallback": SYS_REFUND_FALLBACK, "refund_used": threshold_used, "refund_rule_breached": bool(breached)},
        "picks_total": [{
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
        } for r in fun_picks],
        "singles": fun_singles,
        "system_pool": [{
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
        } for r in system_pool],
        "system": {
            "label": (sys_choice["label"] if sys_choice else None),
            "columns": cols,
            "unit": unit,
            "stake": system_stake,
            "refund_ratio_min_hits": (sys_choice.get("refund_ratio_min_hits") if sys_choice else None),
            "min_hits": (sys_choice.get("min_hits") if sys_choice else None),
            "pool_size_used": (sys_choice.get("n") if sys_choice else None),
            "refund_threshold_used": threshold_used,
            "refund_rule_breached": bool(breached),
            "has_system": bool(has_system),
            "ev_per_euro": None,
        },
        "open": open_amount,
        "after_open": round(DEFAULT_BANKROLL_FUN - open_amount, 2),
        "counts": {"picks_total": len(fun_picks), "singles": len(fun_singles), "system_pool": len(system_pool)},
    }
    return payload

def main():
    fixtures, th_meta = load_thursday()
    fixtures_by_id = {fx.get("fixture_id"): fx for fx in fixtures}

    history = load_history(TUESDAY_HISTORY_PATH)
    window = th_meta.get("window", {}) or {}
    wf = get_week_fields(window, history)

    core_start = safe_float(history.get("core", {}).get("bankroll_current"), None)
    fun_start  = safe_float(history.get("funbet", {}).get("bankroll_current"), None)

    core_bankroll_start = core_start if core_start is not None else DEFAULT_BANKROLL_CORE
    fun_bankroll_start  = fun_start if fun_start is not None else DEFAULT_BANKROLL_FUN

    rows = build_rows(fixtures)

    core_singles, core_double, core_doubles, core_meta = pick_core(rows, fixtures_by_id)
    fun_payload = pick_fun(rows, fixtures_by_id, core_singles)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "week_id": wf["week_id"],
        "week_no": wf["week_no"],
        "week_label": wf["week_label"],
        "window": window,
        "fixtures_total": th_meta.get("fixtures_total", len(fixtures)),

        "core": {
            "bankroll": DEFAULT_BANKROLL_CORE,
            "bankroll_start": core_bankroll_start,
            "bankroll_source": ("history" if core_start is not None else "default"),
            "exposure_cap_pct": CORE_EXPOSURE_CAP,
            "rules": {"no_singles_below": 1.50, "low_odds_policy": "Below 1.50 only as double leg"},
            "singles": core_singles,
            "double": core_double,
            "doubles": core_doubles,
            "open": core_meta["open"],
            "after_open": round(core_bankroll_start - core_meta["open"], 2),
            "picks_count": core_meta["picks_count"],
            "scale_applied": core_meta["scale_applied"],
        },

        "funbet": {
            **fun_payload,
            "bankroll_start": fun_bankroll_start,
            "bankroll_source": ("history" if fun_start is not None else "default"),
        },
    }

    os.makedirs("logs", exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
