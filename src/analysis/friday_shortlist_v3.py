# ============================================================
#  src/analysis/friday_shortlist_v3.py
#  FRIDAY SHORTLIST v3.30 — PRODUCTION (CoreBet + FunBet + DrawBet)
#
#  Reads:
#    - logs/thursday_report_v3.json
#    - (optional) logs/tuesday_history_v3.json   (bankroll carry + week numbering)
#
#  Writes:
#    - logs/friday_shortlist_v3.json
#
#  LOCKED RULES (current agreement)
#   COREBET:
#     - Singles odds: 1.60–2.10 (ONLY)
#     - Stake ladder:
#         1.60–1.75 => 40
#         1.75–1.90 => 30
#         1.90–2.10 => 20
#     - Low odds 1.30–1.60 NEVER single -> forced into doubles bucket
#     - Doubles: target combo 2.10–3.00, up to 2 doubles, stake fixed 15
#     - Avoid "phantom" 1X2 via Market Sanity filter (model vs implied mismatch)
#
#   FUNBET (SYSTEM ONLY):
#     - Pool odds: 1.90–3.60
#     - 7 picks max, quota: max 2 picks with odds > 3.00
#     - Prefer lower odds when EV/conf close
#     - System preference: 4/7 first (threshold 0.65), else 5/7; n=6 => 3/6 then 4/6; n=5 => 3/5
#     - System weekly stake target 25–50 (unit derived)
#
#   DRAWBET (SYSTEM ONLY):
#     - Find 3–5 draws when they exist
#     - Draw odds: 2.80–3.70
#     - Filters: draw_prob >= 0.30, ev_x >= 0.05, and (draw_shape or balance_index high if present)
#     - System: 2/3, 2/4, 2-3/5 (depending on count)
#     - Weekly stake target 25–50 (unit derived)
#
#  Output:
#    - Keeps existing top-level keys "core" and "funbet" for compatibility
#    - Adds "drawbet" (additive)
#    - Adds copy_play (additive) to make presentation easy
# ============================================================

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
DEFAULT_BANKROLL_DRAW = float(os.getenv("BANKROLL_DRAW", "300"))

# Exposure caps (kept, but we DO NOT "weird-scale" stakes; only cap if absolutely needed)
CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.25"))  # allow up to 25% if needed (you can set 0.20)
FUN_EXPOSURE_CAP  = float(os.getenv("FUN_EXPOSURE_CAP",  "0.20"))
DRAW_EXPOSURE_CAP = float(os.getenv("DRAW_EXPOSURE_CAP", "0.20"))

# Gates
ODDS_MATCH_MIN_SCORE_CORE = float(os.getenv("ODDS_MATCH_MIN_SCORE_CORE", "0.75"))
ODDS_MATCH_MIN_SCORE_FUN  = float(os.getenv("ODDS_MATCH_MIN_SCORE_FUN",  "0.70"))
ODDS_MATCH_MIN_SCORE_DRAW = float(os.getenv("ODDS_MATCH_MIN_SCORE_DRAW", "0.70"))

CORE_MIN_CONFIDENCE = float(os.getenv("CORE_MIN_CONFIDENCE", "0.55"))
FUN_MIN_CONFIDENCE  = float(os.getenv("FUN_MIN_CONFIDENCE",  "0.45"))
DRAW_MIN_CONFIDENCE = float(os.getenv("DRAW_MIN_CONFIDENCE", "0.50"))

FUN_AVOID_CORE_OVERLAP = os.getenv("FUN_AVOID_CORE_OVERLAP", "false").lower() == "true"  # allow overlap by default

# ------------------------- MARKETS -------------------------
MARKET_CODE = {"Home":"1","Away":"2","Draw":"X","Over 2.5":"O25","Under 2.5":"U25"}

# ------------------------- COREBET RULES -------------------------
CORE_SINGLES_MIN_ODDS = float(os.getenv("CORE_SINGLES_MIN_ODDS", "1.60"))
CORE_SINGLES_MAX_ODDS = float(os.getenv("CORE_SINGLES_MAX_ODDS", "2.10"))

CORE_LOW_ODDS_MIN = float(os.getenv("CORE_LOW_ODDS_MIN", "1.30"))
CORE_LOW_ODDS_MAX = float(os.getenv("CORE_LOW_ODDS_MAX", "1.60"))

CORE_MIN_SINGLES = int(os.getenv("CORE_MIN_SINGLES", "5"))
CORE_MAX_SINGLES = int(os.getenv("CORE_MAX_SINGLES", "8"))

CORE_MAX_DOUBLES = int(os.getenv("CORE_MAX_DOUBLES", "2"))
CORE_DOUBLE_TARGET_MIN = float(os.getenv("CORE_DOUBLE_TARGET_MIN", "2.10"))
CORE_DOUBLE_TARGET_MAX = float(os.getenv("CORE_DOUBLE_TARGET_MAX", "3.00"))
CORE_DOUBLE_STAKE = float(os.getenv("CORE_DOUBLE_STAKE", "15"))

# Market sanity filter for 1X2 (prevents "phantom value")
# mismatch = model_prob - implied_prob; if abs(mismatch) > threshold => block unless confidence very high
CORE_SANITY_MISMATCH_MAX = float(os.getenv("CORE_SANITY_MISMATCH_MAX", "0.16"))
CORE_SANITY_CONF_OVERRIDE = float(os.getenv("CORE_SANITY_CONF_OVERRIDE", "0.80"))

# Singles stake ladder
def core_single_stake(odds: float) -> float:
    if 1.60 <= odds <= 1.75:
        return 40.0
    if 1.75 < odds <= 1.90:
        return 30.0
    if 1.90 < odds <= 2.10:
        return 20.0
    return 0.0

# Under limitation (use Thursday flags when present)
CORE_MAX_UNDER_SHARE = float(os.getenv("CORE_MAX_UNDER_SHARE", "0.20"))

# ------------------------- FUNBET RULES (SYSTEM ONLY) -------------------------
FUN_PICKS_MIN = int(os.getenv("FUN_PICKS_MIN", "5"))
FUN_PICKS_MAX = int(os.getenv("FUN_PICKS_MAX", "7"))

FUN_ODDS_MIN = float(os.getenv("FUN_ODDS_MIN", "1.90"))
FUN_ODDS_MAX = float(os.getenv("FUN_ODDS_MAX", "3.60"))

FUN_ODDS_GT3_MAX_COUNT = int(os.getenv("FUN_ODDS_GT3_MAX_COUNT", "2"))  # max picks with odds>3.00 when n=7
FUN_COMPARE_CLOSE_EV = float(os.getenv("FUN_COMPARE_CLOSE_EV", "0.02"))
FUN_COMPARE_CLOSE_CONF = float(os.getenv("FUN_COMPARE_CLOSE_CONF", "0.08"))

# System preference / refund
SYS_REFUND_PRIMARY  = float(os.getenv("SYS_REFUND_PRIMARY", "0.65"))
SYS_REFUND_FALLBACK = float(os.getenv("SYS_REFUND_FALLBACK", "0.60"))  # if you want even looser
SYS_UNIT_BASE = float(os.getenv("SYS_UNIT_BASE", "1.0"))
SYS_TARGET_MIN = float(os.getenv("SYS_TARGET_MIN", "25.0"))
SYS_TARGET_MAX = float(os.getenv("SYS_TARGET_MAX", "50.0"))

# ------------------------- DRAWBET RULES (SYSTEM ONLY) -------------------------
DRAW_PICKS_MIN = int(os.getenv("DRAW_PICKS_MIN", "3"))
DRAW_PICKS_MAX = int(os.getenv("DRAW_PICKS_MAX", "5"))

DRAW_ODDS_MIN = float(os.getenv("DRAW_ODDS_MIN", "2.80"))
DRAW_ODDS_MAX = float(os.getenv("DRAW_ODDS_MAX", "3.70"))

DRAW_EV_MIN = float(os.getenv("DRAW_EV_MIN", "0.05"))
DRAW_P_MIN = float(os.getenv("DRAW_P_MIN", "0.30"))
DRAW_BALANCE_MIN = float(os.getenv("DRAW_BALANCE_MIN", "0.85"))  # if balance_index exists
DRAW_REQUIRE_SHAPE = os.getenv("DRAW_REQUIRE_SHAPE", "false").lower() == "true"  # if true, needs draw_shape

# ------------------------- HELPERS -------------------------
def safe_float(v, d=None):
    try:
        return float(v)
    except Exception:
        return d

def safe_int(v, d=None):
    try:
        return int(v)
    except Exception:
        return d

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def implied_prob_from_odds(odds: float):
    o = safe_float(odds, None)
    if o is None or o <= 1.0:
        return None
    return 1.0 / o

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

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_history(path: str) -> dict:
    if not os.path.exists(path):
        return {"week_count": 0, "weeks": {}, "core": {"bankroll_current": None}, "funbet": {"bankroll_current": None}, "drawbet": {"bankroll_current": None}}
    try:
        h = load_json(path)
        h.setdefault("week_count", 0)
        h.setdefault("weeks", {})
        h.setdefault("core", {}).setdefault("bankroll_current", None)
        h.setdefault("funbet", {}).setdefault("bankroll_current", None)
        h.setdefault("drawbet", {}).setdefault("bankroll_current", None)
        return h
    except Exception:
        return {"week_count": 0, "weeks": {}, "core": {"bankroll_current": None}, "funbet": {"bankroll_current": None}, "drawbet": {"bankroll_current": None}}

def get_week_fields(window: dict, history: dict):
    window_from = (window or {}).get("from")
    week_id = iso_week_id_from_window(window)
    if window_from and window_from in (history.get("weeks") or {}):
        week_no = int(history["weeks"][window_from].get("week_no") or 1)
    else:
        week_no = int(history.get("week_count") or 0) + 1
    return {"week_id": week_id, "week_no": week_no, "week_label": f"Week {week_no}", "window_from": window_from}

def odds_match_ok(fx, min_score):
    om = fx.get("odds_match") or {}
    if not om.get("matched"):
        return False
    return (safe_float(om.get("score"), 0.0) or 0.0) >= min_score

def confidence_value(fx):
    return safe_float((fx.get("flags") or {}).get("confidence"), None)

def confidence_ok(fx, min_conf):
    c = confidence_value(fx)
    if c is None:
        return True
    return c >= min_conf

def _odds_from_fx(fx, market_code: str):
    mc = (market_code or "").upper()
    if mc == "1":   return safe_float(fx.get("offered_1"), None)
    if mc == "2":   return safe_float(fx.get("offered_2"), None)
    if mc == "X":   return safe_float(fx.get("offered_x"), None)
    if mc == "O25": return safe_float(fx.get("offered_over_2_5"), None)
    if mc == "U25": return safe_float(fx.get("offered_under_2_5"), None)
    return None

def _prob_from_fx(fx, market_code: str):
    mc = (market_code or "").upper()
    if mc == "1":   return safe_float(fx.get("home_prob"), None)
    if mc == "2":   return safe_float(fx.get("away_prob"), None)
    if mc == "X":   return safe_float(fx.get("draw_prob"), None)
    if mc == "O25": return safe_float(fx.get("over_2_5_prob"), None)
    if mc == "U25": return safe_float(fx.get("under_2_5_prob"), None)
    return None

def _ev_from_fx(fx, market_code: str):
    mc = (market_code or "").upper()
    if mc == "1":   return safe_float(fx.get("ev_1"), None)
    if mc == "2":   return safe_float(fx.get("ev_2"), None)
    if mc == "X":   return safe_float(fx.get("ev_x"), None)
    if mc == "O25": return safe_float(fx.get("ev_over"), None)
    if mc == "U25": return safe_float(fx.get("ev_under"), None)
    return None

def _market_name_from_code(code: str) -> str:
    code = (code or "").upper()
    return {"1":"Home","2":"Away","X":"Draw","O25":"Over 2.5","U25":"Under 2.5"}.get(code, code)

def load_thursday_fixtures():
    data = load_json(THURSDAY_REPORT_PATH)
    if isinstance(data, dict) and "fixtures" in data:
        return data["fixtures"], data
    if isinstance(data, dict) and isinstance(data.get("report"), dict) and "fixtures" in data["report"]:
        return data["report"]["fixtures"], data["report"]
    raise KeyError("fixtures not found in Thursday report")

def build_pick_candidates(fixtures):
    out = []
    for fx in fixtures:
        fid = fx.get("fixture_id")
        if fid is None:
            continue
        for mcode in ["1", "2", "X", "O25", "U25"]:
            odds = _odds_from_fx(fx, mcode)
            if odds is None or odds <= 1.0:
                continue
            out.append({
                "pick_id": f"{fid}:{mcode}",
                "fixture_id": fid,
                "league": fx.get("league"),
                "date": fx.get("date"),
                "time": fx.get("time"),
                "home": fx.get("home"),
                "away": fx.get("away"),
                "match": f'{fx.get("home")} – {fx.get("away")}',
                "market_code": mcode,
                "market": _market_name_from_code(mcode),
                "odds": odds,
                "prob": _prob_from_fx(fx, mcode),
                "ev": _ev_from_fx(fx, mcode),
                "confidence": confidence_value(fx),
                "fx": fx,
            })
    return out

def sort_key_datetime(p):
    # Sort by date/time if present, else at end.
    d = p.get("date") or "9999-12-31"
    t = p.get("time") or "23:59"
    return f"{d} {t}"

# ------------------------- SYSTEM UTILS -------------------------
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

def _system_try(pool, r_try: int, label: str):
    n = len(pool)
    cols = _columns_for_r(n, r_try)
    if cols <= 0:
        return None
    rr = _refund_ratio_worst_case([p["odds"] for p in pool], r_try, cols)
    return {"label": label, "n": n, "columns": cols, "min_hits": r_try, "refund_ratio_min_hits": round(rr, 4)}

def _unit_for_target(columns: int, cap_total: float):
    # target total stake in [25,50] but cannot exceed cap_total
    if columns <= 0:
        return 0.0, 0.0
    base_total = columns * SYS_UNIT_BASE
    target_total = _clamp(base_total, SYS_TARGET_MIN, SYS_TARGET_MAX)
    final_total = min(target_total, cap_total)
    unit = round(final_total / columns, 2)
    stake = round(unit * columns, 2)
    return unit, stake

# ------------------------- COREBET -------------------------
def _strip_core(p, stake: float):
    return {
        "pick_id": p["pick_id"],
        "fixture_id": p["fixture_id"],
        "date": p.get("date"),
        "time": p.get("time"),
        "league": p.get("league"),
        "match": p.get("match"),
        "market": p.get("market"),
        "market_code": p.get("market_code"),
        "odds": round(float(p["odds"]), 3),
        "stake": round(float(stake), 2),
        "prob": p.get("prob"),
        "ev": p.get("ev"),
        "confidence": p.get("confidence"),
        "tag": "core_single",
    }

def corebet_select(picks, bankroll_core):
    cap_amount = bankroll_core * CORE_EXPOSURE_CAP

    # candidates
    singles_pool = []
    low_pool = []

    for p in picks:
        fx = p["fx"]
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_CORE):
            continue
        if not confidence_ok(fx, CORE_MIN_CONFIDENCE):
            continue

        m = p["market_code"]
        if m not in ("1", "2", "O25", "U25"):
            continue  # core excludes Draw

        odds = safe_float(p.get("odds"), None)
        pr = safe_float(p.get("prob"), None)
        evv = safe_float(p.get("ev"), None)
        if odds is None or pr is None or evv is None:
            continue

        # Market sanity filter for 1X2 only
        if m in ("1", "2"):
            imp = implied_prob_from_odds(odds)
            if imp is not None:
                mismatch = pr - imp
                if abs(mismatch) > CORE_SANITY_MISMATCH_MAX:
                    c = safe_float(p.get("confidence"), 0.0) or 0.0
                    if c < CORE_SANITY_CONF_OVERRIDE:
                        continue

        # Under limited & elite-only when possible
        if m == "U25":
            flags = fx.get("flags") or {}
            if flags.get("under_elite") is not True:
                continue

        if CORE_SINGLES_MIN_ODDS <= odds <= CORE_SINGLES_MAX_ODDS:
            st = core_single_stake(odds)
            if st <= 0:
                continue
            singles_pool.append({**p, "stake": st})
        elif CORE_LOW_ODDS_MIN <= odds < CORE_LOW_ODDS_MAX:
            # forced into doubles bucket
            low_pool.append(p)

    # Sort by EV then confidence then lower odds (tie-break)
    def sk(x):
        return (
            safe_float(x.get("ev"), -9999.0),
            safe_float(x.get("confidence"), 0.0),
            -safe_float(x.get("odds"), 99.0),
        )

    singles_pool.sort(key=sk, reverse=True)
    low_pool.sort(key=sk, reverse=True)

    # pick singles unique matches
    singles = []
    used = set()
    under_cnt = 0
    max_under = max(0, int(round(CORE_MAX_UNDER_SHARE * CORE_MAX_SINGLES)))

    for p in singles_pool:
        if len(singles) >= CORE_MAX_SINGLES:
            break
        if p["match"] in used:
            continue
        if p["market_code"] == "U25" and under_cnt >= max_under:
            continue
        singles.append(_strip_core(p, p["stake"]))
        used.add(p["match"])
        if p["market_code"] == "U25":
            under_cnt += 1

    # ensure min singles if possible
    if len(singles) < CORE_MIN_SINGLES:
        for p in singles_pool:
            if len(singles) >= CORE_MIN_SINGLES:
                break
            if p["match"] in used:
                continue
            if p["market_code"] == "U25" and under_cnt >= max_under:
                continue
            singles.append(_strip_core(p, p["stake"]))
            used.add(p["match"])
            if p["market_code"] == "U25":
                under_cnt += 1

    # build doubles from low odds bucket
    doubles = []
    if low_pool and CORE_MAX_DOUBLES > 0:
        # partner pool can include:
        # - low odds picks (1.30-1.60)
        # - or core singles range picks (1.60-2.10) but avoid reusing same match
        partner_pool = []
        for p in picks:
            fx = p["fx"]
            if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_CORE):
                continue
            if not confidence_ok(fx, CORE_MIN_CONFIDENCE):
                continue
            if p["market_code"] not in ("1", "2", "O25"):  # keep doubles simple
                continue
            o = safe_float(p.get("odds"), None)
            pr = safe_float(p.get("prob"), None)
            evv = safe_float(p.get("ev"), None)
            if o is None or pr is None or evv is None:
                continue
            if CORE_LOW_ODDS_MIN <= o <= CORE_SINGLES_MAX_ODDS:
                partner_pool.append(p)

        partner_pool.sort(key=sk, reverse=True)

        used_double = set()
        for leg1 in low_pool:
            if len(doubles) >= CORE_MAX_DOUBLES:
                break
            for leg2 in partner_pool:
                if leg2["match"] == leg1["match"]:
                    continue
                if leg1["match"] in used_double or leg2["match"] in used_double:
                    continue
                combo = safe_float(leg1["odds"], 1.0) * safe_float(leg2["odds"], 1.0)
                if not (CORE_DOUBLE_TARGET_MIN <= combo <= CORE_DOUBLE_TARGET_MAX):
                    continue
                doubles.append({
                    "combo_odds": round(combo, 2),
                    "stake": round(float(CORE_DOUBLE_STAKE), 2),
                    "tag": "core_double_lowodds",
                    "legs": [
                        {
                            "pick_id": leg1["pick_id"],
                            "fixture_id": leg1["fixture_id"],
                            "date": leg1.get("date"),
                            "time": leg1.get("time"),
                            "league": leg1.get("league"),
                            "match": leg1.get("match"),
                            "market": leg1.get("market"),
                            "market_code": leg1.get("market_code"),
                            "odds": round(float(leg1["odds"]), 3),
                        },
                        {
                            "pick_id": leg2["pick_id"],
                            "fixture_id": leg2["fixture_id"],
                            "date": leg2.get("date"),
                            "time": leg2.get("time"),
                            "league": leg2.get("league"),
                            "match": leg2.get("match"),
                            "market": leg2.get("market"),
                            "market_code": leg2.get("market_code"),
                            "odds": round(float(leg2["odds"]), 3),
                        },
                    ]
                })
                used_double.add(leg1["match"])
                used_double.add(leg2["match"])
                break

    # Hard cap only if absolutely needed (scale proportionally)
    open_total = sum(x["stake"] for x in singles) + sum(d["stake"] for d in doubles)
    if open_total > cap_amount and open_total > 0:
        s = cap_amount / open_total
        for x in singles:
            x["stake"] = round(x["stake"] * s, 2)
        for d in doubles:
            d["stake"] = round(d["stake"] * s, 2)
        open_total = sum(x["stake"] for x in singles) + sum(d["stake"] for d in doubles)

    core_double = doubles[0] if doubles else None
    meta = {
        "open": round(open_total, 2),
        "after_open": round(bankroll_core - open_total, 2),
        "picks_count": len(singles),
        "doubles_count": len(doubles),
    }
    return singles, core_double, doubles, meta

# ------------------------- FUNBET (SYSTEM ONLY) -------------------------
def _strip_fun(p):
    return {
        "pick_id": p["pick_id"],
        "fixture_id": p["fixture_id"],
        "date": p.get("date"),
        "time": p.get("time"),
        "league": p.get("league"),
        "match": p.get("match"),
        "market": p.get("market"),
        "market_code": p.get("market_code"),
        "odds": round(float(p["odds"]), 3),
        "prob": p.get("prob"),
        "ev": p.get("ev"),
        "confidence": p.get("confidence"),
    }

def fun_choose_system(pool):
    n = len(pool)
    if n < 5:
        return None, False, None

    # preference order: easiest first
    if n >= 7:
        # prefer 4/7
        t1 = _system_try(pool[:7], 4, "4/7")
        if t1 and (t1["refund_ratio_min_hits"] >= SYS_REFUND_PRIMARY):
            return t1, False, SYS_REFUND_PRIMARY
        t2 = _system_try(pool[:7], 5, "5/7")
        if t2 and (t2["refund_ratio_min_hits"] >= SYS_REFUND_PRIMARY):
            return t2, False, SYS_REFUND_PRIMARY
        # fallback
        if t1 and (t1["refund_ratio_min_hits"] >= SYS_REFUND_FALLBACK):
            return t1, True, SYS_REFUND_FALLBACK
        if t2 and (t2["refund_ratio_min_hits"] >= SYS_REFUND_FALLBACK):
            return t2, True, SYS_REFUND_FALLBACK
        return None, True, None

    if n == 6:
        t1 = _system_try(pool[:6], 3, "3/6")
        if t1 and (t1["refund_ratio_min_hits"] >= SYS_REFUND_PRIMARY):
            return t1, False, SYS_REFUND_PRIMARY
        t2 = _system_try(pool[:6], 4, "4/6")
        if t2 and (t2["refund_ratio_min_hits"] >= SYS_REFUND_PRIMARY):
            return t2, False, SYS_REFUND_PRIMARY
        if t1 and (t1["refund_ratio_min_hits"] >= SYS_REFUND_FALLBACK):
            return t1, True, SYS_REFUND_FALLBACK
        if t2 and (t2["refund_ratio_min_hits"] >= SYS_REFUND_FALLBACK):
            return t2, True, SYS_REFUND_FALLBACK
        return None, True, None

    # n==5
    t = _system_try(pool[:5], 3, "3/5")
    if t and (t["refund_ratio_min_hits"] >= SYS_REFUND_PRIMARY):
        return t, False, SYS_REFUND_PRIMARY
    if t and (t["refund_ratio_min_hits"] >= SYS_REFUND_FALLBACK):
        return t, True, SYS_REFUND_FALLBACK
    return None, True, None

def funbet_select(picks, bankroll_fun, core_fixture_ids):
    cap_total = bankroll_fun * FUN_EXPOSURE_CAP

    candidates = []
    for p in picks:
        fx = p["fx"]
        if p["market_code"] not in ("1", "2", "O25", "U25"):
            continue

        if FUN_AVOID_CORE_OVERLAP and p["fixture_id"] in core_fixture_ids:
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_FUN):
            continue
        if not confidence_ok(fx, FUN_MIN_CONFIDENCE):
            continue

        odds = safe_float(p.get("odds"), None)
        pr = safe_float(p.get("prob"), None)
        evv = safe_float(p.get("ev"), None)
        if odds is None or pr is None or evv is None:
            continue
        if not (FUN_ODDS_MIN <= odds <= FUN_ODDS_MAX):
            continue

        # Under: only if under_elite
        if p["market_code"] == "U25":
            if (fx.get("flags") or {}).get("under_elite") is not True:
                continue

        candidates.append(p)

    # Sorting: EV, confidence, then prefer LOWER odds when close
    # We'll apply a small penalty to high odds, and a stronger penalty when >3.00.
    def fun_rank(x):
        o = safe_float(x.get("odds"), 99.0) or 99.0
        e = safe_float(x.get("ev"), -9999.0) or -9999.0
        c = safe_float(x.get("confidence"), 0.0) or 0.0
        penalty = 0.0
        if o > 3.00:
            penalty = 0.03  # knocks down close EV ties
        # Prefer lower odds slightly in general
        return (e - penalty, c, -o)

    candidates.sort(key=fun_rank, reverse=True)

    # pick up to 7 unique matches
    picks_out = []
    used_matches = set()
    for p in candidates:
        if p["match"] in used_matches:
            continue
        picks_out.append(p)
        used_matches.add(p["match"])
        if len(picks_out) >= FUN_PICKS_MAX:
            break

    # enforce quota: max 2 picks with odds > 3.00 when n==7 (or when >5 overall)
    def enforce_quota(lst):
        hi = [p for p in lst if safe_float(p.get("odds"), 0.0) > 3.00]
        if len(lst) >= 7 and len(hi) > FUN_ODDS_GT3_MAX_COUNT:
            # drop the worst "high odds" items (lowest EV/conf)
            hi_sorted = sorted(hi, key=lambda x: (safe_float(x.get("ev"), -9999.0), safe_float(x.get("confidence"), 0.0)))
            to_drop = len(hi) - FUN_ODDS_GT3_MAX_COUNT
            drop_set = set([id(x) for x in hi_sorted[:to_drop]])
            kept = [p for p in lst if id(p) not in drop_set]
            # backfill with next candidates not used
            for p in candidates:
                if len(kept) >= len(lst):
                    break
                if p["match"] in set([k["match"] for k in kept]):
                    continue
                if safe_float(p.get("odds"), 0.0) > 3.00 and sum(1 for k in kept if safe_float(k.get("odds"), 0.0) > 3.00) >= FUN_ODDS_GT3_MAX_COUNT:
                    continue
                kept.append(p)
            return kept[:len(lst)]
        return lst

    picks_out = enforce_quota(picks_out)

    # choose pool size (prefer 7 if available, else 6, else 5)
    if len(picks_out) >= 7:
        pool = picks_out[:7]
    elif len(picks_out) >= 6:
        pool = picks_out[:6]
    else:
        pool = picks_out[:5]

    sys_choice, breached, refund_used = fun_choose_system(pool)

    unit = stake = 0.0
    if sys_choice:
        unit, stake = _unit_for_target(int(sys_choice["columns"]), cap_total)

    payload = {
        "portfolio": "FunBet",
        "bankroll": DEFAULT_BANKROLL_FUN,
        "bankroll_start": bankroll_fun,
        "bankroll_source": "history" if bankroll_fun != DEFAULT_BANKROLL_FUN else "default",
        "exposure_cap_pct": FUN_EXPOSURE_CAP,
        "rules": {
            "odds_range": [FUN_ODDS_MIN, FUN_ODDS_MAX],
            "quota_max_gt3": FUN_ODDS_GT3_MAX_COUNT,
            "refund_primary": SYS_REFUND_PRIMARY,
            "refund_fallback": SYS_REFUND_FALLBACK,
            "refund_used": refund_used,
            "refund_rule_breached": bool(breached),
            "system_preference": "Prefer easier (4/7, 3/6) first",
            "target_total_range": [SYS_TARGET_MIN, SYS_TARGET_MAX],
        },
        "picks_total": [_strip_fun(p) for p in picks_out],
        "system_pool": [_strip_fun(p) for p in pool],
        "system": {
            "label": sys_choice["label"] if sys_choice else None,
            "columns": int(sys_choice["columns"]) if sys_choice else 0,
            "min_hits": int(sys_choice["min_hits"]) if sys_choice else None,
            "refund_ratio_min_hits": float(sys_choice["refund_ratio_min_hits"]) if sys_choice else None,
            "refund_used": refund_used,
            "refund_rule_breached": bool(breached),
            "unit": unit,
            "stake": stake,
            "has_system": bool(sys_choice is not None),
        },
        "open": round(stake, 2),
        "after_open": round(bankroll_fun - stake, 2),
        "counts": {"picks_total": len(picks_out), "system_pool": len(pool)},
    }
    return payload

# ------------------------- DRAWBET (SYSTEM ONLY) -------------------------
def _strip_draw(p):
    return {
        "pick_id": p["pick_id"],
        "fixture_id": p["fixture_id"],
        "date": p.get("date"),
        "time": p.get("time"),
        "league": p.get("league"),
        "match": p.get("match"),
        "market": "Draw",
        "market_code": "X",
        "odds": round(float(p["odds"]), 3),
        "prob": p.get("prob"),
        "ev": p.get("ev"),
        "confidence": p.get("confidence"),
    }

def draw_system_for_n(n: int):
    if n >= 5:
        # 2-3/5
        cols = comb(5, 2) + comb(5, 3)
        return {"label": "2-3/5", "columns": cols}
    if n == 4:
        cols = comb(4, 2)
        return {"label": "2/4", "columns": cols}
    if n == 3:
        cols = comb(3, 2)
        return {"label": "2/3", "columns": cols}
    return None

def drawbet_select(picks, bankroll_draw):
    cap_total = bankroll_draw * DRAW_EXPOSURE_CAP

    candidates = []
    for p in picks:
        fx = p["fx"]
        if p["market_code"] != "X":
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_DRAW):
            continue
        if not confidence_ok(fx, DRAW_MIN_CONFIDENCE):
            continue

        odds = safe_float(p.get("odds"), None)
        pr = safe_float(p.get("prob"), None)
        evv = safe_float(p.get("ev"), None)
        if odds is None or pr is None or evv is None:
            continue

        if not (DRAW_ODDS_MIN <= odds <= DRAW_ODDS_MAX):
            continue
        if pr < DRAW_P_MIN or evv < DRAW_EV_MIN:
            continue

        # use Thursday shape when present (optional)
        flags = fx.get("flags") or {}
        bal = safe_float(fx.get("balance_index"), None)
        draw_shape = bool(flags.get("draw_shape")) if ("draw_shape" in flags) else False

        if DRAW_REQUIRE_SHAPE and not draw_shape:
            continue

        # If balance exists, prefer higher balance. Do not hard-block unless very low.
        if bal is not None and bal < (DRAW_BALANCE_MIN - 0.10):
            continue

        candidates.append({**p, "balance": bal, "draw_shape": draw_shape})

    # rank by EV, then draw_shape, then balance, then confidence
    candidates.sort(
        key=lambda x: (
            safe_float(x.get("ev"), -9999.0),
            1.0 if x.get("draw_shape") else 0.0,
            safe_float(x.get("balance"), 0.0),
            safe_float(x.get("confidence"), 0.0),
            -safe_float(x.get("odds"), 99.0),
        ),
        reverse=True
    )

    pool = []
    used = set()
    for p in candidates:
        if p["match"] in used:
            continue
        pool.append(p)
        used.add(p["match"])
        if len(pool) >= DRAW_PICKS_MAX:
            break

    # require at least 3
    if len(pool) < DRAW_PICKS_MIN:
        pool = pool[:len(pool)]

    sys = draw_system_for_n(len(pool))
    unit = stake = 0.0
    if sys:
        unit, stake = _unit_for_target(int(sys["columns"]), cap_total)

    payload = {
        "portfolio": "DrawBet",
        "bankroll": DEFAULT_BANKROLL_DRAW,
        "bankroll_start": bankroll_draw,
        "bankroll_source": "history" if bankroll_draw != DEFAULT_BANKROLL_DRAW else "default",
        "exposure_cap_pct": DRAW_EXPOSURE_CAP,
        "rules": {
            "odds_range": [DRAW_ODDS_MIN, DRAW_ODDS_MAX],
            "min_prob": DRAW_P_MIN,
            "min_ev": DRAW_EV_MIN,
            "picks_range": [DRAW_PICKS_MIN, DRAW_PICKS_MAX],
            "target_total_range": [SYS_TARGET_MIN, SYS_TARGET_MAX],
        },
        "picks_total": [_strip_draw(p) for p in pool],
        "system_pool": [_strip_draw(p) for p in pool],
        "system": {
            "label": sys["label"] if sys else None,
            "columns": int(sys["columns"]) if sys else 0,
            "unit": unit,
            "stake": stake,
            "has_system": bool(sys is not None),
        },
        "open": round(stake, 2),
        "after_open": round(bankroll_draw - stake, 2),
        "counts": {"picks_total": len(pool), "system_pool": len(pool)},
    }
    return payload

# ------------------------- COPY PLAY -------------------------
def build_copy_play(core_singles, core_doubles, funbet, drawbet):
    lines = []

    # Core singles
    for p in sorted(core_singles, key=sort_key_datetime):
        lines.append({
            "portfolio": "CORE",
            "date": p.get("date"),
            "time": p.get("time"),
            "league": p.get("league"),
            "match": p.get("match"),
            "market": p.get("market"),
            "odds": p.get("odds"),
            "stake": p.get("stake"),
            "system": None
        })

    # Core doubles
    for d in core_doubles:
        lines.append({
            "portfolio": "CORE_DOUBLE",
            "date": None,
            "time": None,
            "league": None,
            "match": "DOUBLE",
            "market": "—",
            "odds": d.get("combo_odds"),
            "stake": d.get("stake"),
            "system": f'DOUBLE ({d.get("combo_odds")})'
        })

    # Fun system
    sys = (funbet.get("system") or {})
    if sys.get("has_system"):
        lines.append({
            "portfolio": "FUN_SYSTEM",
            "date": None,
            "time": None,
            "league": None,
            "match": "SYSTEM",
            "market": sys.get("label"),
            "odds": None,
            "stake": sys.get("stake"),
            "system": f'{sys.get("label")} | cols={sys.get("columns")} | unit={sys.get("unit")}'
        })

    # Draw system
    dsys = (drawbet.get("system") or {})
    if dsys.get("has_system"):
        lines.append({
            "portfolio": "DRAW_SYSTEM",
            "date": None,
            "time": None,
            "league": None,
            "match": "SYSTEM",
            "market": dsys.get("label"),
            "odds": None,
            "stake": dsys.get("stake"),
            "system": f'{dsys.get("label")} | cols={dsys.get("columns")} | unit={dsys.get("unit")}'
        })

    return lines

# ------------------------- MAIN -------------------------
def main():
    fixtures, th_meta = load_thursday_fixtures()
    history = load_history(TUESDAY_HISTORY_PATH)

    window = th_meta.get("window", {}) if isinstance(th_meta, dict) else {}
    wf = get_week_fields(window, history)

    core_start = safe_float(history.get("core", {}).get("bankroll_current"), None)
    fun_start  = safe_float(history.get("funbet", {}).get("bankroll_current"), None)
    draw_start = safe_float(history.get("drawbet", {}).get("bankroll_current"), None)

    core_bankroll_start = core_start if core_start is not None else DEFAULT_BANKROLL_CORE
    fun_bankroll_start  = fun_start  if fun_start  is not None else DEFAULT_BANKROLL_FUN
    draw_bankroll_start = draw_start if draw_start is not None else DEFAULT_BANKROLL_DRAW

    picks = build_pick_candidates(fixtures)

    core_singles, core_double, core_doubles, core_meta = corebet_select(picks, core_bankroll_start)
    core_fixture_ids = {x["fixture_id"] for x in core_singles}

    funbet = funbet_select(picks, fun_bankroll_start, core_fixture_ids)
    drawbet = drawbet_select(picks, draw_bankroll_start)

    # Copy play (additive)
    copy_play = build_copy_play(core_singles, core_doubles, funbet, drawbet)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "week_id": wf["week_id"],
        "week_no": wf["week_no"],
        "week_label": wf["week_label"],
        "window": window,
        "fixtures_total": th_meta.get("fixtures_total", len(fixtures)),

        "core": {
            "portfolio": "CoreBet",
            "bankroll": DEFAULT_BANKROLL_CORE,
            "bankroll_start": core_bankroll_start,
            "bankroll_source": ("history" if core_start is not None else "default"),
            "exposure_cap_pct": CORE_EXPOSURE_CAP,
            "rules": {
                "singles_odds_range": [CORE_SINGLES_MIN_ODDS, CORE_SINGLES_MAX_ODDS],
                "stake_ladder": {"1.60-1.75": 40, "1.75-1.90": 30, "1.90-2.10": 20},
                "low_odds_to_doubles": [CORE_LOW_ODDS_MIN, CORE_LOW_ODDS_MAX],
                "double_target_combo_odds": [CORE_DOUBLE_TARGET_MIN, CORE_DOUBLE_TARGET_MAX],
                "double_stake": CORE_DOUBLE_STAKE,
                "sanity_mismatch_max": CORE_SANITY_MISMATCH_MAX,
                "sanity_conf_override": CORE_SANITY_CONF_OVERRIDE,
                "max_singles": CORE_MAX_SINGLES,
            },
            "singles": sorted(core_singles, key=sort_key_datetime),
            "double": core_double,
            "doubles": core_doubles,
            "open": core_meta["open"],
            "after_open": core_meta["after_open"],
            "picks_count": core_meta["picks_count"],
            "doubles_count": core_meta["doubles_count"],
        },

        "funbet": {
            **funbet,
            "bankroll": DEFAULT_BANKROLL_FUN,
            "bankroll_start": fun_bankroll_start,
            "bankroll_source": ("history" if fun_start is not None else "default"),
        },

        "drawbet": {
            **drawbet,
            "bankroll": DEFAULT_BANKROLL_DRAW,
            "bankroll_start": draw_bankroll_start,
            "bankroll_source": ("history" if draw_start is not None else "default"),
        },

        "copy_play": copy_play,
    }

    os.makedirs("logs", exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
