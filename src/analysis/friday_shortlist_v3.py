# ============================================================
#  src/analysis/friday_shortlist_v3.py
#  FRIDAY SHORTLIST v3.21 — PRODUCTION
#
#  Fixes in this version:
#   1) FunBet is NOT EV-only: ranking uses Suitability -> Confidence -> EV -> Prob
#   2) Anti-underdog guard for FunBet Home/Away:
#        - odds > 3.00 allowed only if prob>=0.35 AND confidence>=0.70
#        - when 7 picks, max 2 picks can have odds > 3.00
#   3) DrawBet draw probability threshold default: >= 0.28
#
#  Reads:
#    - logs/thursday_report_v3.json
#    - (optional) logs/tuesday_history_v3.json   (bankroll carry + week numbering)
#  Writes:
#    - logs/friday_shortlist_v3.json
#
#  PORTFOLIOS
#   COREBET (bankroll 800, singles 1.70–3.50, stake ladder) + LOW-ODDS DOUBLES (<1.70)
#   FUNBET  (bankroll 400, SYSTEM-ONLY, 5–7 picks, prefer lower system difficulty)
#   DRAWBET (bankroll 300, SYSTEM-ONLY, 2–5 draws if available)
# ============================================================

import os
import json
from datetime import datetime, date
from math import comb

THURSDAY_REPORT_PATH = os.getenv("THURSDAY_REPORT_PATH", "logs/thursday_report_v3.json")
FRIDAY_REPORT_PATH = os.getenv("FRIDAY_REPORT_PATH", "logs/friday_shortlist_v3.json")
TUESDAY_HISTORY_PATH = os.getenv("TUESDAY_HISTORY_PATH", "logs/tuesday_history_v3.json")

# ------------------------- DEFAULT BANKROLLS -------------------------
DEFAULT_BANKROLL_CORE = float(os.getenv("BANKROLL_CORE", "800"))
DEFAULT_BANKROLL_FUN = float(os.getenv("BANKROLL_FUN", "400"))
DEFAULT_BANKROLL_DRAW = float(os.getenv("BANKROLL_DRAW", "300"))

# Exposure caps
CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.15"))
FUN_EXPOSURE_CAP = float(os.getenv("FUN_EXPOSURE_CAP", "0.20"))
DRAW_EXPOSURE_CAP = float(os.getenv("DRAW_EXPOSURE_CAP", "0.20"))

# Gates
ODDS_MATCH_MIN_SCORE_CORE = float(os.getenv("ODDS_MATCH_MIN_SCORE_CORE", "0.75"))
ODDS_MATCH_MIN_SCORE_FUN = float(os.getenv("ODDS_MATCH_MIN_SCORE_FUN", "0.75"))
ODDS_MATCH_MIN_SCORE_DRAW = float(os.getenv("ODDS_MATCH_MIN_SCORE_DRAW", "0.80"))

CORE_MIN_CONFIDENCE = float(os.getenv("CORE_MIN_CONFIDENCE", "0.55"))
FUN_MIN_CONFIDENCE = float(os.getenv("FUN_MIN_CONFIDENCE", "0.45"))
DRAW_MIN_CONFIDENCE = float(os.getenv("DRAW_MIN_CONFIDENCE", "0.55"))

# ✅ define overlap flag
FUN_AVOID_CORE_OVERLAP = os.getenv("FUN_AVOID_CORE_OVERLAP", "true").lower() == "true"

# ------------------------- MARKETS -------------------------
MARKET_CODE = {"Home": "1", "Draw": "X", "Away": "2", "Over 2.5": "O25", "Under 2.5": "U25"}
CORE_ALLOWED_MARKETS = {"Home", "Away", "Over 2.5", "Under 2.5"}
FUN_ALLOWED_MARKETS = {"Home", "Away", "Over 2.5", "Under 2.5"}   # Draw handled by DrawBet
DRAW_ALLOWED_MARKETS = {"Draw"}

def _market_name_from_code(code: str) -> str:
    code = (code or "").upper()
    return {"1": "Home", "2": "Away", "X": "Draw", "O25": "Over 2.5", "U25": "Under 2.5"}.get(code, code)

# ------------------------- BASIC HELPERS -------------------------
def safe_float(v, d=None):
    try:
        return float(v)
    except Exception:
        return d

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_history():
    if not os.path.exists(TUESDAY_HISTORY_PATH):
        return {"week_count": 0, "weeks": {}, "core": {"bankroll_current": None}, "funbet": {"bankroll_current": None}, "drawbet": {"bankroll_current": None}}
    try:
        h = load_json(TUESDAY_HISTORY_PATH)
        h.setdefault("week_count", 0)
        h.setdefault("weeks", {})
        h.setdefault("core", {}).setdefault("bankroll_current", None)
        h.setdefault("funbet", {}).setdefault("bankroll_current", None)
        h.setdefault("drawbet", {}).setdefault("bankroll_current", None)
        return h
    except Exception:
        return {"week_count": 0, "weeks": {}, "core": {"bankroll_current": None}, "funbet": {"bankroll_current": None}, "drawbet": {"bankroll_current": None}}

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

def _total_lambda(fx):
    tl = safe_float(fx.get("total_lambda"), None)
    if tl is not None:
        return tl
    lh = safe_float(fx.get("lambda_home"), 0.0) or 0.0
    la = safe_float(fx.get("lambda_away"), 0.0) or 0.0
    return lh + la

def _abs_gap(fx):
    ag = safe_float(fx.get("abs_lambda_gap"), None)
    if ag is not None:
        return ag
    lh = safe_float(fx.get("lambda_home"), 0.0) or 0.0
    la = safe_float(fx.get("lambda_away"), 0.0) or 0.0
    return abs(lh - la)

def _ev_from_fx(fx, market_code: str):
    mc = (market_code or "").upper()
    if mc == "1": return safe_float(fx.get("ev_1"), None)
    if mc == "2": return safe_float(fx.get("ev_2"), None)
    if mc == "X": return safe_float(fx.get("ev_x"), None)
    if mc == "O25": return safe_float(fx.get("ev_over"), None)
    if mc == "U25": return safe_float(fx.get("ev_under"), None)
    return None

def _prob_from_fx(fx, market_code: str):
    mc = (market_code or "").upper()
    if mc == "1": return safe_float(fx.get("home_prob"), None)
    if mc == "2": return safe_float(fx.get("away_prob"), None)
    if mc == "X": return safe_float(fx.get("draw_prob"), None)
    if mc == "O25": return safe_float(fx.get("over_2_5_prob"), None)
    if mc == "U25": return safe_float(fx.get("under_2_5_prob"), None)
    return None

def _odds_from_fx(fx, market_code: str):
    mc = (market_code or "").upper()
    if mc == "1": return safe_float(fx.get("offered_1"), None)
    if mc == "2": return safe_float(fx.get("offered_2"), None)
    if mc == "X": return safe_float(fx.get("offered_x"), None)
    if mc == "O25": return safe_float(fx.get("offered_over_2_5"), None)
    if mc == "U25": return safe_float(fx.get("offered_under_2_5"), None)
    return None

def _suitability_from_fx(fx, market_code: str):
    mc = (market_code or "").upper()
    if mc == "1": return safe_float(fx.get("suitability_home"), 0.0) or 0.0
    if mc == "2": return safe_float(fx.get("suitability_away"), 0.0) or 0.0
    if mc == "X": return safe_float(fx.get("suitability_draw"), 0.0) or 0.0
    if mc == "O25": return safe_float(fx.get("suitability_over"), 0.0) or 0.0
    if mc == "U25": return safe_float(fx.get("suitability_under"), 0.0) or 0.0
    return 0.0

# ------------------------- THURSDAY LOAD -------------------------
def load_thursday_fixtures():
    data = load_json(THURSDAY_REPORT_PATH)
    if "fixtures" in data:
        return data["fixtures"], data
    if "report" in data and isinstance(data["report"], dict) and "fixtures" in data["report"]:
        return data["report"]["fixtures"], data["report"]
    raise KeyError("fixtures not found in Thursday report")

# ------------------------- BUILD CANDIDATES -------------------------
def build_pick_candidates(fixtures):
    out = []
    for fx in fixtures:
        for mcode in ["1", "2", "X", "O25", "U25"]:
            odds = _odds_from_fx(fx, mcode)
            if odds is None or odds <= 1.0:
                continue
            out.append({
                "pick_id": f'{fx.get("fixture_id")}:{mcode}',
                "fixture_id": fx.get("fixture_id"),
                "date": fx.get("date"),
                "time": fx.get("time"),
                "match": f'{fx.get("home")} – {fx.get("away")}',
                "league": fx.get("league"),
                "market_code": mcode,
                "market": _market_name_from_code(mcode),
                "odds": odds,
                "prob": _prob_from_fx(fx, mcode),
                "ev": _ev_from_fx(fx, mcode),
                "confidence": confidence_value(fx),
                "suitability": _suitability_from_fx(fx, mcode),
                "fx": fx,
            })
    return out

# ============================================================
# COREBET RULES (unchanged here)
# ============================================================
CORE_SINGLES_MIN_ODDS = float(os.getenv("CORE_SINGLES_MIN_ODDS", "1.70"))
CORE_SINGLES_MAX_ODDS = float(os.getenv("CORE_SINGLES_MAX_ODDS", "3.50"))
CORE_LOW_ODDS_MIN = float(os.getenv("CORE_LOW_ODDS_MIN", "1.30"))
CORE_LOW_ODDS_MAX = float(os.getenv("CORE_LOW_ODDS_MAX", "1.69"))

CORE_MAX_SINGLES = int(os.getenv("CORE_MAX_SINGLES", "10"))
CORE_MIN_SINGLES = int(os.getenv("CORE_MIN_SINGLES", "5"))
CORE_MAX_DOUBLES = int(os.getenv("CORE_MAX_DOUBLES", "2"))
CORE_DOUBLE_TARGET_MIN = float(os.getenv("CORE_DOUBLE_TARGET_MIN", "2.10"))
CORE_DOUBLE_TARGET_MAX = float(os.getenv("CORE_DOUBLE_TARGET_MAX", "3.50"))

CORE_MIN_1X2_SHARE = float(os.getenv("CORE_MIN_1X2_SHARE", "0.30"))
CORE_MAX_UNDER_SHARE = float(os.getenv("CORE_MAX_UNDER_SHARE", "0.20"))

def core_single_stake(odds: float) -> float:
    if 1.70 <= odds <= 1.90: return 40.0
    if 1.90 < odds <= 2.20: return 30.0
    if 2.20 < odds <= 3.00: return 20.0
    if 3.00 < odds <= 3.50: return 15.0
    return 0.0

def core_double_stake(combo_odds: float) -> float:
    if combo_odds <= 2.70: return 20.0
    if combo_odds <= 3.50: return 15.0
    return 10.0

CORE_EV_MIN_HOME = float(os.getenv("CORE_EV_MIN_HOME", "0.04"))
CORE_EV_MIN_AWAY = float(os.getenv("CORE_EV_MIN_AWAY", "0.05"))
CORE_EV_MIN_OVER = float(os.getenv("CORE_EV_MIN_OVER", "0.04"))
CORE_EV_MIN_UNDER = float(os.getenv("CORE_EV_MIN_UNDER", "0.08"))

CORE_P_MIN_HOME = float(os.getenv("CORE_P_MIN_HOME", "0.30"))
CORE_P_MIN_AWAY = float(os.getenv("CORE_P_MIN_AWAY", "0.24"))
CORE_P_MIN_OVER = float(os.getenv("CORE_P_MIN_OVER", "0.45"))
CORE_P_MIN_UNDER = float(os.getenv("CORE_P_MIN_UNDER", "0.58"))

CORE_UNDER_LTOTAL_MAX = float(os.getenv("CORE_UNDER_LTOTAL_MAX", "2.30"))
CORE_UNDER_DRAW_MIN = float(os.getenv("CORE_UNDER_DRAW_MIN", "0.30"))
CORE_UNDER_ABS_GAP_MAX = float(os.getenv("CORE_UNDER_ABS_GAP_MAX", "0.35"))

def _strip_core(p):
    return {
        "pick_id": p["pick_id"],
        "fixture_id": p["fixture_id"],
        "date": p.get("date"),
        "time": p.get("time"),
        "market_code": p["market_code"],
        "match": p["match"],
        "league": p["league"],
        "market": p["market"],
        "odds": round(float(p["odds"]), 3),
        "prob": p.get("prob"),
        "ev": p.get("ev"),
        "stake": round(float(p.get("stake", 0.0)), 2),
        "tag": p.get("tag", "core_single"),
    }

def corebet_select(picks, bankroll_core):
    cap_amount = bankroll_core * CORE_EXPOSURE_CAP
    singles_pool = []
    low_pool = []

    for p in picks:
        fx = p["fx"]
        if p["market"] not in CORE_ALLOWED_MARKETS:
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_CORE):
            continue
        if not confidence_ok(fx, CORE_MIN_CONFIDENCE):
            continue

        odds = safe_float(p["odds"], None)
        evv = safe_float(p.get("ev"), None)
        pr = safe_float(p.get("prob"), None)
        if odds is None or evv is None or pr is None:
            continue

        if p["market"] == "Home" and (evv < CORE_EV_MIN_HOME or pr < CORE_P_MIN_HOME): continue
        if p["market"] == "Away" and (evv < CORE_EV_MIN_AWAY or pr < CORE_P_MIN_AWAY): continue
        if p["market"] == "Over 2.5" and (evv < CORE_EV_MIN_OVER or pr < CORE_P_MIN_OVER): continue
        if p["market"] == "Under 2.5":
            if evv < CORE_EV_MIN_UNDER or pr < CORE_P_MIN_UNDER: continue
            if (fx.get("flags") or {}).get("tight_game") is not True: continue
            if safe_float(fx.get("draw_prob"), 0.0) < CORE_UNDER_DRAW_MIN: continue
            if _total_lambda(fx) > CORE_UNDER_LTOTAL_MAX: continue
            if _abs_gap(fx) > CORE_UNDER_ABS_GAP_MAX: continue

        if CORE_SINGLES_MIN_ODDS <= odds <= CORE_SINGLES_MAX_ODDS:
            st = core_single_stake(odds)
            if st > 0:
                singles_pool.append({**p, "stake": st})
        elif CORE_LOW_ODDS_MIN <= odds <= CORE_LOW_ODDS_MAX:
            low_pool.append(p)

    # keep existing core ranking
    singles_pool.sort(key=lambda x: (safe_float(x.get("ev"), -9999.0), safe_float(x.get("confidence"), 0.0), safe_float(x.get("prob"), 0.0)), reverse=True)
    low_pool.sort(key=lambda x: (safe_float(x.get("ev"), -9999.0), safe_float(x.get("confidence"), 0.0), safe_float(x.get("prob"), 0.0)), reverse=True)

    singles = []
    used_matches = set()
    under_count = 0
    max_under = max(1, int(round(CORE_MAX_UNDER_SHARE * CORE_MAX_SINGLES)))
    target_min_1x2 = max(1, int(round(CORE_MIN_1X2_SHARE * CORE_MIN_SINGLES)))

    one_two_count = 0
    for p in singles_pool:
        if len(singles) >= CORE_MAX_SINGLES: break
        if p["match"] in used_matches: continue
        if p["market"] not in ("Home", "Away"): continue
        singles.append(_strip_core(p)); used_matches.add(p["match"]); one_two_count += 1
        if one_two_count >= target_min_1x2 and len(singles) >= CORE_MIN_SINGLES:
            break

    for p in singles_pool:
        if len(singles) >= CORE_MAX_SINGLES: break
        if p["match"] in used_matches: continue
        if p["market"] == "Under 2.5" and under_count >= max_under: continue
        singles.append(_strip_core(p)); used_matches.add(p["match"])
        if p["market"] == "Under 2.5": under_count += 1

    doubles = []
    if low_pool and CORE_MAX_DOUBLES > 0:
        partner_candidates = [p for p in picks if p["market"] in CORE_ALLOWED_MARKETS]
        partner_candidates.sort(key=lambda x: (safe_float(x.get("ev"), -9999.0), safe_float(x.get("confidence"), 0.0), safe_float(x.get("prob"), 0.0)), reverse=True)

        used_double_matches = set()
        for leg1 in low_pool:
            if len(doubles) >= CORE_MAX_DOUBLES: break
            for leg2 in partner_candidates:
                if leg2["match"] == leg1["match"]: continue
                if leg1["match"] in used_double_matches or leg2["match"] in used_double_matches: continue
                combo = safe_float(leg1["odds"], 1.0) * safe_float(leg2["odds"], 1.0)
                if not (CORE_DOUBLE_TARGET_MIN <= combo <= CORE_DOUBLE_TARGET_MAX): continue
                stake = core_double_stake(combo)
                doubles.append({
                    "legs": [
                        {"pick_id": leg1["pick_id"], "fixture_id": leg1["fixture_id"], "match": leg1["match"], "market": leg1["market"], "market_code": leg1["market_code"], "odds": leg1["odds"]},
                        {"pick_id": leg2["pick_id"], "fixture_id": leg2["fixture_id"], "match": leg2["match"], "market": leg2["market"], "market_code": leg2["market_code"], "odds": leg2["odds"]},
                    ],
                    "combo_odds": round(combo, 2),
                    "stake": round(stake, 2),
                    "tag": "core_double_lowodds",
                })
                used_double_matches.add(leg1["match"]); used_double_matches.add(leg2["match"])
                break

    open_total = sum(x["stake"] for x in singles) + sum(d.get("stake", 0.0) for d in doubles)
    if open_total > cap_amount and open_total > 0:
        s = cap_amount / open_total
        for x in singles: x["stake"] = round(x["stake"] * s, 2)
        for d in doubles: d["stake"] = round(float(d.get("stake", 0.0)) * s, 2)
        open_total = sum(x["stake"] for x in singles) + sum(d.get("stake", 0.0) for d in doubles)

    meta = {"open": round(open_total, 2), "after_open": round(bankroll_core - open_total, 2), "picks_count": len(singles), "doubles_count": len(doubles), "scale_applied": None}
    return singles, (doubles[0] if doubles else None), doubles, meta

# ============================================================
# FUNBET — SYSTEM ONLY (modified ranking + anti-underdog guard)
# ============================================================
FUN_PICKS_MIN = int(os.getenv("FUN_PICKS_MIN", "5"))
FUN_PICKS_MAX = int(os.getenv("FUN_PICKS_MAX", "7"))

FUN_EV_MIN_HOME = float(os.getenv("FUN_EV_MIN_HOME", "0.05"))
FUN_EV_MIN_AWAY = float(os.getenv("FUN_EV_MIN_AWAY", "0.05"))
FUN_EV_MIN_OVER = float(os.getenv("FUN_EV_MIN_OVER", "0.05"))
FUN_EV_MIN_UNDER = float(os.getenv("FUN_EV_MIN_UNDER", "0.10"))

FUN_P_MIN_HOME = float(os.getenv("FUN_P_MIN_HOME", "0.28"))
FUN_P_MIN_AWAY = float(os.getenv("FUN_P_MIN_AWAY", "0.22"))
FUN_P_MIN_OVER = float(os.getenv("FUN_P_MIN_OVER", "0.42"))
FUN_P_MIN_UNDER = float(os.getenv("FUN_P_MIN_UNDER", "0.60"))

FUN_UNDER_LTOTAL_MAX = float(os.getenv("FUN_UNDER_LTOTAL_MAX", "2.25"))
FUN_UNDER_DRAW_MIN = float(os.getenv("FUN_UNDER_DRAW_MIN", "0.30"))
FUN_UNDER_ABS_GAP_MAX = float(os.getenv("FUN_UNDER_ABS_GAP_MAX", "0.35"))

# Anti-underdog guard (NEW)
FUN_1X2_ODDS_SOFT_CAP = float(os.getenv("FUN_1X2_ODDS_SOFT_CAP", "3.00"))
FUN_1X2_ODDS_HARD_CAP = float(os.getenv("FUN_1X2_ODDS_HARD_CAP", "3.30"))
FUN_1X2_ODDS_HIGH_MAX_COUNT_IN_7 = int(os.getenv("FUN_1X2_ODDS_HIGH_MAX_COUNT_IN_7", "2"))
FUN_1X2_HIGH_MIN_PROB = float(os.getenv("FUN_1X2_HIGH_MIN_PROB", "0.35"))
FUN_1X2_HIGH_MIN_CONF = float(os.getenv("FUN_1X2_HIGH_MIN_CONF", "0.70"))

SYS_REFUND_PRIMARY = float(os.getenv("SYS_REFUND_PRIMARY", "0.80"))
SYS_REFUND_FALLBACK = float(os.getenv("SYS_REFUND_FALLBACK", "0.65"))

SYS_UNIT_BASE = float(os.getenv("SYS_UNIT_BASE", "1.0"))
SYS_TARGET_MIN = float(os.getenv("SYS_TARGET_MIN", "25.0"))
SYS_TARGET_MAX = float(os.getenv("SYS_TARGET_MAX", "50.0"))

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

def _choose_fun_system(pool):
    """
    Prefer lower difficulty:
      n=5 -> 3/5
      n=6 -> 3/6 else 4/6
      n=7 -> 4/7 else 5/7
    Allow fallback 0.65 and mark breach.
    """
    n = len(pool)
    if n < 5:
        return None, None, True, False

    if n == 5:
        trials = [_system_try(pool, 3, "3/5")]
    elif n == 6:
        trials = [_system_try(pool, 3, "3/6"), _system_try(pool, 4, "4/6")]
    else:
        trials = [_system_try(pool, 4, "4/7"), _system_try(pool, 5, "5/7")]

    for cand in trials:
        if cand and cand["refund_ratio_min_hits"] >= SYS_REFUND_PRIMARY:
            return cand, SYS_REFUND_PRIMARY, False, True

    # fallback ladder (prefer higher r when needed)
    if n == 6:
        fallback = [_system_try(pool, 4, "4/6"), _system_try(pool, 3, "3/6")]
    elif n == 7:
        fallback = [_system_try(pool, 4, "4/7"), _system_try(pool, 5, "5/7")]  # still prefer 4/7 first
    else:
        fallback = [_system_try(pool, 3, "3/5")]

    for cand in fallback:
        if cand and cand["refund_ratio_min_hits"] >= SYS_REFUND_FALLBACK:
            return cand, SYS_REFUND_FALLBACK, True, True

    return None, SYS_REFUND_PRIMARY, True, False

def _system_unit(columns: int, cap_system: float):
    """
    Keep it simple:
      - Start from unit=1.00
      - If columns*1 is within [25,50] and <= cap_system => use unit=1
      - Else scale down to fit cap_system and <=50
    """
    if columns <= 0:
        return 0.0, 0.0

    # prefer 1€/column if it fits targets
    one_total = float(columns) * 1.0
    if (SYS_TARGET_MIN <= one_total <= SYS_TARGET_MAX) and (one_total <= cap_system):
        return 1.0, round(one_total, 2)

    # otherwise target to [25,50] but also <= cap
    target = _clamp(one_total, SYS_TARGET_MIN, SYS_TARGET_MAX)
    target = min(target, cap_system)
    unit = round(target / float(columns), 2)
    stake = round(unit * float(columns), 2)
    return unit, stake

def _strip_pick(p):
    return {
        "pick_id": p["pick_id"],
        "fixture_id": p["fixture_id"],
        "date": p.get("date"),
        "time": p.get("time"),
        "market_code": p["market_code"],
        "match": p["match"],
        "league": p["league"],
        "market": p["market"],
        "odds": round(float(p["odds"]), 3),
        "prob": p.get("prob"),
        "ev": p.get("ev"),
        "confidence": p.get("confidence"),
        "suitability": p.get("suitability"),
    }

def funbet_select(picks, bankroll_fun, core_fixture_ids):
    cap_total = bankroll_fun * FUN_EXPOSURE_CAP

    candidates = []
    for p in picks:
        fx = p["fx"]
        if p["market"] not in FUN_ALLOWED_MARKETS:
            continue
        if FUN_AVOID_CORE_OVERLAP and p["fixture_id"] in core_fixture_ids:
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_FUN):
            continue
        if not confidence_ok(fx, FUN_MIN_CONFIDENCE):
            continue

        odds = safe_float(p["odds"], None)
        evv = safe_float(p.get("ev"), None)
        pr = safe_float(p.get("prob"), None)
        if odds is None or evv is None or pr is None:
            continue

        # market thresholds
        if p["market"] == "Home" and (evv < FUN_EV_MIN_HOME or pr < FUN_P_MIN_HOME): continue
        if p["market"] == "Away" and (evv < FUN_EV_MIN_AWAY or pr < FUN_P_MIN_AWAY): continue
        if p["market"] == "Over 2.5" and (evv < FUN_EV_MIN_OVER or pr < FUN_P_MIN_OVER): continue
        if p["market"] == "Under 2.5":
            if evv < FUN_EV_MIN_UNDER or pr < FUN_P_MIN_UNDER: continue
            if (fx.get("flags") or {}).get("tight_game") is not True: continue
            if safe_float(fx.get("draw_prob"), 0.0) < FUN_UNDER_DRAW_MIN: continue
            if _total_lambda(fx) > FUN_UNDER_LTOTAL_MAX: continue
            if _abs_gap(fx) > FUN_UNDER_ABS_GAP_MAX: continue

        # Anti-underdog guard for 1X2 only (Home/Away)
        if p["market"] in ("Home", "Away"):
            if odds > FUN_1X2_ODDS_HARD_CAP:
                continue
            if odds > FUN_1X2_ODDS_SOFT_CAP:
                conf = safe_float(p.get("confidence"), 0.0) or 0.0
                if not (pr >= FUN_1X2_HIGH_MIN_PROB and conf >= FUN_1X2_HIGH_MIN_CONF):
                    continue

        candidates.append(p)

    # ✅ NOT EV-only: Suitability -> Confidence -> EV -> Prob
    def rank_key(x):
        return (
            safe_float(x.get("suitability"), 0.0) or 0.0,
            safe_float(x.get("confidence"), 0.0) or 0.0,
            safe_float(x.get("ev"), -9999.0) or -9999.0,
            safe_float(x.get("prob"), 0.0) or 0.0,
        )

    candidates.sort(key=rank_key, reverse=True)

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

    # ensure minimum if possible (still unique)
    if len(picks_out) < FUN_PICKS_MIN:
        for p in candidates:
            if p["match"] in used_matches:
                continue
            picks_out.append(p)
            used_matches.add(p["match"])
            if len(picks_out) >= FUN_PICKS_MIN:
                break

    # Apply max-2 high-odds (>3.00) rule when 7 picks
    if len(picks_out) >= 7:
        high = [p for p in picks_out if (p["market"] in ("Home", "Away") and safe_float(p["odds"], 0.0) > FUN_1X2_ODDS_SOFT_CAP)]
        if len(high) > FUN_1X2_ODDS_HIGH_MAX_COUNT_IN_7:
            # drop extra highs from the bottom (lowest rank)
            keep = set()
            # keep the best N highs
            highs_sorted = sorted(high, key=rank_key, reverse=True)
            for hp in highs_sorted[:FUN_1X2_ODDS_HIGH_MAX_COUNT_IN_7]:
                keep.add(hp["pick_id"])
            filtered = []
            for p in picks_out:
                if p in high and p["pick_id"] not in keep:
                    continue
                filtered.append(p)
            picks_out = filtered

    # choose pool size preference: 7 -> 6 -> 5
    if len(picks_out) >= 7:
        pool = picks_out[:7]
    elif len(picks_out) >= 6:
        pool = picks_out[:6]
    else:
        pool = picks_out[:5]

    sys_choice, refund_used, rule_breached, has_system = _choose_fun_system(pool)
    columns = int(sys_choice["columns"]) if sys_choice else 0

    unit, stake = (0.0, 0.0)
    if has_system and columns > 0:
        unit, stake = _system_unit(columns, cap_total)

    payload = {
        "bankroll": bankroll_fun,
        "exposure_cap_pct": FUN_EXPOSURE_CAP,
        "rules": {
            "picks_range": [FUN_PICKS_MIN, FUN_PICKS_MAX],
            "refund_primary": SYS_REFUND_PRIMARY,
            "refund_fallback": SYS_REFUND_FALLBACK,
            "refund_used": refund_used,
            "refund_rule_breached": bool(rule_breached),
            "ranking": "suitability -> confidence -> ev -> prob",
            "anti_underdog": {
                "soft_cap_odds": FUN_1X2_ODDS_SOFT_CAP,
                "hard_cap_odds": FUN_1X2_ODDS_HARD_CAP,
                "max_high_odds_in_7": FUN_1X2_ODDS_HIGH_MAX_COUNT_IN_7,
                "high_odds_requires": {"prob_min": FUN_1X2_HIGH_MIN_PROB, "confidence_min": FUN_1X2_HIGH_MIN_CONF},
            },
        },
        "picks_total": [_strip_pick(p) for p in picks_out],
        "system_pool": [_strip_pick(p) for p in pool],
        "system": {
            "label": None if not sys_choice else sys_choice["label"],
            "columns": columns,
            "min_hits": None if not sys_choice else int(sys_choice["min_hits"]),
            "refund_ratio_min_hits": None if not sys_choice else float(sys_choice["refund_ratio_min_hits"]),
            "unit": unit,
            "stake": stake,
            "refund_used": refund_used,
            "refund_rule_breached": bool(rule_breached),
            "has_system": bool(has_system),
        },
        "open": round(stake, 2),
        "after_open": round(bankroll_fun - stake, 2),
        "counts": {"picks_total": len(picks_out), "system_pool": len(pool)},
    }
    return payload

# ============================================================
# DRAWBET — SYSTEM ONLY (2–5 draws if available)
# ============================================================
DRAW_PICKS_MAX = int(os.getenv("DRAW_PICKS_MAX", "5"))
DRAW_ODDS_MIN = float(os.getenv("DRAW_ODDS_MIN", "2.80"))
DRAW_ODDS_MAX = float(os.getenv("DRAW_ODDS_MAX", "3.80"))
DRAW_EV_MIN = float(os.getenv("DRAW_EV_MIN", "0.07"))
DRAW_P_MIN = float(os.getenv("DRAW_P_MIN", "0.28"))  # ✅ 27-28% as requested
DRAW_LTOTAL_MAX = float(os.getenv("DRAW_LTOTAL_MAX", "2.60"))
DRAW_ABS_GAP_MAX = float(os.getenv("DRAW_ABS_GAP_MAX", "0.25"))

def drawbet_select(picks, bankroll_draw):
    cap_total = bankroll_draw * DRAW_EXPOSURE_CAP

    candidates = []
    for p in picks:
        fx = p["fx"]
        if p["market"] != "Draw":
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_DRAW):
            continue
        if not confidence_ok(fx, DRAW_MIN_CONFIDENCE):
            continue

        odds = safe_float(p["odds"], None)
        evv = safe_float(p.get("ev"), None)
        pr = safe_float(p.get("prob"), None)
        if odds is None or evv is None or pr is None:
            continue

        if not (DRAW_ODDS_MIN <= odds <= DRAW_ODDS_MAX):
            continue
        if evv < DRAW_EV_MIN or pr < DRAW_P_MIN:
            continue
        if _total_lambda(fx) > DRAW_LTOTAL_MAX:
            continue
        if _abs_gap(fx) > DRAW_ABS_GAP_MAX:
            continue

        candidates.append(p)

    candidates.sort(key=lambda x: (safe_float(x.get("suitability"), 0.0), safe_float(x.get("confidence"), 0.0), safe_float(x.get("ev"), -9999.0), safe_float(x.get("prob"), 0.0)), reverse=True)

    pool = []
    used_matches = set()
    for p in candidates:
        if p["match"] in used_matches:
            continue
        pool.append(p)
        used_matches.add(p["match"])
        if len(pool) >= DRAW_PICKS_MAX:
            break

    n = len(pool)
    if n < 2:
        # no system
        return {
            "bankroll": bankroll_draw,
            "exposure_cap_pct": DRAW_EXPOSURE_CAP,
            "picks_total": [],
            "system_pool": [],
            "system": {"label": None, "columns": 0, "unit": 0.0, "stake": 0.0, "has_system": False},
            "open": 0.0,
            "after_open": round(bankroll_draw, 2),
            "counts": {"picks_total": 0, "system_pool": 0},
        }

    # system label by pool size
    if n >= 5:
        label = "2-3/5"
        columns = comb(5, 2) + comb(5, 3)  # 20
        pool = pool[:5]
    elif n == 4:
        label = "2/4"
        columns = comb(4, 2)  # 6
    elif n == 3:
        label = "2/3"
        columns = comb(3, 2)  # 3
    else:
        label = "2/2"
        columns = 1

    # stake targeting 25–50 within cap (unit base 1 if fits)
    one_total = float(columns) * 1.0
    target = _clamp(one_total, 25.0, 50.0)
    target = min(target, cap_total)
    unit = round(target / float(columns), 2) if columns > 0 else 0.0
    stake = round(unit * float(columns), 2)

    return {
        "bankroll": bankroll_draw,
        "exposure_cap_pct": DRAW_EXPOSURE_CAP,
        "rules": {"odds_range": [DRAW_ODDS_MIN, DRAW_ODDS_MAX], "p_min": DRAW_P_MIN, "system_by_n": True},
        "picks_total": [_strip_pick(p) for p in pool],
        "system_pool": [_strip_pick(p) for p in pool],
        "system": {"label": label, "columns": int(columns), "unit": unit, "stake": stake, "has_system": True},
        "open": round(stake, 2),
        "after_open": round(bankroll_draw - stake, 2),
        "counts": {"picks_total": len(pool), "system_pool": len(pool)},
    }

# ------------------------- MAIN -------------------------
def main():
    fixtures, th_meta = load_thursday_fixtures()
    history = load_history()

    window = th_meta.get("window", {}) if isinstance(th_meta, dict) else {}
    wf = get_week_fields(window, history)

    core_start = safe_float(history.get("core", {}).get("bankroll_current"), None)
    fun_start = safe_float(history.get("funbet", {}).get("bankroll_current"), None)
    draw_start = safe_float(history.get("drawbet", {}).get("bankroll_current"), None)

    core_bankroll_start = core_start if core_start is not None else DEFAULT_BANKROLL_CORE
    fun_bankroll_start = fun_start if fun_start is not None else DEFAULT_BANKROLL_FUN
    draw_bankroll_start = draw_start if draw_start is not None else DEFAULT_BANKROLL_DRAW

    picks = build_pick_candidates(fixtures)

    core_singles, core_double, core_doubles, core_meta = corebet_select(picks, core_bankroll_start)
    core_fixture_ids = {x["fixture_id"] for x in core_singles}

    funbet = funbet_select(picks, fun_bankroll_start, core_fixture_ids)
    drawbet = drawbet_select(picks, draw_bankroll_start)

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
            "singles": core_singles,
            "double": core_double,
            "doubles": core_doubles,
            "open": core_meta["open"],
            "after_open": core_meta["after_open"],
            "picks_count": core_meta["picks_count"],
            "doubles_count": core_meta["doubles_count"],
            "scale_applied": core_meta["scale_applied"],
        },

        "funbet": {
            "portfolio": "FunBet",
            "bankroll": DEFAULT_BANKROLL_FUN,
            "bankroll_start": fun_bankroll_start,
            "bankroll_source": ("history" if fun_start is not None else "default"),
            **funbet,
        },

        "drawbet": {
            "portfolio": "DrawBet",
            "bankroll": DEFAULT_BANKROLL_DRAW,
            "bankroll_start": draw_bankroll_start,
            "bankroll_source": ("history" if draw_start is not None else "default"),
            **drawbet,
        },
    }

    os.makedirs("logs", exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
