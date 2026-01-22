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
#  PORTFOLIOS (current locked rules)
#   COREBET:
#     - Singles odds: 1.60–2.10 (max 7–8 picks)
#     - Stake ladder:
#         1.60–1.75 => 40
#         1.75–1.90 => 30
#         1.90–2.10 => 20
#     - LOW ODDS (1.30–1.60): NEVER as singles, only as double legs
#     - Doubles: built only if we have low-odds candidates
#       combo target: configurable; stake default 15
#     - Under: selection capped to 20% of Core singles (and still gated by strict Under rules)
#
#   FUNBET (SYSTEM ONLY):
#     - Pool size target: 5–7
#     - Prefer "lower system" (easier hit-rate):
#         n=7 -> try 4/7 (fallback 5/7 if needed)
#         n=6 -> try 3/6 (fallback 4/6)
#         n=5 -> 3/5
#       Refund threshold is relaxed:
#         primary=0.65 (no 0.80 requirement)
#     - Unit policy: default unit=1.00 when columns in [25..50]
#       else scale up/down to keep total stake inside [25..50] when possible.
#     - NEW locked constraint:
#         If pool has 7 picks -> max 2 picks with odds > 3.00
#         If pool has 5–6 picks -> max 1 pick with odds > 3.00
#       (also hard odds max for fun picks)
#     - Tie-break: when EV+confidence are close, prefer lower odds.
#
#   DRAWBET (SYSTEM ONLY):
#     - Find 2..5 draws (if available)
#     - Odds range: 2.80–3.70 (configurable)
#     - Systems:
#         n=5 -> 2-3/5
#         n=4 -> 2/4
#         n=3 -> 2/3
#         n=2 -> 2/2
#     - Unit policy to keep total in [25..50] when possible.
#
#  Notes:
#   - Uses ONLY Thursday JSON fields (no extra API calls).
#   - Adds date/time fields into picks for chronological ordering (additive).
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

# Exposure caps (we do NOT scale Core singles ladder; exposure is reported)
CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.30"))  # if you want reporting only, keep high enough
FUN_EXPOSURE_CAP = float(os.getenv("FUN_EXPOSURE_CAP", "1.00"))    # system-only, we manage by SYS_TARGET range
DRAW_EXPOSURE_CAP = float(os.getenv("DRAW_EXPOSURE_CAP", "1.00"))  # system-only, we manage by SYS_TARGET range

# Gates
ODDS_MATCH_MIN_SCORE_CORE = float(os.getenv("ODDS_MATCH_MIN_SCORE_CORE", "0.75"))
ODDS_MATCH_MIN_SCORE_FUN = float(os.getenv("ODDS_MATCH_MIN_SCORE_FUN", "0.70"))
ODDS_MATCH_MIN_SCORE_DRAW = float(os.getenv("ODDS_MATCH_MIN_SCORE_DRAW", "0.75"))

CORE_MIN_CONFIDENCE = float(os.getenv("CORE_MIN_CONFIDENCE", "0.55"))
FUN_MIN_CONFIDENCE = float(os.getenv("FUN_MIN_CONFIDENCE", "0.45"))
DRAW_MIN_CONFIDENCE = float(os.getenv("DRAW_MIN_CONFIDENCE", "0.50"))

FUN_AVOID_CORE_OVERLAP = os.getenv("FUN_AVOID_CORE_OVERLAP", "false").lower() == "true"

# ------------------------- MARKETS -------------------------
MARKET_CODE = {"Home": "1", "Draw": "X", "Away": "2", "Over 2.5": "O25", "Under 2.5": "U25"}
CODE_TO_MARKET = {"1": "Home", "2": "Away", "X": "Draw", "O25": "Over 2.5", "U25": "Under 2.5"}

CORE_ALLOWED_MARKETS = {"Home", "Away", "Over 2.5", "Under 2.5"}
FUN_ALLOWED_MARKETS = {"Home", "Away", "Over 2.5", "Under 2.5"}  # Draws handled by DrawBet
DRAW_ALLOWED_MARKETS = {"Draw"}

# ------------------------- COREBET RULES -------------------------
CORE_SINGLES_MIN_ODDS = float(os.getenv("CORE_SINGLES_MIN_ODDS", "1.60"))
CORE_SINGLES_MAX_ODDS = float(os.getenv("CORE_SINGLES_MAX_ODDS", "2.10"))

CORE_LOW_ODDS_MIN = float(os.getenv("CORE_LOW_ODDS_MIN", "1.30"))
CORE_LOW_ODDS_MAX = float(os.getenv("CORE_LOW_ODDS_MAX", "1.60"))

CORE_MAX_SINGLES = int(os.getenv("CORE_MAX_SINGLES", "8"))
CORE_MIN_SINGLES = int(os.getenv("CORE_MIN_SINGLES", "5"))

CORE_MAX_DOUBLES = int(os.getenv("CORE_MAX_DOUBLES", "2"))
CORE_DOUBLE_TARGET_MIN = float(os.getenv("CORE_DOUBLE_TARGET_MIN", "2.00"))
CORE_DOUBLE_TARGET_MAX = float(os.getenv("CORE_DOUBLE_TARGET_MAX", "3.20"))
CORE_DOUBLE_STAKE = float(os.getenv("CORE_DOUBLE_STAKE", "15.0"))

# Under share cap (20%)
CORE_MAX_UNDER_SHARE = float(os.getenv("CORE_MAX_UNDER_SHARE", "0.20"))

# Core EV/prob gates (can be tuned via env)
CORE_EV_MIN_HOME = float(os.getenv("CORE_EV_MIN_HOME", "0.03"))
CORE_EV_MIN_AWAY = float(os.getenv("CORE_EV_MIN_AWAY", "0.04"))
CORE_EV_MIN_OVER = float(os.getenv("CORE_EV_MIN_OVER", "0.03"))
CORE_EV_MIN_UNDER = float(os.getenv("CORE_EV_MIN_UNDER", "0.08"))

CORE_P_MIN_HOME = float(os.getenv("CORE_P_MIN_HOME", "0.30"))
CORE_P_MIN_AWAY = float(os.getenv("CORE_P_MIN_AWAY", "0.24"))
CORE_P_MIN_OVER = float(os.getenv("CORE_P_MIN_OVER", "0.45"))
CORE_P_MIN_UNDER = float(os.getenv("CORE_P_MIN_UNDER", "0.58"))

# Strict Under gates (use Thursday additive fields/flags if present)
CORE_UNDER_LTOTAL_MAX = float(os.getenv("CORE_UNDER_LTOTAL_MAX", "2.30"))
CORE_UNDER_DRAW_MIN = float(os.getenv("CORE_UNDER_DRAW_MIN", "0.30"))
CORE_UNDER_ABS_GAP_MAX = float(os.getenv("CORE_UNDER_ABS_GAP_MAX", "0.35"))

def core_single_stake(odds: float) -> float:
    if 1.60 <= odds <= 1.75:
        return 40.0
    if 1.75 < odds <= 1.90:
        return 30.0
    if 1.90 < odds <= 2.10:
        return 20.0
    return 0.0

# ------------------------- FUNBET RULES (SYSTEM ONLY) -------------------------
FUN_PICKS_MIN = int(os.getenv("FUN_PICKS_MIN", "5"))
FUN_PICKS_MAX = int(os.getenv("FUN_PICKS_MAX", "7"))

FUN_ODDS_MIN = float(os.getenv("FUN_ODDS_MIN", "1.90"))
FUN_ODDS_MAX = float(os.getenv("FUN_ODDS_MAX", "3.60"))

# Max high odds in pool (>3.00)
FUN_HIGH_ODDS_THRESHOLD = float(os.getenv("FUN_HIGH_ODDS_THRESHOLD", "3.00"))
FUN_MAX_HIGH_ODDS_IN_7 = int(os.getenv("FUN_MAX_HIGH_ODDS_IN_7", "2"))
FUN_MAX_HIGH_ODDS_IN_5_6 = int(os.getenv("FUN_MAX_HIGH_ODDS_IN_5_6", "1"))

FUN_EV_MIN_HOME = float(os.getenv("FUN_EV_MIN_HOME", "0.04"))
FUN_EV_MIN_AWAY = float(os.getenv("FUN_EV_MIN_AWAY", "0.04"))
FUN_EV_MIN_OVER = float(os.getenv("FUN_EV_MIN_OVER", "0.04"))
FUN_EV_MIN_UNDER = float(os.getenv("FUN_EV_MIN_UNDER", "0.10"))

FUN_P_MIN_HOME = float(os.getenv("FUN_P_MIN_HOME", "0.28"))
FUN_P_MIN_AWAY = float(os.getenv("FUN_P_MIN_AWAY", "0.22"))
FUN_P_MIN_OVER = float(os.getenv("FUN_P_MIN_OVER", "0.42"))
FUN_P_MIN_UNDER = float(os.getenv("FUN_P_MIN_UNDER", "0.60"))

FUN_UNDER_LTOTAL_MAX = float(os.getenv("FUN_UNDER_LTOTAL_MAX", "2.25"))
FUN_UNDER_DRAW_MIN = float(os.getenv("FUN_UNDER_DRAW_MIN", "0.30"))
FUN_UNDER_ABS_GAP_MAX = float(os.getenv("FUN_UNDER_ABS_GAP_MAX", "0.35"))

# Refund threshold (relaxed)
SYS_REFUND_PRIMARY = float(os.getenv("SYS_REFUND_PRIMARY", "0.65"))

# System spend target
SYS_TARGET_MIN = float(os.getenv("SYS_TARGET_MIN", "25.0"))
SYS_TARGET_MAX = float(os.getenv("SYS_TARGET_MAX", "50.0"))

# Tie-break bucket size (when EV+confidence close)
TIE_EV_STEP = float(os.getenv("TIE_EV_STEP", "0.02"))
TIE_CONF_STEP = float(os.getenv("TIE_CONF_STEP", "0.05"))

# ------------------------- DRAWBET RULES (SYSTEM ONLY) -------------------------
DRAW_PICKS_MAX = int(os.getenv("DRAW_PICKS_MAX", "5"))
DRAW_PICKS_MIN = int(os.getenv("DRAW_PICKS_MIN", "2"))

DRAW_ODDS_MIN = float(os.getenv("DRAW_ODDS_MIN", "2.80"))
DRAW_ODDS_MAX = float(os.getenv("DRAW_ODDS_MAX", "3.70"))

DRAW_EV_MIN = float(os.getenv("DRAW_EV_MIN", "0.03"))
DRAW_P_MIN = float(os.getenv("DRAW_P_MIN", "0.24"))

DRAW_LTOTAL_MAX = float(os.getenv("DRAW_LTOTAL_MAX", "2.70"))
DRAW_ABS_GAP_MAX = float(os.getenv("DRAW_ABS_GAP_MAX", "0.30"))

# ------------------------- BASIC HELPERS -------------------------
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

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

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

# ------------------------- THURSDAY LOAD -------------------------
def load_thursday_fixtures():
    data = load_json(THURSDAY_REPORT_PATH)

    # common shapes:
    # 1) {"fixtures":[...], "window":...}
    if isinstance(data, dict) and "fixtures" in data and isinstance(data["fixtures"], list):
        return data["fixtures"], data

    # 2) {"status":"ok","report":{...}}
    if isinstance(data, dict) and isinstance(data.get("report"), dict):
        rep = data["report"]
        if "fixtures" in rep and isinstance(rep["fixtures"], list):
            return rep["fixtures"], rep

    raise KeyError("fixtures not found in Thursday report")

# ------------------------- BUILD PICK CANDIDATES -------------------------
def build_pick_candidates(fixtures):
    out = []
    for fx in fixtures:
        fx_date = fx.get("date")
        fx_time = fx.get("time")
        for mcode in ["1", "2", "X", "O25", "U25"]:
            odds = _odds_from_fx(fx, mcode)
            if odds is None or odds <= 1.0:
                continue
            out.append({
                "pick_id": f'{fx.get("fixture_id")}:{mcode}',
                "fixture_id": fx.get("fixture_id"),
                "date": fx_date,
                "time": fx_time,
                "match": f'{fx.get("home")} – {fx.get("away")}',
                "league": fx.get("league"),
                "market_code": mcode,
                "market": CODE_TO_MARKET.get(mcode, mcode),
                "odds": odds,
                "prob": _prob_from_fx(fx, mcode),
                "ev": _ev_from_fx(fx, mcode),
                "confidence": confidence_value(fx),
                "odds_match": fx.get("odds_match") or {},
                "flags": fx.get("flags") or {},
                "fx": fx,
            })
    return out

def _dt_key(p):
    # For chronological sort: (date, time)
    d = p.get("date") or ""
    t = p.get("time") or ""
    return (d, t, p.get("league") or "", p.get("match") or "")

# ------------------------- COREBET -------------------------
def _strip_core(p, stake):
    return {
        "pick_id": p["pick_id"],
        "fixture_id": p["fixture_id"],
        "date": p.get("date"),
        "time": p.get("time"),
        "league": p["league"],
        "match": p["match"],
        "market": p["market"],
        "market_code": p["market_code"],
        "odds": round(float(p["odds"]), 3),
        "stake": round(float(stake), 2),
        "prob": p.get("prob"),
        "ev": p.get("ev"),
        "tag": "core_single",
    }

def corebet_select(picks, bankroll_core):
    cap_amount = bankroll_core * CORE_EXPOSURE_CAP  # reported only

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

        # Market-specific gates
        if p["market"] == "Home" and (evv < CORE_EV_MIN_HOME or pr < CORE_P_MIN_HOME):
            continue
        if p["market"] == "Away" and (evv < CORE_EV_MIN_AWAY or pr < CORE_P_MIN_AWAY):
            continue
        if p["market"] == "Over 2.5" and (evv < CORE_EV_MIN_OVER or pr < CORE_P_MIN_OVER):
            continue
        if p["market"] == "Under 2.5":
            if evv < CORE_EV_MIN_UNDER or pr < CORE_P_MIN_UNDER:
                continue
            # strict under gates (prefer Thursday flags if present)
            if (fx.get("flags") or {}).get("under_elite") is True:
                pass
            else:
                if (fx.get("flags") or {}).get("tight_game") is not True:
                    continue
                if safe_float(fx.get("draw_prob"), 0.0) < CORE_UNDER_DRAW_MIN:
                    continue
                if _total_lambda(fx) > CORE_UNDER_LTOTAL_MAX:
                    continue
                if _abs_gap(fx) > CORE_UNDER_ABS_GAP_MAX:
                    continue

        if CORE_SINGLES_MIN_ODDS <= odds <= CORE_SINGLES_MAX_ODDS:
            st = core_single_stake(odds)
            if st <= 0:
                continue
            singles_pool.append((p, st))
        elif CORE_LOW_ODDS_MIN <= odds <= CORE_LOW_ODDS_MAX:
            low_pool.append(p)

    # Sort: EV desc, confidence desc, prob desc, odds asc
    def k_item(item):
        p, _st = item
        return (-safe_float(p.get("ev"), -9999.0), -(safe_float(p.get("confidence"), 0.0)), -(safe_float(p.get("prob"), 0.0)), safe_float(p.get("odds"), 9999.0))

    singles_pool.sort(key=k_item)
    low_pool.sort(key=lambda p: (-safe_float(p.get("ev"), -9999.0), -(safe_float(p.get("confidence"), 0.0)), -(safe_float(p.get("prob"), 0.0)), safe_float(p.get("odds"), 9999.0)))

    # Under cap
    max_under = max(0, int(round(CORE_MAX_UNDER_SHARE * CORE_MAX_SINGLES)))

    singles = []
    used_matches = set()
    under_count = 0

    for p, st in singles_pool:
        if len(singles) >= CORE_MAX_SINGLES:
            break
        if p["match"] in used_matches:
            continue
        if p["market"] == "Under 2.5" and under_count >= max_under:
            continue
        singles.append(_strip_core(p, st))
        used_matches.add(p["match"])
        if p["market"] == "Under 2.5":
            under_count += 1

    # Doubles only from low odds bucket
    doubles = []
    if low_pool and CORE_MAX_DOUBLES > 0:
        # partner candidates: core-eligible odds up to 2.10 (including low odds or singles odds)
        partners = []
        for p in picks:
            fx = p["fx"]
            if p["market"] not in CORE_ALLOWED_MARKETS:
                continue
            if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_CORE):
                continue
            if not confidence_ok(fx, CORE_MIN_CONFIDENCE):
                continue
            o = safe_float(p["odds"], None)
            if o is None:
                continue
            if not (CORE_LOW_ODDS_MIN <= o <= CORE_SINGLES_MAX_ODDS):
                continue
            partners.append(p)

        partners.sort(key=lambda p: (-safe_float(p.get("ev"), -9999.0), -(safe_float(p.get("confidence"), 0.0)), -(safe_float(p.get("prob"), 0.0)), safe_float(p.get("odds"), 9999.0)))

        used_double_matches = set()
        for leg1 in low_pool:
            if len(doubles) >= CORE_MAX_DOUBLES:
                break
            for leg2 in partners:
                if leg2["match"] == leg1["match"]:
                    continue
                if leg1["match"] in used_double_matches or leg2["match"] in used_double_matches:
                    continue
                combo = safe_float(leg1["odds"], 1.0) * safe_float(leg2["odds"], 1.0)
                if not (CORE_DOUBLE_TARGET_MIN <= combo <= CORE_DOUBLE_TARGET_MAX):
                    continue
                doubles.append({
                    "legs": [
                        {
                            "pick_id": leg1["pick_id"], "fixture_id": leg1["fixture_id"],
                            "date": leg1.get("date"), "time": leg1.get("time"),
                            "league": leg1["league"], "match": leg1["match"],
                            "market": leg1["market"], "market_code": leg1["market_code"],
                            "odds": round(float(leg1["odds"]), 3),
                        },
                        {
                            "pick_id": leg2["pick_id"], "fixture_id": leg2["fixture_id"],
                            "date": leg2.get("date"), "time": leg2.get("time"),
                            "league": leg2["league"], "match": leg2["match"],
                            "market": leg2["market"], "market_code": leg2["market_code"],
                            "odds": round(float(leg2["odds"]), 3),
                        },
                    ],
                    "combo_odds": round(combo, 2),
                    "stake": round(float(CORE_DOUBLE_STAKE), 2),
                    "tag": "core_double",
                })
                used_double_matches.add(leg1["match"])
                used_double_matches.add(leg2["match"])
                break

    # Chronological ordering
    singles.sort(key=_dt_key)
    doubles.sort(key=lambda d: (_dt_key(d["legs"][0]) if d.get("legs") else ("", "")))

    open_total = round(sum(x["stake"] for x in singles) + sum(d.get("stake", 0.0) for d in doubles), 2)

    meta = {
        "bankroll_start": bankroll_core,
        "exposure_cap_pct": CORE_EXPOSURE_CAP,
        "open": open_total,
        "after_open": round(bankroll_core - open_total, 2),
        "picks_count": len(singles),
        "doubles_count": len(doubles),
        "cap_amount": round(cap_amount, 2),
        "under_count": sum(1 for x in singles if x["market"] == "Under 2.5"),
    }
    core_double = doubles[0] if doubles else None
    return singles, core_double, doubles, meta

# ------------------------- FUN SYSTEM HELPERS -------------------------
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

def choose_fun_system(pool):
    """
    Prefer LOWER system always:
      n=7 -> 4/7 first, then 5/7
      n=6 -> 3/6 first, then 4/6
      n=5 -> 3/5
    Accept if refund_ratio_min_hits >= SYS_REFUND_PRIMARY (default 0.65).
    """
    n = len(pool)
    if n < 5:
        return None, False

    trials = []
    if n == 7:
        trials = [_system_try(pool, 4, "4/7"), _system_try(pool, 5, "5/7")]
    elif n == 6:
        trials = [_system_try(pool, 3, "3/6"), _system_try(pool, 4, "4/6")]
    else:
        trials = [_system_try(pool, 3, "3/5")]

    for t in trials:
        if t and (t.get("refund_ratio_min_hits") or 0.0) >= SYS_REFUND_PRIMARY:
            return t, True

    # if none passes, still return the first (lowest) for visibility, but mark has_system=False
    return (trials[0] if trials and trials[0] else None), False

def system_unit_for_target(columns: int):
    """
    Unit policy:
      - If columns in [25..50] => unit=1.00
      - If columns < 25 => unit up so stake hits at least 25 (cap 2 decimals)
      - If columns > 50 => unit down so stake is max 50
      - Always clamp stake into [25..50] when columns>0 if possible.
    """
    if columns <= 0:
        return 0.0, 0.0, 0.0

    # start with unit=1 when it fits
    if SYS_TARGET_MIN <= columns <= SYS_TARGET_MAX:
        unit = 1.0
        stake = float(columns) * unit
        return round(unit, 2), round(stake, 2), round(stake, 2)

    if columns < SYS_TARGET_MIN:
        # raise unit
        unit = SYS_TARGET_MIN / float(columns)
        stake = unit * columns
        # keep inside max
        if stake > SYS_TARGET_MAX:
            unit = SYS_TARGET_MAX / float(columns)
            stake = unit * columns
        return round(unit, 2), round(stake, 2), round(stake, 2)

    # columns > max -> lower unit
    unit = SYS_TARGET_MAX / float(columns)
    stake = unit * columns
    return round(unit, 2), round(stake, 2), round(stake, 2)

def _strip_pick(p):
    return {
        "pick_id": p["pick_id"],
        "fixture_id": p["fixture_id"],
        "date": p.get("date"),
        "time": p.get("time"),
        "league": p["league"],
        "match": p["match"],
        "market": p["market"],
        "market_code": p["market_code"],
        "odds": round(float(p["odds"]), 3),
        "prob": p.get("prob"),
        "ev": p.get("ev"),
        "confidence": p.get("confidence"),
    }

def funbet_select(picks, bankroll_fun, core_fixture_ids):
    # candidate filter + ranking
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

        if odds < FUN_ODDS_MIN or odds > FUN_ODDS_MAX:
            continue

        # thresholds by market
        if p["market"] == "Home" and (evv < FUN_EV_MIN_HOME or pr < FUN_P_MIN_HOME):
            continue
        if p["market"] == "Away" and (evv < FUN_EV_MIN_AWAY or pr < FUN_P_MIN_AWAY):
            continue
        if p["market"] == "Over 2.5" and (evv < FUN_EV_MIN_OVER or pr < FUN_P_MIN_OVER):
            continue
        if p["market"] == "Under 2.5":
            if evv < FUN_EV_MIN_UNDER or pr < FUN_P_MIN_UNDER:
                continue
            if (fx.get("flags") or {}).get("under_elite") is True:
                pass
            else:
                if (fx.get("flags") or {}).get("tight_game") is not True:
                    continue
                if safe_float(fx.get("draw_prob"), 0.0) < FUN_UNDER_DRAW_MIN:
                    continue
                if _total_lambda(fx) > FUN_UNDER_LTOTAL_MAX:
                    continue
                if _abs_gap(fx) > FUN_UNDER_ABS_GAP_MAX:
                    continue

        candidates.append(p)

    def tie_bucket(p):
        evv = safe_float(p.get("ev"), 0.0) or 0.0
        conf = safe_float(p.get("confidence"), 0.0) or 0.0
        ev_b = round(evv / TIE_EV_STEP) * TIE_EV_STEP if TIE_EV_STEP > 0 else evv
        cf_b = round(conf / TIE_CONF_STEP) * TIE_CONF_STEP if TIE_CONF_STEP > 0 else conf
        return (-ev_b, -cf_b, safe_float(p.get("odds"), 9999.0), -safe_float(p.get("prob"), 0.0))

    candidates.sort(key=tie_bucket)

    # build unique-match pool, enforce max high-odds count
    picks_out = []
    used_matches = set()

    def max_high_allowed(target_n: int):
        if target_n >= 7:
            return FUN_MAX_HIGH_ODDS_IN_7
        return FUN_MAX_HIGH_ODDS_IN_5_6

    # We don't know final N yet; we build greedily but enforce the 7 or 6/5 rule at the end.
    # Strategy: try build up to 7, then if violation -> drop highest-odds >3 picks first.
    for p in candidates:
        if len(picks_out) >= FUN_PICKS_MAX:
            break
        if p["match"] in used_matches:
            continue
        picks_out.append(p)
        used_matches.add(p["match"])

    # Ensure at least min
    if len(picks_out) < FUN_PICKS_MIN:
        # keep what exists; system may be disabled
        pass

    # Decide pool size we will use (prefer 7 if possible, else 6, else 5)
    if len(picks_out) >= 7:
        pool = picks_out[:7]
        allowed_high = max_high_allowed(7)
    elif len(picks_out) >= 6:
        pool = picks_out[:6]
        allowed_high = max_high_allowed(6)
    else:
        pool = picks_out[:5] if len(picks_out) >= 5 else picks_out[:]
        allowed_high = max_high_allowed(len(pool))

    # Enforce high-odds cap (>3.00)
    high = [p for p in pool if safe_float(p.get("odds"), 0.0) > FUN_HIGH_ODDS_THRESHOLD]
    if len(high) > allowed_high:
        # remove the highest odds among those >3 until satisfied
        pool_sorted = sorted(pool, key=lambda p: safe_float(p.get("odds"), 0.0), reverse=True)
        new_pool = []
        high_count = 0
        for p in pool_sorted:
            o = safe_float(p.get("odds"), 0.0)
            if o > FUN_HIGH_ODDS_THRESHOLD:
                if high_count >= allowed_high:
                    continue
                high_count += 1
            new_pool.append(p)
        # restore original ranking order for presentation (chronological later)
        pool = sorted(new_pool, key=tie_bucket)

    sys_choice, has_system = choose_fun_system(pool)
    cols = int(sys_choice["columns"]) if (sys_choice and has_system) else 0

    unit, stake, final_total = (0.0, 0.0, 0.0)
    if cols > 0:
        unit, stake, final_total = system_unit_for_target(cols)

    # sort pool chronologically for output
    pool_out = sorted(pool, key=_dt_key)

    payload = {
        "bankroll": bankroll_fun,
        "exposure_cap_pct": FUN_EXPOSURE_CAP,
        "rules": {
            "picks_range": [FUN_PICKS_MIN, FUN_PICKS_MAX],
            "odds_range": [FUN_ODDS_MIN, FUN_ODDS_MAX],
            "max_high_odds_threshold": FUN_HIGH_ODDS_THRESHOLD,
            "max_high_odds_in_7": FUN_MAX_HIGH_ODDS_IN_7,
            "max_high_odds_in_5_6": FUN_MAX_HIGH_ODDS_IN_5_6,
            "refund_threshold": SYS_REFUND_PRIMARY,
            "tie_break": "If EV+confidence close -> prefer lower odds",
            "unit_policy": "unit=1 if columns in [25..50] else scale to fit 25..50",
            "avoid_core_overlap": FUN_AVOID_CORE_OVERLAP,
        },
        "picks_total": [_strip_pick(p) for p in sorted(picks_out, key=_dt_key)],
        "system_pool": [_strip_pick(p) for p in pool_out],
        "system": {
            "label": (sys_choice["label"] if (sys_choice and has_system) else None),
            "columns": cols,
            "min_hits": (int(sys_choice["min_hits"]) if (sys_choice and has_system) else None),
            "refund_ratio_min_hits": (float(sys_choice["refund_ratio_min_hits"]) if (sys_choice and has_system) else None),
            "unit": unit,
            "stake": stake,
            "has_system": bool(has_system),
        },
        "open": round(stake, 2),
        "after_open": round(bankroll_fun - stake, 2),
        "counts": {"picks_total": len(picks_out), "system_pool": len(pool_out)},
    }
    return payload

# ------------------------- DRAWBET -------------------------
def choose_draw_system(n: int):
    if n >= 5:
        return "2-3/5", (comb(5, 2) + comb(5, 3)), 5
    if n == 4:
        return "2/4", comb(4, 2), 4
    if n == 3:
        return "2/3", comb(3, 2), 3
    if n == 2:
        return "2/2", comb(2, 2), 2
    return None, 0, n

def drawbet_select(picks, bankroll_draw):
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

        if odds < DRAW_ODDS_MIN or odds > DRAW_ODDS_MAX:
            continue
        if evv < DRAW_EV_MIN or pr < DRAW_P_MIN:
            continue
        if _total_lambda(fx) > DRAW_LTOTAL_MAX:
            continue
        if _abs_gap(fx) > DRAW_ABS_GAP_MAX:
            continue

        candidates.append(p)

    # rank by EV then confidence then lower odds
    candidates.sort(key=lambda p: (-safe_float(p.get("ev"), -9999.0), -(safe_float(p.get("confidence"), 0.0)), safe_float(p.get("odds"), 9999.0)))

    pool = []
    used_matches = set()
    for p in candidates:
        if p["match"] in used_matches:
            continue
        pool.append(p)
        used_matches.add(p["match"])
        if len(pool) >= DRAW_PICKS_MAX:
            break

    if len(pool) < DRAW_PICKS_MIN:
        pool = pool  # no system

    label, cols, n_used = choose_draw_system(len(pool))
    has_system = bool(label and cols > 0 and len(pool) >= 2)

    unit, stake, _final_total = (0.0, 0.0, 0.0)
    if has_system:
        unit, stake, _final_total = system_unit_for_target(cols)

    pool_out = sorted(pool, key=_dt_key)

    payload = {
        "bankroll": bankroll_draw,
        "exposure_cap_pct": DRAW_EXPOSURE_CAP,
        "rules": {
            "picks_min": DRAW_PICKS_MIN,
            "picks_max": DRAW_PICKS_MAX,
            "odds_range": [DRAW_ODDS_MIN, DRAW_ODDS_MAX],
            "systems": {"5": "2-3/5", "4": "2/4", "3": "2/3", "2": "2/2"},
            "unit_policy": "unit=1 if columns in [25..50] else scale to fit 25..50",
        },
        "picks_total": [_strip_pick(p) for p in pool_out],
        "system_pool": [_strip_pick(p) for p in pool_out],
        "system": {
            "label": label if has_system else None,
            "columns": cols if has_system else 0,
            "unit": unit,
            "stake": stake,
            "has_system": bool(has_system),
        },
        "open": round(stake, 2),
        "after_open": round(bankroll_draw - stake, 2),
        "counts": {"picks_total": len(pool_out), "system_pool": len(pool_out)},
    }
    return payload

# ------------------------- MAIN -------------------------
def main():
    fixtures, th_meta = load_thursday_fixtures()
    history = load_history()

    window = th_meta.get("window", {}) if isinstance(th_meta, dict) else {}
    wf = get_week_fields(window, history)

    core_start = safe_float((history.get("core") or {}).get("bankroll_current"), None)
    fun_start = safe_float((history.get("funbet") or {}).get("bankroll_current"), None)
    draw_start = safe_float((history.get("drawbet") or {}).get("bankroll_current"), None)

    core_bankroll_start = core_start if core_start is not None else DEFAULT_BANKROLL_CORE
    fun_bankroll_start = fun_start if fun_start is not None else DEFAULT_BANKROLL_FUN
    draw_bankroll_start = draw_start if draw_start is not None else DEFAULT_BANKROLL_DRAW

    picks = build_pick_candidates(fixtures)

    # CORE
    core_singles, core_double, core_doubles, core_meta = corebet_select(picks, core_bankroll_start)
    core_fixture_ids = {x["fixture_id"] for x in core_singles}

    # FUN (system-only)
    funbet = funbet_select(picks, fun_bankroll_start, core_fixture_ids)

    # DRAW (system-only)
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
            "bankroll_start": round(core_bankroll_start, 2),
            "bankroll_source": ("history" if core_start is not None else "default"),
            "exposure_cap_pct": CORE_EXPOSURE_CAP,
            "rules": {
                "singles_odds_range": [CORE_SINGLES_MIN_ODDS, CORE_SINGLES_MAX_ODDS],
                "low_odds_to_doubles": [CORE_LOW_ODDS_MIN, CORE_LOW_ODDS_MAX],
                "stake_ladder": {"1.60-1.75": 40, "1.75-1.90": 30, "1.90-2.10": 20},
                "max_singles": CORE_MAX_SINGLES,
                "max_under_share": CORE_MAX_UNDER_SHARE,
                "double_target_combo_odds": [CORE_DOUBLE_TARGET_MIN, CORE_DOUBLE_TARGET_MAX],
                "double_stake": CORE_DOUBLE_STAKE,
            },
            "singles": core_singles,
            "double": core_double,
            "doubles": core_doubles,
            "open": core_meta["open"],
            "after_open": core_meta["after_open"],
            "picks_count": core_meta["picks_count"],
            "doubles_count": core_meta["doubles_count"],
            "under_count": core_meta["under_count"],
            "cap_amount": core_meta["cap_amount"],
        },

        "funbet": {
            "portfolio": "FunBet",
            "bankroll": DEFAULT_BANKROLL_FUN,
            "bankroll_start": round(fun_bankroll_start, 2),
            "bankroll_source": ("history" if fun_start is not None else "default"),
            **funbet,
        },

        "drawbet": {
            "portfolio": "DrawBet",
            "bankroll": DEFAULT_BANKROLL_DRAW,
            "bankroll_start": round(draw_bankroll_start, 2),
            "bankroll_source": ("history" if draw_start is not None else "default"),
            **drawbet,
        },
    }

    os.makedirs("logs", exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
