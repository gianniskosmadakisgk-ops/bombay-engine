# ============================================================
#  src/analysis/friday_shortlist_v3.py
#  FRIDAY SHORTLIST v3.30 — PRODUCTION (CoreBet + FunBet + DrawBet)
#
#  What this version fixes (per latest agreement):
#   1) DrawBet: strict + fallback so draws can appear (2–5 picks) instead of often 0.
#   2) Cold doubles: low odds 1.30–1.60 are not lost; they are used in CORE doubles.
#   3) Fun system: prefers the "easier" system first (4/7 before 5/7, 3/6 before 4/6).
#   4) Unit policy: default 1.00 per column when stake stays within target (25–50) and cap.
#   5) Anti-UFO filter: Home/Away with odds > 3.00 are excluded (default), to avoid "ghost" picks.
#   6) No stake scaling for Core singles/doubles: if cap exceeded -> drop lowest-ranked picks.
#   7) Adds report["copy_play"] with Core/Fun/Draw sections + bankroll lines (Start/Open/After).
#
#  Reads:
#    - logs/thursday_report_v3.json
#    - (optional) logs/tuesday_history_v3.json  (week numbering + bankroll carry)
#
#  Writes:
#    - logs/friday_shortlist_v3.json
# ============================================================

import os
import json
from datetime import datetime, date
from math import comb
from typing import Any, Dict, List, Optional, Tuple

THURSDAY_REPORT_PATH = os.getenv("THURSDAY_REPORT_PATH", "logs/thursday_report_v3.json")
FRIDAY_REPORT_PATH = os.getenv("FRIDAY_REPORT_PATH", "logs/friday_shortlist_v3.json")
TUESDAY_HISTORY_PATH = os.getenv("TUESDAY_HISTORY_PATH", "logs/tuesday_history_v3.json")

# ------------------------- BANKROLLS -------------------------
DEFAULT_BANKROLL_CORE = float(os.getenv("BANKROLL_CORE", "800"))
DEFAULT_BANKROLL_FUN  = float(os.getenv("BANKROLL_FUN",  "400"))
DEFAULT_BANKROLL_DRAW = float(os.getenv("BANKROLL_DRAW", "300"))

# Exposure caps (used as hard caps; no scaling for Core stakes)
CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.15"))  # e.g. 120 on 800
FUN_EXPOSURE_CAP  = float(os.getenv("FUN_EXPOSURE_CAP",  "0.20"))  # e.g. 80 on 400
DRAW_EXPOSURE_CAP = float(os.getenv("DRAW_EXPOSURE_CAP", "0.20"))  # e.g. 60 on 300

# ------------------------- GATES -------------------------
ODDS_MATCH_MIN_SCORE_CORE = float(os.getenv("ODDS_MATCH_MIN_SCORE_CORE", "0.75"))
ODDS_MATCH_MIN_SCORE_FUN  = float(os.getenv("ODDS_MATCH_MIN_SCORE_FUN",  "0.70"))
ODDS_MATCH_MIN_SCORE_DRAW = float(os.getenv("ODDS_MATCH_MIN_SCORE_DRAW", "0.70"))

CORE_MIN_CONFIDENCE = float(os.getenv("CORE_MIN_CONFIDENCE", "0.55"))
FUN_MIN_CONFIDENCE  = float(os.getenv("FUN_MIN_CONFIDENCE",  "0.45"))
DRAW_MIN_CONFIDENCE = float(os.getenv("DRAW_MIN_CONFIDENCE", "0.45"))

FUN_AVOID_CORE_OVERLAP = os.getenv("FUN_AVOID_CORE_OVERLAP", "false").lower() == "true"

# ------------------------- MARKETS -------------------------
MARKET_CODE = {"Home": "1", "Draw": "X", "Away": "2", "Over 2.5": "O25", "Under 2.5": "U25"}
CODE_TO_MARKET = {v: k for k, v in MARKET_CODE.items()}

CORE_ALLOWED_MARKETS = {"Home", "Away", "Over 2.5", "Under 2.5"}
FUN_ALLOWED_MARKETS  = {"Home", "Away", "Over 2.5", "Under 2.5"}   # Draw handled by DrawBet
DRAW_ALLOWED_MARKETS = {"Draw"}

# ------------------------- CORE RULES (as agreed) -------------------------
# Singles are 1.60–2.10 (top focus); low odds 1.30–1.60 go to doubles
CORE_SINGLES_MIN_ODDS = float(os.getenv("CORE_SINGLES_MIN_ODDS", "1.60"))
CORE_SINGLES_MAX_ODDS = float(os.getenv("CORE_SINGLES_MAX_ODDS", "2.10"))

CORE_LOW_ODDS_MIN = float(os.getenv("CORE_LOW_ODDS_MIN", "1.30"))
CORE_LOW_ODDS_MAX = float(os.getenv("CORE_LOW_ODDS_MAX", "1.60"))

CORE_MAX_SINGLES = int(os.getenv("CORE_MAX_SINGLES", "8"))
CORE_MIN_SINGLES = int(os.getenv("CORE_MIN_SINGLES", "5"))

CORE_MAX_DOUBLES = int(os.getenv("CORE_MAX_DOUBLES", "2"))
CORE_DOUBLE_TARGET_MIN = float(os.getenv("CORE_DOUBLE_TARGET_MIN", "2.20"))
CORE_DOUBLE_TARGET_MAX = float(os.getenv("CORE_DOUBLE_TARGET_MAX", "3.40"))

# Stakes ladder (exact)
def core_single_stake(odds: float) -> float:
    # 1.60–1.75 => 40
    if 1.60 <= odds <= 1.75:
        return 40.0
    # 1.75–1.90 => 30
    if 1.75 < odds <= 1.90:
        return 30.0
    # 1.90–2.10 => 20
    if 1.90 < odds <= 2.10:
        return 20.0
    return 0.0

# Double stake (simple + stable)
CORE_DOUBLE_STAKE = float(os.getenv("CORE_DOUBLE_STAKE", "15.0"))

# Anti-UFO for Core 1/2 too (optional)
MAX_HOME_AWAY_ODDS = float(os.getenv("MAX_HOME_AWAY_ODDS", "3.00"))

# Core under strictness (kept tight)
CORE_EV_MIN_HOME  = float(os.getenv("CORE_EV_MIN_HOME",  "0.04"))
CORE_EV_MIN_AWAY  = float(os.getenv("CORE_EV_MIN_AWAY",  "0.05"))
CORE_EV_MIN_OVER  = float(os.getenv("CORE_EV_MIN_OVER",  "0.04"))
CORE_EV_MIN_UNDER = float(os.getenv("CORE_EV_MIN_UNDER", "0.10"))

CORE_P_MIN_HOME  = float(os.getenv("CORE_P_MIN_HOME",  "0.30"))
CORE_P_MIN_AWAY  = float(os.getenv("CORE_P_MIN_AWAY",  "0.24"))
CORE_P_MIN_OVER  = float(os.getenv("CORE_P_MIN_OVER",  "0.45"))
CORE_P_MIN_UNDER = float(os.getenv("CORE_P_MIN_UNDER", "0.60"))

# ------------------------- FUN SYSTEM RULES -------------------------
FUN_PICKS_MIN = int(os.getenv("FUN_PICKS_MIN", "7"))   # we aim 7 if available
FUN_PICKS_MAX = int(os.getenv("FUN_PICKS_MAX", "7"))

FUN_EV_MIN_HOME  = float(os.getenv("FUN_EV_MIN_HOME",  "0.05"))
FUN_EV_MIN_AWAY  = float(os.getenv("FUN_EV_MIN_AWAY",  "0.05"))
FUN_EV_MIN_OVER  = float(os.getenv("FUN_EV_MIN_OVER",  "0.05"))
FUN_EV_MIN_UNDER = float(os.getenv("FUN_EV_MIN_UNDER", "0.10"))

FUN_P_MIN_HOME  = float(os.getenv("FUN_P_MIN_HOME",  "0.28"))
FUN_P_MIN_AWAY  = float(os.getenv("FUN_P_MIN_AWAY",  "0.22"))
FUN_P_MIN_OVER  = float(os.getenv("FUN_P_MIN_OVER",  "0.40"))
FUN_P_MIN_UNDER = float(os.getenv("FUN_P_MIN_UNDER", "0.58"))

FUN_ODDS_MIN = float(os.getenv("FUN_ODDS_MIN", "1.90"))
FUN_ODDS_MAX = float(os.getenv("FUN_ODDS_MAX", "3.60"))

# "Only 2 picks >3.00 when 7 picks" (default 2)
FUN_ODDS_GT3_MAX_COUNT = int(os.getenv("FUN_ODDS_GT3_MAX_COUNT", "2"))
FUN_ODDS_GT3_THRESHOLD = float(os.getenv("FUN_ODDS_GT3_THRESHOLD", "3.00"))

# Refund thresholds
SYS_REFUND_PRIMARY  = float(os.getenv("SYS_REFUND_PRIMARY",  "0.80"))
SYS_REFUND_FALLBACK = float(os.getenv("SYS_REFUND_FALLBACK", "0.65"))

# Unit + spending
SYS_UNIT_BASE = float(os.getenv("SYS_UNIT_BASE", "1.00"))  # we want 1€/column when possible
SYS_TARGET_MIN = float(os.getenv("SYS_TARGET_MIN", "25.0"))
SYS_TARGET_MAX = float(os.getenv("SYS_TARGET_MAX", "50.0"))

# ------------------------- DRAWBET RULES -------------------------
DRAW_PICKS_MIN = int(os.getenv("DRAW_PICKS_MIN", "2"))
DRAW_PICKS_MAX = int(os.getenv("DRAW_PICKS_MAX", "5"))

DRAW_ODDS_MIN = float(os.getenv("DRAW_ODDS_MIN", "2.80"))
DRAW_ODDS_MAX = float(os.getenv("DRAW_ODDS_MAX", "3.70"))

# Strict draw thresholds
DRAW_P_MIN_STRICT  = float(os.getenv("DRAW_P_MIN_STRICT", "0.30"))
DRAW_EV_MIN_STRICT = float(os.getenv("DRAW_EV_MIN_STRICT", "0.07"))

# Fallback (to avoid 0 draws)
DRAW_P_MIN_FALLBACK  = float(os.getenv("DRAW_P_MIN_FALLBACK", "0.28"))
DRAW_EV_MIN_FALLBACK = float(os.getenv("DRAW_EV_MIN_FALLBACK", "0.02"))

# Draw shape helper (prefer but not mandatory in fallback)
DRAW_REQUIRE_SHAPE_STRICT = os.getenv("DRAW_REQUIRE_SHAPE_STRICT", "true").lower() == "true"

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

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def odds_match_ok(fx: Dict[str, Any], min_score: float) -> bool:
    om = fx.get("odds_match") or {}
    if not om.get("matched"):
        return False
    return (safe_float(om.get("score"), 0.0) or 0.0) >= min_score

def confidence_value(fx: Dict[str, Any]) -> Optional[float]:
    return safe_float((fx.get("flags") or {}).get("confidence"), None)

def confidence_ok(fx: Dict[str, Any], min_conf: float) -> bool:
    c = confidence_value(fx)
    if c is None:
        return True
    return c >= min_conf

def iso_week_id_from_window(window: Optional[dict]) -> str:
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

def load_history() -> dict:
    if not os.path.exists(TUESDAY_HISTORY_PATH):
        return {"week_count": 0, "weeks": {}, "core": {"bankroll_current": None}, "funbet": {"bankroll_current": None}, "drawbet": {"bankroll_current": None}}
    try:
        h = json.load(open(TUESDAY_HISTORY_PATH, "r", encoding="utf-8"))
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

def _ev_from_fx(fx: Dict[str, Any], market_code: str) -> Optional[float]:
    mc = (market_code or "").upper()
    if mc == "1": return safe_float(fx.get("ev_1"), None)
    if mc == "2": return safe_float(fx.get("ev_2"), None)
    if mc == "X": return safe_float(fx.get("ev_x"), None)
    if mc == "O25": return safe_float(fx.get("ev_over"), None)
    if mc == "U25": return safe_float(fx.get("ev_under"), None)
    return None

def _prob_from_fx(fx: Dict[str, Any], market_code: str) -> Optional[float]:
    mc = (market_code or "").upper()
    if mc == "1": return safe_float(fx.get("home_prob"), None)
    if mc == "2": return safe_float(fx.get("away_prob"), None)
    if mc == "X": return safe_float(fx.get("draw_prob"), None)
    if mc == "O25": return safe_float(fx.get("over_2_5_prob"), None)
    if mc == "U25": return safe_float(fx.get("under_2_5_prob"), None)
    return None

def _odds_from_fx(fx: Dict[str, Any], market_code: str) -> Optional[float]:
    mc = (market_code or "").upper()
    if mc == "1": return safe_float(fx.get("offered_1"), None)
    if mc == "2": return safe_float(fx.get("offered_2"), None)
    if mc == "X": return safe_float(fx.get("offered_x"), None)
    if mc == "O25": return safe_float(fx.get("offered_over_2_5"), None)
    if mc == "U25": return safe_float(fx.get("offered_under_2_5"), None)
    return None

def _market_name_from_code(code: str) -> str:
    return CODE_TO_MARKET.get((code or "").upper(), code)

def load_thursday_fixtures() -> Tuple[List[dict], dict]:
    data = json.load(open(THURSDAY_REPORT_PATH, "r", encoding="utf-8"))
    # supports both raw report and {status, report}
    if isinstance(data, dict) and "fixtures" in data:
        return data["fixtures"], data
    if isinstance(data, dict) and isinstance(data.get("report"), dict) and "fixtures" in data["report"]:
        rep = data["report"]
        return rep["fixtures"], rep
    raise KeyError("fixtures not found in Thursday report")

def build_pick_candidates(fixtures: List[dict]) -> List[dict]:
    out = []
    for fx in fixtures:
        for mcode in ["1", "2", "X", "O25", "U25"]:
            odds = _odds_from_fx(fx, mcode)
            if odds is None or odds <= 1.0:
                continue
            out.append({
                "pick_id": f'{fx.get("fixture_id")}:{mcode}',
                "fixture_id": fx.get("fixture_id"),
                "league": fx.get("league"),
                "date": fx.get("date"),
                "time": fx.get("time"),
                "match": f'{fx.get("home")} – {fx.get("away")}',
                "market_code": mcode,
                "market": _market_name_from_code(mcode),
                "odds": float(odds),
                "prob": _prob_from_fx(fx, mcode),
                "ev": _ev_from_fx(fx, mcode),
                "confidence": confidence_value(fx),
                "fx": fx,
            })
    return out

def _sort_key_datetime(p: dict):
    # robust sort by date+time if present
    d = p.get("date") or "9999-12-31"
    t = p.get("time") or "23:59"
    return f"{d} {t}"

# ------------------------- CORE SELECTION -------------------------
def _core_pick_score(p: dict) -> float:
    # score = EV + 0.25*conf + 0.05*prob; (stable, simple)
    evv = safe_float(p.get("ev"), -9999.0)
    conf = safe_float(p.get("confidence"), 0.0)
    pr = safe_float(p.get("prob"), 0.0)
    return float(evv) + 0.25 * float(conf) + 0.05 * float(pr)

def _strip_core_pick(p: dict, stake: float) -> dict:
    return {
        "pick_id": p["pick_id"],
        "fixture_id": p["fixture_id"],
        "date": p.get("date"),
        "time": p.get("time"),
        "league": p.get("league"),
        "match": p.get("match"),
        "market": p.get("market"),
        "market_code": p.get("market_code"),
        "odds": round(float(p.get("odds")), 2),
        "stake": round(float(stake), 2),
        "ev": p.get("ev"),
        "prob": p.get("prob"),
    }

def _strip_leg(p: dict) -> dict:
    return {
        "pick_id": p["pick_id"],
        "fixture_id": p["fixture_id"],
        "date": p.get("date"),
        "time": p.get("time"),
        "league": p.get("league"),
        "match": p.get("match"),
        "market": p.get("market"),
        "market_code": p.get("market_code"),
        "odds": round(float(p.get("odds")), 2),
    }

def core_select(picks: List[dict], bankroll_core: float) -> Tuple[List[dict], Optional[dict], List[dict], dict]:
    cap_amount = bankroll_core * CORE_EXPOSURE_CAP

    singles_pool: List[Tuple[dict, float]] = []
    low_pool: List[dict] = []

    for p in picks:
        fx = p["fx"]
        m = p["market"]
        if m not in CORE_ALLOWED_MARKETS:
            continue

        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_CORE):
            continue
        if not confidence_ok(fx, CORE_MIN_CONFIDENCE):
            continue

        odds = float(p["odds"])
        evv = safe_float(p.get("ev"), None)
        pr = safe_float(p.get("prob"), None)
        if evv is None or pr is None:
            continue

        # Anti-UFO for 1/2
        if m in ("Home", "Away") and odds > MAX_HOME_AWAY_ODDS:
            continue

        # Market gating
        if m == "Home":
            if evv < CORE_EV_MIN_HOME or pr < CORE_P_MIN_HOME:
                continue
        elif m == "Away":
            if evv < CORE_EV_MIN_AWAY or pr < CORE_P_MIN_AWAY:
                continue
        elif m == "Over 2.5":
            if evv < CORE_EV_MIN_OVER or pr < CORE_P_MIN_OVER:
                continue
        elif m == "Under 2.5":
            # keep under rare & strict: must be under_elite if exists
            flags = fx.get("flags") or {}
            if evv < CORE_EV_MIN_UNDER or pr < CORE_P_MIN_UNDER:
                continue
            if flags.get("under_elite") is False:
                # if the field exists and is false, skip
                continue

        if CORE_SINGLES_MIN_ODDS <= odds <= CORE_SINGLES_MAX_ODDS:
            st = core_single_stake(odds)
            if st > 0:
                singles_pool.append((p, st))
        elif CORE_LOW_ODDS_MIN <= odds <= CORE_LOW_ODDS_MAX:
            # cold-double bucket (do not lose them)
            low_pool.append(p)

    # rank singles by score then by lower odds preference when close
    singles_pool.sort(key=lambda ps: (_while_tie_lower_odds(ps[0]), reverse=True)
                      if False else ( _core_pick_score(ps[0]), -float(ps[0]["odds"]) ), reverse=True)

    # helper: prefer lower odds when score similar (we implement in pair sorting below)
    def better(a: dict, b: dict) -> bool:
        sa = _core_pick_score(a); sb = _core_pick_score(b)
        if abs(sa - sb) <= 0.03:  # close -> prefer lower odds (less "boom-bust")
            return float(a["odds"]) < float(b["odds"])
        return sa > sb

    singles: List[dict] = []
    used_matches = set()

    # pick singles unique match
    for p, st in singles_pool:
        if len(singles) >= CORE_MAX_SINGLES:
            break
        if p["match"] in used_matches:
            continue
        singles.append(_strip_core_pick(p, st))
        used_matches.add(p["match"])

    # ensure minimum singles if possible
    if len(singles) < CORE_MIN_SINGLES:
        for p, st in singles_pool:
            if len(singles) >= CORE_MIN_SINGLES:
                break
            if p["match"] in used_matches:
                continue
            singles.append(_strip_core_pick(p, st))
            used_matches.add(p["match"])

    # ---- Cold doubles ----
    doubles: List[dict] = []
    if CORE_MAX_DOUBLES > 0 and len(low_pool) >= 1:
        # partner candidates: low_pool + some single-range picks (but not same match)
        partner_pool = [p for p, _st in singles_pool] + low_pool
        # unique by match+market
        uniq = {}
        for p in partner_pool:
            uniq[(p["match"], p["market_code"])] = p
        partner_pool = list(uniq.values())

        # sort by score desc, then prefer lower odds on tie
        partner_pool.sort(key=lambda p: (_core_pick_score(p), -float(p["odds"])), reverse=True)
        low_pool.sort(key=lambda p: (_core_pick_score(p), -float(p["odds"])), reverse=True)

        used_double_matches = set()
        for leg1 in low_pool:
            if len(doubles) >= CORE_MAX_DOUBLES:
                break
            for leg2 in partner_pool:
                if leg2["match"] == leg1["match"]:
                    continue
                if leg1["match"] in used_double_matches or leg2["match"] in used_double_matches:
                    continue
                combo = float(leg1["odds"]) * float(leg2["odds"])
                if not (CORE_DOUBLE_TARGET_MIN <= combo <= CORE_DOUBLE_TARGET_MAX):
                    continue
                doubles.append({
                    "legs": [_strip_leg(leg1), _strip_leg(leg2)],
                    "combo_odds": round(combo, 2),
                    "stake": round(float(CORE_DOUBLE_STAKE), 2),
                })
                used_double_matches.add(leg1["match"])
                used_double_matches.add(leg2["match"])
                break

    # ---- Cap enforcement (NO scaling) ----
    def total_open() -> float:
        s = sum(float(x.get("stake", 0.0)) for x in singles)
        d = sum(float(x.get("stake", 0.0)) for x in doubles)
        return s + d

    # drop lowest-ranked singles first if over cap
    if total_open() > cap_amount and singles:
        # rebuild scores for singles by matching original pick_id
        score_by_id = {}
        for p, st in singles_pool:
            score_by_id[f'{p["fixture_id"]}:{p["market_code"]}'] = _core_pick_score(p)
        singles.sort(key=lambda x: (score_by_id.get(x["pick_id"], -9999.0), -float(x.get("odds", 99.0))), reverse=True)
        while total_open() > cap_amount and len(singles) > 0:
            singles.pop()  # drop worst

    # if still over cap, drop doubles last
    while total_open() > cap_amount and len(doubles) > 0:
        doubles.pop()

    # chronological ordering for presentation/copy
    singles.sort(key=_sort_key_datetime)
    for d in doubles:
        # sort legs chronologically
        d["legs"].sort(key=_sort_key_datetime)

    core_open = round(total_open(), 2)
    meta = {
        "bankroll_start": bankroll_core,
        "cap": round(cap_amount, 2),
        "open": core_open,
        "after_open": round(bankroll_core - core_open, 2),
        "counts": {"singles": len(singles), "doubles": len(doubles)},
    }
    core_double = doubles[0] if doubles else None
    return singles, core_double, doubles, meta

# ------------------------- SYSTEM HELPERS -------------------------
def _columns_for_r(n: int, r: int) -> int:
    return comb(n, r) if (n > 0 and 0 < r <= n) else 0

def _refund_ratio_worst_case(odds_list: List[float], r_min: int, columns: int) -> float:
    if columns <= 0 or r_min <= 0:
        return 0.0
    o = sorted([float(x) for x in odds_list])[:r_min]
    prod = 1.0
    for v in o:
        prod *= max(1.01, float(v))
    return prod / float(columns)

def _system_try(pool: List[dict], r_try: int, label: str) -> Optional[dict]:
    n = len(pool)
    cols = _columns_for_r(n, r_try)
    if cols <= 0:
        return None
    rr = _refund_ratio_worst_case([float(p["odds"]) for p in pool], r_try, cols)
    return {"label": label, "n": n, "columns": cols, "min_hits": r_try, "refund_ratio_min_hits": round(rr, 4)}

def choose_fun_system(pool: List[dict]) -> Tuple[Optional[dict], float, bool]:
    """
    Prefer easier first:
      n=7: try 4/7 then 5/7
      n=6: try 3/6 then 4/6
      n=5: try 3/5
    thresholds: primary then fallback
    returns: (choice or None, refund_used, breached)
    """
    n = len(pool)
    if n < 5:
        return None, SYS_REFUND_PRIMARY, True

    def trials_easy_first() -> List[dict]:
        if n >= 7:
            return [t for t in [_system_try(pool[:7], 4, "4/7"), _system_try(pool[:7], 5, "5/7")] if t]
        if n == 6:
            return [t for t in [_system_try(pool[:6], 3, "3/6"), _system_try(pool[:6], 4, "4/6")] if t]
        return [t for t in [_system_try(pool[:5], 3, "3/5")] if t]

    trials = trials_easy_first()

    for thr in [SYS_REFUND_PRIMARY, SYS_REFUND_FALLBACK]:
        for t in trials:
            if (t.get("refund_ratio_min_hits") or 0.0) >= thr:
                return t, thr, (thr != SYS_REFUND_PRIMARY)

    return None, SYS_REFUND_PRIMARY, True

def system_unit(columns: int, cap_system: float) -> Tuple[float, float]:
    """
    Unit policy:
      - default unit = 1.00
      - if stake < target_min, raise unit to hit target_min (rounded 0.01)
      - clamp to target_max and cap_system
    """
    if columns <= 0:
        return 0.0, 0.0

    # try unit=1.00 first
    unit = float(SYS_UNIT_BASE)
    stake = unit * columns

    # raise if below target min
    if stake < SYS_TARGET_MIN:
        unit = round(SYS_TARGET_MIN / columns, 2)
        stake = unit * columns

    # clamp to target max
    if stake > SYS_TARGET_MAX:
        unit = round(SYS_TARGET_MAX / columns, 2)
        stake = unit * columns

    # clamp to cap_system
    if stake > cap_system and cap_system > 0:
        unit = round(cap_system / columns, 2)
        stake = unit * columns

    return float(unit), round(float(stake), 2)

# ------------------------- FUN SELECTION (SYSTEM ONLY) -------------------------
def _fun_pick_score(p: dict) -> float:
    # Slight preference to lower odds when score close (handled in sorting with -odds)
    evv = safe_float(p.get("ev"), -9999.0)
    conf = safe_float(p.get("confidence"), 0.0)
    pr = safe_float(p.get("prob"), 0.0)
    return float(evv) + 0.20 * float(conf) + 0.05 * float(pr)

def fun_select(picks: List[dict], bankroll_fun: float, core_fixture_ids: set) -> Tuple[dict, dict]:
    cap_system = bankroll_fun * FUN_EXPOSURE_CAP

    candidates: List[dict] = []
    for p in picks:
        fx = p["fx"]
        m = p["market"]
        if m not in FUN_ALLOWED_MARKETS:
            continue
        if FUN_AVOID_CORE_OVERLAP and p["fixture_id"] in core_fixture_ids:
            continue

        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_FUN):
            continue
        if not confidence_ok(fx, FUN_MIN_CONFIDENCE):
            continue

        odds = float(p["odds"])
        if odds < FUN_ODDS_MIN or odds > FUN_ODDS_MAX:
            continue

        evv = safe_float(p.get("ev"), None)
        pr = safe_float(p.get("prob"), None)
        if evv is None or pr is None:
            continue

        # Anti-UFO for Home/Away in Fun too
        if m in ("Home", "Away") and odds > FUN_ODDS_GT3_THRESHOLD:
            # we allow at most FUN_ODDS_GT3_MAX_COUNT later, but only if it survives strict shape/conf:
            flags = fx.get("flags") or {}
            if m == "Home" and not flags.get("home_shape", False):
                continue
            if m == "Away" and not flags.get("away_shape", False):
                continue
            if (safe_float(p.get("confidence"), 0.0) or 0.0) < 0.75:
                continue
            # keep, but will be capped in count

        # Market gating
        if m == "Home":
            if evv < FUN_EV_MIN_HOME or pr < FUN_P_MIN_HOME:
                continue
        elif m == "Away":
            if evv < FUN_EV_MIN_AWAY or pr < FUN_P_MIN_AWAY:
                continue
        elif m == "Over 2.5":
            if evv < FUN_EV_MIN_OVER or pr < FUN_P_MIN_OVER:
                continue
        elif m == "Under 2.5":
            # keep under rare: require under_elite when exists
            flags = fx.get("flags") or {}
            if evv < FUN_EV_MIN_UNDER or pr < FUN_P_MIN_UNDER:
                continue
            if flags.get("under_elite") is False:
                continue

        candidates.append(p)

    # rank by score, then prefer lower odds when close (by using -odds second)
    candidates.sort(key=lambda p: (_fun_pick_score(p), -float(p["odds"])), reverse=True)

    # pick unique matches up to 7
    pool: List[dict] = []
    used_matches = set()
    gt3_count = 0

    for p in candidates:
        if p["match"] in used_matches:
            continue

        if p["market"] in ("Home", "Away") and float(p["odds"]) > FUN_ODDS_GT3_THRESHOLD:
            if gt3_count >= FUN_ODDS_GT3_MAX_COUNT:
                continue
            gt3_count += 1

        pool.append(p)
        used_matches.add(p["match"])
        if len(pool) >= FUN_PICKS_MAX:
            break

    # ensure minimum by relaxing only the odds-match gate? (we keep strict; if not enough, run with fewer)
    pool = pool[:FUN_PICKS_MAX]

    # choose system on available size (prefer 7 if we have it, else 6, else 5)
    if len(pool) >= 7:
        sys_pool = pool[:7]
    elif len(pool) == 6:
        sys_pool = pool[:6]
    else:
        sys_pool = pool[:5] if len(pool) >= 5 else pool

    sys_choice, refund_used, breached = choose_fun_system(sys_pool)
    cols = int(sys_choice["columns"]) if sys_choice else 0

    unit, stake = system_unit(cols, cap_system) if cols > 0 else (0.0, 0.0)

    # chronological ordering
    pool.sort(key=_sort_key_datetime)
    sys_pool.sort(key=_sort_key_datetime)

    payload = {
        "portfolio": "FunBet",
        "bankroll": DEFAULT_BANKROLL_FUN,
        "bankroll_start": bankroll_fun,
        "bankroll_source": "history" if bankroll_fun != DEFAULT_BANKROLL_FUN else "default",
        "exposure_cap_pct": FUN_EXPOSURE_CAP,

        "rules": {
            "odds_range": [FUN_ODDS_MIN, FUN_ODDS_MAX],
            "max_gt3_count": FUN_ODDS_GT3_MAX_COUNT,
            "gt3_threshold": FUN_ODDS_GT3_THRESHOLD,
            "refund_primary": SYS_REFUND_PRIMARY,
            "refund_fallback": SYS_REFUND_FALLBACK,
            "refund_used": refund_used,
            "refund_breached": bool(breached),
            "prefer_easy_system": True,
            "unit_base": SYS_UNIT_BASE,
            "target_min": SYS_TARGET_MIN,
            "target_max": SYS_TARGET_MAX,
        },

        "picks_total": [_strip_pick(p) for p in pool],
        "system_pool": [_strip_pick(p) for p in sys_pool],
        "system": {
            "label": sys_choice["label"] if sys_choice else None,
            "columns": cols if sys_choice else None,
            "min_hits": sys_choice["min_hits"] if sys_choice else None,
            "refund_ratio_min_hits": sys_choice["refund_ratio_min_hits"] if sys_choice else None,
            "unit": unit if cols > 0 else None,
            "stake": stake if cols > 0 else None,
        },

        "open": round(stake, 2),
        "after_open": round(bankroll_fun - stake, 2),
    }
    meta = {
        "counts": {"picks_total": len(pool), "system_pool": len(sys_pool)},
    }
    return payload, meta

def _strip_pick(p: dict) -> dict:
    return {
        "pick_id": p["pick_id"],
        "fixture_id": p["fixture_id"],
        "date": p.get("date"),
        "time": p.get("time"),
        "league": p.get("league"),
        "match": p.get("match"),
        "market": p.get("market"),
        "market_code": p.get("market_code"),
        "odds": round(float(p.get("odds")), 2),
        "ev": p.get("ev"),
        "prob": p.get("prob"),
        "confidence": p.get("confidence"),
    }

# ------------------------- DRAW SELECTION (SYSTEM ONLY, STRICT + FALLBACK) -------------------------
def _draw_pick_score(p: dict) -> float:
    evv = safe_float(p.get("ev"), -9999.0)
    conf = safe_float(p.get("confidence"), 0.0)
    pr = safe_float(p.get("prob"), 0.0)
    # prefer slightly lower odds when close -> handled by -odds in sorting
    return float(evv) + 0.20 * float(conf) + 0.05 * float(pr)

def _draw_system_label(n: int) -> Tuple[str, int, List[int]]:
    # returns (label, columns, sizes list)
    if n >= 5:
        # 2-3/5
        cols = comb(5, 2) + comb(5, 3)  # 10 + 10 = 20
        return "2-3/5", int(cols), [2, 3]
    if n == 4:
        # 2/4
        cols = comb(4, 2)  # 6
        return "2/4", int(cols), [2]
    if n == 3:
        # 2/3
        cols = comb(3, 2)  # 3
        return "2/3", int(cols), [2]
    if n == 2:
        # 2/2
        return "2/2", 1, [2]
    return "—", 0, []

def draw_select(picks: List[dict], bankroll_draw: float) -> Tuple[dict, dict]:
    cap_system = bankroll_draw * DRAW_EXPOSURE_CAP

    draw_candidates = []
    for p in picks:
        fx = p["fx"]
        if p["market"] != "Draw":
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_DRAW):
            continue
        if not confidence_ok(fx, DRAW_MIN_CONFIDENCE):
            continue

        odds = float(p["odds"])
        if odds < DRAW_ODDS_MIN or odds > DRAW_ODDS_MAX:
            continue

        evv = safe_float(p.get("ev"), None)
        pr = safe_float(p.get("prob"), None)
        if evv is None or pr is None:
            continue

        flags = fx.get("flags") or {}

        # strict phase
        ok_strict = (pr >= DRAW_P_MIN_STRICT) and (evv >= DRAW_EV_MIN_STRICT)
        if DRAW_REQUIRE_SHAPE_STRICT:
            ok_strict = ok_strict and bool(flags.get("draw_shape", False))

        # fallback phase (no shape requirement)
        ok_fallback = (pr >= DRAW_P_MIN_FALLBACK) and (evv >= DRAW_EV_MIN_FALLBACK)

        if ok_strict or ok_fallback:
            draw_candidates.append((p, ok_strict))

    # sort: strict first, then by score, then prefer lower odds
    draw_candidates.sort(key=lambda x: (1 if x[1] else 0, _draw_pick_score(x[0]), -float(x[0]["odds"])), reverse=True)

    pool = []
    used_matches = set()
    for p, _is_strict in draw_candidates:
        if p["match"] in used_matches:
            continue
        pool.append(p)
        used_matches.add(p["match"])
        if len(pool) >= DRAW_PICKS_MAX:
            break

    # if still 0, keep 0 (no inventing)
    # choose pool size (2..5)
    pool = pool[:DRAW_PICKS_MAX]
    pool.sort(key=_sort_key_datetime)

    label, cols, _sizes = _draw_system_label(len(pool))

    # stake targeting: try unit=1 first, then lift to target_min if needed (cap applies)
    unit, stake = (0.0, 0.0)
    if cols > 0:
        unit = 1.00
        stake = unit * cols
        if stake < SYS_TARGET_MIN:
            unit = round(SYS_TARGET_MIN / cols, 2)
            stake = unit * cols
        if stake > SYS_TARGET_MAX:
            unit = round(SYS_TARGET_MAX / cols, 2)
            stake = unit * cols
        if stake > cap_system and cap_system > 0:
            unit = round(cap_system / cols, 2)
            stake = unit * cols
        stake = round(stake, 2)

    payload = {
        "portfolio": "DrawBet",
        "bankroll": DEFAULT_BANKROLL_DRAW,
        "bankroll_start": bankroll_draw,
        "bankroll_source": "history" if bankroll_draw != DEFAULT_BANKROLL_DRAW else "default",
        "exposure_cap_pct": DRAW_EXPOSURE_CAP,

        "rules": {
            "odds_range": [DRAW_ODDS_MIN, DRAW_ODDS_MAX],
            "strict": {"p_min": DRAW_P_MIN_STRICT, "ev_min": DRAW_EV_MIN_STRICT, "require_draw_shape": DRAW_REQUIRE_SHAPE_STRICT},
            "fallback": {"p_min": DRAW_P_MIN_FALLBACK, "ev_min": DRAW_EV_MIN_FALLBACK},
            "system_sizes": "2/2, 2/3, 2/4, 2-3/5",
        },

        "picks_total": [_strip_pick(p) for p in pool],
        "system_pool": [_strip_pick(p) for p in pool],
        "system": {
            "label": label if cols > 0 else None,
            "columns": cols if cols > 0 else None,
            "unit": unit if cols > 0 else None,
            "stake": stake if cols > 0 else None,
        },

        "open": round(stake, 2),
        "after_open": round(bankroll_draw - stake, 2),
    }
    meta = {"counts": {"picks_total": len(pool), "system_pool": len(pool)}}
    return payload, meta

# ------------------------- COPY PLAY (JSON additive) -------------------------
def build_copy_play(core_singles: List[dict], core_doubles: List[dict], fun_payload: dict, draw_payload: dict,
                    core_meta: dict, fun_meta: dict, draw_meta: dict) -> dict:
    def line_pick(p: dict, stake: Optional[float] = None) -> str:
        d = p.get("date") or "—"
        t = p.get("time") or "—"
        lg = p.get("league") or "—"
        m = p.get("match") or "—"
        mk = p.get("market") or "—"
        od = p.get("odds")
        od_s = f"{od:.2f}" if isinstance(od, (float, int)) else "—"
        st = stake if stake is not None else p.get("stake")
        st_s = f"{float(st):.2f}".rstrip("0").rstrip(".") if st is not None else "—"
        return f"{d} {t} | {lg} | {m} | {mk} | {od_s} | {st_s}"

    core_lines = []
    for p in core_singles:
        core_lines.append(line_pick(p, p.get("stake")))
    for d in core_doubles:
        # show just combo + stake + legs
        legs = d.get("legs") or []
        if len(legs) >= 2:
            l1 = legs[0]; l2 = legs[1]
            combo = d.get("combo_odds")
            combo_s = f"{float(combo):.2f}" if combo is not None else "—"
            st = d.get("stake")
            st_s = f"{float(st):.2f}".rstrip("0").rstrip(".") if st is not None else "—"
            core_lines.append(f"DOUBLE | {combo_s} | {st_s} | {l1.get('match')} ({l1.get('market')} @{l1.get('odds')}) + {l2.get('match')} ({l2.get('market')} @{l2.get('odds')})")

    fun_lines = []
    sys = fun_payload.get("system") or {}
    if sys.get("label") and sys.get("columns") and sys.get("unit") and sys.get("stake") is not None:
        fun_lines.append(f"SYSTEM: {sys.get('label')} | Columns: {sys.get('columns')} | Unit: {sys.get('unit')} | Stake: {sys.get('stake')}")
    for p in (fun_payload.get("system_pool") or []):
        fun_lines.append(line_pick(p))

    draw_lines = []
    dsys = draw_payload.get("system") or {}
    if dsys.get("label") and dsys.get("columns") and dsys.get("unit") and dsys.get("stake") is not None:
        draw_lines.append(f"SYSTEM: {dsys.get('label')} | Columns: {dsys.get('columns')} | Unit: {dsys.get('unit')} | Stake: {dsys.get('stake')}")
    for p in (draw_payload.get("system_pool") or []):
        draw_lines.append(line_pick(p))

    bankroll_lines = {
        "core": f"CORE → Start: {core_meta.get('bankroll_start')} | Open: {core_meta.get('open')} | After: {core_meta.get('after_open')}",
        "fun":  f"FUN  → Start: {fun_payload.get('bankroll_start')} | Open: {fun_payload.get('open')} | After: {fun_payload.get('after_open')}",
        "draw": f"DRAW → Start: {draw_payload.get('bankroll_start')} | Open: {draw_payload.get('open')} | After: {draw_payload.get('after_open')}",
    }

    return {
        "core": core_lines,
        "fun": fun_lines,
        "draw": draw_lines,
        "bankrolls": bankroll_lines,
    }

# ------------------------- MAIN -------------------------
def main():
    fixtures, th_meta = load_thursday_fixtures()
    history = load_history()

    window = th_meta.get("window", {}) if isinstance(th_meta, dict) else {}
    wf = get_week_fields(window, history)

    core_start = safe_float((history.get("core") or {}).get("bankroll_current"), None)
    fun_start  = safe_float((history.get("funbet") or {}).get("bankroll_current"), None)
    draw_start = safe_float((history.get("drawbet") or {}).get("bankroll_current"), None)

    core_bankroll_start = core_start if core_start is not None else DEFAULT_BANKROLL_CORE
    fun_bankroll_start  = fun_start if fun_start is not None else DEFAULT_BANKROLL_FUN
    draw_bankroll_start = draw_start if draw_start is not None else DEFAULT_BANKROLL_DRAW

    picks = build_pick_candidates(fixtures)

    # CORE
    core_singles, core_double, core_doubles, core_meta = core_select(picks, core_bankroll_start)
    core_fixture_ids = {x.get("fixture_id") for x in core_singles}

    # FUN (system only)
    fun_payload, fun_meta = fun_select(picks, fun_bankroll_start, core_fixture_ids)

    # DRAW (system only)
    draw_payload, draw_meta = draw_select(picks, draw_bankroll_start)

    # Copy play (additive)
    copy_play = build_copy_play(core_singles, core_doubles, fun_payload, draw_payload, core_meta, fun_meta, draw_meta)

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
                "cold_double_odds_range": [CORE_LOW_ODDS_MIN, CORE_LOW_ODDS_MAX],
                "double_target_combo": [CORE_DOUBLE_TARGET_MIN, CORE_DOUBLE_TARGET_MAX],
                "max_home_away_odds": MAX_HOME_AWAY_ODDS,
                "no_scaling": True,
            },

            "singles": core_singles,
            "double": core_double,
            "doubles": core_doubles,
            "open": core_meta["open"],
            "after_open": core_meta["after_open"],
            "picks_count": core_meta["counts"]["singles"],
            "doubles_count": core_meta["counts"]["doubles"],
        },

        "funbet": fun_payload,
        "drawbet": draw_payload,

        # additive convenience for presenters
        "copy_play": copy_play,
    }

    os.makedirs("logs", exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
