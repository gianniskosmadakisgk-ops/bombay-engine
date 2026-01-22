# ============================================================
#  src/analysis/friday_shortlist_v3.py
#  FRIDAY SHORTLIST v3.30 — LOCKED RULES (CoreBet / FunBet / DrawBet)
#
#  Reads:
#    - logs/thursday_report_v3.json
#    - (optional) logs/tuesday_history_v3.json (bankroll carry + week numbering)
#
#  Writes:
#    - logs/friday_shortlist_v3.json
#
#  COREBET:
#   - Singles ONLY 1.60–2.10 (no crazy underdogs in core)
#   - Low odds 1.30–1.60 go ONLY to doubles (no singles)
#   - Stake ladder fixed: 40/30/20 (no scaling; prune instead)
#   - Max singles: 8
#
#  FUNBET:
#   - System-only, 5–7 picks
#   - Prefer lower min-hits system: 4/7 over 5/7; 3/6 over 4/6
#   - Refund threshold: 0.65
#   - Stake target 25–50 (unit policy; for 35 columns unit=1 => stake 35)
#   - Anti-underdog: hard cap 3.30; max 2 picks >3.00; >3.00 requires prob>=0.35 & conf>=0.70
#
#  DRAWBET:
#   - Find 2–5 draws (odds 2.80–3.70); system-only (except if only 1 -> single)
#   - System: 2/2, 2/3, 2/4, 2-3/5
#   - Stake target 25–50
# ============================================================

import os
import json
from datetime import datetime, date
from math import comb
from typing import Any, Dict, List, Optional, Tuple

THURSDAY_REPORT_PATH = os.getenv("THURSDAY_REPORT_PATH", "logs/thursday_report_v3.json")
FRIDAY_REPORT_PATH   = os.getenv("FRIDAY_REPORT_PATH",   "logs/friday_shortlist_v3.json")
TUESDAY_HISTORY_PATH = os.getenv("TUESDAY_HISTORY_PATH", "logs/tuesday_history_v3.json")

# ------------------------- DEFAULT BANKROLLS -------------------------
DEFAULT_BANKROLL_CORE = float(os.getenv("BANKROLL_CORE", "800"))
DEFAULT_BANKROLL_FUN  = float(os.getenv("BANKROLL_FUN",  "400"))
DEFAULT_BANKROLL_DRAW = float(os.getenv("BANKROLL_DRAW", "300"))

# Exposure caps (used as hard caps; we PRUNE, not scale)
CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.15"))  # 120 on 800
FUN_EXPOSURE_CAP  = float(os.getenv("FUN_EXPOSURE_CAP",  "0.20"))  # 80 on 400
DRAW_EXPOSURE_CAP = float(os.getenv("DRAW_EXPOSURE_CAP", "0.20"))  # 60 on 300

# Gates (from Thursday odds_match / confidence)
ODDS_MATCH_MIN_SCORE_CORE = float(os.getenv("ODDS_MATCH_MIN_SCORE_CORE", "0.75"))
ODDS_MATCH_MIN_SCORE_FUN  = float(os.getenv("ODDS_MATCH_MIN_SCORE_FUN",  "0.65"))
ODDS_MATCH_MIN_SCORE_DRAW = float(os.getenv("ODDS_MATCH_MIN_SCORE_DRAW", "0.70"))

CORE_MIN_CONFIDENCE = float(os.getenv("CORE_MIN_CONFIDENCE", "0.55"))
FUN_MIN_CONFIDENCE  = float(os.getenv("FUN_MIN_CONFIDENCE",  "0.45"))
DRAW_MIN_CONFIDENCE = float(os.getenv("DRAW_MIN_CONFIDENCE", "0.55"))

FUN_AVOID_CORE_OVERLAP = os.getenv("FUN_AVOID_CORE_OVERLAP", "false").lower() == "true"

# ------------------------- MARKETS -------------------------
CODE_TO_MARKET = {"1": "Home", "2": "Away", "X": "Draw", "O25": "Over 2.5", "U25": "Under 2.5"}
MARKET_TO_CODE = {v: k for k, v in CODE_TO_MARKET.items()}

# ------------------------- CORE RULES -------------------------
CORE_SINGLES_MIN_ODDS = float(os.getenv("CORE_SINGLES_MIN_ODDS", "1.60"))
CORE_SINGLES_MAX_ODDS = float(os.getenv("CORE_SINGLES_MAX_ODDS", "2.10"))

CORE_LOW_ODDS_MIN = float(os.getenv("CORE_LOW_ODDS_MIN", "1.30"))
CORE_LOW_ODDS_MAX = float(os.getenv("CORE_LOW_ODDS_MAX", "1.60"))

CORE_MAX_SINGLES = int(os.getenv("CORE_MAX_SINGLES", "8"))
CORE_MAX_DOUBLES = int(os.getenv("CORE_MAX_DOUBLES", "2"))

CORE_DOUBLE_TARGET_MIN = float(os.getenv("CORE_DOUBLE_TARGET_MIN", "2.00"))
CORE_DOUBLE_TARGET_MAX = float(os.getenv("CORE_DOUBLE_TARGET_MAX", "3.50"))

# fixed stakes (no scaling)
def core_single_stake(odds: float) -> float:
    if 1.60 <= odds <= 1.75:
        return 40.0
    if 1.75 < odds <= 1.90:
        return 30.0
    if 1.90 < odds <= 2.10:
        return 20.0
    return 0.0

def core_double_stake(combo_odds: float) -> float:
    # keep simple; typical will land >=2 so stake 15
    if combo_odds <= 2.70:
        return 15.0
    if combo_odds <= 3.50:
        return 15.0
    return 10.0

# ------------------------- FUN RULES -------------------------
FUN_PICKS_MIN = int(os.getenv("FUN_PICKS_MIN", "5"))
FUN_PICKS_MAX = int(os.getenv("FUN_PICKS_MAX", "7"))

# refund thresholds
SYS_REFUND_THRESHOLD = float(os.getenv("SYS_REFUND_THRESHOLD", "0.65"))

# system spend target (cap applied too)
SYS_TARGET_MIN = float(os.getenv("SYS_TARGET_MIN", "25.0"))
SYS_TARGET_MAX = float(os.getenv("SYS_TARGET_MAX", "50.0"))
SYS_UNIT_BASE  = float(os.getenv("SYS_UNIT_BASE", "1.0"))

# anti-underdog
FUN_SOFT_CAP_ODDS = float(os.getenv("FUN_SOFT_CAP_ODDS", "3.00"))
FUN_HARD_CAP_ODDS = float(os.getenv("FUN_HARD_CAP_ODDS", "3.30"))
FUN_MAX_HIGH_ODDS_IN_7 = int(os.getenv("FUN_MAX_HIGH_ODDS_IN_7", "2"))
FUN_HIGH_ODDS_PROB_MIN = float(os.getenv("FUN_HIGH_ODDS_PROB_MIN", "0.35"))
FUN_HIGH_ODDS_CONF_MIN = float(os.getenv("FUN_HIGH_ODDS_CONF_MIN", "0.70"))

# ------------------------- DRAW RULES -------------------------
DRAW_ODDS_MIN = float(os.getenv("DRAW_ODDS_MIN", "2.80"))
DRAW_ODDS_MAX = float(os.getenv("DRAW_ODDS_MAX", "3.70"))

DRAW_PICKS_MAX = int(os.getenv("DRAW_PICKS_MAX", "5"))
DRAW_PICKS_MIN = int(os.getenv("DRAW_PICKS_MIN", "2"))  # try for 2–5

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

def load_json(path: str) -> dict:
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

def load_history() -> dict:
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

def _odds_from_fx(fx, code: str):
    c = (code or "").upper()
    if c == "1":   return safe_float(fx.get("offered_1"), None)
    if c == "2":   return safe_float(fx.get("offered_2"), None)
    if c == "X":   return safe_float(fx.get("offered_x"), None)
    if c == "O25": return safe_float(fx.get("offered_over_2_5"), None)
    if c == "U25": return safe_float(fx.get("offered_under_2_5"), None)
    return None

def _prob_from_fx(fx, code: str):
    c = (code or "").upper()
    if c == "1":   return safe_float(fx.get("home_prob"), None)
    if c == "2":   return safe_float(fx.get("away_prob"), None)
    if c == "X":   return safe_float(fx.get("draw_prob"), None)
    if c == "O25": return safe_float(fx.get("over_2_5_prob"), None)
    if c == "U25": return safe_float(fx.get("under_2_5_prob"), None)
    return None

def _ev_from_fx(fx, code: str):
    c = (code or "").upper()
    if c == "1":   return safe_float(fx.get("ev_1"), None)
    if c == "2":   return safe_float(fx.get("ev_2"), None)
    if c == "X":   return safe_float(fx.get("ev_x"), None)
    if c == "O25": return safe_float(fx.get("ev_over"), None)
    if c == "U25": return safe_float(fx.get("ev_under"), None)
    return None

def _suitability_from_fx(fx, code: str):
    c = (code or "").upper()
    if c == "1":   return safe_float(fx.get("suitability_home"), None)
    if c == "2":   return safe_float(fx.get("suitability_away"), None)
    if c == "X":   return safe_float(fx.get("suitability_draw"), None)
    if c == "O25": return safe_float(fx.get("suitability_over"), None)
    if c == "U25": return safe_float(fx.get("suitability_under"), None)
    return None

def load_thursday() -> Tuple[List[dict], dict]:
    data = load_json(THURSDAY_REPORT_PATH)
    # support both shapes: {fixtures:...} or {status,timestamp,report:{fixtures:...}}
    if isinstance(data, dict) and "fixtures" in data:
        return data["fixtures"], data
    if isinstance(data, dict) and isinstance(data.get("report"), dict) and "fixtures" in data["report"]:
        return data["report"]["fixtures"], data["report"]
    raise KeyError("fixtures not found in Thursday report")

def build_candidates(fixtures: List[dict]) -> List[dict]:
    out = []
    for fx in fixtures:
        # only keep markets with offered odds
        for code in ["1", "2", "X", "O25", "U25"]:
            odds = _odds_from_fx(fx, code)
            if odds is None or odds <= 1.0:
                continue
            out.append({
                "pick_id": f'{fx.get("fixture_id")}:{code}',
                "fixture_id": fx.get("fixture_id"),
                "date": fx.get("date"),
                "time": fx.get("time"),
                "league": fx.get("league"),
                "match": f'{fx.get("home")} – {fx.get("away")}',
                "market_code": code,
                "market": CODE_TO_MARKET[code],
                "odds": odds,
                "prob": _prob_from_fx(fx, code),
                "ev": _ev_from_fx(fx, code),
                "confidence": confidence_value(fx),
                "suitability": _suitability_from_fx(fx, code),
                "fx": fx,
            })
    return out

def sort_by_datetime(items: List[dict]) -> List[dict]:
    def k(x):
        d = x.get("date") or "9999-12-31"
        t = x.get("time") or "23:59"
        return f"{d} {t}"
    return sorted(items, key=k)

# ------------------------- COREBET -------------------------
def corebet_select(picks: List[dict], bankroll_core: float) -> Tuple[List[dict], Optional[dict], List[dict], dict]:
    cap_amount = bankroll_core * CORE_EXPOSURE_CAP

    singles_pool = []
    low_pool = []

    for p in picks:
        fx = p["fx"]
        if p["market"] not in ("Home", "Away", "Over 2.5", "Under 2.5"):
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_CORE):
            continue
        if not confidence_ok(fx, CORE_MIN_CONFIDENCE):
            continue

        odds = safe_float(p.get("odds"), None)
        pr = safe_float(p.get("prob"), None)
        evv = safe_float(p.get("ev"), None)
        if odds is None or pr is None or evv is None:
            continue

        # Core singles strict range
        if CORE_SINGLES_MIN_ODDS <= odds <= CORE_SINGLES_MAX_ODDS:
            st = core_single_stake(odds)
            if st <= 0:
                continue
            singles_pool.append({**p, "stake": st})
        # low odds only for doubles
        elif CORE_LOW_ODDS_MIN <= odds <= CORE_LOW_ODDS_MAX:
            low_pool.append(p)

    # rank: suitability -> confidence -> ev -> prob, and for close scores prefer LOWER odds
    def score_key(x):
        suit = safe_float(x.get("suitability"), -1.0)
        conf = safe_float(x.get("confidence"), 0.0)
        evv  = safe_float(x.get("ev"), -9999.0)
        pr   = safe_float(x.get("prob"), 0.0)
        odds = safe_float(x.get("odds"), 99.0)
        return (suit, conf, evv, pr, -1.0 * odds)  # lower odds slightly preferred at end

    singles_pool.sort(key=score_key, reverse=True)
    low_pool.sort(key=score_key, reverse=True)

    singles = []
    used_matches = set()
    for p in singles_pool:
        if len(singles) >= CORE_MAX_SINGLES:
            break
        if p["match"] in used_matches:
            continue
        used_matches.add(p["match"])
        singles.append({
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
            "stake": round(float(p["stake"]), 2),
            "tag": "core_single",
        })

    # build doubles from low_pool using partners from (low_pool + singles_pool), target odds window
    doubles = []
    if low_pool and CORE_MAX_DOUBLES > 0:
        partners = [p for p in (low_pool + singles_pool)]
        partners.sort(key=score_key, reverse=True)

        used_d = set()
        for leg1 in low_pool:
            if len(doubles) >= CORE_MAX_DOUBLES:
                break
            for leg2 in partners:
                if leg2["match"] == leg1["match"]:
                    continue
                if leg1["match"] in used_d or leg2["match"] in used_d:
                    continue
                combo = safe_float(leg1["odds"], 1.0) * safe_float(leg2["odds"], 1.0)
                if not (CORE_DOUBLE_TARGET_MIN <= combo <= CORE_DOUBLE_TARGET_MAX):
                    continue
                stake = core_double_stake(combo)
                doubles.append({
                    "legs": [
                        {"pick_id": leg1["pick_id"], "fixture_id": leg1["fixture_id"], "match": leg1["match"], "market": leg1["market"], "market_code": leg1["market_code"], "odds": round(float(leg1["odds"]), 3)},
                        {"pick_id": leg2["pick_id"], "fixture_id": leg2["fixture_id"], "match": leg2["match"], "market": leg2["market"], "market_code": leg2["market_code"], "odds": round(float(leg2["odds"]), 3)},
                    ],
                    "combo_odds": round(combo, 2),
                    "stake": round(stake, 2),
                    "tag": "core_double",
                })
                used_d.add(leg1["match"]); used_d.add(leg2["match"])
                break

    # PRUNE to cap (no scaling)
    def open_amount(sgl, dbl):
        return round(sum(x["stake"] for x in sgl) + sum(d["stake"] for d in dbl), 2)

    # if over cap, drop lowest-ranked singles first (keep doubles if possible)
    if open_amount(singles, doubles) > cap_amount:
        # rank singles again by same score_key, but we already have; rebuild map
        # We'll drop from end (lowest)
        while singles and open_amount(singles, doubles) > cap_amount:
            singles.pop()  # lowest priority
    if open_amount(singles, doubles) > cap_amount:
        while doubles and open_amount(singles, doubles) > cap_amount:
            doubles.pop()  # then drop doubles if still over

    singles = sort_by_datetime(singles)
    # doubles keep order added

    meta_open = open_amount(singles, doubles)
    meta = {
        "bankroll": bankroll_core,
        "exposure_cap_pct": CORE_EXPOSURE_CAP,
        "open": meta_open,
        "after_open": round(bankroll_core - meta_open, 2),
        "picks_count": len(singles),
        "doubles_count": len(doubles),
        "scale_applied": None,
    }

    core_double = doubles[0] if doubles else None
    return singles, core_double, doubles, meta

# ------------------------- FUNBET SYSTEM -------------------------
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

def _system_try(pool: List[dict], r_try: int, label: str):
    n = len(pool)
    cols = _columns_for_r(n, r_try)
    if cols <= 0:
        return None
    rr = _refund_ratio_worst_case([p["odds"] for p in pool], r_try, cols)
    return {"label": label, "n": n, "columns": cols, "min_hits": r_try, "refund_ratio_min_hits": round(rr, 4)}

def choose_fun_system(pool: List[dict]) -> Tuple[Optional[dict], bool]:
    """
    Prefer lower min-hits always:
      n=7: try 4/7 first, else 5/7
      n=6: try 3/6 first, else 4/6
      n=5: 3/5
    Threshold = SYS_REFUND_THRESHOLD (0.65)
    """
    n = len(pool)
    if n < 5:
        return None, False
    if n == 5:
        cand = _system_try(pool, 3, "3/5")
        return (cand if cand and cand["refund_ratio_min_hits"] >= SYS_REFUND_THRESHOLD else None), bool(cand)
    if n == 6:
        cand1 = _system_try(pool, 3, "3/6")
        if cand1 and cand1["refund_ratio_min_hits"] >= SYS_REFUND_THRESHOLD:
            return cand1, True
        cand2 = _system_try(pool, 4, "4/6")
        if cand2 and cand2["refund_ratio_min_hits"] >= SYS_REFUND_THRESHOLD:
            return cand2, True
        return None, True
    # n >= 7 -> use first 7
    cand1 = _system_try(pool, 4, "4/7")
    if cand1 and cand1["refund_ratio_min_hits"] >= SYS_REFUND_THRESHOLD:
        return cand1, True
    cand2 = _system_try(pool, 5, "5/7")
    if cand2 and cand2["refund_ratio_min_hits"] >= SYS_REFUND_THRESHOLD:
        return cand2, True
    return None, True

def _unit_for_target(columns: int, cap_total: float) -> Tuple[float, float]:
    """
    unit base = 1 per column when possible.
    keep total stake in [25..50] and <= cap_total.
    """
    if columns <= 0:
        return 0.0, 0.0

    base_total = columns * SYS_UNIT_BASE

    # if base already inside range and <= cap, keep unit=1
    if SYS_TARGET_MIN <= base_total <= SYS_TARGET_MAX and base_total <= cap_total:
        unit = 1.0
        return unit, round(unit * columns, 2)

    # else choose target = clamp(base_total, 25..50), then cap_total
    target = base_total
    if target < SYS_TARGET_MIN:
        target = SYS_TARGET_MIN
    if target > SYS_TARGET_MAX:
        target = SYS_TARGET_MAX
    if target > cap_total:
        target = cap_total

    unit = round(target / columns, 2)
    stake = round(unit * columns, 2)
    return unit, stake

def funbet_select(picks: List[dict], bankroll_fun: float, core_fixture_ids: set) -> dict:
    cap_total = bankroll_fun * FUN_EXPOSURE_CAP

    candidates = []
    for p in picks:
        fx = p["fx"]
        if p["market"] not in ("Home", "Away", "Over 2.5", "Under 2.5"):
            continue
        if FUN_AVOID_CORE_OVERLAP and p["fixture_id"] in core_fixture_ids:
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_FUN):
            continue
        if not confidence_ok(fx, FUN_MIN_CONFIDENCE):
            continue

        odds = safe_float(p.get("odds"), None)
        pr   = safe_float(p.get("prob"), None)
        evv  = safe_float(p.get("ev"), None)
        conf = safe_float(p.get("confidence"), 0.0)
        suit = safe_float(p.get("suitability"), None)

        if odds is None or pr is None or evv is None:
            continue

        # hard cap
        if odds > FUN_HARD_CAP_ODDS:
            continue

        # if >3.00: require strong prob & confidence
        if odds > FUN_SOFT_CAP_ODDS:
            if pr < FUN_HIGH_ODDS_PROB_MIN or conf < FUN_HIGH_ODDS_CONF_MIN:
                continue

        candidates.append(p)

    # rank: suitability -> confidence -> ev -> prob; if close, prefer LOWER odds
    def k(x):
        suit = safe_float(x.get("suitability"), 0.0)
        conf = safe_float(x.get("confidence"), 0.0)
        evv  = safe_float(x.get("ev"), -9999.0)
        pr   = safe_float(x.get("prob"), 0.0)
        odds = safe_float(x.get("odds"), 99.0)
        return (suit, conf, evv, pr, -1.0 * odds)

    candidates.sort(key=k, reverse=True)

    # select up to 7 unique matches, with max 2 picks >3.00
    picks_out = []
    used_matches = set()
    high_odds_count = 0

    for p in candidates:
        if len(picks_out) >= FUN_PICKS_MAX:
            break
        if p["match"] in used_matches:
            continue
        odds = float(p["odds"])
        if odds > FUN_SOFT_CAP_ODDS:
            if high_odds_count >= FUN_MAX_HIGH_ODDS_IN_7:
                continue
            high_odds_count += 1
        picks_out.append(p)
        used_matches.add(p["match"])

    # if still below min, keep what we have (system may be None)
    # choose pool size preference: 7, else 6, else 5
    if len(picks_out) >= 7:
        pool = picks_out[:7]
    elif len(picks_out) >= 6:
        pool = picks_out[:6]
    elif len(picks_out) >= 5:
        pool = picks_out[:5]
    else:
        pool = picks_out[:]  # <5

    sys_choice, _has_trials = choose_fun_system(pool)

    columns = int(sys_choice["columns"]) if sys_choice else 0
    unit, stake = _unit_for_target(columns, cap_total) if columns > 0 else (0.0, 0.0)

    payload = {
        "bankroll": bankroll_fun,
        "exposure_cap_pct": FUN_EXPOSURE_CAP,
        "rules": {
            "picks_range": [FUN_PICKS_MIN, FUN_PICKS_MAX],
            "refund_threshold": SYS_REFUND_THRESHOLD,
            "unit_base": SYS_UNIT_BASE,
            "target_total_range": [SYS_TARGET_MIN, SYS_TARGET_MAX],
            "ranking": "suitability -> confidence -> ev -> prob (prefer lower odds when close)",
            "anti_underdog": {
                "soft_cap_odds": FUN_SOFT_CAP_ODDS,
                "hard_cap_odds": FUN_HARD_CAP_ODDS,
                "max_high_odds_in_7": FUN_MAX_HIGH_ODDS_IN_7,
                "high_odds_requires": {"prob_min": FUN_HIGH_ODDS_PROB_MIN, "confidence_min": FUN_HIGH_ODDS_CONF_MIN},
            }
        },
        "picks_total": sort_by_datetime([_strip_pick(p) for p in picks_out]),
        "system_pool": sort_by_datetime([_strip_pick(p) for p in pool]),
        "system": {
            "label": (sys_choice["label"] if sys_choice else None),
            "columns": columns,
            "min_hits": (int(sys_choice["min_hits"]) if sys_choice else None),
            "refund_ratio_min_hits": (float(sys_choice["refund_ratio_min_hits"]) if sys_choice else None),
            "unit": unit,
            "stake": stake,
            "refund_used": SYS_REFUND_THRESHOLD,
            "refund_rule_breached": (sys_choice is None),
            "has_system": bool(sys_choice and columns > 0),
        },
        "open": round(stake, 2),
        "after_open": round(bankroll_fun - stake, 2),
        "counts": {"picks_total": len(picks_out), "system_pool": len(pool)},
    }
    return payload

def _strip_pick(p: dict) -> dict:
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

# ------------------------- DRAWBET -------------------------
def _draw_system_for_n(n: int) -> Tuple[Optional[str], Optional[List[int]], int]:
    """
    returns (label, r_list, columns)
    """
    if n == 5:
        cols = comb(5, 2) + comb(5, 3)  # 20
        return "2-3/5", [2, 3], cols
    if n == 4:
        cols = comb(4, 2)  # 6
        return "2/4", [2], cols
    if n == 3:
        cols = comb(3, 2)  # 3
        return "2/3", [2], cols
    if n == 2:
        cols = 1
        return "2/2", [2], cols
    if n == 1:
        return "1/1", [1], 1
    return None, None, 0

def drawbet_select(picks: List[dict], bankroll_draw: float) -> dict:
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

        odds = safe_float(p.get("odds"), None)
        pr   = safe_float(p.get("prob"), None)
        evv  = safe_float(p.get("ev"), None)
        if odds is None or pr is None or evv is None:
            continue

        if not (DRAW_ODDS_MIN <= odds <= DRAW_ODDS_MAX):
            continue

        candidates.append(p)

    # rank: suitability_draw -> confidence -> ev -> prob (prefer lower odds when close)
    def k(x):
        suit = safe_float(x.get("suitability"), 0.0)
        conf = safe_float(x.get("confidence"), 0.0)
        evv  = safe_float(x.get("ev"), -9999.0)
        pr   = safe_float(x.get("prob"), 0.0)
        odds = safe_float(x.get("odds"), 99.0)
        return (suit, conf, evv, pr, -1.0 * odds)

    candidates.sort(key=k, reverse=True)

    pool = []
    used_matches = set()
    for p in candidates:
        if len(pool) >= DRAW_PICKS_MAX:
            break
        if p["match"] in used_matches:
            continue
        pool.append(p)
        used_matches.add(p["match"])

    n = len(pool)
    label, r_list, columns = _draw_system_for_n(n)

    # stake target 25–50, but also <= cap_total
    unit, stake = (0.0, 0.0)
    if columns > 0:
        # base unit=1 if possible
        base_total = columns * SYS_UNIT_BASE
        if SYS_TARGET_MIN <= base_total <= SYS_TARGET_MAX and base_total <= cap_total:
            unit = 1.0
            stake = base_total
        else:
            target = base_total
            if target < SYS_TARGET_MIN:
                target = SYS_TARGET_MIN
            if target > SYS_TARGET_MAX:
                target = SYS_TARGET_MAX
            if target > cap_total:
                target = cap_total
            unit = round(target / columns, 2)
            stake = round(unit * columns, 2)

    payload = {
        "bankroll": bankroll_draw,
        "exposure_cap_pct": DRAW_EXPOSURE_CAP,
        "rules": {
            "odds_range": [DRAW_ODDS_MIN, DRAW_ODDS_MAX],
            "picks_try_up_to": DRAW_PICKS_MAX,
            "system_map": {"2":"2/2", "3":"2/3", "4":"2/4", "5":"2-3/5"},
            "target_total_range": [SYS_TARGET_MIN, SYS_TARGET_MAX],
        },
        "picks_total": sort_by_datetime([_strip_pick(p) for p in pool]),
        "system_pool": sort_by_datetime([_strip_pick(p) for p in pool]),
        "system": {
            "label": label if n >= 2 else None,
            "columns": columns if n >= 2 else 0,
            "unit": unit if n >= 2 else 0.0,
            "stake": stake if n >= 2 else 0.0,
            "has_system": bool(n >= 2 and columns > 0),
            "mode": ("single" if n == 1 else "system"),
        },
        "open": round(stake, 2) if n >= 2 else 0.0,
        "after_open": round(bankroll_draw - (stake if n >= 2 else 0.0), 2),
        "counts": {"picks_total": n, "system_pool": n},
    }
    # if only 1 draw, allow single
    if n == 1:
        # single stake = min(25, cap_total) but not forced; keep 25 as standard
        st = min(25.0, cap_total) if cap_total > 0 else 0.0
        one = pool[0]
        payload["singles"] = [{
            "pick_id": one["pick_id"],
            "fixture_id": one["fixture_id"],
            "date": one.get("date"),
            "time": one.get("time"),
            "market_code": one["market_code"],
            "match": one["match"],
            "league": one["league"],
            "market": one["market"],
            "odds": round(float(one["odds"]), 3),
            "prob": one.get("prob"),
            "ev": one.get("ev"),
            "stake": round(st, 2),
            "tag": "draw_single",
        }]
        payload["open"] = round(st, 2)
        payload["after_open"] = round(bankroll_draw - st, 2)
    return payload

# ------------------------- MAIN -------------------------
def main():
    fixtures, th_meta = load_thursday()
    history = load_history()
    window = th_meta.get("window", {}) if isinstance(th_meta, dict) else {}
    wf = get_week_fields(window, history)

    core_start = safe_float(history.get("core", {}).get("bankroll_current"), None)
    fun_start  = safe_float(history.get("funbet", {}).get("bankroll_current"), None)
    draw_start = safe_float(history.get("drawbet", {}).get("bankroll_current"), None)

    core_bankroll_start = core_start if core_start is not None else DEFAULT_BANKROLL_CORE
    fun_bankroll_start  = fun_start  if fun_start  is not None else DEFAULT_BANKROLL_FUN
    draw_bankroll_start = draw_start if draw_start is not None else DEFAULT_BANKROLL_DRAW

    picks = build_candidates(fixtures)

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
            "rules": {
                "singles_odds_range": [CORE_SINGLES_MIN_ODDS, CORE_SINGLES_MAX_ODDS],
                "low_odds_to_doubles": [CORE_LOW_ODDS_MIN, CORE_LOW_ODDS_MAX],
                "stake_ladder": {"1.60-1.75": 40, "1.75-1.90": 30, "1.90-2.10": 20},
                "max_singles": CORE_MAX_SINGLES,
                "max_doubles": CORE_MAX_DOUBLES,
                "no_scaling": True,
                "cap_policy": "prune_low_priority",
            },
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
