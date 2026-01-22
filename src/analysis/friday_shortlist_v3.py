# ============================================================
#  src/analysis/friday_shortlist_v3.py
#  FRIDAY SHORTLIST v3.30 — PRODUCTION (CoreBet + FunBet + DrawBet)
#
#  LOCKED CHANGES (per your "OK"):
#   1) FunBet Under allowed ONLY if Thursday flags.under_elite == True
#      AND flags.over_friendly_league == False.
#   2) "Underdog brake" on FunBet:
#      - HARD cap odds <= 3.30
#      - SOFT cap odds <= 3.00 with max 2 picks above 3.00 (when 7 picks)
#      - For odds > 3.00: require prob >= 0.35 AND confidence >= 0.70
#   3) FunBet unit: prefer Unit=1 when possible (Columns<=50 and 25<=Stake<=50)
#
#  Reads:
#   - logs/thursday_report_v3.json
#   - (optional) logs/tuesday_history_v3.json  (bankroll carry + week numbering)
#
#  Writes:
#   - logs/friday_shortlist_v3.json
# ============================================================

import os
import json
from datetime import datetime, date
from math import comb

THURSDAY_REPORT_PATH = os.getenv("THURSDAY_REPORT_PATH", "logs/thursday_report_v3.json")
FRIDAY_REPORT_PATH = os.getenv("FRIDAY_REPORT_PATH", "logs/friday_shortlist_v3.json")
TUESDAY_HISTORY_PATH = os.getenv("TUESDAY_HISTORY_PATH", "logs/tuesday_history_v3.json")

# ------------------------- BANKROLL DEFAULTS -------------------------
DEFAULT_BANKROLL_CORE = float(os.getenv("BANKROLL_CORE", "800"))
DEFAULT_BANKROLL_FUN = float(os.getenv("BANKROLL_FUN", "400"))
DEFAULT_BANKROLL_DRAW = float(os.getenv("BANKROLL_DRAW", "300"))

# Exposure caps (used as hard cap; if exceeded, we DROP lowest ranked picks; NO scaling)
CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.25"))   # conservative; change if you want
FUN_EXPOSURE_CAP = float(os.getenv("FUN_EXPOSURE_CAP", "0.20"))     # 80 on 400
DRAW_EXPOSURE_CAP = float(os.getenv("DRAW_EXPOSURE_CAP", "0.20"))   # 60 on 300

# Odds-match / confidence gates
ODDS_MATCH_MIN_SCORE_CORE = float(os.getenv("ODDS_MATCH_MIN_SCORE_CORE", "0.75"))
ODDS_MATCH_MIN_SCORE_FUN = float(os.getenv("ODDS_MATCH_MIN_SCORE_FUN", "0.75"))
ODDS_MATCH_MIN_SCORE_DRAW = float(os.getenv("ODDS_MATCH_MIN_SCORE_DRAW", "0.80"))

CORE_MIN_CONFIDENCE = float(os.getenv("CORE_MIN_CONFIDENCE", "0.55"))
FUN_MIN_CONFIDENCE = float(os.getenv("FUN_MIN_CONFIDENCE", "0.45"))
DRAW_MIN_CONFIDENCE = float(os.getenv("DRAW_MIN_CONFIDENCE", "0.55"))

# Overlap
FUN_AVOID_CORE_OVERLAP = os.getenv("FUN_AVOID_CORE_OVERLAP", "true").lower() == "true"

# ------------------------- MARKETS -------------------------
MARKET_CODE_TO_NAME = {"1": "Home", "2": "Away", "X": "Draw", "O25": "Over 2.5", "U25": "Under 2.5"}
MARKET_NAME_TO_CODE = {v: k for k, v in MARKET_CODE_TO_NAME.items()}

CORE_ALLOWED_MARKETS = {"Home", "Away", "Over 2.5", "Under 2.5"}
FUN_ALLOWED_MARKETS = {"Home", "Away", "Over 2.5", "Under 2.5"}   # draws handled by DrawBet
DRAW_ALLOWED_MARKETS = {"Draw"}

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

def load_history():
    if not os.path.exists(TUESDAY_HISTORY_PATH):
        return {
            "week_count": 0,
            "weeks": {},
            "core": {"bankroll_current": None},
            "funbet": {"bankroll_current": None},
            "drawbet": {"bankroll_current": None},
        }
    try:
        h = load_json(TUESDAY_HISTORY_PATH)
        h.setdefault("week_count", 0)
        h.setdefault("weeks", {})
        h.setdefault("core", {}).setdefault("bankroll_current", None)
        h.setdefault("funbet", {}).setdefault("bankroll_current", None)
        h.setdefault("drawbet", {}).setdefault("bankroll_current", None)
        return h
    except Exception:
        return {
            "week_count": 0,
            "weeks": {},
            "core": {"bankroll_current": None},
            "funbet": {"bankroll_current": None},
            "drawbet": {"bankroll_current": None},
        }

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
                "market": MARKET_CODE_TO_NAME.get(mcode, mcode),
                "odds": odds,
                "prob": _prob_from_fx(fx, mcode),
                "ev": _ev_from_fx(fx, mcode),
                "confidence": confidence_value(fx),
                "flags": fx.get("flags") or {},
                "fx": fx,
            })
    return out

def _pick_sort_key(p):
    # stable: by date/time first, then EV, then confidence, then prob
    dt = f"{p.get('date','')} {p.get('time','')}"
    return (dt, safe_float(p.get("ev"), -9999.0), safe_float(p.get("confidence"), 0.0), safe_float(p.get("prob"), 0.0))

def _rank_key(p):
    # ranking key for quality: EV then confidence then prob
    return (safe_float(p.get("ev"), -9999.0), safe_float(p.get("confidence"), 0.0), safe_float(p.get("prob"), 0.0))

# ============================================================
# COREBET — RULES (your locked ladder)
# ============================================================
CORE_SINGLES_MIN_ODDS = float(os.getenv("CORE_SINGLES_MIN_ODDS", "1.60"))
CORE_SINGLES_MAX_ODDS = float(os.getenv("CORE_SINGLES_MAX_ODDS", "2.10"))

CORE_LOW_ODDS_MIN = float(os.getenv("CORE_LOW_ODDS_MIN", "1.30"))
CORE_LOW_ODDS_MAX = float(os.getenv("CORE_LOW_ODDS_MAX", "1.60"))  # low-odds bucket goes to doubles

CORE_MIN_SINGLES = int(os.getenv("CORE_MIN_SINGLES", "5"))
CORE_MAX_SINGLES = int(os.getenv("CORE_MAX_SINGLES", "8"))
CORE_MAX_DOUBLES = int(os.getenv("CORE_MAX_DOUBLES", "2"))

CORE_DOUBLE_TARGET_MIN = float(os.getenv("CORE_DOUBLE_TARGET_MIN", "2.10"))
CORE_DOUBLE_TARGET_MAX = float(os.getenv("CORE_DOUBLE_TARGET_MAX", "3.50"))

# stake ladder EXACT
def core_single_stake(odds: float) -> float:
    if 1.60 <= odds <= 1.75:
        return 40.0
    if 1.75 < odds <= 1.90:
        return 30.0
    if 1.90 < odds <= 2.10:
        return 20.0
    return 0.0

def core_double_stake(combo_odds: float) -> float:
    # always 15 in your usage, keep ladder if needed
    if combo_odds <= 3.50:
        return 15.0
    return 10.0

# Core gates (tight but not insane)
CORE_EV_MIN = float(os.getenv("CORE_EV_MIN", "0.03"))
CORE_P_MIN = float(os.getenv("CORE_P_MIN", "0.30"))

# Under stricter in core
CORE_UNDER_REQUIRE_ELITE = os.getenv("CORE_UNDER_REQUIRE_ELITE", "true").lower() == "true"

def _strip_core(p, stake: float, tag: str):
    return {
        "pick_id": p["pick_id"],
        "fixture_id": p["fixture_id"],
        "date": p.get("date"),
        "time": p.get("time"),
        "league": p.get("league"),
        "match": p["match"],
        "market": p["market"],
        "market_code": p["market_code"],
        "odds": round(float(p["odds"]), 3),
        "prob": p.get("prob"),
        "ev": p.get("ev"),
        "confidence": p.get("confidence"),
        "stake": float(stake),
        "tag": tag,
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

        odds = safe_float(p.get("odds"), None)
        evv = safe_float(p.get("ev"), None)
        pr = safe_float(p.get("prob"), None)
        if odds is None or evv is None or pr is None:
            continue

        # basic value gate
        if evv < CORE_EV_MIN or pr < CORE_P_MIN:
            continue

        # Core under: prefer elite only (optional)
        if p["market"] == "Under 2.5" and CORE_UNDER_REQUIRE_ELITE:
            if not (p.get("flags") or {}).get("under_elite", False):
                continue

        if CORE_SINGLES_MIN_ODDS <= odds <= CORE_SINGLES_MAX_ODDS:
            st = core_single_stake(odds)
            if st <= 0:
                continue
            singles_pool.append({**p, "_stake": st})
        elif CORE_LOW_ODDS_MIN <= odds <= CORE_LOW_ODDS_MAX:
            low_pool.append(p)

    # rank best first
    singles_pool.sort(key=lambda x: _rank_key(x), reverse=True)
    low_pool.sort(key=lambda x: _rank_key(x), reverse=True)

    singles = []
    used_matches = set()

    # pick up to CORE_MAX_SINGLES unique matches
    for p in singles_pool:
        if len(singles) >= CORE_MAX_SINGLES:
            break
        if p["match"] in used_matches:
            continue
        singles.append(_strip_core(p, p["_stake"], "core_single"))
        used_matches.add(p["match"])

    # ensure minimum (if possible)
    if len(singles) < CORE_MIN_SINGLES:
        for p in singles_pool:
            if len(singles) >= CORE_MIN_SINGLES:
                break
            if p["match"] in used_matches:
                continue
            singles.append(_strip_core(p, p["_stake"], "core_single"))
            used_matches.add(p["match"])

    # Doubles: build from low_pool + best partner (from singles_pool range too)
    doubles = []
    if low_pool and CORE_MAX_DOUBLES > 0:
        # partner candidates: odds up to 2.10 (so combos sit in target)
        partners = []
        for pp in picks:
            fx = pp["fx"]
            if pp["market"] not in CORE_ALLOWED_MARKETS:
                continue
            if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_CORE):
                continue
            if not confidence_ok(fx, CORE_MIN_CONFIDENCE):
                continue
            o = safe_float(pp.get("odds"), None)
            if o is None:
                continue
            if not (CORE_LOW_ODDS_MIN <= o <= CORE_SINGLES_MAX_ODDS):
                continue
            evv = safe_float(pp.get("ev"), None)
            pr = safe_float(pp.get("prob"), None)
            if evv is None or pr is None:
                continue
            if evv < CORE_EV_MIN or pr < CORE_P_MIN:
                continue
            if pp["market"] == "Under 2.5" and CORE_UNDER_REQUIRE_ELITE:
                if not (pp.get("flags") or {}).get("under_elite", False):
                    continue
            partners.append(pp)

        partners.sort(key=lambda x: _rank_key(x), reverse=True)

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
                stake = core_double_stake(combo)
                doubles.append({
                    "legs": [
                        {"pick_id": leg1["pick_id"], "fixture_id": leg1["fixture_id"], "date": leg1.get("date"), "time": leg1.get("time"),
                         "league": leg1.get("league"), "match": leg1["match"], "market": leg1["market"], "market_code": leg1["market_code"], "odds": round(float(leg1["odds"]), 3)},
                        {"pick_id": leg2["pick_id"], "fixture_id": leg2["fixture_id"], "date": leg2.get("date"), "time": leg2.get("time"),
                         "league": leg2.get("league"), "match": leg2["match"], "market": leg2["market"], "market_code": leg2["market_code"], "odds": round(float(leg2["odds"]), 3)},
                    ],
                    "combo_odds": round(combo, 2),
                    "stake": round(stake, 2),
                    "tag": "core_double",
                })
                used_double_matches.add(leg1["match"])
                used_double_matches.add(leg2["match"])
                break

    # Hard cap enforcement WITHOUT scaling: drop lowest-ranked singles/doubles until fits
    def open_total():
        return round(sum(x["stake"] for x in singles) + sum(d.get("stake", 0.0) for d in doubles), 2)

    # rank singles low-to-high for dropping
    singles_ranked = sorted(singles, key=lambda x: (safe_float(x.get("ev"), -9999.0), safe_float(x.get("confidence"), 0.0), safe_float(x.get("prob"), 0.0)))
    # doubles ranked low-to-high by combo odds then stake
    doubles_ranked = sorted(doubles, key=lambda d: (safe_float(d.get("combo_odds"), 0.0), safe_float(d.get("stake"), 0.0)))

    while open_total() > cap_amount and (singles_ranked or doubles_ranked):
        # drop from singles first (most flexible)
        if singles_ranked:
            drop = singles_ranked.pop(0)
            singles = [s for s in singles if s["pick_id"] != drop["pick_id"]]
        elif doubles_ranked:
            dropd = doubles_ranked.pop(0)
            doubles = [d for d in doubles if d.get("combo_odds") != dropd.get("combo_odds") or d.get("stake") != dropd.get("stake")]

    meta = {
        "bankroll": bankroll_core,
        "exposure_cap_pct": CORE_EXPOSURE_CAP,
        "open": open_total(),
        "after_open": round(bankroll_core - open_total(), 2),
        "picks_count": len(singles),
        "doubles_count": len(doubles),
        "scale_applied": None,
    }

    core_double = doubles[0] if doubles else None
    return singles, core_double, doubles, meta

# ============================================================
# FUNBET — SYSTEM ONLY (4/7 preferred, no 5/7 preference unless needed)
# ============================================================
FUN_PICKS_MIN = int(os.getenv("FUN_PICKS_MIN", "5"))
FUN_PICKS_MAX = int(os.getenv("FUN_PICKS_MAX", "7"))

FUN_MIN_ODDS = float(os.getenv("FUN_MIN_ODDS", "1.90"))
FUN_HARD_MAX_ODDS = float(os.getenv("FUN_HARD_MAX_ODDS", "3.30"))
FUN_SOFT_MAX_ODDS = float(os.getenv("FUN_SOFT_MAX_ODDS", "3.00"))
FUN_SOFT_MAX_ODDS_MAXCOUNT_IN_7 = int(os.getenv("FUN_SOFT_MAX_ODDS_MAXCOUNT_IN_7", "2"))

FUN_BIG_ODDS_MIN_PROB = float(os.getenv("FUN_BIG_ODDS_MIN_PROB", "0.35"))
FUN_BIG_ODDS_MIN_CONF = float(os.getenv("FUN_BIG_ODDS_MIN_CONF", "0.70"))

# refund targets
SYS_REFUND_PRIMARY = float(os.getenv("SYS_REFUND_PRIMARY", "0.70"))
SYS_REFUND_FALLBACK = float(os.getenv("SYS_REFUND_FALLBACK", "0.65"))

# system spend (target 25–50) and Unit=1 preference
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
    Prefer lower systems (easier):
      n=7: try 4/7 first, then 5/7 only if needed
      n=6: try 3/6 first, then 4/6
      n=5: 3/5
    Primary threshold first, else fallback threshold.
    """
    n = len(pool)
    if n < 5:
        return None, None, True

    trials = []
    if n == 7:
        trials = [_system_try(pool, 4, "4/7"), _system_try(pool, 5, "5/7")]
    elif n == 6:
        trials = [_system_try(pool, 3, "3/6"), _system_try(pool, 4, "4/6")]
    else:
        trials = [_system_try(pool, 3, "3/5")]

    trials = [t for t in trials if t]

    # primary
    for t in trials:
        if (t.get("refund_ratio_min_hits") or 0.0) >= SYS_REFUND_PRIMARY:
            return t, SYS_REFUND_PRIMARY, False

    # fallback
    for t in trials:
        if (t.get("refund_ratio_min_hits") or 0.0) >= SYS_REFUND_FALLBACK:
            return t, SYS_REFUND_FALLBACK, True

    # if none pass, still pick the "easier" first option but mark breach
    return (trials[0] if trials else None), SYS_REFUND_FALLBACK, True

def _unit_prefer_one(columns: int, cap_system: float):
    """
    If unit=1 gives stake within [25,50] and columns<=50 -> use unit=1.
    Else compute unit to fit in [25,50] but not exceeding cap.
    """
    if columns <= 0:
        return 0.0, 0.0

    if columns <= 50:
        stake1 = float(columns) * 1.0
        if SYS_TARGET_MIN <= stake1 <= SYS_TARGET_MAX and stake1 <= cap_system:
            return 1.0, round(stake1, 2)

    # otherwise compute to target within [25,50]
    target = _clamp(float(columns), SYS_TARGET_MIN, SYS_TARGET_MAX)
    target = min(target, cap_system)
    if target <= 0:
        return 0.0, 0.0
    unit = round(target / float(columns), 2)
    stake = round(unit * float(columns), 2)
    return unit, stake

def _strip_pick(p):
    return {
        "pick_id": p["pick_id"],
        "fixture_id": p["fixture_id"],
        "date": p.get("date"),
        "time": p.get("time"),
        "league": p.get("league"),
        "match": p["match"],
        "market": p["market"],
        "market_code": p["market_code"],
        "odds": round(float(p["odds"]), 3),
        "prob": p.get("prob"),
        "ev": p.get("ev"),
        "confidence": p.get("confidence"),
    }

def funbet_select(picks, bankroll_fun, core_fixture_ids):
    cap_system = bankroll_fun * FUN_EXPOSURE_CAP

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

        odds = safe_float(p.get("odds"), None)
        pr = safe_float(p.get("prob"), None)
        evv = safe_float(p.get("ev"), None)
        if odds is None or pr is None or evv is None:
            continue

        # odds limits
        if odds < FUN_MIN_ODDS:
            continue
        if odds > FUN_HARD_MAX_ODDS:
            continue

        # UNDER: ONLY under_elite + NOT over-friendly league
        if p["market"] == "Under 2.5":
            if not (p.get("flags") or {}).get("under_elite", False):
                continue
            if (p.get("flags") or {}).get("over_friendly_league", False):
                continue

        # "big odds" extra brake (over 3.00)
        conf = safe_float(p.get("confidence"), None)
        if odds > FUN_SOFT_MAX_ODDS:
            if pr < FUN_BIG_ODDS_MIN_PROB:
                continue
            if conf is None or conf < FUN_BIG_ODDS_MIN_CONF:
                continue

        candidates.append(p)

    # rank by EV then confidence then prob
    candidates.sort(key=lambda x: _rank_key(x), reverse=True)

    # pick up to 7 unique matches
    picks_out = []
    used_matches = set()
    over3_count = 0

    for p in candidates:
        if p["match"] in used_matches:
            continue

        # enforce "max 2 picks above 3.00 when we end up with 7 picks"
        if safe_float(p["odds"], 0.0) > FUN_SOFT_MAX_ODDS:
            if over3_count >= FUN_SOFT_MAX_ODDS_MAXCOUNT_IN_7:
                continue

        picks_out.append(p)
        used_matches.add(p["match"])
        if safe_float(p["odds"], 0.0) > FUN_SOFT_MAX_ODDS:
            over3_count += 1

        if len(picks_out) >= FUN_PICKS_MAX:
            break

    # allow smaller if not enough
    if len(picks_out) < FUN_PICKS_MIN:
        # keep whatever we got; system may be None
        pass

    # choose pool size prefer 7, else 6, else 5
    if len(picks_out) >= 7:
        pool = picks_out[:7]
    elif len(picks_out) >= 6:
        pool = picks_out[:6]
    else:
        pool = picks_out[:5]

    sys_choice, refund_used, breached = _choose_fun_system(pool)
    columns = int(sys_choice["columns"]) if sys_choice else 0

    unit, stake = (0.0, 0.0)
    if columns > 0:
        unit, stake = _unit_prefer_one(columns, cap_system)

    payload = {
        "portfolio": "FunBet",
        "bankroll": bankroll_fun,
        "exposure_cap_pct": FUN_EXPOSURE_CAP,
        "rules": {
            "picks_range": [FUN_PICKS_MIN, FUN_PICKS_MAX],
            "odds_min": FUN_MIN_ODDS,
            "hard_max_odds": FUN_HARD_MAX_ODDS,
            "soft_max_odds": FUN_SOFT_MAX_ODDS,
            "max_count_over_soft_in_7": FUN_SOFT_MAX_ODDS_MAXCOUNT_IN_7,
            "big_odds_min_prob": FUN_BIG_ODDS_MIN_PROB,
            "big_odds_min_conf": FUN_BIG_ODDS_MIN_CONF,
            "under_only_elite": True,
            "under_block_over_friendly": True,
            "refund_primary": SYS_REFUND_PRIMARY,
            "refund_fallback": SYS_REFUND_FALLBACK,
            "refund_used": refund_used,
            "refund_rule_breached": bool(breached),
            "unit_prefer_one": True,
            "stake_target_range": [SYS_TARGET_MIN, SYS_TARGET_MAX],
        },
        "picks_total": [_strip_pick(p) for p in picks_out],
        "system_pool": [_strip_pick(p) for p in pool],
        "system": {
            "label": (sys_choice["label"] if sys_choice else None),
            "columns": columns,
            "min_hits": (int(sys_choice["min_hits"]) if sys_choice else None),
            "refund_ratio_min_hits": (float(sys_choice["refund_ratio_min_hits"]) if sys_choice else None),
            "unit": unit,
            "stake": stake,
            "refund_used": refund_used,
            "refund_rule_breached": bool(breached),
            "has_system": bool(columns > 0 and stake > 0),
        },
        "open": round(stake, 2),
        "after_open": round(bankroll_fun - stake, 2),
        "counts": {"picks_total": len(picks_out), "system_pool": len(pool)},
    }
    return payload

# ============================================================
# DRAWBET — system only, 2–5 draws, odds 2.80–3.70, target 25–50
# ============================================================
DRAW_P_MIN = float(os.getenv("DRAW_P_MIN", "0.28"))
DRAW_EV_MIN = float(os.getenv("DRAW_EV_MIN", "0.03"))
DRAW_ODDS_MIN = float(os.getenv("DRAW_ODDS_MIN", "2.80"))
DRAW_ODDS_MAX = float(os.getenv("DRAW_ODDS_MAX", "3.70"))

def _draw_system_label(count: int):
    if count >= 5:
        return "2-3/5"
    if count == 4:
        return "2-3/4"
    if count == 3:
        return "2/3"
    if count == 2:
        return "2/2"
    return None

def _draw_columns(count: int, label: str | None):
    if not label or count < 2:
        return 0
    if label == "2-3/5":
        return comb(5, 2) + comb(5, 3)
    if label == "2-3/4":
        return comb(4, 2) + comb(4, 3)
    if label == "2/3":
        return comb(3, 2)
    if label == "2/2":
        return 1
    return 0

def drawbet_select(picks, bankroll_draw):
    cap_system = bankroll_draw * DRAW_EXPOSURE_CAP

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
        pr = safe_float(p.get("prob"), None)
        evv = safe_float(p.get("ev"), None)
        if odds is None or pr is None or evv is None:
            continue

        if not (DRAW_ODDS_MIN <= odds <= DRAW_ODDS_MAX):
            continue
        if pr < DRAW_P_MIN or evv < DRAW_EV_MIN:
            continue

        candidates.append(p)

    candidates.sort(key=lambda x: _rank_key(x), reverse=True)

    pool = []
    used = set()
    for p in candidates:
        if p["match"] in used:
            continue
        pool.append(p)
        used.add(p["match"])
        if len(pool) >= 5:
            break

    # try sizes 5..2
    chosen_pool = []
    chosen_label = None
    for sz in [5, 4, 3, 2]:
        if len(pool) >= sz:
            chosen_pool = pool[:sz]
            chosen_label = _draw_system_label(sz)
            break

    cols = _draw_columns(len(chosen_pool), chosen_label)

    unit, stake = (0.0, 0.0)
    if cols > 0:
        # prefer unit=1 if it fits
        if cols <= 50 and SYS_TARGET_MIN <= cols <= SYS_TARGET_MAX and cols <= cap_system:
            unit, stake = 1.0, round(float(cols), 2)
        else:
            target = _clamp(float(cols), SYS_TARGET_MIN, SYS_TARGET_MAX)
            target = min(target, cap_system)
            unit = round(target / float(cols), 2) if cols > 0 else 0.0
            stake = round(unit * float(cols), 2)

    payload = {
        "portfolio": "DrawBet",
        "bankroll": bankroll_draw,
        "exposure_cap_pct": DRAW_EXPOSURE_CAP,
        "rules": {
            "odds_range": [DRAW_ODDS_MIN, DRAW_ODDS_MAX],
            "p_min": DRAW_P_MIN,
            "ev_min": DRAW_EV_MIN,
            "pool_size_range": [2, 5],
            "stake_target_range": [SYS_TARGET_MIN, SYS_TARGET_MAX],
        },
        "picks_total": [_strip_pick(p) for p in chosen_pool],
        "system_pool": [_strip_pick(p) for p in chosen_pool],
        "system": {
            "label": chosen_label,
            "columns": cols,
            "unit": unit,
            "stake": stake,
            "has_system": bool(cols > 0 and stake > 0),
        },
        "open": round(stake, 2),
        "after_open": round(bankroll_draw - stake, 2),
        "counts": {"picks_total": len(chosen_pool), "system_pool": len(chosen_pool)},
    }
    return payload

# ============================================================
# MAIN
# ============================================================
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

    # CORE
    core_singles, core_double, core_doubles, core_meta = corebet_select(picks, core_bankroll_start)

    core_fixture_ids = {x["fixture_id"] for x in core_singles}

    # FUN (system only)
    funbet = funbet_select(picks, fun_bankroll_start, core_fixture_ids)

    # DRAW (system only)
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
                "stake_ladder": {"1.60-1.75": 40, "1.75-1.90": 30, "1.90-2.10": 20},
                "low_odds_to_doubles": [CORE_LOW_ODDS_MIN, CORE_LOW_ODDS_MAX],
                "double_target_combo_odds": [CORE_DOUBLE_TARGET_MIN, CORE_DOUBLE_TARGET_MAX],
                "no_scaling": True,
                "cap_policy": "drop_lowest_ranked_until_within_cap",
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
            "bankroll": DEFAULT_BANKROLL_FUN,
            "bankroll_start": fun_bankroll_start,
            "bankroll_source": ("history" if fun_start is not None else "default"),
            **funbet,
        },

        "drawbet": {
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
