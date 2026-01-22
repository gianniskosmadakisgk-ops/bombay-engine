# ============================================================
#  src/analysis/friday_shortlist_v3.py
#  FRIDAY SHORTLIST v3.30 — PRODUCTION (CoreBet + FunBet + DrawBet)
#
#  ΚΛΕΙΔΩΜΕΝΟΙ ΚΑΝΟΝΕΣ (σύμφωνα με όσα κλειδώσαμε):
#  - NO SCALING: τα stakes/units βγαίνουν “καθαρά” από τους κανόνες (όχι αυτόματη κλιμάκωση).
#  - Χρονική σειρά στα outputs: ταξινόμηση ανά Date/Time από το Thursday fixture.
#
#  COREBET (singles + doubles)
#   - Singles odds: 1.60–2.10
#     • 1.60–1.75 => 40€
#     • 1.75–1.90 => 30€
#     • 1.90–2.10 => 20€
#   - Low odds 1.30–1.60 => ΜΟΝΟ για doubles (όχι singles)
#   - Under στο Core: ΜΟΝΟ αν flags.under_elite == true και max 20% των core singles
#   - Over στο Core: flags.over_good_shape == true ΚΑΙ flags.tight_game == false (tight_game κόβει Core Over)
#   - Home/Away στο Core: flags.home_shape / flags.away_shape αντίστοιχα
#   - Max Core picks: 8 singles (default) + έως 2 doubles
#
#  FUNBET (SYSTEM-ONLY)
#   - Picks odds: 1.90–3.60
#   - tight_game ΕΠΙΤΡΕΠΕΤΑΙ στο Fun (δεν κόβει)
#   - Για Over στο Fun: flags.over_good_shape == true (tight_game δεν κόβει)
#   - System preference: προτιμάμε ΜΙΚΡΟΤΕΡΟ min_hits
#       n=7: 4/7 αλλιώς 5/7 (μόνο αν refund>=0.65)
#       n=6: 3/6 αλλιώς 4/6 (μόνο αν refund>=0.65)
#       n=5: 3/5 (μόνο αν refund>=0.65)
#   - Refund threshold: 0.65
#   - Target system spend: 25–50 (unit computed from columns)
#
#  DRAWBET (SYSTEM-ONLY)
#   - Θέλουμε 2–5 draws (αν υπάρχουν) και παίζουμε system ανάλογα:
#       n=5: 2-3/5
#       n=4: 2-3/4
#       n=3: 2/3
#       n=2: 2/2
#   - Odds range: 3.00–4.60 (default)
#   - Target spend: 25–50
#
#  Reads:
#   - logs/thursday_report_v3.json
#   - (optional) logs/tuesday_history_v3.json
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

# ------------------------- DEFAULT BANKROLLS -------------------------
DEFAULT_BANKROLL_CORE = float(os.getenv("BANKROLL_CORE", "800"))
DEFAULT_BANKROLL_FUN = float(os.getenv("BANKROLL_FUN", "400"))
DEFAULT_BANKROLL_DRAW = float(os.getenv("BANKROLL_DRAW", "300"))

# (Δεν κάνουμε scaling, αλλά κρατάμε τα caps ως metadata/πληροφορία)
CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.15"))
FUN_EXPOSURE_CAP = float(os.getenv("FUN_EXPOSURE_CAP", "0.20"))
DRAW_EXPOSURE_CAP = float(os.getenv("DRAW_EXPOSURE_CAP", "0.20"))

# Gates
ODDS_MATCH_MIN_SCORE_CORE = float(os.getenv("ODDS_MATCH_MIN_SCORE_CORE", "0.75"))
ODDS_MATCH_MIN_SCORE_FUN = float(os.getenv("ODDS_MATCH_MIN_SCORE_FUN", "0.65"))
ODDS_MATCH_MIN_SCORE_DRAW = float(os.getenv("ODDS_MATCH_MIN_SCORE_DRAW", "0.75"))

CORE_MIN_CONFIDENCE = float(os.getenv("CORE_MIN_CONFIDENCE", "0.55"))
FUN_MIN_CONFIDENCE = float(os.getenv("FUN_MIN_CONFIDENCE", "0.55"))
DRAW_MIN_CONFIDENCE = float(os.getenv("DRAW_MIN_CONFIDENCE", "0.55"))

# Fun overlap with core: επιτρέπεται overlap (default false means we DO allow overlap)
FUN_AVOID_CORE_OVERLAP = os.getenv("FUN_AVOID_CORE_OVERLAP", "false").lower() == "true"

# ------------------------- LIMITS -------------------------
# Core odds
CORE_SINGLES_MIN_ODDS = float(os.getenv("CORE_SINGLES_MIN_ODDS", "1.60"))
CORE_SINGLES_MAX_ODDS = float(os.getenv("CORE_SINGLES_MAX_ODDS", "2.10"))
CORE_LOW_ODDS_MIN = float(os.getenv("CORE_LOW_ODDS_MIN", "1.30"))
CORE_LOW_ODDS_MAX = float(os.getenv("CORE_LOW_ODDS_MAX", "1.60"))

CORE_MAX_SINGLES = int(os.getenv("CORE_MAX_SINGLES", "8"))
CORE_MIN_SINGLES = int(os.getenv("CORE_MIN_SINGLES", "5"))
CORE_MAX_DOUBLES = int(os.getenv("CORE_MAX_DOUBLES", "2"))

# Core double
CORE_DOUBLE_TARGET_MIN = float(os.getenv("CORE_DOUBLE_TARGET_MIN", "2.00"))
CORE_DOUBLE_TARGET_MAX = float(os.getenv("CORE_DOUBLE_TARGET_MAX", "3.50"))
CORE_DOUBLE_STAKE = float(os.getenv("CORE_DOUBLE_STAKE", "15.0"))  # σταθερό

# Core share caps
CORE_MAX_UNDER_SHARE = float(os.getenv("CORE_MAX_UNDER_SHARE", "0.20"))
CORE_MIN_1X2_SHARE = float(os.getenv("CORE_MIN_1X2_SHARE", "0.30"))

# Fun odds & picks
FUN_ODDS_MIN = float(os.getenv("FUN_ODDS_MIN", "1.90"))
FUN_ODDS_MAX = float(os.getenv("FUN_ODDS_MAX", "3.60"))
FUN_PICKS_MIN = int(os.getenv("FUN_PICKS_MIN", "5"))
FUN_PICKS_MAX = int(os.getenv("FUN_PICKS_MAX", "7"))

# Draw odds & picks
DRAW_ODDS_MIN = float(os.getenv("DRAW_ODDS_MIN", "3.00"))
DRAW_ODDS_MAX = float(os.getenv("DRAW_ODDS_MAX", "4.60"))
DRAW_PICKS_MAX = int(os.getenv("DRAW_PICKS_MAX", "5"))
DRAW_PICKS_MIN = int(os.getenv("DRAW_PICKS_MIN", "2"))

# System policy (Fun/Draw)
SYS_REFUND_MIN = float(os.getenv("SYS_REFUND_MIN", "0.65"))
SYS_UNIT_MIN = float(os.getenv("SYS_UNIT_MIN", "0.50"))
SYS_UNIT_MAX = float(os.getenv("SYS_UNIT_MAX", "2.00"))
SYS_TARGET_MIN = float(os.getenv("SYS_TARGET_MIN", "25.0"))
SYS_TARGET_MAX = float(os.getenv("SYS_TARGET_MAX", "50.0"))

# Thresholds (EV/prob)
CORE_EV_MIN_HOME = float(os.getenv("CORE_EV_MIN_HOME", "0.04"))
CORE_EV_MIN_AWAY = float(os.getenv("CORE_EV_MIN_AWAY", "0.05"))
CORE_EV_MIN_OVER = float(os.getenv("CORE_EV_MIN_OVER", "0.04"))
CORE_EV_MIN_UNDER = float(os.getenv("CORE_EV_MIN_UNDER", "0.08"))

CORE_P_MIN_HOME = float(os.getenv("CORE_P_MIN_HOME", "0.30"))
CORE_P_MIN_AWAY = float(os.getenv("CORE_P_MIN_AWAY", "0.24"))
CORE_P_MIN_OVER = float(os.getenv("CORE_P_MIN_OVER", "0.45"))
CORE_P_MIN_UNDER = float(os.getenv("CORE_P_MIN_UNDER", "0.58"))

FUN_EV_MIN = float(os.getenv("FUN_EV_MIN", "0.05"))
FUN_P_MIN_HOME = float(os.getenv("FUN_P_MIN_HOME", "0.28"))
FUN_P_MIN_AWAY = float(os.getenv("FUN_P_MIN_AWAY", "0.22"))
FUN_P_MIN_OVER = float(os.getenv("FUN_P_MIN_OVER", "0.50"))
FUN_P_MIN_UNDER = float(os.getenv("FUN_P_MIN_UNDER", "0.60"))

DRAW_EV_MIN = float(os.getenv("DRAW_EV_MIN", "0.05"))
DRAW_P_MIN = float(os.getenv("DRAW_P_MIN", "0.28"))

# ------------------------- MARKET MAP -------------------------
MARKET_CODE = {
    "Home": "1",
    "Draw": "X",
    "Away": "2",
    "Over 2.5": "O25",
    "Under 2.5": "U25",
}

CODE_TO_MARKET = {v: k for k, v in MARKET_CODE.items()}


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


def load_thursday_fixtures():
    """
    Accepts either:
      - { fixtures:[...], window:{...}, ... }
      - { status:"ok", report:{ fixtures:[...], window:{...}, ... } }
      - { report:{ report:{ fixtures:[...] } } }  (legacy)
    """
    data = load_json(THURSDAY_REPORT_PATH)
    if isinstance(data, dict) and "fixtures" in data:
        return data["fixtures"], data
    if isinstance(data, dict) and isinstance(data.get("report"), dict):
        rep = data["report"]
        if "fixtures" in rep:
            return rep["fixtures"], rep
        if isinstance(rep.get("report"), dict) and "fixtures" in rep["report"]:
            return rep["report"]["fixtures"], rep["report"]
    raise KeyError("fixtures not found in Thursday report")


def odds_match_ok(fx, min_score):
    om = fx.get("odds_match") or {}
    if not om.get("matched"):
        return False
    return (safe_float(om.get("score"), 0.0) or 0.0) >= min_score


def confidence_ok(fx, min_conf):
    c = safe_float((fx.get("flags") or {}).get("confidence"), None)
    if c is None:
        return True
    return c >= min_conf


def pick_sort_dt(p):
    # p has date/time at top-level (string)
    d = p.get("date") or ""
    t = p.get("time") or ""
    return (d, t, p.get("league") or "", p.get("match") or "")


def _system_columns(n: int, r: int) -> int:
    return comb(n, r) if (n > 0 and 0 < r <= n) else 0


def _refund_ratio_worst_case(odds_list, r_min: int, columns: int) -> float:
    """
    Conservative: worst-case payout when exactly r_min hits using the lowest r_min odds.
    For "r/n" system: payout per 1 unit at min_hits is product(lowest r odds).
    Cost per 1 unit is columns.
    ratio = product / columns
    """
    if columns <= 0 or r_min <= 0:
        return 0.0
    o = sorted([float(x) for x in odds_list])[:r_min]
    prod = 1.0
    for v in o:
        prod *= max(1.01, float(v))
    return prod / float(columns)


def _system_unit_for_target(columns: int) -> tuple[float, float]:
    """
    Unit so that total stake is within [25,50] if possible.
    """
    if columns <= 0:
        return 0.0, 0.0
    base = SYS_TARGET_MIN / float(columns)
    unit = max(SYS_UNIT_MIN, min(SYS_UNIT_MAX, base))
    stake = unit * columns
    # If still below min because unit clamped, push toward max target
    if stake < SYS_TARGET_MIN and unit < SYS_UNIT_MAX:
        unit2 = min(SYS_UNIT_MAX, SYS_TARGET_MIN / float(columns))
        unit = max(unit, unit2)
        stake = unit * columns
    # If above max, reduce
    if stake > SYS_TARGET_MAX and unit > SYS_UNIT_MIN:
        unit2 = max(SYS_UNIT_MIN, SYS_TARGET_MAX / float(columns))
        unit = min(unit, unit2)
        stake = unit * columns
    return round(unit, 2), round(stake, 2)


# ------------------------- CORE STAKE LADDER -------------------------
def core_single_stake(odds: float) -> float:
    if 1.60 <= odds <= 1.75:
        return 40.0
    if 1.75 < odds <= 1.90:
        return 30.0
    if 1.90 < odds <= 2.10:
        return 20.0
    return 0.0


# ------------------------- BUILD CANDIDATES FROM THURSDAY -------------------------
def build_pick_candidates(fixtures):
    """
    Creates per-market pick rows ONLY if offered odds exist.
    Keeps date/time from fixture for chronological sorting.
    """
    out = []
    for fx in fixtures:
        d = fx.get("date")
        t = fx.get("time")
        league = fx.get("league")
        match = f'{fx.get("home")} – {fx.get("away")}'
        fid = fx.get("fixture_id")

        # Helper getters
        def odds_for(code):
            if code == "1":
                return safe_float(fx.get("offered_1"), None)
            if code == "X":
                return safe_float(fx.get("offered_x"), None)
            if code == "2":
                return safe_float(fx.get("offered_2"), None)
            if code == "O25":
                return safe_float(fx.get("offered_over_2_5"), None)
            if code == "U25":
                return safe_float(fx.get("offered_under_2_5"), None)
            return None

        def prob_for(code):
            if code == "1":
                return safe_float(fx.get("home_prob"), None)
            if code == "X":
                return safe_float(fx.get("draw_prob"), None)
            if code == "2":
                return safe_float(fx.get("away_prob"), None)
            if code == "O25":
                return safe_float(fx.get("over_2_5_prob"), None)
            if code == "U25":
                return safe_float(fx.get("under_2_5_prob"), None)
            return None

        def ev_for(code):
            if code == "1":
                return safe_float(fx.get("ev_1"), None)
            if code == "X":
                return safe_float(fx.get("ev_x"), None)
            if code == "2":
                return safe_float(fx.get("ev_2"), None)
            if code == "O25":
                return safe_float(fx.get("ev_over"), None)
            if code == "U25":
                return safe_float(fx.get("ev_under"), None)
            return None

        for code in ["1", "2", "X", "O25", "U25"]:
            o = odds_for(code)
            if o is None or o <= 1.0:
                continue

            out.append(
                {
                    "pick_id": f"{fid}:{code}",
                    "fixture_id": fid,
                    "date": d,
                    "time": t,
                    "league": league,
                    "match": match,
                    "market_code": code,
                    "market": CODE_TO_MARKET.get(code, code),
                    "odds": o,
                    "prob": prob_for(code),
                    "ev": ev_for(code),
                    "flags": fx.get("flags") or {},
                    "odds_match": fx.get("odds_match") or {},
                    "fx": fx,
                }
            )
    return out


# ------------------------- COREBET SELECTION -------------------------
def corebet_select(picks, bankroll_core: float):
    # Pools
    singles_pool = []
    low_pool = []

    for p in picks:
        fx = p["fx"]
        flags = p["flags"] or {}
        market = p["market"]
        code = p["market_code"]

        # Allowed in Core: Home/Away/Over/Under (Draw excluded)
        if market not in ("Home", "Away", "Over 2.5", "Under 2.5"):
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

        # Market gates + SHAPE logic
        if market == "Home":
            if evv < CORE_EV_MIN_HOME or pr < CORE_P_MIN_HOME:
                continue
            if not bool(flags.get("home_shape")):
                continue
        elif market == "Away":
            if evv < CORE_EV_MIN_AWAY or pr < CORE_P_MIN_AWAY:
                continue
            if not bool(flags.get("away_shape")):
                continue
        elif market == "Over 2.5":
            if evv < CORE_EV_MIN_OVER or pr < CORE_P_MIN_OVER:
                continue
            # Core over requires over_good_shape AND NOT tight_game
            if not bool(flags.get("over_good_shape")):
                continue
            if bool(flags.get("tight_game")):
                continue
        elif market == "Under 2.5":
            # Core under only if under_elite true
            if evv < CORE_EV_MIN_UNDER or pr < CORE_P_MIN_UNDER:
                continue
            if not bool(flags.get("under_elite")):
                continue

        # Odds classification
        if CORE_SINGLES_MIN_ODDS <= odds <= CORE_SINGLES_MAX_ODDS:
            st = core_single_stake(odds)
            if st <= 0:
                continue
            singles_pool.append({**p, "stake": st, "tag": "core_single"})
        elif CORE_LOW_ODDS_MIN <= odds < CORE_SINGLES_MIN_ODDS:
            low_pool.append({**p, "tag": "core_low_leg"})

    # Sort by EV desc, then confidence desc, then prob desc
    def k(x):
        c = safe_float((x.get("flags") or {}).get("confidence"), 0.0) or 0.0
        return (safe_float(x.get("ev"), -9999.0), c, safe_float(x.get("prob"), 0.0))

    singles_pool.sort(key=k, reverse=True)
    low_pool.sort(key=k, reverse=True)

    # Build singles with constraints:
    singles = []
    used_matches = set()

    max_under = max(0, int(round(CORE_MAX_UNDER_SHARE * float(CORE_MAX_SINGLES))))
    # keep at least 1 under slot possible only if max_under computed 0 but share>0 and max_singles>=5
    if max_under == 0 and CORE_MAX_UNDER_SHARE > 0 and CORE_MAX_SINGLES >= 5:
        max_under = 1

    under_count = 0

    # Ensure minimum 1X2 share within min_singles
    min_1x2_needed = max(0, int(round(CORE_MIN_1X2_SHARE * float(CORE_MIN_SINGLES))))
    one_two_count = 0

    # Pass 1: grab Home/Away first
    for p in singles_pool:
        if len(singles) >= CORE_MAX_SINGLES:
            break
        if p["match"] in used_matches:
            continue
        if p["market"] not in ("Home", "Away"):
            continue
        singles.append(_strip_pick_for_output(p))
        used_matches.add(p["match"])
        one_two_count += 1
        if one_two_count >= min_1x2_needed and len(singles) >= CORE_MIN_SINGLES:
            break

    # Pass 2: fill remaining (respect under cap)
    for p in singles_pool:
        if len(singles) >= CORE_MAX_SINGLES:
            break
        if p["match"] in used_matches:
            continue
        if p["market"] == "Under 2.5" and under_count >= max_under:
            continue
        singles.append(_strip_pick_for_output(p))
        used_matches.add(p["match"])
        if p["market"] == "Under 2.5":
            under_count += 1

    # Ensure at least CORE_MIN_SINGLES if possible
    # (If not enough candidates, we keep what we have; no padding.)

    # Build doubles from low odds legs (<1.60) with partner from singles-range (1.60–2.10)
    doubles = []
    if low_pool and CORE_MAX_DOUBLES > 0:
        partner_candidates = [p for p in singles_pool]  # already gated + has stakes
        # also allow partners from low_pool? no, keep partner from singles range (clean)
        used_double_matches = set()

        for leg1 in low_pool:
            if len(doubles) >= CORE_MAX_DOUBLES:
                break
            for leg2 in partner_candidates:
                if leg2["match"] == leg1["match"]:
                    continue
                if leg1["match"] in used_double_matches or leg2["match"] in used_double_matches:
                    continue
                combo = safe_float(leg1["odds"], 1.0) * safe_float(leg2["odds"], 1.0)
                if not (CORE_DOUBLE_TARGET_MIN <= combo <= CORE_DOUBLE_TARGET_MAX):
                    continue
                doubles.append(
                    {
                        "legs": [
                            _strip_leg_for_double(leg1),
                            _strip_leg_for_double(leg2),
                        ],
                        "combo_odds": round(combo, 2),
                        "stake": round(float(CORE_DOUBLE_STAKE), 2),
                        "tag": "core_double_lowodds",
                    }
                )
                used_double_matches.add(leg1["match"])
                used_double_matches.add(leg2["match"])
                break

    # Sort singles chronologically for output
    singles.sort(key=pick_sort_dt)

    # Sort doubles by earliest leg date/time
    doubles.sort(key=lambda d: min((leg.get("date") or "", leg.get("time") or "") for leg in d.get("legs", []) if isinstance(leg, dict)) or ("", ""))

    core_open = round(sum(safe_float(x.get("stake"), 0.0) or 0.0 for x in singles) + sum(safe_float(d.get("stake"), 0.0) or 0.0 for d in doubles), 2)
    core_after = round(bankroll_core - core_open, 2)

    meta = {
        "bankroll": bankroll_core,
        "exposure_cap_pct": CORE_EXPOSURE_CAP,
        "open": core_open,
        "after_open": core_after,
        "picks_count": len(singles),
        "doubles_count": len(doubles),
        "scale_applied": None,  # NO SCALING
        "composition": {
            "singles_total": len(singles),
            "singles_1x2": sum(1 for x in singles if x.get("market") in ("Home", "Away")),
            "singles_over": sum(1 for x in singles if x.get("market") == "Over 2.5"),
            "singles_under": sum(1 for x in singles if x.get("market") == "Under 2.5"),
        },
    }

    core_double = doubles[0] if doubles else None
    return singles, core_double, doubles, meta


def _strip_pick_for_output(p):
    fx = p.get("fx") or {}
    return {
        "pick_id": p.get("pick_id"),
        "fixture_id": p.get("fixture_id"),
        "date": p.get("date"),
        "time": p.get("time"),
        "league": p.get("league"),
        "match": p.get("match"),
        "market": p.get("market"),
        "market_code": p.get("market_code"),
        "odds": round(float(p.get("odds") or 0.0), 3),
        "prob": p.get("prob"),
        "ev": p.get("ev"),
        "stake": round(float(p.get("stake") or 0.0), 2),
        "tag": p.get("tag", "core_pick"),
        # keep minimal provenance
        "flags": p.get("flags") or {},
        "odds_match": p.get("odds_match") or {},
    }


def _strip_leg_for_double(p):
    return {
        "pick_id": p.get("pick_id"),
        "fixture_id": p.get("fixture_id"),
        "date": p.get("date"),
        "time": p.get("time"),
        "league": p.get("league"),
        "match": p.get("match"),
        "market": p.get("market"),
        "market_code": p.get("market_code"),
        "odds": round(float(p.get("odds") or 0.0), 3),
    }


# ------------------------- FUNBET (SYSTEM ONLY) -------------------------
def funbet_select(picks, bankroll_fun: float, core_fixture_ids: set[int]):
    candidates = []
    for p in picks:
        fx = p["fx"]
        flags = p["flags"] or {}
        market = p["market"]

        if market not in ("Home", "Away", "Over 2.5", "Under 2.5"):
            continue

        if FUN_AVOID_CORE_OVERLAP and p["fixture_id"] in core_fixture_ids:
            continue

        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_FUN):
            continue
        if not confidence_ok(fx, FUN_MIN_CONFIDENCE):
            continue

        odds = safe_float(p.get("odds"), None)
        evv = safe_float(p.get("ev"), None)
        pr = safe_float(p.get("prob"), None)
        if odds is None or evv is None or pr is None:
            continue

        if not (FUN_ODDS_MIN <= odds <= FUN_ODDS_MAX):
            continue

        # Market thresholds
        if market == "Home":
            if evv < FUN_EV_MIN or pr < FUN_P_MIN_HOME:
                continue
            # shape optional in fun; allow without home_shape
        elif market == "Away":
            if evv < FUN_EV_MIN or pr < FUN_P_MIN_AWAY:
                continue
        elif market == "Over 2.5":
            if evv < FUN_EV_MIN or pr < FUN_P_MIN_OVER:
                continue
            # Fun: allow tight_game; require over_good_shape
            if not bool(flags.get("over_good_shape")):
                continue
        elif market == "Under 2.5":
            if evv < max(FUN_EV_MIN, 0.10) or pr < FUN_P_MIN_UNDER:
                continue
            # Fun: require under_elite (still strict)
            if not bool(flags.get("under_elite")):
                continue

        candidates.append(p)

    # sort by EV desc then confidence then prob
    def k(x):
        c = safe_float((x.get("flags") or {}).get("confidence"), 0.0) or 0.0
        return (safe_float(x.get("ev"), -9999.0), c, safe_float(x.get("prob"), 0.0))

    candidates.sort(key=k, reverse=True)

    # pick 5-7 unique matches
    picks_out = []
    used = set()
    for p in candidates:
        if p["match"] in used:
            continue
        picks_out.append(p)
        used.add(p["match"])
        if len(picks_out) >= FUN_PICKS_MAX:
            break

    # ensure minimum (if insufficient, keep what exists)
    pool = picks_out[: max(FUN_PICKS_MIN, min(len(picks_out), FUN_PICKS_MAX))]
    if len(pool) > FUN_PICKS_MAX:
        pool = pool[:FUN_PICKS_MAX]

    # Choose system type preferring lower min_hits
    sys_choice = _choose_fun_system(pool)
    if sys_choice is None:
        system = {
            "label": None,
            "columns": 0,
            "min_hits": None,
            "refund_ratio_min_hits": None,
            "refund_threshold_used": SYS_REFUND_MIN,
            "has_system": False,
            "unit": 0.0,
            "stake": 0.0,
        }
        open_amount = 0.0
    else:
        unit, stake = _system_unit_for_target(sys_choice["columns"])
        system = {
            "label": sys_choice["label"],
            "columns": int(sys_choice["columns"]),
            "min_hits": int(sys_choice["min_hits"]),
            "refund_ratio_min_hits": float(sys_choice["refund_ratio_min_hits"]),
            "refund_threshold_used": SYS_REFUND_MIN,
            "has_system": True,
            "unit": unit,
            "stake": stake,
        }
        open_amount = stake

    # Output sorted chronologically
    pool_sorted = sorted([_strip_pick_for_system(p) for p in pool], key=pick_sort_dt)
    picks_total_sorted = sorted([_strip_pick_for_system(p) for p in picks_out], key=pick_sort_dt)

    fun_after = round(bankroll_fun - open_amount, 2)

    payload = {
        "portfolio": "FunBet",
        "bankroll": bankroll_fun,
        "bankroll_start": bankroll_fun,
        "bankroll_source": "history_or_default",  # overwritten in main
        "exposure_cap_pct": FUN_EXPOSURE_CAP,
        "rules": {
            "odds_range": [FUN_ODDS_MIN, FUN_ODDS_MAX],
            "picks_range": [FUN_PICKS_MIN, FUN_PICKS_MAX],
            "refund_min": SYS_REFUND_MIN,
            "system_preference": "prefer lower min_hits (4/7 over 5/7, 3/6 over 4/6)",
            "tight_game_allowed": True,
        },
        "picks_total": picks_total_sorted,
        "system_pool": pool_sorted,
        "system": system,
        "open": round(open_amount, 2),
        "after_open": fun_after,
        "counts": {"picks_total": len(picks_total_sorted), "system_pool": len(pool_sorted)},
    }
    return payload


def _strip_pick_for_system(p):
    return {
        "pick_id": p.get("pick_id"),
        "fixture_id": p.get("fixture_id"),
        "date": p.get("date"),
        "time": p.get("time"),
        "league": p.get("league"),
        "match": p.get("match"),
        "market": p.get("market"),
        "market_code": p.get("market_code"),
        "odds": round(float(p.get("odds") or 0.0), 3),
        "prob": p.get("prob"),
        "ev": p.get("ev"),
        "confidence": safe_float((p.get("flags") or {}).get("confidence"), None),
    }


def _choose_fun_system(pool):
    """
    Επιλέγει σύστημα ΜΟΝΟ αν refund_ratio_min_hits >= 0.65.
    Προτεραιότητα: χαμηλότερο min_hits.
      n=7: try 4/7 then 5/7
      n=6: try 3/6 then 4/6
      n=5: try 3/5
    """
    n = len(pool)
    if n < FUN_PICKS_MIN:
        return None

    # always use exactly n picks as pool
    odds_list = [safe_float(p.get("odds"), None) for p in pool]
    if any(o is None for o in odds_list):
        return None

    trials = []
    if n == 7:
        trials = [("4/7", 4), ("5/7", 5)]
    elif n == 6:
        trials = [("3/6", 3), ("4/6", 4)]
    elif n == 5:
        trials = [("3/5", 3)]
    else:
        # if 5< n <7, clamp to 7/6/5 by trimming best EV already done outside
        return None

    for label, rmin in trials:
        cols = _system_columns(n, rmin)
        rr = _refund_ratio_worst_case(odds_list, rmin, cols)
        if rr >= SYS_REFUND_MIN:
            return {"label": label, "min_hits": rmin, "columns": cols, "refund_ratio_min_hits": round(rr, 4)}

    return None


# ------------------------- DRAWBET (SYSTEM ONLY) -------------------------
def drawbet_select(picks, bankroll_draw: float):
    candidates = []
    for p in picks:
        fx = p["fx"]
        flags = p["flags"] or {}

        if p["market"] != "Draw":
            continue

        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_DRAW):
            continue
        if not confidence_ok(fx, DRAW_MIN_CONFIDENCE):
            continue

        odds = safe_float(p.get("odds"), None)
        evv = safe_float(p.get("ev"), None)
        pr = safe_float(p.get("prob"), None)
        if odds is None or evv is None or pr is None:
            continue

        if not (DRAW_ODDS_MIN <= odds <= DRAW_ODDS_MAX):
            continue
        if evv < DRAW_EV_MIN or pr < DRAW_P_MIN:
            continue

        # draw_shape βοηθάει (αν υπάρχει)
        if flags.get("draw_shape") is False:
            continue

        candidates.append(p)

    def k(x):
        c = safe_float((x.get("flags") or {}).get("confidence"), 0.0) or 0.0
        return (safe_float(x.get("ev"), -9999.0), c, safe_float(x.get("prob"), 0.0))

    candidates.sort(key=k, reverse=True)

    pool = []
    used = set()
    for p in candidates:
        if p["match"] in used:
            continue
        pool.append(p)
        used.add(p["match"])
        if len(pool) >= DRAW_PICKS_MAX:
            break

    # Allow 2..5
    n = len(pool)
    if n < DRAW_PICKS_MIN:
        system = {"label": None, "columns": 0, "unit": 0.0, "stake": 0.0, "has_system": False}
        open_amount = 0.0
        pool_out = []
    else:
        label, sizes = _draw_system_label_and_sizes(n)
        columns = sum(comb(n, r) for r in sizes)
        unit, stake = _system_unit_for_target(columns)
        system = {"label": label, "columns": columns, "unit": unit, "stake": stake, "has_system": True}
        open_amount = stake
        pool_out = sorted([_strip_pick_for_system(p) for p in pool], key=pick_sort_dt)

    draw_after = round(bankroll_draw - open_amount, 2)

    payload = {
        "portfolio": "DrawBet",
        "bankroll": bankroll_draw,
        "bankroll_start": bankroll_draw,
        "bankroll_source": "history_or_default",  # overwritten in main
        "exposure_cap_pct": DRAW_EXPOSURE_CAP,
        "rules": {
            "odds_range": [DRAW_ODDS_MIN, DRAW_ODDS_MAX],
            "picks_range": [DRAW_PICKS_MIN, DRAW_PICKS_MAX],
            "system_dynamic": True,
            "target_total_range": [SYS_TARGET_MIN, SYS_TARGET_MAX],
        },
        "picks_total": pool_out,
        "system_pool": pool_out,
        "system": system,
        "open": round(open_amount, 2),
        "after_open": draw_after,
        "counts": {"picks_total": len(pool_out), "system_pool": len(pool_out)},
    }
    return payload


def _draw_system_label_and_sizes(n: int):
    if n >= 5:
        return "2-3/5", [2, 3]
    if n == 4:
        return "2-3/4", [2, 3]
    if n == 3:
        return "2/3", [2]
    if n == 2:
        return "2/2", [2]
    return None, []


# ------------------------- MAIN -------------------------
def main():
    fixtures, th_meta = load_thursday_fixtures()
    history = load_history()

    window = (th_meta.get("window") or {}) if isinstance(th_meta, dict) else {}
    wf = get_week_fields(window, history)

    # bankroll start from history if present else defaults
    core_start = safe_float(history.get("core", {}).get("bankroll_current"), None)
    fun_start = safe_float(history.get("funbet", {}).get("bankroll_current"), None)
    draw_start = safe_float(history.get("drawbet", {}).get("bankroll_current"), None)

    core_bankroll_start = core_start if core_start is not None else DEFAULT_BANKROLL_CORE
    fun_bankroll_start = fun_start if fun_start is not None else DEFAULT_BANKROLL_FUN
    draw_bankroll_start = draw_start if draw_start is not None else DEFAULT_BANKROLL_DRAW

    picks = build_pick_candidates(fixtures)

    # CORE
    core_singles, core_double, core_doubles, core_meta = corebet_select(picks, core_bankroll_start)

    # FUN
    core_fixture_ids = {safe_int(x.get("fixture_id"), None) for x in core_singles if x.get("fixture_id") is not None}
    core_fixture_ids = {x for x in core_fixture_ids if x is not None}
    funbet = funbet_select(picks, fun_bankroll_start, core_fixture_ids)
    funbet["bankroll_start"] = core_bankroll_start if False else fun_bankroll_start  # explicit

    # DRAW
    drawbet = drawbet_select(picks, draw_bankroll_start)
    drawbet["bankroll_start"] = draw_bankroll_start

    # Ensure bankroll_source fields
    core_source = "history" if core_start is not None else "default"
    fun_source = "history" if fun_start is not None else "default"
    draw_source = "history" if draw_start is not None else "default"

    # Construct report
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "week_id": wf["week_id"],
        "week_no": wf["week_no"],
        "week_label": wf["week_label"],
        "window": window,
        "fixtures_total": th_meta.get("fixtures_total", len(fixtures)),
        "engine_leagues": th_meta.get("engine_leagues") or th_meta.get("engine_leagues") or None,

        "core": {
            "portfolio": "CoreBet",
            "bankroll": DEFAULT_BANKROLL_CORE,
            "bankroll_start": core_bankroll_start,
            "bankroll_source": core_source,
            "exposure_cap_pct": CORE_EXPOSURE_CAP,
            "rules": {
                "singles_odds_range": [CORE_SINGLES_MIN_ODDS, CORE_SINGLES_MAX_ODDS],
                "low_odds_to_doubles": [CORE_LOW_ODDS_MIN, CORE_LOW_ODDS_MAX],
                "stake_ladder": {"1.60-1.75": 40, "1.75-1.90": 30, "1.90-2.10": 20},
                "no_scaling": True,
                "under_core_only_under_elite": True,
                "core_over_blocks_tight_game": True,
                "max_under_share": CORE_MAX_UNDER_SHARE,
                "min_1x2_share": CORE_MIN_1X2_SHARE,
            },
            "singles": core_singles,
            "double": core_double,
            "doubles": core_doubles,
            "open": core_meta["open"],
            "after_open": core_meta["after_open"],
            "picks_count": core_meta["picks_count"],
            "doubles_count": core_meta["doubles_count"],
            "scale_applied": core_meta["scale_applied"],
            "composition": core_meta["composition"],
        },

        "funbet": {
            **funbet,
            "bankroll": DEFAULT_BANKROLL_FUN,
            "bankroll_start": fun_bankroll_start,
            "bankroll_source": fun_source,
        },

        "drawbet": {
            **drawbet,
            "bankroll": DEFAULT_BANKROLL_DRAW,
            "bankroll_start": draw_bankroll_start,
            "bankroll_source": draw_source,
        },
    }

    os.makedirs(os.path.dirname(FRIDAY_REPORT_PATH) or ".", exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
