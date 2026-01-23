# ============================================================
# src/analysis/friday_shortlist_v3.py
# FRIDAY SHORTLIST v3.30 — PRODUCTION (CoreBet + FunBet + DrawBet)
#
# KEY RULES (locked):
#  COREBET:
#   - Singles odds: 1.60–2.10 (no singles <1.60)
#   - Singles stakes ladder:
#       1.60–1.75 => 40
#       1.75–1.90 => 30
#       1.90–2.10 => 20
#   - Low-odds (1.30–1.60) ONLY as DOUBLE legs
#   - Under max share: 20% of core singles AND only if flags.under_elite == true
#   - Max singles: 8
#   - No stake scaling (stays exact ladder + fixed double stake)
#
#  FUNBET (SYSTEM ONLY):
#   - Picks odds: 1.90–3.60 (hard cap)
#   - Pool size: 5–7
#   - Prefer lower r: 7->4/7 (else 5/7), 6->3/6 (else 4/6), 5->3/5
#   - Refund threshold primary: 0.65 (worst-case ratio @min_hits)
#   - Unit: if columns in [25,50] => 1.00, else adjust to keep stake within [25,50]
#   - Anti-ghost brakes:
#       * For Home/Away odds >3.00 => require confidence_band=high AND proper shape flag
#       * Max odds>3.00 picks: (n=7 -> 2), (n=6 -> 1), (n<=5 -> 1)
#       * Max 1X2 (Home/Away) picks: (n=7 -> 3), (n=6 -> 3), (n<=5 -> 2)
#       * In over_friendly leagues, 1X2 needs stricter EV + confidence
#
#  DRAWBET (SYSTEM ONLY):
#   - Finds 2–5 draws (X) within odds range 2.80–3.70 (hard cap)
#   - Systems:
#       n=5 => 2-3/5
#       n=4 => 2/4
#       n=3 => 2/3
#       n=2 => 2/2
#   - Budget: 25–50
#
#  HISTORY:
#   - Optional logs/tuesday_history_v3.json for bankroll carry + week numbering
#
# OUTPUT:
#   - logs/friday_shortlist_v3.json
#   - includes: copy_play + blog_preview (ready for Custom GPT)
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

# Optional carry
USE_HISTORY_CARRY = os.getenv("USE_HISTORY_CARRY", "true").lower() == "true"

# ------------------------- ODDS MATCH / CONF GATES -------------------------
ODDS_MATCH_MIN_SCORE_CORE = float(os.getenv("ODDS_MATCH_MIN_SCORE_CORE", "0.75"))
ODDS_MATCH_MIN_SCORE_FUN  = float(os.getenv("ODDS_MATCH_MIN_SCORE_FUN",  "0.75"))
ODDS_MATCH_MIN_SCORE_DRAW = float(os.getenv("ODDS_MATCH_MIN_SCORE_DRAW", "0.80"))

CORE_MIN_CONFIDENCE = float(os.getenv("CORE_MIN_CONFIDENCE", "0.55"))
FUN_MIN_CONFIDENCE  = float(os.getenv("FUN_MIN_CONFIDENCE",  "0.55"))
DRAW_MIN_CONFIDENCE = float(os.getenv("DRAW_MIN_CONFIDENCE", "0.60"))

# Overlap toggle
FUN_ALLOW_CORE_OVERLAP = os.getenv("FUN_ALLOW_CORE_OVERLAP", "true").lower() == "true"

# ------------------------- MARKETS -------------------------
CODE_TO_MARKET = {"1":"Home","2":"Away","X":"Draw","O25":"Over 2.5","U25":"Under 2.5"}
MARKET_TO_CODE = {v:k for k,v in CODE_TO_MARKET.items()}

CORE_ALLOWED = {"Home","Away","Over 2.5","Under 2.5"}
FUN_ALLOWED  = {"Home","Away","Over 2.5","Under 2.5"}    # Draw handled by DrawBet
DRAW_ALLOWED = {"Draw"}

# ------------------------- COREBET SETTINGS -------------------------
CORE_MIN_ODDS = 1.60
CORE_MAX_ODDS = 2.10

CORE_LOW_MIN_ODDS = 1.30
CORE_LOW_MAX_ODDS = 1.60  # exclusive for singles, only doubles

CORE_MAX_SINGLES = 8
CORE_MAX_DOUBLES = 2

CORE_UNDER_MAX_SHARE = 0.20  # max share of under in core singles
CORE_DOUBLE_TARGET_MIN = 2.10
CORE_DOUBLE_TARGET_MAX = 3.50
CORE_DOUBLE_STAKE = 15.0

def core_single_stake(odds: float) -> float:
    if 1.60 <= odds <= 1.75:
        return 40.0
    if 1.75 < odds <= 1.90:
        return 30.0
    if 1.90 < odds <= 2.10:
        return 20.0
    return 0.0

# Core gates (tighten low-odds legs + under)
CORE_EV_MIN_HOME  = float(os.getenv("CORE_EV_MIN_HOME",  "0.04"))
CORE_EV_MIN_AWAY  = float(os.getenv("CORE_EV_MIN_AWAY",  "0.05"))
CORE_EV_MIN_OVER  = float(os.getenv("CORE_EV_MIN_OVER",  "0.04"))
CORE_EV_MIN_UNDER = float(os.getenv("CORE_EV_MIN_UNDER", "0.08"))

CORE_P_MIN_HOME  = float(os.getenv("CORE_P_MIN_HOME",  "0.30"))
CORE_P_MIN_AWAY  = float(os.getenv("CORE_P_MIN_AWAY",  "0.24"))
CORE_P_MIN_OVER  = float(os.getenv("CORE_P_MIN_OVER",  "0.45"))
CORE_P_MIN_UNDER = float(os.getenv("CORE_P_MIN_UNDER", "0.58"))

# Low-odds legs must be "sure"
CORE_LOWLEG_MIN_CONF = float(os.getenv("CORE_LOWLEG_MIN_CONF", "0.70"))
CORE_LOWLEG_MIN_EV   = float(os.getenv("CORE_LOWLEG_MIN_EV",   "0.02"))

# ------------------------- FUNBET SETTINGS -------------------------
FUN_MIN_ODDS = 1.90
FUN_MAX_ODDS = 3.60

FUN_POOL_MIN = 5
FUN_POOL_MAX = 7

FUN_REFUND_MIN = float(os.getenv("FUN_REFUND_MIN", "0.65"))
FUN_BUDGET_MIN = float(os.getenv("FUN_BUDGET_MIN", "25.0"))
FUN_BUDGET_MAX = float(os.getenv("FUN_BUDGET_MAX", "50.0"))

# Anti-ghost quotas
FUN_MAX_GT3_FOR_7 = 2
FUN_MAX_GT3_FOR_6 = 1
FUN_MAX_GT3_FOR_5 = 1

FUN_MAX_1X2_FOR_7 = 3
FUN_MAX_1X2_FOR_6 = 3
FUN_MAX_1X2_FOR_5 = 2

# Fun gates
FUN_EV_MIN_HOME  = float(os.getenv("FUN_EV_MIN_HOME",  "0.05"))
FUN_EV_MIN_AWAY  = float(os.getenv("FUN_EV_MIN_AWAY",  "0.05"))
FUN_EV_MIN_OVER  = float(os.getenv("FUN_EV_MIN_OVER",  "0.05"))
FUN_EV_MIN_UNDER = float(os.getenv("FUN_EV_MIN_UNDER", "0.10"))

FUN_P_MIN_HOME  = float(os.getenv("FUN_P_MIN_HOME",  "0.28"))
FUN_P_MIN_AWAY  = float(os.getenv("FUN_P_MIN_AWAY",  "0.22"))
FUN_P_MIN_OVER  = float(os.getenv("FUN_P_MIN_OVER",  "0.42"))
FUN_P_MIN_UNDER = float(os.getenv("FUN_P_MIN_UNDER", "0.60"))

# Under in fun only if under_elite
FUN_REQUIRE_UNDER_ELITE = os.getenv("FUN_REQUIRE_UNDER_ELITE", "true").lower() == "true"

# League sanity for 1X2 (over-friendly leagues stricter)
OVER_FRIENDLY_1X2_CONF_MIN = float(os.getenv("OVER_FRIENDLY_1X2_CONF_MIN", "0.75"))
OVER_FRIENDLY_1X2_EV_BONUS = float(os.getenv("OVER_FRIENDLY_1X2_EV_BONUS", "0.02"))

# ------------------------- DRAWBET SETTINGS -------------------------
DRAW_MIN_ODDS = 2.80
DRAW_MAX_ODDS = 3.70

DRAW_MIN_P = float(os.getenv("DRAW_MIN_P", "0.28"))
DRAW_MIN_EV = float(os.getenv("DRAW_MIN_EV", "0.05"))

DRAW_BUDGET_MIN = float(os.getenv("DRAW_BUDGET_MIN", "25.0"))
DRAW_BUDGET_MAX = float(os.getenv("DRAW_BUDGET_MAX", "50.0"))

# ------------------------- UTILS -------------------------
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

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_history(path: str):
    if not USE_HISTORY_CARRY or (not os.path.exists(path)):
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}

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
    weeks = (history.get("weeks") or {}) if isinstance(history, dict) else {}
    if window_from and window_from in weeks:
        week_no = int((weeks[window_from] or {}).get("week_no") or 1)
    else:
        week_no = int(history.get("week_count") or 0) + 1 if isinstance(history, dict) else 1
    return {"week_id": week_id, "week_no": week_no, "week_label": f"Week {week_no}"}

def odds_match_ok(fx, min_score):
    om = fx.get("odds_match") or {}
    if not om.get("matched"):
        return False
    return (safe_float(om.get("score"), 0.0) or 0.0) >= min_score

def confidence_val(fx):
    return safe_float((fx.get("flags") or {}).get("confidence"), None)

def confidence_band(fx):
    return (fx.get("flags") or {}).get("confidence_band")

def confidence_ok(fx, min_conf):
    c = confidence_val(fx)
    if c is None:
        return True
    return c >= min_conf

def is_over_friendly(fx):
    return bool((fx.get("flags") or {}).get("over_friendly_league") is True)

def shape_ok_for_1x2(fx, market_code: str):
    flags = fx.get("flags") or {}
    if market_code == "1":
        return bool(flags.get("home_shape") is True)
    if market_code == "2":
        return bool(flags.get("away_shape") is True)
    return False

def ev_from_fx(fx, code: str):
    c = (code or "").upper()
    if c == "1":   return safe_float(fx.get("ev_1"), None)
    if c == "2":   return safe_float(fx.get("ev_2"), None)
    if c == "X":   return safe_float(fx.get("ev_x"), None)
    if c == "O25": return safe_float(fx.get("ev_over"), None)
    if c == "U25": return safe_float(fx.get("ev_under"), None)
    return None

def p_from_fx(fx, code: str):
    c = (code or "").upper()
    if c == "1":   return safe_float(fx.get("home_prob"), None)
    if c == "2":   return safe_float(fx.get("away_prob"), None)
    if c == "X":   return safe_float(fx.get("draw_prob"), None)
    if c == "O25": return safe_float(fx.get("over_2_5_prob"), None)
    if c == "U25": return safe_float(fx.get("under_2_5_prob"), None)
    return None

def odds_from_fx(fx, code: str):
    c = (code or "").upper()
    if c == "1":   return safe_float(fx.get("offered_1"), None)
    if c == "2":   return safe_float(fx.get("offered_2"), None)
    if c == "X":   return safe_float(fx.get("offered_x"), None)
    if c == "O25": return safe_float(fx.get("offered_over_2_5"), None)
    if c == "U25": return safe_float(fx.get("offered_under_2_5"), None)
    return None

def parse_dt_key(fx):
    # best-effort sortable key: date + time string
    d = fx.get("date") or ""
    t = fx.get("time") or "00:00"
    return f"{d}T{t}"

def load_thursday_report():
    data = load_json(THURSDAY_REPORT_PATH)
    # robust: accept {fixtures:...} OR {report:{fixtures:...}} OR {status,report}
    if isinstance(data, dict) and "fixtures" in data:
        return data, data.get("fixtures") or []
    if isinstance(data, dict) and isinstance(data.get("report"), dict) and "fixtures" in data["report"]:
        return data["report"], data["report"].get("fixtures") or []
    raise KeyError("Could not find fixtures in thursday report")

# ------------------------- BUILD PICK UNIVERSE -------------------------
def build_candidates(fixtures):
    out = []
    for fx in fixtures:
        if not isinstance(fx, dict):
            continue
        for code in ["1","2","X","O25","U25"]:
            od = odds_from_fx(fx, code)
            if od is None or od <= 1.0:
                continue
            out.append({
                "fixture_id": fx.get("fixture_id"),
                "pick_id": f'{fx.get("fixture_id")}:{code}',
                "league": fx.get("league"),
                "date": fx.get("date"),
                "time": fx.get("time"),
                "dt_key": parse_dt_key(fx),
                "home": fx.get("home"),
                "away": fx.get("away"),
                "match": f'{fx.get("home")} – {fx.get("away")}',
                "market_code": code,
                "market": CODE_TO_MARKET.get(code, code),
                "odds": od,
                "prob": p_from_fx(fx, code),
                "ev": ev_from_fx(fx, code),
                "confidence": confidence_val(fx),
                "confidence_band": confidence_band(fx),
                "over_friendly_league": is_over_friendly(fx),
                "flags": fx.get("flags") or {},
                "fx": fx,
            })
    return out

def sort_key_quality(p):
    # primary: EV, then confidence, then prob; tie-break prefers LOWER odds when close
    return (safe_float(p.get("ev"), -9999.0),
            safe_float(p.get("confidence"), 0.0),
            safe_float(p.get("prob"), 0.0),
            -safe_float(p.get("odds"), 99.0))

def tie_break_lower_odds_if_close(a, b, ev_eps=0.02):
    ea, eb = safe_float(a.get("ev"), None), safe_float(b.get("ev"), None)
    ca, cb = safe_float(a.get("confidence"), None), safe_float(b.get("confidence"), None)
    if ea is None or eb is None:
        return False
    if abs(ea - eb) <= ev_eps:
        # close EV, prefer lower odds
        oa, ob = safe_float(a.get("odds"), 99.0), safe_float(b.get("odds"), 99.0)
        return oa < ob
    # otherwise prefer higher EV (default)
    return ea > eb

# ------------------------- COREBET -------------------------
def core_pass_gates(p):
    fx = p["fx"]
    if p["market"] not in CORE_ALLOWED:
        return False
    if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_CORE):
        return False
    if not confidence_ok(fx, CORE_MIN_CONFIDENCE):
        return False

    od = safe_float(p.get("odds"), None)
    evv = safe_float(p.get("ev"), None)
    pr  = safe_float(p.get("prob"), None)
    if od is None or evv is None or pr is None:
        return False

    # Under only if under_elite
    if p["market"] == "Under 2.5":
        if (fx.get("flags") or {}).get("under_elite") is not True:
            return False
        if evv < CORE_EV_MIN_UNDER or pr < CORE_P_MIN_UNDER:
            return False

    if p["market"] == "Home":
        if evv < CORE_EV_MIN_HOME or pr < CORE_P_MIN_HOME:
            return False
    if p["market"] == "Away":
        if evv < CORE_EV_MIN_AWAY or pr < CORE_P_MIN_AWAY:
            return False
    if p["market"] == "Over 2.5":
        if evv < CORE_EV_MIN_OVER or pr < CORE_P_MIN_OVER:
            return False

    return True

def core_select(cands, bankroll_core):
    singles_pool = []
    low_pool = []

    for p in cands:
        if not core_pass_gates(p):
            continue
        od = float(p["odds"])
        if CORE_MIN_ODDS <= od <= CORE_MAX_ODDS:
            st = core_single_stake(od)
            if st > 0:
                singles_pool.append({**p, "stake": st, "tag": "core_single"})
        elif CORE_LOW_MIN_ODDS <= od < CORE_LOW_MAX_ODDS:
            # low odds only for doubles; tighten
            if (safe_float(p.get("confidence"), 0.0) or 0.0) < CORE_LOWLEG_MIN_CONF:
                continue
            if (safe_float(p.get("ev"), -9.0) or -9.0) < CORE_LOWLEG_MIN_EV:
                continue
            low_pool.append({**p, "tag": "core_low_leg"})

    singles_pool.sort(key=sort_key_quality, reverse=True)
    low_pool.sort(key=sort_key_quality, reverse=True)

    singles = []
    used_matches = set()
    max_under = max(1, int(round(CORE_UNDER_MAX_SHARE * CORE_MAX_SINGLES)))
    under_used = 0

    for p in singles_pool:
        if len(singles) >= CORE_MAX_SINGLES:
            break
        if p["match"] in used_matches:
            continue
        if p["market"] == "Under 2.5" and under_used >= max_under:
            continue
        singles.append({
            "pick_id": p["pick_id"],
            "fixture_id": p["fixture_id"],
            "league": p["league"],
            "date": p["date"],
            "time": p["time"],
            "dt_key": p["dt_key"],
            "match": p["match"],
            "market": p["market"],
            "market_code": p["market_code"],
            "odds": round(float(p["odds"]), 3),
            "prob": p.get("prob"),
            "ev": p.get("ev"),
            "stake": round(float(p["stake"]), 2),
            "tag": "core",
        })
        used_matches.add(p["match"])
        if p["market"] == "Under 2.5":
            under_used += 1

    # Build doubles (low odds + partner from singles_pool or remaining)
    doubles = []
    if low_pool and CORE_MAX_DOUBLES > 0:
        partners = [p for p in singles_pool if p["match"] not in used_matches]
        partners += [p for p in singles_pool if p["match"] in used_matches]  # allow overlap if needed
        partners.sort(key=sort_key_quality, reverse=True)

        used_double_matches = set()
        for leg1 in low_pool:
            if len(doubles) >= CORE_MAX_DOUBLES:
                break
            for leg2 in partners:
                if leg2["match"] == leg1["match"]:
                    continue
                if leg1["match"] in used_double_matches or leg2["match"] in used_double_matches:
                    continue
                combo = float(leg1["odds"]) * float(leg2["odds"])
                if not (CORE_DOUBLE_TARGET_MIN <= combo <= CORE_DOUBLE_TARGET_MAX):
                    continue
                doubles.append({
                    "legs": [
                        {"pick_id": leg1["pick_id"], "fixture_id": leg1["fixture_id"], "match": leg1["match"], "league": leg1["league"], "date": leg1["date"], "time": leg1["time"], "market": leg1["market"], "market_code": leg1["market_code"], "odds": round(float(leg1["odds"]),3)},
                        {"pick_id": leg2["pick_id"], "fixture_id": leg2["fixture_id"], "match": leg2["match"], "league": leg2["league"], "date": leg2["date"], "time": leg2["time"], "market": leg2["market"], "market_code": leg2["market_code"], "odds": round(float(leg2["odds"]),3)},
                    ],
                    "combo_odds": round(combo, 2),
                    "stake": round(float(CORE_DOUBLE_STAKE), 2),
                    "tag": "core_double",
                })
                used_double_matches.add(leg1["match"])
                used_double_matches.add(leg2["match"])
                break

    # No scaling, compute open
    open_total = round(sum(x["stake"] for x in singles) + sum(d["stake"] for d in doubles), 2)
    after_open = round(bankroll_core - open_total, 2)

    # Sort chronologically for output
    singles.sort(key=lambda x: (x.get("date") or "", x.get("time") or "00:00"))
    for d in doubles:
        d["legs"].sort(key=lambda x: (x.get("date") or "", x.get("time") or "00:00"))

    core_double = doubles[0] if doubles else None
    return singles, core_double, doubles, {"open": open_total, "after_open": after_open}

# ------------------------- FUNBET -------------------------
def fun_pass_gates(p, core_fixture_ids: set):
    fx = p["fx"]
    if p["market"] not in FUN_ALLOWED:
        return False
    if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_FUN):
        return False
    if not confidence_ok(fx, FUN_MIN_CONFIDENCE):
        return False
    if (not FUN_ALLOW_CORE_OVERLAP) and (p["fixture_id"] in core_fixture_ids):
        return False

    od = safe_float(p.get("odds"), None)
    evv = safe_float(p.get("ev"), None)
    pr  = safe_float(p.get("prob"), None)
    if od is None or evv is None or pr is None:
        return False

    # odds band
    if not (FUN_MIN_ODDS <= od <= FUN_MAX_ODDS):
        return False

    # Under only if under_elite (optional)
    if p["market"] == "Under 2.5" and FUN_REQUIRE_UNDER_ELITE:
        if (fx.get("flags") or {}).get("under_elite") is not True:
            return False

    # base thresholds by market
    if p["market"] == "Home":
        if evv < FUN_EV_MIN_HOME or pr < FUN_P_MIN_HOME:
            return False
    if p["market"] == "Away":
        if evv < FUN_EV_MIN_AWAY or pr < FUN_P_MIN_AWAY:
            return False
    if p["market"] == "Over 2.5":
        if evv < FUN_EV_MIN_OVER or pr < FUN_P_MIN_OVER:
            return False
    if p["market"] == "Under 2.5":
        if evv < FUN_EV_MIN_UNDER or pr < FUN_P_MIN_UNDER:
            return False

    # -------- Anti-ghost brake for big 1X2 --------
    if p["market"] in ("Home","Away") and od > 3.00:
        # must be high confidence AND correct shape flag
        cb = p.get("confidence_band") or ""
        if cb.lower() != "high":
            return False
        if not shape_ok_for_1x2(fx, p["market_code"]):
            return False

    # -------- League sanity: over-friendly leagues stricter for 1X2 --------
    if p["market"] in ("Home","Away") and is_over_friendly(fx):
        confv = safe_float(p.get("confidence"), 0.0) or 0.0
        if confv < OVER_FRIENDLY_1X2_CONF_MIN:
            return False
        # require extra EV
        req = (FUN_EV_MIN_HOME if p["market"]=="Home" else FUN_EV_MIN_AWAY) + OVER_FRIENDLY_1X2_EV_BONUS
        if evv < req:
            return False

    return True

def refund_ratio_worst_case(odds_list, r_min, columns):
    if columns <= 0 or r_min <= 0:
        return 0.0
    o = sorted([float(x) for x in odds_list])[:r_min]
    prod = 1.0
    for v in o:
        prod *= max(1.01, float(v))
    return prod / float(columns)

def fun_choose_system(pool):
    n = len(pool)
    if n == 7:
        # prefer 4/7
        cols_4 = comb(7,4)
        rr_4 = refund_ratio_worst_case([p["odds"] for p in pool], 4, cols_4)
        if rr_4 >= FUN_REFUND_MIN:
            return {"label":"4/7","columns":cols_4,"min_hits":4,"refund_ratio":round(rr_4,4)}
        cols_5 = comb(7,5)
        rr_5 = refund_ratio_worst_case([p["odds"] for p in pool], 5, cols_5)
        return {"label":"5/7","columns":cols_5,"min_hits":5,"refund_ratio":round(rr_5,4)}
    if n == 6:
        cols_3 = comb(6,3)
        rr_3 = refund_ratio_worst_case([p["odds"] for p in pool], 3, cols_3)
        if rr_3 >= FUN_REFUND_MIN:
            return {"label":"3/6","columns":cols_3,"min_hits":3,"refund_ratio":round(rr_3,4)}
        cols_4 = comb(6,4)
        rr_4 = refund_ratio_worst_case([p["odds"] for p in pool], 4, cols_4)
        return {"label":"4/6","columns":cols_4,"min_hits":4,"refund_ratio":round(rr_4,4)}
    if n == 5:
        cols_3 = comb(5,3)
        rr_3 = refund_ratio_worst_case([p["odds"] for p in pool], 3, cols_3)
        return {"label":"3/5","columns":cols_3,"min_hits":3,"refund_ratio":round(rr_3,4)}
    return {"label":None,"columns":0,"min_hits":None,"refund_ratio":None}

def fun_unit_for_budget(columns):
    if columns <= 0:
        return 0.0, 0.0
    # if 1€ per column lands inside [25,50], keep it 1€
    if FUN_BUDGET_MIN <= columns <= FUN_BUDGET_MAX:
        return 1.00, round(float(columns), 2)

    # otherwise compute unit to clamp stake into [25,50]
    # unit = target / columns, prefer closer to 1
    target = min(max(columns, FUN_BUDGET_MIN), FUN_BUDGET_MAX)
    unit = round(target / float(columns), 2)
    if unit <= 0:
        unit = 0.50
    stake = round(unit * columns, 2)

    # ensure hard clamp
    if stake < FUN_BUDGET_MIN:
        unit = round(FUN_BUDGET_MIN / float(columns), 2)
        stake = round(unit * columns, 2)
    if stake > FUN_BUDGET_MAX:
        unit = round(FUN_BUDGET_MAX / float(columns), 2)
        stake = round(unit * columns, 2)

    return unit, stake

def fun_select(cands, bankroll_fun, core_fixture_ids: set):
    # build candidates
    pool = []
    used = set()

    candidates = [p for p in cands if fun_pass_gates(p, core_fixture_ids)]
    # primary rank by EV then confidence then prob
    candidates.sort(key=sort_key_quality, reverse=True)

    # enforce quotas while filling
    gt3_limit = FUN_MAX_GT3_FOR_7
    one_two_limit = FUN_MAX_1X2_FOR_7

    def limits_for_n(n):
        if n >= 7:
            return FUN_MAX_GT3_FOR_7, FUN_MAX_1X2_FOR_7
        if n == 6:
            return FUN_MAX_GT3_FOR_6, FUN_MAX_1X2_FOR_6
        return FUN_MAX_GT3_FOR_5, FUN_MAX_1X2_FOR_5

    # we attempt to fill up to 7, then if not possible shrink later
    gt3_count = 0
    one_two_count = 0

    for p in candidates:
        if len(pool) >= FUN_POOL_MAX:
            break
        if p["match"] in used:
            continue

        # dynamic quota based on target max(7) while filling
        od = float(p["odds"])
        if od > 3.00 and gt3_count >= FUN_MAX_GT3_FOR_7:
            continue
        if p["market"] in ("Home","Away") and one_two_count >= FUN_MAX_1X2_FOR_7:
            continue

        pool.append(p)
        used.add(p["match"])
        if od > 3.00:
            gt3_count += 1
        if p["market"] in ("Home","Away"):
            one_two_count += 1

    # if still too small, accept smaller pools (down to 5)
    # also re-apply quotas for final pool size
    for target_n in [7,6,5]:
        if len(pool) >= target_n:
            pool = pool[:target_n]
            break
    if len(pool) < FUN_POOL_MIN:
        # no system this week
        return {
            "bankroll": bankroll_fun,
            "bankroll_start": bankroll_fun,
            "exposure_cap_pct": 0.0,
            "picks_total": [],
            "system_pool": [],
            "system": {"label": None, "columns": 0, "unit": 0.0, "stake": 0.0, "refund_ratio_min_hits": None, "min_hits": None, "has_system": False},
            "open": 0.0,
            "after_open": bankroll_fun,
            "counts": {"picks_total": 0, "system_pool": 0},
            "rules": {"note": "not enough fun picks"},
        }

    # enforce quotas based on final n
    n = len(pool)
    gt3_limit, one_two_limit = limits_for_n(n)
    filtered = []
    gt3_count = 0
    one_two_count = 0
    for p in pool:
        od = float(p["odds"])
        if od > 3.00:
            if gt3_count >= gt3_limit:
                continue
        if p["market"] in ("Home","Away"):
            if one_two_count >= one_two_limit:
                continue
        filtered.append(p)
        if od > 3.00:
            gt3_count += 1
        if p["market"] in ("Home","Away"):
            one_two_count += 1

    # if filtering made it too small, keep original pool
    if len(filtered) >= FUN_POOL_MIN:
        pool = filtered

    # choose system
    sys_choice = fun_choose_system(pool)
    columns = int(sys_choice.get("columns") or 0)
    unit, stake = fun_unit_for_budget(columns)

    open_amount = float(stake)
    after_open = round(bankroll_fun - open_amount, 2)

    # strip output
    picks_out = [{
        "pick_id": p["pick_id"],
        "fixture_id": p["fixture_id"],
        "league": p["league"],
        "date": p["date"],
        "time": p["time"],
        "dt_key": p["dt_key"],
        "match": p["match"],
        "market": p["market"],
        "market_code": p["market_code"],
        "odds": round(float(p["odds"]), 3),
        "prob": p.get("prob"),
        "ev": p.get("ev"),
        "confidence": p.get("confidence"),
    } for p in candidates[:min(len(candidates), 20)]]

    pool_out = [{
        "pick_id": p["pick_id"],
        "fixture_id": p["fixture_id"],
        "league": p["league"],
        "date": p["date"],
        "time": p["time"],
        "dt_key": p["dt_key"],
        "match": p["match"],
        "market": p["market"],
        "market_code": p["market_code"],
        "odds": round(float(p["odds"]), 3),
        "prob": p.get("prob"),
        "ev": p.get("ev"),
        "confidence": p.get("confidence"),
    } for p in pool]

    pool_out.sort(key=lambda x: (x.get("date") or "", x.get("time") or "00:00"))

    return {
        "bankroll": bankroll_fun,
        "bankroll_start": bankroll_fun,
        "exposure_cap_pct": 0.0,
        "rules": {
            "odds_range": [FUN_MIN_ODDS, FUN_MAX_ODDS],
            "refund_min": FUN_REFUND_MIN,
            "budget_range": [FUN_BUDGET_MIN, FUN_BUDGET_MAX],
            "anti_ghost": {
                "home_away_gt3_requires": "high_conf + shape",
                "max_gt3": gt3_limit,
                "max_1x2": one_two_limit
            }
        },
        "picks_total": picks_out,
        "system_pool": pool_out,
        "system": {
            "label": sys_choice.get("label"),
            "columns": columns,
            "min_hits": sys_choice.get("min_hits"),
            "refund_ratio_min_hits": sys_choice.get("refund_ratio"),
            "unit": unit,
            "stake": stake,
            "has_system": bool(columns > 0),
        },
        "open": round(open_amount, 2),
        "after_open": after_open,
        "counts": {"picks_total": len(picks_out), "system_pool": len(pool_out)},
    }

# ------------------------- DRAWBET -------------------------
def draw_pass_gates(p):
    fx = p["fx"]
    if p["market"] != "Draw":
        return False
    if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_DRAW):
        return False
    if not confidence_ok(fx, DRAW_MIN_CONFIDENCE):
        return False
    od = safe_float(p.get("odds"), None)
    evv = safe_float(p.get("ev"), None)
    pr  = safe_float(p.get("prob"), None)
    if od is None or evv is None or pr is None:
        return False
    if not (DRAW_MIN_ODDS <= od <= DRAW_MAX_ODDS):
        return False
    if evv < DRAW_MIN_EV or pr < DRAW_MIN_P:
        return False
    # require draw_shape if exists (soft)
    if (fx.get("flags") or {}).get("draw_shape") is not True:
        return False
    return True

def draw_system_for_n(n: int):
    if n >= 5:
        # 2-3/5
        cols = comb(5,2) + comb(5,3)  # 20
        return {"label":"2-3/5","columns":cols,"pool_size":5}
    if n == 4:
        cols = comb(4,2)  # 6
        return {"label":"2/4","columns":cols,"pool_size":4}
    if n == 3:
        cols = comb(3,2)  # 3
        return {"label":"2/3","columns":cols,"pool_size":3}
    if n == 2:
        cols = comb(2,2)  # 1
        return {"label":"2/2","columns":cols,"pool_size":2}
    return {"label":None,"columns":0,"pool_size":0}

def draw_unit_for_budget(columns: int):
    if columns <= 0:
        return 0.0, 0.0
    # prefer 1 if fits, else compute
    if DRAW_BUDGET_MIN <= columns <= DRAW_BUDGET_MAX:
        return 1.00, round(float(columns), 2)
    target = min(max(columns, DRAW_BUDGET_MIN), DRAW_BUDGET_MAX)
    unit = round(target / float(columns), 2)
    if unit <= 0:
        unit = 1.00
    stake = round(unit * columns, 2)
    # clamp
    if stake < DRAW_BUDGET_MIN:
        unit = round(DRAW_BUDGET_MIN / float(columns), 2)
        stake = round(unit * columns, 2)
    if stake > DRAW_BUDGET_MAX:
        unit = round(DRAW_BUDGET_MAX / float(columns), 2)
        stake = round(unit * columns, 2)
    return unit, stake

def draw_select(cands, bankroll_draw):
    candidates = [p for p in cands if draw_pass_gates(p)]
    candidates.sort(key=sort_key_quality, reverse=True)

    # pick up to 5 unique matches
    pool = []
    used = set()
    for p in candidates:
        if p["match"] in used:
            continue
        pool.append(p)
        used.add(p["match"])
        if len(pool) >= 5:
            break

    n = len(pool)
    sys_choice = draw_system_for_n(n)
    take_n = sys_choice.get("pool_size") or 0
    pool = pool[:take_n] if take_n > 0 else []

    columns = int(sys_choice.get("columns") or 0)
    unit, stake = draw_unit_for_budget(columns)

    open_amount = float(stake)
    after_open = round(bankroll_draw - open_amount, 2)

    pool_out = [{
        "pick_id": p["pick_id"],
        "fixture_id": p["fixture_id"],
        "league": p["league"],
        "date": p["date"],
        "time": p["time"],
        "dt_key": p["dt_key"],
        "match": p["match"],
        "market": "Draw",
        "market_code": "X",
        "odds": round(float(p["odds"]), 3),
        "prob": p.get("prob"),
        "ev": p.get("ev"),
        "confidence": p.get("confidence"),
    } for p in pool]
    pool_out.sort(key=lambda x: (x.get("date") or "", x.get("time") or "00:00"))

    has_system = columns > 0 and len(pool_out) >= 2

    return {
        "bankroll": bankroll_draw,
        "bankroll_start": bankroll_draw,
        "exposure_cap_pct": 0.0,
        "rules": {
            "odds_range": [DRAW_MIN_ODDS, DRAW_MAX_ODDS],
            "budget_range": [DRAW_BUDGET_MIN, DRAW_BUDGET_MAX],
            "requires_draw_shape": True,
        },
        "picks_total": pool_out,
        "system_pool": pool_out,
        "system": {
            "label": sys_choice.get("label") if has_system else None,
            "columns": columns if has_system else 0,
            "unit": unit if has_system else 0.0,
            "stake": stake if has_system else 0.0,
            "has_system": bool(has_system),
        },
        "open": round(open_amount, 2) if has_system else 0.0,
        "after_open": after_open if has_system else bankroll_draw,
        "counts": {"picks_total": len(pool_out), "system_pool": len(pool_out)},
    }

# ------------------------- COPY/BLOG BUILDERS -------------------------
def build_copy_play(report):
    rows = []
    core = report.get("core") or {}
    fun  = report.get("funbet") or {}
    draw = report.get("drawbet") or {}

    for s in (core.get("singles") or []):
        rows.append({
            "date": s.get("date"), "time": s.get("time"), "league": s.get("league"),
            "match": s.get("match"), "market": s.get("market"),
            "odds": s.get("odds"), "stake": s.get("stake"),
            "bucket": "Core"
        })
    for d in (core.get("doubles") or []):
        rows.append({
            "date": None, "time": None, "league": None,
            "match": "Double", "market": d.get("combo_odds"),
            "odds": d.get("combo_odds"), "stake": d.get("stake"),
            "bucket": "CoreDouble"
        })
    # Fun system
    sysf = (fun.get("system") or {})
    if sysf.get("has_system"):
        rows.append({
            "date": None, "time": None, "league": None,
            "match": f'System {sysf.get("label")}', "market": "FunBet",
            "odds": None, "stake": sysf.get("stake"),
            "bucket": "FunSystem"
        })
    # Draw system
    sysd = (draw.get("system") or {})
    if sysd.get("has_system"):
        rows.append({
            "date": None, "time": None, "league": None,
            "match": f'System {sysd.get("label")}', "market": "DrawBet",
            "odds": None, "stake": sysd.get("stake"),
            "bucket": "DrawSystem"
        })

    # stable order: dated first, then systems
    rows.sort(key=lambda x: ((x["date"] or "9999-99-99"), (x["time"] or "99:99"), x["bucket"]))
    return rows

def build_blog_preview(report):
    wk = report.get("week_label") or "—"
    window = report.get("window") or {}
    wfrom = window.get("from") or "—"
    wto   = window.get("to") or "—"

    core = report.get("core") or {}
    fun  = report.get("funbet") or {}
    draw = report.get("drawbet") or {}

    lines = []
    lines.append("[BLOG_PREVIEW]")
    lines.append(f"Week: {wk} | Window: {wfrom} – {wto}")
    lines.append("")
    # CORE
    lines.append("COREBET")
    lines.append("- Singles:")
    for s in (core.get("singles") or []):
        dt = f'{(s.get("date") or "")} {(s.get("time") or "")}'.strip()
        lg = s.get("league") or "—"
        lines.append(f'  • ({dt}) ({lg}) {s.get("match")} — {s.get("market")} — {s.get("odds")} — Stake {s.get("stake")}')
    lines.append("- Doubles:")
    for d in (core.get("doubles") or []):
        legs = d.get("legs") or []
        if len(legs) >= 2:
            l1 = f'{legs[0].get("match")} ({legs[0].get("market")} @{legs[0].get("odds")})'
            l2 = f'{legs[1].get("match")} ({legs[1].get("market")} @{legs[1].get("odds")})'
            lines.append(f'  • {l1} + {l2} — Combo {d.get("combo_odds")} — Stake {d.get("stake")}')
    lines.append("")
    # FUN
    lines.append("FUNBET")
    sysf = fun.get("system") or {}
    lines.append(f'- System: {sysf.get("label") or "—"} | Columns: {sysf.get("columns") or "—"} | Unit: {sysf.get("unit") or "—"} | Stake: {sysf.get("stake") or "—"}')
    lines.append("- Pool:")
    for p in (fun.get("system_pool") or []):
        dt = f'{(p.get("date") or "")} {(p.get("time") or "")}'.strip()
        lg = p.get("league") or "—"
        lines.append(f'  • ({dt}) ({lg}) {p.get("match")} — {p.get("market")} — {p.get("odds")}')
    lines.append("")
    # DRAW
    lines.append("DRAWBET")
    sysd = draw.get("system") or {}
    lines.append(f'- System: {sysd.get("label") or "—"} | Columns: {sysd.get("columns") or "—"} | Unit: {sysd.get("unit") or "—"} | Stake: {sysd.get("stake") or "—"}')
    lines.append("- Pool:")
    for p in (draw.get("system_pool") or []):
        dt = f'{(p.get("date") or "")} {(p.get("time") or "")}'.strip()
        lg = p.get("league") or "—"
        lines.append(f'  • ({dt}) ({lg}) {p.get("match")} — {p.get("market")} — {p.get("odds")}')
    lines.append("")
    # BANKROLLS
    lines.append("BANKROLLS")
    lines.append(f'- Core: Start {core.get("bankroll_start") or core.get("bankroll") or "—"} | Open {core.get("open") or "—"} | After {core.get("after_open") or "—"}')
    lines.append(f'- Fun:  Start {fun.get("bankroll_start") or fun.get("bankroll") or "—"} | Open {fun.get("open") or "—"} | After {fun.get("after_open") or "—"}')
    lines.append(f'- Draw: Start {draw.get("bankroll_start") or draw.get("bankroll") or "—"} | Open {draw.get("open") or "—"} | After {draw.get("after_open") or "—"}')
    lines.append("[/BLOG_PREVIEW]")
    return "\n".join(lines)

# ------------------------- MAIN -------------------------
def main():
    th_meta, fixtures = load_thursday_report()
    history = load_history(TUESDAY_HISTORY_PATH)

    window = th_meta.get("window") or {}
    wf = get_week_fields(window, history if isinstance(history, dict) else {})

    core_start = safe_float(((history.get("core") or {}) if isinstance(history, dict) else {}).get("bankroll_current"), None)
    fun_start  = safe_float(((history.get("funbet") or {}) if isinstance(history, dict) else {}).get("bankroll_current"), None)
    draw_start = safe_float(((history.get("drawbet") or {}) if isinstance(history, dict) else {}).get("bankroll_current"), None)

    core_bankroll_start = core_start if core_start is not None else DEFAULT_BANKROLL_CORE
    fun_bankroll_start  = fun_start if fun_start is not None else DEFAULT_BANKROLL_FUN
    draw_bankroll_start = draw_start if draw_start is not None else DEFAULT_BANKROLL_DRAW

    candidates = build_candidates(fixtures)

    core_singles, core_double, core_doubles, core_meta = core_select(candidates, core_bankroll_start)
    core_fixture_ids = {x.get("fixture_id") for x in core_singles if x.get("fixture_id") is not None}

    funbet = fun_select(candidates, fun_bankroll_start, core_fixture_ids)
    drawbet = draw_select(candidates, draw_bankroll_start)

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
            "rules": {
                "singles_odds_range": [CORE_MIN_ODDS, CORE_MAX_ODDS],
                "singles_stakes": {"1.60-1.75": 40, "1.75-1.90": 30, "1.90-2.10": 20},
                "low_odds_to_doubles": [CORE_LOW_MIN_ODDS, CORE_LOW_MAX_ODDS],
                "max_singles": CORE_MAX_SINGLES,
                "max_doubles": CORE_MAX_DOUBLES,
                "under_max_share": CORE_UNDER_MAX_SHARE,
                "under_requires_under_elite": True,
            },
            "singles": core_singles,
            "double": core_double,
            "doubles": core_doubles,
            "open": core_meta["open"],
            "after_open": core_meta["after_open"],
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

    # Attach ready-to-use blocks for Custom GPT
    report["copy_play"] = build_copy_play(report)
    report["blog_preview"] = build_blog_preview(report)

    os.makedirs("logs", exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
