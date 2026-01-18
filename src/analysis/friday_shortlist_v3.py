# ============================================================
#  src/analysis/friday_shortlist_v3.py
#  FRIDAY SHORTLIST v3.30 — RESET STABLE
#
#  Reads:
#    - logs/thursday_report_v3.json
#    - (optional) logs/tuesday_history_v3.json   (bankroll carry + week numbering)
#
#  Writes:
#    - logs/friday_shortlist_v3.json
#
#  KEY RULES:
#   - NO scaling of stakes (stakes must remain exact 40/30/20/15 etc)
#   - Core: singles odds 1.70–3.50 (stake ladder) + low-odds doubles (<1.70)
#   - Fun: system-only 5–7 picks; prefer 4/7 (avoid 5/7 unless necessary)
#   - Draw: system-only; up to 3 draw picks; system 2/3
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

# Exposure caps (soft guidance; we do NOT scale stakes; we only stop adding when cap reached)
CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.15"))
FUN_EXPOSURE_CAP  = float(os.getenv("FUN_EXPOSURE_CAP",  "0.20"))
DRAW_EXPOSURE_CAP = float(os.getenv("DRAW_EXPOSURE_CAP", "0.20"))

# Gates
ODDS_MATCH_MIN_SCORE_CORE = float(os.getenv("ODDS_MATCH_MIN_SCORE_CORE", "0.75"))
ODDS_MATCH_MIN_SCORE_FUN  = float(os.getenv("ODDS_MATCH_MIN_SCORE_FUN",  "0.75"))
ODDS_MATCH_MIN_SCORE_DRAW = float(os.getenv("ODDS_MATCH_MIN_SCORE_DRAW", "0.80"))

CORE_MIN_CONFIDENCE = float(os.getenv("CORE_MIN_CONFIDENCE", "0.55"))
FUN_MIN_CONFIDENCE  = float(os.getenv("FUN_MIN_CONFIDENCE",  "0.45"))
DRAW_MIN_CONFIDENCE = float(os.getenv("DRAW_MIN_CONFIDENCE", "0.55"))

FUN_AVOID_CORE_OVERLAP = os.getenv("FUN_AVOID_CORE_OVERLAP", "true").lower() == "true"

# ------------------------- PORTFOLIOS -------------------------
MARKET_CODE = {"Home": "1", "Draw": "X", "Away": "2", "Over 2.5": "O25", "Under 2.5": "U25"}
CORE_ALLOWED = {"Home", "Away", "Over 2.5", "Under 2.5"}
FUN_ALLOWED  = {"Home", "Away", "Over 2.5", "Under 2.5"}  # DrawBet handles draws
DRAW_ALLOWED = {"Draw"}

# ------------------------- COREBET RULES -------------------------
CORE_SINGLES_MIN_ODDS = float(os.getenv("CORE_SINGLES_MIN_ODDS", "1.70"))
CORE_SINGLES_MAX_ODDS = float(os.getenv("CORE_SINGLES_MAX_ODDS", "3.50"))

CORE_MAX_SINGLES = int(os.getenv("CORE_MAX_SINGLES", "8"))
CORE_MIN_SINGLES = int(os.getenv("CORE_MIN_SINGLES", "3"))

# Low odds bucket (<1.70) goes ONLY to doubles
CORE_LOW_ODDS_MIN = float(os.getenv("CORE_LOW_ODDS_MIN", "1.30"))
CORE_LOW_ODDS_MAX = float(os.getenv("CORE_LOW_ODDS_MAX", "1.69"))

CORE_MAX_DOUBLES = int(os.getenv("CORE_MAX_DOUBLES", "2"))
CORE_DOUBLE_TARGET_MIN = float(os.getenv("CORE_DOUBLE_TARGET_MIN", "2.10"))
CORE_DOUBLE_TARGET_MAX = float(os.getenv("CORE_DOUBLE_TARGET_MAX", "3.50"))

# Stake ladder (EXACT)
def core_single_stake(odds: float) -> float:
    if 1.70 <= odds <= 1.90:
        return 40.0
    if 1.90 < odds <= 2.20:
        return 30.0
    if 2.20 < odds <= 3.00:
        return 20.0
    if 3.00 < odds <= 3.50:
        return 15.0
    return 0.0

def core_double_stake(combo_odds: float) -> float:
    if combo_odds <= 2.70:
        return 20.0
    if combo_odds <= 3.50:
        return 15.0
    return 10.0

# Hard guardrails
MAX_PICK_ODDS = float(os.getenv("MAX_PICK_ODDS", "3.50"))  # apply to core+fun; draws have own cap

# Core gates (EV/prob)
CORE_EV_MIN_HOME  = float(os.getenv("CORE_EV_MIN_HOME",  "0.04"))
CORE_EV_MIN_AWAY  = float(os.getenv("CORE_EV_MIN_AWAY",  "0.05"))
CORE_EV_MIN_OVER  = float(os.getenv("CORE_EV_MIN_OVER",  "0.04"))
CORE_EV_MIN_UNDER = float(os.getenv("CORE_EV_MIN_UNDER", "0.08"))

CORE_P_MIN_HOME  = float(os.getenv("CORE_P_MIN_HOME",  "0.30"))
CORE_P_MIN_AWAY  = float(os.getenv("CORE_P_MIN_AWAY",  "0.24"))
CORE_P_MIN_OVER  = float(os.getenv("CORE_P_MIN_OVER",  "0.45"))
CORE_P_MIN_UNDER = float(os.getenv("CORE_P_MIN_UNDER", "0.58"))

# Under strict extra gates
CORE_UNDER_LTOTAL_MAX   = float(os.getenv("CORE_UNDER_LTOTAL_MAX", "2.30"))
CORE_UNDER_DRAW_MIN     = float(os.getenv("CORE_UNDER_DRAW_MIN",   "0.30"))
CORE_UNDER_ABS_GAP_MAX  = float(os.getenv("CORE_UNDER_ABS_GAP_MAX","0.35"))

# Core composition
CORE_MAX_UNDER_SHARE = float(os.getenv("CORE_MAX_UNDER_SHARE", "0.20"))

# ------------------------- FUNBET RULES (SYSTEM-ONLY) -------------------------
FUN_PICKS_MIN = int(os.getenv("FUN_PICKS_MIN", "5"))
FUN_PICKS_MAX = int(os.getenv("FUN_PICKS_MAX", "7"))

FUN_EV_MIN_HOME  = float(os.getenv("FUN_EV_MIN_HOME",  "0.05"))
FUN_EV_MIN_AWAY  = float(os.getenv("FUN_EV_MIN_AWAY",  "0.05"))
FUN_EV_MIN_OVER  = float(os.getenv("FUN_EV_MIN_OVER",  "0.05"))
FUN_EV_MIN_UNDER = float(os.getenv("FUN_EV_MIN_UNDER", "0.10"))

FUN_P_MIN_HOME  = float(os.getenv("FUN_P_MIN_HOME",  "0.28"))
FUN_P_MIN_AWAY  = float(os.getenv("FUN_P_MIN_AWAY",  "0.22"))
FUN_P_MIN_OVER  = float(os.getenv("FUN_P_MIN_OVER",  "0.42"))
FUN_P_MIN_UNDER = float(os.getenv("FUN_P_MIN_UNDER", "0.60"))

FUN_UNDER_LTOTAL_MAX  = float(os.getenv("FUN_UNDER_LTOTAL_MAX", "2.25"))
FUN_UNDER_DRAW_MIN    = float(os.getenv("FUN_UNDER_DRAW_MIN",   "0.30"))
FUN_UNDER_ABS_GAP_MAX = float(os.getenv("FUN_UNDER_ABS_GAP_MAX","0.35"))

# System refund thresholds (use 0.80 primary, 0.65 fallback)
SYS_REFUND_PRIMARY  = float(os.getenv("SYS_REFUND_PRIMARY",  "0.80"))
SYS_REFUND_FALLBACK = float(os.getenv("SYS_REFUND_FALLBACK", "0.65"))

# Fun system: prefer 4/7, else 3/6 or 3/5, avoid 5/7 unless last resort
SYS_TARGET_MIN = float(os.getenv("SYS_TARGET_MIN", "25.0"))
SYS_TARGET_MAX = float(os.getenv("SYS_TARGET_MAX", "50.0"))
SYS_UNIT_BASE  = float(os.getenv("SYS_UNIT_BASE",  "1.0"))  # base 1 per column, then clamp by target

# ------------------------- DRAWBET RULES (SYSTEM-ONLY) -------------------------
DRAW_PICKS = int(os.getenv("DRAW_PICKS", "3"))  # only 3 draws max
DRAW_ODDS_MIN = float(os.getenv("DRAW_ODDS_MIN", "3.10"))
DRAW_ODDS_MAX = float(os.getenv("DRAW_ODDS_MAX", "4.60"))
DRAW_EV_MIN   = float(os.getenv("DRAW_EV_MIN",   "0.07"))
DRAW_P_MIN    = float(os.getenv("DRAW_P_MIN",    "0.30"))
DRAW_LTOTAL_MAX = float(os.getenv("DRAW_LTOTAL_MAX", "2.45"))
DRAW_ABS_GAP_MAX = float(os.getenv("DRAW_ABS_GAP_MAX", "0.25"))

# Draw system 2/3
DRAW_SYS_LABEL = "2/3"
DRAW_SYS_COLS  = comb(3, 2)  # 3 columns

# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
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

def _ev_from_fx(fx, code: str):
    code = (code or "").upper()
    return safe_float({
        "1": fx.get("ev_1"),
        "2": fx.get("ev_2"),
        "X": fx.get("ev_x"),
        "O25": fx.get("ev_over"),
        "U25": fx.get("ev_under"),
    }.get(code), None)

def _prob_from_fx(fx, code: str):
    code = (code or "").upper()
    return safe_float({
        "1": fx.get("home_prob"),
        "2": fx.get("away_prob"),
        "X": fx.get("draw_prob"),
        "O25": fx.get("over_2_5_prob"),
        "U25": fx.get("under_2_5_prob"),
    }.get(code), None)

def _odds_from_fx(fx, code: str):
    code = (code or "").upper()
    return safe_float({
        "1": fx.get("offered_1"),
        "2": fx.get("offered_2"),
        "X": fx.get("offered_x"),
        "O25": fx.get("offered_over_2_5"),
        "U25": fx.get("offered_under_2_5"),
    }.get(code), None)

def _market_name(code: str):
    return {"1":"Home","2":"Away","X":"Draw","O25":"Over 2.5","U25":"Under 2.5"}.get(code, code)

def load_thursday():
    data = load_json(THURSDAY_REPORT_PATH)
    if isinstance(data, dict) and "fixtures" in data:
        return data["fixtures"], data
    raise KeyError("fixtures missing in Thursday report")

def build_pick_candidates(fixtures):
    out = []
    for fx in fixtures:
        for code in ["1","2","X","O25","U25"]:
            odds = _odds_from_fx(fx, code)
            if odds is None or odds <= 1.0:
                continue
            out.append({
                "pick_id": f'{fx.get("fixture_id")}:{code}',
                "fixture_id": fx.get("fixture_id"),
                "match": f'{fx.get("home")} – {fx.get("away")}',
                "league": fx.get("league"),
                "market_code": code,
                "market": _market_name(code),
                "odds": odds,
                "prob": _prob_from_fx(fx, code),
                "ev": _ev_from_fx(fx, code),
                "confidence": confidence_value(fx),
                "fx": fx,
            })
    return out

def _strip_pick(p, stake=None, tag=None):
    d = {
        "pick_id": p["pick_id"],
        "fixture_id": p["fixture_id"],
        "market_code": p["market_code"],
        "match": p["match"],
        "league": p["league"],
        "market": p["market"],
        "odds": round(float(p["odds"]), 3),
        "prob": p.get("prob"),
        "ev": p.get("ev"),
        "confidence": p.get("confidence"),
    }
    if stake is not None:
        d["stake"] = float(stake)
    if tag:
        d["tag"] = tag
    return d

# ------------------------------------------------------
# System helpers (refund ratio at min hits, conservative)
# ------------------------------------------------------
def _columns_for_r(n: int, r: int) -> int:
    return comb(n, r) if (n > 0 and 0 < r <= n) else 0

def _refund_ratio_min_hits(odds_list, r_min: int, columns: int) -> float:
    if columns <= 0 or r_min <= 0:
        return 0.0
    o = sorted([float(x) for x in odds_list])[:r_min]
    prod = 1.0
    for v in o:
        prod *= max(1.01, float(v))
    return prod / float(columns)

def _system_try(pool, r_min: int, label: str):
    n = len(pool)
    cols = _columns_for_r(n, r_min)
    if cols <= 0:
        return None
    rr = _refund_ratio_min_hits([p["odds"] for p in pool], r_min, cols)
    return {"label": label, "n": n, "columns": cols, "min_hits": r_min, "refund_ratio_min_hits": round(rr, 4)}

def choose_fun_system(pool):
    """
    Prefer:
      7 picks -> 4/7 (primary). If fails -> 5/7 only if necessary.
      6 picks -> 3/6 (primary) else 4/6.
      5 picks -> 3/5.
    If primary fails, allow fallback threshold 0.65.
    """
    n = len(pool)
    if n < 5:
        return None, None, None, True

    primary = SYS_REFUND_PRIMARY
    fallback = SYS_REFUND_FALLBACK

    # Try primary threshold first
    trials_primary = []
    if n == 7:
        trials_primary = [_system_try(pool, 4, "4/7"), _system_try(pool, 5, "5/7")]
    elif n == 6:
        trials_primary = [_system_try(pool, 3, "3/6"), _system_try(pool, 4, "4/6")]
    else:
        trials_primary = [_system_try(pool, 3, "3/5")]

    for t in trials_primary:
        if t and t["refund_ratio_min_hits"] >= primary:
            # for n=7 this selects 4/7 before 5/7
            return t, primary, False, False

    # If primary fails, prefer higher r (next ladder) BEFORE fallback threshold
    ladder = []
    if n == 7:
        ladder = [_system_try(pool, 4, "4/7"), _system_try(pool, 5, "5/7")]
    elif n == 6:
        ladder = [_system_try(pool, 4, "4/6"), _system_try(pool, 3, "3/6")]
    else:
        ladder = [_system_try(pool, 3, "3/5")]

    for t in ladder:
        if t and t["refund_ratio_min_hits"] >= primary:
            return t, primary, False, False

    # fallback threshold 0.65 (breach flag)
    for t in ladder:
        if t and t["refund_ratio_min_hits"] >= fallback:
            return t, fallback, True, True

    return None, primary, True, True

def system_stake_for_columns(columns: int, cap_amount: float):
    """
    No scaling; we set unit so that stake is within [25..50] but <= cap_amount.
    """
    if columns <= 0:
        return 0.0, 0.0
    base_total = columns * SYS_UNIT_BASE
    target_total = _clamp(base_total, SYS_TARGET_MIN, SYS_TARGET_MAX)
    target_total = min(target_total, cap_amount)
    unit = round(target_total / columns, 2)
    stake = round(unit * columns, 2)
    return unit, stake

# ------------------------------------------------------
# CORE selection (no scaling; stop when cap reached)
# ------------------------------------------------------
def core_select(picks, bankroll_start):
    cap = bankroll_start * CORE_EXPOSURE_CAP

    # Candidate pools
    singles_pool = []
    low_pool = []

    for p in picks:
        fx = p["fx"]
        if p["market"] not in CORE_ALLOWED:
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_CORE):
            continue
        if not confidence_ok(fx, CORE_MIN_CONFIDENCE):
            continue

        odds = safe_float(p["odds"], None)
        if odds is None:
            continue
        if odds > MAX_PICK_ODDS:
            continue

        evv = safe_float(p.get("ev"), None)
        pr = safe_float(p.get("prob"), None)
        if evv is None or pr is None:
            continue

        # market gates
        if p["market"] == "Home" and (evv < CORE_EV_MIN_HOME or pr < CORE_P_MIN_HOME):
            continue
        if p["market"] == "Away" and (evv < CORE_EV_MIN_AWAY or pr < CORE_P_MIN_AWAY):
            continue
        if p["market"] == "Over 2.5" and (evv < CORE_EV_MIN_OVER or pr < CORE_P_MIN_OVER):
            continue
        if p["market"] == "Under 2.5":
            if evv < CORE_EV_MIN_UNDER or pr < CORE_P_MIN_UNDER:
                continue
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
            if st > 0:
                singles_pool.append((p, st))
        elif CORE_LOW_ODDS_MIN <= odds <= CORE_LOW_ODDS_MAX:
            low_pool.append(p)

    # sort by EV desc, then confidence, then prob
    singles_pool.sort(key=lambda x: (safe_float(x[0].get("ev"), -9999.0), safe_float(x[0].get("confidence"), 0.0), safe_float(x[0].get("prob"), 0.0)), reverse=True)
    low_pool.sort(key=lambda x: (safe_float(x.get("ev"), -9999.0), safe_float(x.get("confidence"), 0.0), safe_float(x.get("prob"), 0.0)), reverse=True)

    singles = []
    used = set()
    under_count = 0
    max_under = max(1, int(round(CORE_MAX_UNDER_SHARE * CORE_MAX_SINGLES)))

    open_total = 0.0

    for p, st in singles_pool:
        if len(singles) >= CORE_MAX_SINGLES:
            break
        if p["match"] in used:
            continue
        if p["market"] == "Under 2.5" and under_count >= max_under:
            continue
        if open_total + st > cap:
            continue  # no scaling; just skip
        singles.append(_strip_pick(p, stake=st, tag="core_single"))
        used.add(p["match"])
        open_total += st
        if p["market"] == "Under 2.5":
            under_count += 1

    # Doubles: only if there are low-odds picks and room in cap
    doubles = []
    if low_pool and CORE_MAX_DOUBLES > 0:
        # partner candidates: from singles_pool picks (safer), plus low_pool
        partner = [pp for (pp, _st) in singles_pool] + low_pool
        partner.sort(key=lambda x: (safe_float(x.get("ev"), -9999.0), safe_float(x.get("confidence"), 0.0), safe_float(x.get("prob"), 0.0)), reverse=True)

        used_d = set()
        for leg1 in low_pool:
            if len(doubles) >= CORE_MAX_DOUBLES:
                break
            for leg2 in partner:
                if leg2["match"] == leg1["match"]:
                    continue
                if leg1["match"] in used_d or leg2["match"] in used_d:
                    continue
                combo = safe_float(leg1["odds"], 1.0) * safe_float(leg2["odds"], 1.0)
                if not (CORE_DOUBLE_TARGET_MIN <= combo <= CORE_DOUBLE_TARGET_MAX):
                    continue
                stake = core_double_stake(combo)
                if open_total + stake > cap:
                    continue  # no scaling
                doubles.append({
                    "legs": [
                        {"pick_id": leg1["pick_id"], "fixture_id": leg1["fixture_id"], "match": leg1["match"], "market": leg1["market"], "market_code": leg1["market_code"], "odds": round(float(leg1["odds"]),3)},
                        {"pick_id": leg2["pick_id"], "fixture_id": leg2["fixture_id"], "match": leg2["match"], "market": leg2["market"], "market_code": leg2["market_code"], "odds": round(float(leg2["odds"]),3)},
                    ],
                    "combo_odds": round(combo, 2),
                    "stake": float(stake),
                    "tag": "core_double_lowodds",
                })
                used_d.add(leg1["match"]); used_d.add(leg2["match"])
                open_total += stake
                break

    # enforce min singles softly (no scaling; if too strict => may be below min)
    meta = {
        "bankroll_start": bankroll_start,
        "exposure_cap_pct": CORE_EXPOSURE_CAP,
        "open": round(open_total, 2),
        "after_open": round(bankroll_start - open_total, 2),
        "picks_count": len(singles),
        "doubles_count": len(doubles),
        "stake_policy": "NO_SCALING",
    }
    return singles, (doubles[0] if doubles else None), doubles, meta

# ------------------------------------------------------
# FUN selection (system-only)
# ------------------------------------------------------
def fun_select(picks, bankroll_start, core_fixture_ids):
    cap = bankroll_start * FUN_EXPOSURE_CAP

    cand = []
    for p in picks:
        fx = p["fx"]
        if p["market"] not in FUN_ALLOWED:
            continue
        if FUN_AVOID_CORE_OVERLAP and p["fixture_id"] in core_fixture_ids:
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_FUN):
            continue
        if not confidence_ok(fx, FUN_MIN_CONFIDENCE):
            continue

        odds = safe_float(p["odds"], None)
        if odds is None or odds <= 1.0:
            continue
        if odds > MAX_PICK_ODDS:
            continue

        evv = safe_float(p.get("ev"), None)
        pr = safe_float(p.get("prob"), None)
        if evv is None or pr is None:
            continue

        if p["market"] == "Home" and (evv < FUN_EV_MIN_HOME or pr < FUN_P_MIN_HOME):
            continue
        if p["market"] == "Away" and (evv < FUN_EV_MIN_AWAY or pr < FUN_P_MIN_AWAY):
            continue
        if p["market"] == "Over 2.5" and (evv < FUN_EV_MIN_OVER or pr < FUN_P_MIN_OVER):
            continue
        if p["market"] == "Under 2.5":
            if evv < FUN_EV_MIN_UNDER or pr < FUN_P_MIN_UNDER:
                continue
            if (fx.get("flags") or {}).get("tight_game") is not True:
                continue
            if safe_float(fx.get("draw_prob"), 0.0) < FUN_UNDER_DRAW_MIN:
                continue
            if _total_lambda(fx) > FUN_UNDER_LTOTAL_MAX:
                continue
            if _abs_gap(fx) > FUN_UNDER_ABS_GAP_MAX:
                continue

        cand.append(p)

    cand.sort(key=lambda x: (safe_float(x.get("ev"), -9999.0), safe_float(x.get("confidence"), 0.0), safe_float(x.get("prob"), 0.0)), reverse=True)

    pool = []
    used = set()
    for p in cand:
        if p["match"] in used:
            continue
        pool.append(p)
        used.add(p["match"])
        if len(pool) >= FUN_PICKS_MAX:
            break
    if len(pool) < FUN_PICKS_MIN:
        pool = pool[:max(0, len(pool))]

    # choose pool size preference 7 -> 6 -> 5
    if len(pool) >= 7:
        sys_pool = pool[:7]
    elif len(pool) >= 6:
        sys_pool = pool[:6]
    else:
        sys_pool = pool[:5]

    sys_choice, refund_used, breached, rule_breached = choose_fun_system(sys_pool)

    columns = int(sys_choice["columns"]) if sys_choice else 0
    unit, stake = (0.0, 0.0)
    if sys_choice:
        unit, stake = system_stake_for_columns(columns, cap)

    payload = {
        "bankroll_start": bankroll_start,
        "bankroll": DEFAULT_BANKROLL_FUN,
        "bankroll_source": "history" if bankroll_start != DEFAULT_BANKROLL_FUN else "default",
        "exposure_cap_pct": FUN_EXPOSURE_CAP,
        "rules": {
            "refund_primary": SYS_REFUND_PRIMARY,
            "refund_fallback": SYS_REFUND_FALLBACK,
            "refund_used": refund_used,
            "refund_rule_breached": bool(rule_breached),
            "stake_policy": "NO_SCALING",
            "prefer_4_of_7": True,
            "max_odds": MAX_PICK_ODDS
        },
        "picks_total": [_strip_pick(p) for p in pool],
        "system_pool": [_strip_pick(p) for p in sys_pool],
        "system": {
            "label": sys_choice["label"] if sys_choice else None,
            "columns": columns,
            "min_hits": sys_choice["min_hits"] if sys_choice else None,
            "refund_ratio_min_hits": sys_choice["refund_ratio_min_hits"] if sys_choice else None,
            "unit": unit,
            "stake": stake,
            "refund_threshold_used": refund_used,
            "refund_rule_breached": bool(rule_breached),
            "has_system": bool(sys_choice is not None),
        },
        "open": round(stake, 2),
        "after_open": round(bankroll_start - stake, 2),
        "counts": {"picks_total": len(pool), "system_pool": len(sys_pool)},
    }
    return payload

# ------------------------------------------------------
# DRAW selection (system-only 2/3)
# ------------------------------------------------------
def draw_select(picks, bankroll_start):
    cap = bankroll_start * DRAW_EXPOSURE_CAP

    cand = []
    for p in picks:
        fx = p["fx"]
        if p["market"] != "Draw":
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_DRAW):
            continue
        if not confidence_ok(fx, DRAW_MIN_CONFIDENCE):
            continue

        odds = safe_float(p["odds"], None)
        if odds is None:
            continue
        if not (DRAW_ODDS_MIN <= odds <= DRAW_ODDS_MAX):
            continue

        evv = safe_float(p.get("ev"), None)
        pr = safe_float(p.get("prob"), None)
        if evv is None or pr is None:
            continue
        if evv < DRAW_EV_MIN or pr < DRAW_P_MIN:
            continue

        if _total_lambda(fx) > DRAW_LTOTAL_MAX:
            continue
        if _abs_gap(fx) > DRAW_ABS_GAP_MAX:
            continue

        cand.append(p)

    cand.sort(key=lambda x: (safe_float(x.get("ev"), -9999.0), safe_float(x.get("confidence"), 0.0), safe_float(x.get("prob"), 0.0)), reverse=True)

    pool = []
    used = set()
    for p in cand:
        if p["match"] in used:
            continue
        pool.append(p)
        used.add(p["match"])
        if len(pool) >= DRAW_PICKS:
            break

    has_system = len(pool) == 3
    unit = stake = 0.0
    if has_system:
        columns = DRAW_SYS_COLS
        # target is min(25..50, cap). For 3 cols, unit will be higher, but still capped.
        base_total = columns * SYS_UNIT_BASE
        target = _clamp(base_total, SYS_TARGET_MIN, SYS_TARGET_MAX)
        target = min(target, cap)
        unit = round(target / columns, 2)
        stake = round(unit * columns, 2)

    payload = {
        "bankroll_start": bankroll_start,
        "bankroll": DEFAULT_BANKROLL_DRAW,
        "bankroll_source": "history" if bankroll_start != DEFAULT_BANKROLL_DRAW else "default",
        "exposure_cap_pct": DRAW_EXPOSURE_CAP,
        "rules": {
            "picks": DRAW_PICKS,
            "odds_range": [DRAW_ODDS_MIN, DRAW_ODDS_MAX],
            "system": DRAW_SYS_LABEL,
            "stake_policy": "NO_SCALING",
        },
        "picks_total": [_strip_pick(p) for p in pool],
        "system_pool": [_strip_pick(p) for p in pool],
        "system": {
            "label": DRAW_SYS_LABEL if has_system else None,
            "columns": DRAW_SYS_COLS if has_system else 0,
            "unit": unit,
            "stake": stake,
            "has_system": bool(has_system),
        },
        "open": round(stake, 2),
        "after_open": round(bankroll_start - stake, 2),
        "counts": {"picks_total": len(pool), "system_pool": len(pool)},
    }
    return payload

# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    fixtures, th_meta = load_thursday()
    history = load_history()

    window = th_meta.get("window") if isinstance(th_meta, dict) else None
    wf = get_week_fields(window or {}, history)

    core_start = safe_float((history.get("core") or {}).get("bankroll_current"), None)
    fun_start  = safe_float((history.get("funbet") or {}).get("bankroll_current"), None)
    draw_start = safe_float((history.get("drawbet") or {}).get("bankroll_current"), None)

    core_bankroll_start = core_start if core_start is not None else DEFAULT_BANKROLL_CORE
    fun_bankroll_start  = fun_start  if fun_start  is not None else DEFAULT_BANKROLL_FUN
    draw_bankroll_start = draw_start if draw_start is not None else DEFAULT_BANKROLL_DRAW

    picks = build_pick_candidates(fixtures)

    core_singles, core_double, core_doubles, core_meta = core_select(picks, core_bankroll_start)
    core_fixture_ids = {x["fixture_id"] for x in core_singles}

    funbet = fun_select(picks, fun_bankroll_start, core_fixture_ids)
    drawbet = draw_select(picks, draw_bankroll_start)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "week_id": wf["week_id"],
        "week_no": wf["week_no"],
        "week_label": wf["week_label"],
        "window": th_meta.get("window"),
        "fixtures_total": th_meta.get("fixtures_total", len(fixtures)),

        "core": {
            "portfolio": "CoreBet",
            "bankroll": DEFAULT_BANKROLL_CORE,
            "bankroll_start": core_bankroll_start,
            "bankroll_source": ("history" if core_start is not None else "default"),
            "exposure_cap_pct": CORE_EXPOSURE_CAP,
            "rules": {
                "singles_odds_range": [CORE_SINGLES_MIN_ODDS, CORE_SINGLES_MAX_ODDS],
                "max_singles": CORE_MAX_SINGLES,
                "max_odds": MAX_PICK_ODDS,
                "stake_ladder": {"1.70-1.90": 40, "1.90-2.20": 30, "2.20-3.00": 20, "3.00-3.50": 15},
                "low_odds_to_doubles": [CORE_LOW_ODDS_MIN, CORE_LOW_ODDS_MAX],
                "double_target_combo_odds": [CORE_DOUBLE_TARGET_MIN, CORE_DOUBLE_TARGET_MAX],
                "stake_policy": "NO_SCALING",
            },
            "singles": core_singles,
            "double": core_double,
            "doubles": core_doubles,
            "open": core_meta["open"],
            "after_open": core_meta["after_open"],
            "picks_count": core_meta["picks_count"],
            "doubles_count": core_meta["doubles_count"],
        },

        "funbet": {
            "portfolio": "FunBet",
            **funbet,
        },

        "drawbet": {
            "portfolio": "DrawBet",
            **drawbet,
        },
    }

    os.makedirs("logs", exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
