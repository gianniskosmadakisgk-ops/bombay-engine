# ============================================================
#  FRIDAY SHORTLIST v3.30 — PRODUCTION (CoreBet / FunBet / DrawBet)
#
#  요구:
#   - CoreBet: singles 1.70–3.60, stake ladder EXACT 40/30/20/15, max 8 picks
#            - NO singles below 1.70
#            - odds <1.70 go ONLY to doubles (if possible)
#            - no odds > 3.60 anywhere
#   - FunBet: SYSTEM ONLY, up to 7 picks, choose system preferring:
#            - n=7: 4/7 first, only then 5/7
#            - n=6: 3/6 first, only then 4/6
#            - n=5: 3/5
#            - refund ratio target 0.80 else fallback 0.65 (flag breach)
#   - DrawBet: SYSTEM ONLY, 3 draws, 2/3 system (if not enough -> none)
#   - Reads tuesday_history_v3.json if uploaded (bankroll carry + week numbering)
#   - Writes logs/friday_shortlist_v3.json
# ============================================================

import os
import json
from datetime import datetime, date
from math import comb

THURSDAY_REPORT_PATH = os.getenv("THURSDAY_REPORT_PATH", "logs/thursday_report_v3.json")
FRIDAY_REPORT_PATH   = os.getenv("FRIDAY_REPORT_PATH",   "logs/friday_shortlist_v3.json")
TUESDAY_HISTORY_PATH = os.getenv("TUESDAY_HISTORY_PATH", "logs/tuesday_history_v3.json")

# ------------------------- BANKROLLS -------------------------
DEFAULT_BANKROLL_CORE = float(os.getenv("BANKROLL_CORE", "800"))
DEFAULT_BANKROLL_FUN  = float(os.getenv("BANKROLL_FUN",  "400"))
DEFAULT_BANKROLL_DRAW = float(os.getenv("BANKROLL_DRAW", "300"))

# Exposure caps
CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.15"))
FUN_EXPOSURE_CAP  = float(os.getenv("FUN_EXPOSURE_CAP",  "0.20"))
DRAW_EXPOSURE_CAP = float(os.getenv("DRAW_EXPOSURE_CAP", "0.20"))

# Global odds cap
MAX_ODDS_CAP = float(os.getenv("MAX_ODDS_CAP", "3.60"))

# Odds-match + confidence gates
ODDS_MATCH_MIN_SCORE_CORE = float(os.getenv("ODDS_MATCH_MIN_SCORE_CORE", "0.75"))
ODDS_MATCH_MIN_SCORE_FUN  = float(os.getenv("ODDS_MATCH_MIN_SCORE_FUN",  "0.75"))
ODDS_MATCH_MIN_SCORE_DRAW = float(os.getenv("ODDS_MATCH_MIN_SCORE_DRAW", "0.80"))

CORE_MIN_CONFIDENCE = float(os.getenv("CORE_MIN_CONFIDENCE", "0.55"))
FUN_MIN_CONFIDENCE  = float(os.getenv("FUN_MIN_CONFIDENCE",  "0.45"))
DRAW_MIN_CONFIDENCE = float(os.getenv("DRAW_MIN_CONFIDENCE", "0.55"))

# Avoid overlap between Core and Fun
FUN_AVOID_CORE_OVERLAP = os.getenv("FUN_AVOID_CORE_OVERLAP", "true").lower() == "true"

MARKET_CODE = {"Home":"1","Away":"2","Draw":"X","Over 2.5":"O25","Under 2.5":"U25"}

# ------------------------- HELPERS -------------------------
def safe_float(v, d=None):
    try:
        return float(v)
    except Exception:
        return d

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

def confidence_ok(fx, min_conf):
    c = safe_float((fx.get("flags") or {}).get("confidence"), None)
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

def load_thursday_fixtures():
    data = load_json(THURSDAY_REPORT_PATH)
    if "fixtures" in data:
        return data["fixtures"], data
    if "report" in data and isinstance(data["report"], dict) and "fixtures" in data["report"]:
        return data["report"]["fixtures"], data["report"]
    raise KeyError("fixtures not found in Thursday report")

def _odds_from_fx(fx, code):
    code = (code or "").upper()
    if code == "1": return safe_float(fx.get("offered_1"), None)
    if code == "2": return safe_float(fx.get("offered_2"), None)
    if code == "X": return safe_float(fx.get("offered_x"), None)
    if code == "O25": return safe_float(fx.get("offered_over_2_5"), None)
    if code == "U25": return safe_float(fx.get("offered_under_2_5"), None)
    return None

def _ev_from_fx(fx, code):
    code = (code or "").upper()
    if code == "1": return safe_float(fx.get("ev_1"), None)
    if code == "2": return safe_float(fx.get("ev_2"), None)
    if code == "X": return safe_float(fx.get("ev_x"), None)
    if code == "O25": return safe_float(fx.get("ev_over"), None)
    if code == "U25": return safe_float(fx.get("ev_under"), None)
    return None

def _prob_from_fx(fx, code):
    code = (code or "").upper()
    if code == "1": return safe_float(fx.get("home_prob"), None)
    if code == "2": return safe_float(fx.get("away_prob"), None)
    if code == "X": return safe_float(fx.get("draw_prob"), None)
    if code == "O25": return safe_float(fx.get("over_2_5_prob"), None)
    if code == "U25": return safe_float(fx.get("under_2_5_prob"), None)
    return None

def _market_name(code):
    code = (code or "").upper()
    return {"1":"Home","2":"Away","X":"Draw","O25":"Over 2.5","U25":"Under 2.5"}.get(code, code)

def build_pick_candidates(fixtures):
    out = []
    for fx in fixtures:
        for code in ["1","2","X","O25","U25"]:
            odds = _odds_from_fx(fx, code)
            if odds is None or odds <= 1.0:
                continue
            if odds > MAX_ODDS_CAP:
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
                "confidence": safe_float((fx.get("flags") or {}).get("confidence"), None),
                "fx": fx,
            })
    return out

# ------------------------- COREBET -------------------------
CORE_SINGLES_MIN_ODDS = 1.70
CORE_SINGLES_MAX_ODDS = 3.60
CORE_MAX_SINGLES = int(os.getenv("CORE_MAX_SINGLES", "8"))  # ✅ 7-8 max
CORE_MIN_SINGLES = int(os.getenv("CORE_MIN_SINGLES", "5"))
CORE_LOW_ODDS_MIN = 1.30
CORE_LOW_ODDS_MAX = 1.69

CORE_MAX_DOUBLES = int(os.getenv("CORE_MAX_DOUBLES", "2"))
CORE_DOUBLE_TARGET_MIN = float(os.getenv("CORE_DOUBLE_TARGET_MIN", "2.10"))
CORE_DOUBLE_TARGET_MAX = float(os.getenv("CORE_DOUBLE_TARGET_MAX", "3.50"))

CORE_MIN_1X2_SHARE = float(os.getenv("CORE_MIN_1X2_SHARE", "0.30"))
CORE_MAX_UNDER_SHARE = float(os.getenv("CORE_MAX_UNDER_SHARE", "0.20"))

def core_single_stake(odds: float) -> float:
    # ✅ EXACT ladder
    if 1.70 <= odds <= 1.90: return 40.0
    if 1.90 < odds <= 2.20: return 30.0
    if 2.20 < odds <= 3.00: return 20.0
    if 3.00 < odds <= 3.60: return 15.0
    return 0.0

def core_double_stake(combo_odds: float) -> float:
    if combo_odds <= 2.70: return 20.0
    return 15.0

# gates
CORE_EV_MIN_HOME  = float(os.getenv("CORE_EV_MIN_HOME", "0.04"))
CORE_EV_MIN_AWAY  = float(os.getenv("CORE_EV_MIN_AWAY", "0.05"))
CORE_EV_MIN_OVER  = float(os.getenv("CORE_EV_MIN_OVER", "0.04"))
CORE_EV_MIN_UNDER = float(os.getenv("CORE_EV_MIN_UNDER", "0.08"))

CORE_P_MIN_HOME  = float(os.getenv("CORE_P_MIN_HOME", "0.30"))
CORE_P_MIN_AWAY  = float(os.getenv("CORE_P_MIN_AWAY", "0.24"))
CORE_P_MIN_OVER  = float(os.getenv("CORE_P_MIN_OVER", "0.45"))
CORE_P_MIN_UNDER = float(os.getenv("CORE_P_MIN_UNDER", "0.58"))

CORE_UNDER_LTOTAL_MAX = float(os.getenv("CORE_UNDER_LTOTAL_MAX", "2.30"))
CORE_UNDER_DRAW_MIN   = float(os.getenv("CORE_UNDER_DRAW_MIN", "0.30"))
CORE_UNDER_ABS_GAP_MAX= float(os.getenv("CORE_UNDER_ABS_GAP_MAX", "0.35"))

def _strip_core(p, stake):
    return {
        "pick_id": p["pick_id"],
        "fixture_id": p["fixture_id"],
        "market_code": p["market_code"],
        "match": p["match"],
        "league": p["league"],
        "market": p["market"],
        "odds": round(float(p["odds"]), 3),
        "prob": p.get("prob"),
        "ev": p.get("ev"),
        "stake": round(float(stake), 2),
        "tag": "core_single",
    }

def corebet_select(picks, bankroll_core):
    cap_amount = bankroll_core * CORE_EXPOSURE_CAP

    singles_pool = []
    low_pool = []

    for p in picks:
        fx = p["fx"]
        if p["market"] not in ("Home","Away","Over 2.5","Under 2.5"):
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_CORE):
            continue
        if not confidence_ok(fx, CORE_MIN_CONFIDENCE):
            continue

        odds = safe_float(p.get("odds"), None)
        evv  = safe_float(p.get("ev"), None)
        pr   = safe_float(p.get("prob"), None)
        if odds is None or evv is None or pr is None:
            continue
        if odds > MAX_ODDS_CAP:
            continue

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

    def key(p):
        return (safe_float(p.get("ev"), -9999.0), safe_float(p.get("confidence"), 0.0), safe_float(p.get("prob"), 0.0))

    singles_pool.sort(key=lambda t: key(t[0]), reverse=True)
    low_pool.sort(key=key, reverse=True)

    singles = []
    used = set()

    max_under = max(1, int(round(CORE_MAX_UNDER_SHARE * CORE_MAX_SINGLES)))
    target_min_1x2 = max(1, int(round(CORE_MIN_1X2_SHARE * CORE_MIN_SINGLES)))

    under_cnt = 0
    one_two_cnt = 0

    # pass1: ensure min 1X2
    for p, st in singles_pool:
        if len(singles) >= CORE_MAX_SINGLES:
            break
        if p["match"] in used:
            continue
        if p["market"] not in ("Home","Away"):
            continue
        singles.append(_strip_core(p, st))
        used.add(p["match"])
        one_two_cnt += 1
        if one_two_cnt >= target_min_1x2 and len(singles) >= CORE_MIN_SINGLES:
            break

    # pass2: fill remaining, respect under cap
    for p, st in singles_pool:
        if len(singles) >= CORE_MAX_SINGLES:
            break
        if p["match"] in used:
            continue
        if p["market"] == "Under 2.5" and under_cnt >= max_under:
            continue
        singles.append(_strip_core(p, st))
        used.add(p["match"])
        if p["market"] == "Under 2.5":
            under_cnt += 1

    # doubles: use low odds + partner (from picks) => combo in range
    doubles = []
    if low_pool and CORE_MAX_DOUBLES > 0:
        partners = [p for p in picks if p["market"] in ("Home","Away","Over 2.5","Under 2.5")]
        partners.sort(key=key, reverse=True)

        used_d = set()
        for leg1 in low_pool:
            if len(doubles) >= CORE_MAX_DOUBLES:
                break
            for leg2 in partners:
                if leg2["match"] == leg1["match"]:
                    continue
                if leg1["match"] in used_d or leg2["match"] in used_d:
                    continue
                o1 = safe_float(leg1["odds"], 1.0)
                o2 = safe_float(leg2["odds"], 1.0)
                combo = o1 * o2
                if combo > MAX_ODDS_CAP:
                    continue
                if not (CORE_DOUBLE_TARGET_MIN <= combo <= CORE_DOUBLE_TARGET_MAX):
                    continue
                stake = core_double_stake(combo)
                doubles.append({
                    "legs": [
                        {"pick_id": leg1["pick_id"], "fixture_id": leg1["fixture_id"], "match": leg1["match"], "market": leg1["market"], "market_code": leg1["market_code"], "odds": round(o1, 3)},
                        {"pick_id": leg2["pick_id"], "fixture_id": leg2["fixture_id"], "match": leg2["match"], "market": leg2["market"], "market_code": leg2["market_code"], "odds": round(o2, 3)},
                    ],
                    "combo_odds": round(combo, 2),
                    "stake": round(stake, 2),
                    "tag": "core_double_lowodds",
                })
                used_d.add(leg1["match"]); used_d.add(leg2["match"])
                break

    open_total = sum(x["stake"] for x in singles) + sum(d.get("stake", 0.0) for d in doubles)
    scale_applied = 1.0
    if open_total > cap_amount and open_total > 0:
        s = cap_amount / open_total
        for x in singles:
            x["stake"] = round(x["stake"] * s, 2)
        for d in doubles:
            d["stake"] = round(float(d.get("stake", 0.0)) * s, 2)
        scale_applied = round(s, 3)
        open_total = sum(x["stake"] for x in singles) + sum(d.get("stake", 0.0) for d in doubles)

    meta = {
        "open": round(open_total, 2),
        "after_open": round(bankroll_core - open_total, 2),
        "picks_count": len(singles),
        "doubles_count": len(doubles),
        "scale_applied": scale_applied,
    }
    return singles, (doubles[0] if doubles else None), doubles, meta

# ------------------------- FUNBET (SYSTEM ONLY) -------------------------
FUN_PICKS_MAX = int(os.getenv("FUN_PICKS_MAX", "7"))

SYS_REFUND_PRIMARY  = float(os.getenv("SYS_REFUND_PRIMARY", "0.80"))
SYS_REFUND_FALLBACK = float(os.getenv("SYS_REFUND_FALLBACK", "0.65"))

SYS_UNIT_BASE = float(os.getenv("SYS_UNIT_BASE", "1.0"))
SYS_TARGET_MIN = float(os.getenv("SYS_TARGET_MIN", "25.0"))
SYS_TARGET_MAX = float(os.getenv("SYS_TARGET_MAX", "50.0"))

SYS_CONF_MID  = float(os.getenv("SYS_CONF_MID", "0.55"))
SYS_CONF_HIGH = float(os.getenv("SYS_CONF_HIGH", "0.70"))
SYS_MULT_MID  = float(os.getenv("SYS_MULT_MID", "1.15"))
SYS_MULT_HIGH = float(os.getenv("SYS_MULT_HIGH", "1.30"))

# fun gates (basic)
FUN_EV_MIN = float(os.getenv("FUN_EV_MIN", "0.05"))

def _columns_for_r(n, r):
    return comb(n, r) if (n > 0 and 0 < r <= n) else 0

def _refund_ratio_worst_case(odds_list, r_min, columns):
    if columns <= 0:
        return 0.0
    o = sorted([float(x) for x in odds_list])[:r_min]
    prod = 1.0
    for v in o:
        prod *= max(1.01, float(v))
    return prod / float(columns)

def _system_try(pool, r_try, label):
    n = len(pool)
    cols = _columns_for_r(n, r_try)
    if cols <= 0:
        return None
    rr = _refund_ratio_worst_case([p["odds"] for p in pool], r_try, cols)
    return {"label": label, "columns": cols, "min_hits": r_try, "refund_ratio_min_hits": round(rr, 4)}

def choose_fun_system(pool):
    n = len(pool)
    if n == 7:
        # ✅ prefer 4/7
        primary = [_system_try(pool, 4, "4/7"), _system_try(pool, 5, "5/7")]
    elif n == 6:
        primary = [_system_try(pool, 3, "3/6"), _system_try(pool, 4, "4/6")]
    elif n == 5:
        primary = [_system_try(pool, 3, "3/5")]
    else:
        return None, SYS_REFUND_PRIMARY, True

    # try primary threshold
    for cand in primary:
        if cand and cand["refund_ratio_min_hits"] >= SYS_REFUND_PRIMARY:
            return cand, SYS_REFUND_PRIMARY, False

    # fallback threshold 0.65 (still keep preference order)
    for cand in primary:
        if cand and cand["refund_ratio_min_hits"] >= SYS_REFUND_FALLBACK:
            return cand, SYS_REFUND_FALLBACK, True

    return None, SYS_REFUND_PRIMARY, True

def conf_avg(pool):
    vals = [safe_float(p.get("confidence"), None) for p in pool]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, "na"
    a = sum(vals) / len(vals)
    band = "high" if a >= SYS_CONF_HIGH else ("mid" if a >= SYS_CONF_MID else "low")
    return round(a, 3), band

def system_unit(columns, cap_system, threshold_used, breached, cavg, cband):
    if columns <= 0:
        return 0.0, 0.0, 1.0, 0.0, 0.0

    base_total = columns * SYS_UNIT_BASE
    target_total = max(SYS_TARGET_MIN, min(SYS_TARGET_MAX, base_total))

    mult = 1.0
    if (threshold_used == SYS_REFUND_PRIMARY) and (not breached) and (cavg is not None):
        if cband == "high":
            mult = SYS_MULT_HIGH
        elif cband == "mid":
            mult = SYS_MULT_MID

    boosted = max(SYS_TARGET_MIN, min(SYS_TARGET_MAX, target_total * mult))
    final_total = min(boosted, cap_system)

    unit = round(final_total / columns, 2)
    stake = round(unit * columns, 2)
    return unit, stake, round(mult, 3), round(target_total, 2), round(final_total, 2)

def strip_pick(p):
    return {
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

def funbet_select(picks, bankroll_fun, core_fixture_ids):
    cap_total = bankroll_fun * FUN_EXPOSURE_CAP

    candidates = []
    for p in picks:
        fx = p["fx"]
        if p["market"] == "Draw":
            continue
        if FUN_AVOID_CORE_OVERLAP and p["fixture_id"] in core_fixture_ids:
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_FUN):
            continue
        if not confidence_ok(fx, FUN_MIN_CONFIDENCE):
            continue
        if safe_float(p.get("ev"), -9999.0) < FUN_EV_MIN:
            continue
        candidates.append(p)

    candidates.sort(key=lambda x: (safe_float(x.get("ev"), -9999.0), safe_float(x.get("confidence"), 0.0), safe_float(x.get("prob"), 0.0)), reverse=True)

    pool = []
    used = set()
    for p in candidates:
        if p["match"] in used:
            continue
        pool.append(p)
        used.add(p["match"])
        if len(pool) >= FUN_PICKS_MAX:
            break

    # choose n preference: 7 > 6 > 5
    if len(pool) >= 7:
        sys_pool = pool[:7]
    elif len(pool) >= 6:
        sys_pool = pool[:6]
    else:
        sys_pool = pool[:5]

    sys_choice, thr_used, breached = choose_fun_system(sys_pool)

    cavg, cband = conf_avg(sys_pool)

    unit = stake = 0.0
    mult_used = 1.0
    target_total = final_total = 0.0

    if sys_choice:
        unit, stake, mult_used, target_total, final_total = system_unit(
            columns=int(sys_choice["columns"]),
            cap_system=cap_total,
            threshold_used=thr_used,
            breached=breached,
            cavg=cavg,
            cband=cband
        )

    payload = {
        "portfolio": "FunBet",
        "bankroll": bankroll_fun,
        "exposure_cap_pct": FUN_EXPOSURE_CAP,
        "rules": {
            "max_picks": FUN_PICKS_MAX,
            "refund_primary": SYS_REFUND_PRIMARY,
            "refund_fallback": SYS_REFUND_FALLBACK,
            "refund_used": thr_used,
            "refund_rule_breached": bool(breached),
            "odds_cap": MAX_ODDS_CAP,
        },
        "picks_total": [strip_pick(p) for p in pool],
        "system_pool": [strip_pick(p) for p in sys_pool],
        "system": {
            "label": (sys_choice["label"] if sys_choice else None),
            "columns": (int(sys_choice["columns"]) if sys_choice else 0),
            "min_hits": (int(sys_choice["min_hits"]) if sys_choice else None),
            "refund_ratio_min_hits": (float(sys_choice["refund_ratio_min_hits"]) if sys_choice else None),
            "unit": unit,
            "stake": stake,
            "target_total": target_total,
            "final_total": final_total,
            "confidence_avg": cavg,
            "confidence_band": cband,
            "confidence_multiplier_used": mult_used,
            "refund_used": thr_used,
            "refund_rule_breached": bool(breached),
            "has_system": bool(sys_choice is not None),
        },
        "open": round(stake, 2),
        "after_open": round(bankroll_fun - stake, 2),
        "counts": {"picks_total": len(pool), "system_pool": len(sys_pool)},
        "singles": [],  # ✅ system-only
    }
    return payload

# ------------------------- DRAWBET (3 DRAWS, SYSTEM ONLY) -------------------------
DRAW_PICKS = 3
DRAW_ODDS_MIN = float(os.getenv("DRAW_ODDS_MIN", "3.10"))
DRAW_ODDS_MAX = float(os.getenv("DRAW_ODDS_MAX", "3.60"))  # ✅ keep under global cap
DRAW_EV_MIN   = float(os.getenv("DRAW_EV_MIN", "0.07"))
DRAW_P_MIN    = float(os.getenv("DRAW_P_MIN", "0.30"))
DRAW_LTOTAL_MAX = float(os.getenv("DRAW_LTOTAL_MAX", "2.55"))
DRAW_ABS_GAP_MAX = float(os.getenv("DRAW_ABS_GAP_MAX", "0.30"))

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

        odds = safe_float(p.get("odds"), None)
        evv  = safe_float(p.get("ev"), None)
        pr   = safe_float(p.get("prob"), None)
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

    candidates.sort(key=lambda x: (safe_float(x.get("ev"), -9999.0), safe_float(x.get("confidence"), 0.0), safe_float(x.get("prob"), 0.0)), reverse=True)

    pool = []
    used = set()
    for p in candidates:
        if p["match"] in used:
            continue
        pool.append(p); used.add(p["match"])
        if len(pool) >= DRAW_PICKS:
            break

    has_system = (len(pool) == 3)
    columns = comb(3, 2) if has_system else 0
    label = "2/3" if has_system else None

    # target total smaller (draws are volatile)
    target_total = min(25.0, cap_total) if has_system else 0.0
    unit = round(target_total / columns, 2) if columns > 0 else 0.0
    stake = round(unit * columns, 2) if columns > 0 else 0.0

    payload = {
        "portfolio": "DrawBet",
        "bankroll": bankroll_draw,
        "exposure_cap_pct": DRAW_EXPOSURE_CAP,
        "rules": {"picks": 3, "system": "2/3", "odds_range": [DRAW_ODDS_MIN, DRAW_ODDS_MAX]},
        "picks_total": [strip_pick(p) for p in pool],
        "system_pool": [strip_pick(p) for p in pool],
        "system": {"label": label, "columns": columns, "unit": unit, "stake": stake, "has_system": bool(has_system)},
        "open": stake,
        "after_open": round(bankroll_draw - stake, 2),
        "counts": {"picks_total": len(pool), "system_pool": len(pool)},
        "singles": [],  # ✅ none
    }
    return payload

# ------------------------- MAIN -------------------------
def main():
    fixtures, th_meta = load_thursday_fixtures()
    history = load_history()

    window = th_meta.get("window", {}) if isinstance(th_meta, dict) else {}
    wf = get_week_fields(window, history)

    core_start = safe_float(history.get("core", {}).get("bankroll_current"), None)
    fun_start  = safe_float(history.get("funbet", {}).get("bankroll_current"), None)
    draw_start = safe_float(history.get("drawbet", {}).get("bankroll_current"), None)

    core_bankroll_start = core_start if core_start is not None else DEFAULT_BANKROLL_CORE
    fun_bankroll_start  = fun_start if fun_start is not None else DEFAULT_BANKROLL_FUN
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
            "rules": {
                "odds_cap": MAX_ODDS_CAP,
                "singles_odds_range": [CORE_SINGLES_MIN_ODDS, CORE_SINGLES_MAX_ODDS],
                "stake_ladder": {"1.70-1.90": 40, "1.90-2.20": 30, "2.20-3.00": 20, "3.00-3.60": 15},
                "no_singles_below": 1.70,
                "low_odds_to_doubles": [CORE_LOW_ODDS_MIN, CORE_LOW_ODDS_MAX],
                "max_singles": CORE_MAX_SINGLES,
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
