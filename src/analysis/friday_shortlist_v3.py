# filename: src/analysis/friday_shortlist_v3.py
# ============================================================
# FRIDAY SHORTLIST v3.33 — CoreBet + FunBet + DrawBet (PRODUCTION)
#
# Fixes:
# - Strict odds gating fallback when Thursday lacks flags.odds_strict_ok
#   (prevents Core/Draw from going to zero).
# - EV filters are OPTIONAL (default OFF).
# - copy_play has NO null date/time/league for Double/System lines.
# ============================================================

import os
import json
from datetime import datetime, date
from math import comb

THURSDAY_REPORT_PATH = os.getenv("THURSDAY_REPORT_PATH", "logs/thursday_report_v3.json")
FRIDAY_REPORT_PATH   = os.getenv("FRIDAY_REPORT_PATH",   "logs/friday_shortlist_v3.json")
TUESDAY_HISTORY_PATH = os.getenv("TUESDAY_HISTORY_PATH", "logs/tuesday_history_v3.json")

DEFAULT_BANKROLL_CORE = float(os.getenv("BANKROLL_CORE", "800"))
DEFAULT_BANKROLL_FUN  = float(os.getenv("BANKROLL_FUN",  "400"))
DEFAULT_BANKROLL_DRAW = float(os.getenv("BANKROLL_DRAW", "300"))

# exposure caps
CORE_EXPOSURE_CAP = float(os.getenv("CORE_EXPOSURE_CAP", "0.30"))
FUN_EXPOSURE_CAP  = float(os.getenv("FUN_EXPOSURE_CAP",  "0.25"))
DRAW_EXPOSURE_CAP = float(os.getenv("DRAW_EXPOSURE_CAP", "0.20"))

# gates
ODDS_MATCH_MIN_SCORE_CORE = float(os.getenv("ODDS_MATCH_MIN_SCORE_CORE", "0.70"))
ODDS_MATCH_MIN_SCORE_FUN  = float(os.getenv("ODDS_MATCH_MIN_SCORE_FUN",  "0.70"))
ODDS_MATCH_MIN_SCORE_DRAW = float(os.getenv("ODDS_MATCH_MIN_SCORE_DRAW", "0.70"))

CORE_MIN_CONFIDENCE = float(os.getenv("CORE_MIN_CONFIDENCE", "0.50"))
FUN_MIN_CONFIDENCE  = float(os.getenv("FUN_MIN_CONFIDENCE",  "0.45"))
DRAW_MIN_CONFIDENCE = float(os.getenv("DRAW_MIN_CONFIDENCE", "0.50"))

# overlap
FUN_AVOID_CORE_OVERLAP = os.getenv("FUN_AVOID_CORE_OVERLAP", "false").lower() == "true"

# hard caps
HARD_MAX_ODDS = float(os.getenv("HARD_MAX_ODDS", "3.50"))

# strict odds (Friday side)
CORE_REQUIRE_STRICT_ODDS = os.getenv("CORE_REQUIRE_STRICT_ODDS", "false").lower() == "true"
FUN_REQUIRE_STRICT_ODDS  = os.getenv("FUN_REQUIRE_STRICT_ODDS",  "false").lower() == "true"
DRAW_REQUIRE_STRICT_ODDS = os.getenv("DRAW_REQUIRE_STRICT_ODDS", "false").lower() == "true"

# IMPORTANT: fallback strict threshold when Thursday flags.odds_strict_ok is missing
STRICT_ODDS_SCORE = float(os.getenv("STRICT_ODDS_SCORE", "0.80"))

# instability brake
USE_INSTABILITY_BRAKE = os.getenv("USE_INSTABILITY_BRAKE", "false").lower() == "true"
MAX_PROB_INSTABILITY_CORE = float(os.getenv("MAX_PROB_INSTABILITY_CORE", "0.18"))
MAX_PROB_INSTABILITY_FUN  = float(os.getenv("MAX_PROB_INSTABILITY_FUN",  "0.22"))
MAX_PROB_INSTABILITY_DRAW = float(os.getenv("MAX_PROB_INSTABILITY_DRAW", "0.16"))

# OPTIONAL EV filters (default OFF)
ENABLE_EV_FILTERS = os.getenv("ENABLE_EV_FILTERS", "false").lower() == "true"
FUN_MAX_EV  = float(os.getenv("FUN_MAX_EV", "0.25"))
DRAW_MIN_EV = float(os.getenv("DRAW_MIN_EV", "0.03"))

MARKET_NAME = {"1":"Home","X":"Draw","2":"Away","O25":"Over 2.5","U25":"Under 2.5"}

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

def load_thursday_fixtures():
    data = load_json(THURSDAY_REPORT_PATH)
    if isinstance(data, dict) and "fixtures" in data:
        return data["fixtures"], data
    if isinstance(data, dict) and isinstance(data.get("report"), dict) and "fixtures" in data["report"]:
        return data["report"]["fixtures"], data["report"]
    raise KeyError("fixtures not found in Thursday report")

def _strict_ok_flag_or_none(fx):
    flags = fx.get("flags") or {}
    # returns True/False/None
    return flags.get("odds_strict_ok")

def odds_match_ok(fx, min_score, require_strict: bool):
    om = fx.get("odds_match") or {}
    if not om.get("matched"):
        return False

    sc = safe_float(om.get("score"), 0.0) or 0.0
    if sc < min_score:
        return False

    if require_strict:
        strict_flag = _strict_ok_flag_or_none(fx)
        if strict_flag is True:
            return True
        # if flag missing (old Thursday logs) OR flag false -> fallback to strict-by-score
        return sc >= STRICT_ODDS_SCORE

    return True

def confidence_ok(fx, min_conf):
    c = safe_float((fx.get("flags") or {}).get("confidence"), None)
    if c is None:
        return True
    return c >= min_conf

def instability_ok(fx, max_gap: float):
    if not USE_INSTABILITY_BRAKE:
        return True
    gap = safe_float((fx.get("flags") or {}).get("prob_instability"), None)
    if gap is None:
        return True
    return gap <= max_gap

def _ev_from_fx(fx, code: str):
    return safe_float({
        "1": fx.get("ev_1"),
        "2": fx.get("ev_2"),
        "X": fx.get("ev_x"),
        "O25": fx.get("ev_over"),
        "U25": fx.get("ev_under"),
    }.get(code), None)

def _prob_from_fx(fx, code: str):
    return safe_float({
        "1": fx.get("home_prob"),
        "2": fx.get("away_prob"),
        "X": fx.get("draw_prob"),
        "O25": fx.get("over_2_5_prob"),
        "U25": fx.get("under_2_5_prob"),
    }.get(code), None)

def _odds_from_fx(fx, code: str):
    return safe_float({
        "1": fx.get("offered_1"),
        "2": fx.get("offered_2"),
        "X": fx.get("offered_x"),
        "O25": fx.get("offered_over_2_5"),
        "U25": fx.get("offered_under_2_5"),
    }.get(code), None)

def build_candidates(fixtures):
    out = []
    for fx in fixtures:
        for code in ["1","2","X","O25","U25"]:
            odds = _odds_from_fx(fx, code)
            if odds is None or odds <= 1.0:
                continue
            if odds > HARD_MAX_ODDS:
                continue
            out.append({
                "fixture_id": fx.get("fixture_id"),
                "date": fx.get("date"),
                "time": fx.get("time"),
                "league": fx.get("league"),
                "match": f'{fx.get("home")} – {fx.get("away")}',
                "market_code": code,
                "market": MARKET_NAME.get(code, code),
                "odds": odds,
                "prob": _prob_from_fx(fx, code),
                "ev": _ev_from_fx(fx, code),
                "confidence": safe_float((fx.get("flags") or {}).get("confidence"), None),
                "fx": fx,
            })
    return out

# ---------------- CORE RULES ----------------
CORE_MIN_ODDS = float(os.getenv("CORE_MIN_ODDS", "1.60"))
CORE_MAX_ODDS = float(os.getenv("CORE_MAX_ODDS", "2.10"))
CORE_LOW_MIN  = float(os.getenv("CORE_LOW_MIN", "1.30"))
CORE_LOW_MAX  = float(os.getenv("CORE_LOW_MAX", "1.60"))
CORE_MAX_SINGLES = int(os.getenv("CORE_MAX_SINGLES", "8"))

def core_stake_ladder(odds: float) -> float:
    if 1.60 <= odds <= 1.75:
        return 40.0
    if 1.75 < odds <= 1.90:
        return 30.0
    if 1.90 < odds <= 2.10:
        return 20.0
    return 0.0

def core_double_stake(_combo: float) -> float:
    return 15.0

def pick_core(cands, bankroll_core):
    cap = bankroll_core * CORE_EXPOSURE_CAP

    singles_pool = []
    low_pool = []
    for p in cands:
        fx = p["fx"]
        if p["market_code"] == "X":
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_CORE, CORE_REQUIRE_STRICT_ODDS):
            continue
        if not confidence_ok(fx, CORE_MIN_CONFIDENCE):
            continue
        if not instability_ok(fx, MAX_PROB_INSTABILITY_CORE):
            continue

        odds = safe_float(p["odds"], None)
        evv  = safe_float(p["ev"], None)
        pr   = safe_float(p["prob"], None)
        if odds is None or evv is None or pr is None:
            continue

        if p["market_code"] == "U25" and pr < 0.58:
            continue

        if CORE_MIN_ODDS <= odds <= CORE_MAX_ODDS:
            st = core_stake_ladder(odds)
            if st > 0:
                singles_pool.append({**p, "stake": st})
        elif CORE_LOW_MIN <= odds < CORE_LOW_MAX:
            low_pool.append(p)

    def k(x):
        return (safe_float(x.get("ev"), -9999.0), safe_float(x.get("confidence"), 0.0),
                safe_float(x.get("prob"), 0.0), -safe_float(x.get("odds"), 99.0))

    singles_pool.sort(key=k, reverse=True)
    low_pool.sort(key=k, reverse=True)

    singles = []
    used = set()
    for p in singles_pool:
        if len(singles) >= CORE_MAX_SINGLES:
            break
        if p["match"] in used:
            continue
        singles.append(_strip_pick(p, tag="core_single"))
        used.add(p["match"])

    doubles = []
    if low_pool:
        partner_pool = singles_pool[:]
        partner_pool.sort(key=k, reverse=True)
        used_d = set()
        for leg1 in low_pool:
            if len(doubles) >= 2:
                break
            for leg2 in partner_pool:
                if leg2["match"] == leg1["match"]:
                    continue
                if leg1["match"] in used_d or leg2["match"] in used_d:
                    continue
                combo = safe_float(leg1["odds"], 1.0) * safe_float(leg2["odds"], 1.0)
                if combo < 2.0:
                    continue
                doubles.append({
                    "legs": [_strip_leg(leg1), _strip_leg(leg2)],
                    "combo_odds": round(combo, 2),
                    "stake": core_double_stake(combo),
                    "tag": "core_double",
                })
                used_d.add(leg1["match"]); used_d.add(leg2["match"])
                break

    open_total = sum(x["stake"] for x in singles) + sum(d["stake"] for d in doubles)
    scale = 1.0
    if cap > 0 and open_total > cap and open_total > 0:
        scale = cap / open_total
        for x in singles:
            x["stake"] = round(x["stake"] * scale, 2)
        for d in doubles:
            d["stake"] = round(d["stake"] * scale, 2)
        open_total = sum(x["stake"] for x in singles) + sum(d["stake"] for d in doubles)

    meta = {"open": round(open_total, 2), "after_open": round(bankroll_core - open_total, 2), "scale": round(scale, 3)}
    return singles, (doubles[0] if doubles else None), doubles, meta

# ---------------- FUN RULES (SYSTEM ONLY) ----------------
FUN_MIN_PICKS = int(os.getenv("FUN_MIN_PICKS", "5"))
FUN_MAX_PICKS = int(os.getenv("FUN_MAX_PICKS", "7"))

def fun_pick_filter(p):
    return p["market_code"] != "X"

def choose_fun_system(pool):
    n = len(pool)
    if n >= 7:
        r, label = 4, "4/7"
    elif n == 6:
        r, label = 3, "3/6"
    elif n == 5:
        r, label = 3, "3/5"
    else:
        return None
    return {"label": label, "min_hits": r, "columns": comb(n, r)}

def pick_fun(cands, bankroll_fun, core_fixture_ids):
    cap = bankroll_fun * FUN_EXPOSURE_CAP

    pool_cands = []
    for p in cands:
        fx = p["fx"]
        if not fun_pick_filter(p):
            continue
        if FUN_AVOID_CORE_OVERLAP and p["fixture_id"] in core_fixture_ids:
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_FUN, FUN_REQUIRE_STRICT_ODDS):
            continue
        if not confidence_ok(fx, FUN_MIN_CONFIDENCE):
            continue
        if not instability_ok(fx, MAX_PROB_INSTABILITY_FUN):
            continue
        if p["odds"] > HARD_MAX_ODDS:
            continue

        evv = safe_float(p.get("ev"), None)
        if evv is None:
            continue
        if ENABLE_EV_FILTERS and evv > FUN_MAX_EV:
            continue

        pool_cands.append(p)

    def k(x):
        return (safe_float(x.get("ev"), -9999.0), safe_float(x.get("confidence"), 0.0),
                safe_float(x.get("prob"), 0.0), -safe_float(x.get("odds"), 99.0))

    pool_cands.sort(key=k, reverse=True)

    picks = []
    used = set()
    for p in pool_cands:
        if p["match"] in used:
            continue
        picks.append(p); used.add(p["match"])
        if len(picks) >= FUN_MAX_PICKS:
            break

    if len(picks) >= 7:
        pool = picks[:7]
    elif len(picks) >= 6:
        pool = picks[:6]
    else:
        pool = picks[:5]

    # cap: max 2 odds > 3.00 in FINAL pool
    if len(pool) >= 7:
        high = [p for p in pool if safe_float(p.get("odds"), 0.0) > 3.00]
        if len(high) > 2:
            keep = []
            high_cnt = 0
            for p in pool:
                if safe_float(p.get("odds"), 0.0) > 3.00:
                    if high_cnt < 2:
                        keep.append(p); high_cnt += 1
                    else:
                        continue
                else:
                    keep.append(p)

            used_matches = {p["match"] for p in keep}
            for cand in pool_cands:
                if len(keep) >= 7:
                    break
                if cand["match"] in used_matches:
                    continue
                if safe_float(cand.get("odds"), 0.0) > 3.00:
                    continue
                keep.append(cand)
                used_matches.add(cand["match"])
            pool = keep[:7]

    sys = choose_fun_system(pool)
    stake_total = 0.0
    unit = 0.0
    if sys:
        cols = sys["columns"]
        target = float(cols)
        if target < 25.0:
            target = 25.0
        if target > 50.0:
            target = 50.0
        if cap > 0:
            target = min(target, cap)
        unit = round(target / cols, 2) if cols else 0.0
        stake_total = round(unit * cols, 2)

    return {
        "bankroll": bankroll_fun,
        "exposure_cap_pct": FUN_EXPOSURE_CAP,
        "system_pool": [_strip_pick(p, tag="fun_pool") for p in pool],
        "picks_total": [_strip_pick(p, tag="fun_pick") for p in picks],
        "system": {
            "label": (sys["label"] if sys else None),
            "columns": (sys["columns"] if sys else None),
            "min_hits": (sys["min_hits"] if sys else None),
            "unit": unit if sys else None,
            "stake": stake_total if sys else None,
        },
        "open": stake_total,
        "after_open": round(bankroll_fun - stake_total, 2),
        "counts": {"picks_total": len(picks), "system_pool": len(pool)},
    }

# ---------------- DRAW RULES (SYSTEM ONLY) ----------------
DRAW_ODDS_MIN = float(os.getenv("DRAW_ODDS_MIN", "2.80"))
DRAW_ODDS_MAX = float(os.getenv("DRAW_ODDS_MAX", "3.70"))
DRAW_MIN_DRAWS = int(os.getenv("DRAW_MIN_DRAWS", "2"))
DRAW_MAX_DRAWS = int(os.getenv("DRAW_MAX_DRAWS", "5"))

def choose_draw_system(n: int):
    if n >= 5:
        return {"label":"2/5", "min_hits":2, "columns": comb(5,2)}
    if n == 4:
        return {"label":"2/4", "min_hits":2, "columns": comb(4,2)}
    if n == 3:
        return {"label":"2/3", "min_hits":2, "columns": comb(3,2)}
    return None

def pick_draw(cands, bankroll_draw):
    cap = bankroll_draw * DRAW_EXPOSURE_CAP

    draws = []
    for p in cands:
        fx = p["fx"]
        if p["market_code"] != "X":
            continue
        if not odds_match_ok(fx, ODDS_MATCH_MIN_SCORE_DRAW, DRAW_REQUIRE_STRICT_ODDS):
            continue
        if not confidence_ok(fx, DRAW_MIN_CONFIDENCE):
            continue
        if not instability_ok(fx, MAX_PROB_INSTABILITY_DRAW):
            continue
        if p["odds"] < DRAW_ODDS_MIN or p["odds"] > DRAW_ODDS_MAX:
            continue
        if safe_float(p.get("prob"), 0.0) < float(os.getenv("DRAW_P_MIN", "0.27")):
            continue

        evv = safe_float(p.get("ev"), None)
        if evv is None:
            continue
        if ENABLE_EV_FILTERS and evv < DRAW_MIN_EV:
            continue

        draws.append(p)

    def k(x):
        return (safe_float(x.get("ev"), -9999.0), safe_float(x.get("prob"), 0.0),
                safe_float(x.get("confidence"), 0.0), -safe_float(x.get("odds"), 99.0))
    draws.sort(key=k, reverse=True)

    pool = []
    used = set()
    for p in draws:
        if p["match"] in used:
            continue
        pool.append(p); used.add(p["match"])
        if len(pool) >= DRAW_MAX_DRAWS:
            break

    if len(pool) < DRAW_MIN_DRAWS:
        pool = pool[:len(pool)]

    sys = choose_draw_system(len(pool))
    unit = 0.0
    stake_total = 0.0
    if sys:
        cols = sys["columns"]
        target = 25.0
        if cap > 0:
            target = min(target, cap)
        unit = round(target / cols, 2) if cols else 0.0
        stake_total = round(unit * cols, 2)

    return {
        "bankroll": bankroll_draw,
        "exposure_cap_pct": DRAW_EXPOSURE_CAP,
        "system_pool": [_strip_pick(p, tag="draw_pool") for p in pool],
        "picks_total": [_strip_pick(p, tag="draw_pick") for p in pool],
        "system": {
            "label": (sys["label"] if sys else None),
            "columns": (sys["columns"] if sys else None),
            "min_hits": (sys["min_hits"] if sys else None),
            "unit": unit if sys else None,
            "stake": stake_total if sys else None,
        },
        "open": stake_total,
        "after_open": round(bankroll_draw - stake_total, 2),
        "counts": {"picks_total": len(pool), "system_pool": len(pool)},
    }

def _strip_pick(p, tag="pick"):
    return {
        "pick_id": f'{p.get("fixture_id")}:{p.get("market_code")}',
        "fixture_id": p.get("fixture_id"),
        "date": p.get("date"),
        "time": p.get("time"),
        "league": p.get("league"),
        "match": p.get("match"),
        "market_code": p.get("market_code"),
        "market": p.get("market"),
        "odds": round(float(p.get("odds") or 0.0), 2),
        "prob": p.get("prob"),
        "ev": p.get("ev"),
        "confidence": p.get("confidence"),
        "stake": round(float(p.get("stake") or 0.0), 2) if "stake" in p else None,
        "tag": tag,
    }

def _strip_leg(p):
    return {
        "pick_id": f'{p.get("fixture_id")}:{p.get("market_code")}',
        "fixture_id": p.get("fixture_id"),
        "date": p.get("date"),
        "time": p.get("time"),
        "league": p.get("league"),
        "match": p.get("match"),
        "market": p.get("market"),
        "market_code": p.get("market_code"),
        "odds": round(float(p.get("odds") or 0.0), 2),
    }

def _safe_window_placeholders(window: dict):
    w_to = (window or {}).get("to") or (window or {}).get("from") or datetime.utcnow().date().isoformat()
    return (w_to, "23:59", "MULTI")

def _legs_summary(legs: list):
    if not legs:
        return ""
    parts = []
    for lg in legs[:2]:
        parts.append(f'{lg.get("match","?")} {lg.get("market_code","?")}@{lg.get("odds","?")}')
    return " | ".join(parts)

def build_copy_play(core, fun, draw, window: dict):
    lines = []

    for p in (core.get("singles") or []):
        lines.append({
            "date": p.get("date"),
            "time": p.get("time"),
            "league": p.get("league"),
            "match": p.get("match"),
            "market": p.get("market"),
            "odds": p.get("odds"),
            "stake": p.get("stake"),
            "portfolio": "CoreBet",
            "_sort_key": f"{p.get('date','')} {p.get('time') or '00:00'}",
        })

    ph_date, ph_time, ph_league = _safe_window_placeholders(window)

    for d in (core.get("doubles") or []):
        legs = d.get("legs") or []
        combo_odds = d.get("combo_odds")
        summary = _legs_summary(legs)
        lines.append({
            "date": ph_date,
            "time": ph_time,
            "league": ph_league,
            "match": "Double" if not summary else f"Double: {summary}",
            "market": f'Combo {combo_odds}',
            "odds": combo_odds,
            "stake": d.get("stake"),
            "portfolio": "CoreBet",
            "_sort_key": f"{ph_date} {ph_time}",
        })

    fs = (fun.get("system") or {})
    if fs and fs.get("label"):
        lines.append({
            "date": ph_date,
            "time": ph_time,
            "league": ph_league,
            "match": "System",
            "market": f'FunBet {fs.get("label")}',
            "odds": None,
            "stake": fs.get("stake"),
            "portfolio": "FunBet",
            "_sort_key": f"{ph_date} {ph_time}",
        })

    ds = (draw.get("system") or {})
    if ds and ds.get("label"):
        lines.append({
            "date": ph_date,
            "time": ph_time,
            "league": ph_league,
            "match": "System",
            "market": f'DrawBet {ds.get("label")}',
            "odds": None,
            "stake": ds.get("stake"),
            "portfolio": "DrawBet",
            "_sort_key": f"{ph_date} {ph_time}",
        })

    lines.sort(key=lambda x: x.get("_sort_key") or "9999-12-31 23:59")
    for x in lines:
        x.pop("_sort_key", None)
    return lines

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

    cands = build_candidates(fixtures)

    core_singles, core_double, core_doubles, core_meta = pick_core(cands, core_bankroll_start)
    core_fixture_ids = {p.get("fixture_id") for p in core_singles if p.get("fixture_id") is not None}

    fun_payload = pick_fun(cands, fun_bankroll_start, core_fixture_ids)
    draw_payload = pick_draw(cands, draw_bankroll_start)

    core_section = {
        "portfolio": "CoreBet",
        "bankroll": DEFAULT_BANKROLL_CORE,
        "bankroll_start": core_bankroll_start,
        "bankroll_source": ("history" if core_start is not None else "default"),
        "singles": core_singles,
        "double": core_double,
        "doubles": core_doubles,
        "open": core_meta["open"],
        "after_open": core_meta["after_open"],
    }

    fun_section = {
        "portfolio": "FunBet",
        "bankroll": DEFAULT_BANKROLL_FUN,
        "bankroll_start": fun_bankroll_start,
        "bankroll_source": ("history" if fun_start is not None else "default"),
        **fun_payload
    }

    draw_section = {
        "portfolio": "DrawBet",
        "bankroll": DEFAULT_BANKROLL_DRAW,
        "bankroll_start": draw_bankroll_start,
        "bankroll_source": ("history" if draw_start is not None else "default"),
        **draw_payload
    }

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "week_id": wf["week_id"],
        "week_no": wf["week_no"],
        "week_label": wf["week_label"],
        "window": window,
        "fixtures_total": th_meta.get("fixtures_total", len(fixtures)),
        "core": core_section,
        "funbet": fun_section,
        "drawbet": draw_section,
    }

    report["copy_play"] = build_copy_play(core_section, fun_section, draw_section, window)

    os.makedirs("logs", exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
