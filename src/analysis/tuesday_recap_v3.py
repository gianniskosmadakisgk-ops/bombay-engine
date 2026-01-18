# ============================================================
#  src/analysis/tuesday_recap_v3.py
#  TUESDAY RECAP v3.30 — RESET STABLE (Core + Fun + Draw + Lifetime)
#
#  Reads:  logs/friday_shortlist_v3.json
#  Writes: logs/tuesday_recap_v3.json
#          logs/tuesday_history_v3.json (carry bankroll_current)
#
#  Rule:
#   - If any pick is not finished -> PENDING and history NOT updated
# ============================================================

import os
import json
import time
import requests
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple
from itertools import combinations

API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

FRIDAY_REPORT_PATH = os.getenv("FRIDAY_REPORT_PATH", "logs/friday_shortlist_v3.json")
TUESDAY_RECAP_PATH = os.getenv("TUESDAY_RECAP_PATH", "logs/tuesday_recap_v3.json")
TUESDAY_HISTORY_PATH = os.getenv("TUESDAY_HISTORY_PATH", "logs/tuesday_history_v3.json")

REQUEST_TIMEOUT = int(os.getenv("API_TIMEOUT_SEC", "25"))
REQUEST_SLEEP_MS = int(os.getenv("API_SLEEP_MS", "120"))
MAX_FETCH_RETRIES = int(os.getenv("MAX_FETCH_RETRIES", "2"))

FINAL_STATUSES = {"FT", "AET", "PEN"}

def log(msg: str):
    print(msg, flush=True)

def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

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

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

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

def load_history(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {
            "week_count": 0,
            "last_window_from": None,
            "weeks": {},
            "core": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
            "funbet": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
            "drawbet": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
            "updated_at": now_utc_iso(),
        }
    try:
        h = load_json(path)
        h.setdefault("week_count", 0)
        h.setdefault("last_window_from", None)
        h.setdefault("weeks", {})
        h.setdefault("core", {}).setdefault("stake", 0.0)
        h.setdefault("core", {}).setdefault("profit", 0.0)
        h.setdefault("core", {}).setdefault("bankroll_current", None)
        h.setdefault("funbet", {}).setdefault("stake", 0.0)
        h.setdefault("funbet", {}).setdefault("profit", 0.0)
        h.setdefault("funbet", {}).setdefault("bankroll_current", None)
        h.setdefault("drawbet", {}).setdefault("stake", 0.0)
        h.setdefault("drawbet", {}).setdefault("profit", 0.0)
        h.setdefault("drawbet", {}).setdefault("bankroll_current", None)
        h.setdefault("updated_at", now_utc_iso())
        return h
    except Exception:
        return {
            "week_count": 0,
            "last_window_from": None,
            "weeks": {},
            "core": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
            "funbet": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
            "drawbet": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
            "updated_at": now_utc_iso(),
        }

def _api_get_fixture(fixture_id: int) -> Optional[Dict[str, Any]]:
    if not API_FOOTBALL_KEY:
        return None
    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"id": fixture_id}
    for _attempt in range(MAX_FETCH_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                time.sleep(0.25)
                continue
            js = r.json()
            resp = js.get("response") or []
            if not resp:
                return None
            return resp[0]
        except Exception:
            time.sleep(0.25)
    return None

def fetch_scores(fixture_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    uniq = sorted({int(x) for x in fixture_ids if x is not None})
    if not uniq:
        return out
    if not API_FOOTBALL_KEY:
        log("❌ Missing FOOTBALL_API_KEY – cannot fetch results.")
        return out
    for fid in uniq:
        fx = _api_get_fixture(fid)
        if fx is None:
            continue
        status_short = (((fx.get("fixture") or {}).get("status") or {}).get("short")) or ""
        goals = fx.get("goals") or {}
        out[fid] = {
            "status_short": str(status_short).upper(),
            "home_goals": safe_int(goals.get("home"), None),
            "away_goals": safe_int(goals.get("away"), None),
        }
        time.sleep(max(0.0, REQUEST_SLEEP_MS / 1000.0))
    return out

def market_won(code: str, hg: int, ag: int) -> bool:
    code = (code or "").upper().strip()
    if code == "1": return hg > ag
    if code == "X": return hg == ag
    if code == "2": return ag > hg
    if code == "O25": return (hg + ag) >= 3
    if code == "U25": return (hg + ag) <= 2
    return False

def eval_single(pick: Dict[str, Any], scores: Dict[int, Dict[str, Any]]) -> Tuple[str, float]:
    fid = safe_int(pick.get("fixture_id"), None)
    mcode = (pick.get("market_code") or "").upper().strip()
    stake = safe_float(pick.get("stake"), 0.0) or 0.0
    odds = safe_float(pick.get("odds"), None)
    if fid is None or not mcode:
        return "PENDING", 0.0
    sc = scores.get(fid)
    if not sc:
        return "PENDING", 0.0
    st = (sc.get("status_short") or "").upper()
    hg = sc.get("home_goals")
    ag = sc.get("away_goals")
    if st not in FINAL_STATUSES or hg is None or ag is None:
        return "PENDING", 0.0
    won = market_won(mcode, int(hg), int(ag))
    if odds is None or odds <= 1.0:
        return ("WIN" if won else "LOSE"), 0.0
    if won:
        return "WIN", round(stake * (odds - 1.0), 2)
    return "LOSE", round(-stake, 2)

def eval_double(double_obj: Dict[str, Any], scores: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], str, float]:
    if not double_obj:
        return {}, "PENDING", 0.0
    legs = double_obj.get("legs") or []
    stake = safe_float(double_obj.get("stake"), 0.0) or 0.0
    mult = 1.0
    any_pending = False
    any_lose = False
    legs_out = []

    for leg in legs:
        code = (leg.get("market_code") or "").upper().strip()
        fid = safe_int(leg.get("fixture_id"), None)
        odds = safe_float(leg.get("odds"), None)

        if fid is None or not code:
            any_pending = True
            legs_out.append({**leg, "result": "PENDING"})
            continue

        sc = scores.get(fid)
        if not sc:
            any_pending = True
            legs_out.append({**leg, "result": "PENDING"})
            continue

        st = (sc.get("status_short") or "").upper()
        hg = sc.get("home_goals")
        ag = sc.get("away_goals")
        if st not in FINAL_STATUSES or hg is None or ag is None:
            any_pending = True
            legs_out.append({**leg, "result": "PENDING"})
            continue

        won = market_won(code, int(hg), int(ag))
        if not won:
            any_lose = True
            legs_out.append({**leg, "result": "LOSE"})
        else:
            legs_out.append({**leg, "result": "WIN"})
            if odds and odds > 1.0:
                mult *= float(odds)

    if any_pending:
        status = "PENDING"
        profit = 0.0
    elif any_lose:
        status = "LOSE"
        profit = round(-stake, 2) if stake > 0 else 0.0
    else:
        status = "WIN"
        profit = round((stake * mult) - stake, 2) if stake > 0 else 0.0

    out = {**double_obj, "legs": legs_out, "result": status, "profit": profit}
    return out, status, profit

def parse_system_sizes(label: str) -> List[int]:
    if not label:
        return []
    left = label.split("/")[0].strip()
    parts = [p.strip() for p in left.split("-") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            pass
    return out

def eval_system(system: Dict[str, Any], pool: List[Dict[str, Any]], scores: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], str, float]:
    if not system or not pool:
        return system or {}, "PENDING", 0.0

    label = system.get("label")
    sizes = parse_system_sizes(label or "")
    n = len(pool)

    cols = safe_int(system.get("columns"), 0) or 0
    unit = safe_float(system.get("unit"), 0.0) or 0.0
    stake = safe_float(system.get("stake"), 0.0) or 0.0

    if not sizes or cols <= 0 or unit <= 0 or stake <= 0:
        return {**system, "result": "PENDING", "profit": 0.0}, "PENDING", 0.0

    # determine leg outcomes
    leg_state = []
    for r in pool:
        fid = safe_int(r.get("fixture_id"), None)
        code = (r.get("market_code") or "").upper().strip()
        odds = safe_float(r.get("odds"), None)
        if fid is None or not code:
            leg_state.append(("PENDING", 1.0))
            continue
        sc = scores.get(fid)
        if not sc:
            leg_state.append(("PENDING", 1.0))
            continue
        st = (sc.get("status_short") or "").upper()
        hg = sc.get("home_goals")
        ag = sc.get("away_goals")
        if st not in FINAL_STATUSES or hg is None or ag is None:
            leg_state.append(("PENDING", 1.0))
            continue
        won = market_won(code, int(hg), int(ag))
        if not won:
            leg_state.append(("LOSE", 1.0))
        else:
            leg_state.append(("WIN", float(odds) if odds and odds > 1.0 else 1.0))

    if any(st == "PENDING" for st, _m in leg_state):
        return {**system, "result": "PENDING", "profit": 0.0}, "PENDING", 0.0

    total_payout = 0.0
    idxs = list(range(n))
    for r in sizes:
        if r < 2 or r > n:
            continue
        for combo in combinations(idxs, r):
            if any(leg_state[i][0] == "LOSE" for i in combo):
                continue
            mult = 1.0
            for i in combo:
                mult *= leg_state[i][1]
            total_payout += unit * mult

    profit = round(total_payout - stake, 2)
    result = "WIN" if profit > 0 else "LOSE"

    out = {**system, "result": result, "profit": profit}
    return out, result, profit

def main():
    if not os.path.exists(FRIDAY_REPORT_PATH):
        raise FileNotFoundError(f"Friday report not found: {FRIDAY_REPORT_PATH}")

    friday = load_json(FRIDAY_REPORT_PATH)
    window = friday.get("window") or {}
    window_from = (window or {}).get("from")
    week_id = iso_week_id_from_window(window)

    history = load_history(TUESDAY_HISTORY_PATH)

    # week_no increments only if new window_from
    if window_from and window_from in (history.get("weeks") or {}):
        week_no = int(history["weeks"][window_from].get("week_no") or 1)
    else:
        week_no = int(history.get("week_count") or 0) + 1

    # bankroll starts: use friday bankroll_start if present
    core_bankroll_start = safe_float((friday.get("core") or {}).get("bankroll_start"), None)
    if core_bankroll_start is None:
        core_bankroll_start = safe_float((friday.get("core") or {}).get("bankroll"), 0.0) or 0.0

    fun_bankroll_start = safe_float((friday.get("funbet") or {}).get("bankroll_start"), None)
    if fun_bankroll_start is None:
        fun_bankroll_start = safe_float((friday.get("funbet") or {}).get("bankroll"), 0.0) or 0.0

    draw_bankroll_start = safe_float((friday.get("drawbet") or {}).get("bankroll_start"), None)
    if draw_bankroll_start is None:
        draw_bankroll_start = safe_float((friday.get("drawbet") or {}).get("bankroll"), 0.0) or 0.0

    core = friday.get("core") or {}
    fun  = friday.get("funbet") or {}
    draw = friday.get("drawbet") or {}

    core_singles = core.get("singles") or []
    core_double  = core.get("double") or {}
    core_doubles = core.get("doubles") or []

    fun_system   = fun.get("system") or {}
    fun_pool     = fun.get("system_pool") or []

    draw_system  = draw.get("system") or {}
    draw_pool    = draw.get("system_pool") or []

    fixture_ids = []
    for p in core_singles + fun_pool + draw_pool:
        fid = safe_int(p.get("fixture_id"), None)
        if fid is not None:
            fixture_ids.append(fid)
    # doubles legs
    for d in core_doubles:
        for leg in (d.get("legs") or []):
            fid = safe_int(leg.get("fixture_id"), None)
            if fid is not None:
                fixture_ids.append(fid)

    scores = fetch_scores(fixture_ids)

    # CORE singles
    core_out_singles = []
    core_stake = 0.0
    core_profit = 0.0
    core_pending = 0

    for p in core_singles:
        res, prof = eval_single(p, scores)
        stake = safe_float(p.get("stake"), 0.0) or 0.0
        core_stake += stake
        core_profit += prof
        if res == "PENDING":
            core_pending += 1
        core_out_singles.append({**p, "result": res, "profit": prof})

    # CORE doubles (list)
    core_out_doubles = []
    for d in core_doubles:
        out_d, st, prof = eval_double(d, scores)
        core_stake += safe_float(d.get("stake"), 0.0) or 0.0
        core_profit += prof
        if st == "PENDING":
            core_pending += 1
        core_out_doubles.append(out_d)

    # maintain backward "double"
    core_out_double = core_out_doubles[0] if core_out_doubles else (core_double if core_double else {})

    # FUN system
    fun_out_system, fun_sys_res, fun_sys_profit = eval_system(fun_system, fun_pool, scores)
    fun_stake = safe_float(fun_system.get("stake"), 0.0) or 0.0
    fun_profit = fun_sys_profit
    fun_pending = 1 if fun_sys_res == "PENDING" and fun_system else 0

    # DRAW system
    draw_out_system, draw_sys_res, draw_sys_profit = eval_system(draw_system, draw_pool, scores)
    draw_stake = safe_float(draw_system.get("stake"), 0.0) or 0.0
    draw_profit = draw_sys_profit
    draw_pending = 1 if draw_sys_res == "PENDING" and draw_system else 0

    # bankroll end
    core_bankroll_end = round(core_bankroll_start + core_profit, 2)
    fun_bankroll_end  = round(fun_bankroll_start + fun_profit, 2)
    draw_bankroll_end = round(draw_bankroll_start + draw_profit, 2)

    # update history ONLY if fully settled
    fully_settled = (core_pending == 0 and fun_pending == 0 and draw_pending == 0)

    wk_key = window_from or week_id
    history.setdefault("weeks", {})

    # only increment week_count on new window_from
    if window_from and window_from not in history["weeks"]:
        history["week_count"] = int(history.get("week_count") or 0) + 1

    # write week entry
    history["weeks"][wk_key] = {
        "week_no": int(week_no),
        "week_id": week_id,
        "window": window,
        "core": {"stake": round(core_stake, 2), "profit": round(core_profit, 2)},
        "funbet": {"stake": round(fun_stake, 2), "profit": round(fun_profit, 2)},
        "drawbet": {"stake": round(draw_stake, 2), "profit": round(draw_profit, 2)},
        "fully_settled": bool(fully_settled),
        "updated_at": now_utc_iso(),
    }

    if fully_settled:
        history["core"]["stake"] = round((safe_float(history["core"].get("stake"), 0.0) or 0.0) + core_stake, 2)
        history["core"]["profit"] = round((safe_float(history["core"].get("profit"), 0.0) or 0.0) + core_profit, 2)
        history["core"]["bankroll_current"] = core_bankroll_end

        history["funbet"]["stake"] = round((safe_float(history["funbet"].get("stake"), 0.0) or 0.0) + fun_stake, 2)
        history["funbet"]["profit"] = round((safe_float(history["funbet"].get("profit"), 0.0) or 0.0) + fun_profit, 2)
        history["funbet"]["bankroll_current"] = fun_bankroll_end

        history["drawbet"]["stake"] = round((safe_float(history["drawbet"].get("stake"), 0.0) or 0.0) + draw_stake, 2)
        history["drawbet"]["profit"] = round((safe_float(history["drawbet"].get("profit"), 0.0) or 0.0) + draw_profit, 2)
        history["drawbet"]["bankroll_current"] = draw_bankroll_end

    history["updated_at"] = now_utc_iso()
    save_json(TUESDAY_HISTORY_PATH, history)

    # recap json
    core_roi = round((core_profit / core_stake) * 100.0, 2) if core_stake > 0 else None
    fun_roi  = round((fun_profit / fun_stake) * 100.0, 2) if fun_stake > 0 else None
    draw_roi = round((draw_profit / draw_stake) * 100.0, 2) if draw_stake > 0 else None

    report = {
        "timestamp": now_utc_iso(),
        "week_id": week_id,
        "week_no": int(week_no),
        "week_label": f"Week {int(week_no)}",
        "window": window,
        "source_friday_timestamp": friday.get("timestamp"),

        "core": {
            "stake": round(core_stake, 2),
            "profit": round(core_profit, 2),
            "roi_pct": core_roi,
            "bankroll_start": round(core_bankroll_start, 2),
            "bankroll_end": round(core_bankroll_end, 2),
            "pending": int(core_pending),
            "singles": core_out_singles,
            "double": core_out_double,
            "doubles": core_out_doubles,
        },

        "funbet": {
            "stake": round(fun_stake, 2),
            "profit": round(fun_profit, 2),
            "roi_pct": fun_roi,
            "bankroll_start": round(fun_bankroll_start, 2),
            "bankroll_end": round(fun_bankroll_end, 2),
            "pending": int(fun_pending),
            "system": fun_out_system,
            "system_pool": fun_pool,
        },

        "drawbet": {
            "stake": round(draw_stake, 2),
            "profit": round(draw_profit, 2),
            "roi_pct": draw_roi,
            "bankroll_start": round(draw_bankroll_start, 2),
            "bankroll_end": round(draw_bankroll_end, 2),
            "pending": int(draw_pending),
            "system": draw_out_system,
            "system_pool": draw_pool,
        },

        "lifetime": {
            "week_count": int(history.get("week_count") or 0),
            "core": {
                "stake": round(safe_float(history["core"].get("stake"), 0.0) or 0.0, 2),
                "profit": round(safe_float(history["core"].get("profit"), 0.0) or 0.0, 2),
                "bankroll_current": history["core"].get("bankroll_current"),
            },
            "funbet": {
                "stake": round(safe_float(history["funbet"].get("stake"), 0.0) or 0.0, 2),
                "profit": round(safe_float(history["funbet"].get("profit"), 0.0) or 0.0, 2),
                "bankroll_current": history["funbet"].get("bankroll_current"),
            },
            "drawbet": {
                "stake": round(safe_float(history["drawbet"].get("stake"), 0.0) or 0.0, 2),
                "profit": round(safe_float(history["drawbet"].get("profit"), 0.0) or 0.0, 2),
                "bankroll_current": history["drawbet"].get("bankroll_current"),
            },
        },

        "fully_settled": bool(fully_settled),
    }

    save_json(TUESDAY_RECAP_PATH, report)
    log(f"✅ Tuesday Recap v3 written: {TUESDAY_RECAP_PATH} | settled={fully_settled}")

if __name__ == "__main__":
    main()
