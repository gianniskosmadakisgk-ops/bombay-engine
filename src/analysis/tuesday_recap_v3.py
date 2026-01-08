# ============================================================
#  BOMBAY TUESDAY RECAP v3 — PRODUCTION (UPDATED)
#  Adds: week_id + per-group stats (additive)
# ============================================================

import os
import json
import time
import requests
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple

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

def load_history(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"core": {"stake": 0.0, "profit": 0.0}, "funbet": {"stake": 0.0, "profit": 0.0}, "updated_at": now_utc_iso()}
    try:
        return load_json(path)
    except Exception:
        return {"core": {"stake": 0.0, "profit": 0.0}, "funbet": {"stake": 0.0, "profit": 0.0}, "updated_at": now_utc_iso()}

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

def market_won(market_code: str, hg: int, ag: int) -> bool:
    mc = (market_code or "").upper().strip()
    if mc == "1": return hg > ag
    if mc == "X": return hg == ag
    if mc == "2": return ag > hg
    if mc == "O25": return (hg + ag) >= 3
    if mc == "U25": return (hg + ag) <= 2
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
    hg = sc.get("home_goals"); ag = sc.get("away_goals")
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
    all_final = True
    legs_out = []
    for leg in legs:
        pid = leg.get("pick_id") or ""
        odds = safe_float(leg.get("odds"), None)
        fid = None; code = None
        try:
            a, b = pid.split(":")
            fid = safe_int(a, None)
            code = (b or "").upper().strip()
        except Exception:
            fid = None; code = None
        if fid is None or not code:
            legs_out.append({**leg, "result": "PENDING"}); any_pending = True; all_final = False; continue
        sc = scores.get(fid)
        if not sc:
            legs_out.append({**leg, "result": "PENDING"}); any_pending = True; all_final = False; continue
        st = (sc.get("status_short") or "").upper()
        hg = sc.get("home_goals"); ag = sc.get("away_goals")
        if st not in FINAL_STATUSES or hg is None or ag is None:
            legs_out.append({**leg, "result": "PENDING"}); any_pending = True; all_final = False; continue
        won = market_won(code, int(hg), int(ag))
        if not won:
            any_lose = True; legs_out.append({**leg, "result": "LOSE"})
        else:
            legs_out.append({**leg, "result": "WIN"})
            if odds is not None and odds > 1.0:
                mult *= float(odds)
    if any_pending and not all_final:
        status = "PENDING"; profit = 0.0
    elif any_lose:
        status = "LOSE"; profit = round(-stake, 2) if stake > 0 else 0.0
    else:
        status = "WIN"; profit = round((stake * mult) - stake, 2) if stake > 0 else 0.0
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
    leg_state = []
    for r in pool:
        fid = safe_int(r.get("fixture_id"), None)
        code = (r.get("market_code") or "").upper().strip()
        odds = safe_float(r.get("odds"), None)
        if fid is None or not code:
            leg_state.append(("PENDING", 1.0)); continue
        sc = scores.get(fid)
        if not sc:
            leg_state.append(("PENDING", 1.0)); continue
        st = (sc.get("status_short") or "").upper()
        hg = sc.get("home_goals"); ag = sc.get("away_goals")
        if st not in FINAL_STATUSES or hg is None or ag is None:
            leg_state.append(("PENDING", 1.0)); continue
        won = market_won(code, int(hg), int(ag))
        if not won:
            leg_state.append(("LOSE", 1.0))
        else:
            leg_state.append(("WIN", float(odds) if odds and odds > 1.0 else 1.0))
    if any(st == "PENDING" for st, _m in leg_state):
        return {**system, "result": "PENDING", "profit": 0.0}, "PENDING", 0.0
    from itertools import combinations
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

def _group_stats(items: List[Dict[str, Any]], stake_total: float, profit_total: float):
    picks = len(items)
    hits = sum(1 for p in items if p.get("result") == "WIN")
    settled = [p for p in items if p.get("result") in ("WIN","LOSE")]
    settled_stake = sum(safe_float(p.get("stake"), 0.0) for p in settled)
    settled_profit = sum(safe_float(p.get("profit"), 0.0) for p in settled)
    roi_pct = round((settled_profit / settled_stake) * 100.0, 2) if settled_stake > 0 else 0.0
    return {
        "picks": picks,
        "hits": hits,
        "stake": round(stake_total, 2),
        "profit": round(profit_total, 2),
        "settled_stake": round(settled_stake, 2),
        "settled_profit": round(settled_profit, 2),
        "roi_pct_settled": roi_pct,
        "pending": sum(1 for p in items if p.get("result") == "PENDING"),
    }

def main():
    if not os.path.exists(FRIDAY_REPORT_PATH):
        raise FileNotFoundError(f"Friday report not found: {FRIDAY_REPORT_PATH}")

    friday = load_json(FRIDAY_REPORT_PATH)
    window = friday.get("window") or {}
    week_id = iso_week_id_from_window(window)

    core = friday.get("core", {}) or {}
    fun = friday.get("funbet", {}) or {}

    core_singles = core.get("singles", []) or []
    core_double = core.get("double") or {}

    fun_singles = fun.get("singles", []) or []
    fun_system = fun.get("system") or {}
    fun_pool = fun.get("system_pool", []) or []

    fixture_ids: List[int] = []
    for p in core_singles + fun_singles + fun_pool:
        fid = safe_int(p.get("fixture_id"), None)
        if fid is not None:
            fixture_ids.append(fid)

    for leg in (core_double.get("legs") or []):
        pid = leg.get("pick_id") or ""
        try:
            a, _b = pid.split(":")
            fid = safe_int(a, None)
            if fid is not None:
                fixture_ids.append(fid)
        except Exception:
            pass

    scores = fetch_scores(fixture_ids)

    core_out_singles = []
    core_stake = 0.0
    core_profit = 0.0
    for p in core_singles:
        res, prof = eval_single(p, scores)
        stake = safe_float(p.get("stake"), 0.0) or 0.0
        core_stake += stake
        core_profit += prof
        core_out_singles.append({**p, "result": res, "profit": prof})

    core_out_double, _dres, dprofit = eval_double(core_double, scores)
    core_double_stake = safe_float(core_double.get("stake"), 0.0) or 0.0
    core_stake += core_double_stake
    core_profit += dprofit

    fun_out_singles = []
    fun_stake = 0.0
    fun_profit = 0.0
    for p in fun_singles:
        res, prof = eval_single(p, scores)
        stake = safe_float(p.get("stake"), 0.0) or 0.0
        fun_stake += stake
        fun_profit += prof
        fun_out_singles.append({**p, "result": res, "profit": prof})

    fun_out_system, _sres, sprofit = eval_system(fun_system, fun_pool, scores)
    fun_sys_stake = safe_float(fun_system.get("stake"), 0.0) or 0.0
    fun_stake += fun_sys_stake
    fun_profit += sprofit

    history = load_history(TUESDAY_HISTORY_PATH)
    history["updated_at"] = now_utc_iso()
    save_json(TUESDAY_HISTORY_PATH, history)

    report = {
        "timestamp": now_utc_iso(),
        "week_id": week_id,
        "window": window,
        "source_friday_timestamp": friday.get("timestamp"),

        "core": {
            **_group_stats(core_out_singles, core_stake, core_profit),
            "singles": core_out_singles,
            "double": core_out_double,
        },

        "funbet": {
            **_group_stats(fun_out_singles, fun_stake, fun_profit),
            "singles": fun_out_singles,
            "system": fun_out_system,
            "system_pool": fun_pool,
        },

        "weekly_summary": {
            "core": {
                "history_stake": round(float(history["core"].get("stake", 0.0)), 2),
                "history_profit": round(float(history["core"].get("profit", 0.0)), 2),
            },
            "funbet": {
                "history_stake": round(float(history["funbet"].get("stake", 0.0)), 2),
                "history_profit": round(float(history["funbet"].get("profit", 0.0)), 2),
            },
        },
    }

    save_json(TUESDAY_RECAP_PATH, report)
    log(f"✅ Tuesday Recap v3 written: {TUESDAY_RECAP_PATH}")

if __name__ == "__main__":
    main()
