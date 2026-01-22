# ============================================================
#  BOMBAY TUESDAY RECAP v3.32 — PRODUCTION (Core + Fun + Draw) + History carry
#
#  Reads:
#    - logs/friday_shortlist_v3.json
#  Writes:
#    - logs/tuesday_recap_v3.json
#    - logs/tuesday_history_v3.json
#
#  Features:
#   - Supports portfolios: core, funbet, drawbet
#   - Adds per-bet: final_score, result_icon, result, profit
#   - Adds per-portfolio: stake/profit/roi_pct + bankroll_start/end + score
#   - Updates history with week numbering + bankroll_current (only when settled)
#   - Adds copy_recap list for copy-paste (chronological)
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
HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY} if API_FOOTBALL_KEY else {}

FRIDAY_REPORT_PATH = os.getenv("FRIDAY_REPORT_PATH", "logs/friday_shortlist_v3.json")
TUESDAY_RECAP_PATH = os.getenv("TUESDAY_RECAP_PATH", "logs/tuesday_recap_v3.json")
TUESDAY_HISTORY_PATH = os.getenv("TUESDAY_HISTORY_PATH", "logs/tuesday_history_v3.json")

REQUEST_TIMEOUT = int(os.getenv("API_TIMEOUT_SEC", "25"))
REQUEST_SLEEP_MS = int(os.getenv("API_SLEEP_MS", "120"))
MAX_FETCH_RETRIES = int(os.getenv("MAX_FETCH_RETRIES", "2"))

FINAL_STATUSES = {"FT", "AET", "PEN"}  # settled


# ------------------------- utils -------------------------
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

def sort_key_dt(x: Dict[str, Any]):
    return (x.get("date") or "", x.get("time") or "", x.get("league") or "", x.get("match") or "")


# ------------------------- API Football score fetch -------------------------
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
        hg = goals.get("home")
        ag = goals.get("away")

        out[fid] = {
            "status_short": str(status_short).upper(),
            "home_goals": safe_int(hg, None),
            "away_goals": safe_int(ag, None),
        }

        time.sleep(max(0.0, REQUEST_SLEEP_MS / 1000.0))

    return out


# ------------------------- settlement helpers -------------------------
def market_won(market_code: str, hg: int, ag: int) -> bool:
    mc = (market_code or "").upper().strip()
    if mc == "1":   return hg > ag
    if mc == "X":   return hg == ag
    if mc == "2":   return ag > hg
    if mc == "O25": return (hg + ag) >= 3
    if mc == "U25": return (hg + ag) <= 2
    return False

def icon_for_result(res: str) -> str:
    if res == "WIN": return "🟢"
    if res == "LOSE": return "🔴"
    return "⏳"

def final_score_text(sc: Dict[str, Any]) -> str:
    hg = sc.get("home_goals")
    ag = sc.get("away_goals")
    if hg is None or ag is None:
        return "—"
    return f"{hg}-{ag}"

def eval_single(pick: Dict[str, Any], scores: Dict[int, Dict[str, Any]]) -> Tuple[str, float, str, Optional[int], Optional[int]]:
    fid = safe_int(pick.get("fixture_id"), None)
    mcode = (pick.get("market_code") or "").upper().strip()
    stake = safe_float(pick.get("stake"), 0.0) or 0.0
    odds = safe_float(pick.get("odds"), None)

    if fid is None or not mcode:
        return "PENDING", 0.0, "—", None, None

    sc = scores.get(fid)
    if not sc:
        return "PENDING", 0.0, "—", None, None

    st = (sc.get("status_short") or "").upper()
    hg = sc.get("home_goals")
    ag = sc.get("away_goals")

    if st not in FINAL_STATUSES or hg is None or ag is None:
        return "PENDING", 0.0, "—", hg, ag

    won = market_won(mcode, int(hg), int(ag))
    if odds is None or odds <= 1.0:
        prof = 0.0
    else:
        prof = round(stake * (odds - 1.0), 2) if won else round(-stake, 2)

    res = "WIN" if won else "LOSE"
    return res, prof, f"{hg}-{ag}", int(hg), int(ag)

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
        pid = leg.get("pick_id") or ""
        odds = safe_float(leg.get("odds"), None)

        fid = None
        code = None
        try:
            a, b = pid.split(":")
            fid = safe_int(a, None)
            code = (b or "").upper().strip()
        except Exception:
            fid = None
            code = None

        if fid is None or not code:
            legs_out.append({**leg, "result": "PENDING", "result_icon": "⏳", "final_score": "—"})
            any_pending = True
            continue

        sc = scores.get(fid)
        if not sc:
            legs_out.append({**leg, "result": "PENDING", "result_icon": "⏳", "final_score": "—"})
            any_pending = True
            continue

        st = (sc.get("status_short") or "").upper()
        hg = sc.get("home_goals")
        ag = sc.get("away_goals")

        if st not in FINAL_STATUSES or hg is None or ag is None:
            legs_out.append({**leg, "result": "PENDING", "result_icon": "⏳", "final_score": "—"})
            any_pending = True
            continue

        won = market_won(code, int(hg), int(ag))
        if not won:
            any_lose = True
            legs_out.append({**leg, "result": "LOSE", "result_icon": "🔴", "final_score": f"{hg}-{ag}"})
        else:
            legs_out.append({**leg, "result": "WIN", "result_icon": "🟢", "final_score": f"{hg}-{ag}"})
            if odds is not None and odds > 1.0:
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

    out = {**double_obj, "legs": legs_out, "result": status, "result_icon": icon_for_result(status), "profit": profit}
    return out, status, profit

def parse_system_sizes(label: str) -> List[int]:
    """
    label examples:
      "4/7" -> [4]
      "2-3/5" -> [2,3]
      "3/6" -> [3]
    """
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
        return {**system, "result": "PENDING", "result_icon": "⏳", "profit": 0.0}, "PENDING", 0.0

    # leg states
    leg_state = []
    legs_out = []

    for r in pool:
        fid = safe_int(r.get("fixture_id"), None)
        code = (r.get("market_code") or "").upper().strip()
        odds = safe_float(r.get("odds"), None)

        if fid is None or not code:
            leg_state.append(("PENDING", 1.0))
            legs_out.append({**r, "result": "PENDING", "result_icon": "⏳", "final_score": "—"})
            continue

        sc = scores.get(fid)
        if not sc:
            leg_state.append(("PENDING", 1.0))
            legs_out.append({**r, "result": "PENDING", "result_icon": "⏳", "final_score": "—"})
            continue

        st = (sc.get("status_short") or "").upper()
        hg = sc.get("home_goals")
        ag = sc.get("away_goals")

        if st not in FINAL_STATUSES or hg is None or ag is None:
            leg_state.append(("PENDING", 1.0))
            legs_out.append({**r, "result": "PENDING", "result_icon": "⏳", "final_score": "—"})
            continue

        won = market_won(code, int(hg), int(ag))
        if not won:
            leg_state.append(("LOSE", 1.0))
            legs_out.append({**r, "result": "LOSE", "result_icon": "🔴", "final_score": f"{hg}-{ag}"})
        else:
            leg_state.append(("WIN", float(odds) if odds and odds > 1.0 else 1.0))
            legs_out.append({**r, "result": "WIN", "result_icon": "🟢", "final_score": f"{hg}-{ag}"})

    if any(st == "PENDING" for st, _m in leg_state):
        out = {**system, "result": "PENDING", "result_icon": "⏳", "profit": 0.0, "pool_results": legs_out}
        return out, "PENDING", 0.0

    # settled system profit
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

    out = {**system, "result": result, "result_icon": icon_for_result(result), "profit": profit, "pool_results": legs_out}
    return out, result, profit

def score_ratio(wins: int, total: int) -> str:
    if total <= 0:
        return "0/0"
    return f"{wins}/{total}"


# ------------------------- copy recap builder -------------------------
def build_copy_recap(portfolio_name: str, singles: List[Dict[str, Any]], doubles: List[Dict[str, Any]], system_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []

    for s in singles or []:
        out.append({
            "date": s.get("date"), "time": s.get("time"),
            "portfolio": portfolio_name,
            "type": "single",
            "icon": s.get("result_icon"),
            "match": s.get("match"),
            "market": s.get("market"),
            "odds": s.get("odds"),
            "stake": s.get("stake"),
            "final_score": s.get("final_score"),
            "profit": s.get("profit"),
        })

    for d in doubles or []:
        out.append({
            "date": (d.get("legs") or [{}])[0].get("date"),
            "time": (d.get("legs") or [{}])[0].get("time"),
            "portfolio": portfolio_name,
            "type": "double",
            "icon": d.get("result_icon"),
            "match": " + ".join([f'{x.get("match")} ({x.get("market")})' for x in (d.get("legs") or [])]),
            "market": "DOUBLE",
            "odds": d.get("combo_odds"),
            "stake": d.get("stake"),
            "final_score": "—",
            "profit": d.get("profit"),
        })

    if system_obj and system_obj.get("label"):
        out.append({
            "date": (system_obj.get("pool_results") or [{}])[0].get("date"),
            "time": (system_obj.get("pool_results") or [{}])[0].get("time"),
            "portfolio": portfolio_name,
            "type": "system",
            "icon": system_obj.get("result_icon"),
            "match": f'{system_obj.get("label")} ({system_obj.get("columns")} cols)',
            "market": "SYSTEM",
            "odds": None,
            "stake": system_obj.get("stake"),
            "final_score": "—",
            "profit": system_obj.get("profit"),
        })

    out.sort(key=lambda x: (x.get("date") or "", x.get("time") or "", x.get("portfolio") or "", x.get("type") or ""))
    return out


# ------------------------- main -------------------------
def main():
    if not os.path.exists(FRIDAY_REPORT_PATH):
        raise FileNotFoundError(f"Friday report not found: {FRIDAY_REPORT_PATH}")

    friday = load_json(FRIDAY_REPORT_PATH)
    window = friday.get("window") or {}
    window_from = (window or {}).get("from")
    week_id = iso_week_id_from_window(window)

    history = load_history(TUESDAY_HISTORY_PATH)

    # week_no increments only when window.from changes
    wk_key = window_from or week_id
    if wk_key in (history.get("weeks") or {}):
        week_no = int(history["weeks"][wk_key].get("week_no") or 1)
    else:
        week_no = int(history.get("week_count") or 0) + 1

    # --- Collect portfolios from Friday ---
    core = friday.get("core") or {}
    fun  = friday.get("funbet") or {}
    draw = friday.get("drawbet") or {}

    # bankroll start: prefer history bankroll_current, else friday bankroll_start, else friday bankroll, else 0
    def bankroll_start_for(name: str, friday_obj: dict) -> float:
        hs = safe_float((history.get(name) or {}).get("bankroll_current"), None)
        if hs is not None:
            return float(hs)
        fs = safe_float(friday_obj.get("bankroll_start"), None)
        if fs is not None:
            return float(fs)
        fb = safe_float(friday_obj.get("bankroll"), None)
        if fb is not None:
            return float(fb)
        return 0.0

    core_bankroll_start = bankroll_start_for("core", core)
    fun_bankroll_start  = bankroll_start_for("funbet", fun)
    draw_bankroll_start = bankroll_start_for("drawbet", draw)

    # --- Build fixture id list ---
    fixture_ids: List[int] = []

    core_singles = core.get("singles") or []
    core_doubles = core.get("doubles") or ([] if not core.get("double") else [core.get("double")])
    fun_system_pool = fun.get("system_pool") or []
    fun_system = fun.get("system") or {}
    draw_system_pool = draw.get("system_pool") or []
    draw_system = draw.get("system") or {}

    # add all fixture ids from singles + system pools
    for p in core_singles + fun_system_pool + draw_system_pool:
        fid = safe_int(p.get("fixture_id"), None)
        if fid is not None:
            fixture_ids.append(fid)

    # add fixture ids from double legs
    for d in core_doubles:
        for leg in (d.get("legs") or []):
            fid = safe_int(leg.get("fixture_id"), None)
            if fid is not None:
                fixture_ids.append(fid)
            else:
                pid = leg.get("pick_id") or ""
                try:
                    a, _b = pid.split(":")
                    fid2 = safe_int(a, None)
                    if fid2 is not None:
                        fixture_ids.append(fid2)
                except Exception:
                    pass

    scores = fetch_scores(fixture_ids)

    # ---------------- CORE settle ----------------
    core_out_singles = []
    core_stake = 0.0
    core_profit = 0.0
    core_pending = 0

    for p in core_singles:
        res, prof, fs, hg, ag = eval_single(p, scores)
        stake = safe_float(p.get("stake"), 0.0) or 0.0
        core_stake += stake
        core_profit += prof
        if res == "PENDING":
            core_pending += 1

        core_out_singles.append({
            **p,
            "result": res,
            "result_icon": icon_for_result(res),
            "profit": prof,
            "final_score": fs,
            "home_goals": hg,
            "away_goals": ag,
        })

    core_out_doubles = []
    double_wins = 0
    double_total = 0
    for d in core_doubles:
        d_out, d_res, d_prof = eval_double(d, scores)
        core_profit += d_prof
        stake = safe_float(d.get("stake"), 0.0) or 0.0
        core_stake += stake
        if d_res == "PENDING":
            core_pending += 1
        if d_res in ("WIN", "LOSE"):
            double_total += 1
            if d_res == "WIN":
                double_wins += 1
        core_out_doubles.append(d_out)

    core_out_singles.sort(key=sort_key_dt)
    core_out_doubles.sort(key=lambda x: ((x.get("legs") or [{}])[0].get("date") or "", (x.get("legs") or [{}])[0].get("time") or ""))

    core_bankroll_end = round(core_bankroll_start + core_profit, 2)
    core_roi = round((core_profit / core_stake) * 100.0, 2) if core_stake > 0 else None

    # Score strings
    core_single_wins = sum(1 for p in core_out_singles if p.get("result") == "WIN")
    core_score_singles = score_ratio(core_single_wins, len(core_out_singles))
    core_score_doubles = score_ratio(double_wins, double_total) if double_total > 0 else "—"

    # ---------------- FUN settle (system-only) ----------------
    fun_out_system, fun_sys_res, fun_sys_profit = eval_system(fun_system, fun_system_pool, scores)
    fun_pending = 1 if fun_sys_res == "PENDING" else 0
    fun_stake = safe_float(fun_system.get("stake"), 0.0) or 0.0
    fun_profit = fun_sys_profit
    fun_bankroll_end = round(fun_bankroll_start + fun_profit, 2)
    fun_roi = round((fun_profit / fun_stake) * 100.0, 2) if fun_stake > 0 else None

    fun_pool_wins = sum(1 for r in (fun_out_system.get("pool_results") or []) if r.get("result") == "WIN")
    fun_score_pool = score_ratio(fun_pool_wins, len(fun_system_pool))

    # ---------------- DRAW settle (system-only) ----------------
    draw_out_system, draw_sys_res, draw_sys_profit = eval_system(draw_system, draw_system_pool, scores)
    draw_pending = 1 if draw_sys_res == "PENDING" else 0
    draw_stake = safe_float(draw_system.get("stake"), 0.0) or 0.0
    draw_profit = draw_sys_profit
    draw_bankroll_end = round(draw_bankroll_start + draw_profit, 2)
    draw_roi = round((draw_profit / draw_stake) * 100.0, 2) if draw_stake > 0 else None

    draw_pool_wins = sum(1 for r in (draw_out_system.get("pool_results") or []) if r.get("result") == "WIN")
    draw_score_pool = score_ratio(draw_pool_wins, len(draw_system_pool))

    # ---------------- history update ----------------
    history.setdefault("weeks", {})
    prev_week = history["weeks"].get(wk_key)

    fully_settled_core = (core_pending == 0)
    fully_settled_fun  = (fun_pending == 0)
    fully_settled_draw = (draw_pending == 0)

    # If re-run same week, remove previous contributions before adding new (per portfolio)
    def subtract_prev(port: str):
        nonlocal history, prev_week
        if not isinstance(prev_week, dict):
            return
        old = (prev_week.get(port) or {})
        old_stake = safe_float(old.get("stake"), 0.0) or 0.0
        old_profit = safe_float(old.get("profit"), 0.0) or 0.0
        history[port]["stake"] = round((safe_float(history[port].get("stake"), 0.0) or 0.0) - old_stake, 2)
        history[port]["profit"] = round((safe_float(history[port].get("profit"), 0.0) or 0.0) - old_profit, 2)

    def add_new(port: str, stake: float, profit: float):
        history[port]["stake"] = round((safe_float(history[port].get("stake"), 0.0) or 0.0) + stake, 2)
        history[port]["profit"] = round((safe_float(history[port].get("profit"), 0.0) or 0.0) + profit, 2)

    # Ensure base keys exist
    history.setdefault("core", {}).setdefault("stake", 0.0)
    history.setdefault("core", {}).setdefault("profit", 0.0)
    history.setdefault("core", {}).setdefault("bankroll_current", None)
    history.setdefault("funbet", {}).setdefault("stake", 0.0)
    history.setdefault("funbet", {}).setdefault("profit", 0.0)
    history.setdefault("funbet", {}).setdefault("bankroll_current", None)
    history.setdefault("drawbet", {}).setdefault("stake", 0.0)
    history.setdefault("drawbet", {}).setdefault("profit", 0.0)
    history.setdefault("drawbet", {}).setdefault("bankroll_current", None)

    # Only update totals + bankroll_current when that portfolio fully settled
    if fully_settled_core:
        subtract_prev("core")
        add_new("core", core_stake, core_profit)
        history["core"]["bankroll_current"] = core_bankroll_end

    if fully_settled_fun:
        subtract_prev("funbet")
        add_new("funbet", fun_stake, fun_profit)
        history["funbet"]["bankroll_current"] = fun_bankroll_end

    if fully_settled_draw:
        subtract_prev("drawbet")
        add_new("drawbet", draw_stake, draw_profit)
        history["drawbet"]["bankroll_current"] = draw_bankroll_end

    # week_count increments only if new wk_key
    if wk_key not in history["weeks"]:
        history["week_count"] = int(history.get("week_count") or 0) + 1

    history["last_window_from"] = window_from
    history["last_week_id"] = week_id
    history["updated_at"] = now_utc_iso()

    history["weeks"][wk_key] = {
        "week_no": int(week_no),
        "week_id": week_id,
        "window": window,
        "core": {"stake": round(core_stake, 2), "profit": round(core_profit, 2), "fully_settled": bool(fully_settled_core)},
        "funbet": {"stake": round(fun_stake, 2), "profit": round(fun_profit, 2), "fully_settled": bool(fully_settled_fun)},
        "drawbet": {"stake": round(draw_stake, 2), "profit": round(draw_profit, 2), "fully_settled": bool(fully_settled_draw)},
        "updated_at": now_utc_iso(),
    }

    save_json(TUESDAY_HISTORY_PATH, history)

    # ---------------- copy recap ----------------
    # doubles list for recap = core_out_doubles
    # systems: fun_out_system, draw_out_system
    copy_recap = []
    copy_recap += build_copy_recap("CoreBet", core_out_singles, core_out_doubles, {})
    copy_recap += build_copy_recap("FunBet", [], [], fun_out_system)
    copy_recap += build_copy_recap("DrawBet", [], [], draw_out_system)
    copy_recap.sort(key=lambda x: (x.get("date") or "", x.get("time") or "", x.get("portfolio") or "", x.get("type") or ""))

    # ---------------- recap JSON ----------------
    report = {
        "timestamp": now_utc_iso(),
        "week_id": week_id,
        "week_no": int(week_no),
        "week_label": f"Week {int(week_no)}",
        "window": window,
        "source_friday_timestamp": friday.get("timestamp"),

        "core": {
            "portfolio": "CoreBet",
            "stake": round(core_stake, 2),
            "profit": round(core_profit, 2),
            "roi_pct": core_roi,
            "bankroll_start": round(core_bankroll_start, 2),
            "bankroll_end": round(core_bankroll_end, 2),
            "pending": int(core_pending),
            "score": {"singles": core_score_singles, "doubles": core_score_doubles},
            "singles": core_out_singles,
            "doubles": core_out_doubles,
        },

        "funbet": {
            "portfolio": "FunBet",
            "stake": round(fun_stake, 2),
            "profit": round(fun_profit, 2),
            "roi_pct": fun_roi,
            "bankroll_start": round(fun_bankroll_start, 2),
            "bankroll_end": round(fun_bankroll_end, 2),
            "pending": int(fun_pending),
            "score": {"system_pool": fun_score_pool},
            "system": fun_out_system,
            "system_pool": fun_system_pool,
        },

        "drawbet": {
            "portfolio": "DrawBet",
            "stake": round(draw_stake, 2),
            "profit": round(draw_profit, 2),
            "roi_pct": draw_roi,
            "bankroll_start": round(draw_bankroll_start, 2),
            "bankroll_end": round(draw_bankroll_end, 2),
            "pending": int(draw_pending),
            "score": {"system_pool": draw_score_pool},
            "system": draw_out_system,
            "system_pool": draw_system_pool,
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

        "copy_recap": copy_recap,
    }

    save_json(TUESDAY_RECAP_PATH, report)
    log(f"✅ Tuesday Recap v3 written: {TUESDAY_RECAP_PATH}")

if __name__ == "__main__":
    main()
