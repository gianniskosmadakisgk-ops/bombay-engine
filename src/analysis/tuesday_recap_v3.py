# ============================================================
#  src/analysis/tuesday_recap_v3.py
#  BOMBAY TUESDAY RECAP v3 — PRODUCTION (CoreBet + FunBet + DrawBet)
#
#  Reads:
#    - logs/friday_shortlist_v3.json
#    - (optional) logs/tuesday_history_v3.json   (lifetime + bankroll carry)
#
#  Writes:
#    - logs/tuesday_recap_v3.json
#    - logs/tuesday_history_v3.json  (updated if fully settled per-portfolio)
#
#  Rules:
#   - If any pick is not finished -> result MUST be "PENDING" and profit=0 for that item.
#   - Lifetime totals + bankroll_current update ONLY when a portfolio is fully settled (no PENDING).
#
#  Fixes:
#   - system_pool score is now real hits/total (WIN count), not "n/n".
#   - Adds settled vs total ROI fields (additive).
# ============================================================

import os
import json
import time
import requests
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple
from itertools import combinations as it_combinations

API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

FRIDAY_REPORT_PATH = os.getenv("FRIDAY_REPORT_PATH", "logs/friday_shortlist_v3.json")
TUESDAY_RECAP_PATH = os.getenv("TUESDAY_RECAP_PATH", "logs/tuesday_recap_v3.json")
TUESDAY_HISTORY_PATH = os.getenv("TUESDAY_HISTORY_PATH", "logs/tuesday_history_v3.json")

REQUEST_TIMEOUT = int(os.getenv("API_TIMEOUT_SEC", "25"))
REQUEST_SLEEP_MS = int(os.getenv("API_SLEEP_MS", "120"))
MAX_FETCH_RETRIES = int(os.getenv("MAX_FETCH_RETRIES", "2"))

FINAL_STATUSES = {"FT", "AET", "PEN"}  # settled

# Defaults (used only if history missing and Friday lacks bankroll_start)
DEFAULT_BANKROLL_CORE = float(os.getenv("BANKROLL_CORE", "800"))
DEFAULT_BANKROLL_FUN = float(os.getenv("BANKROLL_FUN", "400"))
DEFAULT_BANKROLL_DRAW = float(os.getenv("BANKROLL_DRAW", "300"))


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


def iso_week_id_from_window(window: Optional[Dict[str, Any]]) -> str:
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


def _default_history() -> Dict[str, Any]:
    return {
        "week_count": 0,
        "last_window_from": None,
        "weeks": {},
        "core": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
        "funbet": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
        "drawbet": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
        "updated_at": now_utc_iso(),
    }


def load_history(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return _default_history()
    try:
        h = load_json(path)
        if not isinstance(h, dict):
            return _default_history()

        h.setdefault("week_count", 0)
        h.setdefault("last_window_from", None)
        if "weeks" not in h or not isinstance(h["weeks"], dict):
            h["weeks"] = {}

        h.setdefault("core", {"stake": 0.0, "profit": 0.0, "bankroll_current": None})
        h.setdefault("funbet", {"stake": 0.0, "profit": 0.0, "bankroll_current": None})
        h.setdefault("drawbet", {"stake": 0.0, "profit": 0.0, "bankroll_current": None})

        h["core"].setdefault("stake", 0.0)
        h["core"].setdefault("profit", 0.0)
        h["core"].setdefault("bankroll_current", None)

        h["funbet"].setdefault("stake", 0.0)
        h["funbet"].setdefault("profit", 0.0)
        h["funbet"].setdefault("bankroll_current", None)

        h["drawbet"].setdefault("stake", 0.0)
        h["drawbet"].setdefault("profit", 0.0)
        h["drawbet"].setdefault("bankroll_current", None)

        return h
    except Exception:
        return _default_history()


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
    if mc == "1":
        return hg > ag
    if mc == "X":
        return hg == ag
    if mc == "2":
        return ag > hg
    if mc == "O25":
        return (hg + ag) >= 3
    if mc == "U25":
        return (hg + ag) <= 2
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
    all_final = True

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
            legs_out.append({**leg, "result": "PENDING"})
            any_pending = True
            all_final = False
            continue

        sc = scores.get(fid)
        if not sc:
            legs_out.append({**leg, "result": "PENDING"})
            any_pending = True
            all_final = False
            continue

        st = (sc.get("status_short") or "").upper()
        hg = sc.get("home_goals")
        ag = sc.get("away_goals")

        if st not in FINAL_STATUSES or hg is None or ag is None:
            legs_out.append({**leg, "result": "PENDING"})
            any_pending = True
            all_final = False
            continue

        won = market_won(code, int(hg), int(ag))
        if not won:
            any_lose = True
            legs_out.append({**leg, "result": "LOSE"})
        else:
            legs_out.append({**leg, "result": "WIN"})
            if odds is not None and odds > 1.0:
                mult *= float(odds)

    if any_pending and not all_final:
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
    out: List[int] = []
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

    leg_state: List[Tuple[str, float]] = []
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

    for rsize in sizes:
        if rsize < 2 or rsize > n:
            continue
        for combo in it_combinations(idxs, rsize):
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


def _hits_total(items: List[Dict[str, Any]]) -> Tuple[int, int, int]:
    total = len(items)
    wins = sum(1 for p in items if p.get("result") == "WIN")
    pending = sum(1 for p in items if p.get("result") == "PENDING")
    return wins, total, pending


def _roi_pct(profit: float, stake: float) -> Optional[float]:
    if stake and stake > 0:
        return round((profit / stake) * 100.0, 2)
    return None


def _bankroll_start_from(friday_section: Dict[str, Any], history_section: Dict[str, Any], default_bankroll: float) -> float:
    hs = safe_float((history_section or {}).get("bankroll_current"), None)
    if hs is not None:
        return float(hs)
    fs = safe_float((friday_section or {}).get("bankroll_start"), None)
    if fs is not None:
        return float(fs)
    fb = safe_float((friday_section or {}).get("bankroll"), None)
    if fb is not None:
        return float(fb)
    return float(default_bankroll)


def main():
    if not os.path.exists(FRIDAY_REPORT_PATH):
        raise FileNotFoundError(f"Friday report not found: {FRIDAY_REPORT_PATH}")

    friday = load_json(FRIDAY_REPORT_PATH)
    window = friday.get("window") or {}
    window_from = (window or {}).get("from")
    week_id = iso_week_id_from_window(window)

    history = load_history(TUESDAY_HISTORY_PATH)

    # week_no increments only when new window.from appears
    if window_from and window_from in history.get("weeks", {}):
        week_no = int(history["weeks"][window_from].get("week_no") or 1)
    else:
        week_no = int(history.get("week_count") or 0) + 1

    # Sections (Friday may or may not include drawbet)
    core = friday.get("core", {}) or {}
    fun = friday.get("funbet", {}) or {}
    draw = friday.get("drawbet", {}) or {}

    # Bankroll starts (carry from history if present)
    core_bankroll_start = _bankroll_start_from(core, history.get("core", {}), DEFAULT_BANKROLL_CORE)
    fun_bankroll_start = _bankroll_start_from(fun, history.get("funbet", {}), DEFAULT_BANKROLL_FUN)
    draw_bankroll_start = _bankroll_start_from(draw, history.get("drawbet", {}), DEFAULT_BANKROLL_DRAW)

    # Collect picks
    core_singles = core.get("singles", []) or []
    core_double = core.get("double") or {}
    core_doubles = core.get("doubles", []) or []
    if not core_doubles and core_double:
        core_doubles = [core_double]

    fun_singles = fun.get("singles", []) or []  # expected empty in new design, but safe
    fun_system = fun.get("system") or {}
    fun_pool = fun.get("system_pool", []) or []

    draw_singles = draw.get("singles", []) or []  # expected empty
    draw_system = (draw.get("system") or {})
    draw_pool = (draw.get("system_pool") or [])

    # Fixture ids for fetching scores
    fixture_ids: List[int] = []

    def _add_fids_from_picks(pl):
        for p in pl:
            fid = safe_int(p.get("fixture_id"), None)
            if fid is not None:
                fixture_ids.append(fid)

    _add_fids_from_picks(core_singles)
    _add_fids_from_picks(fun_singles)
    _add_fids_from_picks(fun_pool)
    _add_fids_from_picks(draw_singles)
    _add_fids_from_picks(draw_pool)

    for dbl in core_doubles:
        for leg in (dbl.get("legs") or []):
            pid = leg.get("pick_id") or ""
            try:
                a, _b = pid.split(":")
                fid = safe_int(a, None)
                if fid is not None:
                    fixture_ids.append(fid)
            except Exception:
                pass

    scores = fetch_scores(fixture_ids)

    # ---------------- CORE eval ----------------
    core_out_singles = []
    core_stake = 0.0
    core_profit = 0.0
    core_pending = 0
    core_settled_stake = 0.0
    core_settled_profit = 0.0

    for p in core_singles:
        res, prof = eval_single(p, scores)
        stake = safe_float(p.get("stake"), 0.0) or 0.0
        core_stake += stake
        core_profit += prof
        if res == "PENDING":
            core_pending += 1
        else:
            core_settled_stake += stake
            core_settled_profit += prof
        core_out_singles.append({**p, "result": res, "profit": prof})

    core_out_doubles = []
    for dbl in core_doubles:
        out_d, dres, dprof = eval_double(dbl, scores)
        dstake = safe_float(dbl.get("stake"), 0.0) or 0.0
        core_stake += dstake
        core_profit += dprof
        if dres == "PENDING":
            core_pending += 1
        else:
            core_settled_stake += dstake
            core_settled_profit += dprof
        core_out_doubles.append(out_d)

    core_bankroll_end = round(core_bankroll_start + core_profit, 2)
    core_bankroll_end_settled = round(core_bankroll_start + core_settled_profit, 2)

    # ---------------- FUN eval ----------------
    fun_out_singles = []
    fun_stake = 0.0
    fun_profit = 0.0
    fun_pending = 0
    fun_settled_stake = 0.0
    fun_settled_profit = 0.0

    for p in fun_singles:
        res, prof = eval_single(p, scores)
        stake = safe_float(p.get("stake"), 0.0) or 0.0
        fun_stake += stake
        fun_profit += prof
        if res == "PENDING":
            fun_pending += 1
        else:
            fun_settled_stake += stake
            fun_settled_profit += prof
        fun_out_singles.append({**p, "result": res, "profit": prof})

    fun_out_system, fun_sys_res, fun_sys_profit = eval_system(fun_system, fun_pool, scores)
    fun_sys_stake = safe_float(fun_system.get("stake"), 0.0) or 0.0
    fun_stake += fun_sys_stake
    fun_profit += fun_sys_profit
    if fun_system and fun_sys_res == "PENDING":
        fun_pending += 1
    else:
        fun_settled_stake += fun_sys_stake
        fun_settled_profit += fun_sys_profit

    fun_bankroll_end = round(fun_bankroll_start + fun_profit, 2)
    fun_bankroll_end_settled = round(fun_bankroll_start + fun_settled_profit, 2)

    # ---------------- DRAW eval ----------------
    draw_out_singles = []
    draw_stake = 0.0
    draw_profit = 0.0
    draw_pending = 0
    draw_settled_stake = 0.0
    draw_settled_profit = 0.0

    for p in draw_singles:
        res, prof = eval_single(p, scores)
        stake = safe_float(p.get("stake"), 0.0) or 0.0
        draw_stake += stake
        draw_profit += prof
        if res == "PENDING":
            draw_pending += 1
        else:
            draw_settled_stake += stake
            draw_settled_profit += prof
        draw_out_singles.append({**p, "result": res, "profit": prof})

    draw_out_system, draw_sys_res, draw_sys_profit = eval_system(draw_system, draw_pool, scores)
    draw_sys_stake = safe_float(draw_system.get("stake"), 0.0) or 0.0
    draw_stake += draw_sys_stake
    draw_profit += draw_sys_profit
    if draw_system and draw_sys_res == "PENDING":
        draw_pending += 1
    else:
        draw_settled_stake += draw_sys_stake
        draw_settled_profit += draw_sys_profit

    draw_bankroll_end = round(draw_bankroll_start + draw_profit, 2)
    draw_bankroll_end_settled = round(draw_bankroll_start + draw_settled_profit, 2)

    # ---------------- Scores (FIXED) ----------------
    core_single_w, core_single_t, _ = _hits_total(core_out_singles)
    # For doubles: count winning legs total across all doubles (for display only)
    dbl_legs = []
    for d in core_out_doubles:
        dbl_legs.extend(d.get("legs") or [])
    dbl_leg_wins = sum(1 for l in dbl_legs if l.get("result") == "WIN")
    dbl_leg_total = len(dbl_legs)

    fun_single_w, fun_single_t, _ = _hits_total(fun_out_singles)
    fun_pool_wins = 0
    for r in fun_pool:
        fid = safe_int(r.get("fixture_id"), None)
        code = (r.get("market_code") or "").upper().strip()
        if fid is None or not code:
            continue
        sc = scores.get(fid)
        if not sc:
            continue
        st = (sc.get("status_short") or "").upper()
        hg = sc.get("home_goals")
        ag = sc.get("away_goals")
        if st not in FINAL_STATUSES or hg is None or ag is None:
            continue
        if market_won(code, int(hg), int(ag)):
            fun_pool_wins += 1
    fun_pool_total = len(fun_pool)

    draw_pool_wins = 0
    for r in draw_pool:
        fid = safe_int(r.get("fixture_id"), None)
        code = (r.get("market_code") or "").upper().strip()
        if fid is None or not code:
            continue
        sc = scores.get(fid)
        if not sc:
            continue
        st = (sc.get("status_short") or "").upper()
        hg = sc.get("home_goals")
        ag = sc.get("away_goals")
        if st not in FINAL_STATUSES or hg is None or ag is None:
            continue
        if market_won(code, int(hg), int(ag)):
            draw_pool_wins += 1
    draw_pool_total = len(draw_pool)

    # ---------------- Update history ----------------
    wk_key = window_from or week_id
    history.setdefault("weeks", {})
    prev_week = history["weeks"].get(wk_key)

    def _remove_prev(port_key: str):
        if not isinstance(prev_week, dict):
            return
        old = (prev_week.get(port_key) or {})
        old_stake = safe_float(old.get("stake"), 0.0) or 0.0
        old_profit = safe_float(old.get("profit"), 0.0) or 0.0
        history[port_key]["stake"] = round((safe_float(history[port_key].get("stake"), 0.0) or 0.0) - old_stake, 2)
        history[port_key]["profit"] = round((safe_float(history[port_key].get("profit"), 0.0) or 0.0) - old_profit, 2)

    def _add_new(port_key: str, stake_v: float, profit_v: float, bankroll_end_v: float):
        history[port_key]["stake"] = round((safe_float(history[port_key].get("stake"), 0.0) or 0.0) + stake_v, 2)
        history[port_key]["profit"] = round((safe_float(history[port_key].get("profit"), 0.0) or 0.0) + profit_v, 2)
        history[port_key]["bankroll_current"] = round(bankroll_end_v, 2)

    # Update per-portfolio only when fully settled
    core_fully_settled = (core_pending == 0)
    fun_fully_settled = (fun_pending == 0)
    draw_fully_settled = (draw_pending == 0)

    # If rerun same week_key and it was fully settled before, remove previous contributions before adding new
    if core_fully_settled:
        _remove_prev("core")
        _add_new("core", core_stake, core_profit, core_bankroll_end)
    if fun_fully_settled:
        _remove_prev("funbet")
        _add_new("funbet", fun_stake, fun_profit, fun_bankroll_end)
    if draw_fully_settled and ("drawbet" in history):
        _remove_prev("drawbet")
        _add_new("drawbet", draw_stake, draw_profit, draw_bankroll_end)

    # week_count increments only when new window.from is first seen
    if window_from and window_from not in history.get("weeks", {}):
        history["week_count"] = int(history.get("week_count") or 0) + 1

    history["last_window_from"] = window_from
    history["last_week_id"] = week_id
    history["updated_at"] = now_utc_iso()

    history["weeks"][wk_key] = {
        "week_no": int(week_no),
        "week_id": week_id,
        "window": window,
        "core": {"stake": round(core_stake, 2), "profit": round(core_profit, 2), "fully_settled": core_fully_settled},
        "funbet": {"stake": round(fun_stake, 2), "profit": round(fun_profit, 2), "fully_settled": fun_fully_settled},
        "drawbet": {"stake": round(draw_stake, 2), "profit": round(draw_profit, 2), "fully_settled": draw_fully_settled},
        "fully_settled_all": bool(core_fully_settled and fun_fully_settled and draw_fully_settled),
        "updated_at": now_utc_iso(),
    }

    save_json(TUESDAY_HISTORY_PATH, history)

    # ---------------- Tuesday recap JSON ----------------
    report = {
        "timestamp": now_utc_iso(),
        "week_id": week_id,
        "week_no": int(week_no),
        "week_label": f"Week {int(week_no)}",
        "window": window,
        "source_friday_timestamp": friday.get("timestamp"),

        "core": {
            "portfolio": (core.get("portfolio") or "CoreBet"),
            "stake": round(core_stake, 2),
            "profit": round(core_profit, 2),
            "roi_pct_total": _roi_pct(core_profit, core_stake),
            "settled_stake": round(core_settled_stake, 2),
            "settled_profit": round(core_settled_profit, 2),
            "roi_pct_settled": _roi_pct(core_settled_profit, core_settled_stake),
            "bankroll_start": round(core_bankroll_start, 2),
            "bankroll_end": round(core_bankroll_end, 2),
            "bankroll_end_settled": round(core_bankroll_end_settled, 2),
            "pending": int(core_pending),
            "singles": core_out_singles,
            "double": (core_out_doubles[0] if core_out_doubles else {}),
            "doubles": core_out_doubles,
            "score": {
                "singles": f"{core_single_w}/{core_single_t}",
                "double_legs": (f"{dbl_leg_wins}/{dbl_leg_total}" if dbl_leg_total > 0 else "—"),
            },
        },

        "funbet": {
            "portfolio": (fun.get("portfolio") or "FunBet"),
            "stake": round(fun_stake, 2),
            "profit": round(fun_profit, 2),
            "roi_pct_total": _roi_pct(fun_profit, fun_stake),
            "settled_stake": round(fun_settled_stake, 2),
            "settled_profit": round(fun_settled_profit, 2),
            "roi_pct_settled": _roi_pct(fun_settled_profit, fun_settled_stake),
            "bankroll_start": round(fun_bankroll_start, 2),
            "bankroll_end": round(fun_bankroll_end, 2),
            "bankroll_end_settled": round(fun_bankroll_end_settled, 2),
            "pending": int(fun_pending),
            "singles": fun_out_singles,
            "system": fun_out_system,
            "system_pool": fun_pool,
            "score": {
                "singles": f"{fun_single_w}/{fun_single_t}",
                "system_pool": f"{fun_pool_wins}/{fun_pool_total}",
            },
        },

        "drawbet": {
            "portfolio": (draw.get("portfolio") or "DrawBet"),
            "stake": round(draw_stake, 2),
            "profit": round(draw_profit, 2),
            "roi_pct_total": _roi_pct(draw_profit, draw_stake),
            "settled_stake": round(draw_settled_stake, 2),
            "settled_profit": round(draw_settled_profit, 2),
            "roi_pct_settled": _roi_pct(draw_settled_profit, draw_settled_stake),
            "bankroll_start": round(draw_bankroll_start, 2),
            "bankroll_end": round(draw_bankroll_end, 2),
            "bankroll_end_settled": round(draw_bankroll_end_settled, 2),
            "pending": int(draw_pending),
            "singles": draw_out_singles,
            "system": draw_out_system,
            "system_pool": draw_pool,
            "score": {
                "system_pool": f"{draw_pool_wins}/{draw_pool_total}",
            },
        },

        "lifetime": {
            "week_count": int(history.get("week_count") or 0),
            "core": {
                "stake": round(safe_float((history.get("core") or {}).get("stake"), 0.0) or 0.0, 2),
                "profit": round(safe_float((history.get("core") or {}).get("profit"), 0.0) or 0.0, 2),
                "bankroll_current": (history.get("core") or {}).get("bankroll_current"),
            },
            "funbet": {
                "stake": round(safe_float((history.get("funbet") or {}).get("stake"), 0.0) or 0.0, 2),
                "profit": round(safe_float((history.get("funbet") or {}).get("profit"), 0.0) or 0.0, 2),
                "bankroll_current": (history.get("funbet") or {}).get("bankroll_current"),
            },
            "drawbet": {
                "stake": round(safe_float((history.get("drawbet") or {}).get("stake"), 0.0) or 0.0, 2),
                "profit": round(safe_float((history.get("drawbet") or {}).get("profit"), 0.0) or 0.0, 2),
                "bankroll_current": (history.get("drawbet") or {}).get("bankroll_current"),
            },
        },
    }

    save_json(TUESDAY_RECAP_PATH, report)
    log(f"✅ Tuesday Recap v3 written: {TUESDAY_RECAP_PATH}")


if __name__ == "__main__":
    main()
