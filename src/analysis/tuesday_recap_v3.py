# ============================================================
# src/analysis/tuesday_recap_v3.py
# BOMBAY TUESDAY RECAP v3.30 — PRODUCTION
#
# Reads:
#   - logs/friday_shortlist_v3.json
#   - logs/tuesday_history_v3.json (carry bankrolls + lifetime)
#
# Writes:
#   - logs/tuesday_recap_v3.json
#   - updates logs/tuesday_history_v3.json
#
# Rules:
#   - If fixture not finished => result MUST be "PENDING" and profit 0.
#   - Updates bankroll_current ONLY when that portfolio has no PENDING.
#   - Adds drawbet support.
#   - Adds copy_recap (array of strings) for the Custom GPT.
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

# ---------- utils ----------
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
            "weeks": {},
            "core": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
            "funbet": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
            "drawbet": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
            "updated_at": now_utc_iso(),
        }
    try:
        h = load_json(path)
        h.setdefault("week_count", 0)
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
            "weeks": {},
            "core": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
            "funbet": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
            "drawbet": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
            "updated_at": now_utc_iso(),
        }

# ---------- API-Football ----------
def _api_get_fixture(fixture_id: int) -> Optional[Dict[str, Any]]:
    if not API_FOOTBALL_KEY:
        return None

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"id": fixture_id}

    for _ in range(MAX_FETCH_RETRIES + 1):
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

# ---------- evaluation ----------
def market_won(market_code: str, hg: int, ag: int) -> bool:
    mc = (market_code or "").upper().strip()
    if mc == "1": return hg > ag
    if mc == "X": return hg == ag
    if mc == "2": return ag > hg
    if mc == "O25": return (hg + ag) >= 3
    if mc == "U25": return (hg + ag) <= 2
    return False

def eval_single(pick: Dict[str, Any], scores: Dict[int, Dict[str, Any]]) -> Tuple[str, float, Optional[str]]:
    fid = safe_int(pick.get("fixture_id"), None)
    mcode = (pick.get("market_code") or "").upper().strip()
    stake = safe_float(pick.get("stake"), 0.0) or 0.0
    odds = safe_float(pick.get("odds"), None)

    if fid is None or not mcode:
        return "PENDING", 0.0, None

    sc = scores.get(fid)
    if not sc:
        return "PENDING", 0.0, None

    st = (sc.get("status_short") or "").upper()
    hg = sc.get("home_goals"); ag = sc.get("away_goals")
    if st not in FINAL_STATUSES or hg is None or ag is None:
        return "PENDING", 0.0, None

    won = market_won(mcode, int(hg), int(ag))
    final_score = f"{int(hg)}-{int(ag)}"
    if odds is None or odds <= 1.0:
        return ("WIN" if won else "LOSE"), 0.0, final_score

    if won:
        return "WIN", round(stake * (odds - 1.0), 2), final_score
    return "LOSE", round(-stake, 2), final_score

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
            fid = None; code = None

        if fid is None or not code:
            legs_out.append({**leg, "result": "PENDING", "final_score": None})
            any_pending = True; all_final = False
            continue

        sc = scores.get(fid)
        if not sc:
            legs_out.append({**leg, "result": "PENDING", "final_score": None})
            any_pending = True; all_final = False
            continue

        st = (sc.get("status_short") or "").upper()
        hg = sc.get("home_goals"); ag = sc.get("away_goals")
        if st not in FINAL_STATUSES or hg is None or ag is None:
            legs_out.append({**leg, "result": "PENDING", "final_score": None})
            any_pending = True; all_final = False
            continue

        won = market_won(code, int(hg), int(ag))
        fs = f"{int(hg)}-{int(ag)}"
        if not won:
            any_lose = True
            legs_out.append({**leg, "result": "LOSE", "final_score": fs})
        else:
            legs_out.append({**leg, "result": "WIN", "final_score": fs})
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

def eval_system(system: Dict[str, Any], pool: List[Dict[str, Any]], scores: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], str, float, List[Dict[str, Any]]]:
    """
    Returns: (system_out, result, profit, pool_out_with_results)
    Pool picks can be shown with WIN/LOSE/PENDING + final_score.
    """
    if not system or not pool:
        return system or {}, "PENDING", 0.0, pool or []

    label = system.get("label")
    sizes = parse_system_sizes(label or "")
    n = len(pool)

    cols = safe_int(system.get("columns"), 0) or 0
    unit = safe_float(system.get("unit"), 0.0) or 0.0
    stake = safe_float(system.get("stake"), 0.0) or 0.0

    if not sizes or cols <= 0 or unit <= 0 or stake <= 0:
        return {**system, "result": "PENDING", "profit": 0.0}, "PENDING", 0.0, pool

    leg_state = []
    pool_out = []
    for r in pool:
        fid = safe_int(r.get("fixture_id"), None)
        code = (r.get("market_code") or "").upper().strip()
        odds = safe_float(r.get("odds"), None)

        if fid is None or not code:
            leg_state.append(("PENDING", 1.0))
            pool_out.append({**r, "result": "PENDING", "final_score": None})
            continue

        sc = scores.get(fid)
        if not sc:
            leg_state.append(("PENDING", 1.0))
            pool_out.append({**r, "result": "PENDING", "final_score": None})
            continue

        st = (sc.get("status_short") or "").upper()
        hg = sc.get("home_goals"); ag = sc.get("away_goals")
        if st not in FINAL_STATUSES or hg is None or ag is None:
            leg_state.append(("PENDING", 1.0))
            pool_out.append({**r, "result": "PENDING", "final_score": None})
            continue

        fs = f"{int(hg)}-{int(ag)}"
        won = market_won(code, int(hg), int(ag))
        if not won:
            leg_state.append(("LOSE", 1.0))
            pool_out.append({**r, "result": "LOSE", "final_score": fs})
        else:
            leg_state.append(("WIN", float(odds) if odds and odds > 1.0 else 1.0))
            pool_out.append({**r, "result": "WIN", "final_score": fs})

    if any(st == "PENDING" for st, _m in leg_state):
        return {**system, "result": "PENDING", "profit": 0.0}, "PENDING", 0.0, pool_out

    total_payout = 0.0
    idxs = list(range(n))
    for rsize in sizes:
        if rsize < 2 or rsize > n:
            continue
        for combo in combinations(idxs, rsize):
            if any(leg_state[i][0] == "LOSE" for i in combo):
                continue
            mult = 1.0
            for i in combo:
                mult *= leg_state[i][1]
            total_payout += unit * mult

    profit = round(total_payout - stake, 2)
    result = "WIN" if profit > 0 else "LOSE"

    out = {**system, "result": result, "profit": profit}
    return out, result, profit, pool_out

# ---------- helper recap + copy ----------
def score_hits(items: List[Dict[str, Any]]) -> Tuple[int,int,int]:
    wins = sum(1 for p in items if p.get("result") == "WIN")
    loses = sum(1 for p in items if p.get("result") == "LOSE")
    pend = sum(1 for p in items if p.get("result") == "PENDING")
    return wins, loses, pend

def build_copy_recap(recap: Dict[str, Any]) -> List[str]:
    lines = []
    wk = recap.get("week_label") or "—"
    w = recap.get("window") or {}
    lines.append(f'WEEK {wk} | Window {w.get("from","—")} – {w.get("to","—")}')

    for key in ["core","funbet","drawbet"]:
        if key not in recap:
            continue
        sec = recap.get(key) or {}
        lines.append(f'[{key.upper()}] stake={sec.get("stake","—")} profit={sec.get("profit","—")} roi={sec.get("roi_pct","—")} bankroll={sec.get("bankroll_start","—")}→{sec.get("bankroll_end","—")} pending={sec.get("pending","—")}')

    return lines

# ---------- main ----------
def main():
    if not os.path.exists(FRIDAY_REPORT_PATH):
        raise FileNotFoundError(f"Friday report not found: {FRIDAY_REPORT_PATH}")

    friday = load_json(FRIDAY_REPORT_PATH)
    window = friday.get("window") or {}
    window_from = window.get("from")
    week_id = iso_week_id_from_window(window)

    history = load_history(TUESDAY_HISTORY_PATH)

    # week_no increments when new window.from
    if window_from and window_from in history.get("weeks", {}):
        week_no = int(history["weeks"][window_from].get("week_no") or 1)
    else:
        week_no = int(history.get("week_count") or 0) + 1

    # bankroll start from history if present, else from friday bankroll_start
    core_bankroll_start = safe_float((history.get("core") or {}).get("bankroll_current"), None)
    if core_bankroll_start is None:
        core_bankroll_start = safe_float((friday.get("core") or {}).get("bankroll_start"), None)
    if core_bankroll_start is None:
        core_bankroll_start = safe_float((friday.get("core") or {}).get("bankroll"), 0.0) or 0.0

    fun_bankroll_start = safe_float((history.get("funbet") or {}).get("bankroll_current"), None)
    if fun_bankroll_start is None:
        fun_bankroll_start = safe_float((friday.get("funbet") or {}).get("bankroll_start"), None)
    if fun_bankroll_start is None:
        fun_bankroll_start = safe_float((friday.get("funbet") or {}).get("bankroll"), 0.0) or 0.0

    draw_bankroll_start = safe_float((history.get("drawbet") or {}).get("bankroll_current"), None)
    if draw_bankroll_start is None:
        draw_bankroll_start = safe_float((friday.get("drawbet") or {}).get("bankroll_start"), None)
    if draw_bankroll_start is None:
        draw_bankroll_start = safe_float((friday.get("drawbet") or {}).get("bankroll"), 0.0) or 0.0

    core = friday.get("core", {}) or {}
    fun  = friday.get("funbet", {}) or {}
    draw = friday.get("drawbet", {}) or {}

    core_singles = core.get("singles", []) or []
    core_doubles = core.get("doubles", []) or []
    core_double_primary = core.get("double") or {}

    fun_system = fun.get("system") or {}
    fun_pool   = fun.get("system_pool", []) or []

    draw_system = draw.get("system") or {}
    draw_pool   = draw.get("system_pool", []) or []

    # collect fixture ids
    fixture_ids: List[int] = []
    for p in core_singles + fun_pool + draw_pool:
        fid = safe_int(p.get("fixture_id"), None)
        if fid is not None:
            fixture_ids.append(fid)
    for d in core_doubles:
        for leg in (d.get("legs") or []):
            pid = leg.get("pick_id") or ""
            try:
                a, _b = pid.split(":")
                fid = safe_int(a, None)
                if fid is not None:
                    fixture_ids.append(fid)
            except Exception:
                pass

    scores = fetch_scores(fixture_ids)

    # CORE singles
    core_out_singles = []
    core_stake = 0.0
    core_profit = 0.0
    core_pending = 0

    for p in core_singles:
        res, prof, fs = eval_single(p, scores)
        stake = safe_float(p.get("stake"), 0.0) or 0.0
        core_stake += stake
        core_profit += prof
        if res == "PENDING":
            core_pending += 1
        core_out_singles.append({**p, "result": res, "profit": prof, "final_score": fs})

    # CORE doubles (list)
    core_out_doubles = []
    for d in core_doubles:
        outd, status, prof = eval_double(d, scores)
        st = safe_float(d.get("stake"), 0.0) or 0.0
        core_stake += st
        core_profit += prof
        if status == "PENDING":
            core_pending += 1
        core_out_doubles.append(outd)

    # FUN system
    fun_out_system, fun_sys_res, fun_sys_profit, fun_pool_out = eval_system(fun_system, fun_pool, scores)
    fun_sys_stake = safe_float(fun_system.get("stake"), 0.0) or 0.0
    fun_stake = fun_sys_stake
    fun_profit = fun_sys_profit
    fun_pending = 1 if fun_sys_res == "PENDING" else 0

    # DRAW system
    draw_out_system, draw_sys_res, draw_sys_profit, draw_pool_out = eval_system(draw_system, draw_pool, scores)
    draw_sys_stake = safe_float(draw_system.get("stake"), 0.0) or 0.0
    draw_stake = draw_sys_stake
    draw_profit = draw_sys_profit
    draw_pending = 1 if draw_sys_res == "PENDING" else 0

    # bankroll ends
    core_bankroll_end = round(core_bankroll_start + core_profit, 2)
    fun_bankroll_end  = round(fun_bankroll_start + fun_profit, 2)
    draw_bankroll_end = round(draw_bankroll_start + draw_profit, 2)

    # update history (per-portfolio only if no pending)
    wk_key = window_from or week_id
    history.setdefault("weeks", {})
    prev_week = history["weeks"].get(wk_key)

    # If rerun same week and fully settled, subtract old contributions then add new
    def subtract_prev(section: str, stake_new: float, profit_new: float, settled_ok: bool):
        if not settled_ok:
            return
        nonlocal history, prev_week
        if isinstance(prev_week, dict):
            old = (prev_week.get(section) or {})
            old_stake = safe_float(old.get("stake"), 0.0) or 0.0
            old_profit = safe_float(old.get("profit"), 0.0) or 0.0
            history[section]["stake"] = round((safe_float(history[section].get("stake"), 0.0) or 0.0) - old_stake, 2)
            history[section]["profit"] = round((safe_float(history[section].get("profit"), 0.0) or 0.0) - old_profit, 2)

        history[section]["stake"] = round((safe_float(history[section].get("stake"), 0.0) or 0.0) + stake_new, 2)
        history[section]["profit"] = round((safe_float(history[section].get("profit"), 0.0) or 0.0) + profit_new, 2)

    core_settled_ok = (core_pending == 0)
    fun_settled_ok  = (fun_pending == 0)
    draw_settled_ok = (draw_pending == 0)

    subtract_prev("core", core_stake, core_profit, core_settled_ok)
    subtract_prev("funbet", fun_stake, fun_profit, fun_settled_ok)
    subtract_prev("drawbet", draw_stake, draw_profit, draw_settled_ok)

    if core_settled_ok:
        history["core"]["bankroll_current"] = core_bankroll_end
    if fun_settled_ok:
        history["funbet"]["bankroll_current"] = fun_bankroll_end
    if draw_settled_ok:
        history["drawbet"]["bankroll_current"] = draw_bankroll_end

    if window_from and window_from not in history["weeks"]:
        history["week_count"] = int(history.get("week_count") or 0) + 1

    history["weeks"][wk_key] = {
        "week_no": int(week_no),
        "week_id": week_id,
        "window": window,
        "core": {"stake": round(core_stake, 2), "profit": round(core_profit, 2), "pending": core_pending},
        "funbet": {"stake": round(fun_stake, 2), "profit": round(fun_profit, 2), "pending": fun_pending},
        "drawbet": {"stake": round(draw_stake, 2), "profit": round(draw_profit, 2), "pending": draw_pending},
        "updated_at": now_utc_iso(),
    }
    history["updated_at"] = now_utc_iso()
    save_json(TUESDAY_HISTORY_PATH, history)

    core_roi = round((core_profit / core_stake) * 100.0, 2) if core_stake > 0 else None
    fun_roi  = round((fun_profit / fun_stake) * 100.0, 2) if fun_stake > 0 else None
    draw_roi = round((draw_profit / draw_stake) * 100.0, 2) if draw_stake > 0 else None

    # scores
    cw, cl, cp = score_hits(core_out_singles)
    fw, fl, fp = score_hits(fun_pool_out)
    dw, dl, dp = score_hits(draw_pool_out)

    recap = {
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
            "score": {
                "singles": f"{cw}/{len(core_out_singles)}" if core_out_singles else "0/0",
                "doubles": "—" if not core_out_doubles else f"{sum(1 for d in core_out_doubles if d.get('result')=='WIN')}/{len(core_out_doubles)}",
            },
            "singles": core_out_singles,
            "doubles": core_out_doubles,
        },

        "funbet": {
            "stake": round(fun_stake, 2),
            "profit": round(fun_profit, 2),
            "roi_pct": fun_roi,
            "bankroll_start": round(fun_bankroll_start, 2),
            "bankroll_end": round(fun_bankroll_end, 2),
            "pending": int(fun_pending),
            "score": {
                "system_pool": f"{fw}/{len(fun_pool_out)}" if fun_pool_out else "0/0",
            },
            "system": fun_out_system,
            "system_pool": fun_pool_out,
        },

        "drawbet": {
            "stake": round(draw_stake, 2),
            "profit": round(draw_profit, 2),
            "roi_pct": draw_roi,
            "bankroll_start": round(draw_bankroll_start, 2),
            "bankroll_end": round(draw_bankroll_end, 2),
            "pending": int(draw_pending),
            "score": {
                "system_pool": f"{dw}/{len(draw_pool_out)}" if draw_pool_out else "0/0",
            },
            "system": draw_out_system,
            "system_pool": draw_pool_out,
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
    }

    recap["copy_recap"] = build_copy_recap(recap)

    save_json(TUESDAY_RECAP_PATH, recap)
    log(f"✅ Tuesday Recap v3 written: {TUESDAY_RECAP_PATH}")

if __name__ == "__main__":
    main()
