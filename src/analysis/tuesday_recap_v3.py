# ============================================================
#  src/analysis/tuesday_recap_v3.py
#  BOMBAY TUESDAY RECAP v3.30 — PRODUCTION (CoreBet + FunBet + DrawBet)
#
#  Reads:
#   - logs/friday_shortlist_v3.json
#  Writes:
#   - logs/tuesday_recap_v3.json
#   - logs/tuesday_history_v3.json (carry bankrolls + lifetime)
#
#  Key rules:
#   - If ANY relevant match is not settled -> result MUST be "PENDING" and
#     history bankrolls MUST NOT advance for that portfolio.
#   - Week numbering increments ONLY when window.from changes (reruns same window overwrite).
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


# ------------------------- UTIL -------------------------
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
    """
    History structure:
      week_count, last_window_from, last_week_id, weeks{key}, core{stake,profit,bankroll_current}, funbet{...}, drawbet{...}
    """
    if not os.path.exists(path):
        return {
            "week_count": 0,
            "last_window_from": None,
            "last_week_id": None,
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
        h.setdefault("last_week_id", None)
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
        return h
    except Exception:
        return {
            "week_count": 0,
            "last_window_from": None,
            "last_week_id": None,
            "weeks": {},
            "core": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
            "funbet": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
            "drawbet": {"stake": 0.0, "profit": 0.0, "bankroll_current": None},
            "updated_at": now_utc_iso(),
        }


# ------------------------- API-FOOTBALL -------------------------
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


# ------------------------- SETTLEMENT LOGIC -------------------------
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


def _final_score_str(hg: Optional[int], ag: Optional[int]) -> Optional[str]:
    if hg is None or ag is None:
        return None
    return f"{int(hg)}-{int(ag)}"


def eval_single(pick: Dict[str, Any], scores: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], str, float]:
    fid = safe_int(pick.get("fixture_id"), None)
    mcode = (pick.get("market_code") or "").upper().strip()
    stake = safe_float(pick.get("stake"), 0.0) or 0.0
    odds = safe_float(pick.get("odds"), None)

    if fid is None or not mcode:
        out = {**pick, "result": "PENDING", "profit": 0.0, "final_score": None, "status_short": None}
        return out, "PENDING", 0.0

    sc = scores.get(fid)
    if not sc:
        out = {**pick, "result": "PENDING", "profit": 0.0, "final_score": None, "status_short": None}
        return out, "PENDING", 0.0

    st = (sc.get("status_short") or "").upper()
    hg = sc.get("home_goals")
    ag = sc.get("away_goals")

    if st not in FINAL_STATUSES or hg is None or ag is None:
        out = {**pick, "result": "PENDING", "profit": 0.0, "final_score": _final_score_str(hg, ag), "status_short": st}
        return out, "PENDING", 0.0

    won = market_won(mcode, int(hg), int(ag))
    if odds is None or odds <= 1.0:
        prof = 0.0
    else:
        prof = round(stake * (odds - 1.0), 2) if won else round(-stake, 2)

    res = "WIN" if won else "LOSE"
    out = {**pick, "result": res, "profit": prof, "final_score": _final_score_str(hg, ag), "status_short": st}
    return out, res, prof


def eval_double(double_obj: Dict[str, Any], scores: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], str, float]:
    if not double_obj:
        return {}, "PENDING", 0.0

    legs = double_obj.get("legs") or []
    stake = safe_float(double_obj.get("stake"), 0.0) or 0.0
    combo_odds = safe_float(double_obj.get("combo_odds"), None)

    legs_out = []
    any_pending = False
    any_lose = False

    mult = 1.0
    for leg in legs:
        fid = safe_int(leg.get("fixture_id"), None)
        mcode = (leg.get("market_code") or "").upper().strip()

        # fallback parse from pick_id if missing
        if (fid is None or not mcode) and leg.get("pick_id"):
            try:
                a, b = str(leg.get("pick_id")).split(":")
                fid = safe_int(a, fid)
                mcode = (b or mcode).upper().strip()
            except Exception:
                pass

        if fid is None or not mcode:
            legs_out.append({**leg, "result": "PENDING", "final_score": None})
            any_pending = True
            continue

        sc = scores.get(fid)
        if not sc:
            legs_out.append({**leg, "result": "PENDING", "final_score": None})
            any_pending = True
            continue

        st = (sc.get("status_short") or "").upper()
        hg = sc.get("home_goals")
        ag = sc.get("away_goals")

        if st not in FINAL_STATUSES or hg is None or ag is None:
            legs_out.append({**leg, "result": "PENDING", "final_score": _final_score_str(hg, ag)})
            any_pending = True
            continue

        won = market_won(mcode, int(hg), int(ag))
        if not won:
            any_lose = True
            legs_out.append({**leg, "result": "LOSE", "final_score": _final_score_str(hg, ag)})
        else:
            legs_out.append({**leg, "result": "WIN", "final_score": _final_score_str(hg, ag)})
            o = safe_float(leg.get("odds"), None)
            if o is not None and o > 1.0:
                mult *= float(o)

    if any_pending:
        out = {**double_obj, "legs": legs_out, "result": "PENDING", "profit": 0.0}
        return out, "PENDING", 0.0

    if any_lose:
        prof = round(-stake, 2) if stake > 0 else 0.0
        out = {**double_obj, "legs": legs_out, "result": "LOSE", "profit": prof}
        return out, "LOSE", prof

    # WIN
    if combo_odds is not None and combo_odds > 1.0:
        mult_used = combo_odds
    else:
        mult_used = mult

    prof = round(stake * (mult_used - 1.0), 2) if stake > 0 else 0.0
    out = {**double_obj, "legs": legs_out, "result": "WIN", "profit": prof}
    return out, "WIN", prof


def parse_system_sizes(label: str, pool_n: int) -> List[int]:
    """
    label examples:
      "4/7" => [4]
      "2-3/5" => [2,3]
      "3/6" => [3]
      "2/2" => [2]
    """
    if not label:
        return []
    left = label.split("/")[0].strip()
    parts = [p.strip() for p in left.split("-") if p.strip()]
    sizes = []
    for p in parts:
        try:
            sizes.append(int(p))
        except Exception:
            pass
    # keep only valid
    return [r for r in sizes if 1 <= r <= pool_n]


def eval_system(system: Dict[str, Any], pool: List[Dict[str, Any]], scores: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], str, float, List[Dict[str, Any]]]:
    """
    Evaluates system using:
      payout = sum(unit * product(odds in winning combos)) for each size r in sizes
      profit = payout - stake
    Returns updated system object + result + profit + pool_with_results
    """
    if not system or not pool:
        return system or {}, "PENDING", 0.0, pool

    label = system.get("label")
    unit = safe_float(system.get("unit"), 0.0) or 0.0
    stake = safe_float(system.get("stake"), 0.0) or 0.0
    columns = safe_int(system.get("columns"), 0) or 0

    n = len(pool)
    sizes = parse_system_sizes(label or "", n)

    # If missing basics => treat as no system
    if not label or not sizes or unit <= 0 or stake <= 0 or columns <= 0:
        return {**system, "result": "PENDING", "profit": 0.0}, "PENDING", 0.0, pool

    # resolve pool leg outcomes
    pool_out = []
    leg_state: List[Tuple[str, float]] = []  # ("WIN"/"LOSE"/"PENDING", odds_if_win)
    for r in pool:
        fid = safe_int(r.get("fixture_id"), None)
        code = (r.get("market_code") or "").upper().strip()
        odds = safe_float(r.get("odds"), None)

        if fid is None or not code:
            pool_out.append({**r, "result": "PENDING", "final_score": None})
            leg_state.append(("PENDING", 1.0))
            continue

        sc = scores.get(fid)
        if not sc:
            pool_out.append({**r, "result": "PENDING", "final_score": None})
            leg_state.append(("PENDING", 1.0))
            continue

        st = (sc.get("status_short") or "").upper()
        hg = sc.get("home_goals")
        ag = sc.get("away_goals")

        if st not in FINAL_STATUSES or hg is None or ag is None:
            pool_out.append({**r, "result": "PENDING", "final_score": _final_score_str(hg, ag)})
            leg_state.append(("PENDING", 1.0))
            continue

        won = market_won(code, int(hg), int(ag))
        if not won:
            pool_out.append({**r, "result": "LOSE", "final_score": _final_score_str(hg, ag)})
            leg_state.append(("LOSE", 1.0))
        else:
            pool_out.append({**r, "result": "WIN", "final_score": _final_score_str(hg, ag)})
            leg_state.append(("WIN", float(odds) if odds and odds > 1.0 else 1.0))

    if any(st == "PENDING" for st, _o in leg_state):
        return {**system, "result": "PENDING", "profit": 0.0}, "PENDING", 0.0, pool_out

    # settled payout
    idxs = list(range(n))
    payout = 0.0
    for rsize in sizes:
        for combo in combinations(idxs, rsize):
            if any(leg_state[i][0] == "LOSE" for i in combo):
                continue
            mult = 1.0
            for i in combo:
                mult *= leg_state[i][1]
            payout += unit * mult

    profit = round(payout - stake, 2)
    result = "WIN" if profit > 0 else "LOSE"
    out = {**system, "result": result, "profit": profit, "payout": round(payout, 2)}
    return out, result, profit, pool_out


def roi_pct(profit: float, stake: float) -> Optional[float]:
    if stake <= 0:
        return None
    return round((profit / stake) * 100.0, 2)


def wins_over_total(items: List[Dict[str, Any]]) -> str:
    total = len(items)
    wins = sum(1 for x in items if x.get("result") == "WIN")
    return f"{wins}/{total}"


# ------------------------- MAIN -------------------------
def main():
    if not os.path.exists(FRIDAY_REPORT_PATH):
        raise FileNotFoundError(f"Friday report not found: {FRIDAY_REPORT_PATH}")

    friday = load_json(FRIDAY_REPORT_PATH)
    window = friday.get("window") or {}
    window_from = (window or {}).get("from")
    week_id = iso_week_id_from_window(window)

    history = load_history(TUESDAY_HISTORY_PATH)

    # week_no increments ONLY when window.from changes
    if window_from and window_from in (history.get("weeks") or {}):
        week_no = int(history["weeks"][window_from].get("week_no") or 1)
    else:
        week_no = int(history.get("week_count") or 0) + 1

    # Sections
    core = friday.get("core", {}) or {}
    fun = friday.get("funbet", {}) or {}
    draw = friday.get("drawbet", {}) or {}

    # bankroll start: history carry else friday bankroll_start else friday bankroll
    def bankroll_start_for(port_key: str, friday_block: Dict[str, Any]) -> float:
        from_hist = safe_float((history.get(port_key) or {}).get("bankroll_current"), None)
        if from_hist is not None:
            return float(from_hist)
        fb = safe_float(friday_block.get("bankroll_start"), None)
        if fb is not None:
            return float(fb)
        b = safe_float(friday_block.get("bankroll"), 0.0) or 0.0
        return float(b)

    core_bankroll_start = bankroll_start_for("core", core)
    fun_bankroll_start = bankroll_start_for("funbet", fun)
    draw_bankroll_start = bankroll_start_for("drawbet", draw)

    # Picks
    core_singles = core.get("singles", []) or []
    core_double = core.get("double") or {}
    core_doubles = core.get("doubles", []) or []
    # use "doubles" list if exists; else fall back to "double"
    if not core_doubles and core_double:
        core_doubles = [core_double]

    fun_system = fun.get("system") or {}
    fun_pool = fun.get("system_pool", []) or []
    fun_singles = fun.get("singles", []) or []  # just in case older JSON includes singles

    draw_system = draw.get("system") or {}
    draw_pool = draw.get("system_pool", []) or []
    draw_singles = draw.get("singles", []) or []

    # Collect fixture ids
    fixture_ids: List[int] = []
    for p in core_singles + fun_singles + fun_pool + draw_singles + draw_pool:
        fid = safe_int(p.get("fixture_id"), None)
        if fid is not None:
            fixture_ids.append(fid)

    for dbl in core_doubles:
        for leg in (dbl.get("legs") or []):
            fid = safe_int(leg.get("fixture_id"), None)
            if fid is None and leg.get("pick_id"):
                try:
                    a, _b = str(leg.get("pick_id")).split(":")
                    fid = safe_int(a, None)
                except Exception:
                    fid = None
            if fid is not None:
                fixture_ids.append(fid)

    scores = fetch_scores(fixture_ids)

    # ---------------- CORE eval ----------------
    core_out_singles = []
    core_stake = 0.0
    core_profit = 0.0
    core_pending = 0

    for p in core_singles:
        out, res, prof = eval_single(p, scores)
        st = safe_float(p.get("stake"), 0.0) or 0.0
        core_stake += st
        core_profit += prof
        if res == "PENDING":
            core_pending += 1
        core_out_singles.append(out)

    core_out_doubles = []
    for dbl in core_doubles:
        out, res, prof = eval_double(dbl, scores)
        st = safe_float(dbl.get("stake"), 0.0) or 0.0
        core_stake += st
        core_profit += prof
        if res == "PENDING":
            core_pending += 1
        core_out_doubles.append(out)

    core_bankroll_end = round(core_bankroll_start + core_profit, 2)
    core_roi = roi_pct(core_profit, core_stake)

    # ---------------- FUN eval ----------------
    fun_out_singles = []
    fun_stake = 0.0
    fun_profit = 0.0
    fun_pending = 0

    for p in fun_singles:
        out, res, prof = eval_single(p, scores)
        st = safe_float(p.get("stake"), 0.0) or 0.0
        fun_stake += st
        fun_profit += prof
        if res == "PENDING":
            fun_pending += 1
        fun_out_singles.append(out)

    fun_out_system, fun_sys_res, fun_sys_profit, fun_pool_out = eval_system(fun_system, fun_pool, scores)
    fun_sys_stake = safe_float(fun_system.get("stake"), 0.0) or 0.0
    fun_stake += fun_sys_stake
    fun_profit += fun_sys_profit
    if fun_system and fun_sys_res == "PENDING":
        fun_pending += 1

    fun_bankroll_end = round(fun_bankroll_start + fun_profit, 2)
    fun_roi = roi_pct(fun_profit, fun_stake)

    # ---------------- DRAW eval ----------------
    draw_out_singles = []
    draw_stake = 0.0
    draw_profit = 0.0
    draw_pending = 0

    for p in draw_singles:
        out, res, prof = eval_single(p, scores)
        st = safe_float(p.get("stake"), 0.0) or 0.0
        draw_stake += st
        draw_profit += prof
        if res == "PENDING":
            draw_pending += 1
        draw_out_singles.append(out)

    draw_out_system, draw_sys_res, draw_sys_profit, draw_pool_out = eval_system(draw_system, draw_pool, scores)
    draw_sys_stake = safe_float(draw_system.get("stake"), 0.0) or 0.0
    draw_stake += draw_sys_stake
    draw_profit += draw_sys_profit
    if draw_system and draw_sys_res == "PENDING":
        draw_pending += 1

    draw_bankroll_end = round(draw_bankroll_start + draw_profit, 2)
    draw_roi = roi_pct(draw_profit, draw_stake)

    # ---------------- HISTORY UPDATE ----------------
    wk_key = window_from or week_id
    history.setdefault("weeks", {})
    prev_week = history["weeks"].get(wk_key)

    def subtract_prev_if_settled(port: str):
        if not isinstance(prev_week, dict):
            return
        if not bool(prev_week.get("fully_settled")):
            return
        old = (prev_week.get(port) or {})
        old_stake = safe_float(old.get("stake"), 0.0) or 0.0
        old_profit = safe_float(old.get("profit"), 0.0) or 0.0
        history[port]["stake"] = round((safe_float(history[port].get("stake"), 0.0) or 0.0) - old_stake, 2)
        history[port]["profit"] = round((safe_float(history[port].get("profit"), 0.0) or 0.0) - old_profit, 2)

    # Only advance bankroll_current for portfolios that are fully settled (no pending in that portfolio)
    core_settled = (core_pending == 0)
    fun_settled = (fun_pending == 0)
    draw_settled = (draw_pending == 0)

    fully_settled_all = bool(core_settled and fun_settled and draw_settled)

    # If re-run same window and previous was settled, remove previous contributions first
    if fully_settled_all and isinstance(prev_week, dict) and bool(prev_week.get("fully_settled")):
        subtract_prev_if_settled("core")
        subtract_prev_if_settled("funbet")
        subtract_prev_if_settled("drawbet")

    # Add new contributions (only for settled portfolios)
    if core_settled:
        history["core"]["stake"] = round((safe_float(history["core"].get("stake"), 0.0) or 0.0) + core_stake, 2)
        history["core"]["profit"] = round((safe_float(history["core"].get("profit"), 0.0) or 0.0) + core_profit, 2)
        history["core"]["bankroll_current"] = core_bankroll_end

    if fun_settled:
        history["funbet"]["stake"] = round((safe_float(history["funbet"].get("stake"), 0.0) or 0.0) + fun_stake, 2)
        history["funbet"]["profit"] = round((safe_float(history["funbet"].get("profit"), 0.0) or 0.0) + fun_profit, 2)
        history["funbet"]["bankroll_current"] = fun_bankroll_end

    if draw_settled:
        history["drawbet"]["stake"] = round((safe_float(history["drawbet"].get("stake"), 0.0) or 0.0) + draw_stake, 2)
        history["drawbet"]["profit"] = round((safe_float(history["drawbet"].get("profit"), 0.0) or 0.0) + draw_profit, 2)
        history["drawbet"]["bankroll_current"] = draw_bankroll_end

    # week_count increments only when NEW window_from key appears
    if window_from and window_from not in history["weeks"]:
        history["week_count"] = int(history.get("week_count") or 0) + 1

    history["last_window_from"] = window_from
    history["last_week_id"] = week_id

    history["weeks"][wk_key] = {
        "week_no": int(week_no),
        "week_id": week_id,
        "window": window,
        "core": {"stake": round(core_stake, 2), "profit": round(core_profit, 2), "pending": core_pending},
        "funbet": {"stake": round(fun_stake, 2), "profit": round(fun_profit, 2), "pending": fun_pending},
        "drawbet": {"stake": round(draw_stake, 2), "profit": round(draw_profit, 2), "pending": draw_pending},
        "fully_settled": bool(fully_settled_all),
        "updated_at": now_utc_iso(),
    }

    history["updated_at"] = now_utc_iso()
    save_json(TUESDAY_HISTORY_PATH, history)

    # ---------------- RECAP JSON ----------------
    # Weekly totals (computed by engine)
    total_stake = round(core_stake + fun_stake + draw_stake, 2)
    total_profit = round(core_profit + fun_profit + draw_profit, 2)
    total_roi = roi_pct(total_profit, total_stake)

    total_bankroll_start = round(core_bankroll_start + fun_bankroll_start + draw_bankroll_start, 2)
    total_bankroll_end = round(core_bankroll_end + fun_bankroll_end + draw_bankroll_end, 2)

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
            "roi_pct": core_roi,
            "bankroll_start": round(core_bankroll_start, 2),
            "bankroll_end": round(core_bankroll_end, 2),
            "pending": int(core_pending),
            "score": {
                "singles": wins_over_total(core_out_singles),
                "doubles": wins_over_total(core_out_doubles),
            },
            "singles": sorted(core_out_singles, key=lambda x: (x.get("date") or "", x.get("time") or "", x.get("league") or "", x.get("match") or "")),
            "doubles": core_out_doubles,
            "double": (core_out_doubles[0] if core_out_doubles else None),
        },

        "funbet": {
            "portfolio": (fun.get("portfolio") or "FunBet"),
            "stake": round(fun_stake, 2),
            "profit": round(fun_profit, 2),
            "roi_pct": fun_roi,
            "bankroll_start": round(fun_bankroll_start, 2),
            "bankroll_end": round(fun_bankroll_end, 2),
            "pending": int(fun_pending),
            "score": {
                "singles": wins_over_total(fun_out_singles),
                "system_pool": wins_over_total(fun_pool_out),
            },
            "singles": sorted(fun_out_singles, key=lambda x: (x.get("date") or "", x.get("time") or "", x.get("league") or "", x.get("match") or "")),
            "system": fun_out_system,
            "system_pool": sorted(fun_pool_out, key=lambda x: (x.get("date") or "", x.get("time") or "", x.get("league") or "", x.get("match") or "")),
        },

        "drawbet": {
            "portfolio": (draw.get("portfolio") or "DrawBet"),
            "stake": round(draw_stake, 2),
            "profit": round(draw_profit, 2),
            "roi_pct": draw_roi,
            "bankroll_start": round(draw_bankroll_start, 2),
            "bankroll_end": round(draw_bankroll_end, 2),
            "pending": int(draw_pending),
            "score": {
                "singles": wins_over_total(draw_out_singles),
                "system_pool": wins_over_total(draw_pool_out),
            },
            "singles": sorted(draw_out_singles, key=lambda x: (x.get("date") or "", x.get("time") or "", x.get("league") or "", x.get("match") or "")),
            "system": draw_out_system,
            "system_pool": sorted(draw_pool_out, key=lambda x: (x.get("date") or "", x.get("time") or "", x.get("league") or "", x.get("match") or "")),
        },

        "weekly_summary": {
            "this_week": {
                "stake": total_stake,
                "profit": total_profit,
                "roi_pct": total_roi,
                "bankroll_start": total_bankroll_start,
                "bankroll_end": total_bankroll_end,
                "pending_any": bool(core_pending > 0 or fun_pending > 0 or draw_pending > 0),
            },
            "lifetime": {
                "week_count": int(history.get("week_count") or 0),
                "core": {
                    "stake": round(safe_float(history["core"].get("stake"), 0.0) or 0.0, 2),
                    "profit": round(safe_float(history["core"].get("profit"), 0.0) or 0.0, 2),
                    "roi_pct": roi_pct(
                        float(safe_float(history["core"].get("profit"), 0.0) or 0.0),
                        float(safe_float(history["core"].get("stake"), 0.0) or 0.0),
                    ),
                    "bankroll_current": history["core"].get("bankroll_current"),
                },
                "funbet": {
                    "stake": round(safe_float(history["funbet"].get("stake"), 0.0) or 0.0, 2),
                    "profit": round(safe_float(history["funbet"].get("profit"), 0.0) or 0.0, 2),
                    "roi_pct": roi_pct(
                        float(safe_float(history["funbet"].get("profit"), 0.0) or 0.0),
                        float(safe_float(history["funbet"].get("stake"), 0.0) or 0.0),
                    ),
                    "bankroll_current": history["funbet"].get("bankroll_current"),
                },
                "drawbet": {
                    "stake": round(safe_float(history["drawbet"].get("stake"), 0.0) or 0.0, 2),
                    "profit": round(safe_float(history["drawbet"].get("profit"), 0.0) or 0.0, 2),
                    "roi_pct": roi_pct(
                        float(safe_float(history["drawbet"].get("profit"), 0.0) or 0.0),
                        float(safe_float(history["drawbet"].get("stake"), 0.0) or 0.0),
                    ),
                    "bankroll_current": history["drawbet"].get("bankroll_current"),
                },
            },
        },
    }

    save_json(TUESDAY_RECAP_PATH, report)
    log(f"✅ Tuesday Recap written: {TUESDAY_RECAP_PATH}")
    log(f"✅ Tuesday History updated: {TUESDAY_HISTORY_PATH}")


if __name__ == "__main__":
    main()
