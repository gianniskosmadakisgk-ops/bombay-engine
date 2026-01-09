# ============================================================
#  BOMBAY TUESDAY RECAP v3 — PRODUCTION (BANKROLL + SCORE + LIFETIME)
#
#  Reads:  logs/friday_shortlist_v3.json
#  Writes: logs/tuesday_recap_v3.json
#          logs/tuesday_history_v3.json
#
#  Presenter requirements:
#   - If match not finished -> result MUST be "PENDING" in JSON.
#   - NO calculations by Presenter; therefore ROI%, bankroll, score must be produced here.
#
#  Additive fields (schema-safe):
#   - week_id, week_label
#   - bankroll_start / bankroll_end_settled / bankroll_end_final (per Core/Fun)
#   - score strings: "hits/total" for singles, system pool, double legs
#   - lifetime totals (stake/profit/roi_pct) + lifetime bankroll start/end
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

FINAL_STATUSES = {"FT", "AET", "PEN"}  # settled


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
    History schema (self-contained, additive, migration-safe):
    {
      "initial_bankroll": {"core": 800.0, "funbet": 400.0},
      "week_index": {"2026-W02": 1, ...},
      "lifetime": {
        "core": {"stake": 0.0, "profit": 0.0},
        "funbet": {"stake": 0.0, "profit": 0.0}
      },
      "updated_at": "..."
    }
    """
    if not os.path.exists(path):
        return {
            "initial_bankroll": {"core": None, "funbet": None},
            "week_index": {},
            "lifetime": {"core": {"stake": 0.0, "profit": 0.0}, "funbet": {"stake": 0.0, "profit": 0.0}},
            "updated_at": now_utc_iso(),
        }
    try:
        h = load_json(path)
    except Exception:
        h = {}

    # migrate older formats
    if "lifetime" not in h:
        # older file might be {"core":{stake,profit}, "funbet":{...}}
        core = h.get("core") or {"stake": 0.0, "profit": 0.0}
        fun = h.get("funbet") or {"stake": 0.0, "profit": 0.0}
        h = {
            "initial_bankroll": {"core": None, "funbet": None},
            "week_index": {},
            "lifetime": {"core": core, "funbet": fun},
            "updated_at": h.get("updated_at", now_utc_iso()),
        }

    h.setdefault("initial_bankroll", {"core": None, "funbet": None})
    h.setdefault("week_index", {})
    h.setdefault("lifetime", {"core": {"stake": 0.0, "profit": 0.0}, "funbet": {"stake": 0.0, "profit": 0.0}})
    h.setdefault("updated_at", now_utc_iso())
    return h


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


def eval_double(double_obj: Dict[str, Any], scores: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], str, float, str]:
    """
    Returns:
      - enriched double obj (legs results)
      - overall status
      - profit
      - score string "wins/legs" or "PENDING"
    """
    if not double_obj:
        return {}, "PENDING", 0.0, "—"

    legs = double_obj.get("legs") or []
    stake = safe_float(double_obj.get("stake"), 0.0) or 0.0

    mult = 1.0
    any_pending = False
    any_lose = False
    wins = 0
    total_legs = 0

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

        total_legs += 1

        if fid is None or not code:
            legs_out.append({**leg, "result": "PENDING"})
            any_pending = True
            continue

        sc = scores.get(fid)
        if not sc:
            legs_out.append({**leg, "result": "PENDING"})
            any_pending = True
            continue

        st = (sc.get("status_short") or "").upper()
        hg = sc.get("home_goals")
        ag = sc.get("away_goals")

        if st not in FINAL_STATUSES or hg is None or ag is None:
            legs_out.append({**leg, "result": "PENDING"})
            any_pending = True
            continue

        won = market_won(code, int(hg), int(ag))
        if not won:
            any_lose = True
            legs_out.append({**leg, "result": "LOSE"})
        else:
            wins += 1
            legs_out.append({**leg, "result": "WIN"})
            if odds is not None and odds > 1.0:
                mult *= float(odds)

    if any_pending:
        status = "PENDING"
        profit = 0.0
        score = "PENDING"
    elif any_lose:
        status = "LOSE"
        profit = round(-stake, 2) if stake > 0 else 0.0
        score = f"{wins}/{max(1,total_legs)}"
    else:
        status = "WIN"
        profit = round((stake * mult) - stake, 2) if stake > 0 else 0.0
        score = f"{wins}/{max(1,total_legs)}"

    out = {**double_obj, "legs": legs_out, "result": status, "profit": profit}
    return out, status, profit, score


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


def eval_system(system: Dict[str, Any], pool: List[Dict[str, Any]], scores: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], str, float, str]:
    """
    Returns:
      - enriched system obj (with result+profit)
      - result
      - profit
      - score_pool string "wins/n" or "PENDING"
    """
    if not system or not pool:
        return system or {}, "PENDING", 0.0, "—"

    # Pool hit score
    wins = 0
    pend = 0
    for r in pool:
        fid = safe_int(r.get("fixture_id"), None)
        code = (r.get("market_code") or "").upper().strip()
        if fid is None or not code:
            pend += 1
            continue
        sc = scores.get(fid)
        if not sc:
            pend += 1
            continue
        st = (sc.get("status_short") or "").upper()
        hg = sc.get("home_goals")
        ag = sc.get("away_goals")
        if st not in FINAL_STATUSES or hg is None or ag is None:
            pend += 1
            continue
        if market_won(code, int(hg), int(ag)):
            wins += 1

    if pend > 0:
        pool_score = "PENDING"
    else:
        pool_score = f"{wins}/{len(pool)}"

    label = system.get("label")
    sizes = parse_system_sizes(label or "")
    n = len(pool)

    cols = safe_int(system.get("columns"), 0) or 0
    unit = safe_float(system.get("unit"), 0.0) or 0.0
    stake = safe_float(system.get("stake"), 0.0) or 0.0

    if not sizes or cols <= 0 or unit <= 0 or stake <= 0:
        return {**system, "result": "PENDING", "profit": 0.0}, "PENDING", 0.0, pool_score

    # leg states for settlement
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
        return {**system, "result": "PENDING", "profit": 0.0}, "PENDING", 0.0, pool_score

    # settled system profit
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
    return out, result, profit, pool_score


def roi_pct(profit: float, stake: float) -> Optional[float]:
    s = safe_float(stake, 0.0) or 0.0
    p = safe_float(profit, 0.0) or 0.0
    if s <= 0:
        return None
    return round((p / s) * 100.0, 2)


def main():
    if not os.path.exists(FRIDAY_REPORT_PATH):
        raise FileNotFoundError(f"Friday report not found: {FRIDAY_REPORT_PATH}")

    friday = load_json(FRIDAY_REPORT_PATH)
    window = friday.get("window") or {}
    week_id = iso_week_id_from_window(window)

    history = load_history(TUESDAY_HISTORY_PATH)

    # Assign week_label (Week 1, Week 2...) deterministically per week_id
    week_index = history.get("week_index") or {}
    if week_id not in week_index:
        week_index[week_id] = int(len(week_index) + 1)
    history["week_index"] = week_index
    week_label = f"Week {week_index[week_id]}"

    # Initialize baseline bankrolls once
    init_b = history.get("initial_bankroll") or {"core": None, "funbet": None}
    core_bankroll_start = safe_float((friday.get("core") or {}).get("bankroll"), None)
    fun_bankroll_start = safe_float((friday.get("funbet") or {}).get("bankroll"), None)

    if init_b.get("core") is None and core_bankroll_start is not None:
        init_b["core"] = core_bankroll_start
    if init_b.get("funbet") is None and fun_bankroll_start is not None:
        init_b["funbet"] = fun_bankroll_start
    history["initial_bankroll"] = init_b

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

    # ---- CORE singles ----
    core_out_singles = []
    core_stake_singles = 0.0
    core_profit_singles = 0.0
    core_pending_singles = 0
    core_hits_singles = 0

    for p in core_singles:
        res, prof = eval_single(p, scores)
        stake = safe_float(p.get("stake"), 0.0) or 0.0
        core_stake_singles += stake
        core_profit_singles += prof
        if res == "PENDING":
            core_pending_singles += 1
        if res == "WIN":
            core_hits_singles += 1
        core_out_singles.append({**p, "result": res, "profit": prof})

    core_score_singles = "PENDING" if core_pending_singles > 0 else f"{core_hits_singles}/{len(core_singles)}"

    # ---- CORE double ----
    core_out_double, core_double_res, core_double_profit, core_score_double = eval_double(core_double, scores)
    core_double_stake = safe_float(core_double.get("stake"), 0.0) or 0.0

    # Totals
    core_stake = core_stake_singles + core_double_stake
    core_profit = core_profit_singles + core_double_profit
    core_pending = core_pending_singles + (1 if (core_double and core_double_res == "PENDING") else 0)

    # Bankroll outputs
    core_bankroll_start = safe_float(core.get("bankroll"), None)
    core_bankroll_end_settled = None
    core_bankroll_end_final = None
    if core_bankroll_start is not None:
        # settled = only if no pending
        if core_pending == 0:
            core_bankroll_end_final = round(core_bankroll_start + core_profit, 2)
        # settled-only end: subtract pending unknowns -> use current computed profit (which is 0 for pending)
        core_bankroll_end_settled = round(core_bankroll_start + core_profit, 2)

    core_roi_pct = None if core_pending > 0 else roi_pct(core_profit, core_stake)
    core_roi_pct_settled = roi_pct(core_profit, core_stake)  # profit includes 0 for pending; still OK as "current"

    # ---- FUN singles ----
    fun_out_singles = []
    fun_stake_singles = 0.0
    fun_profit_singles = 0.0
    fun_pending_singles = 0
    fun_hits_singles = 0

    for p in fun_singles:
        res, prof = eval_single(p, scores)
        stake = safe_float(p.get("stake"), 0.0) or 0.0
        fun_stake_singles += stake
        fun_profit_singles += prof
        if res == "PENDING":
            fun_pending_singles += 1
        if res == "WIN":
            fun_hits_singles += 1
        fun_out_singles.append({**p, "result": res, "profit": prof})

    fun_score_singles = "PENDING" if fun_pending_singles > 0 else f"{fun_hits_singles}/{len(fun_singles)}"

    # ---- FUN system ----
    fun_out_system, fun_sys_res, fun_sys_profit, fun_score_system_pool = eval_system(fun_system, fun_pool, scores)
    fun_sys_stake = safe_float(fun_system.get("stake"), 0.0) or 0.0

    fun_stake = fun_stake_singles + fun_sys_stake
    fun_profit = fun_profit_singles + fun_sys_profit
    fun_pending = fun_pending_singles + (1 if (fun_system and fun_sys_res == "PENDING") else 0)

    fun_bankroll_start = safe_float(fun.get("bankroll"), None)
    fun_bankroll_end_settled = None
    fun_bankroll_end_final = None
    if fun_bankroll_start is not None:
        if fun_pending == 0:
            fun_bankroll_end_final = round(fun_bankroll_start + fun_profit, 2)
        fun_bankroll_end_settled = round(fun_bankroll_start + fun_profit, 2)

    fun_roi_pct = None if fun_pending > 0 else roi_pct(fun_profit, fun_stake)
    fun_roi_pct_settled = roi_pct(fun_profit, fun_stake)

    # ---- LIFETIME ----
    lifetime = history.get("lifetime") or {"core": {"stake": 0.0, "profit": 0.0}, "funbet": {"stake": 0.0, "profit": 0.0}}

    if core_pending == 0:
        lifetime["core"]["stake"] = round(float(lifetime["core"].get("stake", 0.0)) + core_stake, 2)
        lifetime["core"]["profit"] = round(float(lifetime["core"].get("profit", 0.0)) + core_profit, 2)

    if fun_pending == 0:
        lifetime["funbet"]["stake"] = round(float(lifetime["funbet"].get("stake", 0.0)) + fun_stake, 2)
        lifetime["funbet"]["profit"] = round(float(lifetime["funbet"].get("profit", 0.0)) + fun_profit, 2)

    history["lifetime"] = lifetime
    history["updated_at"] = now_utc_iso()
    save_json(TUESDAY_HISTORY_PATH, history)

    lifetime_core_bankroll_start = safe_float(init_b.get("core"), None)
    lifetime_fun_bankroll_start = safe_float(init_b.get("funbet"), None)

    lifetime_core_bankroll_end = None
    lifetime_fun_bankroll_end = None

    if lifetime_core_bankroll_start is not None:
        lifetime_core_bankroll_end = round(lifetime_core_bankroll_start + float(lifetime["core"].get("profit", 0.0)), 2)
    if lifetime_fun_bankroll_start is not None:
        lifetime_fun_bankroll_end = round(lifetime_fun_bankroll_start + float(lifetime["funbet"].get("profit", 0.0)), 2)

    # ---- Final report ----
    report = {
        "timestamp": now_utc_iso(),
        "week_id": week_id,
        "week_label": week_label,
        "window": window,
        "source_friday_timestamp": friday.get("timestamp"),

        "core": {
            "bankroll_start": core_bankroll_start,
            "bankroll_end_settled": core_bankroll_end_settled,
            "bankroll_end_final": core_bankroll_end_final,
            "stake": round(core_stake, 2),
            "profit": round(core_profit, 2),
            "roi_pct": core_roi_pct,
            "roi_pct_settled": core_roi_pct_settled,
            "pending": int(core_pending),
            "score_singles": core_score_singles,
            "score_double": core_score_double,
            "singles": core_out_singles,
            "double": core_out_double,
        },

        "funbet": {
            "bankroll_start": fun_bankroll_start,
            "bankroll_end_settled": fun_bankroll_end_settled,
            "bankroll_end_final": fun_bankroll_end_final,
            "stake": round(fun_stake, 2),
            "profit": round(fun_profit, 2),
            "roi_pct": fun_roi_pct,
            "roi_pct_settled": fun_roi_pct_settled,
            "pending": int(fun_pending),
            "score_singles": fun_score_singles,
            "score_system_pool": fun_score_system_pool,
            "singles": fun_out_singles,
            "system": fun_out_system,
            "system_pool": fun_pool,
        },

        "weekly_summary": {
            "lifetime": {
                "core": {
                    "stake": round(float(lifetime["core"].get("stake", 0.0)), 2),
                    "profit": round(float(lifetime["core"].get("profit", 0.0)), 2),
                    "roi_pct": roi_pct(float(lifetime["core"].get("profit", 0.0)), float(lifetime["core"].get("stake", 0.0))),
                    "bankroll_start": lifetime_core_bankroll_start,
                    "bankroll_end": lifetime_core_bankroll_end,
                },
                "funbet": {
                    "stake": round(float(lifetime["funbet"].get("stake", 0.0)), 2),
                    "profit": round(float(lifetime["funbet"].get("profit", 0.0)), 2),
                    "roi_pct": roi_pct(float(lifetime["funbet"].get("profit", 0.0)), float(lifetime["funbet"].get("stake", 0.0))),
                    "bankroll_start": lifetime_fun_bankroll_start,
                    "bankroll_end": lifetime_fun_bankroll_end,
                },
            }
        },
    }

    save_json(TUESDAY_RECAP_PATH, report)
    log(f"✅ Tuesday Recap v3 written: {TUESDAY_RECAP_PATH}")


if __name__ == "__main__":
    main()
