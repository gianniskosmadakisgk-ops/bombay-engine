# ============================================================
#  BOMBAY TUESDAY RECAP v3 — RESULTS + ROI + BANKROLL HISTORY
#  Reads:  logs/friday_shortlist_v3.json
#  Writes: logs/tuesday_recap_v3.json  (+ logs/tuesday_history_v3.json)
#
#  Notes:
#   - Robust to missing odds/stakes/system fields.
#   - Voids: treated as push (profit 0 on singles; leg multiplier 1.0 in combos).
# ============================================================

import os
import json
import time
import requests
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

FRIDAY_REPORT_PATH = os.getenv("FRIDAY_REPORT_PATH", "logs/friday_shortlist_v3.json")
TUESDAY_RECAP_PATH = os.getenv("TUESDAY_RECAP_PATH", "logs/tuesday_recap_v3.json")
TUESDAY_HISTORY_PATH = os.getenv("TUESDAY_HISTORY_PATH", "logs/tuesday_history_v3.json")

REQUEST_TIMEOUT = int(os.getenv("API_TIMEOUT_SEC", "25"))
REQUEST_SLEEP_MS = int(os.getenv("API_SLEEP_MS", "140"))  # small throttle
MAX_FETCH_RETRIES = int(os.getenv("MAX_FETCH_RETRIES", "2"))

FINAL_STATUSES = {"FT", "AET", "PEN"}  # treat these as settled


def log(msg: str):
    print(msg, flush=True)


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


def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_history(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {
            "core": {"picks": 0, "hits": 0, "stake": 0.0, "profit": 0.0},
            "funbet": {"picks": 0, "hits": 0, "stake": 0.0, "profit": 0.0},
            "updated_at": now_utc_iso(),
        }
    try:
        return load_json(path)
    except Exception:
        return {
            "core": {"picks": 0, "hits": 0, "stake": 0.0, "profit": 0.0},
            "funbet": {"picks": 0, "hits": 0, "stake": 0.0, "profit": 0.0},
            "updated_at": now_utc_iso(),
        }


def _api_get_fixture(fixture_id: int) -> Optional[Dict[str, Any]]:
    if not API_FOOTBALL_KEY:
        return None

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"id": fixture_id}

    for attempt in range(MAX_FETCH_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                log(f"⚠️ API-Football status={r.status_code} for fixture_id={fixture_id}")
                time.sleep(0.25)
                continue
            js = r.json()
            resp = js.get("response") or []
            if not resp:
                return None
            return resp[0]
        except Exception as e:
            log(f"⚠️ API-Football error fixture_id={fixture_id}: {e}")
            time.sleep(0.25)
    return None


def fetch_scores(fixture_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Returns dict:
      fixture_id -> {
        status_short, home_goals, away_goals,
        home_name, away_name, date
      }
    """
    out: Dict[int, Dict[str, Any]] = {}
    uniq = sorted({int(x) for x in fixture_ids if x is not None})

    if not uniq:
        return out

    if not API_FOOTBALL_KEY:
        log("❌ Missing FOOTBALL_API_KEY – cannot fetch results.")
        return out

    log(f"Fetching results for {len(uniq)} fixtures...")

    for fid in uniq:
        fx = _api_get_fixture(fid)
        if fx is None:
            continue

        status_short = (((fx.get("fixture") or {}).get("status") or {}).get("short")) or ""
        goals = fx.get("goals") or {}
        hg = goals.get("home")
        ag = goals.get("away")

        teams = fx.get("teams") or {}
        home = (teams.get("home") or {}).get("name")
        away = (teams.get("away") or {}).get("name")
        date = (fx.get("fixture") or {}).get("date")

        out[fid] = {
            "status_short": status_short,
            "home_goals": safe_int(hg, None),
            "away_goals": safe_int(ag, None),
            "home_name": home,
            "away_name": away,
            "date": date,
        }

        time.sleep(max(0.0, REQUEST_SLEEP_MS / 1000.0))

    return out


def market_result(market_code: str, hg: int, ag: int) -> bool:
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


def evaluate_single_pick(pick: Dict[str, Any], score_lookup: Dict[int, Dict[str, Any]]) -> Tuple[str, float]:
    """
    Returns (status, profit)
      status: "win" | "lose" | "void" | "missing"
    """
    fid = safe_int(pick.get("fixture_id"), None)
    odds = safe_float(pick.get("odds"), None)
    stake = safe_float(pick.get("stake"), 0.0) or 0.0
    mcode = (pick.get("market_code") or "").upper().strip()

    if fid is None or not mcode:
        return "missing", 0.0

    sc = score_lookup.get(fid)
    if not sc:
        return "missing", 0.0

    st = (sc.get("status_short") or "").upper()
    hg = sc.get("home_goals")
    ag = sc.get("away_goals")

    if st not in FINAL_STATUSES or hg is None or ag is None:
        # Treat non-final / missing goals as void (push)
        return "void", 0.0

    # settle
    won = market_result(mcode, int(hg), int(ag))
    if odds is None or odds <= 1.0:
        # odds missing -> we can still mark win/lose, profit 0
        return ("win" if won else "lose"), 0.0

    if won:
        return "win", round(stake * (odds - 1.0), 2)
    return "lose", round(-stake, 2)


def evaluate_double(double_obj: Dict[str, Any], score_lookup: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], int, float, float]:
    """
    Returns:
      double_out, hits, stake, profit
    """
    if not double_obj:
        return {}, 0, 0.0, 0.0

    legs = double_obj.get("legs") or []
    stake = safe_float(double_obj.get("stake"), 0.0) or 0.0

    leg_out = []
    mult = 1.0
    any_lose = False
    all_void = True

    for leg in legs:
        pid = leg.get("pick_id") or ""
        # pick_id is like "fixture_id:CODE"
        fid = None
        code = None
        try:
            a, b = pid.split(":")
            fid = safe_int(a, None)
            code = (b or "").upper().strip()
        except Exception:
            fid = None
            code = None

        odds = safe_float(leg.get("odds"), None)

        if fid is None or not code:
            leg_out.append({**leg, "result": "missing"})
            continue

        sc = score_lookup.get(fid)
        if not sc:
            leg_out.append({**leg, "result": "missing"})
            continue

        st = (sc.get("status_short") or "").upper()
        hg = sc.get("home_goals")
        ag = sc.get("away_goals")

        if st not in FINAL_STATUSES or hg is None or ag is None:
            leg_out.append({**leg, "result": "void"})
            # void leg -> multiplier 1.0
            continue

        all_void = False
        won = market_result(code, int(hg), int(ag))
        if not won:
            any_lose = True
            leg_out.append({**leg, "result": "lose"})
        else:
            leg_out.append({**leg, "result": "win"})
            if odds is not None and odds > 1.0:
                mult *= odds

    if stake <= 0:
        # no stake -> can't compute money, but can still give logical result
        status = "missing_stake"
        profit = 0.0
        hits = 1 if (not any_lose and not all_void) else 0
        out = {**double_obj, "legs": leg_out, "status": status, "profit": profit}
        return out, hits, 0.0, 0.0

    if any_lose:
        profit = round(-stake, 2)
        hits = 0
        status = "lose"
    else:
        if all_void:
            profit = 0.0
            hits = 0
            status = "void"
        else:
            payout = stake * mult
            profit = round(payout - stake, 2)
            hits = 1 if profit > 0 else 0
            status = "win"

    out = {**double_obj, "legs": leg_out, "status": status, "profit": profit}
    return out, hits, stake, profit


def parse_system_sizes(label: str) -> List[int]:
    """
    Examples:
      "3-4-5/8" -> [3,4,5]
      "4/6"     -> [4]
      "2-3/5"   -> [2,3]
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


def evaluate_system(system: Dict[str, Any], pool: List[Dict[str, Any]], score_lookup: Dict[int, Dict[str, Any]]) -> Tuple[Dict[str, Any], int, float, float]:
    """
    Evaluates combo system:
      profit = Σ(payout_combo) - total_stake
      payout_combo = unit * Π(odds_leg) for all-win legs (void legs multiply by 1.0)
      losing leg -> payout 0 for that combo
    """
    if not system or not pool:
        return {"label": system.get("label") if system else None, "columns": 0, "unit": 0.0, "stake": 0.0, "profit": 0.0}, 0, 0.0, 0.0

    label = system.get("label")
    sizes = parse_system_sizes(label or "")
    n = len(pool)

    cols = safe_int(system.get("columns"), None)
    unit = safe_float(system.get("unit"), None)
    stake = safe_float(system.get("stake"), None)

    # derive columns/unit if missing
    if not sizes:
        return {**system, "profit": 0.0, "note": "no_sizes"}, 0, 0.0, 0.0

    if cols is None:
        cols = 0
        for r in sizes:
            if 2 <= r <= n:
                cols += comb(n, r)

    if unit is None:
        if stake is not None and cols and cols > 0:
            unit = stake / cols
        else:
            unit = 0.0

    if stake is None:
        stake = unit * cols

    if cols <= 0 or unit <= 0:
        return {**system, "profit": 0.0, "note": "no_unit_or_cols"}, 0, 0.0, 0.0

    # precompute leg outcomes for pool
    leg_eval = []
    for r in pool:
        fid = safe_int(r.get("fixture_id"), None)
        code = (r.get("market_code") or "").upper().strip()
        odds = safe_float(r.get("odds"), None)

        status = "missing"
        mult = 1.0
        if fid is not None and code:
            sc = score_lookup.get(fid)
            if sc:
                st = (sc.get("status_short") or "").upper()
                hg = sc.get("home_goals")
                ag = sc.get("away_goals")
                if st in FINAL_STATUSES and hg is not None and ag is not None:
                    won = market_result(code, int(hg), int(ag))
                    status = "win" if won else "lose"
                    if won and odds and odds > 1.0:
                        mult = float(odds)
                    elif won:
                        mult = 1.0
                else:
                    status = "void"
                    mult = 1.0

        leg_eval.append({"status": status, "mult": mult})

    total_payout = 0.0
    hit_combos = 0

    idxs = list(range(n))
    for r in sizes:
        if r < 2 or r > n:
            continue
        for combo in combinations(idxs, r):
            lose = False
            combo_mult = 1.0
            all_void = True
            for i in combo:
                st = leg_eval[i]["status"]
                if st == "lose":
                    lose = True
                    break
                if st == "win":
                    all_void = False
                    combo_mult *= leg_eval[i]["mult"]
                # void -> multiplier 1.0, doesn't change all_void unless all legs void
            if lose:
                continue
            if all_void:
                # full void combo -> push, payout = unit (profit 0 for that column)
                total_payout += unit
                continue
            payout = unit * combo_mult
            total_payout += payout
            if payout > unit + 1e-9:
                hit_combos += 1

    profit = round(total_payout - stake, 2)

    out = {
        **system,
        "label": label,
        "columns": int(cols),
        "unit": round(float(unit), 4),
        "stake": round(float(stake), 2),
        "profit": profit,
        "hit_combos": int(hit_combos),
    }
    return out, hit_combos, float(stake), float(profit)


def main():
    if not os.path.exists(FRIDAY_REPORT_PATH):
        raise FileNotFoundError(f"Friday report not found: {FRIDAY_REPORT_PATH}")

    shortlist = load_json(FRIDAY_REPORT_PATH)

    core = shortlist.get("core", {}) or {}
    fun = shortlist.get("funbet", {}) or {}

    core_singles = core.get("singles", []) or []
    core_double = core.get("double") or {}

    fun_singles = fun.get("singles", []) or []
    fun_system = fun.get("system") or {}
    fun_pool = fun.get("system_pool", []) or []

    # collect fixture ids
    fixture_ids: List[int] = []
    for p in core_singles:
        fid = safe_int(p.get("fixture_id"), None)
        if fid is not None:
            fixture_ids.append(fid)

    # double legs fixture ids are inside pick_id "fid:CODE"
    for leg in (core_double.get("legs") or []):
        pid = leg.get("pick_id") or ""
        try:
            a, _b = pid.split(":")
            fid = safe_int(a, None)
            if fid is not None:
                fixture_ids.append(fid)
        except Exception:
            pass

    for p in fun_singles:
        fid = safe_int(p.get("fixture_id"), None)
        if fid is not None:
            fixture_ids.append(fid)

    for p in fun_pool:
        fid = safe_int(p.get("fixture_id"), None)
        if fid is not None:
            fixture_ids.append(fid)

    scores = fetch_scores(fixture_ids)

    # ---- CORE singles ----
    enriched_core_singles: List[Dict[str, Any]] = []
    core_hits = 0
    core_picks = 0
    core_stake = 0.0
    core_profit = 0.0

    for pick in core_singles:
        status, prof = evaluate_single_pick(pick, scores)
        enriched = dict(pick)
        enriched["result"] = status
        enriched["profit"] = prof
        enriched_core_singles.append(enriched)

        stake = safe_float(pick.get("stake"), 0.0) or 0.0
        core_picks += 1
        core_stake += stake
        core_profit += prof
        if status == "win":
            core_hits += 1

    # ---- CORE double ----
    core_double_out, double_hits, double_stake, double_profit = evaluate_double(core_double, scores)
    # Only count into money totals if stake exists
    core_profit += double_profit
    core_stake += double_stake
    if double_hits > 0:
        core_hits += 1
    if core_double:
        core_picks += 1  # treat the double as a pick for reporting

    # ---- FUN singles ----
    enriched_fun_singles: List[Dict[str, Any]] = []
    fun_hits = 0
    fun_picks = 0
    fun_stake = 0.0
    fun_profit = 0.0

    for pick in fun_singles:
        status, prof = evaluate_single_pick(pick, scores)
        enriched = dict(pick)
        enriched["result"] = status
        enriched["profit"] = prof
        enriched_fun_singles.append(enriched)

        stake = safe_float(pick.get("stake"), 0.0) or 0.0
        fun_picks += 1
        fun_stake += stake
        fun_profit += prof
        if status == "win":
            fun_hits += 1

    # ---- FUN system ----
    fun_system_out, sys_hits, sys_stake, sys_profit = evaluate_system(fun_system, fun_pool, scores)
    fun_profit += sys_profit
    fun_stake += sys_stake
    if fun_system and fun_pool:
        fun_picks += 1  # treat system as a pick for reporting
        # hits = successful combos, but to keep same semantics as old recap, count combos as hits:
        fun_hits += sys_hits

    # ---- ROIs ----
    core_roi = (core_profit / core_stake) if core_stake > 0 else 0.0
    fun_roi = (fun_profit / fun_stake) if fun_stake > 0 else 0.0

    # ---- HISTORY ----
    history = load_history(TUESDAY_HISTORY_PATH)
    history["core"]["picks"] = int(history["core"].get("picks", 0)) + int(core_picks)
    history["core"]["hits"] = int(history["core"].get("hits", 0)) + int(core_hits)
    history["core"]["stake"] = float(history["core"].get("stake", 0.0)) + float(core_stake)
    history["core"]["profit"] = float(history["core"].get("profit", 0.0)) + float(core_profit)

    history["funbet"]["picks"] = int(history["funbet"].get("picks", 0)) + int(fun_picks)
    history["funbet"]["hits"] = int(history["funbet"].get("hits", 0)) + int(fun_hits)
    history["funbet"]["stake"] = float(history["funbet"].get("stake", 0.0)) + float(fun_stake)
    history["funbet"]["profit"] = float(history["funbet"].get("profit", 0.0)) + float(fun_profit)

    history["updated_at"] = now_utc_iso()

    total_core_roi = (history["core"]["profit"] / history["core"]["stake"]) if history["core"]["stake"] > 0 else 0.0
    total_fun_roi = (history["funbet"]["profit"] / history["funbet"]["stake"]) if history["funbet"]["stake"] > 0 else 0.0

    save_json(TUESDAY_HISTORY_PATH, history)

    # ---- REPORT (Tuesday Recap V3) ----
    report = {
        "timestamp": now_utc_iso(),
        "source_friday_timestamp": shortlist.get("timestamp"),
        "week": {
            "core": {
                "picks": int(core_picks),
                "hits": int(core_hits),
                "stake": round(core_stake, 2),
                "profit": round(core_profit, 2),
                "roi": round(core_roi * 100, 2),
            },
            "funbet": {
                "picks": int(fun_picks),
                "hits": int(fun_hits),
                "stake": round(fun_stake, 2),
                "profit": round(fun_profit, 2),
                "roi": round(fun_roi * 100, 2),
            },
        },
        "total": {
            "core": {
                "picks": int(history["core"]["picks"]),
                "hits": int(history["core"]["hits"]),
                "stake": round(float(history["core"]["stake"]), 2),
                "profit": round(float(history["core"]["profit"]), 2),
                "roi": round(total_core_roi * 100, 2),
            },
            "funbet": {
                "picks": int(history["funbet"]["picks"]),
                "hits": int(history["funbet"]["hits"]),
                "stake": round(float(history["funbet"]["stake"]), 2),
                "profit": round(float(history["funbet"]["profit"]), 2),
                "roi": round(total_fun_roi * 100, 2),
            },
        },
        "outcomes": {
            "core": {
                "singles": enriched_core_singles,
                "double": core_double_out,
            },
            "funbet": {
                "singles": enriched_fun_singles,
                "system": fun_system_out,
            },
        },
    }

    save_json(TUESDAY_RECAP_PATH, report)
    log(f"✅ Tuesday Recap v3 written: {TUESDAY_RECAP_PATH}")


if __name__ == "__main__":
    main()
