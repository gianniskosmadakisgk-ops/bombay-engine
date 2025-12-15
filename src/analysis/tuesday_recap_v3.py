# =========================
# FILE: src/analysis/tuesday_recap_v3.py
# =========================
import os
import json
from datetime import datetime

FRIDAY_REPORT_PATH = "logs/friday_shortlist_v3.json"
TUESDAY_RESULTS_PATH = "logs/tuesday_results.json"   # optional input from you
TUESDAY_RECAP_PATH = "logs/tuesday_recap_v3.json"

def log(msg: str):
    print(msg, flush=True)

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def result_map(results_json):
    """
    Primary key: pick_id (fixture_id:market_code)
    Example pick_id: "123456:O25"
    """
    m = {}
    if not results_json:
        return m

    for r in results_json.get("matches", []) or []:
        pid = (r.get("pick_id") or "").strip()
        res = (r.get("result") or "").upper().strip()
        if pid and res in ("WIN", "LOSS", "VOID"):
            m[pid] = res

    return m

def settle_single(stake, odds, res):
    if res == "WIN":
        return round(stake * (odds - 1.0), 2)
    if res == "LOSS":
        return round(-stake, 2)
    return 0.0

def main():
    friday = load_json(FRIDAY_REPORT_PATH)
    if not friday:
        raise FileNotFoundError(f"Missing Friday report: {FRIDAY_REPORT_PATH}")

    results = load_json(TUESDAY_RESULTS_PATH)  # may be None
    rmap = result_map(results)

    core = friday.get("core", {}) or {}
    fun = friday.get("funbet", {}) or {}

    # -------- CORE recap --------
    core_singles = core.get("singles", []) or []
    core_double = core.get("double", None)

    core_rows = []
    core_pl = 0.0
    core_w = core_l = core_v = 0

    for p in core_singles:
        pid = (p.get("pick_id") or "").strip()
        stake = float(p.get("stake") or 0.0)
        odds = float(p.get("odds") or 0.0)

        res = rmap.get(pid, "PENDING") if pid else "PENDING"
        pl = None

        if res != "PENDING":
            pl = settle_single(stake, odds, res)
            core_pl += pl
            if res == "WIN":
                core_w += 1
            elif res == "LOSS":
                core_l += 1
            elif res == "VOID":
                core_v += 1

        core_rows.append({
            "pick_id": pid or None,
            "fixture_id": p.get("fixture_id"),
            "market_code": p.get("market_code"),

            "match": p.get("match"),
            "league": p.get("league"),
            "market": p.get("market"),
            "odds": odds,
            "stake": stake,
            "result": res,
            "p_l": pl,
        })

    # Double stays informational unless you extend results schema to settle combos
    double_row = None
    if core_double:
        double_row = {
            "type": "Double",
            "combo_odds": core_double.get("combo_odds"),
            "stake": core_double.get("stake"),
            "legs": core_double.get("legs", []),
            "result": "PENDING" if not results else "NOT_SETTLED_BY_DEFAULT",
        }

    # -------- FUN recap --------
    fun_picks = fun.get("picks", []) or []
    fun_rows = []
    fun_w = fun_l = fun_v = 0

    for p in fun_picks:
        pid = (p.get("pick_id") or "").strip()
        odds = float(p.get("odds") or 0.0)

        res = rmap.get(pid, "PENDING") if pid else "PENDING"
        if res == "WIN":
            fun_w += 1
        elif res == "LOSS":
            fun_l += 1
        elif res == "VOID":
            fun_v += 1

        fun_rows.append({
            "pick_id": pid or None,
            "fixture_id": p.get("fixture_id"),
            "market_code": p.get("market_code"),

            "match": p.get("match"),
            "league": p.get("league"),
            "market": p.get("market"),
            "odds": odds,
            "result": res,
        })

    recap = {
        "timestamp": datetime.utcnow().isoformat(),
        "window": friday.get("window", {}),
        "fixtures_total": friday.get("fixtures_total"),

        "core_recap": {
            "bankroll_start": core.get("bankroll"),
            "open": core.get("open"),
            "after_open": core.get("after_open"),
            "picks_count": core.get("picks_count"),
            "picks": core_rows,
            "double": double_row,
            "settled": bool(results),
            "wins": core_w,
            "losses": core_l,
            "voids": core_v,
            "p_l_total": round(core_pl, 2) if results else None,
        },

        "fun_recap": {
            "bankroll_start": fun.get("bankroll"),
            "open": fun.get("open"),
            "after_open": fun.get("after_open"),
            "system": fun.get("system"),
            "columns": fun.get("columns"),
            "unit": fun.get("unit"),
            "total_stake": fun.get("total_stake"),
            "picks_count": fun.get("picks_count"),
            "picks": fun_rows,
            "settled": bool(results),
            "wins": fun_w,
            "losses": fun_l,
            "voids": fun_v,
            "note": "Fun payout δεν υπολογίζεται χωρίς line-by-line settlement του συστήματος. Εδώ κρατάμε outcomes ανά pick.",
        },

        "weekly_summary": {
            "core_open": core.get("open"),
            "fun_open": fun.get("open"),
            "core_p_l": round(core_pl, 2) if results else None,
            "fun_p_l": None,
        },
    }

    os.makedirs("logs", exist_ok=True)
    with open(TUESDAY_RECAP_PATH, "w", encoding="utf-8") as f:
        json.dump(recap, f, ensure_ascii=False, indent=2)

    log(f"✅ Tuesday Recap saved → {TUESDAY_RECAP_PATH}")

if __name__ == "__main__":
    main()
