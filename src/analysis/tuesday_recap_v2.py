import os
import json
from datetime import datetime

FRIDAY_REPORT_PATH = "logs/friday_shortlist_v3.json"
TUESDAY_RESULTS_PATH = "logs/tuesday_results.json"   # optional input
TUESDAY_RECAP_PATH = "logs/tuesday_recap_v3.json"

def log(msg: str):
    print(msg, flush=True)

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def key_of(match, market):
    return f"{(match or '').strip()}||{(market or '').strip()}"

def result_map(results_json):
    """
    Expected schema (simple):
    {
      "matches": [
        {"match": "...", "market": "...", "result": "WIN|LOSS|VOID"}
      ]
    }
    """
    m = {}
    if not results_json:
        return m
    for r in results_json.get("matches", []) or []:
        k = key_of(r.get("match"), r.get("market"))
        res = (r.get("result") or "").upper().strip()
        if res in ("WIN", "LOSS", "VOID"):
            m[k] = res
    return m

def settle_single(stake, odds, res):
    """
    Settlement:
      WIN: + stake*(odds-1)
      LOSS: - stake
      VOID: 0
    """
    stake = float(stake or 0.0)
    odds = float(odds or 0.0)
    if res == "WIN":
        return round(stake * (odds - 1.0), 2)
    if res == "LOSS":
        return round(-stake, 2)
    return 0.0

def settle_double(double_obj, rmap):
    """
    Double settles ONLY if both legs have explicit results in tuesday_results.json.
    - If any leg is LOSS => LOSS
    - If any leg is VOID and no LOSS => treat as NOT_SETTLED (you can later define rules)
    - If both WIN => WIN (profit by combo_odds)
    Returns (status, p_l)
    """
    if not double_obj:
        return None, None

    legs = double_obj.get("legs", []) or []
    if len(legs) != 2:
        return "NOT_SETTLED", None

    leg_results = []
    for leg in legs:
        k = key_of(leg.get("match"), leg.get("market"))
        leg_results.append(rmap.get(k, "PENDING"))

    if "PENDING" in leg_results:
        return "PENDING", None

    if "LOSS" in leg_results:
        return "LOSS", settle_single(double_obj.get("stake"), 0.0, "LOSS")

    if "VOID" in leg_results:
        return "NOT_SETTLED_VOID_RULE", None

    # both WIN
    combo_odds = float(double_obj.get("combo_odds") or 0.0)
    stake = float(double_obj.get("stake") or 0.0)
    if combo_odds <= 1.0:
        return "NOT_SETTLED_BAD_ODDS", None
    pl = round(stake * (combo_odds - 1.0), 2)
    return "WIN", pl

def main():
    friday = load_json(FRIDAY_REPORT_PATH)
    if not friday:
        raise FileNotFoundError(f"Missing Friday report: {FRIDAY_REPORT_PATH}")

    results = load_json(TUESDAY_RESULTS_PATH)  # optional
    rmap = result_map(results)

    core = friday.get("core", {}) or {}
    fun = friday.get("funbet", {}) or {}

    # ---------------- CORE recap ----------------
    core_singles = core.get("singles", []) or []
    core_double = core.get("double", None)

    core_rows = []
    core_pl = 0.0
    core_w = core_l = core_v = core_pending = 0

    for p in core_singles:
        match = p.get("match")
        market = p.get("market")
        stake = float(p.get("stake") or 0.0)
        odds = float(p.get("odds") or 0.0)

        res = rmap.get(key_of(match, market), "PENDING")
        pl = None
        if res == "PENDING":
            core_pending += 1
        else:
            pl = settle_single(stake, odds, res)
            core_pl += pl
            if res == "WIN": core_w += 1
            elif res == "LOSS": core_l += 1
            elif res == "VOID": core_v += 1

        core_rows.append({
            "match": match,
            "league": p.get("league"),
            "market": market,
            "odds": odds,
            "stake": stake,
            "result": res,
            "p_l": pl,
        })

    # Double
    double_row = None
    double_pl = None
    if core_double:
        status, pl = settle_double(core_double, rmap)
        double_pl = pl
        double_row = {
            "type": "Double",
            "combo_odds": core_double.get("combo_odds"),
            "stake": core_double.get("stake"),
            "legs": core_double.get("legs", []),
            "result": status,
            "p_l": pl,
        }
        if pl is not None:
            core_pl += pl

    core_settled = bool(results) and (core_pending == 0) and (double_row is None or double_row["result"] != "PENDING")

    # ---------------- FUN recap ----------------
    fun_picks = fun.get("picks", []) or []
    fun_rows = []
    fun_w = fun_l = fun_v = fun_pending = 0

    for p in fun_picks:
        match = p.get("match")
        market = p.get("market")
        odds = float(p.get("odds") or 0.0)
        res = rmap.get(key_of(match, market), "PENDING")
        if res == "PENDING": fun_pending += 1
        elif res == "WIN": fun_w += 1
        elif res == "LOSS": fun_l += 1
        elif res == "VOID": fun_v += 1

        fun_rows.append({
            "match": match,
            "league": p.get("league"),
            "market": market,
            "odds": odds,
            "result": res,
        })

    # NOTE: Δεν κάνουμε settlement συστήματος χωρίς line-by-line κουπόνια.
    # (Το “picks hit-rate” είναι αρκετό για πρώτο κύκλο.)
    fun_settled = bool(results) and (fun_pending == 0)

    recap = {
        "timestamp": datetime.utcnow().isoformat(),
        "window": friday.get("window", {}),
        "fixtures_total": friday.get("fixtures_total"),

        "core_recap": {
            "bankroll_start": core.get("bankroll"),
            "open": core.get("open"),
            "after_open": core.get("after_open"),
            "picks": core_rows,
            "double": double_row,
            "settled": core_settled,
            "wins": core_w,
            "losses": core_l,
            "voids": core_v,
            "pending": core_pending,
            "p_l_total": round(core_pl, 2) if bool(results) else None,
        },

        "fun_recap": {
            "bankroll_start": fun.get("bankroll"),
            "open": fun.get("open"),
            "after_open": fun.get("after_open"),
            "system": fun.get("system"),
            "columns": fun.get("columns"),
            "unit": fun.get("unit"),
            "total_stake": fun.get("total_stake"),
            "picks": fun_rows,
            "settled": fun_settled,
            "wins": fun_w,
            "losses": fun_l,
            "voids": fun_v,
            "pending": fun_pending,
            "note": "Δεν γίνεται payout settlement συστήματος χωρίς explicit lines/results schema. Προς το παρόν κρατάμε hit-rate ανά pick.",
        },

        "weekly_summary": {
            "core_open": core.get("open"),
            "fun_open": fun.get("open"),
            "core_p_l": round(core_pl, 2) if bool(results) else None,
            "fun_p_l": None,
        },
    }

    os.makedirs("logs", exist_ok=True)
    with open(TUESDAY_RECAP_PATH, "w", encoding="utf-8") as f:
        json.dump(recap, f, ensure_ascii=False, indent=2)

    log(f"✅ Tuesday Recap saved → {TUESDAY_RECAP_PATH}")

if __name__ == "__main__":
    main()
