import os
import json
import re
import unicodedata
from datetime import datetime

FRIDAY_REPORT_PATH = "logs/friday_shortlist_v3.json"
TUESDAY_RESULTS_PATH = "logs/tuesday_results.json"     # optional
TUESDAY_RECAP_PATH = "logs/tuesday_recap_v3.json"

# ---------- Normalization helpers ----------
def strip_accents(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def norm_spaces(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s

def normalize_match(match: str) -> str:
    """
    Canonicalize match string:
    - lowercase
    - remove accents
    - normalize dashes (–, —, -) to " - "
    - collapse spaces
    """
    s = strip_accents(match or "").lower()
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s*-\s*", " - ", s)
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)   # kill punctuation
    return norm_spaces(s)

def normalize_market(market: str) -> str:
    """
    Map different labels to same canonical market.
    """
    s = strip_accents(market or "").lower().strip()

    # unify common variants
    s = s.replace("o2.5", "over 2.5").replace("u2.5", "under 2.5")
    s = s.replace("over2.5", "over 2.5").replace("under2.5", "under 2.5")
    s = s.replace("over 2,5", "over 2.5").replace("under 2,5", "under 2.5")

    aliases = {
        "1": "home",
        "home": "home",
        "h": "home",
        "2": "away",
        "away": "away",
        "a": "away",
        "x": "draw",
        "draw": "draw",
        "d": "draw",
        "over 2.5": "over 2.5",
        "under 2.5": "under 2.5",
    }
    s = norm_spaces(s)
    return aliases.get(s, s)

def key_of(match, market):
    return f"{normalize_match(match)}||{normalize_market(market)}"

# ---------- IO ----------
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def result_map(results_json):
    """
    results_json schema expected:
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
        match = p.get("match")
        market = p.get("market")
        stake = float(p.get("stake") or 0.0)
        odds = float(p.get("odds") or 0.0)

        res = rmap.get(key_of(match, market), "PENDING")
        pl = None
        if res != "PENDING":
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
        match = p.get("match")
        market = p.get("market")
        odds = float(p.get("odds") or 0.0)

        res = rmap.get(key_of(match, market), "PENDING")
        if res == "WIN": fun_w += 1
        elif res == "LOSS": fun_l += 1
        elif res == "VOID": fun_v += 1

        fun_rows.append({
            "match": match,
            "league": p.get("league"),
            "market": market,
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
            "note": "Fun payout δεν υπολογίζεται χωρίς settlement ανά στήλη. Εδώ κρατάμε outcomes ανά pick.",
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

    print(f"✅ Tuesday Recap saved → {TUESDAY_RECAP_PATH}", flush=True)

if __name__ == "__main__":
    main()
