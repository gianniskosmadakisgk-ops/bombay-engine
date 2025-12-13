import os
import json
import math
from datetime import datetime

THURSDAY_REPORT_PATH = "logs/thursday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v3.json"

# ============================================================
#  FRIDAY SHORTLIST v4.0
#  - Keeps legacy sections: draw_singles, over_singles, funbet_draw, funbet_over, bankrolls
#  - NEW: "core" section:
#       * core_singles (odds 1.50-2.20, scaling by odds)
#       * core_doubles (2 strong favorites <1.50 combined)
#  - FunBet: dynamic r-of-N (3/4/5 of up to 7) based on confidence
#  - Kelly: OFF by default (kelly = [])
#  - Output: logs/friday_shortlist_v3.json
# ============================================================

# ------------------------- BANKROLLS (units = €) -------------------------
BANKROLL_CORE = float(os.getenv("BANKROLL_CORE", "1000"))
BANKROLL_FUN = float(os.getenv("BANKROLL_FUN", "500"))

# legacy bankrolls (kept for compatibility)
BANKROLL_DRAW = float(os.getenv("BANKROLL_DRAW", "1000"))
BANKROLL_OVER = float(os.getenv("BANKROLL_OVER", "1000"))
BANKROLL_FUN_DRAW = float(os.getenv("BANKROLL_FUN_DRAW", "300"))
BANKROLL_FUN_OVER = float(os.getenv("BANKROLL_FUN_OVER", "300"))
BANKROLL_KELLY = float(os.getenv("BANKROLL_KELLY", "600"))

MAX_FUN_EXPOSURE_PCT = float(os.getenv("MAX_FUN_EXPOSURE_PCT", "0.20"))

# Limits
MAX_CORE_SINGLES = int(os.getenv("MAX_CORE_SINGLES", "6"))
MAX_FUN_PICKS = int(os.getenv("MAX_FUN_PICKS", "7"))

# Common thresholds
MIN_EDGE = float(os.getenv("MIN_EDGE", "0.01"))
CORE_MIN_EDGE = float(os.getenv("CORE_MIN_EDGE", "0.02"))
CORE_MIN_PROB = float(os.getenv("CORE_MIN_PROB", "0.55"))  # minimum model confidence to consider core
CORE_ODDS_MIN = float(os.getenv("CORE_ODDS_MIN", "1.50"))
CORE_ODDS_MAX = float(os.getenv("CORE_ODDS_MAX", "2.20"))

# Double rules
DOUBLE_ODDS_MAX_SINGLE = float(os.getenv("DOUBLE_ODDS_MAX_SINGLE", "1.50"))
DOUBLE_MIN_PROB = float(os.getenv("DOUBLE_MIN_PROB", "0.70"))
DOUBLE_MIN_EDGE = float(os.getenv("DOUBLE_MIN_EDGE", "0.02"))
DOUBLE_STAKE = float(os.getenv("DOUBLE_STAKE", "25"))

# Draw/Over legacy filters (kept)
DRAW_MIN_PROB = float(os.getenv("DRAW_MIN_PROB", "0.22"))
DRAW_MIN_EDGE = float(os.getenv("DRAW_MIN_EDGE", "0.03"))
DRAW_LAMBDA_DIFF_MAX = float(os.getenv("DRAW_LAMBDA_DIFF_MAX", "0.35"))
DRAW_STAKE = float(os.getenv("DRAW_STAKE", "30"))

OVER_MIN_PROB = float(os.getenv("OVER_MIN_PROB", "0.55"))
OVER_MIN_EDGE = float(os.getenv("OVER_MIN_EDGE", "0.03"))
OVER_LAMBDA_TOTAL_MIN = float(os.getenv("OVER_LAMBDA_TOTAL_MIN", "2.75"))
MAX_FAIR_OVER = float(os.getenv("MAX_FAIR_OVER", "1.82"))

# Optional: disable Kelly entirely
ENABLE_KELLY = os.getenv("ENABLE_KELLY", "false").lower() == "true"

DRAW_PRIORITY_LEAGUES = {
    "Ligue 1", "Serie A", "La Liga", "Championship",
    "Serie B", "Ligue 2", "Liga Portugal 2", "Swiss Super League",
}
OVER_PRIORITY_LEAGUES = {
    "Bundesliga", "Eredivisie", "Jupiler Pro League", "Superliga",
    "Allsvenskan", "Eliteserien", "Swiss Super League", "Liga Portugal 1",
}

def log(msg: str):
    print(msg, flush=True)

def safe_float(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default

def edge_value(p: float, odds: float) -> float:
    if p is None or odds is None:
        return 0.0
    return (p * odds) - 1.0

def compute_draw_score(draw_prob, league):
    score = (draw_prob or 0.0) * 100.0
    if league in DRAW_PRIORITY_LEAGUES:
        score *= 1.05
    return score

def compute_over_score(over_prob, league):
    score = (over_prob or 0.0) * 100.0
    if league in OVER_PRIORITY_LEAGUES:
        score *= 1.05
    return score

def classify_over_stake(over_prob, fair_over, league):
    score = compute_over_score(over_prob, league)
    if over_prob >= 0.66 and fair_over is not None and fair_over <= 1.60 and score >= 66:
        return "premium", 16.0
    return "standard", 8.0

# ------------------------- CORE STAKE SCALING (your ranges) -------------------------
def core_stake_by_odds(odds: float) -> float:
    if odds is None:
        return 0.0
    # piecewise scaling (units)
    if odds < 1.50:
        return 0.0
    if odds <= 1.60:
        return 40.0
    if odds <= 1.70:
        return 34.0
    if odds <= 1.80:
        return 30.0
    if odds <= 1.90:
        return 26.0
    if odds <= 2.00:
        return 22.0
    if odds <= 2.10:
        return 18.0
    if odds <= 2.20:
        return 14.0
    if odds <= 2.30:
        return 10.0
    return 8.0

def load_thursday_fixtures():
    if not os.path.exists(THURSDAY_REPORT_PATH):
        raise FileNotFoundError(f"Thursday report not found: {THURSDAY_REPORT_PATH}")
    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    fixtures = data.get("fixtures", []) or []
    return fixtures, data

# ------------------------- FUNBET (dynamic r-of-N) -------------------------
def funbet_dynamic(picks, bankroll=500.0, max_exposure_pct=0.20):
    """
    Picks: list of dicts with at least score + odds
    Choose N up to 7. Choose r in {3,4,5} by avg score.
    Stake per column chosen so total <= exposure cap.
    """
    picks = sorted(picks, key=lambda x: x.get("score", 0.0), reverse=True)[:MAX_FUN_PICKS]
    n = len(picks)
    if n < 3:
        return {"system": None, "columns": 0, "unit": 0.0, "total_stake": 0.0, "picks": []}

    avg_score = sum(p.get("score", 0.0) for p in picks) / n

    if avg_score >= 75:
        r = min(5, n)
    elif avg_score >= 68:
        r = min(4, n)
    else:
        r = min(3, n)

    if r < 3:
        r = 3

    cols = math.comb(n, r)
    max_exposure = bankroll * max_exposure_pct

    unit = max_exposure / cols
    unit = int(unit)
    if unit < 1:
        unit = 1
    if unit > 5:
        unit = 5

    total = unit * cols
    if total > max_exposure:
        unit = max(1, int(max_exposure // cols))
        total = unit * cols

    return {"system": f"{r}/{n}", "columns": cols, "unit": float(unit), "total_stake": float(total), "picks": picks}

# ------------------------- PICK GENERATION -------------------------
def generate_picks(fixtures):
    draw_singles = []
    over_singles = []

    core_candidates = []   # 1X2 favorites (Home/Away) singles
    double_candidates = [] # favorites candidates for double

    for f in fixtures:
        home = f.get("home")
        away = f.get("away")
        league = f.get("league")
        match_label = f"{home} – {away}"

        # probs
        p_home = safe_float(f.get("home_prob"), None)
        p_draw = safe_float(f.get("draw_prob"), None)
        p_away = safe_float(f.get("away_prob"), None)
        p_over = safe_float(f.get("over_2_5_prob"), None)

        # lambdas
        lam_h = safe_float(f.get("lambda_home"), None)
        lam_a = safe_float(f.get("lambda_away"), None)
        lam_total = (lam_h + lam_a) if (lam_h is not None and lam_a is not None) else None

        # fair
        fair_x = safe_float(f.get("fair_x"), None)
        fair_over = safe_float(f.get("fair_over_2_5"), None)
        fair_1 = safe_float(f.get("fair_1"), None)
        fair_2 = safe_float(f.get("fair_2"), None)

        # offered
        offered_x = safe_float(f.get("offered_x"), None)
        offered_over = safe_float(f.get("offered_over_2_5"), None)
        offered_1 = safe_float(f.get("offered_1"), None)
        offered_2 = safe_float(f.get("offered_2"), None)

        # ---------------- DRAW (legacy) ----------------
        if p_draw is not None and offered_x is not None and offered_x > 1.0:
            e = edge_value(p_draw, offered_x)
            lam_diff = abs(lam_h - lam_a) if (lam_h is not None and lam_a is not None) else 999.0
            if p_draw >= DRAW_MIN_PROB and e >= DRAW_MIN_EDGE and lam_diff <= DRAW_LAMBDA_DIFF_MAX:
                draw_singles.append({
                    "match": match_label,
                    "league": league,
                    "prob": round(p_draw, 3),
                    "fair": fair_x,
                    "odds": offered_x,
                    "score": round(compute_draw_score(p_draw, league), 1),
                    "stake": float(DRAW_STAKE),
                })

        # ---------------- OVER (legacy) ----------------
        if p_over is not None and offered_over is not None and offered_over > 1.0:
            e = edge_value(p_over, offered_over)
            if (
                p_over >= OVER_MIN_PROB
                and e >= OVER_MIN_EDGE
                and (lam_total is None or lam_total >= OVER_LAMBDA_TOTAL_MIN)
                and (fair_over is None or fair_over <= MAX_FAIR_OVER)
            ):
                tier, stake = classify_over_stake(p_over, fair_over, league)
                over_singles.append({
                    "match": match_label,
                    "league": league,
                    "prob": round(p_over, 3),
                    "fair": fair_over,
                    "odds": offered_over,
                    "score": round(compute_over_score(p_over, league), 1),
                    "tier": tier,
                    "stake": float(stake),
                })

        # ---------------- CORE (1X2 singles) ----------------
        def consider_core(market, p, offered, fair):
            if p is None or offered is None or offered <= 1.0:
                return
            e = edge_value(p, offered)
            if e < CORE_MIN_EDGE or p < CORE_MIN_PROB:
                return
            if offered < CORE_ODDS_MIN or offered > CORE_ODDS_MAX:
                return

            stake = core_stake_by_odds(offered)
            if stake <= 0:
                return

            core_candidates.append({
                "match": match_label,
                "league": league,
                "market": market,
                "prob": round(p, 3),
                "fair": fair,
                "odds": offered,
                "edge": round(e * 100.0, 1),
                "score": round(p * 100.0, 1),
                "stake": float(stake),
            })

        consider_core("Home", p_home, offered_1, fair_1)
        consider_core("Away", p_away, offered_2, fair_2)

        # ---------------- DOUBLE CANDIDATES (<1.50) ----------------
        def consider_double(market, p, offered, fair):
            if p is None or offered is None or offered <= 1.0:
                return
            e = edge_value(p, offered)
            if offered <= DOUBLE_ODDS_MAX_SINGLE and p >= DOUBLE_MIN_PROB and e >= DOUBLE_MIN_EDGE:
                double_candidates.append({
                    "match": match_label,
                    "league": league,
                    "market": market,
                    "prob": round(p, 3),
                    "fair": fair,
                    "odds": offered,
                    "edge": round(e * 100.0, 1),
                    "score": round(p * 100.0, 1),
                })

        consider_double("Home", p_home, offered_1, fair_1)
        consider_double("Away", p_away, offered_2, fair_2)

    # sort + cap
    core_singles = sorted(core_candidates, key=lambda x: (x["edge"], x["score"]), reverse=True)[:MAX_CORE_SINGLES]
    draw_singles = sorted(draw_singles, key=lambda x: x["score"], reverse=True)[:7]
    over_singles = sorted(over_singles, key=lambda x: x["score"], reverse=True)[:7]

    # build one DOUBLE from top2 doubles (if available)
    core_doubles = []
    double_candidates = sorted(double_candidates, key=lambda x: (x["edge"], x["score"]), reverse=True)
    if len(double_candidates) >= 2:
        a = double_candidates[0]
        b = double_candidates[1]
        combined_odds = round((a["odds"] or 1.0) * (b["odds"] or 1.0), 2)
        core_doubles.append({
            "legs": [
                {"match": a["match"], "league": a["league"], "market": a["market"], "odds": a["odds"]},
                {"match": b["match"], "league": b["league"], "market": b["market"], "odds": b["odds"]},
            ],
            "combined_odds": combined_odds,
            "stake": float(DOUBLE_STAKE),
        })

    # FUNBET pools: use “core_singles + over_singles + draw_singles” as candidates, pick top scores
    fun_pool = []
    for x in core_singles:
        fun_pool.append({"match": x["match"], "league": x["league"], "odds": x["odds"], "score": x["score"], "market": x["market"]})
    for x in over_singles:
        fun_pool.append({"match": x["match"], "league": x["league"], "odds": x["odds"], "score": x["score"], "market": "Over 2.5"})
    for x in draw_singles:
        fun_pool.append({"match": x["match"], "league": x["league"], "odds": x["odds"], "score": x["score"], "market": "Draw"})

    fun_pool = sorted(fun_pool, key=lambda x: x["score"], reverse=True)
    funbet_core = funbet_dynamic(fun_pool, bankroll=BANKROLL_FUN, max_exposure_pct=MAX_FUN_EXPOSURE_PCT)

    return draw_singles, over_singles, core_singles, core_doubles, funbet_core

def main():
    log("🚀 Running Friday Shortlist v4.0 (Core + Dynamic FunBet, Kelly OFF)")

    fixtures, th_report = load_thursday_fixtures()
    log(f"Loaded {len(fixtures)} fixtures from {THURSDAY_REPORT_PATH}")

    draw_singles, over_singles, core_singles, core_doubles, funbet_core = generate_picks(fixtures)

    # legacy funbets (kept, simple mirror)
    funbet_draw = funbet_core
    funbet_over = funbet_core

    # bankroll summary (kept keys)
    draw_open = sum(d.get("stake", 0.0) for d in draw_singles)
    over_open = sum(o.get("stake", 0.0) for o in over_singles)
    core_open = sum(c.get("stake", 0.0) for c in core_singles) + sum(d.get("stake", 0.0) for d in core_doubles)
    fun_open = funbet_core.get("total_stake", 0.0)

    bankrolls = {
        "core": {"bank_start": BANKROLL_CORE, "week_start": BANKROLL_CORE, "open": round(core_open, 1), "after_open": round(BANKROLL_CORE - core_open, 1), "picks": len(core_singles) + len(core_doubles)},
        "fun":  {"bank_start": BANKROLL_FUN, "week_start": BANKROLL_FUN, "open": round(fun_open, 1),  "after_open": round(BANKROLL_FUN - fun_open, 1),  "picks": len(funbet_core.get("picks", []))},

        # legacy categories (still filled)
        "draw": {"bank_start": BANKROLL_DRAW, "week_start": BANKROLL_DRAW, "open": round(draw_open, 1), "after_open": round(BANKROLL_DRAW - draw_open, 1), "picks": len(draw_singles)},
        "over": {"bank_start": BANKROLL_OVER, "week_start": BANKROLL_OVER, "open": round(over_open, 1), "after_open": round(BANKROLL_OVER - over_open, 1), "picks": len(over_singles)},
        "fun_draw": {"bank_start": BANKROLL_FUN_DRAW, "week_start": BANKROLL_FUN_DRAW, "open": round(fun_open, 1), "after_open": round(BANKROLL_FUN_DRAW - fun_open, 1), "picks": len(funbet_core.get("picks", []))},
        "fun_over": {"bank_start": BANKROLL_FUN_OVER, "week_start": BANKROLL_FUN_OVER, "open": round(fun_open, 1), "after_open": round(BANKROLL_FUN_OVER - fun_open, 1), "picks": len(funbet_core.get("picks", []))},
        "kelly": {"bank_start": BANKROLL_KELLY, "week_start": BANKROLL_KELLY, "open": 0.0, "after_open": BANKROLL_KELLY, "picks": 0},
    }

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "fixtures_total": len(fixtures),
        "window": th_report.get("window", {}),

        # legacy sections (keep)
        "draw_singles": draw_singles,
        "over_singles": over_singles,
        "funbet_draw": funbet_draw,
        "funbet_over": funbet_over,
        "kelly": [],

        # NEW core section
        "core": {
            "core_singles": core_singles,
            "core_doubles": core_doubles,
            "funbet_core": funbet_core,
        },

        "bankrolls": bankrolls,
    }

    os.makedirs(os.path.dirname(FRIDAY_REPORT_PATH), exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log(f"✅ Friday Shortlist v4.0 saved → {FRIDAY_REPORT_PATH}")

if __name__ == "__main__":
    main()
