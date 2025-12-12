import os
import json
from datetime import datetime

THURSDAY_REPORT_PATH = "logs/thursday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v3.json"

# ------------------------- BANKROLLS (units = €) -------------------------
BANKROLL_DRAW = float(os.getenv("BANKROLL_DRAW", "1000"))
BANKROLL_OVER = float(os.getenv("BANKROLL_OVER", "1000"))
BANKROLL_FUN_DRAW = float(os.getenv("BANKROLL_FUN_DRAW", "300"))
BANKROLL_FUN_OVER = float(os.getenv("BANKROLL_FUN_OVER", "300"))
BANKROLL_KELLY = float(os.getenv("BANKROLL_KELLY", "600"))

MAX_FUN_EXPOSURE_PCT = float(os.getenv("MAX_FUN_EXPOSURE_PCT", "0.20"))

# "Not forced" caps
MAX_DRAWS = int(os.getenv("MAX_DRAWS", "7"))
MAX_OVERS = int(os.getenv("MAX_OVERS", "7"))
MAX_KELLY = int(os.getenv("MAX_KELLY", "6"))

# ------------------------- SELECTION RULES (production) -------------------------
# Minimum edges (value)
MIN_EDGE_DRAWS = float(os.getenv("MIN_EDGE_DRAWS", "0.03"))   # 3%
MIN_EDGE_OVERS = float(os.getenv("MIN_EDGE_OVERS", "0.03"))   # 3%
MIN_EDGE_KELLY = float(os.getenv("MIN_EDGE_KELLY", "0.015"))  # 1.5%

# Draw quality gates
MIN_DRAW_PROB = float(os.getenv("MIN_DRAW_PROB", "0.22"))
MIN_DRAW_ODDS = float(os.getenv("MIN_DRAW_ODDS", "2.60"))
MAX_LAM_DIFF_DRAW = float(os.getenv("MAX_LAM_DIFF_DRAW", "0.35"))  # closeness gate

# Over quality gates
MIN_OVER_PROB = float(os.getenv("MIN_OVER_PROB", "0.55"))
MIN_Z_TOTAL_OVER = float(os.getenv("MIN_Z_TOTAL_OVER", "0.65"))
MIN_LAM_TOTAL_OVER = float(os.getenv("MIN_LAM_TOTAL_OVER", "2.75"))
MAX_FAIR_OVER = float(os.getenv("MAX_FAIR_OVER", "1.90"))

# Stakes (flat for singles)
DRAW_STAKE = float(os.getenv("DRAW_STAKE", "30.0"))

# Kelly caps
KELLY_MAX_ODDS = float(os.getenv("KELLY_MAX_ODDS", "4.0"))
KELLY_MIN_PROB = float(os.getenv("KELLY_MIN_PROB", "0.18"))

# ------------------------- LEAGUE PRIORITIES (light bump) -------------------------
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

def compute_draw_score(draw_prob, league):
    s = (draw_prob or 0.0) * 100.0
    if league in DRAW_PRIORITY_LEAGUES:
        s *= 1.05
    return s

def compute_over_score(over_prob, league):
    s = (over_prob or 0.0) * 100.0
    if league in OVER_PRIORITY_LEAGUES:
        s *= 1.05
    return s

def classify_over_stake(over_prob, fair_over, z_total, league):
    # Tiering: probability + price quality + confidence
    score = compute_over_score(over_prob, league)
    if over_prob >= 0.62 and fair_over <= 1.65 and z_total >= 0.90 and score >= 62:
        return "premium", 16.0
    return "standard", 8.0

def compute_system_stake(bankroll, columns, max_exposure_pct=MAX_FUN_EXPOSURE_PCT,
                         min_unit=1.0, max_unit=5.0):
    if columns <= 0:
        return 0.0, 0.0
    max_exposure = bankroll * max_exposure_pct
    base_unit = max_exposure / columns
    unit = int(base_unit)
    if unit < min_unit: unit = min_unit
    if unit > max_unit: unit = max_unit
    total = unit * columns
    if total > max_exposure:
        unit = max(min_unit, int(max_exposure // columns))
        total = unit * columns
    return float(unit), float(total)

def load_thursday_fixtures():
    if not os.path.exists(THURSDAY_REPORT_PATH):
        raise FileNotFoundError(f"Thursday report not found: {THURSDAY_REPORT_PATH}")
    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    fixtures = data.get("fixtures", []) or []
    return fixtures, data

# ------------------------- FUNBET SYSTEMS -------------------------
def funbet_draw(draw_singles):
    picks = sorted(draw_singles, key=lambda x: x["score"], reverse=True)[:7]
    n = len(picks)
    if n < 3:
        return {"system": None, "columns": 0, "unit": 0.0, "total_stake": 0.0, "picks": []}
    if n == 3: sys, cols = "3/3", 1
    elif n == 4: sys, cols = "3/4", 4
    elif n == 5: sys, cols = "3/5", 10
    elif n == 6: sys, cols = "4/6", 15
    else: sys, cols = "4/7", 35
    unit, total = compute_system_stake(BANKROLL_FUN_DRAW, cols)
    return {"system": sys, "columns": cols, "unit": unit, "total_stake": total, "picks": picks}

def funbet_over(over_singles):
    picks = sorted(over_singles, key=lambda x: x["score"], reverse=True)[:7]
    n = len(picks)
    if n < 3:
        return {"system": None, "columns": 0, "unit": 0.0, "total_stake": 0.0, "picks": []}
    if n == 3: sys, cols = "3/3", 1
    elif n == 4: sys, cols = "2/4", 6
    elif n == 5: sys, cols = "2/5", 10
    elif n == 6: sys, cols = "3/6", 20
    else: sys, cols = "3/7", 35
    unit, total = compute_system_stake(BANKROLL_FUN_OVER, cols)
    return {"system": sys, "columns": cols, "unit": unit, "total_stake": total, "picks": picks}

# ------------------------- EDGE HELPERS -------------------------
def edge_from_prob_and_odds(p: float, offered: float):
    if p is None or offered is None or offered <= 1.0:
        return None
    return (offered * p) - 1.0  # e.g. 0.03 = +3%

# ------------------------- KELLY (scaled by edge buckets) -------------------------
def scaled_kelly_fraction(p: float, offered: float, edge: float):
    if p is None or offered is None:
        return 0.0
    if offered <= 1.0:
        return 0.0

    b = offered - 1.0
    q = 1.0 - p
    f_star = (b * p - q) / b
    if f_star <= 0:
        return 0.0

    # Scaling by edge bucket (from your Gemini spec)
    if edge < 0.015:
        k = 0.0
    elif edge < 0.03:
        k = 0.25
    elif edge < 0.06:
        k = 0.50
    else:
        k = 0.75

    f = f_star * k

    # hard caps (tight, production-safe)
    if offered <= 2.5:
        cap = 0.05
    else:
        cap = 0.03
    f = min(f, cap)

    return max(0.0, f)

# ------------------------- MAIN PICK GENERATION -------------------------
def generate_picks(fixtures):
    draw_singles = []
    over_singles = []
    kelly_candidates = []

    for f in fixtures:
        home = f.get("home")
        away = f.get("away")
        league = f.get("league")
        match_label = f"{home} – {away}"

        # probs
        p_draw = safe_float(f.get("draw_prob"))
        p_over = safe_float(f.get("over_2_5_prob"))
        p_home = safe_float(f.get("home_prob"))
        p_away = safe_float(f.get("away_prob"))

        # fair
        fair_x = safe_float(f.get("fair_x"))
        fair_over = safe_float(f.get("fair_over_2_5"))
        fair_1 = safe_float(f.get("fair_1"))
        fair_2 = safe_float(f.get("fair_2"))

        # offered
        offered_x = safe_float(f.get("offered_x"))
        offered_over = safe_float(f.get("offered_over_2_5"))
        offered_1 = safe_float(f.get("offered_1"))
        offered_2 = safe_float(f.get("offered_2"))

        # lambda diagnostics
        lam_h = safe_float(f.get("lambda_home"))
        lam_a = safe_float(f.get("lambda_away"))
        lam_total = safe_float(f.get("lambda_total"))
        z_total = safe_float(f.get("z_total"), 0.0) or 0.0

        # ---------------- DRAW SINGLES ----------------
        if p_draw is not None and offered_x is not None and fair_x is not None:
            edge = edge_from_prob_and_odds(p_draw, offered_x)
            lam_diff = abs((lam_h or 0.0) - (lam_a or 0.0))
            if (
                edge is not None and edge >= MIN_EDGE_DRAWS
                and p_draw >= MIN_DRAW_PROB
                and offered_x >= MIN_DRAW_ODDS
                and lam_diff <= MAX_LAM_DIFF_DRAW
            ):
                draw_singles.append({
                    "match": match_label,
                    "league": league,
                    "prob": round(p_draw, 3),
                    "fair": fair_x,
                    "odds": offered_x,
                    "edge": round(edge * 100.0, 1),
                    "score": round(compute_draw_score(p_draw, league), 1),
                    "stake": DRAW_STAKE,
                })

        # ---------------- OVER SINGLES ----------------
        if p_over is not None and offered_over is not None and fair_over is not None:
            edge = edge_from_prob_and_odds(p_over, offered_over)
            if (
                edge is not None and edge >= MIN_EDGE_OVERS
                and p_over >= MIN_OVER_PROB
                and z_total >= MIN_Z_TOTAL_OVER
                and (lam_total is None or lam_total >= MIN_LAM_TOTAL_OVER)
                and fair_over <= MAX_FAIR_OVER
                and offered_over > 1.01
            ):
                tier, stake = classify_over_stake(p_over, fair_over, z_total, league)
                over_singles.append({
                    "match": match_label,
                    "league": league,
                    "prob": round(p_over, 3),
                    "fair": fair_over,
                    "odds": offered_over,
                    "edge": round(edge * 100.0, 1),
                    "z_total": round(z_total, 3),
                    "score": round(compute_over_score(p_over, league), 1),
                    "tier": tier,
                    "stake": float(stake),
                })

        # ---------------- KELLY (ONLY 1 & 2, scaled, not flat) ----------------
        def add_kelly(market_label, p, fair, offered):
            if p is None or fair is None or offered is None:
                return
            if offered > KELLY_MAX_ODDS:
                return
            if p < KELLY_MIN_PROB:
                return
            edge = edge_from_prob_and_odds(p, offered)
            if edge is None or edge < MIN_EDGE_KELLY:
                return

            f = scaled_kelly_fraction(p, offered, edge)
            if f <= 0:
                return

            stake = max(3.0, round(BANKROLL_KELLY * f, 1))
            kelly_candidates.append({
                "match": match_label,
                "league": league,
                "market": market_label,
                "prob": round(p, 3),
                "fair": fair,
                "odds": offered,
                "edge": round(edge * 100.0, 1),
                "stake": stake,
                "f_fraction": round(f, 4),
            })

        add_kelly("Home", p_home, fair_1, offered_1)
        add_kelly("Away", p_away, fair_2, offered_2)

    # "not forced" caps — παίρνεις μόνο τα καλύτερα
    draw_singles = sorted(draw_singles, key=lambda d: (d["edge"], d["score"]), reverse=True)[:MAX_DRAWS]
    over_singles = sorted(over_singles, key=lambda o: (o["edge"], o["z_total"], o["score"]), reverse=True)[:MAX_OVERS]
    kelly_candidates = sorted(kelly_candidates, key=lambda k: (k["edge"], k["f_fraction"]), reverse=True)[:MAX_KELLY]

    return draw_singles, over_singles, kelly_candidates

def main():
    log("🚀 Running Friday Shortlist v3.3 (edge+confidence, scaled Kelly, not forced caps)")

    fixtures, th_report = load_thursday_fixtures()
    log(f"Loaded {len(fixtures)} fixtures from {THURSDAY_REPORT_PATH}")

    draw_singles, over_singles, kelly_picks = generate_picks(fixtures)

    fb_draw = funbet_draw(draw_singles)
    fb_over = funbet_over(over_singles)

    draw_open = sum(d["stake"] for d in draw_singles)
    over_open = sum(o["stake"] for o in over_singles)
    fun_draw_open = fb_draw["total_stake"]
    fun_over_open = fb_over["total_stake"]
    kelly_open = sum(k["stake"] for k in kelly_picks)

    bankrolls = {
        "draw": {"bank_start": BANKROLL_DRAW, "week_start": BANKROLL_DRAW, "open": round(draw_open, 1), "after_open": round(BANKROLL_DRAW - draw_open, 1), "picks": len(draw_singles)},
        "over": {"bank_start": BANKROLL_OVER, "week_start": BANKROLL_OVER, "open": round(over_open, 1), "after_open": round(BANKROLL_OVER - over_open, 1), "picks": len(over_singles)},
        "fun_draw": {"bank_start": BANKROLL_FUN_DRAW, "week_start": BANKROLL_FUN_DRAW, "open": round(fun_draw_open, 1), "after_open": round(BANKROLL_FUN_DRAW - fun_draw_open, 1), "picks": len(fb_draw["picks"])},
        "fun_over": {"bank_start": BANKROLL_FUN_OVER, "week_start": BANKROLL_FUN_OVER, "open": round(fun_over_open, 1), "after_open": round(BANKROLL_FUN_OVER - fun_over_open, 1), "picks": len(fb_over["picks"])},
        "kelly": {"bank_start": BANKROLL_KELLY, "week_start": BANKROLL_KELLY, "open": round(kelly_open, 1), "after_open": round(BANKROLL_KELLY - kelly_open, 1), "picks": len(kelly_picks)},
    }

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "fixtures_total": len(fixtures),
        "window": th_report.get("window", {}),
        "draw_singles": draw_singles,
        "over_singles": over_singles,
        "funbet_draw": fb_draw,
        "funbet_over": fb_over,
        "kelly": kelly_picks,
        "bankrolls": bankrolls,
    }

    os.makedirs(os.path.dirname(FRIDAY_REPORT_PATH), exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log(f"✅ Friday Shortlist v3.3 saved → {FRIDAY_REPORT_PATH}")

if __name__ == "__main__":
    main()
