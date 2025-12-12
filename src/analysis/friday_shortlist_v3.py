import os
import json
from datetime import datetime

THURSDAY_REPORT_PATH = "logs/thursday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v3.json"

# ------------------------- BANKROLLS (units = €) -------------------------
BANKROLL_DRAW = 1000.0
BANKROLL_OVER = 1000.0
BANKROLL_FUN_DRAW = 300.0
BANKROLL_FUN_OVER = 300.0
BANKROLL_KELLY = 600.0

MAX_FUN_EXPOSURE_PCT = 0.20

# soft caps (not forced)
MAX_DRAWS = int(os.getenv("MAX_DRAWS", "7"))
MAX_OVERS = int(os.getenv("MAX_OVERS", "7"))
MAX_KELLY = int(os.getenv("MAX_KELLY", "6"))

# ------------------------- SELECTION DEFAULTS (calibrated) -------------------------
MIN_EDGE = float(os.getenv("MIN_EDGE", "0.01"))  # 1% noise floor

# Draw filters
DRAW_MIN_PROB = float(os.getenv("DRAW_MIN_PROB", "0.22"))
DRAW_MIN_EDGE = float(os.getenv("DRAW_MIN_EDGE", "0.03"))
DRAW_LAMBDA_DIFF_MAX = float(os.getenv("DRAW_LAMBDA_DIFF_MAX", "0.35"))
DRAW_STAKE = float(os.getenv("DRAW_STAKE", "30"))

# Over filters
OVER_MIN_PROB = float(os.getenv("OVER_MIN_PROB", "0.55"))
OVER_MIN_EDGE = float(os.getenv("OVER_MIN_EDGE", "0.03"))
OVER_LAMBDA_TOTAL_MIN = float(os.getenv("OVER_LAMBDA_TOTAL_MIN", "2.75"))
MAX_FAIR_OVER = float(os.getenv("MAX_FAIR_OVER", "1.82"))  # optional sanity

# Portfolio risk
PORTFOLIO_CAP = float(os.getenv("PORTFOLIO_CAP", "0.20"))  # 20% of bankroll exposure cap

# ------------------------- LEAGUE PRIORITIES -------------------------
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

# ------------------------- NEW: EDGE-BANDED KELLY -------------------------
def kelly_fraction(p: float, odds: float) -> float:
    if p is None or odds is None or odds <= 1.0:
        return 0.0
    b = odds - 1.0
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f)

def edge_value(p: float, odds: float) -> float:
    if p is None or odds is None:
        return 0.0
    return (p * odds) - 1.0

def band_multiplier(edge: float) -> float:
    # edge in decimal (0.03 = 3%)
    if edge <= 0.01:
        return 0.00
    if edge <= 0.03:
        return 0.10
    if edge <= 0.10:
        return 0.25
    return 0.50

def generate_picks(fixtures):
    draw_singles = []
    over_singles = []
    kelly_candidates = []

    for f in fixtures:
        home = f.get("home")
        away = f.get("away")
        league = f.get("league")
        match_label = f"{home} – {away}"

        # model probs (Thursday)
        p_home = safe_float(f.get("home_prob"), None)
        p_draw = safe_float(f.get("draw_prob"), None)
        p_away = safe_float(f.get("away_prob"), None)
        p_over = safe_float(f.get("over_2_5_prob"), None)

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

        draw_score = compute_draw_score(p_draw or 0.0, league)
        over_score = compute_over_score(p_over or 0.0, league)

        # ---------------- DRAW SINGLES ----------------
        if p_draw is not None and offered_x is not None and offered_x > 1.0:
            e = edge_value(p_draw, offered_x)
            lam_diff = abs(lam_h - lam_a) if (lam_h is not None and lam_a is not None) else 999.0

            if (
                p_draw >= DRAW_MIN_PROB
                and e >= DRAW_MIN_EDGE
                and lam_diff <= DRAW_LAMBDA_DIFF_MAX
            ):
                draw_singles.append({
                    "match": match_label,
                    "league": league,
                    "fair": fair_x,
                    "prob": round(p_draw, 3),
                    "score": round(draw_score, 1),
                    "odds": offered_x,
                    "stake": float(DRAW_STAKE),
                    "edge": round(e * 100.0, 1),
                })

        # ---------------- OVER SINGLES ----------------
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
                    "fair": fair_over,
                    "prob": round(p_over, 3),
                    "score": round(over_score, 1),
                    "odds": offered_over,
                    "tier": tier,
                    "stake": float(stake),
                    "edge": round(e * 100.0, 1),
                })

        # ---------------- KELLY (HOME/AWAY) ----------------
        def add_kelly(market, p, offered, fair):
            if p is None or offered is None or offered <= 1.0:
                return
            e = edge_value(p, offered)
            if e <= MIN_EDGE:
                return
            f_full = kelly_fraction(p, offered)
            if f_full <= 0:
                return

            mult = band_multiplier(e)
            if mult <= 0:
                return

            f_raw = f_full * mult

            # store fraction for later portfolio scaling
            kelly_candidates.append({
                "match": match_label,
                "league": league,
                "market": market,
                "prob": round(p, 3),
                "fair": fair,
                "odds": offered,
                "edge": round(e * 100.0, 1),
                "f_raw": f_raw,
            })

        add_kelly("Home", p_home, offered_1, fair_1)
        add_kelly("Away", p_away, offered_2, fair_2)

    # cap counts (not forced, just slice)
    draw_singles = sorted(draw_singles, key=lambda d: d["score"], reverse=True)[:MAX_DRAWS]
    over_singles = sorted(over_singles, key=lambda o: o["score"], reverse=True)[:MAX_OVERS]
    kelly_candidates = sorted(kelly_candidates, key=lambda k: k["edge"], reverse=True)[:MAX_KELLY]

    # ---------------- PORTFOLIO CAP SCALING (Kelly only) ----------------
    total_f = sum(k["f_raw"] for k in kelly_candidates)
    cap = PORTFOLIO_CAP
    scale = 1.0
    if total_f > cap and total_f > 0:
        scale = cap / total_f

    kelly_picks = []
    for k in kelly_candidates:
        f_final = k["f_raw"] * scale
        stake = round(BANKROLL_KELLY * f_final, 1)
        if stake < 3.0:
            stake = 3.0

        kelly_picks.append({
            "match": k["match"],
            "league": k["league"],
            "market": k["market"],
            "prob": k["prob"],
            "fair": k["fair"],
            "odds": k["odds"],
            "edge": k["edge"],
            "stake": stake,
            "f_fraction": round(f_final, 5),
        })

    return draw_singles, over_singles, kelly_picks

def main():
    log("🚀 Running Friday Shortlist v3.6 (Edge-Banded Kelly + Portfolio Cap)")

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

    log(f"✅ Friday Shortlist v3.6 saved → {FRIDAY_REPORT_PATH}")

if __name__ == "__main__":
    main()
