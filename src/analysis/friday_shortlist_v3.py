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

# "Not forced" caps (can be <10 naturally)
MAX_DRAWS = int(os.getenv("MAX_DRAWS", "7"))
MAX_OVERS = int(os.getenv("MAX_OVERS", "7"))
MAX_KELLY = int(os.getenv("MAX_KELLY", "6"))

# Kelly (SAFE)
KELLY_FRACTION = 0.30
KELLY_MIN_EDGE = 0.15
KELLY_MAX_ODDS = 4.0
KELLY_MIN_PROB = 0.18

# Draw engine
MIN_DRAW_PROB = float(os.getenv("MIN_DRAW_PROB", "0.34"))
MIN_DRAW_ODDS = float(os.getenv("MIN_DRAW_ODDS", "2.80"))
DRAW_STAKE = float(os.getenv("DRAW_STAKE", "30"))

# Over engine
MIN_OVER_PROB = float(os.getenv("MIN_OVER_PROB", "0.65"))
MAX_FAIR_OVER = float(os.getenv("MAX_FAIR_OVER", "1.75"))

# Optional extra quality gate (prevents “meh” picks)
MIN_VALUE_PCT_DRAWS = float(os.getenv("MIN_VALUE_PCT_DRAWS", "0"))   # set 5 if you want stricter
MIN_VALUE_PCT_OVERS = float(os.getenv("MIN_VALUE_PCT_OVERS", "0"))   # set 3 if you want stricter

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
    if over_prob >= 0.70 and fair_over <= 1.55 and score >= 70:
        return "monster", 24.0
    if over_prob >= 0.67 and fair_over <= 1.65 and score >= 67:
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

def generate_picks(fixtures):
    draw_singles = []
    over_singles = []
    kelly_candidates = []

    for f in fixtures:
        home = f.get("home")
        away = f.get("away")
        league = f.get("league")
        match_label = f"{home} – {away}"

        fair_1 = safe_float(f.get("fair_1"))
        fair_x = safe_float(f.get("fair_x"))
        fair_2 = safe_float(f.get("fair_2"))
        fair_over = safe_float(f.get("fair_over_2_5"))

        draw_prob = safe_float(f.get("draw_prob"), 0.0) or 0.0
        over_prob = safe_float(f.get("over_2_5_prob"), 0.0) or 0.0

        offered_1 = safe_float(f.get("offered_1"))
        offered_x = safe_float(f.get("offered_x"))
        offered_2 = safe_float(f.get("offered_2"))
        offered_over = safe_float(f.get("offered_over_2_5"))

        value_pct_x = safe_float(f.get("value_pct_x"))
        value_pct_over = safe_float(f.get("value_pct_over"))

        draw_score = compute_draw_score(draw_prob, league)
        over_score = compute_over_score(over_prob, league)

        # ---------------- DRAW SINGLES ----------------
        if (
            draw_prob >= MIN_DRAW_PROB
            and offered_x is not None and offered_x >= MIN_DRAW_ODDS
            and fair_x is not None
            and (value_pct_x is None or value_pct_x >= MIN_VALUE_PCT_DRAWS)
        ):
            draw_singles.append(
                {
                    "match": match_label,
                    "league": league,
                    "fair": fair_x,
                    "prob": round(draw_prob, 3),
                    "score": round(draw_score, 1),
                    "odds": offered_x,
                    "stake": DRAW_STAKE,
                }
            )

        # ---------------- OVER SINGLES ----------------
        if (
            fair_over is not None
            and over_prob >= MIN_OVER_PROB
            and fair_over <= MAX_FAIR_OVER
            and offered_over is not None and offered_over > 1.01
            and (value_pct_over is None or value_pct_over >= MIN_VALUE_PCT_OVERS)
        ):
            tier, stake = classify_over_stake(over_prob, fair_over, league)
            over_singles.append(
                {
                    "match": match_label,
                    "league": league,
                    "fair": fair_over,
                    "prob": round(over_prob, 3),
                    "score": round(over_score, 1),
                    "odds": offered_over,
                    "tier": tier,
                    "stake": float(stake),
                }
            )

        # ---------------- KELLY (ONLY 1 & 2, MAX ODDS 4.0) ----------------
        def add_kelly_candidate(market_label, fair, offered, prob_model):
            if fair is None or offered is None:
                return
            if offered > KELLY_MAX_ODDS:
                return
            if prob_model < KELLY_MIN_PROB:
                return

            edge_ratio = (offered / fair) - 1.0
            if edge_ratio < KELLY_MIN_EDGE:
                return

            p = prob_model
            q = 1.0 - p
            b = offered - 1.0
            if b <= 0:
                return

            f_full = (b * p - q) / b
            if f_full <= 0:
                return

            f = f_full * KELLY_FRACTION

            if offered <= 2.5:
                cap = 0.05
            else:
                cap = 0.03
            f = min(f, cap)
            if f <= 0:
                return

            stake = max(3.0, round(BANKROLL_KELLY * f, 1))

            kelly_candidates.append(
                {
                    "match": match_label,
                    "league": league,
                    "market": market_label,
                    "fair": fair,
                    "odds": offered,
                    "prob": round(prob_model, 3),
                    "edge": round(edge_ratio * 100.0, 1),
                    "stake": stake,
                    "f_fraction": round(f, 4),
                }
            )

        p_home = (1.0 / fair_1) if fair_1 and fair_1 > 0 else 0.0
        p_away = (1.0 / fair_2) if fair_2 and fair_2 > 0 else 0.0

        if offered_1 is not None and fair_1 is not None:
            add_kelly_candidate("Home", fair_1, offered_1, p_home)
        if offered_2 is not None and fair_2 is not None:
            add_kelly_candidate("Away", fair_2, offered_2, p_away)

    # sort + "not forced" caps
    draw_singles = sorted(draw_singles, key=lambda d: d["score"], reverse=True)[:MAX_DRAWS]
    over_singles = sorted(over_singles, key=lambda o: o["score"], reverse=True)[:MAX_OVERS]
    kelly_candidates = sorted(kelly_candidates, key=lambda k: k["edge"], reverse=True)[:MAX_KELLY]

    return draw_singles, over_singles, kelly_candidates

def main():
    log("🚀 Running Friday Shortlist v3.2 (not-forced caps, uses Thursday value_pct)")

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

    log(f"✅ Friday Shortlist v3.2 saved → {FRIDAY_REPORT_PATH}")

if __name__ == "__main__":
    main()
