import os
import json
from datetime import datetime

# ============================================================
#  FRIDAY SHORTLIST v3 — PRODUCTION (UNITS VERSION)
#  - Διαβάζει το Thursday report v3 (με fair + offered odds)
#  - ΔΕΝ χτυπάει ξανά TheOddsAPI
#  - Χτίζει:
#       * Draw Singles (flat 30u, με min prob & min odds)
#       * Over Singles (8 / 16 / 24u, standard/premium/monster)
#       * FunBet Draw (dynamic stake, max 20% bankroll)
#       * FunBet Over (dynamic stake, max 20% bankroll)
#       * Kelly value bets (ΜΟΝΟ 1 & 2) με ασφαλές Kelly
# ============================================================

THURSDAY_REPORT_PATH = "logs/thursday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v3.json"

# ------------------------------------------------------------
# BANKROLLS (σε μονάδες = ευρώ)
# ------------------------------------------------------------
BANKROLL_DRAW = 1000.0
BANKROLL_OVER = 1001000.0 if False else 1000.0  # ignore
BANKROLL_FUN_DRAW = 300.0
BANKROLL_FUN_OVER = 300.0
BANKROLL_KELLY = 600.0

UNIT = 1.0

MAX_FUN_EXPOSURE_PCT = 0.20      # 20% ανά κύκλο
KELLY_FRACTION = 0.30            # κλασματικό Kelly 30%
KELLY_MIN_EDGE = 0.15            # 15%+ value
KELLY_MAX_ODDS = 8.0
KELLY_MAX_PICKS = 6
KELLY_MIN_PROB = 0.18            # >= 18% πιθανότητα από το μοντέλο

MIN_DRAW_PROB = 0.38             # ≥ 38% για Draw Engine
MIN_DRAW_ODDS = 2.80             # προσφερόμενη απόδοση Χ ≥ 2.80

# ------------------------------------------------------------
# LEAGUE PRIORITIES
# ------------------------------------------------------------
DRAW_PRIORITY_LEAGUES = {
    "Ligue 1", "Serie A", "La Liga", "Championship",
    "Serie B", "Ligue 2", "Liga Portugal 2", "Swiss Super League",
}

OVER_PRIORITY_LEAGUES = {
    "Bundesliga", "Eredivisie", "Jupiler Pro League", "Superliga",
    "Allsvenskan", "Eliteserien", "Swiss Super League", "Liga Portugal 1",
}

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def log(msg):
    print(msg, flush=True)

def parse_match(match: str):
    if "–" in match:
        home, away = match.split("–")
    elif "-" in match:
        home, away = match.split("-")
    else:
        return match.strip(), ""
    return home.strip(), away.strip()

# ------------------------------------------------------------
# LOAD THURSDAY REPORT
# ------------------------------------------------------------

def load_thursday_fixtures():
    if not os.path.exists(THURSDAY_REPORT_PATH):
        raise FileNotFoundError(f"Thursday report not found: {THURSDAY_REPORT_PATH}")
    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("fixtures", []), data

# ------------------------------------------------------------
# SCORING
# ------------------------------------------------------------

def compute_draw_score(draw_prob, league):
    score = draw_prob * 100.0
    if league in DRAW_PRIORITY_LEAGUES:
        score *= 1.05
    return score

def compute_over_score(over_prob, league):
    score = over_prob * 100.0
    if league in OVER_PRIORITY_LEAGUES:
        score *= 1.05
    return score

# ------------------------------------------------------------
# OVER STAKE CLASSIFICATION
# ------------------------------------------------------------

def classify_over_stake(over_prob, fair_over, league):
    score = compute_over_score(over_prob, league)

    if over_prob >= 0.70 and fair_over <= 1.55 and score >= 70:
        return "monster", 24.0

    if over_prob >= 0.67 and fair_over <= 1.65 and score >= 67:
        return "premium", 16.0

    return "standard", 8.0

# ------------------------------------------------------------
# FUNBET HELPER
# ------------------------------------------------------------

def compute_system_stake(bankroll, columns, max_exposure_pct=MAX_FUN_EXPOSURE_PCT,
                         min_unit=1.0, max_unit=5.0):
    if columns <= 0:
        return 0.0, 0.0

    max_exposure = bankroll * max_exposure_pct
    base_unit = max_exposure / columns

    unit = int(base_unit)
    if unit < min_unit:
        unit = min_unit
    if unit > max_unit:
        unit = max_unit

    total = unit * columns

    if total > max_exposure:
        unit = max(min_unit, int(max_exposure // columns))
        total = unit * columns

    return float(unit), float(total)

# ------------------------------------------------------------
# GENERATE PICKS
# ------------------------------------------------------------

def generate_picks(fixtures):
    draw_singles = []
    over_singles = []
    kelly_candidates = []

    for f in fixtures:
        match = f.get("match", "")
        league = f.get("league", "")
        home, away = parse_match(match)

        probs = f.get("probs", {})
        fair = f.get("fair", {})
        offered = f.get("offered", {})

        fair_1 = fair.get("home")
        fair_x = fair.get("draw")
        fair_2 = fair.get("away")
        fair_over = fair.get("over")

        draw_prob = probs.get("draw_prob") or 0.0
        over_prob = probs.get("over_prob") or 0.0
        home_prob = probs.get("home_prob") or 0.0
        away_prob = probs.get("away_prob") or 0.0

        offered_1 = offered.get("home")
        offered_x = offered.get("draw")
        offered_2 = offered.get("away")
        offered_over = offered.get("over") or offered.get("over_2_5")

        draw_score = compute_draw_score(draw_prob, league)
        over_score = compute_over_score(over_prob, league)

        # ---------------- DRAW SINGLES ----------------
        if (
            draw_prob >= MIN_DRAW_PROB
            and offered_x is not None
            and offered_x >= MIN_DRAW_ODDS
        ):
            draw_singles.append(
                {
                    "match": match,
                    "league": league,
                    "fair": fair_x,
                    "prob": round(draw_prob, 3),
                    "score": round(draw_score, 1),
                    "odds": offered_x,
                    "stake": 30.0,
                }
            )

        # ---------------- OVER SINGLES ----------------
        if over_prob >= 0.65 and fair_over and fair_over <= 1.75:
            tier, stake = classify_over_stake(over_prob, fair_over, league)
            over_singles.append(
                {
                    "match": match,
                    "league": league,
                    "fair": fair_over,
                    "prob": round(over_prob, 3),
                    "score": round(over_score, 1),
                    "odds": offered_over,
                    "tier": tier,
                    "stake": stake,
                }
            )

        # ---------------- KELLY CANDIDATES ----------------
        def add_kelly_candidate(label, fair_price, offered_price, prob_model):
            if not fair_price or not offered_price:
                return

            if prob_model < KELLY_MIN_PROB:
                return

            edge_ratio = (offered_price / fair_price) - 1
            if edge_ratio < KELLY_MIN_EDGE:
                return

            if offered_price > KELLY_MAX_ODDS:
                return

            b = offered_price - 1
            p = prob_model
            q = 1 - p

            f_full = (b * p - q) / b
            if f_full <= 0:
                return

            f = min(f_full * KELLY_FRACTION,
                    0.05 if offered_price <= 2.5 else
                    0.03 if offered_price <= 4 else
                    0.02 if offered_price <= 6 else 0.01)

            if f <= 0:
                return

            stake_units = max(3.0, round(BANKROLL_KELLY * f, 1))

            kelly_candidates.append(
                {
                    "match": match,
                    "league": league,
                    "market": label,
                    "fair": fair_price,
                    "odds": offered_price,
                    "prob": round(prob_model, 3),
                    "edge": round(edge_ratio * 100, 1),
                    "stake": stake_units,
                    "f_fraction": round(f, 4),
                }
            )

        if offered_1:
            add_kelly_candidate("Home", fair_1, offered_1, home_prob)
        if offered_2:
            add_kelly_candidate("Away", fair_2, offered_2, away_prob)

    draw_singles = sorted(draw_singles, key=lambda d: d["score"], reverse=True)[:10]
    over_singles = sorted(over_singles, key=lambda o: o["score"], reverse=True)[:10]
    kelly_candidates = sorted(kelly_candidates, key=lambda k: k["edge"], reverse=True)[:6]

    return draw_singles, over_singles, kelly_candidates

# ------------------------------------------------------------
# FUNBET SYSTEMS
# ------------------------------------------------------------

def funbet_draw(draw_singles):
    picks = sorted(draw_singles, key=lambda x: x["score"], reverse=True)[:7]
    n = len(picks)

    if n < 3:
        return {"system": None, "columns": 0, "unit": 0.0,
                "total_stake": 0.0, "picks": []}

    if n == 3:
        sys, cols = "3/3", 1
    elif n == 4:
        sys, cols = "3/4", 4
    elif n == 5:
        sys, cols = "3/5", 10
    elif n == 6:
        sys, cols = "4/6", 15
    else:
        sys, cols = "4/7", 35

    unit, total = compute_system_stake(BANKROLL_FUN_DRAW, cols)
    return {"system": sys, "columns": cols, "unit": unit,
            "total_stake": total, "picks": picks}


def funbet_over(over_singles):
    picks = sorted(over_singles, key=lambda x: x["score"], reverse=True)[:7]
    n = len(picks)

    if n < 3:
        return {"system": None, "columns": 0, "unit": 0.0,
                "total_stake": 0.0, "picks": []}

    if n == 3:
        sys, cols = "3/3", 1
    elif n == 4:
        sys, cols = "2/4", 6
    elif n == 5:
        sys, cols = "2/5", 10
    elif n == 6:
        sys, cols = "3/6", 20
    else:
        sys, cols = "3/7", 35

    unit, total = compute_system_stake(BANKROLL_FUN_OVER, cols)
    return {"system": sys, "columns": cols, "unit": unit,
            "total_stake": total, "picks": picks}

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    log("🚀 Running Friday Shortlist v3 (no extra odds calls, tightened draw/Kelly logic)")

    fixtures, th_report = load_thursday_fixtures()

    draw_singles, over_singles, kelly_picks = generate_picks(fixtures)

    fb_draw = funbet_draw(draw_singles)
    fb_over = funbet_over(over_singles)

    draw_open = sum(d["stake"] for d in draw_singles)
    over_open = sum(o["stake"] for o in over_singles)
    fun_draw_open = fb_draw["total_stake"]
    fun_over_open = fb_over["total_stake"]
    kelly_open = sum(k["stake"] for k in kelly_picks)

    bankrolls = {
        "draw": {
            "bank_start": BANKROLL_DRAW,
            "week_start": BANKROLL_DRAW,
            "open": round(draw_open, 1),
            "after_open": round(BANKROLL_DRAW - draw_open, 1),
            "picks": len(draw_singles),
        },
        "over": {
            "bank_start": BANKROLL_OVER,
            "week_start": BANKROLL_OVER,
            "open": round(over_open, 1),
            "after_open": round(BANKROLL_OVER - over_open, 1),
            "picks": len(over_singles),
        },
        "fun_draw": {
            "bank_start": BANKROLL_FUN_DRAW,
            "week_start": BANKROLL_FUN_DRAW,
            "open": round(fun_draw_open, 1),
            "after_open": round(BANKROLL_FUN_DRAW - fun_draw_open, 1),
            "picks": len(fb_draw["picks"]),
        },
        "fun_over": {
            "bank_start": BANKROLL_FUN_OVER,
            "week_start": BANKROLL_FUN_OVER,
            "open": round(fun_over_open, 1),
            "after_open": round(BANKROLL_FUN_OVER - fun_over_open, 1),
            "picks": len(fb_over["picks"]),
        },
        "kelly": {
            "bank_start": BANKROLL_KELLY,
            "week_start": BANKROLL_KELLY,
            "open": round(kelly_open, 1),
            "after_open": round(BANKROLL_KELLY - kelly_open, 1),
            "picks": len(kelly_picks),
        },
    }

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "fixtures_total": len(fixtures),
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

    log(f"✅ Friday Shortlist v3 saved → {FRIDAY_REPORT_PATH}")


if __name__ == "__main__":
    main()
