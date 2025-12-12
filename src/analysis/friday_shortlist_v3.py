import os
import json
from datetime import datetime

# ============================================================
#  FRIDAY SHORTLIST v3 — UNITS VERSION (NO extra TheOddsAPI calls)
#  - Reads logs/thursday_report_v3.json
#  - Uses fair_* / probs / offered_*
#  - Builds:
#       * Draw Singles (flat stake)
#       * Over Singles (8/16/24u tiers)
#       * FunBet Draw/Over systems
#       * Kelly value bets (ONLY 1 & 2)
#
#  FIXES v3.1:
#    - Draw picks were impossible due to Thursday draw_prob cap (<=0.35).
#      So MIN_DRAW_PROB is set to a REALISTIC value.
#    - Over picks require offered_over_2_5 to exist.
# ============================================================

THURSDAY_REPORT_PATH = "logs/thursday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v3.json"

# ------------------------- BANKROLLS (units = €) -------------------------
BANKROLL_DRAW = 1000.0
BANKROLL_OVER = 1000.0
BANKROLL_FUN_DRAW = 300.0
BANKROLL_FUN_OVER = 300.0
BANKROLL_KELLY = 600.0

MAX_FUN_EXPOSURE_PCT = 0.20
KELLY_FRACTION = 0.30
KELLY_MIN_EDGE = 0.15
KELLY_MAX_ODDS = 8.0
KELLY_MAX_PICKS = 6
KELLY_MIN_PROB = 0.18

# IMPORTANT: Thursday caps draw_prob to <=0.35, so 0.38 was impossible.
MIN_DRAW_PROB = 0.26          # realistic for this model
MIN_DRAW_ODDS = 2.80

# Optional "value draw" fallback (to avoid zero draws when market offers value)
VALUE_DRAW_MIN_PROB = 0.22
VALUE_DRAW_MIN_EDGE = 0.08    # 8% value vs fair
VALUE_DRAW_MIN_ODDS = 2.60

# ------------------------- LEAGUE PRIORITIES -------------------------
DRAW_PRIORITY_LEAGUES = {
    "Ligue 1", "Serie A", "La Liga", "Championship",
    "Serie B", "Ligue 2", "Liga Portugal 2", "Swiss Super League",
}

OVER_PRIORITY_LEAGUES = {
    "Bundesliga", "Eredivisie", "Jupiler Pro League", "Superliga",
    "Allsvenskan", "Eliteserien", "Swiss Super League", "Liga Portugal 1",
}

# ------------------------- HELPERS -------------------------
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
    score = draw_prob * 100.0
    if league in DRAW_PRIORITY_LEAGUES:
        score *= 1.05
    return score

def compute_over_score(over_prob, league):
    score = over_prob * 100.0
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
    if unit < min_unit:
        unit = min_unit
    if unit > max_unit:
        unit = max_unit

    total = unit * columns

    if total > max_exposure:
        unit = max(min_unit, int(max_exposure // columns))
        total = unit * columns

    return float(unit), float(total)

# ------------------------- LOAD THURSDAY REPORT -------------------------
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

    return {"system": sys, "columns": cols, "unit": unit, "total_stake": total, "picks": picks}

def funbet_over(over_singles):
    picks = sorted(over_singles, key=lambda x: x["score"], reverse=True)[:7]
    n = len(picks)

    if n < 3:
        return {"system": None, "columns": 0, "unit": 0.0, "total_stake": 0.0, "picks": []}

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

    return {"system": sys, "columns": cols, "unit": unit, "total_stake": total, "picks": picks}

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

        draw_score = compute_draw_score(draw_prob, league)
        over_score = compute_over_score(over_prob, league)

        # ---------------- DRAW SINGLES ----------------
        # Core draw rule (realistic)
        core_draw_ok = (
            draw_prob >= MIN_DRAW_PROB
            and offered_x is not None
            and offered_x >= MIN_DRAW_ODDS
        )

        # Value draw fallback (prevents "0 draws" weeks)
        value_draw_ok = False
        if (not core_draw_ok) and fair_x and offered_x:
            edge = (offered_x / fair_x) - 1.0
            value_draw_ok = (
                draw_prob >= VALUE_DRAW_MIN_PROB
                and offered_x >= VALUE_DRAW_MIN_ODDS
                and edge >= VALUE_DRAW_MIN_EDGE
            )

        if core_draw_ok or value_draw_ok:
            draw_singles.append(
                {
                    "match": match_label,
                    "league": league,
                    "fair": fair_x,
                    "prob": round(draw_prob, 3),
                    "score": round(draw_score, 1),
                    "odds": offered_x,
                    "stake": 30.0,
                }
            )

        # ---------------- OVER SINGLES ----------------
        # Require offered_over to exist (otherwise it's not a real pick)
        if (
            fair_over is not None
            and over_prob >= 0.65
            and fair_over <= 1.75
            and offered_over is not None
            and offered_over > 1.01
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

        # ---------------- KELLY (ONLY 1 & 2) ----------------
        def add_kelly_candidate(market_label, fair, offered, prob_model):
            if fair is None or offered is None:
                return
            if prob_model < KELLY_MIN_PROB:
                return

            edge_ratio = (offered / fair) - 1.0
            if edge_ratio < KELLY_MIN_EDGE:
                return
            if offered > KELLY_MAX_ODDS:
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
            elif offered <= 4.0:
                cap = 0.03
            elif offered <= 6.0:
                cap = 0.02
            else:
                cap = 0.01

            f = min(f, cap)
            if f <= 0:
                return

            raw_stake = BANKROLL_KELLY * f
            stake = max(3.0, round(raw_stake, 1))

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

        p_home = 1.0 / fair_1 if fair_1 and fair_1 > 0 else 0.0
        p_away = 1.0 / fair_2 if fair_2 and fair_2 > 0 else 0.0

        if offered_1:
            add_kelly_candidate("Home", fair_1, offered_1, p_home)
        if offered_2:
            add_kelly_candidate("Away", fair_2, offered_2, p_away)

    draw_singles = sorted(draw_singles, key=lambda d: d["score"], reverse=True)[:10]
    over_singles = sorted(over_singles, key=lambda o: o["score"], reverse=True)[:10]
    kelly_candidates = sorted(kelly_candidates, key=lambda k: k["edge"], reverse=True)[:KELLY_MAX_PICKS]

    return draw_singles, over_singles, kelly_candidates

# ------------------------- MAIN -------------------------
def main():
    log("🚀 Running Friday Shortlist v3 (units, no extra odds calls)")

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
        "draw": {"bank_start": 1000.0, "week_start": 1000.0, "open": round(draw_open, 1), "after_open": round(1000.0 - draw_open, 1), "picks": len(draw_singles)},
        "over": {"bank_start": 1000.0, "week_start": 1000.0, "open": round(over_open, 1), "after_open": round(1000.0 - over_open, 1), "picks": len(over_singles)},
        "fun_draw": {"bank_start": 300.0, "week_start": 300.0, "open": round(fun_draw_open, 1), "after_open": round(300.0 - fun_draw_open, 1), "picks": len(fb_draw["picks"])},
        "fun_over": {"bank_start": 300.0, "week_start": 300.0, "open": round(fun_over_open, 1), "after_open": round(300.0 - fun_over_open, 1), "picks": len(fb_over["picks"])},
        "kelly": {"bank_start": 600.0, "week_start": 600.0, "open": round(kelly_open, 1), "after_open": round(600.0 - kelly_open, 1), "picks": len(kelly_picks)},
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

    log(f"✅ Friday Shortlist v3 saved → {FRIDAY_REPORT_PATH}")

if __name__ == "__main__":
    main()
