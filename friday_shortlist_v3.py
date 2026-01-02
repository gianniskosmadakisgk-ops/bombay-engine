import os
import json
from datetime import datetime
import itertools

TUESDAY_REPORT_PATH = "logs/tuesday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v3.json"

os.makedirs("logs", exist_ok=True)

DRAW_WALLET = 300
OVER_WALLET = 300
FANBET_DRAW_WALLET = 100
FANBET_OVER_WALLET = 100
KELLY_WALLET = 300.0

MAX_DRAW_PICKS = 10
MAX_OVER_PICKS = 10

FUNBET_DRAW_STAKE_PER_COL = 2.0
FUNBET_OVER_STAKE_PER_COL = 4.0

DRAW_MIN_SCORE = 7.5
DRAW_MIN_ODDS = 2.70

OVER_MIN_SCORE = 7.5
OVER_MIN_FAIR = 1.70

OVER_AUTO_SCORE = 9.0
OVER_NEG_EDGE_LIMIT = -0.10

KELLY_VALUE_THRESHOLD = 0.15
KELLY_FRACTION = 0.40
KELLY_MIN_PROB = 0.25
KELLY_MAX_EXPOSURE_PCT = 0.35


def log(msg: str):
    print(msg, flush=True)


def load_tuesday_fixtures():
    if not os.path.exists(TUESDAY_REPORT_PATH):
        raise FileNotFoundError(f"Tuesday report not found: {TUESDAY_REPORT_PATH}")

    with open(TUESDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixtures = data.get("fixtures", [])
    log(f"Loaded {len(fixtures)} fixtures from Tuesday v3.")
    return fixtures, data


def flat_stake(score):
    if score >= 8.5:
        return 20
    elif score >= 7.5:
        return 15
    return 0


def generate_picks(fixtures):
    draw_singles = []
    over_singles = []
    kelly_candidates = []

    def add_kelly(match_label, league, label, fair, offered):
        if fair is None or offered is None:
            return
        try:
            fair = float(fair)
            offered = float(offered)
        except (TypeError, ValueError):
            return

        if fair <= 0 or offered <= 1:
            return

        p = 1.0 / fair
        if p < KELLY_MIN_PROB:
            return

        edge = (offered - fair) / fair
        if edge < KELLY_VALUE_THRESHOLD:
            return

        b = offered - 1
        q = 1 - p
        fk = (b * p - q) / b
        if fk <= 0:
            return

        stake_raw = fk * KELLY_FRACTION * KELLY_WALLET
        if stake_raw > 0:
            kelly_candidates.append({
                "match": match_label,
                "league": league,
                "market": label,
                "fair": round(fair, 2),
                "offered": round(offered, 2),
                "edge": f"{edge:+.0%}",
                "stake_raw": stake_raw,
            })

    for f in fixtures:
        odds_match = f.get("odds_match") or {}
        if not odds_match.get("matched"):
            continue

        league = f.get("league")
        match_label = f"{f.get('home')} - {f.get('away')}"

        fair_1, fair_x, fair_2 = f.get("fair_1"), f.get("fair_x"), f.get("fair_2")
        fair_over = f.get("fair_over_2_5")

        offered_1 = f.get("offered_1")
        offered_x = f.get("offered_x")
        offered_2 = f.get("offered_2")
        offered_over = f.get("offered_over_2_5")

        # ✅ safe on None
        score_draw = float(f.get("draw_prob") or 0) * 10
        score_over = float(f.get("over_2_5_prob") or 0) * 10

        if fair_x and offered_x and score_draw >= DRAW_MIN_SCORE:
            if float(offered_x) >= DRAW_MIN_ODDS:
                stake = flat_stake(score_draw)
                if stake:
                    diff = (float(offered_x) - float(fair_x)) / float(fair_x)
                    draw_singles.append({
                        "match": match_label,
                        "league": league,
                        "odds": round(float(offered_x), 2),
                        "fair": round(float(fair_x), 2),
                        "diff": f"{diff:+.0%}",
                        "score": round(score_draw, 2),
                        "stake": stake,
                    })

        if fair_over and offered_over and score_over >= OVER_MIN_SCORE:
            offered_over_f = float(offered_over)
            fair_over_f = float(fair_over)

            value_ok = offered_over_f >= fair_over_f
            monster = (
                score_over >= OVER_AUTO_SCORE
                and offered_over_f >= OVER_MIN_FAIR
                and (offered_over_f - fair_over_f) / fair_over_f >= OVER_NEG_EDGE_LIMIT
            )

            if offered_over_f >= OVER_MIN_FAIR and (value_ok or monster):
                stake = flat_stake(score_over)
                if stake:
                    diff = (offered_over_f - fair_over_f) / fair_over_f
                    over_singles.append({
                        "match": match_label,
                        "league": league,
                        "odds": round(offered_over_f, 2),
                        "fair": round(fair_over_f, 2),
                        "diff": f"{diff:+.0%}",
                        "score": round(score_over, 2),
                        "stake": stake,
                    })

        add_kelly(match_label, league, "Home", fair_1, offered_1)
        add_kelly(match_label, league, "Draw", fair_x, offered_x)
        add_kelly(match_label, league, "Away", fair_2, offered_2)
        add_kelly(match_label, league, "Over 2.5", fair_over, offered_over)

    draw_singles = sorted(draw_singles, key=lambda x: x["score"], reverse=True)[:MAX_DRAW_PICKS]
    over_singles = sorted(over_singles, key=lambda x: x["score"], reverse=True)[:MAX_OVER_PICKS]

    kelly_candidates = sorted(kelly_candidates, key=lambda x: x["stake_raw"], reverse=True)[:10]
    total_raw = sum(k["stake_raw"] for k in kelly_candidates)
    max_exposure = KELLY_WALLET * KELLY_MAX_EXPOSURE_PCT
    scale = max_exposure / total_raw if total_raw > max_exposure and total_raw > 0 else 1.0

    kelly_final = []
    for k in kelly_candidates:
        stake = round(k["stake_raw"] * scale, 2)
        kelly_final.append({
            "match": k["match"],
            "league": k["league"],
            "market": k["market"],
            "fair": k["fair"],
            "offered": k["offered"],
            "edge": k["edge"],
            "stake (€)": stake,
        })

    return draw_singles, over_singles, kelly_final


def build_funbet_draw(draw_singles):
    sorted_draws = sorted(draw_singles, key=lambda x: x["score"], reverse=True)
    picks = sorted_draws[:6]
    n = len(picks)

    if n >= 6:
        sizes = [4, 5, 6]
        system = "4-5-6"
    elif n == 5:
        sizes = [3, 4, 5]
        system = "3-4-5"
    else:
        return {"picks": picks, "system": None, "columns": 0, "total_stake": 0}

    cols = sum(1 for r in sizes for _ in itertools.combinations(range(n), r))
    total = cols * FUNBET_DRAW_STAKE_PER_COL

    return {
        "picks": picks,
        "system": system,
        "columns": cols,
        "stake_per_column": FUNBET_DRAW_STAKE_PER_COL,
        "total_stake": total,
    }


def build_funbet_over(over_singles):
    sorted_ = sorted(over_singles, key=lambda x: x["score"], reverse=True)
    picks = sorted_[:6]
    n = len(picks)

    if n < 3:
        return {"picks": picks, "system": None, "columns": 0, "total_stake": 0}

    cols = sum(1 for _ in itertools.combinations(range(n), 2))
    total = cols * FUNBET_OVER_STAKE_PER_COL

    return {
        "picks": picks,
        "system": f"2-from-{n}",
        "columns": cols,
        "stake_per_column": FUNBET_OVER_STAKE_PER_COL,
        "total_stake": total,
    }


def bankroll_summary(draw_singles, over_singles, fun_draw, fun_over, kelly):
    draw_spent = sum(x["stake"] for x in draw_singles)
    over_spent = sum(x["stake"] for x in over_singles)
    fun_draw_spent = fun_draw.get("total_stake") or 0
    fun_over_spent = fun_over.get("total_stake") or 0
    kelly_spent = sum(k["stake (€)"] for k in kelly)

    return [
        {"Wallet": "Draw Singles", "Before": f"{DRAW_WALLET}€", "After": f"{DRAW_WALLET-draw_spent:.2f}€", "Open Bets": f"{draw_spent:.2f}€"},
        {"Wallet": "Over Singles", "Before": f"{OVER_WALLET}€", "After": f"{OVER_WALLET-over_spent:.2f}€", "Open Bets": f"{over_spent:.2f}€"},
        {"Wallet": "FanBet Draw", "Before": f"{FANBET_DRAW_WALLET}€", "After": f"{FANBET_DRAW_WALLET-fun_draw_spent:.2f}€", "Open Bets": f"{fun_draw_spent:.2f}€"},
        {"Wallet": "FanBet Over", "Before": f"{FANBET_OVER_WALLET}€", "After": f"{FANBET_OVER_WALLET-fun_over_spent:.2f}€", "Open Bets": f"{fun_over_spent:.2f}€"},
        {"Wallet": "Kelly", "Before": f"{KELLY_WALLET:.0f}€", "After": f"{KELLY_WALLET-kelly_spent:.2f}€", "Open Bets": f"{kelly_spent:.2f}€"},
    ]


def main():
    log("🎯 Running Friday Shortlist v3 (Tuesday Edition)...")

    fixtures, meta_src = load_tuesday_fixtures()

    draw_singles, over_singles, kelly = generate_picks(fixtures)
    fun_draw = build_funbet_draw(draw_singles)
    fun_over = build_funbet_over(over_singles)
    banks = bankroll_summary(draw_singles, over_singles, fun_draw, fun_over, kelly)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "source": {
            "path": TUESDAY_REPORT_PATH,
            "generated_at": meta_src.get("generated_at"),
            "window": meta_src.get("window"),
            "fixtures_total": meta_src.get("fixtures_total", len(fixtures)),
        },
        "meta": {
            "fixtures_used": sum(1 for f in fixtures if (f.get("odds_match") or {}).get("matched")),
            "draw_singles": len(draw_singles),
            "over_singles": len(over_singles),
            "kelly_picks": len(kelly),
            "funbet_draw_cols": fun_draw.get("columns", 0),
            "funbet_over_cols": fun_over.get("columns", 0),
        },
        "draw_singles": draw_singles,
        "over_singles": over_singles,
        "funbet_draw": fun_draw,
        "funbet_over": fun_over,
        "kelly": kelly,
        "bankroll_status": banks,
    }

    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log(f"✅ Saved Friday shortlist → {FRIDAY_REPORT_PATH}")


if __name__ == "__main__":
    main()
