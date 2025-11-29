import os
import json
from datetime import datetime
from pathlib import Path
import itertools
import re

import requests

# ======================================================
#  FRIDAY SHORTLIST v1  (Giannis Edition)
#
#  - Διαβάζει το Thursday report (fair odds + scores)
#  - Τραβάει ΠΡΑΓΜΑΤΙΚΕΣ αποδόσεις από TheOddsAPI
#  - Φτιάχνει:
#       * Draw singles
#       * Over singles
#       * FunBet Draw system
#       * FunBet Over system
#       * Kelly value bets (1 / X / 2 / Over)
#       * Bankroll summary
#  - Σώζει: logs/friday_shortlist_v1.json
# ======================================================

THURSDAY_REPORT_PATH = "logs/thursday_report_v1.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v1.json"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

os.makedirs("logs", exist_ok=True)

# ---------------------- BANKROLLS ----------------------
DRAW_WALLET = 400
OVER_WALLET = 300
FANBET_DRAW_WALLET = 100
FANBET_OVER_WALLET = 100
KELLY_WALLET = 300

# ---------------------- THRESHOLDS ---------------------
DRAW_MIN_SCORE = 7.5
DRAW_MIN_ODDS = 2.70

OVER_MIN_SCORE = 7.5
OVER_MIN_FAIR = 1.70

KELLY_VALUE_THRESHOLD = 0.15   # +15%
KELLY_FRACTION = 0.40

FUNBET_DRAW_STAKE_PER_COL = 3.0
FUNBET_OVER_STAKE_PER_COL = 4.0

# league name -> TheOddsAPI sport_key
LEAGUE_TO_SPORT = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_1",
}

# ------------------------------------------------------
# Helper logging
# ------------------------------------------------------
def log(msg: str):
    print(msg, flush=True)


# ------------------------------------------------------
# Load Thursday data
# ------------------------------------------------------
def load_thursday_fixtures():
    if not os.path.exists(THURSDAY_REPORT_PATH):
        raise FileNotFoundError(f"Thursday report not found: {THURSDAY_REPORT_PATH}")

    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixtures = data.get("fixtures", [])
    log(f"Loaded {len(fixtures)} fixtures from Thursday report.")
    return fixtures


# ------------------------------------------------------
# Odds API helpers
# ------------------------------------------------------
def api_get_odds(sport_key: str):
    """
    Φέρνει odds για συγκεκριμένο sport_key από TheOddsAPI.
    """
    if not ODDS_API_KEY:
        log("⚠️ ODDS_API_KEY not set, returning empty odds.")
        return []

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
    }
    url = f"{ODDS_BASE_URL}/{sport_key}/odds"

    try:
        res = requests.get(url, params=params, timeout=20)
    except Exception as e:
        log(f"⚠️ Error requesting odds for {sport_key}: {e}")
        return []

    if res.status_code != 200:
        log(f"⚠️ Odds API error {res.status_code} for {sport_key}: {res.text[:200]}")
        return []

    try:
        return res.json()
    except Exception as e:
        log(f"⚠️ JSON decode error for {sport_key}: {e}")
        return []


def normalize_team(name: str) -> str:
    """
    Απλοποιημένο normalizer για να ταιριάζουμε ονόματα ομάδων
    μεταξύ API-Football και TheOddsAPI.
    """
    if not name:
        return ""
    s = name.lower()
    # πετάμε κλασικά suffix (fc, cf, ac κλπ)
    s = re.sub(r"\b(fc|cf|afc|cfc|ac|sc|bk)\b", "", s)
    # αφαιρούμε σημεία στίξης / διπλά κενά
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_odds_index(fixtures):
    """
    Χτυπάει TheOddsAPI μόνο για τα leagues που χρειαζόμαστε
    και χτίζει index:
        (norm_home, norm_away) -> { "draw": price, "over_2_5": price }
    """
    # leagues που εμφανίζονται στα fixtures και έχουμε mapping σε sport_key
    leagues_used = sorted({f["league"] for f in fixtures if f.get("league") in LEAGUE_TO_SPORT})
    log(f"Leagues in Thursday report (with odds support): {leagues_used}")

    odds_index = {}
    total_events = 0

    for league_name in leagues_used:
        sport_key = LEAGUE_TO_SPORT[league_name]
        events = api_get_odds(sport_key)
        log(f"Fetched {len(events)} odds events for {league_name} ({sport_key})")
        total_events += len(events)

        for ev in events:
            home = normalize_team(ev.get("home_team", ""))
            away = normalize_team(ev.get("away_team", ""))
            if not home or not away:
                continue

            best_draw = None
            best_over = None

            for b in ev.get("bookmakers", []):
                for m in b.get("markets", []):
                    key = m.get("key")
                    if key == "h2h":
                        for o in m.get("outcomes", []):
                            if o.get("name", "").lower() == "draw":
                                price = float(o.get("price", 0))
                                if price > 0:
                                    if best_draw is None or price > best_draw:
                                        best_draw = price

                    elif key == "totals":
                        for o in m.get("outcomes", []):
                            name = o.get("name", "").lower()
                            if "over" in name and "2.5" in name:
                                price = float(o.get("price", 0))
                                if price > 0:
                                    if best_over is None or price > best_over:
                                        best_over = price

            odds_index[(home, away)] = {
                "odds_draw": best_draw,
                "odds_over_2_5": best_over,
            }

    log(f"Built odds index for {len(odds_index)} (home, away) pairs. Total events fetched: {total_events}")
    return odds_index


# ------------------------------------------------------
# Pick generators
# ------------------------------------------------------
def flat_stake(score: float) -> int:
    if score >= 8.0:
        return 20
    elif score >= 7.5:
        return 15
    return 0


def generate_picks(fixtures, odds_index):
    draw_singles = []
    over_singles = []
    kelly_picks = []

    matched_count = 0

    for f in fixtures:
        league = f.get("league", "")
        match_label = f.get("match", "")
        fair_1 = f.get("fair_1")
        fair_x = f.get("fair_x")
        fair_2 = f.get("fair_2")
        fair_over = f.get("fair_over")
        score_draw = f.get("score_draw", 0.0)
        score_over = f.get("score_over", 0.0)

        try:
            home_name, away_name = [x.strip() for x in match_label.split("-")]
        except ValueError:
            # περίεργο format αγώνα
            continue

        home_norm = normalize_team(home_name)
        away_norm = normalize_team(away_name)

        odds = odds_index.get((home_norm, away_norm)) or odds_index.get((away_norm, home_norm))
        if odds:
            matched_count += 1

        odds_x = odds.get("odds_draw") if odds else None
        odds_over = odds.get("odds_over_2_5") if odds else None

        # -------- DRAW SINGLES --------
        if fair_x and odds_x and odds_x >= DRAW_MIN_ODDS and score_draw >= DRAW_MIN_SCORE:
            diff_x = (odds_x - fair_x) / fair_x
            stake = flat_stake(score_draw)
            if stake > 0:
                draw_singles.append({
                    "match": match_label,
                    "league": league,
                    "odds": round(odds_x, 2),
                    "fair": round(fair_x, 2),
                    "diff": f"{diff_x:+.0%}",
                    "score": round(score_draw, 2),
                    "stake": stake,
                    "wallet": "Draw",
                })

        # -------- OVER SINGLES --------
        if fair_over and odds_over and fair_over >= OVER_MIN_FAIR and score_over >= OVER_MIN_SCORE:
            diff_over = (odds_over - fair_over) / fair_over
            stake = flat_stake(score_over)
            if stake > 0:
                over_singles.append({
                    "match": match_label,
                    "league": league,
                    "odds": round(odds_over, 2),
                    "fair": round(fair_over, 2),
                    "diff": f"{diff_over:+.0%}",
                    "score": round(score_over, 2),
                    "stake": stake,
                    "wallet": "Over",
                })

        # -------- KELLY (1 / X / 2 / OVER) --------
        def maybe_add_kelly(market_label, fair, offered):
            if not fair or not offered:
                return
            diff = (offered - fair) / fair
            if diff < KELLY_VALUE_THRESHOLD:
                return

            p = 1.0 / fair
            b = offered - 1.0
            q = 1.0 - p
            k_fraction = (b * p - q) / b
            if k_fraction <= 0:
                return
            stake = round(KELLY_WALLET * k_fraction * KELLY_FRACTION, 2)
            if stake <= 0:
                return

            kelly_picks.append({
                "match": match_label,
                "market": market_label,
                "fair": round(fair, 2),
                "offered": round(offered, 2),
                "diff": f"{diff:+.0%}",
                "kelly%": f"{KELLY_FRACTION*100:.0f}%",
                "stake (€)": stake,
            })

        # Kelly για 1 / X / 2 / Over (αν έχουμε odds — εδώ έχουμε μόνο draw & over από odds API)
        if fair_x and odds_x:
            maybe_add_kelly("Draw", fair_x, odds_x)
        if fair_over and odds_over:
            maybe_add_kelly("Over 2.5", fair_over, odds_over)

        # Σημείωση: Αν αργότερα προσθέσουμε odds για 1 & 2, απλά θα καλέσουμε:
        # maybe_add_kelly("Home", fair_1, odds_home)
        # maybe_add_kelly("Away", fair_2, odds_away)

    # Top 10 Kelly
    kelly_picks = sorted(
        kelly_picks,
        key=lambda x: x["stake (€)"],
        reverse=True
    )[:10]

    log(f"Matched odds for {matched_count} / {len(fixtures)} fixtures.")
    log(f"Draw singles: {len(draw_singles)}, Over singles: {len(over_singles)}, Kelly picks: {len(kelly_picks)}")

    return draw_singles, over_singles, kelly_picks


# ------------------------------------------------------
# FunBet systems
# ------------------------------------------------------
def build_funbet_draw(draw_singles):
    """
    Παίρνει τις καλύτερες ισοπαλίες και φτιάχνει σύστημα 3-4-5 ή 4-5-6.
    """
    # Κρατάμε τις καλύτερες ανά score
    sorted_draws = sorted(draw_singles, key=lambda x: x["score"], reverse=True)
    picks = sorted_draws[:6]  # max 6

    n = len(picks)
    system = None
    columns = 0

    if n >= 6:
        # 4-5-6 πάνω σε 6 picks
        sizes = [4, 5, 6]
        system = "4-5-6"
    elif n == 5:
        # 3-4-5 πάνω σε 5 picks
        sizes = [3, 4, 5]
        system = "3-4-5"
    else:
        sizes = []

    if sizes:
        for r in sizes:
            for _ in itertools.combinations(range(n), r):
                columns += 1

    total_stake = columns * FUNBET_DRAW_STAKE_PER_COL

    return {
        "picks": picks,
        "system": system,
        "columns": columns,
        "stake_per_column": FUNBET_DRAW_STAKE_PER_COL,
        "total_stake": total_stake,
    }


def build_funbet_over(over_singles):
    """
    Σύστημα 2-from-X για τα καλύτερα Over.
    """
    sorted_overs = sorted(over_singles, key=lambda x: x["score"], reverse=True)
    picks = sorted_overs[:6]  # μέχρι 6

    n = len(picks)
    if n < 3:
        columns = 0
    else:
        # όλες οι δυάδες
        columns = 0
        for _ in itertools.combinations(range(n), 2):
            columns += 1

    total_stake = columns * FUNBET_OVER_STAKE_PER_COL

    return {
        "picks": picks,
        "system": f"2-from-{n}" if n >= 3 else None,
        "columns": columns,
        "stake_per_column": FUNBET_OVER_STAKE_PER_COL,
        "total_stake": total_stake,
    }


# ------------------------------------------------------
# Bankroll summary
# ------------------------------------------------------
def bankroll_summary(draw_singles, over_singles, funbet_draw, funbet_over, kelly_picks):
    draw_spent = sum(p["stake"] for p in draw_singles)
    over_spent = sum(p["stake"] for p in over_singles)
    funbet_draw_spent = funbet_draw.get("total_stake", 0) or 0
    funbet_over_spent = funbet_over.get("total_stake", 0) or 0
    kelly_spent = sum(p["stake (€)"] for p in kelly_picks)

    return [
        {
            "Wallet": "Draw Singles",
            "Before": f"{DRAW_WALLET}€",
            "After": f"{DRAW_WALLET - draw_spent:.2f}€",
            "Open Bets": f"{draw_spent:.2f}€",
        },
        {
            "Wallet": "Over Singles",
            "Before": f"{OVER_WALLET}€",
            "After": f"{OVER_WALLET - over_spent:.2f}€",
            "Open Bets": f"{over_spent:.2f}€",
        },
        {
            "Wallet": "FanBet Draw",
            "Before": f"{FANBET_DRAW_WALLET}€",
            "After": f"{FANBET_DRAW_WALLET - funbet_draw_spent:.2f}€",
            "Open Bets": f"{funbet_draw_spent:.2f}€",
        },
        {
            "Wallet": "FunBet Over",
            "Before": f"{FANBET_OVER_WALLET}€",
            "After": f"{FANBET_OVER_WALLET - funbet_over_spent:.2f}€",
            "Open Bets": f"{funbet_over_spent:.2f}€",
        },
        {
            "Wallet": "Kelly",
            "Before": f"{KELLY_WALLET}€",
            "After": f"{KELLY_WALLET - kelly_spent:.2f}€",
            "Open Bets": f"{kelly_spent:.2f}€",
        },
    ]


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    log("🎯 Running Friday Shortlist (v1)...")

    fixtures = load_thursday_fixtures()
    odds_index = build_odds_index(fixtures)

    draw_singles, over_singles, kelly_picks = generate_picks(fixtures, odds_index)
    funbet_draw = build_funbet_draw(draw_singles)
    funbet_over = build_funbet_over(over_singles)
    banks = bankroll_summary(draw_singles, over_singles, funbet_draw, funbet_over, kelly_picks)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "draw_singles": draw_singles,
        "over_singles": over_singles,
        "funbet_draw": funbet_draw,
        "funbet_over": funbet_over,
        "fraction_kelly": {"picks": kelly_picks},
        "bankroll_status": banks,
    }

    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log(f"✅ Friday shortlist report saved: {FRIDAY_REPORT_PATH}")
    log(
        f"Summary → Draw singles: {len(draw_singles)}, "
        f"Over singles: {len(over_singles)}, "
        f"Kelly picks: {len(kelly_picks)}, "
        f"FunBet Draw cols: {funbet_draw.get('columns', 0)}, "
        f"FunBet Over cols: {funbet_over.get('columns', 0)}"
    )


if __name__ == "__main__":
    main()
