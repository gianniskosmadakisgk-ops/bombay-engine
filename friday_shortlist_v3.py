import os
import json
from datetime import datetime
import itertools
import re

import requests

# ======================================================
#  FRIDAY SHORTLIST v3  (σύμφωνο με Thursday v3)
#
#  - Διαβάζει Thursday report v3 (fair odds + probabilities)
#  - Τραβάει πραγματικές αποδόσεις από TheOddsAPI
#  - Επιλογές:
#       * Draw singles  (έως 10 picks / όλες οι λίγκες)
#       * Over singles  (έως 10 picks / όλες οι λίγκες)
#       * FanBet Draw system (3-4-5 ή 4-5-6)
#       * FunBet Over system (2-from-X)
#       * Kelly value bets (1 / X / 2 / Over 2.5)
#       * Bankroll summary
# ======================================================

THURSDAY_REPORT_PATH = "logs/thursday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v3.json"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

os.makedirs("logs", exist_ok=True)

# ---------------------- BANKROLLS ----------------------
DRAW_WALLET = 300
OVER_WALLET = 300
FANBET_DRAW_WALLET = 100
FANBET_OVER_WALLET = 100
KELLY_WALLET = 300.0       # αρχικό Kelly κεφάλαιο (reference)

# ---------------------- LIMITS -------------------------
MAX_DRAW_PICKS = 10
MAX_OVER_PICKS = 10

FUNBET_DRAW_STAKE_PER_COL = 2.0
FUNBET_OVER_STAKE_PER_COL = 4.0

# ---------------------- THRESHOLDS ---------------------
DRAW_MIN_SCORE = 7.5
DRAW_MIN_ODDS = 2.70

OVER_MIN_SCORE = 7.5
OVER_MIN_FAIR = 1.70

# auto-play over rules (για “τέρατα” σκορ)
OVER_AUTO_SCORE = 9.0
OVER_NEG_EDGE_LIMIT = -0.10   # δεχόμαστε μέχρι -10% εις βάρος μας

# Kelly rules
KELLY_VALUE_THRESHOLD = 0.15      # +15% min edge vs fair
KELLY_FRACTION = 0.40             # 40% του πλήρους Kelly
KELLY_MIN_PROB = 0.25             # min probability 25% (fair <= 4.00)
KELLY_MAX_EXPOSURE_PCT = 0.35     # 35% του αρχικού bank → max ~105€

# bonus προτεραιότητας για “αγαπημένες” λίγκες
DRAW_LEAGUE_BONUS = 0.20
OVER_LEAGUE_BONUS = 0.20

# ---------------------- LEAGUES ------------------------
# Λίγκες όπου ο Draw / Over engine είναι πιο αξιόπιστος
DRAW_LEAGUES = {
    "Ligue 1",
    "Serie A",
    "La Liga",
    "Championship",
    "Serie B",
    "Ligue 2",
    "Liga Portugal 2",
    "Swiss Super League",
}

OVER_LEAGUES = {
    "Bundesliga",
    "Eredivisie",
    "Jupiler Pro League",
    "Superliga",
    "Allsvenskan",
    "Eliteserien",
    "Swiss Super League",
    "Liga Portugal 1",
}

# league name -> TheOddsAPI sport_key
LEAGUE_TO_SPORT = {
    # Αγγλία
    "Premier League": "soccer_epl",
    "Championship": "soccer_efl_champ",

    # Ισπανία
    "La Liga": "soccer_spain_la_liga",
    "La Liga 2": "soccer_spain_segunda_division",

    # Ιταλία
    "Serie A": "soccer_italy_serie_a",
    "Serie B": "soccer_italy_serie_b",

    # Γερμανία
    "Bundesliga": "soccer_germany_bundesliga",
    "Bundesliga 2": "soccer_germany_bundesliga2",

    # Γαλλία
    "Ligue 1": "soccer_france_ligue_one",
    "Ligue 2": "soccer_france_ligue_two",

    # Πορτογαλία
    "Liga Portugal 1": "soccer_portugal_primeira_liga",
    # Liga Portugal 2 δεν υπάρχει στο TheOddsAPI → fair only

    # Ελβετία
    "Swiss Super League": "soccer_switzerland_superleague",

    # Ολλανδία
    "Eredivisie": "soccer_netherlands_eredivisie",

    # Βέλγιο
    "Jupiler Pro League": "soccer_belgium_first_div",

    # Δανία
    "Superliga": "soccer_denmark_superliga",

    # Σουηδία
    "Allsvenskan": "soccer_sweden_allsvenskan",

    # Νορβηγία
    "Eliteserien": "soccer_norway_eliteserien",
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
    log(f"Loaded {len(fixtures)} fixtures from Thursday report v3.")
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
        res = requests.get(url, params=params, timeout=25)
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
    Normalizer για να ταιριάζουμε ονόματα ομάδων
    μεταξύ API-Football και TheOddsAPI.
    """
    if not name:
        return ""
    s = name.lower()
    s = re.sub(r"\b(fc|cf|afc|cfc|ac|sc|bk)\b", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_odds_index(fixtures):
    """
    Χτίζει index:
        (norm_home, norm_away) -> odds dict
    μόνο για λίγκες που υποστηρίζει το TheOddsAPI.
    """
    leagues_used = sorted(
        {f.get("league") for f in fixtures if f.get("league") in LEAGUE_TO_SPORT}
    )
    log(f"Leagues in Thursday report (with odds support): {leagues_used}")

    odds_index = {}
    total_events = 0

    for league_name in leagues_used:
        sport_key = LEAGUE_TO_SPORT[league_name]
        events = api_get_odds(sport_key)
        log(f"Fetched {len(events)} odds events for {league_name} ({sport_key})")
        total_events += len(events)

        for ev in events:
            home_raw = ev.get("home_team", "")
            away_raw = ev.get("away_team", "")
            home = normalize_team(home_raw)
            away = normalize_team(away_raw)
            if not home or not away:
                continue

            best_home = None
            best_away = None
            best_draw = None
            best_over = None

            for b in ev.get("bookmakers", []):
                for m in b.get("markets", []):
                    key = m.get("key")
                    if key == "h2h":
                        for o in m.get("outcomes", []):
                            name = o.get("name", "")
                            name_norm = normalize_team(name)
                            price = float(o.get("price", 0) or 0)
                            if price <= 0:
                                continue

                            if name_norm == home:
                                if best_home is None or price > best_home:
                                    best_home = price
                            elif name_norm == away:
                                if best_away is None or price > best_away:
                                    best_away = price
                            elif name.lower() == "draw":
                                if best_draw is None or price > best_draw:
                                    best_draw = price

                    elif key == "totals":
                        for o in m.get("outcomes", []):
                            name = o.get("name", "").lower()
                            price = float(o.get("price", 0) or 0)
                            if price <= 0:
                                continue
                            if "over" in name and "2.5" in name:
                                if best_over is None or price > best_over:
                                    best_over = price

            odds_index[(home, away)] = {
                "odds_home": best_home,
                "odds_draw": best_draw,
                "odds_away": best_away,
                "odds_over_2_5": best_over,
            }

    log(
        f"Built odds index for {len(odds_index)} (home, away) pairs. "
        f"Total events fetched: {total_events}"
    )
    return odds_index


# ------------------------------------------------------
# Helpers για scores
# ------------------------------------------------------
def prob_to_score(p: float) -> float:
    """
    Μετατρέπει probability (0–1) σε score 1–10 (όπως στο prompt του GPT).
    """
    try:
        p = float(p)
    except (TypeError, ValueError):
        return 0.0
    s = p * 10.0
    s = max(1.0, min(10.0, s))
    return round(s, 2)


# ------------------------------------------------------
# Pick generators
# ------------------------------------------------------
def flat_stake(score: float) -> int:
    """
    8.5+ → 20€
    7.5–8.49 → 15€
    <7.5  →  0€ (skip)
    """
    if score >= 8.5:
        return 20
    elif score >= 7.5:
        return 15
    return 0


def generate_picks(fixtures, odds_index):
    draw_singles = []
    over_singles = []
    kelly_candidates = []

    matched_count = 0

    for f in fixtures:
        league = f.get("league", "")
        home_name = f.get("home", "")
        away_name = f.get("away", "")

        if not home_name or not away_name:
            continue

        match_label = f"{home_name} - {away_name}"

        fair_1 = f.get("fair_1")
        fair_x = f.get("fair_x")
        fair_2 = f.get("fair_2")
        fair_over = f.get("fair_over_2_5")

        draw_prob = f.get("draw_prob")
        over_prob = f.get("over_2_5_prob")

        score_draw = prob_to_score(draw_prob)
        score_over = prob_to_score(over_prob)

        home_norm = normalize_team(home_name)
        away_norm = normalize_team(away_name)

        odds = odds_index.get((home_norm, away_norm)) or odds_index.get((away_norm, home_norm))
        if odds:
            matched_count += 1
        else:
            odds = {}

        odds_home = odds.get("odds_home")
        odds_x = odds.get("odds_draw")
        odds_away = odds.get("odds_away")
        odds_over = odds.get("odds_over_2_5")

        # -------------- DRAW SINGLES  (όλες οι λίγκες) --------------
        if fair_x and score_draw >= DRAW_MIN_SCORE:

            if odds_x:
                market_odds_x = float(odds_x)
                diff_x = (market_odds_x - fair_x) / fair_x
                diff_label = f"{diff_x:+.0%}"
                value_raw = diff_x
                odds_source = "market"
            else:
                market_odds_x = float(fair_x)
                diff_label = "—"
                value_raw = 0.0
                odds_source = "model"

            if market_odds_x >= DRAW_MIN_ODDS:
                stake = flat_stake(score_draw)
                if stake > 0:
                    priority = score_draw + (DRAW_LEAGUE_BONUS if league in DRAW_LEAGUES else 0.0)
                    draw_singles.append({
                        "match": match_label,
                        "league": league,
                        "odds": round(market_odds_x, 2),
                        "fair": round(fair_x, 2),
                        "diff": diff_label,
                        "value_raw": round(value_raw, 4),
                        "score": round(score_draw, 2),
                        "stake": stake,
                        "wallet": "Draw Singles",
                        "odds_source": odds_source,
                        "priority": priority,
                    })

        # -------------- OVER SINGLES  (όλες οι λίγκες) --------------
        if fair_over and score_over >= OVER_MIN_SCORE:

            if odds_over:
                market_odds_over = float(odds_over)
                diff_over = (market_odds_over - fair_over) / fair_over
                diff_label = f"{diff_over:+.0%}"
                value_raw = diff_over
                odds_source = "market"
            else:
                market_odds_over = float(fair_over)
                diff_label = "—"
                value_raw = 0.0
                odds_source = "model"

            # βασικός κανόνας value (παίζουμε μόνο αν >= fair)
            edge_ok = value_raw >= 0.0

            # special rule για σκορ ≥ 9:
            # παίζουμε οπωσδήποτε αν η απόδοση είναι τουλάχιστον 1.70
            # και το value δεν είναι χειρότερο από -10%
            auto_over_monster = (
                score_over >= OVER_AUTO_SCORE
                and market_odds_over >= OVER_MIN_FAIR
                and value_raw >= OVER_NEG_EDGE_LIMIT
            )

            if market_odds_over >= OVER_MIN_FAIR and (edge_ok or auto_over_monster):
                stake = flat_stake(score_over)
                if stake > 0:
                    priority = score_over + (OVER_LEAGUE_BONUS if league in OVER_LEAGUES else 0.0)
                    over_singles.append({
                        "match": match_label,
                        "league": league,
                        "odds": round(market_odds_over, 2),
                        "fair": round(fair_over, 2),
                        "diff": diff_label,
                        "value_raw": round(value_raw, 4),
                        "score": round(score_over, 2),
                        "stake": stake,
                        "wallet": "Over Singles",
                        "odds_source": odds_source,
                        "priority": priority,
                    })

        # -------------- KELLY CANDIDATES (1 / X / 2 / Over 2.5) --------------
        def add_kelly_candidate(market_label, fair, offered):
            if not fair or not offered:
                return

            fair = float(fair)
            offered = float(offered)
            if fair <= 1.01 or offered <= 1.01:
                return

            # probability από fair odds
            p = 1.0 / fair
            if p < KELLY_MIN_PROB:
                return

            diff = (offered - fair) / fair
            if diff < KELLY_VALUE_THRESHOLD:
                return

            b = offered - 1.0
            q = 1.0 - p
            if b <= 0:
                return

            full_kelly_fraction = (b * p - q) / b
            if full_kelly_fraction <= 0:
                return

            raw_stake = full_kelly_fraction * KELLY_FRACTION * KELLY_WALLET
            if raw_stake <= 0:
                return

            kelly_candidates.append({
                "match": match_label,
                "league": league,
                "market": market_label,
                "fair": fair,
                "offered": offered,
                "edge": diff,
                "prob": p,
                "stake_raw": raw_stake,
            })

        # Kelly μόνο όταν έχουμε πραγματικά odds:
        if odds_home and fair_1:
            add_kelly_candidate("Home", fair_1, odds_home)
        if odds_x and fair_x:
            add_kelly_candidate("Draw", fair_x, odds_x)
        if odds_away and fair_2:
            add_kelly_candidate("Away", fair_2, odds_away)
        if odds_over and fair_over:
            add_kelly_candidate("Over 2.5", fair_over, odds_over)

    # ------- LIMIT & SORT DRAWS / OVERS (έως 10 picks) -------
    draw_singles = sorted(
        draw_singles,
        key=lambda x: x["priority"],
        reverse=True
    )[:MAX_DRAW_PICKS]
    for d in draw_singles:
        d.pop("priority", None)

    over_singles = sorted(
        over_singles,
        key=lambda x: x["priority"],
        reverse=True
    )[:MAX_OVER_PICKS]
    for o in over_singles:
        o.pop("priority", None)

    # ------- KELLY: scale ώστε συνολικό exposure <= 35% -------
    kelly_candidates = sorted(
        kelly_candidates,
        key=lambda x: x["stake_raw"],
        reverse=True
    )[:10]

    total_raw = sum(p["stake_raw"] for p in kelly_candidates)
    max_exposure = KELLY_WALLET * KELLY_MAX_EXPOSURE_PCT

    if total_raw > 0 and total_raw > max_exposure:
        scale = max_exposure / total_raw
    else:
        scale = 1.0

    kelly_picks = []
    for c in kelly_candidates:
        stake_final = round(c["stake_raw"] * scale, 2)
        kelly_picks.append({
            "match": c["match"],
            "league": c["league"],
            "market": c["market"],
            "fair": round(c["fair"], 2),
            "offered": round(c["offered"], 2),
            "diff": f"{c['edge']:+.0%}",
            "kelly%": f"{KELLY_FRACTION * 100:.0f}%",
            "stake (€)": stake_final,
        })

    log(f"Matched odds for {matched_count} / {len(fixtures)} fixtures.")
    log(
        f"Draw singles: {len(draw_singles)}, "
        f"Over singles: {len(over_singles)}, "
        f"Kelly picks: {len(kelly_picks)}"
    )

    return draw_singles, over_singles, kelly_picks


# ------------------------------------------------------
# FunBet systems
# ------------------------------------------------------
def build_funbet_draw(draw_singles):
    """
    Παίρνει τις καλύτερες ισοπαλίες και φτιάχνει σύστημα 3-4-5 ή 4-5-6.
    """
    sorted_draws = sorted(draw_singles, key=lambda x: x["score"], reverse=True)
    picks = sorted_draws[:6]  # max 6

    n = len(picks)
    system = None
    columns = 0

    if n >= 6:
        sizes = [4, 5, 6]
        system = "4-5-6"
    elif n == 5:
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
        system = None
    else:
        columns = 0
        for _ in itertools.combinations(range(n), 2):
            columns += 1
        system = f"2-from-{n}"

    total_stake = columns * FUNBET_OVER_STAKE_PER_COL

    return {
        "picks": picks,
        "system": system,
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
            "Before": f"{KELLY_WALLET:.0f}€",
            "After": f"{KELLY_WALLET - kelly_spent:.2f}€",
            "Open Bets": f"{kelly_spent:.2f}€",
        },
    ]


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    log("🎯 Running Friday Shortlist (v3 – Thursday v3 compatible)...")

    fixtures = load_thursday_fixtures()
    odds_index = build_odds_index(fixtures)

    draw_singles, over_singles, kelly_picks = generate_picks(fixtures, odds_index)
    funbet_draw = build_funbet_draw(draw_singles)
    funbet_over = build_funbet_over(over_singles)
    banks = bankroll_summary(draw_singles, over_singles, funbet_draw, funbet_over, kelly_picks)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "meta": {
            "fixtures_total": len(fixtures),
            "draw_singles": len(draw_singles),
            "over_singles": len(over_singles),
            "kelly_picks": len(kelly_picks),
            "funbet_draw_cols": funbet_draw.get("columns", 0),
            "funbet_over_cols": funbet_over.get("columns", 0),
        },
        "draw_singles": draw_singles,
        "over_singles": over_singles,
        "funbet_draw": funbet_draw,
        "funbet_over": funbet_over,
        "kelly": {"picks": kelly_picks},
        "bankroll_status": banks,
    }

    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log(f"✅ Friday shortlist report saved: {FRIDAY_REPORT_PATH}")
    log(
        "Summary → Draw singles: "
        f"{len(draw_singles)}, Over singles: {len(over_singles)}, "
        f"Kelly picks: {len(kelly_picks)}, "
        f"FunBet Draw cols: {funbet_draw.get('columns', 0)}, "
        f"FunBet Over cols: {funbet_over.get('columns', 0)}"
    )


if __name__ == "__main__":
    main()
