import os
import json
from datetime import datetime
from pathlib import Path
import itertools
import re

import requests

# ======================================================
#  FRIDAY SHORTLIST v2  (Giannis Edition)
#
#  - Διαβάζει το Thursday report (fair odds + scores)
#  - Τραβάει ΠΡΑΓΜΑΤΙΚΕΣ αποδόσεις από TheOddsAPI
#  - Φτιάχνει:
#       * Draw singles
#       * Over singles
#       * FunBet Draw system
#       * FunBet Over system
#       * Kelly value bets (1 / X / 2 / Over 2.5)
#       * Bankroll summary
#  - Fallback:
#       * Αν δεν βρεθούν odds για έναν αγώνα, τα singles
#         μπορούν να βγουν με fair_odd (χωρίς Kelly)
#  - Σώζει: logs/friday_shortlist_v2.json
# ======================================================

THURSDAY_REPORT_PATH = "logs/thursday_report_v1.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v2.json"

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

KELLY_VALUE_THRESHOLD = 0.15
KELLY_FRACTION = 0.40

FUNBET_DRAW_STAKE_PER_COL = 3.0
FUNBET_OVER_STAKE_PER_COL = 4.0

# Λίγκες σύμφωνα με το blueprint
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
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",

    # 🔥 FIXED: Correct TheOddsAPI key
    "Ligue 1": "soccer_france_ligue_1",
}

# ------------------------------------------------------
def log(msg: str):
    print(msg, flush=True)

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
def api_get_odds(sport_key: str):
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
    if not name:
        return ""
    s = name.lower()
    s = re.sub(r"\b(fc|cf|afc|cfc|ac|sc|bk)\b", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# ------------------------------------------------------
def build_odds_index(fixtures):
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
                                best_home = max(best_home or 0, price)
                            elif name_norm == away:
                                best_away = max(best_away or 0, price)
                            elif name.lower() == "draw":
                                best_draw = max(best_draw or 0, price)

                    elif key == "totals":
                        for o in m.get("outcomes", []):
                            name = o.get("name", "").lower()
                            price = float(o.get("price", 0) or 0)
                            if price > 0 and "over" in name and "2.5" in name:
                                best_over = max(best_over or 0, price)

            odds_index[(home, away)] = {
                "odds_home": best_home,
                "odds_draw": best_draw,
                "odds_away": best_away,
                "odds_over_2_5": best_over,
            }

    log(f"Built odds index for {len(odds_index)} (home, away) pairs. Total events fetched: {total_events}")
    return odds_index

# ------------------------------------------------------
def flat_stake(score: float) -> int:
    if score >= 8.5:
        return 20
    elif score >= 7.5:
        return 15
    return 0

# ------------------------------------------------------
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
        score_draw = float(f.get("score_draw", 0.0) or 0.0)
        score_over = float(f.get("score_over", 0.0) or 0.0)

        try:
            home_name, away_name = [x.strip() for x in match_label.split("-")]
        except ValueError:
            continue

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

        # DRAW SINGLES
        if league in DRAW_LEAGUES and fair_x and score_draw >= DRAW_MIN_SCORE:

            if odds_x:
                market_odds_x = float(odds_x)
                diff_x = (market_odds_x - fair_x) / fair_x
                diff_label = f"{diff_x:+.0%}"
                value_raw = diff_x
                odds_source = "market"
            else:
                market_odds_x = float(fair_x)
                diff_label = "n/a"
                value_raw = 0.0
                odds_source = "model"

            if market_odds_x >= DRAW_MIN_ODDS:
                stake = flat_stake(score_draw)
                if stake > 0:
                    draw_singles.append({
                        "match": match_label,
                        "league": league,
                        "odds": round(market_odds_x, 2),
                        "fair": round(fair_x, 2),
                        "diff": diff_label,
                        "value_raw": round(value_raw, 4),
                        "
