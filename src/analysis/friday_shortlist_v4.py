import os
import json
import itertools
from datetime import datetime
import re
import requests

# ==============================================================================
#  FRIDAY SHORTLIST V4 — PRODUCTION VERSION (UNITS-BASED STAKING)
#  - Loads Thursday report (v3)
#  - Pulls offered odds from TheOddsAPI
#  - Builds:
#       * Draw Singles (flat stake)
#       * Over Singles (standard / premium / monster)
#       * FanBet Draw systems
#       * FanBet Over systems
#       * Kelly value bets
#  - Saves clean JSON report for UI / Custom GPT
# ==============================================================================

THURSDAY_REPORT_PATH = "logs/thursday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v4.json"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

# ------------------------------------------------------------------------------
# BANKROLLS (IN UNITS)
# ------------------------------------------------------------------------------
BANKROLL_DRAW = 1000
BANKROLL_OVER = 1000
BANKROLL_FUN_DRAW = 300
BANKROLL_FUN_OVER = 300
BANKROLL_KELLY = 600

UNIT = 1.0

# Draw Singles stake: flat 20 units
DRAW_STAKE_FLAT = 20 * UNIT

# Over Singles stakes:
OVER_STAKE_STANDARD = 4 * UNIT
OVER_STAKE_PREMIUM = 8 * UNIT
OVER_STAKE_MONSTER = 12 * UNIT

# Stake per column for all FunBets
FUNBET_STAKE_PER_COLUMN = 1 * UNIT

# ------------------------------------------------------------------------------
# LEAGUE PRIORITIES (tie-break / bonus, NOT hard filters)
# ------------------------------------------------------------------------------
DRAW_PRIORITY_LEAGUES = {
    "Ligue 1",
    "Serie A",
    "La Liga",
    "Championship",
    "Serie B",
    "Ligue 2",
    "Liga Portugal 2",
    "Swiss Super League",
}

OVER_PRIORITY_LEAGUES = {
    "Bundesliga",
    "Eredivisie",
    "Jupiler Pro League",
    "Superliga",
    "Allsvenskan",
    "Eliteserien",
    "Swiss Super League",
    "Liga Portugal 1",
}

# ------------------------------------------------------------------------------
# ODDS SUPPORT MAP
# ------------------------------------------------------------------------------
LEAGUE_TO_SPORT = {
    "Premier League": "soccer_epl",
    "Championship": "soccer_efl_champ",
    "La Liga": "soccer_spain_la_liga",
    "La Liga 2": "soccer_spain_segunda_division",
    "Serie A": "soccer_italy_serie_a",
    "Serie B": "soccer_italy_serie_b",
    "Bundesliga": "soccer_germany_bundesliga",
    "Bundesliga 2": "soccer_germany_bundesliga2",
    "Ligue 1": "soccer_france_ligue_one",
    "Ligue 2": "soccer_france_ligue_two",
    "Liga Portugal 1": "soccer_portugal_primeira_liga",
    "Swiss Super League": "soccer_switzerland_superleague",
    "Eredivisie": "soccer_netherlands_eredivisie",
    "Jupiler Pro League": "soccer_belgium_first_div",
    "Superliga": "soccer_denmark_superliga",
    "Allsvenskan": "soccer_sweden_allsvenskan",
    "Eliteserien": "soccer_norway_eliteserien",
    "Argentina Primera": "soccer_argentina_primera_division",
    "Brazil Serie A": "soccer_brazil_serie_a",  # future use
}

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------
def log(msg: str):
    print(msg, flush=True)


def normalize_team(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\b(fc|cf|afc|cfc|ac|sc|bk)\b", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# ------------------------------------------------------------------------------
# Load Thursday fixtures
# ------------------------------------------------------------------------------
def load_thursday():
    if not os.path.exists(THURSDAY_REPORT_PATH):
        raise FileNotFoundError(f"Thursday report missing: {THURSDAY_REPORT_PATH}")

    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f).get("fixtures", [])


# ------------------------------------------------------------------------------
# ODDS API
# ------------------------------------------------------------------------------
def get_odds_for_league(sport_key: str):
    if not ODDS_API_KEY:
        log("⚠️ Missing ODDS_API_KEY → skipping odds for this league")
        return []

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
    }

    try:
        res = requests.get(
            f"{ODDS_BASE_URL}/{sport_key}/odds",
            params=params,
            timeout=20,
        )
    except Exception as e:
        log(f"
