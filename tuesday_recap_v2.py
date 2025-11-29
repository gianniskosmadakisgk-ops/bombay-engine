import os
import json
from datetime import datetime
from itertools import combinations

import requests

# ======================================================
#  TUESDAY RECAP v2  (Giannis Edition)
#
#  - Διαβάζει:
#       * logs/thursday_report_v1.json  (για window + leagues)
#       * logs/friday_shortlist_v2.json (όλα τα bets)
#  - Τραβάει τελικά αποτελέσματα από API-Football
#  - Κλείνει:
#       * Draw singles
#       * Over singles
#       * FunBet Draw system
#       * FunBet Over system
#       * Kelly (Home / Draw / Away / Over 2.5)
#  - Υπολογίζει P&L ανά πορτοφόλι + συνολικά
#  - Σώζει: logs/tuesday_recap_v2.json
# ======================================================

FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

THURSDAY_REPORT_PATH = "logs/thursday_report_v1.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v2.json"
TUESDAY_RECAP_PATH = "logs/tuesday_recap_v2.json"

# Ίδια wallets με Friday
DRAW_WALLET = 400
OVER_WALLET = 300
FANBET_DRAW_WALLET = 100
FANBET_OVER_WALLET = 100
KELLY_WALLET = 300

# mapping League name -> API-Football league id
LEAGUE_NAME_TO_ID = {
    "Premier League": 39,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61,
}

FINISHED_STATUSES = {"FT", "AET", "PEN"}


def log(msg: str):
    print(msg, flush=True)


# ------------------------------------------------------
# Helpers: API-Football
# ------------------------------------------------------
def api_get(path: str, params: dict) -> list:
    headers = {"x-apisports-key": FOOTBALL_API_KEY}
    url = f"{FOOTBALL_BASE_URL}{path}"
    try:
        res = requests.get(url, headers=headers, params=params, timeout=25)
    except Exception as e:
        log(f"⚠️ Request error on {path}: {e}")
        return []

    if res.status_code != 200:
        log(f"⚠️ API error {res.status_code} on {path} with params {params}")
        try:
            log(res.text[:300])
        except Exception:
            pass
        return []

    try:
        data = res.json()
    except Exception as e:
        log(f"⚠️ JSON decode error on {path}: {e}")
        return []

    return data.get("response", [])


def normalize_team(name: str) -> str:
    if not name:
        return ""
    s = name.lower()
    import re

    s = re.sub(r"\b(fc|cf|afc|cfc|ac|sc|bk)\b", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ------------------------------------------------------
# Load input reports
# ------------------------------------------------------
def load_thursday_report():
    if not os.path.exists(THURSDAY_REPORT_PATH):
        raise FileNotFoundError(f"Thursday report not found: {THURSDAY_REPORT_PATH}")
    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_friday_report():
    if not os.path.exists(FRIDAY_REPORT_PATH):
        raise FileNotFoundError(f"Friday shortlist not found: {FRIDAY_REPORT_PATH}")
    with open(FRIDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------
# Fetch final results για τα fixtures του παραθύρου
# ------------------------------------------------------
def build_results_index(thursday_data, leagues_needed):
    window = thursday_data.get("source_window", {})
    date_from = window.get("date_from")
    date_to = window.get("date_to")
    season = window.get("season")

    log(f"📅 Tuesday Recap window: {date_from} to {date_to} (season {season})")

    results_index = {}
    total_fixtures = 0

    for league_name in sorted(leagues_needed):
        league_id = LEAGUE_NAME_TO_ID.get(league_name)
        if not league_id:
            log(f"⚠️ League {league_name} has no league_id mapping, skipping for results.")
            continue

        params = {
            "league": league_id,
            "season": season,
            "from": date_from,
            "to": date_to,
        }
        resp = api_get("/fixtures", params)
        log(f"Results: league {league_name} ({league_id}) → {len(resp)} fixtures")
        total_fixtures += len(resp)

        for fx in resp:
            try:
                l_name = fx["league"]["name"]
                home_name = fx["teams"]["home"]["name"]
                away_name = fx["teams"]["away"]["name"]
                status = fx["fixture"]["status"]["short"]
                g_home = fx["goals"]["home"]
                g_away = fx["goals"]["away"]
            except Exception:
                continue

            home_norm = normalize_team(home_name)
            away_norm = normalize_team(away_name)

            if g_home is None or g_away is None or status not in FINISHED_STATUSES:
                outcome = "pending"
                total_goals = None
            else:
                total_goals = g_home + g_away
                if g_home > g_away:
                    outcome = "Home"
                elif g_home < g_away:
                    outcome = "Away"
                else:
                    outcome = "Draw"

            key = (l_name, home_norm, away_norm)
            results_index[key] = {
                "league": l_name,
                "home": home_name,
                "away": away_name,
                "status": status,
                "outcome": outcome,
                "goals_home": g_home,
                "goals_away": g_away,
                "total_goals": total_goals,
            }

    log(f"Built results index for {len(results_index)} fixtures (total fetched: {total_fixtures})")
    return results_index, window


def lookup_result(results_index, league, match_label):
    try:
        home_name, away_name = [x.strip() for x in match_label.split("-")]
    except ValueError:
        return None

    home_norm = normalize_team(home_name)
    away_norm = normalize_team(away_name)
    return results_index.get((league, home_norm, away_norm)) or results_index.get(
        (league, away_norm, home_norm)
    )


# ------------------------------------------------------
# Settlement helpers
# ------------------------------------------------------
def settle_single_draw(pick, result):
    stake = float(pick["stake"])
    odds = float(pick["odds"])
    if not result or result["outcome"] == "pending":
        return {**pick, "result": "unmatched", "payout": 0.0, "profit": -0.0}

    if result["outcome"] == "Draw":
        payout = round(stake * odds, 2)
        profit = round(payout - stake, 2)
        return {**pick, "result": "win", "payout": payout, "profit": profit}
    else:
        return {**pick, "result": "loss", "payout": 0.0, "profit": -stake}


def settle_single_over(pick, result):
    stake = float(pick["stake"])
    odds = float(pick["odds"])
    if not result or result["outcome"] == "pending" or result["total_goals"] is None:
        return {**pick, "result": "unmatched", "payout": 0.0, "profit": -0.0}

    if result["total_goals"] > 2.5:
        payout = round(stake * odds, 2)
        profit = round(payout - stake, 2)
        return {**pick, "result": "win", "payout": payout, "profit": profit}
    else:
        return {**pick, "result": "loss", "payout": 0.0,
