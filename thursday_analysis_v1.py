import os
import json
from datetime import datetime, timedelta
from pathlib import Path

import requests
import yaml

# ======================================================
#  THURSDAY ANALYSIS v1  (Giannis Edition)
#
#  - Παίρνει fixtures & team stats από API-Football
#  - ΔΕΝ χρησιμοποιεί bookmaker odds
#  - Υπολογίζει:
#       fair_1, fair_x, fair_2, fair_over
#       score_draw, score_over
#  - Σώζει: logs/thursday_report_v1.json
# ======================================================

FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

# Κύριες λίγκες – μπορείς να αλλάξεις / επεκτείνεις
LEAGUES = [39, 140, 135, 78, 61]  # EPL, LaLiga, Serie A, Bundesliga, Ligue 1

# Από την ημέρα που τρέχει → επόμενες 4 μέρες (συμπερ. σήμερα)
DAYS_FORWARD = 4
REPORT_PATH = "logs/thursday_report_v1.json"

os.makedirs("logs", exist_ok=True)


# ------------------------------------------------------
# Helper: logging
# ------------------------------------------------------
def log(msg: str):
    print(msg, flush=True)


# ------------------------------------------------------
# Φόρτωμα core YAML (για sanity check)
# ------------------------------------------------------
def load_core_configs():
    try:
        root = Path(__file__).resolve().parent
        core_path = root / "core" / "bombay_rules_v4.yaml"
        engine_core_path = root / "engines" / "Bombay_Core_v6.yaml"
        bookmaker_path = root / "engines" / "bookmaker_logic.yaml"

        if core_path.exists():
            with open(core_path, "r", encoding="utf-8") as f:
                yaml.safe_load(f)
            log("✅ Loaded bombay_rules_v4.yaml")

        if engine_core_path.exists():
            with open(engine_core_path, "r", encoding="utf-8") as f:
                yaml.safe_load(f)
            log("✅ Loaded Bombay_Core_v6.yaml")

        if bookmaker_path.exists():
            with open(bookmaker_path, "r", encoding="utf-8") as f:
                yaml.safe_load(f)
            log("✅ Loaded bookmaker_logic.yaml")

    except Exception as e:
        log(f"⚠️ Skipped loading core configs: {e}")


# ------------------------------------------------------
# Season helper
# ------------------------------------------------------
def get_current_season(day: datetime) -> str:
    """
    API-Football χρησιμοποιεί ως season το έτος έναρξης της σεζόν.
    Π.χ. σεζόν 2025-26 → season = 2025.

    Λογική:
    - Ιανουάριος–Ιούνιος  → season = previous year
    - Ιούλιος–Δεκέμβριος → season = current year
    """
    if day.month >= 7:
        year = day.year
    else:
        year = day.year - 1
    return str(year)


# ------------------------------------------------------
# API-Football helpers
# ------------------------------------------------------
def api_get(path: str, params: dict) -> list:
    headers = {"x-apisports-key": FOOTBALL_API_KEY}
    url = f"{FOOTBALL_BASE_URL}{path}"
    try:
        res = requests.get(url, headers=headers, params=params, timeout=20)
    except Exception as e:
        log(f"⚠️ Request error on {path}: {e}")
        return []

    if res.status_code != 200:
        log(f"⚠️ API error {res.status_code} on {path} with params {params}")
        return []

    try:
        data = res.json()
    except Exception as e:
        log(f"⚠️ JSON decode error on {path}: {e}")
        return []

    return data.get("response", [])


def fetch_fixtures(date_from: str, date_to: str, season: str) -> list:
    fixtures = []
    for league_id in LEAGUES:
        params = {
            "league": league_id,
            "season": season,
            "from": date_from,
            "to": date_to,
        }
        resp = api_get("/fixtures", params)
        log(f"✅ Fixtures: league {league_id} → {len(resp)} matches")
        fixtures.extend(resp)
    return fixtures


# Cache για team statistics ώστε να μην χτυπάμε συνέχεια το ίδιο endpoint
_team_stats_cache = {}


def fetch_team_stats(league_id: int, team_id: int, season: str) -> dict:
    key = (league_id, team_id, season)
    if key in _team_stats_cache:
        return _team_stats_cache[key]

    params = {
        "league": league_id,
        "season": season,
        "team": team_id,
    }
    resp = api_get("/teams/statistics", params)
    if not resp:
        _team_stats_cache[key] = {}
        return {}

    stats = resp[0] if isinstance(resp, list) else resp
    _team_stats_cache[key] = stats
    return stats


# ------------------------------------------------------
#  Fair odds & score helpers
# ------------------------------------------------------
def clamp(x, low, high):
    return max(low, min(high, x))


def compute_probabilities_and_scores(home_stats: dict, away_stats: dict):
    """
    Απλοποιημένο μοντέλο fair πιθανότητας:
    - χρησιμοποιεί avg goals for/against
    - εκτιμά p_home, p_draw, p_away, p_over
    - επιστρέφει fair odds + scores (0–10)
    """

    try:
        gf_home = float(home_stats["goals"]["for"]["average"]["total"])
        ga_home = float(home_stats["goals"]["against"]["average"]["total"])
        gf_away = float(away_stats["goals"]["for"]["average"]["total"])
        ga_away = float(away_stats["goals"]["against"]["average"]["total"])
    except Exception:
        # fallback σε ουδέτερα values
        gf_home = 1.4
        ga_home = 1.1
        gf_away = 1.2
        ga_away = 1.3

    # Ratings (attack - defence)
    rating_home = gf_home - ga_home
    rating_away = gf_away - ga_away
    diff = rating_home - rating_away  # home - away

    # --- Draw probability ---
    base_draw = 0.26
    balance_factor = clamp(1.0 - abs(diff), 0.0, 1.0)  # ισορροπία
    p_draw = base_draw + 0.06 * balance_factor          # ~0.26–0.32

    # --- Home / Away probability ---
    import math

    if diff >= 0:
        r = 1 / (1 + math.exp(-diff))
    else:
        r = 1 - (1 / (1 + math.exp(diff)))

    remaining = max(0.0, 1.0 - p_draw)
    p_home = remaining * r
    p_away = remaining * (1.0 - r)

    # normalize
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    # --- Over 2.5 probability ---
    total_goals_level = gf_home + gf_away
    if total_goals_level <= 2.2:
        p_over = 0.48
    elif total_goals_level <= 2.5:
        p_over = 0.55
    elif total_goals_level <= 2.8:
        p_over = 0.62
    else:
        p_over = 0.68

    # --- Fair odds (χωρίς γκανιότα) ---
    def fair_from_prob(p):
        p = clamp(p, 0.05, 0.90)
        return round(1.0 / p, 2)

    fair_1 = fair_from_prob(p_home)
    fair_x = fair_from_prob(p_draw)
    fair_2 = fair_from_prob(p_away)
    fair_over = fair_from_prob(p_over)

    # --- Scores (0–10) ---
    score_draw_raw = 5.0 + (p_draw - 0.22) / 0.12 * 4.0  # ~6–10
    score_draw = round(clamp(score_draw_raw, 0.0, 10.0), 2)

    score_over_raw = 5.5 + (p_over - 0.50) / 0.18 * 4.0  # ~6–10
    score_over = round(clamp(score_over_raw, 0.0, 10.0), 2)

    return fair_1, fair_x, fair_2, fair_over, score_draw, score_over


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    if not FOOTBALL_API_KEY:
        raise RuntimeError("FOOTBALL_API_KEY is not set in environment.")

    load_core_configs()

    # 1) Primary window με βάση το server time
    today = datetime.utcnow()
    season_primary = get_current_season(today)
    date_from = today.strftime("%Y-%m-%d")
    date_to = (today + timedelta(days=DAYS_FORWARD)).strftime("%Y-%m-%d")

    log(f"📅 Primary window: {date_from} to {date_to} (season {season_primary})")
    fixtures_raw = fetch_fixtures(date_from, date_to, season_primary)

    # 2) Αν δεν βρούμε fixtures, κάνουμε fallback ένα έτος πίσω
    if not fixtures_raw:
        fallback_day = today - timedelta(days=365)
        season_fallback = get_current_season(fallback_day)
        date_from = fallback_day.strftime("%Y-%m-%d")
        date_to = (fallback_day + timedelta(days=DAYS_FORWARD)).strftime("%Y-%m-%d")
        log(
            f"⚠️ No fixtures in primary window. "
            f"Falling back to {date_from} to {date_to} (season {season_fallback})"
        )
        fixtures_raw = fetch_fixtures(date_from, date_to, season_fallback)
        season_used = season_fallback
    else:
        season_used = season_primary

    processed = []

    for f in fixtures_raw:
        try:
            league_name = f["league"]["name"]
            league_id = int(f["league"]["id"])
            home_team = f["teams"]["home"]["name"]
            away_team = f["teams"]["away"]["name"]
            home_id = int(f["teams"]["home"]["id"])
            away_id = int(f["teams"]["away"]["id"])

            match_label = f"{home_team} - {away_team}"

            # Fetch team statistics (cached)
            home_stats = fetch_team_stats(league_id, home_id, season_used)
            away_stats = fetch_team_stats(league_id, away_id, season_used)

            if not home_stats or not away_stats:
                log(f"⚠️ Missing stats for {match_label}, skipping.")
                continue

            (
                fair_1,
                fair_x,
                fair_2,
                fair_over,
                score_draw,
                score_over,
            ) = compute_probabilities_and_scores(home_stats, away_stats)

            processed.append(
                {
                    "league": league_name,
                    "match": match_label,
                    "fair_1": fair_1,
                    "fair_x": fair_x,
                    "fair_2": fair_2,
                    "fair_over": fair_over,
                    "score_draw": score_draw,
                    "score_over": score_over,
                }
            )

        except Exception as e:
            log(f"⚠️ Error processing fixture: {e}")

    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "source_window": {
            "date_from": date_from,
            "date_to": date_to,
            "season": season_used,
        },
        "fixtures_analyzed": len(processed),
        "fixtures": processed,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    log(f"✅ Thursday analysis complete — {len(processed)} fixtures analyzed.")
    log(f"📝 Report saved at {REPORT_PATH}")


if __name__ == "__main__":
    main()
