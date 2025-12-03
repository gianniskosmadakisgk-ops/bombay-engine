# ==============================================================
#  BOMBAY ENGINE — THURSDAY ANALYSIS FULL v3
#  (Full engine: fixtures + team stats + fair odds + scores)
#
#  - Παίρνει fixtures από API-Football
#  - Παίρνει team statistics για κάθε ομάδα
#  - Υπολογίζει:
#       * fair_1, fair_x, fair_2, fair_over_2_5
#       * score_draw, score_over  (0–10)
#  - Φιλτράρει μόνο τις λίγκες-στόχους (TARGET_LEAGUES)
#  - Χρησιμοποιεί local cache για /teams/statistics
#  - Σώζει: logs/thursday_report_v3.json
#
#  Σημαντικό:
#  - Το season μπορεί να γίνει override από env var FOOTBALL_SEASON
#    ώστε να μπορείς να βάλεις 2024 ή 2025 χωρίς να αλλάζεις κώδικα.
# ==============================================================

import os
import json
import time
import math
from datetime import datetime, timedelta
from pathlib import Path

import requests
import yaml

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------
API_KEY = os.getenv("FOOTBALL_API_KEY")
BASE_URL = "https://v3.football.api-sports.io"

# Αν θες συγκεκριμένο season, το ορίζεις στο Render:
# FOOTBALL_SEASON = "2024" ή "2025"
FOOTBALL_SEASON_OVERRIDE = os.getenv("FOOTBALL_SEASON")

REPORT_PATH = "logs/thursday_report_v3.json"
CACHE_PATH = "logs/team_stats_cache_v3.json"

os.makedirs("logs", exist_ok=True)

# --------------------------------------------------------------
# TARGET LEAGUES (με βάση league.name του API-Football)
# --------------------------------------------------------------
TARGET_LEAGUES = {
    # Draw Engine leagues
    "Ligue 1",
    "Serie A",
    "La Liga",
    "Championship",
    "Serie B",
    "Ligue 2",
    "Liga Portugal 2",
    "Swiss Super League",

    # Over Engine leagues
    "Bundesliga",
    "Eredivisie",
    "Jupiler Pro League",
    "Superliga",
    "Allsvenskan",
    "Eliteserien",
    "Liga Portugal 1",

    # Extra για Kelly / γενική εικόνα
    "Premier League",
    "La Liga 2",
    "Bundesliga 2",
}

# Πόσες μέρες μπροστά κοιτάμε από "σήμερα"
DAYS_FORWARD = 4


# --------------------------------------------------------------
# Helper logging
# --------------------------------------------------------------
def log(msg: str):
    print(msg, flush=True)


# --------------------------------------------------------------
# Load core YAMLs (sanity check — δεν χρησιμοποιούνται άμεσα εδώ)
# --------------------------------------------------------------
def load_core_configs():
    try:
        root = Path(__file__).resolve().parent.parent  # πάμε ένα επίπεδο πάνω (src/)
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


# --------------------------------------------------------------
# Season helper
# --------------------------------------------------------------
def get_current_season(day: datetime) -> str:
    """
    API-Football: season = έτος έναρξης σεζόν (π.χ. 2024 για 2024-25).

    - Αν υπάρχει FOOTBALL_SEASON στο environment → το παίρνουμε αυτούσιο.
    - Αλλιώς:
        Ιούλιος–Δεκέμβριος  → season = current year
        Ιανουάριος–Ιούνιος  → season = previous year
    """
    if FOOTBALL_SEASON_OVERRIDE:
        log(f"ℹ️ Using FOOTBALL_SEASON override from env: {FOOTBALL_SEASON_OVERRIDE}")
        return FOOTBALL_SEASON_OVERRIDE

    if day.month >= 7:
        year = day.year
    else:
        year = day.year - 1

    season = str(year)
    log(f"ℹ️ Using inferred season based on date: {season}")
    return season


# --------------------------------------------------------------
# Cache helpers
# --------------------------------------------------------------
def load_stats_cache() -> dict:
    if not os.path.exists(CACHE_PATH):
        log("ℹ️ No existing team stats cache, starting fresh.")
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
        log(f"ℹ️ Loaded team stats cache from {CACHE_PATH} ({len(cache)} entries)")
        return cache
    except Exception as e:
        log(f"⚠️ Failed to load cache {CACHE_PATH}: {e}")
        return {}


def save_stats_cache(cache: dict):
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        log(f"💾 Team stats cache saved ({len(cache)} entries) → {CACHE_PATH}")
    except Exception as e:
        log(f"⚠️ Failed to save cache {CACHE_PATH}: {e}")


_team_stats_cache = {}  # in-memory cache


def cache_key(league_id: int, team_id: int, season: str) -> str:
    return f"{season}:{league_id}:{team_id}"


# --------------------------------------------------------------
# API-Football helpers
# --------------------------------------------------------------
def api_get(path: str, params: dict) -> list:
    if not API_KEY:
        log("❌ FOOTBALL_API_KEY is not set.")
        return []

    headers = {"x-apisports-key": API_KEY}
    url = f"{BASE_URL}{path}"

    try:
        res = requests.get(url, headers=headers, params=params, timeout=25)
    except Exception as e:
        log(f"⚠️ Request error on {path}: {e}")
        return []

    if res.status_code != 200:
        log(f"⚠️ API error {res.status_code} on {path} with params {params}")
        try:
            log(f"⚠️ Body: {res.text[:300]}")
        except Exception:
            pass
        return []

    try:
        data = res.json()
    except Exception as e:
        log(f"⚠️ JSON decode error on {path}: {e}")
        return []

    errors = data.get("errors") or data.get("error")
    if errors:
        log(f"⚠️ API errors on {path}: {errors}")

    resp = data.get("response", [])
    return resp


def fetch_fixtures(date_from: str, date_to: str, season: str) -> list:
    """
    Φέρνει ΟΛΑ τα fixtures από το API στο window
    και μετά φιλτράρει μόνο τις λίγκες-στόχους με βάση league.name.
    """
    params = {
        "season": season,
        "from": date_from,
        "to": date_to,
    }
    resp = api_get("/fixtures", params)
    log(f"📥 Raw fixtures fetched from API: {len(resp)}")

    fixtures = []
    for f in resp:
        league = f.get("league", {}) or {}
        league_name = league.get("name")
        if league_name in TARGET_LEAGUES:
            fixtures.append(f)

    log(f"🎯 Fixtures in target leagues: {len(fixtures)}")
    return fixtures


def fetch_team_stats(league_id: int, team_id: int, season: str) -> dict:
    """
    Team statistics με cache (για να μην τρώμε rate-limit).
    """
    key = cache_key(league_id, team_id, season)
    if key in _team_stats_cache:
        return _team_stats_cache[key]

    # μικρό delay όταν χτυπάμε API
    time.sleep(0.4)

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


# --------------------------------------------------------------
# Fair odds & scoring helpers
# --------------------------------------------------------------
def clamp(x, low, high):
    return max(low, min(high, x))


def compute_probabilities_and_scores(home_stats: dict, away_stats: dict):
    """
    Απλοποιημένο αλλά σταθερό μοντέλο fair πιθανότητας:

    - Παίρνουμε average goals for/against από το API-Football
    - Φτιάχνουμε ένα rating για κάθε ομάδα (attack - defence)
    - Εκτιμούμε p_home, p_draw, p_away, p_over_2_5
    - Επιστρέφουμε fair odds + scores (0–10)
    """

    try:
        gf_home = float(home_stats["goals"]["for"]["average"]["total"])
        ga_home = float(home_stats["goals"]["against"]["average"]["total"])
        gf_away = float(away_stats["goals"]["for"]["average"]["total"])
        ga_away = float(away_stats["goals"]["against"]["average"]["total"])
    except Exception:
        # fallback σε ουδέτερα values, αν λείπουν δεδομένα
        gf_home = 1.4
        ga_home = 1.1
        gf_away = 1.2
        ga_away = 1.3

    rating_home = gf_home - ga_home
    rating_away = gf_away - ga_away
    diff = rating_home - rating_away  # home - away

    # ---- Draw probability ----
    base_draw = 0.26
    balance_factor = clamp(1.0 - abs(diff), 0.0, 1.0)  # πιο ισορροπημένος = πιο πιθανό Χ
    p_draw = base_draw + 0.06 * balance_factor          # ~0.26–0.32

    # ---- Home/Away probabilities (logistic) ----
    if diff >= 0:
        r = 1 / (1 + math.exp(-diff))
    else:
        r = 1 - (1 / (1 + math.exp(diff)))

    remaining = max(0.0, 1.0 - p_draw)
    p_home = remaining * r
    p_away = remaining * (1.0 - r)

    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    # ---- Over 2.5 probability ----
    total_goals_level = gf_home + gf_away
    if total_goals_level <= 2.2:
        p_over = 0.48
    elif total_goals_level <= 2.5:
        p_over = 0.55
    elif total_goals_level <= 2.8:
        p_over = 0.62
    else:
        p_over = 0.68

    # ---- Fair odds χωρίς γκανιότα ----
    def fair_from_prob(p):
        p = clamp(p, 0.05, 0.90)
        return round(1.0 / p, 2)

    fair_1 = fair_from_prob(p_home)
    fair_x = fair_from_prob(p_draw)
    fair_2 = fair_from_prob(p_away)
    fair_over = fair_from_prob(p_over)

    # ---- Scores 0–10 ----
    score_draw_raw = 5.0 + (p_draw - 0.22) / 0.12 * 4.0  # περίπου 6–10 στις καλές περιπτώσεις
    score_draw = round(clamp(score_draw_raw, 0.0, 10.0), 2)

    score_over_raw = 5.5 + (p_over - 0.50) / 0.18 * 4.0
    score_over = round(clamp(score_over_raw, 0.0, 10.0), 2)

    return fair_1, fair_x, fair_2, fair_over, score_draw, score_over


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
def main():
    global _team_stats_cache

    if not API_KEY:
        raise RuntimeError("FOOTBALL_API_KEY is not set in environment.")

    load_core_configs()

    # φόρτωμα cache
    _team_stats_cache = load_stats_cache()

    # 1) Primary window (UTC σήμερα + 4 μέρες)
    today_utc = datetime.utcnow()
    season = get_current_season(today_utc)

    date_from = today_utc.strftime("%Y-%m-%d")
    date_to = (today_utc + timedelta(days=DAYS_FORWARD)).strftime("%Y-%m-%d")

    log(f"📅 Thursday v3 window: {date_from} → {date_to} (season {season})")

    fixtures_raw = fetch_fixtures(date_from, date_to, season)

    # 2) Αν παρ' όλα αυτά δεν βρούμε τίποτα, κάνουμε ένα fallback 7 μέρες μπροστά,
    #    απλά για να μη γυρίσουμε εντελώς άδειο report.
    if not fixtures_raw:
        fallback_from = (today_utc + timedelta(days=1)).strftime("%Y-%m-%d")
        fallback_to = (today_utc + timedelta(days=7)).strftime("%Y-%m-%d")
        log(
            f"⚠️ No fixtures in primary window. "
            f"Trying fallback window {fallback_from} → {fallback_to} (season {season})"
        )
        fixtures_raw = fetch_fixtures(fallback_from, fallback_to, season)
        date_from = fallback_from
        date_to = fallback_to

    processed = []

    for f in fixtures_raw:
        try:
            league_info = f.get("league", {}) or {}
            league_name = league_info.get("name")
            league_id = int(league_info.get("id"))

            fixture_info = f.get("fixture", {}) or {}
            kickoff_iso = fixture_info.get("date")  # ISO UTC string
            kickoff_ts = fixture_info.get("timestamp")  # UNIX timestamp

            home_info = f.get("teams", {}).get("home", {}) or {}
            away_info = f.get("teams", {}).get("away", {}) or {}

            home_team = home_info.get("name")
            away_team = away_info.get("name")
            home_id = int(home_info.get("id"))
            away_id = int(away_info.get("id"))

            match_label = f"{home_team} - {away_team}"

            # team statistics (cached)
            home_stats = fetch_team_stats(league_id, home_id, season)
            away_stats = fetch_team_stats(league_id, away_id, season)

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
                    "league_id": league_id,
                    "match": match_label,
                    "date_utc": kickoff_iso,
                    "timestamp": kickoff_ts,
                    "fair_1": fair_1,
                    "fair_x": fair_x,
                    "fair_2": fair_2,
                    "fair_over_2_5": fair_over,
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
            "season": season,
        },
        "fixtures_analyzed": len(processed),
        "fixtures": processed,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    save_stats_cache(_team_stats_cache)

    log(f"✅ Thursday v3 ready → {len(processed)} fixtures analysed.")
    log(f"📝 Saved → {REPORT_PATH}")

    if processed:
        sample = processed[:3]
        log("📌 Sample fixtures from report:")
        log(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
