# ================================================================
#  BOMBAY ENGINE — THURSDAY ANALYSIS FULL v3
#  (Complete script — έτοιμο για Render)
#
#  - Τραβάει fixtures ανά λίγκα από API-FOOTBALL
#  - Χρησιμοποιεί FOOTBALL_SEASON από environment
#  - Χτίζει full model για:
#       * p_home, p_draw, p_away
#       * p_over_2_5, p_under_2_5
#  - Βασισμένο σε team statistics + standings
#  - Caching για /teams/statistics και /standings
#  - Υποστηρίζει Draw Engine / Over Engine ανά λίγκα
#  - Σώζει JSON report → logs/thursday_report_v3.json
# ================================================================

import os
import json
import time
import math
from datetime import datetime, timedelta
from pathlib import Path

import requests
import yaml

API_KEY = os.getenv("FOOTBALL_API_KEY")
API_URL = "https://v3.football.api-sports.io"

# -------------------------------------------------
#  Season από environment
# -------------------------------------------------
FOOTBALL_SEASON_ENV = os.getenv("FOOTBALL_SEASON")


def resolve_season() -> str:
    """
    Αν υπάρχει FOOTBALL_SEASON στο περιβάλλον → το χρησιμοποιούμε.
    Αλλιώς κάνουμε classic ευρωπαϊκή λογική:
      - Ιούλιος–Δεκέμβριος → season = current year
      - Ιανουάριος–Ιούνιος → season = previous year
    """
    if FOOTBALL_SEASON_ENV:
        return FOOTBALL_SEASON_ENV

    today = datetime.utcnow()
    if today.month >= 7:
        year = today.year
    else:
        year = today.year - 1
    return str(year)


SEASON = resolve_season()

# -------------------------------------------------
#  Paths
# -------------------------------------------------
REPORT_PATH = "logs/thursday_report_v3.json"
TEAM_CACHE_PATH = "logs/team_stats_cache_v3.json"
STANDINGS_CACHE_PATH = "logs/standings_cache_v3.json"

os.makedirs("logs", exist_ok=True)

# -------------------------------------------------
#  ΛΙΓΚΕΣ & ΤΥΠΟΙ ENGINE
# -------------------------------------------------
# Draw Engine leagues
DRAW_LEAGUES = {
    61: "Ligue 1",          # France
    135: "Serie A",         # Italy
    140: "La Liga",         # Spain
    40: "Championship",     # England
    136: "Serie B",         # Italy
    62: "Ligue 2",          # France
    95: "Liga Portugal 2",  # Portugal 2
    207: "Swiss Super League",  # Shared με Over
}

# Over Engine leagues
OVER_LEAGUES = {
    78: "Bundesliga",          # Germany
    88: "Eredivisie",          # Netherlands
    144: "Jupiler Pro League", # Belgium
    271: "Superliga",          # Denmark (Superliga)  ⚠️ έλεγξε ότι το ID ταιριάζει
    113: "Allsvenskan",        # Sweden               ⚠️ "
    103: "Eliteserien",        # Norway               ⚠️ "
    207: "Swiss Super League", # shared
    94: "Liga Portugal 1",     # Portugal 1
}

# Ενιαίο mapping: league_id → {name, engines}
LEAGUES = {}
for lid, name in DRAW_LEAGUES.items():
    LEAGUES.setdefault(lid, {"name": name, "engines": set()})
    LEAGUES[lid]["engines"].add("draw")

for lid, name in OVER_LEAGUES.items():
    LEAGUES.setdefault(lid, {"name": name, "engines": set()})
    LEAGUES[lid]["engines"].add("over")

# -------------------------------------------------
#  Χρήσιμο logging
# -------------------------------------------------
def log(msg: str):
    print(msg, flush=True)


# -------------------------------------------------
#  Load core YAMLs (sanity only)
# -------------------------------------------------
def load_core_configs():
    try:
        root = Path(__file__).resolve().parent
        core_path = root / "core" / "bombay_rules_v4.yaml"
        engine_core_path = root / "engines" / "Bombay_Core_v6.yaml"
        bookmaker_path = root / "engines" / "bookmaker_logic.yaml"

        if core_path.exists():
            yaml.safe_load(core_path.read_text(encoding="utf-8"))
            log("✅ Loaded bombay_rules_v4.yaml")

        if engine_core_path.exists():
            yaml.safe_load(engine_core_path.read_text(encoding="utf-8"))
            log("✅ Loaded Bombay_Core_v6.yaml")

        if bookmaker_path.exists():
            yaml.safe_load(bookmaker_path.read_text(encoding="utf-8"))
            log("✅ Loaded bookmaker_logic.yaml")

    except Exception as e:
        log(f"⚠️ Skipped loading core configs: {e}")


# -------------------------------------------------
#  Simple helpers
# -------------------------------------------------
def clamp(x, low, high):
    return max(low, min(high, x))


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def get_nested(d: dict, path, default=0.0):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return safe_float(cur, default=default)


# -------------------------------------------------
#  Cache helpers
# -------------------------------------------------
def load_json_cache(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log(f"⚠️ Failed to load cache {path}: {e}")
        return {}


def save_json_cache(path: str, data: dict):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        log(f"💾 Saved cache {path} ({len(data)} entries)")
    except Exception as e:
        log(f"⚠️ Failed to save cache {path}: {e}")


TEAM_STATS_CACHE = {}
STANDINGS_CACHE = {}


def team_cache_key(league_id: int, team_id: int, season: str) -> str:
    return f"{season}:{league_id}:{team_id}"


def standings_cache_key(league_id: int, season: str) -> str:
    return f"{season}:{league_id}"


# -------------------------------------------------
#  API helper
# -------------------------------------------------
def api_get(path: str, params: dict) -> dict:
    headers = {"x-apisports-key": API_KEY}
    url = f"{API_URL}{path}"
    try:
        res = requests.get(url, headers=headers, params=params, timeout=25)
    except Exception as e:
        log(f"❌ Request error on {path}: {e}")
        return {}

    if res.status_code != 200:
        log(f"⚠️ API status {res.status_code} on {path} params={params}")
        try:
            log(res.text[:300])
        except Exception:
            pass
        return {}

    try:
        data = res.json()
    except Exception as e:
        log(f"⚠️ JSON decode error on {path}: {e}")
        return {}

    errors = data.get("errors") or data.get("error")
    if errors:
        log(f"⚠️ API errors on {path}: {errors}")

    return data


# -------------------------------------------------
#  Fetchers
# -------------------------------------------------
def fetch_fixtures_for_league(league_id: int, season: str,
                              date_from: str, date_to: str) -> list:
    """
    Τραβάμε fixtures για συγκεκριμένη λίγκα, season, window.
    Χρησιμοποιούμε ΜΟΝΟ from/to (όχι date) για να μη γκρινιάζει το API.
    """
    info = LEAGUES[league_id]
    log(f"🥇 Fetching fixtures for {info['name']} ({league_id})")

    params = {
        "league": league_id,
        "season": season,
        "from": date_from,
        "to": date_to,
    }
    data = api_get("/fixtures", params)
    resp = data.get("response", []) if data else []
    log(f"   → {len(resp)} fixtures retrieved for league={league_id}")
    return resp


def fetch_team_stats(league_id: int, team_id: int, season: str) -> dict:
    key = team_cache_key(league_id, team_id, season)
    if key in TEAM_STATS_CACHE:
        return TEAM_STATS_CACHE[key]

    time.sleep(0.35)  # μικρό throttle

    params = {
        "league": league_id,
        "team": team_id,
        "season": season,
    }
    data = api_get("/teams/statistics", params)
    resp = data.get("response") if data else None
    if not resp:
        log(f"⚠️ Empty team statistics for league={league_id}, team={team_id}")
        TEAM_STATS_CACHE[key] = {}
        return {}

    # /teams/statistics επιστρέφει object, όχι list
    stats = resp if isinstance(resp, dict) else resp[0]
    TEAM_STATS_CACHE[key] = stats
    return stats


def fetch_league_standings(league_id: int, season: str) -> dict:
    key = standings_cache_key(league_id, season)
    if key in STANDINGS_CACHE:
        return STANDINGS_CACHE[key]

    time.sleep(0.35)

    params = {
        "league": league_id,
        "season": season,
    }
    data = api_get("/standings", params)
    resp = data.get("response") if data else None
    if not resp:
        log(f"⚠️ Empty standings for league={league_id}, season={season}")
        STANDINGS_CACHE[key] = {}
        return {}

    # API-Football structure: response[0]["league"]["standings"][0] → list of teams
    try:
        league_block = resp[0]["league"]
        standings_list = league_block["standings"][0]
        table = {row["team"]["id"]: row for row in standings_list}
    except Exception as e:
        log(f"⚠️ Unexpected standings format for league={league_id}: {e}")
        table = {}

    STANDINGS_CACHE[key] = table
    return table


# -------------------------------------------------
#  Model helpers
# -------------------------------------------------
def build_team_profile(stats: dict, standing_row: dict,
                       league_id: int, side: str) -> dict:
    """
    Φτιάχνει προφίλ ομάδας:
      - attack_index
      - defence_index
      - tempo_index
      - prestige_factor
      - motivation_factor
    side: "home" / "away"
    """

    # --- Basic production (goals, xG, shots) ---
    # averages per game
    gf_total = get_nested(stats, ["goals", "for", "average", "total"], 1.3)
    ga_total = get_nested(stats, ["goals", "against", "average", "total"], 1.3)

    gf_home = get_nested(stats, ["goals", "for", "average", "home"], gf_total)
    ga_home = get_nested(stats, ["goals", "against", "average", "home"], ga_total)
    gf_away = get_nested(stats, ["goals", "for", "average", "away"], gf_total)
    ga_away = get_nested(stats, ["goals", "against", "average", "away"], ga_total)

    # xG – αν δεν υπάρχει, fallback στα goals
    xg_for = get_nested(stats, ["expected", "goals", "for", "average", "total"], gf_total)
    xg_against = get_nested(
        stats, ["expected", "goals", "against", "average", "total"], ga_total
    )

    shots_for = get_nested(stats, ["shots", "total", "total"], 10.0)
    shots_on = get_nested(stats, ["shots", "on", "total"], 4.0)
    shots_against = get_nested(stats, ["shots", "total", "against"], 10.0)

    big_chances = get_nested(stats, ["big_chances", "for", "total"], 3.0)
    big_chances_against = get_nested(
        stats, ["big_chances", "against", "total"], 3.0
    )

    # tempo / pace approx: σύνολο shots ανά game και total goals expectation
    tempo_raw = (shots_for + shots_against) / 20.0 + (gf_total + ga_total) / 4.0
    tempo_index = clamp(tempo_raw, 0.4, 1.8)

    # attack / defence index
    attack_raw = (
        0.35 * gf_total
        + 0.25 * xg_for
        + 0.15 * (shots_on / 5.0)
        + 0.15 * (big_chances / 4.0)
        + 0.10 * tempo_index
    )

    defence_raw = (
        0.35 * ga_total
        + 0.25 * xg_against
        + 0.15 * (shots_against / 10.0)
        + 0.15 * (big_chances_against / 4.0)
        + 0.10 * tempo_index
    )

    attack_index = clamp(attack_raw, 0.4, 2.5)
    defence_index = clamp(defence_raw, 0.4, 2.5)

    # side-adjust για home/away
    if side == "home":
        attack_index *= clamp(1.0 + (gf_home - gf_away) * 0.15, 0.85, 1.25)
        defence_index *= clamp(1.0 + (ga_home - ga_away) * 0.10, 0.80, 1.20)
    else:
        attack_index *= clamp(1.0 + (gf_away - gf_home) * 0.15, 0.85, 1.25)
        defence_index *= clamp(1.0 + (ga_away - ga_home) * 0.10, 0.80, 1.20)

    # --- Prestige & Motivation from standings ---
    total_teams = 20
    rank = None
    points = None
    goal_diff = 0

    if standing_row:
        try:
            rank = int(standing_row.get("rank") or 0)
        except Exception:
            rank = None
        try:
            points = int(standing_row.get("points") or 0)
        except Exception:
            points = None
        try:
            goals_for = standing_row.get("all", {}).get("goals", {}).get("for", 0)
            goals_against = standing_row.get("all", {}).get("goals", {}).get("against", 0)
            goal_diff = safe_float(goals_for) - safe_float(goals_against)
        except Exception:
            goal_diff = 0

        try:
            total_teams = int(
                standing_row.get("group_total")
                or standing_row.get("total_teams")
                or 20
            )
        except Exception:
            total_teams = 20

    # prestige: πάνω οι “μεγάλοι” + λίγη ενίσχυση από goal_diff
    if rank is None or rank <= 0:
        prestige = 0.9
    else:
        # 1ος → 1.15, τελευταίος → 0.75
        prestige = 1.15 - 0.40 * (rank - 1) / max(1, total_teams - 1)
        prestige += clamp(goal_diff / 40.0, -0.05, 0.05)

    prestige = clamp(prestige, 0.70, 1.20)

    # motivation: μάχη τίτλου / Ευρώπη / υποβιβασμός
    motivation = 1.0
    if rank is not None and total_teams >= 10:
        if rank <= 4:
            motivation += 0.10  # title / Europe
        if rank <= 2:
            motivation += 0.05  # title fight

        if rank >= total_teams - 2:
            motivation += 0.15  # direct relegation fight
        elif rank >= total_teams - 4:
            motivation += 0.08  # play-out zone

    motivation = clamp(motivation, 0.85, 1.25)

    # μικρό league-specific tweak
    engines = LEAGUES.get(league_id, {}).get("engines", set())
    if "draw" in engines:
        # πιο αργές λίγκες
        tempo_index *= 0.95
    if "over" in engines:
        # πιο γρήγορες
        tempo_index *= 1.05

    return {
        "attack_index": attack_index,
        "defence_index": defence_index,
        "tempo_index": tempo_index,
        "prestige": prestige,
        "motivation": motivation,
    }


def compute_match_model(home_profile: dict, away_profile: dict,
                        league_id: int) -> dict:
    """
    Παίρνει τα δύο profiles και παράγει:
      - p_home, p_draw, p_away
      - p_over_2_5, p_under_2_5
    """

    # home advantage baseline
    home_adv_base = 0.10

    # league type tweaks
    engines = LEAGUES.get(league_id, {}).get("engines", set())
    draw_league = "draw" in engines
    over_league = "over" in engines

    if draw_league:
        home_adv_base -= 0.02  # πιο ισορροπημένες
    if over_league:
        home_adv_base += 0.01  # λίγο παραπάνω home edge

    # effective strength
    def strength(p):
        return (
            1.4 * p["attack_index"]
            - 1.0 * p["defence_index"]
        ) * p["prestige"] * p["motivation"]

    s_home = strength(home_profile)
    s_away = strength(away_profile)

    # normalise a bit
    scale = max(1.0, (abs(s_home) + abs(s_away)) / 3.5)
    s_home /= scale
    s_away /= scale

    diff = s_home - s_away + home_adv_base

    # logistic for home win prob
    p_home_raw = 1.0 / (1.0 + math.exp(-diff * 1.45))
    p_away_raw = 1.0 / (1.0 + math.exp(diff * 1.45))
    # symmetrical
    p_away_raw = 1.0 - p_home_raw

    # draw probability: base + ισορροπία
    balance = 1.0 - clamp(abs(diff), 0.0, 1.5) / 1.5
    p_draw_base = 0.25
    if draw_league:
        p_draw_base += 0.03
    if over_league:
        p_draw_base -= 0.02

    p_draw = clamp(p_draw_base + 0.07 * balance, 0.18, 0.35)

    remaining = max(0.0, 1.0 - p_draw)
    p_home = clamp(remaining * p_home_raw, 0.05, 0.80)
    p_away = clamp(remaining * p_away_raw, 0.05, 0.80)

    # normalise
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    # Over 2.5 model
    tempo_avg = (home_profile["tempo_index"] + away_profile["tempo_index"]) / 2.0
    attack_sum = home_profile["attack_index"] + away_profile["attack_index"]
    defence_sum = home_profile["defence_index"] + away_profile["defence_index"]

    base_over = 0.52
    if over_league:
        base_over += 0.06
    if draw_league:
        base_over -= 0.02

    # attack vs defence signal
    attack_signal = clamp((attack_sum - defence_sum) / 4.0, -0.08, 0.10)
    tempo_signal = clamp((tempo_avg - 1.0) * 0.12, -0.05, 0.07)

    p_over = clamp(base_over + attack_signal + tempo_signal, 0.40, 0.78)
    p_under = 1.0 - p_over

    return {
        "home_win": round(p_home, 3),
        "draw_win": round(p_draw, 3),
        "away_win": round(p_away, 3),
        "over_2_5": round(p_over, 3),
        "under_2_5": round(p_under, 3),
    }


# -------------------------------------------------
#  MAIN
# -------------------------------------------------
def main():
    global TEAM_STATS_CACHE, STANDINGS_CACHE

    if not API_KEY:
        raise RuntimeError("FOOTBALL_API_KEY is not set in environment")
    log(f"🔑 Using FOOTBALL_SEASON={SEASON}")

    load_core_configs()

    TEAM_STATS_CACHE = load_json_cache(TEAM_CACHE_PATH)
    STANDINGS_CACHE = load_json_cache(STANDINGS_CACHE_PATH)

    # Window: από σήμερα + 4 ημέρες
    today = datetime.utcnow().date()
    date_from = today.strftime("%Y-%m-%d")
    date_to = (today + timedelta(days=4)).strftime("%Y-%m-%d")
    log("==============================================")
    log(f"🗓  Window: {date_from} → {date_to} (season {SEASON})")

    all_fixtures = []

    # 1) Τραβάμε fixtures ανά λίγκα
    for league_id in sorted(LEAGUES.keys()):
        league_fixtures = fetch_fixtures_for_league(
            league_id, SEASON, date_from, date_to
        )
        all_fixtures.extend(league_fixtures)

    log(f"📊 Total fixtures found: {len(all_fixtures)}")

    processed = []

    # 2) Standings cache per league
    standings_per_league = {}
    for league_id in sorted(LEAGUES.keys()):
        standings_per_league[league_id] = fetch_league_standings(league_id, SEASON)

    # 3) Process κάθε fixture
    for f in all_fixtures:
        try:
            fixture = f["fixture"]
            league = f["league"]
            teams = f["teams"]

            league_id = int(league["id"])
            if league_id not in LEAGUES:
                continue

            league_info = LEAGUES[league_id]
            engines = league_info["engines"]

            home_team = teams["home"]
            away_team = teams["away"]

            home_id = int(home_team["id"])
            away_id = int(away_team["id"])

            home_name = home_team["name"]
            away_name = away_team["name"]

            fixture_id = int(fixture["id"])
            kickoff_iso = fixture.get("date")
            # timestamp = fixture.get("timestamp")

            standings_table = standings_per_league.get(league_id, {})
            home_standing = standings_table.get(home_id, {})
            away_standing = standings_table.get(away_id, {})

            # Φέρνουμε team statistics
            home_stats = fetch_team_stats(league_id, home_id, SEASON)
            away_stats = fetch_team_stats(league_id, away_id, SEASON)

            if not home_stats or not away_stats:
                log(f"⚠️ Missing stats for fixture {fixture_id} ({home_name} - {away_name})")
                continue

            home_profile = build_team_profile(
                home_stats, home_standing, league_id, side="home"
            )
            away_profile = build_team_profile(
                away_stats, away_standing, league_id, side="away"
            )

            model = compute_match_model(home_profile, away_profile, league_id)

            expected_goals = round(
                home_profile["attack_index"] + away_profile["attack_index"], 3
            )

            strength_home = round(
                home_profile["attack_index"] * home_profile["prestige"], 3
            )
            strength_away = round(
                away_profile["attack_index"] * away_profile["prestige"], 3
            )

            if "draw" in engines:
                engine_tag = "draw"
            elif "over" in engines:
                engine_tag = "over"
            else:
                engine_tag = "other"

            processed.append(
                {
                    "fixture_id": fixture_id,
                    "date": kickoff_iso,
                    "league_id": league_id,
                    "league": league_info["name"],
                    "engine": list(sorted(engines)),
                    "home": home_name,
                    "away": away_name,
                    "model": model,
                    "expected_goals": expected_goals,
                    "strength_home": strength_home,
                    "strength_away": strength_away,
                    "profile_home": home_profile,
                    "profile_away": away_profile,
                }
            )

        except Exception as e:
            log(f"⚠️ Error processing fixture: {e}")

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "window": {
            "date_from": date_from,
            "date_to": date_to,
            "season": SEASON,
        },
        "fixtures_analyzed": len(processed),
        "fixtures": processed,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False)

    save_json_cache(TEAM_CACHE_PATH, TEAM_STATS_CACHE)
    save_json_cache(STANDINGS_CACHE_PATH, STANDINGS_CACHE)

    log(f"✅ Thursday v3 ready → {len(processed)} fixtures analysed.")
    log(f"📝 Saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
