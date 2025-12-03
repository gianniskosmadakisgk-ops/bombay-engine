# ================================================================
#  BOMBAY ENGINE — THURSDAY ANALYSIS FULL v3
#  - Fixtures από API-FOOTBALL (per league)
#  - Season από ENV (FOOTBALL_SEASON) ή auto fallback
#  - Team statistics + standings
#  - Full Bookmaker-style model:
#       P(home win), P(draw), P(away win), P(over 2.5), P(under 2.5)
#  - Ασφαλές fallback αν κάτι λείπει
#  - Αποθήκευση report: logs/thursday_report_v3.json
# ================================================================

import os
import json
import time
import math
from datetime import datetime, timedelta
from pathlib import Path

import requests

# ------------------------------------------------
#  CONFIG & CONSTANTS
# ------------------------------------------------

API_KEY = os.getenv("FOOTBALL_API_KEY")
BASE_URL = "https://v3.football.api-sports.io"

# Season από ENV (π.χ. 2025). Αν λείπει → auto.
FOOTBALL_SEASON_ENV = os.getenv("FOOTBALL_SEASON")

REPORT_PATH = "logs/thursday_report_v3.json"
CACHE_PATH = "logs/team_stats_cache_v3.json"
STANDINGS_CACHE_PATH = "logs/standings_cache_v3.json"

os.makedirs("logs", exist_ok=True)

# Λίγκες που δουλεύουμε (IDs του API-FOOTBALL)
LEAGUES = [
    {"name": "Premier League", "id": 39},
    {"name": "La Liga", "id": 140},
    {"name": "Serie A", "id": 135},
    {"name": "Bundesliga", "id": 78},
    {"name": "Ligue 1", "id": 61},
]

# Πόσες μέρες μπροστά κοιτάμε (συμπεριλαμβάνει σήμερα)
DAYS_FORWARD = 4

# Βάρη μοντέλου (παιχνιδάκι για tuning αργότερα)
W_FORM = 0.25
W_GOAL_DIFF = 0.20
W_ATTACK = 0.15
W_DEFENCE = 0.15
W_STABILITY = 0.10
W_PRESTIGE = 0.08
W_MOTIVATION = 0.07

HOME_ADVANTAGE = 0.20  # extra rating για γηπεδούχο


# ------------------------------------------------
#  LOG HELPER
# ------------------------------------------------

def log(msg: str):
    print(msg, flush=True)


# ------------------------------------------------
#  SEASON HELPER
# ------------------------------------------------

def get_current_season(today: datetime) -> str:
    """
    Αν δεν έχουμε FOOTBALL_SEASON στο ENV, βρίσκουμε σεζόν:
    - Ιούλιος–Δεκέμβριος  → season = current year
    - Ιανουάριος–Ιούνιος  → season = previous year
    """
    if today.month >= 7:
        year = today.year
    else:
        year = today.year - 1
    return str(year)


def resolve_season() -> str:
    today = datetime.utcnow()
    if FOOTBALL_SEASON_ENV:
        log(f"ℹ️ Using season from ENV FOOTBALL_SEASON={FOOTBALL_SEASON_ENV}")
        return FOOTBALL_SEASON_ENV
    season = get_current_season(today)
    log(f"ℹ️ Using auto-detected season={season}")
    return season


# ------------------------------------------------
#  CACHE HELPERS (TEAM STATS + STANDINGS)
# ------------------------------------------------

def load_json_cache(path: str) -> dict:
    if not os.path.exists(path):
        log(f"ℹ️ No cache at {path}, starting empty.")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log(f"ℹ️ Loaded cache {path} ({len(data)} entries)")
        return data
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


def stats_cache_key(league_id: int, team_id: int, season: str) -> str:
    return f"{season}:{league_id}:{team_id}"


def standings_cache_key(league_id: int, season: str) -> str:
    return f"{season}:{league_id}"


# ------------------------------------------------
#  API HELPER
# ------------------------------------------------

def api_get(path: str, params: dict) -> dict:
    headers = {"x-apisports-key": API_KEY}
    url = f"{BASE_URL}{path}"
    try:
        res = requests.get(url, headers=headers, params=params, timeout=20)
    except Exception as e:
        log(f"❌ Request error {path} {params}: {e}")
        return {"response": [], "errors": {"exception": str(e)}}

    try:
        data = res.json()
    except Exception as e:
        log(f"❌ JSON decode error {path}: {e}")
        return {"response": [], "errors": {"json": str(e)}}

    errors = data.get("errors") or data.get("error")
    if errors:
        log(f"⚠️ API errors on {path}: {errors}")

    return data


# ------------------------------------------------
#  FETCH FIXTURES / STATS / STANDINGS
# ------------------------------------------------

def fetch_fixtures_for_league(league_id: int, season: str,
                              date_from: str, date_to: str) -> list:
    params = {
        "league": league_id,
        "season": season,
        "from": date_from,
        "to": date_to,
    }
    data = api_get("/fixtures", params)
    response = data.get("response", [])
    log(f"   → {len(response)} fixtures retrieved for league={league_id}")
    return response


def fetch_team_stats(league_id: int, team_id: int, season: str) -> dict:
    key = stats_cache_key(league_id, team_id, season)
    if key in TEAM_STATS_CACHE:
        return TEAM_STATS_CACHE[key]

    # Μικρό delay για να μη βαράμε το API σαν τρελοί
    time.sleep(0.25)

    params = {
        "league": league_id,
        "team": team_id,
        "season": season,
    }
    data = api_get("/teams/statistics", params)
    response = data.get("response", {})

    if isinstance(response, list):
        response = response[0] if response else {}

    TEAM_STATS_CACHE[key] = response or {}
    return TEAM_STATS_CACHE[key]


def fetch_standings(league_id: int, season: str) -> dict:
    """
    Γυρίζει dict: team_id -> {"rank": int, "points": int}
    """
    key = standings_cache_key(league_id, season)
    if key in STANDINGS_CACHE:
        return STANDINGS_CACHE[key]

    params = {"league": league_id, "season": season}
    data = api_get("/standings", params)
    response = data.get("response", [])

    standings_map = {}

    try:
        # API-Football: response[0]["league"]["standings"] είναι list of groups
        if response:
            groups = response[0]["league"]["standings"]
            # groups είναι list από λίστες
            for group in groups:
                for row in group:
                    team = row.get("team", {}) or {}
                    team_id = int(team.get("id"))
                    rank = int(row.get("rank", 0) or 0)
                    points = int(row.get("points", 0) or 0)
                    standings_map[team_id] = {
                        "rank": rank,
                        "points": points,
                    }
    except Exception as e:
        log(f"⚠️ Error parsing standings for league {league_id}: {e}")

    STANDINGS_CACHE[key] = standings_map
    return standings_map


# ------------------------------------------------
#  FEATURE ENGINE (per team)
# ------------------------------------------------

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def compute_team_features(stats: dict, team_id: int,
                          is_home: bool,
                          standings: dict) -> dict:
    """
    Βγάζει τα βασικά features για την ομάδα:
    attack, defence, form, stability, prestige, motivation, tempo/xG proxy.
    Όλα σε "rating" γύρω από 0.
    """
    fixtures = stats.get("fixtures", {}) or {}
    played_total = safe_float(fixtures.get("played", {}).get("total"), 0.0)
    wins_total = safe_float(fixtures.get("wins", {}).get("total"), 0.0)
    draws_total = safe_float(fixtures.get("draws", {}).get("total"), 0.0)
    loses_total = safe_float(fixtures.get("loses", {}).get("total"), 0.0)

    goals = stats.get("goals", {}) or {}
    gf_avg = safe_float(goals.get("for", {}).get("average", {}).get("total"), 1.3)
    ga_avg = safe_float(goals.get("against", {}).get("average", {}).get("total"), 1.2)

    clean_sheet_total = safe_float(stats.get("clean_sheet", {}).get("total"), 0.0)
    failed_to_score_total = safe_float(stats.get("failed_to_score", {}).get("total"), 0.0)

    # --- Form (win rate) ---
    if played_total > 0:
        win_rate = wins_total / played_total
        draw_rate = draws_total / played_total
        loss_rate = loses_total / played_total
    else:
        win_rate = 0.33
        draw_rate = 0.33
        loss_rate = 0.34

    form_rating = (win_rate - loss_rate)  # -1 .. +1 περίπου

    # --- Attack / Defence / Goal diff ---
    goal_diff_avg = gf_avg - ga_avg  # θετικό = επικίνδυνη ομάδα
    attack_rating = gf_avg  # όσο βάζει τόσο το καλύτερο
    defence_rating = -ga_avg  # όσο λιγότερα τρώει τόσο καλύτερα (άρα αρνητικό)

    # --- Stability (clean sheets vs failed to score) ---
    if played_total > 0:
        cs_rate = clean_sheet_total / played_total
        fts_rate = failed_to_score_total / played_total
    else:
        cs_rate = 0.2
        fts_rate = 0.2

    stability_rating = cs_rate - fts_rate  # -1 .. +1 περίπου

    # --- Prestige (proxy) ---
    # Αν δεν έχουμε τίποτα άλλο, χρησιμοποιούμε:
    #   περισσότερα points + καλύτερο goal diff => μεγαλύτερο prestige
    standing_row = standings.get(team_id, {})
    rank = safe_float(standing_row.get("rank"), 10.0)
    points = safe_float(standing_row.get("points"), 0.0)

    # Χοντρικό prestige: περισσότερα points, μικρότερο rank -> καλύτερο.
    prestige_raw = (points / 90.0) + (1.0 - min(rank, 20.0) / 20.0)
    # normalise γύρω από 0
    prestige_rating = prestige_raw * 0.5  # ~0..1

    # --- Motivation ---
    # Τίτλος / Ευρώπη / Σωτηρία παίρνουν boost.
    motivation_rating = 0.0
    if rank > 0:
        if rank <= 3:
            motivation_rating += 0.4  # τίτλος
        elif rank <= 6:
            motivation_rating += 0.25  # Ευρώπη
        elif rank >= 17:  # ζώνη υποβιβασμού (προσεγγιστικά)
            motivation_rating += 0.35  # σωτηρία
        elif 7 <= rank <= 10:
            motivation_rating += 0.1  # κυνήγι καλύτερης θέσης
        else:
            motivation_rating += 0.0   # καθαρά mid-table

    # --- Tempo / xG proxy ---
    # Δεν έχουμε πραγματικά xG στο API-FOOTBALL, άρα proxy:
    tempo_rating = gf_avg + ga_avg          # πόσο "ανοικτά" παιχνίδια
    xg_proxy = gf_avg                       # proxy επίθεσης

    # Normalize tempo/xg γύρω από 0
    tempo_rating = (tempo_rating - 2.5) / 2.0   # περίπου -1..+1
    xg_rating = (xg_proxy - 1.5) / 1.5

    # --- Home advantage ---
    home_bonus = HOME_ADVANTAGE if is_home else 0.0

    return {
        "form": form_rating,
        "goal_diff": goal_diff_avg,
        "attack": attack_rating,
        "defence": defence_rating,
        "stability": stability_rating,
        "prestige": prestige_rating,
        "motivation": motivation_rating,
        "tempo": tempo_rating,
        "xg": xg_rating,
        "home_bonus": home_bonus,
        "gf_avg": gf_avg,
        "ga_avg": ga_avg,
    }


# ------------------------------------------------
#  MATCH MODEL — PROBABILITIES
# ------------------------------------------------

def combine_team_rating(features: dict) -> float:
    """
    Παίρνει όλα τα features και τα κάνει ένα συνολικό rating.
    """
    rating = 0.0
    rating += W_FORM * features["form"]
    rating += W_GOAL_DIFF * features["goal_diff"]
    rating += W_ATTACK * features["attack"]
    rating += W_DEFENCE * features["defence"]
    rating += W_STABILITY * features["stability"]
    rating += W_PRESTIGE * features["prestige"]
    rating += W_MOTIVATION * features["motivation"]
    rating += 0.10 * features["tempo"]
    rating += 0.10 * features["xg"]
    rating += features["home_bonus"]
    return rating


def clamp_prob(p: float) -> float:
    return max(0.03, min(0.93, p))


def compute_match_probabilities(home_feat: dict, away_feat: dict) -> dict:
    """
    Βγάζει:
      - P(home win), P(draw), P(away win)
      - P(over 2.5), P(under 2.5)
    με λογιστικές συναρτήσεις + heuristics.
    """
    # --- Strength ratings ---
    r_home = combine_team_rating(home_feat)
    r_away = combine_team_rating(away_feat)
    diff = r_home - r_away

    # Logistic για home vs away
    p_home_raw = 1.0 / (1.0 + math.exp(-diff))
    p_away_raw = 1.0 - p_home_raw

    # Base draw probability ανάλογα με ισορροπία & ρυθμό
    balance = max(0.0, 1.0 - abs(diff))              # όσο πιο κοντά τόσο πιο μεγάλο
    tempo_factor = (home_feat["tempo"] + away_feat["tempo"]) / 2.0
    # χαμηλό tempo → λίγο πιο πολλά X, υψηλό tempo → λίγο λιγότερα
    base_draw = 0.24 + 0.06 * balance - 0.03 * tempo_factor
    base_draw = clamp_prob(base_draw)

    remaining = 1.0 - base_draw
    # αναλογία home/away με βάση τα raw
    total_raw = p_home_raw + p_away_raw
    if total_raw <= 0:
        p_home = remaining * 0.5
        p_away = remaining * 0.5
    else:
        p_home = remaining * (p_home_raw / total_raw)
        p_away = remaining * (p_away_raw / total_raw)

    # Normalize
    total = p_home + base_draw + p_away
    if total > 0:
        p_home /= total
        p_draw = base_draw / total
        p_away /= total
    else:
        p_home, p_draw, p_away = 0.33, 0.34, 0.33

    # Clamp
    p_home = clamp_prob(p_home)
    p_draw = clamp_prob(p_draw)
    p_away = clamp_prob(p_away)

    # --- Over 2.5 goals probability ---
    gf_home = home_feat["gf_avg"]
    ga_home = home_feat["ga_avg"]
    gf_away = away_feat["gf_avg"]
    ga_away = away_feat["ga_avg"]

    attack_level = (gf_home + gf_away) / 2.0
    defence_level = (ga_home + ga_away) / 2.0
    tempo = (home_feat["tempo"] + away_feat["tempo"]) / 2.0

    # Προσεγγιστικό expected goals για το match
    expected_goals = attack_level + max(0.0, defence_level) + 0.4 * tempo

    # Map expected_goals → P(over2.5) με logistic
    p_over = 1.0 / (1.0 + math.exp(-(expected_goals - 2.6)))
    p_over = clamp_prob(p_over)
    p_under = 1.0 - p_over

    return {
        "home_win": round(p_home, 4),
        "draw_win": round(p_draw, 4),
        "away_win": round(p_away, 4),
        "over_2_5": round(p_over, 4),
        "under_2_5": round(p_under, 4),
        "expected_goals": round(expected_goals, 3),
        "strength_home": round(r_home, 3),
        "strength_away": round(r_away, 3),
    }


# ------------------------------------------------
#  MAIN
# ------------------------------------------------

def main():
    global TEAM_STATS_CACHE, STANDINGS_CACHE

    if not API_KEY:
        raise RuntimeError("FOOTBALL_API_KEY not set")

    # Load caches
    TEAM_STATS_CACHE = load_json_cache(CACHE_PATH)
    STANDINGS_CACHE = load_json_cache(STANDINGS_CACHE_PATH)

    # Resolve season & date window
    today = datetime.utcnow()
    season = resolve_season()
    date_from = today.strftime("%Y-%m-%d")
    date_to = (today + timedelta(days=DAYS_FORWARD)).strftime("%Y-%m-%d")

    log("===========================================")
    log(f"📅 Window: {date_from} → {date_to} (season {season})")
    log("===========================================")

    fixtures_out = []

    for league in LEAGUES:
        league_id = league["id"]
        league_name = league["name"]
        log(f"🏆 Fetching fixtures for {league_name} ({league_id})")

        fixtures_raw = fetch_fixtures_for_league(
            league_id=league_id,
            season=season,
            date_from=date_from,
            date_to=date_to,
        )

        if not fixtures_raw:
            log(f"⚠️ No fixtures for {league_name} in window.")
            continue

        # Standings (για motivation / prestige)
        standings = fetch_standings(league_id, season)

        for fx in fixtures_raw:
            try:
                fixture_info = fx.get("fixture", {}) or {}
                teams_info = fx.get("teams", {}) or {}

                fixture_id = int(fixture_info.get("id"))
                kickoff_iso = fixture_info.get("date")
                league_obj = fx.get("league", {}) or {}
                lg_name = league_obj.get("name", league_name)

                home = teams_info.get("home", {}) or {}
                away = teams_info.get("away", {}) or {}

                home_id = int(home.get("id"))
                away_id = int(away.get("id"))
                home_name = home.get("name", "Home")
                away_name = away.get("name", "Away")

                # Team statistics
                home_stats = fetch_team_stats(league_id, home_id, season)
                away_stats = fetch_team_stats(league_id, away_id, season)

                if not home_stats or not away_stats:
                    log(f"⚠️ Missing stats for {home_name} vs {away_name}, using fallback probs.")
                    model_probs = {
                        "home_win": 0.40,
                        "draw_win": 0.28,
                        "away_win": 0.32,
                        "over_2_5": 0.5,
                        "under_2_5": 0.5,
                        "expected_goals": 2.5,
                        "strength_home": 0.0,
                        "strength_away": 0.0,
                    }
                else:
                    # Features
                    home_feat = compute_team_features(home_stats, home_id, True, standings)
                    away_feat = compute_team_features(away_stats, away_id, False, standings)

                    model_probs = compute_match_probabilities(home_feat, away_feat)

                fixtures_out.append(
                    {
                        "league": lg_name,
                        "league_id": league_id,
                        "fixture_id": fixture_id,
                        "date": kickoff_iso,
                        "home": home_name,
                        "away": away_name,
                        "model": model_probs,
                    }
                )

            except Exception as e:
                log(f"❌ Error processing fixture: {e}")
                continue

    # Τελικό report
    report = {
        "fixtures": fixtures_out,
        "generated_at": datetime.utcnow().isoformat(),
        "source_window": {
            "date_from": date_from,
            "date_to": date_to,
            "season": season,
        },
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Save caches
    save_json_cache(CACHE_PATH, TEAM_STATS_CACHE)
    save_json_cache(STANDINGS_CACHE_PATH, STANDINGS_CACHE)

    log("===========================================")
    log(f"✅ Thursday v3 ready — {len(fixtures_out)} fixtures analysed.")
    log(f"📝 Saved → {REPORT_PATH}")
    log("===========================================")


if __name__ == "__main__":
    main()
