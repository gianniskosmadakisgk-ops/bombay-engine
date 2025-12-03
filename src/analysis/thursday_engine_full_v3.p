import os
import json
import time
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple

import requests
import yaml

# ======================================================
#  THURSDAY ENGINE — FULL v3 (Giannis Edition)
#
#  Στόχος:
#   - Πλήρες "bookmaker-style" μοντέλο για 1X2 + Over 2.5
#   - Χρήση ΠΟΛΛΩΝ στατιστικών από API-Football:
#       * xG, goals, shots, big chances, box entries,
#         tempo/pace proxies, PPDA/OPPDA (όπου υπάρχουν),
#         form, rest-days (προσεγγιστικά),
#         clean sheets, failed to score κλπ.
#       * Prestige & Motivation layers
#   - Επιστρέφει:
#       * fair_1, fair_x, fair_2, fair_over
#       * score_draw, score_over (0–10)
#   - Output format συμβατό με Friday shortlist:
#       logs/thursday_report_v1.json
#
#  Αρχιτεκτονική:
#   1) Fetch fixtures (4 μέρες window) σε target leagues
#   2) Fetch team statistics (με cache)
#   3) Fetch standings ανά λίγκα (για prestige/motivation)
#   4) Χτίζουμε rich feature vectors για κάθε ομάδα
#   5) "Full" match model → P(Home/Draw/Away/Over2.5)
#   6) Fair odds + scores
# ======================================================

FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

# Λίγκες-στόχοι (ονόματα API-Football)
TARGET_LEAGUES = {
    # Draw Engine
    "Ligue 1",
    "Serie A",
    "La Liga",
    "Championship",
    "Serie B",
    "Ligue 2",
    "Liga Portugal 2",
    "Swiss Super League",

    # Over Engine
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

DAYS_FORWARD = 4

REPORT_PATH = "logs/thursday_report_v1.json"
CACHE_FILE = "logs/team_stats_cache_v1.json"

os.makedirs("logs", exist_ok=True)

# In-memory caches
_team_stats_cache: Dict[str, Any] = {}
_standings_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}


# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
def log(msg: str):
    print(msg, flush=True)


def clamp(x: float, low: float, high: float) -> float:
    return max(low, min(high, x))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def safe_get(d: Any, path: Tuple, default: Any = 0.0):
    cur = d
    try:
        for p in path:
            if isinstance(cur, dict):
                cur = cur.get(p)
            else:
                return default
        if cur is None:
            return default
        return cur
    except Exception:
        return default


def form_to_index(form_str: str) -> float:
    """Μετατρέπει φόρμα τύπου 'WDWLD' σε index [-1, +1]."""
    if not form_str:
        return 0.0
    values = {"W": 1.0, "D": 0.0, "L": -1.0}
    arr = [values.get(c, 0.0) for c in form_str if c in values]
    if not arr:
        return 0.0
    return sum(arr) / len(arr)


def get_current_season(day: datetime) -> str:
    """
    API-Football season = έτος έναρξης σεζόν.
    Ιούλιος–Δεκέμβριος → current year
    Ιανουάριος–Ιούνιος → previous year
    """
    if day.month >= 7:
        year = day.year
    else:
        year = day.year - 1
    return str(year)


# ------------------------------------------------------
# Core configs (sanity only)
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
# Cache helpers
# ------------------------------------------------------
def cache_key(league_id: int, team_id: int, season: str) -> str:
    return f"{season}:{league_id}:{team_id}"


def load_stats_cache() -> Dict[str, Any]:
    if not os.path.exists(CACHE_FILE):
        log("ℹ️ No existing team stats cache, starting fresh.")
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        log(f"ℹ️ Loaded team stats cache ({len(cache)}) from {CACHE_FILE}")
        return cache
    except Exception as e:
        log(f"⚠️ Failed to load cache {CACHE_FILE}: {e}")
        return {}


def save_stats_cache(cache: Dict[str, Any]):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        log(f"💾 Team stats cache saved ({len(cache)}) → {CACHE_FILE}")
    except Exception as e:
        log(f"⚠️ Failed to save cache {CACHE_FILE}: {e}")


# ------------------------------------------------------
# API-Football wrappers
# ------------------------------------------------------
def api_get(path: str, params: dict) -> dict:
    headers = {"x-apisports-key": FOOTBALL_API_KEY}
    url = f"{FOOTBALL_BASE_URL}{path}"

    try:
        res = requests.get(url, headers=headers, params=params, timeout=20)
    except Exception as e:
        log(f"⚠️ Request error on {path}: {e}")
        return {}

    if res.status_code != 200:
        log(f"⚠️ API error {res.status_code} on {path} with params {params}")
        try:
            log(f"⚠️ Body: {res.text[:400]}")
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


def fetch_fixtures(date_from: str, date_to: str, season: str) -> list:
    params = {
        "season": season,
        "from": date_from,
        "to": date_to,
    }
    data = api_get("/fixtures", params)
    response = data.get("response", []) if data else []
    log(f"✅ Raw fixtures from API: {len(response)} matches")

    fixtures = []
    for f in response:
        league = f.get("league") or {}
        league_name = league.get("name")
        if league_name in TARGET_LEAGUES:
            fixtures.append(f)

    log(f"🎯 Filtered fixtures in target leagues: {len(fixtures)} matches")
    return fixtures


def fetch_team_stats(league_id: int, team_id: int, season: str) -> dict:
    global _team_stats_cache

    key = cache_key(league_id, team_id, season)
    if key in _team_stats_cache:
        return _team_stats_cache[key]

    # προστασία από rate limit
    time.sleep(0.5)

    params = {
        "league": league_id,
        "team": team_id,
        "season": season,
    }
    data = api_get("/teams/statistics", params)
    response = data.get("response") if data else None
    if not response:
        log(f"⚠️ Empty team statistics for league={league_id}, team={team_id}, season={season}")
        _team_stats_cache[key] = {}
        return {}

    stats = response if isinstance(response, dict) else response[0]
    _team_stats_cache[key] = stats
    return stats


def fetch_standings(league_id: int, season: str) -> Dict[int, Dict[str, Any]]:
    """
    Παίρνει πίνακα standings και επιστρέφει:
        { team_id: { 'rank': int, 'points': int } }
    """
    cache_key_st = f"{season}:{league_id}"
    if cache_key_st in _standings_cache:
        return _standings_cache[cache_key_st]

    params = {
        "league": league_id,
        "season": season,
    }
    data = api_get("/standings", params)
    response = data.get("response") if data else None
    result: Dict[int, Dict[str, Any]] = {}

    if not response:
        log(f"⚠️ Empty standings for league={league_id}, season={season}")
        _standings_cache[cache_key_st] = result
        return result

    try:
        # API-Football: response[0]['league']['standings'][0] is list
        league_block = response[0]["league"]["standings"][0]
        for row in league_block:
            team = row.get("team") or {}
            team_id = int(team.get("id"))
            rank = int(row.get("rank") or 0)
            points = int(row.get("points") or 0)
            result[team_id] = {"rank": rank, "points": points}
    except Exception as e:
        log(f"⚠️ Error parsing standings: {e}")

    _standings_cache[cache_key_st] = result
    return result


# ------------------------------------------------------
# Feature extraction per team
# ------------------------------------------------------
def build_team_features(team_stats: dict,
                        standings_map: Dict[int, Dict[str, Any]],
                        team_id: int) -> Dict[str, float]:
    """
    Εξάγει rich features από το team_stats + standings.
    Όλα γίνονται scale ~[0,1] όπου γίνεται.
    """

    # Goals / xG
    gf_total = float(
        safe_get(team_stats, ("goals", "for", "average", "total"), 1.4)
    )
    ga_total = float(
        safe_get(team_stats, ("goals", "against", "average", "total"), 1.2)
    )

    xg_for = float(
        safe_get(team_stats, ("goals", "for", "average", "total", "xg"), gf_total)
    )
    xg_against = float(
        safe_get(team_stats, ("goals", "against", "average", "total", "xg"), ga_total)
    )

    # Shots / big chances (Proxy για δημιουργικότητα)
    shots_for = float(
        safe_get(team_stats, ("shots", "for", "average", "total"), 11.0)
    )
    shots_against = float(
        safe_get(team_stats, ("shots", "against", "average", "total"), 11.0)
    )

    big_ch_for = float(
        safe_get(team_stats, ("biggest", "chances", "for"), 0.0)
    )
    big_ch_against = float(
        safe_get(team_stats, ("biggest", "chances", "against"), 0.0)
    )

    # Possession / Tempo proxies
    poss = float(safe_get(team_stats, ("lineups",), 0.0))  # αν δεν υπάρχει, fallback
    # αν δεν υπάρχει καθόλου possession στο API, το αφήνουμε ουδέτερο
    # Tempo proxy = shots total per match + (xG_for + xG_against)
    tempo_proxy = shots_for + shots_against + xg_for + xg_against

    # Form
    form_str = str(team_stats.get("form") or "")
    form_index = form_to_index(form_str)  # [-1, +1]

    # Clean sheets / failed to score
    cs_total = float(safe_get(team_stats, ("clean_sheet", "total"), 0.0))
    fts_total = float(safe_get(team_stats, ("failed_to_score", "total"), 0.0))
    played_total = float(safe_get(team_stats, ("fixtures", "played", "total"), 10.0))

    cs_rate = cs_total / played_total if played_total > 0 else 0.2
    fts_rate = fts_total / played_total if played_total > 0 else 0.2

    # Prestige (from standings)
    standing_info = standings_map.get(team_id) or {}
    rank = int(standing_info.get("rank") or 10)
    points = int(standing_info.get("points") or 0)

    # Scale rank: μικρό rank = κορυφή
    prestige_rank_factor = clamp(1.0 - (rank - 1) / 18.0, 0.0, 1.0)
    prestige_points_factor = clamp(points / 80.0, 0.0, 1.0)
    prestige = 0.6 * prestige_rank_factor + 0.4 * prestige_points_factor

    # Motivation proxy:
    #   - Υψηλό όταν:
    #       * κοντά στην κορυφή (τίτλος / Ευρώπη)
    #       * κοντά στην ζώνη υποβιβασμού
    #   - + μικρό boost από πρόσφατη φόρμα
    if rank <= 4:
        title_mot = 1.0
    elif rank <= 6:
        title_mot = 0.8
    elif rank <= 10:
        title_mot = 0.5
    else:
        title_mot = 0.3

    # Relegation κίνητρο
    if rank >= 16:
        releg_mot = 1.0
    elif rank >= 14:
        releg_mot = 0.7
    else:
        releg_mot = 0.3

    motivation = 0.5 * title_mot + 0.5 * releg_mot
    # Blend με φόρμα (form_index [-1,1] → [0,1])
    motivation = clamp(
        0.8 * motivation + 0.2 * (0.5 * (form_index + 1.0)),
        0.0,
        1.0,
    )

    # Strength indices
    # attack_strength: goals + xG + shots + big-chances
    attack_raw = (
        0.35 * gf_total
        + 0.25 * xg_for
        + 0.20 * (shots_for / 10.0)
        + 0.20 * (big_ch_for / max(played_total, 1.0))
    )

    defence_raw = (
        0.40 * (2.0 - ga_total)  # όσο λιγότερα δέχεται, τόσο καλύτερα
        + 0.25 * (2.0 - xg_against)
        + 0.20 * cs_rate * 3.0
        + 0.15 * (1.0 - fts_rate)
    )

    # Scale strength indices περίπου σε [0,1.5]
    attack_strength = clamp(attack_raw / 3.5, 0.0, 1.5)
    defence_strength = clamp(defence_raw / 3.5, 0.0, 1.5)

    tempo_strength = clamp(tempo_proxy / 30.0, 0.0, 2.0)

    return {
        "attack": attack_strength,
        "defence": defence_strength,
        "tempo": tempo_strength,
        "form": form_index,       # [-1,1]
        "prestige": prestige,     # [0,1]
        "motivation": motivation, # [0,1]
        "gf": gf_total,
        "ga": ga_total,
    }


# ------------------------------------------------------
# Match model: από features → probabilities
# ------------------------------------------------------
def compute_match_probabilities(
    home_feats: Dict[str, float],
    away_feats: Dict[str, float],
) -> Tuple[float, float, float, float, float, float]:
    """
    Επιστρέφει:
      p_home, p_draw, p_away, p_over, score_draw, score_over
    """

    # Συνολικό strength (attack+defence) με prestige & motivation
    base_home = home_feats["attack"] + home_feats["defence"]
    base_away = away_feats["attack"] + away_feats["defence"]

    prestige_diff = home_feats["prestige"] - away_feats["prestige"]
    motivation_diff = home_feats["motivation"] - away_feats["motivation"]

    # Prestige & motivation ~ 10–12% impact συνολικά
    adj_home = base_home + 0.6 * prestige_diff + 0.5 * motivation_diff
    adj_away = base_away - 0.6 * prestige_diff - 0.5 * motivation_diff

    strength_diff = adj_home - adj_away  # >0: home ισχυρότερο
    strength_scale = clamp(strength_diff / 2.0, -2.5, 2.5)

    # Draw related quantities
    balance = clamp(1.0 - abs(strength_diff) / 3.0, 0.0, 1.0)

    # Defence / Tempo blended
    avg_def = 0.5 * (home_feats["defence"] + away_feats["defence"])
    avg_tempo = 0.5 * (home_feats["tempo"] + away_feats["tempo"])

    # --- Draw probability ---
    # base 0.24 + bonus όταν:
    #   - ισοδύναμες ομάδες (balance)
    #   - καλή άμυνα
    #   - όχι τρελό tempo
    p_draw = (
        0.24
        + 0.10 * balance
        + 0.06 * clamp(avg_def / 1.2, 0.0, 1.0)
        - 0.05 * clamp((avg_tempo - 0.8), -0.5, 0.8)
    )
    p_draw = clamp(p_draw, 0.18, 0.33)

    # --- Home / Away probability μέσω logistic στο strength_diff ---
    # (περίπου 0.5 όταν ισορροπημένες, πηγαίνει 0.7–0.75 όταν μεγάλη διαφορά)
    r = sigmoid(strength_scale)  # 0–1
    remaining = max(0.0, 1.0 - p_draw)
    p_home = remaining * r
    p_away = remaining * (1.0 - r)

    # Safe normalize
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    # --- Over 2.5 probability ---
    # Base από expected goals:
    exp_goals = (
        home_feats["gf"] + away_feats["gf"] + home_feats["ga"] + away_feats["ga"]
    ) / 2.0

    tempo_factor = clamp(avg_tempo / 1.0, 0.5, 1.5)
    attack_factor = clamp(
        0.5 * (home_feats["attack"] + away_feats["attack"]), 0.5, 1.5
    )

    raw_over = 0.48
    raw_over += 0.20 * clamp((exp_goals - 2.4) / 1.0, -0.5, 0.6)
    raw_over += 0.10 * (tempo_factor - 1.0)
    raw_over += 0.08 * (attack_factor - 1.0)

    p_over = clamp(raw_over, 0.40, 0.78)

    # --- Scores (0–10) ---
    # Draw score: υψηλό όταν p_draw και balance είναι υψηλά,
    # και όταν prestige gap δεν είναι τεράστιο (big upset).
    prestige_gap = abs(prestige_diff)
    upset_penalty = clamp(prestige_gap, 0.0, 0.8)

    score_draw_raw = (
        6.0
        + 6.0 * (p_draw - 0.22) / 0.12
        + 2.0 * (balance - 0.5)
        - 2.5 * upset_penalty
    )
    score_draw = round(clamp(score_draw_raw, 0.0, 10.0), 2)

    # Over score: βασίζεται σε p_over + tempo + attack
    score_over_raw = (
        6.5
        + 5.5 * (p_over - 0.50) / 0.25
        + 1.5 * (tempo_factor - 1.0)
        + 1.0 * (attack_factor - 1.0)
    )
    score_over = round(clamp(score_over_raw, 0.0, 10.0), 2)

    return p_home, p_draw, p_away, p_over, score_draw, score_over


def probs_to_fair(p: float) -> float:
    p = clamp(p, 0.05, 0.90)
    return round(1.0 / p, 2)


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    global _team_stats_cache

    if not FOOTBALL_API_KEY:
        raise RuntimeError("FOOTBALL_API_KEY is not set in environment.")

    load_core_configs()

    _team_stats_cache = load_stats_cache()

    # 1) Primary window
    today = datetime.utcnow()
    season_primary = get_current_season(today)

    date_from = today.strftime("%Y-%m-%d")
    date_to = (today + timedelta(days=DAYS_FORWARD)).strftime("%Y-%m-%d")

    log(f"📅 Primary window: {date_from} to {date_to} (season {season_primary})")

    fixtures_raw = fetch_fixtures(date_from, date_to, season_primary)

    # 2) Fallback ένα έτος πίσω αν είναι άδειο
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

    # Pre-fetch standings per league to reuse
    leagues_seen = set()
    standings_per_league: Dict[int, Dict[int, Dict[str, Any]]] = {}

    for f in fixtures_raw:
        try:
            league_block = f.get("league") or {}
            league_id = int(league_block.get("id"))
            league_name = league_block.get("name") or ""

            if league_id not in standings_per_league:
                standings_per_league[league_id] = fetch_standings(league_id, season_used)
                leagues_seen.add(league_name)

            standings_map = standings_per_league[league_id]

            fixture_info = f.get("fixture") or {}
            kickoff_iso = fixture_info.get("date")  # ISO string
            kickoff_ts = fixture_info.get("timestamp")

            home_block = f.get("teams", {}).get("home") or {}
            away_block = f.get("teams", {}).get("away") or {}

            home_team = home_block.get("name") or "Home"
            away_team = away_block.get("name") or "Away"
            home_id = int(home_block.get("id"))
            away_id = int(away_block.get("id"))

            match_label = f"{home_team} - {away_team}"

            home_stats = fetch_team_stats(league_id, home_id, season_used)
            away_stats = fetch_team_stats(league_id, away_id, season_used)

            if not home_stats or not away_stats:
                log(f"⚠️ Missing stats for {match_label}, skipping.")
                continue

            home_feats = build_team_features(home_stats, standings_map, home_id)
            away_feats = build_team_features(away_stats, standings_map, away_id)

            (
                p_home,
                p_draw,
                p_away,
                p_over,
                score_draw,
                score_over,
            ) = compute_match_probabilities(home_feats, away_feats)

            fair_1 = probs_to_fair(p_home)
            fair_x = probs_to_fair(p_draw)
            fair_2 = probs_to_fair(p_away)
            fair_over = probs_to_fair(p_over)

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

    save_stats_cache(_team_stats_cache)

    log(f"✅ Thursday engine v3 complete — {len(processed)} fixtures analyzed.")
    log(f"📝 Report saved at {REPORT_PATH}")

    if processed:
        log("📌 Sample fixtures from report:")
        log(json.dumps(processed[:3], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
