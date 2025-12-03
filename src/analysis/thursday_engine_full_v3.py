# ==============================================================
#  BOMBAY ENGINE — THURSDAY ANALYSIS FULL v3
#  (Merged logic — Bookmaker-style model for 1X2 + Over 2.5)
#
#  Created for: Giannis — Full Professional Model
#
#  - Παίρνει fixtures από API-Football
#  - Για κάθε αγώνα φέρνει team statistics (με cache)
#  - Υπολογίζει:
#       * p_home, p_draw, p_away, p_over
#       * fair_1, fair_x, fair_2, fair_over
#       * score_draw, score_over  (0–10)
#  - Φιλτράρει μόνο τις TARGET_LEAGUES
#  - Σώζει: logs/thursday_report_v3.json
# ==============================================================

import os
import json
import time
import math
from datetime import datetime, timedelta
from pathlib import Path

import requests

# ----------------------- CONFIG -----------------------

API_KEY = os.getenv("FOOTBALL_API_KEY")
BASE_URL = "https://v3.football.api-sports.io"

# Αν υπάρχει, ΠΑΝΤΑ χρησιμοποιούμε αυτή τη σεζόν
SEASON_OVERRIDE = os.getenv("FOOTBALL_SEASON")

REPORT_PATH = "logs/thursday_report_v3.json"
CACHE_PATH = "logs/team_stats_cache_v3.json"

os.makedirs("logs", exist_ok=True)

# Λίγκες-στόχοι με βάση το league.name του API
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

# Πόσες μέρες μπροστά κοιτάμε από σήμερα
DAYS_FORWARD = 4

# ----------------------- HELPERS -----------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def get_current_season(day: datetime) -> str:
    """
    API-Football: season = έτος έναρξης.
    Αν δεν έχουμε SEASON_OVERRIDE, χρησιμοποιούμε αυτό.
    Ιούλιος–Δεκέμβριος → τρέχον έτος
    Ιανουάριος–Ιούνιος → προηγούμενο
    """
    if day.month >= 7:
        year = day.year
    else:
        year = day.year - 1
    return str(year)


def safe_get(d, path, default=None):
    """
    Ασφαλές nested get: safe_get(stats, ["goals", "for", "average", "total"], 1.3)
    """
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        if key not in cur:
            return default
        cur = cur[key]
    if cur is None:
        return default
    return cur


# ----------------------- CACHE -----------------------


def load_stats_cache() -> dict:
    if not os.path.exists(CACHE_PATH):
        log("ℹ️ No existing stats cache, starting fresh.")
        return {}
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
        log(f"ℹ️ Loaded team stats cache ({len(cache)} entries).")
        return cache
    except Exception as e:
        log(f"⚠️ Failed to load cache {CACHE_PATH}: {e}")
        return {}


def save_stats_cache(cache: dict) -> None:
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        log(f"💾 Team stats cache saved ({len(cache)} entries) → {CACHE_PATH}")
    except Exception as e:
        log(f"⚠️ Failed to save cache {CACHE_PATH}: {e}")


def cache_key(league_id: int, team_id: int, season: str) -> str:
    return f"{season}:{league_id}:{team_id}"


_stats_cache: dict = {}


# ----------------------- API WRAPPERS -----------------------


def api_get(path: str, params: dict) -> list:
    if not API_KEY:
        raise RuntimeError("FOOTBALL_API_KEY is not set in environment.")

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
            log(f"⚠️ Body: {res.text[:400]}")
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
        log(f"⚠️ API reported errors on {path}: {errors}")

    response = data.get("response", [])
    return response


def fetch_fixtures(date_from: str, date_to: str, season: str) -> list:
    """
    Φέρνει ΟΛΑ τα fixtures της σεζόν στο συγκεκριμένο window
    και μετά φιλτράρει μόνο TARGET_LEAGUES.
    """
    params = {
        "season": season,
        "from": date_from,
        "to": date_to,
    }
    resp = api_get("/fixtures", params)
    log(f"✅ Raw fixtures fetched: {len(resp)}")

    fixtures = []
    for f in resp:
        league = f.get("league") or {}
        league_name = league.get("name")
        if league_name in TARGET_LEAGUES:
            fixtures.append(f)

    log(f"🎯 Fixtures in target leagues: {len(fixtures)}")
    return fixtures


def fetch_team_stats(league_id: int, team_id: int, season: str) -> dict:
    """
    – Πρώτα κοιτάμε local cache (_stats_cache)
    – Αν δεν υπάρχει, χτυπάμε /teams/statistics
    """
    global _stats_cache
    key = cache_key(league_id, team_id, season)

    if key in _stats_cache:
        return _stats_cache[key]

    # ‼️ Μικρό delay για να μην βαράμε limits
    time.sleep(0.4)

    params = {
        "league": league_id,
        "team": team_id,
        "season": season,
    }
    resp = api_get("/teams/statistics", params)
    if not resp:
        log(f"⚠️ Empty statistics for league={league_id}, team={team_id}, season={season}")
        _stats_cache[key] = {}
        return {}

    stats = resp[0] if isinstance(resp, list) else resp
    _stats_cache[key] = stats
    return stats


# ----------------------- RATING MODEL -----------------------


def compute_probs_and_scores(home_stats: dict, away_stats: dict):
    """
    Full-ish μοντέλο bookmaker:
    - Χρησιμοποιεί goals, xG, shots, big chances, PPDA, pace κλπ
    - Βγάζει p_home, p_draw, p_away, p_over
    - Μετατρέπει σε fair odds + scores 0–10
    """

    # ---------- BASIC GOALS / xG ----------
    gf_h = float(safe_get(home_stats, ["goals", "for", "average", "total"], 1.40))
    ga_h = float(safe_get(home_stats, ["goals", "against", "average", "total"], 1.20))
    gf_a = float(safe_get(away_stats, ["goals", "for", "average", "total"], 1.30))
    ga_a = float(safe_get(away_stats, ["goals", "against", "average", "total"], 1.30))

    xg_f_h = float(safe_get(home_stats, ["expected", "goals", "for", "average"], gf_h))
    xg_a_h = float(safe_get(home_stats, ["expected", "goals", "against", "average"], ga_h))
    xg_f_a = float(safe_get(away_stats, ["expected", "goals", "for", "average"], gf_a))
    xg_a_a = float(safe_get(away_stats, ["expected", "goals", "against", "average"], ga_a))

    # ---------- SHOTS / BIG CHANCES / BOX ENTRIES ----------
    shots_on_h = float(
        safe_get(home_stats, ["shots", "on", "average", "total"], 4.5)
    )
    shots_on_a = float(
        safe_get(away_stats, ["shots", "on", "average", "total"], 4.5)
    )

    big_ch_h = float(safe_get(home_stats, ["big_chances", "for", "average"], 1.2))
    big_ch_a = float(safe_get(away_stats, ["big_chances", "for", "average"], 1.2))
    big_ch_conc_h = float(
        safe_get(home_stats, ["big_chances", "against", "average"], 1.1)
    )
    big_ch_conc_a = float(
        safe_get(away_stats, ["big_chances", "against", "average"], 1.1)
    )

    pace_h = float(safe_get(home_stats, ["tempo", "pace"], 1.0))
    pace_a = float(safe_get(away_stats, ["tempo", "pace"], 1.0))

    # ---------- PRESS / PPDA ----------
    ppda_h = float(safe_get(home_stats, ["ppda", "for", "average"], 10.0))
    ppda_a = float(safe_get(away_stats, ["ppda", "for", "average"], 10.0))
    oppda_h = float(safe_get(home_stats, ["ppda", "against", "average"], 11.0))
    oppda_a = float(safe_get(away_stats, ["ppda", "against", "average"], 11.0))

    # μικρό normalisation (χαμηλότερο ppda → πιο aggressive pressing)
    press_index_h = 1.0 + (12.0 - min(ppda_h, 18.0)) / 40.0
    press_index_a = 1.0 + (12.0 - min(ppda_a, 18.0)) / 40.0

    # ---------- FORM (τελευταία 5 ματς) ----------
    def form_score(stats: dict) -> float:
        form_str = safe_get(stats, ["form"], "") or ""
        # API style: "WDWLW"
        score = 0.0
        for ch in form_str[-5:]:
            if ch == "W":
                score += 1.0
            elif ch == "D":
                score += 0.4
        return score / 5.0  # 0–1

    form_h = form_score(home_stats)
    form_a = form_score(away_stats)

    # ---------- PRESTIGE / MOTIVATION (coarse) ----------
    # Prestige proxy: overall rank this season (χαμηλότερο = καλύτερο)
    rank_h = safe_get(home_stats, ["league", "standings", "overall", "rank"], 10)
    rank_a = safe_get(away_stats, ["league", "standings", "overall", "rank"], 10)
    try:
        rank_h = float(rank_h)
        rank_a = float(rank_a)
    except Exception:
        rank_h = 10.0
        rank_a = 10.0

    prestige_h = 1.0 + (12.0 - min(rank_h, 20.0)) / 40.0
    prestige_a = 1.0 + (12.0 - min(rank_a, 20.0)) / 40.0

    # Motivation proxy: home/away points last 5 games
    mot_h = 1.0 + form_h * 0.5
    mot_a = 1.0 + form_a * 0.5

    # ---------- OFFENCE / DEFENCE INDEX ----------

    attack_h = (
        0.35 * gf_h
        + 0.35 * xg_f_h
        + 0.10 * shots_on_h
        + 0.10 * big_ch_h
        + 0.10 * pace_h
    ) * press_index_h

    attack_a = (
        0.35 * gf_a
        + 0.35 * xg_f_a
        + 0.10 * shots_on_a
        + 0.10 * big_ch_a
        + 0.10 * pace_a
    ) * press_index_a

    defence_h = (
        0.45 * ga_h + 0.35 * xg_a_h + 0.20 * big_ch_conc_h
    )
    defence_a = (
        0.45 * ga_a + 0.35 * xg_a_a + 0.20 * big_ch_conc_a
    )

    rating_h = (attack_h - defence_h) * mot_h * prestige_h
    rating_a = (attack_a - defence_a) * mot_a * prestige_a

    # ---------- 1X2 PROBABILITIES ----------

    # relative strength
    diff = rating_h - rating_a  # >0 = πιο δυνατό το home

    # logistic για μετατροπή diff → baseline home prob
    base_home = 1.0 / (1.0 + math.exp(-diff / 3.0))  # 0–1

    # draw baseline
    # όσο πιο κοντά τα ratings, τόσο μεγαλύτερη πιθανότητα ισοπαλίας
    balance = max(0.0, 1.0 - abs(diff) / 4.0)
    p_draw = 0.20 + 0.12 * balance  # ~ 0.20–0.32

    # απομένει για 1 & 2
    remaining = max(0.0, 1.0 - p_draw)
    p_home = remaining * base_home
    p_away = remaining * (1.0 - base_home)

    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    # ---------- OVER 2.5 PROBABILITY ----------

    total_attack = attack_h + attack_a
    total_xg = xg_f_h + xg_f_a

    base_over = 0.55
    # πιο επιθετικά / υψηλό xG → ανεβαίνει το over
    if total_attack + total_xg <= 4.0:
        p_over = base_over - 0.07
    elif total_attack + total_xg <= 5.0:
        p_over = base_over
    elif total_attack + total_xg <= 6.0:
        p_over = base_over + 0.08
    else:
        p_over = base_over + 0.13

    p_over = max(0.40, min(0.82, p_over))

    # ---------- FAIR ODDS ----------

    def fair_from_prob(p: float) -> float:
        p = max(0.04, min(0.92, p))
        return round(1.0 / p, 2)

    fair_1 = fair_from_prob(p_home)
    fair_x = fair_from_prob(p_draw)
    fair_2 = fair_from_prob(p_away)
    fair_over = fair_from_prob(p_over)

    # ---------- SCORES 0–10 ----------

    score_draw_raw = 5.5 + (p_draw - 0.24) / 0.10 * 3.5
    score_over_raw = 5.5 + (p_over - 0.50) / 0.18 * 4.0

    score_draw = round(max(0.0, min(10.0, score_draw_raw)), 2)
    score_over = round(max(0.0, min(10.0, score_over_raw)), 2)

    return {
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over": p_over,
        "fair_1": fair_1,
        "fair_x": fair_x,
        "fair_2": fair_2,
        "fair_over": fair_over,
        "score_draw": score_draw,
        "score_over": score_over,
    }


# ----------------------- MAIN -----------------------


def main():
    global _stats_cache

    if not API_KEY:
        raise RuntimeError("FOOTBALL_API_KEY is not set in environment.")

    # Φόρτωμα cache στην μνήμη
    _stats_cache = load_stats_cache()

    today = datetime.utcnow()
    date_from = today.strftime("%Y-%m-%d")
    date_to = (today + timedelta(days=DAYS_FORWARD)).strftime("%Y-%m-%d")

    if SEASON_OVERRIDE:
        season_primary = SEASON_OVERRIDE
        log(f"📅 Using season from env FOOTBALL_SEASON={season_primary}")
    else:
        season_primary = get_current_season(today)
        log(f"📅 Using inferred season={season_primary}")

    log(f"📅 Window: {date_from} → {date_to}")

    fixtures_raw = fetch_fixtures(date_from, date_to, season_primary)

    # Αν *δεν* έχουμε override και βγάλει 0 fixtures, κάνουμε fallback ένα χρόνο πίσω
    if not fixtures_raw and not SEASON_OVERRIDE:
        fallback_day = today - timedelta(days=365)
        fallback_season = get_current_season(fallback_day)
        fb_from = fallback_day.strftime("%Y-%m-%d")
        fb_to = (fallback_day + timedelta(days=DAYS_FORWARD)).strftime("%Y-%m-%d")
        log(
            f"⚠️ No fixtures in primary window. "
            f"Fallback to {fb_from} → {fb_to}, season={fallback_season}"
        )
        fixtures_raw = fetch_fixtures(fb_from, fb_to, fallback_season)
        season_used = fallback_season
        date_from_used = fb_from
        date_to_used = fb_to
    else:
        season_used = season_primary
        date_from_used = date_from
        date_to_used = date_to

    processed = []

    for f in fixtures_raw:
        try:
            league = f.get("league") or {}
            league_name = league.get("name")
            league_id = int(league.get("id"))

            fixture_info = f.get("fixture") or {}
            kickoff_iso = fixture_info.get("date")
            kickoff_ts = fixture_info.get("timestamp")

            home_team = (f.get("teams") or {}).get("home") or {}
            away_team = (f.get("teams") or {}).get("away") or {}
            home_name = home_team.get("name")
            away_name = away_team.get("name")
            home_id = int(home_team.get("id"))
            away_id = int(away_team.get("id"))

            match_label = f"{home_name} - {away_name}"

            home_stats = fetch_team_stats(league_id, home_id, season_used)
            away_stats = fetch_team_stats(league_id, away_id, season_used)

            if not home_stats or not away_stats:
                log(f"⚠️ Missing stats for {match_label}, skipping.")
                continue

            res = compute_probs_and_scores(home_stats, away_stats)

            processed.append(
                {
                    "league": league_name,
                    "league_id": league_id,
                    "match": match_label,
                    "date_utc": kickoff_iso,
                    "timestamp": kickoff_ts,
                    "fair_1": res["fair_1"],
                    "fair_x": res["fair_x"],
                    "fair_2": res["fair_2"],
                    "fair_over": res["fair_over"],
                    "score_draw": res["score_draw"],
                    "score_over": res["score_over"],
                    "p_home": round(res["p_home"], 4),
                    "p_draw": round(res["p_draw"], 4),
                    "p_away": round(res["p_away"], 4),
                    "p_over": round(res["p_over"], 4),
                }
            )

        except Exception as e:
            log(f"⚠️ Error processing fixture: {e}")

    output = {
        "generated_at": datetime.utcnow().isoformat(),
        "source_window": {
            "date_from": date_from_used,
            "date_to": date_to_used,
            "season": season_used,
        },
        "fixtures_analyzed": len(processed),
        "fixtures": processed,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    save_stats_cache(_stats_cache)

    log(
        f"✅ Thursday v3 ready → {len(processed)} fixtures analysed. "
        f"Saved → {REPORT_PATH}"
    )

    if processed:
        log("📌 Sample fixtures:")
        log(json.dumps(processed[:3], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
