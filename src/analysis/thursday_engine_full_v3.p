# ======================================================
#  BOMBAY ENGINE — THURSDAY ANALYSIS FULL v3
#  (Block 1 + Block 2 + Block 3 merged)
#  Created for: Giannis — Full Professional Model
# ======================================================

import os
import json
import time
import math
from datetime import datetime, timedelta
from pathlib import Path
import requests

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------

API_KEY = os.getenv("FOOTBALL_API_KEY")
BASE_URL = "https://v3.football.api-sports.io"

REPORT_PATH = "logs/thursday_report_v3.json"
CACHE_PATH = "logs/team_stats_cache_v3.json"

os.makedirs("logs", exist_ok=True)

# ------------------------------------------------------
# TARGET LEAGUES
# ------------------------------------------------------

TARGET_LEAGUES = {
    "Premier League",
    "Championship",
    "Serie A",
    "Serie B",
    "La Liga",
    "La Liga 2",
    "Bundesliga",
    "Bundesliga 2",
    "Ligue 1",
    "Ligue 2",
    "Eredivisie",
    "Swiss Super League",
    "Liga Portugal 1",
    "Liga Portugal 2",
    "Jupiler Pro League",
    "Superliga",
    "Allsvenskan",
    "Eliteserien",
}

# ------------------------------------------------------
# INTERNAL CACHE
# ------------------------------------------------------

if os.path.exists(CACHE_PATH):
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            TEAM_CACHE = json.load(f)
    except:
        TEAM_CACHE = {}
else:
    TEAM_CACHE = {}

# ------------------------------------------------------
# LOGGING
# ------------------------------------------------------
def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------
# API CALLS
# ------------------------------------------------------
def api_get(path, params):
    headers = {"x-apisports-key": API_KEY}

    try:
        r = requests.get(BASE_URL + path, params=params, headers=headers, timeout=20)
    except Exception as e:
        log(f"⚠️ Network error: {e}")
        return []

    if r.status_code != 200:
        log(f"⚠️ API status {r.status_code}: {r.text[:200]}")
        return []

    try:
        data = r.json()
    except:
        log("⚠️ JSON decode error")
        return []

    return data.get("response", [])


# ------------------------------------------------------
# FETCH FIXTURES
# ------------------------------------------------------
def fetch_fixtures():
    today = datetime.utcnow()
    date_from = today.strftime("%Y-%m-%d")
    date_to = (today + timedelta(days=4)).strftime("%Y-%m-%d")

    # Determine season
    if today.month >= 7:
        season = today.year
    else:
        season = today.year - 1

    params = {"season": season, "from": date_from, "to": date_to}
    raw = api_get("/fixtures", params)
    log(f"📌 Raw fixtures fetched: {len(raw)}")

    filtered = []
    for f in raw:
        league_name = f.get("league", {}).get("name")
        if league_name in TARGET_LEAGUES:
            filtered.append(f)

    log(f"🎯 Fixtures in target leagues: {len(filtered)}")
    return filtered, season


# ------------------------------------------------------
# FETCH TEAM STATISTICS (with cache)
# ------------------------------------------------------
def get_stats(league_id, team_id, season):
    key = f"{season}:{league_id}:{team_id}"

    if key in TEAM_CACHE:
        return TEAM_CACHE[key]

    time.sleep(0.4)  # anti-rate limit
    params = {"league": league_id, "team": team_id, "season": season}
    resp = api_get("/teams/statistics", params)

    if not resp:
        TEAM_CACHE[key] = {}
        return {}

    TEAM_CACHE[key] = resp[0]
    return resp[0]


# ------------------------------------------------------
# BOMBAY ENGINE — FULL MODEL
# ------------------------------------------------------

def safe(v, default=0.0):
    try:
        return float(v)
    except:
        return float(default)


def extract_metrics(stats):
    """ Extracts >14 metrics from stats safely. """

    g_for = safe(stats["goals"]["for"]["average"]["total"])
    g_against = safe(stats["goals"]["against"]["average"]["total"])
    xg_for = safe(stats["xG"]["for"])
    xg_against = safe(stats["xG"]["against"])
    shots_for = safe(stats["shots"]["total"]["average"])
    shots_against = safe(stats["shots"]["against"]["average"])
    ppda = safe(stats.get("ppda", {}).get("att", 12))
    oppda = safe(stats.get("ppda", {}).get("def", 12))

    big_chances = safe(stats.get("big_chances", 1))
    box_entries = safe(stats.get("box_entries", 8))
    deep_compl = safe(stats.get("deep_completions", 5))
    nsxg = safe(stats.get("nsxg", 0.8))

    form = safe(stats.get("form_points", 6))
    rest = safe(stats.get("rest_days", 5))

    return {
        "g_for": g_for,
        "g_against": g_against,
        "xg_for": xg_for,
        "xg_against": xg_against,
        "shots_for": shots_for,
        "shots_against": shots_against,
        "ppda": ppda,
        "oppda": oppda,
        "big_chances": big_chances,
        "box_entries": box_entries,
        "deep_compl": deep_compl,
        "nsxg": nsxg,
        "form": form,
        "rest": rest,
    }


def compute_bombay_probabilities(home, away):
    """ FULL MODEL: 14 metrics + prestige + motivation """

    # Normalize advantage
    def adv(a, b):
        return (a - b) / max(1.0, abs(a) + abs(b))

    weights = {
        "xg": 0.20,
        "xga": 0.20,
        "shots": 0.12,
        "nsxg": 0.08,
        "big": 0.08,
        "box": 0.08,
        "deep": 0.06,
        "ppda": 0.05,
        "oppda": 0.05,
        "form": 0.05,
        "rest": 0.03,
    }

    h_adv = (
        adv(home["xg_for"], away["xg_against"]) * weights["xg"] +
        adv(home["xg_against"], away["xg_for"]) * weights["xga"] +
        adv(home["shots_for"], away["shots_against"]) * weights["shots"] +
        adv(home["nsxg"], away["nsxg"]) * weights["nsxg"] +
        adv(home["big_chances"], away["big_chances"]) * weights["big"] +
        adv(home["box_entries"], away["box_entries"]) * weights["box"] +
        adv(home["deep_compl"], away["deep_compl"]) * weights["deep"] +
        adv(home["ppda"], away["ppda"]) * weights["ppda"] +
        adv(home["oppda"], away["oppda"]) * weights["oppda"] +
        adv(home["form"], away["form"]) * weights["form"] +
        adv(home["rest"], away["rest"]) * weights["rest"]
    )

    prestige_home = adv(home["form"], 5) * 0.10
    prestige_away = adv(away["form"], 5) * 0.10

    motivation_home = adv(home["rest"], away["rest"]) * 0.10
    motivation_away = adv(away["rest"], home["rest"]) * 0.10

    rating_home = h_adv + prestige_home + motivation_home
    rating_away = -h_adv + prestige_away + motivation_away

    # Convert ratings → probabilities via softmax
    exp_h = math.exp(rating_home)
    exp_a = math.exp(rating_away)
    base = exp_h + exp_a + 1.0

    p_home = exp_h / base
    p_away = exp_a / base
    p_draw = 1.0 / base

    # Over probability
    base_goals = home["g_for"] + away["g_for"] + home["xg_for"] + away["xg_for"]
    p_over = min(0.90, max(0.10, base_goals / 6.0))

    # Fair odds
    fair_1 = round(1 / max(p_home, 0.05), 2)
    fair_x = round(1 / max(p_draw, 0.05), 2)
    fair_2 = round(1 / max(p_away, 0.05), 2)
    fair_over = round(1 / max(p_over, 0.05), 2)

    # Engine scores 0–10
    score_draw = round(max(0, min(10, (p_draw - 0.20) * 40)), 2)
    score_over = round(max(0, min(10, (p_over - 0.45) * 30)), 2)

    return fair_1, fair_x, fair_2, fair_over, p_home, p_draw, p_away, p_over, score_draw, score_over


# ------------------------------------------------------
# MAIN ANALYSIS
# ------------------------------------------------------
def main():
    fixtures, season = fetch_fixtures()
    output = []

    for f in fixtures:
        league_id = f["league"]["id"]
        league_name = f["league"]["name"]

        home_name = f["teams"]["home"]["name"]
        away_name = f["teams"]["away"]["name"]
        home_id = f["teams"]["home"]["id"]
        away_id = f["teams"]["away"]["id"]

        date_utc = f["fixture"]["date"]
        ts = f["fixture"]["timestamp"]

        home_stats = get_stats(league_id, home_id, season)
        away_stats = get_stats(league_id, away_id, season)

        if not home_stats or not away_stats:
            log(f"⚠️ Missing stats: {home_name} vs {away_name}")
            continue

        H = extract_metrics(home_stats)
        A = extract_metrics(away_stats)

        (
            fair_1,
            fair_x,
            fair_2,
            fair_over,
            p1,
            px,
            p2,
            pover,
            score_draw,
            score_over,
        ) = compute_bombay_probabilities(H, A)

        output.append({
            "league": league_name,
            "league_id": league_id,
            "match": f"{home_name} - {away_name}",
            "date_utc": date_utc,
            "timestamp": ts,
            "fair_1": fair_1,
            "fair_x": fair_x,
            "fair_2": fair_2,
            "fair_over": fair_over,
            "p_home": p1,
            "p_draw": px,
            "p_away": p2,
            "p_over": pover,
            "score_draw": score_draw,
            "score_over": score_over,
        })

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": datetime.utcnow().isoformat(),
                "fixtures_analyzed": len(output),
                "fixtures": output,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(TEAM_CACHE, f, ensure_ascii=False)

    log(f"✅ Thursday v3 ready — {len(output)} fixtures analysed.")
    log(f"📝 Saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
