# ================================================================
#  BOMBAY ENGINE — THURSDAY ANALYSIS FULL v3.5
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
#  Season resolver
# -------------------------------------------------
FOOTBALL_SEASON_ENV = os.getenv("FOOTBALL_SEASON")

def resolve_season() -> str:
    if FOOTBALL_SEASON_ENV:
        return FOOTBALL_SEASON_ENV

    today = datetime.utcnow()
    return str(today.year if today.month >= 7 else today.year - 1)

SEASON = resolve_season()

# -------------------------------------------------
#  Paths
# -------------------------------------------------
REPORT_PATH = "logs/thursday_report_v3.json"
TEAM_CACHE_PATH = "logs/team_stats_cache_v3.json"
STANDINGS_CACHE_PATH = "logs/standings_cache_v3.json"
os.makedirs("logs", exist_ok=True)

# -------------------------------------------------
#  LEAGUES
# -------------------------------------------------

DRAW_LEAGUES = {
    39: "Premier League",
    40: "Championship",
    61: "Ligue 1",
    62: "Ligue 2",
    95: "Liga Portugal 2",
    135: "Serie A",
    136: "Serie B",
    140: "La Liga",
    98: "Japan J1",
    99: "Japan J2",
}

OVER_LEAGUES = {
    78: "Bundesliga",
    88: "Eredivisie",
    94: "Liga Portugal 1",
    144: "Belgium Jupiler League",
    271: "Denmark Superliga",
    103: "Eliteserien (Norway)",
    113: "Allsvenskan (Sweden)",
    207: "Swiss Super League",
    253: "MLS",
    98: "Japan J1",
    99: "Japan J2",
    71: "Brazil Serie A",
    72: "Brazil Serie B",
    128: "Argentina Liga Profesional",
}

LEAGUES = {}
for lid, name in DRAW_LEAGUES.items():
    LEAGUES.setdefault(lid, {"name": name, "engines": set()})
    LEAGUES[lid]["engines"].add("draw")

for lid, name in OVER_LEAGUES.items():
    LEAGUES.setdefault(lid, {"name": name, "engines": set()})
    LEAGUES[lid]["engines"].add("over")

# -------------------------------------------------
#  Logging utils
# -------------------------------------------------

def log(msg: str):
    print(msg, flush=True)

def clamp(x, low, high):
    return max(low, min(high, x))

def safe_float(x, default=0.0):
    try: return float(x)
    except: return default

def get_nested(d: dict, path, default=0.0):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur: 
            return default
        cur = cur[key]
    return safe_float(cur, default=default)

# -------------------------------------------------
#  Cache
# -------------------------------------------------

def load_json_cache(path: str):
    if not os.path.exists(path):
        return {}
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except:
        return {}

def save_json_cache(path: str, data: dict):
    try:
        json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False)
    except:
        pass

TEAM_STATS_CACHE = {}
STANDINGS_CACHE = {}

def team_cache_key(league_id, team_id, season):
    return f"{season}:{league_id}:{team_id}"

def standings_cache_key(league_id, season):
    return f"{season}:{league_id}"
    # -------------------------------------------------
#  API helper
# -------------------------------------------------
def api_get(path: str, params: dict) -> dict:
    headers = {"x-apisports-key": API_KEY}
    url = f"{API_URL}{path}"
    try:
        r = requests.get(url, headers=headers, params=params, timeout=25)
    except Exception as e:
        log(f"❌ Request error on {path}: {e}")
        return {}

    if r.status_code != 200:
        log(f"⚠️ API status {r.status_code} on {path}: {params}")
        return {}
    try:
        return r.json()
    except:
        return {}

# -------------------------------------------------
#  FETCHERS
# -------------------------------------------------
def fetch_fixtures_for_league(league_id, season, date_from, date_to):
    params = {
        "league": league_id,
        "season": int(season),
        "from": date_from,
        "to": date_to,
    }
    data = api_get("/fixtures", params)
    return data.get("response", []) if data else []

def fetch_team_stats(league_id, team_id, season):
    key = team_cache_key(league_id, team_id, season)
    if key in TEAM_STATS_CACHE:
        return TEAM_STATS_CACHE[key]

    time.sleep(0.35)
    params = {"league": league_id, "team": team_id, "season": int(season)}
    data = api_get("/teams/statistics", params)
    resp = data.get("response") if data else None

    stats = resp if isinstance(resp, dict) else (resp[0] if resp else {})
    TEAM_STATS_CACHE[key] = stats or {}
    return stats or {}

def fetch_league_standings(league_id, season):
    key = standings_cache_key(league_id, season)
    if key in STANDINGS_CACHE:
        return STANDINGS_CACHE[key]

    time.sleep(0.35)
    params = {"league": league_id, "season": int(season)}
    data = api_get("/standings", params)
    resp = data.get("response") if data else None

    try:
        standings = resp[0]["league"]["standings"][0]
        table = {row["team"]["id"]: row for row in standings}
    except:
        table = {}

    STANDINGS_CACHE[key] = table
    return table

# -------------------------------------------------
#  MIN MATCHDAY CHECK
# -------------------------------------------------
def league_ready(league_id):
    """Αν η λίγκα έχει < 3 αγωνιστικές → την αγνοούμε."""
    table = STANDINGS_CACHE.get(standings_cache_key(league_id, SEASON), {})
    if not table:
        return False

    # κάθε row έχει 'all': {'played': X}
    played = []
    for row in table.values():
        try:
            played.append(int(row.get("all", {}).get("played", 0)))
        except:
            pass

    if not played:
        return False

    avg_played = sum(played) / len(played)
    return avg_played >= 3

# -------------------------------------------------
#  TEAM PROFILE BUILDER
# -------------------------------------------------
def build_team_profile(stats, standing, league_id, side):
    # basic numbers
    gf = get_nested(stats, ["goals", "for", "average", "total"], 1.3)
    ga = get_nested(stats, ["goals", "against", "average", "total"], 1.3)

    xgf = get_nested(stats, ["expected", "goals", "for", "average", "total"], gf)
    xga = get_nested(stats, ["expected", "goals", "against", "average", "total"], ga)

    shots_for = get_nested(stats, ["shots", "total", "total"], 10)
    shots_against = get_nested(stats, ["shots", "total", "against"], 10)
    big_for = get_nested(stats, ["big_chances", "for", "total"], 3)
    big_against = get_nested(stats, ["big_chances", "against", "total"], 3)

    tempo_raw = (shots_for + shots_against)/22 + (gf + ga)/4
    tempo = clamp(tempo_raw, 0.4, 1.8)

    attack = (
        0.40 * gf +
        0.30 * xgf +
        0.15 * (big_for / 4.0) +
        0.15 * tempo
    )

    defence = (
        0.40 * ga +
        0.30 * xga +
        0.15 * (big_against / 4.0) +
        0.15 * tempo
    )

    # home/away tweak
    try:
        gf_h = get_nested(stats, ["goals", "for", "average", side], gf)
        ga_h = get_nested(stats, ["goals", "against", "average", side], ga)
        attack *= clamp(1.0 + (gf_h - gf)*0.10, 0.85, 1.25)
        defence *= clamp(1.0 + (ga_h - ga)*0.10, 0.85, 1.25)
    except:
        pass

    # prestige & motivation
    if standing:
        rank = safe_float(standing.get("rank", 10))
        total = safe_float(standing.get("group_total", 20))
        gd = safe_float(standing.get("all", {}).get("goals", {}).get("for", 0)) - \
             safe_float(standing.get("all", {}).get("goals", {}).get("against", 0))
    else:
        rank, total, gd = 10, 20, 0

    prestige = 1.15 - 0.40*((rank-1)/max(1,total-1))
    prestige += clamp(gd/40, -0.05, 0.05)
    prestige = clamp(prestige, 0.70, 1.20)

    motivation = 1.0
    if rank <= 4: motivation += 0.10
    if rank <= 2: motivation += 0.05
    if rank >= total-2: motivation += 0.12
    motivation = clamp(motivation, 0.85, 1.25)

    # league-type tweak
    engines = LEAGUES.get(league_id, {}).get("engines", set())
    if "draw" in engines:
        tempo *= 0.95
    if "over" in engines:
        tempo *= 1.05

    return {
        "attack": clamp(attack, 0.4, 3.0),
        "defence": clamp(defence, 0.4, 3.0),
        "tempo": tempo,
        "prestige": prestige,
        "motivation": motivation,
    }
    # -------------------------------------------------
#  MATCH MODEL (probabilities)
# -------------------------------------------------
def compute_match_model(home, away, league_id):
    # Home advantage baseline
    home_adv = 0.10
    engines = LEAGUES.get(league_id, {}).get("engines", set())

    if "draw" in engines:
        home_adv -= 0.02
    if "over" in engines:
        home_adv += 0.01

    # Team strengths (simple, deterministic)
    def strength(p):
        return (
            1.4 * p["attack"]
            - 1.0 * p["defence"]
        ) * p["prestige"] * p["motivation"]

    s_h = strength(home)
    s_a = strength(away)

    # Normalization
    scale = max(1.0, (abs(s_h) + abs(s_a)) / 3.5)
    s_h /= scale
    s_a /= scale

    diff = s_h - s_a + home_adv

    # 1X2 PROBABILITIES
    p_home_raw = 1 / (1 + math.exp(-diff * 1.45))
    p_away_raw = 1 - p_home_raw

    # Draw logic
    balance = 1 - clamp(abs(diff), 0, 1.5)/1.5
    p_d_base = 0.25
    if "draw" in engines:
        p_d_base += 0.03
    if "over" in engines:
        p_d_base -= 0.02

    p_draw = clamp(p_d_base + 0.07*balance, 0.17, 0.35)

    # re-distribute
    remaining = max(0, 1 - p_draw)
    p_home = remaining * p_home_raw
    p_away = remaining * p_away_raw

    # normalize
    total = p_home + p_draw + p_away
    p_home /= total
    p_draw /= total
    p_away /= total

    # OVER/UNDER MODEL
    tempo_avg = (home["tempo"] + away["tempo"]) / 2
    atk_sum = home["attack"] + away["attack"]
    def_sum = home["defence"] + away["defence"]

    base_over = 0.52
    if "over" in engines:
        base_over += 0.05
    if "draw" in engines:
        base_over -= 0.02

    atk_signal = clamp((atk_sum - def_sum)/4, -0.08, 0.10)
    tempo_signal = clamp((tempo_avg - 1)*0.12, -0.05, 0.07)

    p_over = clamp(base_over + atk_signal + tempo_signal, 0.38, 0.80)
    p_under = 1 - p_over

    # SCORE ENGINE (draw-score + over-score)
    # --- DRAW SCORE ---
    # Better peak zones for realistic selection
    draw_score = (
        15
        - abs(p_draw - 0.33) * 35
        - abs(p_home - 0.33) * 15
        - abs(p_away - 0.33) * 15
    )
    draw_score = clamp(draw_score, 0, 10)

    # --- OVER SCORE ---
    over_score = (
        16 * p_over               # grows faster 0.40→6.4, 0.55→8.8, 0.65→10.4
        + (tempo_avg - 1) * 5     # tempo boost
    )
    over_score = clamp(over_score, 0, 10)

    return {
        "p_home": round(p_home, 3),
        "p_draw": round(p_draw, 3),
        "p_away": round(p_away, 3),
        "p_over": round(p_over, 3),
        "p_under": round(p_under, 3),
        "draw_score": round(draw_score, 2),
        "over_score": round(over_score, 2),
    }

# -------------------------------------------------
#  PROBABILITY → FAIR ODDS WITH SMALL MARGIN (2.5%)
# -------------------------------------------------
def apply_margin(p_home, p_draw, p_away, margin=1.025):
    """
    We add a tiny 2.5% overround so the odds are realistic.
    """
    total = p_home + p_draw + p_away
    p_home /= total
    p_draw /= total
    p_away /= total

    # stretch
    factor = margin
    p_home *= factor
    p_draw *= factor
    p_away *= factor

    # final renorm to the overround
    new_total = p_home + p_draw + p_away
    p_home /= new_total
    p_draw /= new_total
    p_away /= new_total

    return p_home, p_draw, p_away

def fair_odds(p):
    p = clamp(p, 0.03, 0.95)
    return round(1/p, 2)
    # -------------------------------------------------
#  MAIN
# -------------------------------------------------
def main():
    global TEAM_STATS_CACHE, STANDINGS_CACHE

    if not API_KEY:
        raise RuntimeError("FOOTBALL_API_KEY is not set")

    load_core_configs()

    TEAM_STATS_CACHE = load_json_cache(TEAM_CACHE_PATH)
    STANDINGS_CACHE = load_json_cache(STANDINGS_CACHE_PATH)

    # Window — 3 full days ahead
    today = datetime.utcnow().date()
    date_from = today.strftime("%Y-%m-%d")
    date_to = (today + timedelta(days=3)).strftime("%Y-%m-%d")

    log("==============================================")
    log(f"🗓  Window: {date_from} → {date_to}")
    log(f"🏆  Season used: {SEASON}")
    log("==============================================")

    # WEEK COUNTER
    state_file = "logs/week_state.json"
    try:
        if os.path.exists(state_file):
            state = json.load(open(state_file, "r"))
            week = int(state.get("week", 1))
        else:
            week = 1
    except:
        week = 1

    # Save back increment (Thursday always increments week)
    new_state = {"week": week + 1}
    json.dump(new_state, open(state_file, "w"))

    # fetch all fixtures
    all_fixtures = []
    for league_id in sorted(LEAGUES.keys()):
        fixtures = fetch_fixtures_for_league(
            league_id, SEASON, date_from, date_to
        )
        all_fixtures.extend(fixtures)

    log(f"📊 Total fixtures picked: {len(all_fixtures)}")

    # standings cache
    standings_per_league = {
        lid: fetch_league_standings(lid, SEASON)
        for lid in LEAGUES.keys()
    }

    processed = []

    # process each fixture
    for f in all_fixtures:
        try:
            fixture = f["fixture"]
            league = f["league"]
            teams = f["teams"]

            league_id = int(league["id"])
            if league_id not in LEAGUES:
                continue

            # date/time
            kickoff_iso = fixture.get("date")
            match_date, match_time = "", ""
            if kickoff_iso:
                try:
                    dt = datetime.fromisoformat(kickoff_iso.replace("Z","+00:00"))
                    match_date = dt.strftime("%Y-%m-%d")
                    match_time = dt.strftime("%H:%M")
                except:
                    if "T" in kickoff_iso:
                        parts = kickoff_iso.split("T")
                        match_date = parts[0]
                        match_time = parts[1][:5]

            home = teams["home"]
            away = teams["away"]
            home_id, away_id = int(home["id"]), int(away["id"])

            # discard leagues with <2 matches played
            standings = standings_per_league.get(league_id, {})
            hrow = standings.get(home_id, {})
            arow = standings.get(away_id, {})

            games_h = safe_float(hrow.get("all", {}).get("played", 0))
            games_a = safe_float(arow.get("all", {}).get("played", 0))
            if games_h < 2 or games_a < 2:
                continue

            # stats
            hstats = fetch_team_stats(league_id, home_id, SEASON)
            astats = fetch_team_stats(league_id, away_id, SEASON)
            if not hstats or not astats:
                continue

            # build profiles
            hp = build_team_profile(hstats, hrow, league_id, "home")
            ap = build_team_profile(astats, arow, league_id, "away")

            # model
            m = compute_match_model(hp, ap, league_id)

            p_home, p_draw, p_away = m["p_home"], m["p_draw"], m["p_away"]
            p_over, p_under = m["p_over"], m["p_under"]
            draw_score, over_score = m["draw_score"], m["over_score"]

            # apply 2.5% margin
            p_home_b, p_draw_b, p_away_b = apply_margin(p_home, p_draw, p_away, margin=1.025)
            p_over_b = clamp(p_over * 1.02, 0.04, 0.92)
            p_under_b = 1 - p_over_b

            fair_1 = fair_odds(p_home_b)
            fair_x = fair_odds(p_draw_b)
            fair_2 = fair_odds(p_away_b)
            fair_over = fair_odds(p_over_b)
            fair_under = fair_odds(p_under_b)

            engines = LEAGUES[league_id]["engines"]
            engine_tag = (
                "Draw + Over Engine"
                if "draw" in engines and "over" in engines else
                "Draw Engine"
                if "draw" in engines else
                "Over Engine"
                if "over" in engines else
                "Other"
            )

            processed.append({
                "week": week,
                "fixture_id": int(fixture["id"]),
                "date": match_date,
                "time": match_time,
                "league_id": league_id,
                "league": LEAGUES[league_id]["name"],
                "home": home["name"],
                "away": away["name"],
                "model": engine_tag,
                # probabilites
                "draw_prob": round(p_draw,3),
                "over_2_5_prob": round(p_over,3),
                "under_2_5_prob": round(p_under,3),
                # fair odds
                "fair_1": fair_1,
                "fair_x": fair_x,
                "fair_2": fair_2,
                "fair_over_2_5": fair_over,
                "fair_under_2_5": fair_under,
                # scoring engine
                "score_draw": draw_score,
                "score_over": over_score,
            })

        except Exception as e:
            log(f"⚠️ Error: {e}")

    # sort by date/time
    processed.sort(key=lambda x: (x["date"], x["time"]))

    # final report
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "week": week,
        "window": {"from": date_from, "to": date_to},
        "fixtures_analyzed": len(processed),
        "fixtures": processed,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    save_json_cache(TEAM_CACHE_PATH, TEAM_STATS_CACHE)
    save_json_cache(STANDINGS_CACHE_PATH, STANDINGS_CACHE)

    log(f"✔ Thursday Engine v3.5 done → {len(processed)} fixtures")
    log(f"📄 Saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
