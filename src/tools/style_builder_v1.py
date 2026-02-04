# ============================================================
# src/tools/style_builder_v1.py
# STYLE BUILDER v1.1 — PRODUCTION
#
# Goal:
# - Build "data/team_style_metrics.json" used by Thursday engine.
# - Fetch teams per locked league via API-Football.
# - Compute simple recency-weighted proxies from last finished matches:
#     xch_for_90, xch_against_90, tempo_index
#
# Important:
# - If a league returns 0 teams for a season, we auto-try fallback seasons.
# - If a team has insufficient data, we write neutral defaults and mark _computed_neutral.
# ============================================================

import os
import json
import datetime
import unicodedata
import re
import time
import requests

API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_FOOTBALL_KEY} if API_FOOTBALL_KEY else {}

FOOTBALL_SEASON_ENV = os.getenv("FOOTBALL_SEASON", "").strip()
STYLE_LAST_N = int(os.getenv("STYLE_LAST_N", "12"))
STYLE_OUT_PATH = os.getenv("STYLE_OUT_PATH", "data/team_style_metrics.json")

REQUEST_TIMEOUT = int(os.getenv("API_TIMEOUT_SEC", "25"))
REQUEST_SLEEP_MS = int(os.getenv("API_SLEEP_MS", "120"))
MAX_FETCH_RETRIES = int(os.getenv("MAX_FETCH_RETRIES", "2"))

# Same locked leagues (Thursday)
LEAGUES = {
    "Premier League": 39,
    "Championship": 40,
    "Ligue 1": 61,
    "Ligue 2": 62,
    "Bundesliga": 78,
    "Serie A": 135,
    "La Liga": 140,
    "Eredivisie": 88,
    "Belgium First Division A": 144,
}

LEAGUE_COUNTRY_MAP = {
    "Belgium First Division A": "Belgium",
    "Jupiler Pro League": "Belgium",
    "Ligue 1": "France",
    "Ligue 2": "France",
    "Premier League": "England",
    "Championship": "England",
    "Serie A": "Italy",
    "La Liga": "Spain",
    "Bundesliga": "Germany",
    "Eredivisie": "Netherlands",
}

DEFAULT_STYLE = {"xch_for_90": 1.0, "xch_against_90": 1.0, "tempo_index": 1.0}


def log(msg: str):
    print(msg, flush=True)


def safe_float(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _strip_accents(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def normalize_team_name(raw: str) -> str:
    if not raw:
        return ""
    s = _strip_accents(raw).lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    kill = {"fc", "afc", "cf", "sc", "sv", "ssc", "ac", "cd", "ud", "bk", "fk", "if"}
    parts = [p for p in s.split() if p not in kill]
    s = " ".join(parts).strip()

    aliases = {
        "wolverhampton wanderers": "wolves",
        "wolverhampton": "wolves",
        "brighton and hove albion": "brighton",
        "west bromwich albion": "west brom",
        "manchester united": "man utd",
        "manchester city": "man city",
        "newcastle united": "newcastle",
        "tottenham hotspur": "tottenham",
        "bayern munchen": "bayern munich",
        "paris saint germain": "psg",
        "internazionale": "inter",
    }
    return aliases.get(s, s)


def resolve_season(now_utc: datetime.datetime) -> str:
    # If env provided, use it.
    if FOOTBALL_SEASON_ENV:
        return FOOTBALL_SEASON_ENV

    # European season heuristic: Feb 2026 => season 2025 (start year).
    y = now_utc.year
    if now_utc.month < 7:
        y = y - 1
    return str(y)


def _recency_weights(n: int):
    # more recent -> higher weight (max 12)
    base = [1.0, 0.85, 0.72, 0.61, 0.52, 0.45, 0.40, 0.36, 0.33, 0.30, 0.28, 0.26]
    w = base[: max(1, min(len(base), n))]
    s = sum(w)
    return [x / s for x in w]


def api_get(path: str, params: dict):
    if not API_FOOTBALL_KEY:
        raise RuntimeError("Missing FOOTBALL_API_KEY")

    url = f"{API_FOOTBALL_BASE}{path}"
    last_err = None

    for _ in range(MAX_FETCH_RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                time.sleep(0.25)
                continue
            return r.json() or {}
        except Exception as e:
            last_err = str(e)
            time.sleep(0.25)

    log(f"❌ api_get failed: {path} params={params} err={last_err}")
    return {}


def fetch_teams_in_league(league_id: int, season: str):
    js = api_get("/teams", {"league": league_id, "season": season})
    resp = js.get("response") or []
    out = []
    for it in resp:
        t = (it or {}).get("team") or {}
        tid = t.get("id")
        name = t.get("name")
        if tid and name:
            out.append({"id": tid, "name": name, "norm": normalize_team_name(name)})
    return out


def fetch_teams_in_league_with_fallbacks(league_name: str, league_id: int, season: str):
    # try season, then season-1, then season+1 if empty
    tried = []
    seasons_to_try = [season]
    try:
        s_int = int(season)
        seasons_to_try += [str(s_int - 1), str(s_int + 1)]
    except Exception:
        pass

    for s in seasons_to_try:
        tried.append(s)
        teams = fetch_teams_in_league(league_id, s)
        if teams:
            if s != season:
                log(f"   ⚠️ {league_name}: season {season} returned 0 teams; using fallback season={s}")
            return teams, s

    log(f"   ❌ {league_name}: no teams for seasons tried: {tried}")
    return [], season


def fetch_last_finished_fixtures(team_id: int, league_id: int, season: str, last_n: int):
    # Prefer status=FT; if account ignores it, we still filter.
    js = api_get(
        "/fixtures",
        {"team": team_id, "league": league_id, "season": season, "last": max(last_n, 5), "status": "FT"},
    )
    resp = js.get("response") or []
    out = []

    for fx in resp:
        st = (fx.get("fixture", {}).get("status", {}).get("short") or "").upper()
        if st != "FT":
            continue

        goals = fx.get("goals") or {}
        hg = goals.get("home")
        ag = goals.get("away")
        if hg is None or ag is None:
            continue

        teams = fx.get("teams") or {}
        home_id = (teams.get("home") or {}).get("id")
        away_id = (teams.get("away") or {}).get("id")
        is_home = (home_id == team_id)

        gf = int(hg) if is_home else int(ag)
        ga = int(ag) if is_home else int(hg)
        out.append({"gf": gf, "ga": ga})

    return out[:last_n]


def compute_style_from_last(fixtures):
    if not fixtures:
        return None

    m = len(fixtures)
    w = _recency_weights(m)

    gf = ga = tg = 0.0
    for i, row in enumerate(fixtures):
        gf_i = safe_float(row.get("gf"), 0.0) or 0.0
        ga_i = safe_float(row.get("ga"), 0.0) or 0.0
        tg_i = gf_i + ga_i
        gf += w[i] * gf_i
        ga += w[i] * ga_i
        tg += w[i] * tg_i

    return {
        "xch_for_90": round(gf, 4),
        "xch_against_90": round(ga, 4),
        "tempo_index": round(tg, 4),
    }


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    season = resolve_season(now)
    log(f"✅ Style Builder season selected: {season}")

    countries = {}
    totals = {"teams": 0, "filled": 0, "missing": 0}

    for league_name, league_id in LEAGUES.items():
        country = LEAGUE_COUNTRY_MAP.get(league_name)
        if not country:
            continue

        countries.setdefault(country, {"teams": {}})

        log(f"→ League: {league_name} (id={league_id}) season={season}")

        teams, season_used = fetch_teams_in_league_with_fallbacks(league_name, league_id, season)
        log(f"   teams fetched: {len(teams)} (season_used={season_used})")

        for t in teams:
            totals["teams"] += 1
            last_fx = fetch_last_finished_fixtures(t["id"], league_id, season_used, STYLE_LAST_N)
            style = compute_style_from_last(last_fx)

            if style is None:
                totals["missing"] += 1
                style = {**DEFAULT_STYLE, "_computed_neutral": True}
            else:
                totals["filled"] += 1

            countries[country]["teams"][t["norm"]] = style

            # gentle rate-limit
            time.sleep(max(0.0, REQUEST_SLEEP_MS / 1000.0))

    out = {
        "as_of": now.date().isoformat(),
        "countries": countries,
        "meta": {
            "league_country_map": LEAGUE_COUNTRY_MAP,
            "defaults": DEFAULT_STYLE,
            "builder": {
                "season": season,
                "last_n_finished": STYLE_LAST_N,
                "note": "Proxies from last finished fixtures: goals_for, goals_against, total_goals (tempo).",
            },
            "counts": totals,
        },
    }

    os.makedirs(os.path.dirname(STYLE_OUT_PATH) or ".", exist_ok=True)
    with open(STYLE_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    log(f"✅ Wrote: {STYLE_OUT_PATH}")
    log(f"   totals: {totals}")


if __name__ == "__main__":
    main()
