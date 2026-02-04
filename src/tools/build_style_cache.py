import os, json, time, datetime, math, unicodedata, re
import requests
from dateutil import parser

API_KEY = os.getenv("FOOTBALL_API_KEY")
BASE = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY} if API_KEY else {}

SEASON = os.getenv("FOOTBALL_SEASON", "").strip()  # π.χ. "2024"
LAST_N = int(os.getenv("STYLE_LAST_N", "20"))
SLEEP_SEC = float(os.getenv("STYLE_SLEEP_SEC", "0.25"))

OUT_PATH = os.getenv("STYLE_CACHE_OUT", "data/team_style_metrics.json")

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
    "Premier League": "England",
    "Championship": "England",
    "Serie A": "Italy",
    "La Liga": "Spain",
    "Bundesliga": "Germany",
    "Ligue 1": "France",
    "Ligue 2": "France",
    "Eredivisie": "Netherlands",
    "Belgium First Division A": "Belgium",
}

DEFAULTS = {"xch_for_90": 1.0, "xch_against_90": 1.0, "tempo_index": 1.0}

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def normalize_team_name(raw: str) -> str:
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

def req(path, params):
    if not API_KEY:
        raise SystemExit("Missing FOOTBALL_API_KEY")
    url = f"{BASE}{path}"
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"{path} status={r.status_code} body={r.text[:200]}")
    return r.json()

def recency_weights(n: int):
    base = [1.0, 0.85, 0.72, 0.61, 0.52, 0.45, 0.40, 0.36, 0.33, 0.30]
    w = base[:max(1, min(len(base), n))]
    s = sum(w)
    return [x / s for x in w]

def fetch_teams(league_id: int, season: str):
    js = req("/teams", {"league": league_id, "season": season})
    out = []
    for item in js.get("response") or []:
        t = (item or {}).get("team") or {}
        tid = t.get("id")
        name = t.get("name")
        if tid and name:
            out.append({"id": int(tid), "name": str(name), "norm": normalize_team_name(str(name))})
    return out

def fetch_last_fixtures(team_id: int, league_id: int, season: str, last_n: int):
    js = req("/fixtures", {
        "team": team_id,
        "league": league_id,
        "season": season,
        "status": "FT",
        "last": last_n
    })
    return js.get("response") or []

def compute_style_from_fixtures(team_id: int, fixtures: list):
    # Weighted avg of goals for/against from last fixtures
    # (If fewer than 5 matches, we still compute but can fallback to defaults.)
    sample = fixtures[:LAST_N]
    if not sample:
        return dict(DEFAULTS), {"matches": 0, "missing": True}

    w = recency_weights(min(len(sample), 10))
    # use up to 10 for stable weights; if more, treat older as low weight lumped
    used = sample[:len(w)]

    gf = ga = 0.0
    m = 0
    for idx, fx in enumerate(used):
        g = (fx.get("goals") or {})
        hg, ag = g.get("home"), g.get("away")
        if hg is None or ag is None:
            continue

        home = ((fx.get("teams") or {}).get("home") or {}).get("id")
        away = ((fx.get("teams") or {}).get("away") or {}).get("id")
        if home is None or away is None:
            continue

        hg, ag = int(hg), int(ag)
        is_home = int(home) == int(team_id)

        gf_i = hg if is_home else ag
        ga_i = ag if is_home else hg

        gf += w[idx] * gf_i
        ga += w[idx] * ga_i
        m += 1

    if m == 0:
        return dict(DEFAULTS), {"matches": 0, "missing": True}

    tempo = gf + ga
    # clamp for sanity
    gf = max(0.2, min(3.5, gf))
    ga = max(0.2, min(3.5, ga))
    tempo = max(0.6, min(6.0, tempo))

    return {"xch_for_90": round(gf, 3), "xch_against_90": round(ga, 3), "tempo_index": round(tempo, 3)}, {"matches": m, "missing": False}

def main():
    if not SEASON:
        raise SystemExit("Set FOOTBALL_SEASON (e.g. 2024)")

    out = {
        "as_of": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
        "countries": {},
        "meta": {
            "league_country_map": dict(LEAGUE_COUNTRY_MAP),
            "defaults": dict(DEFAULTS),
            "source": "api-football:/teams + /fixtures(status=FT,last=N)",
            "note": f"Proxies from last {LAST_N} finished fixtures per team (weighted). Not true pressing data."
        }
    }

    # init buckets
    for c in set(LEAGUE_COUNTRY_MAP.values()):
        out["countries"][c] = {"teams": {}}

    for league_name, league_id in LEAGUES.items():
        country = LEAGUE_COUNTRY_MAP.get(league_name)
        if not country:
            continue

        print(f"== {league_name} ({league_id}) / {country} ==")
        teams = fetch_teams(league_id, SEASON)
        time.sleep(SLEEP_SEC)

        for t in teams:
            fixtures = fetch_last_fixtures(t["id"], league_id, SEASON, LAST_N)
            time.sleep(SLEEP_SEC)

            metrics, dbg = compute_style_from_fixtures(t["id"], fixtures)
            out["countries"][country]["teams"][t["norm"]] = metrics

        print(f"   teams: {len(teams)}")

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote: {OUT_PATH}")

if __name__ == "__main__":
    main()
