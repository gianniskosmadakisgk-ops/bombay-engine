import os
import json
import math
import requests
import datetime
import unicodedata
from dateutil import parser

# ============================================================
#  THURSDAY ENGINE v3 (Balanced Model, real team stats)
#  - Fixtures & team stats από API-FOOTBALL
#  - Poisson + league adjustments (balanced)
#  - Fair odds = 1 / prob (ΑΠΑΡΑΛΛΑΧΤΟ)
#  - Offered odds από TheOddsAPI (αν USE_ODDS_API=true)
#  - Γράφει logs/thursday_report_v3.json
#
#  FIX v3.2:
#   - TheOddsAPI: Z time format + fallback retries if 0 events
#   - Logging: status/events/remaining
#   - Matching: same normalization + swapped home/away support
# ============================================================

API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

FOOTBALL_SEASON = os.getenv("FOOTBALL_SEASON", "2025")
HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

USE_ODDS_API = os.getenv("USE_ODDS_API", "true").lower() == "true"
WINDOW_HOURS = int(os.getenv("WINDOW_HOURS", "72"))

LEAGUES = {
    "Premier League": 39,
    "Championship": 40,
    "Ligue 1": 61,
    "Ligue 2": 62,
    "Bundesliga": 78,
    "Serie A": 135,
    "Serie B": 136,
    "La Liga": 140,
    "Liga Portugal 1": 94,
}

LEAGUE_TO_SPORT = {
    "Premier League": "soccer_epl",
    "Championship": "soccer_efl_champ",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Serie B": "soccer_italy_serie_b",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
    "Ligue 2": "soccer_france_ligue_two",
    "Liga Portugal 1": "soccer_portugal_primeira_liga",
}

TEAM_STATS_CACHE = {}


def log(msg: str):
    print(msg, flush=True)


def implied(p: float):
    """FAIR = 1 / prob (spec) — ΜΗΝ ΑΛΛΑΞΕΙ."""
    return 1.0 / p if p and p > 0 else None


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

    out = []
    for ch in s:
        if ch.isalnum() or ch.isspace():
            out.append(ch)
    s = "".join(out)
    s = " ".join(s.split())

    # kill common tokens anywhere
    kill = {"fc", "afc", "cf", "sc", "sv", "ssc", "ac", "cd", "ud"}
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
        "borussia dortmund": "dortmund",
        "bayer leverkusen": "leverkusen",
        "paris saint germain": "psg",
    }

    return aliases.get(s, s)


def iso_z(dt: datetime.datetime) -> str:
    dt = dt.astimezone(datetime.timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


# ------------------------- FIXTURES -------------------------
def fetch_fixtures(league_id: int, league_name: str):
    if not API_FOOTBALL_KEY:
        log("⚠️ Missing FOOTBALL_API_KEY – NO fixtures will be fetched!")
        return []

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"league": league_id, "season": FOOTBALL_SEASON}

    try:
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=20).json()
    except Exception as e:
        log(f"⚠️ Error fetching fixtures for {league_name}: {e}")
        return []

    if not r.get("response"):
        log(f"⚠️ No fixtures response for league {league_name}")
        return []

    out = []
    now = datetime.datetime.now(datetime.timezone.utc)

    for fx in r["response"]:
        if fx["fixture"]["status"]["short"] != "NS":
            continue

        dt = parser.isoparse(fx["fixture"]["date"]).astimezone(datetime.timezone.utc)
        diff_hours = (dt - now).total_seconds() / 3600.0
        if not (0 <= diff_hours <= WINDOW_HOURS):
            continue

        home = fx["teams"]["home"]
        away = fx["teams"]["away"]

        out.append(
            {
                "id": fx["fixture"]["id"],
                "league_id": league_id,
                "league_name": league_name,
                "home": home["name"],
                "away": away["name"],
                "home_id": home["id"],
                "away_id": away["id"],
                "home_norm": normalize_team_name(home["name"]),
                "away_norm": normalize_team_name(away["name"]),
                "date_raw": fx["fixture"]["date"],
            }
        )

    log(f"→ {league_name}: {len(out)} fixtures within window")
    return out


# ------------------------- TEAM STATS -------------------------
def fetch_team_recent_stats(team_id: int, league_id: int):
    ck = (team_id, league_id)
    if ck in TEAM_STATS_CACHE:
        return TEAM_STATS_CACHE[ck]

    if not API_FOOTBALL_KEY:
        TEAM_STATS_CACHE[ck] = {}
        return TEAM_STATS_CACHE[ck]

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"team": team_id, "league": league_id, "season": FOOTBALL_SEASON, "last": 5}

    try:
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=20).json()
    except Exception as e:
        log(f"⚠️ Error fetching team stats team_id={team_id}: {e}")
        TEAM_STATS_CACHE[ck] = {}
        return TEAM_STATS_CACHE[ck]

    if not r.get("response"):
        TEAM_STATS_CACHE[ck] = {}
        return TEAM_STATS_CACHE[ck]

    gf = ga = m = 0
    for fx in r["response"]:
        m += 1
        g_home = fx["goals"]["home"] or 0
        g_away = fx["goals"]["away"] or 0
        is_home = fx["teams"]["home"]["id"] == team_id
        if is_home:
            gf += g_home
            ga += g_away
        else:
            gf += g_away
            ga += g_home

    stats = {
        "matches_count": m,
        "avg_goals_for": (gf / m) if m else None,
        "avg_goals_against": (ga / m) if m else None,
    }
    TEAM_STATS_CACHE[ck] = stats
    return stats


def fetch_league_baselines(league_id: int):
    overrides = {
        39: {"avg_goals_per_match": 2.9, "avg_draw_rate": 0.24, "avg_over25_rate": 0.58},
        40: {"avg_goals_per_match": 2.5, "avg_draw_rate": 0.28, "avg_over25_rate": 0.52},
        78: {"avg_goals_per_match": 3.1, "avg_draw_rate": 0.25, "avg_over25_rate": 0.60},
        135: {"avg_goals_per_match": 2.5, "avg_draw_rate": 0.30, "avg_over25_rate": 0.52},
        140: {"avg_goals_per_match": 2.6, "avg_draw_rate": 0.27, "avg_over25_rate": 0.55},
    }
    base = {"avg_goals_per_match": 2.6, "avg_draw_rate": 0.26, "avg_over25_rate": 0.55, "home_advantage": 0.18}
    if league_id in overrides:
        base.update(overrides[league_id])
    return base


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def compute_expected_goals(home_stats: dict, away_stats: dict, league_baseline: dict):
    league_avg = safe_float(league_baseline.get("avg_goals_per_match"), 2.6) or 2.6
    home_adv = safe_float(league_baseline.get("home_advantage"), 0.18) or 0.18

    def savg(stats, key, default):
        v = safe_float(stats.get(key), None)
        return default if v is None else v

    denom = max(0.5, league_avg / 2)
    home_for = savg(home_stats, "avg_goals_for", denom)
    home_against = savg(home_stats, "avg_goals_against", denom)
    away_for = savg(away_stats, "avg_goals_for", denom)
    away_against = savg(away_stats, "avg_goals_against", denom)

    home_att = home_for / denom
    away_att = away_for / denom
    home_defw = home_against / denom
    away_defw = away_against / denom

    lam_h = denom * (0.6 * home_att + 0.4 * away_defw)
    lam_a = denom * (0.6 * away_att + 0.4 * home_defw)

    lam_h *= (1.0 + home_adv)

    lam_h = max(0.2, min(3.5, lam_h))
    lam_a = max(0.2, min(3.5, lam_a))
    return lam_h, lam_a


def compute_probabilities(lambda_home: float, lambda_away: float, context: dict):
    max_goals = 7
    ph = pd = pa = 0.0
    po = 0.0

    for i in range(max_goals + 1):
        p_i = poisson_pmf(i, lambda_home)
        for j in range(max_goals + 1):
            p_j = poisson_pmf(j, lambda_away)
            p = p_i * p_j
            if i > j:
                ph += p
            elif i == j:
                pd += p
            else:
                pa += p
            if i + j >= 3:
                po += p

    tot = ph + pd + pa
    if tot > 0:
        ph, pd, pa = ph / tot, pd / tot, pa / tot
    else:
        ph, pd, pa = 0.4, 0.2, 0.4

    po = min(0.99, max(0.01, po))
    pu = 1.0 - po

    lb = context.get("league_baseline", {}) or {}
    league_draw = safe_float(lb.get("avg_draw_rate"), None)
    league_over = safe_float(lb.get("avg_over25_rate"), None)

    if league_draw is not None:
        alpha_d = 0.25
        pd = (1 - alpha_d) * pd + alpha_d * league_draw
        pd = min(0.35, max(0.18, pd))
        rest = max(1e-6, ph + pa)
        scale = (1.0 - pd) / rest
        ph *= scale
        pa *= scale

    if league_over is not None:
        alpha_o = 0.25
        po = (1 - alpha_o) * po + alpha_o * league_over
        po = min(0.80, max(0.40, po))
        pu = 1.0 - po

    ph = min(0.80, max(0.10, ph))
    pa = min(0.80, max(0.10, pa))
    pd = min(0.35, max(0.15, pd))
    s = ph + pd + pa
    ph, pd, pa = ph / s, pd / s, pa / s

    po = min(0.85, max(0.35, po))
    pu = 1.0 - po

    return {"home_prob": ph, "draw_prob": pd, "away_prob": pa, "over_2_5_prob": po, "under_2_5_prob": pu}


# ------------------------- ODDS (TheOddsAPI) -------------------------
def _odds_request(sport_key: str, params: dict):
    url = f"{ODDS_BASE_URL}/{sport_key}/odds"
    try:
        res = requests.get(url, params=params, timeout=25)
        rem = res.headers.get("x-requests-remaining")
        used = res.headers.get("x-requests-used")
        log(f"   TheOddsAPI status={res.status_code} remaining={rem} used={used}")
        if res.status_code != 200:
            log(f"   body={res.text[:200]}")
            return []
        data = res.json()
        return data or []
    except Exception as e:
        log(f"   TheOddsAPI request error: {e}")
        return []


def fetch_odds_for_league(league_name: str, window_from: datetime.datetime, window_to: datetime.datetime):
    if not USE_ODDS_API:
        return []
    if not ODDS_API_KEY:
        log("⚠️ Missing ODDS_API_KEY – skipping odds")
        return []

    sport_key = LEAGUE_TO_SPORT.get(league_name)
    if not sport_key:
        return []

    base_params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu,uk",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    # Attempt 1: with time window (Z format)
    params1 = dict(base_params)
    params1["commenceTimeFrom"] = iso_z(window_from)
    params1["commenceTimeTo"] = iso_z(window_to)

    log(f"→ Odds fetch {league_name} [{sport_key}] (windowed)")
    data = _odds_request(sport_key, params1)
    if data:
        return data

    # Attempt 2: no window (some feeds fail with commenceTime filters)
    log(f"→ Odds fetch {league_name} [{sport_key}] (no-window fallback)")
    data = _odds_request(sport_key, base_params)
    if data:
        return data

    # Attempt 3: eu only (fallback)
    log(f"→ Odds fetch {league_name} [{sport_key}] (eu-only fallback)")
    params3 = dict(base_params)
    params3["regions"] = "eu"
    return _odds_request(sport_key, params3)


def build_odds_index(odds_data):
    index = {}

    for ev in odds_data or []:
        home_raw = ev.get("home_team", "")
        away_raw = ev.get("away_team", "")

        h = normalize_team_name(home_raw)
        a = normalize_team_name(away_raw)

        best_home = best_draw = best_away = None
        best_over = best_under = None

        for bm in ev.get("bookmakers", []) or []:
            for m in bm.get("markets", []) or []:
                mk = (m.get("key") or "").lower()

                if mk == "h2h":
                    for o in m.get("outcomes", []) or []:
                        name_norm = normalize_team_name(o.get("name", ""))
                        price = safe_float(o.get("price"), None)
                        if price is None or price <= 1.0:
                            continue
                        if name_norm == h:
                            best_home = max(best_home or 0.0, price)
                        elif name_norm == a:
                            best_away = max(best_away or 0.0, price)
                        elif name_norm in ["draw", "x", "tie"]:
                            best_draw = max(best_draw or 0.0, price)

                elif mk == "totals":
                    for o in m.get("outcomes", []) or []:
                        price = safe_float(o.get("price"), None)
                        if price is None or price <= 1.0:
                            continue
                        name = (o.get("name") or "").lower()
                        point = safe_float(o.get("point"), None)
                        if point is not None and abs(point - 2.5) > 1e-6:
                            continue
                        if "over" in name and ("2.5" in name or point == 2.5):
                            best_over = max(best_over or 0.0, price)
                        elif "under" in name and ("2.5" in name or point == 2.5):
                            best_under = max(best_under or 0.0, price)

        # forward
        index[(h, a)] = {"home": best_home, "draw": best_draw, "away": best_away, "over": best_over, "under": best_under}
        # reverse (swap 1/2)
        index[(a, h)] = {"home": best_away, "draw": best_draw, "away": best_home, "over": best_over, "under": best_under}

    return index


def build_fixture_blocks():
    fixtures_out = []
    now = datetime.datetime.now(datetime.timezone.utc)
    to_dt = now + datetime.timedelta(hours=WINDOW_HOURS)

    log(f"Using FOOTBALL_SEASON={FOOTBALL_SEASON}")
    log(f"Window: next {WINDOW_HOURS} hours")
    log(f"USE_ODDS_API={USE_ODDS_API}")

    if not API_FOOTBALL_KEY:
        log("❌ FOOTBALL_API_KEY is missing. Aborting fixture fetch.")
        return []

    all_fixtures = []
    for lg_name, lg_id in LEAGUES.items():
        all_fixtures.extend(fetch_fixtures(lg_id, lg_name))

    log(f"Total raw fixtures collected: {len(all_fixtures)}")

    odds_index_global = {}
    if USE_ODDS_API:
        total_events = 0
        for lg_name in LEAGUES.keys():
            odds_data = fetch_odds_for_league(lg_name, now, to_dt)
            total_events += len(odds_data or [])
            odds_index_global.update(build_odds_index(odds_data))
        log(f"Odds events fetched: {total_events}, index keys: {len(odds_index_global)}")
    else:
        log("⚠️ USE_ODDS_API=False → skipping TheOddsAPI.")

    for fx in all_fixtures:
        league_id = fx["league_id"]
        league_name = fx["league_name"]

        league_baseline = fetch_league_baselines(league_id)
        home_stats = fetch_team_recent_stats(fx["home_id"], league_id)
        away_stats = fetch_team_recent_stats(fx["away_id"], league_id)

        lam_h, lam_a = compute_expected_goals(home_stats, away_stats, league_baseline)

        probs = compute_probabilities(lam_h, lam_a, {"league_baseline": league_baseline})

        p_home = probs["home_prob"]
        p_draw = probs["draw_prob"]
        p_away = probs["away_prob"]
        p_over = probs["over_2_5_prob"]
        p_under = probs["under_2_5_prob"]

        fair_1 = implied(p_home)
        fair_x = implied(p_draw)
        fair_2 = implied(p_away)
        fair_over = implied(p_over)
        fair_under = implied(p_under)

        offered = odds_index_global.get((fx["home_norm"], fx["away_norm"]), {}) or {}

        dt = parser.isoparse(fx["date_raw"]).astimezone(datetime.timezone.utc)
        fixtures_out.append(
            {
                "fixture_id": fx["id"],
                "date": dt.date().isoformat(),
                "time": dt.strftime("%H:%M"),
                "league_id": league_id,
                "league": league_name,
                "home": fx["home"],
                "away": fx["away"],
                "model": "bombay_balanced_v1",
                "lambda_home": round(lam_h, 3),
                "lambda_away": round(lam_a, 3),

                "fair_1": fair_1,
                "fair_x": fair_x,
                "fair_2": fair_2,
                "fair_over_2_5": fair_over,
                "fair_under_2_5": fair_under,

                "draw_prob": round(p_draw, 3),
                "over_2_5_prob": round(p_over, 3),
                "under_2_5_prob": round(p_under, 3),

                "offered_1": offered.get("home"),
                "offered_x": offered.get("draw"),
                "offered_2": offered.get("away"),
                "offered_over_2_5": offered.get("over"),
                "offered_under_2_5": offered.get("under"),
            }
        )

    log(f"Thursday fixtures_out: {len(fixtures_out)}")
    return fixtures_out


def main():
    fixtures = build_fixture_blocks()

    now = datetime.datetime.now(datetime.timezone.utc)
    to_dt = now + datetime.timedelta(hours=WINDOW_HOURS)

    out = {
        "generated_at": now.isoformat(),
        "window": {"from": now.date().isoformat(), "to": to_dt.date().isoformat(), "hours": WINDOW_HOURS},
        "fixtures_total": len(fixtures),
        "fixtures": fixtures,
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/thursday_report_v3.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    log(f"✅ Thursday v3 READY. Fixtures: {len(fixtures)}")


if __name__ == "__main__":
    main()
