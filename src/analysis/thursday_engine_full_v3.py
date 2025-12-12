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
#  FIXES v3.1:
#   - Better normalization (diacritics/aliases/tokens)
#   - Odds lookup supports swapped home/away safely
#   - TheOddsAPI: wider regions + commenceTimeFrom/To window
# ============================================================

# ------------------------- CONFIG / KEYS -------------------------
API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

FOOTBALL_SEASON = os.getenv("FOOTBALL_SEASON", "2025")

HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

USE_ODDS_API = os.getenv("USE_ODDS_API", "true").lower() == "true"

# 3 ημέρες (72 ώρες)
WINDOW_HOURS = int(os.getenv("WINDOW_HOURS", "72"))

# ------------------------- LEAGUES -------------------------
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

# ------------------------- LEAGUE → SPORT KEY (TheOddsAPI) -------------------------
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

# ------------------------- HELPERS -------------------------
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
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def normalize_team_name(raw: str) -> str:
    """
    Normalization για matching με TheOddsAPI.
    Στόχος: ΠΕΡΙΣΣΟΤΕΡΑ matches, όχι "τέλειο NLP".

    Rules:
    - lower, strip accents
    - replace & -> and
    - remove punctuation
    - remove common tokens (fc, afc, cf, sc, sv, 1., etc)
    - collapse spaces
    - aliases
    """
    if not raw:
        return ""

    s = _strip_accents(raw).lower().strip()
    s = s.replace("&", "and")

    # keep alnum + spaces only
    out = []
    for ch in s:
        if ch.isalnum() or ch.isspace():
            out.append(ch)
    s = "".join(out)
    s = " ".join(s.split())

    # remove leading numeric tokens (π.χ. "1 fc koln" -> "fc koln")
    while s and s.split()[0].isdigit():
        s = " ".join(s.split()[1:])

    # remove common trailing tokens
    kill_tokens = {"fc", "afc", "cf", "sc", "sv", "ssc", "ac", "cd", "ud"}
    parts = s.split()
    parts = [p for p in parts if p not in kill_tokens]
    s = " ".join(parts).strip()

    # aliases (add as you discover pain)
    aliases = {
        # england
        "wolverhampton wanderers": "wolves",
        "wolverhampton": "wolves",
        "brighton and hove albion": "brighton",
        "west bromwich albion": "west brom",
        "manchester united": "man utd",
        "manchester city": "man city",
        "newcastle united": "newcastle",
        "nottingham forest": "nottingham forest",
        "tottenham hotspur": "tottenham",
        # germany
        "bayern munchen": "bayern munich",
        "borussia monchengladbach": "monchengladbach",
        "rb leipzig": "leipzig",
        "1 fc koln": "koln",
        "vfb stuttgart": "stuttgart",
        "vfl wolfsburg": "wolfsburg",
        "eintracht frankfurt": "frankfurt",
        "borussia dortmund": "dortmund",
        "bayer leverkusen": "leverkusen",
        "sc freiburg": "freiburg",
        # france
        "paris saint germain": "psg",
        "stade brestois 29": "brest",
        "red star fc 93": "red star",
    }

    return aliases.get(s, s)


# ------------------------- FIXTURES -------------------------
def fetch_fixtures(league_id: int, league_name: str):
    """Fixtures από API-FOOTBALL για τη λίγκα, μέσα στο WINDOW_HOURS."""
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
        status_short = fx["fixture"]["status"]["short"]
        if status_short != "NS":
            continue

        dt = parser.isoparse(fx["fixture"]["date"]).astimezone(datetime.timezone.utc)
        diff_hours = (dt - now).total_seconds() / 3600.0
        if not (0 <= diff_hours <= WINDOW_HOURS):
            continue

        home_team = fx["teams"]["home"]
        away_team = fx["teams"]["away"]

        home_name = home_team["name"]
        away_name = away_team["name"]
        home_id = home_team["id"]
        away_id = away_team["id"]

        out.append(
            {
                "id": fx["fixture"]["id"],
                "league_id": league_id,
                "league_name": league_name,
                "home": home_name,
                "away": away_name,
                "home_id": home_id,
                "away_id": away_id,
                "home_norm": normalize_team_name(home_name),
                "away_norm": normalize_team_name(away_name),
                "date_raw": fx["fixture"]["date"],
                "timestamp_utc": dt.isoformat(),
            }
        )

    log(f"→ {league_name}: {len(out)} fixtures within window")
    return out


# ------------------------- TEAM STATS (last 5) -------------------------
def fetch_team_recent_stats(team_id: int, league_id: int):
    cache_key = (team_id, league_id)
    if cache_key in TEAM_STATS_CACHE:
        return TEAM_STATS_CACHE[cache_key]

    if not API_FOOTBALL_KEY:
        TEAM_STATS_CACHE[cache_key] = {}
        return TEAM_STATS_CACHE[cache_key]

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {
        "team": team_id,
        "league": league_id,
        "season": FOOTBALL_SEASON,
        "last": 5,
    }

    try:
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=20).json()
    except Exception as e:
        log(f"⚠️ Error fetching team stats team_id={team_id}: {e}")
        TEAM_STATS_CACHE[cache_key] = {}
        return TEAM_STATS_CACHE[cache_key]

    if not r.get("response"):
        TEAM_STATS_CACHE[cache_key] = {}
        return TEAM_STATS_CACHE[cache_key]

    goals_for = 0
    goals_against = 0
    matches = 0

    for fx in r["response"]:
        matches += 1
        g_home = fx["goals"]["home"] or 0
        g_away = fx["goals"]["away"] or 0

        is_home = fx["teams"]["home"]["id"] == team_id
        if is_home:
            goals_for += g_home
            goals_against += g_away
        else:
            goals_for += g_away
            goals_against += g_home

    avg_for = (goals_for / matches) if matches else None
    avg_against = (goals_against / matches) if matches else None

    stats = {
        "matches_count": matches,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "avg_goals_for": avg_for,
        "avg_goals_against": avg_against,
    }

    TEAM_STATS_CACHE[cache_key] = stats
    return stats


# ------------------------- LEAGUE BASELINES -------------------------
def fetch_league_baselines(league_id: int):
    league_overrides = {
        39: {"avg_goals_per_match": 2.9, "avg_draw_rate": 0.24, "avg_over25_rate": 0.58},
        40: {"avg_goals_per_match": 2.5, "avg_draw_rate": 0.28, "avg_over25_rate": 0.52},
        78: {"avg_goals_per_match": 3.1, "avg_draw_rate": 0.25, "avg_over25_rate": 0.60},
        135: {"avg_goals_per_match": 2.5, "avg_draw_rate": 0.30, "avg_over25_rate": 0.52},
        140: {"avg_goals_per_match": 2.6, "avg_draw_rate": 0.27, "avg_over25_rate": 0.55},
    }

    base = {
        "avg_goals_per_match": 2.6,
        "avg_draw_rate": 0.26,
        "avg_over25_rate": 0.55,
        "home_advantage": 0.18,
    }

    if league_id in league_overrides:
        base.update(league_overrides[league_id])

    return base


# ------------------------- POISSON -------------------------
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


# ------------------------- MODEL: EXPECTED GOALS -------------------------
def compute_expected_goals(home_stats: dict, away_stats: dict, league_baseline: dict):
    """
    Κρατάμε τη λογική σου (balanced last5) — δεν αλλάζουμε "φιλοσοφία".
    """
    league_avg_goals = safe_float(league_baseline.get("avg_goals_per_match"), 2.6) or 2.6
    home_advantage = safe_float(league_baseline.get("home_advantage"), 0.18) or 0.18

    def safe_avg(stats: dict, key: str, default: float):
        v = safe_float(stats.get(key), None)
        return default if v is None else v

    home_for = safe_avg(home_stats, "avg_goals_for", league_avg_goals * 0.5)
    home_against = safe_avg(home_stats, "avg_goals_against", league_avg_goals * 0.5)
    away_for = safe_avg(away_stats, "avg_goals_for", league_avg_goals * 0.5)
    away_against = safe_avg(away_stats, "avg_goals_against", league_avg_goals * 0.5)

    denom = max(0.5, league_avg_goals / 2)

    home_attack_strength = home_for / denom
    away_attack_strength = away_for / denom

    home_def_weakness = home_against / denom
    away_def_weakness = away_against / denom

    lambda_home = denom * (0.6 * home_attack_strength + 0.4 * away_def_weakness)
    lambda_away = denom * (0.6 * away_attack_strength + 0.4 * home_def_weakness)

    lambda_home *= (1.0 + home_advantage)

    lambda_home = max(0.2, min(3.5, lambda_home))
    lambda_away = max(0.2, min(3.5, lambda_away))

    return lambda_home, lambda_away


# ------------------------- MODEL: PROBABILITIES -------------------------
def compute_probabilities(lambda_home: float, lambda_away: float, context: dict):
    """
    1X2 + O/U 2.5 από Poisson matrix.
    League blending μικρό weight.
    """
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

    total_1x2 = ph + pd + pa
    if total_1x2 <= 0:
        ph, pd, pa = 0.4, 0.2, 0.4
    else:
        ph /= total_1x2
        pd /= total_1x2
        pa /= total_1x2

    po = min(0.99, max(0.01, po))
    pu = 1.0 - po

    league_baseline = context.get("league_baseline", {}) or {}
    league_draw = safe_float(league_baseline.get("avg_draw_rate"), None)
    league_over = safe_float(league_baseline.get("avg_over25_rate"), None)

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
    ph /= s
    pd /= s
    pa /= s

    po = min(0.85, max(0.35, po))
    pu = 1.0 - po

    return {
        "home_prob": ph,
        "draw_prob": pd,
        "away_prob": pa,
        "over_2_5_prob": po,
        "under_2_5_prob": pu,
    }


# ------------------------- ODDS (TheOddsAPI) -------------------------
def fetch_odds_for_league(league_name: str, window_from: datetime.datetime, window_to: datetime.datetime):
    """
    Τραβάει odds *μία φορά* για τη λίγκα.
    FIX: regions widened + time window.
    """
    if not USE_ODDS_API:
        return []

    sport_key = LEAGUE_TO_SPORT.get(league_name)
    if not sport_key:
        return []

    if not ODDS_API_KEY:
        log("⚠️ Missing ODDS_API_KEY – skipping odds")
        return []

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu,uk",              # περισσότερη κάλυψη από σκέτο eu
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
        # σφίγγουμε στο ίδιο window με τα fixtures
        "commenceTimeFrom": window_from.replace(microsecond=0).isoformat(),
        "commenceTimeTo": window_to.replace(microsecond=0).isoformat(),
    }

    try:
        url = f"{ODDS_BASE_URL}/{sport_key}/odds"
        res = requests.get(url, params=params, timeout=25)
        if res.status_code != 200:
            log(f"⚠️ Odds error [{league_name}] status={res.status_code} body={res.text[:200]}")
            return []
        return res.json()
    except Exception as e:
        log(f"⚠️ Odds request error for {league_name}: {e}")
        return []


def build_odds_index(odds_data):
    """
    Builds BOTH:
      - directional index[(home_norm, away_norm)] = {home,draw,away,over,under}
      - and also supports swapped order by inserting the reverse key with swapped home/away prices.

    Αυτό αυξάνει matches ΤΡΕΛΑ, γιατί home/away naming μερικές φορές δεν είναι aligned.
    """
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
        index[(h, a)] = {
            "home": best_home,
            "draw": best_draw,
            "away": best_away,
            "over": best_over,
            "under": best_under,
        }

        # reverse (swap home/away)
        index[(a, h)] = {
            "home": best_away,
            "draw": best_draw,
            "away": best_home,
            "over": best_over,
            "under": best_under,
        }

    return index


# ------------------------- MAIN PIPELINE -------------------------
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

    # 1) Fixtures
    all_fixtures = []
    for lg_name, lg_id in LEAGUES.items():
        fx_list = fetch_fixtures(lg_id, lg_name)
        all_fixtures.extend(fx_list)

    log(f"Total raw fixtures collected: {len(all_fixtures)}")

    # 2) Odds index
    odds_index_global = {}
    if USE_ODDS_API:
        total_events = 0
        for lg_name in LEAGUES.keys():
            odds_data = fetch_odds_for_league(lg_name, now, to_dt)
            total_events += len(odds_data or [])
            league_index = build_odds_index(odds_data)
            odds_index_global.update(league_index)
        log(f"Odds events fetched: {total_events}, index keys: {len(odds_index_global)}")
    else:
        log("⚠️ USE_ODDS_API=False → skipping TheOddsAPI.")

    # 3) Loop fixtures
    for fx in all_fixtures:
        home_name = fx["home"]
        away_name = fx["away"]
        league_name = fx["league_name"]
        league_id = fx["league_id"]
        home_id = fx["home_id"]
        away_id = fx["away_id"]

        home_norm = fx["home_norm"]
        away_norm = fx["away_norm"]

        league_baseline = fetch_league_baselines(league_id)

        home_stats = fetch_team_recent_stats(home_id, league_id)
        away_stats = fetch_team_recent_stats(away_id, league_id)

        lambda_home, lambda_away = compute_expected_goals(home_stats, away_stats, league_baseline)

        context = {
            "league_name": league_name,
            "league_id": league_id,
            "home": home_name,
            "away": away_name,
            "league_baseline": league_baseline,
            "home_stats": home_stats,
            "away_stats": away_stats,
        }
        probs = compute_probabilities(lambda_home, lambda_away, context)

        p_home = probs.get("home_prob")
        p_draw = probs.get("draw_prob")
        p_away = probs.get("away_prob")
        p_over = probs.get("over_2_5_prob")
        p_under = probs.get("under_2_5_prob")

        # FAIR (spec) — ΜΗΝ ΑΛΛΑΞΕΙ
        fair_1 = implied(p_home)
        fair_x = implied(p_draw)
        fair_2 = implied(p_away)
        fair_over = implied(p_over)
        fair_under = implied(p_under)

        offered = odds_index_global.get((home_norm, away_norm), {}) or {}
        off_home = offered.get("home")
        off_draw = offered.get("draw")
        off_away = offered.get("away")
        off_over = offered.get("over")
        off_under = offered.get("under")

        dt = parser.isoparse(fx["date_raw"]).astimezone(datetime.timezone.utc)
        date_str = dt.date().isoformat()
        time_str = dt.strftime("%H:%M")

        fixtures_out.append(
            {
                "fixture_id": fx["id"],
                "date": date_str,
                "time": time_str,
                "league_id": league_id,
                "league": league_name,
                "home": home_name,
                "away": away_name,
                "model": "bombay_balanced_v1",
                "lambda_home": round(lambda_home, 3),
                "lambda_away": round(lambda_away, 3),

                "fair_1": fair_1,
                "fair_x": fair_x,
                "fair_2": fair_2,
                "fair_over_2_5": fair_over,
                "fair_under_2_5": fair_under,

                "draw_prob": round(p_draw, 3) if isinstance(p_draw, (int, float)) else None,
                "over_2_5_prob": round(p_over, 3) if isinstance(p_over, (int, float)) else None,
                "under_2_5_prob": round(p_under, 3) if isinstance(p_under, (int, float)) else None,

                "offered_1": off_home,
                "offered_x": off_draw,
                "offered_2": off_away,
                "offered_over_2_5": off_over,
                "offered_under_2_5": off_under,
            }
        )

    log(f"Thursday fixtures_out: {len(fixtures_out)}")
    return fixtures_out


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    to_dt = now + datetime.timedelta(hours=WINDOW_HOURS)

    fixtures = build_fixture_blocks()

    out = {
        "generated_at": now.isoformat(),
        "window": {
            "from": now.date().isoformat(),
            "to": to_dt.date().isoformat(),
            "hours": WINDOW_HOURS,
        },
        "fixtures_total": len(fixtures),
        "fixtures": fixtures,
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/thursday_report_v3.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    log(f"✅ Thursday v3 READY. Fixtures: {len(fixtures)}")


if __name__ == "__main__":
    main()
