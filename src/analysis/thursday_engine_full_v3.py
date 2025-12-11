import os
import json
import math
import unicodedata
import datetime
from typing import Dict, Tuple

import requests
from dateutil import parser

# ============================================================
#  THURSDAY ENGINE v4 — FULL MODEL + ODDS MAPPING
#  - Fixtures από API-FOOTBALL
#  - Πραγματικά team stats από /teams/statistics
#  - Poisson goal model -> P(1/X/2, Over/Under 2.5)
#  - FAIR = 1 / prob (όπως στο master spec)
#  - Offered odds από TheOddsAPI με καλύτερο name-mapping
#  - Γράφει logs/thursday_report_v3.json (backwards compatible)
# ============================================================

# ------------------------- CONFIG / KEYS -------------------------
API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

FOOTBALL_SEASON = os.getenv("FOOTBALL_SEASON", "2025")

HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

USE_ODDS_API = os.getenv("USE_ODDS_API", "true").lower() == "true"

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

# 3 μέρες (72 ώρες)
WINDOW_HOURS = 72

# ------------------------- LEAGUE → SPORT KEY (TheOddsAPI) -------------------------
LEAGUE_TO_SPORT = {
    "Premier League": "soccer_epl",
    "Championship": "soccer_efl_champ",
    "La Liga": "soccer_spain_la_liga",
    "La Liga 2": "soccer_spain_segunda_division",
    "Serie A": "soccer_italy_serie_a",
    "Serie B": "soccer_italy_serie_b",
    "Bundesliga": "soccer_germany_bundesliga",
    "Bundesliga 2": "soccer_germany_bundesliga2",
    "Ligue 1": "soccer_france_ligue_one",
    "Ligue 2": "soccer_france_ligue_two",
    "Liga Portugal 1": "soccer_portugal_primeira_liga",
    "Swiss Super League": "soccer_switzerland_superleague",
    "Eredivisie": "soccer_netherlands_eredivisie",
    "Jupiler Pro League": "soccer_belgium_first_div",
    "Superliga": "soccer_denmark_superliga",
    "Allsvenskan": "soccer_sweden_allsvenskan",
    "Eliteserien": "soccer_norway_eliteserien",
    "Argentina Primera": "soccer_argentina_primera_division",
    "Brazil Serie A": "soccer_brazil_serie_a",
}

# ============================================================
# HELPERS
# ============================================================

def log(msg: str):
    print(msg, flush=True)


def implied(p: float):
    """FAIR = 1 / prob (όπως στο spec)."""
    return 1.0 / p if p and p > 0 else None


def normalize_team_name(s: str) -> str:
    """
    Normalisation για να ταιριάζουν API-Football vs TheOddsAPI.

    Παραδείγματα:
      "West Bromwich Albion FC" → "west bromwich albion"
      "Stoke City"              → "stoke"
    """
    if not s:
        return ""

    # Lowercase & strip accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()

    # Πετάμε κοινά suffixes
    TRASH = [
        " fc", " afc", " cf", " cfc", " sc", " ac",
        " u19", " u21", " u23",
        " women", " wfc",
    ]
    for t in TRASH:
        if s.endswith(t):
            s = s[: -len(t)]

    # Πετάμε “football club” κλπ
    for t in [" football club", " club", " soccer club"]:
        s = s.replace(t, "")

    # Κρατάμε μόνο γράμματα/αριθμούς/κενά
    cleaned = []
    for ch in s:
        if ch.isalnum() or ch.isspace():
            cleaned.append(ch)
    s = "".join(cleaned)

    # Συμπτύσσουμε κενά
    s = " ".join(s.split())

    # Κάποια χειροκίνητα aliases
    ALIASES = {
        "manchester utd": "manchester united",
        "man united": "manchester united",
        "newcastle utd": "newcastle",
        "psg": "paris saint germain",
        "inter milan": "inter",
        "wolves": "wolverhampton",
    }
    if s in ALIASES:
        s = ALIASES[s]

    return s


# ============================================================
# FIXTURES
# ============================================================

def fetch_fixtures(league_id: int, league_name: str):
    """
    Fixtures από API-FOOTBALL για συγκεκριμένη λίγκα
    μέσα στο παράθυρο των WINDOW_HOURS ωρών.
    """
    if not API_FOOTBALL_KEY:
        log("⚠️ Missing FOOTBALL_API_KEY – NO fixtures will be fetched!")
        return []

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {
        "league": league_id,
        "season": FOOTBALL_SEASON,
    }

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
        if status_short != "NS":  # μόνο not started
            continue

        dt = parser.isoparse(fx["fixture"]["date"]).astimezone(datetime.timezone.utc)
        diff_hours = (dt - now).total_seconds() / 3600.0
        if not (0 <= diff_hours <= WINDOW_HOURS):
            continue

        home_name = fx["teams"]["home"]["name"]
        away_name = fx["teams"]["away"]["name"]
        home_id = fx["teams"]["home"]["id"]
        away_id = fx["teams"]["away"]["id"]

        out.append(
            {
                "id": fx["fixture"]["id"],
                "league_id": league_id,
                "league_name": league_name,
                "home": home_name,
                "away": away_name,
                "home_id": home_id,
                "away_id": away_id,
                "date_raw": fx["fixture"]["date"],
                "timestamp_utc": dt.isoformat(),
            }
        )

    log(f"→ {league_name}: {len(out)} fixtures within window")
    return out


# ============================================================
# TEAM STATS (API-FOOTBALL /teams/statistics)
# ============================================================

_team_stats_cache: Dict[Tuple[int, int], dict] = {}


def fetch_team_stats(team_id: int, league_id: int) -> dict:
    """
    Τραβάει aggregate στατιστικά ομάδας από API-FOOTBALL.

    Χρησιμοποιεί endpoint:
      GET /teams/statistics?season=YYYY&team=ID&league=ID

    Μας ενδιαφέρουν:
      - avg_goals_for
      - avg_goals_against
      - form_string (π.χ. "WDLDW")
    """
    if not team_id:
        return {}

    key = (team_id, league_id)
    if key in _team_stats_cache:
        return _team_stats_cache[key]

    if not API_FOOTBALL_KEY:
        return {}

    url = f"{API_FOOTBALL_BASE}/teams/statistics"
    params = {
        "season": FOOTBALL_SEASON,
        "team": team_id,
        "league": league_id,
    }

    try:
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=20).json()
    except Exception as e:
        log(f"⚠️ Error fetching team statistics team={team_id}, league={league_id}: {e}")
        _team_stats_cache[key] = {}
        return {}

    resp = r.get("response")
    if not resp:
        _team_stats_cache[key] = {}
        return {}

    goals_for_avg = resp.get("goals", {}).get("for", {}).get("average", {}).get("total")
    goals_against_avg = resp.get("goals", {}).get("against", {}).get("average", {}).get("total")
    form_str = resp.get("form")  # π.χ. "WDLDW"

    def to_float(x):
        try:
            return float(str(x))
        except Exception:
            return None

    stats = {
        "avg_goals_for": to_float(goals_for_avg),
        "avg_goals_against": to_float(goals_against_avg),
        "form": form_str,
    }

    _team_stats_cache[key] = stats
    return stats


def fetch_league_baselines(league_id: int) -> dict:
    """
    Σταθερά baselines ανά λίγκα.
    Αν θες, μπορείς αργότερα να τα βγάλεις δυναμικά.
    """
    # Συντηρητικά default
    baseline = {
        "avg_goals_per_match": 2.6,
        "avg_draw_rate": 0.25,
        "avg_over25_rate": 0.55,
        "home_advantage": 0.20,
    }

    # Αν θες ειδικές τιμές ανά λίγκα, τις βάζεις εδώ (π.χ. Serie A πιο under κλπ.)
    if league_id in (39, 140):  # EPL, La Liga
        baseline["avg_goals_per_match"] = 2.7
        baseline["avg_draw_rate"] = 0.24
    if league_id in (61, 71):  # Ligue 1...
        baseline["avg_goals_per_match"] = 2.5
        baseline["avg_draw_rate"] = 0.27

    return baseline


# ============================================================
# CORE MODEL
# ============================================================

def compute_expected_goals(home_stats: dict, away_stats: dict, league_baseline: dict):
    """
    Μετατροπή team stats σε λ_home, λ_away (Poisson goals).

    - home_strength_attack  = avg_goals_for_home / league_avg_per_team
    - away_strength_attack  = avg_goals_for_away / league_avg_per_team
    - home_strength_defence = avg_goals_against_home / league_avg_per_team
    - away_strength_defence = avg_goals_against_away / league_avg_per_team

    λ_home = league_avg_per_team * home_attack * away_defence * (1 + home_advantage)
    λ_away = league_avg_per_team * away_attack * home_defence
    """
    league_avg_goals = league_baseline.get("avg_goals_per_match", 2.6)
    home_advantage = league_baseline.get("home_advantage", 0.20)

    league_avg_team = league_avg_goals / 2.0

    def safe(v, default):
        return v if isinstance(v, (int, float)) and v > 0 else default

    home_gf = safe(home_stats.get("avg_goals_for"), league_avg_team)
    home_ga = safe(home_stats.get("avg_goals_against"), league_avg_team)
    away_gf = safe(away_stats.get("avg_goals_for"), league_avg_team)
    away_ga = safe(away_stats.get("avg_goals_against"), league_avg_team)

    home_attack = home_gf / league_avg_team
    away_attack = away_gf / league_avg_team
    home_defence = home_ga / league_avg_team
    away_defence = away_ga / league_avg_team

    lambda_home = league_avg_team * home_attack * away_defence * (1.0 + home_advantage)
    lambda_away = league_avg_team * away_attack * home_defence

    # Λογικά όρια για να μην ξεφύγει
    lambda_home = max(0.2, min(lambda_home, 3.5))
    lambda_away = max(0.2, min(lambda_away, 3.5))

    return lambda_home, lambda_away


def poisson_prob(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def compute_probabilities(lambda_home: float, lambda_away: float, context: dict):
    """
    Poisson model για goals → probabilities 1 / X / 2 / Over / Under 2.5.

    Βήματα:
      - Grid 0..6 goals για κάθε ομάδα.
      - Υπολογισμός joint P(h, a).
      - P(Home) = sum P(h > a), P(Draw) = sum P(h == a), κ.ο.κ.
      - P(Over 2.5) = sum P(h + a >= 3)
      - Renormalize ώστε home + draw + away = 1
    """
    max_goals = 6

    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    p_over = 0.0

    for h in range(max_goals + 1):
        p_h = poisson_prob(h, lambda_home)
        for a in range(max_goals + 1):
            p_a = poisson_prob(a, lambda_away)
            p = p_h * p_a

            if h > a:
                p_home += p
            elif h == a:
                p_draw += p
            else:
                p_away += p

            if h + a >= 3:
                p_over += p

    # Υπάρχει μικρή μάζα πέρα από 6 γκολ – renormalize
    total_1x2 = p_home + p_draw + p_away
    if total_1x2 > 0:
        p_home /= total_1x2
        p_draw /= total_1x2
        p_away /= total_1x2

    p_over = min(max(p_over, 0.05), 0.95)
    p_under = 1.0 - p_over

    return {
        "home_prob": p_home,
        "draw_prob": p_draw,
        "away_prob": p_away,
        "over_2_5_prob": p_over,
        "under_2_5_prob": p_under,
    }


# ============================================================
# ODDS (TheOddsAPI) + MAPPING
# ============================================================

def fetch_odds_for_league(league_name: str):
    """Τραβάει odds *μία φορά* από TheOddsAPI για τη συγκεκριμένη λίγκα."""
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
        "regions": "eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
    }

    try:
        url = f"{ODDS_BASE_URL}/{sport_key}/odds"
        res = requests.get(url, params=params, timeout=20)
        if res.status_code != 200:
            log(f"⚠️ Odds error [{league_name}] status={res.status_code}")
            return []
        return res.json()
    except Exception as e:
        log(f"⚠️ Odds request error for {league_name}: {e}")
        return []


def build_odds_index(odds_data):
    """
    Index by NORMALIZED team names:

      index[(norm_home, norm_away)] = {
          'home': best_home,
          'draw': best_draw,
          'away': best_away,
          'over': best_over_2_5,
          'under': best_under_2_5
      }
    """
    index = {}

    for ev in odds_data or []:
        home_raw = ev.get("home_team", "")
        away_raw = ev.get("away_team", "")

        home_norm = normalize_team_name(home_raw)
        away_norm = normalize_team_name(away_raw)

        best_home = best_draw = best_away = None
        best_over = best_under = None

        for bm in ev.get("bookmakers", []):
            for mkt in bm.get("markets", []):
                mk = mkt.get("key")

                if mk == "h2h":
                    outs = mkt.get("outcomes", [])
                    for o in outs:
                        name = normalize_team_name(o.get("name", ""))
                        try:
                            price = float(o["price"])
                        except Exception:
                            continue

                        if name == home_norm:
                            best_home = max(best_home or 0, price)
                        elif name == away_norm:
                            best_away = max(best_away or 0, price)
                        elif name == "draw":
                            best_draw = max(best_draw or 0, price)

                elif mk == "totals":
                    for o in mkt.get("outcomes", []):
                        raw_name = o.get("name", "").lower()
                        try:
                            price = float(o["price"])
                        except Exception:
                            continue
                        if "over" in raw_name and "2.5" in raw_name:
                            best_over = max(best_over or 0, price)
                        elif "under" in raw_name and "2.5" in raw_name:
                            best_under = max(best_under or 0, price)

        index[(home_norm, away_norm)] = {
            "home": best_home,
            "draw": best_draw,
            "away": best_away,
            "over": best_over,
            "under": best_under,
        }

    log(f"Odds index size: {len(index)}")
    return index


# ============================================================
# MAIN PIPELINE
# ============================================================

def build_fixture_blocks():
    fixtures_out = []

    log(f"Using FOOTBALL_SEASON={FOOTBALL_SEASON}")
    log(f"Window: next {WINDOW_HOURS} hours")

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
        for lg_name in LEAGUES.keys():
            odds_data = fetch_odds_for_league(lg_name)
            league_index = build_odds_index(odds_data)
            odds_index_global.update(league_index)
    else:
        log("⚠️ USE_ODDS_API = False → δεν τραβάμε odds από TheOddsAPI.")

    # 3) Loop fixtures → stats → λ → probabilities → fair → offered
    for fx in all_fixtures:
        home_name = fx["home"]
        away_name = fx["away"]
        home_id = fx["home_id"]
        away_id = fx["away_id"]
        league_name = fx["league_name"]
        league_id = fx["league_id"]

        home_norm = normalize_team_name(home_name)
        away_norm = normalize_team_name(away_name)

        league_baseline = fetch_league_baselines(league_id)

        # ---- TEAM STATS ----
        home_stats = fetch_team_stats(home_id, league_id)
        away_stats = fetch_team_stats(away_id, league_id)

        # ---- EXPECTED GOALS ----
        lambda_home, lambda_away = compute_expected_goals(
            home_stats, away_stats, league_baseline
        )

        # ---- PROBABILITIES ----
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

        p_home = probs["home_prob"]
        p_draw = probs["draw_prob"]
        p_away = probs["away_prob"]
        p_over = probs["over_2_5_prob"]
        p_under = probs["under_2_5_prob"]

        # ---- FAIR ----
        fair_1 = implied(p_home)
        fair_x = implied(p_draw)
        fair_2 = implied(p_away)
        fair_over = implied(p_over)
        fair_under = implied(p_under)

        # ---- OFFERED ----
        offered = odds_index_global.get((home_norm, away_norm), {})
        off_home = offered.get("home")
        off_draw = offered.get("draw")
        off_away = offered.get("away")
        off_over = offered.get("over")
        off_under = offered.get("under")

        # ---- DATE/TIME ----
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
                "model": "poisson_v1_teamstats",
                "lambda_home": round(lambda_home, 3),
                "lambda_away": round(lambda_away, 3),
                "fair_1": fair_1,
                "fair_x": fair_x,
                "fair_2": fair_2,
                "fair_over_2_5": fair_over,
                "fair_under_2_5": fair_under,
                "draw_prob": round(p_draw, 3),
                "over_2_5_prob": round(p_over, 3),
                "under_2_5_prob": round(p_under, 3),
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
    fixtures = build_fixture_blocks()

    now = datetime.datetime.now(datetime.timezone.utc)
    to_dt = now + datetime.timedelta(hours=WINDOW_HOURS)

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

    log(f"Thursday v4 READY. Fixtures: {len(fixtures)}")


if __name__ == "__main__":
    main()
