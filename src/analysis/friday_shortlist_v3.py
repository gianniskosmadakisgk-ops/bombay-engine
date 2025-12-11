import os
import re
import json
import math
import requests
import datetime
from dateutil import parser

# ============================================================
#  THURSDAY ENGINE v4 — FULL MODEL (Poisson + real team stats)
#  με βελτιωμένο mapping TheOddsAPI (team names + Over 2.5 Goals)
# ============================================================

# ------------------------- CONFIG / KEYS -------------------------
API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

FOOTBALL_SEASON = os.getenv("FOOTBALL_SEASON", "2025")

HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

# Μπορείς να το ελέγχεις από το Render (USE_ODDS_API=true/false)
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

# 3 ημέρες (72 ώρες)
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

MAX_GOALS = 7  # για Poisson sum (0..7)


# ------------------------- HELPERS -------------------------
def implied(p: float):
    """Υπολογισμός fair από πιθανότητα (1/p) — όπως στο master spec."""
    return 1.0 / p if p and p > 0 else None


def poisson_pmf(k: int, lam: float) -> float:
    """Poisson pmf P(X=k) με lam = expected goals."""
    if lam <= 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def normalize_team(s: str) -> str:
    """
    Ενιαίο normalize για ονόματα ομάδων,
    ώστε να ταιριάζουν API-FOOTBALL και TheOddsAPI.

    Παράδειγμα:
      "West Bromwich Albion FC" -> "west bromwich albion"
      "Stoke City"              -> "stoke city"
      "Real Sociedad de Fútbol" -> "real sociedad de futbol"
    """
    if not s:
        return ""
    s = s.lower()

    # σβήσε συνηθισμένα suffix/prefix
    s = s.replace("football club", " ")
    s = s.replace("futebol clube", " ")
    s = s.replace("futbol club", " ")
    s = s.replace("sociedad deportiva", " ")

    s = re.sub(r"\b(fc|cf|afc|cfc|ac|sc|bk)\b", " ", s)

    # άφησε μόνο γράμματα/νούμερα, όλα τα άλλα γίνονται space
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ------------------------- FIXTURES -------------------------
def fetch_fixtures(league_id: int, league_name: str):
    """
    Τραβάει fixtures από API-FOOTBALL για συγκεκριμένη λίγκα
    μέσα στο παράθυρο των WINDOW_HOURS ωρών.
    Κρατάει ΚΑΙ team ids (home_id / away_id).
    """
    if not API_FOOTBALL_KEY:
        print("⚠️ Missing FOOTBALL_API_KEY – NO fixtures will be fetched!", flush=True)
        return []

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {
        "league": league_id,
        "season": FOOTBALL_SEASON,
    }

    try:
        r = requests.get(
            url, headers=HEADERS_FOOTBALL, params=params, timeout=20
        ).json()
    except Exception as e:
        print(f"⚠️ Error fetching fixtures for {league_name}: {e}", flush=True)
        return []

    if not r.get("response"):
        print(f"⚠️ No fixtures response for league {league_name}", flush=True)
        return []

    out = []
    now = datetime.datetime.now(datetime.timezone.utc)

    for fx in r["response"]:
        status_short = fx["fixture"]["status"]["short"]
        if status_short != "NS":  # μόνο not started
            continue

        dt = parser.isoparse(fx["fixture"]["date"]).astimezone(
            datetime.timezone.utc
        )
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
                "date_raw": fx["fixture"]["date"],
                "timestamp_utc": dt.isoformat(),
            }
        )

    print(f"→ {league_name}: {len(out)} fixtures within window", flush=True)
    return out


# ------------------------- TEAM STATS (API-FOOTBALL) -------------------------
def fetch_team_stats(team_id: int, league_id: int):
    """
    Τραβάει συνολικά στατιστικά ομάδας από API-FOOTBALL /teams/statistics.
    Χρησιμοποιούμε:
      - fixtures.played.total
      - goals.for.total.total
      - goals.against.total.total
    """
    if not API_FOOTBALL_KEY:
        return {}

    url = f"{API_FOOTBALL_BASE}/teams/statistics"
    params = {
        "team": team_id,
        "league": league_id,
        "season": FOOTBALL_SEASON,
    }

    try:
        r = requests.get(
            url, headers=HEADERS_FOOTBALL, params=params, timeout=20
        ).json()
    except Exception as e:
        print(f"⚠️ Error fetching team stats team_id={team_id}: {e}", flush=True)
        return {}

    data = r.get("response")
    if not data:
        return {}

    fixtures = data.get("fixtures", {})
    played = fixtures.get("played", {}).get("total") or 0

    goals = data.get("goals", {})
    gf_total = (
        goals.get("for", {}).get("total", {}).get("total", 0) or 0
    )
    ga_total = (
        goals.get("against", {}).get("total", {}).get("total", 0) or 0
    )

    if played > 0:
        avg_gf = gf_total / played
        avg_ga = ga_total / played
    else:
        avg_gf = 0.0
        avg_ga = 0.0

    return {
        "matches_played": played,
        "avg_goals_for": avg_gf,
        "avg_goals_against": avg_ga,
    }


def fetch_league_baselines(league_id: int):
    """
    Βασικά στατιστικά λίγκας (fixed προς το παρόν).
    """
    return {
        "avg_goals_per_match": 2.6,  # combined
        "avg_draw_rate": 0.24,
        "avg_over25_rate": 0.55,
        "home_advantage": 0.20,  # +20% στο λ_home
    }


# ------------------------- CORE MODEL -------------------------
def compute_expected_goals(home_stats: dict, away_stats: dict, league_baseline: dict):
    """
    Μετατρέπει team stats σε expected goals (λ_home, λ_away) για Poisson.
    """
    avg_goals_match = league_baseline.get("avg_goals_per_match", 2.6)
    home_adv = league_baseline.get("home_advantage", 0.2)

    base_team_goals = avg_goals_match / 2.0 if avg_goals_match > 0 else 1.3

    home_gf = home_stats.get("avg_goals_for", base_team_goals)
    home_ga = home_stats.get("avg_goals_against", base_team_goals)
    away_gf = away_stats.get("avg_goals_for", base_team_goals)
    away_ga = away_stats.get("avg_goals_against", base_team_goals)

    attack_home = home_gf / base_team_goals if base_team_goals > 0 else 1.0
    attack_away = away_gf / base_team_goals if base_team_goals > 0 else 1.0

    defence_weakness_home = home_ga / base_team_goals if base_team_goals > 0 else 1.0
    defence_weakness_away = away_ga / base_team_goals if base_team_goals > 0 else 1.0

    attack_home = min(max(attack_home, 0.5), 1.8)
    attack_away = min(max(attack_away, 0.5), 1.8)
    defence_weakness_home = min(max(defence_weakness_home, 0.5), 1.8)
    defence_weakness_away = min(max(defence_weakness_away, 0.5), 1.8)

    lambda_home = (
        base_team_goals
        * attack_home
        * defence_weakness_away
        * (1.0 + home_adv)
    )
    lambda_away = base_team_goals * attack_away * defence_weakness_home

    lambda_home = min(max(lambda_home, 0.2), 3.5)
    lambda_away = min(max(lambda_away, 0.2), 3.5)

    return float(lambda_home), float(lambda_away)


def compute_probabilities(lambda_home: float, lambda_away: float, context: dict):
    """
    Poisson model:
      - Υπολογίζουμε όλα τα σκορ 0–7.
      - Μαζεύουμε:
          * home_prob, draw_prob, away_prob
          * over_2_5_prob
      - Renormalize 1X2 ώστε να κάνει άθροισμα 1.
    """
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0
    p_over25 = 0.0

    for gh in range(0, MAX_GOALS + 1):
        ph = poisson_pmf(gh, lambda_home)
        for ga in range(0, MAX_GOALS + 1):
            pa = poisson_pmf(ga, lambda_away)
            p = ph * pa

            if gh > ga:
                p_home += p
            elif gh == ga:
                p_draw += p
            else:
                p_away += p

            if gh + ga >= 3:
                p_over25 += p

    total_1x2 = p_home + p_draw + p_away
    if total_1x2 > 0:
        p_home /= total_1x2
        p_draw /= total_1x2
        p_away /= total_1x2

    p_over25 = min(max(p_over25, 0.0), 1.0)
    p_under25 = max(0.0, 1.0 - p_over25)

    return {
        "home_prob": p_home,
        "draw_prob": p_draw,
        "away_prob": p_away,
        "over_2_5_prob": p_over25,
        "under_2_5_prob": p_under25,
    }


# ------------------------- ODDS (TheOddsAPI) -------------------------
def fetch_odds_for_league(league_name: str):
    """Τραβάει odds *μία φορά* από TheOddsAPI για τη συγκεκριμένη λίγκα."""
    if not USE_ODDS_API:
        return []

    sport_key = LEAGUE_TO_SPORT.get(league_name)
    if not sport_key:
        return []

    if not ODDS_API_KEY:
        print("⚠️ Missing ODDS_API_KEY – skipping odds", flush=True)
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
            print(f"⚠️ Odds error [{league_name}] status={res.status_code}", flush=True)
            return []
        return res.json()
    except Exception as e:
        print(f"⚠️ Odds request error for {league_name}: {e}", flush=True)
        return []


def build_odds_index(odds_data):
    """
    Index:
      index[(home_norm, away_norm)] = {
          'home': best_home,
          'draw': best_draw,
          'away': best_away,
          'over': best_over_2_5,
          'under': best_under_2_5
      }
    Χρησιμοποιεί normalize_team τόσο για teams όσο και για outcomes.
    Πιάνει επίσης "Over 2.5 Goals" κ.λπ.
    """
    index = {}

    for ev in odds_data or []:
        home_raw = ev.get("home_team", "")
        away_raw = ev.get("away_team", "")
        home_key = normalize_team(home_raw)
        away_key = normalize_team(away_raw)

        best_home = best_draw = best_away = None
        best_over = best_under = None

        for bm in ev.get("bookmakers", []):
            for m in bm.get("markets", []):
                mk = m.get("key")

                if mk == "h2h":
                    for o in m.get("outcomes", []):
                        nm_raw = o.get("name", "") or ""
                        nm = normalize_team(nm_raw)
                        try:
                            price = float(o["price"])
                        except Exception:
                            continue

                        if nm == home_key:
                            best_home = max(best_home or 0, price)
                        elif nm == away_key:
                            best_away = max(best_away or 0, price)
                        else:
                            low = nm_raw.lower()
                            if "draw" in low or "tie" in low:
                                best_draw = max(best_draw or 0, price)

                elif mk == "totals":
                    for o in m.get("outcomes", []):
                        name = (o.get("name", "") or "").lower()
                        try:
                            price = float(o["price"])
                        except Exception:
                            continue

                        if "over" in name and "2.5" in name:
                            best_over = max(best_over or 0, price)
                        elif "under" in name and "2.5" in name:
                            best_under = max(best_under or 0, price)

        index[(home_key, away_key)] = {
            "home": best_home,
            "draw": best_draw,
            "away": best_away,
            "over": best_over,
            "under": best_under,
        }

    return index


# ------------------------- MAIN PIPELINE -------------------------
def build_fixture_blocks():
    """
    Κύρια ροή:
      fixtures → team stats → λ → probabilities → fair → odds → JSON rows
    """
    fixtures_out = []

    print(f"Using FOOTBALL_SEASON={FOOTBALL_SEASON}", flush=True)
    print(f"Window: next {WINDOW_HOURS} hours", flush=True)

    if not API_FOOTBALL_KEY:
        print("❌ FOOTBALL_API_KEY is missing. Aborting fixture fetch.", flush=True)
        return []

    # 1) Fixtures από όλες τις λίγκες
    all_fixtures = []
    for lg_name, lg_id in LEAGUES.items():
        fx_list = fetch_fixtures(lg_id, lg_name)
        all_fixtures.extend(fx_list)

    print(f"Total raw fixtures collected: {len(all_fixtures)}", flush=True)

    # 2) Odds index
    odds_index_global = {}
    if USE_ODDS_API:
        for lg_name in LEAGUES.keys():
            odds_data = fetch_odds_for_league(lg_name)
            league_index = build_odds_index(odds_data)
            odds_index_global.update(league_index)
        print(f"Odds index built for {len(odds_index_global)} matches", flush=True)
    else:
        print("⚠️ USE_ODDS_API = False → δεν τραβάμε odds από TheOddsAPI.", flush=True)

    # 3) Cache για team stats ώστε να μην βαράμε 1000 φορές το API
    team_stats_cache = {}

    # 4) Loop fixtures → probabilities → fair
    for fx in all_fixtures:
        home_name = fx["home"]
        away_name = fx["away"]
        league_name = fx["league_name"]
        league_id = fx["league_id"]
        home_id = fx["home_id"]
        away_id = fx["away_id"]

        match_key_tuple = (normalize_team(home_name), normalize_team(away_name))

        league_baseline = fetch_league_baselines(league_id)

        # --- Team stats με caching ---
        if home_id not in team_stats_cache:
            team_stats_cache[home_id] = fetch_team_stats(home_id, league_id)
        if away_id not in team_stats_cache:
            team_stats_cache[away_id] = fetch_team_stats(away_id, league_id)

        home_stats = team_stats_cache.get(home_id, {})
        away_stats = team_stats_cache.get(away_id, {})

        # ---- STEP 1: Expected goals (λ) ----
        lambda_home, lambda_away = compute_expected_goals(
            home_stats, away_stats, league_baseline
        )

        # ---- STEP 2: Probabilities από Poisson ----
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

        # ---- STEP 3: FAIR odds από πιθανότητες (όπως στο spec) ----
        fair_1 = implied(p_home)
        fair_x = implied(p_draw)
        fair_2 = implied(p_away)
        fair_over = implied(p_over)
        fair_under = implied(p_under)

        # ---- STEP 4: Offered odds από TheOddsAPI ----
        offered = odds_index_global.get(match_key_tuple, {})
        off_home = offered.get("home")
        off_draw = offered.get("draw")
        off_away = offered.get("away")
        off_over = offered.get("over")
        off_under = offered.get("under")

        # ---- STEP 5: Formatting ημερομηνίας/ώρας ----
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
                "model": "poisson_v1",
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

    print(f"Thursday fixtures_out: {len(fixtures_out)}", flush=True)
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

    print(f"Thursday v4 FULL MODEL READY. Fixtures: {len(fixtures)}", flush=True)


if __name__ == "__main__":
    main()
