import os
import json
import requests
import datetime
from math import exp, factorial
from dateutil import parser

# ============================================================
#  THURSDAY ENGINE v4 — PRODUCTION MODEL
#  - Fixtures από API-FOOTBALL
#  - Πραγματικά team stats (goals + shots + xG όπου υπάρχει)
#  - Poisson core model
#  - FAIR odds = 1 / prob
#  - Offered odds από TheOddsAPI
#  - Γράφει logs/thursday_report_v3.json (backwards compatible)
# ============================================================

# ------------------------- CONFIG / KEYS -------------------------
API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

FOOTBALL_SEASON = os.getenv("FOOTBALL_SEASON", "2025")

HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

# Αν θες να είναι ΠΑΝΤΑ on, μπορείς να το κάνεις USE_ODDS_API = True
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

# ============================================================
#               FAIR / POISSON HELPERS
# ============================================================

def implied(p: float):
    """Υπολογισμός fair από πιθανότητα (1/p) — όπως στο master spec."""
    return 1.0 / p if p and p > 0 else None


def P(k: int, lam: float) -> float:
    """Poisson PMF."""
    try:
        return exp(-lam) * (lam ** k) / factorial(k)
    except Exception:
        return 0.0


def safe_float(v, default=0.0):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


# ============================================================
#                 FIXTURE & DATA HELPERS
# ============================================================

def fetch_fixtures(league_id: int, league_name: str):
    """
    Τραβάει fixtures από API-FOOTBALL για συγκεκριμένη λίγκα
    μέσα στο παράθυρο των WINDOW_HOURS ωρών.
    Κρατάει ΚΑΙ τα team_ids για να φέρνουμε team stats.
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
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=20).json()
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

    print(f"→ {league_name}: {len(out)} fixtures within window", flush=True)
    return out


def fetch_fixture_statistics(fixture_id: int):
    """
    Τραβάει /fixtures/statistics για ένα fixture και γυρίζει
    dict team_id -> list[ {type, value}, ... ]
    """
    if not API_FOOTBALL_KEY or not fixture_id:
        return {}

    url = f"{API_FOOTBALL_BASE}/fixtures/statistics"
    params = {"fixture": fixture_id}

    try:
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=20).json()
    except Exception as e:
        print(f"⚠️ Error fetching fixture statistics fixture={fixture_id}: {e}", flush=True)
        return {}

    if not r.get("response"):
        return {}

    out = {}
    for entry in r["response"]:
        team = entry.get("team", {})
        t_id = team.get("id")
        stats_list = entry.get("statistics", [])
        out[t_id] = stats_list

    return out


def fetch_team_recent_stats(team_id: int, league_id: int):
    """
    ΠΡΑΓΜΑΤΙΚΑ team stats (goals, shots, xG όπου υπάρχει) από API-FOOTBALL.

    Χρησιμοποιεί:
      /fixtures?team={team_id}&league={league_id}&season=...&last=5
    και για κάθε fixture:
      /fixtures/statistics?fixture={fixture_id}

    Επιστρέφει aggregate:
      - goals_for / goals_against
      - xg_for / xg_against (όπου υπάρχει "Expected goals" αλλιώς 0)
      - shots_for / shots_against
      - shots_on_target_for / shots_on_target_against
      - big_chances_for / big_chances_against (αν υπάρχει σαν stat τύπου "Big Chances")
      - matches_count
    """
    if not API_FOOTBALL_KEY or not team_id:
        return {}

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
        print(f"⚠️ Error fetching team fixtures team_id={team_id}: {e}", flush=True)
        return {}

    if not r.get("response"):
        return {}

    gf = ga = 0.0
    xg_for = xg_against = 0.0
    sh_for = sh_against = 0.0
    sot_for = sot_against = 0.0
    bigc_for = bigc_against = 0.0

    matches = 0

    for fx in r["response"]:
        fixture_id = fx["fixture"]["id"]
        matches += 1

        goals_home = fx["goals"]["home"] or 0
        goals_away = fx["goals"]["away"] or 0

        t_home = fx["teams"]["home"]["id"]
        t_away = fx["teams"]["away"]["id"]

        # Goals for/against
        if t_home == team_id:
            gf += goals_home
            ga += goals_away
        elif t_away == team_id:
            gf += goals_away
            ga += goals_home

        # --- Fixture statistics call ---
        stats_by_team = fetch_fixture_statistics(fixture_id)
        if not stats_by_team:
            continue

        # Διαβάζουμε stats για την ομάδα μας και τον αντίπαλο
        for t_id, stats_list in stats_by_team.items():
            is_our_team = (t_id == team_id)

            for st in stats_list:
                st_type = (st.get("type") or "").lower()
                value = st.get("value")

                if value is None:
                    continue

                # Προσπαθούμε να το κάνουμε float
                v = safe_float(value, 0.0)

                # Total Shots
                if "total shots" in st_type:
                    if is_our_team:
                        sh_for += v
                    else:
                        sh_against += v

                # Shots on Goal / Shots on Target
                if "shots on goal" in st_type or "shots on target" in st_type:
                    if is_our_team:
                        sot_for += v
                    else:
                        sot_against += v

                # Big chances (παίζει να γράφεται λίγο διαφορετικά ανά provider)
                if "big chances" in st_type:
                    if is_our_team:
                        bigc_for += v
                    else:
                        bigc_against += v

                # Expected goals (αν υπάρχει)
                if "expected" in st_type and "goal" in st_type:
                    if is_our_team:
                        xg_for += v
                    else:
                        xg_against += v

    return {
        "goals_for": gf,
        "goals_against": ga,
        "xg_for": xg_for,
        "xg_against": xg_against,
        "shots_for": sh_for,
        "shots_against": sh_against,
        "shots_on_target_for": sot_for,
        "shots_on_target_against": sot_against,
        "big_chances_for": bigc_for,
        "big_chances_against": bigc_against,
        "matches_count": matches,
    }


def fetch_league_baselines(league_id: int):
    """
    Απλό league baseline.
    Μπορείς αργότερα να το κάνεις dynamic από ιστορικά fixtures.
    """
    return {
        "avg_goals_per_match": 2.6,
        "avg_draw_rate": 0.24,
        "avg_over25_rate": 0.55,
        "home_advantage": 0.20,
    }

# ============================================================
#            CORE MODEL: λ + PROBABILITIES
# ============================================================

def compute_expected_goals(home_stats: dict, away_stats: dict, league_baseline: dict):
    """
    Από aggregated stats + league baselines → λ_home, λ_away.

    Χρησιμοποιεί:
      - goals_for / goals_against
      - xg_for / xg_against (όπου υπάρχει)
      - shots_on_target / big chances
      - league_average + home_advantage

    Όταν δεν υπάρχουν αρκετά δεδομένα, fallback σε league baseline.
    """
    league_avg = safe_float(league_baseline.get("avg_goals_per_match", 2.60))
    home_adv = safe_float(league_baseline.get("home_advantage", 0.20))

    h_matches = max(1, int(home_stats.get("matches_count") or 0))
    a_matches = max(1, int(away_stats.get("matches_count") or 0))

    # --- HOME team aggregates ---
    h_gf_gpm = safe_float(home_stats.get("goals_for")) / h_matches
    h_ga_gpm = safe_float(home_stats.get("goals_against")) / h_matches
    h_xg_gpm = safe_float(home_stats.get("xg_for")) / h_matches if home_stats.get("xg_for") else 0.0
    h_xga_gpm = safe_float(home_stats.get("xg_against")) / h_matches if home_stats.get("xg_against") else 0.0
    h_sot_gpm = safe_float(home_stats.get("shots_on_target_for")) / h_matches
    h_bigc_gpm = safe_float(home_stats.get("big_chances_for")) / h_matches

    # --- AWAY team aggregates ---
    a_gf_gpm = safe_float(away_stats.get("goals_for")) / a_matches
    a_ga_gpm = safe_float(away_stats.get("goals_against")) / a_matches
    a_xg_gpm = safe_float(away_stats.get("xg_for")) / a_matches if away_stats.get("xg_for") else 0.0
    a_xga_gpm = safe_float(away_stats.get("xg_against")) / a_matches if away_stats.get("xg_against") else 0.0
    a_sot_gpm = safe_float(away_stats.get("shots_on_target_for")) / a_matches
    a_bigc_gpm = safe_float(away_stats.get("big_chances_for")) / a_matches

    # Αν δεν έχουμε σχεδόν τίποτα, fallback σε συμμετρικό split του league_avg
    if (home_stats.get("matches_count") or 0) == 0 and (away_stats.get("matches_count") or 0) == 0:
        # 55% της επίθεσης στον γηπεδούχο, 45% στον φιλοξενούμενο
        h_gf_gpm = league_avg * 0.55
        a_gf_gpm = league_avg * 0.45
        h_ga_gpm = a_gf_gpm
        a_ga_gpm = h_gf_gpm

    # Attack strength proxies
    attack_home = (
        0.45 * h_gf_gpm +
        0.25 * h_xg_gpm +
        0.15 * (h_sot_gpm * 0.10) +
        0.15 * (h_bigc_gpm * 0.15)
    )

    attack_away = (
        0.45 * a_gf_gpm +
        0.25 * a_xg_gpm +
        0.15 * (a_sot_gpm * 0.10) +
        0.15 * (a_bigc_gpm * 0.15)
    )

    # Defensive strength proxies (μικρότερα = καλύτερη άμυνα)
    defense_home = (
        0.50 * h_ga_gpm +
        0.30 * h_xga_gpm +
        0.20 * (safe_float(home_stats.get("big_chances_against")) / h_matches if h_matches else 0.0)
    )

    defense_away = (
        0.50 * a_ga_gpm +
        0.30 * a_xga_gpm +
        0.20 * (safe_float(away_stats.get("big_chances_against")) / a_matches if a_matches else 0.0)
    )

    # Κρατάμε την άμυνα σε λογικά όρια
    defense_home = max(0.3, min(defense_home, 3.0))
    defense_away = max(0.3, min(defense_away, 3.0))

    # Λογική mix: επίθεση γηπεδούχου vs άμυνα φιλοξενούμενου, γύρω από το league_avg
    lambda_home = (
        0.55 * attack_home +
        0.30 * max(0.1, league_avg - defense_away) +
        0.15 * (league_avg / 2.0)
    )

    lambda_away = (
        0.55 * attack_away +
        0.30 * max(0.1, league_avg - defense_home) +
        0.15 * (league_avg / 2.2)
    )

    # Home advantage
    lambda_home *= (1.0 + home_adv * 0.5)

    # Safety clamp
    lambda_home = max(0.25, min(lambda_home, 3.20))
    lambda_away = max(0.20, min(lambda_away, 3.00))

    return lambda_home, lambda_away


def compute_probabilities(lambda_home: float, lambda_away: float, context: dict):
    """
    Πλήρες core μοντέλο:
      - Poisson goal matrix
      - League draw/over blending
      - Light form/momentum/injury/fatigue logic (placeholders)
    """
    # ----------------------------------------------------
    # POISSON MATRIX
    # ----------------------------------------------------
    max_goals = 7
    p_home = p_draw = p_away = 0.0
    p_over = 0.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = P(i, lambda_home) * P(j, lambda_away)

            if i > j:
                p_home += p
            elif i < j:
                p_away += p
            else:
                p_draw += p

            if i + j >= 3:
                p_over += p

    p_under = 1.0 - p_over

    # ----------------------------------------------------
    # LEAGUE CORRECTIONS
    # ----------------------------------------------------
    baseline = context.get("league_baseline", {})

    lg_draw = safe_float(baseline.get("avg_draw_rate", 0.24))
    lg_over = safe_float(baseline.get("avg_over25_rate", 0.55))

    # Blend προς τα league trends (ελαφριά)
    p_draw = 0.85 * p_draw + 0.15 * lg_draw
    p_over = 0.85 * p_over + 0.15 * lg_over
    p_under = 1.0 - p_over

    # ----------------------------------------------------
    # MOMENTUM / FORM (placeholders)
    # ----------------------------------------------------
    home_stats = context.get("home_stats", {}) or {}
    away_stats = context.get("away_stats", {}) or {}

    form_home = safe_float(home_stats.get("form_points", 0.0))
    form_away = safe_float(away_stats.get("form_points", 0.0))

    if form_home - form_away > 5:
        p_home *= 1.04
    elif form_away - form_home > 5:
        p_away *= 1.04

    # ----------------------------------------------------
    # H2H ADJUSTMENT (αν περάσεις h2h_draw_rate στο context)
    # ----------------------------------------------------
    h2h_draw_rate = context.get("h2h_draw_rate", None)
    if isinstance(h2h_draw_rate, (int, float)):
        p_draw = 0.90 * p_draw + 0.10 * h2h_draw_rate

    # ----------------------------------------------------
    # INJURY / FATIGUE PLACEHOLDER (αν περάσεις δεδομένα)
    # ----------------------------------------------------
    inj_home = safe_float(home_stats.get("injuries", 0.0))
    inj_away = safe_float(away_stats.get("injuries", 0.0))

    if inj_home >= 2:
        p_home *= 0.93
    if inj_away >= 2:
        p_away *= 0.93

    rest_home = home_stats.get("rest_days", 5)
    rest_away = away_stats.get("rest_days", 5)

    if rest_home is not None and rest_home < 3:
        p_home *= 0.95
    if rest_away is not None and rest_away < 3:
        p_away *= 0.95

    # ----------------------------------------------------
    # NORMALIZATION 1X2
    # ----------------------------------------------------
    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    # Safety clamps
    p_home = max(0.02, min(p_home, 0.90))
    p_draw = max(0.04, min(p_draw, 0.40))
    p_away = max(0.02, min(p_away, 0.90))

    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total

    # Ξαναυπολογίζουμε το under μετά τα adjustments
    p_under = max(0.0, min(1.0, 1.0 - p_over))

    return {
        "home_prob": p_home,
        "draw_prob": p_draw,
        "away_prob": p_away,
        "over_2_5_prob": p_over,
        "under_2_5_prob": p_under,
    }

# ============================================================
#                     ODDS (TheOddsAPI)
# ============================================================

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
      index["Home – Away"] = {
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
        match_key = f"{home_raw} – {away_raw}"

        best_home = best_draw = best_away = None
        best_over = best_under = None

        for bm in ev.get("bookmakers", []):
            for m in bm.get("markets", []):
                mk = m.get("key")

                if mk == "h2h":
                    outs = m.get("outcomes", [])
                    if len(outs) == 3:
                        try:
                            best_home = max(best_home or 0, float(outs[0]["price"]))
                            best_away = max(best_away or 0, float(outs[1]["price"]))
                            best_draw = max(best_draw or 0, float(outs[2]["price"]))
                        except Exception:
                            pass

                elif mk == "totals":
                    for o in m.get("outcomes", []):
                        name = o.get("name", "")
                        try:
                            price = float(o["price"])
                        except Exception:
                            continue
                        if name == "Over 2.5":
                            best_over = max(best_over or 0, price)
                        elif name == "Under 2.5":
                            best_under = max(best_under or 0, price)

        index[match_key] = {
            "home": best_home,
            "draw": best_draw,
            "away": best_away,
            "over": best_over,
            "under": best_under,
        }

    return index

# ============================================================
#                     MAIN PIPELINE
# ============================================================

def build_fixture_blocks():
    """
    Κύρια ροή:
      fixtures → (team stats) → λ → probabilities → fair → odds → JSON rows
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

    # 3) Cache για team stats ώστε να μην τα φέρνουμε 10 φορές
    team_stats_cache = {}

    # 4) Loop fixtures → team stats → probabilities → fair
    for fx in all_fixtures:
        home_name = fx["home"]
        away_name = fx["away"]
        league_name = fx["league_name"]
        league_id = fx["league_id"]
        home_id = fx.get("home_id")
        away_id = fx.get("away_id")

        match_key = f"{home_name} – {away_name}"

        # Πραγματικά team stats από τα τελευταία 5 ματς της κάθε ομάδας (με caching)
        home_key = (home_id, league_id)
        away_key = (away_id, league_id)

        if home_key not in team_stats_cache:
            team_stats_cache[home_key] = fetch_team_recent_stats(home_id, league_id) if home_id else {}
        if away_key not in team_stats_cache:
            team_stats_cache[away_key] = fetch_team_recent_stats(away_id, league_id) if away_id else {}

        home_stats = team_stats_cache.get(home_key, {}) or {}
        away_stats = team_stats_cache.get(away_key, {}) or {}

        league_baseline = fetch_league_baselines(league_id)

        # ---- STEP 1: Expected goals (λ) ----
        lambda_home, lambda_away = compute_expected_goals(
            home_stats, away_stats, league_baseline
        )

        # ---- STEP 2: Probabilities από το core μοντέλο ----
        context = {
            "league_name": league_name,
            "league_id": league_id,
            "home": home_name,
            "away": away_name,
            "league_baseline": league_baseline,
            "home_stats": home_stats,
            "away_stats": away_stats,
            # εδώ αργότερα μπορείς να περάσεις h2h_draw_rate, injuries, rest_days κλπ.
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
        offered = odds_index_global.get(match_key, {})
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
                "model": "production_v4_poisson_teamstats_xg",
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

    print(f"Thursday v4 PRODUCTION READY. Fixtures: {len(fixtures)}", flush=True)


if __name__ == "__main__":
    main()
