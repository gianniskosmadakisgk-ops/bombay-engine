import os
import json
import math
import requests
import datetime
from dateutil import parser

# ============================================================
#  THURSDAY ENGINE v4 — FULL MODEL
#  - Fixtures από API-FOOTBALL
#  - Πραγματικά team stats (last N matches)
#  - Poisson + league calibration για 1X2 & Over/Under
#  - FAIR odds = 1/p (όπως στο master spec)
#  - Προαιρετικά offered odds από TheOddsAPI
#  - Γράφει logs/thursday_report_v3.json (backwards compatible)
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

# Πόσα πρόσφατα ματς ανά ομάδα για stats
TEAM_RECENT_MATCHES = 5

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

# Cache για team stats ώστε να μη βαράμε 200 φορές το ίδιο team
TEAM_STATS_CACHE = {}

# ------------------------- HELPERS -------------------------
def implied(p: float):
    """Υπολογισμός fair από πιθανότητα (1/p) — όπως στο master spec."""
    return 1.0 / p if p and p > 0 else None


def normalize_team_name(name: str) -> str:
    """
    Προσπαθεί να φέρει κοντά τα ονόματα API-Football & TheOddsAPI.
    π.χ. 'West Bromwich Albion FC' -> 'west bromwich albion'
         'West Brom' -> 'west brom'
    """
    import re

    if not name:
        return ""

    n = name.lower()

    # Πετάμε διάφορα suffixes / τυπικές λέξεις
    junk_words = [
        "fc", "afc", "cf", "sc", "ac", "calcio",
        "deportivo", "club", "football club",
    ]
    for w in junk_words:
        n = n.replace(w, " ")

    # Πετάμε μη-αλφαριθμητικούς
    n = re.sub(r"[^a-z0-9 ]+", " ", n)
    n = re.sub(r"\s+", " ", n).strip()

    # Συγκεκριμένα aliases
    aliases = {
        "stoke city": "stoke",
        "stoke": "stoke",
        "west bromwich albion": "west brom",
        "west bromwich": "west brom",
        "west bromwich albion fc": "west brom",
        "1 fc koln": "koln",
        "fc koln": "koln",
        "1 koln": "koln",
        "real sociedad de futbol": "real sociedad",
    }
    return aliases.get(n, n)


def fetch_fixtures(league_id: int, league_name: str):
    """
    Τραβάει fixtures από API-FOOTBALL για συγκεκριμένη λίγκα
    μέσα στο παράθυρο των WINDOW_HOURS ωρών.
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
                "date_raw": fx["fixture"]["date"],
                "timestamp_utc": dt.isoformat(),
            }
        )

    print(f"→ {league_name}: {len(out)} fixtures within window", flush=True)
    return out


def fetch_team_recent_stats(team_id: int, league_id: int):
    """
    Τραβάει πρόσφατα ματς ομάδας από API-FOOTBALL και τα μετατρέπει σε aggregate stats.

    ΕΠΙΣΤΡΕΦΕΙ dict με:
      - goals_for / goals_against
      - matches_count

    (Shots/xG μπορούν να προστεθούν αργότερα – κρατάμε το API-safe προς το παρόν)
    """
    if not API_FOOTBALL_KEY or not team_id:
        return {
            "goals_for": 0.0,
            "goals_against": 0.0,
            "matches_count": 0,
        }

    cache_key = (team_id, league_id)
    if cache_key in TEAM_STATS_CACHE:
        return TEAM_STATS_CACHE[cache_key]

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {
        "team": team_id,
        "league": league_id,
        "season": FOOTBALL_SEASON,
        "last": TEAM_RECENT_MATCHES,
    }

    try:
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=20).json()
    except Exception as e:
        print(f"⚠️ Error fetching team stats team_id={team_id}: {e}", flush=True)
        stats = {
            "goals_for": 0.0,
            "goals_against": 0.0,
            "matches_count": 0,
        }
        TEAM_STATS_CACHE[cache_key] = stats
        return stats

    if not r.get("response"):
        stats = {
            "goals_for": 0.0,
            "goals_against": 0.0,
            "matches_count": 0,
        }
        TEAM_STATS_CACHE[cache_key] = stats
        return stats

    gf = ga = 0.0
    matches = 0

    for fx in r["response"]:
        try:
            goals_home = fx["goals"]["home"] or 0
            goals_away = fx["goals"]["away"] or 0
            home_id = fx["teams"]["home"]["id"]
            away_id = fx["teams"]["away"]["id"]
        except Exception:
            continue

        matches += 1
        if team_id == home_id:
            gf += goals_home
            ga += goals_away
        elif team_id == away_id:
            gf += goals_away
            ga += goals_home

    stats = {
        "goals_for": gf,
        "goals_against": ga,
        "matches_count": matches,
    }
    TEAM_STATS_CACHE[cache_key] = stats
    return stats


def fetch_league_baselines(league_id: int):
    """
    Βασικά στατιστικά λίγκας – placeholders, μπορούν να γίνουν δυναμικά αργότερα.
    """
    # Μπορείς να φτιάξεις dict per-league, προς το παρόν generic baseline
    return {
        "avg_goals_per_match": 2.6,
        "avg_draw_rate": 0.24,
        "avg_over25_rate": 0.55,
        "home_advantage": 0.20,
    }


# ------------------------- MODEL: λ (Expected Goals) -------------------------
def compute_expected_goals(home_stats: dict, away_stats: dict, league_baseline: dict):
    """
    Μετατρέπει team stats σε expected goals (Poisson λ_home, λ_away).

    • Χρησιμοποιεί goals_for / goals_against των τελευταίων N αγώνων.
    • Καλιμπράρεται πάνω σε avg_goals_per_match της λίγκας & home_advantage.
    """

    avg_goals_match = league_baseline.get("avg_goals_per_match", 2.6)
    home_adv = league_baseline.get("home_advantage", 0.20)

    baseline_team_goals = avg_goals_match / 2.0  # μέσος όρος γκολ ανά ομάδα

    def team_attack_defense(stats):
        m = max(stats.get("matches_count", 0), 1)
        gf = stats.get("goals_for", 0.0) / m
        ga = stats.get("goals_against", 0.0) / m

        attack_index = gf / baseline_team_goals if baseline_team_goals > 0 else 1.0
        defense_index = ga / baseline_team_goals if baseline_team_goals > 0 else 1.0
        return attack_index, defense_index, gf, ga

    home_att_idx, home_def_idx, home_gf, home_ga = team_attack_defense(home_stats)
    away_att_idx, away_def_idx, away_gf, away_ga = team_attack_defense(away_stats)

    # Αν δεν έχουμε δεδομένα, πάμε σε ουδέτερο 1.0
    if home_stats.get("matches_count", 0) == 0:
        home_att_idx = home_def_idx = 1.0
    if away_stats.get("matches_count", 0) == 0:
        away_att_idx = away_def_idx = 1.0

    # Combined indices
    # Home: επιθετική δύναμη home + αμυντική αδυναμία away + home advantage
    lambda_home = baseline_team_goals * (home_att_idx + away_def_idx) / 2.0
    lambda_home *= (1.0 + home_adv)

    # Away: επιθετική δύναμη away + αμυντική αδυναμία home (χωρίς full home_adv)
    lambda_away = baseline_team_goals * (away_att_idx + home_def_idx) / 2.0
    lambda_away *= (1.0 - 0.3 * home_adv)

    # Safety clamps (να μην ξεφεύγουν)
    lambda_home = max(0.2, min(3.5, lambda_home))
    lambda_away = max(0.2, min(3.5, lambda_away))

    return float(lambda_home), float(lambda_away)


# ------------------------- MODEL: Probabilities from λ -------------------------
def poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF."""
    if k < 0:
        return 0.0
    try:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except OverflowError:
        return 0.0


def compute_probabilities(lambda_home: float, lambda_away: float, context: dict):
    """
    Core Poisson-based μοντέλο που βγάζει πιθανότητες:

        {
            "home_prob": float,
            "draw_prob": float,
            "away_prob": float,
            "over_2_5_prob": float,
            "under_2_5_prob": float,
        }

    Βήματα:
      1. Poisson scoreline probabilities (0–10 goals).
      2. P(home/draw/away), P(>2.5).
      3. League calibration προς avg_draw_rate & avg_over25_rate.
      4. Renormalization.
    """
    league_baseline = context.get("league_baseline", {})
    avg_draw_rate = league_baseline.get("avg_draw_rate", 0.24)
    avg_over25_rate = league_baseline.get("avg_over25_rate", 0.55)

    max_goals = 10

    home_probs = [poisson_pmf(i, lambda_home) for i in range(max_goals + 1)]
    away_probs = [poisson_pmf(j, lambda_away) for j in range(max_goals + 1)]

    p_home = p_draw = p_away = p_over = 0.0
    total = 0.0

    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p_ij = home_probs[i] * away_probs[j]
            total += p_ij
            if i > j:
                p_home += p_ij
            elif i == j:
                p_draw += p_ij
            else:
                p_away += p_ij
            if i + j >= 3:
                p_over += p_ij

    # Αν το grid 0–10 δεν καλύπτει όλο το mass (πολύ μικρό tail), κάνουμε normalize
    if total > 0:
        p_home /= total
        p_draw /= total
        p_away /= total
        p_over /= total

    # League calibration (blend Poisson με league baselines)
    alpha_draw = 0.3
    alpha_over = 0.3

    draw_cal = (1 - alpha_draw) * p_draw + alpha_draw * avg_draw_rate
    over_cal = (1 - alpha_over) * p_over + alpha_over * avg_over25_rate

    # Ανακατανομή home/away ώστε home+draw+away=1
    non_draw_poisson = max(p_home + p_away, 1e-9)
    scale_non_draw = (1.0 - draw_cal) / non_draw_poisson

    home_cal = p_home * scale_non_draw
    away_cal = p_away * scale_non_draw

    # Τελική 1X2 renormalization
    total_1x2 = home_cal + draw_cal + away_cal
    if total_1x2 > 0:
        home_cal /= total_1x2
        draw_cal /= total_1x2
        away_cal /= total_1x2

    # Under 2.5
    under_cal = max(0.0, min(1.0, 1.0 - over_cal))

    # Safety clamp
    def clamp01(x):
        return max(0.0001, min(0.9999, x))

    home_cal = clamp01(home_cal)
    draw_cal = clamp01(draw_cal)
    away_cal = clamp01(away_cal)
    over_cal = clamp01(over_cal)
    under_cal = clamp01(under_cal)

    return {
        "home_prob": home_cal,
        "draw_prob": draw_cal,
        "away_prob": away_cal,
        "over_2_5_prob": over_cal,
        "under_2_5_prob": under_cal,
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
      index["home_key – away_key"] = {
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

        home_key = normalize_team_name(home_raw)
        away_key = normalize_team_name(away_raw)
        match_key = f"{home_key} – {away_key}"

        best_home = best_draw = best_away = None
        best_over = best_under = None

        for bm in ev.get("bookmakers", []):
            for m in bm.get("markets", []):
                mk = m.get("key")

                if mk == "h2h":
                    outs = m.get("outcomes", [])
                    if len(outs) == 3:
                        try:
                            # Η σειρά στα docs είναι [home, away, draw] αλλά κάνουμε safe mapping
                            # προσπαθούμε να πιάσουμε home/away μέσω name αν υπάρχει
                            o0, o1, o2 = outs
                            prices = {
                                o0.get("name", "home").lower(): float(o0["price"]),
                                o1.get("name", "away").lower(): float(o1["price"]),
                                o2.get("name", "draw").lower(): float(o2["price"]),
                            }
                            # απλά παίρνουμε max ανά κατηγορία
                            best_home = max(best_home or 0, prices.get("home", prices.get(home_raw.lower(), 0)))
                            best_away = max(best_away or 0, prices.get("away", prices.get(away_raw.lower(), 0)))
                            best_draw = max(best_draw or 0, prices.get("draw", prices.get("x", 0)))
                        except Exception:
                            pass

                elif mk == "totals":
                    for o in m.get("outcomes", []):
                        name = (o.get("name") or "").lower()
                        try:
                            price = float(o["price"])
                        except Exception:
                            continue
                        # Πιάνουμε και "Over 2.5 Goals" κλπ
                        if "over 2.5" in name:
                            best_over = max(best_over or 0, price)
                        elif "under 2.5" in name:
                            best_under = max(best_under or 0, price)

        index[match_key] = {
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

    # 3) Loop fixtures → team stats → probabilities → fair
    for fx in all_fixtures:
        home_name = fx["home"]
        away_name = fx["away"]
        league_name = fx["league_name"]
        league_id = fx["league_id"]

        home_id = fx.get("home_id")
        away_id = fx.get("away_id")

        match_key_odds = f"{normalize_team_name(home_name)} – {normalize_team_name(away_name)}"

        # Πραγματικά team stats
        home_stats = fetch_team_recent_stats(home_id, league_id)
        away_stats = fetch_team_recent_stats(away_id, league_id)

        league_baseline = fetch_league_baselines(league_id)

        # ---- STEP 1: Expected goals (λ) ----
        lambda_home, lambda_away = compute_expected_goals(
            home_stats, away_stats, league_baseline
        )

        # ---- STEP 2: Probabilities από το μοντέλο ----
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

        # ---- STEP 3: FAIR odds από πιθανότητες ----
        fair_1 = implied(p_home)
        fair_x = implied(p_draw)
        fair_2 = implied(p_away)
        fair_over = implied(p_over)
        fair_under = implied(p_under)

        # ---- STEP 4: Offered odds από TheOddsAPI ----
        offered = odds_index_global.get(match_key_odds, {})
        off_home = offered.get("home")
        off_draw = offered.get("draw")
        off_away = offered.get("away")
        off_over = offered.get("over")
        off_under = offered.get("under")

        # ---- STEP 5: Formatting ημερομηνίας/ώρας ----
        dt = parser.isoparse(fx["date_raw"]).astimezone(datetime.timezone.utc)
        date_str = dt.date().isoformat()
        time_str = dt.strftime("%H:%M")

        # ---- STEP 6: Scores 1–10 (draw / over) ----
        def score_from_prob(p):
            if p is None:
                return None
            raw = p * 10.0
            raw = max(1.0, min(10.0, raw))
            return round(raw, 1)

        score_draw = score_from_prob(p_draw)
        score_over = score_from_prob(p_over)

        fixtures_out.append(
            {
                "fixture_id": fx["id"],
                "date": date_str,
                "time": time_str,
                "league_id": league_id,
                "league": league_name,
                "home": home_name,
                "away": away_name,
                "model": "bombay_production_v4",
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
                "score_draw": score_draw,
                "score_over": score_over,
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

    print(f"Thursday v4 READY. Fixtures: {len(fixtures)}", flush=True)


if __name__ == "__main__":
    main()
