import os
import json
import requests
import datetime
from dateutil import parser

# ============================================================
#  THURSDAY ENGINE v3 (refined)
#  - Τραβάει fixtures από API-FOOTBALL
#  - Υπολογίζει dummy fair odds / probabilities
#  - (ΠΡΟΣΩΡΙΝΑ) ΔΕΝ χτυπάει TheOddsAPI για odds,
#    για να μην καίμε credits όσο κάνουμε debug.
#  - Γράφει logs/thursday_report_v3.json
# ============================================================

# ------------------------- API KEYS -------------------------
API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

FOOTBALL_SEASON = os.getenv("FOOTBALL_SEASON", "2025")

HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

# Αν θες να αρχίσει να χτυπάει TheOddsAPI, κάντο True.
USE_ODDS_API = False

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

# 3 ημέρες ακριβώς (72 ώρες)
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


# ------------------------- FAIR MODEL (dummy) -------------------------
def dummy_fair_model(match_key: str):
    """
    PROVISIONAL μοντέλο – placeholder μέχρι να κουμπώσει το κανονικό Poisson.

    Επιστρέφει probabilities που χρησιμοποιούνται για:
      - fair odds
      - draw_prob / over_prob

    Προς το παρόν ΙΔΙΟ για όλα τα ματς.
    """
    home_prob = 0.38
    draw_prob = 0.33
    away_prob = 0.29
    over_prob = 0.58  # P(Over 2.5)
    return {
        "home_prob": home_prob,
        "draw_prob": draw_prob,
        "away_prob": away_prob,
        "over_prob": over_prob,
    }


def implied(p):
    return 1.0 / p if p and p > 0 else None


# ------------------------- HELPERS: FIXTURES -------------------------
def fetch_fixtures(league_id, league_name):
    """
    Τραβάει fixtures από API-FOOTBALL για συγκεκριμένη λίγκα
    μέσα στο παράθυρο των 72 ωρών.
    """
    if not API_FOOTBALL_KEY:
        print("⚠️ Missing FOOTBALL_API_KEY – NO fixtures will be fetched!", flush=True)
        return []

    url = f"{API_FOOTBALL_BASE}/fixtures?league={league_id}&season={FOOTBALL_SEASON}"

    try:
        r = requests.get(url, headers=HEADERS_FOOTBALL, timeout=20).json()
    except Exception as e:
        print(f"⚠️ Error fetching fixtures for {league_name}: {e}", flush=True)
        return []

    if not r.get("response"):
        print(f"⚠️ No fixtures response for league {league_name}", flush=True)
        return []

    out = []
    # AWARE UTC datetime
    now = datetime.datetime.now(datetime.timezone.utc)

    for fx in r["response"]:
        # Θέλουμε μόνο μη-ξεκινήμενα ματς
        if fx["fixture"]["status"]["short"] != "NS":
            continue

        # Ημερομηνία fixture (ISO με timezone)
        dt = parser.isoparse(fx["fixture"]["date"]).astimezone(datetime.timezone.utc)
        diff = (dt - now).total_seconds() / 3600.0

        # Φίλτρο στο παράθυρο 0–WINDOW_HOURS
        if not (0 <= diff <= WINDOW_HOURS):
            continue

        home_name = fx["teams"]["home"]["name"]
        away_name = fx["teams"]["away"]["name"]

        out.append(
            {
                "id": fx["fixture"]["id"],
                "league_id": league_id,
                "league_name": league_name,
                "home": home_name,
                "away": away_name,
                "date_raw": fx["fixture"]["date"],
            }
        )

    print(f"→ {league_name}: {len(out)} fixtures within window", flush=True)
    return out


# ------------------------- HELPERS: ODDS (TheOddsAPI) -------------------------
def fetch_odds_for_league(league_name):
    """
    Αν USE_ODDS_API = True, τραβάει odds *μία φορά* από TheOddsAPI
    για τη συγκεκριμένη λίγκα.
    """
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
    Φτιάχνει index:
      index["Home – Away"] = {
          'home': best_home,
          'draw': best_draw,
          'away': best_away,
          'over': best_over_2_5,
          'under': best_under_2_5
      }

    ΠΡΟΣΟΧΗ: Τα ονόματα ομάδων μπορεί να μην ταιριάζουν 100% με API-FOOTBALL.
    Αργότερα μπορούμε να βάλουμε normalisation / mapping.
    """
    index = {}
    for ev in odds_data:
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
                        price = float(o["price"])
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


# ------------------------- BUILD FIXTURE BLOCKS -------------------------
def build_fixture_blocks():
    fixtures_out = []

    print(f"Using FOOTBALL_SEASON={FOOTBALL_SEASON}", flush=True)
    print(f"Window: next {WINDOW_HOURS} hours", flush=True)

    if not API_FOOTBALL_KEY:
        print("❌ FOOTBALL_API_KEY is missing. Aborting fixture fetch.", flush=True)
        return []

    # 1) Μαζεύουμε fixtures από όλες τις λίγκες
    all_fixtures = []
    for lg_name, lg_id in LEAGUES.items():
        fx_list = fetch_fixtures(lg_id, lg_name)
        all_fixtures.extend(fx_list)

    print(f"Total raw fixtures collected: {len(all_fixtures)}", flush=True)

    # 2) Odds index
    if USE_ODDS_API:
        odds_index = {}
        for lg_name in LEAGUES.keys():
            odds_data = fetch_odds_for_league(lg_name)
            league_index = build_odds_index(odds_data)
            odds_index.update(league_index)
        print(f"Odds index built for {len(odds_index)} matches", flush=True)
    else:
        odds_index = {}
        print("⚠️ USE_ODDS_API = False → δεν τραβάμε odds από TheOddsAPI.", flush=True)

    # 3) Δέσιμο fixtures + μοντέλο + odds
    for fx in all_fixtures:
        home = fx["home"]
        away = fx["away"]
        league_name = fx["league_name"]
        league_id = fx["league_id"]

        match_key = f"{home} – {away}"

        # Μοντέλο fair probabilities (dummy προς το παρόν)
        probs = dummy_fair_model(match_key)
        p_home = probs["home_prob"]
        p_draw = probs["draw_prob"]
        p_away = probs["away_prob"]
        p_over = probs["over_prob"]
        p_under = max(0.0, 1.0 - p_over)

        fair_1 = implied(p_home)
        fair_x = implied(p_draw)
        fair_2 = implied(p_away)
        fair_over = implied(p_over)
        fair_under = implied(p_under)

        offered = odds_index.get(match_key, {})
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
                "home": home,
                "away": away,
                "model": "dummy_v1",
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

    print(f"Thursday fixtures_out: {len(fixtures_out)}", flush=True)
    return fixtures_out


def main():
    fixtures = build_fixture_blocks()

    # Αν απέτυχαν όλα (π.χ. δεν υπάρχει API key), μην γράψεις άδειο αρχείο
    if fixtures is None:
        print("No fixtures generated. Skipping report write.", flush=True)
        return

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

    print(f"Thursday v3 READY. Fixtures: {len(fixtures)}", flush=True)


if __name__ == "__main__":
    main()
