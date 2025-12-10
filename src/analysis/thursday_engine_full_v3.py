import json
import requests
import datetime
from dateutil import parser

API_FOOTBALL_KEY = "<YOUR_API_KEY>"
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
ODDS_API_KEY = "<YOUR_ODDS_API_KEY>"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

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

# 3 ημέρες (για info)
WINDOW_HOURS = 72


# ------------------------- FAIR MODEL (dummy) -------------------------
def dummy_fair_model(match_key: str):
    """
    PROVISIONAL μοντέλο – placeholder μέχρι να κουμπώσει το κανονικό Poisson.
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


# ------------------------- FIXTURES -------------------------
def fetch_fixtures(league_id, date_from, date_to):
    """
    Τραβάει fixtures ΜΟΝΟ για τις επόμενες 3 μέρες (from/to),
    άρα ΔΕΝ ξαναφιλτράρουμε με ώρες μετά.
    """
    url = (
        f"{API_FOOTBALL_BASE}/fixtures?"
        f"league={league_id}&season=2025&from={date_from}&to={date_to}"
    )
    r = requests.get(url, headers=HEADERS_FOOTBALL).json()
    if not r.get("response"):
        return []

    out = []
    for fx in r["response"]:
        if fx["fixture"]["status"]["short"] != "NS":
            continue

        home_name = fx["teams"]["home"]["name"]
        away_name = fx["teams"]["away"]["name"]

        out.append(
            {
                "id": fx["fixture"]["id"],
                "league_id": league_id,
                "home": home_name,
                "away": away_name,
                "date_raw": fx["fixture"]["date"],
            }
        )
    return out


# ------------------------- ODDS -------------------------
def fetch_odds_for_league(league_name):
    """
    Τραβάει odds *μία φορά* από TheOddsAPI για τη συγκεκριμένη λίγκα.
    """
    league_to_sport = {
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

    sport_key = league_to_sport.get(league_name)
    if not sport_key or not ODDS_API_KEY:
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
    index["Home – Away"] = {... καλύτερες αποδόσεις ...}
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
def build_fixture_blocks(date_from, date_to):
    fixtures_out = []

    # 1) Μαζεύουμε fixtures από όλες τις λίγκες
    all_fixtures = []
    for lg_name, lg_id in LEAGUES.items():
        fx_list = fetch_fixtures(lg_id, date_from, date_to)
        for f in fx_list:
            f["league_name"] = lg_name
        all_fixtures.extend(fx_list)

    # 2) Odds: μία φορά ανά λίγκα -> index Home–Away
    odds_index = {}
    for lg_name in LEAGUES.keys():
        odds_data = fetch_odds_for_league(lg_name)
        league_index = build_odds_index(odds_data)
        odds_index.update(league_index)

    # 3) Χτίζουμε το τελικό block ανά fixture
    for fx in all_fixtures:
        home = fx["home"]
        away = fx["away"]
        league_name = fx["league_name"]
        league_id = fx["league_id"]

        match_key = f"{home} – {away}"

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

        dt = parser.isoparse(fx["date_raw"])
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

    return fixtures_out


# ------------------------- MAIN -------------------------
def main():
    today = datetime.datetime.utcnow().date()
    date_from = today.isoformat()
    date_to = (today + datetime.timedelta(days=3)).isoformat()

    fixtures = build_fixture_blocks(date_from, date_to)

    out = {
        "generated_at": datetime.datetime.utcnow().isoformat(),
        "window": {"from": date_from, "to": date_to, "hours": WINDOW_HOURS},
        "fixtures_total": len(fixtures),
        "fixtures": fixtures,
    }

    import os
    os.makedirs("logs", exist_ok=True)
    with open("logs/thursday_report_v3.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Thursday v3 READY.", flush=True)


if __name__ == "__main__":
    main()
