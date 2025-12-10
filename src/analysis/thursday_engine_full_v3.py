import json
import requests
import datetime
from dateutil import parser

API_FOOTBALL_KEY = "<YOUR_API_KEY>"
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

ODDS_API_KEY = "<YOUR_ODDS_API_KEY>"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

# ------------------------- LEAGUES (API-FOOTBALL IDs) -------------------------
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

# ------------------------- LEAGUE → TheOddsAPI sport key ----------------------
LEAGUE_TO_SPORT = {
    "Premier League": "soccer_epl",
    "Championship": "soccer_efl_champ",
    "Ligue 1": "soccer_france_ligue_one",
    "Ligue 2": "soccer_france_ligue_two",
    "Bundesliga": "soccer_germany_bundesliga",
    "Serie A": "soccer_italy_serie_a",
    "Serie B": "soccer_italy_serie_b",
    "La Liga": "soccer_spain_la_liga",
    "Liga Portugal 1": "soccer_portugal_primeira_liga",
}

# ------------------------- FILTERS (για αργότερα) ----------------------------
DRAW_PROB_THRESHOLD = 0.35
DRAW_MIN_OFFERED = 2.90
KELLY_MIN_PROB = 0.20
KELLY_MIN_EDGE = 0.10

# 3 ημέρες ακριβώς (72 ώρες)
WINDOW_HOURS = 72


# ------------------------- HELPERS -------------------------------------------
def fetch_fixtures(league_id: int):
    """Φέρνει fixtures 3ημέρου από API-Football για συγκεκριμένη λίγκα."""
    url = f"{API_FOOTBALL_BASE}/fixtures?league={league_id}&season=2025"
    try:
        r = requests.get(url, headers=HEADERS_FOOTBALL, timeout=20)
        data = r.json()
    except Exception:
        return []

    if not data.get("response"):
        return []

    out = []
    now = datetime.datetime.utcnow()
    for fx in data["response"]:
        if fx["fixture"]["status"]["short"] != "NS":
            continue
        dt = parser.isoparse(fx["fixture"]["date"])
        diff_hours = (dt - now).total_seconds() / 3600.0
        if 0 <= diff_hours <= WINDOW_HOURS:
            out.append(
                {
                    "id": fx["fixture"]["id"],
                    "league": league_id,
                    "match": f"{fx['teams']['home']['name']} – {fx['teams']['away']['name']}",
                    "date": fx["fixture"]["date"],
                }
            )
    return out


def fetch_odds_for_league(sport_key: str):
    """ΜΙΑ κλήση στο TheOddsAPI ανά λίγκα (sport_key)."""
    if not ODDS_API_KEY:
        return []

    url = f"{ODDS_BASE_URL}/{sport_key}/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return []
        return r.json()
    except Exception:
        return []


def build_odds_index(odds_data):
    """
    Φτιάχνει index:
      odds_index["Home – Away"] = {
          "home": ..., "draw": ..., "away": ..., "over": ..., "under": ...
      }
    """
    index = {}
    for ev in odds_data:
        home_raw = ev.get("home_team", "") or ""
        away_raw = ev.get("away_team", "") or ""
        match_name = f"{home_raw} – {away_raw}"

        best_home = best_draw = best_away = None
        best_over = best_under = None

        for bm in ev.get("bookmakers", []):
            for mk in bm.get("markets", []):
                key = mk.get("key")

                if key == "h2h":
                    outcomes = mk.get("outcomes", [])
                    for o in outcomes:
                        name = o.get("name", "")
                        price = float(o["price"])
                        if name == home_raw:
                            best_home = max(best_home or 0, price)
                        elif name == away_raw:
                            best_away = max(best_away or 0, price)
                        elif name.lower() == "draw":
                            best_draw = max(best_draw or 0, price)

                elif key == "totals":
                    for o in mk.get("outcomes", []):
                        name = o.get("name", "").lower()
                        price = float(o["price"])
                        if "over" in name and "2.5" in name:
                            best_over = max(best_over or 0, price)
                        if "under" in name and "2.5" in name:
                            best_under = max(best_under or 0, price)

        index[match_name] = {
            "home": best_home,
            "draw": best_draw,
            "away": best_away,
            "over": best_over,
            "under": best_under,
        }

    return index


def dummy_fair_model(match: str):
    """Προσωρινό μοντέλο fair probabilities – να αντικατασταθεί με Poisson."""
    return {
        "home_prob": 0.38,
        "draw_prob": 0.33,
        "away_prob": 0.29,
        "over_prob": 0.58,
    }


def implied(p: float):
    return 1.0 / p if p > 0 else None


def build_fixture_blocks():
    all_fixtures = []

    # 1) Fixtures 3ημέρου από όλες τις λίγκες
    for lg_name, lg_id in LEAGUES.items():
        fx_list = fetch_fixtures(lg_id)
        for f in fx_list:
            f["league_name"] = lg_name
        all_fixtures.extend(fx_list)

    # 2) Odds ανά λίγκα (sport_key) – ΜΙΑ κλήση ανά λίγκα
    odds_index = {}
    for lg_name in LEAGUES:
        sport_key = LEAGUE_TO_SPORT.get(lg_name)
        if not sport_key:
            continue
        odds_data = fetch_odds_for_league(sport_key)
        odds_index.update(build_odds_index(odds_data))

    # 3) Δένουμε fair + offered ανά παιχνίδι
    output = []
    for fx in all_fixtures:
        match_name = fx["match"]
        probs = dummy_fair_model(match_name)

        fair_1 = implied(probs["home_prob"])
        fair_x = implied(probs["draw_prob"])
        fair_2 = implied(probs["away_prob"])
        fair_over = implied(probs["over_prob"])

        offered = odds_index.get(match_name, {})

        output.append(
            {
                "match": match_name,
                "league": fx["league_name"],
                "date": fx["date"],
                "probs": probs,
                "fair": {
                    "home": fair_1,
                    "draw": fair_x,
                    "away": fair_2,
                    "over": fair_over,
                },
                "offered": offered,
            }
        )

    return output


def main():
    fixtures = build_fixture_blocks()
    out = {
        "timestamp": str(datetime.datetime.utcnow()),
        "window_hours": WINDOW_HOURS,
        "fixtures_total": len(fixtures),
        "fixtures": fixtures,
    }

    # 🔴 ΣΗΜΑΝΤΙΚΟ: γράφουμε στο v3 γιατί αυτό διαβάζει το Friday script
    with open("logs/thursday_report_v3.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("Thursday v3 READY.")


if __name__ == "__main__":
    main()
