import os
import json, requests, datetime, math
from dateutil import parser

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

# ------------------------- LEAGUES -------------------------
LEAGUES = {
    "Premier League": 39, "Championship": 40,
    "Ligue 1": 61, "Ligue 2": 62,
    "Bundesliga": 78, "Serie A": 135,
    "Serie B": 136, "La Liga": 140,
    "Liga Portugal 1": 94,
}

# ------------------------- FILTERS -------------------------
DRAW_PROB_THRESHOLD = 0.35
DRAW_MIN_OFFERED = 2.90
KELLY_MIN_PROB = 0.20
KELLY_MIN_EDGE = 0.10

# 3 ημέρες ακριβώς (72 ώρες)
WINDOW_HOURS = 72

# ------------------------- HELPERS -------------------------
def fetch_fixtures(league_id):
    url = f"{API_FOOTBALL_BASE}/fixtures?league={league_id}&season=2025"
    r = requests.get(url, headers=HEADERS_FOOTBALL).json()
    if not r.get("response"):
        return []
    out = []
    now = datetime.datetime.utcnow()
    for fx in r["response"]:
        if fx["fixture"]["status"]["short"] != "NS":
            continue
        dt = parser.isoparse(fx["fixture"]["date"])
        diff = (dt - now).total_seconds() / 3600
        if 0 <= diff <= WINDOW_HOURS:   # 3-day window filter
            out.append(
                {
                    "id": fx["fixture"]["id"],
                    "league": league_id,
                    "match": f"{fx['teams']['home']['name']} – {fx['teams']['away']['name']}",
                    "date": fx["fixture"]["date"],
                }
            )
    return out


def fetch_odds_for_league(league_name):
    url = f"{ODDS_BASE_URL}/soccer/{league_name}/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": "eu", "markets": "h2h,totals"}
    try:
        r = requests.get(url, params=params)
        if r.status_code != 200:
            return {}
        return r.json()
    except Exception:
        return {}


def build_odds_index(odds_data):
    index = {}
    for item in odds_data:
        match = f"{item.get('home_team', '')} – {item.get('away_team', '')}"
        m = {"home": None, "draw": None, "away": None, "over": None, "under": None}

        for b in item.get("bookmakers", []):
            for mk in b.get("markets", []):
                if mk["key"] == "h2h":
                    try:
                        m["home"] = float(mk["outcomes"][0]["price"])
                        m["away"] = float(mk["outcomes"][1]["price"])
                        m["draw"] = float(mk["outcomes"][2]["price"])
                    except Exception:
                        pass
                if mk["key"] == "totals":
                    for o in mk["outcomes"]:
                        if o["name"] == "Over 2.5":
                            m["over"] = float(o["price"])
                        if o["name"] == "Under 2.5":
                            m["under"] = float(o["price"])
        index[match] = m
    return index


def dummy_fair_model(match):
    """Προσωρινό μοντέλο – θα το αντικαταστήσουμε με Poisson."""
    return {
        "home_prob": 0.38,
        "draw_prob": 0.33,
        "away_prob": 0.29,
        "over_prob": 0.58,
    }


def implied(p):
    return 1 / p if p > 0 else None


def build_fixture_blocks():
    output = []
    now = datetime.datetime.utcnow()

    all_fixtures = []
    for lg_name, lg_id in LEAGUES.items():
        fx = fetch_fixtures(lg_id)
        for f in fx:
            f["league_name"] = lg_name
        all_fixtures.extend(fx)

    # Fetch odds ONCE per league
    odds_index = {}
    for lg_name in LEAGUES:
        data = fetch_odds_for_league(lg_name)
        odds_index.update(build_odds_index(data))

    for fx in all_fixtures:
        probs = dummy_fair_model(fx["match"])

        fair_1 = implied(probs["home_prob"])
        fair_x = implied(probs["draw_prob"])
        fair_2 = implied(probs["away_prob"])
        fair_over = implied(probs["over_prob"])

        offered = odds_index.get(fx["match"], {})

        output.append(
            {
                "match": fx["match"],
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
    # ΣΩΣΤΟ ΟΝΟΜΑ για να το διαβάζει το Friday
    with open("logs/thursday_report_v3.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Thursday v3 READY.")


if __name__ == "__main__":
    main()
