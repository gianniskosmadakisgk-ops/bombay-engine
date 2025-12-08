import os
import json
import itertools
from datetime import datetime
import re
import requests

# ==============================================================================
#  FRIDAY SHORTLIST V4 — FULL PRODUCTION VERSION
#  - Loads Thursday Report
#  - Pulls offered odds from TheOddsAPI
#  - Computes Draw Singles, Over Singles, FunBet Draw, FunBet Over, Kelly
#  - Produces clean JSON for UI
# ==============================================================================

THURSDAY_REPORT_PATH = "logs/thursday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v4.json"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

# ------------------------------------------------------------------------------
# BANKROLLS (in units)
# ------------------------------------------------------------------------------
BANKROLL_DRAW = 1000
BANKROLL_OVER = 1000
BANKROLL_FUN_DRAW = 300
BANKROLL_FUN_OVER = 300
BANKROLL_KELLY = 600

UNIT = 1.0

# ------------------------------------------------------------------------------
# LEAGUE PRIORITIES
# ------------------------------------------------------------------------------
DRAW_PRIORITY_LEAGUES = {
    "Ligue 1",
    "Serie A",
    "La Liga",
    "Championship",
    "Serie B",
    "Ligue 2",
    "Liga Portugal 2",
    "Swiss Super League",
}

OVER_PRIORITY_LEAGUES = {
    "Bundesliga",
    "Eredivisie",
    "Jupiler Pro League",
    "Superliga",
    "Allsvenskan",
    "Eliteserien",
    "Swiss Super League",
    "Liga Portugal 1",
}

# ------------------------------------------------------------------------------
# ODDS SUPPORT MAP
# ------------------------------------------------------------------------------
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
    "Brazil Serie A": "soccer_brazil_serie_a"  # plus if needed later
}

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------
def log(msg):
    print(msg, flush=True)

def normalize_team(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\b(fc|cf|afc|cfc|ac|sc|bk)\b", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# ------------------------------------------------------------------------------
# Load Thursday fixtures
# ------------------------------------------------------------------------------
def load_thursday():
    if not os.path.exists(THURSDAY_REPORT_PATH):
        raise FileNotFoundError("Thursday report missing")

    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f).get("fixtures", [])

# ------------------------------------------------------------------------------
# ODDS API
# ------------------------------------------------------------------------------
def get_odds_for_league(sport_key: str):
    if not ODDS_API_KEY:
        log("⚠️ Missing ODDS_API_KEY")
        return []

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
    }

    try:
        res = requests.get(f"{ODDS_BASE_URL}/{sport_key}/odds", params=params, timeout=20)
        if res.status_code != 200:
            log(f"⚠️ Odds error [{sport_key}] status={res.status_code}")
            return []
        return res.json()
    except Exception as e:
        log(f"⚠️ Odds request error: {e}")
        return []

def build_odds_index(fixtures):
    odds_index = {}

    leagues = sorted({f["league"] for f in fixtures if f.get("league") in LEAGUE_TO_SPORT})

    for lg in leagues:
        key = LEAGUE_TO_SPORT[lg]
        events = get_odds_for_league(key)
        log(f"Fetched {len(events)} odds events for {lg}")

        for ev in events:
            h_raw = ev.get("home_team", "")
            a_raw = ev.get("away_team", "")
            h = normalize_team(h_raw)
            a = normalize_team(a_raw)

            best_home = best_draw = best_away = best_over = None

            for bm in ev.get("bookmakers", []):
                for m in bm.get("markets", []):
                    mk = m.get("key")

                    if mk == "h2h":
                        for o in m.get("outcomes", []):
                            nm = normalize_team(o["name"])
                            price = float(o["price"])
                            if nm == h:
                                best_home = max(best_home or 0, price)
                            elif nm == a:
                                best_away = max(best_away or 0, price)
                            elif nm == "draw":
                                best_draw = max(best_draw or 0, price)

                    elif mk == "totals":
                        for o in m.get("outcomes", []):
                            name = o.get("name", "").lower()
                            price = float(o["price"])
                            if "over" in name and "2.5" in name:
                                best_over = max(best_over or 0, price)

            odds_index[(h, a)] = {
                "home": best_home,
                "draw": best_draw,
                "away": best_away,
                "over_2_5": best_over,
            }

    return odds_index

# ------------------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------------------
def compute_draw_score(draw_prob, league):
    score = draw_prob * 100
    if league in DRAW_PRIORITY_LEAGUES:
        score *= 1.05
    return score

def compute_over_score(over_prob, league):
    score = over_prob * 100
    if league in OVER_PRIORITY_LEAGUES:
        score *= 1.05
    return score

# ------------------------------------------------------------------------------
# PICKS
# ------------------------------------------------------------------------------
def generate_picks(fixtures, odds_index):
    draw_singles = []
    over_singles = []
    kelly = []

    for f in fixtures:
        home = f["home"]
        away = f["away"]
        league = f["league"]

        fair_x = f["fair_x"]
        fair_over = f["fair_over_2_5"]

        draw_prob = f["draw_prob"]
        over_prob = f["over_2_5_prob"]

        h = normalize_team(home)
        a = normalize_team(away)
        odds = odds_index.get((h, a), {})

        offered_x = odds.get("draw") or None
        offered_over = odds.get("over_2_5") or None

        draw_score = compute_draw_score(draw_prob, league)
        over_score = compute_over_score(over_prob, league)

        # ---------------- DRAW SINGLES ----------------
        if draw_prob >= 0.30 and fair_x <= 3.40:
            draw_singles.append({
                "match": f"{home} - {away}",
                "league": league,
                "fair": fair_x,
                "prob": draw_prob,
                "score": round(draw_score, 1),
                "offered": offered_x,
                "stake": UNIT
            })

        # ---------------- OVER SINGLES ----------------
        if over_prob >= 0.60 and fair_over <= 1.75:
            over_singles.append({
                "match": f"{home} - {away}",
                "league": league,
                "fair": fair_over,
                "prob": over_prob,
                "score": round(over_score, 1),
                "offered": offered_over,
                "stake": UNIT
            })

        # ---------------- KELLY ----------------
        def add_kelly(label, fair, offered, prob):
            if not offered:
                return
            p = prob
            q = 1 - p
            b = offered - 1
            edge = (p * offered) - 1
            if edge < 0.10:
                return
            stake = BANKROLL_KELLY * (b * p - q) / b
            if stake > 0:
                kelly.append({
                    "match": f"{home} - {away}",
                    "league": league,
                    "market": label,
                    "fair": fair,
                    "offered": offered,
                    "prob": p,
                    "stake": round(stake, 2)
                })

        if offered_x:
            add_kelly("Draw", fair_x, offered_x, draw_prob)

        if offered_over:
            add_kelly("Over 2.5", fair_over, offered_over, over_prob)

    return draw_singles, over_singles, kelly

# ------------------------------------------------------------------------------
# FUNBET SYSTEMS
# ------------------------------------------------------------------------------
def funbet_draw(draw_singles):
    picks = sorted(draw_singles, key=lambda x: x["score"], reverse=True)
    picks = picks[:7]

    n = len(picks)
    if n < 3:
        return {"system": None, "columns": 0, "total_stake": 0, "picks": []}

    if n == 3:
        sys = "3/3"
        cols = 1
    elif n == 4:
        sys = "3/4"
        cols = 4
    elif n == 5:
        sys = "3/5"
        cols = 10
    elif n == 6:
        sys = "4/6"
        cols = 15
    else:
        sys = "4/7"
        cols = 35

    total = 5  # always 5 units
    return {
        "system": sys,
        "columns": cols,
        "total_stake": total,
        "picks": picks
    }

def funbet_over(over_singles):
    picks = sorted(over_singles, key=lambda x: x["score"], reverse=True)
    picks = picks[:7]

    n = len(picks)
    if n < 3:
        return {"system": None, "columns": 0, "total_stake": 0, "picks": []}

    if n == 3:
        sys = "3/3"
        cols = 1
    elif n == 4:
        sys = "2/4"
        cols = 6
    elif n == 5:
        sys = "2/5"
        cols = 10
    elif n == 6:
        sys = "3/6"
        cols = 20
    else:
        sys = "3/7"
        cols = 35

    total = 5
    return {
        "system": sys,
        "columns": cols,
        "total_stake": total,
        "picks": picks
    }

# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    log("🚀 Running Friday Shortlist V4")

    fixtures = load_thursday()
    odds_index = build_odds_index(fixtures)

    draw_singles, over_singles, kelly = generate_picks(fixtures, odds_index)

    fb_draw = funbet_draw(draw_singles)
    fb_over = funbet_over(over_singles)

    bankrolls = {
        "draw": {
            "before": BANKROLL_DRAW,
            "after": BANKROLL_DRAW - len(draw_singles) * UNIT,
            "open": len(draw_singles) * UNIT,
        },
        "over": {
            "before": BANKROLL_OVER,
            "after": BANKROLL_OVER - len(over_singles) * UNIT,
            "open": len(over_singles) * UNIT,
        },
        "fun_draw": {
            "before": BANKROLL_FUN_DRAW,
            "after": BANKROLL_FUN_DRAW - fb_draw["total_stake"],
            "open": fb_draw["total_stake"],
        },
        "fun_over": {
            "before": BANKROLL_FUN_OVER,
            "after": BANKROLL_FUN_OVER - fb_over["total_stake"],
            "open": fb_over["total_stake"],
        },
        "kelly": {
            "before": BANKROLL_KELLY,
            "after": BANKROLL_KELLY - sum(k["stake"] for k in kelly),
            "open": sum(k["stake"] for k in kelly),
        }
    }

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "fixtures_total": len(fixtures),
        "draw_singles": draw_singles,
        "over_singles": over_singles,
        "funbet_draw": fb_draw,
        "funbet_over": fb_over,
        "kelly": kelly,
        "bankrolls": bankrolls,
    }

    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log(f"✅ Friday Shortlist V4 saved → {FRIDAY_REPORT_PATH}")


if __name__ == "__main__":
    main()
