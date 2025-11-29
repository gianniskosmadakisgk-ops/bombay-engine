import os
import json
from datetime import datetime
import requests
import itertools
import re

# ======================================================
#  FRIDAY SHORTLIST v2  (Production)
# ======================================================

THURSDAY_REPORT_PATH = "logs/thursday_report_v1.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v2.json"
HISTORY_PATH = "logs/bets_history_v2.json"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

os.makedirs("logs", exist_ok=True)

# ---------------------- BANKROLLS ----------------------
DRAW_WALLET = 400
OVER_WALLET = 300
FANBET_DRAW_WALLET = 100
FANBET_OVER_WALLET = 100
KELLY_WALLET = 300

# ---------------------- THRESHOLDS ---------------------
DRAW_MIN_SCORE = 7.5
DRAW_MIN_ODDS = 2.70
OVER_MIN_SCORE = 7.5
OVER_MIN_FAIR = 1.70

KELLY_VALUE_THRESHOLD = 0.15
KELLY_FRACTION = 0.40

FUNBET_DRAW_STAKE_PER_COL = 3.0
FUNBET_OVER_STAKE_PER_COL = 4.0

DRAW_LEAGUES = {
    "Ligue 1", "Serie A", "La Liga", "Championship",
    "Serie B", "Ligue 2", "Swiss Super League"
}

OVER_LEAGUES = {
    "Bundesliga", "Eredivisie", "Jupiler Pro League",
    "Superliga", "Allsvenskan", "Eliteserien",
    "Swiss Super League", "Liga Portugal 1"
}

# league name -> sport_key
LEAGUE_TO_SPORT = {
    "Premier League": "soccer_epl",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
}

def log(msg: str):
    print(msg, flush=True)

# ------------------------------------------------------
# Load Thursday data
# ------------------------------------------------------
def load_thursday_fixtures():
    if not os.path.exists(THURSDAY_REPORT_PATH):
        raise FileNotFoundError(f"Thursday report not found: {THURSDAY_REPORT_PATH}")

    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    log(f"Loaded {len(data.get('fixtures', []))} fixtures from Thursday report.")
    return data.get("fixtures", [])

# ------------------------------------------------------
# Odds API
# ------------------------------------------------------
def api_get_odds(sport_key: str):
    if not ODDS_API_KEY:
        log("⚠️ No ODDS_API_KEY set — cannot fetch odds.")
        return []

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
    }

    url = f"{ODDS_BASE_URL}/{sport_key}/odds"

    try:
        res = requests.get(url, params=params, timeout=20)
    except Exception as e:
        log(f"⚠️ Request error for {sport_key}: {e}")
        return []

    if res.status_code != 200:
        log(f"⚠️ Odds API {res.status_code} for {sport_key}: {res.text[:150]}")
        return []

    return res.json()

def normalize_team(name: str) -> str:
    if not name:
        return ""
    s = name.lower()
    s = re.sub(r"\b(fc|afc|cf|sc|bk|ac)\b", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()

def build_odds_index(fixtures):
    leagues_used = sorted({f["league"] for f in fixtures if f["league"] in LEAGUE_TO_SPORT})

    log(f"Leagues with odds support: {leagues_used}")

    odds_index = {}
    total_events = 0

    for league in leagues_used:
        sport_key = LEAGUE_TO_SPORT[league]
        events = api_get_odds(sport_key)
        log(f"Fetched {len(events)} odds events for {league} ({sport_key})")
        total_events += len(events)

        for ev in events:
            home = normalize_team(ev.get("home_team", ""))
            away = normalize_team(ev.get("away_team", ""))

            best_home = best_away = best_draw = best_over = None

            for b in ev.get("bookmakers", []):
                for m in b.get("markets", []):
                    if m["key"] == "h2h":
                        for o in m["outcomes"]:
                            price = float(o.get("price") or 0)
                            if price <= 0:
                                continue

                            name = normalize_team(o["name"])
                            if name == home:
                                best_home = max(best_home or 0, price)
                            elif name == away:
                                best_away = max(best_away or 0, price)
                            elif o["name"].lower() == "draw":
                                best_draw = max(best_draw or 0, price)

                    elif m["key"] == "totals":
                        for o in m["outcomes"]:
                            name = o["name"].lower()
                            price = float(o.get("price") or 0)
                            if "over" in name and "2.5" in name:
                                best_over = max(best_over or 0, price)

            odds_index[(home, away)] = {
                "odds_home": best_home,
                "odds_draw": best_draw,
                "odds_away": best_away,
                "odds_over_2_5": best_over,
            }

    log(f"Built odds index for {len(odds_index)} matches. Total events: {total_events}")
    return odds_index

# ------------------------------------------------------
# Pick Generators
# ------------------------------------------------------
def flat_stake(score):
    if score >= 8.5:
        return 20
    if score >= 7.5:
        return 15
    return 0

def generate_picks(fixtures, odds_index):
    draw_singles = []
    over_singles = []
    kelly_picks = []
    matched = 0

    for f in fixtures:
        league = f["league"]
        match = f["match"]

        fair_1 = f["fair_1"]
        fair_x = f["fair_x"]
        fair_2 = f["fair_2"]
        fair_over = f["fair_over"]

        score_draw = float(f["score_draw"])
        score_over = float(f["score_over"])

        home_name, away_name = [x.strip() for x in match.split("-")]
        home_norm = normalize_team(home_name)
        away_norm = normalize_team(away_name)

        odds = odds_index.get((home_norm, away_norm)) or odds_index.get((away_norm, home_norm))
        if odds:
            matched += 1
        else:
            odds = {}

        odds_home = odds.get("odds_home")
        odds_x = odds.get("odds_draw")
        odds_away = odds.get("odds_away")
        odds_over = odds.get("odds_over_2_5")

        # DRAW singles
        if league in DRAW_LEAGUES and fair_x and score_draw >= DRAW_MIN_SCORE:
            source = "market" if odds_x else "model"
            offered = odds_x or fair_x
            diff = (offered - fair_x) / fair_x if odds_x else 0
            if offered >= DRAW_MIN_ODDS:
                stake = flat_stake(score_draw)
                if stake:
                    draw_singles.append({
                        "match": match,
                        "league": league,
                        "odds": round(offered, 2),
                        "fair": fair_x,
                        "score": score_draw,
                        "stake": stake,
                        "value_raw": diff,
                        "odds_source": source,
                    })

        # OVER singles
        if league in OVER_LEAGUES and fair_over and score_over >= OVER_MIN_SCORE:
            source = "market" if odds_over else "model"
            offered = odds_over or fair_over
            diff = (offered - fair_over) / fair_over if odds_over else 0
            if fair_over >= OVER_MIN_FAIR:
                stake = flat_stake(score_over)
                if stake:
                    over_singles.append({
                        "match": match,
                        "league": league,
                        "odds": round(offered, 2),
                        "fair": fair_over,
                        "score": score_over,
                        "stake": stake,
                        "value_raw": diff,
                        "odds_source": source,
                    })

        # ----------------------------------------
        # KELLY
        # ----------------------------------------
        def add_kelly(label, fair, offered):
            if not fair or not offered:
                return
            diff = (offered - fair) / fair
            if diff < KELLY_VALUE_THRESHOLD:
                return

            p = 1.0 / fair
            q = 1.0 - p
            b = offered - 1
            k = (b * p - q) / b
            if k <= 0:
                return

            stake = round(KELLY_WALLET * k * KELLY_FRACTION, 2)
            if stake <= 0:
                return

            kelly_picks.append({
                "match": match,
                "league": league,
                "market": label,
                "fair": fair,
                "offered": offered,
                "diff": f"{diff:+.0%}",
                "stake (€)": stake,
            })

        if odds_home:
            add_kelly("Home", fair_1, odds_home)
        if odds_x:
            add_kelly("Draw", fair_x, odds_x)
        if odds_away:
            add_kelly("Away", fair_2, odds_away)
        if odds_over:
            add_kelly("Over 2.5", fair_over, odds_over)

    draw_singles = sorted(draw_singles, key=lambda x: (x["score"], x["value_raw"]), reverse=True)[:10]
    over_singles = sorted(over_singles, key=lambda x: (x["score"], x["value_raw"]), reverse=True)[:10]
    kelly_picks = sorted(kelly_picks, key=lambda x: x["stake (€)"], reverse=True)[:10]

    log(f"Matched odds: {matched}/{len(fixtures)}")
    return draw_singles, over_singles, kelly_picks

# ------------------------------------------------------
# Funbet systems
# ------------------------------------------------------
def build_funbet_draw(draws):
    picks = sorted(draws, key=lambda x: x["score"], reverse=True)[:6]
    n = len(picks)

    if n >= 6:
        sizes = [4, 5, 6]
        system = "4-5-6"
    elif n == 5:
        sizes = [3, 4, 5]
        system = "3-4-5"
    else:
        return {"picks": picks, "system": None, "columns": 0, "total_stake": 0}

    cols = sum(1 for r in sizes for _ in itertools.combinations(range(n), r))
    total = cols * FUNBET_DRAW_STAKE_PER_COL

    return {
        "picks": picks,
        "system": system,
        "columns": cols,
        "total_stake": total,
    }

def build_funbet_over(overs):
    picks = sorted(overs, key=lambda x: x["score"], reverse=True)[:6]
    n = len(picks)
    if n < 3:
        return {"picks": picks, "system": None, "columns": 0, "total_stake": 0}

    cols = sum(1 for _ in itertools.combinations(range(n), 2))
    total = cols * FUNBET_OVER_STAKE_PER_COL

    return {
        "picks": picks,
        "system": f"2-from-{n}",
        "columns": cols,
        "total_stake": total,
    }

# ------------------------------------------------------
# Bankroll Summary
# ------------------------------------------------------
def bankroll_summary(draw_s, over_s, fbd, fbo, kelly):
    draw_spent = sum(x["stake"] for x in draw_s)
    over_spent = sum(x["stake"] for x in over_s)
    fb_draw_spent = fbd.get("total_stake", 0)
    fb_over_spent = fbo.get("total_stake", 0)
    kelly_spent = sum(x["stake (€)"] for x in kelly)

    return [
        {"wallet": "Draw Singles", "spent": draw_spent, "after": DRAW_WALLET - draw_spent},
        {"wallet": "Over Singles", "spent": over_spent, "after": OVER_WALLET - over_spent},
        {"wallet": "FunBet Draw", "spent": fb_draw_spent, "after": FANBET_DRAW_WALLET - fb_draw_spent},
        {"wallet": "FunBet Over", "spent": fb_over_spent, "after": FANBET_OVER_WALLET - fb_over_spent},
        {"wallet": "Kelly", "spent": kelly_spent, "after": KELLY_WALLET - kelly_spent},
    ]

# ------------------------------------------------------
# History Logging
# ------------------------------------------------------
def append_week_to_history(draw_s, over_s, fbd, fbo, kelly):
    if not os.path.exists(HISTORY_PATH):
        log("🟢 No existing history → creating fresh file.")
        hist = []
    else:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            hist = json.load(f)

    week = f"{datetime.utcnow().date()}"

    hist.append({
        "week": week,
        "draw_singles": draw_s,
        "over_singles": over_s,
        "funbet_draw": fbd,
        "funbet_over": fbo,
        "kelly": kelly,
        "generated_at": datetime.utcnow().isoformat(),
    })

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2, ensure_ascii=False)

    log(f"🟢 Appended Friday snapshot to history ({week}).")

# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    log("🎯 Running Friday Shortlist (v2)...")

    fixtures = load_thursday_fixtures()
    odds_index = build_odds_index(fixtures)
    draw_s, over_s, kelly = generate_picks(fixtures, odds_index)

    fbd = build_funbet_draw(draw_s)
    fbo = build_funbet_over(over_s)
    banks = bankroll_summary(draw_s, over_s, fbd, fbo, kelly)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "draw_singles": draw_s,
        "over_singles": over_s,
        "funbet_draw": fbd,
        "funbet_over": fbo,
        "kelly": kelly,
        "bankroll": banks,
    }

    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log(f"🟢 Friday shortlist saved → {FRIDAY_REPORT_PATH}")

    append_week_to_history(draw_s, over_s, fbd, fbo, kelly)

    log(f"Summary → Draw: {len(draw_s)}, Over: {len(over_s)}, Kelly: {len(kelly)}, FB Draw Cols: {fbd.get('columns', 0)}, FB Over Cols: {fbo.get('columns', 0)}")


if __name__ == "__main__":
    main()
