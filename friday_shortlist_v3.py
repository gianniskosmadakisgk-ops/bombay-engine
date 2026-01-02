import os
import json
from datetime import datetime
import itertools
import re
import requests

THURSDAY_REPORT_PATH = "logs/thursday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v3.json"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

os.makedirs("logs", exist_ok=True)

# ---------------------- BANKROLLS ----------------------
DRAW_WALLET = 300
OVER_WALLET = 300
FANBET_DRAW_WALLET = 100
FANBET_OVER_WALLET = 100
KELLY_WALLET = 300.0

# ---------------------- LIMITS -------------------------
MAX_DRAW_PICKS = 10
MAX_OVER_PICKS = 10

FUNBET_DRAW_STAKE_PER_COL = 2.0
FUNBET_OVER_STAKE_PER_COL = 4.0

# ---------------------- THRESHOLDS ---------------------
DRAW_MIN_SCORE = 7.5
DRAW_MIN_ODDS = 2.70

OVER_MIN_SCORE = 7.5
OVER_MIN_FAIR = 1.70

OVER_AUTO_SCORE = 9.0
OVER_NEG_EDGE_LIMIT = -0.10

KELLY_VALUE_THRESHOLD = 0.15
KELLY_FRACTION = 0.40
KELLY_MIN_PROB = 0.25
KELLY_MAX_EXPOSURE_PCT = 0.35

# preferred leagues
DRAW_LEAGUES = {
    "Ligue 1", "Serie A", "La Liga", "Championship", "Serie B",
    "Ligue 2", "Liga Portugal 2", "Swiss Super League",
}

OVER_LEAGUES = {
    "Bundesliga", "Eredivisie", "Jupiler Pro League", "Superliga",
    "Allsvenskan", "Eliteserien", "Swiss Super League", "Liga Portugal 1",
}

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
}

def log(msg: str):
    print(msg, flush=True)

def load_thursday_fixtures():
    if not os.path.exists(THURSDAY_REPORT_PATH):
        raise FileNotFoundError(f"Thursday report not found: {THURSDAY_REPORT_PATH}")
    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    fixtures = data.get("fixtures", [])
    log(f"Loaded {len(fixtures)} fixtures from Thursday v3.")
    return fixtures

def api_get_odds(sport_key: str):
    if not ODDS_API_KEY:
        log("⚠️ ODDS_API_KEY not set → returning empty.")
        return []

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
    }

    url = f"{ODDS_BASE_URL}/{sport_key}/odds"
    try:
        res = requests.get(url, params=params, timeout=25)
    except Exception as e:
        log(f"⚠️ Request error for odds {sport_key}: {e}")
        return []

    if res.status_code != 200:
        log(f"⚠️ Status {res.status_code}: {res.text[:200]}")
        return []

    try:
        return res.json()
    except Exception:
        return []

def normalize_team(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\b(fc|cf|afc|cfc|ac|sc|bk)\b", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def build_odds_index(fixtures):
    leagues = sorted({f.get("league") for f in fixtures if f.get("league") in LEAGUE_TO_SPORT})
    log(f"Odds supported leagues: {leagues}")

    odds_index = {}
    for lg in leagues:
        sport_key = LEAGUE_TO_SPORT[lg]
        events = api_get_odds(sport_key)
        log(f"Fetched {len(events)} events for {lg}")

        for ev in events:
            home_raw = ev.get("home_team", "")
            away_raw = ev.get("away_team", "")
            home = normalize_team(home_raw)
            away = normalize_team(away_raw)
            if not home or not away:
                continue

            best_home = best_draw = best_away = best_over25 = None

            for bm in ev.get("bookmakers", []) or []:
                for m in bm.get("markets", []) or []:
                    k = m.get("key")

                    if k == "h2h":
                        for o in m.get("outcomes", []) or []:
                            nm = normalize_team(o.get("name", ""))
                            price = float(o.get("price", 0) or 0)
                            if price <= 0:
                                continue
                            if nm == home:
                                best_home = price if best_home is None else max(best_home, price)
                            elif nm == away:
                                best_away = price if best_away is None else max(best_away, price)
                            elif nm == "draw":
                                best_draw = price if best_draw is None else max(best_draw, price)

                    elif k == "totals":
                        # TheOddsAPI v4: outcomes have name Over/Under and point 2.5
                        for o in m.get("outcomes", []) or []:
                            name = (o.get("name") or "").strip().lower()
                            point = o.get("point", None)
                            price = float(o.get("price", 0) or 0)
                            if price <= 0:
                                continue
                            if name == "over" and point == 2.5:
                                best_over25 = price if best_over25 is None else max(best_over25, price)

            odds_index[(home, away)] = {
                "home": best_home,
                "draw": best_draw,
                "away": best_away,
                "over_2_5": best_over25,
                "source_home": home_raw,
                "source_away": away_raw,
            }

    log(f"Odds index size: {len(odds_index)}")
    return odds_index

def clamp_score(x: float) -> float:
    if x < 1:
        return 1.0
    if x > 10:
        return 10.0
    return x

def flat_stake(score):
    if score >= 8.5:
        return 20
    elif score >= 7.5:
        return 15
    return 0

def get_event_odds(odds_index, home_name, away_name):
    home_norm = normalize_team(home_name)
    away_norm = normalize_team(away_name)

    direct = odds_index.get((home_norm, away_norm))
    if direct:
        return direct, False  # swap=False

    rev = odds_index.get((away_norm, home_norm))
    if rev:
        # swap needed: rev.home actually belongs to away team
        swapped = {
            "home": rev.get("away"),
            "draw": rev.get("draw"),
            "away": rev.get("home"),
            "over_2_5": rev.get("over_2_5"),
            "source_home": rev.get("source_away"),
            "source_away": rev.get("source_home"),
        }
        return swapped, True

    return {}, False

def generate_picks(fixtures, odds_index):
    draw_singles = []
    over_singles = []
    kelly_candidates = []

    for f in fixtures:
        league = f.get("league")
        home = f.get("home")
        away = f.get("away")
        fixture_id = f.get("fixture_id")
        match_label = f"{home} - {away}"

        fair_1, fair_x, fair_2 = f.get("fair_1"), f.get("fair_x"), f.get("fair_2")
        fair_over = f.get("fair_over_2_5")

        draw_prob = float(f.get("draw_prob") or 0)
        over_prob = float(f.get("over_2_5_prob") or 0)

        score_draw = clamp_score(draw_prob * 10)
        score_over = clamp_score(over_prob * 10)

        odds, swapped = get_event_odds(odds_index, home, away)
        odds_home = odds.get("home")
        odds_x = odds.get("draw")
        odds_away = odds.get("away")
        odds_over = odds.get("over_2_5")

        # ---------------- DRAW singles ----------------
        if league in DRAW_LEAGUES and fair_x and score_draw >= DRAW_MIN_SCORE:
            offered = odds_x if odds_x else float(fair_x)
            if offered >= DRAW_MIN_ODDS:
                stake = flat_stake(score_draw)
                if stake:
                    diff = (offered - float(fair_x)) / float(fair_x)
                    draw_singles.append({
                        "fixture_id": fixture_id,
                        "match": match_label,
                        "league": league,
                        "odds": round(offered, 2),
                        "fair": round(float(fair_x), 2),
                        "diff": f"{diff:+.0%}",
                        "score": round(score_draw, 2),
                        "stake": stake,
                        "odds_swapped": swapped,
                    })

        # ---------------- OVER singles ----------------
        if league in OVER_LEAGUES and fair_over and score_over >= OVER_MIN_SCORE:
            offered = odds_over if odds_over else float(fair_over)
            value_ok = offered >= float(fair_over)
            monster = (
                score_over >= OVER_AUTO_SCORE
                and offered >= OVER_MIN_FAIR
                and (offered - float(fair_over)) / float(fair_over) >= OVER_NEG_EDGE_LIMIT
            )

            if offered >= OVER_MIN_FAIR and (value_ok or monster):
                stake = flat_stake(score_over)
                if stake:
                    diff = (offered - float(fair_over)) / float(fair_over)
                    over_singles.append({
                        "fixture_id": fixture_id,
                        "match": match_label,
                        "league": league,
                        "odds": round(offered, 2),
                        "fair": round(float(fair_over), 2),
                        "diff": f"{diff:+.0%}",
                        "score": round(score_over, 2),
                        "stake": stake,
                        "odds_swapped": swapped,
                    })

        # ---------------- KELLY candidates ----------------
        def add_kelly(label, fair, offered):
            if fair is None or offered is None:
                return
            fair = float(fair)
            offered = float(offered)
            if fair <= 1.0001 or offered <= 1.0001:
                return

            p = 1.0 / fair
            if p < KELLY_MIN_PROB:
                return

            edge = (offered - fair) / fair
            if edge < KELLY_VALUE_THRESHOLD:
                return

            b = offered - 1
            q = 1 - p
            fk = (b * p - q) / b
            if fk <= 0:
                return

            stake_raw = fk * KELLY_FRACTION * KELLY_WALLET
            if stake_raw > 0:
                kelly_candidates.append({
                    "fixture_id": fixture_id,
                    "match": match_label,
                    "league": league,
                    "market": label,
                    "fair": round(fair, 2),
                    "offered": round(offered, 2),
                    "edge": f"{edge:+.0%}",
                    "stake_raw": stake_raw,
                })

        if odds_home: add_kelly("Home", fair_1, odds_home)
        if odds_x: add_kelly("Draw", fair_x, odds_x)
        if odds_away: add_kelly("Away", fair_2, odds_away)
        if odds_over: add_kelly("Over 2.5", fair_over, odds_over)

    draw_singles = sorted(draw_singles, key=lambda x: x["score"], reverse=True)[:MAX_DRAW_PICKS]
    over_singles = sorted(over_singles, key=lambda x: x["score"], reverse=True)[:MAX_OVER_PICKS]

    kelly_candidates = sorted(kelly_candidates, key=lambda x: x["stake_raw"], reverse=True)[:10]
    total_raw = sum(k["stake_raw"] for k in kelly_candidates)
    max_exposure = KELLY_WALLET * KELLY_MAX_EXPOSURE_PCT
    scale = (max_exposure / total_raw) if (total_raw and total_raw > max_exposure) else 1.0

    kelly_final = []
    for k in kelly_candidates:
        stake = round(k["stake_raw"] * scale, 2)
        kelly_final.append({
            "fixture_id": k["fixture_id"],
            "match": k["match"],
            "league": k["league"],
            "market": k["market"],
            "fair": k["fair"],
            "offered": k["offered"],
            "edge": k["edge"],
            "stake (€)": stake,
        })

    return draw_singles, over_singles, kelly_final

def build_funbet_draw(draw_singles):
    sorted_draws = sorted(draw_singles, key=lambda x: x["score"], reverse=True)
    picks = sorted_draws[:6]
    n = len(picks)

    if n >= 6:
        sizes = [4, 5, 6]
        system = "4-5-6"
    elif n == 5:
        sizes = [3, 4, 5]
        system = "3-4-5"
    else:
        return {"picks": picks, "system": None, "columns": 0, "stake_per_column": 0, "total_stake": 0}

    cols = sum(1 for r in sizes for _ in itertools.combinations(range(n), r))
    total = cols * FUNBET_DRAW_STAKE_PER_COL

    return {"picks": picks, "system": system, "columns": cols, "stake_per_column": FUNBET_DRAW_STAKE_PER_COL, "total_stake": total}

def build_funbet_over(over_singles):
    sorted_ = sorted(over_singles, key=lambda x: x["score"], reverse=True)
    picks = sorted_[:6]
    n = len(picks)

    if n < 3:
        return {"picks": picks, "system": None, "columns": 0, "stake_per_column": 0, "total_stake": 0}

    cols = sum(1 for _ in itertools.combinations(range(n), 2))
    total = cols * FUNBET_OVER_STAKE_PER_COL

    return {"picks": picks, "system": f"2-from-{n}", "columns": cols, "stake_per_column": FUNBET_OVER_STAKE_PER_COL, "total_stake": total}

def bankroll_summary(draw_singles, over_singles, fun_draw, fun_over, kelly):
    draw_spent = sum(x["stake"] for x in draw_singles)
    over_spent = sum(x["stake"] for x in over_singles)
    fun_draw_spent = float(fun_draw.get("total_stake") or 0)
    fun_over_spent = float(fun_over.get("total_stake") or 0)
    kelly_spent = sum(float(k["stake (€)"]) for k in kelly)

    return [
        {"Wallet": "Draw Singles", "Before": f"{DRAW_WALLET}€", "After": f"{DRAW_WALLET-draw_spent:.2f}€", "Open Bets": f"{draw_spent:.2f}€"},
        {"Wallet": "Over Singles", "Before": f"{OVER_WALLET}€", "After": f"{OVER_WALLET-over_spent:.2f}€", "Open Bets": f"{over_spent:.2f}€"},
        {"Wallet": "FanBet Draw", "Before": f"{FANBET_DRAW_WALLET}€", "After": f"{FANBET_DRAW_WALLET-fun_draw_spent:.2f}€", "Open Bets": f"{fun_draw_spent:.2f}€"},
        {"Wallet": "FanBet Over", "Before": f"{FANBET_OVER_WALLET}€", "After": f"{FANBET_OVER_WALLET-fun_over_spent:.2f}€", "Open Bets": f"{fun_over_spent:.2f}€"},
        {"Wallet": "Kelly", "Before": f"{KELLY_WALLET:.0f}€", "After": f"{KELLY_WALLET-kelly_spent:.2f}€", "Open Bets": f"{kelly_spent:.2f}€"},
    ]

def main():
    log("🎯 Running Friday Shortlist v3...")

    fixtures = load_thursday_fixtures()
    odds_index = build_odds_index(fixtures)

    draw_singles, over_singles, kelly = generate_picks(fixtures, odds_index)
    fun_draw = build_funbet_draw(draw_singles)
    fun_over = build_funbet_over(over_singles)
    banks = bankroll_summary(draw_singles, over_singles, fun_draw, fun_over, kelly)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "meta": {
            "fixtures_total": len(fixtures),
            "draw_singles": len(draw_singles),
            "over_singles": len(over_singles),
            "kelly_picks": len(kelly),
            "funbet_draw_cols": fun_draw.get("columns", 0),
            "funbet_over_cols": fun_over.get("columns", 0),
        },
        "draw_singles": draw_singles,
        "over_singles": over_singles,
        "funbet_draw": fun_draw,
        "funbet_over": fun_over,
        "kelly": kelly,
        "bankroll_status": banks,
    }

    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log(f"✅ Saved Friday shortlist → {FRIDAY_REPORT_PATH}")

if __name__ == "__main__":
    main()
