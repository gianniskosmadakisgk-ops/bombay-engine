import os
import requests
import json
import math

# === CONFIG ===
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
BOOKMAKER = "bet365"
HEADERS = {"x-apisports-key": FOOTBALL_API_KEY}

REPORT_PATH = "logs/thursday_report_v1.json"
OUTPUT_PATH = "logs/friday_shortlist_v1.json"

DEBUG = True  # 👈 Ενεργό Debug Mode

# === WALLET SETTINGS ===
WALLETS = {
    "Draw Engine": 400,
    "Over Engine": 300,
    "FunBet Draw": 100,
    "FunBet Over": 100,
    "Fraction Kelly": 300
}

# === FILTER PARAMETERS ===
DRAW_THRESHOLD = 7.5
OVER_THRESHOLD = 7.5
VALUE_THRESHOLD = 0.10
KELLY_THRESHOLD = 0.15
KELLY_FRACTION = 0.40
BANKROLL_KELLY = WALLETS["Fraction Kelly"]

# === FUNCTION: Fetch Real Odds ===
def get_real_odds():
    print("📡 Fetching real bookmaker odds from The Odds API...")
    try:
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "eu",
            "markets": "h2h,totals",
            "oddsFormat": "decimal",
            "bookmakers": BOOKMAKER
        }
        res = requests.get(ODDS_URL, params=params, timeout=20)
        data = res.json()
        odds_map = {}

        for game in data:
            home = game["home_team"]
            away = game["away_team"]
            key = f"{home} - {away}"

            bookmakers = game.get("bookmakers", [])
            if not bookmakers:
                continue

            markets = bookmakers[0].get("markets", [])
            h2h, totals = None, None

            for m in markets:
                if m["key"] == "h2h":
                    outcomes = {o["name"]: o["price"] for o in m["outcomes"]}
                    h2h = outcomes
                elif m["key"] == "totals":
                    if m["outcomes"]:
                        totals = m["outcomes"][0]["price"]

            odds_map[key] = {
                "1": h2h.get(home) if h2h else None,
                "X": h2h.get("Draw") if h2h else None,
                "2": h2h.get(away) if h2h else None,
                "Over": totals
            }

        print(f"✅ Real odds fetched for {len(odds_map)} matches.")
        return odds_map

    except Exception as e:
        print(f"⚠️ Error fetching odds: {e}")
        return {}

# === HELPER FUNCTIONS ===
def calc_diff(fair, offered):
    try:
        return round((offered - fair) / fair, 3)
    except:
        return 0.0

def kelly_stake(fair, offered, bankroll=BANKROLL_KELLY, fraction=KELLY_FRACTION):
    try:
        p = 1 / fair
        b = offered - 1
        q = 1 - p
        kelly = (p * b - q) / b
        kelly = max(kelly, 0)
        stake = bankroll * fraction * kelly
        return round(stake, 2), round(kelly, 4)
    except:
        return 0.0, 0.0

def classify_score(score):
    if score >= 8.0:
        return "A"
    elif score >= 7.5:
        return "B"
    else:
        return "C"

def boost_if_value(score, fair, offered, threshold=VALUE_THRESHOLD):
    diff = calc_diff(fair, offered)
    if diff > threshold:
        return round(score + 0.5, 2)
    return score

# === LOAD THURSDAY DATA ===
if not os.path.exists(REPORT_PATH):
    raise FileNotFoundError(f"❌ Missing Thursday report at {REPORT_PATH}")

with open(REPORT_PATH, "r", encoding="utf-8") as f:
    thursday_data = json.load(f)

fixtures = thursday_data.get("data_sample", [])
print(f"📘 Thursday fixtures loaded: {len(fixtures)} matches")

# === LOAD REAL ODDS ===
real_odds = get_real_odds()

# === INIT OUTPUT LISTS ===
draw_picks, over_picks, funbet_draw, funbet_over, fraction_kelly = [], [], [], [], []

# === PROCESS FIXTURES ===
for f in fixtures:
    match = f["match"]
    league = f.get("league", "")
    fair_x = f.get("fair_x", 3.0)
    fair_over = f.get("fair_over", 1.9)

    real = real_odds.get(match, {})
    offered_x = real.get("X", fair_x)
    offered_over = real.get("Over", fair_over)

    score_draw = f.get("score_draw", 5)
    score_over = f.get("score_over", 5)

    if DEBUG:
        print(f"\n🎯 {match} — {league}")
        print(f"   Fair X: {fair_x} | Offered X: {offered_x}")
        print(f"   Fair Over: {fair_over} | Offered Over: {offered_over}")
        print(f"   Score Draw: {score_draw} | Score Over: {score_over}")

    # --- DRAW ENGINE ---
    boosted_draw = boost_if_value(score_draw, fair_x, offered_x)
    if boosted_draw >= DRAW_THRESHOLD:
        draw_picks.append({
            "match": match,
            "league": league,
            "fair": fair_x,
            "offered": offered_x,
            "diff%": round(calc_diff(fair_x, offered_x) * 100, 1),
            "score": boosted_draw,
            "category": classify_score(boosted_draw),
            "stake (€)": 20.0 if boosted_draw >= 8.0 else 15.0
        })

    # --- OVER ENGINE ---
    boosted_over = boost_if_value(score_over, fair_over, offered_over)
    if boosted_over >= OVER_THRESHOLD:
        over_picks.append({
            "match": match,
            "league": league,
            "fair": fair_over,
            "offered": offered_over,
            "diff%": round(calc_diff(fair_over, offered_over) * 100, 1),
            "score": boosted_over,
            "category": classify_score(boosted_over),
            "stake (€)": 20.0 if boosted_over >= 8.0 else 15.0
        })

# === FUNBET SYSTEMS ===
funbet_draw = sorted(draw_picks, key=lambda x: x["score"], reverse=True)[:5]
funbet_over = sorted(over_picks, key=lambda x: x["score"], reverse=True)[:5]

# === FRACTION KELLY ===
value_candidates = []
for f in fixtures:
    for market in ["home", "away", "draw", "over"]:
        fair = f.get(f"fair_{market}", None)
        if not fair:
            continue
        offered = (real_odds.get(f["match"], {}).get(market.title(), fair))
        diff = calc_diff(fair, offered)
        if diff >= KELLY_THRESHOLD:
            stake, kelly_f = kelly_stake(fair, offered)
            value_candidates.append({
                "match": f["match"],
                "market": market,
                "fair": fair,
                "offered": offered,
                "diff%": round(diff * 100, 1),
                "kelly_f": kelly_f,
                "stake (€)": stake
            })

fraction_kelly = sorted(value_candidates, key=lambda x: x["diff%"], reverse=True)[:10]

# === FINAL OUTPUT ===
output = {
    "summary": {
        "draw_count": len(draw_picks),
        "over_count": len(over_picks),
        "funbet_draw": len(funbet_draw),
        "funbet_over": len(funbet_over),
        "fraction_kelly": len(fraction_kelly)
    },
    "wallets": WALLETS,
    "draw_picks": draw_picks,
    "over_picks": over_picks,
    "funbet_draw": funbet_draw,
    "funbet_over": funbet_over,
    "fraction_kelly": fraction_kelly
}

# === SAVE OUTPUT ===
os.makedirs("logs", exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("\n✅ Friday Shortlist complete — saved to logs/friday_shortlist_v1.json")
print(f"📊 Draw: {len(draw_picks)} | Over: {len(over_picks)} | Kelly: {len(fraction_kelly)}")
