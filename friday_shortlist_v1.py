import os
import json
from datetime import datetime
from itertools import combinations

import requests

# ============================================
#  FRIDAY SHORTLIST v1  (Giannis Edition)
#
#  Input : logs/thursday_report_v1.json
#          (fair_1, fair_x, fair_2, fair_over, score_draw, score_over)
#  Extra : πραγματικές αποδόσεις από TheOddsAPI
#
#  Output: logs/friday_shortlist_v1.json
#          - draw_engine (singles)
#          - over_engine (singles)
#          - funbet_draw (συστήματα)
#          - funbet_over (συστήματα)
#          - fraction_kelly (top 10 value picks)
#          - bankroll_status (5 wallets)
# ============================================

THURSDAY_REPORT = "logs/thursday_report_v1.json"
FRIDAY_REPORT = "logs/friday_shortlist_v1.json"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

# --- Wallets ---
DRAW_WALLET = 400.0
OVER_WALLET = 300.0
FANBET_DRAW_WALLET = 100.0
FANBET_OVER_WALLET = 100.0
KELLY_WALLET = 300.0

# --- Rules ---
MIN_SCORE = 7.5
HIGH_CONF_SCORE = 8.5

DRAW_FLAT_LOW = 15.0
DRAW_FLAT_HIGH = 20.0

OVER_FLAT_LOW = 15.0
OVER_FLAT_HIGH = 20.0

FUNBET_DRAW_STAKE_PER_COL = 3.0
FUNBET_OVER_STAKE_PER_COL = 4.0

KELLY_MIN_DIFF = 0.15     # +15% value
KELLY_FRACTION = 0.40     # παίζουμε 40% του full Kelly


def log(msg: str):
    print(msg, flush=True)


# --------------------------------------------
# Load Thursday report
# --------------------------------------------
def load_thursday_fixtures():
    if not os.path.exists(THURSDAY_REPORT):
        raise FileNotFoundError(f"Thursday report not found: {THURSDAY_REPORT}")

    with open(THURSDAY_REPORT, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixtures = data.get("fixtures", [])
    log(f"Loaded {len(fixtures)} fixtures from Thursday report.")
    return fixtures


# --------------------------------------------
# Fetch odds from TheOddsAPI
# --------------------------------------------
def fetch_odds_map():
    """
    Επιστρέφει:
      'Home - Away' -> {'1': .., 'X': .., '2': .., 'O2.5': ..}
    """
    if not ODDS_API_KEY:
        log("⚠️ ODDS_API_KEY not set – no real odds will be used.")
        return {}

    url = f"{ODDS_BASE_URL}/soccer/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals",
    }

    try:
        res = requests.get(url, params=params, timeout=20)
    except Exception as e:
        log(f"⚠️ Error calling TheOddsAPI: {e}")
        return {}

    if res.status_code != 200:
        log(f"⚠️ TheOddsAPI returned {res.status_code}: {res.text[:200]}")
        return {}

    odds_map = {}
    try:
        for ev in res.json():
            home = ev.get("home_team")
            away = ev.get("away_team")
            if not home or not away:
                continue

            label = f"{home} - {away}"
            market_data = {"1": None, "X": None, "2": None, "O2.5": None}

            for book in ev.get("bookmakers", []):
                for m in book.get("markets", []):
                    if m.get("key") == "h2h":
                        for o in m.get("outcomes", []):
                            name = o.get("name")
                            price = o.get("price")
                            if name == home:
                                market_data["1"] = float(price)
                            elif name == "Draw":
                                market_data["X"] = float(price)
                            elif name == away:
                                market_data["2"] = float(price)
                    elif m.get("key") == "totals":
                        for o in m.get("outcomes", []):
                            if o.get("name") == "Over 2.5":
                                market_data["O2.5"] = float(o.get("price"))

            odds_map[label] = market_data

    except Exception as e:
        log(f"⚠️ Error parsing TheOddsAPI response: {e}")
        return {}

    log(f"Fetched odds for {len(odds_map)} matches from TheOddsAPI.")
    return odds_map


def attach_odds(fixtures, odds_map):
    enriched = []
    for f in fixtures:
        label = f.get("match")
        o = odds_map.get(label, {})
        f["odds_1"] = o.get("1")
        f["odds_x"] = o.get("X")
        f["odds_2"] = o.get("2")
        f["odds_over"] = o.get("O2.5")
        enriched.append(f)
    return enriched


# --------------------------------------------
# Singles – Draw & Over
# --------------------------------------------
def build_draw_singles(fixtures):
    picks = []
    for f in fixtures:
        score = float(f.get("score_draw", 0))
        fair_x = f.get("fair_x")
        odds_x = f.get("odds_x")

        if score < MIN_SCORE:
            continue
        if not fair_x or not odds_x:
            continue

        diff = (odds_x - fair_x) / fair_x
        stake = DRAW_FLAT_HIGH if score > HIGH_CONF_SCORE else DRAW_FLAT_LOW

        picks.append({
            "match": f["match"],
            "league": f["league"],
            "odds": round(odds_x, 2),
            "fair": round(fair_x, 2),
            "diff": f"{diff:+.0%}",
            "score": round(score, 2),
            "stake": f"{stake:.0f}€",
            "stake_value": stake,
            "wallet": "Draw",
        })

    # Αν είναι πάνω από 10, κρατάμε τις 10 καλύτερες (score + value)
    picks.sort(key=lambda x: (x["score"], float(x["diff"].replace('%', ''))), reverse=True)
    return picks[:10]


def build_over_singles(fixtures):
    picks = []
    for f in fixtures:
        score = float(f.get("score_over", 0))
        fair_over = f.get("fair_over")
        odds_over = f.get("odds_over")

        if score < MIN_SCORE:
            continue
        if not fair_over or not odds_over:
            continue

        diff = (odds_over - fair_over) / fair_over
        stake = OVER_FLAT_HIGH if score > HIGH_CONF_SCORE else OVER_FLAT_LOW

        picks.append({
            "match": f["match"],
            "league": f["league"],
            "odds": round(odds_over, 2),
            "fair": round(fair_over, 2),
            "diff": f"{diff:+.0%}",
            "score": round(score, 2),
            "stake": f"{stake:.0f}€",
            "stake_value": stake,
            "wallet": "Over",
        })

    picks.sort(key=lambda x: (x["score"], float(x["diff"].replace('%', ''))), reverse=True)
    return picks[:10]


# --------------------------------------------
# FunBet Draw – 3-4-5 ή 4-5-6
# --------------------------------------------
def build_funbet_draw(draw_picks):
    """
    Αν έχουμε:
      - 6+ picks → top 6, σύστημα 4-5-6
      - 5    picks → top 5, σύστημα 3-4-5
      - <5   picks → δεν παίζουμε FunBet Draw
    """
    if len(draw_picks) < 5:
        return None

    if len(draw_picks) >= 6:
        used = draw_picks[:6]
        system = "4-5-6"
        sizes = [4, 5, 6]
    else:
        used = draw_picks[:5]
        system = "3-4-5"
        sizes = [3, 4, 5]

    matches = [p["match"] for p in used]

    columns = []
    for n in sizes:
        for combo in combinations(matches, n):
            columns.append(list(combo))

    total_stake = len(columns) * FUNBET_DRAW_STAKE_PER_COL

    # base_picks: για να ξέρουμε ποια ματς μπαίνουν στο σύστημα
    base_picks = [
        {
            "match": p["match"],
            "league": p["league"],
            "odds": p["odds"],
            "fair": p["fair"],
            "diff": p["diff"],
            "score": p["score"],
        }
        for p in used
    ]

    return {
        "system": system,
        "picks_count": len(used),
        "stake_per_column": FUNBET_DRAW_STAKE_PER_COL,
        "columns_count": len(columns),
        "total_stake": total_stake,
        "wallet": "FanBet Draw",
        "base_picks": base_picks,
        "columns": columns,
    }


# --------------------------------------------
# FunBet Over – 2-from-X
# --------------------------------------------
def build_funbet_over(over_picks):
    """
    Χρησιμοποιεί 4–6 καλύτερα over.
    Αν <4 picks → δεν παίζουμε FunBet Over.
    Σύστημα: όλες οι δυάδες (2-from-X).
    """
    if len(over_picks) < 4:
        return None

    used_n = min(6, len(over_picks))
    used = over_picks[:used_n]
    matches = [p["match"] for p in used]

    columns = []
    for combo in combinations(matches, 2):
        columns.append(list(combo))

    total_stake = len(columns) * FUNBET_OVER_STAKE_PER_COL

    base_picks = [
        {
            "match": p["match"],
            "league": p["league"],
            "odds": p["odds"],
            "fair": p["fair"],
            "diff": p["diff"],
            "score": p["score"],
        }
        for p in used
    ]

    return {
        "system": "2-from-X",
        "picks_count": used_n,
        "stake_per_column": FUNBET_OVER_STAKE_PER_COL,
        "columns_count": len(columns),
        "total_stake": total_stake,
        "wallet": "FunBet Over",
        "base_picks": base_picks,
        "columns": columns,
    }


# --------------------------------------------
# Kelly Engine – top 10 value picks
# --------------------------------------------
def kelly_stake(bankroll, fair, offered):
    if fair <= 0 or offered <= 1:
        return 0.0

    p = 1.0 / fair
    b = offered - 1.0
    q = 1.0 - p
    f = (b * p - q) / b
    if f <= 0:
        return 0.0

    stake_fraction = f * KELLY_FRACTION
    return round(bankroll * stake_fraction, 2)


def build_kelly_picks(fixtures):
    picks = []

    for f in fixtures:
        match = f["match"]

        markets = [
            ("home", "fair_1", "odds_1"),
            ("draw", "fair_x", "odds_x"),
            ("away", "fair_2", "odds_2"),
            ("over", "fair_over", "odds_over"),
        ]

        for label, fair_key, odds_key in markets:
            fair = f.get(fair_key)
            offered = f.get(odds_key)
            if not fair or not offered:
                continue

            diff = (offered - fair) / fair
            if diff < KELLY_MIN_DIFF:
                continue

            stake = kelly_stake(KELLY_WALLET, fair, offered)
            if stake <= 0:
                continue

            picks.append({
                "match": match,
                "market": label,
                "fair": round(fair, 2),
                "offered": round(offered, 2),
                "diff": f"{diff:+.0%}",
                "kelly%": f"{int(KELLY_FRACTION * 100)}%",
                "stake (€)": stake,
            })

    picks.sort(key=lambda x: float(x["diff"].replace('%', '')), reverse=True)
    return picks[:10]


# --------------------------------------------
# Bankroll summary
# --------------------------------------------
def bankroll_summary(draw_singles, over_singles, funbet_draw, funbet_over, kelly_picks):
    draw_spent = sum(p["stake_value"] for p in draw_singles)
    over_spent = sum(p["stake_value"] for p in over_singles)
    kelly_spent = sum(p["stake (€)"] for p in kelly_picks)

    funbet_draw_spent = funbet_draw["total_stake"] if funbet_draw else 0.0
    funbet_over_spent = funbet_over["total_stake"] if funbet_over else 0.0

    summary = [
        {
            "Wallet": "Draw Engine",
            "Before": f"{DRAW_WALLET:.0f}€",
            "After": f"{DRAW_WALLET - draw_spent:.2f}€",
            "Open Bets": f"{draw_spent:.2f}€",
        },
        {
            "Wallet": "Over Engine",
            "Before": f"{OVER_WALLET:.0f}€",
            "After": f"{OVER_WALLET - over_spent:.2f}€",
            "Open Bets": f"{over_spent:.2f}€",
        },
        {
            "Wallet": "FanBet Draw",
            "Before": f"{FANBET_DRAW_WALLET:.0f}€",
            "After": f"{FANBET_DRAW_WALLET - funbet_draw_spent:.2f}€",
            "Open Bets": f"{funbet_draw_spent:.2f}€",
        },
        {
            "Wallet": "FunBet Over",
            "Before": f"{FANBET_OVER_WALLET:.0f}€",
            "After": f"{FANBET_OVER_WALLET - funbet_over_spent:.2f}€",
            "Open Bets": f"{funbet_over_spent:.2f}€",
        },
        {
            "Wallet": "Fraction Kelly",
            "Before": f"{KELLY_WALLET:.0f}€",
            "After": f"{KELLY_WALLET - kelly_spent:.2f}€",
            "Open Bets": f"{kelly_spent:.2f}€",
        },
    ]

    return summary


# --------------------------------------------
# MAIN
# --------------------------------------------
if __name__ == "__main__":
    print("🎯 Running Friday Shortlist (v1)...", flush=True)

    # 1) Thursday fixtures
    fixtures = load_thursday_fixtures()

    # 2) Real odds
    odds_map = fetch_odds_map()
    fixtures = attach_odds(fixtures, odds_map)

    # 3) Singles
    draw_singles = build_draw_singles(fixtures)
    over_singles = build_over_singles(fixtures)

    # 4) FunBets
    funbet_draw = build_funbet_draw(draw_singles)
    funbet_over = build_funbet_over(over_singles)

    # 5) Kelly Engine
    kelly_picks = build_kelly_picks(fixtures)

    # 6) Bankroll
    banks = bankroll_summary(draw_singles, over_singles, funbet_draw, funbet_over, kelly_picks)

    # 7) Report
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "draw_engine": draw_singles,
        "over_engine": over_singles,
        "funbet_draw": funbet_draw or {},
        "funbet_over": funbet_over or {},
        "fraction_kelly": {"picks": kelly_picks},
        "bankroll_status": banks,
    }

    os.makedirs("logs", exist_ok=True)
    with open(FRIDAY_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ Friday shortlist report saved: {FRIDAY_REPORT}", flush=True)
    print(
        f"🎯 Draw singles: {len(draw_singles)}, "
        f"Over singles: {len(over_singles)}, "
        f"Kelly picks: {len(kelly_picks)}, "
        f"FunBet Draw cols: {funbet_draw['columns_count'] if funbet_draw else 0}, "
        f"FunBet Over cols: {funbet_over['columns_count'] if funbet_over else 0}",
        flush=True,
    )
