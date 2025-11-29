import os
import json
import itertools
from datetime import datetime
from pathlib import Path

import requests

# ======================================================
#  FRIDAY SHORTLIST v1  (Giannis Edition)
#
#  - Διαβάζει το Thursday report (fair odds + scores)
#  - Παίρνει ΠΡΑΓΜΑΤΙΚΕΣ αποδόσεις από TheOddsAPI
#  - Φτιάχνει:
#       * Draw singles
#       * Over singles
#       * FunBet Draw (3-4-5 σύστημα)
#       * FunBet Over (2-from-X, δυάδες)
#       * Kelly top 10 value picks
#       * Bankroll summary (5 πορτοφόλια)
#  - Σώζει: logs/friday_shortlist_v1.json
# ======================================================

THURSDAY_REPORT_PATH = "logs/thursday_report_v1.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v1.json"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

# ----- Πορτοφόλια -----
DRAW_WALLET = 400
OVER_WALLET = 300
FANBET_DRAW_WALLET = 100
FANBET_OVER_WALLET = 100
KELLY_WALLET = 300

# ----- thresholds / stakes -----
DRAW_MIN_SCORE = 7.5
OVER_MIN_SCORE = 7.5
DRAW_MIN_ODDS = 2.70
OVER_MIN_ODDS = 1.70

HIGH_CONF_SCORE = 8.5         # 20€
MID_CONF_SCORE = 7.5          # 15€

FUNBET_DRAW_STAKE_PER_COL = 3
FUNBET_OVER_STAKE_PER_COL = 4

KELLY_DIFF_THRESHOLD = 0.15   # +15% value
KELLY_FRACTION = 0.30         # παίζουμε 30% του πλήρους Kelly

BOOKMAKER_PRIORITY = ["bet365", "pinnacle", "unibet", "williamhill"]


# ------------------------------------------------------
# helpers
# ------------------------------------------------------
def log(msg: str):
    print(msg, flush=True)


def load_thursday_fixtures():
    if not os.path.exists(THURSDAY_REPORT_PATH):
        raise FileNotFoundError(
            f"Thursday report not found: {THURSDAY_REPORT_PATH}"
        )
    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    fixtures = data.get("fixtures", [])
    log(f"Loaded {len(fixtures)} fixtures from the Thursday report.")
    return fixtures


def normalize_team_name(name: str) -> str:
    name = name.lower()
    junk = [" fc", " cf", " ac", " sc", " bc", " bk", " u19", " u21"]
    for j in junk:
        name = name.replace(j, " ")
    for ch in [".", "-", "_"]:
        name = name.replace(ch, " ")
    name = " ".join(name.split())
    return name


def match_key_from_label(match_label: str) -> str:
    # "Manchester City - Leeds"
    if " - " in match_label:
        home, away = match_label.split(" - ", 1)
    else:
        parts = match_label.split("-")
        home = parts[0]
        away = parts[1] if len(parts) > 1 else ""
    return f"{normalize_team_name(home)}|{normalize_team_name(away)}"


def match_key_from_teams(home: str, away: str) -> str:
    return f"{normalize_team_name(home)}|{normalize_team_name(away)}"


def fetch_odds_from_theoddsapi():
    if not ODDS_API_KEY:
        log("⚠️ ODDS_API_KEY is not set. Friday shortlist will have no real odds.")
        return {}

    url = f"{ODDS_BASE_URL}/soccer/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals",
    }
    try:
        res = requests.get(url, params=params, timeout=25)
    except Exception as e:
        log(f"⚠️ Error calling TheOddsAPI: {e}")
        return {}

    if res.status_code != 200:
        log(f"⚠️ TheOddsAPI returned status {res.status_code}: {res.text[:200]}")
        return {}

    data = res.json()
    log(f"Fetched {len(data)} events from TheOddsAPI.")

    index = {}
    for ev in data:
        home = ev.get("home_team", "")
        away = ev.get("away_team", "")
        key = match_key_from_teams(home, away)
        index[key] = ev

    return index


def extract_price(bookmakers, market_key, outcome_name, point_target=None):
    """
    Προσπαθεί πρώτα στους priority bookmakers, μετά μέσο όρο από όλους.
    """
    # 1) Priority search
    for bm in bookmakers:
        bm_key = str(bm.get("key", "")).lower()
        if bm_key not in BOOKMAKER_PRIORITY:
            continue
        for m in bm.get("markets", []):
            if m.get("key") != market_key:
                continue
            for o in m.get("outcomes", []):
                name = o.get("name", "")
                point = o.get("point")
                price = o.get("price")
                if market_key == "totals":
                    # Over 2.5
                    if "over" in name.lower() and (point_target is None or point == point_target):
                        return float(price)
                else:
                    if name.lower() == outcome_name.lower():
                        return float(price)

    # 2) Average from all bookmakers
    prices = []
    for bm in bookmakers:
        for m in bm.get("markets", []):
            if m.get("key") != market_key:
                continue
            for o in m.get("outcomes", []):
                name = o.get("name", "")
                point = o.get("point")
                price = o.get("price")
                if market_key == "totals":
                    if "over" in name.lower() and (point_target is None or point == point_target):
                        prices.append(float(price))
                else:
                    if name.lower() == outcome_name.lower():
                        prices.append(float(price))

    if prices:
        return round(sum(prices) / len(prices), 2)

    return None


def kelly_stake(bankroll, fair, offered):
    if not fair or not offered:
        return 0.0
    p = 1.0 / float(fair)
    b = float(offered) - 1.0
    q = 1.0 - p
    if b <= 0:
        return 0.0
    f = (b * p - q) / b
    if f <= 0:
        return 0.0
    stake = bankroll * f * KELLY_FRACTION
    return round(stake, 2)


# ------------------------------------------------------
# build shortlist
# ------------------------------------------------------
def build_shortlist(fixtures, odds_index):
    enriched = []
    matched = 0

    for f in fixtures:
        league = f.get("league")
        match_label = f.get("match")
        fair_1 = f.get("fair_1")
        fair_x = f.get("fair_x")
        fair_2 = f.get("fair_2")
        fair_over = f.get("fair_over")
        score_draw = float(f.get("score_draw", 0))
        score_over = float(f.get("score_over", 0))

        odds_1 = odds_x = odds_2 = odds_over = None

        key = match_key_from_label(match_label)
        ev = odds_index.get(key)
        if ev:
            matched += 1
            bookmakers = ev.get("bookmakers", [])
            # h2h
            odds_1 = extract_price(bookmakers, "h2h", "Home", None)
            odds_2 = extract_price(bookmakers, "h2h", "Away", None)
            odds_x = extract_price(bookmakers, "h2h", "Draw", None)
            # totals: Over 2.5
            odds_over = extract_price(bookmakers, "totals", "Over", 2.5)

        enriched.append(
            {
                "league": league,
                "match": match_label,
                "fair_1": fair_1,
                "fair_x": fair_x,
                "fair_2": fair_2,
                "fair_over": fair_over,
                "score_draw": score_draw,
                "score_over": score_over,
                "odds_1": odds_1,
                "odds_x": odds_x,
                "odds_2": odds_2,
                "odds_over": odds_over,
            }
        )

    log(f"Matched odds for {matched} / {len(fixtures)} fixtures.")
    return enriched


def generate_draw_singles(fixtures):
    candidates = []
    for f in fixtures:
        if f["odds_x"] is None:
            continue
        if f["score_draw"] < DRAW_MIN_SCORE:
            continue
        if f["odds_x"] < DRAW_MIN_ODDS:
            continue
        fair_x = f.get("fair_x") or 0
        if fair_x <= 0:
            continue
        diff = (f["odds_x"] - fair_x) / fair_x
        candidates.append((diff, f))

    # sort by value diff desc
    candidates.sort(key=lambda t: t[0], reverse=True)
    picks = []
    for diff, f in candidates[:10]:
        score = f["score_draw"]
        if score >= HIGH_CONF_SCORE:
            stake = 20
        else:
            stake = 15
        picks.append(
            {
                "match": f["match"],
                "league": f["league"],
                "odds": f["odds_x"],
                "fair": f["fair_x"],
                "score": round(score, 2),
                "value_diff": f"{diff:+.0%}",
                "stake": stake,
                "wallet": "Draw",
            }
        )

    return picks


def generate_over_singles(fixtures):
    candidates = []
    for f in fixtures:
        if f["odds_over"] is None:
            continue
        if f["score_over"] < OVER_MIN_SCORE:
            continue
        if f["odds_over"] < OVER_MIN_ODDS:
            continue
        fair_over = f.get("fair_over") or 0
        if fair_over <= 0:
            continue
        diff = (f["odds_over"] - fair_over) / fair_over
        candidates.append((diff, f))

    candidates.sort(key=lambda t: t[0], reverse=True)
    picks = []
    for diff, f in candidates[:10]:
        score = f["score_over"]
        if score >= HIGH_CONF_SCORE:
            stake = 20
        else:
            stake = 15
        picks.append(
            {
                "match": f["match"],
                "league": f["league"],
                "odds": f["odds_over"],
                "fair": f["fair_over"],
                "score": round(score, 2),
                "value_diff": f"{diff:+.0%}",
                "stake": stake,
                "wallet": "Over",
            }
        )

    return picks


def generate_funbet_draw(draw_singles):
    """
    Παίρνουμε τις 5 καλύτερες draws (αν υπάρχουν)
    και παίζουμε σύστημα 3-4-5 με 3€/στήλη.
    """
    top = draw_singles[:5]
    matches = [p["match"] for p in top]
    columns = []

    for n in [3, 4, 5]:
        if len(matches) >= n:
            for combo in itertools.combinations(matches, n):
                columns.append(list(combo))

    total_stake = len(columns) * FUNBET_DRAW_STAKE_PER_COL
    return {
        "picks_used": matches,
        "system": "3-4-5",
        "columns_count": len(columns),
        "stake_per_column": FUNBET_DRAW_STAKE_PER_COL,
        "total_stake": total_stake,
        "wallet": "FanBet Draw",
        "columns": columns,
    }


def generate_funbet_over(over_singles):
    """
    Παίρνουμε μέχρι 6 καλύτερα overs και παίζουμε
    όλες τις δυάδες (2-from-X) με 4€/στήλη.
    """
    top = over_singles[:6]
    matches = [p["match"] for p in top]
    columns = []

    if len(matches) >= 2:
        for combo in itertools.combinations(matches, 2):
            columns.append(list(combo))

    total_stake = len(columns) * FUNBET_OVER_STAKE_PER_COL
    return {
        "picks_used": matches,
        "system": "2-from-X (all doubles)",
        "columns_count": len(columns),
        "stake_per_column": FUNBET_OVER_STAKE_PER_COL,
        "total_stake": total_stake,
        "wallet": "FunBet Over",
        "columns": columns,
    }


def generate_kelly_picks(fixtures):
    picks = []

    for f in fixtures:
        markets = [
            ("Home", "fair_1", "odds_1", "1"),
            ("Draw", "fair_x", "odds_x", "X"),
            ("Away", "fair_2", "odds_2", "2"),
            ("Over2.5", "fair_over", "odds_over", "Over"),
        ]
        for label, fair_key, odds_key, market_tag in markets:
            fair = f.get(fair_key)
            offered = f.get(odds_key)
            if not fair or not offered:
                continue
            diff = (offered - fair) / fair
            if diff < KELLY_DIFF_THRESHOLD:
                continue
            stake = kelly_stake(KELLY_WALLET, fair, offered)
            if stake <= 0:
                continue
            picks.append(
                {
                    "match": f["match"],
                    "league": f["league"],
                    "market": market_tag,
                    "fair": fair,
                    "offered": offered,
                    "diff": f"{diff:+.0%}",
                    "kelly_fraction": f"{int(KELLY_FRACTION*100)}%",
                    "stake": stake,
                }
            )

    # Top 10 by diff
    picks.sort(key=lambda x: float(x["diff"].replace("%", "")), reverse=True)
    return picks[:10]


def bankroll_status(draw_singles, over_singles, funbet_draw, funbet_over, kelly_picks):
    draw_spent = sum(p["stake"] for p in draw_singles)
    over_spent = sum(p["stake"] for p in over_singles)
    funbet_draw_spent = funbet_draw["total_stake"] if funbet_draw else 0
    funbet_over_spent = funbet_over["total_stake"] if funbet_over else 0
    kelly_spent = sum(p["stake"] for p in kelly_picks)

    return [
        {
            "wallet": "Draw",
            "before": f"{DRAW_WALLET}€",
            "after": f"{DRAW_WALLET - draw_spent}€",
            "open_bets": f"{draw_spent}€",
        },
        {
            "wallet": "Over",
            "before": f"{OVER_WALLET}€",
            "after": f"{OVER_WALLET - over_spent}€",
            "open_bets": f"{over_spent}€",
        },
        {
            "wallet": "FanBet Draw",
            "before": f"{FANBET_DRAW_WALLET}€",
            "after": f"{FANBET_DRAW_WALLET - funbet_draw_spent}€",
            "open_bets": f"{funbet_draw_spent}€",
        },
        {
            "wallet": "FunBet Over",
            "before": f"{FANBET_OVER_WALLET}€",
            "after": f"{FANBET_OVER_WALLET - funbet_over_spent}€",
            "open_bets": f"{funbet_over_spent}€",
        },
        {
            "wallet": "Kelly",
            "before": f"{KELLY_WALLET}€",
            "after": f"{KELLY_WALLET - kelly_spent:.2f}€",
            "open_bets": f"{kelly_spent:.2f}€",
        },
    ]


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    log("🎯 Running Friday Shortlist (v1)...")

    fixtures = load_thursday_fixtures()
    odds_index = fetch_odds_from_theoddsapi()
    enriched = build_shortlist(fixtures, odds_index)

    draw_singles = generate_draw_singles(enriched)
    over_singles = generate_over_singles(enriched)

    funbet_draw = generate_funbet_draw(draw_singles) if len(draw_singles) >= 5 else None
    funbet_over = generate_funbet_over(over_singles) if len(over_singles) >= 3 else None

    kelly_picks = generate_kelly_picks(enriched)
    banks = bankroll_status(
        draw_singles, over_singles, funbet_draw or {"total_stake": 0},
        funbet_over or {"total_stake": 0}, kelly_picks
    )

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "source_report": THURSDAY_REPORT_PATH,
        "totals": {
            "fixtures": len(enriched),
            "draw_singles": len(draw_singles),
            "over_singles": len(over_singles),
            "kelly_picks": len(kelly_picks),
            "funbet_draw_columns": funbet_draw["columns_count"] if funbet_draw else 0,
            "funbet_over_columns": funbet_over["columns_count"] if funbet_over else 0,
        },
        "draw_singles": draw_singles,
        "over_singles": over_singles,
        "funbet_draw": funbet_draw,
        "funbet_over": funbet_over,
        "kelly_picks": kelly_picks,
        "bankroll_status": banks,
    }

    Path("logs").mkdir(parents=True, exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log(f"✅ Friday shortlist report saved: {FRIDAY_REPORT_PATH}")
    log(
        f"Draw singles: {len(draw_singles)}, "
        f"Over singles: {len(over_singles)}, "
        f"Kelly picks: {len(kelly_picks)}, "
        f"FunBet Draw cols: {report['totals']['funbet_draw_columns']}, "
        f"FunBet Over cols: {report['totals']['funbet_over_columns']}"
    )


if __name__ == "__main__":
    main()
