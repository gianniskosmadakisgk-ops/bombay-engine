import os
import json
import itertools
from datetime import datetime
import re
import requests

# ==============================================================================
#  FRIDAY SHORTLIST V3 — PRODUCTION (UNITS VERSION)
#  - Loads Thursday report v3 (fixtures with fair odds & probs)
#  - Pulls offered odds from TheOddsAPI (prefers Bet365 when available)
#  - Builds:
#       * Draw Singles (flat 30u stake)
#       * Over Singles (8 / 16 / 24u staking)
#       * FanBet Draw systems
#       * FanBet Over systems
#       * (Kelly stub – reserved for future)
#  - Saves clean JSON report → logs/friday_shortlist_v3.json
# ==============================================================================

THURSDAY_REPORT_PATH = "logs/thursday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v3.json"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

os.makedirs("logs", exist_ok=True)

# ------------------------------------------------------------------------------
# BANKROLLS (in units)
# ------------------------------------------------------------------------------
BANKROLL_DRAW = 1000.0
BANKROLL_OVER = 1000.0
BANKROLL_FUN_DRAW = 300.0
BANKROLL_FUN_OVER = 300.0
BANKROLL_KELLY = 600.0  # reserved, but not used yet

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
# ODDS SUPPORT MAP (TheOddsAPI sport keys)
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
    "Brazil Serie A": "soccer_brazil_serie_a",
}

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------
def log(msg: str):
    print(msg, flush=True)


def normalize_team(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\b(fc|cf|afc|cfc|ac|sc|bk)\b", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


# ------------------------------------------------------------------------------
# Load Thursday fixtures
# ------------------------------------------------------------------------------
def load_thursday_fixtures():
    if not os.path.exists(THURSDAY_REPORT_PATH):
        raise FileNotFoundError(f"Thursday report missing: {THURSDAY_REPORT_PATH}")

    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixtures = data.get("fixtures", [])
    log(f"Loaded {len(fixtures)} fixtures from Thursday v3.")
    return fixtures


# ------------------------------------------------------------------------------
# ODDS API
# ------------------------------------------------------------------------------
def get_odds_for_sport(sport_key: str):
    if not ODDS_API_KEY:
        log("⚠️ ODDS_API_KEY not set → skipping live odds.")
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
        log(f"⚠️ Odds request error for {sport_key}: {e}")
        return []

    if res.status_code != 200:
        log(f"⚠️ Odds API status {res.status_code} for {sport_key}")
        return []

    try:
        return res.json()
    except Exception:
        log("⚠️ Failed to decode odds JSON.")
        return []


def build_odds_index(fixtures):
    """
    Returns dict keyed by (home_norm, away_norm) with:
      { "home": ..., "draw": ..., "away": ..., "over_2_5": ... }
    Prefers Bet365 odds when available; otherwise best overall.
    """
    leagues = sorted({f["league"] for f in fixtures if f.get("league") in LEAGUE_TO_SPORT})
    log(f"Leagues with odds support: {leagues}")

    odds_index = {}

    for lg in leagues:
        sport_key = LEAGUE_TO_SPORT[lg]
        events = get_odds_for_sport(sport_key)
        log(f"Fetched {len(events)} odds events for {lg}")

        for ev in events:
            h_raw = ev.get("home_team", "")
            a_raw = ev.get("away_team", "")
            home = normalize_team(h_raw)
            away = normalize_team(a_raw)
            if not home or not away:
                continue

            overall = {"home": None, "draw": None, "away": None, "over_2_5": None}
            bet365 = {"home": None, "draw": None, "away": None, "over_2_5": None}

            for bm in ev.get("bookmakers", []):
                bm_key = (bm.get("key") or "").lower()
                is_bet365 = "bet365" in bm_key

                for m in bm.get("markets", []):
                    mk = m.get("key")

                    if mk == "h2h":
                        for o in m.get("outcomes", []):
                            nm = normalize_team(o.get("name", ""))
                            price = safe_float(o.get("price"))
                            if price is None:
                                continue

                            # overall best
                            if nm == home:
                                overall["home"] = max(overall["home"] or 0, price)
                                if is_bet365:
                                    bet365["home"] = max(bet365["home"] or 0, price)
                            elif nm == away:
                                overall["away"] = max(overall["away"] or 0, price)
                                if is_bet365:
                                    bet365["away"] = max(bet365["away"] or 0, price)
                            elif nm == "draw":
                                overall["draw"] = max(overall["draw"] or 0, price)
                                if is_bet365:
                                    bet365["draw"] = max(bet365["draw"] or 0, price)

                    elif mk == "totals":
                        for o in m.get("outcomes", []):
                            name = (o.get("name") or "").lower()
                            price = safe_float(o.get("price"))
                            if price is None:
                                continue
                            if "over" in name and "2.5" in name:
                                overall["over_2_5"] = max(overall["over_2_5"] or 0, price)
                                if is_bet365:
                                    bet365["over_2_5"] = max(bet365["over_2_5"] or 0, price)

            # choose bet365 if present, else overall
            chosen = {}
            for k in overall.keys():
                chosen[k] = bet365[k] if bet365[k] is not None else overall[k]

            odds_index[(home, away)] = chosen

    log(f"Odds index size: {len(odds_index)}")
    return odds_index


# ------------------------------------------------------------------------------
# SCORING
# ------------------------------------------------------------------------------
def draw_score(draw_prob: float, league: str) -> float:
    # base 0–100 scale
    score = draw_prob * 100.0
    if league in DRAW_PRIORITY_LEAGUES:
        score *= 1.05
    return score


def over_score(over_prob: float, league: str) -> float:
    score = over_prob * 100.0
    if league in OVER_PRIORITY_LEAGUES:
        score *= 1.05
    return score


# ------------------------------------------------------------------------------
# OVER STAKING (8 / 16 / 24)
# ------------------------------------------------------------------------------
def compute_over_stake(prob: float, odds: float) -> float:
    """
    - prob >= 0.65 (φίλτρο πριν φτάσουμε εδώ)
    - Bucket στο probability + bucket στα odds
    - Index 0/1/2 → stake 8 / 16 / 24 units
    """
    if odds is None:
        odds = 1.75  # neutral fallback

    # Probability buckets
    # 0.65–0.71   → 0
    # 0.72–0.80   → 1
    # >0.80       → 2
    if prob < 0.72:
        prob_bucket = 0
    elif prob < 0.80:
        prob_bucket = 1
    else:
        prob_bucket = 2

    # Odds buckets
    # 1.40–1.59 → +1 (πολύ χαμηλή απόδοση)
    # 1.60–1.85 → 0
    # >1.85     → -1
    if odds < 1.60:
        odds_bucket = 1
    elif odds <= 1.85:
        odds_bucket = 0
    else:
        odds_bucket = -1

    index = prob_bucket + odds_bucket
    if index < 0:
        index = 0
    if index > 2:
        index = 2

    tier_to_stake = {0: 8.0, 1: 16.0, 2: 24.0}
    return tier_to_stake[index]


# ------------------------------------------------------------------------------
# PICK GENERATION
# ------------------------------------------------------------------------------
def generate_picks(fixtures, odds_index):
    draw_singles = []
    over_singles = []
    kelly_picks = []  # placeholder; not used yet

    for f in fixtures:
        league = f.get("league")
        home = f.get("home")
        away = f.get("away")

        fair_x = safe_float(f.get("fair_x"))
        fair_over = safe_float(f.get("fair_over_2_5"))

        draw_prob = safe_float(f.get("draw_prob"), 0.0)
        over_prob = safe_float(f.get("over_2_5_prob"), 0.0)

        home_norm = normalize_team(home)
        away_norm = normalize_team(away)
        odds = odds_index.get((home_norm, away_norm)) or odds_index.get(
            (away_norm, home_norm), {}
        )

        offered_x = safe_float(odds.get("draw"))
        offered_over = safe_float(odds.get("over_2_5"))

        match_label = f"{home} - {away}"

        # ------------------------------------------------------------------ #
        # DRAW SINGLES
        # ------------------------------------------------------------------ #
        if draw_prob is not None and draw_prob >= 0.35 and fair_x is not None:
            score = draw_score(draw_prob, league)
            edge = None
            if offered_x and offered_x > 0 and fair_x > 0:
                edge = offered_x / fair_x - 1.0

            draw_singles.append(
                {
                    "match": match_label,
                    "league": league,
                    "fair": round(fair_x, 2),
                    "prob": round(draw_prob, 3),
                    "score": round(score, 1),
                    "offered": round(offered_x, 2) if offered_x else None,
                    "market_edge": round(edge, 3) if edge is not None else None,
                    "stake": 30.0,  # flat 30 units per draw
                }
            )

        # ------------------------------------------------------------------ #
        # OVER SINGLES
        # ------------------------------------------------------------------ #
        if over_prob is not None and over_prob >= 0.65 and fair_over is not None:
            score = over_score(over_prob, league)
            stake = compute_over_stake(over_prob, offered_over or fair_over)

            edge = None
            if offered_over and offered_over > 0 and fair_over > 0:
                edge = offered_over / fair_over - 1.0

            over_singles.append(
                {
                    "match": match_label,
                    "league": league,
                    "fair": round(fair_over, 2),
                    "prob": round(over_prob, 3),
                    "score": round(score, 1),
                    "offered": round(offered_over, 2) if offered_over else None,
                    "market_edge": round(edge, 3) if edge is not None else None,
                    "stake": float(stake),
                }
            )

    # --------------------------------------------------------------------------
    # RANKING & LIMITS
    # --------------------------------------------------------------------------
    def draw_sort_key(p):
        edge = p["market_edge"] if p["market_edge"] is not None else -999.0
        priority = 1 if p["league"] in DRAW_PRIORITY_LEAGUES else 0
        return (-p["score"], -edge, -priority)

    def over_sort_key(p):
        edge = p["market_edge"] if p["market_edge"] is not None else -999.0
        priority = 1 if p["league"] in OVER_PRIORITY_LEAGUES else 0
        return (-p["score"], -edge, -priority)

    draw_singles = sorted(draw_singles, key=draw_sort_key)[:10]
    over_singles = sorted(over_singles, key=over_sort_key)[:10]

    return draw_singles, over_singles, kelly_picks


# ------------------------------------------------------------------------------
# FUNBET SYSTEMS
# ------------------------------------------------------------------------------
def build_funbet_draw(draw_singles):
    """
    Uses top 3–6 draw picks from Draw Singles.
    Systems:
      3 picks → 2/3  (3 cols)
      4 picks → 2/4  (6 cols)
      5 picks → 3/5  (10 cols)
      6 picks → 3/6  (20 cols)
    Stake per column = 1 unit, total stake capped at 60 units.
    """
    picks = sorted(draw_singles, key=lambda x: x["score"], reverse=True)[:6]
    n = len(picks)

    if n < 3:
        return {"system": None, "columns": 0, "stake_per_column": 0.0, "total_stake": 0.0, "picks": []}

    if n == 3:
        system = "2/3"
        cols = 3
    elif n == 4:
        system = "2/4"
        cols = 6
    elif n == 5:
        system = "3/5"
        cols = 10
    else:  # n == 6
        system = "3/6"
        cols = 20

    base_stake_per_col = 1.0
    total = cols * base_stake_per_col

    if total > 60.0:
        total = 60.0
        stake_per_col = total / cols
    else:
        stake_per_col = base_stake_per_col

    return {
        "system": system,
        "columns": cols,
        "stake_per_column": round(stake_per_col, 2),
        "total_stake": round(total, 2),
        "picks": picks,
    }


def build_funbet_over(over_singles):
    """
    Uses top 3–6 over picks from Over Singles.
    Systems:
      3 picks → 2/3
      4 picks → 2/4
      5 picks → 3/5
      6 picks → 3/6
    Base stake per column ≈ 3 units, total stake capped at 60 units.
    """
    picks = sorted(over_singles, key=lambda x: x["score"], reverse=True)[:6]
    n = len(picks)

    if n < 3:
        return {"system": None, "columns": 0, "stake_per_column": 0.0, "total_stake": 0.0, "picks": []}

    if n == 3:
        system = "2/3"
        cols = 3
    elif n == 4:
        system = "2/4"
        cols = 6
    elif n == 5:
        system = "3/5"
        cols = 10
    else:  # n == 6
        system = "3/6"
        cols = 20

    base_stake_per_col = 3.0
    total = cols * base_stake_per_col

    if total > 60.0:
        total = 60.0
        stake_per_col = total / cols
    else:
        stake_per_col = base_stake_per_col

    return {
        "system": system,
        "columns": cols,
        "stake_per_column": round(stake_per_col, 2),
        "total_stake": round(total, 2),
        "picks": picks,
    }


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    log("🚀 Running Friday Shortlist V3 (final rules)")

    fixtures = load_thursday_fixtures()
    odds_index = build_odds_index(fixtures)

    draw_singles, over_singles, kelly_picks = generate_picks(fixtures, odds_index)

    fun_draw = build_funbet_draw(draw_singles)
    fun_over = build_funbet_over(over_singles)

    # Bankroll accounting
    draw_open = sum(p["stake"] for p in draw_singles)
    over_open = sum(p["stake"] for p in over_singles)
    fun_draw_open = fun_draw["total_stake"]
    fun_over_open = fun_over["total_stake"]
    kelly_open = sum(p.get("stake", 0.0) for p in kelly_picks)

    bankrolls = {
        "draw": {
            "before": BANKROLL_DRAW,
            "open": round(draw_open, 2),
            "after": round(BANKROLL_DRAW - draw_open, 2),
        },
        "over": {
            "before": BANKROLL_OVER,
            "open": round(over_open, 2),
            "after": round(BANKROLL_OVER - over_open, 2),
        },
        "fun_draw": {
            "before": BANKROLL_FUN_DRAW,
            "open": round(fun_draw_open, 2),
            "after": round(BANKROLL_FUN_DRAW - fun_draw_open, 2),
        },
        "fun_over": {
            "before": BANKROLL_FUN_OVER,
            "open": round(fun_over_open, 2),
            "after": round(BANKROLL_FUN_OVER - fun_over_open, 2),
        },
        "kelly": {
            "before": BANKROLL_KELLY,
            "open": round(kelly_open, 2),
            "after": round(BANKROLL_KELLY - kelly_open, 2),
        },
    }

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "fixtures_total": len(fixtures),
        "draw_singles": draw_singles,
        "over_singles": over_singles,
        "funbet_draw": fun_draw,
        "funbet_over": fun_over,
        "kelly": kelly_picks,
        "bankrolls": bankrolls,
    }

    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log(f"✅ Friday Shortlist V3 saved → {FRIDAY_REPORT_PATH}")


if __name__ == "__main__":
    main()
