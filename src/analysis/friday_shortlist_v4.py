import os
import json
import itertools
from datetime import datetime
import re
import requests

# ==============================================================================
#  FRIDAY SHORTLIST V4 — PRODUCTION VERSION (UNITS-BASED STAKING)
#  - Loads Thursday report (v3)
#  - Pulls offered odds from TheOddsAPI
#  - Builds:
#       * Draw Singles (flat stake)
#       * Over Singles (standard / premium / monster)
#       * FanBet Draw systems
#       * FanBet Over systems
#       * Kelly value bets
#  - Saves clean JSON report for UI / Custom GPT
# ==============================================================================

THURSDAY_REPORT_PATH = "logs/thursday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v4.json"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

# ------------------------------------------------------------------------------
# BANKROLLS (IN UNITS)
# ------------------------------------------------------------------------------
BANKROLL_DRAW = 1000
BANKROLL_OVER = 1000
BANKROLL_FUN_DRAW = 300
BANKROLL_FUN_OVER = 300
BANKROLL_KELLY = 600

UNIT = 1.0

# Draw Singles stake: flat 20 units
DRAW_STAKE_FLAT = 20 * UNIT

# Over Singles stakes:
OVER_STAKE_STANDARD = 4 * UNIT
OVER_STAKE_PREMIUM = 8 * UNIT
OVER_STAKE_MONSTER = 12 * UNIT

# Stake per column for all FunBets
FUNBET_STAKE_PER_COLUMN = 1 * UNIT

# ------------------------------------------------------------------------------
# LEAGUE PRIORITIES (tie-break / bonus, NOT hard filters)
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
    "Brazil Serie A": "soccer_brazil_serie_a",  # future use
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

# ------------------------------------------------------------------------------
# Load Thursday fixtures
# ------------------------------------------------------------------------------
def load_thursday():
    if not os.path.exists(THURSDAY_REPORT_PATH):
        raise FileNotFoundError(f"Thursday report missing: {THURSDAY_REPORT_PATH}")

    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f).get("fixtures", [])


# ------------------------------------------------------------------------------
# ODDS API
# ------------------------------------------------------------------------------
def get_odds_for_league(sport_key: str):
    if not ODDS_API_KEY:
        log("⚠️ Missing ODDS_API_KEY → skipping odds for this league")
        return []

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
    }

    try:
        res = requests.get(
            f"{ODDS_BASE_URL}/{sport_key}/odds",
            params=params,
            timeout=20,
        )
    except Exception as e:
        log(f"⚠️ Odds request error for {sport_key}: {e}")
        return []

    if res.status_code != 200:
        log(f"⚠️ Odds error [{sport_key}] status={res.status_code}")
        return []

    try:
        return res.json()
    except Exception as e:
        log(f"⚠️ Odds JSON decode error for {sport_key}: {e}")
        return []


def build_odds_index(fixtures):
    """Builds a (home_norm, away_norm) → odds dict index."""
    odds_index = {}

    leagues = sorted({
        f["league"] for f in fixtures
        if f.get("league") in LEAGUE_TO_SPORT
    })

    for lg in leagues:
        key = LEAGUE_TO_SPORT[lg]
        events = get_odds_for_league(key)
        log(f"Fetched {len(events)} odds events for {lg}")

        for ev in events:
            h_raw = ev.get("home_team", "")
            a_raw = ev.get("away_team", "")
            h = normalize_team(h_raw)
            a = normalize_team(a_raw)

            if not h or not a:
                continue

            best_home = best_draw = best_away = best_over = None

            for bm in ev.get("bookmakers", []):
                for m in bm.get("markets", []):
                    mk = m.get("key")

                    if mk == "h2h":
                        for o in m.get("outcomes", []):
                            nm = normalize_team(o.get("name", ""))
                            price = float(o.get("price", 0) or 0)
                            if price <= 0:
                                continue

                            if nm == h:
                                best_home = max(best_home or 0, price)
                            elif nm == a:
                                best_away = max(best_away or 0, price)
                            elif nm == "draw":
                                best_draw = max(best_draw or 0, price)

                    elif mk == "totals":
                        for o in m.get("outcomes", []):
                            name = o.get("name", "").lower()
                            price = float(o.get("price", 0) or 0)
                            if price <= 0:
                                continue
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
# SCORING
# ------------------------------------------------------------------------------
def compute_draw_score(draw_prob: float, league: str) -> float:
    score = draw_prob * 100.0
    if league in DRAW_PRIORITY_LEAGUES:
        score *= 1.05
    return score


def compute_over_score(over_prob: float, league: str) -> float:
    score = over_prob * 100.0
    if league in OVER_PRIORITY_LEAGUES:
        score *= 1.05
    return score


def classify_over_tier(over_prob: float) -> str:
    """
    Simple tiering rule:
      - monster  ≥ 0.70
      - premium  ≥ 0.64
      - standard otherwise (but still above entry threshold)
    """
    if over_prob >= 0.70:
        return "monster"
    if over_prob >= 0.64:
        return "premium"
    return "standard"


def over_stake_for_tier(tier: str) -> float:
    if tier == "monster":
        return OVER_STAKE_MONSTER
    if tier == "premium":
        return OVER_STAKE_PREMIUM
    return OVER_STAKE_STANDARD


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

        draw_prob = float(f["draw_prob"])
        over_prob = float(f["over_2_5_prob"])

        h = normalize_team(home)
        a = normalize_team(away)
        odds = odds_index.get((h, a), {})

        offered_x = odds.get("draw") or None
        offered_over = odds.get("over_2_5") or None

        draw_score = compute_draw_score(draw_prob, league)
        over_score = compute_over_score(over_prob, league)

        match_label = f"{home} - {away}"

        # ---------------- DRAW SINGLES ----------------
        # Entry rule: p(draw) >= 0.30 & fair_x <= 3.40
        if draw_prob >= 0.30 and fair_x <= 3.40:
            draw_singles.append({
                "match": match_label,
                "league": league,
                "fair": fair_x,
                "prob": draw_prob,
                "score": round(draw_score, 1),
                "offered": offered_x,
                "stake": DRAW_STAKE_FLAT,
            })

        # ---------------- OVER SINGLES ----------------
        # Entry rule: p(over) >= 0.60 & fair_over <= 1.75
        if over_prob >= 0.60 and fair_over <= 1.75:
            tier = classify_over_tier(over_prob)
            stake = over_stake_for_tier(tier)
            over_singles.append({
                "match": match_label,
                "league": league,
                "fair": fair_over,
                "prob": over_prob,
                "score": round(over_score, 1),
                "offered": offered_over,
                "tier": tier,
                "stake": stake,
            })

        # ---------------- KELLY VALUE BETS ----------------
        def add_kelly(label, fair, offered, p):
            if not offered:
                return
            if p <= 0 or p >= 1:
                return

            q = 1.0 - p
            b = offered - 1.0
            if b <= 0:
                return

            edge = p * offered - 1.0
            if edge < 0.10:  # min value threshold
                return

            f_kelly = (b * p - q) / b
            if f_kelly <= 0:
                return

            stake = BANKROLL_KELLY * f_kelly
            if stake <= 0:
                return

            kelly.append({
                "match": match_label,
                "league": league,
                "market": label,
                "fair": fair,
                "offered": offered,
                "prob": p,
                "edge": round(edge, 3),
                "stake": round(stake, 2),
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
    """
    FanBet Draw (wallet 300 units):
      - Take up to 6 best draws by score.
      - If 3 picks  → system 2-3
      - If 4 picks  → system 3-4
      - If 5-6 picks→ system 3-4-5-6 (trimmed to available picks)
      - Stake per column = 1 unit.
    """
    picks = sorted(draw_singles, key=lambda x: x["score"], reverse=True)[:6]
    n = len(picks)

    if n < 3:
        return {"system": None, "columns": 0, "total_stake": 0, "picks": []}

    if n == 3:
        sizes = [2, 3]
        system_str = "2-3"
    elif n == 4:
        sizes = [3, 4]
        system_str = "3-4"
    else:  # 5 ή 6 picks
        max_size = n
        sizes = [s for s in [3, 4, 5, 6] if s <= max_size]
        system_str = "-".join(str(s) for s in sizes)

    columns = 0
    for r in sizes:
        columns += sum(1 for _ in itertools.combinations(range(n), r))

    total_stake = columns * FUNBET_STAKE_PER_COLUMN

    return {
        "system": system_str,
        "columns": columns,
        "stake_per_column": FUNBET_STAKE_PER_COLUMN,
        "total_stake": total_stake,
        "picks": picks,
    }


def funbet_over(over_singles):
    """
    FanBet Over (wallet 300 units):
      - Take up to 6 best overs by score.
      - 3 picks → 2/3
      - 4 picks → 2/4
      - 5 picks → 3/5
      - 6 picks → 3/6
      - Stake per column = 1 unit.
    """
    picks = sorted(over_singles, key=lambda x: x["score"], reverse=True)[:6]
    n = len(picks)

    if n < 3:
        return {"system": None, "columns": 0, "total_stake": 0, "picks": []}

    if n == 3:
        sizes = [2, 3]
        system_str = "2-3"
    elif n == 4:
        sizes = [2, 4]
        system_str = "2-4"
    elif n == 5:
        sizes = [3, 5]
        system_str = "3-5"
    else:  # n == 6
        sizes = [3, 6]
        system_str = "3-6"

    columns = 0
    for r in sizes:
        columns += sum(1 for _ in itertools.combinations(range(n), r))

    total_stake = columns * FUNBET_STAKE_PER_COLUMN

    return {
        "system": system_str,
        "columns": columns,
        "stake_per_column": FUNBET_STAKE_PER_COLUMN,
        "total_stake": total_stake,
        "picks": picks,
    }


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    log("🚀 Running Friday Shortlist V4 (units version)")

    fixtures = load_thursday()
    odds_index = build_odds_index(fixtures)

    draw_singles, over_singles, kelly = generate_picks(fixtures, odds_index)

    fb_draw = funbet_draw(draw_singles)
    fb_over = funbet_over(over_singles)

    draw_open = sum(d["stake"] for d in draw_singles)
    over_open = sum(o["stake"] for o in over_singles)
    kelly_open = sum(k["stake"] for k in kelly)
    fun_draw_open = fb_draw["total_stake"]
    fun_over_open = fb_over["total_stake"]

    bankrolls = {
        "draw": {
            "before": BANKROLL_DRAW,
            "after": BANKROLL_DRAW - draw_open,
            "open": draw_open,
        },
        "over": {
            "before": BANKROLL_OVER,
            "after": BANKROLL_OVER - over_open,
            "open": over_open,
        },
        "fun_draw": {
            "before": BANKROLL_FUN_DRAW,
            "after": BANKROLL_FUN_DRAW - fun_draw_open,
            "open": fun_draw_open,
        },
        "fun_over": {
            "before": BANKROLL_FUN_OVER,
            "after": BANKROLL_FUN_OVER - fun_over_open,
            "open": fun_over_open,
        },
        "kelly": {
            "before": BANKROLL_KELLY,
            "after": BANKROLL_KELLY - kelly_open,
            "open": kelly_open,
        },
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
