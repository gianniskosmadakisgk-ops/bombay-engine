import os
import json
import math
import re
from datetime import datetime

import requests

# ==============================================================================
#  FRIDAY SHORTLIST V3 — FINAL MULTI-WALLET ENGINE (UNITS = €1)
#  - Reads Thursday model output (fair odds & probabilities)
#  - Pulls odds from TheOddsAPI (preferring Bet365)
#  - Builds:
#       • Draw Singles (flat 30u)
#       • Over 2.5 Singles (8 / 16 / 24u, odds-sensitive)
#       • FunBet Draw (system, dynamic stake per column)
#       • FunBet Over (system, dynamic stake per column)
#       • Kelly Value Wallet (Home / Draw / Away / Over 2.5)
#  - Writes a single JSON payload used by the Custom GPT front-end.
#
#  All staking here is in UNITS. 1 unit = €1.
# ==============================================================================

THURSDAY_REPORT_PATH = "logs/thursday_report_v3.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v3.json"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

# ------------------------------------------------------------------------------
# BANKROLLS (in units)
# ------------------------------------------------------------------------------
BANKROLL_DRAW = 1000      # Draw Engine singles
BANKROLL_OVER = 1000      # Over 2.5 singles
BANKROLL_FUN_DRAW = 300   # FunBet Draw systems
BANKROLL_FUN_OVER = 300   # FunBet Over systems
BANKROLL_KELLY = 600      # Kelly value wallet

UNIT = 1.0  # 1 unit = €1

# ------------------------------------------------------------------------------
# LEAGUE PRIORITIES
# (priority only affects score / ordering, not eligibility)
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
# THEODDSAPI – league key map
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
def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_team(name: str) -> str:
    """Normalize team names for joining model fixtures with odds API."""
    if not name:
        return ""
    s = name.lower()
    s = re.sub(r"\b(fc|cf|afc|cfc|ac|sc|bk)\b", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------------------
# LOAD THURSDAY REPORT
# ------------------------------------------------------------------------------
def load_thursday_fixtures():
    if not os.path.exists(THURSDAY_REPORT_PATH):
        raise FileNotFoundError(f"Thursday report missing at {THURSDAY_REPORT_PATH}")

    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("fixtures", [])


# ------------------------------------------------------------------------------
# THEODDSAPI – ODDS FETCHING (PREFER BET365)
# ------------------------------------------------------------------------------
def get_odds_for_league(sport_key: str):
    if not ODDS_API_KEY:
        log("⚠️ Missing ODDS_API_KEY – odds will be empty.")
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
            log(f"⚠️ Odds error [{sport_key}] status={res.status_code} body={res.text[:200]}")
            return []
        return res.json()
    except Exception as e:
        log(f"⚠️ Odds request error for {sport_key}: {e}")
        return []


def build_odds_index(fixtures):
    """Return dict[(home_norm, away_norm)] -> dict of best odds, preferring Bet365."""
    odds_index = {}

    leagues = sorted({f["league"] for f in fixtures if f.get("league") in LEAGUE_TO_SPORT})

    for league in leagues:
        sport_key = LEAGUE_TO_SPORT[league]
        events = get_odds_for_league(sport_key)
        log(f"Fetched {len(events)} odds events for {league} ({sport_key})")

        for ev in events:
            h_raw = ev.get("home_team", "")
            a_raw = ev.get("away_team", "")
            h = normalize_team(h_raw)
            a = normalize_team(a_raw)

            best_home = best_draw = best_away = best_over = None
            b365_home = b365_draw = b365_away = b365_over = None

            for bm in ev.get("bookmakers", []):
                title = (bm.get("title") or bm.get("key") or "").lower()
                is_b365 = "365" in title or "bet365" in title

                for m in bm.get("markets", []):
                    mk = m.get("key")

                    if mk == "h2h":
                        for o in m.get("outcomes", []):
                            name = normalize_team(o.get("name", ""))
                            price = float(o.get("price", 0))
                            if not price:
                                continue
                            if name == h:
                                best_home = max(best_home or 0, price)
                                if is_b365:
                                    b365_home = max(b365_home or 0, price)
                            elif name == a:
                                best_away = max(best_away or 0, price)
                                if is_b365:
                                    b365_away = max(b365_away or 0, price)
                            elif name == "draw":
                                best_draw = max(best_draw or 0, price)
                                if is_b365:
                                    b365_draw = max(b365_draw or 0, price)

                    elif mk == "totals":
                        for o in m.get("outcomes", []):
                            nm = o.get("name", "").lower()
                            price = float(o.get("price", 0))
                            if not price:
                                continue
                            if "over" in nm and "2.5" in nm:
                                best_over = max(best_over or 0, price)
                                if is_b365:
                                    b365_over = max(b365_over or 0, price)

            odds_index[(h, a)] = {
                "home": b365_home or best_home,
                "draw": b365_draw or best_draw,
                "away": b365_away or best_away,
                "over_2_5": b365_over or best_over,
                "bookmaker": "Bet365" if any([b365_home, b365_draw, b365_away, b365_over]) else "best_available",
            }

    return odds_index


# ------------------------------------------------------------------------------
# SCORING
# ------------------------------------------------------------------------------
def compute_draw_score(p_draw: float, league: str) -> float:
    score = p_draw * 100.0
    if league in DRAW_PRIORITY_LEAGUES:
        score *= 1.05
    return score


def compute_over_score(p_over: float, league: str) -> float:
    score = p_over * 100.0
    if league in OVER_PRIORITY_LEAGUES:
        score *= 1.05
    return score


def over_stake_units(effective_odds: float) -> int:
    """Dynamic staking for Over 2.5 singles.
    Lower odds (safer) → higher stake.
    """
    if effective_odds is None or effective_odds <= 0:
        return 0
    if effective_odds <= 1.55:
        return 24
    if effective_odds <= 1.75:
        return 16
    if effective_odds <= 2.05:
        return 8
    return 0  # we ignore very high-odds overs in the singles engine


# ------------------------------------------------------------------------------
# PICK GENERATION (SINGLES + KELLY)
# ------------------------------------------------------------------------------
def generate_picks(fixtures, odds_index):
    draw_singles = []
    over_singles = []
    kelly_candidates = []

    for f in fixtures:
        home = f["home"]
        away = f["away"]
        league = f["league"]
        date = f.get("date")
        time = f.get("time")
        model = f.get("model", "")

        fair_1 = float(f.get("fair_1"))
        fair_x = float(f.get("fair_x"))
        fair_2 = float(f.get("fair_2"))
        fair_over = float(f.get("fair_over_2_5"))

        p_draw = float(f.get("draw_prob", 0.0))
        p_over = float(f.get("over_2_5_prob", 0.0))

        key = (normalize_team(home), normalize_team(away))
        odds = odds_index.get(key, {})
        offered_1 = odds.get("home") or None
        offered_x = odds.get("draw") or None
        offered_2 = odds.get("away") or None
        offered_over = odds.get("over_2_5") or None
        bookmaker = odds.get("bookmaker") or "unknown"

        # ---------------- DRAW SINGLES ----------------
        #   • Min model probability: 33%
        #   • Fair odds window: [2.40, 3.80]
        #   • Flat stake: 30u
        if p_draw >= 0.33 and 2.40 <= fair_x <= 3.80:
            score = compute_draw_score(p_draw, league)
            use_odds = offered_x or fair_x
            value_diff = ((use_odds - fair_x) / fair_x) if fair_x > 0 else 0.0

            draw_singles.append({
                "match": f"{home} - {away}",
                "league": league,
                "date": date,
                "time": time,
                "model": model,
                "prob": round(p_draw, 3),
                "score": round(score, 1),
                "fair": round(fair_x, 2),
                "odds": round(use_odds, 2),
                "bookmaker": bookmaker,
                "diff": f"{value_diff:+.0%}",
                "stake": 30,   # units
            })

        # ---------------- OVER 2.5 SINGLES ----------------
        #   • Min model probability: 65%
        #   • Fair odds window: [1.40, 2.10]
        #   • Stake buckets (by offered odds if available, else fair):
        #       ≤1.55 → 24u
        #       1.56–1.75 → 16u
        #       1.76–2.05 → 8u
        if p_over >= 0.65 and 1.40 <= fair_over <= 2.10:
            effective = offered_over or fair_over
            stake_units = over_stake_units(effective)
            if stake_units > 0:
                score = compute_over_score(p_over, league)
                value_diff = ((effective - fair_over) / fair_over) if fair_over > 0 else 0.0

                over_singles.append({
                    "match": f"{home} - {away}",
                    "league": league,
                    "date": date,
                    "time": time,
                    "model": model,
                    "prob": round(p_over, 3),
                    "score": round(score, 1),
                    "fair": round(fair_over, 2),
                    "odds": round(effective, 2),
                    "bookmaker": bookmaker,
                    "diff": f"{value_diff:+.0%}",
                    "stake": stake_units,
                })

        # ---------------- KELLY VALUE WALLET ----------------
        # We consider four markets where we have odds:
        #   • 1X2: Home / Draw / Away
        #   • Totals: Over 2.5
        # Rules:
        #   • Value threshold: offered ≥ fair * 1.15  (~15% edge)
        #   • Fractional Kelly: 30% of full Kelly fraction
        #   • Cap per bet: 5% of BANKROLL_KELLY
        #   • Odds window: [1.40, 8.00]
        def add_kelly(label: str, fair: float, offered: float):
            if not offered or offered <= 1.40 or offered > 8.0:
                return

            # Model probability from fair price
            p = 1.0 / fair
            if p <= 0 or p >= 0.80:
                return

            # Value edge vs fair
            edge = offered / fair - 1.0
            if edge < 0.15:
                return

            b = offered - 1.0
            q = 1.0 - p
            f_full = (b * p - q) / b  # full Kelly fraction
            if f_full <= 0:
                return

            f_frac = f_full * 0.30  # 30% Kelly
            f_frac = min(f_frac, 0.05)  # hard cap 5% of bankroll per bet
            if f_frac <= 0:
                return

            stake = BANKROLL_KELLY * f_frac
            if stake < 1.0:
                stake = 1.0  # minimal actionable stake

            kelly_candidates.append({
                "match": f"{home} - {away}",
                "league": league,
                "date": date,
                "time": time,
                "model": model,
                "market": label,
                "fair": round(fair, 2),
                "odds": round(offered, 2),
                "bookmaker": bookmaker,
                "edge": f"{edge:.0%}",
                "prob": round(p, 3),
                "stake": round(stake, 1),
                "edge_raw": edge,
            })

        if offered_1:
            add_kelly("Home", fair_1, offered_1)
        if offered_x:
            add_kelly("Draw", fair_x, offered_x)
        if offered_2:
            add_kelly("Away", fair_2, offered_2)
        if offered_over:
            add_kelly("Over 2.5", fair_over, offered_over)

    # ---------------- HARD CUTS & SORTING ----------------
    # Singles: limit to top 10 by score
    draw_singles = sorted(draw_singles, key=lambda x: x["score"], reverse=True)[:10]
    over_singles = sorted(over_singles, key=lambda x: x["score"], reverse=True)[:10]

    # Kelly: keep at most 6 strongest edges
    kelly_candidates = sorted(kelly_candidates, key=lambda x: x["edge_raw"], reverse=True)
    kelly = []
    for pick in kelly_candidates:
        if len(kelly) >= 6:
            break
        kelly.append({k: v for k, v in pick.items() if k != "edge_raw"})

    return draw_singles, over_singles, kelly


# ------------------------------------------------------------------------------
# FUNBET SYSTEMS
# ------------------------------------------------------------------------------
def build_funbet_draw(draw_singles):
    """System betting on top draw picks.
    Dynamic system choice + 1/2/3 units per column, capped at ~20% bankroll.
    """
    picks = sorted(draw_singles, key=lambda x: x["score"], reverse=True)
    if len(picks) < 3:
        return {"system": None, "columns": 0, "stake_per_column": 0, "total_stake": 0, "picks": []}

    n = min(len(picks), 7)
    picks = picks[:n]

    # Dynamic k based on n (validated by blueprint)
    if n == 3:
        k = 3          # 3/3
    elif n == 4:
        k = 3          # 3/4
    elif n == 5:
        k = 3          # 3/5
    elif n == 6:
        k = 4          # 4/6
    else:
        k = 4          # 4/7 on top 7 picks
        n = 7
        picks = picks[:7]

    cols = math.comb(n, k)
    max_exposure = BANKROLL_FUN_DRAW * 0.20  # 20% of FunBet Draw bankroll

    stake_per_column = 3
    for su in (3, 2, 1):
        if cols * su <= max_exposure:
            stake_per_column = su
            break

    total_stake = cols * stake_per_column

    return {
        "system": f"{k}/{n}",
        "columns": cols,
        "stake_per_column": stake_per_column,
        "total_stake": total_stake,
        "picks": picks,
    }


def build_funbet_over(over_singles):
    """System betting on top Over 2.5 picks.
    Same dynamic staking logic, different system map.
    """
    picks = sorted(over_singles, key=lambda x: x["score"], reverse=True)
    if len(picks) < 3:
        return {"system": None, "columns": 0, "stake_per_column": 0, "total_stake": 0, "picks": []}

    n = min(len(picks), 7)
    picks = picks[:n]

    # System choice (2-from / 3-from structure)
    if n == 3:
        k = 3          # 3/3
    elif n == 4:
        k = 2          # 2/4
    elif n == 5:
        k = 2          # 2/5
    elif n == 6:
        k = 3          # 3/6
    else:
        k = 3          # 3/7 on top 7
        n = 7
        picks = picks[:7]

    cols = math.comb(n, k)
    max_exposure = BANKROLL_FUN_OVER * 0.20  # 20% of FunBet Over bankroll

    stake_per_column = 3
    for su in (3, 2, 1):
        if cols * su <= max_exposure:
            stake_per_column = su
            break

    total_stake = cols * stake_per_column

    return {
        "system": f"{k}/{n}",
        "columns": cols,
        "stake_per_column": stake_per_column,
        "total_stake": total_stake,
        "picks": picks,
    }


# ------------------------------------------------------------------------------
# BANKROLL SUMMARY
# ------------------------------------------------------------------------------
def build_bankroll_summary(draw_singles, over_singles, fb_draw, fb_over, kelly):
    draw_open = sum(p["stake"] for p in draw_singles)
    over_open = sum(p["stake"] for p in over_singles)
    kelly_open = sum(p["stake"] for p in kelly)

    return [
        {
            "Wallet": "Draw Singles",
            "Before": f"{BANKROLL_DRAW}u",
            "Open": f"{draw_open:.1f}u",
            "After": f"{BANKROLL_DRAW - draw_open:.1f}u",
        },
        {
            "Wallet": "Over Singles",
            "Before": f"{BANKROLL_OVER}u",
            "Open": f"{over_open:.1f}u",
            "After": f"{BANKROLL_OVER - over_open:.1f}u",
        },
        {
            "Wallet": "FunBet Draw",
            "Before": f"{BANKROLL_FUN_DRAW}u",
            "Open": f"{fb_draw['total_stake']:.1f}u",
            "After": f"{BANKROLL_FUN_DRAW - fb_draw['total_stake']:.1f}u",
        },
        {
            "Wallet": "FunBet Over",
            "Before": f"{BANKROLL_FUN_OVER}u",
            "Open": f"{fb_over['total_stake']:.1f}u",
            "After": f"{BANKROLL_FUN_OVER - fb_over['total_stake']:.1f}u",
        },
        {
            "Wallet": "Kelly",
            "Before": f"{BANKROLL_KELLY}u",
            "Open": f"{kelly_open:.1f}u",
            "After": f"{BANKROLL_KELLY - kelly_open:.1f}u",
        },
    ]


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main():
    log("🚀 Running Friday Shortlist V3 (final engine, units mode)")

    fixtures = load_thursday_fixtures()
    log(f"Loaded {len(fixtures)} fixtures from Thursday report")

    odds_index = build_odds_index(fixtures)
    log(f"Built odds index for {len(odds_index)} fixtures")

    draw_singles, over_singles, kelly = generate_picks(fixtures, odds_index)

    fb_draw = build_funbet_draw(draw_singles)
    fb_over = build_funbet_over(over_singles)

    bankrolls = build_bankroll_summary(draw_singles, over_singles, fb_draw, fb_over, kelly)

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "fixtures_total": len(fixtures),
        "draw_singles": draw_singles,
        "over_singles": over_singles,
        "funbet_draw": fb_draw,
        "funbet_over": fb_over,
        "kelly": kelly,
        "bankroll_status": bankrolls,
    }

    os.makedirs(os.path.dirname(FRIDAY_REPORT_PATH), exist_ok=True)
    with open(FRIDAY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log(f"✅ Friday Shortlist saved → {FRIDAY_REPORT_PATH}")


if __name__ == "__main__":
    main()
