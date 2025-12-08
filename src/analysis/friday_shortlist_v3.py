import os
import json
import itertools
import re
from datetime import datetime
from pathlib import Path

import requests

# ==============================================================================
#  FRIDAY SHORTLIST V3  (units engine)
#  - Διαβάζει Thursday v3 report
#  - Τραβάει αποδόσεις από TheOddsAPI
#  - Φτιάχνει:
#       * Draw Singles (flat 20 units)
#       * Over Singles (4 / 8 / 12 units)
#       * FanBet Draw (συστήματα 2-3 / 3-4 / 3-4-5-6)
#       * FanBet Over  (2-from-X / 3-from-X)
#       * Kelly value bets (ξεχωριστό wallet 600)
#  - Σώζει report → logs/friday_shortlist_v3.json
# ==============================================================================

# ---- paths -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]      # project root (/opt/render/project)
THURSDAY_REPORT_PATH = ROOT_DIR / "logs" / "thursday_report_v3.json"
FRIDAY_REPORT_PATH   = ROOT_DIR / "logs" / "friday_shortlist_v3.json"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

# ---- bankrolls (σε μονάδες) --------------------------------------------------
BANKROLL_DRAW      = 1000
BANKROLL_OVER      = 1000
BANKROLL_FUN_DRAW  = 300
BANKROLL_FUN_OVER  = 300
BANKROLL_KELLY     = 600

UNIT = 1.0  # 1 μονάδα

# ---- league priorities -------------------------------------------------------
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

# ---- mapping λίγκας → TheOddsAPI sport_key -----------------------------------
LEAGUE_TO_SPORT = {
    # Αγγλία
    "Premier League": "soccer_epl",
    "Championship": "soccer_efl_champ",
    # Ισπανία
    "La Liga": "soccer_spain_la_liga",
    "La Liga 2": "soccer_spain_segunda_division",
    # Ιταλία
    "Serie A": "soccer_italy_serie_a",
    "Serie B": "soccer_italy_serie_b",
    # Γερμανία
    "Bundesliga": "soccer_germany_bundesliga",
    "Bundesliga 2": "soccer_germany_bundesliga2",
    # Γαλλία
    "Ligue 1": "soccer_france_ligue_one",
    "Ligue 2": "soccer_france_ligue_two",
    # Πορτογαλία
    "Liga Portugal 1": "soccer_portugal_primeira_liga",
    # Ελβετία
    "Swiss Super League": "soccer_switzerland_superleague",
    # Ολλανδία
    "Eredivisie": "soccer_netherlands_eredivisie",
    # Βέλγιο
    "Jupiler Pro League": "soccer_belgium_first_div",
    # Δανία
    "Superliga": "soccer_denmark_superliga",
    # Σουηδία
    "Allsvenskan": "soccer_sweden_allsvenskan",
    # Νορβηγία
    "Eliteserien": "soccer_norway_eliteserien",
    # Λατινική Αμερική (αν τα βάλεις αργότερα στο Thursday)
    "Argentina Primera": "soccer_argentina_primera_division",
    "Brazil Serie A": "soccer_brazil_serie_a",
}

# ---- helpers -----------------------------------------------------------------
def log(msg: str):
    print(msg, flush=True)


def normalize_team(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"\b(fc|cf|afc|cfc|ac|sc|bk)\b", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---- Thursday loader ---------------------------------------------------------
def load_thursday_fixtures():
    if not THURSDAY_REPORT_PATH.exists():
        raise FileNotFoundError(f"Thursday report missing: {THURSDAY_REPORT_PATH}")

    with THURSDAY_REPORT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    fixtures = data.get("fixtures", [])
    log(f"Loaded {len(fixtures)} fixtures from {THURSDAY_REPORT_PATH}")
    return fixtures


# ---- Odds API ----------------------------------------------------------------
def get_odds_for_league(sport_key: str):
    if not ODDS_API_KEY:
        log("⚠️ ODDS_API_KEY not set → skipping odds.")
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
        log(f"⚠️ Odds HTTP {res.status_code} for {sport_key}: {res.text[:200]}")
        return []

    try:
        return res.json()
    except Exception:
        return []


def build_odds_index(fixtures):
    odds_index = {}

    leagues = sorted({f["league"] for f in fixtures if f.get("league") in LEAGUE_TO_SPORT})
    log(f"Leagues with odds support: {leagues}")

    for lg in leagues:
        sport_key = LEAGUE_TO_SPORT[lg]
        events = get_odds_for_league(sport_key)
        log(f"Fetched {len(events)} odds events for {lg}")

        for ev in events:
            home_raw = ev.get("home_team", "")
            away_raw = ev.get("away_team", "")
            h = normalize_team(home_raw)
            a = normalize_team(away_raw)

            if not h or not a:
                continue

            best_home = best_draw = best_away = best_over = None

            for bm in ev.get("bookmakers", []):
                for m in bm.get("markets", []):
                    key = m.get("key")

                    if key == "h2h":
                        for o in m.get("outcomes", []):
                            nm = normalize_team(o.get("name", ""))
                            price = float(o.get("price", 0) or 0)
                            if price <= 1.01:
                                continue
                            if nm == h:
                                best_home = max(best_home or 0.0, price)
                            elif nm == a:
                                best_away = max(best_away or 0.0, price)
                            elif nm == "draw":
                                best_draw = max(best_draw or 0.0, price)

                    elif key == "totals":
                        for o in m.get("outcomes", []):
                            name = o.get("name", "").lower()
                            price = float(o.get("price", 0) or 0)
                            if price <= 1.01:
                                continue
                            if "over" in name and "2.5" in name:
                                best_over = max(best_over or 0.0, price)

            odds_index[(h, a)] = {
                "home": best_home,
                "draw": best_draw,
                "away": best_away,
                "over_2_5": best_over,
            }

    log(f"Odds index size: {len(odds_index)}")
    return odds_index


# ---- scoring -----------------------------------------------------------------
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


def classify_over_tier(score: float) -> str:
    """
    Tiering για stakes:
      - score >= 80  → monster (12u)
      - 75–79.9      → premium (8u)
      - 70–74.9      → standard (4u)
    """
    if score >= 80:
        return "monster"
    if score >= 75:
        return "premium"
    if score >= 70:
        return "standard"
    return "none"


# ---- picks + Kelly -----------------------------------------------------------
def generate_picks(fixtures, odds_index):
    draw_singles = []
    over_singles = []
    kelly = []

    for f in fixtures:
        home = f["home"]
        away = f["away"]
        league = f["league"]

        fair_1 = f["fair_1"]
        fair_x = f["fair_x"]
        fair_2 = f["fair_2"]
        fair_over = f["fair_over_2_5"]

        draw_prob = float(f["draw_prob"])
        over_prob = float(f["over_2_5_prob"])

        h = normalize_team(home)
        a = normalize_team(away)
        odds = odds_index.get((h, a)) or odds_index.get((a, h), {})

        offered_home = odds.get("home")
        offered_x = odds.get("draw")
        offered_away = odds.get("away")
        offered_over = odds.get("over_2_5")

        draw_score = compute_draw_score(draw_prob, league)
        over_score = compute_over_score(over_prob, league)

        # ---------------- DRAW SINGLES (flat 20 units) ----------------
        if draw_prob >= 0.30 and fair_x <= 3.40:
            draw_singles.append({
                "match": f"{home} - {away}",
                "league": league,
                "fair": round(fair_x, 2),
                "prob": round(draw_prob, 3),
                "score": round(draw_score, 1),
                "offered": offered_x,
                "market_edge": (
                    round((offered_x - fair_x) / fair_x, 3)
                    if offered_x else None
                ),
                "stake": 20.0,   # 20 units flat
            })

        # ---------------- OVER SINGLES (4 / 8 / 12 units) -------------
        if over_prob >= 0.60 and fair_over <= 1.75:
            tier = classify_over_tier(over_score)
            if tier == "monster":
                stake = 12.0
            elif tier == "premium":
                stake = 8.0
            elif tier == "standard":
                stake = 4.0
            else:
                stake = 0.0

            if stake > 0:
                over_singles.append({
                    "match": f"{home} - {away}",
                    "league": league,
                    "fair": round(fair_over, 2),
                    "prob": round(over_prob, 3),
                    "score": round(over_score, 1),
                    "tier": tier,
                    "offered": offered_over,
                    "market_edge": (
                        round((offered_over - fair_over) / fair_over, 3)
                        if offered_over else None
                    ),
                    "stake": stake,
                })

        # ---------------- KELLY (ξεχωριστό wallet 600) ----------------
        def add_kelly(label, fair, offered, p):
            if not offered:
                return
            if p <= 0 or fair <= 1.01:
                return

            b = offered - 1.0
            q = 1.0 - p
            # Kelly fraction
            k_full = (b * p - q) / b
            if k_full <= 0:
                return

            edge = (offered / fair) - 1.0
            if edge < 0.15:   # θέλουμε σοβαρό value
                return

            fraction = 0.40 * k_full   # 40% Kelly
            stake_raw = BANKROLL_KELLY * fraction
            if stake_raw <= 0:
                return

            kelly.append({
                "match": f"{home} - {away}",
                "league": league,
                "market": label,
                "fair": round(fair, 2),
                "offered": round(offered, 2),
                "prob": round(p, 3),
                "edge": round(edge, 3),
                "stake": round(stake_raw, 2),
            })

        if offered_home:
            add_kelly("Home", fair_1, offered_home, 1.0 / fair_1)
        if offered_x:
            add_kelly("Draw", fair_x, offered_x, draw_prob)
        if offered_away:
            add_kelly("Away", fair_2, offered_away, 1.0 / fair_2)
        if offered_over:
            add_kelly("Over 2.5", fair_over, offered_over, over_prob)

    # limit Kelly total exposure to 35% του wallet
    max_exposure = BANKROLL_KELLY * 0.35
    total_raw = sum(k["stake"] for k in kelly)
    scale = max_exposure / total_raw if total_raw > max_exposure and total_raw > 0 else 1.0
    if scale < 1.0:
        for k in kelly:
            k["stake"] = round(k["stake"] * scale, 2)

    return draw_singles, over_singles, kelly


# ---- FanBet systems ----------------------------------------------------------
def build_funbet_draw(draw_singles):
    """
    Wallet: 300 units
    - 5–6 picks → σύστημα 3-4-5-6
    - 4 picks   → 3-4
    - 3 picks   → 2-3
    - <3        → off
    1 unit / column
    """
    picks = sorted(draw_singles, key=lambda x: x["score"], reverse=True)[:6]
    n = len(picks)

    if n < 3:
        return {"system": None, "columns": 0, "stake_per_column": 1.0,
                "total_stake": 0.0, "picks": []}

    if n == 3:
        sizes = [2, 3]
        system = "2-3"
    elif n == 4:
        sizes = [3, 4]
        system = "3-4"
    else:  # 5 ή 6
        sizes = [3, 4, 5, 6] if n >= 6 else [3, 4, 5]
        system = "3-4-5-6" if n >= 6 else "3-4-5"

    cols = sum(1 for r in sizes if r <= n for _ in itertools.combinations(range(n), r))
    stake_per_col = 1.0
    total = cols * stake_per_col

    return {
        "system": system,
        "columns": cols,
        "stake_per_column": stake_per_col,
        "total_stake": total,
        "picks": picks,
    }


def build_funbet_over(over_singles):
    """
    Wallet: 300 units
      - 3 picks → 2/3
      - 4 picks → 2/4
      - 5 picks → 3/5
      - 6 picks → 3/6 (αν τα κάνουμε πιο aggressive στο μέλλον → 4/6)
    1 unit / column
    """
    picks = sorted(over_singles, key=lambda x: x["score"], reverse=True)[:6]
    n = len(picks)

    if n < 3:
        return {"system": None, "columns": 0, "stake_per_column": 1.0,
                "total_stake": 0.0, "picks": []}

    if n == 3:
        sizes = [2]
        system = "2/3"
    elif n == 4:
        sizes = [2]
        system = "2/4"
    elif n == 5:
        sizes = [3]
        system = "3/5"
    else:  # 6 picks
        # απλό rule: αν ο μέσος score είναι πολύ ψηλά, μπορούμε αργότερα να πάμε 4/6
        avg_score = sum(p["score"] for p in picks) / n
        if avg_score >= 80:
            sizes = [4]
            system = "4/6"
        else:
            sizes = [3]
            system = "3/6"

    cols = sum(1 for r in sizes for _ in itertools.combinations(range(n), r))
    stake_per_col = 1.0
    total = cols * stake_per_col

    return {
        "system": system,
        "columns": cols,
        "stake_per_column": stake_per_col,
        "total_stake": total,
        "picks": picks,
    }


# ---- main --------------------------------------------------------------------
def main():
    log("🚀 Running Friday Shortlist V3 (units engine)")

    fixtures = load_thursday_fixtures()
    odds_index = build_odds_index(fixtures)

    draw_singles, over_singles, kelly = generate_picks(fixtures, odds_index)
    fb_draw = build_funbet_draw(draw_singles)
    fb_over = build_funbet_over(over_singles)

    # bankroll snapshot
    draw_open = sum(p["stake"] for p in draw_singles)
    over_open = sum(p["stake"] for p in over_singles)
    kelly_open = sum(k["stake"] for k in kelly)

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
            "after": round(BANKROLL_KELLY - kelly_open, 2),
            "open": round(kelly_open, 2),
        },
    }

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "fixtures_total": len(fixtures),
        "draw_singles": draw_singles,
        "over_singles": over_singles,
        "funbet_draw": fb_draw,
        "funbet_over": fb_over,
        "kelly": kelly,
        "bankrolls": bankrolls,
    }

    FRIDAY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FRIDAY_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log(f"✅ Friday Shortlist V3 saved → {FRIDAY_REPORT_PATH}")


if __name__ == "__main__":
    main()
