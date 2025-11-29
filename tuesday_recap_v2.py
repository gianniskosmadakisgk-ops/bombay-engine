import os
import json
from datetime import datetime
from collections import defaultdict
from typing import Dict, Tuple, Any, List

import requests

# ======================================================
#  TUESDAY RECAP v2  (Giannis Edition)
#
#  - Διαβάζει:
#       * logs/thursday_report_v1.json  (για ημερομηνίες / season)
#       * logs/friday_shortlist_v2.json (όλα τα bets της Παρασκευής)
#  - Τραβάει ΤΕΛΙΚΑ αποτελέσματα από API-Football:
#       * /leagues  (για να βρει league_id από league name)
#       * /fixtures (για FT σκορ στο ίδιο date window)
#  - Υπολογίζει:
#       * Αποτελέσματα για:
#           - Draw singles
#           - Over singles
#           - FunBet Draw system
#           - FunBet Over system
#           - Kelly bets (Home / Draw / Away / Over 2.5)
#       * Net profit per wallet + τελικό ταμείο
#  - Σώζει: logs/tuesday_recap_v2.json
# ======================================================

FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"

THURSDAY_REPORT_PATH = "logs/thursday_report_v1.json"
FRIDAY_REPORT_PATH = "logs/friday_shortlist_v2.json"
TUESDAY_RECAP_PATH = "logs/tuesday_recap_v2.json"

os.makedirs("logs", exist_ok=True)

# Τα ίδια αρχικά πορτοφόλια με Friday Shortlist
DRAW_WALLET = 400
OVER_WALLET = 300
FUNBET_DRAW_WALLET = 100
FUNBET_OVER_WALLET = 100
KELLY_WALLET = 300


# ------------------------------------------------------
# Helper logging
# ------------------------------------------------------
def log(msg: str):
    print(msg, flush=True)


# ------------------------------------------------------
# API-Football helpers
# ------------------------------------------------------
def api_get(path: str, params: dict) -> list:
    """
    Γενικό helper για API-Football.
    Επιστρέφει data.get("response", []) ή [] σε error.
    """
    if not FOOTBALL_API_KEY:
        raise RuntimeError("FOOTBALL_API_KEY is not set in environment.")

    headers = {"x-apisports-key": FOOTBALL_API_KEY}
    url = f"{FOOTBALL_BASE_URL}{path}"

    try:
        res = requests.get(url, headers=headers, params=params, timeout=25)
    except Exception as e:
        log(f"⚠️ Request error on {path}: {e}")
        return []

    if res.status_code != 200:
        log(f"⚠️ API error {res.status_code} on {path} with params {params}")
        try:
            log(f"⚠️ Body: {res.text[:300]}")
        except Exception:
            pass
        return []

    try:
        data = res.json()
    except Exception as e:
        log(f"⚠️ JSON decode error on {path}: {e}")
        return []

    errors = data.get("errors") or data.get("error")
    if errors:
        log(f"⚠️ API errors on {path}: {errors}")

    return data.get("response", [])


def get_league_id(league_name: str, season: str) -> int:
    """
    Βρίσκει το league_id από το όνομα της λίγκας + season.
    Αν δεν βρεθεί, επιστρέφει 0.
    """
    resp = api_get("/leagues", {"name": league_name, "season": season})
    if not resp:
        log(f"⚠️ No league_id found for '{league_name}' (season {season})")
        return 0

    try:
        league_id = int(resp[0]["league"]["id"])
        log(f"✅ League '{league_name}' → id {league_id}")
        return league_id
    except Exception as e:
        log(f"⚠️ Failed to extract league_id for '{league_name}': {e}")
        return 0


def fetch_fixtures_for_league(
    league_name: str,
    season: str,
    date_from: str,
    date_to: str,
) -> list:
    """
    Παίρνει όλα τα fixtures για μια συγκεκριμένη λίγκα,
    στο ίδιο date window με το Thursday report.
    """
    league_id = get_league_id(league_name, season)
    if not league_id:
        return []

    params = {
        "league": league_id,
        "season": season,
        "from": date_from,
        "to": date_to,
    }
    resp = api_get("/fixtures", params)
    log(f"✅ Fixtures for '{league_name}' (id {league_id}) → {len(resp)} matches")
    return resp


# ------------------------------------------------------
# Load Thursday + Friday data
# ------------------------------------------------------
def load_thursday_window() -> dict:
    if not os.path.exists(THURSDAY_REPORT_PATH):
        raise FileNotFoundError(f"Thursday report not found: {THURSDAY_REPORT_PATH}")

    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    window = data.get("source_window") or {}
    if not window:
        raise RuntimeError("source_window missing from Thursday report.")

    log(
        "📅 Using source_window from Thursday: "
        f"{window.get('date_from')} to {window.get('date_to')} "
        f"(season {window.get('season')})"
    )
    return window


def load_friday_bets() -> dict:
    if not os.path.exists(FRIDAY_REPORT_PATH):
        raise FileNotFoundError(f"Friday shortlist report not found: {FRIDAY_REPORT_PATH}")

    with open(FRIDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    log(
        "📥 Loaded Friday shortlist v2: "
        f"{len(data.get('draw_singles', []))} draw singles, "
        f"{len(data.get('over_singles', []))} over singles, "
        f"{len(data.get('kelly', {}).get('picks', []))} Kelly picks."
    )
    return data


# ------------------------------------------------------
# Results index
# ------------------------------------------------------
def build_results_index(
    leagues_used: List[str],
    season: str,
    date_from: str,
    date_to: str,
) -> Dict[Tuple[str, str], dict]:
    """
    Χτίζει index:
        (league_name, "Home - Away") -> {
            "home_goals": int,
            "away_goals": int,
            "status": str (short),
            "finished": bool,
            "is_draw": bool,
            "is_over_2_5": bool,
            "winner": "home"|"away"|"draw"|"unknown"
        }
    """
    index: Dict[Tuple[str, str], dict] = {}

    for league_name in sorted(leagues_used):
        fixtures = fetch_fixtures_for_league(league_name, season, date_from, date_to)
        for f in fixtures:
            try:
                lg_name = f["league"]["name"]
                home_name = f["teams"]["home"]["name"]
                away_name = f["teams"]["away"]["name"]
                match_label = f"{home_name} - {away_name}"

                goals_home = f["goals"]["home"]
                goals_away = f["goals"]["away"]
                status_short = f["fixture"]["status"]["short"]

                finished = status_short in {"FT", "AET", "PEN"}
                if goals_home is None or goals_away is None:
                    finished = False

                if finished:
                    hg = int(goals_home)
                    ag = int(goals_away)
                    is_draw = hg == ag
                    is_over_2_5 = (hg + ag) >= 3
                    if is_draw:
                        winner = "draw"
                    elif hg > ag:
                        winner = "home"
                    else:
                        winner = "away"
                else:
                    hg = goals_home if goals_home is not None else 0
                    ag = goals_away if goals_away is not None else 0
                    is_draw = False
                    is_over_2_5 = False
                    winner = "unknown"

                index[(lg_name, match_label)] = {
                    "home_goals": hg,
                    "away_goals": ag,
                    "status": status_short,
                    "finished": finished,
                    "is_draw": is_draw,
                    "is_over_2_5": is_over_2_5,
                    "winner": winner,
                }
            except Exception as e:
                log(f"⚠️ Error building result for a fixture: {e}")

    log(f"📊 Built results index for {len(index)} matches.")
    return index


def get_match_result(results_index: dict, league: str, match_label: str) -> dict:
    """
    Επιστρέφει αποτέλεσμα για συγκεκριμένο αγώνα.
    Αν δεν βρεθεί → status 'missing'.
    """
    data = results_index.get((league, match_label))
    if not data:
        return {
            "status": "missing",
            "finished": False,
            "winner": "unknown",
            "is_draw": False,
            "is_over_2_5": False,
            "home_goals": None,
            "away_goals": None,
        }
    return data


# ------------------------------------------------------
# Settlement helpers
# ------------------------------------------------------
def settle_single_draw(bet: dict, result: dict) -> Tuple[str, float]:
    """
    Επιστρέφει (status, net_profit) για Draw single.
    net_profit είναι σε € σε σχέση με initial bankroll:
        win  → stake*(odds-1)
        lose → -stake
        void/pending/missing → 0
    """
    stake = float(bet.get("stake", 0) or 0)
    odds = float(bet.get("odds", 0) or 0)

    if result["status"] == "missing":
        return "missing", 0.0
    if not result["finished"]:
        return "pending", 0.0

    if result["is_draw"]:
        return "win", stake * (odds - 1.0)
    else:
        return "lose", -stake


def settle_single_over(bet: dict, result: dict) -> Tuple[str, float]:
    stake = float(bet.get("stake", 0) or 0)
    odds = float(bet.get("odds", 0) or 0)

    if result["status"] == "missing":
        return "missing", 0.0
    if not result["finished"]:
        return "pending", 0.0

    if result["is_over_2_5"]:
        return "win", stake * (odds - 1.0)
    else:
        return "lose", -stake


def settle_kelly(bet: dict, result: dict) -> Tuple[str, float]:
    """
    Kelly bet μπορεί να είναι:
      - Home
      - Draw
      - Away
      - Over 2.5
    """
    stake = float(bet.get("stake (€)", 0) or 0)
    odds = float(bet.get("offered", 0) or 0)
    market = (bet.get("market") or "").lower()

    if result["status"] == "missing":
        return "missing", 0.0
    if not result["finished"]:
        return "pending", 0.0

    win = False
    if market == "home":
        win = result["winner"] == "home"
    elif market == "draw":
        win = result["winner"] == "draw"
    elif market == "away":
        win = result["winner"] == "away"
    elif market.startswith("over"):
        win = result["is_over_2_5"]

    if win:
        return "win", stake * (odds - 1.0)
    else:
        return "lose", -stake


def settle_funbet_column(
    picks: List[dict],
    results_index: dict,
    market_type: str,
    stake_per_column: float,
) -> Tuple[str, float]:
    """
    Υπολογίζει μία στήλη συστήματος (funbet).
    market_type: "draw" ή "over".
    Κανόνας:
      - Αν ΟΛΑ τα picks έχουν αποτέλεσμα και είναι κερδισμένα → win
      - Αν ΚΑΠΟΙΟ έχει χάσει → lose
      - Αν λείπει αποτέλεσμα για κάποιο → pending (0 profit)
    """
    # outcomes per pick
    statuses = []
    wins_flags = []

    for bet in picks:
        league = bet.get("league", "")
        match_label = bet.get("match", "")
        result = get_match_result(results_index, league, match_label)

        if result["status"] == "missing":
            statuses.append("missing")
            wins_flags.append(False)
            continue

        if not result["finished"]:
            statuses.append("pending")
            wins_flags.append(False)
            continue

        if market_type == "draw":
            win = result["is_draw"]
        else:
            win = result["is_over_2_5"]

        statuses.append("finished")
        wins_flags.append(win)

    # Αν κάποιο pending/missing → δεν το μετράμε (0), σαν void
    if any(s in {"pending", "missing"} for s in statuses):
        return "pending", 0.0

    # Αν έστω ένα χαμένο → χάνεται όλη η στήλη
    if not all(wins_flags):
        return "lose", -stake_per_column

    # Όλα κερδισμένα → multi-odds
    product_odds = 1.0
    for bet in picks:
        product_odds *= float(bet.get("odds", 0) or 0)

    return "win", stake_per_column * (product_odds - 1.0)


# ------------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------------
def main():
    log("🎯 Running Tuesday Recap (v2)...")

    # 1) Load Thursday window + Friday bets
    window = load_thursday_window()
    friday = load_friday_bets()

    date_from = window.get("date_from")
    date_to = window.get("date_to")
    season = str(window.get("season"))

    draw_singles = friday.get("draw_singles", []) or []
    over_singles = friday.get("over_singles", []) or []
    funbet_draw = friday.get("funbet_draw", {}) or {}
    funbet_over = friday.get("funbet_over", {}) or {}
    kelly_picks = friday.get("kelly", {}).get("picks", []) or []

    # 2) Βρίσκουμε ποιες λίγκες μας ενδιαφέρουν
    leagues_used = set()
    for bet in draw_singles + over_singles + kelly_picks:
        lg = bet.get("league")
        if lg:
            leagues_used.add(lg)

    for bet in funbet_draw.get("picks", []) or []:
        lg = bet.get("league")
        if lg:
            leagues_used.add(lg)

    for bet in funbet_over.get("picks", []) or []:
        lg = bet.get("league")
        if lg:
            leagues_used.add(lg)

    log(f"📚 Leagues in bets: {sorted(leagues_used)}")

    # 3) Χτίζουμε index αποτελεσμάτων
    results_index = build_results_index(
        leagues_used=sorted(leagues_used),
        season=season,
        date_from=date_from,
        date_to=date_to,
    )

    # 4) Wallet trackers
    wallets = {
        "Draw Singles": {
            "initial": DRAW_WALLET,
            "staked": 0.0,
            "profit": 0.0,
        },
        "Over Singles": {
            "initial": OVER_WALLET,
            "staked": 0.0,
            "profit": 0.0,
        },
        "FunBet Draw": {
            "initial": FUNBET_DRAW_WALLET,
            "staked": float(funbet_draw.get("total_stake", 0) or 0),
            "profit": 0.0,
        },
        "FunBet Over": {
            "initial": FUNBET_OVER_WALLET,
            "staked": float(funbet_over.get("total_stake", 0) or 0),
            "profit": 0.0,
        },
        "Kelly": {
            "initial": KELLY_WALLET,
            "staked": sum(float(b.get("stake (€)", 0) or 0) for b in kelly_picks),
            "profit": 0.0,
        },
    }

    # 5) Ledgers
    draw_results = []
    over_results = []
    funbet_draw_results = []
    funbet_over_results = []
    kelly_results = []

    # ---- Draw singles ----
    for bet in draw_singles:
        league = bet.get("league", "")
        match_label = bet.get("match", "")
        result = get_match_result(results_index, league, match_label)
        status, pnl = settle_single_draw(bet, result)

        stake = float(bet.get("stake", 0) or 0)
        wallets["Draw Singles"]["staked"] += stake
        wallets["Draw Singles"]["profit"] += pnl

        draw_results.append({
            **bet,
            "result_status": status,
            "home_goals": result.get("home_goals"),
            "away_goals": result.get("away_goals"),
            "pnl": round(pnl, 2),
        })

    # ---- Over singles ----
    for bet in over_singles:
        league = bet.get("league", "")
        match_label = bet.get("match", "")
        result = get_match_result(results_index, league, match_label)
        status, pnl = settle_single_over(bet, result)

        stake = float(bet.get("stake", 0) or 0)
        wallets["Over Singles"]["staked"] += stake
        wallets["Over Singles"]["profit"] += pnl

        over_results.append({
            **bet,
            "result_status": status,
            "home_goals": result.get("home_goals"),
            "away_goals": result.get("away_goals"),
            "pnl": round(pnl, 2),
        })

    # ---- FunBet Draw system ----
    fb_draw_picks = funbet_draw.get("picks", []) or []
    fb_draw_system = funbet_draw.get("system")
    fb_draw_stake_col = float(funbet_draw.get("stake_per_column", 0) or 0)

    if fb_draw_picks and fb_draw_system and fb_draw_stake_col > 0:
        n = len(fb_draw_picks)
        import itertools

        if fb_draw_system == "4-5-6":
            sizes = [4, 5, 6]
        elif fb_draw_system == "3-4-5":
            sizes = [3, 4, 5]
        else:
            sizes = []

        for r in sizes:
            for idxs in itertools.combinations(range(n), r):
                picks = [fb_draw_picks[i] for i in idxs]
                status, pnl = settle_funbet_column(
                    picks, results_index, market_type="draw",
                    stake_per_column=fb_draw_stake_col,
                )
                wallets["FunBet Draw"]["profit"] += pnl

                funbet_draw_results.append({
                    "system": fb_draw_system,
                    "combo_size": r,
                    "matches": [p["match"] for p in picks],
                    "leagues": [p["league"] for p in picks],
                    "status": status,
                    "stake": fb_draw_stake_col,
                    "pnl": round(pnl, 2),
                })

    # ---- FunBet Over system ----
    fb_over_picks = funbet_over.get("picks", []) or []
    fb_over_system = funbet_over.get("system")
    fb_over_stake_col = float(funbet_over.get("stake_per_column", 0) or 0)

    if fb_over_picks and fb_over_system and fb_over_stake_col > 0:
        n = len(fb_over_picks)
        import itertools

        # system = "2-from-n"
        for i, j in itertools.combinations(range(n), 2):
            picks = [fb_over_picks[i], fb_over_picks[j]]
            status, pnl = settle_funbet_column(
                picks, results_index, market_type="over",
                stake_per_column=fb_over_stake_col,
            )
            wallets["FunBet Over"]["profit"] += pnl

            funbet_over_results.append({
                "system": fb_over_system,
                "combo_size": 2,
                "matches": [p["match"] for p in picks],
                "leagues": [p["league"] for p in picks],
                "status": status,
                "stake": fb_over_stake_col,
                "pnl": round(pnl, 2),
            })

    # ---- Kelly bets ----
    for bet in kelly_picks:
        league = bet.get("league", "")
        match_label = bet.get("match", "")
        result = get_match_result(results_index, league, match_label)
        status, pnl = settle_kelly(bet, result)

        wallets["Kelly"]["profit"] += pnl

        kelly_results.append({
            **bet,
            "result_status": status,
            "home_goals": result.get("home_goals"),
            "away_goals": result.get("away_goals"),
            "pnl": round(pnl, 2),
        })

    # 6) Τελικό summary wallets
    wallets_summary = []
    for name, w in wallets.items():
        initial = float(w["initial"])
        staked = float(w["staked"])
        profit = float(w["profit"])
        final = initial + profit
        if staked > 0:
            yield_pct = profit / staked
        else:
            yield_pct = 0.0

        wallets_summary.append({
            "Wallet": name,
            "Initial": f"{initial:.2f}€",
            "Staked": f"{staked:.2f}€",
            "Profit": f"{profit:+.2f}€",
            "Final": f"{final:.2f}€",
            "Yield%": f"{yield_pct:+.1%}",
        })

    # 7) Χτίζουμε το τελικό report
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "source_window": window,
        "meta": {
            "draw_singles": len(draw_singles),
            "over_singles": len(over_singles),
            "kelly_picks": len(kelly_picks),
            "funbet_draw_cols": funbet_draw.get("columns", 0),
            "funbet_over_cols": funbet_over.get("columns", 0),
        },
        "wallets": wallets_summary,
        "draw_singles": draw_results,
        "over_singles": over_results,
        "funbet_draw_columns": funbet_draw_results,
        "funbet_over_columns": funbet_over_results,
        "kelly_bets": kelly_results,
    }

    with open(TUESDAY_RECAP_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log(f"✅ Tuesday recap report saved: {TUESDAY_RECAP_PATH}")

    # Μικρή σύνοψη για τα logs
    total_profit = sum(
        float(w["profit"]) for w in wallets.values()
    )
    log("📊 Wallets summary:")
    for w in wallets_summary:
        log(
            f"  - {w['Wallet']}: Staked {w['Staked']}, "
            f"Profit {w['Profit']}, Final {w['Final']}, Yield {w['Yield%']}"
        )
    log(f"💰 Total profit across all wallets: {total_profit:+.2f}€")


if __name__ == "__main__":
    main()
