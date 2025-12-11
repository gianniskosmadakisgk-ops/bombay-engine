"""
Thursday Engine V3
Production 1X2 & Over 2.5 Poisson Model
- xG-based expected goals (API-Football data)
- Momentum / Form / Injury / Rest / H2H / League draw bias
- 6x6 Poisson score matrix
- 1X2 probabilities (Draw No-Bet + draw caps)
- Over 2.5 probabilities (caps)
- Fair odds (no margin)
- The Odds API v4 integration (1X2 + Over 2.5)
"""

from __future__ import annotations

import math
from typing import Dict, Any, List, Tuple, Optional

import requests


# ============================================================
# CONSTANTS (σύμφωνα με τα specs σου)
# ============================================================

# Λόγω table:
#   xG For cap: 2.8
#   xG Against cap: 2.3
LAMBDA_HOME_MIN = 0.30
LAMBDA_HOME_MAX = 2.80
LAMBDA_AWAY_MIN = 0.30
LAMBDA_AWAY_MAX = 2.30

MAX_GOALS = 6  # 0..6 + tail lumped

DRAW_MIN = 0.22
DRAW_MAX = 0.32

OVER25_MIN = 0.35
OVER25_MAX = 0.72


# ============================================================
# HELPERS
# ============================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def poisson_pmf(lmbda: float, max_goals: int = MAX_GOALS) -> List[float]:
    """
    Poisson 0..max_goals + tail at index max_goals+1.
    """
    pmf = []
    for k in range(max_goals + 1):
        pmf.append(math.exp(-lmbda) * (lmbda ** k) / math.factorial(k))
    tail = 1.0 - sum(pmf)
    pmf.append(tail)
    return pmf


def implied_prob(odds: float) -> Optional[float]:
    if odds is None or odds <= 0:
        return None
    return 1.0 / odds


def normalize_three_way(p1: float, px: float, p2: float) -> Tuple[float, float, float]:
    s = p1 + px + p2
    if s <= 0:
        return 0.0, 0.0, 0.0
    return p1 / s, px / s, p2 / s


# ============================================================
# 1. EXPECTED GOALS (compute_expected_goals)
# ============================================================

def compute_expected_goals(
    *,
    # team season stats
    home_goals_scored: float,
    home_goals_conceded: float,
    home_games: int,
    away_goals_scored: float,
    away_goals_conceded: float,
    away_games: int,
    # league averages
    league_avg_home_goals: float,
    league_avg_away_goals: float,
    league_avg_home_goals_conceded: float,
    league_avg_away_goals_conceded: float,
    league_ov25_pct: float,
    league_draw_pct: float,
    # form / momentum (ratio vs 1.0 baseline)
    home_recent_xg_ratio: float = 1.0,   # last 5
    away_recent_xg_ratio: float = 1.0,
    home_form10_ratio: float = 1.0,      # last 10
    away_form10_ratio: float = 1.0,
    # injuries (count of key players)
    home_key_injured: int = 0,
    away_key_injured: int = 0,
    # rest days
    home_rest_days: float = 3.0,
    away_rest_days: float = 3.0,
    # draw biases
    h2h_draw_pct: float = 0.0,          # 0..1
) -> Dict[str, Any]:
    """
    Υλοποίηση του production spec:

      atk_home  = (home_goals_scored / home_games) / league_avg_home_goals
      def_away  = (away_goals_conceded / away_games) / league_avg_away_goals_conceded
      λ_home    = atk_home * def_away * league_avg_home_goals

      atk_away  = (away_goals_scored / away_games) / league_avg_away_goals
      def_home  = (home_goals_conceded / home_games) / league_avg_home_goals_conceded
      λ_away    = atk_away * def_home * league_avg_away_goals

    Adjustments / weights:
      Momentum (last 5)    : 0.12, cap ±0.3, Δλ = 0.12 * (recent_xg_ratio - 1.0)
      Form (last 10)       : 0.08, cap ±0.2, Δλ = 0.08 * (form10_ratio     - 1.0)
      Injury               : -0.25 per key, cap -0.5
      Rest Days            : -0.05 per missing day <3, cap -0.15
      H2H Draw Bias        : if h2h_draw_pct ≥ 0.60 → λ_both *= 0.90
      League Draw Bias     : if league_draw_pct ≥ 0.28 → λ_both *= 0.92
      Tempo Over Bias      : over_bias = min(0.15, 0.05 * (league_ov25_pct - 0.52))
                             λ_both *= (1 + over_bias)

    Caps:
      λ_home ∈ [0.30, 2.80]
      λ_away ∈ [0.30, 2.30]
    """

    debug: Dict[str, Any] = {}

    # --- base season per-game stats
    home_scored_pg = home_goals_scored / max(1, home_games)
    home_conceded_pg = home_goals_conceded / max(1, home_games)
    away_scored_pg = away_goals_scored / max(1, away_games)
    away_conceded_pg = away_goals_conceded / max(1, away_games)

    # --- atk / def strengths vs league
    atk_home = home_scored_pg / league_avg_home_goals
    def_away = away_conceded_pg / league_avg_away_goals_conceded

    atk_away = away_scored_pg / league_avg_away_goals
    def_home = home_conceded_pg / league_avg_home_goals_conceded

    debug["atk_home"] = atk_home
    debug["def_away"] = def_away
    debug["atk_away"] = atk_away
    debug["def_home"] = def_home

    # --- base lambdas
    λ_home = atk_home * def_away * league_avg_home_goals
    λ_away = atk_away * def_home * league_avg_away_goals

    debug["λ_home_base"] = λ_home
    debug["λ_away_base"] = λ_away

    # --------------------------------------------------------
    # Adjustments (βάσει πίνακα)
    # --------------------------------------------------------

    # Momentum (last 5)
    Δλ_home_mom = 0.12 * (home_recent_xg_ratio - 1.0)
    Δλ_away_mom = 0.12 * (away_recent_xg_ratio - 1.0)
    Δλ_home_mom = clamp(Δλ_home_mom, -0.3, 0.3)
    Δλ_away_mom = clamp(Δλ_away_mom, -0.3, 0.3)

    λ_home += Δλ_home_mom
    λ_away += Δλ_away_mom

    debug["Δλ_home_momentum"] = Δλ_home_mom
    debug["Δλ_away_momentum"] = Δλ_away_mom

    # Form (last 10)
    Δλ_home_form = 0.08 * (home_form10_ratio - 1.0)
    Δλ_away_form = 0.08 * (away_form10_ratio - 1.0)
    Δλ_home_form = clamp(Δλ_home_form, -0.2, 0.2)
    Δλ_away_form = clamp(Δλ_away_form, -0.2, 0.2)

    λ_home += Δλ_home_form
    λ_away += Δλ_away_form

    debug["Δλ_home_form"] = Δλ_home_form
    debug["Δλ_away_form"] = Δλ_away_form

    # Injury: -0.25 per key, cap -0.5
    Δλ_home_injury = -0.25 * home_key_injured
    Δλ_away_injury = -0.25 * away_key_injured
    Δλ_home_injury = clamp(Δλ_home_injury, -0.5, 0.0)
    Δλ_away_injury = clamp(Δλ_away_injury, -0.5, 0.0)

    λ_home += Δλ_home_injury
    λ_away += Δλ_away_injury

    debug["Δλ_home_injury"] = Δλ_home_injury
    debug["Δλ_away_injury"] = Δλ_away_injury

    # Rest Days: -0.05 per missing day below 3, cap -0.15
    Δλ_home_rest = 0.0
    Δλ_away_rest = 0.0
    if home_rest_days < 3:
        Δλ_home_rest = -0.05 * (3 - home_rest_days)
    if away_rest_days < 3:
        Δλ_away_rest = -0.05 * (3 - away_rest_days)
    Δλ_home_rest = clamp(Δλ_home_rest, -0.15, 0.0)
    Δλ_away_rest = clamp(Δλ_away_rest, -0.15, 0.0)

    λ_home += Δλ_home_rest
    λ_away += Δλ_away_rest

    debug["Δλ_home_rest"] = Δλ_home_rest
    debug["Δλ_away_rest"] = Δλ_away_rest

    # --- clamp before biases
    λ_home = clamp(λ_home, LAMBDA_HOME_MIN, LAMBDA_HOME_MAX)
    λ_away = clamp(λ_away, LAMBDA_AWAY_MIN, LAMBDA_AWAY_MAX)

    # H2H Draw Bias: if ≥ 60% draws → λ_both * 0.90
    if h2h_draw_pct >= 0.60:
        λ_home *= 0.90
        λ_away *= 0.90
        debug["h2h_draw_mul"] = 0.90
    else:
        debug["h2h_draw_mul"] = 1.0

    # League Draw Bias: if league_draw_pct ≥ 28% → λ_both * 0.92
    if league_draw_pct >= 0.28:
        λ_home *= 0.92
        λ_away *= 0.92
        debug["league_draw_mul"] = 0.92
    else:
        debug["league_draw_mul"] = 1.0

    # Tempo Adjustment (League Over Bias)
    over_bias = min(0.15, 0.05 * (league_ov25_pct - 0.52))
    λ_home *= (1.0 + over_bias)
    λ_away *= (1.0 + over_bias)

    debug["over_bias"] = over_bias

    # Final caps (safety)
    λ_home = clamp(λ_home, LAMBDA_HOME_MIN, LAMBDA_HOME_MAX)
    λ_away = clamp(λ_away, LAMBDA_AWAY_MIN, LAMBDA_AWAY_MAX)

    debug["λ_home_final"] = λ_home
    debug["λ_away_final"] = λ_away

    return {
        "λ_home": λ_home,
        "λ_away": λ_away,
        "debug": debug,
    }


# ============================================================
# 2. PROBABILITIES (compute_probabilities)
# ============================================================

def compute_probabilities(
    λ_home: float,
    λ_away: float,
    draw_bias: float = 1.0,
) -> Dict[str, Any]:
    """
    Υλοποίηση του production 1X2 & Over 2.5 μοντέλου.

    1X2:
      home_prob = Σ(h>a)P(h:a) + 0.5 Σ(h=a)P(h:a)
      draw_prob = Σ(h=a)P(h:a)
      away_prob = Σ(h<a)P(h:a) + 0.5 Σ(h=a)P(h:a)

      total = home_prob + away_prob
      home_prob /= total
      away_prob /= total

      draw_prob = clamp(draw_prob * 1.15 * draw_bias, 0.22, 0.32)

      final normalization: divide όλα με το sum(home+draw+away)

    Over 2.5:
      total_goals(n) = Σ(h+a=n)P(h:a)
      over_2_5 = Σ_{n>=3} total_goals(n)
      under_2_5 = 1 - over_2_5
      over_2_5 capped in [0.35, 0.72]
    """

    debug: Dict[str, Any] = {}

    pmf_home = poisson_pmf(λ_home, MAX_GOALS)
    pmf_away = poisson_pmf(λ_away, MAX_GOALS)

    debug["pmf_home"] = pmf_home
    debug["pmf_away"] = pmf_away

    # 6×6 score matrix
    matrix = [[0.0] * (MAX_GOALS + 1) for _ in range(MAX_GOALS + 1)]
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            matrix[h][a] = pmf_home[h] * pmf_away[a]

    # 1X2 raw
    home_raw = 0.0
    away_raw = 0.0
    draw_raw = 0.0
    draw_mass = 0.0

    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = matrix[h][a]
            if h > a:
                home_raw += p
            elif h < a:
                away_raw += p
            else:
                draw_raw += p
                draw_mass += p

    home_prob = home_raw + 0.5 * draw_mass
    away_prob = away_raw + 0.5 * draw_mass
    draw_prob = draw_raw

    debug["home_raw"] = home_raw
    debug["away_raw"] = away_raw
    debug["draw_raw"] = draw_raw
    debug["home_plus_half_draw"] = home_prob
    debug["away_plus_half_draw"] = away_prob

    # Draw No-Bet normalization (home & away only)
    total_dnb = home_prob + away_prob
    if total_dnb > 0:
        home_prob /= total_dnb
        away_prob /= total_dnb
    else:
        home_prob = away_prob = 0.0

    debug["home_dnb"] = home_prob
    debug["away_dnb"] = away_prob

    # Draw adjustment with bias
    draw_prob = draw_prob * 1.15 * draw_bias
    draw_prob = clamp(draw_prob, DRAW_MIN, DRAW_MAX)

    debug["draw_adj"] = draw_prob

    # Final normalization 1X2
    home_final, draw_final, away_final = normalize_three_way(
        home_prob, draw_prob, away_prob
    )

    debug["home_final"] = home_final
    debug["draw_final"] = draw_final
    debug["away_final"] = away_final

    # Over / Under 2.5
    total_goals_prob = [0.0] * (2 * MAX_GOALS + 1)  # 0..12

    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            tg = h + a
            if tg < len(total_goals_prob):
                total_goals_prob[tg] += matrix[h][a]

    over_2_5_raw = sum(total_goals_prob[3:])
    over_2_5 = clamp(over_2_5_raw, OVER25_MIN, OVER25_MAX)
    under_2_5 = 1.0 - over_2_5

    debug["over_2_5_raw"] = over_2_5_raw
    debug["over_2_5_capped"] = over_2_5
    debug["under_2_5"] = under_2_5

    # Fair odds (no margin)
    def fair_odds(p: float) -> Optional[float]:
        return None if p <= 0 else 1.0 / p

    odds = {
        "home": fair_odds(home_final),
        "draw": fair_odds(draw_final),
        "away": fair_odds(away_final),
        "over_2_5": fair_odds(over_2_5),
        "under_2_5": fair_odds(under_2_5),
    }

    return {
        "probs": {
            "home": home_final,
            "draw": draw_final,
            "away": away_final,
            "over_2_5": over_2_5,
            "under_2_5": under_2_5,
        },
        "odds": odds,
        "debug": debug,
    }


# ============================================================
# 3. THE ODDS API v4 INTEGRATION
# ============================================================

def fetch_theoddsapi_events(
    api_key: str,
    sport_key: str,
    regions: str = "eu",
    markets: str = "h2h,totals",
    odds_format: str = "decimal",
    date_format: str = "iso",
) -> List[Dict[str, Any]]:
    """
    GET /v4/sports/{sport_key}/odds from The Odds API.

    - sport_key π.χ. "soccer_epl", "soccer_greece_super_league"
    - επιστρέφει λίστα από events
    """
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }

    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def extract_1x2_and_over25_from_event(event: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """
    Από ένα event της The Odds API:

    - διαβάζει ΟΛΟΥΣ τους bookmakers
    - μαζεύει όλες τις τιμές για:
        home, draw, away (h2h)
        over_2_5, under_2_5 (totals point=2.5)
    - επιστρέφει AVERAGE odds για κάθε αγορά.

    Αν λείπει κάτι → None.
    """

    home_prices: List[float] = []
    draw_prices: List[float] = []
    away_prices: List[float] = []
    over25_prices: List[float] = []
    under25_prices: List[float] = []

    home_team = (event.get("home_team") or "").lower()
    away_team = (event.get("away_team") or "").lower()

    for book in event.get("bookmakers", []):
        for market in book.get("markets", []):
            key = market.get("key")
            outcomes = market.get("outcomes", [])

            if key == "h2h":
                # 3-way: home, draw, away
                for o in outcomes:
                    name = (o.get("name") or "").lower()
                    price = o.get("price")
                    if price is None:
                        continue

                    if name in ["home", home_team]:
                        home_prices.append(price)
                    elif name in ["draw", "tie", "x"]:
                        draw_prices.append(price)
                    elif name in ["away", away_team]:
                        away_prices.append(price)

            elif key == "totals":
                for o in outcomes:
                    point = o.get("point")
                    name = (o.get("name") or "").lower()
                    price = o.get("price")
                    if price is None or point != 2.5:
                        continue
                    if name == "over":
                        over25_prices.append(price)
                    elif name == "under":
                        under25_prices.append(price)

    if not home_prices or not draw_prices or not away_prices:
        return None
    if not over25_prices or not under25_prices:
        return None

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs)

    return {
        "home": avg(home_prices),
        "draw": avg(draw_prices),
        "away": avg(away_prices),
        "over_2_5": avg(over25_prices),
        "under_2_5": avg(under25_prices),
    }


def theoddsapi_market_probs_no_vig(odds: Dict[str, float]) -> Dict[str, float]:
    """
    Μετατρέπει τις average odds σε implied probabilities
    και τις normalizes (no vig) ξεχωριστά για 1X2 και O/U.
    """

    raw = {k: (implied_prob(v) or 0.0) for k, v in odds.items()}

    # 1X2
    s_1x2 = raw["home"] + raw["draw"] + raw["away"]
    if s_1x2 > 0:
        home = raw["home"] / s_1x2
        draw = raw["draw"] / s_1x2
        away = raw["away"] / s_1x2
    else:
        home = draw = away = 0.0

    # O/U 2.5
    s_ou = raw["over_2_5"] + raw["under_2_5"]
    if s_ou > 0:
        over25 = raw["over_2_5"] / s_ou
        under25 = raw["under_2_5"] / s_ou
    else:
        over25 = under25 = 0.0

    return {
        "home": home,
        "draw": draw,
        "away": away,
        "over_2_5": over25,
        "under_2_5": under25,
    }


def compare_model_vs_market(
    model_probs: Dict[str, float],
    market_probs_nv: Dict[str, float],
) -> Dict[str, Any]:
    """
    Υπολογίζει ratio model / market για edge analysis.
    Δεν αλλάζει το μοντέλο.
    """

    def edge_ratio(m: float, b: float) -> Optional[float]:
        if b <= 0:
            return None
        return m / b

    edges = {
        "home": edge_ratio(model_probs["home"], market_probs_nv["home"]),
        "draw": edge_ratio(model_probs["draw"], market_probs_nv["draw"]),
        "away": edge_ratio(model_probs["away"], market_probs_nv["away"]),
        "over_2_5": edge_ratio(model_probs["over_2_5"], market_probs_nv["over_2_5"]),
        "under_2_5": edge_ratio(model_probs["under_2_5"], market_probs_nv["under_2_5"]),
    }

    return {"edges": edges}


# ============================================================
# 4. MINI DEMO (μπορείς να το σβήσεις στο production)
# ============================================================

if __name__ == "__main__":
    # Dummy παράδειγμα για να δεις ότι τρέχει
    eg = compute_expected_goals(
        home_goals_scored=25,
        home_goals_conceded=18,
        home_games=15,
        away_goals_scored=20,
        away_goals_conceded=22,
        away_games=15,
        league_avg_home_goals=1.45,
        league_avg_away_goals=1.20,
        league_avg_home_goals_conceded=1.20,
        league_avg_away_goals_conceded=1.45,
        league_ov25_pct=0.54,
        league_draw_pct=0.27,
        home_recent_xg_ratio=1.05,
        away_recent_xg_ratio=0.98,
        home_form10_ratio=1.02,
        away_form10_ratio=0.97,
        home_key_injured=1,
        away_key_injured=0,
        home_rest_days=2,
        away_rest_days=4,
        h2h_draw_pct=0.50,
    )

    λ_home = eg["λ_home"]
    λ_away = eg["λ_away"]

    res = compute_probabilities(λ_home, λ_away, draw_bias=1.0)
    print("λ_home, λ_away:", λ_home, λ_away)
    print("probs:", res["probs"])
    print("fair odds:", res["odds"])
