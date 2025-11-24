import json
import numpy as np
import requests
from datetime import datetime
import os

# ---------------------------------------
# Config
# ---------------------------------------
API_URL = "https://bombay-engine.onrender.com/run_thursday_analysis"
FAIR_KEYS = ["fair_1", "fair_x", "fair_2", "fair_over"]
ACTUAL_KEYS = ["odd_1", "odd_x", "odd_2", "odd_over"]

# ---------------------------------------
# Core scoring engines
# ---------------------------------------

def calc_draw_score(match):
    """Υπολογίζει Draw Score βάσει των παραμέτρων"""
    weights = {
        "avg_xg_diff": 0.25,
        "spi_diff": 0.15,
        "form_similarity": 0.15,
        "h2h_draw_rate": 0.10,
        "league_draw_rate": 0.10,
        "motivation_balance": 0.05,
        "injury_impact": 0.05,
        "weather_balance": 0.05,
        "fair_odds_alignment": 0.10
    }

    # παράδειγμα normalized παραμέτρων (0–1)
    params = {
        "avg_xg_diff": match.get("avg_xg_diff", 0.5),
        "spi_diff": match.get("spi_diff", 0.5),
        "form_similarity": match.get("form_similarity", 0.5),
        "h2h_draw_rate": match.get("h2h_draw_rate", 0.5),
        "league_draw_rate": match.get("league_draw_rate", 0.5),
        "motivation_balance": match.get("motivation_balance", 0.5),
        "injury_impact": match.get("injury_impact", 0.5),
        "weather_balance": match.get("weather_balance", 0.5),
        "fair_odds_alignment": match.get("fair_odds_alignment", 0.5),
    }

    score = sum(params[k] * v for k, v in weights.items()) * 10
    return round(score, 2)

def calc_over_score(match):
    """Υπολογίζει Over Score βάσει των παραμέτρων"""
    weights = {
        "avg_xg_total": 0.25,
        "league_over_rate": 0.15,
        "form_over_rate": 0.10,
        "attack_strength": 0.10,
        "defense_weakness": 0.10,
        "weather_effect": 0.05,
        "spi_diff": 0.05,
        "motivation_index": 0.05,
        "injury_impact": 0.05,
        "fair_odds_calibration": 0.10
    }

    params = {
        "avg_xg_total": match.get("avg_xg_total", 0.5),
        "league_over_rate": match.get("league_over_rate", 0.5),
        "form_over_rate": match.get("form_over_rate", 0.5),
        "attack_strength": match.get("attack_strength", 0.5),
        "defense_weakness": match.get("defense_weakness", 0.5),
        "weather_effect": match.get("weather_effect", 0.5),
        "spi_diff": match.get("spi_diff", 0.5),
        "motivation_index": match.get("motivation_index", 0.5),
        "injury_impact": match.get("injury_impact", 0.5),
        "fair_odds_calibration": match.get("fair_odds_calibration", 0.5),
    }

    score = sum(params[k] * v for k, v in weights.items()) * 10
    return round(score, 2)

# ---------------------------------------
# Kelly Criterion
# ---------------------------------------

def kelly_fraction(p, b):
    """Υπολογίζει το ποσοστό Kelly"""
    return max(((p * (b + 1) - 1) / b), 0)

def calc_kelly_value(match):
    """Υπολογίζει τη διαφορά fair-actual και το stake"""
    results = {}
    bankroll = 300
    fraction = 0.4  # 40% του Kelly
    for fair_key, odd_key in zip(FAIR_KEYS, ACTUAL_KEYS):
        fair = match.get(fair_key)
        actual = match.get(odd_key)
        if not fair or not actual:
            continue
        fair_prob = 1 / fair
        actual_prob = 1 / actual
        edge = fair_prob - actual_prob
        value_diff = round(((actual - fair) / fair) * 100, 2)
        k = kelly_fraction(fair_prob, actual - 1)
        stake = round(bankroll * k * fraction, 2)
        results[odd_key] = {
            "value_diff_%": value_diff,
            "kelly_fraction": round(k, 3),
            "stake": stake
        }
    return results

# ---------------------------------------
# Friday Shortlist Generator
# ---------------------------------------

def generate_friday_shortlist():
    """Φέρνει τα δεδομένα της Thursday και δημιουργεί Shortlists"""
    try:
        response = requests.get(API_URL, timeout=20)
        data = response.json().get("data", [])
    except Exception as e:
        return {"status": "error", "message": str(e)}

    draw_list, over_list, kelly_list = [], [], []

    for match in data:
        draw_score = calc_draw_score(match)
        over_score = calc_over_score(match)
        kelly_info = calc_kelly_value(match)

        match["draw_score"] = draw_score
        match["over_score"] = over_score
        match["kelly_info"] = kelly_info

        if draw_score >= 7.5:
            draw_list.append(match)
        if over_score >= 7.5:
            over_list.append(match)
        if any(v["value_diff_%"] > 10 for v in kelly_info.values()):
            kelly_list.append(match)

    draw_top = sorted(draw_list, key=lambda x: x["draw_score"], reverse=True)[:10]
    over_top = sorted(over_list, key=lambda x: x["over_score"], reverse=True)[:10]
    kelly_top = sorted(kelly_list, key=lambda x: max(v["value_diff_%"] for v in x["kelly_info"].values()), reverse=True)[:10]

    return {
        "status": "success",
        "timestamp": datetime.utcnow().isoformat(),
        "draw_shortlist": draw_top,
        "over_shortlist": over_top,
        "kelly_shortlist": kelly_top
    }

# ---------------------------------------
# Manual Test
# ---------------------------------------
if __name__ == "__main__":
    result = generate_friday_shortlist()
    print(json.dumps(result, indent=2))
