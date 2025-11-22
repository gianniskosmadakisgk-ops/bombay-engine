from flask import Flask, jsonify
import numpy as np

app = Flask(__name__)

# -------------------------------
# SCORE CALCULATION MODULES
# -------------------------------

def calculate_draw_score(h2h_draw_rate, league_draw_rate, balance_index, recent_draw_form,
                         spi_diff, motivation_index, fatigue_index, weather_index,
                         injury_index, fair_x_norm, h2h_recent_draws):
    h2h_bonus = min(1, h2h_recent_draws / 10)
    score = (
        (h2h_draw_rate * 0.20) +
        (league_draw_rate * 0.15) +
        (balance_index * 0.15) +
        (recent_draw_form * 0.10) +
        ((1 - (spi_diff / 100)) * 0.10) +
        (motivation_index * 0.10) +
        (fatigue_index * 0.05) +
        (weather_index * 0.05) +
        (injury_index * 0.05) +
        (fair_x_norm * 0.05) +
        (h2h_bonus * 0.05)
    )
    return round(score * 10, 1)

def calculate_over_score(avg_xg_total, league_over_rate, form_over_rate, attack_strength,
                         defense_weakness, weather_index, spi_diff, motivation_index,
                         injury_index, fair_over_norm, h2h_recent_overs):
    h2h_bonus = min(1, h2h_recent_overs / 10)
    score = (
        (avg_xg_total * 0.25) +
        (league_over_rate * 0.15) +
        (form_over_rate * 0.10) +
        (attack_strength * 0.10) +
        (defense_weakness * 0.10) +
        (weather_index * 0.05) +
        ((spi_diff / 100) * 0.05) +
        (motivation_index * 0.05) +
        (injury_index * 0.05) +
        (fair_over_norm * 0.10) +
        (h2h_bonus * 0.05)
    )
    return round(score * 10, 1)

# -------------------------------
# ADAPTIVE LEARNING MODULE
# -------------------------------

def adaptive_weight_update(metrics):
    global WEIGHTS
    draw_corr = metrics.get("draw_correlation", 0)
    over_corr = metrics.get("over_correlation", 0)

    if draw_corr > 0.3:
        WEIGHTS["H2H_DRAW"] = min(WEIGHTS["H2H_DRAW"] + 0.05, 0.25)
    if over_corr < 0.1:
        WEIGHTS["WEATHER_OVER"] = max(WEIGHTS["WEATHER_OVER"] - 0.05, 0.05)
    return WEIGHTS

WEIGHTS = {
    "H2H_DRAW": 0.20,
    "WEATHER_OVER": 0.05
}

# -------------------------------
# MAIN ANALYSIS ENDPOINT
# -------------------------------

@app.route('/run_thursday_analysis', methods=['GET'])
def run_thursday_analysis():
    # Example data (μελλοντικά θα τραβά από το API)
    example_fixture = {
        "h2h_draw_rate": 0.3,
        "league_draw_rate": 0.25,
        "balance_index": 0.8,
        "recent_draw_form": 0.2,
        "spi_diff": 8,
        "motivation_index": 0.6,
        "fatigue_index": 0.5,
        "weather_index": 0.4,
        "injury_index": 0.5,
        "fair_x_norm": 0.7,
        "h2h_recent_draws": 4,
        "avg_xg_total": 1.8,
        "league_over_rate": 0.6,
        "form_over_rate": 0.5,
        "attack_strength": 0.7,
        "defense_weakness": 0.6,
        "fair_over_norm": 0.7,
        "h2h_recent_overs": 6
    }

    draw_score = calculate_draw_score(**{k: example_fixture[k] for k in [
        "h2h_draw_rate", "league_draw_rate", "balance_index", "recent_draw_form",
        "spi_diff", "motivation_index", "fatigue_index", "weather_index",
        "injury_index", "fair_x_norm", "h2h_recent_draws"
    ]})

    over_score = calculate_over_score(**{k: example_fixture[k] for k in [
        "avg_xg_total", "league_over_rate", "form_over_rate", "attack_strength",
        "defense_weakness", "weather_index", "spi_diff", "motivation_index",
        "injury_index", "fair_over_norm", "h2h_recent_overs"
    ]})

    return jsonify({
        "status": "success",
        "draw_score": draw_score,
        "over_score": over_score,
        "adaptive_weights": WEIGHTS
    })

if __name__ == '__main__':
    app.run(debug=True)
