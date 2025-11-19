from flask import Flask, jsonify, send_from_directory
import os
import json
import random
import datetime

app = Flask(__name__)

# --- Thursday Analysis Simulation ---
@app.route('/thursday-analysis', methods=['GET'])
def thursday_analysis():
    analysis_data = {
        "league": "Ligue 2 France",
        "matches_analyzed": 84,
        "draw_probability": 0.29,
        "avg_goals": 2.34,
        "comment": "Stable balance between Over/Under; watch late odds drift."
    }
    return jsonify(analysis_data)


# --- Friday Shortlist Simulation ---
@app.route('/friday-shortlist', methods=['GET'])
def friday_shortlist():
    sample_matches = [
        {"match": "Bordeaux vs Ajaccio", "fair_odds": {"1": 2.10, "X": 3.10, "2": 3.70}, "edge": "+14%"},
        {"match": "Parma vs Pisa", "fair_odds": {"1": 1.95, "X": 3.25, "2": 4.10}, "edge": "+12%"},
        {"match": "Granada vs Levante", "fair_odds": {"1": 2.40, "X": 3.00, "2": 3.00}, "edge": "+15%"},
        {"match": "Caen vs Amiens", "fair_odds": {"1": 2.05, "X": 3.20, "2": 3.60}, "edge": "+16%"},
    ]
    shortlist = {
        "date": str(datetime.date.today()),
        "matches": sample_matches,
        "note": "Top 4 value differences for this Friday shortlist."
    }
    return jsonify(shortlist)


# --- Tuesday Recap Simulation ---
@app.route('/tuesday-recap', methods=['GET'])
def tuesday_recap():
    recap = {
        "week": "Week 47",
        "bets_placed": 10,
        "wins": 6,
        "roi": "+8.7%",
        "bankroll_growth": "+26.1%",
        "comment": "Kelly fraction (0.5) remains optimal; maintain selection discipline."
    }
    return jsonify(recap)


# --- Serve OpenAPI YAML for ChatGPT Integration ---
@app.route('/openapi.yaml', methods=['GET'])
def serve_openapi():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'openapi.yaml')


# --- Health check route ---
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "Bombay Engine is running",
        "routes": [
            "/thursday-analysis",
            "/friday-shortlist",
            "/tuesday-recap",
            "/openapi.yaml"
        ]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
