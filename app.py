import pandas as pd
from flask import Flask, jsonify
import random
from datetime import datetime

app = Flask(__name__)

# -----------------------------------------------------------
#  MATCH ENGINE – προσωρινό για δοκιμή χωρίς API
#  (μετά θα αντικατασταθεί από τα πραγματικά fixtures)
# -----------------------------------------------------------

def generate_matches():
    leagues = [
        "Premier League", "Serie A", "La Liga",
        "Super League (GR)", "Bundesliga", "Ligue 1"
    ]

    teams = [
        ["Arsenal", "Brighton"], ["Milan", "Lazio"],
        ["Betis", "Girona"], ["AEK", "PAOK"],
        ["Bayern", "Leipzig"], ["PSG", "Lyon"]
    ]

    data = []
    for i in range(len(teams)):
        home, away = teams[i]
        league = leagues[i]
        fair_1 = round(random.uniform(1.6, 2.6), 2)
        fair_x = round(random.uniform(3.0, 3.8), 2)
        fair_2 = round(random.uniform(2.8, 4.0), 2)
        fair_over = round(random.uniform(1.7, 2.1), 2)
        fair_under = round(random.uniform(1.9, 2.2), 2)
        draw_score = random.randint(6, 10)
        over_score = random.randint(5, 10)

        data.append({
            "league": league,
            "match": f"{home} - {away}",
            "fair_1": fair_1,
            "fair_x": fair_x,
            "fair_2": fair_2,
            "fair_over": fair_over,
            "fair_under": fair_under,
            "draw_conf": draw_score,
            "over_conf": over_score
        })
    return data


# -----------------------------------------------------------
#  ENDPOINTS — συνδέονται απευθείας με το Chat
# -----------------------------------------------------------

@app.route('/')
def home():
    return jsonify({"status": "Bombay Engine live"})

@app.route('/thursday-analysis')
def thursday_analysis():
    matches = generate_matches()
    timestamp = datetime.utcnow().isoformat()

    # Δημιουργία πινάκων για Draw Engine & Over Engine
    draw_engine = []
    over_engine = []

    for m in matches:
        draw_engine.append({
            "league": m["league"],
            "match": m["match"],
            "fair_x": m["fair_x"],
            "draw_conf": m["draw_conf"]
        })
        over_engine.append({
            "league": m["league"],
            "match": m["match"],
            "fair_over": m["fair_over"],
            "over_conf": m["over_conf"]
        })

    report = {
        "status": "Thursday Analysis complete",
        "timestamp": timestamp,
        "fixtures_count": len(matches),
        "draw_engine": draw_engine,
        "over_engine": over_engine
    }

    return jsonify(report)


@app.route('/friday-shortlist')
def friday_shortlist():
    timestamp = datetime.utcnow().isoformat()
    return jsonify({
        "status": "Friday Shortlist complete",
        "timestamp": timestamp
    })


@app.route('/tuesday-recap')
def tuesday_recap():
    timestamp = datetime.utcnow().isoformat()
    return jsonify({
        "status": "Tuesday Recap complete",
        "timestamp": timestamp
    })


# -----------------------------------------------------------
#  RUN SERVER
# -----------------------------------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
