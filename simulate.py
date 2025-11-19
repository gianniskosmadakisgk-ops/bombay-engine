from flask import Flask, jsonify
import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({
        "status": "Bombay Engine v6 active",
        "version": 6,
        "last_update": str(datetime.date.today()),
        "modes": ["Thursday Full Analysis", "Friday Shortlist", "Tuesday Recap"]
    })

@app.route('/simulate')
def simulate():
    return jsonify({
        "date": str(datetime.date.today() + datetime.timedelta(days=1)),
        "matches_analyzed": 40,
        "min_edge": "10%",
        "method": "Half-Kelly",
        "fund": 300,
        "top_differences": [
            {"match": "AEK - PAOK", "fair_1x2": [1.95, 3.40, 4.10], "book_1x2": [2.25, 3.10, 3.60], "edge": "+15.4%"},
            {"match": "Panathinaikos - Aris", "fair_1x2": [1.80, 3.60, 4.50], "book_1x2": [2.05, 3.30, 3.95], "edge": "+13.9%"},
            {"match": "Olympiacos - Volos", "fair_ou": [1.85, 1.95], "book_ou": [2.15, 1.75], "edge": "+16.2%"}
        ],
        "status": "Thursday Simulation ready"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
