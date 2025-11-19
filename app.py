from flask import Flask, jsonify
import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({
        "status": "Bombay Engine v6 active",
        "version": 6,
        "last_update": str(datetime.date.today()),
        "modes": ["Thursday Full Analysis", "Friday ShortList", "Tuesday Recap"]
    })

@app.route('/simulate')
def simulate():
    return jsonify({
        "date": str(datetime.date.today() + datetime.timedelta(days=1)),
        "matches_analyzed": 40,
        "min_edge": "10%",
        "method": "Half-Kelly",
        "fund": 300,
        "status": "Thursday Simulation ready"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
