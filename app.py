from flask import Flask, jsonify
import datetime
from simulate import simulate

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
def run_simulation():
    return simulate()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
