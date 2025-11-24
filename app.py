from flask import Flask, jsonify, request
from datetime import datetime
from friday import generate_friday_shortlist
import requests

app = Flask(__name__)

# ----------------------------------------------------------
# Root
# ----------------------------------------------------------
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "Bombay Engine API is running",
        "timestamp": datetime.utcnow().isoformat()
    })

# ----------------------------------------------------------
# Thursday Analysis
# ----------------------------------------------------------
@app.route("/run_thursday_analysis", methods=["GET"])
def run_thursday_analysis():
    try:
        # Σημείωση: εδώ μιλάμε με τον Thursday engine
        response = requests.get("https://bombay-engine.onrender.com/run_thursday_analysis", timeout=25)
        data = response.json()
        return jsonify({
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# ----------------------------------------------------------
# Friday Shortlist
# ----------------------------------------------------------
@app.route("/run_friday_shortlist", methods=["GET"])
def run_friday_shortlist():
    try:
        result = generate_friday_shortlist()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

# ----------------------------------------------------------
# Health Check
# ----------------------------------------------------------
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({
        "pong": True,
        "timestamp": datetime.utcnow().isoformat()
    })

# ----------------------------------------------------------
# Run Flask app
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
