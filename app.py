from flask import Flask, jsonify
import requests
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# Load API key from environment
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")

HEADERS = {
    "x-apisports-key": FOOTBALL_API_KEY
}

BASE_URL = "https://v3.football.api-sports.io"

# Utility function to fetch fixtures for given days
def fetch_fixtures(days_forward=4):
    today = datetime.now()
    friday = today + timedelta(days=days_forward)
    monday = friday + timedelta(days=3)

    params = {
        "from": friday.strftime("%Y-%m-%d"),
        "to": monday.strftime("%Y-%m-%d"),
        "season": 2024
    }

    response = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS, params=params, timeout=20)
    data = response.json()
    return data.get("response", [])

# Route for Thursday Analysis
@app.route("/run_thursday_analysis", methods=["GET"])
def run_thursday_analysis():
    try:
        fixtures = fetch_fixtures()
        return jsonify({"count": len(fixtures), "data": fixtures, "status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Route for Friday Shortlist
@app.route("/run_friday_shortlist", methods=["GET"])
def run_friday_shortlist():
    try:
        fixtures = fetch_fixtures()
        shortlist = fixtures[:10]  # Mock shortlist for now
        return jsonify({"count": len(shortlist), "data": shortlist, "status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Route for health check
@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "Server running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
