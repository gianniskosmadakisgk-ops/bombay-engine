import os
import requests
from datetime import datetime, timedelta
from flask import Flask, jsonify

app = Flask(__name__)

# Environment variables
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
API_URL = "https://v3.football.api-sports.io/fixtures"

HEADERS = {
    "x-apisports-key": FOOTBALL_API_KEY
}

# Υπολογισμός επόμενης Παρασκευής - Δευτέρας
def next_weekend_dates():
    today = datetime.utcnow()
    days_ahead = (4 - today.weekday()) % 7  # Παρασκευή
    friday = today + timedelta(days=days_ahead)
    monday = friday + timedelta(days=3)
    return friday.strftime("%Y-%m-%d"), monday.strftime("%Y-%m-%d")

@app.route("/healthcheck", methods=["GET"])
def healthcheck():
    return jsonify({"message": "Server running", "status": "ok"})

@app.route("/run_thursday_analysis", methods=["GET"])
def run_thursday_analysis():
    friday, monday = next_weekend_dates()
    params = {
        "from": friday,
        "to": monday,
        "season": 2024
    }

    response = requests.get(API_URL, headers=HEADERS, params=params)
    try:
        data = response.json().get("response", [])
        return jsonify({
            "count": len(data),
            "range": {"from": friday, "to": monday},
            "data_sample": data[:2],  # δείχνουμε μόνο 2 για να μην είναι τεράστιο
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "fail"}), 500


@app.route("/run_friday_shortlist", methods=["GET"])
def run_friday_shortlist():
    friday, monday = next_weekend_dates()
    params = {
        "from": friday,
        "to": monday,
        "season": 2024
    }

    response = requests.get(API_URL, headers=HEADERS, params=params)
    try:
        data = response.json().get("response", [])
        shortlist = [{"home": f["teams"]["home"]["name"], "away": f["teams"]["away"]["name"]}
                     for f in data[:10]]
        return jsonify({
            "count": len(shortlist),
            "shortlist": shortlist,
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "fail"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
