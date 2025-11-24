import os
import requests
from datetime import datetime, timedelta
from flask import Flask, jsonify

app = Flask(__name__)

# Environment variables
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
API_URL = "https://v3.football.api-sports.io/fixtures"

HEADERS = {
    "x-apisports-key": FOOTBALL_API_KEY,
    "x-rapidapi-host": "v3.football.api-sports.io"
}

# Υπολογισμός Παρασκευής–Δευτέρας
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
        "season": 2025  # ✅ Διόρθωση σε 2025
    }

    try:
        response = requests.get(API_URL, headers=HEADERS, params=params, timeout=30)
        data = response.json().get("response", [])
        return jsonify({
            "count": len(data),
            "range": {"from": friday, "to": monday},
            "data_sample": data[:3],
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "fail"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
