import os
import requests
from flask import Flask, jsonify
from datetime import datetime

# Εισάγουμε τα modules των λειτουργιών
from friday import run_friday_shortlist
from simulate import run_thursday_analysis
from logger import run_tuesday_recap

app = Flask(__name__)

# ===== CONFIGURATION =====
API_KEY_FOOTBALL = os.getenv("FOOTBALL_API_KEY")
API_KEY_ODDS = os.getenv("ODDS_API_KEY")
WEBHOOK_URL = os.getenv("CHATGPT_WEBHOOK_URL")

BASE_URL_FOOTBALL = "https://v3.football.api-sports.io"
HEADERS_FOOTBALL = {"x-apisports-key": API_KEY_FOOTBALL}


# ===== BASIC ROUTE =====
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Bombay Engine Active",
        "version": "6.0",
        "timestamp": datetime.utcnow().isoformat()
    })


# ===== THURSDAY ANALYSIS =====
@app.route("/thursday-analysis", methods=["GET"])
def thursday_analysis():
    try:
        result = run_thursday_analysis()
        return jsonify({
            "status": "Thursday Analysis complete",
            "timestamp": datetime.utcnow().isoformat(),
            "data": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== FRIDAY SHORTLIST =====
@app.route("/friday-shortlist", methods=["GET"])
def friday_shortlist():
    try:
        result = run_friday_shortlist()
        return jsonify({
            "status": "Friday Shortlist complete",
            "timestamp": datetime.utcnow().isoformat(),
            "data": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===== TUESDAY RECAP =====
@app.route("/tuesday-recap", methods=["GET"])
def tuesday_recap():
    try:
        result = run_tuesday_recap()
        return jsonify({
            "status": "Tuesday Recap complete",
            "timestamp": datetime.utcnow().isoformat(),
            "data": result
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ====== START APP ======
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
