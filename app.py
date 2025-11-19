from flask import Flask, jsonify, request
from datetime import datetime
import os
import requests

app = Flask(__name__)

# --- Base route ---
@app.route("/")
def home():
    return jsonify({
        "status": "Bombay Engine running",
        "timestamp": datetime.utcnow().isoformat()
    })

# --- Friday simulation endpoint ---
@app.route("/friday", methods=["GET"])
def friday():
    # dummy response (παράδειγμα — μπορεί να συνδέεται με simulate.py)
    data = {
        "date": str(datetime.utcnow().date()),
        "fund": 300,
        "method": "Half-Kelly",
        "min_edge": "10%",
        "status": "Friday shortlist ready (EPL, LaLiga, SerieA)",
        "top_value_bets": [
            {"match": "Liverpool vs Newcastle", "odds": {"Home": 1.9, "Draw": 3.8, "Away": 4.2}},
            {"match": "Real Sociedad vs Betis", "odds": {"Home": 2.4, "Draw": 3.3, "Away": 3.1}}
        ]
    }
    return jsonify(data)

# --- Thursday analysis endpoint ---
@app.route("/thursday", methods=["GET"])
def thursday():
    data = {
        "status": "Thursday analysis completed",
        "matches_analyzed": 38,
        "leagues": ["EPL", "LaLiga", "Serie A"]
    }
    return jsonify(data)

# --- Tuesday recap endpoint ---
@app.route("/tuesday", methods=["GET"])
def tuesday():
    data = {
        "status": "Tuesday recap ready",
        "summary": {
            "win_rate": "64%",
            "profit": "+7.8 units",
            "sample_size": 52
        }
    }
    return jsonify(data)

# --- Notification endpoint (used by GitHub Actions) ---
@app.route("/api/notify", methods=["POST"])
def notify():
    data = request.get_json()
    secret_key = os.getenv("BOMBAY_CHAT_KEY")

    if not data or data.get("key") != secret_key:
        return jsonify({"error": "unauthorized"}), 403

    message = data.get("message", "")
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[BOMBAY NOTIFY] {timestamp} — {message}")

    # Εδώ στο μέλλον θα προστεθεί το direct chat hook
    return jsonify({"status": "ok", "received": message})

# --- Run app ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
