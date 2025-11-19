import pandas as pd
from flask import Flask, jsonify, request
import random

app = Flask(__name__)

# -----------------------------
# 🧠 FAKE MATCH ENGINE – για δοκιμές χωρίς API
# (Όταν κουμπώσει το API, αντικαθιστούμε το generate_matches)
# -----------------------------
def generate_matches():
    leagues = ["Premier League", "Serie A", "La Liga", "Super League (GR)", "Bundesliga", "Ligue 1"]
    teams = [
        ["Arsenal", "Brighton"], ["Milan", "Lazio"], ["Betis", "Girona"],
        ["AEK", "PAOK"], ["Bayern", "Leipzig"], ["PSG", "Lyon"]
    ]
    data = []
    for i in range(len(teams)):
        home, away = teams[i]
        league = leagues[i]
        fair_1 = round(random.uniform(1.6, 2.5), 2)
        fair_x = round(random.uniform(3.1, 4.2), 2)
        fair_2 = round(random.uniform(3.2, 5.0), 2)
        fair_over = round(random.uniform(1.85, 2.10), 2)
        fair_under = round(random.uniform(1.80, 2.10), 2)
        draw_score = round(random.uniform(5.0, 9.0), 1)
        over_score = round(random.uniform(4.0, 9.0), 1)
        under_score = round(random.uniform(4.0, 9.0), 1)
        ou_balance = round(over_score - under_score, 1)
        data.append({
            "League": league,
            "Match": f"{home} - {away}",
            "Fair 1": fair_1,
            "Fair X": fair_x,
            "Fair 2": fair_2,
            "Fair Over": fair_over,
            "Fair Under": fair_under,
            "Draw Score": draw_score,
            "Over Score": over_score,
            "Under Score": under_score,
            "OU Balance": ou_balance
        })
    return pd.DataFrame(data)

# -----------------------------
# 🧩 MAIN ROUTE – Thursday Analysis
# -----------------------------
@app.route("/thursday_analysis", methods=["GET"])
def thursday_analysis():
    df = generate_matches()
    # Μετατροπή πίνακα σε καθαρή markdown μορφή
    table_md = df.to_markdown(index=False)
    return jsonify({
        "message": "Thursday Analysis (Fair Odds + Scoring Model)",
        "table": table_md
    })

# -----------------------------
# 🔹 ChatGPT Trigger Route
# -----------------------------
@app.route("/trigger", methods=["POST"])
def trigger():
    data = request.json
    command = data.get("command", "").lower()

    if "thursday" in command:
        df = generate_matches()
        table_md = df.to_markdown(index=False)
        return jsonify({
            "message": "📊 Thursday Analysis Completed",
            "table": table_md
        })
    else:
        return jsonify({"message": "Command not recognized"})

# -----------------------------
# 🟢 MAIN
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
