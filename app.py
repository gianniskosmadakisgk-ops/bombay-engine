from flask import Flask, request, jsonify
import datetime
import json
import os
import requests

app = Flask(__name__)

# -----------------------------
# ROUTES
# -----------------------------

@app.route('/')
def home():
    return "🚀 Bombay Engine is live and running!"

# -----------------------------
# FRIDAY SHORTLIST ENDPOINT
# -----------------------------

@app.route('/friday_shortlist', methods=['POST'])
def friday_shortlist():
    message = """
🎯 **Bombay Friday Shortlist**
Τα 10 κορυφαία picks της εβδομάδας:
────────────────────────────
⚽ **Draw Engine**
• Top 10 ισχυρότερα X
• Ενεργό FanBet System (4-5 επιλογές)

🔥 **Over/Under Engine**
• Top 10 ισχυρότερα Over/Under
• Ενεργό FanBet System (4-6 επιλογές)

💰 **Bankroll Update**
• Εφαρμόζεται Half-Kelly με min edge 10%
• ROI και ενεργά ταμεία ενημερωμένα

📩 Απεστάλη αυτόματα στο ChatGPT.
"""
    send_chat_message(message)
    return jsonify({"status": "Friday shortlist sent"}), 200


# -----------------------------
# THURSDAY ANALYSIS ENDPOINT
# -----------------------------

@app.route('/thursday_analysis', methods=['POST'])
def thursday_analysis():
    message = """
📊 **Bombay Thursday Analysis**
Ανάλυση εβδομάδας:
────────────────────────────
⚙️ **Performance Metrics**
• Draw Engine accuracy
• Over/Under success rate
• Bankroll evolution (7-day)

📈 **Upcoming Signals**
• Matches με υψηλό confidence για Παρασκευή
• Dynamic Odds Tracking ενεργό

📩 Η ανάλυση στάλθηκε στο ChatGPT.
"""
    send_chat_message(message)
    return jsonify({"status": "Thursday analysis sent"}), 200


# -----------------------------
# TUESDAY RECAP ENDPOINT
# -----------------------------

@app.route('/tuesday_recap', methods=['POST'])
def tuesday_recap():
    message = """
📅 **Bombay Tuesday Recap**
Σύνοψη και ROI update:
────────────────────────────
📊 **Results Summary**
• Εβδομαδιαία απόδοση ανά engine
• ROI % και strike rates

⚽ **Upcoming Schedule**
• Προετοιμασία για μεσοβδόμαδα simulations
• Τελευταία ενημέρωση bankroll

📩 Το recap στάλθηκε αυτόματα στο ChatGPT.
"""
    send_chat_message(message)
    return jsonify({"status": "Tuesday recap sent"}), 200


# -----------------------------
# NOTIFICATION ENDPOINT (GENERIC)
# -----------------------------

@app.route('/notify', methods=['POST'])
def notify():
    try:
        data = request.get_json(force=True)
        print("Notification received:", data)
        with open("logs/last_notification.json", "w") as f:
            json.dump(data, f, indent=4)
        return jsonify({"message": "Notification received OK"}), 200
    except Exception as e:
        print("Notify error:", e)
        return jsonify({"error": str(e)}), 500


# -----------------------------
# CHATGPT MESSAGE SENDER
# -----------------------------

def send_chat_message(content):
    """Send message directly to ChatGPT via webhook"""
    try:
        webhook_url = os.getenv("CHATGPT_WEBHOOK_URL")
        payload = {"text": content}
        headers = {"Content-Type": "application/json"}
        response = requests.post(webhook_url, json=payload, headers=headers)
        response.raise_for_status()
        print("✅ Chat message sent:", content)
    except Exception as e:
        print("❌ Error sending message:", e)


# -----------------------------
# MAIN
# -----------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
