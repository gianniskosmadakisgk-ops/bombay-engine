@app.route("/friday_shortlist", methods=["POST"])
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

📩 Απεστάλη στο ChatGPT
"""
    send_chat_message(message)
    return jsonify({"status": "Friday shortlist sent"}), 200
