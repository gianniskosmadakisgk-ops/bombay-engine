from flask import Flask, request, jsonify
import os
import json
import datetime

app = Flask(__name__)

# --- ROUTES ---

@app.route('/')
def home():
    return "Bombay Engine is live 🧠🔥"

@app.route('/friday', methods=['GET'])
def friday():
    return jsonify({
        "status": "Friday shortlist endpoint working",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

@app.route('/notify', methods=['POST'])
def notify():
    try:
        data = request.get_json(force=True)
        print("📩 Notification received:", data)

        # Log the notification for debugging
        with open("logs/last_notification.json", "w") as f:
            json.dump(data, f, indent=4)

        return jsonify({"message": "Notification received OK"}), 200
    except Exception as e:
        print("❌ Notify error:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat_notify():
    try:
        data = request.get_json(force=True)
        print("💬 Chat notification received:", data)

        # Save to file so you can see it worked
        with open("logs/chat_notification.json", "w") as f:
            json.dump(data, f, indent=4)

        return jsonify({"message": "Chat notification delivered"}), 200
    except Exception as e:
        print("❌ Chat error:", e)
        return jsonify({"error": str(e)}), 500

# --- MAIN ---

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
