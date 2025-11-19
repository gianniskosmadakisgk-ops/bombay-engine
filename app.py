from flask import Flask, request, jsonify
import json
import datetime

app = Flask(__name__)

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
        print("Notification received:", data)

        # Log the notification for debugging
        with open("logs/last_notification.json", "w") as f:
            json.dump(data, f, indent=4)

        return jsonify({"message": "Notification received OK"}), 200
    except Exception as e:
        print("Notify error:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat_notify():
    try:
        data = request.get_json(force=True)
        print("Chat webhook received:", data)
        return jsonify({"message": "Chat notification received OK"}), 200
    except Exception as e:
        print("Chat error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
