from flask import Flask, jsonify, request, send_from_directory
import os
import json
import datetime

app = Flask(__name__)

# --------------------------
# ROUTES
# --------------------------

@app.route('/')
def home():
    return "Bombay Engine is live 🔥"

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
        # Save notification log
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
        print("Chat notification received:", data)
        return jsonify({"message": "Chat message received"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------
# Serve plugin manifest
# --------------------------
@app.route('/.well-known/ai-plugin.json', methods=['GET'])
def serve_ai_plugin():
    return send_from_directory('.well-known', 'ai-plugin.json')

@app.route('/openapi.yaml', methods=['GET'])
def serve_openapi():
    return send_from_directory('.', 'openapi.yaml')

# --------------------------
# MAIN
# --------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
