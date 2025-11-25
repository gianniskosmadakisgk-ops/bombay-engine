import os
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

CHAT_WEBHOOK_URL = os.getenv("CHAT_WEBHOOK_URL", "https://api.openai.com/v1/chat/completions")  # placeholder

@app.route("/chat_forward", methods=["POST"])
def chat_forward():
    try:
        data = request.get_json()
        message = data.get("message", "No message")

        payload = {
            "model": "gpt-5",
            "messages": [{"role": "user", "content": message}]
        }

        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
            "Content-Type": "application/json"
        }

        response = requests.post(CHAT_WEBHOOK_URL, json=payload, headers=headers)
        if response.status_code == 200:
            return jsonify({"status": "sent", "response": response.json()}), 200
        else:
            return jsonify({"status": "fail", "error": response.text}), 500

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
