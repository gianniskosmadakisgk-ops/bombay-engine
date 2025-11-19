from flask import Flask, request, jsonify
import os
import json
import datetime
import requests

app = Flask(__name__)

# ---------------------------------------
# MAIN ROUTES
# ---------------------------------------

@app.route('/')
def home():
    return "🚀 Bombay Engine is live and connected to ChatGPT!"

# --- Friday endpoint ---
@app.route('/friday', methods=['GET'])
def friday():
    return jsonify({
        "status": "Friday shortlist endpoint working",
        "timestamp": datetime.datetime.utcnow().isoformat()
    })

# --- Notification endpoint (from GitHub) ---
@app.route('/notify', methods=['POST'])
def notify():
    try:
        data = request.get_json(force=True)
        print("🔔 Notification received:", data)

        # Save notification for reference
        os.makedirs("logs", exist_ok=True)
        with open("logs/last_notification.json", "w") as f:
            json.dump(data, f, indent=4)

        # Send update to Chat
        send_to_chat("📬 Bombay Engine received a new workflow update:\n" + json.dumps(data, indent=2))

        return jsonify({"message": "Notification received OK"}), 200
    except Exception as e:
        print("❌ Notify error:", e)
        return jsonify({"error": str(e)}), 500

# --- Chat webhook endpoint ---
@app.route('/chat', methods=['POST'])
def chat_notify():
    try:
        data = request.get_json(force=True)
        print("💬 Chat webhook received:", data)

        send_to_chat("✅ Thursday Analysis completed and forwarded from Bombay Engine.")

        return jsonify({"status": "Message delivered to Chat"}), 200
    except Exception as e:
        print("❌ Chat error:", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------
# FUNCTION TO FORWARD MESSAGES TO CHATGPT
# ---------------------------------------

def send_to_chat(message):
    chat_url = os.getenv("CHATGPT_WEBHOOK_URL")
    if not chat_url:
        print("⚠️ CHATGPT_WEBHOOK_URL not found in environment.")
        return False

    payload = {
        "text": message,
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

    try:
        r = requests.post(chat_url, json=payload, timeout=10)
        if r.status_code == 200:
            print("✅ Message sent successfully to ChatGPT.")
            return True
        else:
            print(f"⚠️ ChatGPT responded with {r.status_code}: {r.text}")
            return False
    except Exception as e:
        print("❌ Error sending to ChatGPT:", e)
        return False


# ---------------------------------------
# APP RUNNER
# ---------------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
