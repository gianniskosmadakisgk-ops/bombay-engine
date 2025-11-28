from flask import Flask, request, jsonify
import os
import json
import requests

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "Listener service running", "status": "ok"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"message": "Listener running", "status": "ok"}), 200

@app.route("/listener", methods=["POST"])
def listener():
    try:
        data = request.get_json(force=True)
        print("📩 Received report from Render:", json.dumps(data, indent=2, ensure_ascii=False))

        chat_url = os.environ.get("CHAT_FORWARD_URL", "")
        if chat_url:
            print(f"📤 Forwarding report to chat at {chat_url}")
            r = requests.post(chat_url, json=data, timeout=20)
            print(f"✅ Forwarded with status: {r.status_code}")
        else:
            print("⚠️ CHAT_FORWARD_URL not set")

        return jsonify({"status": "ok", "message": "Report received"}), 200
    except Exception as e:
        print(f"❌ Error in listener: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10001))
    print(f"🟢 Starting listener service on port {port}...")
    app.run(host="0.0.0.0", port=port, use_reloader=False)

# Required for gunicorn
application = app
