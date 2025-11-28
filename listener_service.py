from flask import Flask, request, jsonify
import os
import json
import requests

app = Flask(__name__)

# === Root route ===
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Listener service running",
        "status": "ok"
    }), 200

# === Health check ===
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "message": "Listener running",
        "status": "ok"
    }), 200

# === Main listener endpoint ===
@app.route("/listener", methods=["POST"])
def listener():
    try:
        data = request.get_json(force=True)
        print("📩 Received report from Render:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

        # --- Load CHAT_FORWARD_URL safely ---
        chat_url = os.getenv("CHAT_FORWARD_URL", "").strip()
        print(f"🔍 Loaded CHAT_FORWARD_URL: {chat_url}")

        # --- Forward if set ---
        if chat_url:
            try:
                print(f"📤 Forwarding report to chat at {chat_url}")
                r = requests.post(chat_url, json=data, timeout=20)
                print(f"✅ Forwarded successfully (status {r.status_code})")
            except Exception as fwd_error:
                print(f"⚠️ Forwarding error: {fwd_error}")
        else:
            print("⚠️ CHAT_FORWARD_URL not set or empty — skipping forward")

        return jsonify({"status": "ok", "message": "Report received"}), 200

    except Exception as e:
        print(f"❌ Error in listener: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10001))
    print(f"🟢 Starting Bombay Listener service on port {port}...")
    app.run(host="0.0.0.0", port=port, use_reloader=False)

# Required for Gunicorn
application = app
