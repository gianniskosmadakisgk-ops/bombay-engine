from flask import Flask, request, jsonify
import requests
import os

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.route("/listener", methods=["POST"])
def listener():
    try:
        data = request.get_json()
        message = data.get("message", "")
        summary = data.get("data", {}).get("summary", "")

        print("📥 Received report from Render:")
        print(message)
        print(summary)

        # === Send report to ChatGPT ===
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "Εμφάνισε καθαρά αναφορές Bombay Engine"},
                {"role": "user", "content": f"{message}\n\n{summary}"}
            ]
        }

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }

        r = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        print("✅ Forwarded to ChatGPT API")

        return jsonify({"status": "ok", "openai_response": r.json()}), 200

    except Exception as e:
        print(f"❌ Listener error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/listener/health", methods=["GET"])
def healthcheck():
    return jsonify({"message": "Listener running", "status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10001))
    print(f"🟢 Starting listener service on port {port}")
    app.run(host="0.0.0.0", port=port)
