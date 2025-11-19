import requests
import os
from datetime import datetime

def send_to_chat(message):
    chat_url = os.getenv("CHATGPT_WEBHOOK_URL")  # Παίρνει το secret από GitHub
    if not chat_url:
        print("❌ CHATGPT_WEBHOOK_URL not found.")
        return False

    payload = {
        "text": message,
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        r = requests.post(chat_url, json=payload, timeout=10)
        if r.status_code == 200:
            print("✅ Message sent to ChatGPT successfully.")
            return True
        else:
            print(f"⚠️ ChatGPT responded with {r.status_code}: {r.text}")
            return False
    except Exception as e:
        print("❌ Error sending to ChatGPT:", e)
        return False


if __name__ == "__main__":
    send_to_chat("📡 Bombay Engine live test message at " + datetime.utcnow().isoformat())
