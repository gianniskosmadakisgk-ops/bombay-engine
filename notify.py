import os
import requests
import json
from datetime import datetime

def notify_bombay_chat(message: str):
    webhook_url = os.getenv("CHATGPT_WEBHOOK_URL")
    if not webhook_url:
        print("[Notify] ❌ No webhook URL found in environment variables.")
        return

    payload = {
        "text": message,
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 200:
            print("[Notify] ✅ Message sent successfully.")
        else:
            print(f"[Notify] ⚠️ Failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[Notify] ❌ Error sending message: {e}")


if __name__ == "__main__":
    notify_bombay_chat("📢 Test notification from Bombay Engine.")
