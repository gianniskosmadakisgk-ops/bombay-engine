import os
import requests
from datetime import datetime

# απλό endpoint για ειδοποιήσεις προς ChatGPT Bombay
def send_notification(message: str):
    try:
        webhook_url = os.getenv("CHATGPT_WEBHOOK_URL")
        if not webhook_url:
            print("⚠️ Δεν υπάρχει webhook URL στο περιβάλλον.")
            return
        
        payload = {
            "text": f"📣 Bombay Notification:\n{message}\n🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        }
        requests.post(webhook_url, json=payload)
        print("✅ Εστάλη ειδοποίηση στο ChatGPT chat.")
    except Exception as e:
        print(f"❌ Σφάλμα αποστολής: {e}")
