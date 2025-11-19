import json
import os
from datetime import datetime
import requests

def save_friday_log():
    """
    Fetch Friday data from the Render app and save it to /logs with timestamp.
    """
    # Endpoint του Render app σου
    FRIDAY_URL = "https://bombay-engine.onrender.com/friday"
    LOGS_DIR = "logs"

    try:
        response = requests.get(FRIDAY_URL)
        if response.status_code == 200:
            data = response.json()
            # Δημιουργεί filename με ημερομηνία
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
            file_path = os.path.join(LOGS_DIR, f"friday_{date_str}.json")

            # Αποθήκευση JSON
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ Friday log saved: {file_path}")
        else:
            print(f"⚠️ Error fetching Friday data. Status: {response.status_code}")
    except Exception as e:
        print(f"❌ Exception in save_friday_log: {e}")
