import json
import os
import datetime
import requests

LOGS_DIR = "logs"
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)

def fetch_friday_data():
    try:
        response = requests.get("https://bombay-engine.onrender.com/friday")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def save_log(data):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{LOGS_DIR}/friday_{timestamp}.json"

    # Καταγραφή του log σε readable μορφή
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ Log saved as: {filename}")
    cleanup_old_logs()

def cleanup_old_logs():
    files = sorted(
        [f for f in os.listdir(LOGS_DIR) if f.endswith(".json")],
        key=lambda x: os.path.getmtime(os.path.join(LOGS_DIR, x))
    )
    if len(files) > 10:
        for old_file in files[:-10]:
            os.remove(os.path.join(LOGS_DIR, old_file))
            print(f"🗑️ Removed old log: {old_file}")

if __name__ == "__main__":
    data = fetch_friday_data()
    save_log(data)
