import requests
import json
from datetime import datetime
import os

def fetch_friday_data():
    url = "https://bombay-engine.onrender.com/friday"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}")
        return None

def save_log(data):
    # Δημιουργούμε τον φάκελο logs αν δεν υπάρχει
    os.makedirs("logs", exist_ok=True)

    # Δημιουργούμε όνομα αρχείου με ημερομηνία
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"logs/friday_{timestamp}.json"

    # Αποθήκευση σε αρχείο JSON
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Log saved as {filename}")

def main():
    data = fetch_friday_data()
    if data:
        save_log(data)
    else:
        print("⚠️ No data fetched.")

if __name__ == "__main__":
    main()
