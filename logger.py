import os
import json
from datetime import datetime
from friday import run_friday_simulation

def save_friday_log():
    result = run_friday_simulation()
    date_str = str(datetime.now().date())
    filename = f"logs/friday_{date_str}.json"

    # Αν δεν υπάρχει φάκελος logs, τον δημιουργεί
    os.makedirs("logs", exist_ok=True)

    # Αποθηκεύει το αρχείο JSON
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Friday simulation log saved: {filename}")
    return filename

if __name__ == "__main__":
    save_friday_log()
