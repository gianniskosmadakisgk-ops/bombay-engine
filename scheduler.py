import schedule
import time
import requests
import datetime

# Βάλε εδώ το URL του Render app σου
RENDER_BASE_URL = "https://bombay-engine.onrender.com"

def run_thursday():
    print(f"[{datetime.datetime.now()}] Running Thursday Analysis...")
    requests.get(f"{RENDER_BASE_URL}/thursday-analysis")

def run_friday():
    print(f"[{datetime.datetime.now()}] Running Friday Shortlist...")
    requests.get(f"{RENDER_BASE_URL}/friday-shortlist")

def run_tuesday():
    print(f"[{datetime.datetime.now()}] Running Tuesday Recap...")
    requests.get(f"{RENDER_BASE_URL}/tuesday-recap")

# Προγραμματισμός
schedule.every().tuesday.at("12:00").do(run_tuesday)
schedule.every().thursday.at("18:00").do(run_thursday)
schedule.every().friday.at("12:00").do(run_friday)

print("Scheduler is active and waiting...")

while True:
    schedule.run_pending()
    time.sleep(60)
