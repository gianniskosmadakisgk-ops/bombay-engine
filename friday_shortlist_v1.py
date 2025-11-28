import os
import json
import subprocess
from datetime import datetime

# === CONFIGURATION ===
ENGINES = [
    "engines/draw_engine.py",
    "engines/over_engine.py",
    "engines/funbet_draw.py",
    "engines/funbet_over.py",
    "engines/kelly_engine.py"
]

LOG_DIR = "logs"
OUTPUT_FILE = os.path.join(LOG_DIR, "friday_shortlist_v1.json")

# === Helper to run each engine ===
def run_engine(script_path):
    print(f"\n🚀 Running {script_path} ...")
    result = subprocess.run(
        ["python3", script_path],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("⚠️ STDERR:")
        print(result.stderr)
    print(f"✅ Finished {script_path}\n")

# === Step 1: Run all engines ===
os.makedirs(LOG_DIR, exist_ok=True)

for script in ENGINES:
    run_engine(script)

# === Step 2: Merge all partial results ===
summary = {
    "report_name": "Friday Shortlist Summary",
    "generated_at": datetime.utcnow().isoformat(),
    "status": "processing"
}

# Load all expected sub-reports
reports = {
    "draw_engine": "friday_draw_shortlist.json",
    "over_engine": "friday_over_shortlist.json",
    "funbet_draw": "friday_funbet_draw.json",
    "funbet_over": "friday_funbet_over.json",
    "fraction_kelly": "friday_kelly.json"
}

for key, filename in reports.items():
    path = os.path.join(LOG_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                summary[key] = data
                print(f"✅ Loaded {filename} ({len(data.get('picks', []))} picks)")
            except Exception as e:
                print(f"⚠️ Error reading {filename}: {e}")
    else:
        print(f"⚠️ Missing expected file: {filename}")
        summary[key] = {"error": "file not found", "picks": []}

# === Step 3: Add bankroll information ===
summary["wallets"] = {
    "Draw Engine": 400,
    "Over Engine": 300,
    "FunBet Draw": 200,
    "FunBet Over": 200,
    "Fraction Kelly": 300
}

summary["exposure"] = "calculated dynamically"
summary["status"] = "Friday shortlist complete ✅"

# === Step 4: Save merged report ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n📤 Friday shortlist generation complete — reports ready.")
print(f"📁 Output file: {OUTPUT_FILE}")
# === SAVE REPORT ===
os.makedirs("logs", exist_ok=True)
with open("logs/friday_shortlist_v1.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# === ALSO SAVE INDIVIDUAL ENGINE OUTPUTS FOR RECAP ===
for fund, filename in [
    ("Draw", "friday_draw_shortlist.json"),
    ("Over", "friday_over_shortlist.json"),
    ("FunBet Draw", "friday_funbet_draw.json"),
    ("FunBet Over", "friday_funbet_over.json"),
    ("Kelly", "friday_kelly.json"),
]:
    path = os.path.join("logs", filename)
    if fund.lower().replace(" ", "_") in summary:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary[fund.lower().replace(" ", "_")], f, indent=2, ensure_ascii=False)

print("📊 Friday shortlist and individual JSONs saved successfully.")
