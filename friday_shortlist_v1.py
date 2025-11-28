import os
import json
import subprocess

# === CONFIG ===
ENGINES = [
    "engines/draw_engine.py",
    "engines/over_engine.py",
    "engines/funbet_draw.py",
    "engines/funbet_over.py",
    "engines/kelly_engine.py"
]

def run_engine(script_path):
    print(f"🚀 Running {script_path} ...")
    result = subprocess.run(
        ["python3", script_path],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    print(f"✅ Finished {script_path}\n")

# === RUN ALL ENGINES ===
for script in ENGINES:
    run_engine(script)

# === MERGE RESULTS ===
summary = {}
for name in [
    "friday_draw_shortlist.json",
    "friday_over_shortlist.json",
    "friday_funbet_draw.json",
    "friday_funbet_over.json",
    "friday_kelly.json"
]:
    path = os.path.join("logs", name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            key = name.replace("friday_", "").replace(".json", "")
            summary[key] = data.get("count", 0)

summary["wallets"] = {
    "Draw Engine": 400,
    "Over Engine": 300,
    "FunBet Draw": 200,
    "FunBet Over": 200,
    "Fraction Kelly": 300
}
summary["exposure"] = "calculated dynamically"
summary["status"] = "Friday shortlist complete"

os.makedirs("logs", exist_ok=True)
with open("logs/friday_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("📤 Friday shortlist generation complete — reports ready.")
