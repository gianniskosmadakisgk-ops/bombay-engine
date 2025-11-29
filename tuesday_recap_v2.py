import os
import json
from datetime import datetime
from pathlib import Path

HISTORY_PATH = "logs/bets_history_v2.json"
RECAP_PATH = "logs/tuesday_recap_v2.json"

os.makedirs("logs", exist_ok=True)


# --------------------------------------------
# Load history
# --------------------------------------------
def load_history():
    if not os.path.exists(HISTORY_PATH):
        print("⚠️ No history found. Creating empty recap.")
        return []

    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------
# Helper: Compute stats for an engine
# --------------------------------------------
def compute_engine_stats(weeks_data, engine_name):
    played = 0
    won = 0
    profit = 0.0
    stake_total = 0.0

    for w in weeks_data:
        engine = w.get(engine_name, {})
        played += engine.get("played", 0)
        won += engine.get("won", 0)
        profit += engine.get("profit", 0.0)
        stake_total += engine.get("stake", 0.0)

    roi = (profit / stake_total * 100) if stake_total > 0 else 0.0

    return {
        "played": played,
        "won": won,
        "profit": round(profit, 2),
        "roi": f"{roi:.1f}%",
        "new_balance": round(1000 + profit, 2),  # placeholder balance logic
    }


# --------------------------------------------
# MAIN
# --------------------------------------------
def main():
    print("🎯 Running Tuesday Recap (v2)...")

    weeks = load_history()

    if not weeks:
        recap = {"message": "No weeks available.", "generated_at": datetime.utcnow().isoformat()}
        with open(RECAP_PATH, "w", encoding="utf-8") as f:
            json.dump(recap, f, ensure_ascii=False, indent=2)
        print(f"✅ Empty recap saved: {RECAP_PATH}")
        return

    # Latest week
    latest = weeks[-1]
    week_id = latest.get("week")

    # Engines
    engines = ["draw", "over", "funbet_draw", "funbet_over", "kelly"]

    latest_stats = {}
    lifetime_stats = {}

    for eng in engines:
        latest_stats[eng] = compute_engine_stats([latest], eng)
        lifetime_stats[eng] = compute_engine_stats(weeks, eng)

    recap = {
        "week": week_id,
        "generated_at": datetime.utcnow().isoformat(),
        "latest": latest_stats,
        "lifetime": lifetime_stats,
    }

    with open(RECAP_PATH, "w", encoding="utf-8") as f:
        json.dump(recap, f, ensure_ascii=False, indent=2)

    print(f"✅ Tuesday recap saved: {RECAP_PATH}")
    print(f"Week: {week_id} → Done.")


if __name__ == "__main__":
    main()
