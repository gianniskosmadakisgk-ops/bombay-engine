import os
import json
from datetime import datetime

# ==========================================
#  TUESDAY RECAP v2  — Giannis Edition
#
#  - Διαβάζει bets history από:
#       logs/bets_history_v2.json
#    (το αρχείο που γεμίζει το friday_shortlist_v2.py)
#
#  - Βγάζει:
#       * Ανα εβδομάδα: πόσα picks / πόσα stake
#         για Draw, Over, FunBet Draw, FunBet Over, Kelly
#       * Συνολικά (lifetime) αριθμούς και stakes
#
#  - Προς το παρόν ΔΕΝ έχει πραγματικά αποτελέσματα
#    (won/lost/ROI = 0.0 placeholder) μέχρι να
#    φτιάξουμε settlement pipeline.
#
#  - Σώζει report:
#       logs/tuesday_recap_v2.json
# ==========================================

HISTORY_PATH = "logs/bets_history_v2.json"
REPORT_PATH = "logs/tuesday_recap_v2.json"

os.makedirs("logs", exist_ok=True)


def log(msg: str):
    print(msg, flush=True)


# ------------------------------------------------------
# Helpers για ασφαλές διάβασμα αριθμών
# ------------------------------------------------------
def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def load_history():
    """Φορτώνει το bets history. Αν δεν υπάρχει, γυρίζει []."""
    if not os.path.exists(HISTORY_PATH):
        log(f"⚠️ No history file found at {HISTORY_PATH}. Creating empty recap.")
        return []

    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            log(f"ℹ️ Loaded {len(data)} weeks from {HISTORY_PATH}")
            return data
        else:
            log(f"⚠️ History file is not a list, ignoring malformed content.")
            return []
    except Exception as e:
        log(f"⚠️ Failed to load history {HISTORY_PATH}: {e}")
        return []


# ------------------------------------------------------
#  Weekly + Lifetime stats
# ------------------------------------------------------
def compute_weekly_stats(snapshot: dict):
    """
    Παίρνει ένα εβδομαδιαίο snapshot από το Friday Shortlist history
    και βγάζει συνοπτικά νούμερα για κάθε engine.
    """
    week_id = snapshot.get("week") or snapshot.get("week_id") or "unknown"

    draw_list = snapshot.get("draw_singles", []) or []
    over_list = snapshot.get("over_singles", []) or []
    funbet_draw = snapshot.get("funbet_draw", {}) or {}
    funbet_over = snapshot.get("funbet_over", {}) or {}
    kelly_block = snapshot.get("kelly", {}) or {}
    kelly_list = kelly_block.get("picks", []) or []

    # --- Draw Singles ---
    draw_played = len(draw_list)
    draw_stake = sum(safe_float(p.get("stake", 0)) for p in draw_list)

    # --- Over Singles ---
    over_played = len(over_list)
    over_stake = sum(safe_float(p.get("stake", 0)) for p in over_list)

    # --- FunBet Draw ---
    fdraw_cols = int(funbet_draw.get("columns", 0) or 0)
    fdraw_stake = safe_float(funbet_draw.get("total_stake", 0))

    # --- FunBet Over ---
    fover_cols = int(funbet_over.get("columns", 0) or 0)
    fover_stake = safe_float(funbet_over.get("total_stake", 0))

    # --- Kelly ---
    kelly_played = len(kelly_list)
    kelly_stake = sum(safe_float(p.get("stake (€)", 0)) for p in kelly_list)

    # Προς το παρόν δεν έχουμε αποτελέσματα → όλα 0
    zero = {
        "won": 0,
        "lost": 0,
        "roi": 0.0,
        "profit": 0.0,
    }

    return {
        "week": week_id,
        "draw_engine": {
            "played": draw_played,
            "stake": draw_stake,
            **zero,
        },
        "over_engine": {
            "played": over_played,
            "stake": over_stake,
            **zero,
        },
        "funbet_draw": {
            "columns": fdraw_cols,
            "stake": fdraw_stake,
            **zero,
        },
        "funbet_over": {
            "columns": fover_cols,
            "stake": fover_stake,
            **zero,
        },
        "kelly": {
            "played": kelly_played,
            "stake": kelly_stake,
            **zero,
        },
    }


def aggregate_lifetime(weekly_stats: list):
    """
    Μαζεύει τα weekly stats και βγάζει lifetime σύνοψη.
    Προσοχή: ROI/profit ακόμα 0 μέχρι να μπουν αποτελέσματα.
    """
    lifetime = {
        "draw_engine": {"played": 0, "stake": 0.0, "won": 0, "lost": 0, "profit": 0.0, "roi": 0.0},
        "over_engine": {"played": 0, "stake": 0.0, "won": 0, "lost": 0, "profit": 0.0, "roi": 0.0},
        "funbet_draw": {"columns": 0, "stake": 0.0, "profit": 0.0, "roi": 0.0},
        "funbet_over": {"columns": 0, "stake": 0.0, "profit": 0.0, "roi": 0.0},
        "kelly": {"played": 0, "stake": 0.0, "won": 0, "lost": 0, "profit": 0.0, "roi": 0.0},
    }

    for w in weekly_stats:
        de = w["draw_engine"]
        oe = w["over_engine"]
        fd = w["funbet_draw"]
        fo = w["funbet_over"]
        ke = w["kelly"]

        lifetime["draw_engine"]["played"] += de["played"]
        lifetime["draw_engine"]["stake"] += de["stake"]

        lifetime["over_engine"]["played"] += oe["played"]
        lifetime["over_engine"]["stake"] += oe["stake"]

        lifetime["funbet_draw"]["columns"] += fd["columns"]
        lifetime["funbet_draw"]["stake"] += fd["stake"]

        lifetime["funbet_over"]["columns"] += fo["columns"]
        lifetime["funbet_over"]["stake"] += fo["stake"]

        lifetime["kelly"]["played"] += ke["played"]
        lifetime["kelly"]["stake"] += ke["stake"]

    # ROI/profit μένουν 0.0 μέχρι να έχουμε πραγματικά αποτελέσματα
    return lifetime


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    log("🎯 Running Tuesday Recap (v2)...")

    history = load_history()
    if not history:
        recap = {
            "generated_at": datetime.utcnow().isoformat(),
            "weeks": [],
            "lifetime": {
                "message": "No betting history yet. Recap is empty.",
            },
        }
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(recap, f, ensure_ascii=False, indent=2)
        log(f"✅ Empty Tuesday recap saved: {REPORT_PATH}")
        return

    weekly_stats = []
    for snap in history:
        try:
            week_stats = compute_weekly_stats(snap)
            weekly_stats.append(week_stats)
        except Exception as e:
            log(f"⚠️ Failed to compute weekly stats for snapshot: {e}")

    lifetime = aggregate_lifetime(weekly_stats)

    recap = {
        "generated_at": datetime.utcnow().isoformat(),
        "weeks": weekly_stats,
        "lifetime": lifetime,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(recap, f, ensure_ascii=False, indent=2)

    log(f"✅ Tuesday recap saved: {REPORT_PATH}")
    log(f"Summary → weeks: {len(weekly_stats)}")


if __name__ == "__main__":
    main()
