import os
import json
from datetime import datetime
from collections import defaultdict

# ======================================================
#  TUESDAY RECAP v2  (Giannis Edition)
#
#  - Διαβάζει ιστορικό στοιχημάτων από:
#       logs/bets_history_v2.json
#  - Υπολογίζει για την ΤΕΛΕΥΤΑΙΑ εβδομάδα (max week):
#       * πόσα παίχτηκαν / πόσα επιβεβαιώθηκαν
#       * ποντάρισμα, επιστροφές, PnL, ROI
#       * ΝΕΟ ΤΑΜΕΙΟ μετά την εβδομάδα
#  - Υπολογίζει και ΣΥΝΟΛΙΚΑ από Week 1 μέχρι σήμερα:
#       * αντίστοιχα metrics + τρέχον ταμείο
#  - Σώζει report σε:
#       logs/tuesday_recap_v2.json
#
#  ΔΟΜΗ INPUT (logs/bets_history_v2.json):
#  [
#    {
#      "week": 1,
#      "wallet": "Draw Singles",      # ή "Over Singles", "FanBet Draw",
#                                     #    "FunBet Over", "Kelly"
#      "bet_id": "W1-DR-001",
#      "stake": 15.0,
#      "payout": 0.0                  # 0 αν χάθηκε, αλλιώς ποσό επιστροφής
#    },
#    ...
#  ]
# ======================================================

HISTORY_PATH = "logs/bets_history_v2.json"
OUTPUT_PATH = "logs/tuesday_recap_v2.json"

os.makedirs("logs", exist_ok=True)

# ΑΡΧΙΚΑ ΠΟΡΤΟΦΟΛΙΑ (όπως συμφωνήσαμε)
INITIAL_WALLETS = {
    "Draw Singles": 400.0,
    "Over Singles": 300.0,
    "FanBet Draw": 100.0,
    "FunBet Over": 100.0,
    "Kelly": 300.0,
}


def log(msg: str):
    print(msg, flush=True)


# ------------------------------------------------------
# Φόρτωμα ιστορικού
# ------------------------------------------------------
def load_history():
    if not os.path.exists(HISTORY_PATH):
        log(f"⚠️ No history file found at {HISTORY_PATH}. Creating empty recap.")
        return []

    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            log("⚠️ History file is not a list. Using empty list.")
            return []
        log(f"📚 Loaded {len(data)} bets from history.")
        return data
    except Exception as e:
        log(f"❌ Error loading history: {e}")
        return []


# ------------------------------------------------------
# Aggregation helpers
# ------------------------------------------------------
def aggregate_by_wallet_and_week(bets):
    """
    Επιστρέφει:
    - weeks: sorted list με όλα τα weeks που βρέθηκαν
    - by_wallet_week[(wallet, week)] = dict με totals
    """
    by_wallet_week = defaultdict(lambda: {
        "bets": 0,
        "wins": 0,
        "stake": 0.0,
        "payout": 0.0,
    })
    weeks = set()

    for b in bets:
        try:
            week = int(b.get("week", 0) or 0)
        except Exception:
            continue
        wallet = b.get("wallet", "").strip() or "Unknown"
        stake = float(b.get("stake", 0.0) or 0.0)
        payout = float(b.get("payout", 0.0) or 0.0)

        key = (wallet, week)
        weeks.add(week)

        agg = by_wallet_week[key]
        agg["bets"] += 1
        agg["stake"] += stake
        agg["payout"] += payout
        if payout > 0:
            agg["wins"] += 1

    weeks = sorted(w for w in weeks if w > 0)
    return weeks, by_wallet_week


def compute_pnl_roi(stake, payout):
    pnl = payout - stake
    roi = 0.0
    if stake > 0:
        roi = pnl / stake
    return pnl, roi


# ------------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------------
def main():
    log("🎯 Running Tuesday Recap (v2)...")

    bets = load_history()
    if not bets:
        # Σώζουμε άδειο αλλά έγκυρο report
        empty = {
            "generated_at": datetime.utcnow().isoformat(),
            "current_week": None,
            "wallets": {},
            "note": "No bets history found. Report is empty.",
        }
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(empty, f, ensure_ascii=False, indent=2)
        log(f"✅ Empty Tuesday recap saved: {OUTPUT_PATH}")
        return

    weeks, by_wallet_week = aggregate_by_wallet_and_week(bets)
    if not weeks:
        log("⚠️ No valid week numbers in history. Aborting recap.")
        empty = {
            "generated_at": datetime.utcnow().isoformat(),
            "current_week": None,
            "wallets": {},
            "note": "No valid week numbers in bets history.",
        }
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(empty, f, ensure_ascii=False, indent=2)
        log(f"✅ Tuesday recap saved (no valid weeks): {OUTPUT_PATH}")
        return

    current_week = max(weeks)
    log(f"📆 Current week detected from history: Week {current_week}")

    wallets_report = {}

    for wallet_name, initial_amount in INITIAL_WALLETS.items():
        # --- Overall totals μέχρι current_week ---
        total_bets = 0
        total_wins = 0
        total_stake = 0.0
        total_payout = 0.0

        for w in weeks:
            if w > current_week:
                continue
            agg = by_wallet_week.get((wallet_name, w))
            if not agg:
                continue
            total_bets += agg["bets"]
            total_wins += agg["wins"]
            total_stake += agg["stake"]
            total_payout += agg["payout"]

        total_pnl, total_roi = compute_pnl_roi(total_stake, total_payout)
        current_bankroll = initial_amount + total_pnl

        # --- Αυτό το week μόνο ---
        week_agg = by_wallet_week.get((wallet_name, current_week), None)
        if week_agg:
            week_bets = week_agg["bets"]
            week_wins = week_agg["wins"]
            week_stake = week_agg["stake"]
            week_payout = week_agg["payout"]
        else:
            week_bets = week_wins = 0
            week_stake = week_payout = 0.0

        week_pnl, week_roi = compute_pnl_roi(week_stake, week_payout)

        # Ταμείο πριν την εβδομάδα = initial + PnL μέχρι προηγούμενο week
        prev_pnl = total_pnl - week_pnl
        bankroll_before_week = initial_amount + prev_pnl
        bankroll_after_week = bankroll_before_week + week_pnl  # ίσο με current_bankroll

        wallets_report[wallet_name] = {
            "initial_wallet": round(initial_amount, 2),
            "this_week": {
                "week": current_week,
                "bets": week_bets,
                "wins": week_wins,
                "stake": round(week_stake, 2),
                "payout": round(week_payout, 2),
                "pnl": round(week_pnl, 2),
                "roi": f"{week_roi*100:.1f}%",
                "wallet_before": round(bankroll_before_week, 2),
                "new_wallet": round(bankroll_after_week, 2),
            },
            "overall": {
                "weeks_covered": weeks,
                "bets": total_bets,
                "wins": total_wins,
                "stake": round(total_stake, 2),
                "payout": round(total_payout, 2),
                "pnl": round(total_pnl, 2),
                "roi": f"{total_roi*100:.1f}%",
                "current_wallet": round(current_bankroll, 2),
            },
        }

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "current_week": current_week,
        "wallets": wallets_report,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    log(f"✅ Tuesday recap v2 saved: {OUTPUT_PATH}")

    # Μικρή σύνοψη στα logs
    log("📌 Summary per wallet (this week):")
    for wname, wdata in wallets_report.items():
        tw = wdata["this_week"]
        log(
            f"- {wname}: Week {tw['week']} "
            f"bets={tw['bets']}, wins={tw['wins']}, "
            f"pnl={tw['pnl']}€, new_wallet={tw['new_wallet']}€"
        )


if __name__ == "__main__":
    main()
