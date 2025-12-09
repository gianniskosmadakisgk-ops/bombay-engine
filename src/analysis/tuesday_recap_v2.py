import os
import json
from datetime import datetime

# ============================================================
#  TUESDAY RECAP v2
#
#  - Διαβάζει τα εβδομαδιαία αποτελέσματα από
#       logs/tuesday_results_input_v2.json
#  - Αν υπάρχει προηγούμενο recap (logs/tuesday_recap_v2.json),
#       συνεχίζει τα totals.
#  - Βγάζει ανά wallet:
#       bank_start (αρχή εβδομάδας)
#       bank_end (τέλος εβδομάδας)
#       week_picks / week_hits
#       total_picks / total_hits
#       pnl_week / pnl_total
#       stake_week / stake_total
#       roi_week / roi_total
#  - Φτιάχνει και total_system γραμμή.
#  - Γράφει το τελικό report σε logs/tuesday_recap_v2.json
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(BASE_DIR, "logs")

INPUT_PATH = os.path.join(LOGS_DIR, "tuesday_results_input_v2.json")
RECAP_PATH = os.path.join(LOGS_DIR, "tuesday_recap_v2.json")

# Σταθερά αρχικά bankrolls (units)
INITIAL_BANKROLLS = {
    "draw": 1000.0,
    "over": 1000.0,
    "fun_draw": 300.0,
    "fun_over": 300.0,
    "kelly": 600.0,
}

WALLET_ORDER = ["draw", "over", "fun_draw", "fun_over", "kelly"]


def log(msg: str):
    print(msg, flush=True)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_div(num, den):
    if den == 0:
        return 0.0
    return num / den


def main():
    log("🚀 Running Tuesday Recap v2")

    os.makedirs(LOGS_DIR, exist_ok=True)

    # --------------------------------------------------------
    # 1. Φόρτωση weekly input
    # --------------------------------------------------------
    weekly_input = load_json(INPUT_PATH)
    if weekly_input is None:
        raise FileNotFoundError(
            f"Weekly input file not found: {INPUT_PATH}\n"
            f"Δημιούργησε ένα JSON με week_label και wallets."
        )

    week_label = weekly_input.get("week_label", "")
    weekly_wallets = weekly_input.get("wallets", {})

    # --------------------------------------------------------
    # 2. Φόρτωση προηγούμενου recap (αν υπάρχει)
    # --------------------------------------------------------
    prev_recap = load_json(RECAP_PATH)

    if prev_recap is None:
        week_index = 1
        prev_wallet_state = {}
        log("No previous recap found – starting Week 1.")
    else:
        week_index = int(prev_recap.get("week_index", 0)) + 1
        prev_wallet_state = prev_recap.get("wallets", {})
        log(f"Previous recap found – continuing to Week {week_index}.")

    # --------------------------------------------------------
    # 3. Υπολογισμός ανά wallet
    # --------------------------------------------------------
    wallets_out = {}

    # Αθροιστικά για total_system
    total_bank_start = 0.0
    total_bank_end = 0.0
    total_stake_week = 0.0
    total_stake_total = 0.0
    total_pnl_week = 0.0
    total_pnl_total = 0.0
    total_picks_week = 0
    total_hits_week = 0
    total_picks_total = 0
    total_hits_total = 0

    for wallet in WALLET_ORDER:
        init_bank = INITIAL_BANKROLLS[wallet]

        prev = prev_wallet_state.get(wallet, {})
        prev_bank_end = float(prev.get("bank_end", init_bank))
        prev_total_stake = float(prev.get("stake_total", 0.0))
        prev_total_pnl = float(prev.get("pnl_total", 0.0))
        prev_total_picks = int(prev.get("total_picks", 0))
        prev_total_hits = int(prev.get("total_hits", 0))

        w_week = weekly_wallets.get(wallet, {})
        picks_week = int(w_week.get("picks", 0))
        hits_week = int(w_week.get("hits", 0))
        stake_week = float(w_week.get("stake_week", 0.0))
        pnl_week = float(w_week.get("pnl_week", 0.0))

        bank_start = prev_bank_end  # αρχή εβδομάδας
        bank_end = bank_start + pnl_week

        total_picks = prev_total_picks + picks_week
        total_hits = prev_total_hits + hits_week
        stake_total = prev_total_stake + stake_week
        pnl_total = prev_total_pnl + pnl_week

        roi_week = safe_div(pnl_week, stake_week)
        roi_total = safe_div(pnl_total, stake_total)

        week_ph = f"{hits_week}/{picks_week}" if picks_week > 0 else "0/0"
        total_ph = f"{total_hits}/{total_picks}" if total_picks > 0 else "0/0"

        wallets_out[wallet] = {
            "bank_initial": init_bank,            # σταθερό reference
            "bank_start": round(bank_start, 2),  # αρχή εβδομάδας
            "bank_end": round(bank_end, 2),      # τέλος εβδομάδας

            "week_picks": picks_week,
            "week_hits": hits_week,
            "week_ph": week_ph,

            "total_picks": total_picks,
            "total_hits": total_hits,
            "total_ph": total_ph,

            "stake_week": round(stake_week, 2),
            "stake_total": round(stake_total, 2),

            "pnl_week": round(pnl_week, 2),
            "pnl_total": round(pnl_total, 2),

            "roi_week": round(roi_week, 4),      # π.χ. 0.185 = 18.5%
            "roi_total": round(roi_total, 4),
        }

        # Αθροίσματα για total system
        total_bank_start += bank_start
        total_bank_end += bank_end
        total_stake_week += stake_week
        total_stake_total += stake_total
        total_pnl_week += pnl_week
        total_pnl_total += pnl_total
        total_picks_week += picks_week
        total_hits_week += hits_week
        total_picks_total += total_picks
        total_hits_total += total_hits

    total_roi_week = safe_div(total_pnl_week, total_stake_week)
    total_roi_total = safe_div(total_pnl_total, total_stake_total)

    total_system = {
        "bank_initial": sum(INITIAL_BANKROLLS.values()),
        "bank_start": round(total_bank_start, 2),
        "bank_end": round(total_bank_end, 2),

        "week_picks": total_picks_week,
        "week_hits": total_hits_week,
        "week_ph": f"{total_hits_week}/{total_picks_week}" if total_picks_week > 0 else "0/0",

        "total_picks": total_picks_total,
        "total_hits": total_hits_total,
        "total_ph": f"{total_hits_total}/{total_picks_total}" if total_picks_total > 0 else "0/0",

        "stake_week": round(total_stake_week, 2),
        "stake_total": round(total_stake_total, 2),

        "pnl_week": round(total_pnl_week, 2),
        "pnl_total": round(total_pnl_total, 2),

        "roi_week": round(total_roi_week, 4),
        "roi_total": round(total_roi_total, 4),
    }

    recap = {
        "version": 2,
        "generated_at": datetime.utcnow().isoformat(),
        "week_index": week_index,
        "week_label": week_label,
        "wallets": wallets_out,
        "total_system": total_system,
    }

    with open(RECAP_PATH, "w", encoding="utf-8") as f:
        json.dump(recap, f, ensure_ascii=False, indent=2)

    log(f"✅ Tuesday recap saved → {RECAP_PATH}")


if __name__ == "__main__":
    main()
