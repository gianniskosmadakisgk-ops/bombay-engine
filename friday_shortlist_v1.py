import json
import os
from datetime import datetime

# === SETTINGS ===
REPORT_PATH = "logs/friday_shortlist_v1.json"
THURSDAY_REPORT = "logs/thursday_report_v1.json"

DRAW_WALLET = 400
OVER_WALLET = 300
FANBET_DRAW_WALLET = 100
FANBET_OVER_WALLET = 100
KELLY_WALLET = 300

# === THRESHOLDS ===
DRAW_MIN_SCORE = 7.5
DRAW_MIN_ODDS = 2.70
OVER_MIN_SCORE = 7.5
OVER_MIN_FAIR = 1.70
KELLY_VALUE_THRESHOLD = 0.15   # +15%
KELLY_FRACTION = 0.40

# === LOAD PREVIOUS REPORT ===
def load_thursday_data():
    if not os.path.exists(THURSDAY_REPORT):
        return []
    with open(THURSDAY_REPORT, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("fixtures", [])

# === FLAT STAKE RULES ===
def flat_stake(confidence):
    if confidence >= 8.0:
        return 20
    elif confidence >= 7.5:
        return 15
    return 0

# === KELLY STAKE CALC ===
def kelly_stake(bankroll, fair, offered, prob):
    b = offered - 1
    q = 1 - prob
    kelly_fraction = ((b * prob - q) / b)
    if kelly_fraction < 0:
        return 0
    stake_fraction = kelly_fraction * KELLY_FRACTION
    return round(bankroll * stake_fraction, 2)

# === ENGINE FILTERS ===
def generate_shortlist(fixtures):
    draw_picks = []
    over_picks = []
    kelly_picks = []

    for f in fixtures:
        league = f.get("league", "")
        match = f.get("match", "")
        score_draw = f.get("score_draw", 0)
        score_over = f.get("score_over", 0)
        fair_x = f.get("fair_x", 0)
        fair_over = f.get("fair_over", 0)
        odds_x = f.get("odds_x", 0)
        odds_over = f.get("odds_over", 0)

        diff_x = ((odds_x - fair_x) / fair_x) if fair_x else 0
        diff_over = ((odds_over - fair_over) / fair_over) if fair_over else 0

        # === DRAW ENGINE ===
        if odds_x >= DRAW_MIN_ODDS and score_draw >= DRAW_MIN_SCORE:
            stake = flat_stake(score_draw)
            if stake > 0:
                draw_picks.append({
                    "match": match,
                    "league": league,
                    "odds": odds_x,
                    "fair": fair_x,
                    "diff": f"{diff_x:+.0%}",
                    "score": round(score_draw, 1),
                    "stake": f"{stake}€",
                    "wallet": "Draw"
                })

        # === OVER ENGINE ===
        if fair_over >= OVER_MIN_FAIR and score_over >= OVER_MIN_SCORE:
            stake = flat_stake(score_over)
            if stake > 0:
                over_picks.append({
                    "match": match,
                    "league": league,
                    "odds": odds_over,
                    "fair": fair_over,
                    "diff": f"{diff_over:+.0%}",
                    "score": round(score_over, 1),
                    "stake": f"{stake}€",
                    "wallet": "Over"
                })

        # === FRACTION KELLY (Top 10) ===
        if diff_x >= KELLY_VALUE_THRESHOLD:
            prob = 1 / fair_x if fair_x > 0 else 0
            stake = kelly_stake(KELLY_WALLET, fair_x, odds_x, prob)
            if stake > 0:
                kelly_picks.append({
                    "match": match,
                    "market": "Draw",
                    "fair": fair_x,
                    "offered": odds_x,
                    "diff": f"{diff_x:+.0%}",
                    "kelly%": f"{KELLY_FRACTION*100:.0f}%",
                    "stake (€)": stake
                })

        if diff_over >= KELLY_VALUE_THRESHOLD:
            prob = 1 / fair_over if fair_over > 0 else 0
            stake = kelly_stake(KELLY_WALLET, fair_over, odds_over, prob)
            if stake > 0:
                kelly_picks.append({
                    "match": match,
                    "market": "Over",
                    "fair": fair_over,
                    "offered": odds_over,
                    "diff": f"{diff_over:+.0%}",
                    "kelly%": f"{KELLY_FRACTION*100:.0f}%",
                    "stake (€)": stake
                })

    # Limit Top 10 for Kelly
    kelly_picks = sorted(kelly_picks, key=lambda x: float(x["diff"].replace("%", "")), reverse=True)[:10]

    return draw_picks, over_picks, kelly_picks

# === BANKROLL SUMMARY ===
def bankroll_summary(draw_picks, over_picks, kelly_picks):
    draw_spent = sum([int(p["stake"].replace("€", "")) for p in draw_picks])
    over_spent = sum([int(p["stake"].replace("€", "")) for p in over_picks])
    kelly_spent = sum([p["stake (€)"] for p in kelly_picks])

    summary = [
        {"Wallet": "Draw Engine", "Before": f"{DRAW_WALLET}€", "After": f"{DRAW_WALLET - draw_spent}€", "Open Bets": f"{draw_spent}€"},
        {"Wallet": "Over Engine", "Before": f"{OVER_WALLET}€", "After": f"{OVER_WALLET - over_spent}€", "Open Bets": f"{over_spent}€"},
        {"Wallet": "FanBet Draws", "Before": f"{FANBET_DRAW_WALLET}€", "After": "≈52€", "Open Bets": "≈48€"},
        {"Wallet": "FanBet Overs", "Before": f"{FANBET_OVER_WALLET}€", "After": "≈70€", "Open Bets": "≈30€"},
        {"Wallet": "Fraction Kelly", "Before": f"{KELLY_WALLET}€", "After": f"{KELLY_WALLET - kelly_spent:.2f}€", "Open Bets": f"{kelly_spent:.2f}€"},
    ]
    return summary

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("🎯 Running Friday Shortlist (v1)...")

    fixtures = load_thursday_data()
    draw_picks, over_picks, kelly_picks = generate_shortlist(fixtures)
    banks = bankroll_summary(draw_picks, over_picks, kelly_picks)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "draw_engine": draw_picks,
        "over_engine": over_picks,
        "fraction_kelly": {"picks": kelly_picks},
        "bankroll_status": banks
    }

    os.makedirs("logs", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ Friday shortlist report saved: {REPORT_PATH}")
    print(f"🎯 Draw picks: {len(draw_picks)}, Over picks: {len(over_picks)}, Kelly picks: {len(kelly_picks)}")import json
import os
from datetime import datetime

# === SETTINGS ===
REPORT_PATH = "logs/friday_shortlist_v1.json"
THURSDAY_REPORT = "logs/thursday_report_v1.json"

DRAW_WALLET = 400
OVER_WALLET = 300
FANBET_DRAW_WALLET = 100
FANBET_OVER_WALLET = 100
KELLY_WALLET = 300

# === THRESHOLDS ===
DRAW_MIN_SCORE = 7.5
DRAW_MIN_ODDS = 2.70
OVER_MIN_SCORE = 7.5
OVER_MIN_FAIR = 1.70
KELLY_VALUE_THRESHOLD = 0.15   # +15%
KELLY_FRACTION = 0.40

# === LOAD PREVIOUS REPORT ===
def load_thursday_data():
    if not os.path.exists(THURSDAY_REPORT):
        return []
    with open(THURSDAY_REPORT, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("fixtures", [])

# === FLAT STAKE RULES ===
def flat_stake(confidence):
    if confidence >= 8.0:
        return 20
    elif confidence >= 7.5:
        return 15
    return 0

# === KELLY STAKE CALC ===
def kelly_stake(bankroll, fair, offered, prob):
    b = offered - 1
    q = 1 - prob
    kelly_fraction = ((b * prob - q) / b)
    if kelly_fraction < 0:
        return 0
    stake_fraction = kelly_fraction * KELLY_FRACTION
    return round(bankroll * stake_fraction, 2)

# === ENGINE FILTERS ===
def generate_shortlist(fixtures):
    draw_picks = []
    over_picks = []
    kelly_picks = []

    for f in fixtures:
        league = f.get("league", "")
        match = f.get("match", "")
        score_draw = f.get("score_draw", 0)
        score_over = f.get("score_over", 0)
        fair_x = f.get("fair_x", 0)
        fair_over = f.get("fair_over", 0)
        odds_x = f.get("odds_x", 0)
        odds_over = f.get("odds_over", 0)

        diff_x = ((odds_x - fair_x) / fair_x) if fair_x else 0
        diff_over = ((odds_over - fair_over) / fair_over) if fair_over else 0

        # === DRAW ENGINE ===
        if odds_x >= DRAW_MIN_ODDS and score_draw >= DRAW_MIN_SCORE:
            stake = flat_stake(score_draw)
            if stake > 0:
                draw_picks.append({
                    "match": match,
                    "league": league,
                    "odds": odds_x,
                    "fair": fair_x,
                    "diff": f"{diff_x:+.0%}",
                    "score": round(score_draw, 1),
                    "stake": f"{stake}€",
                    "wallet": "Draw"
                })

        # === OVER ENGINE ===
        if fair_over >= OVER_MIN_FAIR and score_over >= OVER_MIN_SCORE:
            stake = flat_stake(score_over)
            if stake > 0:
                over_picks.append({
                    "match": match,
                    "league": league,
                    "odds": odds_over,
                    "fair": fair_over,
                    "diff": f"{diff_over:+.0%}",
                    "score": round(score_over, 1),
                    "stake": f"{stake}€",
                    "wallet": "Over"
                })

        # === FRACTION KELLY (Top 10) ===
        if diff_x >= KELLY_VALUE_THRESHOLD:
            prob = 1 / fair_x if fair_x > 0 else 0
            stake = kelly_stake(KELLY_WALLET, fair_x, odds_x, prob)
            if stake > 0:
                kelly_picks.append({
                    "match": match,
                    "market": "Draw",
                    "fair": fair_x,
                    "offered": odds_x,
                    "diff": f"{diff_x:+.0%}",
                    "kelly%": f"{KELLY_FRACTION*100:.0f}%",
                    "stake (€)": stake
                })

        if diff_over >= KELLY_VALUE_THRESHOLD:
            prob = 1 / fair_over if fair_over > 0 else 0
            stake = kelly_stake(KELLY_WALLET, fair_over, odds_over, prob)
            if stake > 0:
                kelly_picks.append({
                    "match": match,
                    "market": "Over",
                    "fair": fair_over,
                    "offered": odds_over,
                    "diff": f"{diff_over:+.0%}",
                    "kelly%": f"{KELLY_FRACTION*100:.0f}%",
                    "stake (€)": stake
                })

    # Limit Top 10 for Kelly
    kelly_picks = sorted(kelly_picks, key=lambda x: float(x["diff"].replace("%", "")), reverse=True)[:10]

    return draw_picks, over_picks, kelly_picks

# === BANKROLL SUMMARY ===
def bankroll_summary(draw_picks, over_picks, kelly_picks):
    draw_spent = sum([int(p["stake"].replace("€", "")) for p in draw_picks])
    over_spent = sum([int(p["stake"].replace("€", "")) for p in over_picks])
    kelly_spent = sum([p["stake (€)"] for p in kelly_picks])

    summary = [
        {"Wallet": "Draw Engine", "Before": f"{DRAW_WALLET}€", "After": f"{DRAW_WALLET - draw_spent}€", "Open Bets": f"{draw_spent}€"},
        {"Wallet": "Over Engine", "Before": f"{OVER_WALLET}€", "After": f"{OVER_WALLET - over_spent}€", "Open Bets": f"{over_spent}€"},
        {"Wallet": "FanBet Draws", "Before": f"{FANBET_DRAW_WALLET}€", "After": "≈52€", "Open Bets": "≈48€"},
        {"Wallet": "FanBet Overs", "Before": f"{FANBET_OVER_WALLET}€", "After": "≈70€", "Open Bets": "≈30€"},
        {"Wallet": "Fraction Kelly", "Before": f"{KELLY_WALLET}€", "After": f"{KELLY_WALLET - kelly_spent:.2f}€", "Open Bets": f"{kelly_spent:.2f}€"},
    ]
    return summary

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("🎯 Running Friday Shortlist (v1)...")

    fixtures = load_thursday_data()
    draw_picks, over_picks, kelly_picks = generate_shortlist(fixtures)
    banks = bankroll_summary(draw_picks, over_picks, kelly_picks)

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "draw_engine": draw_picks,
        "over_engine": over_picks,
        "fraction_kelly": {"picks": kelly_picks},
        "bankroll_status": banks
    }

    os.makedirs("logs", exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"✅ Friday shortlist report saved: {REPORT_PATH}")
    print(f"🎯 Draw picks: {len(draw_picks)}, Over picks: {len(over_picks)}, Kelly picks: {len(kelly_picks)}")
