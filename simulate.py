import random
import datetime

# Δημιουργία ψεύτικων αποδόσεων fair και bookmaker
def get_match_data():
    matches = []
    for i in range(1, 40):
        fair_odds_1 = round(random.uniform(1.8, 3.5), 2)
        fair_odds_x = round(random.uniform(2.8, 4.0), 2)
        fair_odds_2 = round(random.uniform(2.0, 4.5), 2)

        # Bookmaker δίνει λίγο διαφορετικές αποδόσεις
        book_odds_1 = round(fair_odds_1 * random.uniform(0.9, 1.15), 2)
        book_odds_x = round(fair_odds_x * random.uniform(0.9, 1.15), 2)
        book_odds_2 = round(fair_odds_2 * random.uniform(0.9, 1.15), 2)

        matches.append({
            "match": f"Match {i}",
            "fair_odds": {"1": fair_odds_1, "X": fair_odds_x, "2": fair_odds_2},
            "book_odds": {"1": book_odds_1, "X": book_odds_x, "2": book_odds_2},
        })
    return matches


# Υπολογισμός value bets με Kelly criterion
def calculate_value_matches(matches):
    bankroll = 300
    kelly_fraction = 0.5
    min_edge = 0.10

    value_bets = []

    for m in matches:
        for outcome in ["1", "X", "2"]:
            fair = 1 / m["fair_odds"][outcome]
            book = 1 / m["book_odds"][outcome]
            edge = (fair / book) - 1

            if edge >= min_edge:
                stake = round(bankroll * kelly_fraction * edge, 2)
                value_bets.append({
                    "match": m["match"],
                    "outcome": outcome,
                    "fair_odds": m["fair_odds"][outcome],
                    "book_odds": m["book_odds"][outcome],
                    "edge": round(edge * 100, 1),
                    "stake": stake
                })
    return value_bets


def simulate():
    matches = get_match_data()
    bets = calculate_value_matches(matches)
    return {
        "date": str(datetime.date.today()),
        "fund": 300,
        "method": "Half-Kelly",
        "min_edge": "10%",
        "status": "Thursday Simulation full run complete",
        "top_value_bets": bets[:5]
    }


def simulate_friday():
    matches = get_match_data()
    bets = calculate_value_matches(matches)
    return {
        "date": str(datetime.date.today()),
        "fund": 300,
        "method": "Half-Kelly",
        "min_edge": "10%",
        "status": "Friday shortlist ready",
        "top_value_bets": bets[:10]
    }
