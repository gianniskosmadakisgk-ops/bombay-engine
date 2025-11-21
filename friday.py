import datetime
import json

def runFridayShortlist():
    # Προσωρινά δεδομένα για δοκιμή – αντικαθίστανται με κανονικά μετά
    shortlist_data = {
        "Draw Engine": [
            {"Match": "Team A vs Team B", "Odds": 3.25, "Confidence": "High"},
            {"Match": "Team C vs Team D", "Odds": 3.10, "Confidence": "Medium"}
        ],
        "Over/Under Engine": [
            {"Match": "Team E vs Team F", "Line": "O2.5", "Value": "+0.42"},
            {"Match": "Team G vs Team H", "Line": "U2.5", "Value": "+0.31"}
        ],
        "Bankroll Allocation": {
            "Draws": "40%",
            "Over/Under": "30%",
            "Value Picks": "30%"
        },
        "Generated": datetime.datetime.utcnow().isoformat()
    }

    # Αποθηκεύεται και σε logs αν χρειαστεί
    with open("logs/friday_shortlist.json", "w") as f:
        json.dump(shortlist_data, f, indent=2)

    return shortlist_data
