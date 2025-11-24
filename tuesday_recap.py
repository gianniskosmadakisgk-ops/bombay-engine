import json
import requests
from datetime import datetime, timedelta

def run_tuesday_recap():
    today = datetime.now()
    friday = today - timedelta(days=3)
    monday = today - timedelta(days=1)

    # Dummy recap summary for example
    recap_data = {
        "status": "success",
        "from": friday.strftime("%Y-%m-%d"),
        "to": monday.strftime("%Y-%m-%d"),
        "summary": [
            {"category": "Draws", "played": 25, "won": 14, "success_rate": "56%"},
            {"category": "Over 2.5", "played": 30, "won": 18, "success_rate": "60%"},
            {"category": "Kelly Bank", "profit": "+12.4%", "active_bets": 8}
        ]
    }

    print(json.dumps(recap_data, indent=2))
    return recap_data

if __name__ == "__main__":
    run_tuesday_recap()
