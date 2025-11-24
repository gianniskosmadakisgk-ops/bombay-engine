import json
import numpy as np
from datetime import datetime

# -----------------------------
# Friday Shortlist Generator
# -----------------------------

THRESHOLD_SCORE = 7.5
KELLY_FRACTION = 0.40

def load_data():
    try:
        with open('logs/thursday_output.json', 'r') as f:
            data = json.load(f)
        return data.get('data', [])
    except FileNotFoundError:
        print("⚠️ Thursday analysis file not found.")
        return []

def calc_kelly(fair_odds, real_odds, win_prob):
    b = real_odds - 1
    q = 1 - win_prob
    k = ((b * win_prob - q) / b) * KELLY_FRACTION
    return max(0, k)

def shortlist_games(data):
    draw_picks, over_picks, kelly_picks = [], [], []

    for match in data:
        try:
            teams = match.get('teams', {})
            fair = match.get('fair_odds', {})
            real = match.get('real_odds', {})
            scores = match.get('scores', {})

            draw_score = scores.get('draw', 0)
            over_score = scores.get('over', 0)

            if draw_score >= THRESHOLD_SCORE:
                draw_picks.append({
                    "home": teams.get('home', {}).get('name', ''),
                    "away": teams.get('away', {}).get('name', ''),
                    "score": draw_score,
                    "league": match.get('league', {}).get('name', ''),
                    "date": match.get('fixture', {}).get('date', '')
                })

            if over_score >= THRESHOLD_SCORE:
                over_picks.append({
                    "home": teams.get('home', {}).get('name', ''),
                    "away": teams.get('away', {}).get('name', ''),
                    "score": over_score,
                    "league": match.get('league', {}).get('name', ''),
                    "date": match.get('fixture', {}).get('date', '')
                })

            for key in ['home', 'draw', 'away', 'over']:
                fair_odd = fair.get(key)
                real_odd = real.get(key)
                if fair_odd and real_odd and real_odd > fair_odd:
                    diff = (real_odd - fair_odd) / fair_odd * 100
                    win_prob = 1 / fair_odd
                    kelly = calc_kelly(fair_odd, real_odd, win_prob)
                    if kelly > 0:
                        kelly_picks.append({
                            "bet_type": key,
                            "home": teams.get('home', {}).get('name', ''),
                            "away": teams.get('away', {}).get('name', ''),
                            "diff_pct": round(diff, 2),
                            "kelly_stake": round(kelly, 4),
                            "league": match.get('league', {}).get('name', ''),
                            "date": match.get('fixture', {}).get('date', '')
                        })
        except Exception as e:
            print(f"⚠️ Error processing match: {e}")
            continue

    draw_picks = sorted(draw_picks, key=lambda x: x['score'], reverse=True)[:10]
    over_picks = sorted(over_picks, key=lambda x: x['score'], reverse=True)[:10]
    kelly_picks = sorted(kelly_picks, key=lambda x: x['diff_pct'], reverse=True)[:10]

    return {
        "timestamp": datetime.now().isoformat(),
        "draw_top10": draw_picks,
        "over_top10": over_picks,
        "kelly_top10": kelly_picks
    }

def save_shortlist(result):
    with open('logs/friday_shortlist.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("✅ Friday shortlist saved successfully!")

if __name__ == "__main__":
    data = load_data()
    if not data:
        print("⚠️ No Thursday data to process.")
    else:
        result = shortlist_games(data)
        save_shortlist(result)
