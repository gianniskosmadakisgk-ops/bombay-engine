import os
import json
from datetime import datetime

THURSDAY_REPORT = "logs/thursday_report_v1.json"
FRIDAY_REPORT   = "logs/friday_shortlist_v2.json"
TUESDAY_REPORT  = "logs/tuesday_recap_v2.json"

os.makedirs("logs", exist_ok=True)


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_results(fixtures):
    """
    Από το Thursday report παίρνουμε:
    - τα fair odds
    - τα goal lines
    - score_draw / score_over
    """
    results = []

    for fx in fixtures:
        results.append({
            "match": fx.get("match"),
            "league": fx.get("league"),
            "fair_1": fx.get("fair_1"),
            "fair_x": fx.get("fair_x"),
            "fair_2": fx.get("fair_2"),
            "fair_over": fx.get("fair_over"),
            "score_draw": fx.get("score_draw"),
            "score_over": fx.get("score_over"),
        })

    return results


def extract_friday_bets(friday):
    """
    Από το Friday shortlist παίρνουμε:
    - draw singles
    - over singles
    - kelly picks
    - funbet draw
    - funbet over
    - bankroll
    """
    if not friday:
        return {}

    return {
        "draw_singles": friday.get("draw_singles", []),
        "over_singles": friday.get("over_singles", []),
        "kelly": friday.get("kelly", {}).get("picks", []),
        "funbet_draw": friday.get("funbet_draw", {}),
        "funbet_over": friday.get("funbet_over", {}),
        "bankroll_status": friday.get("bankroll_status", []),
    }


def compute_summary(friday, results):
    """
    Γράφει συνοπτικά:
    - Πόσα ματς είχαμε
    - Πόσα picks παίξαμε
    - Πόσο value βρήκαμε
    - Τι bankroll καταναλώσαμε
    """
    summary = {}

    if friday:
        summary["draw_singles"] = len(friday.get("draw_singles", []))
        summary["over_singles"] = len(friday.get("over_singles", []))
        summary["kelly_picks"] = len(friday.get("kelly", {}).get("picks", []))

        fb_d = friday.get("funbet_draw", {})
        fb_o = friday.get("funbet_over", {})
        summary["funbet_draw_cols"] = fb_d.get("columns", 0)
        summary["funbet_over_cols"] = fb_o.get("columns", 0)

    summary["fixtures_total"] = len(results)

    return summary


def main():
    print("🟦 Running Tuesday Recap (v2)...")

    thursday = load_json(THURSDAY_REPORT)
    friday   = load_json(FRIDAY_REPORT)

    if not thursday:
        raise FileNotFoundError(f"Thursday report not found: {THURSDAY_REPORT}")

    results = extract_results(thursday.get("fixtures", []))
    friday_bets = extract_friday_bets(friday)
    summary = compute_summary(friday_bets, results)

    final_report = {
        "timestamp": datetime.utcnow().isoformat(),
        "fixtures_results": results,
        "friday_bets": friday_bets,
        "summary": summary,
    }

    with open(TUESDAY_REPORT, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    print(f"✅ Tuesday Recap saved: {TUESDAY_REPORT}")
    print("Summary:", summary)


if __name__ == "__main__":
    main()
