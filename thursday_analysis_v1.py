import os
import json
from datetime import datetime
from pathlib import Path

# ---------- ΡΥΘΜΙΣΕΙΣ ----------
THURSDAY_REPORT_PATH = "logs/thursday_report_v1.json"
FRIDAY_SHORTLIST_PATH = "logs/friday_shortlist_v1.json"

MIN_EDGE = 0.05          # 5% value (odds / fair - 1)
FLAT_STAKE = 10.0        # μονάδες ανά pick
# -------------------------------


def load_thursday_report():
    """
    Διαβάζει το logs/thursday_report_v1.json
    και επιστρέφει (meta, fixtures_list)
    """
    if not os.path.exists(THURSDAY_REPORT_PATH):
        raise FileNotFoundError(
            f"Thursday report not found: {THURSDAY_REPORT_PATH}"
        )

    with open(THURSDAY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixtures = data.get("fixtures", [])
    meta = data.get("meta", {})
    return meta, fixtures


def build_shortlist(fixtures):
    """
    Παίρνει τα fixtures από το Thursday και φτιάχνει
    δύο λίστες:
      - draw_picks
      - over_picks
    με βασικά πεδία και edge.
    """
    draw_picks = []
    over_picks = []

    for m in fixtures:
        league = m.get("league")
        match_label = m.get("match")

        fair_x = m.get("fair_x")
        odds_x = m.get("odds_x")

        fair_over = m.get("fair_over")
        odds_over = m.get("odds_over")

        # --- Draw edge ---
        edge_draw = None
        if fair_x and odds_x:
            try:
                edge_draw = (float(odds_x) / float(fair_x)) - 1.0
            except Exception:
                edge_draw = None

        if edge_draw is not None and edge_draw >= MIN_EDGE:
            draw_picks.append(
                {
                    "league": league,
                    "match": match_label,
                    "market": "DRAW",
                    "book_odds": odds_x,
                    "fair_odds": fair_x,
                    "edge": edge_draw,
                    "stake": FLAT_STAKE,
                }
            )

        # --- Over 2.5 edge ---
        edge_over = None
        if fair_over and odds_over:
            try:
                edge_over = (float(odds_over) / float(fair_over)) - 1.0
            except Exception:
                edge_over = None

        if edge_over is not None and edge_over >= MIN_EDGE:
            over_picks.append(
                {
                    "league": league,
                    "match": match_label,
                    "market": "OVER 2.5",
                    "book_odds": odds_over,
                    "fair_odds": fair_over,
                    "edge": edge_over,
                    "stake": FLAT_STAKE,
                }
            )

    return draw_picks, over_picks


def save_friday_shortlist(source_meta, draw_picks, over_picks):
    """
    Αποθηκεύει το Friday shortlist σε JSON.
    """
    total_flat_stake = FLAT_STAKE * (len(draw_picks) + len(over_picks))

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "source_report": THURSDAY_REPORT_PATH,
        "source_meta": source_meta,
        "summary": {
            "draw_picks": len(draw_picks),
            "over_picks": len(over_picks),
            "flat_stake_per_pick": FLAT_STAKE,
            "total_flat_stake": total_flat_stake,
        },
        "draw_picks": draw_picks,
        "over_picks": over_picks,
    }

    Path(os.path.dirname(FRIDAY_SHORTLIST_PATH)).mkdir(
        parents=True, exist_ok=True
    )

    with open(FRIDAY_SHORTLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main():
    source_meta, fixtures = load_thursday_report()
    draw_picks, over_picks = build_shortlist(fixtures)
    report = save_friday_shortlist(source_meta, draw_picks, over_picks)

    s = report["summary"]
    print(
        f"Friday shortlist report saved: {FRIDAY_SHORTLIST_PATH}\n"
        f"Draw picks: {s['draw_picks']}, Over picks: {s['over_picks']}"
    )


if __name__ == "__main__":
    main()
