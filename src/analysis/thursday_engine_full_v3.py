import os
import json
from datetime import datetime

OUTPUT_PATH = "logs/thursday_report_v3.json"

# ---------------- SANITY CAPS ----------------
CAPS = {
    "over": 0.70,
    "under": 0.75,
    "draw_min": 0.20,
    "draw_max": 0.32,
    "any_min": 0.05,
}

LOW_TEMPO_LEAGUES = {"Serie B", "Ligue 2"}

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def snap_fair_to_market(fair, offered):
    if not offered or offered <= 1:
        return fair
    if fair < offered / 1.35:
        return round(offered / 1.35, 2)
    if fair > offered * 1.35:
        return round(offered * 1.35, 2)
    return fair

def normalize_probs(f):
    # --- Any outcome min ---
    for k in ["home_prob", "draw_prob", "away_prob", "over_2_5_prob", "under_2_5_prob"]:
        if f.get(k) is not None:
            f[k] = max(CAPS["any_min"], f[k])

    # --- Draw caps ---
    if f.get("draw_prob") is not None:
        f["draw_prob"] = clamp(f["draw_prob"], CAPS["draw_min"], CAPS["draw_max"])

    # --- Over / Under caps ---
    if f.get("over_2_5_prob") is not None:
        f["over_2_5_prob"] = min(f["over_2_5_prob"], CAPS["over"])
    if f.get("under_2_5_prob") is not None:
        f["under_2_5_prob"] = min(f["under_2_5_prob"], CAPS["under"])

def favorite_protection(f):
    for market, prob_key in [
        ("offered_1", "home_prob"),
        ("offered_2", "away_prob"),
    ]:
        off = f.get(market)
        if off is None or f.get(prob_key) is None:
            continue
        if off <= 1.40:
            f[prob_key] = max(f[prob_key], 0.62)
        elif off <= 1.60:
            f[prob_key] = max(f[prob_key], 0.55)
        elif off <= 1.80:
            f[prob_key] = max(f[prob_key], 0.50)

def over_under_logic(f):
    lam_h = f.get("lambda_home")
    lam_a = f.get("lambda_away")
    league = f.get("league")

    if lam_h is None or lam_a is None:
        return

    lam_total = lam_h + lam_a

    # Over blocks
    if lam_total < 2.4 or lam_h < 0.9 or lam_a < 0.9:
        f["over_2_5_prob"] = min(f.get("over_2_5_prob", 0), 0.60)

    if league in LOW_TEMPO_LEAGUES:
        f["over_2_5_prob"] = min(f.get("over_2_5_prob", 0), 0.65)

    # Under blocks
    if lam_total > 3.0 and lam_h > 1.4 and lam_a > 1.4:
        f["under_2_5_prob"] = min(f.get("under_2_5_prob", 0), 0.60)

def fair_from_prob(p):
    return round(1 / p, 2) if p and p > 0 else None

def process_fixture(f):
    normalize_probs(f)
    favorite_protection(f)
    over_under_logic(f)

    # Fair odds recompute
    for prob_key, fair_key, off_key in [
        ("home_prob", "fair_1", "offered_1"),
        ("draw_prob", "fair_x", "offered_x"),
        ("away_prob", "fair_2", "offered_2"),
        ("over_2_5_prob", "fair_over_2_5", "offered_over_2_5"),
        ("under_2_5_prob", "fair_under_2_5", "offered_under_2_5"),
    ]:
        p = f.get(prob_key)
        if p:
            fair = fair_from_prob(p)
            fair = snap_fair_to_market(fair, f.get(off_key))
            f[fair_key] = fair

    return f

def main():
    # ⚠️ Εδώ υποθέτουμε ότι τα fixtures έρχονται ήδη φορτωμένα από upstream
    with open("logs/raw_fixtures.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    fixtures = data.get("fixtures", [])
    out = []

    for fx in fixtures:
        out.append(process_fixture(fx))

    output = {
        "timestamp": datetime.utcnow().isoformat(),
        "fixtures": out,
        "window": data.get("window", {}),
    }

    os.makedirs("logs", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Thursday stabilized report saved → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
