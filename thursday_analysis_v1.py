import json
import os
from datetime import datetime

# === Settings ===
OUTPUT_PATH = "logs/thursday_report_v1.json"
os.makedirs("logs", exist_ok=True)

# === Sample leagues and fixtures ===
fixtures = [
    {"league": "Bundesliga", "match": "Dortmund – Freiburg"},
    {"league": "Serie A", "match": "Atalanta – Bologna"},
    {"league": "Ligue 1", "match": "Lille – Marseille"},
    {"league": "Eredivisie", "match": "Utrecht – Twente"},
    {"league": "Swiss Super League", "match": "Basel – Luzern"},
]

# === Fair Odds + Confidence simulation ===
# Normally from model or database — here it's mock logic for structure testing
data = []
for f in fixtures:
    fair_home = round(1.70 + (hash(f['match']) % 25) / 100, 2)
    fair_draw = round(2.80 + (hash(f['match']) % 40) / 100, 2)
    fair_away = round(2.10 + (hash(f['match']) % 35) / 100, 2)
    fair_over = round(1.65 + (hash(f['match']) % 30) / 100, 2)

    conf_draw = round(7.5 + (hash(f['league']) % 25) / 10, 1)
    conf_over = round(7.8 + (hash(f['match']) % 20) / 10, 1)

    data.append({
        "league": f["league"],
        "match": f["match"],
        "Fair 1–X–2": [fair_home, fair_draw, fair_away],
        "Fair Over": fair_over,
        "Conf Draw": conf_draw,
        "Conf Over": conf_over,
        "Status": "ok"
    })

# === Compose report ===
report = {
    "report_name": "Thursday Main Analysis Report",
    "generated_at": datetime.utcnow().isoformat(),
    "fixtures_total": len(data),
    "fixtures": data
}

# === Save report ===
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"✅ Thursday analysis completed successfully.")
print(f"📁 Report saved at: {OUTPUT_PATH}")
print(f"📊 Fixtures analyzed: {len(data)}")
