import datetime
import random
import json

# ----------------------------------------------------------
#  Thursday Analysis — Bombay Engine v6.3 (Bookmaker Logic)
# ----------------------------------------------------------

LEAGUES = [
    "Ligue 1", "Serie A", "La Liga", "Championship", "Serie B",
    "Ligue 2", "Liga Portugal 2", "Swiss Super League",
    "Eredivisie", "Jupiler Pro League"
]

# League baseline draw and over tendencies
LEAGUE_FACTORS = {
    "Ligue 1": {"draw": 0.29, "over": 0.54},
    "Serie A": {"draw": 0.27, "over": 0.56},
    "La Liga": {"draw": 0.28, "over": 0.51},
    "Championship": {"draw": 0.30, "over": 0.49},
    "Serie B": {"draw": 0.31, "over": 0.46},
    "Ligue 2": {"draw": 0.32, "over": 0.47},
    "Liga Portugal 2": {"draw": 0.30, "over": 0.48},
    "Swiss Super League": {"draw": 0.27, "over": 0.59},
    "Eredivisie": {"draw": 0.25, "over": 0.63},
    "Jupiler Pro League": {"draw": 0.26, "over": 0.61},
}

def compute_team_strength():
    return round(random.uniform(0.4, 1.6), 2)

def bookmaker_fair_odds(str_home, str_away, league):
    total = str_home + str_away
    prob_home = str_home / total
    prob_away = str_away / total
    prob_draw = LEAGUE_FACTORS[league]["draw"]

    margin = 0.04
    total_prob = prob_home + prob_away + prob_draw
    prob_home = prob_home / total_prob * (1 - margin)
    prob_draw = prob_draw / total_prob * (1 - margin)
    prob_away = prob_away / total_prob * (1 - margin)

    fair_1 = round(1 / prob_home, 2)
    fair_x = round(1 / prob_draw, 2)
    fair_2 = round(1 / prob_away, 2)

    base_over = LEAGUE_FACTORS[league]["over"]
    adj_over = base_over + random.uniform(-0.05, 0.05)
    fair_over = round(1 / adj_over, 2)

    return fair_1, fair_x, fair_2, fair_over

def draw_score_formula(league):
    base = LEAGUE_FACTORS[league]["draw"] * 10
    h2h_factor = random.uniform(0, 2)
    form_factor = random.uniform(-1, 1)
    xg_balance = random.uniform(0, 1)
    weather_factor = random.choice([0, 0.3, 0.5])
    score = base * 0.5 + h2h_factor + form_factor + xg_balance + weather_factor
    return round(min(max(score, 0), 10), 1)

def over_score_formula(league):
    base = LEAGUE_FACTORS[league]["over"] * 10
    xg_factor = random.uniform(0, 2)
    momentum = random.uniform(-0.5, 1)
    h2h_over = random.uniform(0, 1.5)
    weather_factor = random.choice([0, 0.2, 0.3])
    score = base * 0.5 + xg_factor + momentum + h2h_over + weather_factor
    return round(min(max(score, 0), 10), 1)

def generate_fixture_report():
    today = datetime.date.today()
    from_date = today + datetime.timedelta(days=4)
    to_date = from_date + datetime.timedelta(days=3)

    fixtures = []
    for league in LEAGUES:
        for _ in range(random.randint(3, 6)):
            home = random.choice(["Lille", "Marseille", "Roma", "Porto", "PSV", "Anderlecht", "Basel", "Toulouse", "Palermo", "Levante"])
            away = random.choice(["Nice", "Juventus", "Benfica", "Twente", "Club Brugge", "Young Boys", "Lorient", "Atalanta", "Cadiz", "Empoli"])
            str_home, str_away = compute_team_strength(), compute_team_strength()
            fair_1, fair_x, fair_2, fair_over = bookmaker_fair_odds(str_home, str_away, league)
            score_draw = draw_score_formula(league)
            score_over = over_score_formula(league)
            category = "Draw-prone" if score_draw >= 7.5 else "Open" if score_over >= 7.5 else "Balanced"

            fixtures.append({
                "match": f"{home} - {away}",
                "league": league,
                "fair_1": fair_1,
                "fair_x": fair_x,
                "fair_2": fair_2,
                "fair_over": fair_over,
                "score_draw": score_draw,
                "score_over": score_over,
                "category": category
            })

    report = {
        "count": len(fixtures),
        "range": {"from": str(from_date), "to": str(to_date)},
        "fixtures": fixtures
    }

    return json.dumps(report, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    print(generate_fixture_report())
