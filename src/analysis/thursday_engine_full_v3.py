import os
import json
import math
import requests
import datetime
import unicodedata
import re
from dateutil import parser

# ============================================================
#  THURSDAY ENGINE v3.6 (Stats Layer Upgrade + Better Lambdas + Better Scores)
#
#  Key upgrades:
#   1) Team stats via /teams/statistics (home/away splits) + fallback prev season
#   2) Expected goals uses: home attack vs away defense, away attack vs home defense
#   3) Probabilities: Poisson, light league blending (NO hard clamps that kill draws/overs)
#   4) Scores: NOT just prob*10. Uses:
#        - draw: draw_prob + lambda closeness
#        - over/under: total lambda + prob
#   5) Odds matching: keeps your v3.5 fuzzy + time proximity
#
#  Output: logs/thursday_report_v3.json (compatible)
# ============================================================

API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

FOOTBALL_SEASON = os.getenv("FOOTBALL_SEASON", "2025")
USE_ODDS_API = os.getenv("USE_ODDS_API", "true").lower() == "true"
WINDOW_HOURS = int(os.getenv("WINDOW_HOURS", "72"))

# When stats sample is tiny, fallback to prev season
MIN_STATS_MATCHES = int(os.getenv("MIN_STATS_MATCHES", "6"))

LEAGUES = {
    "Premier League": 39,
    "Championship": 40,
    "Ligue 1": 61,
    "Ligue 2": 62,
    "Bundesliga": 78,
    "Serie A": 135,
    "Serie B": 136,
    "La Liga": 140,
    "Liga Portugal 1": 94,
}

LEAGUE_TO_SPORT = {
    "Premier League": "soccer_epl",
    "Championship": "soccer_efl_champ",
    "La Liga": "soccer_spain_la_liga",
    "Serie A": "soccer_italy_serie_a",
    "Serie B": "soccer_italy_serie_b",
    "Bundesliga": "soccer_germany_bundesliga",
    "Ligue 1": "soccer_france_ligue_one",
    "Ligue 2": "soccer_france_ligue_two",
    "Liga Portugal 1": "soccer_portugal_primeira_liga",
}

TEAM_STATS_CACHE = {}

def log(msg: str):
    print(msg, flush=True)

def safe_float(v, default=None):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default

def implied(p: float):
    # FAIR = 1/prob (SPEC – DO NOT CHANGE)
    return 1.0 / p if p and p > 0 else None

# ------------------------- NAME NORMALIZATION -------------------------
def _strip_accents(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def normalize_team_name(raw: str) -> str:
    if not raw:
        return ""
    s = _strip_accents(raw).lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    kill = {"fc", "afc", "cf", "sc", "sv", "ssc", "ac", "cd", "ud", "bk", "fk"}
    parts = [p for p in s.split() if p not in kill]
    s = " ".join(parts).strip()

    aliases = {
        "wolverhampton wanderers": "wolves",
        "wolverhampton": "wolves",
        "brighton and hove albion": "brighton",
        "west bromwich albion": "west brom",
        "manchester united": "man utd",
        "manchester city": "man city",
        "newcastle united": "newcastle",
        "tottenham hotspur": "tottenham",
        "bayern munchen": "bayern munich",
        "borussia dortmund": "dortmund",
        "bayer leverkusen": "leverkusen",
        "paris saint germain": "psg",
        "internazionale": "inter",
        "sporting clube de portugal": "sporting",
    }
    return aliases.get(s, s)

def token_set(s: str):
    return set([t for t in s.split() if t])

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0

def iso_z(dt: datetime.datetime) -> str:
    dt = dt.astimezone(datetime.timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")

# ------------------------- FIXTURES (API-FOOTBALL) -------------------------
def fetch_fixtures(league_id: int, league_name: str):
    if not API_FOOTBALL_KEY:
        log("⚠️ Missing FOOTBALL_API_KEY – NO fixtures will be fetched!")
        return []

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"league": league_id, "season": FOOTBALL_SEASON}

    try:
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=25).json()
    except Exception as e:
        log(f"⚠️ Error fetching fixtures for {league_name}: {e}")
        return []

    resp = r.get("response") or []
    if not resp:
        log(f"⚠️ No fixtures response for league {league_name}")
        return []

    out = []
    now = datetime.datetime.now(datetime.timezone.utc)

    for fx in resp:
        if fx["fixture"]["status"]["short"] != "NS":
            continue

        dt = parser.isoparse(fx["fixture"]["date"]).astimezone(datetime.timezone.utc)
        diff_hours = (dt - now).total_seconds() / 3600.0
        if not (0 <= diff_hours <= WINDOW_HOURS):
            continue

        home = fx["teams"]["home"]
        away = fx["teams"]["away"]

        out.append(
            {
                "id": fx["fixture"]["id"],
                "league_id": league_id,
                "league_name": league_name,
                "home": home["name"],
                "away": away["name"],
                "home_id": home["id"],
                "away_id": away["id"],
                "home_norm": normalize_team_name(home["name"]),
                "away_norm": normalize_team_name(away["name"]),
                "date_raw": fx["fixture"]["date"],
                "commence_utc": dt,
            }
        )

    log(f"→ {league_name}: {len(out)} fixtures within window")
    return out

# ------------------------- LEAGUE BASELINES -------------------------
def fetch_league_baselines(league_id: int):
    overrides = {
        39: {"avg_goals_per_match": 2.9, "avg_draw_rate": 0.24, "avg_over25_rate": 0.58, "home_advantage": 0.18},
        40: {"avg_goals_per_match": 2.5, "avg_draw_rate": 0.28, "avg_over25_rate": 0.52, "home_advantage": 0.16},
        78: {"avg_goals_per_match": 3.1, "avg_draw_rate": 0.25, "avg_over25_rate": 0.60, "home_advantage": 0.16},
        135: {"avg_goals_per_match": 2.5, "avg_draw_rate": 0.30, "avg_over25_rate": 0.52, "home_advantage": 0.17},
        140: {"avg_goals_per_match": 2.6, "avg_draw_rate": 0.27, "avg_over25_rate": 0.55, "home_advantage": 0.18},
    }
    base = {"avg_goals_per_match": 2.6, "avg_draw_rate": 0.26, "avg_over25_rate": 0.55, "home_advantage": 0.18}
    if league_id in overrides:
        base.update(overrides[league_id])
    return base

# ------------------------- TEAM STATS via /teams/statistics -------------------------
def _football_get(path: str, params: dict):
    url = f"{API_FOOTBALL_BASE}/{path}"
    try:
        res = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=25)
        if res.status_code != 200:
            return None
        return res.json()
    except Exception:
        return None

def fetch_team_statistics(team_id: int, league_id: int, season: str):
    """
    Returns dict with:
      - played_home, played_away
      - gf_home_pg, ga_home_pg
      - gf_away_pg, ga_away_pg
    """
    r = _football_get("teams/statistics", {"team": team_id, "league": league_id, "season": season})
    if not r or not r.get("response"):
        return None

    resp = r["response"]
    played = resp.get("fixtures", {}).get("played", {})
    goals_for = resp.get("goals", {}).get("for", {}).get("average", {})
    goals_against = resp.get("goals", {}).get("against", {}).get("average", {})

    ph = safe_float(played.get("home"), 0.0) or 0.0
    pa = safe_float(played.get("away"), 0.0) or 0.0

    gf_h = safe_float(goals_for.get("home"), None)
    gf_a = safe_float(goals_for.get("away"), None)
    ga_h = safe_float(goals_against.get("home"), None)
    ga_a = safe_float(goals_against.get("away"), None)

    return {
        "played_home": int(ph),
        "played_away": int(pa),
        "gf_home_pg": gf_h,
        "ga_home_pg": ga_h,
        "gf_away_pg": gf_a,
        "ga_away_pg": ga_a,
        "season": season,
    }

def get_team_stats(team_id: int, league_id: int):
    """
    Cached. Uses current season, falls back to previous season if sample too small.
    """
    ck = (team_id, league_id, FOOTBALL_SEASON)
    if ck in TEAM_STATS_CACHE:
        return TEAM_STATS_CACHE[ck]

    if not API_FOOTBALL_KEY:
        TEAM_STATS_CACHE[ck] = {"ok": False, "reason": "missing_api_key"}
        return TEAM_STATS_CACHE[ck]

    cur = fetch_team_statistics(team_id, league_id, FOOTBALL_SEASON)
    if cur:
        sample = (cur.get("played_home", 0) or 0) + (cur.get("played_away", 0) or 0)
    else:
        sample = 0

    used = cur
    used_reason = "current_season"

    if (not used) or sample < MIN_STATS_MATCHES:
        # fallback previous season
        try:
            prev_season = str(int(FOOTBALL_SEASON) - 1)
        except Exception:
            prev_season = None

        if prev_season:
            prev = fetch_team_statistics(team_id, league_id, prev_season)
            if prev:
                used = prev
                used_reason = "fallback_prev_season"

    if not used:
        TEAM_STATS_CACHE[ck] = {"ok": False, "reason": "no_stats"}
        return TEAM_STATS_CACHE[ck]

    TEAM_STATS_CACHE[ck] = {"ok": True, "reason": used_reason, **used}
    return TEAM_STATS_CACHE[ck]

# ------------------------- POISSON MODEL -------------------------
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def compute_expected_goals(home_stats: dict, away_stats: dict, league_baseline: dict):
    """
    Uses home team HOME attack vs away team AWAY defense.
    Uses away team AWAY attack vs home team HOME defense.
    Adds home advantage.
    Shrinks to league average if stats missing.
    """
    league_avg = safe_float(league_baseline.get("avg_goals_per_match"), 2.6) or 2.6
    denom = max(0.6, league_avg / 2.0)
    home_adv = safe_float(league_baseline.get("home_advantage"), 0.18) or 0.18

    # priors (shrinkage): if missing, use denom
    h_gf = safe_float(home_stats.get("gf_home_pg"), None)
    h_ga = safe_float(home_stats.get("ga_home_pg"), None)
    a_gf = safe_float(away_stats.get("gf_away_pg"), None)
    a_ga = safe_float(away_stats.get("ga_away_pg"), None)

    if h_gf is None: h_gf = denom
    if h_ga is None: h_ga = denom
    if a_gf is None: a_gf = denom
    if a_ga is None: a_ga = denom

    # attack/defense factors
    h_att = h_gf / denom
    h_def = h_ga / denom
    a_att = a_gf / denom
    a_def = a_ga / denom

    # stronger separation than before, but still stable
    lam_h = denom * (0.70 * h_att + 0.30 * a_def)
    lam_a = denom * (0.70 * a_att + 0.30 * h_def)

    lam_h *= (1.0 + home_adv)

    # soft caps only (avoid insane extremes)
    lam_h = max(0.20, min(3.60, lam_h))
    lam_a = max(0.20, min(3.30, lam_a))

    return lam_h, lam_a

def compute_probabilities(lambda_home: float, lambda_away: float, league_baseline: dict):
    max_goals = 7
    ph = pd = pa = 0.0
    po = 0.0

    for i in range(max_goals + 1):
        p_i = poisson_pmf(i, lambda_home)
        for j in range(max_goals + 1):
            p_j = poisson_pmf(j, lambda_away)
            p = p_i * p_j

            if i > j:
                ph += p
            elif i == j:
                pd += p
            else:
                pa += p

            if i + j >= 3:
                po += p

    tot = ph + pd + pa
    if tot <= 0:
        ph, pd, pa = 0.40, 0.20, 0.40
    else:
        ph, pd, pa = ph / tot, pd / tot, pa / tot

    lg_draw = safe_float(league_baseline.get("avg_draw_rate"), None)
    lg_over = safe_float(league_baseline.get("avg_over25_rate"), None)

    # light blending
    if lg_draw is not None:
        alpha_d = 0.10
        pd = (1 - alpha_d) * pd + alpha_d * lg_draw
        rest = max(1e-12, ph + pa)
        scale = (1.0 - pd) / rest
        ph *= scale
        pa *= scale

    if lg_over is not None:
        alpha_o = 0.10
        po = (1 - alpha_o) * po + alpha_o * lg_over

    # normalize & guard
    eps = 1e-6
    ph = max(eps, min(1.0 - 2*eps, ph))
    pd = max(eps, min(1.0 - 2*eps, pd))
    pa = max(eps, min(1.0 - 2*eps, pa))
    s = ph + pd + pa
    ph, pd, pa = ph / s, pd / s, pa / s

    po = max(eps, min(1.0 - eps, po))
    pu = 1.0 - po

    return {"home_prob": ph, "draw_prob": pd, "away_prob": pa, "over_2_5_prob": po, "under_2_5_prob": pu}

# ------------------------- ODDS (TheOddsAPI) -------------------------
def _odds_request(sport_key: str, params: dict):
    url = f"{ODDS_BASE_URL}/{sport_key}/odds"
    try:
        res = requests.get(url, params=params, timeout=25)
        rem = res.headers.get("x-requests-remaining")
        used = res.headers.get("x-requests-used")
        log(f"   TheOddsAPI status={res.status_code} remaining={rem} used={used}")
        if res.status_code != 200:
            log(f"   body={res.text[:220]}")
            return []
        data = res.json()
        return data or []
    except Exception as e:
        log(f"   TheOddsAPI request error: {e}")
        return []

def fetch_odds_for_league(league_name: str, window_from: datetime.datetime, window_to: datetime.datetime):
    if not USE_ODDS_API:
        return []
    if not ODDS_API_KEY:
        log("⚠️ Missing ODDS_API_KEY – skipping odds")
        return []

    sport_key = LEAGUE_TO_SPORT.get(league_name)
    if not sport_key:
        return []

    base_params = {
        "apiKey": ODDS_API_KEY,
        "regions": "eu,uk,us",
        "markets": "h2h,totals",
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }

    params1 = dict(base_params)
    params1["commenceTimeFrom"] = iso_z(window_from)
    params1["commenceTimeTo"] = iso_z(window_to)

    log(f"→ Odds fetch {league_name} [{sport_key}] (windowed)")
    data = _odds_request(sport_key, params1)
    if data:
        return data

    log(f"→ Odds fetch {league_name} [{sport_key}] (no-window fallback)")
    data = _odds_request(sport_key, base_params)
    if data:
        return data

    log(f"→ Odds fetch {league_name} [{sport_key}] (eu-only fallback)")
    params3 = dict(base_params)
    params3["regions"] = "eu"
    return _odds_request(sport_key, params3)

def build_events_cache(odds_events):
    out = []
    for ev in odds_events or []:
        h_raw = ev.get("home_team", "") or ""
        a_raw = ev.get("away_team", "") or ""
        h = normalize_team_name(h_raw)
        a = normalize_team_name(a_raw)
        if not h or not a:
            continue

        try:
            ct = parser.isoparse(ev.get("commence_time")).astimezone(datetime.timezone.utc)
        except Exception:
            ct = None

        out.append({
            "home_norm": h,
            "away_norm": a,
            "home_tokens": token_set(h),
            "away_tokens": token_set(a),
            "commence_time": ct,
            "raw": ev,
        })
    return out

def pick_best_odds_for_fixture(fx, league_events_cache):
    if not league_events_cache:
        return {}, {"matched": False, "reason": "no_odds_events"}

    fx_h = fx["home_norm"]
    fx_a = fx["away_norm"]
    fx_ht = token_set(fx_h)
    fx_at = token_set(fx_a)
    fx_time = fx.get("commence_utc")

    best = None
    best_score = -1.0
    best_swap = False

    for ev in league_events_cache:
        ct = ev["commence_time"]
        time_pen = 0.0
        if fx_time and ct:
            diff_h = abs((ct - fx_time).total_seconds()) / 3600.0
            time_pen = min(1.0, diff_h / 8.0)

        s1 = jaccard(fx_ht, ev["home_tokens"])
        s2 = jaccard(fx_at, ev["away_tokens"])
        score_norm = (s1 + s2) / 2.0 - 0.25 * time_pen

        s1s = jaccard(fx_ht, ev["away_tokens"])
        s2s = jaccard(fx_at, ev["home_tokens"])
        score_swap = (s1s + s2s) / 2.0 - 0.25 * time_pen

        if score_norm > best_score:
            best_score = score_norm
            best = ev
            best_swap = False
        if score_swap > best_score:
            best_score = score_swap
            best = ev
            best_swap = True

    if best is None or best_score < 0.55:
        return {}, {"matched": False, "reason": f"no_match(score={best_score:.2f})"}

    ev_raw = best["raw"]

    best_home = best_draw = best_away = None
    best_over = best_under = None

    for bm in ev_raw.get("bookmakers", []) or []:
        for m in bm.get("markets", []) or []:
            mk = (m.get("key") or "").lower()

            if mk == "h2h":
                for o in m.get("outcomes", []) or []:
                    price = safe_float(o.get("price"), None)
                    if price is None or price <= 1.0:
                        continue
                    nm = normalize_team_name(o.get("name", ""))
                    if nm in ("draw", "x", "tie"):
                        best_draw = max(best_draw or 0.0, price)
                    else:
                        if not best_swap:
                            if nm == best["home_norm"]:
                                best_home = max(best_home or 0.0, price)
                            elif nm == best["away_norm"]:
                                best_away = max(best_away or 0.0, price)
                        else:
                            if nm == best["home_norm"]:
                                best_away = max(best_away or 0.0, price)
                            elif nm == best["away_norm"]:
                                best_home = max(best_home or 0.0, price)

            elif mk == "totals":
                for o in m.get("outcomes", []) or []:
                    price = safe_float(o.get("price"), None)
                    if price is None or price <= 1.0:
                        continue
                    point = safe_float(o.get("point"), None)
                    if point is not None and abs(point - 2.5) > 1e-6:
                        continue
                    name = (o.get("name") or "").lower()
                    if "over" in name and ("2.5" in name or point == 2.5):
                        best_over = max(best_over or 0.0, price)
                    elif "under" in name and ("2.5" in name or point == 2.5):
                        best_under = max(best_under or 0.0, price)

    return {
        "home": best_home,
        "draw": best_draw,
        "away": best_away,
        "over": best_over,
        "under": best_under,
    }, {"matched": True, "score": round(best_score, 3), "swap": best_swap}

# ------------------------- SCORES (better) -------------------------
def clamp_1_10(x: float) -> float:
    x = round(float(x), 1)
    if x < 1.0: return 1.0
    if x > 10.0: return 10.0
    return x

def score_draw(draw_prob: float, lam_h: float, lam_a: float) -> float:
    # closeness boosts draw score when teams expected goals are similar
    closeness = 1.0 - min(1.0, abs(lam_h - lam_a) / 1.25)  # 0..1
    s = (draw_prob * 7.0 + closeness * 3.0) * 10.0 / 10.0  # keep scale stable
    return clamp_1_10(s)

def score_over(over_prob: float, lam_sum: float) -> float:
    # total expected goals boosts over score
    lam_boost = min(1.0, max(0.0, (lam_sum - 2.2) / 1.6))  # 0..1 roughly
    s = (over_prob * 7.0 + lam_boost * 3.0) * 10.0 / 10.0
    return clamp_1_10(s)

def score_under(under_prob: float, lam_sum: float) -> float:
    lam_boost = min(1.0, max(0.0, (2.6 - lam_sum) / 1.6))  # favors lower totals
    s = (under_prob * 7.0 + lam_boost * 3.0) * 10.0 / 10.0
    return clamp_1_10(s)

# ------------------------- MAIN PIPELINE -------------------------
def build_fixture_blocks():
    fixtures_out = []
    now = datetime.datetime.now(datetime.timezone.utc)
    to_dt = now + datetime.timedelta(hours=WINDOW_HOURS)

    odds_from = now - datetime.timedelta(hours=6)

    log(f"Using FOOTBALL_SEASON={FOOTBALL_SEASON}")
    log(f"Window: next {WINDOW_HOURS} hours")
    log(f"USE_ODDS_API={USE_ODDS_API}")

    if not API_FOOTBALL_KEY:
        log("❌ FOOTBALL_API_KEY is missing. Aborting fixture fetch.")
        return []

    all_fixtures = []
    for lg_name, lg_id in LEAGUES.items():
        all_fixtures.extend(fetch_fixtures(lg_id, lg_name))

    log(f"Total raw fixtures collected: {len(all_fixtures)}")

    odds_cache_by_league = {}
    if USE_ODDS_API:
        total_events = 0
        for lg_name in LEAGUES.keys():
            odds_events = fetch_odds_for_league(lg_name, odds_from, to_dt)
            total_events += len(odds_events or [])
            odds_cache_by_league[lg_name] = build_events_cache(odds_events)
        log(f"Odds events fetched total: {total_events}")
    else:
        log("⚠️ USE_ODDS_API=False → skipping TheOddsAPI.")

    matched_cnt = 0
    stats_fallback_cnt = 0
    stats_missing_cnt = 0

    for fx in all_fixtures:
        league_id = fx["league_id"]
        league_name = fx["league_name"]
        lb = fetch_league_baselines(league_id)

        hs = get_team_stats(fx["home_id"], league_id)
        aw = get_team_stats(fx["away_id"], league_id)

        if not hs.get("ok"):
            stats_missing_cnt += 1
        if hs.get("reason") == "fallback_prev_season" or aw.get("reason") == "fallback_prev_season":
            stats_fallback_cnt += 1

        lam_h, lam_a = compute_expected_goals(hs, aw, lb)
        probs = compute_probabilities(lam_h, lam_a, lb)

        p_home = probs["home_prob"]
        p_draw = probs["draw_prob"]
        p_away = probs["away_prob"]
        p_over = probs["over_2_5_prob"]
        p_under = probs["under_2_5_prob"]

        fair_1 = implied(p_home)
        fair_x = implied(p_draw)
        fair_2 = implied(p_away)
        fair_over = implied(p_over)
        fair_under = implied(p_under)

        offered = {}
        match_debug = {"matched": False, "reason": "odds_off"}
        if USE_ODDS_API:
            league_cache = odds_cache_by_league.get(league_name, [])
            offered, match_debug = pick_best_odds_for_fixture(fx, league_cache)
            if match_debug.get("matched"):
                matched_cnt += 1

        dt = fx["commence_utc"]
        lam_sum = lam_h + lam_a

        fixtures_out.append(
            {
                "fixture_id": fx["id"],
                "date": dt.date().isoformat(),
                "time": dt.strftime("%H:%M"),
                "league_id": league_id,
                "league": league_name,
                "home": fx["home"],
                "away": fx["away"],
                "model": "bombay_balanced_v3_6",

                "lambda_home": round(lam_h, 3),
                "lambda_away": round(lam_a, 3),

                "home_prob": round(p_home, 3),
                "draw_prob": round(p_draw, 3),
                "away_prob": round(p_away, 3),
                "over_2_5_prob": round(p_over, 3),
                "under_2_5_prob": round(p_under, 3),

                "fair_1": fair_1,
                "fair_x": fair_x,
                "fair_2": fair_2,
                "fair_over_2_5": fair_over,
                "fair_under_2_5": fair_under,

                "offered_1": offered.get("home"),
                "offered_x": offered.get("draw"),
                "offered_2": offered.get("away"),
                "offered_over_2_5": offered.get("over"),
                "offered_under_2_5": offered.get("under"),

                # better scores
                "score_draw": score_draw(p_draw, lam_h, lam_a),
                "score_over": score_over(p_over, lam_sum),
                "score_under": score_under(p_under, lam_sum),

                # stats debug (small but useful)
                "stats_home": {"ok": hs.get("ok"), "reason": hs.get("reason"), "season": hs.get("season")},
                "stats_away": {"ok": aw.get("ok"), "reason": aw.get("reason"), "season": aw.get("season")},

                # odds debug
                "odds_match": match_debug,
            }
        )

    log(f"Thursday fixtures_out: {len(fixtures_out)} | odds matched: {matched_cnt}")
    log(f"Stats: fallback_prev_season={stats_fallback_cnt} | missing_stats={stats_missing_cnt}")
    return fixtures_out

def main():
    fixtures = build_fixture_blocks()

    now = datetime.datetime.now(datetime.timezone.utc)
    to_dt = now + datetime.timedelta(hours=WINDOW_HOURS)

    out = {
        "generated_at": now.isoformat(),
        "window": {"from": now.date().isoformat(), "to": to_dt.date().isoformat(), "hours": WINDOW_HOURS},
        "fixtures_total": len(fixtures),
        "fixtures": fixtures,
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/thursday_report_v3.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    log(f"✅ Thursday v3.6 READY. Fixtures: {len(fixtures)}")

if __name__ == "__main__":
    main()
