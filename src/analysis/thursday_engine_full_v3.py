import os
import json
import math
import requests
import datetime
import unicodedata
import re
from statistics import mean, pstdev
from dateutil import parser

# ============================================================
#  THURSDAY ENGINE v3.7 (Production Poisson + Dixon-Coles + Shrinkage + Sanity)
#
#  Goals:
#   - Stable lambdas (no crazy fair odds like <1.15)
#   - Dixon–Coles low-score correction -> more realistic draws (0-0, 1-1)
#   - Sanity rules on probabilities (cap favorites, draw floor)
#   - Over/Under scoring uses probability + z-score of lambda_total within league window
#   - Odds matching: gated by time + similarity (prevents wrong match)
#   - Adds value_pct_* fields for UI/Friday selection
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

# ---- Dixon-Coles + sanity controls (tweakable without code change)
DC_RHO = float(os.getenv("DC_RHO", "0.12"))                 # typical 0.10 - 0.13
DRAW_PROB_FLOOR = float(os.getenv("DRAW_PROB_FLOOR", "0.18"))  # keep draw alive
FAV_PROB_CAP = float(os.getenv("FAV_PROB_CAP", "0.78"))        # avoids fair < ~1.28

# ---- Lambda shrinkage controls
MIN_STATS_MATCHES = int(os.getenv("MIN_STATS_MATCHES", "6"))
SHRINK_K = float(os.getenv("SHRINK_K", "8.0"))  # higher => more shrinkage to league avg
LAMBDA_MIN = float(os.getenv("LAMBDA_MIN", "0.30"))
LAMBDA_MAX_HOME = float(os.getenv("LAMBDA_MAX_HOME", "3.00"))
LAMBDA_MAX_AWAY = float(os.getenv("LAMBDA_MAX_AWAY", "2.80"))

# ---- Odds matching controls
ODDS_TIME_GATE_HOURS = float(os.getenv("ODDS_TIME_GATE_HOURS", "6"))
ODDS_TIME_SOFT_HOURS = float(os.getenv("ODDS_TIME_SOFT_HOURS", "10"))
ODDS_SIM_THRESHOLD = float(os.getenv("ODDS_SIM_THRESHOLD", "0.62"))

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

    kill = {"fc", "afc", "cf", "sc", "sv", "ssc", "ac", "cd", "ud", "bk", "fk", "if"}
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
        "sporting cp": "sporting",
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

# ------------------------- TEAM STATS (API-FOOTBALL /teams/statistics) -------------------------
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
    r = _football_get("teams/statistics", {"team": team_id, "league": league_id, "season": season})
    if not r or not r.get("response"):
        return None

    resp = r["response"]
    played = resp.get("fixtures", {}).get("played", {})
    goals_for = resp.get("goals", {}).get("for", {}).get("average", {})
    goals_against = resp.get("goals", {}).get("against", {}).get("average", {})

    ph = int(safe_float(played.get("home"), 0.0) or 0)
    pa = int(safe_float(played.get("away"), 0.0) or 0)

    gf_h = safe_float(goals_for.get("home"), None)
    gf_a = safe_float(goals_for.get("away"), None)
    ga_h = safe_float(goals_against.get("home"), None)
    ga_a = safe_float(goals_against.get("away"), None)

    return {
        "played_home": ph,
        "played_away": pa,
        "gf_home_pg": gf_h,
        "ga_home_pg": ga_h,
        "gf_away_pg": gf_a,
        "ga_away_pg": ga_a,
        "season": season,
    }

def get_team_stats(team_id: int, league_id: int):
    ck = (team_id, league_id, FOOTBALL_SEASON)
    if ck in TEAM_STATS_CACHE:
        return TEAM_STATS_CACHE[ck]

    if not API_FOOTBALL_KEY:
        TEAM_STATS_CACHE[ck] = {"ok": False, "reason": "missing_api_key"}
        return TEAM_STATS_CACHE[ck]

    cur = fetch_team_statistics(team_id, league_id, FOOTBALL_SEASON)
    sample = 0
    if cur:
        sample = (cur.get("played_home", 0) or 0) + (cur.get("played_away", 0) or 0)

    used = cur
    used_reason = "current_season"

    if (not used) or sample < MIN_STATS_MATCHES:
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
        return res.json() or []
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

def _best_odds_from_event(ev_raw, event_home_norm, event_away_norm, swapped: bool):
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
                        if not swapped:
                            if nm == event_home_norm:
                                best_home = max(best_home or 0.0, price)
                            elif nm == event_away_norm:
                                best_away = max(best_away or 0.0, price)
                        else:
                            if nm == event_home_norm:
                                best_away = max(best_away or 0.0, price)
                            elif nm == event_away_norm:
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

    return {"home": best_home, "draw": best_draw, "away": best_away, "over": best_over, "under": best_under}

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
    best_diff = None

    for ev in league_events_cache:
        ct = ev["commence_time"]
        diff_h = None
        if fx_time and ct:
            diff_h = abs((ct - fx_time).total_seconds()) / 3600.0

        # HARD time gate
        if diff_h is not None and diff_h > ODDS_TIME_GATE_HOURS:
            continue

        time_pen = 0.0
        if diff_h is not None:
            time_pen = min(1.0, diff_h / max(1e-6, ODDS_TIME_SOFT_HOURS))

        s_norm = (jaccard(fx_ht, ev["home_tokens"]) + jaccard(fx_at, ev["away_tokens"])) / 2.0
        s_swap = (jaccard(fx_ht, ev["away_tokens"]) + jaccard(fx_at, ev["home_tokens"])) / 2.0

        score_norm = s_norm - 0.20 * time_pen
        score_swap = s_swap - 0.20 * time_pen

        if score_norm > best_score:
            best_score = score_norm
            best = ev
            best_swap = False
            best_diff = diff_h

        if score_swap > best_score:
            best_score = score_swap
            best = ev
            best_swap = True
            best_diff = diff_h

    if best is None:
        return {}, {"matched": False, "reason": f"time_gate_no_candidates(>{ODDS_TIME_GATE_HOURS}h)"}
    if best_score < ODDS_SIM_THRESHOLD:
        return {}, {"matched": False, "reason": f"low_similarity(score={best_score:.2f})"}

    odds = _best_odds_from_event(best["raw"], best["home_norm"], best["away_norm"], best_swap)

    debug = {
        "matched": True,
        "score": round(best_score, 3),
        "swap": best_swap,
        "time_diff_h": None if best_diff is None else round(best_diff, 2),
    }
    return odds, debug

# ------------------------- VALUE % -------------------------
def value_pct(offered, fair):
    if offered is None or fair is None:
        return None
    try:
        if offered <= 0 or fair <= 0:
            return None
        return round((offered / fair - 1.0) * 100.0, 1)
    except Exception:
        return None

# ------------------------- LAMBDA MODEL (shrinkage + tempo) -------------------------
def shrink(v: float, n: float, prior: float, k: float = SHRINK_K) -> float:
    # Bayesian-style shrinkage: weighted average of observed vs prior
    # n: sample size (matches)
    if v is None:
        return prior
    n = max(0.0, float(n or 0.0))
    w = n / (n + k) if (n + k) > 0 else 0.0
    return w * float(v) + (1.0 - w) * float(prior)

def compute_expected_goals(home_stats: dict, away_stats: dict, league_baseline: dict):
    """
    Production xG:
      λ_home = atk_home * def_away * league_avg_home
      λ_away = atk_away * def_home * league_avg_away
    with shrinkage to league priors and soft tempo adjustment.
    """
    league_avg = safe_float(league_baseline.get("avg_goals_per_match"), 2.6) or 2.6
    home_adv = safe_float(league_baseline.get("home_advantage"), 0.18) or 0.18
    lg_over = safe_float(league_baseline.get("avg_over25_rate"), 0.55) or 0.55

    denom = max(0.60, league_avg / 2.0)  # baseline goals per team
    league_avg_home = denom * (1.0 + 0.20)  # small structural home bias
    league_avg_away = denom * (1.0 - 0.05)  # small away drag

    # sample sizes
    h_n = float((home_stats.get("played_home", 0) or 0) + (home_stats.get("played_away", 0) or 0))
    a_n = float((away_stats.get("played_home", 0) or 0) + (away_stats.get("played_away", 0) or 0))

    # observed per-game
    h_gf_home = safe_float(home_stats.get("gf_home_pg"), None)
    h_ga_home = safe_float(home_stats.get("ga_home_pg"), None)
    a_gf_away = safe_float(away_stats.get("gf_away_pg"), None)
    a_ga_away = safe_float(away_stats.get("ga_away_pg"), None)

    # shrink to priors
    h_gf_home = shrink(h_gf_home, h_n, league_avg_home)
    h_ga_home = shrink(h_ga_home, h_n, denom)
    a_gf_away = shrink(a_gf_away, a_n, league_avg_away)
    a_ga_away = shrink(a_ga_away, a_n, denom)

    # strengths (multiplicative)
    atk_home = max(0.40, h_gf_home / max(0.40, league_avg_home))
    def_away = max(0.40, a_ga_away / max(0.40, denom))  # higher conceded => weaker defense => boosts home goals

    atk_away = max(0.40, a_gf_away / max(0.40, league_avg_away))
    def_home = max(0.40, h_ga_home / max(0.40, denom))

    lam_h = atk_home * def_away * league_avg_home
    lam_a = atk_away * def_home * league_avg_away

    # home advantage (final)
    lam_h *= (1.0 + home_adv)

    # tempo adjustment (soft)
    # over_bias = min(0.15, 0.05 * (league_ov25 - 0.52))
    over_bias = min(0.15, 0.05 * ((lg_over or 0.55) - 0.52))
    if over_bias > 0:
        lam_h *= (1.0 + over_bias)
        lam_a *= (1.0 + over_bias)

    # clamps
    lam_h = max(LAMBDA_MIN, min(LAMBDA_MAX_HOME, lam_h))
    lam_a = max(LAMBDA_MIN, min(LAMBDA_MAX_AWAY, lam_a))

    return lam_h, lam_a

# ------------------------- POISSON + DIXON-COLES -------------------------
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def poisson_probs_0_6_plus_tail(lam: float):
    # returns probs for [0..6] and tail bucket [>=7] at index 7
    p = [poisson_pmf(i, lam) for i in range(7)]
    tail = max(0.0, 1.0 - sum(p))
    p.append(tail)
    return p  # len=8

def dc_tau(x: int, y: int, lam_h: float, lam_a: float, rho: float):
    # Dixon–Coles correction term for low scores
    if x == 0 and y == 0:
        return 1.0 - (lam_h * lam_a * rho)
    if x == 1 and y == 0:
        return 1.0 + (lam_a * rho)
    if x == 0 and y == 1:
        return 1.0 + (lam_h * rho)
    if x == 1 and y == 1:
        return 1.0 - ((lam_h + lam_a) * rho)
    return 1.0

def compute_probabilities_dc(lam_h: float, lam_a: float):
    # Build 8x8 score matrix using buckets 0..6 and tail(7= >=7)
    ph = pd = pa = 0.0
    po = 0.0

    pH = poisson_probs_0_6_plus_tail(lam_h)
    pA = poisson_probs_0_6_plus_tail(lam_a)

    for x in range(8):
        for y in range(8):
            base = pH[x] * pA[y]

            # apply DC only for true low scores (0/1), not tail bucket
            tau = 1.0
            if x <= 1 and y <= 1:
                tau = dc_tau(x, y, lam_h, lam_a, DC_RHO)

            p = base * tau
            # safeguard if tau goes negative (can happen if extreme λ and rho)
            if p < 0:
                p = 0.0

            # interpret tail as 7 for comparisons (good enough for production)
            xg = 7 if x == 7 else x
            yg = 7 if y == 7 else y

            if xg > yg:
                ph += p
            elif xg == yg:
                pd += p
            else:
                pa += p

            if (xg + yg) >= 3:
                po += p

    tot = ph + pd + pa
    if tot <= 0:
        ph, pd, pa = 0.40, 0.20, 0.40
        tot = 1.0
    else:
        ph, pd, pa = ph / tot, pd / tot, pa / tot

    # --- Sanity: draw floor & favorite cap ---
    # cap favorites
    ph = min(ph, FAV_PROB_CAP)
    pa = min(pa, FAV_PROB_CAP)

    # enforce draw floor
    if pd < DRAW_PROB_FLOOR:
        need = DRAW_PROB_FLOOR - pd
        pd = DRAW_PROB_FLOOR
        rest = max(1e-12, ph + pa)
        # take proportionally from ph/pa
        take_h = need * (ph / rest)
        take_a = need * (pa / rest)
        ph = max(1e-6, ph - take_h)
        pa = max(1e-6, pa - take_a)

    # renormalize after rules
    s = ph + pd + pa
    ph, pd, pa = ph / s, pd / s, pa / s

    # clamp over (soft only)
    po = max(1e-6, min(1.0 - 1e-6, po))
    pu = 1.0 - po

    return {
        "home_prob": ph,
        "draw_prob": pd,
        "away_prob": pa,
        "over_2_5_prob": po,
        "under_2_5_prob": pu,
    }

# ------------------------- SCORES (prob + zscore / closeness) -------------------------
def clamp_1_10(x: float) -> float:
    x = round(float(x), 1)
    if x < 1.0: return 1.0
    if x > 10.0: return 10.0
    return x

def score_draw(draw_prob: float, lam_h: float, lam_a: float) -> float:
    closeness = 1.0 - min(1.0, abs(lam_h - lam_a) / 1.20)  # 0..1
    s = (draw_prob * 7.0 + closeness * 3.0) * 10.0 / 1.0
    return clamp_1_10(s)

def score_over(over_prob: float, z_total: float) -> float:
    z = max(0.0, min(1.0, z_total / 1.5))  # scale 0..~1
    s = (over_prob * 7.0 + z * 3.0) * 10.0 / 1.0
    return clamp_1_10(s)

def score_under(under_prob: float, z_total: float) -> float:
    z = max(0.0, min(1.0, (-z_total) / 1.5))
    s = (under_prob * 7.0 + z * 3.0) * 10.0 / 1.0
    return clamp_1_10(s)

# ------------------------- MAIN PIPELINE -------------------------
def build_fixture_blocks():
    fixtures_out = []
    now = datetime.datetime.now(datetime.timezone.utc)
    to_dt = now + datetime.timedelta(hours=WINDOW_HOURS)

    odds_from = now - datetime.timedelta(hours=8)  # widened

    log(f"Using FOOTBALL_SEASON={FOOTBALL_SEASON}")
    log(f"Window: next {WINDOW_HOURS} hours")
    log(f"USE_ODDS_API={USE_ODDS_API}")
    log(f"DC_RHO={DC_RHO} | DRAW_PROB_FLOOR={DRAW_PROB_FLOOR} | FAV_PROB_CAP={FAV_PROB_CAP}")
    log(f"ODDS_TIME_GATE_HOURS={ODDS_TIME_GATE_HOURS} | ODDS_SIM_THRESHOLD={ODDS_SIM_THRESHOLD}")

    if not API_FOOTBALL_KEY:
        log("❌ FOOTBALL_API_KEY is missing. Aborting fixture fetch.")
        return []

    all_fixtures = []
    for lg_name, lg_id in LEAGUES.items():
        all_fixtures.extend(fetch_fixtures(lg_id, lg_name))

    log(f"Total raw fixtures collected: {len(all_fixtures)}")

    # odds cache per league
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

    # First pass: compute lambdas + probs; collect lambda_total per league for z-scoring
    temp_rows = []
    lam_totals_by_league = {}

    matched_cnt = 0
    stats_fallback_cnt = 0
    stats_missing_cnt = 0

    for fx in all_fixtures:
        league_id = fx["league_id"]
        league_name = fx["league_name"]
        lb = fetch_league_baselines(league_id)

        hs = get_team_stats(fx["home_id"], league_id)
        aw = get_team_stats(fx["away_id"], league_id)

        if not hs.get("ok") or not aw.get("ok"):
            stats_missing_cnt += 1
        if hs.get("reason") == "fallback_prev_season" or aw.get("reason") == "fallback_prev_season":
            stats_fallback_cnt += 1

        lam_h, lam_a = compute_expected_goals(hs, aw, lb)
        probs = compute_probabilities_dc(lam_h, lam_a)

        offered = {}
        match_debug = {"matched": False, "reason": "odds_off"}
        if USE_ODDS_API:
            league_cache = odds_cache_by_league.get(league_name, [])
            offered, match_debug = pick_best_odds_for_fixture(fx, league_cache)
            if match_debug.get("matched"):
                matched_cnt += 1

        dt = fx["commence_utc"]
        lam_sum = lam_h + lam_a

        lam_totals_by_league.setdefault(league_name, []).append(lam_sum)

        temp_rows.append({
            "fx": fx,
            "lb": lb,
            "hs": hs,
            "aw": aw,
            "lam_h": lam_h,
            "lam_a": lam_a,
            "lam_sum": lam_sum,
            "probs": probs,
            "offered": offered,
            "match_debug": match_debug,
            "dt": dt,
        })

    # league z-score params
    league_z = {}
    for lg, arr in lam_totals_by_league.items():
        if not arr:
            league_z[lg] = (0.0, 1.0)
            continue
        m = mean(arr)
        sd = pstdev(arr) if len(arr) >= 2 else 0.35
        if sd < 0.15:
            sd = 0.15
        league_z[lg] = (m, sd)

    # Second pass: build output with scores + value_pct
    for row in temp_rows:
        fx = row["fx"]
        league_name = fx["league_name"]
        probs = row["probs"]
        offered = row["offered"]
        match_debug = row["match_debug"]
        dt = row["dt"]

        lam_h = row["lam_h"]
        lam_a = row["lam_a"]
        lam_sum = row["lam_sum"]

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

        off_1 = offered.get("home")
        off_x = offered.get("draw")
        off_2 = offered.get("away")
        off_o = offered.get("over")
        off_u = offered.get("under")

        m, sd = league_z.get(league_name, (0.0, 1.0))
        z_total = (lam_sum - m) / sd if sd > 0 else 0.0

        fixtures_out.append(
            {
                "fixture_id": fx["id"],
                "date": dt.date().isoformat(),
                "time": dt.strftime("%H:%M"),
                "league_id": fx["league_id"],
                "league": league_name,
                "home": fx["home"],
                "away": fx["away"],
                "model": "bombay_production_v3_7_dc",

                "lambda_home": round(lam_h, 3),
                "lambda_away": round(lam_a, 3),
                "lambda_total": round(lam_sum, 3),
                "z_total": round(z_total, 3),

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

                "offered_1": off_1,
                "offered_x": off_x,
                "offered_2": off_2,
                "offered_over_2_5": off_o,
                "offered_under_2_5": off_u,

                "value_pct_1": value_pct(off_1, fair_1),
                "value_pct_x": value_pct(off_x, fair_x),
                "value_pct_2": value_pct(off_2, fair_2),
                "value_pct_over": value_pct(off_o, fair_over),
                "value_pct_under": value_pct(off_u, fair_under),

                "score_draw": score_draw(p_draw, lam_h, lam_a),
                "score_over": score_over(p_over, z_total),
                "score_under": score_under(p_under, z_total),

                "stats_home": {"ok": row["hs"].get("ok"), "reason": row["hs"].get("reason"), "season": row["hs"].get("season")},
                "stats_away": {"ok": row["aw"].get("ok"), "reason": row["aw"].get("reason"), "season": row["aw"].get("season")},

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

    log(f"✅ Thursday v3.7 READY. Fixtures: {len(fixtures)}")

if __name__ == "__main__":
    main()
