import os
import json
import math
import requests
import datetime
import unicodedata
import re
from dateutil import parser

# ============================================================
#  THURSDAY ENGINE v4 (Production-style λ + sane probabilities)
#
#  Goals:
#   - Fix "crazy fair odds" by fixing lambdas, not by post-clamping probs
#   - Bookmaker-ish lambda pipeline:
#       base λ from home/away attack/defense vs league home/away avgs
#       + momentum/form/rest adjustments (Δλ with caps)
#       + draw bias applied to λ (not by forcing draw_prob later)
#       + tempo/over bias applied to λ_total
#       + clamp final λ into realistic range
#   - Probabilities from Poisson 0..6 + tail
#   - Draw bias light clamp (optional) can be disabled via env
#   - TheOddsAPI: 1 call / league
#   - Odds matching: gated by time + similarity to avoid wrong matches
#
#  Output: logs/thursday_report_v3.json (backwards compatible)
# ============================================================

API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

FOOTBALL_SEASON = os.getenv("FOOTBALL_SEASON", "2025")
USE_ODDS_API = os.getenv("USE_ODDS_API", "true").lower() == "true"
WINDOW_HOURS = int(os.getenv("WINDOW_HOURS", "72"))

# --- model toggles ---
USE_TEAM_STATISTICS = os.getenv("USE_TEAM_STATISTICS", "true").lower() == "true"  # /teams/statistics
USE_FORM_ADJ = os.getenv("USE_FORM_ADJ", "true").lower() == "true"
USE_REST_ADJ = os.getenv("USE_REST_ADJ", "true").lower() == "true"
USE_DRAW_LAMBDA_BIAS = os.getenv("USE_DRAW_LAMBDA_BIAS", "true").lower() == "true"
USE_TEMPO_OVER_BIAS = os.getenv("USE_TEMPO_OVER_BIAS", "true").lower() == "true"

# draw probability clamp (OPTIONAL) - keep soft
USE_DRAW_PROB_CLAMP = os.getenv("USE_DRAW_PROB_CLAMP", "true").lower() == "true"
DRAW_PROB_MIN = float(os.getenv("DRAW_PROB_MIN", "0.20"))
DRAW_PROB_MAX = float(os.getenv("DRAW_PROB_MAX", "0.34"))
DRAW_PROB_BOOST = float(os.getenv("DRAW_PROB_BOOST", "1.10"))  # mild

# lambda clamps (core)
LAMBDA_MIN = float(os.getenv("LAMBDA_MIN", "0.30"))
LAMBDA_MAX = float(os.getenv("LAMBDA_MAX", "3.00"))

# adjustments (from your research)
W_MOM5 = float(os.getenv("W_MOM5", "0.12"))
CAP_MOM5 = float(os.getenv("CAP_MOM5", "0.30"))
W_FORM10 = float(os.getenv("W_FORM10", "0.08"))
CAP_FORM10 = float(os.getenv("CAP_FORM10", "0.20"))

REST_PENALTY_PER_DAY = float(os.getenv("REST_PENALTY_PER_DAY", "0.05"))  # day < 3
REST_CAP = float(os.getenv("REST_CAP", "0.15"))

# draw lambda biases
LEAGUE_DRAW_GATE = float(os.getenv("LEAGUE_DRAW_GATE", "0.28"))  # if league draw% >= 28%
LEAGUE_DRAW_LAMBDA_MULT = float(os.getenv("LEAGUE_DRAW_LAMBDA_MULT", "0.92"))

# tempo / over bias
# over_bias = min(0.15, 0.05*(league_over25 - 0.52))
OVER_BIAS_MAX = float(os.getenv("OVER_BIAS_MAX", "0.15"))
OVER_BIAS_K = float(os.getenv("OVER_BIAS_K", "0.05"))
OVER_BIAS_BASE = float(os.getenv("OVER_BIAS_BASE", "0.52"))

# Poisson truncation
POI_MAX = int(os.getenv("POI_MAX", "6"))

# Odds matching controls
ODDS_TIME_GATE_HOURS = float(os.getenv("ODDS_TIME_GATE_HOURS", "6"))
ODDS_TIME_SOFT_HOURS = float(os.getenv("ODDS_TIME_SOFT_HOURS", "10"))
ODDS_SIM_THRESHOLD = float(os.getenv("ODDS_SIM_THRESHOLD", "0.62"))

# Stats controls
MIN_STATS_MATCHES = int(os.getenv("MIN_STATS_MATCHES", "6"))  # if too small -> fallback prev season
FORM_LAST5 = int(os.getenv("FORM_LAST5", "5"))
FORM_LAST10 = int(os.getenv("FORM_LAST10", "10"))

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
TEAM_FORM_CACHE = {}

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

# ------------------------- API HELPERS -------------------------
def _football_get(path: str, params: dict):
    if not API_FOOTBALL_KEY:
        return None
    url = f"{API_FOOTBALL_BASE}/{path}"
    try:
        res = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=25)
        if res.status_code != 200:
            return None
        return res.json()
    except Exception:
        return None

# ------------------------- FIXTURES -------------------------
def fetch_fixtures(league_id: int, league_name: str):
    if not API_FOOTBALL_KEY:
        log("⚠️ Missing FOOTBALL_API_KEY – NO fixtures will be fetched!")
        return []

    r = _football_get("fixtures", {"league": league_id, "season": FOOTBALL_SEASON})
    resp = (r or {}).get("response") or []
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
    # Keep your overrides (stable) + derive home/away split
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

    # derive split: typical home share ~55%
    g = safe_float(base.get("avg_goals_per_match"), 2.6) or 2.6
    base["avg_home_goals"] = round(g * 0.55, 4)
    base["avg_away_goals"] = round(g * 0.45, 4)
    return base

# ------------------------- TEAM STATS (/teams/statistics + fallback) -------------------------
def fetch_team_statistics(team_id: int, league_id: int, season: str):
    r = _football_get("teams/statistics", {"team": team_id, "league": league_id, "season": season})
    if not r or not r.get("response"):
        return None

    resp = r["response"]
    played = resp.get("fixtures", {}).get("played", {})
    goals_for = resp.get("goals", {}).get("for", {}).get("average", {})
    goals_against = resp.get("goals", {}).get("against", {}).get("average", {})

    ph = int(safe_float(played.get("home"), 0.0) or 0.0)
    pa = int(safe_float(played.get("away"), 0.0) or 0.0)

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

    used = None
    used_reason = "current_season"
    if USE_TEAM_STATISTICS:
        cur = fetch_team_statistics(team_id, league_id, FOOTBALL_SEASON)
        sample = 0
        if cur:
            sample = (cur.get("played_home", 0) or 0) + (cur.get("played_away", 0) or 0)
        if (not cur) or sample < MIN_STATS_MATCHES:
            try:
                prev_season = str(int(FOOTBALL_SEASON) - 1)
            except Exception:
                prev_season = None
            if prev_season:
                prev = fetch_team_statistics(team_id, league_id, prev_season)
                if prev:
                    used = prev
                    used_reason = "fallback_prev_season"
        else:
            used = cur

    if not used:
        # fallback minimal
        TEAM_STATS_CACHE[ck] = {"ok": False, "reason": "no_stats"}
        return TEAM_STATS_CACHE[ck]

    TEAM_STATS_CACHE[ck] = {"ok": True, "reason": used_reason, **used}
    return TEAM_STATS_CACHE[ck]

# ------------------------- TEAM FORM (last N fixtures) -------------------------
def fetch_team_last_fixtures(team_id: int, league_id: int, last_n: int):
    ck = (team_id, league_id, last_n, FOOTBALL_SEASON)
    if ck in TEAM_FORM_CACHE:
        return TEAM_FORM_CACHE[ck]

    r = _football_get("fixtures", {"team": team_id, "league": league_id, "season": FOOTBALL_SEASON, "last": last_n})
    resp = (r or {}).get("response") or []
    TEAM_FORM_CACHE[ck] = resp
    return resp

def _form_factor_from_fixtures(team_id: int, fixtures: list):
    """
    Returns factor around 1.0:
      wins -> >1, losses -> <1
    weighted decay like your spec (simple)
    """
    if not fixtures:
        return 1.0

    weights = []
    vals = []
    # newest first in API usually; still stable
    for i, fx in enumerate(fixtures[:FORM_LAST10], start=1):
        w = max(0.3, 1.0 - 0.05 * i)  # decay
        weights.append(w)

        # result as 1.10 / 1.00 / 0.90
        home_id = fx["teams"]["home"]["id"]
        away_id = fx["teams"]["away"]["id"]
        g_home = fx["goals"]["home"] or 0
        g_away = fx["goals"]["away"] or 0

        is_home = (home_id == team_id)
        gf = g_home if is_home else g_away
        ga = g_away if is_home else g_home

        if gf > ga:
            v = 1.10
        elif gf == ga:
            v = 1.00
        else:
            v = 0.90
        vals.append(v)

    sw = sum(weights) if weights else 1.0
    avg = sum(v * w for v, w in zip(vals, weights)) / sw
    return float(avg)

def _rest_days(team_id: int, fixtures: list):
    """
    Days since last match (UTC).
    """
    if not fixtures:
        return None
    try:
        dt = parser.isoparse(fixtures[0]["fixture"]["date"]).astimezone(datetime.timezone.utc)
        now = datetime.datetime.now(datetime.timezone.utc)
        days = (now - dt).total_seconds() / 86400.0
        return max(0.0, float(days))
    except Exception:
        return None

# ------------------------- LAMBDA MODEL (bookmaker-ish) -------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def compute_expected_goals(home_stats: dict, away_stats: dict, lb: dict, home_form: dict, away_form: dict):
    """
    Base:
      atk_home = (home gf_home_pg) / league_avg_home_goals
      def_away = (away ga_away_pg) / league_avg_home_goals   (away conceded correlates to home scored)
      λ_home = atk_home * def_away * league_avg_home_goals
    symmetric:
      atk_away = (away gf_away_pg) / league_avg_away_goals
      def_home = (home ga_home_pg) / league_avg_away_goals   (home conceded correlates to away scored)
      λ_away = atk_away * def_home * league_avg_away_goals
    Adjustments:
      Δλ from momentum/form/rest, capped
      draw bias applied to λ (both down a bit)
      tempo/over bias applied to λ_total (both up)
    Final clamp.
    """
    avg_home = safe_float(lb.get("avg_home_goals"), 1.4) or 1.4
    avg_away = safe_float(lb.get("avg_away_goals"), 1.2) or 1.2
    lg_draw = safe_float(lb.get("avg_draw_rate"), 0.26) or 0.26
    lg_over = safe_float(lb.get("avg_over25_rate"), 0.55) or 0.55

    # pull stats with shrinkage
    h_gf_home = safe_float(home_stats.get("gf_home_pg"), None)
    h_ga_home = safe_float(home_stats.get("ga_home_pg"), None)
    a_gf_away = safe_float(away_stats.get("gf_away_pg"), None)
    a_ga_away = safe_float(away_stats.get("ga_away_pg"), None)

    # shrinkage to league avgs
    if h_gf_home is None: h_gf_home = avg_home
    if a_ga_away is None: a_ga_away = avg_home
    if a_gf_away is None: a_gf_away = avg_away
    if h_ga_home is None: h_ga_home = avg_away

    atk_home = h_gf_home / max(0.3, avg_home)
    def_away = a_ga_away / max(0.3, avg_home)
    lam_h = atk_home * def_away * avg_home

    atk_away = a_gf_away / max(0.3, avg_away)
    def_home = h_ga_home / max(0.3, avg_away)
    lam_a = atk_away * def_home * avg_away

    # --- Δλ adjustments (form/momentum/rest) ---
    if USE_FORM_ADJ:
        # momentum last 5
        mom_h = safe_float(home_form.get("mom5_factor"), 1.0) or 1.0
        mom_a = safe_float(away_form.get("mom5_factor"), 1.0) or 1.0
        d_h = clamp(W_MOM5 * (mom_h - 1.0), -CAP_MOM5, CAP_MOM5)
        d_a = clamp(W_MOM5 * (mom_a - 1.0), -CAP_MOM5, CAP_MOM5)
        lam_h += d_h
        lam_a += d_a

        # form last 10
        f10_h = safe_float(home_form.get("form10_factor"), 1.0) or 1.0
        f10_a = safe_float(away_form.get("form10_factor"), 1.0) or 1.0
        d2_h = clamp(W_FORM10 * (f10_h - 1.0), -CAP_FORM10, CAP_FORM10)
        d2_a = clamp(W_FORM10 * (f10_a - 1.0), -CAP_FORM10, CAP_FORM10)
        lam_h += d2_h
        lam_a += d2_a

    if USE_REST_ADJ:
        rd_h = home_form.get("rest_days")
        rd_a = away_form.get("rest_days")

        # penalty if rest < 3 days
        if rd_h is not None and rd_h < 3.0:
            pen = REST_PENALTY_PER_DAY * max(0.0, 3.0 - rd_h)
            lam_h -= min(REST_CAP, pen)
        if rd_a is not None and rd_a < 3.0:
            pen = REST_PENALTY_PER_DAY * max(0.0, 3.0 - rd_a)
            lam_a -= min(REST_CAP, pen)

    # draw bias -> apply to lambdas (NOT draw_prob later)
    if USE_DRAW_LAMBDA_BIAS and lg_draw >= LEAGUE_DRAW_GATE:
        lam_h *= LEAGUE_DRAW_LAMBDA_MULT
        lam_a *= LEAGUE_DRAW_LAMBDA_MULT

    # tempo/over bias -> scale both lambdas
    if USE_TEMPO_OVER_BIAS:
        over_bias = min(OVER_BIAS_MAX, OVER_BIAS_K * (lg_over - OVER_BIAS_BASE))
        if over_bias > 0:
            lam_h *= (1.0 + over_bias)
            lam_a *= (1.0 + over_bias)

    # final clamp
    lam_h = clamp(lam_h, LAMBDA_MIN, LAMBDA_MAX)
    lam_a = clamp(lam_a, LAMBDA_MIN, LAMBDA_MAX)
    return float(lam_h), float(lam_a)

# ------------------------- POISSON PROBS (0..6 + tail) -------------------------
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def poisson_vec_with_tail(lam: float, kmax: int):
    # 0..kmax plus tail at index kmax+1
    probs = [poisson_pmf(k, lam) for k in range(kmax + 1)]
    s = sum(probs)
    tail = max(0.0, 1.0 - s)
    probs.append(tail)
    # renormalize tiny drift
    t = sum(probs)
    if t > 0:
        probs = [p / t for p in probs]
    return probs

def compute_probabilities(lam_h: float, lam_a: float, lb: dict):
    ph = pd = pa = 0.0
    over = 0.0

    vh = poisson_vec_with_tail(lam_h, POI_MAX)
    va = poisson_vec_with_tail(lam_a, POI_MAX)

    # matrix with tail bucket
    for i, p_i in enumerate(vh):
        for j, p_j in enumerate(va):
            p = p_i * p_j
            if i > j:
                ph += p
            elif i == j:
                pd += p
            else:
                pa += p

            # total goals >=3
            # NOTE: tail bucket makes total goals approximate but stable
            if (i + j) >= 3:
                over += p

    # normalize 1X2
    tot = ph + pd + pa
    if tot <= 0:
        ph, pd, pa = 0.40, 0.20, 0.40
    else:
        ph, pd, pa = ph / tot, pd / tot, pa / tot

    # optional draw probability clamp (soft sanity)
    if USE_DRAW_PROB_CLAMP:
        pd2 = clamp(pd * DRAW_PROB_BOOST, DRAW_PROB_MIN, DRAW_PROB_MAX)
        rest = max(1e-12, ph + pa)
        scale = (1.0 - pd2) / rest
        ph *= scale
        pa *= scale
        pd = pd2

    # Over clamp (per your spec)
    over = clamp(over, 0.35, 0.72)
    under = 1.0 - over

    # final normalization
    s = ph + pd + pa
    if s > 0:
        ph, pd, pa = ph / s, pd / s, pa / s

    return {
        "home_prob": ph,
        "draw_prob": pd,
        "away_prob": pa,
        "over_2_5_prob": over,
        "under_2_5_prob": under,
    }

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

    fx_ht = token_set(fx["home_norm"])
    fx_at = token_set(fx["away_norm"])
    fx_time = fx.get("commence_utc")

    best = None
    best_score = -1.0
    best_swap = False
    best_diff = None

    for ev in league_events_cache:
        ct = ev["commence_time"]
        if fx_time and ct:
            diff_h = abs((ct - fx_time).total_seconds()) / 3600.0
        else:
            diff_h = None

        # HARD TIME GATE
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

# ------------------------- VALUE% helper -------------------------
def value_pct(offered, fair):
    if offered is None or fair is None:
        return None
    try:
        if offered <= 0 or fair <= 0:
            return None
        return round((offered / fair - 1.0) * 100.0, 1)
    except Exception:
        return None

# ------------------------- SCORES (simple, stable 1..10) -------------------------
def score_1_10(p: float) -> float:
    s = round((p or 0.0) * 10.0, 1)
    if s < 1.0: s = 1.0
    if s > 10.0: s = 10.0
    return s

# ------------------------- MAIN PIPELINE -------------------------
def build_fixture_blocks():
    fixtures_out = []
    now = datetime.datetime.now(datetime.timezone.utc)
    to_dt = now + datetime.timedelta(hours=WINDOW_HOURS)

    odds_from = now - datetime.timedelta(hours=8)  # widened back
    log(f"Using FOOTBALL_SEASON={FOOTBALL_SEASON}")
    log(f"Window: next {WINDOW_HOURS} hours")
    log(f"USE_ODDS_API={USE_ODDS_API}")
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

        # form
        home_last5 = fetch_team_last_fixtures(fx["home_id"], league_id, FORM_LAST5) if USE_FORM_ADJ or USE_REST_ADJ else []
        away_last5 = fetch_team_last_fixtures(fx["away_id"], league_id, FORM_LAST5) if USE_FORM_ADJ or USE_REST_ADJ else []
        home_last10 = fetch_team_last_fixtures(fx["home_id"], league_id, FORM_LAST10) if USE_FORM_ADJ else []
        away_last10 = fetch_team_last_fixtures(fx["away_id"], league_id, FORM_LAST10) if USE_FORM_ADJ else []

        home_form = {
            "mom5_factor": _form_factor_from_fixtures(fx["home_id"], home_last5),
            "form10_factor": _form_factor_from_fixtures(fx["home_id"], home_last10),
            "rest_days": _rest_days(fx["home_id"], home_last5) if USE_REST_ADJ else None,
        }
        away_form = {
            "mom5_factor": _form_factor_from_fixtures(fx["away_id"], away_last5),
            "form10_factor": _form_factor_from_fixtures(fx["away_id"], away_last10),
            "rest_days": _rest_days(fx["away_id"], away_last5) if USE_REST_ADJ else None,
        }

        lam_h, lam_a = compute_expected_goals(hs, aw, lb, home_form, away_form)
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
        off_1 = offered.get("home")
        off_x = offered.get("draw")
        off_2 = offered.get("away")
        off_o = offered.get("over")
        off_u = offered.get("under")

        fixtures_out.append(
            {
                "fixture_id": fx["id"],
                "date": dt.date().isoformat(),
                "time": dt.strftime("%H:%M"),
                "league_id": league_id,
                "league": league_name,
                "home": fx["home"],
                "away": fx["away"],
                "model": "bombay_balanced_v4",

                "lambda_home": round(lam_h, 3),
                "lambda_away": round(lam_a, 3),

                # probs
                "home_prob": round(p_home, 3),
                "draw_prob": round(p_draw, 3),
                "away_prob": round(p_away, 3),
                "over_2_5_prob": round(p_over, 3),
                "under_2_5_prob": round(p_under, 3),

                # fair
                "fair_1": fair_1,
                "fair_x": fair_x,
                "fair_2": fair_2,
                "fair_over_2_5": fair_over,
                "fair_under_2_5": fair_under,

                # offered
                "offered_1": off_1,
                "offered_x": off_x,
                "offered_2": off_2,
                "offered_over_2_5": off_o,
                "offered_under_2_5": off_u,

                # value pct (ONLY shown if present; helpful for UI)
                "value_pct_1": value_pct(off_1, fair_1),
                "value_pct_x": value_pct(off_x, fair_x),
                "value_pct_2": value_pct(off_2, fair_2),
                "value_pct_over": value_pct(off_o, fair_over),
                "value_pct_under": value_pct(off_u, fair_under),

                # scores (simple)
                "score_draw": score_1_10(p_draw),
                "score_over": score_1_10(p_over),
                "score_under": score_1_10(p_under),

                # debug meta (doesn't affect presenter rules)
                "stats_home": {"ok": hs.get("ok"), "reason": hs.get("reason"), "season": hs.get("season")},
                "stats_away": {"ok": aw.get("ok"), "reason": aw.get("reason"), "season": aw.get("season")},
                "form_home": {"mom5": round(home_form["mom5_factor"], 3), "form10": round(home_form["form10_factor"], 3), "rest_days": None if home_form["rest_days"] is None else round(home_form["rest_days"], 2)},
                "form_away": {"mom5": round(away_form["mom5_factor"], 3), "form10": round(away_form["form10_factor"], 3), "rest_days": None if away_form["rest_days"] is None else round(away_form["rest_days"], 2)},
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

    log(f"✅ Thursday v4 READY. Fixtures: {len(fixtures)}")

if __name__ == "__main__":
    main()
