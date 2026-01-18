import os
import json
import math
import requests
import datetime
import unicodedata
import re
from dateutil import parser

# ============================================================
#  BOMBAY THURSDAY FULL ENGINE v3 — PRODUCTION (LOCKED LEAGUES)
#
#  Writes: logs/thursday_report_v3.json
#
#  IMPORTANT:
#   - League allowlist is HARD-LOCKED (no extra leagues can appear).
#   - Adds additive helper fields for Friday (no schema breaks):
#       total_lambda, abs_lambda_gap
#       suitability_home/away/draw/over/under (0..1)
#       season_used, engine_leagues
#
#  Also keeps EV fields:
#       ev_1 / ev_x / ev_2 / ev_over / ev_under
# ============================================================

API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY} if API_FOOTBALL_KEY else {}

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

# If env is wrong (e.g. 2026 instead of 2025/26), we will auto-fallback.
FOOTBALL_SEASON_ENV = os.getenv("FOOTBALL_SEASON", "").strip()
USE_ODDS_API = os.getenv("USE_ODDS_API", "true").lower() == "true"
WINDOW_HOURS = int(os.getenv("WINDOW_HOURS", "72"))

# Odds matching controls
ODDS_TIME_GATE_HOURS = float(os.getenv("ODDS_TIME_GATE_HOURS", "6"))
ODDS_TIME_SOFT_HOURS = float(os.getenv("ODDS_TIME_SOFT_HOURS", "10"))
ODDS_SIM_THRESHOLD = float(os.getenv("ODDS_SIM_THRESHOLD", "0.62"))

# Model controls
SHRINKAGE_K = float(os.getenv("SHRINKAGE_K", "8"))
DC_RHO = float(os.getenv("DC_RHO", "-0.13"))

LAMBDA_MIN = float(os.getenv("LAMBDA_MIN", "0.40"))
LAMBDA_MAX_HOME = float(os.getenv("LAMBDA_MAX_HOME", "3.00"))
LAMBDA_MAX_AWAY = float(os.getenv("LAMBDA_MAX_AWAY", "3.00"))

# Stabilization caps
CAP_OVER = float(os.getenv("CAP_OVER", "0.70"))
CAP_UNDER = float(os.getenv("CAP_UNDER", "0.75"))
CAP_DRAW_MAX = float(os.getenv("CAP_DRAW_MAX", "0.32"))
CAP_DRAW_MIN = float(os.getenv("CAP_DRAW_MIN", "0.20"))
CAP_OUTCOME_MIN = float(os.getenv("CAP_OUTCOME_MIN", "0.05"))

FAIR_SNAP_RATIO = float(os.getenv("FAIR_SNAP_RATIO", "1.35"))

LOW_TEMPO_LEAGUES = set([x.strip() for x in os.getenv("LOW_TEMPO_LEAGUES", "Serie B,Ligue 2").split(",") if x.strip()])
LOW_TEMPO_OVER_CAP = float(os.getenv("LOW_TEMPO_OVER_CAP", "0.65"))

# Over blockers
OVER_BLOCK_LTOTAL = float(os.getenv("OVER_BLOCK_LTOTAL", "2.4"))
OVER_BLOCK_LMIN = float(os.getenv("OVER_BLOCK_LMIN", "0.9"))
# Under blockers
UNDER_BLOCK_LTOTAL = float(os.getenv("UNDER_BLOCK_LTOTAL", "3.0"))
UNDER_BLOCK_BOTH_GT = float(os.getenv("UNDER_BLOCK_BOTH_GT", "1.4"))

# Draw normalization
DRAW_LAMBDA_GAP_MAX = float(os.getenv("DRAW_LAMBDA_GAP_MAX", "0.40"))
DRAW_IF_GAP_CAP = float(os.getenv("DRAW_IF_GAP_CAP", "0.26"))
DRAW_LEAGUE_PLUS = float(os.getenv("DRAW_LEAGUE_PLUS", "0.03"))

# Baselines
USE_DYNAMIC_LEAGUE_BASELINES = os.getenv("USE_DYNAMIC_LEAGUE_BASELINES", "false").lower() == "true"
BASELINES_LAST_N = int(os.getenv("BASELINES_LAST_N", "180"))

# Selection hints (no model distortion)
TIGHT_DRAW_THRESHOLD = float(os.getenv("TIGHT_DRAW_THRESHOLD", "0.28"))
TIGHT_LTOTAL_THRESHOLD = float(os.getenv("TIGHT_LTOTAL_THRESHOLD", "2.55"))

OVER_TIGHT_PENALTY_PTS = float(os.getenv("OVER_TIGHT_PENALTY_PTS", "12.0"))
OVER_LOW_TEMPO_EXTRA_PENALTY_PTS = float(os.getenv("OVER_LOW_TEMPO_EXTRA_PENALTY_PTS", "6.0"))

# Candidate hints (stored only in flags)
CORE_ODDS_MIN = float(os.getenv("CORE_ODDS_MIN", "1.50"))
CORE_ODDS_MAX = float(os.getenv("CORE_ODDS_MAX", "1.90"))
CORE_VALUE_MIN_PCT = float(os.getenv("CORE_VALUE_MIN_PCT", "10.0"))

FUN_ODDS_MIN = float(os.getenv("FUN_ODDS_MIN", "2.10"))
FUN_ODDS_MAX = float(os.getenv("FUN_ODDS_MAX", "4.00"))
FUN_VALUE_MIN_PCT = float(os.getenv("FUN_VALUE_MIN_PCT", "5.0"))
FUN_MIN_PROB = float(os.getenv("FUN_MIN_PROB", "0.25"))

# -------------------- HARD-LOCKED LEAGUES (ONLY THESE) --------------------
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
# ------------------------------------------------------------------------

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

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

def iso_z(dt: datetime.datetime) -> str:
    dt = dt.astimezone(datetime.timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")

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

# ------------------------- API FOOTBALL: fixtures -------------------------
def fetch_fixtures(league_id: int, league_name: str, season_used: str):
    if not API_FOOTBALL_KEY:
        log("❌ Missing FOOTBALL_API_KEY – NO fixtures will be fetched!")
        return []

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"league": league_id, "season": season_used}

    try:
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=25).json()
    except Exception as e:
        log(f"⚠️ Error fetching fixtures for {league_name}: {e}")
        return []

    resp = r.get("response") or []
    if not resp:
        return []

    out = []
    now = datetime.datetime.now(datetime.timezone.utc)

    for fx in resp:
        # HARD LOCK: ignore any fixture that isn't exactly in our allowlist mapping by league_id call anyway.
        if fx.get("league", {}).get("id") != league_id:
            continue

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

    log(f"→ {league_name}: kept {len(out)} fixtures within {WINDOW_HOURS}h (season={season_used})")
    return out

# ------------------------- TEAM STATS (API-FOOTBALL) -------------------------
def _recency_weights(n: int):
    base = [1.0, 0.85, 0.72, 0.61, 0.52]
    w = base[: max(0, min(5, n))]
    s = sum(w) if w else 1.0
    return [x / s for x in w]

def fetch_team_recent_stats(team_id: int, league_id: int, season_used: str, want_home_context: bool = None):
    ck = (team_id, league_id, season_used, want_home_context)
    if ck in TEAM_STATS_CACHE:
        return TEAM_STATS_CACHE[ck]

    if not API_FOOTBALL_KEY:
        TEAM_STATS_CACHE[ck] = {}
        return TEAM_STATS_CACHE[ck]

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"team": team_id, "league": league_id, "season": season_used, "last": 10}

    try:
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=25).json()
    except Exception as e:
        log(f"⚠️ Error fetching team stats team_id={team_id}: {e}")
        TEAM_STATS_CACHE[ck] = {}
        return TEAM_STATS_CACHE[ck]

    resp_all = r.get("response") or []
    if not resp_all:
        TEAM_STATS_CACHE[ck] = {}
        return TEAM_STATS_CACHE[ck]

    resp = []
    if want_home_context is None:
        resp = resp_all
    else:
        for fx in resp_all:
            is_home = fx["teams"]["home"]["id"] == team_id
            if want_home_context and is_home:
                resp.append(fx)
            if (want_home_context is False) and (not is_home):
                resp.append(fx)
        if len(resp) < 3:
            resp = resp_all

    sample = resp[:5]
    m = len(sample)
    if m == 0:
        TEAM_STATS_CACHE[ck] = {}
        return TEAM_STATS_CACHE[ck]

    w = _recency_weights(m)
    gf = ga = 0.0
    gf_raw = ga_raw = 0

    for idx, fx in enumerate(sample):
        g_home = fx.get("goals", {}).get("home")
        g_away = fx.get("goals", {}).get("away")
        if g_home is None or g_away is None:
            continue
        g_home = int(g_home)
        g_away = int(g_away)

        is_home = fx["teams"]["home"]["id"] == team_id
        if is_home:
            gf_i, ga_i = g_home, g_away
        else:
            gf_i, ga_i = g_away, g_home

        gf_raw += gf_i
        ga_raw += ga_i
        gf += w[idx] * gf_i
        ga += w[idx] * ga_i

    stats = {
        "matches_count": m,
        "avg_goals_for": gf,
        "avg_goals_against": ga,
        "avg_goals_for_unweighted": (gf_raw / m) if m else None,
        "avg_goals_against_unweighted": (ga_raw / m) if m else None,
    }
    TEAM_STATS_CACHE[ck] = stats
    return stats

# ------------------------- LEAGUE BASELINES -------------------------
def fetch_league_baselines_static(league_id: int):
    overrides = {
        39: {"avg_goals_per_match": 2.9, "home_advantage": 0.18, "avg_draw_rate": 0.24, "avg_over25_rate": 0.58},
        40: {"avg_goals_per_match": 2.5, "home_advantage": 0.16, "avg_draw_rate": 0.28, "avg_over25_rate": 0.52},
        78: {"avg_goals_per_match": 3.1, "home_advantage": 0.17, "avg_draw_rate": 0.25, "avg_over25_rate": 0.60},
        135: {"avg_goals_per_match": 2.5, "home_advantage": 0.15, "avg_draw_rate": 0.30, "avg_over25_rate": 0.52},
        136: {"avg_goals_per_match": 2.45, "home_advantage": 0.15, "avg_draw_rate": 0.30, "avg_over25_rate": 0.50},
        140: {"avg_goals_per_match": 2.6, "home_advantage": 0.16, "avg_draw_rate": 0.27, "avg_over25_rate": 0.55},
        94: {"avg_goals_per_match": 2.55, "home_advantage": 0.15, "avg_draw_rate": 0.28, "avg_over25_rate": 0.54},
        61: {"avg_goals_per_match": 2.70, "home_advantage": 0.16, "avg_draw_rate": 0.26, "avg_over25_rate": 0.55},
        62: {"avg_goals_per_match": 2.35, "home_advantage": 0.15, "avg_draw_rate": 0.29, "avg_over25_rate": 0.49},
    }
    base = {"avg_goals_per_match": 2.6, "home_advantage": 0.16, "avg_draw_rate": 0.26, "avg_over25_rate": 0.55}
    if league_id in overrides:
        base.update(overrides[league_id])
    return base

def fetch_league_baselines_dynamic(league_id: int, season_used: str):
    if not API_FOOTBALL_KEY:
        return fetch_league_baselines_static(league_id)

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"league": league_id, "season": season_used, "status": "FT", "last": BASELINES_LAST_N}

    try:
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=25).json()
    except Exception:
        return fetch_league_baselines_static(league_id)

    resp = r.get("response") or []
    if not resp or len(resp) < 40:
        return fetch_league_baselines_static(league_id)

    total = 0
    home_goals = 0
    away_goals = 0
    draws = 0
    overs = 0

    for fx in resp:
        hg = fx.get("goals", {}).get("home")
        ag = fx.get("goals", {}).get("away")
        if hg is None or ag is None:
            continue
        total += 1
        home_goals += int(hg)
        away_goals += int(ag)
        if int(hg) == int(ag):
            draws += 1
        if (int(hg) + int(ag)) >= 3:
            overs += 1

    if total < 30:
        return fetch_league_baselines_static(league_id)

    avg_goals_per_match = (home_goals + away_goals) / total
    hgpg = home_goals / total
    agpg = away_goals / total
    ha = (hgpg - agpg) / avg_goals_per_match if avg_goals_per_match > 0 else 0.16
    ha = _clamp(ha, 0.05, 0.25)

    return {
        "avg_goals_per_match": float(round(avg_goals_per_match, 3)),
        "home_advantage": float(round(ha, 3)),
        "avg_draw_rate": float(round(draws / total, 3)),
        "avg_over25_rate": float(round(overs / total, 3)),
    }

def fetch_league_baselines(league_id: int, season_used: str):
    return fetch_league_baselines_dynamic(league_id, season_used) if USE_DYNAMIC_LEAGUE_BASELINES else fetch_league_baselines_static(league_id)

# ------------------------- POISSON -------------------------
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def compute_expected_goals(home_stats: dict, away_stats: dict, league_baseline: dict):
    avg_match = safe_float(league_baseline.get("avg_goals_per_match"), 2.6) or 2.6
    league_avg_team = max(0.65, avg_match / 2.0)

    home_adv = safe_float(league_baseline.get("home_advantage"), 0.16) or 0.16
    home_adv_factor = 1.0 + home_adv

    def get_rates(stats):
        n = int(stats.get("matches_count") or 0)
        gf_mle = safe_float(stats.get("avg_goals_for"), None)
        ga_mle = safe_float(stats.get("avg_goals_against"), None)
        if gf_mle is None:
            gf_mle = league_avg_team
        if ga_mle is None:
            ga_mle = league_avg_team

        k = max(0.0, SHRINKAGE_K)
        denom = (n + k) if (n + k) > 0 else 1.0

        gf_shrunk = (n * gf_mle + k * league_avg_team) / denom
        ga_shrunk = (n * ga_mle + k * league_avg_team) / denom

        att = _clamp(gf_shrunk / league_avg_team, 0.55, 1.85)
        dff = _clamp(ga_shrunk / league_avg_team, 0.55, 1.85)

        return {"n": n, "att": att, "def": dff}

    h = get_rates(home_stats or {})
    a = get_rates(away_stats or {})

    lam_h = league_avg_team * h["att"] * a["def"] * home_adv_factor
    lam_a = league_avg_team * a["att"] * h["def"]

    lam_h = _clamp(lam_h, LAMBDA_MIN, LAMBDA_MAX_HOME)
    lam_a = _clamp(lam_a, LAMBDA_MIN, LAMBDA_MAX_AWAY)
    return lam_h, lam_a

def compute_probabilities(lambda_home: float, lambda_away: float):
    max_goals = 6
    pmf_h = [poisson_pmf(k, lambda_home) for k in range(max_goals)]
    pmf_a = [poisson_pmf(k, lambda_away) for k in range(max_goals)]
    tail_h = max(0.0, 1.0 - sum(pmf_h))
    tail_a = max(0.0, 1.0 - sum(pmf_a))
    pmf_h.append(tail_h)
    pmf_a.append(tail_a)

    mat = [[pmf_h[i] * pmf_a[j] for j in range(max_goals + 1)] for i in range(max_goals + 1)]

    rho = DC_RHO
    lamH = lambda_home
    lamA = lambda_away

    def tau(x, y):
        if x == 0 and y == 0:
            return 1.0 - (lamH * lamA * rho)
        if x == 1 and y == 0:
            return 1.0 + (lamA * rho)
        if x == 0 and y == 1:
            return 1.0 + (lamH * rho)
        if x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    for (x, y) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        mat[x][y] *= tau(x, y)

    s = sum(sum(row) for row in mat)
    if s <= 0:
        return {"home_prob": 0.40, "draw_prob": 0.26, "away_prob": 0.34, "over_2_5_prob": 0.55, "under_2_5_prob": 0.45}

    mat = [[v / s for v in row] for row in mat]

    ph = pd = pa = 0.0
    po = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = mat[i][j]
            if i > j:
                ph += p
            elif i == j:
                pd += p
            else:
                pa += p
            if (i + j) >= 3:
                po += p

    po = _clamp(po, 1e-6, 1.0 - 1e-6)
    pu = 1.0 - po
    return {"home_prob": ph, "draw_prob": pd, "away_prob": pa, "over_2_5_prob": po, "under_2_5_prob": pu}

# ------------------------- ODDS API -------------------------
def _odds_request(sport_key: str, params: dict):
    url = f"{ODDS_BASE_URL}/{sport_key}/odds"
    try:
        res = requests.get(url, params=params, timeout=25)
        if res.status_code != 200:
            return []
        return res.json() or []
    except Exception:
        return []

def fetch_odds_for_league(league_name: str, window_from: datetime.datetime, window_to: datetime.datetime):
    if not USE_ODDS_API:
        return []
    if not ODDS_API_KEY:
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

    data = _odds_request(sport_key, params1)
    if data:
        return data
    return _odds_request(sport_key, base_params)

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
                    if "over" in name:
                        best_over = max(best_over or 0.0, price)
                    elif "under" in name:
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
            if diff_h > ODDS_TIME_GATE_HOURS:
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
        return {}, {"matched": False, "reason": "no_candidates"}
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

# ------------------------- VALUE% + EV -------------------------
def value_pct(offered, fair):
    if offered is None or fair is None:
        return None
    try:
        if offered <= 0 or fair <= 0:
            return None
        return round((offered / fair - 1.0) * 100.0, 1)
    except Exception:
        return None

def ev_per_unit(prob, offered):
    prob = safe_float(prob, None)
    offered = safe_float(offered, None)
    if prob is None or offered is None:
        return None
    if prob <= 0 or offered <= 1.0:
        return None
    return round((prob * offered) - 1.0, 4)

# ------------------------- STABILIZATION -------------------------
def _renorm_1x2(ph, pd, pa):
    ph = max(CAP_OUTCOME_MIN, ph)
    pd = max(CAP_OUTCOME_MIN, pd)
    pa = max(CAP_OUTCOME_MIN, pa)
    pd = _clamp(pd, CAP_DRAW_MIN, CAP_DRAW_MAX)
    s = ph + pd + pa
    if s <= 0:
        return 0.40, 0.26, 0.34
    return ph / s, pd / s, pa / s

def _apply_favorite_protection(ph, pd, pa, off1, off2):
    if off1 is not None:
        if off1 <= 1.40: ph = max(ph, 0.62)
        elif off1 <= 1.60: ph = max(ph, 0.55)
        elif off1 <= 1.80: ph = max(ph, 0.50)

    if off2 is not None:
        if off2 <= 1.40: pa = max(pa, 0.62)
        elif off2 <= 1.60: pa = max(pa, 0.55)
        elif off2 <= 1.80: pa = max(pa, 0.50)

    return _renorm_1x2(ph, pd, pa)

def stabilize_probs(league_name, league_baseline, lam_h, lam_a, ph, pd, pa, po, pu, off1, offx, off2, offo, offu):
    ph = _clamp(ph, CAP_OUTCOME_MIN, 1.0)
    pa = _clamp(pa, CAP_OUTCOME_MIN, 1.0)
    pd = _clamp(pd, CAP_DRAW_MIN, CAP_DRAW_MAX)

    gap = abs((lam_h or 0) - (lam_a or 0))
    if gap > DRAW_LAMBDA_GAP_MAX:
        pd = min(pd, DRAW_IF_GAP_CAP)

    league_draw = safe_float((league_baseline or {}).get("avg_draw_rate"), None)
    if league_draw is not None:
        pd = min(pd, league_draw + DRAW_LEAGUE_PLUS, CAP_DRAW_MAX)
        pd = max(pd, CAP_DRAW_MIN)

    ph, pd, pa = _renorm_1x2(ph, pd, pa)
    ph, pd, pa = _apply_favorite_protection(ph, pd, pa, off1, off2)

    ltot = (lam_h or 0) + (lam_a or 0)

    if ltot < OVER_BLOCK_LTOTAL or (lam_h or 0) < OVER_BLOCK_LMIN or (lam_a or 0) < OVER_BLOCK_LMIN:
        po = min(po, min(CAP_OVER, 0.66))
    if league_name in LOW_TEMPO_LEAGUES:
        po = min(po, LOW_TEMPO_OVER_CAP)

    if ltot > UNDER_BLOCK_LTOTAL or ((lam_h or 0) > UNDER_BLOCK_BOTH_GT and (lam_a or 0) > UNDER_BLOCK_BOTH_GT):
        pu = min(pu, 0.70)

    po = _clamp(po, CAP_OUTCOME_MIN, CAP_OVER)
    pu = _clamp(pu, CAP_OUTCOME_MIN, CAP_UNDER)
    s2 = po + pu
    if s2 > 0:
        po, pu = po / s2, pu / s2

    return ph, pd, pa, po, pu

# ------------------------- CONFIDENCE + SUITABILITY -------------------------
def confidence_score(home_stats, away_stats, match_debug, lam_h, lam_a):
    n_h = int((home_stats or {}).get("matches_count") or 0)
    n_a = int((away_stats or {}).get("matches_count") or 0)
    n = min(n_h, n_a)

    score = 0.40
    score += 0.20 * _clamp(n / 5.0, 0.0, 1.0)
    if (match_debug or {}).get("matched"):
        score += 0.15

    ltot = (lam_h or 0.0) + (lam_a or 0.0)
    if ltot < 2.1: score -= 0.10
    elif ltot < 2.3: score -= 0.06
    if ltot > 3.6: score -= 0.08
    elif ltot > 3.3: score -= 0.05

    return _clamp(score, 0.05, 0.95)

def suitability_from(ev_val, conf_val):
    # 0..1, stable mapping
    ev_val = safe_float(ev_val, None)
    conf_val = safe_float(conf_val, 0.5) or 0.5
    if ev_val is None:
        return None
    base = _clamp(0.5 + (ev_val * 2.0), 0.0, 1.0)
    mult = _clamp(0.6 + 0.4 * conf_val, 0.0, 1.0)
    return round(base * mult, 3)

# ------------------------- MAIN PIPELINE -------------------------
def resolve_season_candidates(now_utc: datetime.datetime):
    # Preferred: env. If bad -> fallback to (now.year-1) then (now.year)
    candidates = []
    if FOOTBALL_SEASON_ENV:
        candidates.append(FOOTBALL_SEASON_ENV)
        try:
            y = int(FOOTBALL_SEASON_ENV)
            candidates.append(str(y - 1))
            candidates.append(str(y + 1))
        except Exception:
            pass
    candidates.append(str(now_utc.year - 1))
    candidates.append(str(now_utc.year))
    # unique keep order
    seen = set()
    out = []
    for c in candidates:
        if c and c not in seen:
            out.append(c)
            seen.add(c)
    return out[:4]

def build_fixture_blocks():
    fixtures_out = []
    now = datetime.datetime.now(datetime.timezone.utc)
    to_dt = now + datetime.timedelta(hours=WINDOW_HOURS)
    odds_from = now - datetime.timedelta(hours=8)

    if not API_FOOTBALL_KEY:
        log("❌ FOOTBALL_API_KEY is missing. Aborting.")
        return [], None

    season_candidates = resolve_season_candidates(now)

    season_used = None
    all_fixtures = []
    # Try seasons until we get something (for Jan this avoids env=2026 -> 0 fixtures)
    for s in season_candidates:
        tmp = []
        for lg_name, lg_id in LEAGUES.items():
            tmp.extend(fetch_fixtures(lg_id, lg_name, s))
        if len(tmp) > 0:
            season_used = s
            all_fixtures = tmp
            break

    if season_used is None:
        season_used = season_candidates[0] if season_candidates else ""
        all_fixtures = []

    log(f"Season used: {season_used} | Total fixtures collected: {len(all_fixtures)}")

    odds_cache_by_league = {}
    if USE_ODDS_API and ODDS_API_KEY:
        for lg_name in LEAGUES.keys():
            odds_events = fetch_odds_for_league(lg_name, odds_from, to_dt)
            odds_cache_by_league[lg_name] = build_events_cache(odds_events)

    for fx in all_fixtures:
        league_id = fx["league_id"]
        league_name = fx["league_name"]

        # HARD LOCK final: only allow if in our dict
        if league_name not in LEAGUES:
            continue

        league_baseline = fetch_league_baselines(league_id, season_used)

        home_stats = fetch_team_recent_stats(fx["home_id"], league_id, season_used, want_home_context=True)
        away_stats = fetch_team_recent_stats(fx["away_id"], league_id, season_used, want_home_context=False)

        lam_h, lam_a = compute_expected_goals(home_stats, away_stats, league_baseline)
        probs = compute_probabilities(lam_h, lam_a)

        offered = {}
        match_debug = {"matched": False, "reason": "odds_off"}
        if USE_ODDS_API and ODDS_API_KEY:
            league_cache = odds_cache_by_league.get(league_name, [])
            offered, match_debug = pick_best_odds_for_fixture(fx, league_cache)

        off_1 = offered.get("home")
        off_x = offered.get("draw")
        off_2 = offered.get("away")
        off_o = offered.get("over")
        off_u = offered.get("under")

        m_ph, m_pd, m_pa, m_po, m_pu = stabilize_probs(
            league_name=league_name,
            league_baseline=league_baseline,
            lam_h=lam_h,
            lam_a=lam_a,
            ph=probs["home_prob"],
            pd=probs["draw_prob"],
            pa=probs["away_prob"],
            po=probs["over_2_5_prob"],
            pu=probs["under_2_5_prob"],
            off1=off_1, offx=off_x, off2=off_2, offo=off_o, offu=off_u,
        )

        fair_1 = implied(m_ph)
        fair_x = implied(m_pd)
        fair_2 = implied(m_pa)
        fair_over = implied(m_po)
        fair_under = implied(m_pu)

        v1 = value_pct(off_1, fair_1)
        vx = value_pct(off_x, fair_x)
        v2 = value_pct(off_2, fair_2)
        vo = value_pct(off_o, fair_over)
        vu = value_pct(off_u, fair_under)

        ev1 = ev_per_unit(m_ph, off_1)
        evx = ev_per_unit(m_pd, off_x)
        ev2 = ev_per_unit(m_pa, off_2)
        evo = ev_per_unit(m_po, off_o)
        evu = ev_per_unit(m_pu, off_u)

        ltot = (lam_h or 0.0) + (lam_a or 0.0)
        abs_gap = abs((lam_h or 0.0) - (lam_a or 0.0))

        tight_game = (m_pd >= TIGHT_DRAW_THRESHOLD) or (ltot <= TIGHT_LTOTAL_THRESHOLD)
        low_tempo = league_name in LOW_TEMPO_LEAGUES

        over_penalty_pts = 0.0
        if tight_game:
            over_penalty_pts += OVER_TIGHT_PENALTY_PTS
        if low_tempo:
            over_penalty_pts += OVER_LOW_TEMPO_EXTRA_PENALTY_PTS

        selection_vo = None
        if vo is not None:
            selection_vo = round(vo - over_penalty_pts, 1)

        conf = confidence_score(home_stats, away_stats, match_debug, lam_h, lam_a)
        conf_band = "high" if conf >= 0.70 else ("mid" if conf >= 0.55 else "low")

        flags = {
            "tight_game": bool(tight_game),
            "low_tempo_league": bool(low_tempo),
            "odds_matched": bool(match_debug.get("matched")),
            "confidence": round(conf, 3),
            "confidence_band": conf_band,
            "core_1": (off_1 is not None and v1 is not None and m_ph is not None and (CORE_ODDS_MIN <= off_1 <= CORE_ODDS_MAX) and (v1 >= CORE_VALUE_MIN_PCT)),
            "core_x": (off_x is not None and vx is not None and m_pd is not None and (CORE_ODDS_MIN <= off_x <= CORE_ODDS_MAX) and (vx >= CORE_VALUE_MIN_PCT)),
            "core_2": (off_2 is not None and v2 is not None and m_pa is not None and (CORE_ODDS_MIN <= off_2 <= CORE_ODDS_MAX) and (v2 >= CORE_VALUE_MIN_PCT)),
            "core_over": (off_o is not None and vo is not None and m_po is not None and (CORE_ODDS_MIN <= off_o <= CORE_ODDS_MAX) and (vo >= CORE_VALUE_MIN_PCT)),
            "core_under": (off_u is not None and vu is not None and m_pu is not None and (CORE_ODDS_MIN <= off_u <= CORE_ODDS_MAX) and (vu >= CORE_VALUE_MIN_PCT)),
            "fun_1": (off_1 is not None and v1 is not None and m_ph is not None and (FUN_ODDS_MIN <= off_1 <= FUN_ODDS_MAX) and (v1 >= FUN_VALUE_MIN_PCT) and (m_ph >= FUN_MIN_PROB)),
            "fun_x": (off_x is not None and vx is not None and m_pd is not None and (FUN_ODDS_MIN <= off_x <= FUN_ODDS_MAX) and (vx >= FUN_VALUE_MIN_PCT) and (m_pd >= FUN_MIN_PROB)),
            "fun_2": (off_2 is not None and v2 is not None and m_pa is not None and (FUN_ODDS_MIN <= off_2 <= FUN_ODDS_MAX) and (v2 >= FUN_VALUE_MIN_PCT) and (m_pa >= FUN_MIN_PROB)),
            "fun_over": (off_o is not None and vo is not None and m_po is not None and (FUN_ODDS_MIN <= off_o <= FUN_ODDS_MAX) and (vo >= FUN_VALUE_MIN_PCT) and (m_po >= FUN_MIN_PROB)),
            "fun_under": (off_u is not None and vu is not None and m_pu is not None and (FUN_ODDS_MIN <= off_u <= FUN_ODDS_MAX) and (vu >= FUN_VALUE_MIN_PCT) and (m_pu >= FUN_MIN_PROB)),
        }

        dt = fx["commence_utc"]

        fixtures_out.append(
            {
                "fixture_id": fx["id"],
                "date": dt.date().isoformat(),
                "time": dt.strftime("%H:%M"),
                "league_id": league_id,
                "league": league_name,
                "home": fx["home"],
                "away": fx["away"],
                "model": "bombay_multiplicative_dc_v3_locked",

                "lambda_home": round(lam_h, 3),
                "lambda_away": round(lam_a, 3),

                # additive helpers
                "total_lambda": round(ltot, 3),
                "abs_lambda_gap": round(abs_gap, 3),

                # MODEL probabilities
                "home_prob": round(m_ph, 3),
                "draw_prob": round(m_pd, 3),
                "away_prob": round(m_pa, 3),
                "over_2_5_prob": round(m_po, 3),
                "under_2_5_prob": round(m_pu, 3),

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

                "value_pct_1": v1,
                "value_pct_x": vx,
                "value_pct_2": v2,
                "value_pct_over": vo,
                "value_pct_under": vu,

                "ev_1": ev1,
                "ev_x": evx,
                "ev_2": ev2,
                "ev_over": evo,
                "ev_under": evu,

                # suitability (additive)
                "suitability_home": suitability_from(ev1, conf),
                "suitability_away": suitability_from(ev2, conf),
                "suitability_draw": suitability_from(evx, conf),
                "suitability_over": suitability_from(evo, conf),
                "suitability_under": suitability_from(evu, conf),

                # aliases
                "score_draw": evx,
                "score_over": evo,

                "selection_value_pct_over": selection_vo,
                "over_value_penalty_pts": round(over_penalty_pts, 2),
                "flags": flags,
                "odds_match": match_debug,
                "league_baseline": league_baseline,
            }
        )

    return fixtures_out, season_used

def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    to_dt = now + datetime.timedelta(hours=WINDOW_HOURS)

    fixtures, season_used = build_fixture_blocks()

    out = {
        "generated_at": now.isoformat(),
        "season_used": season_used,
        "engine_leagues": list(LEAGUES.keys()),
        "window": {"from": now.date().isoformat(), "to": to_dt.date().isoformat(), "hours": WINDOW_HOURS},
        "fixtures_total": len(fixtures),
        "fixtures": fixtures,
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/thursday_report_v3.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    log(f"✅ Thursday v3 written. Season={season_used} Fixtures={len(fixtures)}")

if __name__ == "__main__":
    main()
