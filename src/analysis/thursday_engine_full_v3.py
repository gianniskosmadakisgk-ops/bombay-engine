import os
import json
import math
import requests
import datetime
import unicodedata
import re
from dateutil import parser

# ============================================================
#  BOMBAY THURSDAY ENGINE v3 (STABILIZED) + MIS CUT (DEFAULT ON)
#  - Fetch fixtures (API-FOOTBALL) within WINDOW_HOURS
#  - Base model: multiplicative Poisson + shrinkage + Dixon-Coles
#  - Odds: TheOddsAPI (1 call per league) + robust matching
#  - STABILIZATION LAYER:
#      * Hard probability caps
#      * Favorite protection (market-driven floor)
#      * Fair odds deviation guard (snap-to-market within 1.35x)
#      * Over/Under blockers + low-tempo caps
#      * Draw normalization
#
#  - MIS LAYER (DEFAULT ON):
#      * scores odds matching quality + market availability
#      * assigns status: reject / fun_only / core_allowed / elite
#      * by default: CUT reject fixtures from final output
#
#  - Output: logs/thursday_report_v3.json
# ============================================================

API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

FOOTBALL_SEASON = os.getenv("FOOTBALL_SEASON", "2025")
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

# --- Stabilization ---
CAP_OVER = float(os.getenv("CAP_OVER", "0.70"))
CAP_UNDER = float(os.getenv("CAP_UNDER", "0.75"))
CAP_DRAW_MAX = float(os.getenv("CAP_DRAW_MAX", "0.32"))
CAP_DRAW_MIN = float(os.getenv("CAP_DRAW_MIN", "0.20"))
CAP_OUTCOME_MIN = float(os.getenv("CAP_OUTCOME_MIN", "0.05"))

FAIR_SNAP_RATIO = float(os.getenv("FAIR_SNAP_RATIO", "1.35"))

LOW_TEMPO_LEAGUES = set(
    [x.strip() for x in os.getenv("LOW_TEMPO_LEAGUES", "Serie B,Ligue 2").split(",") if x.strip()]
)
LOW_TEMPO_OVER_CAP = float(os.getenv("LOW_TEMPO_OVER_CAP", "0.65"))

OVER_BLOCK_LTOTAL = float(os.getenv("OVER_BLOCK_LTOTAL", "2.4"))
OVER_BLOCK_LMIN = float(os.getenv("OVER_BLOCK_LMIN", "0.9"))

UNDER_BLOCK_LTOTAL = float(os.getenv("UNDER_BLOCK_LTOTAL", "3.0"))
UNDER_BLOCK_BOTH_GT = float(os.getenv("UNDER_BLOCK_BOTH_GT", "1.4"))

DRAW_LAMBDA_GAP_MAX = float(os.getenv("DRAW_LAMBDA_GAP_MAX", "0.40"))
DRAW_IF_GAP_CAP = float(os.getenv("DRAW_IF_GAP_CAP", "0.26"))
DRAW_LEAGUE_PLUS = float(os.getenv("DRAW_LEAGUE_PLUS", "0.03"))

USE_DYNAMIC_LEAGUE_BASELINES = os.getenv("USE_DYNAMIC_LEAGUE_BASELINES", "false").lower() == "true"
BASELINES_LAST_N = int(os.getenv("BASELINES_LAST_N", "180"))

# --- MIS CONTROLS (DEFAULT ON) ---
MIS_ENABLED = os.getenv("MIS_ENABLED", "true").lower() == "true"
MIS_CUT_REJECT = os.getenv("MIS_CUT_REJECT", "true").lower() == "true"

# Quality thresholds (tune later)
MIS_MIN_MATCH_SCORE = float(os.getenv("MIS_MIN_MATCH_SCORE", "0.68"))      # similarity score
MIS_MAX_TIME_DIFF_H = float(os.getenv("MIS_MAX_TIME_DIFF_H", "3.0"))       # odds time diff
MIS_MIN_MARKETS_ANY = int(os.getenv("MIS_MIN_MARKETS_ANY", "2"))           # at least N offered markets
MIS_MIN_1X2_MARKETS = int(os.getenv("MIS_MIN_1X2_MARKETS", "2"))           # at least 2 of (1,X,2) offered
MIS_ELITE_SCORE = float(os.getenv("MIS_ELITE_SCORE", "82"))                # elite if score>=
MIS_CORE_SCORE = float(os.getenv("MIS_CORE_SCORE", "70"))                  # core_allowed if score>=
MIS_FUN_SCORE = float(os.getenv("MIS_FUN_SCORE", "58"))                    # fun_only if score>= else reject

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

# ------------------------- FIXTURES (API-FOOTBALL) -------------------------
def fetch_fixtures(league_id: int, league_name: str):
    if not API_FOOTBALL_KEY:
        log("❌ Missing FOOTBALL_API_KEY – NO fixtures will be fetched!")
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

# ------------------------- TEAM RECENT GOALS (API-FOOTBALL) -------------------------
def fetch_team_recent_stats(team_id: int, league_id: int, want_home_context: bool = None):
    ck = (team_id, league_id, want_home_context)
    if ck in TEAM_STATS_CACHE:
        return TEAM_STATS_CACHE[ck]

    if not API_FOOTBALL_KEY:
        TEAM_STATS_CACHE[ck] = {}
        return TEAM_STATS_CACHE[ck]

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"team": team_id, "league": league_id, "season": FOOTBALL_SEASON, "last": 10}

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

    gf = ga = m = 0
    for fx in resp[:5]:
        m += 1
        g_home = fx["goals"]["home"] or 0
        g_away = fx["goals"]["away"] or 0
        is_home = fx["teams"]["home"]["id"] == team_id
        if is_home:
            gf += g_home
            ga += g_away
        else:
            gf += g_away
            ga += g_home

    stats = {
        "matches_count": m,
        "avg_goals_for": (gf / m) if m else None,
        "avg_goals_against": (ga / m) if m else None,
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

def fetch_league_baselines_dynamic(league_id: int):
    if not API_FOOTBALL_KEY:
        return fetch_league_baselines_static(league_id)

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"league": league_id, "season": FOOTBALL_SEASON, "status": "FT", "last": BASELINES_LAST_N}

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

def fetch_league_baselines(league_id: int):
    return fetch_league_baselines_dynamic(league_id) if USE_DYNAMIC_LEAGUE_BASELINES else fetch_league_baselines_static(league_id)

# ------------------------- POISSON HELPERS -------------------------
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

# ------------------------- MULTIPLICATIVE LAMBDAS + SHRINKAGE -------------------------
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

        att = gf_shrunk / league_avg_team
        dff = ga_shrunk / league_avg_team

        att = _clamp(att, 0.55, 1.85)
        dff = _clamp(dff, 0.55, 1.85)

        return {"n": n, "att": att, "def": dff}

    h = get_rates(home_stats or {})
    a = get_rates(away_stats or {})

    lam_h = league_avg_team * h["att"] * a["def"] * home_adv_factor
    lam_a = league_avg_team * a["att"] * h["def"]

    lam_h = _clamp(lam_h, LAMBDA_MIN, LAMBDA_MAX_HOME)
    lam_a = _clamp(lam_a, LAMBDA_MIN, LAMBDA_MAX_AWAY)
    return lam_h, lam_a

# ------------------------- DC-ADJUSTED PROBS -------------------------
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

# ------------------------- ODDS (TheOddsAPI) -------------------------
def _odds_request(sport_key: str, params: dict):
    url = f"{ODDS_BASE_URL}/{sport_key}/odds"
    try:
        res = requests.get(url, params=params, timeout=25)
        rem = res.headers.get("x-requests-remaining")
        used = res.headers.get("x-requests-used")
        log(f"   TheOddsAPI status={res.status_code} remaining={rem} used={used}")
        if res.status_code != 200:
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

# ------------------------- VALUE% -------------------------
def value_pct(offered, fair):
    if offered is None or fair is None:
        return None
    try:
        if offered <= 0 or fair <= 0:
            return None
        return round((offered / fair - 1.0) * 100.0, 1)
    except Exception:
        return None

# ============================================================
#  STABILIZATION LAYER
# ============================================================
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
        if off1 <= 1.40:
            ph = max(ph, 0.62)
        elif off1 <= 1.60:
            ph = max(ph, 0.55)
        elif off1 <= 1.80:
            ph = max(ph, 0.50)

    if off2 is not None:
        if off2 <= 1.40:
            pa = max(pa, 0.62)
        elif off2 <= 1.60:
            pa = max(pa, 0.55)
        elif off2 <= 1.80:
            pa = max(pa, 0.50)

    return _renorm_1x2(ph, pd, pa)

def _snap_prob_to_market(prob, offered):
    if prob is None or offered is None or offered <= 1.0:
        return prob
    fair = implied(prob)
    if fair is None:
        return prob
    lo = offered / FAIR_SNAP_RATIO
    hi = offered * FAIR_SNAP_RATIO
    if fair < lo:
        return 1.0 / lo
    if fair > hi:
        return 1.0 / hi
    return prob

def stabilize_probs(
    league_name: str,
    league_baseline: dict,
    lam_h: float,
    lam_a: float,
    ph: float,
    pd: float,
    pa: float,
    po: float,
    pu: float,
    off1,
    offx,
    off2,
    offo,
    offu,
):
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

    ph2 = _snap_prob_to_market(ph, off1)
    pd2 = _snap_prob_to_market(pd, offx)
    pa2 = _snap_prob_to_market(pa, off2)
    ph, pd, pa = _renorm_1x2(ph2, pd2, pa2)

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

    po2 = _snap_prob_to_market(po, offo)
    pu2 = _snap_prob_to_market(pu, offu)
    if po2 is not None and pu2 is not None and (po2 + pu2) > 0:
        po, pu = po2 / (po2 + pu2), pu2 / (po2 + pu2)

    po = _clamp(po, CAP_OUTCOME_MIN, CAP_OVER)
    pu = _clamp(pu, CAP_OUTCOME_MIN, CAP_UNDER)
    s2 = po + pu
    if s2 > 0:
        po, pu = po / s2, pu / s2

    return ph, pd, pa, po, pu

# ============================================================
#  MIS (MATCH INTEGRITY SCORE)
#  Uses ONLY: odds_match debug + offered market availability + time diff
# ============================================================
def compute_mis(match_debug: dict, offered: dict):
    # offered markets count
    offered_keys = ["home", "draw", "away", "over", "under"]
    offered_cnt = sum(1 for k in offered_keys if safe_float(offered.get(k), None) is not None)

    one_x_two_cnt = sum(
        1 for k in ["home", "draw", "away"] if safe_float(offered.get(k), None) is not None
    )

    matched = bool(match_debug.get("matched"))
    sim = safe_float(match_debug.get("score"), 0.0) or 0.0
    tdiff = safe_float(match_debug.get("time_diff_h"), None)

    # gating flags
    gates = {
        "odds_matched": matched,
        "sim_ok": (sim >= MIS_MIN_MATCH_SCORE),
        "time_ok": (tdiff is None or tdiff <= MIS_MAX_TIME_DIFF_H),
        "markets_any_ok": (offered_cnt >= MIS_MIN_MARKETS_ANY),
        "markets_1x2_ok": (one_x_two_cnt >= MIS_MIN_1X2_MARKETS),
    }

    # score (0..100)
    score = 0.0
    score += 45.0 if matched else 0.0
    score += _clamp(sim, 0.0, 1.0) * 35.0
    score += _clamp(offered_cnt / 5.0, 0.0, 1.0) * 15.0
    score += 5.0 if (tdiff is None or tdiff <= 1.5) else (3.0 if tdiff <= MIS_MAX_TIME_DIFF_H else 0.0)

    score = round(_clamp(score, 0.0, 100.0), 1)

    # status
    if not (gates["odds_matched"] and gates["sim_ok"] and gates["time_ok"] and gates["markets_any_ok"]):
        status = "reject"
    else:
        if score >= MIS_ELITE_SCORE and gates["markets_1x2_ok"]:
            status = "elite"
        elif score >= MIS_CORE_SCORE and gates["markets_1x2_ok"]:
            status = "core_allowed"
        elif score >= MIS_FUN_SCORE:
            status = "fun_only"
        else:
            status = "reject"

    return {
        "enabled": MIS_ENABLED,
        "cut_reject": MIS_CUT_REJECT,
        "score": score,
        "status": status,
        "inputs": {
            "matched": matched,
            "match_score": round(sim, 3),
            "time_diff_h": tdiff,
            "offered_markets_count": offered_cnt,
            "offered_1x2_count": one_x_two_cnt,
        },
        "gates": gates,
        "thresholds": {
            "min_match_score": MIS_MIN_MATCH_SCORE,
            "max_time_diff_h": MIS_MAX_TIME_DIFF_H,
            "min_markets_any": MIS_MIN_MARKETS_ANY,
            "min_1x2_markets": MIS_MIN_1X2_MARKETS,
            "elite_score": MIS_ELITE_SCORE,
            "core_score": MIS_CORE_SCORE,
            "fun_score": MIS_FUN_SCORE,
        },
    }

# ------------------------- MAIN PIPELINE -------------------------
def build_fixture_blocks():
    fixtures_out = []
    rejected = []
    now = datetime.datetime.now(datetime.timezone.utc)
    to_dt = now + datetime.timedelta(hours=WINDOW_HOURS)
    odds_from = now - datetime.timedelta(hours=8)

    log(f"Using FOOTBALL_SEASON={FOOTBALL_SEASON}")
    log(f"Window: next {WINDOW_HOURS} hours | USE_ODDS_API={USE_ODDS_API}")
    log(f"MODEL: ShrinkageK={SHRINKAGE_K} | DC_RHO={DC_RHO}")
    log(f"STABILIZE: snap_ratio={FAIR_SNAP_RATIO} caps(over={CAP_OVER}, under={CAP_UNDER}, draw=[{CAP_DRAW_MIN},{CAP_DRAW_MAX}])")
    log(f"MIS: enabled={MIS_ENABLED} cut_reject={MIS_CUT_REJECT} minMatchScore={MIS_MIN_MATCH_SCORE}")

    if not API_FOOTBALL_KEY:
        log("❌ FOOTBALL_API_KEY is missing. Aborting.")
        return [], []

    all_fixtures = []
    for lg_name, lg_id in LEAGUES.items():
        all_fixtures.extend(fetch_fixtures(lg_id, lg_name))
    log(f"Total fixtures collected: {len(all_fixtures)}")

    odds_cache_by_league = {}
    matched_cnt = 0

    if USE_ODDS_API:
        total_events = 0
        for lg_name in LEAGUES.keys():
            odds_events = fetch_odds_for_league(lg_name, odds_from, to_dt)
            total_events += len(odds_events or [])
            odds_cache_by_league[lg_name] = build_events_cache(odds_events)
        log(f"Odds events fetched total: {total_events}")
    else:
        log("⚠️ USE_ODDS_API=False → skipping odds.")

    for fx in all_fixtures:
        league_id = fx["league_id"]
        league_name = fx["league_name"]
        league_baseline = fetch_league_baselines(league_id)

        home_stats = fetch_team_recent_stats(fx["home_id"], league_id, want_home_context=True)
        away_stats = fetch_team_recent_stats(fx["away_id"], league_id, want_home_context=False)

        lam_h, lam_a = compute_expected_goals(home_stats, away_stats, league_baseline)
        probs = compute_probabilities(lam_h, lam_a)

        offered = {}
        match_debug = {"matched": False, "reason": "odds_off"}
        if USE_ODDS_API:
            league_cache = odds_cache_by_league.get(league_name, [])
            offered, match_debug = pick_best_odds_for_fixture(fx, league_cache)
            if match_debug.get("matched"):
                matched_cnt += 1

        off_1 = offered.get("home")
        off_x = offered.get("draw")
        off_2 = offered.get("away")
        off_o = offered.get("over")
        off_u = offered.get("under")

        ph, pd, pa, po, pu = stabilize_probs(
            league_name=league_name,
            league_baseline=league_baseline,
            lam_h=lam_h,
            lam_a=lam_a,
            ph=probs["home_prob"],
            pd=probs["draw_prob"],
            pa=probs["away_prob"],
            po=probs["over_2_5_prob"],
            pu=probs["under_2_5_prob"],
            off1=off_1,
            offx=off_x,
            off2=off_2,
            offo=off_o,
            offu=off_u,
        )

        fair_1 = implied(ph)
        fair_x = implied(pd)
        fair_2 = implied(pa)
        fair_over = implied(po)
        fair_under = implied(pu)

        # MIS
        mis = None
        if MIS_ENABLED:
            mis = compute_mis(
                match_debug if isinstance(match_debug, dict) else {},
                {"home": off_1, "draw": off_x, "away": off_2, "over": off_o, "under": off_u},
            )

        dt = fx["commence_utc"]

        row = {
            "fixture_id": fx["id"],
            "date": dt.date().isoformat(),
            "time": dt.strftime("%H:%M"),
            "league_id": league_id,
            "league": league_name,
            "home": fx["home"],
            "away": fx["away"],
            "model": "bombay_multiplicative_dc_v3_stabilized_mis",

            "lambda_home": round(lam_h, 3),
            "lambda_away": round(lam_a, 3),

            "home_prob": round(ph, 3),
            "draw_prob": round(pd, 3),
            "away_prob": round(pa, 3),
            "over_2_5_prob": round(po, 3),
            "under_2_5_prob": round(pu, 3),

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

            "odds_match": match_debug,
            "league_baseline": league_baseline,
            "stabilization": {
                "snap_ratio": FAIR_SNAP_RATIO,
                "caps": {
                    "over": CAP_OVER,
                    "under": CAP_UNDER,
                    "draw_min": CAP_DRAW_MIN,
                    "draw_max": CAP_DRAW_MAX,
                    "outcome_min": CAP_OUTCOME_MIN,
                },
                "low_tempo_league": (league_name in LOW_TEMPO_LEAGUES),
            },
            "mis": mis,
        }

        if MIS_ENABLED and MIS_CUT_REJECT and (mis or {}).get("status") == "reject":
            rejected.append(row)
            continue

        fixtures_out.append(row)

    log(f"Thursday fixtures_out: {len(fixtures_out)} | rejected: {len(rejected)} | odds matched: {matched_cnt}")
    return fixtures_out, rejected

def main():
    fixtures, rejected = build_fixture_blocks()
    now = datetime.datetime.now(datetime.timezone.utc)
    to_dt = now + datetime.timedelta(hours=WINDOW_HOURS)

    out = {
        "generated_at": now.isoformat(),
        "window": {"from": now.date().isoformat(), "to": to_dt.date().isoformat(), "hours": WINDOW_HOURS},
        "fixtures_total": len(fixtures),
        "fixtures_rejected": len(rejected),
        "mis": {
            "enabled": MIS_ENABLED,
            "cut_reject": MIS_CUT_REJECT,
        },
        "fixtures": fixtures,
        "rejected": rejected,  # keep for debugging / audit
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/thursday_report_v3.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    log(f"✅ Thursday v3 (MIS CUT) READY. Fixtures: {len(fixtures)} | Rejected: {len(rejected)}")

if __name__ == "__main__":
    main()
