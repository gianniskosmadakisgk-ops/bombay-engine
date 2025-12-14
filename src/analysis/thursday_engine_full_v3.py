import os
import json
import math
import requests
import datetime
import unicodedata
import re
from dateutil import parser

# ============================================================
#  THURSDAY ENGINE v3.11 (STABILIZED + MARKET-ANCHORED FAIR)
#  - API-FOOTBALL fixtures + recent goals
#  - Multiplicative Poisson model (Att/Def factors) + Shrinkage
#  - Dixon-Coles low-score correction
#  - Market-Anchored blending (when offered odds exist) to prevent "alien" fair odds
#  - Sanity caps + favorite protection floors
#  - Output: logs/thursday_report_v3.json (same path)
# ============================================================

API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

FOOTBALL_SEASON = os.getenv("FOOTBALL_SEASON", "2025")
USE_ODDS_API = os.getenv("USE_ODDS_API", "true").lower() == "true"
WINDOW_HOURS = int(os.getenv("WINDOW_HOURS", "72"))

# Matching controls
ODDS_TIME_GATE_HOURS = float(os.getenv("ODDS_TIME_GATE_HOURS", "6"))
ODDS_TIME_SOFT_HOURS = float(os.getenv("ODDS_TIME_SOFT_HOURS", "10"))
ODDS_SIM_THRESHOLD = float(os.getenv("ODDS_SIM_THRESHOLD", "0.62"))

# Model controls
SHRINKAGE_K = float(os.getenv("SHRINKAGE_K", "8"))
DC_RHO = float(os.getenv("DC_RHO", "-0.13"))

LAMBDA_MIN = float(os.getenv("LAMBDA_MIN", "0.40"))
LAMBDA_MAX_HOME = float(os.getenv("LAMBDA_MAX_HOME", "3.00"))
LAMBDA_MAX_AWAY = float(os.getenv("LAMBDA_MAX_AWAY", "3.00"))

# NEW: Stabilization / caps
CAP_OVER_MAX = float(os.getenv("CAP_OVER_MAX", "0.70"))
CAP_UNDER_MAX = float(os.getenv("CAP_UNDER_MAX", "0.75"))
CAP_DRAW_MAX = float(os.getenv("CAP_DRAW_MAX", "0.32"))
CAP_DRAW_MIN = float(os.getenv("CAP_DRAW_MIN", "0.20"))
CAP_OUTCOME_MIN = float(os.getenv("CAP_OUTCOME_MIN", "0.05"))

# NEW: Market-anchored blending (0..1)
W_MIN = float(os.getenv("BLEND_W_MIN", "0.20"))
W_MAX = float(os.getenv("BLEND_W_MAX", "0.75"))
BLEND_K2 = float(os.getenv("BLEND_K2", "10"))  # larger => trust market more

# NEW: Favorite protection floors when market shows clear fave
FAV_FLOOR_180 = float(os.getenv("FAV_FLOOR_180", "0.50"))
FAV_FLOOR_160 = float(os.getenv("FAV_FLOOR_160", "0.55"))
FAV_FLOOR_140 = float(os.getenv("FAV_FLOOR_140", "0.62"))

# Over/Under anti-meat rules
OVER_LTOTAL_MIN = float(os.getenv("OVER_LTOTAL_MIN", "2.4"))
OVER_TEAM_LMIN = float(os.getenv("OVER_TEAM_LMIN", "0.9"))
UNDER_LTOTAL_MAX = float(os.getenv("UNDER_LTOTAL_MAX", "3.0"))
UNDER_BOTH_LMIN = float(os.getenv("UNDER_BOTH_LMIN", "1.4"))
LOW_TEMPO_LEAGUES = set((os.getenv("LOW_TEMPO_LEAGUES", "Serie B,Ligue 2").split(",")))
LOW_TEMPO_OVER_CAP = float(os.getenv("LOW_TEMPO_OVER_CAP", "0.65"))

# Baselines
USE_DYNAMIC_LEAGUE_BASELINES = os.getenv("USE_DYNAMIC_LEAGUE_BASELINES", "false").lower() == "true"
BASELINES_LAST_N = int(os.getenv("BASELINES_LAST_N", "180"))

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

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

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

    # filter by venue context if requested
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
        140: {"avg_goals_per_match": 2.6, "home_advantage": 0.16, "avg_draw_rate": 0.27, "avg_over25_rate": 0.55},
        94: {"avg_goals_per_match": 2.55, "home_advantage": 0.15, "avg_draw_rate": 0.28, "avg_over25_rate": 0.54},
        61: {"avg_goals_per_match": 2.70, "home_advantage": 0.16, "avg_draw_rate": 0.26, "avg_over25_rate": 0.55},
        62: {"avg_goals_per_match": 2.35, "home_advantage": 0.15, "avg_draw_rate": 0.29, "avg_over25_rate": 0.49},
        136: {"avg_goals_per_match": 2.45, "home_advantage": 0.15, "avg_draw_rate": 0.30, "avg_over25_rate": 0.50},
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

    total = home_goals = away_goals = draws = overs = 0
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
    if USE_DYNAMIC_LEAGUE_BASELINES:
        return fetch_league_baselines_dynamic(league_id)
    return fetch_league_baselines_static(league_id)

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
    return lam_h, lam_a, h["n"], a["n"]

# ------------------------- DC-ADJUSTED PROBS -------------------------
def compute_probabilities(lambda_home: float, lambda_away: float):
    max_goals = 6
    pmf_h = [poisson_pmf(k, lambda_home) for k in range(max_goals)]
    pmf_a = [poisson_pmf(k, lambda_away) for k in range(max_goals)]

    tail_h = max(0.0, 1.0 - sum(pmf_h))
    tail_a = max(0.0, 1.0 - sum(pmf_a))
    pmf_h.append(tail_h)
    pmf_a.append(tail_a)

    mat = [[0.0 for _ in range(max_goals + 1)] for __ in range(max_goals + 1)]
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            mat[i][j] = pmf_h[i] * pmf_a[j]

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

    po = max(1e-6, min(1.0 - 1e-6, po))
    return {"home_prob": ph, "draw_prob": pd, "away_prob": pa, "over_2_5_prob": po, "under_2_5_prob": 1.0 - po}

# ------------------------- MARKET IMPLIED (remove margin) -------------------------
def implied_probs_1x2(o1, ox, o2):
    if not o1 or not ox or not o2:
        return None
    if o1 <= 1.0 or ox <= 1.0 or o2 <= 1.0:
        return None
    p1, px, p2 = 1.0 / o1, 1.0 / ox, 1.0 / o2
    s = p1 + px + p2
    if s <= 0:
        return None
    return {"home": p1 / s, "draw": px / s, "away": p2 / s}

def implied_probs_ou(o_over, o_under):
    if not o_over or not o_under:
        return None
    if o_over <= 1.0 or o_under <= 1.0:
        return None
    po, pu = 1.0 / o_over, 1.0 / o_under
    s = po + pu
    if s <= 0:
        return None
    return {"over": po / s, "under": pu / s}

def blend_weight(n_home, n_away):
    # n_eff: 0..10 περίπου (δουλεύουμε με last5)
    n_eff = min(int(n_home or 0), 5) + min(int(n_away or 0), 5)
    w = n_eff / (n_eff + max(1e-6, BLEND_K2))
    return _clamp(w, W_MIN, W_MAX)

# ------------------------- STABILIZERS -------------------------
def apply_ou_logic_caps(league_name, lam_h, lam_a, p_over, p_under):
    lt = (lam_h or 0) + (lam_a or 0)
    # Over not allowed if basic lambda constraints fail
    if lt < OVER_LTOTAL_MIN or (lam_h is not None and lam_h < OVER_TEAM_LMIN) or (lam_a is not None and lam_a < OVER_TEAM_LMIN):
        p_over = min(p_over, 0.58)  # soft push down
        p_under = 1.0 - p_over

    # low-tempo leagues tighter over cap
    if league_name in LOW_TEMPO_LEAGUES:
        p_over = min(p_over, LOW_TEMPO_OVER_CAP)
        p_under = 1.0 - p_over

    # Under not allowed if game projects too open
    if lt > UNDER_LTOTAL_MAX and (lam_h or 0) > UNDER_BOTH_LMIN and (lam_a or 0) > UNDER_BOTH_LMIN:
        p_under = min(p_under, 0.62)  # soft push down
        p_over = 1.0 - p_under

    # hard caps
    p_over = min(p_over, CAP_OVER_MAX)
    p_under = min(p_under, CAP_UNDER_MAX)

    # renorm (keep complement)
    p_over = max(0.05, min(0.95, p_over))
    p_under = max(0.05, min(0.95, p_under))
    s = p_over + p_under
    p_over, p_under = p_over / s, p_under / s
    return p_over, p_under

def apply_1x2_caps_and_fave_floors(o1, o2, ph, pd, pa, lam_h, lam_a):
    # base mins
    ph = max(CAP_OUTCOME_MIN, ph)
    pd = max(CAP_OUTCOME_MIN, pd)
    pa = max(CAP_OUTCOME_MIN, pa)

    # draw bounds
    pd = min(pd, CAP_DRAW_MAX)
    pd = max(pd, CAP_DRAW_MIN)

    # draw normalization vs lambda difference (soft)
    if lam_h is not None and lam_a is not None:
        if abs(lam_h - lam_a) > 0.40:
            pd = min(pd, 0.26)

    # favorite protection: if market shows strong fav, ensure prob floor
    def floor_for_odds(odds):
        if odds is None:
            return None
        if odds <= 1.40:
            return FAV_FLOOR_140
        if odds <= 1.60:
            return FAV_FLOOR_160
        if odds <= 1.80:
            return FAV_FLOOR_180
        return None

    floor_home = floor_for_odds(o1)
    floor_away = floor_for_odds(o2)

    # apply floors then renorm
    if floor_home is not None:
        ph = max(ph, floor_home)
    if floor_away is not None:
        pa = max(pa, floor_away)

    s = ph + pd + pa
    if s <= 0:
        return 0.40, 0.26, 0.34
    ph, pd, pa = ph / s, pd / s, pa / s

    # re-enforce draw min after renorm
    if pd < CAP_DRAW_MIN:
        pd = CAP_DRAW_MIN
        rest = max(1e-6, ph + pa)
        scale = (1.0 - pd) / rest
        ph *= scale
        pa *= scale

    return ph, pd, pa

def value_pct(offered, fair):
    if offered is None or fair is None:
        return None
    try:
        if offered <= 0 or fair <= 0:
            return None
        return round((offered / fair - 1.0) * 100.0, 1)
    except Exception:
        return None

def score_1_10(p: float) -> float:
    s = round((p or 0.0) * 10.0, 1)
    if s < 1.0: s = 1.0
    if s > 10.0: s = 10.0
    return s

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

def _best_odds_from_event_for_fixture(ev_raw, event_home_norm, event_away_norm, swapped: bool):
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
        if fx_time and ct:
            diff_h = abs((ct - fx_time).total_seconds()) / 3600.0
        else:
            diff_h = None

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

    odds = _best_odds_from_event_for_fixture(best["raw"], best["home_norm"], best["away_norm"], best_swap)
    debug = {"matched": True, "score": round(best_score, 3), "swap": best_swap, "time_diff_h": None if best_diff is None else round(best_diff, 2)}
    return odds, debug

# ------------------------- MAIN PIPELINE -------------------------
def build_fixture_blocks():
    fixtures_out = []
    now = datetime.datetime.now(datetime.timezone.utc)
    to_dt = now + datetime.timedelta(hours=WINDOW_HOURS)
    odds_from = now - datetime.timedelta(hours=8)

    log(f"Using FOOTBALL_SEASON={FOOTBALL_SEASON}")
    log(f"Window: next {WINDOW_HOURS} hours")
    log(f"USE_ODDS_API={USE_ODDS_API}")
    log(f"MODEL: ShrinkageK={SHRINKAGE_K} | DC_RHO={DC_RHO}")
    log(f"BLENDING: W in [{W_MIN},{W_MAX}] K2={BLEND_K2}")

    if not API_FOOTBALL_KEY:
        log("❌ FOOTBALL_API_KEY is missing. Aborting fixture fetch.")
        return []

    all_fixtures = []
    for lg_name, lg_id in LEAGUES.items():
        all_fixtures.extend(fetch_fixtures(lg_id, lg_name))

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

    for fx in all_fixtures:
        league_id = fx["league_id"]
        league_name = fx["league_name"]
        league_baseline = fetch_league_baselines(league_id)

        home_stats = fetch_team_recent_stats(fx["home_id"], league_id, want_home_context=True)
        away_stats = fetch_team_recent_stats(fx["away_id"], league_id, want_home_context=False)

        lam_h, lam_a, n_home, n_away = compute_expected_goals(home_stats, away_stats, league_baseline)

        model = compute_probabilities(lam_h, lam_a)
        ph_m, pd_m, pa_m = model["home_prob"], model["draw_prob"], model["away_prob"]
        po_m, pu_m = model["over_2_5_prob"], model["under_2_5_prob"]

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

        # -------- Market-anchored blending (only if odds exist) --------
        w = blend_weight(n_home, n_away)

        # 1X2
        mkt_1x2 = implied_probs_1x2(off_1, off_x, off_2)
        if mkt_1x2:
            ph = (w * ph_m) + ((1.0 - w) * mkt_1x2["home"])
            pd = (w * pd_m) + ((1.0 - w) * mkt_1x2["draw"])
            pa = (w * pa_m) + ((1.0 - w) * mkt_1x2["away"])
        else:
            ph, pd, pa = ph_m, pd_m, pa_m

        # OU
        mkt_ou = implied_probs_ou(off_o, off_u)
        if mkt_ou:
            po = (w * po_m) + ((1.0 - w) * mkt_ou["over"])
            pu = (w * pu_m) + ((1.0 - w) * mkt_ou["under"])
        else:
            po, pu = po_m, pu_m

        # Stabilizers
        po, pu = apply_ou_logic_caps(league_name, lam_h, lam_a, po, pu)
        ph, pd, pa = apply_1x2_caps_and_fave_floors(off_1, off_2, ph, pd, pa, lam_h, lam_a)

        # FAIR from FINAL probs
        fair_1 = implied(ph)
        fair_x = implied(pd)
        fair_2 = implied(pa)
        fair_over = implied(po)
        fair_under = implied(pu)

        dt = fx["commence_utc"]

        fixtures_out.append({
            "fixture_id": fx["id"],
            "date": dt.date().isoformat(),
            "time": dt.strftime("%H:%M"),
            "league_id": league_id,
            "league": league_name,
            "home": fx["home"],
            "away": fx["away"],
            "model": "bombay_multiplicative_dc_marketblend_v3_11",

            "lambda_home": round(lam_h, 3),
            "lambda_away": round(lam_a, 3),

            # FINAL probs (these drive fair + Friday)
            "home_prob": round(ph, 3),
            "draw_prob": round(pd, 3),
            "away_prob": round(pa, 3),
            "over_2_5_prob": round(po, 3),
            "under_2_5_prob": round(pu, 3),

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

            # value pct
            "value_pct_1": value_pct(off_1, fair_1),
            "value_pct_x": value_pct(off_x, fair_x),
            "value_pct_2": value_pct(off_2, fair_2),
            "value_pct_over": value_pct(off_o, fair_over),
            "value_pct_under": value_pct(off_u, fair_under),

            # scores (kept)
            "score_draw": score_1_10(pd),
            "score_over": score_1_10(po),
            "score_under": score_1_10(pu),

            # debug (safe extra fields)
            "blend_w": round(w, 3),
            "odds_match": match_debug,
        })

    log(f"Thursday fixtures_out: {len(fixtures_out)} | odds matched: {matched_cnt}")
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

    log(f"✅ Thursday v3.11 READY. Fixtures: {len(fixtures)}")

if __name__ == "__main__":
    main()
