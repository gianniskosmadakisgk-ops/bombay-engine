import os
import json
import math
import requests
import datetime
import unicodedata
import re
from dateutil import parser

# ============================================================
#  BOMBAY THURSDAY — STABILIZED ENGINE (v1, V3 output)
#  Output: logs/thursday_report_v3.json  (unchanged schema)
#  Purpose: realistic probs + fair odds, cut the "crazy"
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

# Dixon-Coles rho
DC_RHO = float(os.getenv("DC_RHO", "-0.13"))

# Lambda clamps
LAMBDA_MIN = float(os.getenv("LAMBDA_MIN", "0.40"))
LAMBDA_MAX_HOME = float(os.getenv("LAMBDA_MAX_HOME", "3.00"))
LAMBDA_MAX_AWAY = float(os.getenv("LAMBDA_MAX_AWAY", "3.00"))

# Two-stage shrinkage (stability)
K_LAST_TO_SEASON = float(os.getenv("K_LAST_TO_SEASON", "12"))
K_SEASON_TO_LEAGUE = float(os.getenv("K_SEASON_TO_LEAGUE", "6"))

# Factor clamps (stop absurd favorites flips)
ATT_MIN = float(os.getenv("ATT_MIN", "0.60"))
ATT_MAX = float(os.getenv("ATT_MAX", "1.70"))
DEF_MIN = float(os.getenv("DEF_MIN", "0.60"))
DEF_MAX = float(os.getenv("DEF_MAX", "1.70"))

# ---------------- HARD SANITY RULES (as you defined) ----------------
CAP_OVER = float(os.getenv("CAP_OVER", "0.70"))
CAP_UNDER = float(os.getenv("CAP_UNDER", "0.75"))
CAP_DRAW_MAX = float(os.getenv("CAP_DRAW_MAX", "0.32"))
CAP_DRAW_MIN = float(os.getenv("CAP_DRAW_MIN", "0.20"))
CAP_ANY_MIN = float(os.getenv("CAP_ANY_MIN", "0.05"))

# Favorite protection thresholds (market -> prob floor)
FAV_T1_ODDS = float(os.getenv("FAV_T1_ODDS", "1.80"))
FAV_T1_PROB = float(os.getenv("FAV_T1_PROB", "0.50"))
FAV_T2_ODDS = float(os.getenv("FAV_T2_ODDS", "1.60"))
FAV_T2_PROB = float(os.getenv("FAV_T2_PROB", "0.55"))
FAV_T3_ODDS = float(os.getenv("FAV_T3_ODDS", "1.40"))
FAV_T3_PROB = float(os.getenv("FAV_T3_PROB", "0.62"))

# Fair deviation guard factor
FAIR_DEV_GUARD = float(os.getenv("FAIR_DEV_GUARD", "1.35"))

# Over/Under logic thresholds
OVER_LTOTAL_MIN = float(os.getenv("OVER_LTOTAL_MIN", "2.4"))
OVER_TEAM_L_MIN = float(os.getenv("OVER_TEAM_L_MIN", "0.9"))
UNDER_LTOTAL_MAX = float(os.getenv("UNDER_LTOTAL_MAX", "3.0"))
UNDER_BOTH_L_MIN = float(os.getenv("UNDER_BOTH_L_MIN", "1.4"))

LOW_TEMPO_LEAGUES = set([s.strip() for s in os.getenv("LOW_TEMPO_LEAGUES", "Serie B,Ligue 2").split(",") if s.strip()])
LOW_TEMPO_OVER_CAP = float(os.getenv("LOW_TEMPO_OVER_CAP", "0.65"))

# Draw normalization
DRAW_LDIFF_HARD = float(os.getenv("DRAW_LDIFF_HARD", "0.40"))
DRAW_MAX_WHEN_LDIFF = float(os.getenv("DRAW_MAX_WHEN_LDIFF", "0.26"))

USE_DYNAMIC_LEAGUE_BASELINES = os.getenv("USE_DYNAMIC_LEAGUE_BASELINES", "true").lower() == "true"
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

TEAM_LAST_CACHE = {}
TEAM_SEASON_CACHE = {}

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

def iso_z(dt: datetime.datetime) -> str:
    dt = dt.astimezone(datetime.timezone.utc).replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")

# ------------------------- FIXTURES -------------------------
def fetch_fixtures(league_id: int, league_name: str):
    if not API_FOOTBALL_KEY:
        log("⚠️ Missing FOOTBALL_API_KEY – NO fixtures fetched!")
        return []

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"league": league_id, "season": FOOTBALL_SEASON}

    try:
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=25).json()
    except Exception as e:
        log(f"⚠️ Error fixtures {league_name}: {e}")
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

        out.append({
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
        })

    return out

# ------------------------- TEAM LAST-5 (context) -------------------------
def fetch_team_last(team_id: int, league_id: int, want_home_context: bool):
    ck = (team_id, league_id, want_home_context)
    if ck in TEAM_LAST_CACHE:
        return TEAM_LAST_CACHE[ck]

    if not API_FOOTBALL_KEY:
        TEAM_LAST_CACHE[ck] = {}
        return TEAM_LAST_CACHE[ck]

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"team": team_id, "league": league_id, "season": FOOTBALL_SEASON, "last": 12}

    try:
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=25).json()
    except Exception:
        TEAM_LAST_CACHE[ck] = {}
        return TEAM_LAST_CACHE[ck]

    resp_all = r.get("response") or []
    if not resp_all:
        TEAM_LAST_CACHE[ck] = {}
        return TEAM_LAST_CACHE[ck]

    resp = []
    for fx in resp_all:
        is_home = fx["teams"]["home"]["id"] == team_id
        if want_home_context and is_home:
            resp.append(fx)
        if (not want_home_context) and (not is_home):
            resp.append(fx)
    if len(resp) < 4:
        resp = resp_all

    gf = ga = m = 0
    for fx in resp[:5]:
        m += 1
        hg = fx["goals"]["home"] or 0
        ag = fx["goals"]["away"] or 0
        is_home = fx["teams"]["home"]["id"] == team_id
        if is_home:
            gf += hg; ga += ag
        else:
            gf += ag; ga += hg

    out = {"matches_count": m, "avg_goals_for": (gf / m) if m else None, "avg_goals_against": (ga / m) if m else None}
    TEAM_LAST_CACHE[ck] = out
    return out

# ------------------------- TEAM SEASON STATS (context) -------------------------
def fetch_team_season(team_id: int, league_id: int):
    ck = (team_id, league_id)
    if ck in TEAM_SEASON_CACHE:
        return TEAM_SEASON_CACHE[ck]

    if not API_FOOTBALL_KEY:
        TEAM_SEASON_CACHE[ck] = {}
        return TEAM_SEASON_CACHE[ck]

    url = f"{API_FOOTBALL_BASE}/teams/statistics"
    params = {"team": team_id, "league": league_id, "season": FOOTBALL_SEASON}

    try:
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=25).json()
    except Exception:
        TEAM_SEASON_CACHE[ck] = {}
        return TEAM_SEASON_CACHE[ck]

    resp = r.get("response") or {}
    goals = resp.get("goals") or {}
    gf = ((goals.get("for") or {}).get("average") or {})
    ga = ((goals.get("against") or {}).get("average") or {})

    out = {
        "gf_home": safe_float(gf.get("home"), None),
        "gf_away": safe_float(gf.get("away"), None),
        "ga_home": safe_float(ga.get("home"), None),
        "ga_away": safe_float(ga.get("away"), None),
    }
    TEAM_SEASON_CACHE[ck] = out
    return out

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

    total = 0
    hg_sum = 0
    ag_sum = 0
    draws = 0
    overs = 0

    for fx in resp:
        hg = (fx.get("goals") or {}).get("home")
        ag = (fx.get("goals") or {}).get("away")
        if hg is None or ag is None:
            continue
        total += 1
        hg_sum += int(hg); ag_sum += int(ag)
        if int(hg) == int(ag):
            draws += 1
        if int(hg) + int(ag) >= 3:
            overs += 1

    if total < 30:
        return fetch_league_baselines_static(league_id)

    avg_goals = (hg_sum + ag_sum) / total
    hgpg = hg_sum / total
    agpg = ag_sum / total
    ha = (hgpg - agpg) / avg_goals if avg_goals > 0 else 0.16
    ha = _clamp(ha, 0.05, 0.25)

    return {
        "avg_goals_per_match": float(round(avg_goals, 3)),
        "home_advantage": float(round(ha, 3)),
        "avg_draw_rate": float(round(draws / total, 3)),
        "avg_over25_rate": float(round(overs / total, 3)),
    }

def fetch_league_baselines(league_id: int):
    return fetch_league_baselines_dynamic(league_id) if USE_DYNAMIC_LEAGUE_BASELINES else fetch_league_baselines_static(league_id)

# ------------------------- POISSON + DC -------------------------
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def compute_probabilities_dc(lam_h: float, lam_a: float):
    max_goals = 6
    pmf_h = [poisson_pmf(k, lam_h) for k in range(max_goals)]
    pmf_a = [poisson_pmf(k, lam_a) for k in range(max_goals)]
    pmf_h.append(max(0.0, 1.0 - sum(pmf_h)))
    pmf_a.append(max(0.0, 1.0 - sum(pmf_a)))

    mat = [[pmf_h[i] * pmf_a[j] for j in range(max_goals + 1)] for i in range(max_goals + 1)]

    rho = DC_RHO
    def tau(x, y):
        if x == 0 and y == 0:
            return 1.0 - (lam_h * lam_a * rho)
        if x == 1 and y == 0:
            return 1.0 + (lam_a * rho)
        if x == 0 and y == 1:
            return 1.0 + (lam_h * rho)
        if x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    for (x, y) in [(0,0), (1,0), (0,1), (1,1)]:
        mat[x][y] *= tau(x, y)

    s = sum(sum(row) for row in mat)
    if s <= 0:
        return {"home": 0.40, "draw": 0.26, "away": 0.34, "over": 0.55, "under": 0.45}

    mat = [[v / s for v in row] for row in mat]

    ph = pd = pa = 0.0
    po = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = mat[i][j]
            if i > j: ph += p
            elif i == j: pd += p
            else: pa += p
            if i + j >= 3: po += p

    po = _clamp(po, 1e-6, 1 - 1e-6)
    return {"home": ph, "draw": pd, "away": pa, "over": po, "under": 1.0 - po}

# ------------------------- STABILIZED EXPECTED GOALS -------------------------
def stabilized_rate(last_stats, season_stats, last_key, season_key, league_avg):
    n_last = int((last_stats or {}).get("matches_count") or 0)
    r_last = safe_float((last_stats or {}).get(last_key), None)

    r_season = safe_float((season_stats or {}).get(season_key), None)
    if r_season is None:
        r_season = league_avg

    # Stage A: last -> season
    if r_last is None or n_last <= 0:
        r_a = r_season
    else:
        k1 = max(0.0, K_LAST_TO_SEASON)
        r_a = (n_last * r_last + k1 * r_season) / (n_last + k1)

    # Stage B: season -> league
    k2 = max(0.0, K_SEASON_TO_LEAGUE)
    n_season_equiv = 10.0
    r_b = (n_season_equiv * r_a + k2 * league_avg) / (n_season_equiv + k2)

    return r_b

def compute_expected_goals(home_last, away_last, home_season, away_season, base):
    avg_match = safe_float(base.get("avg_goals_per_match"), 2.6) or 2.6
    league_avg_team = max(0.70, avg_match / 2.0)
    home_adv = safe_float(base.get("home_advantage"), 0.16) or 0.16
    home_adv_factor = 1.0 + home_adv

    gf_h = stabilized_rate(home_last, home_season, "avg_goals_for", "gf_home", league_avg_team)
    ga_h = stabilized_rate(home_last, home_season, "avg_goals_against", "ga_home", league_avg_team)

    gf_a = stabilized_rate(away_last, away_season, "avg_goals_for", "gf_away", league_avg_team)
    ga_a = stabilized_rate(away_last, away_season, "avg_goals_against", "ga_away", league_avg_team)

    att_h = _clamp(gf_h / league_avg_team, ATT_MIN, ATT_MAX)
    def_h = _clamp(ga_h / league_avg_team, DEF_MIN, DEF_MAX)
    att_a = _clamp(gf_a / league_avg_team, ATT_MIN, ATT_MAX)
    def_a = _clamp(ga_a / league_avg_team, DEF_MIN, DEF_MAX)

    lam_h = league_avg_team * att_h * def_a * home_adv_factor
    lam_a = league_avg_team * att_a * def_h

    lam_h = _clamp(lam_h, LAMBDA_MIN, LAMBDA_MAX_HOME)
    lam_a = _clamp(lam_a, LAMBDA_MIN, LAMBDA_MAX_AWAY)
    return lam_h, lam_a

# ------------------------- ODDS (TheOddsAPI) -------------------------
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
    if not USE_ODDS_API or not ODDS_API_KEY:
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

    p1 = dict(base_params)
    p1["commenceTimeFrom"] = iso_z(window_from)
    p1["commenceTimeTo"] = iso_z(window_to)

    data = _odds_request(sport_key, p1)
    if data:
        return data
    return _odds_request(sport_key, base_params)

def build_events_cache(events):
    out = []
    for ev in events or []:
        h = normalize_team_name(ev.get("home_team", "") or "")
        a = normalize_team_name(ev.get("away_team", "") or "")
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
    best = {"home": None, "draw": None, "away": None, "over": None, "under": None}
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
                        best["draw"] = max(best["draw"] or 0.0, price)
                    else:
                        if not swapped:
                            if nm == event_home_norm:
                                best["home"] = max(best["home"] or 0.0, price)
                            elif nm == event_away_norm:
                                best["away"] = max(best["away"] or 0.0, price)
                        else:
                            if nm == event_home_norm:
                                best["away"] = max(best["away"] or 0.0, price)
                            elif nm == event_away_norm:
                                best["home"] = max(best["home"] or 0.0, price)
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
                        best["over"] = max(best["over"] or 0.0, price)
                    elif "under" in name:
                        best["under"] = max(best["under"] or 0.0, price)
    return best

def pick_best_odds_for_fixture(fx, cache):
    if not cache:
        return {}, {"matched": False, "reason": "no_odds_events"}

    fx_ht = token_set(fx["home_norm"])
    fx_at = token_set(fx["away_norm"])
    fx_time = fx.get("commence_utc")

    best = None
    best_score = -1.0
    best_swap = False
    best_diff = None

    for ev in cache:
        ct = ev["commence_time"]
        diff_h = abs((ct - fx_time).total_seconds()) / 3600.0 if (fx_time and ct) else None
        if diff_h is not None and diff_h > ODDS_TIME_GATE_HOURS:
            continue

        time_pen = min(1.0, diff_h / max(1e-6, ODDS_TIME_SOFT_HOURS)) if diff_h is not None else 0.0

        s_norm = (jaccard(fx_ht, ev["home_tokens"]) + jaccard(fx_at, ev["away_tokens"])) / 2.0
        s_swap = (jaccard(fx_ht, ev["away_tokens"]) + jaccard(fx_at, ev["home_tokens"])) / 2.0

        sc_norm = s_norm - 0.20 * time_pen
        sc_swap = s_swap - 0.20 * time_pen

        if sc_norm > best_score:
            best_score = sc_norm; best = ev; best_swap = False; best_diff = diff_h
        if sc_swap > best_score:
            best_score = sc_swap; best = ev; best_swap = True; best_diff = diff_h

    if best is None:
        return {}, {"matched": False, "reason": "no_candidate"}
    if best_score < ODDS_SIM_THRESHOLD:
        return {}, {"matched": False, "reason": f"low_similarity(score={best_score:.2f})"}

    odds = _best_odds_from_event(best["raw"], best["home_norm"], best["away_norm"], best_swap)
    dbg = {"matched": True, "score": round(best_score, 3), "swap": best_swap, "time_diff_h": None if best_diff is None else round(best_diff, 2)}
    return odds, dbg

# ------------------------- VALUE + SCORES (keep in JSON) -------------------------
def value_pct(offered, fair):
    if offered is None or fair is None:
        return None
    if offered <= 0 or fair <= 0:
        return None
    return round((offered / fair - 1.0) * 100.0, 1)

def score_1_10(p: float) -> float:
    s = round((p or 0.0) * 10.0, 1)
    return float(_clamp(s, 1.0, 10.0))

# ------------------------- SANITY + SNAP (THE KEY) -------------------------
def apply_prob_caps(ph, pd, pa, po, pu, league_name):
    # Any min
    ph = max(CAP_ANY_MIN, ph)
    pd = max(CAP_ANY_MIN, pd)
    pa = max(CAP_ANY_MIN, pa)

    # Draw min/max
    pd = _clamp(pd, CAP_DRAW_MIN, CAP_DRAW_MAX)

    # Over/Under caps (league low-tempo tighter over cap)
    over_cap = LOW_TEMPO_OVER_CAP if league_name in LOW_TEMPO_LEAGUES else CAP_OVER
    po = min(po, over_cap)
    pu = min(pu, CAP_UNDER)

    # Renormalize 1X2
    s = ph + pd + pa
    ph, pd, pa = ph / s, pd / s, pa / s

    # Renormalize O/U (not strict sum if both capped, so normalize)
    t = po + pu
    po, pu = po / t, pu / t
    return ph, pd, pa, po, pu

def favorite_protection(ph, pd, pa, offered_1, offered_2):
    # If market shows clear favorite, enforce prob floor
    # Home favorite
    if offered_1 is not None:
        if offered_1 <= FAV_T3_ODDS: ph = max(ph, FAV_T3_PROB)
        elif offered_1 <= FAV_T2_ODDS: ph = max(ph, FAV_T2_PROB)
        elif offered_1 <= FAV_T1_ODDS: ph = max(ph, FAV_T1_PROB)
    # Away favorite
    if offered_2 is not None:
        if offered_2 <= FAV_T3_ODDS: pa = max(pa, FAV_T3_PROB)
        elif offered_2 <= FAV_T2_ODDS: pa = max(pa, FAV_T2_PROB)
        elif offered_2 <= FAV_T1_ODDS: pa = max(pa, FAV_T1_PROB)

    # keep any min on the rest, then renormalize
    ph = max(CAP_ANY_MIN, ph)
    pd = max(CAP_ANY_MIN, pd)
    pa = max(CAP_ANY_MIN, pa)
    s = ph + pd + pa
    return ph / s, pd / s, pa / s

def fair_deviation_guard(fair, offered):
    # If fair deviates too far from market, "snap" toward market by capping deviation
    if fair is None or offered is None:
        return fair
    if offered <= 1.0 or fair <= 1.0:
        return fair
    lo = offered / FAIR_DEV_GUARD
    hi = offered * FAIR_DEV_GUARD
    return _clamp(fair, lo, hi)

def over_under_logic(po, lam_h, lam_a, league_name):
    lt = (lam_h + lam_a)
    # Over not allowed conditions
    if lt < OVER_LTOTAL_MIN:
        po = min(po, 0.60)
    if min(lam_h, lam_a) < OVER_TEAM_L_MIN:
        po = min(po, 0.60)
    if league_name in LOW_TEMPO_LEAGUES:
        po = min(po, LOW_TEMPO_OVER_CAP)

    # Under not allowed conditions
    pu = 1.0 - po
    if lt > UNDER_LTOTAL_MAX:
        pu = min(pu, 0.60)
    if (lam_h > UNDER_BOTH_L_MIN) and (lam_a > UNDER_BOTH_L_MIN):
        pu = min(pu, 0.60)

    # normalize back
    t = po + pu
    return po / t, pu / t

def draw_normalization(pd, lam_h, lam_a, base, league_name):
    ldiff = abs(lam_h - lam_a)
    if ldiff > DRAW_LDIFF_HARD:
        pd = min(pd, DRAW_MAX_WHEN_LDIFF)
    # league bound: <= league_avg + 0.03 (unless draw-heavy, still bounded)
    avg_draw = safe_float(base.get("avg_draw_rate"), 0.26) or 0.26
    pd = min(pd, avg_draw + 0.03)
    pd = _clamp(pd, CAP_DRAW_MIN, CAP_DRAW_MAX)
    return pd

# ------------------------- MAIN PIPELINE -------------------------
def build_fixture_blocks():
    now = datetime.datetime.now(datetime.timezone.utc)
    to_dt = now + datetime.timedelta(hours=WINDOW_HOURS)
    odds_from = now - datetime.timedelta(hours=8)

    if not API_FOOTBALL_KEY:
        log("❌ FOOTBALL_API_KEY missing. Abort.")
        return []

    all_fx = []
    for lg_name, lg_id in LEAGUES.items():
        all_fx.extend(fetch_fixtures(lg_id, lg_name))

    odds_cache = {}
    if USE_ODDS_API and ODDS_API_KEY:
        for lg_name in LEAGUES.keys():
            events = fetch_odds_for_league(lg_name, odds_from, to_dt)
            odds_cache[lg_name] = build_events_cache(events)

    out = []
    for fx in all_fx:
        lg_name = fx["league_name"]
        lg_id = fx["league_id"]
        base = fetch_league_baselines(lg_id)

        home_last = fetch_team_last(fx["home_id"], lg_id, want_home_context=True)
        away_last = fetch_team_last(fx["away_id"], lg_id, want_home_context=False)
        home_season = fetch_team_season(fx["home_id"], lg_id)
        away_season = fetch_team_season(fx["away_id"], lg_id)

        lam_h, lam_a = compute_expected_goals(home_last, away_last, home_season, away_season, base)
        probs = compute_probabilities_dc(lam_h, lam_a)

        ph, pd, pa = probs["home"], probs["draw"], probs["away"]
        po, pu = probs["over"], probs["under"]

        offered = {}
        match_debug = {"matched": False, "reason": "odds_off"}
        if USE_ODDS_API and ODDS_API_KEY:
            offered, match_debug = pick_best_odds_for_fixture(fx, odds_cache.get(lg_name, []))

        off_1 = offered.get("home")
        off_x = offered.get("draw")
        off_2 = offered.get("away")
        off_o = offered.get("over")
        off_u = offered.get("under")

        # --- RULES APPLICATION ORDER ---
        # 1) Over/Under logic constraints (uses lambdas + league)
        po, pu = over_under_logic(po, lam_h, lam_a, lg_name)

        # 2) Draw normalization (uses lambdas + league baseline)
        pd = draw_normalization(pd, lam_h, lam_a, base, lg_name)

        # 3) Prob caps (hard)
        ph, pd, pa, po, pu = apply_prob_caps(ph, pd, pa, po, pu, lg_name)

        # 4) Favorite protection (market-aware)
        ph, pd, pa = favorite_protection(ph, pd, pa, off_1, off_2)

        # 5) Final caps again (safety)
        ph, pd, pa, po, pu = apply_prob_caps(ph, pd, pa, po, pu, lg_name)

        # --- FAIR (from prob), then deviation guard snap to market ---
        fair_1 = implied(ph); fair_x = implied(pd); fair_2 = implied(pa)
        fair_over = implied(po); fair_under = implied(pu)

        # Market deviation guard (book-like)
        fair_1 = fair_deviation_guard(fair_1, off_1)
        fair_x = fair_deviation_guard(fair_x, off_x)
        fair_2 = fair_deviation_guard(fair_2, off_2)
        fair_over = fair_deviation_guard(fair_over, off_o)
        fair_under = fair_deviation_guard(fair_under, off_u)

        # If we snapped fair, probs must follow fair (still "book-like" and stable)
        # Rebuild probs from fair where snapped happened (only when offered exists)
        def prob_from_fair(f):
            return (1.0 / f) if (f is not None and f > 0) else None

        ph2 = prob_from_fair(fair_1) if off_1 is not None else ph
        pd2 = prob_from_fair(fair_x) if off_x is not None else pd
        pa2 = prob_from_fair(fair_2) if off_2 is not None else pa

        # normalize 1X2 after snap
        if (ph2 is not None) and (pd2 is not None) and (pa2 is not None):
            ph2 = max(CAP_ANY_MIN, ph2); pd2 = _clamp(pd2, CAP_DRAW_MIN, CAP_DRAW_MAX); pa2 = max(CAP_ANY_MIN, pa2)
            s = ph2 + pd2 + pa2
            ph, pd, pa = ph2 / s, pd2 / s, pa2 / s

        po2 = prob_from_fair(fair_over) if off_o is not None else po
        pu2 = prob_from_fair(fair_under) if off_u is not None else pu
        if (po2 is not None) and (pu2 is not None):
            po2 = min(po2, LOW_TEMPO_OVER_CAP if lg_name in LOW_TEMPO_LEAGUES else CAP_OVER)
            pu2 = min(pu2, CAP_UNDER)
            t = po2 + pu2
            po, pu = po2 / t, pu2 / t

        # Scores kept (Friday can use them if needed, but you can ignore in presentation)
        score_draw = score_1_10(pd)
        score_over = score_1_10(po)
        score_under = score_1_10(pu)

        dt = fx["commence_utc"]

        out.append({
            "fixture_id": fx["id"],
            "date": dt.date().isoformat(),
            "time": dt.strftime("%H:%M"),
            "league_id": lg_id,
            "league": lg_name,
            "home": fx["home"],
            "away": fx["away"],
            "model": "bombay_stabilized_v1",

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

            "score_draw": score_draw,
            "score_over": score_over,
            "score_under": score_under,

            "odds_match": match_debug,
            "league_baseline": base,
        })

    return out

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

    log(f"✅ Bombay Thursday Stabilized (V3) saved → logs/thursday_report_v3.json | fixtures={len(fixtures)}")

if __name__ == "__main__":
    main()
