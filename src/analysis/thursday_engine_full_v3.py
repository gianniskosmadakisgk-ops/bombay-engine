import os
import json
import math
import requests
import datetime
import unicodedata
import re
from dateutil import parser

# ============================================================
# BOMBAY THURSDAY ENGINE v3 (LOCKED LEAGUES) — PRODUCTION
#
# Writes: logs/thursday_report_v3.json
# Output includes:
#   - engine_leagues: list of leagues actually processed (LOCKED)
#   - season_used: FOOTBALL_SEASON
#   - fixtures_total + fixtures[]
#
# IMPORTANT:
#   - Only leagues in LEAGUES dict exist. No Belgium/Netherlands/Greece/Turkey, ever.
# ============================================================

API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

FOOTBALL_SEASON = os.getenv("FOOTBALL_SEASON", "2025")  # API-Football season label (e.g. 2025 for 25/26)
WINDOW_HOURS = int(os.getenv("WINDOW_HOURS", "72"))
USE_ODDS_API = os.getenv("USE_ODDS_API", "true").lower() == "true"

# Odds matching controls
ODDS_TIME_GATE_HOURS = float(os.getenv("ODDS_TIME_GATE_HOURS", "8"))
ODDS_TIME_SOFT_HOURS = float(os.getenv("ODDS_TIME_SOFT_HOURS", "12"))
ODDS_SIM_THRESHOLD = float(os.getenv("ODDS_SIM_THRESHOLD", "0.60"))

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

LOW_TEMPO_LEAGUES = set([x.strip() for x in os.getenv("LOW_TEMPO_LEAGUES", "Serie B,Ligue 2").split(",") if x.strip()])
LOW_TEMPO_OVER_CAP = float(os.getenv("LOW_TEMPO_OVER_CAP", "0.65"))

# Over/Under blockers
OVER_BLOCK_LTOTAL = float(os.getenv("OVER_BLOCK_LTOTAL", "2.4"))
OVER_BLOCK_LMIN = float(os.getenv("OVER_BLOCK_LMIN", "0.9"))
UNDER_BLOCK_LTOTAL = float(os.getenv("UNDER_BLOCK_LTOTAL", "3.0"))
UNDER_BLOCK_BOTH_GT = float(os.getenv("UNDER_BLOCK_BOTH_GT", "1.4"))

# Draw normalization
DRAW_LAMBDA_GAP_MAX = float(os.getenv("DRAW_LAMBDA_GAP_MAX", "0.40"))
DRAW_IF_GAP_CAP = float(os.getenv("DRAW_IF_GAP_CAP", "0.26"))
DRAW_LEAGUE_PLUS = float(os.getenv("DRAW_LEAGUE_PLUS", "0.03"))

# Selection hints (stored only)
TIGHT_DRAW_THRESHOLD = float(os.getenv("TIGHT_DRAW_THRESHOLD", "0.28"))
TIGHT_LTOTAL_THRESHOLD = float(os.getenv("TIGHT_LTOTAL_THRESHOLD", "2.55"))
OVER_TIGHT_PENALTY_PTS = float(os.getenv("OVER_TIGHT_PENALTY_PTS", "12.0"))
OVER_LOW_TEMPO_EXTRA_PENALTY_PTS = float(os.getenv("OVER_LOW_TEMPO_EXTRA_PENALTY_PTS", "6.0"))

# ✅ LOCKED LEAGUES (ONLY THESE)
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

# TheOddsAPI mapping (only for the above)
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
    return len(a & b) / max(1, len(a | b))

# ------------------------- FIXTURES -------------------------
def fetch_fixtures(league_id: int, league_name: str):
    if not API_FOOTBALL_KEY:
        log("❌ Missing FOOTBALL_API_KEY")
        return []

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"league": league_id, "season": FOOTBALL_SEASON}

    try:
        r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=25).json()
    except Exception as e:
        log(f"⚠️ fixtures fetch error {league_name}: {e}")
        return []

    resp = r.get("response") or []
    if not resp:
        return []

    now = datetime.datetime.now(datetime.timezone.utc)
    out = []

    for fx in resp:
        if (fx.get("fixture") or {}).get("status", {}).get("short") != "NS":
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
            "commence_utc": dt,
        })

    log(f"→ {league_name}: kept {len(out)} within {WINDOW_HOURS}h")
    return out

# ------------------------- TEAM STATS -------------------------
def _recency_weights(n: int):
    base = [1.0, 0.85, 0.72, 0.61, 0.52]
    w = base[: max(0, min(5, n))]
    s = sum(w) if w else 1.0
    return [x / s for x in w]

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
    except Exception:
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
            is_home = (fx.get("teams") or {}).get("home", {}).get("id") == team_id
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

    for idx, fx in enumerate(sample):
        g_home = (fx.get("goals") or {}).get("home")
        g_away = (fx.get("goals") or {}).get("away")
        if g_home is None or g_away is None:
            continue
        g_home = int(g_home)
        g_away = int(g_away)
        is_home = (fx.get("teams") or {}).get("home", {}).get("id") == team_id
        gf_i, ga_i = (g_home, g_away) if is_home else (g_away, g_home)
        gf += w[idx] * gf_i
        ga += w[idx] * ga_i

    stats = {"matches_count": m, "avg_goals_for": gf, "avg_goals_against": ga}
    TEAM_STATS_CACHE[ck] = stats
    return stats

# ------------------------- LEAGUE BASELINES (STATIC) -------------------------
def league_baseline(league_id: int):
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
    base.update(overrides.get(league_id, {}))
    return base

# ------------------------- MODEL -------------------------
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def compute_expected_goals(home_stats: dict, away_stats: dict, lb: dict):
    avg_match = safe_float(lb.get("avg_goals_per_match"), 2.6) or 2.6
    league_avg_team = max(0.65, avg_match / 2.0)

    home_adv = safe_float(lb.get("home_advantage"), 0.16) or 0.16
    home_adv_factor = 1.0 + home_adv

    def get_rates(stats):
        n = int(stats.get("matches_count") or 0)
        gf = safe_float(stats.get("avg_goals_for"), league_avg_team) or league_avg_team
        ga = safe_float(stats.get("avg_goals_against"), league_avg_team) or league_avg_team

        k = max(0.0, SHRINKAGE_K)
        denom = (n + k) if (n + k) > 0 else 1.0

        gf_s = (n * gf + k * league_avg_team) / denom
        ga_s = (n * ga + k * league_avg_team) / denom

        att = _clamp(gf_s / league_avg_team, 0.55, 1.85)
        dff = _clamp(ga_s / league_avg_team, 0.55, 1.85)
        return {"att": att, "def": dff}

    h = get_rates(home_stats or {})
    a = get_rates(away_stats or {})

    lam_h = league_avg_team * h["att"] * a["def"] * home_adv_factor
    lam_a = league_avg_team * a["att"] * h["def"]

    lam_h = _clamp(lam_h, LAMBDA_MIN, LAMBDA_MAX_HOME)
    lam_a = _clamp(lam_a, LAMBDA_MIN, LAMBDA_MAX_AWAY)
    return lam_h, lam_a

def compute_probabilities(lam_h: float, lam_a: float):
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
    return {"home_prob": ph, "draw_prob": pd, "away_prob": pa, "over_2_5_prob": po, "under_2_5_prob": 1.0 - po}

def _renorm_1x2(ph, pd, pa):
    ph = max(CAP_OUTCOME_MIN, ph)
    pd = _clamp(max(CAP_OUTCOME_MIN, pd), CAP_DRAW_MIN, CAP_DRAW_MAX)
    pa = max(CAP_OUTCOME_MIN, pa)
    s = ph + pd + pa
    return (ph / s, pd / s, pa / s) if s > 0 else (0.40, 0.26, 0.34)

def stabilize_probs(league_name: str, lb: dict, lam_h: float, lam_a: float, ph: float, pd: float, pa: float, po: float, pu: float):
    gap = abs((lam_h or 0) - (lam_a or 0))
    if gap > DRAW_LAMBDA_GAP_MAX:
        pd = min(pd, DRAW_IF_GAP_CAP)

    league_draw = safe_float((lb or {}).get("avg_draw_rate"), None)
    if league_draw is not None:
        pd = min(pd, league_draw + DRAW_LEAGUE_PLUS, CAP_DRAW_MAX)
        pd = max(pd, CAP_DRAW_MIN)

    ph, pd, pa = _renorm_1x2(ph, pd, pa)

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
    if (not USE_ODDS_API) or (not ODDS_API_KEY):
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
        "commenceTimeFrom": iso_z(window_from),
        "commenceTimeTo": iso_z(window_to),
    }
    data = _odds_request(sport_key, base_params)
    if data:
        return data
    # fallback without time filters
    base_params.pop("commenceTimeFrom", None)
    base_params.pop("commenceTimeTo", None)
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

    fx_h = fx["home_norm"]; fx_a = fx["away_norm"]
    fx_ht = token_set(fx_h); fx_at = token_set(fx_a)
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
            best_score = score_norm; best = ev; best_swap = False; best_diff = diff_h
        if score_swap > best_score:
            best_score = score_swap; best = ev; best_swap = True; best_diff = diff_h

    if best is None:
        return {}, {"matched": False, "reason": "no_candidates_time_gate"}
    if best_score < ODDS_SIM_THRESHOLD:
        return {}, {"matched": False, "reason": f"low_similarity(score={best_score:.2f})"}

    odds = _best_odds_from_event(best["raw"], best["home_norm"], best["away_norm"], best_swap)
    return odds, {"matched": True, "score": round(best_score, 3), "swap": best_swap, "time_diff_h": None if best_diff is None else round(best_diff, 2)}

# ------------------------- VALUE + EV -------------------------
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

# ------------------------- CONFIDENCE -------------------------
def _confidence_score(home_stats, away_stats, match_debug, lam_h, lam_a):
    n_h = int((home_stats or {}).get("matches_count") or 0)
    n_a = int((away_stats or {}).get("matches_count") or 0)
    n = min(n_h, n_a)

    score = 0.40
    score += 0.20 * _clamp(n / 5.0, 0.0, 1.0)
    if (match_debug or {}).get("matched"):
        score += 0.15

    ltot = (lam_h or 0.0) + (lam_a or 0.0)
    if ltot < 2.1:
        score -= 0.10
    elif ltot < 2.3:
        score -= 0.06
    if ltot > 3.6:
        score -= 0.08
    elif ltot > 3.3:
        score -= 0.05

    return _clamp(score, 0.05, 0.95)

# ------------------------- MAIN -------------------------
def build_fixture_blocks():
    now = datetime.datetime.now(datetime.timezone.utc)
    to_dt = now + datetime.timedelta(hours=WINDOW_HOURS)
    odds_from = now - datetime.timedelta(hours=8)

    if not API_FOOTBALL_KEY:
        log("❌ FOOTBALL_API_KEY missing. Abort.")
        return []

    # 1) Collect fixtures (LOCKED leagues only)
    all_fixtures = []
    for lg_name, lg_id in LEAGUES.items():
        all_fixtures.extend(fetch_fixtures(lg_id, lg_name))
    log(f"Total fixtures collected: {len(all_fixtures)}")

    # 2) Odds caches per locked league
    odds_cache_by_league = {}
    if USE_ODDS_API and ODDS_API_KEY:
        for lg_name in LEAGUES.keys():
            odds_events = fetch_odds_for_league(lg_name, odds_from, to_dt)
            odds_cache_by_league[lg_name] = build_events_cache(odds_events)

    fixtures_out = []
    matched_cnt = 0

    for fx in all_fixtures:
        league_id = fx["league_id"]
        league_name = fx["league_name"]
        lb = league_baseline(league_id)

        home_stats = fetch_team_recent_stats(fx["home_id"], league_id, want_home_context=True)
        away_stats = fetch_team_recent_stats(fx["away_id"], league_id, want_home_context=False)

        lam_h, lam_a = compute_expected_goals(home_stats, away_stats, lb)
        probs = compute_probabilities(lam_h, lam_a)

        offered = {}
        match_debug = {"matched": False, "reason": "odds_off"}
        if USE_ODDS_API and ODDS_API_KEY:
            offered, match_debug = pick_best_odds_for_fixture(fx, odds_cache_by_league.get(league_name, []))
            if match_debug.get("matched"):
                matched_cnt += 1

        off_1 = offered.get("home")
        off_x = offered.get("draw")
        off_2 = offered.get("away")
        off_o = offered.get("over")
        off_u = offered.get("under")

        m_ph, m_pd, m_pa, m_po, m_pu = stabilize_probs(
            league_name, lb,
            lam_h, lam_a,
            probs["home_prob"], probs["draw_prob"], probs["away_prob"],
            probs["over_2_5_prob"], probs["under_2_5_prob"]
        )

        # fair from MODEL probs
        fair_1 = implied(m_ph); fair_x = implied(m_pd); fair_2 = implied(m_pa)
        fair_over = implied(m_po); fair_under = implied(m_pu)

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

        total_lambda = round((lam_h or 0.0) + (lam_a or 0.0), 3)
        abs_gap = round(abs((lam_h or 0.0) - (lam_a or 0.0)), 3)

        tight_game = (m_pd >= TIGHT_DRAW_THRESHOLD) or (total_lambda <= TIGHT_LTOTAL_THRESHOLD)
        low_tempo = league_name in LOW_TEMPO_LEAGUES

        over_penalty_pts = 0.0
        if tight_game:
            over_penalty_pts += OVER_TIGHT_PENALTY_PTS
        if low_tempo:
            over_penalty_pts += OVER_LOW_TEMPO_EXTRA_PENALTY_PTS

        selection_vo = None
        if vo is not None:
            selection_vo = round(vo - over_penalty_pts, 1)

        conf = _confidence_score(home_stats, away_stats, match_debug, lam_h, lam_a)
        conf_band = "high" if conf >= 0.70 else ("mid" if conf >= 0.55 else "low")

        dt = fx["commence_utc"]

        fixtures_out.append({
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
            "total_lambda": total_lambda,
            "abs_lambda_gap": abs_gap,

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

            # compatibility aliases
            "score_draw": evx,
            "score_over": evo,

            "selection_value_pct_over": selection_vo,
            "over_value_penalty_pts": round(over_penalty_pts, 2),

            "flags": {
                "tight_game": bool(tight_game),
                "low_tempo_league": bool(low_tempo),
                "odds_matched": bool(match_debug.get("matched")),
                "confidence": round(conf, 3),
                "confidence_band": conf_band,
            },

            "odds_match": match_debug,
            "league_baseline": lb,
        })

    log(f"Fixtures out: {len(fixtures_out)} | odds matched: {matched_cnt}")
    return fixtures_out

def main():
    now = datetime.datetime.now(datetime.timezone.utc)
    to_dt = now + datetime.timedelta(hours=WINDOW_HOURS)

    fixtures = build_fixture_blocks()

    out = {
        "generated_at": now.isoformat(),
        "season_used": str(FOOTBALL_SEASON),
        "window": {"from": now.date().isoformat(), "to": to_dt.date().isoformat(), "hours": WINDOW_HOURS},
        "engine_leagues": list(LEAGUES.keys()),
        "fixtures_total": len(fixtures),
        "fixtures": fixtures,
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/thursday_report_v3.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    log("✅ Thursday v3 written: logs/thursday_report_v3.json")

if __name__ == "__main__":
    main()
