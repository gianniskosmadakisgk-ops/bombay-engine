import os
import json
import math
import requests
import datetime
import unicodedata
import re
from dateutil import parser

# ============================================================
#  BOMBAY THURSDAY FULL ENGINE v3 (STABILIZED) — PRODUCTION
#
#  DROP-IN REPLACEMENT (fix: pagination στο fetch_fixtures)
#
#  Output contract:
#   - Writes: logs/thursday_report_v3.json
#   - Keeps existing fixture JSON keys intact (schema-safe)
#   - Adds (schema-additive):
#       ev_1 / ev_x / ev_2 / ev_over / ev_under  (prob * offered - 1)
#       score_draw / score_over                 (aliases for ev_x / ev_over)
#
#  Notes:
#   - FAIR + VALUE% computed from MODEL probabilities (stabilized), NOT snap probs.
#   - snap_* probabilities are display-only.
#   - Selection-only hints (Over penalty + flags) DO NOT change model probabilities.
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

# -------------------- SELECTION HINTS (NO MODEL DISTORTION) --------------------
TIGHT_DRAW_THRESHOLD = float(os.getenv("TIGHT_DRAW_THRESHOLD", "0.28"))
TIGHT_LTOTAL_THRESHOLD = float(os.getenv("TIGHT_LTOTAL_THRESHOLD", "2.55"))

# Penalty (VALUE% points) applied ONLY to selection_value_pct_over (not to model fair/value)
OVER_TIGHT_PENALTY_PTS = float(os.getenv("OVER_TIGHT_PENALTY_PTS", "12.0"))
OVER_LOW_TEMPO_EXTRA_PENALTY_PTS = float(os.getenv("OVER_LOW_TEMPO_EXTRA_PENALTY_PTS", "6.0"))

# Candidate hints (stored in flags)
CORE_ODDS_MIN = float(os.getenv("CORE_ODDS_MIN", "1.50"))
CORE_ODDS_MAX = float(os.getenv("CORE_ODDS_MAX", "1.90"))
CORE_VALUE_MIN_PCT = float(os.getenv("CORE_VALUE_MIN_PCT", "10.0"))

FUN_ODDS_MIN = float(os.getenv("FUN_ODDS_MIN", "2.10"))
FUN_ODDS_MAX = float(os.getenv("FUN_ODDS_MAX", "4.00"))
FUN_VALUE_MIN_PCT = float(os.getenv("FUN_VALUE_MIN_PCT", "5.0"))
FUN_MIN_PROB = float(os.getenv("FUN_MIN_PROB", "0.25"))
# ---------------------------------------------------------------------------

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
    """
    IMPORTANT FIX:
    Το παλιό endpoint (league+season) δουλεύει για σένα, αλλά χωρίς pagination
    μπορεί να “κόβει” fixtures. Εδώ προσθέτουμε pagination (page=1..total).
    """
    if not API_FOOTBALL_KEY:
        log("❌ Missing FOOTBALL_API_KEY – NO fixtures will be fetched!")
        return []

    url = f"{API_FOOTBALL_BASE}/fixtures"
    now = datetime.datetime.now(datetime.timezone.utc)

    out = []
    raw_total = 0
    page = 1
    total_pages = None

    # guardrail: δεν πρέπει ποτέ να χρειαστούν πολλές σελίδες
    MAX_PAGES = 30

    while True:
        params = {"league": league_id, "season": FOOTBALL_SEASON, "page": page}

        try:
            r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=25).json()
        except Exception as e:
            log(f"⚠️ Error fetching fixtures for {league_name} page={page}: {e}")
            break

        resp = r.get("response") or []
        paging = r.get("paging") or {}

        if total_pages is None:
            try:
                total_pages = int(paging.get("total")) if paging.get("total") is not None else None
            except Exception:
                total_pages = None

        if not resp:
            break

        raw_total += len(resp)

        for fx in resp:
            try:
                if fx["fixture"]["status"]["short"] != "NS":
                    continue
            except Exception:
                continue

            try:
                dt = parser.isoparse(fx["fixture"]["date"]).astimezone(datetime.timezone.utc)
            except Exception:
                continue

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

        if total_pages is not None and page >= total_pages:
            break

        page += 1
        if page > MAX_PAGES:
            break

    log(
        f"→ {league_name}: raw={raw_total} pages={total_pages if total_pages is not None else '?'} | "
        f"kept={len(out)} fixtures within window"
    )
    return out


# ------------------------- TEAM RECENT GOALS (API-FOOTBALL) -------------------------
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

        att = _clamp(gf_shrunk / league_avg_team, 0.55, 1.85)
        dff = _clamp(ga_shrunk / league_avg_team, 0.55, 1.85)

        return {"att": att, "def": dff}

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


# ------------------------- MAIN PIPELINE -------------------------
def build_fixture_blocks():
    fixtures_out = []
    now = datetime.datetime.now(datetime.timezone.utc)
    to_dt = now + datetime.timedelta(hours=WINDOW_HOURS)
    odds_from = now - datetime.timedelta(hours=8)

    log(f"Using FOOTBALL_SEASON={FOOTBALL_SEASON}")
    log(f"Window: next {WINDOW_HOURS} hours | USE_ODDS_API={USE_ODDS_API}")

    if not API_FOOTBALL_KEY:
        log("❌ FOOTBALL_API_KEY is missing. Aborting.")
        return []

    all_fixtures = []
    for lg_name, lg_id in LEAGUES.items():
        all_fixtures.extend(fetch_fixtures(lg_id, lg_name))
    log(f"Total fixtures collected: {len(all_fixtures)}")

    # ---- odds cache ----
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
            off1=off_1,
            offx=off_x,
            off2=off_2,
            offo=off_o,
            offu=off_u,
        )

        s_ph, s_pd, s_pa, s_po, s_pu = market_snap_probs(
            m_ph, m_pd, m_pa, m_po, m_pu, off_1, off_x, off_2, off_o, off_u
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

        conf = _confidence_score(home_stats, away_stats, match_debug, lam_h, lam_a)
        conf_band = "high" if conf >= 0.70 else ("mid" if conf >= 0.55 else "low")

        flags = {
            "tight_game": bool(tight_game),
            "low_tempo_league": bool(low_tempo),
            "odds_matched": bool(match_debug.get("matched")),
            "confidence": round(conf, 3),
            "confidence_band": conf_band,

            "core_1": _candidate(off_1, v1, m_ph, CORE_ODDS_MIN, CORE_ODDS_MAX, CORE_VALUE_MIN_PCT, 0.0),
            "core_x": _candidate(off_x, vx, m_pd, CORE_ODDS_MIN, CORE_ODDS_MAX, CORE_VALUE_MIN_PCT, 0.0),
            "core_2": _candidate(off_2, v2, m_pa, CORE_ODDS_MIN, CORE_ODDS_MAX, CORE_VALUE_MIN_PCT, 0.0),
            "core_over": _candidate(off_o, vo, m_po, CORE_ODDS_MIN, CORE_ODDS_MAX, CORE_VALUE_MIN_PCT, 0.0),
            "core_under": _candidate(off_u, vu, m_pu, CORE_ODDS_MIN, CORE_ODDS_MAX, CORE_VALUE_MIN_PCT, 0.0),

            "fun_1": _candidate(off_1, v1, m_ph, FUN_ODDS_MIN, FUN_ODDS_MAX, FUN_VALUE_MIN_PCT, FUN_MIN_PROB),
            "fun_x": _candidate(off_x, vx, m_pd, FUN_ODDS_MIN, FUN_ODDS_MAX, FUN_VALUE_MIN_PCT, FUN_MIN_PROB),
            "fun_2": _candidate(off_2, v2, m_pa, FUN_ODDS_MIN, FUN_ODDS_MAX, FUN_VALUE_MIN_PCT, FUN_MIN_PROB),
            "fun_over": _candidate(off_o, vo, m_po, FUN_ODDS_MIN, FUN_ODDS_MAX, FUN_VALUE_MIN_PCT, FUN_MIN_PROB),
            "fun_under": _candidate(off_u, vu, m_pu, FUN_ODDS_MIN, FUN_ODDS_MAX, FUN_VALUE_MIN_PCT, FUN_MIN_PROB),
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
                "model": "bombay_multiplicative_dc_v3_stabilized",

                "lambda_home": round(lam_h, 3),
                "lambda_away": round(lam_a, 3),

                "home_prob": round(m_ph, 3),
                "draw_prob": round(m_pd, 3),
                "away_prob": round(m_pa, 3),
                "over_2_5_prob": round(m_po, 3),
                "under_2_5_prob": round(m_pu, 3),

                "snap_home_prob": round(s_ph, 3),
                "snap_draw_prob": round(s_pd, 3),
                "snap_away_prob": round(s_pa, 3),
                "snap_over_2_5_prob": round(s_po, 3),
                "snap_under_2_5_prob": round(s_pu, 3),

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

                "score_draw": evx,
                "score_over": evo,

                "selection_value_pct_over": selection_vo,
                "over_value_penalty_pts": round(over_penalty_pts, 2),
                "flags": flags,

                "odds_match": match_debug,
                "league_baseline": league_baseline,
            }
        )

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

    log(f"✅ Thursday v3 FULL READY. Fixtures: {len(fixtures)}")


if __name__ == "__main__":
    main()
