# src/analysis/tuesday_recap_v3.py
from __future__ import annotations

import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY} if API_FOOTBALL_KEY else {}


# -------------------------
# IO / paths
# -------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _find_project_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "logs").is_dir() and (p / "src").is_dir():
            return p
    return start.parents[2] if len(start.parents) >= 3 else start


_THIS = Path(__file__).resolve()
ROOT = _find_project_root(_THIS.parent)
LOGS = ROOT / "logs"


def _load_latest_friday() -> Dict[str, Any]:
    p1 = LOGS / "friday_shortlist_v3.json"
    if p1.exists():
        return _read_json(p1)
    files = sorted(LOGS.glob("friday_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError("No friday_shortlist_v3.json or friday_*.json found in logs/")
    return _read_json(files[0])


def _history_path() -> Path:
    return LOGS / "tuesday_history_v3.json"


def _load_history() -> Dict[str, Any]:
    p = _history_path()
    if p.exists():
        try:
            data = _read_json(p)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


# -------------------------
# Tiny helpers
# -------------------------

def _sf(v: Any, d: Optional[float] = None) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return d


def _si(v: Any, d: Optional[int] = None) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return d


def _sb_env(name: str, default: bool) -> bool:
    v = os.getenv(name, "true" if default else "false").strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _fmt_tick(won: Optional[bool]) -> str:
    if won is True:
        return "â"
    if won is False:
        return "â"
    return "â"


def _chrono_key(x: Dict[str, Any]) -> Tuple[str, str]:
    return (str(x.get("date") or ""), str(x.get("time_gr") or ""))


def _profit(staked: float, returned: float) -> float:
    return round(float(returned) - float(staked), 2)


def _roi(staked: float, returned: float) -> Optional[float]:
    st = float(staked)
    if st <= 0:
        return None
    return round((float(returned) - st) / st, 4)


# -------------------------
# Learning memory v1
# -------------------------

LEARNING_MEMORY_PATH = LOGS / "market_learning_v1.json"


def _learning_market_key(market: Any) -> str:
    m = str(market or "").strip().lower()
    aliases = {
        "home": "HOME",
        "home (1)": "HOME",
        "1": "HOME",
        "away": "AWAY",
        "away (2)": "AWAY",
        "2": "AWAY",
        "draw": "DRAW",
        "draw (x)": "DRAW",
        "x": "DRAW",
        "over": "OVER_2_5",
        "over 2.5": "OVER_2_5",
        "under": "UNDER_2_5",
        "under 2.5": "UNDER_2_5",
    }
    return aliases.get(m, (str(market or "UNKNOWN").strip().upper().replace(" ", "_") or "UNKNOWN"))


def _learning_odds_band(odds: Any) -> str:
    o = _sf(odds, 0.0) or 0.0
    if o <= 0:
        return "UNKNOWN"
    if o < 1.70:
        return "1.00-1.69"
    if o < 1.90:
        return "1.70-1.89"
    if o < 2.10:
        return "1.90-2.09"
    if o < 2.40:
        return "2.10-2.39"
    return "2.40+"


def _learning_quality_band(quality: Any) -> str:
    q = _sf(quality, None)
    if q is None:
        return "UNKNOWN"
    if q < 0.60:
        return "<0.60"
    if q < 0.70:
        return "0.60-0.69"
    if q < 0.80:
        return "0.70-0.79"
    if q < 0.90:
        return "0.80-0.89"
    return "0.90+"


def _learning_empty_bucket() -> Dict[str, Any]:
    return {
        "picks": 0,
        "settled": 0,
        "wins": 0,
        "losses": 0,
        "unit_stake": 0.0,
        "unit_return": 0.0,
        "unit_profit": 0.0,
        "odds_sum": 0.0,
        "avg_odds": None,
        "hit_rate": None,
        "unit_roi": None,
    }


def _learning_add_bucket(bucket: Dict[str, Any], row: Dict[str, Any]) -> None:
    won = row.get("won")
    odds = float(_sf(row.get("odds"), 0.0) or 0.0)

    bucket["picks"] = int(bucket.get("picks") or 0) + 1
    bucket["odds_sum"] = float(bucket.get("odds_sum") or 0.0) + odds

    if won is True:
        bucket["settled"] = int(bucket.get("settled") or 0) + 1
        bucket["wins"] = int(bucket.get("wins") or 0) + 1
        bucket["unit_stake"] = float(bucket.get("unit_stake") or 0.0) + 1.0
        bucket["unit_return"] = float(bucket.get("unit_return") or 0.0) + max(odds, 0.0)
    elif won is False:
        bucket["settled"] = int(bucket.get("settled") or 0) + 1
        bucket["losses"] = int(bucket.get("losses") or 0) + 1
        bucket["unit_stake"] = float(bucket.get("unit_stake") or 0.0) + 1.0

    bucket["unit_profit"] = float(bucket.get("unit_return") or 0.0) - float(bucket.get("unit_stake") or 0.0)


def _learning_finalize_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(bucket)
    picks = int(out.get("picks") or 0)
    settled = int(out.get("settled") or 0)
    wins = int(out.get("wins") or 0)
    unit_stake = float(out.get("unit_stake") or 0.0)
    unit_return = float(out.get("unit_return") or 0.0)
    odds_sum = float(out.get("odds_sum") or 0.0)

    out["unit_stake"] = round(unit_stake, 2)
    out["unit_return"] = round(unit_return, 2)
    out["unit_profit"] = round(unit_return - unit_stake, 2)
    out["odds_sum"] = round(odds_sum, 2)
    out["avg_odds"] = round(odds_sum / picks, 3) if picks > 0 else None
    out["hit_rate"] = round((wins / settled) * 100.0, 2) if settled > 0 else None
    out["unit_roi"] = round(((unit_return - unit_stake) / unit_stake) * 100.0, 2) if unit_stake > 0 else None
    return out


def _learning_add_group(groups: Dict[str, Dict[str, Any]], key: Any, row: Dict[str, Any]) -> None:
    k = str(key or "UNKNOWN")
    if k not in groups:
        groups[k] = _learning_empty_bucket()
    _learning_add_bucket(groups[k], row)


def _learning_finalize_groups(groups: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {k: _learning_finalize_bucket(v) for k, v in sorted(groups.items(), key=lambda x: x[0])}


def _learning_build_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    markets: Dict[str, Dict[str, Any]] = {}
    leagues: Dict[str, Dict[str, Any]] = {}
    wallets: Dict[str, Dict[str, Any]] = {}
    odds_bands: Dict[str, Dict[str, Any]] = {}
    quality_bands: Dict[str, Dict[str, Any]] = {}
    market_by_league: Dict[str, Dict[str, Any]] = {}
    market_by_odds_band: Dict[str, Dict[str, Any]] = {}

    total = _learning_empty_bucket()

    for row in rows:
        market = _learning_market_key(row.get("market"))
        league = str(row.get("league") or "UNKNOWN")
        wallet = str(row.get("wallet") or "UNKNOWN")
        ob = _learning_odds_band(row.get("odds"))
        qb = _learning_quality_band(row.get("quality"))

        _learning_add_bucket(total, row)
        _learning_add_group(markets, market, row)
        _learning_add_group(leagues, league, row)
        _learning_add_group(wallets, wallet, row)
        _learning_add_group(odds_bands, ob, row)
        _learning_add_group(quality_bands, qb, row)
        _learning_add_group(market_by_league, f"{market} | {league}", row)
        _learning_add_group(market_by_odds_band, f"{market} | {ob}", row)

    return {
        "summary": _learning_finalize_bucket(total),
        "markets": _learning_finalize_groups(markets),
        "leagues": _learning_finalize_groups(leagues),
        "wallets": _learning_finalize_groups(wallets),
        "odds_bands": _learning_finalize_groups(odds_bands),
        "quality_bands": _learning_finalize_groups(quality_bands),
        "market_by_league": _learning_finalize_groups(market_by_league),
        "market_by_odds_band": _learning_finalize_groups(market_by_odds_band),
    }


def _learning_merge_bucket(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(dst or _learning_empty_bucket())
    raw["picks"] = int(raw.get("picks") or 0) + int(src.get("picks") or 0)
    raw["settled"] = int(raw.get("settled") or 0) + int(src.get("settled") or 0)
    raw["wins"] = int(raw.get("wins") or 0) + int(src.get("wins") or 0)
    raw["losses"] = int(raw.get("losses") or 0) + int(src.get("losses") or 0)
    raw["unit_stake"] = float(raw.get("unit_stake") or 0.0) + float(src.get("unit_stake") or 0.0)
    raw["unit_return"] = float(raw.get("unit_return") or 0.0) + float(src.get("unit_return") or 0.0)
    raw["odds_sum"] = float(raw.get("odds_sum") or 0.0) + float(src.get("odds_sum") or 0.0)
    return _learning_finalize_bucket(raw)


def _learning_merge_groups(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(dst or {})
    for key, bucket in (src or {}).items():
        out[key] = _learning_merge_bucket(out.get(key) or _learning_empty_bucket(), bucket)
    return {k: out[k] for k in sorted(out.keys())}


def _load_learning_memory() -> Dict[str, Any]:
    if LEARNING_MEMORY_PATH.exists():
        try:
            data = _read_json(LEARNING_MEMORY_PATH)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _update_learning_memory(current_week: int, friday: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    previous = _load_learning_memory()
    applied_weeks = previous.get("applied_weeks") or []
    fp = _friday_fingerprint(friday)
    week_key = f"week_{current_week}:{fp.get('generated_at') or ''}"

    if week_key in applied_weeks:
        previous["last_update_status"] = "skipped_duplicate"
        previous["generated_at"] = _utc_now_iso()
        _write_json(LEARNING_MEMORY_PATH, previous)
        return previous

    week_memory = _learning_build_from_rows(rows)

    cumulative = {
        "summary": _learning_merge_bucket(previous.get("summary") or _learning_empty_bucket(), week_memory.get("summary") or _learning_empty_bucket()),
        "markets": _learning_merge_groups(previous.get("markets") or {}, week_memory.get("markets") or {}),
        "leagues": _learning_merge_groups(previous.get("leagues") or {}, week_memory.get("leagues") or {}),
        "wallets": _learning_merge_groups(previous.get("wallets") or {}, week_memory.get("wallets") or {}),
        "odds_bands": _learning_merge_groups(previous.get("odds_bands") or {}, week_memory.get("odds_bands") or {}),
        "quality_bands": _learning_merge_groups(previous.get("quality_bands") or {}, week_memory.get("quality_bands") or {}),
        "market_by_league": _learning_merge_groups(previous.get("market_by_league") or {}, week_memory.get("market_by_league") or {}),
        "market_by_odds_band": _learning_merge_groups(previous.get("market_by_odds_band") or {}, week_memory.get("market_by_odds_band") or {}),
    }

    applied_weeks = list(applied_weeks) + [week_key]

    out = {
        "title": "Bombay Market Learning Memory V1",
        "version": "market_learning_v1",
        "generated_at": _utc_now_iso(),
        "status": "ok",
        "note": "Unit ROI treats every settled pick as 1 unit. Core bankroll ROI remains in Tuesday recap.",
        "source": {
            "learning_path": str(LEARNING_MEMORY_PATH),
            "current_week": current_week,
            "friday_fingerprint": fp,
            "rows_added_this_run": len(rows),
        },
        "applied_weeks": applied_weeks,
        "last_week": week_memory,
        **cumulative,
        "last_update_status": "updated",
    }

    _write_json(LEARNING_MEMORY_PATH, out)
    return out


def _learning_row(wallet: str, source: Dict[str, Any], won: Optional[bool], fixture_id: Optional[int]) -> Dict[str, Any]:
    return {
        "wallet": wallet,
        "date": str(source.get("date") or ""),
        "time_gr": str(source.get("time_gr") or ""),
        "league": str(source.get("league") or ""),
        "match": str(source.get("match") or ""),
        "market": str(source.get("market") or ""),
        "odds": float(_sf(source.get("odds"), 0.0) or 0.0),
        "won": won,
        "tier": source.get("tier"),
        "risk_tag": source.get("risk_tag"),
        "quality": _sf(source.get("quality")),
        "fixture_id": fixture_id,
    }


# -------------------------
# Team-name normalizer (legacy fallback)
# -------------------------

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
    }
    return aliases.get(s, s)


# -------------------------
# Settlement fetch
# -------------------------

class FixtureFT:
    def __init__(self, fixture_id: int, home: str, away: str, hg: int, ag: int, status: str):
        self.fixture_id = int(fixture_id)
        self.home = str(home)
        self.away = str(away)
        self.hg = int(hg)
        self.ag = int(ag)
        self.status = str(status)


def fetch_fixture_ft(fixture_id: int) -> Optional[FixtureFT]:
    if not API_FOOTBALL_KEY:
        raise RuntimeError("Missing FOOTBALL_API_KEY for Tuesday recap.")

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"id": int(fixture_id)}
    r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=25)
    r.raise_for_status()
    data = r.json()
    resp = data.get("response") or []
    if not resp:
        return None

    fx = resp[0]
    st = (fx.get("fixture", {}).get("status", {}).get("short") or "")
    hg = fx.get("goals", {}).get("home")
    ag = fx.get("goals", {}).get("away")
    if hg is None or ag is None:
        return None

    home = fx.get("teams", {}).get("home", {}).get("name") or ""
    away = fx.get("teams", {}).get("away", {}).get("name") or ""
    return FixtureFT(
        fixture_id=int(fixture_id),
        home=str(home),
        away=str(away),
        hg=int(hg),
        ag=int(ag),
        status=str(st),
    )


def settle_market(ft: FixtureFT, market: str) -> Optional[bool]:
    m = (market or "").strip()
    total = ft.hg + ft.ag

    if m == "Over 2.5":
        return total >= 3
    if m == "Under 2.5":
        return total <= 2
    if m == "Home (1)":
        return ft.hg > ft.ag
    if m == "Away (2)":
        return ft.ag > ft.hg
    if m == "Draw":
        return ft.hg == ft.ag
    return None


# -------------------------
# Legacy fixture_id resolver
# -------------------------

def _api_fixtures_by_date(date_yyyy_mm_dd: str) -> List[Dict[str, Any]]:
    if not API_FOOTBALL_KEY:
        raise RuntimeError("Missing FOOTBALL_API_KEY for Tuesday recap.")

    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"date": date_yyyy_mm_dd}
    r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=25)
    r.raise_for_status()
    data = r.json()
    resp = data.get("response") or []

    out = []
    for fx in resp:
        st = (fx.get("fixture", {}).get("status", {}).get("short") or "")
        if st not in ("FT", "AET", "PEN"):
            continue
        fid = fx.get("fixture", {}).get("id")
        h = fx.get("teams", {}).get("home", {}).get("name") or ""
        a = fx.get("teams", {}).get("away", {}).get("name") or ""
        if fid is None or not h or not a:
            continue
        out.append({
            "fixture_id": int(fid),
            "home_norm": normalize_team_name(h),
            "away_norm": normalize_team_name(a),
        })
    return out


def _resolve_fixture_id_legacy(line: Dict[str, Any], date_cache: Dict[str, List[Dict[str, Any]]]) -> Optional[int]:
    date = str(line.get("date") or "").strip()
    match = str(line.get("match") or "")
    if not date or "â" not in match:
        return None

    parts = [p.strip() for p in match.split("â", 1)]
    if len(parts) != 2:
        return None

    home_norm = normalize_team_name(parts[0])
    away_norm = normalize_team_name(parts[1])
    if not home_norm or not away_norm:
        return None

    if date not in date_cache:
        date_cache[date] = _api_fixtures_by_date(date)

    for fx in date_cache[date]:
        if fx["home_norm"] == home_norm and fx["away_norm"] == away_norm:
            return int(fx["fixture_id"])

    return None


# -------------------------
# System payout
# -------------------------

def _comb_indices(n: int, k: int) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []

    def rec(start: int, comb: List[int]) -> None:
        if len(comb) == k:
            out.append(tuple(comb))
            return
        for i in range(start, n):
            comb.append(i)
            rec(i + 1, comb)
            comb.pop()

    rec(0, [])
    return out


def system_payout(lines: List[Dict[str, Any]], k: int, columns: int, stake_total: float) -> Dict[str, Any]:
    if not lines or columns <= 0 or stake_total <= 0 or k <= 0:
        return {
            "staked": round(float(stake_total), 2),
            "returned": 0.0,
            "profit": round(-float(stake_total), 2) if stake_total > 0 else 0.0,
            "roi": _roi(float(stake_total), 0.0),
            "winning_columns": 0,
            "stake_per_column": 0.0,
        }

    combos = _comb_indices(len(lines), k)
    used_columns = int(columns) if int(columns) > 0 else len(combos)
    if used_columns <= 0:
        used_columns = len(combos)

    stake_per_col = float(stake_total) / float(used_columns)
    returned = 0.0
    win_cols = 0

    for c in combos:
        ok = True
        prod = 1.0
        for ix in c:
            if lines[ix].get("won") is not True:
                ok = False
                break
            prod *= float(lines[ix].get("odds") or 0.0)
        if ok:
            win_cols += 1
            returned += prod * stake_per_col

    staked = float(stake_total)
    returned = round(returned, 2)
    return {
        "staked": round(staked, 2),
        "returned": returned,
        "profit": _profit(staked, returned),
        "roi": _roi(staked, returned),
        "winning_columns": win_cols,
        "stake_per_column": round(stake_per_col, 4),
    }


# -------------------------
# Friday extraction
# -------------------------

def _extract_all_lines(friday: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    core = (friday.get("core") or {}).get("singles") or []
    fun = (friday.get("funbet") or {}).get("system_pool") or []
    superfun = (friday.get("drawbet") or {}).get("system_pool") or []
    return list(core), list(fun), list(superfun)


# -------------------------
# History guards
# -------------------------

def _friday_fingerprint(friday: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": str(friday.get("title") or ""),
        "generated_at": str(friday.get("generated_at") or ""),
        "window": friday.get("window") or {},
    }


def _assert_history_update_allowed(friday: Dict[str, Any], hist: Dict[str, Any]) -> int:
    history_week_count = _si(hist.get("week_count"), 0) or 0
    current_week = int(
        _si((friday.get("history") or {}).get("next_week"), history_week_count + 1)
        or (history_week_count + 1)
    )

    if history_week_count >= current_week:
        raise ValueError(
            f"Refusing to update history: week {current_week} has already been applied "
            f"(history week_count={history_week_count})."
        )

    last_applied = hist.get("last_applied_friday") or {}
    fp = _friday_fingerprint(friday)
    if (
        str(last_applied.get("generated_at") or "") == fp["generated_at"]
        and fp["generated_at"]
    ):
        raise ValueError(
            f"Refusing to update history: friday generated_at={fp['generated_at']} "
            f"already exists in history."
        )

    return current_week


# -------------------------
# Main recap builder
# -------------------------

def build_tuesday_recap() -> Dict[str, Any]:
    friday = _load_latest_friday()
    hist = _load_history()

    update_history = _sb_env("TUESDAY_UPDATE_HISTORY", True)
    current_week = _assert_history_update_allowed(friday, hist) if update_history else int(
        _si((friday.get("history") or {}).get("next_week"), (_si(hist.get("week_count"), 0) or 0) + 1)
        or ((_si(hist.get("week_count"), 0) or 0) + 1)
    )

    core_lines, fun_lines, superfun_lines = _extract_all_lines(friday)

    date_cache: Dict[str, List[Dict[str, Any]]] = {}
    learning_rows: List[Dict[str, Any]] = []

    def get_fixture_id(line: Dict[str, Any]) -> Optional[int]:
        fxid = line.get("fixture_id") or line.get("id") or line.get("fixtureId")
        if fxid is not None:
            try:
                return int(fxid)
            except Exception:
                pass
        return _resolve_fixture_id_legacy(line, date_cache)

    all_fxids: List[int] = []
    for group in (core_lines, fun_lines, superfun_lines):
        for ln in group:
            fid = get_fixture_id(ln)
            if fid is not None:
                all_fxids.append(fid)
    all_fxids = sorted(set(all_fxids))

    ft_map: Dict[int, FixtureFT] = {}
    missing_fxids: List[int] = []
    for fxid in all_fxids:
        ft = fetch_fixture_ft(fxid)
        if ft is None:
            missing_fxids.append(fxid)
        else:
            ft_map[fxid] = ft

    def settle_line(line: Dict[str, Any]) -> Optional[bool]:
        fid = get_fixture_id(line)
        if fid is None:
            return None
        ft = ft_map.get(fid)
        if not ft:
            return None
        return settle_market(ft, str(line.get("market") or ""))

    # -------------------------
    # CORE singles
    # -------------------------
    core_staked = 0.0
    core_returned = 0.0
    core_won = 0
    core_lost = 0
    core_unsettled = 0
    core_out_lines: List[Dict[str, Any]] = []

    for x in core_lines:
        won = settle_line(x)
        odds = float(_sf(x.get("odds"), 0.0) or 0.0)
        stake = float(_sf(x.get("stake"), 0.0) or 0.0)

        core_staked += stake
        if won is True:
            core_won += 1
            core_returned += stake * odds
        elif won is False:
            core_lost += 1
        else:
            core_unsettled += 1

        learning_rows.append(_learning_row("core", x, won, get_fixture_id(x)))

        core_out_lines.append({
            "tick": _fmt_tick(won),
            "date": str(x.get("date") or ""),
            "time_gr": str(x.get("time_gr") or ""),
            "league": str(x.get("league") or ""),
            "match": str(x.get("match") or ""),
            "market": str(x.get("market") or ""),
            "odds": round(odds, 2),
            "stake": round(stake, 2),
            "tier": x.get("tier"),
            "risk_tag": x.get("risk_tag"),
            "quality": _sf(x.get("quality")),
            "fixture_id": get_fixture_id(x),
        })

    core_out_lines.sort(key=_chrono_key)
    core_profit = _profit(core_staked, core_returned)
    core_roi = _roi(core_staked, core_returned)

    # -------------------------
    # FUN system
    # -------------------------
    fun_out_lines: List[Dict[str, Any]] = []
    for x in fun_lines:
        won = settle_line(x)
        learning_rows.append(_learning_row("fun", x, won, get_fixture_id(x)))
        fun_out_lines.append({
            "tick": _fmt_tick(won),
            "date": str(x.get("date") or ""),
            "time_gr": str(x.get("time_gr") or ""),
            "league": str(x.get("league") or ""),
            "match": str(x.get("match") or ""),
            "market": str(x.get("market") or ""),
            "odds": float(_sf(x.get("odds"), 0.0) or 0.0),
            "won": won,
            "tier": x.get("tier"),
            "risk_tag": x.get("risk_tag"),
            "quality": _sf(x.get("quality")),
            "fixture_id": get_fixture_id(x),
        })
    fun_out_lines.sort(key=_chrono_key)

    fun_sys = (friday.get("funbet") or {}).get("system") or {}
    fun_stake_total = float(_sf(fun_sys.get("stake"), 0.0) or 0.0)
    fun_n = int(_si(fun_sys.get("n"), 0) or 0)
    fun_k = int(_si(fun_sys.get("k"), 0) or 0)
    fun_columns = int(_si(fun_sys.get("columns"), 0) or 0)

    fun_payout = system_payout(fun_out_lines, k=fun_k, columns=fun_columns, stake_total=fun_stake_total)

    # -------------------------
    # SUPERFUN system
    # -------------------------
    superfun_out_lines: List[Dict[str, Any]] = []
    for x in superfun_lines:
        won = settle_line(x)
        learning_rows.append(_learning_row("superfun", x, won, get_fixture_id(x)))
        superfun_out_lines.append({
            "tick": _fmt_tick(won),
            "date": str(x.get("date") or ""),
            "time_gr": str(x.get("time_gr") or ""),
            "league": str(x.get("league") or ""),
            "match": str(x.get("match") or ""),
            "market": str(x.get("market") or ""),
            "odds": float(_sf(x.get("odds"), 0.0) or 0.0),
            "won": won,
            "tier": x.get("tier"),
            "risk_tag": x.get("risk_tag"),
            "quality": _sf(x.get("quality")),
            "source": x.get("source"),
            "fixture_id": get_fixture_id(x),
        })
    superfun_out_lines.sort(key=_chrono_key)

    sf_sys = (friday.get("drawbet") or {}).get("system") or {}
    sf_stake_total = float(_sf(sf_sys.get("stake_total"), _sf(sf_sys.get("stake"), 0.0)) or 0.0)
    sf_n = int(_si(sf_sys.get("n"), 0) or 0)
    sf_k = int(_si(sf_sys.get("k"), 0) or 0)
    sf_columns = int(_si(sf_sys.get("columns"), 0) or 0)

    superfun_payout = system_payout(superfun_out_lines, k=sf_k, columns=sf_columns, stake_total=sf_stake_total)

    # -------------------------
    # Bankroll updates
    # -------------------------
    core_start = float(_sf((friday.get("core") or {}).get("bankroll_start"), 0.0) or 0.0)
    core_open = float(_sf((friday.get("core") or {}).get("open"), 0.0) or 0.0)
    core_after_open = core_start - core_open
    core_end = core_after_open + core_returned

    fun_start = float(_sf((friday.get("funbet") or {}).get("bankroll_start"), 0.0) or 0.0)
    fun_open = float(_sf((friday.get("funbet") or {}).get("open"), 0.0) or 0.0)
    fun_after_open = fun_start - fun_open
    fun_end = fun_after_open + float(fun_payout["returned"] or 0.0)

    sf_start = float(_sf((friday.get("drawbet") or {}).get("bankroll_start"), 0.0) or 0.0)
    sf_open = float(_sf((friday.get("drawbet") or {}).get("open"), 0.0) or 0.0)
    sf_after_open = sf_start - sf_open
    sf_end = sf_after_open + float(superfun_payout["returned"] or 0.0)

    # -------------------------
    # Weekly totals
    # -------------------------
    week_stake_total = round(core_staked + fun_payout["staked"] + superfun_payout["staked"], 2)
    week_return_total = round(core_returned + fun_payout["returned"] + superfun_payout["returned"], 2)
    week_profit_total = _profit(week_stake_total, week_return_total)
    week_roi_total = _roi(week_stake_total, week_return_total)

    # -------------------------
    # Learning memory update
    # -------------------------
    learning_memory = _update_learning_memory(current_week, friday, learning_rows) if update_history else {
        "status": "skipped",
        "reason": "TUESDAY_UPDATE_HISTORY=false",
        "learning_path": str(LEARNING_MEMORY_PATH),
    }

    # -------------------------
    # Cumulative from history
    # -------------------------
    prev_stats = hist.get("stats") or {}
    prev_wallets = prev_stats.get("wallets") or {}
    prev_system = prev_stats.get("system") or {}

    prev_core_stake = float(_sf((prev_wallets.get("core") or {}).get("stake_total"), 0.0) or 0.0)
    prev_core_return = float(_sf((prev_wallets.get("core") or {}).get("return_total"), 0.0) or 0.0)

    prev_fun_stake = float(_sf((prev_wallets.get("fun") or {}).get("stake_total"), 0.0) or 0.0)
    prev_fun_return = float(_sf((prev_wallets.get("fun") or {}).get("return_total"), 0.0) or 0.0)

    prev_sf_stake = float(_sf((prev_wallets.get("superfun") or {}).get("stake_total"), 0.0) or 0.0)
    prev_sf_return = float(_sf((prev_wallets.get("superfun") or {}).get("return_total"), 0.0) or 0.0)

    prev_system_stake = float(_sf(prev_system.get("stake_total"), 0.0) or 0.0)
    prev_system_return = float(_sf(prev_system.get("return_total"), 0.0) or 0.0)

    cum_core_stake = round(prev_core_stake + core_staked, 2)
    cum_core_return = round(prev_core_return + core_returned, 2)

    cum_fun_stake = round(prev_fun_stake + float(fun_payout["staked"]), 2)
    cum_fun_return = round(prev_fun_return + float(fun_payout["returned"]), 2)

    cum_sf_stake = round(prev_sf_stake + float(superfun_payout["staked"]), 2)
    cum_sf_return = round(prev_sf_return + float(superfun_payout["returned"]), 2)

    cum_system_stake = round(prev_system_stake + week_stake_total, 2)
    cum_system_return = round(prev_system_return + week_return_total, 2)

    out = {
        "title": "Bombay Tuesday Recap â v4",
        "generated_at": _utc_now_iso(),
        "window": friday.get("window") or {},
        "week": current_week,

        "corebet": {
            "bankroll_start": round(core_start, 2),
            "open": round(core_open, 2),
            "after_open": round(core_after_open, 2),
            "staked": round(core_staked, 2),
            "returned": round(core_returned, 2),
            "profit_week": core_profit,
            "roi_week": core_roi,
            "won": core_won,
            "lost": core_lost,
            "unsettled": core_unsettled,
            "bankroll_end": round(core_end, 2),
            "lines": core_out_lines,
        },

        "funbet": {
            "bankroll_start": round(fun_start, 2),
            "open": round(fun_open, 2),
            "after_open": round(fun_after_open, 2),
            "system": {
                "n": fun_n,
                "k": fun_k,
                "columns": fun_columns,
                "stake": round(fun_stake_total, 2),
            },
            "payout": fun_payout,
            "profit_week": fun_payout["profit"],
            "roi_week": fun_payout["roi"],
            "bankroll_end": round(fun_end, 2),
            "lines": [{k: v for k, v in x.items() if k != "won"} for x in fun_out_lines],
        },

        "superfunbet": {
            "bankroll_start": round(sf_start, 2),
            "open": round(sf_open, 2),
            "after_open": round(sf_after_open, 2),
            "system": {
                "n": sf_n,
                "k": sf_k,
                "columns": sf_columns,
                "stake_total": round(sf_stake_total, 2),
                "target": (sf_sys.get("target") or None),
                "mode": sf_sys.get("mode"),
            },
            "payout": superfun_payout,
            "profit_week": superfun_payout["profit"],
            "roi_week": superfun_payout["roi"],
            "bankroll_end": round(sf_end, 2),
            "lines": [{k: v for k, v in x.items() if k != "won"} for x in superfun_out_lines],
        },

        "system_week": {
            "stake_total": week_stake_total,
            "return_total": week_return_total,
            "profit": week_profit_total,
            "roi": week_roi_total,
        },

        "learning_memory": {
            "status": learning_memory.get("status"),
            "last_update_status": learning_memory.get("last_update_status"),
            "learning_path": str(LEARNING_MEMORY_PATH),
            "rows_added_this_run": ((learning_memory.get("source") or {}).get("rows_added_this_run")),
            "summary": learning_memory.get("summary"),
            "markets": learning_memory.get("markets"),
        },

        "cumulative": {
            "wallets": {
                "core": {
                    "stake_total": cum_core_stake,
                    "return_total": cum_core_return,
                    "profit": _profit(cum_core_stake, cum_core_return),
                    "roi": _roi(cum_core_stake, cum_core_return),
                },
                "fun": {
                    "stake_total": cum_fun_stake,
                    "return_total": cum_fun_return,
                    "profit": _profit(cum_fun_stake, cum_fun_return),
                    "roi": _roi(cum_fun_stake, cum_fun_return),
                },
                "superfun": {
                    "stake_total": cum_sf_stake,
                    "return_total": cum_sf_return,
                    "profit": _profit(cum_sf_stake, cum_sf_return),
                    "roi": _roi(cum_sf_stake, cum_sf_return),
                },
            },
            "system": {
                "stake_total": cum_system_stake,
                "return_total": cum_system_return,
                "profit": _profit(cum_system_stake, cum_system_return),
                "roi": _roi(cum_system_stake, cum_system_return),
            },
        },

        "summary": {
            "missing_fixtures": missing_fxids,
            "bankrolls_end": {
                "corebet": round(core_end, 2),
                "funbet": round(fun_end, 2),
                "superfunbet": round(sf_end, 2),
            },
            "history_update_enabled": update_history,
        },
    }

    out_hist = {
        "week_count": current_week,
        "as_of": out["generated_at"][:10],
        "note": (
            f"Updated from Week {current_week} real results. "
            f"End-of-week bankrolls: Core {round(core_end, 2)} / "
            f"Fun {round(fun_end, 2)} / SuperFun {round(sf_end, 2)}. "
            f"Total {round(core_end + fun_end + sf_end, 2)}. "
            f"Next run should be Week {current_week + 1}."
        ),
        "bankroll_current": {
            "core": round(core_end, 2),
            "fun": round(fun_end, 2),
            "draw": round(sf_end, 2),
            "total": round(core_end + fun_end + sf_end, 2),
        },
        "stats": {
            "wallets": {
                "core": {
                    "stake_total": cum_core_stake,
                    "return_total": cum_core_return,
                    "profit": _profit(cum_core_stake, cum_core_return),
                    "roi": _roi(cum_core_stake, cum_core_return),
                },
                "fun": {
                    "stake_total": cum_fun_stake,
                    "return_total": cum_fun_return,
                    "profit": _profit(cum_fun_stake, cum_fun_return),
                    "roi": _roi(cum_fun_stake, cum_fun_return),
                },
                "superfun": {
                    "stake_total": cum_sf_stake,
                    "return_total": cum_sf_return,
                    "profit": _profit(cum_sf_stake, cum_sf_return),
                    "roi": _roi(cum_sf_stake, cum_sf_return),
                },
            },
            "system": {
                "stake_total": cum_system_stake,
                "return_total": cum_system_return,
                "profit": _profit(cum_system_stake, cum_system_return),
                "roi": _roi(cum_system_stake, cum_system_return),
            },
        },
        "last_window": out.get("window") or {},
        "last_applied_friday": _friday_fingerprint(friday),
    }

    _write_json(LOGS / "tuesday_recap_v3.json", out)
    if update_history:
        _write_json(_history_path(), out_hist)

    return out


def main() -> int:
    try:
        out = build_tuesday_recap()
        print(json.dumps({
            "status": "ok",
            "saved": str(LOGS / "tuesday_recap_v3.json"),
            "history_saved": str(_history_path()) if _sb_env("TUESDAY_UPDATE_HISTORY", True) else None,
            "learning_saved": str(LEARNING_MEMORY_PATH) if _sb_env("TUESDAY_UPDATE_HISTORY", True) else None,
            "generated_at": out["generated_at"],
            "week": out["week"],
            "system_roi_week": (out.get("system_week") or {}).get("roi"),
            "system_roi_cumulative": ((out.get("cumulative") or {}).get("system") or {}).get("roi"),
            "history_update_enabled": (out.get("summary") or {}).get("history_update_enabled"),
        }))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e), "history_path": str(_history_path())}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
