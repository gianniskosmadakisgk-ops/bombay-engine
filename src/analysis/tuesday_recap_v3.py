# src/analysis/tuesday_recap_v3.py
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
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

# -------------------------
# Settlement helpers
# -------------------------

def _fmt_tick(won: Optional[bool]) -> str:
    if won is True: return "✅"
    if won is False: return "❌"
    return "—"

def _sf(v: Any, d: Optional[float] = None) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return d

@dataclass
class FixtureFT:
    fixture_id: int
    home: str
    away: str
    hg: int
    ag: int
    status: str  # FT, AET, etc

def fetch_fixture_ft(fixture_id: int) -> Optional[FixtureFT]:
    if not API_FOOTBALL_KEY:
        raise RuntimeError("Missing FOOTBALL_API_KEY for Tuesday recap.")
    url = f"{API_FOOTBALL_BASE}/fixtures"
    params = {"id": int(fixture_id)}
    r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=25).json()
    resp = r.get("response") or []
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

    # unknown market type
    return None

# -------------------------
# System payout calculators
# -------------------------

def _comb_indices(n: int, k: int) -> List[Tuple[int, ...]]:
    # small n only (n<=7) — brute force ok
    out: List[Tuple[int, ...]] = []
    idx = list(range(k))

    def rec(start: int, comb: List[int]):
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
    """
    lines: each has {won(bool|None), odds(float)}
    We assume stake_total is TOTAL spend across all columns.
    payout = sum(product_odds * stake_per_col) for each winning combo.
    """
    if not lines or columns <= 0 or stake_total <= 0:
        return {"staked": float(stake_total), "returned": 0.0, "roi": None, "winning_columns": 0}

    stake_per_col = stake_total / float(columns)
    n = len(lines)
    combos = _comb_indices(n, k)

    returned = 0.0
    win_cols = 0
    for c in combos:
        ok = True
        prod = 1.0
        for ix in c:
            w = lines[ix].get("won")
            if w is not True:
                ok = False
                break
            prod *= float(lines[ix].get("odds") or 0.0)
        if ok:
            win_cols += 1
            returned += prod * stake_per_col

    staked = float(stake_total)
    roi = None
    if staked > 0:
        roi = round((returned - staked) / staked, 4)

    return {
        "staked": round(staked, 2),
        "returned": round(returned, 2),
        "roi": roi,
        "winning_columns": win_cols,
        "stake_per_column": round(stake_per_col, 4),
    }

# -------------------------
# Extract fixtures from Friday
# -------------------------

def _extract_all_lines(friday: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    core = (friday.get("core") or {}).get("singles") or []
    fun = (friday.get("funbet") or {}).get("system_pool") or []
    draw = (friday.get("drawbet") or {}).get("system_pool") or []
    return list(core), list(fun), list(draw)

def _fixture_ids_from_lines(lines: List[Dict[str, Any]]) -> List[int]:
    ids = []
    for x in lines:
        fxid = x.get("fixture_id") or x.get("id") or x.get("fixtureId")
        if fxid is None:
            continue
        try:
            ids.append(int(fxid))
        except Exception:
            continue
    return ids

def _chrono_key(x: Dict[str, Any]) -> Tuple[str, str]:
    return (str(x.get("date") or ""), str(x.get("time_gr") or ""))

# -------------------------
# Main recap builder
# -------------------------

def build_tuesday_recap() -> Dict[str, Any]:
    friday = _load_latest_friday()

    core_lines, fun_lines, draw_lines = _extract_all_lines(friday)
    all_fxids = sorted(set(_fixture_ids_from_lines(core_lines) + _fixture_ids_from_lines(fun_lines) + _fixture_ids_from_lines(draw_lines)))

    # fetch FT once per fixture_id
    ft_map: Dict[int, FixtureFT] = {}
    missing = []
    for fxid in all_fxids:
        ft = fetch_fixture_ft(fxid)
        if ft is None:
            missing.append(fxid)
        else:
            ft_map[fxid] = ft

    # settle a line
    def settle_line(line: Dict[str, Any]) -> Optional[bool]:
        fxid = line.get("fixture_id") or line.get("id") or line.get("fixtureId")
        if fxid is None:
            return None
        fxid = int(fxid)
        ft = ft_map.get(fxid)
        if not ft:
            return None
        return settle_market(ft, str(line.get("market") or ""))

    # ---- CORE ROI (singles only) ----
    core_staked = 0.0
    core_returned = 0.0
    core_won = 0
    core_lost = 0

    core_out_lines = []
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

        core_out_lines.append({
            "tick": _fmt_tick(won),
            "date": str(x.get("date") or ""),
            "time_gr": str(x.get("time_gr") or ""),
            "league": str(x.get("league") or ""),
            "match": str(x.get("match") or ""),
            "market": str(x.get("market") or ""),
            "odds": round(odds, 2),
            "stake": round(stake, 2),
        })

    core_out_lines.sort(key=_chrono_key)
    core_roi = None
    if core_staked > 0:
        core_roi = round((core_returned - core_staked) / core_staked, 4)

    # ---- FUN system payout (4/7 columns 35) ----
    fun_out_lines = []
    for x in fun_lines:
        won = settle_line(x)
        fun_out_lines.append({
            "tick": _fmt_tick(won),
            "date": str(x.get("date") or ""),
            "time_gr": str(x.get("time_gr") or ""),
            "league": str(x.get("league") or ""),
            "match": str(x.get("match") or ""),
            "market": str(x.get("market") or ""),
            "odds": float(_sf(x.get("odds"), 0.0) or 0.0),
            "won": won,
        })
    fun_out_lines.sort(key=_chrono_key)

    fun_stake_total = float(_sf(((friday.get("funbet") or {}).get("system") or {}).get("stake"), 0.0) or 0.0)
    fun_payout = system_payout(fun_out_lines, k=4, columns=35, stake_total=fun_stake_total)

    # ---- DRAW system payout (2/5 columns 10) ----
    draw_out_lines = []
    for x in draw_lines:
        won = settle_line(x)
        draw_out_lines.append({
            "tick": _fmt_tick(won),
            "date": str(x.get("date") or ""),
            "time_gr": str(x.get("time_gr") or ""),
            "league": str(x.get("league") or ""),
            "match": str(x.get("match") or ""),
            "market": str(x.get("market") or ""),
            "odds": float(_sf(x.get("odds"), 0.0) or 0.0),
            "won": won,
        })
    draw_out_lines.sort(key=_chrono_key)

    draw_stake_total = float(_sf(((friday.get("drawbet") or {}).get("system") or {}).get("stake"), 0.0) or 0.0)
    draw_payout = system_payout(draw_out_lines, k=2, columns=10, stake_total=draw_stake_total)

    # ---- Bankroll updates ----
    core_start = float(_sf((friday.get("core") or {}).get("bankroll_start"), 0.0) or 0.0)
    core_open  = float(_sf((friday.get("core") or {}).get("open"), 0.0) or 0.0)
    core_after_open = core_start - core_open
    core_end = core_after_open + core_returned

    fun_start = float(_sf((friday.get("funbet") or {}).get("bankroll_start"), 0.0) or 0.0)
    fun_open  = float(_sf((friday.get("funbet") or {}).get("open"), 0.0) or 0.0)
    fun_after_open = fun_start - fun_open
    fun_end = fun_after_open + float(fun_payout["returned"] or 0.0)

    draw_start = float(_sf((friday.get("drawbet") or {}).get("bankroll_start"), 0.0) or 0.0)
    draw_open  = float(_sf((friday.get("drawbet") or {}).get("open"), 0.0) or 0.0)
    draw_after_open = draw_start - draw_open
    draw_end = draw_after_open + float(draw_payout["returned"] or 0.0)

    # Weekly ROI per bankroll (real)
    def _roi(staked: float, returned: float) -> Optional[float]:
        if staked <= 0:
            return None
        return round((returned - staked) / staked, 4)

    out = {
        "title": "Bombay Tuesday Recap — v3 (API-Football settlement)",
        "generated_at": _utc_now_iso(),
        "window": friday.get("window") or {},

        "corebet": {
            "bankroll_start": round(core_start, 2),
            "open": round(core_open, 2),
            "after_open": round(core_after_open, 2),
            "staked": round(core_staked, 2),
            "returned": round(core_returned, 2),
            "won": core_won,
            "lost": core_lost,
            "roi_week": _roi(core_staked, core_returned),
            "bankroll_end": round(core_end, 2),
            "lines": core_out_lines,
        },

        "funbet": {
            "bankroll_start": round(fun_start, 2),
            "open": round(fun_open, 2),
            "after_open": round(fun_after_open, 2),
            "system": {"n": 7, "k": 4, "columns": 35, "stake": round(fun_stake_total, 2)},
            "payout": fun_payout,
            "roi_week": fun_payout.get("roi"),
            "bankroll_end": round(fun_end, 2),
            "lines": [
                {k: v for k, v in x.items() if k != "won"} for x in fun_out_lines
            ],
        },

        "drawbet": {
            "bankroll_start": round(draw_start, 2),
            "open": round(draw_open, 2),
            "after_open": round(draw_after_open, 2),
            "system": {"n": 5, "k": 2, "columns": 10, "stake": round(draw_stake_total, 2)},
            "payout": draw_payout,
            "roi_week": draw_payout.get("roi"),
            "bankroll_end": round(draw_end, 2),
            "lines": [
                {k: v for k, v in x.items() if k != "won"} for x in draw_out_lines
            ],
        },

        "summary": {
            "missing_fixtures": missing,
            "bankrolls_end": {
                "corebet": round(core_end, 2),
                "funbet": round(fun_end, 2),
                "drawbet": round(draw_end, 2),
            },
        },
    }

    # Next-week portable history
    out_hist = {
        "as_of": out["generated_at"][:10],
        "bankrolls": out["summary"]["bankrolls_end"],
        "last_window": out.get("window") or {},
    }

    _write_json(LOGS / "tuesday_recap_v3.json", out)
    _write_json(LOGS / "tuesday_history_v3_next.json", out_hist)
    return out

def main() -> int:
    try:
        out = build_tuesday_recap()
        print(json.dumps({"status": "ok", "saved": str(LOGS / "tuesday_recap_v3.json"), "generated_at": out["generated_at"]}))
        return 0
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
