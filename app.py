from flask import request  # αν δεν υπάρχει ήδη

@app.route("/thursday-analysis-v3", methods=["GET"])
def gpt_thursday():
    report, error = load_json_report("logs/thursday_report_v3.json")
    if report is None:
        return jsonify({
            "status":"error",
            "message":"Thursday report not available",
            "error": error,
            "timestamp": datetime.utcnow().isoformat(),
            "report": None
        }), 404

    # ---- Chunking controls ----
    # per_page = πόσες λίγκες ανά απάντηση (default 3)
    try:
        per_page = int(request.args.get("per_page", "3"))
    except Exception:
        per_page = 3
    per_page = max(1, min(5, per_page))

    # cursor = index στο league list (default 0)
    try:
        cursor = int(request.args.get("cursor", "0"))
    except Exception:
        cursor = 0
    cursor = max(0, cursor)

    # optional explicit leagues=Premier League,La Liga
    leagues_param = (request.args.get("leagues") or "").strip()
    leagues_filter = [x.strip() for x in leagues_param.split(",") if x.strip()] if leagues_param else []

    fixtures = report.get("fixtures") or []
    engine_leagues = report.get("engine_leagues") or sorted({f.get("league") for f in fixtures if f.get("league")})

    # choose which leagues to return
    if leagues_filter:
        chosen_leagues = [lg for lg in engine_leagues if lg in leagues_filter]
        next_cursor = None
    else:
        chosen_leagues = engine_leagues[cursor:cursor + per_page]
        next_cursor = cursor + per_page if (cursor + per_page) < len(engine_leagues) else None

    # filter fixtures by chosen leagues
    chunk_fixtures = [f for f in fixtures if f.get("league") in set(chosen_leagues)]

    # optional lite=1 to drop heavy fields (default 1 to avoid size issues)
    lite = (request.args.get("lite", "1").lower() in ("1", "true", "yes"))
    if lite:
        keep = {
            "fixture_id","date","time","league_id","league","home","away","model",
            "lambda_home","lambda_away","total_lambda","abs_lambda_gap",
            "home_prob","draw_prob","away_prob","over_2_5_prob","under_2_5_prob",
            "fair_1","fair_x","fair_2","fair_over_2_5","fair_under_2_5",
            "offered_1","offered_x","offered_2","offered_over_2_5","offered_under_2_5",
            "value_pct_1","value_pct_x","value_pct_2","value_pct_over","value_pct_under",
            "ev_1","ev_x","ev_2","ev_over","ev_under",
            "flags","odds_match"
        }
        def slim(x):
            return {k: x.get(k) for k in keep if k in x}
        chunk_fixtures = [slim(f) for f in chunk_fixtures]

    chunk_report = {
        "generated_at": report.get("generated_at"),
        "season_used": report.get("season_used"),
        "window": report.get("window"),
        "engine_leagues": engine_leagues,
        "fixtures_total": len(fixtures),
        "chunk": {
            "cursor": cursor,
            "per_page": per_page,
            "leagues": chosen_leagues,
            "next_cursor": next_cursor
        },
        "fixtures": chunk_fixtures
    }

    return jsonify({
        "status":"ok",
        "timestamp": datetime.utcnow().isoformat(),
        "report": chunk_report
    })
