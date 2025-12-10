import os
import json
import requests
import datetime
from dateutil import parser

# ============================================================
#  THURSDAY ENGINE v3 (with TheOddsAPI)
#  - Τραβάει fixtures από API-FOOTBALL
#  - Υπολογίζει dummy fair odds / probabilities
#  - Τραβάει odds από TheOddsAPI (TheOddChappie 😄)
#  - Γράφει logs/thursday_report_v3.json
# ============================================================

# ------------------------- API KEYS -------------------------
API_FOOTBALL_KEY = os.getenv("FOOTBALL_API_KEY")
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4/sports"

FOOTBALL_SEASON = os.getenv("FOOTBALL_SEASON", "2025")

HEADERS_FOOTBALL = {"x-apisports-key": API_FOOTBALL_KEY}

# Χρησιμοποιούμε TheOddsAPI
USE_ODDS_API = True

# ------------------------- LEAGUES -------------------------
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

# 3 ημέρες ακριβώς (72 ώρες)
WINDOW_HOURS = 72

# ------------------------- LEAGUE → SPORT KEY (TheOddsAPI) -------------------------
LEAGUE_TO_SPORT = {
    "Premier League": "soccer_epl",
    "Championship": "soccer_efl_champ",
    "La Liga": "soccer_spain_la_liga",
    "La Liga 2": "soccer_spain_segunda_division",
    "Serie A": "soccer_italy_serie_a",
    "Serie B": "soccer_italy_serie_b",
    "Bundesliga": "soccer_germany_bundesliga",
    "Bundesliga 2": "s
