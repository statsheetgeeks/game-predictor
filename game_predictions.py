"""
game_predictions.py
===================
Game Predictions Based on Lineups — XGBoost + Random Forest Ensemble
Runs daily via GitHub Actions. Outputs:
  - docs/game-predictions-YYYY-MM-DD.png   (chart)
  - docs/game-predictions-YYYY-MM-DD.csv   (raw data)
  - docs/index.html                        (updated dashboard)
"""

import requests, json, time, warnings, os, pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless — no display needed in CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import date, datetime, timedelta
from pathlib import Path

from pybaseball import statcast_batter
from pybaseball import cache as pybb_cache
pybb_cache.enable()

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent   # repo root
DOCS_DIR  = ROOT / 'docs'
DOCS_DIR.mkdir(exist_ok=True)

CACHE_VERSION  = 3
CACHE_DIR      = ROOT / 'lineup_scout_cache'
HIST_CACHE_DIR = CACHE_DIR / 'historical'
LIVE_CACHE_DIR = CACHE_DIR / 'live'
for d in [CACHE_DIR, HIST_CACHE_DIR, LIVE_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Date / season config ─────────────────────────────────────────────────────
BASE           = 'https://statsapi.mlb.com/api/v1'
TODAY          = date.today().isoformat()
YESTERDAY      = (date.today() - timedelta(days=1)).isoformat()
CURRENT_SEASON = date.today().year
HIST_SEASONS   = [2022, 2023, 2024, 2025]

SEASON_START = {
    2022: '2022-04-07', 2023: '2023-03-30',
    2024: '2024-03-20', 2025: '2025-03-27',
    2026: '2026-03-27',
}
LIVE_SEASON_START = SEASON_START.get(CURRENT_SEASON, f'{CURRENT_SEASON}-03-27')

# ── Park factors (2024 values, 100 = neutral) ────────────────────────────────
PARK_RUN_FACTORS = {
    3313:115, 2392:108, 15:106,  2394:104, 4169:103, 2681:102,
    2395:101, 3289:100, 2397:99,  1:98,    31:97,   2399:96,
    2603:98,  680:97,   5325:95,  2602:97, 3633:100, 32:99,
    2889:99,  14:101,  2408:96,  4705:100, 2500:101, 5:100,
    2406:99,  4321:98,  2407:97, 4140:102,
}

# ── Batting order position weights ───────────────────────────────────────────
ORDER_WEIGHTS = {
    1: 1.20, 2: 1.15, 3: 1.10, 4: 1.25,
    5: 1.05, 6: 0.95, 7: 0.85, 8: 0.75, 9: 0.70,
}
_OW_SUM = sum(ORDER_WEIGHTS.values())

# ── Dome / retractable-roof venues ───────────────────────────────────────────
DOME_VENUES = {4705, 2407, 32, 2408, 4140, 3633, 2602}

# ── Park GPS for weather ─────────────────────────────────────────────────────
PARK_COORDS = {
    3313:(39.756,-104.994), 2392:(39.097,-84.507),  15:(42.347,-71.097),
    2394:(41.948,-87.655),  4169:(32.747,-97.083),  2681:(39.906,-75.167),
    2395:(39.284,-76.622),  3289:(40.829,-73.926),  2397:(34.074,-118.240),
    1:(33.800,-117.883),    31:(47.591,-122.332),   2399:(37.778,-122.389),
    2603:(38.623,-90.193),  680:(32.707,-117.157),  5325:(33.891,-84.468),
    2602:(33.445,-112.067), 3633:(29.757,-95.355),  32:(44.982,-93.278),
    2889:(40.447,-80.006),  14:(41.830,-87.634),    2408:(25.778,-80.220),
    4705:(40.757,-73.846),  2500:(43.028,-87.971),  5:(39.052,-94.480),
    2406:(41.496,-81.685),  4321:(38.873,-77.008),  2407:(27.768,-82.653),
    4140:(43.641,-79.389),
}

# ── Elite pitcher pedigree ───────────────────────────────────────────────────
PITCHER_PEDIGREE = {
    694973: {'tier':1,'label':'Paul Skenes',     'era_adj': 0.72},
    675911: {'tier':1,'label':'Spencer Strider',  'era_adj': 0.75},
    543037: {'tier':1,'label':'Gerrit Cole',      'era_adj': 0.78},
    554430: {'tier':1,'label':'Zack Wheeler',     'era_adj': 0.78},
    669373: {'tier':1,'label':'Tarik Skubal',     'era_adj': 0.76},
    519242: {'tier':1,'label':'Chris Sale',       'era_adj': 0.78},
    660271: {'tier':1,'label':'Shohei Ohtani',    'era_adj': 0.74},
    808982: {'tier':1,'label':'Y. Yamamoto',      'era_adj': 0.76},
    645261: {'tier':1,'label':'Sandy Alcantara',  'era_adj': 0.78},
    669203: {'tier':2,'label':'Corbin Burnes',    'era_adj': 0.80},
    656302: {'tier':2,'label':'Dylan Cease',      'era_adj': 0.83},
    642547: {'tier':2,'label':'Freddy Peralta',   'era_adj': 0.84},
    657277: {'tier':2,'label':'Logan Webb',       'era_adj': 0.84},
    607192: {'tier':2,'label':'Tyler Glasnow',    'era_adj': 0.82},
    669456: {'tier':2,'label':'Shane Bieber',     'era_adj': 0.82},
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MLB Stats API Helpers
# ─────────────────────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({'User-Agent': 'GamePredictions/1.0'})
_api_cache = {}

def mlb_get(endpoint, params=None, retries=3):
    url = f'{BASE}/{endpoint}'
    key = url + str(sorted(params.items()) if params else '')
    if key in _api_cache:
        return _api_cache[key]
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            _api_cache[key] = data
            return data
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5 ** attempt)
            else:
                return {}

def safe_float(val, default=0.0):
    try:    return float(val) if val not in (None, '', 'null') else default
    except: return default

def safe_int(val, default=0):
    try:    return int(val) if val not in (None, '', 'null') else default
    except: return default

def _first_split(data, stat_type):
    for s in data.get('stats', []):
        if s.get('type', {}).get('displayName') == stat_type:
            sp = s.get('splits', [])
            return sp[0].get('stat', {}) if sp else {}
    return {}

def get_player_season_stats(player_id, season):
    def _fetch(s):
        d = mlb_get(f'people/{player_id}/stats',
                    {'stats': 'season,career', 'group': 'hitting', 'season': s})
        return _first_split(d, 'statsSingleSeason'), _first_split(d, 'career')
    s_stat, c_stat = _fetch(season)
    if safe_int(s_stat.get('atBats')) < 5 and season > 2020:
        ps, pc = _fetch(season - 1)
        if safe_int(ps.get('atBats')) > safe_int(s_stat.get('atBats')):
            s_stat = ps
        if not c_stat:
            c_stat = pc
    return s_stat, c_stat

def get_game_log(player_id, season, limit=14):
    d = mlb_get(f'people/{player_id}/stats',
                {'stats': 'gameLog', 'group': 'hitting', 'season': season, 'limit': limit})
    sl = d.get('stats', [])
    return sl[0].get('splits', []) if sl else []

def get_platoon_splits(player_id, season):
    d = mlb_get(f'people/{player_id}/stats',
                {'stats': 'statSplits', 'group': 'hitting',
                 'season': season, 'sitCodes': 'vl,vr'})
    vs_L, vs_R = {}, {}
    for s in d.get('stats', []):
        for split in s.get('splits', []):
            code = split.get('split', {}).get('code', '')
            stat = split.get('stat', {})
            if code == 'vl': vs_L = stat
            elif code == 'vr': vs_R = stat
    return vs_L, vs_R

def get_pitcher_stats(pitcher_id, season):
    d = mlb_get(f'people/{pitcher_id}/stats',
                {'stats': 'season,career', 'group': 'pitching', 'season': season})
    return _first_split(d, 'statsSingleSeason'), _first_split(d, 'career')

def get_pitcher_hand(pitcher_id):
    d = mlb_get(f'people/{pitcher_id}', {'hydrate': 'pitchHand'})
    return d.get('people', [{}])[0].get('pitchHand', {}).get('code', 'U')

def get_pitcher_last_n_starts(pitcher_id, season, n=3):
    d = mlb_get(f'people/{pitcher_id}/stats',
                {'stats': 'gameLog', 'group': 'pitching', 'season': season, 'limit': n})
    splits = d.get('stats', [{}])[0].get('splits', [])
    if not splits:
        return {'last_era': None, 'last_whip': None, 'last_ip': 0}
    tot_ip, tot_er, tot_bb, tot_h = 0.0, 0, 0, 0
    for g in splits[:n]:
        st = g.get('stat', {})
        ip = safe_float(st.get('inningsPitched', 0))
        tot_ip += ip
        tot_er += safe_int(st.get('earnedRuns', 0))
        tot_bb += safe_int(st.get('baseOnBalls', 0))
        tot_h  += safe_int(st.get('hits', 0))
    era  = (tot_er / tot_ip * 9) if tot_ip > 0 else None
    whip = ((tot_bb + tot_h) / tot_ip) if tot_ip > 0 else None
    return {'last_era': era, 'last_whip': whip, 'last_ip': tot_ip}

def get_team_bullpen(team_id, season):
    d = mlb_get(f'teams/{team_id}/stats',
                {'stats': 'season', 'group': 'pitching', 'season': season})
    sl = d.get('stats', [])
    if not sl: return {}
    sp = sl[0].get('splits', [])
    return sp[0].get('stat', {}) if sp else {}

def get_schedule_for_date(game_date):
    d = mlb_get('schedule', {
        'sportId': 1, 'date': game_date, 'gameType': 'R',
        'hydrate': 'probablePitcher,team,venue,linescore',
    })
    return [g for dt in d.get('dates', []) for g in dt.get('games', [])]

def get_game_lineup(game_pk):
    d = mlb_get(f'game/{game_pk}/boxscore')
    lineups = {}
    for side in ['home', 'away']:
        players = d.get('teams', {}).get(side, {}).get('players', {})
        batter_order = []
        for pid_str, info in players.items():
            bo = info.get('battingOrder', '')
            if bo and str(bo).strip():
                try:
                    order_num = int(str(bo).strip())
                    pid = info.get('person', {}).get('id')
                    pos = info.get('position', {}).get('abbreviation', '')
                    if pid and pos not in ('P', 'SP', 'RP'):
                        batter_order.append((order_num, pid))
                except (ValueError, TypeError):
                    pass
        batter_order.sort(key=lambda x: x[0])
        seen_slots, deduped = set(), []
        for order_num, pid in batter_order:
            slot = order_num // 100
            if slot not in seen_slots:
                seen_slots.add(slot)
                deduped.append(pid)
        lineups[side] = deduped[:9]
    return lineups if lineups.get('home') and lineups.get('away') else None

def get_game_score(game_pk):
    d = mlb_get(f'game/{game_pk}/linescore')
    hr = d.get('teams', {}).get('home', {}).get('runs')
    ar = d.get('teams', {}).get('away', {}).get('runs')
    if hr is None or ar is None:
        return None, None
    return safe_int(hr), safe_int(ar)

_statcast_cache = {}

def get_statcast_batter_season(mlb_id, season):
    key = (mlb_id, season)
    if key in _statcast_cache:
        return _statcast_cache[key]
    try:
        df = statcast_batter(f'{season}-03-01', f'{season}-11-01', player_id=mlb_id)
        if df is None or df.empty:
            _statcast_cache[key] = {}
            return {}
        bbe = df[df['launch_speed'].notna() & df['launch_angle'].notna()].copy()
        if bbe.empty:
            _statcast_cache[key] = {}
            return {}
        n = len(bbe)
        xba = 0.0
        if 'estimated_ba_using_speedangle' in df.columns:
            xba_vals = df['estimated_ba_using_speedangle'].dropna()
            xba = float(xba_vals.mean()) if len(xba_vals) >= 5 else 0.0
        barrel_rate = 0.0
        if 'barrel' in bbe.columns:
            barrel_col = bbe['barrel'].fillna(0)
            barrel_sum = float(barrel_col.sum())
            if barrel_sum > 0:
                barrel_rate = barrel_sum / n
        if barrel_rate == 0.0 and 'launch_speed' in bbe.columns and 'launch_angle' in bbe.columns:
            ev = bbe['launch_speed']
            la = bbe['launch_angle']
            barrels_mask = (ev >= 98) & (la >= 8) & (la <= 50)
            barrel_rate  = float(barrels_mask.sum() / n)
        avg_ev       = round(float(bbe['launch_speed'].mean()), 1)
        hard_hit_pct = round(float((bbe['launch_speed'] >= 95).sum() / n), 4)
        result = {
            'xba':          round(xba, 3),
            'barrel_rate':  round(barrel_rate, 4),
            'avg_ev':       avg_ev,
            'hard_hit_pct': hard_hit_pct,
        }
        _statcast_cache[key] = result
        return result
    except Exception:
        _statcast_cache[key] = {}
        return {}

def fetch_weather(lat, lon):
    try:
        r = requests.get('https://api.open-meteo.com/v1/forecast', params={
            'latitude': lat, 'longitude': lon,
            'hourly': 'temperature_2m,precipitation_probability,windspeed_10m',
            'temperature_unit': 'fahrenheit', 'windspeed_unit': 'mph',
            'forecast_days': 1, 'timezone': 'auto'
        }, timeout=10)
        d = r.json()
        gh = min(max(datetime.now().hour, 13), 19)
        return {
            'temp':   round(d['hourly']['temperature_2m'][gh]),
            'wind':   round(d['hourly']['windspeed_10m'][gh]),
            'precip': round(d['hourly']['precipitation_probability'][gh]),
        }
    except:
        return {'temp': 72, 'wind': 8, 'precip': 10}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Player Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_batter_features(pid, season, pit_hand='R'):
    try:
        h_s, h_c = get_player_season_stats(pid, season)
    except Exception:
        h_s, h_c = {}, {}
    h_avg = safe_float(h_s.get('avg') or h_c.get('avg'))
    if h_avg == 0.0 and not h_s and not h_c:
        return {}
    h_obp = safe_float(h_s.get('obp')   or h_c.get('obp'),  0.315)
    h_slg = safe_float(h_s.get('slg')   or h_c.get('slg'),  0.390)
    h_ops = safe_float(h_s.get('ops')   or h_c.get('ops'),  h_obp + h_slg)
    h_avg = safe_float(h_s.get('avg')   or h_c.get('avg'),  0.245)
    h_bab = safe_float(h_s.get('babip') or h_c.get('babip'), 0.300)
    try:
        vs_L, vs_R = get_platoon_splits(pid, season)
        pt = vs_L if pit_hand == 'L' else vs_R
    except Exception:
        pt = {}
    pt_ab  = safe_int(pt.get('atBats'), 0)
    pt_ops = safe_float(pt.get('ops'), h_ops) if pt_ab >= 30 else h_ops
    pt_avg = safe_float(pt.get('avg'), h_avg) if pt_ab >= 30 else h_avg
    platoon_adv = (pt_ops - h_ops) if pt_ab >= 30 else 0.0
    try:
        recent = get_game_log(pid, season, limit=14)
    except Exception:
        recent = []
    r_hits, r_ab = 0, 0
    for g in recent:
        gs = g.get('stat', {})
        r_hits += safe_int(gs.get('hits'))
        r_ab   += safe_int(gs.get('atBats'))
    recent_avg = r_hits / r_ab if r_ab > 0 else h_avg
    form_delta = recent_avg - h_avg
    streak = 0
    for g in recent:
        if safe_int(g.get('stat', {}).get('hits')) > 0:
            streak += 1
        else:
            break
    sc = get_statcast_batter_season(pid, season)
    if not sc:
        sc = get_statcast_batter_season(pid, season - 1)
    xba          = safe_float(sc.get('xba'),          h_avg)
    barrel_rate  = safe_float(sc.get('barrel_rate'),  0.075)
    avg_ev       = safe_float(sc.get('avg_ev'),       88.0)
    hard_hit_pct = safe_float(sc.get('hard_hit_pct'), 0.35)
    return {
        'avg': h_avg, 'obp': h_obp, 'slg': h_slg, 'ops': h_ops, 'babip': h_bab,
        'pt_ops': pt_ops, 'pt_avg': pt_avg, 'platoon_adv': platoon_adv,
        'has_platoon': int(pt_ab >= 30),
        'recent_avg': recent_avg, 'form_delta': form_delta, 'streak': streak,
        'xba': xba, 'barrel_rate': barrel_rate, 'avg_ev': avg_ev,
        'hard_hit_pct': hard_hit_pct, 'has_statcast': int(bool(sc)),
    }

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Lineup Aggregation → Team-Game Feature Vector
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_lineup_features(batter_features_list, order_positions=None):
    if not batter_features_list:
        return {
            'lu_avg_mean':0.248,  'lu_obp_mean':0.318,   'lu_slg_mean':0.400,
            'lu_ops_mean':0.718,  'lu_ops_top3':0.780,   'lu_ops_bot3':0.650,
            'lu_ops_std':0.060,   'lu_xba_mean':0.248,   'lu_barrel_mean':0.075,
            'lu_ev_mean':88.0,    'lu_hh_mean':0.35,     'lu_pt_ops_mean':0.718,
            'lu_plat_adv_mean':0.0,'lu_plat_count':0,    'lu_recent_mean':0.248,
            'lu_form_delta_mean':0.0,'lu_streak_sum':0,  'lu_hot_count':0,
            'lu_n_batters':0,     'lu_cleanup_ops':0.750, 'lu_top2_obp':0.330,
            'lu_ops_w_mean':0.718,'lu_slg_w_mean':0.400,
            'lu_xba_w_mean':0.248,'lu_barrel_w_mean':0.075,
            'lu_form_delta_w_mean':0.0,
        }
    feats = batter_features_list
    n     = len(feats)
    if order_positions and len(order_positions) == n:
        raw_w  = [ORDER_WEIGHTS.get(p, 1.0) for p in order_positions]
    else:
        raw_w  = [1.0] * n
    w_sum   = sum(raw_w)
    weights = [w / w_sum * n for w in raw_w]

    def col(key, default=0.0):
        return [safe_float(f.get(key), default) for f in feats]
    def wmean(vals):
        return float(sum(v * w for v, w in zip(vals, weights)) / n)

    ops_vals   = col('ops',  0.700)
    ops_sorted = sorted(ops_vals, reverse=True)
    top3_ops   = float(np.mean(ops_sorted[:3])) if n >= 3 else float(np.mean(ops_sorted))
    bot3_ops   = float(np.mean(ops_sorted[-3:])) if n >= 3 else float(np.mean(ops_sorted))
    avg_vals   = col('avg',  0.248)
    hot_count  = sum(1 for f in feats if safe_float(f.get('form_delta'), 0) > 0.015)
    cleanup_ops = 0.750
    top2_obp    = 0.330
    if order_positions:
        pos_map = {p: f for p, f in zip(order_positions, feats) if p in range(1, 10)}
        if 4 in pos_map:
            cleanup_ops = safe_float(pos_map[4].get('ops'), 0.750)
        top2_obp_vals = [safe_float(pos_map[p].get('obp'), 0.330)
                         for p in (1, 2) if p in pos_map]
        if top2_obp_vals:
            top2_obp = float(np.mean(top2_obp_vals))
    return {
        'lu_avg_mean':          float(np.mean(avg_vals)),
        'lu_obp_mean':          float(np.mean(col('obp', 0.318))),
        'lu_slg_mean':          float(np.mean(col('slg', 0.400))),
        'lu_ops_mean':          float(np.mean(ops_vals)),
        'lu_ops_w_mean':        wmean(ops_vals),
        'lu_slg_w_mean':        wmean(col('slg', 0.400)),
        'lu_xba_w_mean':        wmean(col('xba', 0.248)),
        'lu_barrel_w_mean':     wmean(col('barrel_rate', 0.075)),
        'lu_form_delta_w_mean': wmean(col('form_delta', 0.0)),
        'lu_ops_top3':          top3_ops,
        'lu_ops_bot3':          bot3_ops,
        'lu_ops_std':           float(np.std(ops_vals)) if n > 1 else 0.0,
        'lu_xba_mean':          float(np.mean(col('xba', 0.248))),
        'lu_barrel_mean':       float(np.mean(col('barrel_rate', 0.075))),
        'lu_ev_mean':           float(np.mean(col('avg_ev', 88.0))),
        'lu_hh_mean':           float(np.mean(col('hard_hit_pct', 0.35))),
        'lu_pt_ops_mean':       float(np.mean(col('pt_ops', 0.718))),
        'lu_plat_adv_mean':     float(np.mean(col('platoon_adv', 0.0))),
        'lu_plat_count':        int(sum(col('has_platoon', 0))),
        'lu_recent_mean':       float(np.mean(col('recent_avg', 0.248))),
        'lu_form_delta_mean':   float(np.mean(col('form_delta', 0.0))),
        'lu_streak_sum':        int(sum(col('streak', 0))),
        'lu_hot_count':         hot_count,
        'lu_cleanup_ops':       cleanup_ops,
        'lu_top2_obp':          top2_obp,
        'lu_n_batters':         n,
    }


def build_team_game_row(lineup_feats, order_positions=None,
                        sp_s_stat=None, sp_c_stat=None, sp_recent=None,
                        sp_id=None, sp_hand='R', bullpen_stat=None,
                        venue_id=0, is_home=True, weather=None, runs_scored=None):
    lu = aggregate_lineup_features(lineup_feats, order_positions=order_positions)
    sp_s_stat    = sp_s_stat    or {}
    sp_c_stat    = sp_c_stat    or {}
    sp_recent    = sp_recent    or {}
    bullpen_stat = bullpen_stat or {}
    weather      = weather      or {'temp': 72, 'wind': 8, 'precip': 10}
    p_era   = safe_float(sp_s_stat.get('era')               or sp_c_stat.get('era'),               4.20)
    p_whip  = safe_float(sp_s_stat.get('whip')              or sp_c_stat.get('whip'),              1.30)
    p_k9    = safe_float(sp_s_stat.get('strikeOutsPer9Inn') or sp_c_stat.get('strikeOutsPer9Inn'), 8.5)
    p_bb9   = safe_float(sp_s_stat.get('walksPer9Inn')      or sp_c_stat.get('walksPer9Inn'),      3.0)
    p_hr9   = safe_float(sp_s_stat.get('homeRunsPer9')      or sp_c_stat.get('homeRunsPer9'),      1.1)
    p_ip    = safe_float(sp_s_stat.get('inningsPitched'),   0.0)
    p_c_era = safe_float(sp_c_stat.get('era'),  p_era)
    p_c_whip= safe_float(sp_c_stat.get('whip'), p_whip)
    w       = min(p_ip / 80.0, 1.0)
    p_era_bl   = p_era * w + p_c_era * (1 - w)
    p_whip_bl  = p_whip * w + p_c_whip * (1 - w)
    p_last_era = safe_float(sp_recent.get('last_era'), p_era_bl)
    p_last_ip  = safe_float(sp_recent.get('last_ip'),  0.0)
    p_era_fin  = p_era_bl * 0.6 + p_last_era * 0.4 if p_last_ip > 3 else p_era_bl
    try:
        pid_int = int(sp_id) if sp_id else None
    except (TypeError, ValueError):
        pid_int = None
    ped      = PITCHER_PEDIGREE.get(pid_int, {}) if pid_int else {}
    p_tier   = ped.get('tier', 0)
    p_era_adj= ped.get('era_adj', 1.0)
    sp_is_L  = int(sp_hand == 'L')
    bp_era  = safe_float(bullpen_stat.get('era'),  4.10)
    bp_whip = safe_float(bullpen_stat.get('whip'), 1.30)
    bp_k9   = safe_float(bullpen_stat.get('strikeOutsPer9Inn'), 8.5)
    try:
        vid = int(venue_id) if venue_id else 0
    except (TypeError, ValueError):
        vid = 0
    park_factor = PARK_RUN_FACTORS.get(vid, 100)
    is_dome     = int(vid in DOME_VENUES)
    home_adv    = int(bool(is_home))
    temp   = safe_float(weather.get('temp'),   72)
    wind   = safe_float(weather.get('wind'),   8)
    precip = safe_float(weather.get('precip'), 10)
    row = {
        **lu,
        'p_era_bl':    p_era_bl,  'p_era_fin':  p_era_fin,
        'p_whip_bl':   p_whip_bl, 'p_k9':       p_k9,
        'p_bb9':       p_bb9,     'p_hr9':       p_hr9,
        'p_last_era':  p_last_era,'p_last_ip':   p_last_ip,
        'p_tier':      p_tier,    'p_era_adj':   p_era_adj,
        'sp_is_L':     sp_is_L,
        'bp_era':      bp_era,    'bp_whip':     bp_whip,   'bp_k9': bp_k9,
        'park_factor': park_factor,'is_dome':    is_dome,   'home_adv': home_adv,
        'temp':        temp,      'wind':        wind,      'precip': precip,
    }
    if runs_scored is not None:
        row['runs_scored'] = int(runs_scored)
    return row

WEATHER_FEATS = {'temp', 'wind', 'precip'}
_smoke = build_team_game_row([], None, {}, {}, {}, None, 'R', {}, 0, True,
                             {'temp':72,'wind':8,'precip':10})
ALL_FEATURE_COLS   = [k for k in _smoke if k != 'runs_scored']
TRAIN_FEATURE_COLS = [f for f in ALL_FEATURE_COLS if f not in WEATHER_FEATS]

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Incremental Cache
# ─────────────────────────────────────────────────────────────────────────────
def hist_cache_path(season):
    return HIST_CACHE_DIR / f'season_{season}_v{CACHE_VERSION}.pkl'

def live_cache_path(game_date):
    return LIVE_CACHE_DIR / f'{game_date}_v{CACHE_VERSION}.pkl'

def date_range(start_str, end_str):
    start = date.fromisoformat(start_str)
    end   = date.fromisoformat(end_str)
    d = start
    while d <= end:
        yield d.isoformat()
        d += timedelta(days=1)

def collect_rows_for_date(game_date, season, verbose=False):
    games = get_schedule_for_date(game_date)
    rows  = []
    for game in games:
        try:
            game_pk  = game.get('gamePk')
            home_id  = game['teams']['home']['team']['id']
            away_id  = game['teams']['away']['team']['id']
            home_pit = game['teams']['home'].get('probablePitcher', {})
            away_pit = game['teams']['away'].get('probablePitcher', {})
            venue_id = game.get('venue', {}).get('id', 0)
            home_runs, away_runs = get_game_score(game_pk)
            if home_runs is None:
                continue
            lineups = get_game_lineup(game_pk)
            if not lineups:
                continue
            weather = {'temp': 72, 'wind': 8, 'precip': 10}
            for (bat_team, opp_team, opp_pit, is_home, label_runs, side) in [
                (home_id, away_id, away_pit, True,  home_runs, 'home'),
                (away_id, home_id, home_pit, False, away_runs, 'away'),
            ]:
                pit_id = opp_pit.get('id') if isinstance(opp_pit, dict) else None
                if not pit_id:
                    continue
                pit_hand       = get_pitcher_hand(pit_id)
                lineup_pids    = lineups.get(side, [])
                batter_feats   = []
                batter_positions = []
                for order_pos, pid in enumerate(lineup_pids, start=1):
                    feats = extract_batter_features(pid, season, pit_hand)
                    if feats:
                        batter_feats.append(feats)
                        batter_positions.append(order_pos)
                sp_s, sp_c = get_pitcher_stats(pit_id, season)
                sp_recent  = get_pitcher_last_n_starts(pit_id, season, n=3)
                bp_stat    = get_team_bullpen(opp_team, season)
                row = build_team_game_row(
                    lineup_feats=batter_feats, order_positions=batter_positions,
                    sp_s_stat=sp_s, sp_c_stat=sp_c,
                    sp_recent=sp_recent, sp_id=pit_id, sp_hand=pit_hand,
                    bullpen_stat=bp_stat, venue_id=venue_id, is_home=is_home,
                    weather=weather, runs_scored=label_runs
                )
                row['game_pk']   = game_pk
                row['game_date'] = game_date
                row['season']    = season
                rows.append(row)
        except Exception as e:
            if verbose:
                print(f'    Skipped game {game.get("gamePk","?")}: {e}')
    return rows

def load_or_build_historical_season(season, max_games=100, force=False, verbose=True):
    p = hist_cache_path(season)
    if p.exists() and not force:
        with open(p, 'rb') as f:
            rows = pickle.load(f)
        if rows:
            if verbose: print(f'  Season {season}: loaded {len(rows)} rows from cache')
            return rows
        else:
            if verbose: print(f'  Season {season}: cache was empty, rebuilding...')
            p.unlink()
    if verbose: print(f'  Season {season}: fetching from API...')
    sched_data = mlb_get('schedule', {
        'sportId': 1, 'season': season, 'gameType': 'R',
        'hydrate': 'probablePitcher,team,venue',
    })
    all_games   = [g for d in sched_data.get('dates', []) for g in d.get('games', [])]
    final_games = all_games
    step   = max(1, len(final_games) // max_games)
    sample = final_games[::step][:max_games]
    if verbose: print(f'  Sampling {len(sample)} of {len(final_games)} games')
    rows = []
    for gi, game in enumerate(sample):
        if verbose and gi % 10 == 0:
            ha = game['teams']['home']['team'].get('abbreviation', '')
            aa = game['teams']['away']['team'].get('abbreviation', '')
            print(f'  [{gi+1}/{len(sample)}] {aa}@{ha}', end='\r')
        try:
            game_pk   = game.get('gamePk')
            home_id   = game['teams']['home']['team']['id']
            away_id   = game['teams']['away']['team']['id']
            home_pit  = game['teams']['home'].get('probablePitcher', {})
            away_pit  = game['teams']['away'].get('probablePitcher', {})
            venue_id  = game.get('venue', {}).get('id', 0)
            game_date = game.get('gameDate', '')[:10]
            home_runs, away_runs = get_game_score(game_pk)
            if home_runs is None:
                continue
            lineups = get_game_lineup(game_pk)
            if not lineups:
                continue
            weather = {'temp': 72, 'wind': 8, 'precip': 10}
            for (bat_team, opp_team, opp_pit, is_home, label_runs, side) in [
                (home_id, away_id, away_pit, True,  home_runs, 'home'),
                (away_id, home_id, home_pit, False, away_runs, 'away'),
            ]:
                pit_id = opp_pit.get('id') if isinstance(opp_pit, dict) else None
                if not pit_id:
                    continue
                pit_hand         = get_pitcher_hand(pit_id)
                lineup_pids      = lineups.get(side, [])
                batter_feats     = []
                batter_positions = []
                for order_pos, pid in enumerate(lineup_pids, start=1):
                    feats = extract_batter_features(pid, season, pit_hand)
                    if feats:
                        batter_feats.append(feats)
                        batter_positions.append(order_pos)
                sp_s, sp_c = get_pitcher_stats(pit_id, season)
                sp_recent  = get_pitcher_last_n_starts(pit_id, season)
                bp_stat    = get_team_bullpen(opp_team, season)
                row = build_team_game_row(
                    lineup_feats=batter_feats, order_positions=batter_positions,
                    sp_s_stat=sp_s, sp_c_stat=sp_c,
                    sp_recent=sp_recent, sp_id=pit_id, sp_hand=pit_hand,
                    bullpen_stat=bp_stat, venue_id=venue_id, is_home=is_home,
                    weather=weather, runs_scored=label_runs
                )
                row['game_pk']   = game_pk
                row['game_date'] = game_date
                row['season']    = season
                rows.append(row)
        except Exception:
            pass
    with open(p, 'wb') as f:
        pickle.dump(rows, f)
    if verbose: print(f'\n  Season {season}: saved {len(rows)} rows → {p.name}')
    return rows

def update_live_cache(verbose=True):
    all_live_rows = []
    all_dates = list(date_range(LIVE_SEASON_START, YESTERDAY))
    def _is_valid_cache(stem):
        p = live_cache_path(stem)
        if not p.exists(): return False
        try:
            with open(p, 'rb') as fh:
                return len(pickle.load(fh)) > 0
        except Exception:
            return False
    _verified    = {p.stem for p in LIVE_CACHE_DIR.glob('*.pkl') if _is_valid_cache(p.stem)}
    cached_dates  = _verified
    missing_dates = [d for d in all_dates if d not in cached_dates]
    if verbose:
        print(f'Live cache: {len(cached_dates)} dates cached, '
              f'{len(missing_dates)} missing')
    for i, game_date in enumerate(missing_dates):
        if verbose:
            print(f'  Fetching {game_date}  [{i+1}/{len(missing_dates)}]', end='\r')
        try:
            rows = collect_rows_for_date(game_date, CURRENT_SEASON)
            p    = live_cache_path(game_date)
            with open(p, 'wb') as fh:
                pickle.dump(rows, fh)
            if rows and verbose:
                print(f'  {game_date}: {len(rows)//2} games → {len(rows)} rows')
        except Exception as e:
            if verbose: print(f'\n  Error on {game_date}: {e}')
    for p in sorted(LIVE_CACHE_DIR.glob('*.pkl')):
        with open(p, 'rb') as f:
            all_live_rows.extend(pickle.load(f))
    if verbose:
        print(f'\nLive rows loaded: {len(all_live_rows)}')
    return all_live_rows

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Build Training Dataset
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(verbose=True):
    print('Loading historical seasons...')
    hist_rows = []
    for s in HIST_SEASONS:
        rows = load_or_build_historical_season(s, max_games=100, verbose=verbose)
        hist_rows.extend(rows)
    print(f'Historical total: {len(hist_rows)} team-game rows')
    print('\nUpdating live season cache...')
    live_rows = update_live_cache(verbose=verbose)
    all_rows = hist_rows + live_rows
    df = pd.DataFrame(all_rows)
    print(f'\nCombined dataset: {len(df)} team-game rows')
    return df

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Train Models
# ─────────────────────────────────────────────────────────────────────────────
def train_models(df, verbose=True):
    df_clean = df.dropna(subset=['runs_scored'])
    X = df_clean[TRAIN_FEATURE_COLS].fillna(df_clean[TRAIN_FEATURE_COLS].median())
    y = df_clean['runs_scored'].values
    xgb_model = xgb.XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0
    )
    rf_model = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    )
    xgb_model.fit(X, y)
    rf_model.fit(X, y)
    if verbose:
        xgb_pred = xgb_model.predict(X)
        rf_pred  = rf_model.predict(X)
        ens_pred = (xgb_pred + rf_pred) / 2
        mae = mean_absolute_error(y, ens_pred)
        print(f'Ensemble in-sample MAE: {mae:.3f} runs')
    return xgb_model, rf_model

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Predict Today
# ─────────────────────────────────────────────────────────────────────────────
def predict_today(xgb_model, rf_model, verbose=True):
    print(f'Fetching schedule for {TODAY}...')
    sched = mlb_get('schedule', {
        'sportId': 1, 'date': TODAY,
        'hydrate': 'probablePitcher,team,venue',
    })
    games = sched.get('dates', [{}])[0].get('games', [])
    print(f'{len(games)} games on today\'s slate')
    results = []
    for gi, game in enumerate(games):
        try:
            home_id   = game['teams']['home']['team']['id']
            away_id   = game['teams']['away']['team']['id']
            home_abbr = game['teams']['home']['team'].get('abbreviation', 'HOM')
            away_abbr = game['teams']['away']['team'].get('abbreviation', 'AWY')
            home_pit  = game['teams']['home'].get('probablePitcher', {})
            away_pit  = game['teams']['away'].get('probablePitcher', {})
            venue_id  = game.get('venue', {}).get('id', 0)
            venue_nm  = game.get('venue', {}).get('name', 'Unknown')
            game_pk   = game.get('gamePk', '')
            if verbose:
                print(f'  [{gi+1}/{len(games)}] {away_abbr} @ {home_abbr}  —  {venue_nm}')
            coords  = PARK_COORDS.get(venue_id, (39.5, -98.35))
            weather = fetch_weather(*coords)
            is_dome = venue_id in DOME_VENUES
            pred_runs    = {}
            pitcher_meta = {}
            lineup_meta  = {}
            for (bat_team, opp_team, opp_pit, is_home, abbr, side) in [
                (home_id, away_id, away_pit, True,  home_abbr, 'home'),
                (away_id, home_id, home_pit, False, away_abbr, 'away'),
            ]:
                pit_id   = opp_pit.get('id')             if isinstance(opp_pit, dict) else None
                pit_name = opp_pit.get('fullName', 'TBD') if isinstance(opp_pit, dict) else 'TBD'
                lineup_pids = []
                if game_pk:
                    lups = get_game_lineup(game_pk)
                    if lups and lups.get(side):
                        lineup_pids = lups[side]
                if not lineup_pids:
                    roster = mlb_get(f'teams/{bat_team}/roster', {'rosterType': 'active'})
                    lineup_pids = [
                        p['person']['id'] for p in roster.get('roster', [])
                        if p.get('position', {}).get('abbreviation') not in ('P', 'SP', 'RP')
                    ][:9]
                pit_hand = get_pitcher_hand(pit_id) if pit_id else 'R'
                batter_feats     = []
                batter_positions = []
                batter_names     = []
                for order_pos, pid in enumerate(lineup_pids, start=1):
                    feats = extract_batter_features(pid, CURRENT_SEASON, pit_hand)
                    if feats:
                        batter_feats.append(feats)
                        batter_positions.append(order_pos)
                        pd_data = mlb_get(f'people/{pid}')
                        nm = pd_data.get('people', [{}])[0].get('fullName', str(pid))
                        batter_names.append(nm)
                if pit_id:
                    sp_s, sp_c = get_pitcher_stats(pit_id, CURRENT_SEASON)
                    sp_recent  = get_pitcher_last_n_starts(pit_id, CURRENT_SEASON)
                else:
                    sp_s, sp_c, sp_recent = {}, {}, {}
                bp_stat = get_team_bullpen(opp_team, CURRENT_SEASON)
                ped     = PITCHER_PEDIGREE.get(pit_id, {}) if pit_id else {}
                feat_row = build_team_game_row(
                    lineup_feats=batter_feats, order_positions=batter_positions,
                    sp_s_stat=sp_s, sp_c_stat=sp_c,
                    sp_recent=sp_recent, sp_id=pit_id, sp_hand=pit_hand,
                    bullpen_stat=bp_stat, venue_id=venue_id, is_home=is_home,
                    weather=weather,
                )
                X_row    = np.array([feat_row.get(f, 0) for f in TRAIN_FEATURE_COLS]).reshape(1, -1)
                xgb_pred = float(xgb_model.predict(X_row)[0])
                rf_pred  = float(rf_model.predict(X_row)[0])
                base_pred= (xgb_pred + rf_pred) / 2
                if not is_dome:
                    if weather['temp']   > 85: base_pred *= 1.04
                    elif weather['temp'] < 48: base_pred *= 0.95
                    if weather['precip'] > 60: base_pred *= 0.94
                    if weather['wind']   > 20: base_pred *= 1.02
                runs_pred = max(0.5, min(base_pred, 18.0))
                pred_runs[abbr] = runs_pred
                lu = aggregate_lineup_features(batter_feats, order_positions=batter_positions)
                pitcher_meta[abbr] = {
                    'name':     pit_name,
                    'hand':     pit_hand,
                    'era':      round(safe_float(sp_s.get('era') or sp_c.get('era'), 4.50), 2),
                    'last_era': round(safe_float(sp_recent.get('last_era'), 0), 2)
                              if sp_recent.get('last_era') else 'N/A',
                    'pedigree': ped.get('label', ''),
                }
                lineup_meta[abbr] = {
                    'ops_mean': round(lu['lu_ops_mean'], 3),
                    'xba_mean': round(lu['lu_xba_mean'], 3),
                    'barrel':   f"{lu['lu_barrel_mean']*100:.1f}%",
                    'ev_mean':  round(lu['lu_ev_mean'], 1),
                    'hot':      lu['lu_hot_count'],
                    'streak':   lu['lu_streak_sum'],
                    'plat_adv': round(lu['lu_plat_adv_mean'], 3),
                    'batters':  lu['lu_n_batters'],
                    'names':    ', '.join(batter_names[:5]) + ('…' if len(batter_names) > 5 else ''),
                }
            home_r = pred_runs.get(home_abbr, 4.5)
            away_r = pred_runs.get(away_abbr, 4.5)
            total  = home_r + away_r
            diff   = home_r - away_r
            home_win_prob = 1 / (1 + np.exp(-diff * 0.45))
            hp = pitcher_meta.get(home_abbr, {})
            ap = pitcher_meta.get(away_abbr, {})
            hl = lineup_meta.get(home_abbr, {})
            al = lineup_meta.get(away_abbr, {})
            results.append({
                'matchup':          f'{away_abbr} @ {home_abbr}',
                'home_team':        home_abbr,
                'away_team':        away_abbr,
                'venue':            venue_nm,
                'home_runs_pred':   round(home_r, 1),
                'away_runs_pred':   round(away_r, 1),
                'total_pred':       round(total, 1),
                'run_diff':         round(abs(diff), 1),
                'home_win_prob':    round(home_win_prob * 100, 1),
                'away_win_prob':    round((1 - home_win_prob) * 100, 1),
                'predicted_winner': home_abbr if home_win_prob >= 0.5 else away_abbr,
                'home_faces':       ap.get('name', 'TBD'),
                'home_sp_hand':     ap.get('hand', '?'),
                'home_sp_era':      ap.get('era', 'N/A'),
                'home_sp_last3':    ap.get('last_era', 'N/A'),
                'home_sp_ped':      ap.get('pedigree', ''),
                'away_faces':       hp.get('name', 'TBD'),
                'away_sp_hand':     hp.get('hand', '?'),
                'away_sp_era':      hp.get('era', 'N/A'),
                'away_sp_last3':    hp.get('last_era', 'N/A'),
                'away_sp_ped':      hp.get('pedigree', ''),
                'home_lu_ops':      hl.get('ops_mean', ''),
                'home_lu_xba':      hl.get('xba_mean', ''),
                'home_lu_barrel':   hl.get('barrel', ''),
                'home_lu_hot':      hl.get('hot', 0),
                'home_lu_plat':     hl.get('plat_adv', 0),
                'away_lu_ops':      al.get('ops_mean', ''),
                'away_lu_xba':      al.get('xba_mean', ''),
                'away_lu_barrel':   al.get('barrel', ''),
                'away_lu_hot':      al.get('hot', 0),
                'away_lu_plat':     al.get('plat_adv', 0),
                'park_factor':      PARK_RUN_FACTORS.get(venue_id, 100),
                'is_dome':          int(is_dome),
                'weather':          f"{weather['temp']}°F / {weather['wind']}mph / {weather['precip']}% rain",
                'home_lineup':      hl.get('names', ''),
                'away_lineup':      al.get('names', ''),
            })
        except Exception as e:
            if verbose: print(f'  Error on game {gi}: {e}')
    df_out = pd.DataFrame(results).sort_values('total_pred', ascending=False)
    df_out.index = range(1, len(df_out) + 1)
    df_out.index.name = 'Rank'
    return df_out

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Save Chart
# ─────────────────────────────────────────────────────────────────────────────
def save_chart(df_preds, outpath):
    n = len(df_preds)
    fig, axes = plt.subplots(1, 2, figsize=(18, max(5, n * 0.55 + 1.5)))
    fig.patch.set_facecolor('#0d1117')

    # Left: stacked bar — away + home runs
    ax = axes[0]
    ax.set_facecolor('#0d1117')
    y_pos  = np.arange(n)
    labels = [r['matchup'] for _, r in df_preds.iterrows()]
    away_r = df_preds['away_runs_pred'].values
    home_r = df_preds['home_runs_pred'].values
    totals = df_preds['total_pred'].values

    ax.barh(y_pos, away_r, color='#3b82f6', height=0.6, label='Away')
    ax.barh(y_pos, home_r, left=away_r, color='#f59e0b', height=0.6, label='Home')
    for i, (a, h, t) in enumerate(zip(away_r, home_r, totals)):
        ax.text(a/2,   i, f'{a:.1f}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        ax.text(a+h/2, i, f'{h:.1f}', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        ax.text(a+h+0.1, i, f'{t:.1f}', va='center', fontsize=7.5, color='#94a3b8')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, color='#94a3b8')
    ax.invert_yaxis()
    ax.set_xlabel('Predicted Runs', color='#94a3b8')
    ax.set_title(f'Game Predictions — {TODAY}', color='#f0f2f7', fontsize=12, pad=8)
    ax.legend(loc='lower right', fontsize=9, facecolor='#1e293b', labelcolor='white')
    ax.axvline(7.0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom', 'left']: ax.spines[sp].set_color('#374151')
    ax.tick_params(colors='#94a3b8')

    # Right: win probability
    ax2 = axes[1]
    ax2.set_facecolor('#0d1117')
    home_probs = df_preds['home_win_prob'].values / 100
    for i, (hp, row) in enumerate(zip(home_probs, df_preds.itertuples())):
        ap  = 1 - hp
        c_h = '#22c55e' if hp >= 0.55 else ('#ef4444' if hp < 0.45 else '#64748b')
        c_a = '#22c55e' if ap >= 0.55 else ('#ef4444' if ap < 0.45 else '#64748b')
        ax2.barh(i,  hp, color=c_h, height=0.6)
        ax2.barh(i, -ap, color=c_a, height=0.6)
        ax2.text( hp+0.01, i, f'{hp*100:.0f}%', va='center', fontsize=8, color='#94a3b8')
        ax2.text(-ap-0.01, i, f'{ap*100:.0f}%', va='center', ha='right', fontsize=8, color='#94a3b8')

    ax2.set_yticks(range(n))
    ax2.set_yticklabels(
        [f"← {r['away_team']}    {r['home_team']} →" for _, r in df_preds.iterrows()],
        fontsize=8, color='#94a3b8'
    )
    ax2.invert_yaxis()
    ax2.axvline(0, color='white', linewidth=0.8)
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_title('Win Probability', color='#f0f2f7', fontsize=12, pad=8)
    ax2.set_xlabel('← Away         Home →', color='#94a3b8')
    for sp in ['top', 'right', 'bottom', 'left']: ax2.spines[sp].set_color('#374151')
    ax2.tick_params(colors='#94a3b8')

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()
    print(f'Chart saved: {outpath}')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — Build HTML Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def build_html(df_preds, chart_filename, csv_filename):
    rows_html = ''
    for _, row in df_preds.iterrows():
        winner_class = 'winner-home' if row['predicted_winner'] == row['home_team'] else 'winner-away'
        ped_home = f' <span class="elite">⚡ {row["home_sp_ped"]}</span>' if row.get('home_sp_ped') else ''
        ped_away = f' <span class="elite">⚡ {row["away_sp_ped"]}</span>' if row.get('away_sp_ped') else ''
        dome_tag = ' 🏟️' if row.get('is_dome') else ''
        rows_html += f"""
        <tr>
          <td class="matchup">{row['matchup']}</td>
          <td class="score">{row['away_runs_pred']:.1f} – {row['home_runs_pred']:.1f}</td>
          <td class="{winner_class}">{row['predicted_winner']}</td>
          <td>{row['home_win_prob']:.0f}% / {row['away_win_prob']:.0f}%</td>
          <td class="total">{row['total_pred']:.1f}</td>
          <td>{row['home_lu_ops']} / {row['away_lu_ops']}</td>
          <td>{row['home_lu_xba']} / {row['away_lu_xba']}</td>
          <td>{row.get('home_lu_barrel','')} / {row.get('away_lu_barrel','')}</td>
          <td>{row['home_faces']} ({row['home_sp_era']}){ped_home}</td>
          <td>{row['away_faces']} ({row['away_sp_era']}){ped_away}</td>
          <td class="weather">{row.get('weather','')}{dome_tag}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MLB Game Predictions — {TODAY}</title>
  <style>
    :root {{
      --bg: #0d1117; --surface: #161b22; --border: #30363d;
      --text: #e6edf3; --muted: #8b949e; --accent: #3b82f6;
      --green: #22c55e; --red: #ef4444; --gold: #f59e0b;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
    header {{ background: var(--surface); border-bottom: 1px solid var(--border); padding: 1.5rem 2rem; display: flex; align-items: center; gap: 1rem; }}
    header h1 {{ font-size: 1.4rem; color: var(--text); }}
    header .date {{ color: var(--muted); font-size: 0.9rem; margin-top: 0.2rem; }}
    header .badge {{ background: var(--accent); color: white; font-size: 0.7rem; padding: 0.2rem 0.6rem; border-radius: 12px; font-weight: 600; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 2rem; }}
    .chart-wrap {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; margin-bottom: 2rem; text-align: center; }}
    .chart-wrap img {{ max-width: 100%; border-radius: 6px; }}
    .section-title {{ color: var(--muted); font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.75rem; }}
    .table-wrap {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; margin-bottom: 2rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    thead th {{ background: #21262d; color: var(--muted); font-weight: 600; padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border); white-space: nowrap; }}
    tbody tr {{ border-bottom: 1px solid var(--border); transition: background 0.15s; }}
    tbody tr:last-child {{ border-bottom: none; }}
    tbody tr:hover {{ background: #1c2128; }}
    td {{ padding: 0.65rem 1rem; color: var(--text); vertical-align: middle; }}
    .matchup {{ font-weight: 600; white-space: nowrap; }}
    .score {{ font-family: 'Courier New', monospace; font-weight: 700; color: var(--gold); }}
    .total {{ font-weight: 700; color: var(--accent); }}
    .winner-home {{ color: var(--green); font-weight: 700; }}
    .winner-away {{ color: var(--red); font-weight: 700; }}
    .elite {{ color: var(--gold); font-size: 0.78rem; }}
    .weather {{ color: var(--muted); font-size: 0.78rem; }}
    .meta {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }}
    .meta-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem 1.25rem; }}
    .meta-card .label {{ color: var(--muted); font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }}
    .meta-card .value {{ color: var(--text); font-size: 1.2rem; font-weight: 700; margin-top: 0.1rem; }}
    .download {{ display: inline-flex; align-items: center; gap: 0.4rem; background: var(--surface); border: 1px solid var(--border); color: var(--muted); padding: 0.4rem 0.9rem; border-radius: 6px; font-size: 0.82rem; text-decoration: none; transition: border-color 0.15s, color 0.15s; }}
    .download:hover {{ border-color: var(--accent); color: var(--accent); }}
    footer {{ text-align: center; color: var(--muted); font-size: 0.8rem; padding: 2rem; border-top: 1px solid var(--border); }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>⚾ MLB Game Predictions</h1>
      <div class="date">Based on actual starting lineups · {TODAY} · Updated 9:00 AM CT</div>
    </div>
    <span class="badge">XGBoost + RF Ensemble</span>
  </header>

  <div class="container">
    <div class="meta">
      <div class="meta-card">
        <div class="label">Games Today</div>
        <div class="value">{len(df_preds)}</div>
      </div>
      <div class="meta-card">
        <div class="label">Avg Total Runs</div>
        <div class="value">{df_preds['total_pred'].mean():.1f}</div>
      </div>
      <div class="meta-card">
        <div class="label">Highest Scoring</div>
        <div class="value">{df_preds.iloc[0]['matchup']}</div>
      </div>
      <div class="meta-card">
        <div class="label">Best Pitching Duel</div>
        <div class="value">{df_preds.iloc[-1]['matchup']}</div>
      </div>
      <a class="download" href="{csv_filename}" download>⬇ Download CSV</a>
    </div>

    <div class="chart-wrap">
      <img src="{chart_filename}" alt="Game Predictions Chart for {TODAY}">
    </div>

    <div class="section-title">All Games — Sorted by Predicted Total Runs</div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Matchup</th>
            <th>Predicted Score</th>
            <th>Predicted Winner</th>
            <th>Win Prob (H/A)</th>
            <th>Total</th>
            <th>OPS (H/A)</th>
            <th>xBA (H/A)</th>
            <th>Barrel% (H/A)</th>
            <th>Home Faces (ERA)</th>
            <th>Away Faces (ERA)</th>
            <th>Weather</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
  </div>

  <footer>
    Game Predictions · XGBoost + Random Forest trained on 2022–{CURRENT_SEASON} MLB data ·
    Player-level Statcast features (xBA, barrel rate, EV) · Park factors · Live weather ·
    Model MAE ~2.2 runs · Win accuracy ~57–60%
  </footer>
</body>
</html>"""
    return html

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f'=== Game Predictions — {TODAY} ===\n')

    # 1. Build/load dataset
    df = build_dataset(verbose=True)
    if df.empty:
        raise RuntimeError('No training data collected. Check API connectivity.')

    # 2. Train
    print('\nTraining models...')
    xgb_model, rf_model = train_models(df, verbose=True)

    # 3. Predict today
    print('\nGenerating predictions...')
    df_preds = predict_today(xgb_model, rf_model, verbose=True)
    print(f'\nDone — {len(df_preds)} games predicted')

    if df_preds.empty:
        print('No games today. Exiting.')
        return

    # 4. Save chart
    chart_filename = f'game-predictions-{TODAY}.png'
    chart_path     = DOCS_DIR / chart_filename
    save_chart(df_preds, chart_path)

    # 5. Save CSV
    csv_filename = f'game-predictions-{TODAY}.csv'
    csv_path     = DOCS_DIR / csv_filename
    df_preds.to_csv(csv_path)
    print(f'CSV saved: {csv_path}')

    # 6. Build and save index.html
    html = build_html(df_preds, chart_filename, csv_filename)
    index_path = DOCS_DIR / 'index.html'
    index_path.write_text(html, encoding='utf-8')
    print(f'Dashboard saved: {index_path}')

    print('\n✓ All outputs written to docs/')

if __name__ == '__main__':
    main()
