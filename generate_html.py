import os
import re
import sys
import json
import logging
import pandas as pd
import numpy as np
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kelly3 v12 強化学習モジュールのインポート
if r"C:\Users\kyoui\keiba" not in sys.path:
    sys.path.append(r"C:\Users\kyoui\keiba")

try:
    # Kelly3 の買い目は強化学習(DQN)推論ではなく Kelly3.ipynb の
    # 「中央値重視・堅牢ポートフォリオ」(generate_smart_balanced_bets) を使用する。
    from modules.rl_betting import (
        generate_smart_balanced_bets, VENUE_MAP, VENUE_NAME_MAP
    )
    rl_agent_available = True
    logger.info("Successfully loaded modules.rl_betting (Kelly3 portfolio)")
except Exception as e:
    logger.warning(f"Could not load modules.rl_betting: {e}")
    rl_agent_available = False

PLACE_DICT_CHUOH = {
    '札幌': '01',
    '函館': '02',
    '福島': '03',
    '新潟': '04',
    '東京': '05',
    '中山': '06',
    '中京': '07',
    '京都': '08',
    '阪神': '09',
    '小倉': '10'
}
REVERSE_PLACE_DICT = {v: k for k, v in PLACE_DICT_CHUOH.items()}

# ==========================================================================
# Kelly2 / Kelly3 本家ノートブック準拠の買い目生成
#  - PICKUP     : Kelly2.ipynb (winning_strategies_v13.csv + TCSV\v13 モデルスコア + 単勝オッズ)
#  - PICKUP 2   : Kelly3.ipynb (中央値重視・堅牢ポートフォリオ generate_smart_balanced_bets)
# ==========================================================================
KELLY_CONFIG_DIR = r"C:\Users\kyoui\keiba\config"
KELLY_STRATEGIES_V13_CSV = os.path.join(KELLY_CONFIG_DIR, "winning_strategies_v13.csv")
KELLY_TCSV_DIR = r"C:\keibasoftcom\KSCAutoBetPlus\TCSV"
KELLY_TCSV_V13_DIR = r"C:\keibasoftcom\KSCAutoBetPlus\TCSV\v13"
KELLY_TANSHO_DB_URL = "postgresql://postgres:zatenn@localhost/postgres"
KELLY_TRACK_MAPPING = {
    '01': 'SAPPORO', '02': 'HAKODATE', '03': 'FUKUSHIMA', '04': 'NIIGATA', '05': 'TOKYO',
    '06': 'NAKAYAMA', '07': 'CHUKYO', '08': 'KYOTO', '09': 'HANSHIN', '10': 'KOKURA'
}
# 買い目表示用の券種コード→日本語名 (ipatgo CSV の券種コード)
KELLY_BET_CODE_JP = {
    'TANSYO': '単勝', 'FUKUSYO': '複勝', 'WAKUREN': '枠連', 'UMAREN': '馬連',
    'UMATAN': '馬単', 'WIDE': 'ワイド', 'SANRENPUKU': '3連複', 'SANRENTAN': '3連単',
}
KELLY_ORDERED_BET_CODES = {'UMATAN', 'SANRENTAN'}
# 合意度表示に使う主要4モデル (Kelly2.ipynb / 画面の「4モデル平均」と同じ)
KELLY_CONF_MODELS = ['LightGBM', 'CatBoost', 'RandomForest', 'TabNet']


def _softmax_np(x):
    """Kelly2.ipynb と同一のソフトマックス (EV算出用)"""
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum(axis=0) + 1e-12)


def _load_tansho_odds_df(day):
    """odds1_tansho から当日の単勝オッズ(実値)・人気を取得する。
    Kelly2.ipynb / Kelly3.ipynb と同じく DB を一次ソースとする。
    取得できない場合は空 DataFrame を返し、ノートブックと同じく EV 計算をスキップする。"""
    empty = pd.DataFrame(columns=['race_code', 'umaban', 'odds_val', 'ninki'])
    try:
        from sqlalchemy import create_engine
        engine = create_engine(KELLY_TANSHO_DB_URL)
        nen = str(day)[:4]
        gappi = str(day)[4:8]
        sql = ("SELECT race_code, umaban, odds, ninki FROM odds1_tansho "
               f"WHERE kaisai_nen = '{nen}' AND kaisai_gappi = '{gappi}';")
        df_o = pd.read_sql(sql, engine)
        if df_o.empty:
            logger.warning(f"No tansho odds found in DB for {day}")
            return empty
        df_o['race_code'] = df_o['race_code'].astype(str)
        df_o['umaban'] = pd.to_numeric(df_o['umaban'], errors='coerce')
        df_o['odds_val'] = pd.to_numeric(df_o['odds'], errors='coerce') / 10.0
        df_o['ninki'] = pd.to_numeric(df_o['ninki'], errors='coerce')
        df_o = df_o.dropna(subset=['umaban'])
        logger.info(f"Loaded {len(df_o)} tansho odds records for {day} from odds1_tansho")
        return df_o[['race_code', 'umaban', 'odds_val', 'ninki']]
    except Exception as e:
        logger.warning(f"tansho odds (odds1_tansho) load failed for {day}: {e}")
        return empty


def _kelly_model_csv_path(model, day, base_dir):
    """モデルスコアCSVのパスを解決する (Ensemble は _raw 無し)"""
    for name in (f"{model}_raw_{day}.csv", f"{model}_{day}.csv"):
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p
    return None


def _kelly2_bet_specs(bt, nos):
    """Kelly2.ipynb の bet_type から ipatgo 形式の買い目仕様リストを返す。
    nos: モデルスコア降順の馬番リスト
    各要素: code / sel_mode / multi / nums_str / combs / axis1 / axis2 / partners / eyes"""
    pad = lambda n: f"{int(n):02d}"
    specs = []

    if bt in ('単勝', '複勝'):
        code = 'TANSYO' if bt == '単勝' else 'FUKUSYO'
        specs.append({'code': code, 'sel_mode': 'NORMAL', 'multi': '', 'nums_str': pad(nos[0]),
                      'combs': 1, 'axis1': nos[0], 'axis2': None, 'partners': [],
                      'eyes': pad(nos[0])})
    elif bt == '馬連-1頭軸3頭ながし' and len(nos) >= 4:
        partners = sorted(nos[1:4])
        specs.append({'code': 'UMAREN', 'sel_mode': 'WHEEL', 'multi': '',
                      'nums_str': f"{pad(nos[0])}-{''.join(pad(x) for x in partners)}",
                      'combs': 3, 'axis1': nos[0], 'axis2': None, 'partners': partners,
                      'eyes': f"{pad(nos[0])} - {', '.join(pad(x) for x in partners)}"})
    elif bt == '馬単-1頭軸3頭ながし' and len(nos) >= 4:
        partners = sorted(nos[1:4])
        specs.append({'code': 'UMATAN', 'sel_mode': 'WHEEL1', 'multi': '',
                      'nums_str': f"{pad(nos[0])}-{''.join(pad(x) for x in partners)}",
                      'combs': 3, 'axis1': nos[0], 'axis2': None, 'partners': partners,
                      'eyes': f"{pad(nos[0])} → {', '.join(pad(x) for x in partners)}"})
    elif bt == '3連複-3頭BOX' and len(nos) >= 3:
        box = sorted(nos[:3])
        specs.append({'code': 'SANRENPUKU', 'sel_mode': 'BOX', 'multi': '',
                      'nums_str': ''.join(pad(x) for x in box), 'combs': 1,
                      'axis1': None, 'axis2': None, 'partners': box,
                      'eyes': f"{', '.join(pad(x) for x in box)} BOX"})
    elif bt == '3連複-4頭BOX' and len(nos) >= 4:
        box = sorted(nos[:4])
        specs.append({'code': 'SANRENPUKU', 'sel_mode': 'BOX', 'multi': '',
                      'nums_str': ''.join(pad(x) for x in box), 'combs': 4,
                      'axis1': None, 'axis2': None, 'partners': box,
                      'eyes': f"{', '.join(pad(x) for x in box)} BOX"})
    elif bt == '3連複-1頭軸3頭ながし' and len(nos) >= 4:
        partners = sorted(nos[1:4])
        specs.append({'code': 'SANRENPUKU', 'sel_mode': 'WHEEL1B', 'multi': '',
                      'nums_str': f"{pad(nos[0])}-{''.join(pad(x) for x in partners)}",
                      'combs': 3, 'axis1': nos[0], 'axis2': None, 'partners': partners,
                      'eyes': f"{pad(nos[0])} - {', '.join(pad(x) for x in partners)}"})
    elif bt == '3連単-2通り' and len(nos) >= 3:
        # サイト表示は「01 ↔ 12 → 04」の1エントリに統一
        # 連携時は parseSmappyEyes で is2Touri=true として2通りに展開される
        specs.append({'code': 'SANRENTAN', 'sel_mode': 'NORMAL', 'multi': '',
                      'nums_str': f"{pad(nos[0])}-{pad(nos[1])}-{pad(nos[2])}", 'combs': 2,
                      'axis1': nos[0], 'axis2': nos[1], 'partners': [nos[2]],
                      'eyes': f"{pad(nos[0])} ↔ {pad(nos[1])} → {pad(nos[2])}"})
    elif bt == '3連単-3頭BOX' and len(nos) >= 3:
        box = sorted(nos[:3])
        specs.append({'code': 'SANRENTAN', 'sel_mode': 'BOX', 'multi': '',
                      'nums_str': ''.join(pad(x) for x in box), 'combs': 6,
                      'axis1': None, 'axis2': None, 'partners': box,
                      'eyes': f"{', '.join(pad(x) for x in box)} BOX"})
    elif bt == '3連単-1頭軸3頭流し' and len(nos) >= 4:
        partners = sorted(nos[1:4])
        specs.append({'code': 'SANRENTAN', 'sel_mode': 'WHEEL1', 'multi': '',
                      'nums_str': f"{pad(nos[0])}-{''.join(pad(x) for x in partners)}",
                      'combs': 6, 'axis1': nos[0], 'axis2': None, 'partners': partners,
                      'eyes': f"{pad(nos[0])} → {', '.join(pad(x) for x in partners)}"})
    elif bt == '3連単-1頭軸3頭マルチ' and len(nos) >= 4:
        partners = sorted(nos[1:4])
        specs.append({'code': 'SANRENTAN', 'sel_mode': 'WHEEL1', 'multi': 'MULTI',
                      'nums_str': f"{pad(nos[0])}-{''.join(pad(x) for x in partners)}",
                      'combs': 18, 'axis1': nos[0], 'axis2': None, 'partners': partners,
                      'eyes': f"{pad(nos[0])} ↔ {', '.join(pad(x) for x in partners)}"})
    elif bt == '3連単-2頭軸3頭マルチ' and len(nos) >= 5:
        partners = sorted(nos[2:5])
        specs.append({'code': 'SANRENTAN', 'sel_mode': 'WHEEL12', 'multi': 'MULTI',
                      'nums_str': f"{pad(nos[0])}-{pad(nos[1])}-{''.join(pad(x) for x in partners)}",
                      'combs': 18, 'axis1': nos[0], 'axis2': nos[1], 'partners': partners,
                      'eyes': f"{pad(nos[0])}, {pad(nos[1])} ↔ {', '.join(pad(x) for x in partners)}"})
    return specs


def build_kelly2_bets_for_day(day, races_of_day):
    """Kelly2.ipynb (Cell3) と同一ロジックで当日の買い目を生成する。
    - winning_strategies_v13.csv の BUY / EXCLUDE ルールを race meta で判定
    - TCSV\\v13 のモデルスコアCSVで z_score、DB(odds1_tansho)で EV を算出
    - score_th / EV_th を軸1頭目に適用し、bet_type ごとの買い目を生成
    モデルスコアCSVが無い日は None を返す (従来の近似表示にフォールバック)。"""
    if not os.path.exists(KELLY_STRATEGIES_V13_CSV):
        logger.warning(f"Kelly2: strategy master not found: {KELLY_STRATEGIES_V13_CSV}")
        return None
    try:
        raw_strat = pd.read_csv(KELLY_STRATEGIES_V13_CSV)
    except Exception:
        try:
            raw_strat = pd.read_csv(KELLY_STRATEGIES_V13_CSV, encoding='utf-8-sig')
        except Exception as e:
            logger.error(f"Kelly2: strategy CSV read failed: {e}")
            return None
    if 'action' not in raw_strat.columns:
        return None

    buy_df = raw_strat[raw_strat['action'] == 'BUY'].copy()
    ex_df = raw_strat[raw_strat['action'] == 'EXCLUDE'].copy()
    if buy_df.empty:
        return None

    day_str = str(day)

    # v13 モデルスコアCSVが無い日は Kelly2 の買い目を再現できないため早期リターン
    # (画面側は従来の近似ロジックで表示を継続する)
    has_v13 = any(_kelly_model_csv_path(m, day_str, KELLY_TCSV_V13_DIR)
                  for m in buy_df['model'].dropna().unique())
    if not has_v13:
        logger.debug(f"Kelly2: v13 score CSV not found for {day}")
        return None

    odds_df = _load_tansho_odds_df(day_str)

    # --- モデルスコアの読み込み (TCSV\v13) + z_score / EV 計算 ---
    model_dfs = {}
    for m in buy_df['model'].dropna().unique():
        m_path = _kelly_model_csv_path(m, day_str, KELLY_TCSV_V13_DIR)
        if m_path is None:
            continue
        try:
            sdf = pd.read_csv(m_path)
        except Exception as e:
            logger.warning(f"Kelly2: score CSV read failed ({m_path}): {e}")
            continue
        if 'race_horse_id' not in sdf.columns or 'score' not in sdf.columns:
            continue
        sdf['race_horse_id'] = sdf['race_horse_id'].astype(str)
        sdf['racecode'] = sdf['race_horse_id'].str[:16]
        sdf['umaban'] = sdf['race_horse_id'].str[16:18].astype(int)
        sdf['z_score'] = sdf.groupby('racecode')['score'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
        sdf['ev'] = np.nan
        if not odds_df.empty:
            sdf = sdf.merge(odds_df[['race_code', 'umaban', 'odds_val']],
                            left_on=['racecode', 'umaban'],
                            right_on=['race_code', 'umaban'], how='left')
            for rc, group in sdf.groupby('racecode'):
                probs = _softmax_np(group['z_score'].values * 2.0)
                evs = probs * np.log1p(group['odds_val'].fillna(10.0).values)
                sdf.loc[group.index, 'ev'] = evs
        model_dfs[m] = sdf

    if not model_dfs:
        logger.warning(f"Kelly2: no v13 score CSV for {day} -> PICKUP は近似表示にフォールバック")
        return None

    # 合意度表示用 (主要4モデルの順位)
    conf_rank = {}
    for m, sdf in model_dfs.items():
        if m not in KELLY_CONF_MODELS:
            continue
        for rc, group in sdf.groupby('racecode'):
            order = group.sort_values('score', ascending=False)['umaban'].tolist()
            entry = conf_rank.setdefault(str(rc), {})
            for i, un in enumerate(order):
                entry.setdefault(int(un), []).append(i + 1)

    # レースメタ (サイトの JSON より。Kelly2.ipynb の race_meta と同じキー構成)
    meta_by_race = {}
    for rid, r_info in races_of_day.items():
        meta = dict(r_info.get('meta') or {})
        rid_str = str(rid)
        meta.setdefault('venue_code', rid_str[4:6] if len(rid_str) >= 6 else '')
        meta_by_race[rid_str] = meta

    candidates = []
    for _, strat in buy_df.iterrows():
        m_name = strat['model']
        if m_name not in model_dfs:
            continue
        df = model_dfs[m_name]
        bt = str(strat.get('bet_type') or '')
        unit = int(strat['unit_price']) if pd.notnull(strat.get('unit_price')) else 100
        cat_col = str(strat.get('cat_col') or '')
        cat_val = strat.get('val')
        score_th = strat.get('score_th')
        ev_th = strat.get('EV_th')

        for r_code, group in df.groupby('racecode'):
            r_code = str(r_code)
            s_r12 = (r_code[:4] + r_code[8:16]) if len(r_code) == 16 else r_code
            meta = meta_by_race.get(s_r12)
            if not meta:
                continue
            # 条件マッチ (Kelly2.ipynb Cell3 と同一)
            if cat_col == 'all':
                matched = True
            else:
                matched = (cat_col in meta and str(meta.get(cat_col)) == str(cat_val))
            if not matched:
                continue
            # EXCLUDE ルール
            is_excluded = False
            for _, ex in ex_df.iterrows():
                e_col = str(ex.get('cat_col') or '')
                if e_col in meta and str(meta.get(e_col)) == str(ex.get('val')):
                    if str(ex.get('model')) in ['all', '全モデル', m_name]:
                        is_excluded = True
                        break
            if is_excluded:
                continue

            sorted_g = group.sort_values('score', ascending=False)
            nos = [int(x) for x in sorted_g['umaban'].tolist()]
            if len(nos) < 3:
                continue
            axis1_row = sorted_g.iloc[0]
            if pd.notnull(score_th) and float(score_th) > -90:
                if float(axis1_row['z_score']) < float(score_th):
                    continue
            if pd.notnull(ev_th) and float(ev_th) > -90:
                if pd.notnull(axis1_row['ev']) and float(axis1_row['ev']) < float(ev_th):
                    continue

            h1_num = int(nos[0])
            h1_conf = conf_rank.get(r_code, {}).get(h1_num, [])
            pop_rank = None
            if not odds_df.empty:
                hit = odds_df[(odds_df['race_code'] == r_code) & (odds_df['umaban'] == h1_num)]
                if len(hit) > 0 and pd.notnull(hit.iloc[0]['ninki']):
                    pop_rank = int(hit.iloc[0]['ninki'])

            date_str = r_code[:8]
            track_name = KELLY_TRACK_MAPPING.get(r_code[8:10], 'UNKNOWN')
            race_no = int(r_code[-2:])
            for spec in _kelly2_bet_specs(bt, nos):
                line = (f"{date_str},{track_name},{race_no},{spec['code']},{spec['sel_mode']},"
                        f"{spec['multi']},{spec['nums_str']},{unit}")
                candidates.append({
                    'line': line,
                    'race_id': s_r12,
                    'strategy_id': str(strat.get('strategy_id') or ''),
                    'model': m_name,
                    'bet_type': bt,
                    'rawType': bt,
                    'unit': unit,
                    'combs': spec['combs'],
                    'cost': unit * spec['combs'],
                    'axis1Num': spec['axis1'],
                    'axis2Num': spec['axis2'],
                    'partnerNums': spec['partners'],
                    'bettingEyesText': spec['eyes'],
                    'roi': float(strat.get('roi_total') or strat.get('roi') or 0),
                    'hitRate': float(strat.get('hit_rate') or 0),
                    'h1Info': {
                        'avgRank': float(sum(h1_conf) / len(h1_conf)) if h1_conf else 99.0,
                        'top3Count': sum(1 for x in h1_conf if x <= 3),
                    },
                    'h1PopRank': pop_rank,
                })

    if not candidates:
        return None

    # 買い目の重複排除 (Kelly2.ipynb: drop_duplicates('line', keep='first'))
    by_race = {}
    seen = set()
    for c in candidates:
        if c['line'] in seen:
            continue
        seen.add(c['line'])
        by_race.setdefault(c['race_id'], []).append(c)

    result = {}
    for rid, bets in by_race.items():
        result[rid] = {'bets': bets, 'total_cost': sum(b['cost'] for b in bets)}
    logger.info(f"Kelly2: {len(seen)} bets / {len(result)} races for {day}")
    return result


def _load_kelly3_score_df(day):
    """Kelly3.ipynb (Cell4) と同一の TCSV 直下モデルスコアCSVを読み込み、
    race_id / horse_num を付与した DataFrame を返す。
    Ensemble を優先し、無い場合は他モデルの平均で補完する。"""
    score_dfs = {}
    for m in ['Ensemble', 'AutoGluon', 'LightGBM', 'CatBoost', 'XGBoost', 'TabNet']:
        name = f"Ensemble_{day}.csv" if m == 'Ensemble' else f"{m}_raw_{day}.csv"
        fp = os.path.join(KELLY_TCSV_DIR, name)
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            logger.warning(f"Kelly3: score CSV read failed ({fp}): {e}")
            continue
        if 'race_horse_id' not in df.columns or 'score' not in df.columns:
            continue
        df['race_horse_id'] = df['race_horse_id'].astype(str)
        score_dfs[m] = df.set_index('race_horse_id')['score']

    if not score_dfs:
        return None
    if 'Ensemble' not in score_dfs:
        score_dfs['Ensemble'] = pd.concat(list(score_dfs.values()), axis=1).mean(axis=1)

    model_names = list(score_dfs.keys())
    df_scores = pd.DataFrame(score_dfs).reset_index()
    id_col = df_scores.columns[0]
    df_scores['race_horse_id'] = df_scores[id_col].astype(str)
    year_prefix = str(day)[:4]
    df_scores['race_id'] = year_prefix + df_scores['race_horse_id'].str[8:16]
    df_scores['horse_num'] = df_scores['race_horse_id'].str[16:18].astype(int)
    return df_scores, model_names


def build_kelly3_portfolio_for_day(day, races_of_day):
    """Kelly3.ipynb (Cell4) の「中央値重視・堅牢ポートフォリオ」と同一ロジックで
    当日の買い目を生成し、画面表示用の strat2 (sub_items 付き) を返す。
    - モデルスコア : TCSV 直下の {model}_raw_{day}.csv / Ensemble_{day}.csv
    - オッズ/人気  : DB(odds1_tansho)
    - 買い目決定   : modules.rl_betting.generate_smart_balanced_bets (Kelly3 本家ロジック)
    """
    if not rl_agent_available:
        logger.warning("Kelly3: modules.rl_betting が利用できないためポートフォリオ生成をスキップ")
        return {}

    loaded = _load_kelly3_score_df(day)
    if loaded is None:
        logger.warning(f"Kelly3: no score CSV for {day} -> PICKUP 2 は生成しません")
        return {}
    df_scores, models = loaded

    # --- オッズ・人気 (DB) ---
    odds_df = _load_tansho_odds_df(day)
    odds_map = {}
    for _, row in odds_df.iterrows():
        rc = str(row['race_code'])
        if len(rc) < 16:
            continue
        rid = rc[:4] + rc[8:16]
        odds_map[(rid, int(row['umaban']))] = (
            float(row['odds_val']) if pd.notnull(row['odds_val']) else 5.0,
            int(row['ninki']) if pd.notnull(row['ninki']) else 99,
        )

    # --- レースごとの Ensemble 順位・z_score を集計 (Kelly3.ipynb Cell4 と同一) ---
    race_ranks = {}
    all_model_ranks = {}
    for rid, group in df_scores.groupby('race_id'):
        all_model_ranks[rid] = {}
        for m in models:
            if m in group.columns:
                all_model_ranks[rid][m] = group.sort_values(m, ascending=False)['horse_num'].tolist()
        if 'Ensemble' not in group.columns:
            continue
        sorted_ens = group.sort_values('Ensemble', ascending=False)
        s_vals = sorted_ens['Ensemble'].values
        std_val = s_vals.std() if len(s_vals) > 1 and s_vals.std() > 0 else 1.0
        z_scores = (s_vals - s_vals.mean()) / std_val
        r_list = []
        for i, (_, row) in enumerate(sorted_ens.iterrows()):
            h_num = int(row['horse_num'])
            o_val, n_val = odds_map.get((str(rid), h_num), (5.0, 99))
            r_list.append((h_num, float(row['Ensemble']), float(z_scores[i]), 1.0, o_val, n_val))
        race_ranks[str(rid)] = r_list

    # --- レーン割り当て (Kelly3.ipynb: 発走時刻順 mod3) ---
    sorted_rids = sorted(races_of_day.keys(),
                         key=lambda x: (races_of_day[x].get('start_time', '99:99'), x))
    lane_info = {}
    for idx, rid in enumerate(sorted_rids):
        lane = 'Lane_A' if idx % 3 == 0 else ('Lane_B' if idx % 3 == 1 else 'Lane_C')
        lane_jp = 'レーンA' if idx % 3 == 0 else ('レーンB' if idx % 3 == 1 else 'レーンC')
        lane_info[str(rid)] = (lane, lane_jp)

    result = {}
    for rid, r_info in races_of_day.items():
        rid_str = str(rid)
        h_list = race_ranks.get(rid_str)
        if not h_list or len(h_list) < 4:
            continue
        meta = r_info.get('meta') or {}
        track_name = meta.get('track_name') or KELLY_TRACK_MAPPING.get(rid_str[4:6], 'UNKNOWN')
        round_val = str(r_info.get('round', '1'))
        race_num = int(round_val) if round_val.isdigit() else 1

        top1, top2 = h_list[0], h_list[1]
        top1_pop = top1[5] if len(top1) > 5 and top1[5] < 99 else 1
        top1_odds = top1[4]
        score_gap = top1[1] - top2[1]

        lines, combs, cost, act_name = generate_smart_balanced_bets(
            str(day), track_name, race_num, [h[:5] for h in h_list],
            top1_pop=top1_pop, score_gap=score_gap, top1_odds=top1_odds
        )
        if not lines:
            continue

        # 券種ごとに集約して sub_items を作る
        grouped = {}
        order = []
        for ln in lines:
            parts = ln.split(',')
            if len(parts) < 8:
                continue
            code = parts[3]
            nums = [int(x) for x in re.findall(r'\d{2}', parts[6])]
            amount = int(parts[7]) if parts[7].isdigit() else 100
            if not nums:
                continue
            combo = tuple(nums) if code in KELLY_ORDERED_BET_CODES else tuple(sorted(nums))
            if code not in grouped:
                grouped[code] = {'combos': [], 'amounts': []}
                order.append(code)
            grouped[code]['combos'].append(combo)
            grouped[code]['amounts'].append(amount)

        sub_items = []
        for code in order:
            g = grouped[code]
            uniq, seen_c = [], set()
            for c in g['combos']:
                if c not in seen_c:
                    seen_c.add(c)
                    uniq.append(c)
            sub_items.append(_kelly3_sub_item(code, uniq, g['amounts']))
        if not sub_items:
            continue

        lane, lane_jp = lane_info.get(rid_str, ('Lane_A', 'レーンA'))
        result[rid_str] = {
            'action_id': 1,
            'action_name': f"Kelly3 {act_name}" if act_name else "Kelly3 中央値重視ポートフォリオ",
            'is_pickup': True,
            'cost': int(sum(s['cost'] for s in sub_items)),
            'combs': int(sum(s['combs'] for s in sub_items)),
            'axis1Num': top1[0],
            'axis2Num': None,
            'partnerNums': [h[0] for h in h_list[1:4]],
            'bettingEyesText': ' / '.join(s['bettingEyesText'] for s in sub_items),
            'rawType': 'Kelly3-ポートフォリオ',
            'lines': lines,
            'model': 'Kelly3 (中央値重視ポートフォリオ)',
            'sub_items': sub_items,
            'lane': lane,
            'lane_jp': lane_jp,
            'top_horses': '-'.join(str(h[0]) for h in h_list[:4]),
        }

    logger.info(f"Kelly3: {len(result)} races with portfolio bets for {day}")
    return result


def _kelly3_sub_item(code, combos, amounts):
    """券種ごとに集約した買い目を Web 表示用 sub_item に変換する。
    combos : [(1,2,3), ...] 順序あり券種はそのままの順、順不同券種はソート済み
    amounts: [100, 100, ...] 各点の購入金額"""
    jp = KELLY_BET_CODE_JP.get(code, code)
    ordered = code in KELLY_ORDERED_BET_CODES
    pad = lambda n: f"{int(n):02d}"
    total_amount = int(sum(int(a) for a in amounts))
    n_combs = len(combos)
    axis1 = axis2 = None
    partners = []
    eyes = ""
    raw_type = jp

    if ordered:
        firsts = {c[0] for c in combos}
        if (n_combs == 2 and len(combos[0]) == 2
                and tuple(combos[1]) == (combos[0][1], combos[0][0])):
            # 折り返し (a  b)
            axis1, axis2 = combos[0][0], combos[0][1]
            partners = [axis2]
            eyes = f"{pad(axis1)} ↔ {pad(axis2)}"
            raw_type = f"{jp}-折り返し"
        elif len(firsts) == 1 and len(combos[0]) >= 3:
            axis1 = combos[0][0]
            s2 = list(dict.fromkeys(c[1] for c in combos))
            s3 = list(dict.fromkeys(c[2] for c in combos))
            if sorted(s2) == sorted(s3) and len(s2) >= 2:
                # フォーメーション (軸1頭 → 2着候補 → 3着候補)
                partners = sorted(s2)
                eyes = (f"{pad(axis1)} → {', '.join(pad(x) for x in sorted(s2))}"
                        f" → {', '.join(pad(x) for x in sorted(s3))}")
                raw_type = f"{jp}-フォーメーション"
            else:
                partners = sorted(set(s2) | set(s3))
                eyes = " / ".join("→".join(pad(x) for x in c) for c in combos)
                raw_type = f"{jp}-1頭軸ながし"
        elif len(firsts) == 2 and all(len(c) == 3 for c in combos):
            axis1, axis2 = sorted(firsts)
            partners = sorted({c[2] for c in combos})
            eyes = " / ".join("→".join(pad(x) for x in c) for c in combos)
            raw_type = f"{jp}-2頭軸ながし"
        else:
            partners = sorted({x for c in combos for x in c})
            eyes = " / ".join("→".join(pad(x) for x in c) for c in combos)
    else:
        if len(combos) == 1 and len(combos[0]) == 1:
            # 単勝/複勝などの 1点買い
            axis1 = combos[0][0]
            eyes = pad(axis1)
            raw_type = f"{jp}-1点"
        else:
            common = set(combos[0])
            for c in combos[1:]:
                common &= set(c)
            all_nums = sorted({x for c in combos for x in c})
            if common and len(all_nums) > len(common):
                axes = sorted(common)
                others = [n for n in all_nums if n not in common]
                axis1 = axes[0]
                axis2 = axes[1] if len(axes) > 1 else None
                partners = others
                axes_txt = " - ".join(pad(x) for x in axes)
                eyes = f"{axes_txt} - {', '.join(pad(x) for x in others)}"
                raw_type = f"{jp}-{len(axes)}頭軸{len(others)}頭ながし"
            elif common:
                # 全点共通の組み合わせ (実質1点)
                axes = sorted(common)
                axis1 = axes[0]
                axis2 = axes[1] if len(axes) > 1 else None
                partners = []
                eyes = " - ".join(pad(x) for x in axes)
                raw_type = f"{jp}-1点"
            else:
                partners = all_nums
                eyes = f"{', '.join(pad(x) for x in all_nums)} BOX"
                raw_type = f"{jp}-{len(all_nums)}頭BOX"

    return {
        'type': code,
        'rawType': raw_type,
        'bet_type_jp': jp,
        'combs': n_combs,
        'cost': total_amount,
        'bettingEyesText': eyes,
        'axis1Num': axis1,
        'axis2Num': axis2,
        'partnerNums': partners,
    }


def generate_static_html():
    eval_dir = r"C:\Users\kyoui\keiba\data\eval"
    output_html_path = r"C:\Users\kyoui\tohshin_keiba\index.html"
    strategies_csv_path = r"C:\Users\kyoui\keiba\config\winning_strategies_v13.csv"
    strategies2_csv_path = r"C:\Users\kyoui\keiba\config\winning_strategies2.csv"
    race_id_list_path = r"C:\Users\kyoui\keiba\data\raceid\raceIdList.csv"
    race_meta_cache = {}

    # PICKUP / PICKUP 2 の買い目は本家ノートブック (Kelly2.ipynb / Kelly3.ipynb) 準拠で生成する
    # (強化学習DQNによる旧アクション推論は廃止)
    
    # 発走時刻データの読み込み (raceIdList.csv)
    race_time_dict = {}
    if os.path.exists(race_id_list_path):
        try:
            rid_df = pd.read_csv(race_id_list_path, dtype={'race_id': str, 'time': str})
            rid_df['race_id_str'] = rid_df['race_id'].astype(str).str.zfill(12)
            race_time_dict = dict(zip(rid_df['race_id_str'], rid_df['time'].astype(str)))
            logger.info(f"Loaded {len(race_time_dict)} race post times from {race_id_list_path}")
        except Exception as e:
            logger.error(f"Error loading raceIdList.csv: {e}")
    else:
        logger.warning(f"raceIdList.csv not found: {race_id_list_path}")
    
    # 戦略データの読み込み (戦略1: winning_strategies_v13.csv)
    strategies_dict = {}
    shubetsu_order = {'3連単': 1, '3連複': 2, '馬単': 3, '馬連': 4, 'ワイド': 5, '単勝': 6}
    type_order = {
        '2頭軸ながし': 1,
        '1頭軸ながし': 2,
        'ながし': 3,
        '2頭軸マルチ': 4,
        '1頭軸マルチ': 5,
        'マルチ': 6,
        'BOX': 7
    }
    if os.path.exists(strategies_csv_path):
        try:
            sdf = pd.read_csv(strategies_csv_path, encoding='utf-8-sig')
            # NaNをNoneに置き換える (JSONでnullとして出力される)
            sdf = sdf.astype(object).where(pd.notnull(sdf), None)
            # BUY行のみ使用 (EXCLUDE行は除外フィルタとして別途使われる想定)
            if 'action' in sdf.columns:
                sdf = sdf[sdf['action'] == 'BUY'].copy()
            
            for _, row in sdf.iterrows():
                row_dict = row.to_dict()
                # v13形式: bet_type -> type にマッピング
                raw_type = str(row_dict.get('bet_type') or row_dict.get('type') or '')
                row_dict['type'] = raw_type
                parts = raw_type.split('-')
                shubetsu = parts[0].strip() if len(parts) > 0 else ''
                type_sub = parts[1].strip() if len(parts) > 1 else ''
                row_dict['shubetsu'] = shubetsu
                row_dict['type_sub'] = type_sub
                row_dict['s_rank'] = shubetsu_order.get(shubetsu, 99)
                row_dict['t_rank'] = type_order.get(type_sub, 99)
                # ROI/的中率カラムのマッピング
                if 'roi' not in row_dict:
                    row_dict['roi'] = row_dict.get('roi_total', 0)
                if 'hit_rate' not in row_dict:
                    row_dict['hit_rate'] = row_dict.get('monthly_win_rate', 0)

                # v13: cat_col='venue_name'の場合はvalで会場別グループ化、それ以外は'全場'
                cat_col = str(row_dict.get('cat_col') or '')
                if cat_col == 'venue_name':
                    v_name = str(row_dict.get('val') or '全場')
                elif 'venue_name' in row_dict:
                    v_name = str(row_dict['venue_name'])
                else:
                    v_name = '全場'
                if v_name not in strategies_dict:
                    strategies_dict[v_name] = []
                strategies_dict[v_name].append(row_dict)
            logger.info(f"Loaded {sum(len(v) for v in strategies_dict.values())} BUY strategies from {strategies_csv_path}")
        except Exception as e:
            logger.error(f"Error loading strategies CSV: {e}")
    else:
        logger.warning(f"Strategies CSV not found: {strategies_csv_path}")

    # 戦略2データの読み込み (戦略2: winning_strategies2.csv)
    strategies2_list = []
    if os.path.exists(strategies2_csv_path):
        try:
            s2df = pd.read_csv(strategies2_csv_path, encoding='utf-8-sig')
            s2df = s2df.astype(object).where(pd.notnull(s2df), None)
            strategies2_list = s2df.to_dict('records')
            logger.info(f"Loaded {len(strategies2_list)} strategies2 from {strategies2_csv_path}")
        except Exception as e:
            logger.error(f"Error loading strategies2 CSV: {e}")
    else:
        logger.warning(f"Strategies2 CSV not found: {strategies2_csv_path}")

    # 評価理由データの読み込み (eval_reasons.json)
    eval_reasons_dict = {}
    eval_reasons_candidates = [
        r"C:\Users\kyoui\keiba\data\eval\eval_reasons.json",
        r"C:\Users\kyoui\keiba\eval_reasons.json",
        r"C:\Users\kyoui\tohshin_keiba\eval_reasons.json",
        r"C:\Users\kyoui\tohshin_keiba\jsons\eval_reasons.json"
    ]
    for er_path in eval_reasons_candidates:
        if os.path.exists(er_path):
            try:
                with open(er_path, "r", encoding="utf-8") as f:
                    loaded_reasons = json.load(f)
                    eval_reasons_dict.update(loaded_reasons)
                logger.info(f"Loaded {len(loaded_reasons)} eval reasons from {er_path}")
            except Exception as e:
                logger.error(f"Error loading {er_path}: {e}")

    # jsons/eval_reasons.json にも保存
    if eval_reasons_dict:
        try:
            eval_reasons_out = r"C:\Users\kyoui\tohshin_keiba\jsons\eval_reasons.json"
            os.makedirs(os.path.dirname(eval_reasons_out), exist_ok=True)
            with open(eval_reasons_out, "w", encoding="utf-8") as f:
                json.dump(eval_reasons_dict, f, ensure_ascii=False)
            logger.info(f"Saved merged eval_reasons to {eval_reasons_out}")
        except Exception as e:
            logger.error(f"Failed to write eval_reasons.json: {e}")

    logger.info(f"Loading all picke files from {eval_dir}...")
    
    if not os.path.exists(eval_dir):
        logger.error(f"Directory not found: {eval_dir}")
        return

    import glob
    from datetime import datetime, timedelta
    all_pickle_files = glob.glob(os.path.join(eval_dir, "*.pickle"))
    if not all_pickle_files:
        logger.error(f"No pickle files found in {eval_dir}")
        return

    # 直近1ヶ月以内に更新されたpickleファイルのみ対象
    pickle_mtime_cutoff = (datetime.now() - timedelta(days=31)).timestamp()
    pickle_files = [f for f in all_pickle_files if os.path.getmtime(f) >= pickle_mtime_cutoff]
    logger.info(f"Pickle files: {len(pickle_files)} recent / {len(all_pickle_files)} total (cutoff: {datetime.fromtimestamp(pickle_mtime_cutoff).strftime('%Y-%m-%d')})")
    if not pickle_files:
        logger.warning(f"No recently modified pickle files found. Using all files.")
        pickle_files = all_pickle_files
    
    # Extract direct features from pickle
    try:
        import re
        all_dfs = []
        for pf in pickle_files:
            logger.info(f"  Reading {os.path.basename(pf)}...")
            df_part = pd.read_pickle(pf)
            
            # カラム名の揺れを吸収
            if 'id' in df_part.columns and 'race_id' not in df_part.columns:
                df_part = df_part.rename(columns={'id': 'race_id'})
                
            # race_id が欠損している場合の補完
            r_id_col = next((c for c in ['race_id', 'レースID'] if c in df_part.columns), None)
            
            if r_id_col is None or df_part[r_id_col].isna().all():
                # 1. race_horse_id から抽出 (上4桁 + 9-16桁目)
                if 'race_horse_id' in df_part.columns:
                    # 例: 202603070901051001 -> 2026 + 09010510 = 202609010510
                    def extract_rid(val):
                        s = str(val)
                        if len(s) >= 16:
                            return s[:4] + s[8:16]
                        return val
                    df_part['race_id'] = df_part['race_horse_id'].apply(extract_rid)
                    logger.info(f"    Restored race_id from race_horse_id for {os.path.basename(pf)}")
                # 2. ファイル名から抽出 (12桁の数値)
                else:
                    match = re.search(r'(\d{12})', os.path.basename(pf))
                    if match:
                        df_part['race_id'] = match.group(1)
                        logger.info(f"    Restored race_id from filename for {os.path.basename(pf)}")
            
            # コンカット高速化のため必要なカラム関連のみ残す
            keep_cols = [c for c in df_part.columns if any(k in str(c).lower() for k in ['id', 'date', 'race', 'horse', '番', '名', 'raw', 'lightgbm', 'xgboost', 'catboost', 'lstm', 'randomforest', 'decisiontree', 'transformer', 'tabnet', 'ensemble', 'python', 'lgbm', 'レース', '馬'])]
            all_dfs.append(df_part[keep_cols] if keep_cols else df_part)
        
        df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Total records before deduplication: {len(df)}")
        
        # Deduplicate by race_id and horse_number if available
        # Find ID columns for deduplication
        h_num_col = None
        for col in ['horse_number', '馬番']:
            if col in df.columns:
                h_num_col = col
                break
        
        r_id_col = None
        for col in ['race_id', 'レースID']:
            if col in df.columns:
                r_id_col = col
                break
        
        if r_id_col and h_num_col:
            df = df.drop_duplicates(subset=[r_id_col, h_num_col], keep='last')
            logger.info(f"Total records after deduplication: {len(df)}")

        date_col = None
        if 'date' in df.columns:
            date_col = 'date'
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col])
            # 2026以降のものだけにフィルタリング
            df = df[df[date_col].dt.year >= 2026].copy()
            # 常に全データ（評価対象全て）を表示したい場合、フィルタリングは緩くするか最新日に合わせる
            # ここでは最新の日付から数日分を表示するようにフィルタを調整
            latest_date = df[date_col].max()
            logger.info(f"Latest date in data: {latest_date}")
            # df = df[df[date_col] >= latest_date - pd.Timedelta(days=7)].copy()
            df['date_str'] = df[date_col].dt.strftime('%Y-%m-%d')
        else:
            df['date_str'] = ""
    except Exception as e:
        logger.error(f"Data loading/processing error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return
        
    # Identity columns logic
    race_id_col = None
    for col in ['race_id', 'レースID']:
        if col in df.columns:
            race_id_col = col
            break
    
    if not race_id_col:
        for col in df.columns:
            if 'ID' in str(col) or 'id' in str(col).lower():
                race_id_col = col
                break
    
    if not race_id_col:
        logger.error("No 'race_id' column found in pickle")
        return
        
    # Get horse number
    horse_num_col = None
    for col in ['horse_number', '馬番', '鬥ｬ逡ｪ', 'umaban']:
        if col in df.columns:
            horse_num_col = col
            break
            
    # Get horse name
    horse_name_col = None
    for col in ['horse_name', '馬名', '鬥ｬ蜷', 'horse_name_latest']:
        if col in df.columns:
            horse_name_col = col
            break
            
    # Fill missing names/numbers
    if not horse_num_col:
        df['馬番_temp'] = range(1, len(df) + 1)
        horse_num_col = '馬番_temp'
        
    if not horse_name_col:
        df['馬名_temp'] = "No Name"
        horse_name_col = '馬名_temp'

    # AutoGluon スコアの読み込み (C:\keibasoftcom\KSCAutoBetPlus\TCSV)
    tcsv_dir = r"C:\keibasoftcom\KSCAutoBetPlus\TCSV"
    if os.path.exists(tcsv_dir):
        unique_dates = df['date_str'].unique()
        ag_dfs = []
        for d_str in unique_dates:
            if not d_str: continue
            r_day = d_str.replace('-', '')
            fp_ag = os.path.join(tcsv_dir, f"AutoGluon_raw_{r_day}.csv")
            if os.path.exists(fp_ag):
                try:
                    df_ag = pd.read_csv(fp_ag)
                    s_id = df_ag['race_horse_id'].astype(str)
                    df_ag['ag_race_id'] = s_id.apply(lambda s: s[:4] + s[8:16] if len(s) >= 16 else "")
                    df_ag['ag_horse_num'] = s_id.apply(lambda s: int(s[16:18]) if len(s) >= 18 else None)
                    df_ag['AutoGluon_raw'] = pd.to_numeric(df_ag['score'], errors='coerce')
                    valid_ag = df_ag[['ag_race_id', 'ag_horse_num', 'AutoGluon_raw']].dropna(subset=['ag_race_id', 'ag_horse_num'])
                    ag_dfs.append(valid_ag)
                except Exception as e:
                    logger.error(f"Error reading {fp_ag}: {e}")
        if ag_dfs:
            all_ag = pd.concat(ag_dfs, ignore_index=True).drop_duplicates(subset=['ag_race_id', 'ag_horse_num'])
            df['temp_rid'] = df[race_id_col].astype(str)
            df['temp_hnum'] = pd.to_numeric(df[horse_num_col], errors='coerce')
            df = df.reset_index(drop=True)
            if 'AutoGluon_raw' in df.columns:
                df.drop(columns=['AutoGluon_raw'], inplace=True)
            df = pd.merge(df, all_ag, left_on=['temp_rid', 'temp_hnum'], right_on=['ag_race_id', 'ag_horse_num'], how='left')
            if 'AutoGluon_raw' not in df.columns:
                df['AutoGluon_raw'] = 0.0
            else:
                df['AutoGluon_raw'] = df['AutoGluon_raw'].fillna(0.0)
            df.drop(columns=['temp_rid', 'temp_hnum', 'ag_race_id', 'ag_horse_num'], errors='ignore', inplace=True)
            logger.info(f"Merged AutoGluon scores: {(df['AutoGluon_raw'] > 0).sum()} valid entries")
        else:
            df['AutoGluon_raw'] = 0.0
    else:
        df['AutoGluon_raw'] = 0.0

    # Get scores and map column names
    score_mapping = {
        'LightGBM': 'LightGBM_raw',
        'XGBoost': 'XGBoost_raw',
        'CatBoost': 'CatBoost_raw',
        'LSTM': 'LSTM_raw',
        'RandomForest': 'RandomForest_raw',
        'DecisionTree': 'DecisionTree_raw',
        'Transformer': 'Transformer_raw',
        'TabNet': 'TabNet_raw',
        'Ensemble': 'Python',
        'AutoGluon': 'AutoGluon_raw'
    }
    
    # Mapping logic to capture scores from various possible column names
    for base_name, raw_name in score_mapping.items():
        # raw_name (LGBM_raw etc) が存在していても、中身が全て 0 の場合は base_name (LGBM etc) からの取得を試みる
        take_from_base = False
        if raw_name not in df.columns:
            take_from_base = True
        elif raw_name in df.columns:
            # 数値変換して全て 0 かチェック
            try:
                temp_vals = pd.to_numeric(df[raw_name], errors='coerce').fillna(0)
                # 全ての要素が 0 もしくは欠損値である場合
                if (temp_vals == 0).all():
                    take_from_base = True
            except:
                take_from_base = True
        
        if take_from_base and base_name in df.columns:
            df[raw_name] = df[base_name]
            logger.info(f"  Captured {raw_name} from {base_name}")
        elif raw_name in df.columns:
            logger.info(f"  {raw_name} already contains data or {base_name} is missing")

    req_scores = ['LightGBM_raw', 'XGBoost_raw', 'CatBoost_raw', 'LSTM_raw', 'RandomForest_raw', 'DecisionTree_raw', 'Transformer_raw', 'TabNet_raw', 'Ensemble', 'AutoGluon_raw']
    for s in req_scores:
        if s not in df.columns:
            df[s] = 0.0

    # Construct final dataset
    df_out = pd.DataFrame()
    df_out['race_id'] = df[race_id_col].astype(str)
    df_out['date_str'] = df['date_str']
    df_out['horse_number'] = pd.to_numeric(df[horse_num_col], errors='coerce')
    df_out['horse_name'] = df[horse_name_col].astype(str)
    
    for s in req_scores:
        df_out[s] = pd.to_numeric(df[s], errors='coerce')
    
    df_out = df_out.fillna({s: 0.0 for s in req_scores})

    # group by race_id
    races = {}
    grouped = df_out.groupby('race_id')
    for name, group in grouped:
        records = group.to_dict('records')
        race_id_str = str(name)
        
        # Determine Date, Place, Round
        date_val = records[0].get('date_str', '')
        
        # Determine Place name
        # Netkeiba ID: YYYY(0:4) Place(4:6) Times(6:8) Day(8:10) Round(10:12)
        place_code = race_id_str[4:6] if len(race_id_str) >= 6 else ""
        place_name = REVERSE_PLACE_DICT.get(place_code, "")
        
        # 見つからない場合はフェイルセーフ
        if not place_name:
            place_name = place_code if place_code else "Unknown"

        round_no = race_id_str[10:12] if len(race_id_str) >= 12 else ''
        try:
            round_int = int(round_no)
        except ValueError:
            round_int = round_no
            
        weekday_ja = ""
        if date_val:
            try:
                from datetime import datetime
                dt = datetime.strptime(date_val, "%Y-%m-%d")
                weekdays_ja = ["月", "火", "水", "木", "金", "土", "日"]
                weekday_ja = weekdays_ja[dt.weekday()]
            except Exception as e:
                logger.error(f"Error parsing weekday: {e}")

        if date_val and place_name and round_int:
            race_title = f"{date_val} {place_name} {round_int}R"
        elif place_name and round_int:
            race_title = f"{place_name} {round_int}R"
        else:
            race_title = f"Race {race_id_str}"
            
        # この会場に対応する推奨戦略を取得
        race_strategies = strategies_dict.get(place_name, []) + strategies_dict.get('全場', [])
        
        # 評価理由データの取得
        race_reasons = eval_reasons_dict.get(race_id_str, {})
        if not race_reasons:
            for k in [race_id_str.zfill(12), race_id_str[:12] if len(race_id_str) >= 12 else race_id_str]:
                if k in eval_reasons_dict:
                    race_reasons = eval_reasons_dict[k]
                    break
            
        # レースメタデータ（戦略2用）の取得
        race_meta_item = {}
        if date_val:
            r_day = date_val.replace('-', '')
            if r_day not in race_meta_cache:
                shutuba_p = os.path.join(r"C:\Users\kyoui\keiba\data\tmp", f"shutuba_{r_day}.pickle")
                day_meta = {}
                if os.path.exists(shutuba_p):
                    try:
                        s_df = pd.read_pickle(shutuba_p)
                        for s_rid_idx, s_grp in s_df.groupby(s_df.index):
                            f_row = s_grp.iloc[0]
                            s_rid_str = str(s_rid_idx)
                            v_code = s_rid_str[4:6] if len(s_rid_str) >= 6 else ""
                            v_name = REVERSE_PLACE_DICT.get(v_code, 'その他')
                            
                            r_cls = str(f_row.get('race_class', ''))
                            if '新馬' in r_cls: c_cat = '新馬'
                            elif '未勝利' in r_cls: c_cat = '未勝利'
                            elif '1勝' in r_cls: c_cat = '1勝クラス'
                            elif '2勝' in r_cls: c_cat = '2勝クラス'
                            elif '3勝' in r_cls: c_cat = '3勝クラス'
                            elif 'オープン' in r_cls or 'OP' in r_cls: c_cat = 'オープン'
                            elif any(g in r_cls for g in ['G1', 'G2', 'G3', '重賞']): c_cat = '重賞'
                            elif '障害' in r_cls: c_cat = '障害'
                            else: c_cat = 'その他'

                            r_trk = str(f_row.get('race_type', ''))
                            if '芝' in r_trk: t_trk = '芝'
                            elif 'ダート' in r_trk or 'ダ' in r_trk: t_trk = 'ダート'
                            elif '障害' in r_trk: t_trk = '障害'
                            else: t_trk = 'その他'

                            dist_val = f_row.get('distance', 0)
                            try:
                                dist_num = float(dist_val)
                            except:
                                dist_num = 0
                            
                            if dist_num > 2400: d_cat = '長距離 (>2400m)'
                            elif 1400 <= dist_num <= 1600: d_cat = 'マイル (1400-1600m)'
                            elif 0 < dist_num < 1400: d_cat = '短距離 (<1400m)'
                            elif 1800 <= dist_num <= 2200: d_cat = '中距離 (1800-2200m)'
                            else: d_cat = 'その他'

                            baba_val = str(f_row.get('baba', f_row.get('track_condition', '')))
                            baba_cond = '不良' if '不' in baba_val else ('重' if '重' in baba_val else ('稍' if '稍' in baba_val else '良'))

                            day_meta[s_rid_str] = {
                                'venue_name': v_name,
                                'track_name': VENUE_MAP.get(v_code, 'UNKNOWN') if 'VENUE_MAP' in globals() else v_name,
                                'class_cat': c_cat,
                                'track_type': t_trk,
                                'course_len': dist_num if dist_num > 0 else 1600.0,
                                'n_horses': len(s_grp),
                                'venue_track': f"{v_name}_{t_trk}",
                                'class_venue': f"{c_cat}_{v_name}",
                                'class_venue_track': f"{c_cat}_{v_name}_{t_trk}",
                                'class_track': f"{c_cat}_{t_trk}",
                                'dist_track': f"{d_cat}_{t_trk}",
                                'track_ground': f"{t_trk}_{baba_cond}"
                            }
                    except Exception as e:
                        logger.error(f"Error reading {shutuba_p}: {e}")
                race_meta_cache[r_day] = day_meta
            
            race_meta_item = race_meta_cache.get(r_day, {}).get(race_id_str, {})
        
        if not race_meta_item:
            race_meta_item = {
                'venue_name': place_name,
                'track_name': VENUE_MAP.get(place_code, 'UNKNOWN') if 'VENUE_MAP' in globals() else place_name,
                'class_cat': 'その他',
                'track_type': 'その他',
                'course_len': 1600.0,
                'n_horses': 14,
                'venue_track': f"{place_name}_その他",
                'class_venue': f"その他_{place_name}",
                'class_venue_track': f"その他_{place_name}_その他",
                'class_track': 'その他_その他',
                'dist_track': 'その他_その他',
                'track_ground': 'その他_良'
            }

        races[race_id_str] = {
            "race_id": race_id_str,
            "title": race_title,
            "date": date_val,
            "place": place_name,
            "weekday": weekday_ja,
            "round": str(round_int),
            "horses": records,
            "strategies": race_strategies,
            "meta": race_meta_item,
            "reasons": race_reasons
        }

    # データを日付ごとにグループ化
    dates_data = {}
    for r_id, r_info in races.items():
        d = r_info.get('date', 'unknown')
        if d not in dates_data:
            dates_data[d] = {}
        dates_data[d][r_id] = r_info

    # JRA 標準発走時刻テーブル
    POST_TIMES_3 = {
        0: {1: "09:50", 2: "10:20", 3: "10:50", 4: "11:20", 5: "12:10", 6: "12:40", 7: "13:10", 8: "13:40", 9: "14:15", 10: "14:50", 11: "15:25", 12: "16:05"},
        1: {1: "10:05", 2: "10:35", 3: "11:05", 4: "11:35", 5: "12:25", 6: "12:55", 7: "13:25", 8: "13:55", 9: "14:25", 10: "15:00", 11: "15:35", 12: "16:15"},
        2: {1: "10:15", 2: "10:45", 3: "11:15", 4: "11:45", 5: "12:35", 6: "13:05", 7: "13:35", 8: "14:05", 9: "14:35", 10: "15:10", 11: "15:45", 12: "16:30"}
    }
    POST_TIMES_2 = {
        0: {1: "10:00", 2: "10:30", 3: "11:00", 4: "11:30", 5: "12:20", 6: "12:50", 7: "13:20", 8: "13:50", 9: "14:25", 10: "15:00", 11: "15:35", 12: "16:10"},
        1: {1: "10:15", 2: "10:45", 3: "11:15", 4: "11:45", 5: "12:35", 6: "13:05", 7: "13:35", 8: "14:05", 9: "14:40", 10: "15:15", 11: "15:45", 12: "16:25"}
    }
    POST_TIMES_1 = {
        0: {1: "10:00", 2: "10:35", 3: "11:05", 4: "11:35", 5: "12:25", 6: "12:55", 7: "13:25", 8: "13:55", 9: "14:30", 10: "15:05", 11: "15:40", 12: "16:20"}
    }

    # 各日付ごとに発走時刻を付与・タイトル更新
    for d, d_races in dates_data.items():
        place_codes = sorted(list(set(r_id[4:6] for r_id in d_races.keys() if len(r_id) >= 6)))
        num_places = len(place_codes)
        
        for r_id, r_info in d_races.items():
            p_code = r_id[4:6] if len(r_id) >= 6 else ""
            p_idx = place_codes.index(p_code) if p_code in place_codes else 0
            try:
                r_num = int(r_info.get('round', '1'))
            except:
                r_num = 1
                
            if r_id in race_time_dict and race_time_dict[r_id] and str(race_time_dict[r_id]).strip():
                s_time = str(race_time_dict[r_id]).strip()
            elif num_places >= 3:
                s_time = POST_TIMES_3.get(p_idx, POST_TIMES_3[0]).get(r_num, "10:00")
            elif num_places == 2:
                s_time = POST_TIMES_2.get(p_idx, POST_TIMES_2[0]).get(r_num, "10:00")
            else:
                s_time = POST_TIMES_1[0].get(r_num, "10:00")
                
            r_info['start_time'] = s_time
            d_str = r_info.get('date', '')
            p_str = r_info.get('place', '')
            rd_str = r_info.get('round', '')
            r_info['title'] = f"{p_str}{rd_str}R {s_time}".strip()

        # ==================================================================
        # PICKUP (戦略1) : Kelly2.ipynb と同一ロジックの買い目
        # PICKUP 2 (戦略2): Kelly3.ipynb (中央値重視ポートフォリオ) の買い目
        # ==================================================================
        day_str = d.replace('-', '')
        kelly2_map = build_kelly2_bets_for_day(day_str, d_races)
        if kelly2_map is not None:
            for k_rid in d_races:
                d_races[k_rid]['kelly2'] = kelly2_map.get(k_rid, {'bets': [], 'total_cost': 0})

        kelly3_map = build_kelly3_portfolio_for_day(day_str, d_races)
        if kelly3_map:
            for k_rid, strat2_item in kelly3_map.items():
                if k_rid in d_races:
                    d_races[k_rid]['strat2'] = strat2_item

    # 1. 各日付のデータを保存
    jsons_dir = r"C:\Users\kyoui\tohshin_keiba\jsons"
    for d, d_races in dates_data.items():
        out_json = os.path.join(jsons_dir, f"data_{d}.json")
        try:
            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(d_races, f, ensure_ascii=False)
            logger.info(f"Generated daily JSON: {out_json}")
        except Exception as e:
            logger.error(f"Failed to write daily JSON {out_json}: {e}")

    # 2. メタデータ（日付リスト）を保存
    # pickleフィルタにより dates_data は直近分のみの場合があるため、
    # 既存JSONファイルからも日付を収集して完全なリストを維持する
    import re as _re
    existing_dates = set()
    for jf in glob.glob(os.path.join(jsons_dir, "data_*.json")):
        m = _re.search(r'data_(\d{4}-\d{2}-\d{2})\.json$', os.path.basename(jf))
        if m:
            existing_dates.add(m.group(1))
    all_dates = sorted(existing_dates | set(dates_data.keys()))
    meta_data = {
        "dates": all_dates,
        "latest": max(all_dates) if all_dates else ""
    }
    meta_json_path = os.path.join(jsons_dir, "meta.json")
    try:
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False)
        logger.info(f"Generated meta.json at {meta_json_path} ({len(all_dates)} dates, {len(dates_data)} updated)")
    except Exception as e:
        logger.error(f"Failed to write meta.json: {e}")

    scriptable_file = os.path.join(os.path.dirname(__file__), "scriptable_smappy.js")
    _scriptable_app_code = ""
    if os.path.exists(scriptable_file):
        try:
            with open(scriptable_file, "r", encoding="utf-8") as sf:
                _scriptable_app_code = sf.read()
        except Exception as e:
            logger.warning(f"Failed to read scriptable_smappy.js: {e}")
    _scriptable_app_code_json = json.dumps(_scriptable_app_code)

    scriptable_ipat_file = os.path.join(os.path.dirname(__file__), "scriptable_ipat.js")
    _scriptable_ipat_code = ""
    if os.path.exists(scriptable_ipat_file):
        try:
            with open(scriptable_ipat_file, "r", encoding="utf-8") as sf:
                _scriptable_ipat_code = sf.read()
        except Exception as e:
            logger.warning(f"Failed to read scriptable_ipat.js: {e}")
    _scriptable_ipat_code_json = json.dumps(_scriptable_ipat_code)

    scriptable_umaca_file = os.path.join(os.path.dirname(__file__), "scriptable_umaca.js")
    _scriptable_umaca_code = ""
    if os.path.exists(scriptable_umaca_file):
        try:
            with open(scriptable_umaca_file, "r", encoding="utf-8") as sf:
                _scriptable_umaca_code = sf.read()
        except Exception as e:
            logger.warning(f"Failed to read scriptable_umaca.js: {e}")
    _scriptable_umaca_code_json = json.dumps(_scriptable_umaca_code)

    _smappy_fixed_bml = 'javascript:void((async function(){var sn={"1":"単勝","2":"複勝","3":"枠連","4":"馬連","5":"ワイド","6":"馬単","7":"3連複","8":"3連単"};var text="";if(navigator.clipboard&&navigator.clipboard.readText){try{text=await navigator.clipboard.readText();}catch(e){}}if(!text||text.indexOf("steps")<0){text=prompt("買い目データを貼り付けてください:",text||"");}if(!text)return;var data;try{data=JSON.parse(text);}catch(e){alert("データ形式が正しくありません");return;}var s=data.steps;var vn=data.venueName||"";var wd=data.weekday||"";if(!s||!s.length){alert("買い目データが空です");return;}var i=0,r=0,d=false,T=Date.now();function dg(m){var x=document.getElementById("smappy-diag");if(!x){x=document.createElement("div");x.id="smappy-diag";x.style="position:fixed;top:0;left:0;width:100%;z-index:100000;background:rgba(0,0,0,0.9);color:#0f0;font-size:10px;padding:4px;pointer-events:none;";document.body.appendChild(x);}x.innerText=m;}function fi(ok){if(d)return;d=true;dg("FINISH:"+ok);}function tp(e){var r=e.getBoundingClientRect();var x=r.left+r.width/2;var y=r.top+r.height/2;var o={bubbles:true,cancelable:true,clientX:x,clientY:y,view:window};try{var t=new Touch({identifier:Date.now(),target:e,clientX:x,clientY:y,radiusX:2,radiusY:2});var to={bubbles:true,cancelable:true,touches:[t],targetTouches:[t],changedTouches:[t],view:window};e.dispatchEvent(new TouchEvent("touchstart",to));e.dispatchEvent(new TouchEvent("touchend",to));}catch(err){}e.dispatchEvent(new MouseEvent("mousedown",o));e.dispatchEvent(new MouseEvent("mouseup",o));e.dispatchEvent(new MouseEvent("click",o));try{e.click();}catch(err){}}function cf(){var k=["金額","セット","次へ","決定"];var a=document.querySelectorAll("a,button");for(var j=0;j<a.length;j++){var b=a[j].getBoundingClientRect();if(b.width>0&&b.height>0){for(var l=0;l<k.length;l++){if(a[j].textContent.indexOf(k[l])>=0){tp(a[j]);return;}}}}}function nx(){try{if(Date.now()-T>25000){fi(false);return;}var p="";if(document.getElementById("jyo"))p="V";else if(document.getElementById("race"))p="R";else if(document.getElementById("siki"))p="S";else if(document.getElementById("hou"))p="M";else{var c=(document.body.innerText||"");if(c.indexOf("会場")>=0||c.indexOf("開催")>=0)p="V";if(c.indexOf("レース")>=0||c.indexOf("回次")>=0)p="R";if(c.indexOf("式別")>=0)p="S";if(c.indexOf("方式")>=0)p="M";}if(i>=s.length){dg("Done");cf();fi(true);return;}var v=s[i];var f=false;var vs=[v];var n=parseInt(v);if(!isNaN(n)){if(i===1){vs=[String(n-1),(n-1<10?"0"+(n-1):String(n-1))];}else{vs=[v,String(n),(n<10?"0"+n:String(n)),String(n-1),(n-1<10?"0"+(n-1):String(n-1))];}}dg("S"+i+":"+v+" r:"+r+" p:"+p);var okP=(i===0&&(p==="V"||p===""||r>1))||(i===1&&(p==="R"||p==="V"||p===""||r>1))||(i===2&&(p==="S"||r>1))||(i===3&&(p==="M"||p==="S"||r>1))||(i>3);if(okP){if(i===0){var bs=document.querySelectorAll("a,button");for(var k2=0;k2<bs.length;k2++){var b2=bs[k2].getBoundingClientRect();if(b2.width<=4||b2.height<=4||bs[k2].classList.contains("disabled"))continue;var t=(bs[k2].innerText||bs[k2].textContent||"").trim();if(vn&&t.indexOf(vn)>=0){tp(bs[k2]);i++;r=0;setTimeout(nx,450);f=true;break;}}if(!f){for(var k=0;k<vs.length;k++){var es=document.querySelectorAll("a[data-value=\'"+vs[k]+"\'],button[data-value=\'"+vs[k]+"\']");for(var j=0;j<es.length;j++){var b=es[j].getBoundingClientRect();if(b.width>3&&b.height>3){tp(es[j]);i++;r=0;setTimeout(nx,450);f=true;break;}}if(f)break;}}}else{for(var k=0;k<vs.length;k++){var es=document.querySelectorAll("a[data-value=\'"+vs[k]+"\'],button[data-value=\'"+vs[k]+"\']");for(var j=0;j<es.length;j++){var b=es[j].getBoundingClientRect();if(b.width>3&&b.height>3){tp(es[j]);i++;r=0;setTimeout(nx,450);f=true;break;}}if(f)break;}if(!f){var bs=document.querySelectorAll("a,button");for(var k2=0;k2<bs.length;k2++){var b2=bs[k2].getBoundingClientRect();if(b2.width<=4||b2.height<=4)continue;var t=(bs[k2].innerText||bs[k2].textContent||"").trim();if(i===1&&(t===v+"R"||t===v+"レース"||t.indexOf(v+"R")>=0)){tp(bs[k2]);i++;r=0;setTimeout(nx,450);f=true;break;}if(i===2&&sn[v]&&t.indexOf(sn[v])>=0){tp(bs[k2]);i++;r=0;setTimeout(nx,450);f=true;break;}}}}}if(!f){r++;setTimeout(nx,200);}}catch(e){dg("E:"+e.message);fi(false);}}nx();})());'
    _smappy_fixed_bml_json = json.dumps(_smappy_fixed_bml)
    _smappy_part2_js = 'try{if(typeof completion==="function")completion("OK");}catch(e){}var sn={"1":"単勝","2":"複勝","3":"枠連","4":"馬連","5":"ワイド","6":"馬単","7":"3連複","8":"3連単"};var i=0,r=0,d=false,T=Date.now();function dg(m){var x=document.getElementById("smappy-diag");if(!x){x=document.createElement("div");x.id="smappy-diag";x.style="position:fixed;top:0;left:0;width:100%;z-index:100000;background:rgba(0,0,0,0.9);color:#0f0;font-size:10px;padding:4px;pointer-events:none;";document.body.appendChild(x);}x.innerText=m;}function fi(ok){if(d)return;d=true;dg("FINISH:"+ok);}function tp(e){var r=e.getBoundingClientRect();var x=r.left+r.width/2;var y=r.top+r.height/2;var o={bubbles:true,cancelable:true,clientX:x,clientY:y,view:window};try{var t=new Touch({identifier:Date.now(),target:e,clientX:x,clientY:y,radiusX:2,radiusY:2});var to={bubbles:true,cancelable:true,touches:[t],targetTouches:[t],changedTouches:[t],view:window};e.dispatchEvent(new TouchEvent("touchstart",to));e.dispatchEvent(new TouchEvent("touchend",to));}catch(err){}e.dispatchEvent(new MouseEvent("mousedown",o));e.dispatchEvent(new MouseEvent("mouseup",o));e.dispatchEvent(new MouseEvent("click",o));try{e.click();}catch(err){}}function cf(){var k=["金額","セット","次へ","決定"];var a=document.querySelectorAll("a,button");for(var j=0;j<a.length;j++){var b=a[j].getBoundingClientRect();if(b.width>0&&b.height>0){for(var l=0;l<k.length;l++){if(a[j].textContent.indexOf(k[l])>=0){tp(a[j]);return;}}}}}function nx(){try{if(Date.now()-T>25000){fi(false);return;}var p="";if(document.getElementById("jyo"))p="V";else if(document.getElementById("race"))p="R";else if(document.getElementById("siki"))p="S";else if(document.getElementById("hou"))p="M";else{var c=(document.body.innerText||"");if(c.indexOf("会場")>=0||c.indexOf("開催")>=0)p="V";if(c.indexOf("レース")>=0||c.indexOf("回次")>=0)p="R";if(c.indexOf("式別")>=0)p="S";if(c.indexOf("方式")>=0)p="M";}if(i>=s.length){dg("Done");cf();fi(true);return;}var v=s[i];var f=false;var vs=[v];var n=parseInt(v);if(!isNaN(n)){if(i===1){vs=[String(n-1),(n-1<10?"0"+(n-1):String(n-1))];}else{vs=[v,String(n),(n<10?"0"+n:String(n)),String(n-1),(n-1<10?"0"+(n-1):String(n-1))];}}dg("S"+i+":"+v+" r:"+r+" p:"+p);var okP=(i===0&&(p==="V"||p===""||r>1))||(i===1&&(p==="R"||p==="V"||p===""||r>1))||(i===2&&(p==="S"||r>1))||(i===3&&(p==="M"||p==="S"||r>1))||(i>3);if(okP){if(i===0){var bs=document.querySelectorAll("a,button");for(var k2=0;k2<bs.length;k2++){var b2=bs[k2].getBoundingClientRect();if(b2.width<=4||b2.height<=4||bs[k2].classList.contains("disabled"))continue;var t=(bs[k2].innerText||bs[k2].textContent||"").trim();if(vn&&t.indexOf(vn)>=0){tp(bs[k2]);i++;r=0;setTimeout(nx,450);f=true;break;}}if(!f){for(var k=0;k<vs.length;k++){var es=document.querySelectorAll("a[data-value=\'"+vs[k]+"\'],button[data-value=\'"+vs[k]+"\']");for(var j=0;j<es.length;j++){var b=es[j].getBoundingClientRect();if(b.width>3&&b.height>3){tp(es[j]);i++;r=0;setTimeout(nx,450);f=true;break;}}if(f)break;}}}else{for(var k=0;k<vs.length;k++){var es=document.querySelectorAll("a[data-value=\'"+vs[k]+"\'],button[data-value=\'"+vs[k]+"\']");for(var j=0;j<es.length;j++){var b=es[j].getBoundingClientRect();if(b.width>3&&b.height>3){tp(es[j]);i++;r=0;setTimeout(nx,450);f=true;break;}}if(f)break;}if(!f){var bs=document.querySelectorAll("a,button");for(var k2=0;k2<bs.length;k2++){var b2=bs[k2].getBoundingClientRect();if(b2.width<=4||b2.height<=4)continue;var t=(bs[k2].innerText||bs[k2].textContent||"").trim();if(i===1&&(t===v+"R"||t===v+"レース"||t.indexOf(v+"R")>=0)){tp(bs[k2]);i++;r=0;setTimeout(nx,450);f=true;break;}if(i===2&&sn[v]&&t.indexOf(sn[v])>=0){tp(bs[k2]);i++;r=0;setTimeout(nx,450);f=true;break;}}}}}if(!f){r++;setTimeout(nx,200);}}catch(e){dg("E:"+e.message);fi(false);}}nx();})();'
    _smappy_part2_js_json = json.dumps(_smappy_part2_js)
    strategies2_json = json.dumps(strategies2_list, ensure_ascii=False)


    html_template = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="robots" content="noindex, nofollow">
    <title>Keiba AI Predictions</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Noto+Sans+JP:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f19;
            --primary: #4ade80;
            --primary-glow: rgba(74, 222, 128, 0.4);
            --card-bg: rgba(255, 255, 255, 0.03);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            -webkit-tap-highlight-color: transparent;
        }}

        body {{
            font-family: 'Outfit', 'Noto Sans JP', sans-serif;
            background: radial-gradient(circle at top right, #1a2333, #0b0f19);
            color: var(--text-main);
            min-height: 100vh;
            padding: 20px 16px;
            padding-bottom: 80px;
        }}

        header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 30px;
            animation: fadeInDown 0.8s ease;
        }}

        h1 {{
            font-size: 1.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #4ade80, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
        }}

        .controls-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
            animation: fadeIn 1s ease;
        }}

        select {{
            appearance: none;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            color: var(--text-main);
            padding: 14px 16px;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            outline: none;
            backdrop-filter: blur(10px);
            cursor: pointer;
            transition: all 0.3s ease;
        }}

        select:focus {{
            border-color: var(--primary);
            box-shadow: 0 0 15px var(--primary-glow);
        }}

        select option {{
            background-color: #0b0f19;
            color: #f8fafc;
        }}

        /* Next Race Button (Fixed Bottom-Right FAB) */
        .next-race-btn {{
            position: fixed;
            bottom: 24px;
            right: 20px;
            z-index: 900;
            background: linear-gradient(135deg, #10b981, #059669);
            color: #ffffff;
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 50px;
            padding: 10px 18px;
            font-size: 0.88rem;
            font-weight: 700;
            cursor: pointer;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            transition: all 0.25s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4), 0 2px 8px rgba(0, 0, 0, 0.4);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            letter-spacing: 0.02em;
        }}

        .next-race-btn:hover {{
            transform: translateY(-3px) scale(1.05);
            box-shadow: 0 10px 25px rgba(16, 185, 129, 0.6);
            background: linear-gradient(135deg, #34d399, #10b981);
        }}

        .next-race-btn:active {{
            transform: translateY(0) scale(0.96);
        }}

        .race-card.highlight-target {{
            animation: pulseGlow 1.5s ease-in-out 2 !important;
            border-color: #4ade80 !important;
        }}

        @keyframes pulseGlow {{
            0% {{ box-shadow: 0 0 0 rgba(74, 222, 128, 0); }}
            50% {{ box-shadow: 0 0 30px rgba(74, 222, 128, 0.8); }}
            100% {{ box-shadow: 0 0 0 rgba(74, 222, 128, 0); }}
        }}

        .race-list {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .race-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 18px 20px;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease, padding 0.25s ease;
            scroll-margin-top: 20px;
            position: relative;
        }}

        .race-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
            border-color: rgba(255, 255, 255, 0.15);
        }}

        .race-card.collapsed {{
            padding: 14px 20px;
        }}

        .race-card.collapsed .race-body {{
            display: none;
        }}

        .race-info-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            cursor: pointer;
            user-select: none;
            transition: margin-bottom 0.25s ease, padding-bottom 0.25s ease;
        }}

        .race-card:not(.collapsed) .race-info-header {{
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }}

        .race-card.collapsed .race-info-header {{
            margin-bottom: 0;
            padding-bottom: 0;
            border-bottom: none;
        }}

        .race-header-row-top {{
            display: flex;
            align-items: center;
            gap: 12px;
            flex-shrink: 0;
        }}

        .race-title-text {{
            font-size: 1.2rem;
            font-weight: 800;
            color: var(--primary);
            letter-spacing: 0.5px;
            white-space: nowrap;
        }}

        .race-ext-links {{
            display: inline-flex;
            gap: 8px;
            align-items: center;
        }}

        .race-link-btn {{
            text-decoration: none;
            font-size: 0.8rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 6px;
            transition: all 0.2s ease;
            white-space: nowrap;
            display: inline-flex;
            align-items: center;
        }}

        .race-link-btn.web-link {{
            color: var(--text-muted);
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .race-link-btn.web-link:hover {{
            color: #fff;
            background: rgba(255, 255, 255, 0.12);
        }}

        .race-link-btn.app-link {{
            color: var(--primary);
            background: rgba(74, 222, 128, 0.1);
            border: 1px solid var(--primary);
        }}

        .race-link-btn.app-link:hover {{
            background: rgba(74, 222, 128, 0.2);
        }}

        .race-header-row-bottom {{
            display: flex;
            align-items: center;
            gap: 10px;
            flex-shrink: 0;
        }}

        .race-badges-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .accordion-toggle-btn {{
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-muted);
            width: 32px;
            height: 32px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.25s ease;
            padding: 0;
        }}

        .race-info-header:hover .accordion-toggle-btn {{
            background: rgba(74, 222, 128, 0.15);
            border-color: var(--primary);
            color: var(--primary);
        }}

        .chevron-svg {{
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            transform: rotate(180deg);
        }}

        .race-card.collapsed .chevron-svg {{
            transform: rotate(0deg);
        }}

        @media (max-width: 768px) {{
            .race-card {{
                padding: 14px 14px;
                border-radius: 16px;
            }}

            .race-card.collapsed {{
                padding: 12px 14px;
            }}

            .race-info-header {{
                display: flex;
                flex-direction: column;
                align-items: stretch;
                gap: 10px;
            }}

            .race-card:not(.collapsed) .race-info-header {{
                margin-bottom: 12px;
                padding-bottom: 10px;
            }}

            .race-header-row-top {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                width: 100%;
                gap: 8px;
            }}

            .race-title-text {{
                font-size: 1.15rem;
                font-weight: 800;
                white-space: nowrap;
            }}

            .race-ext-links {{
                display: flex;
                gap: 6px;
            }}

            .race-link-btn {{
                font-size: 0.75rem;
                padding: 3px 7px;
            }}

            .race-header-row-bottom {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                width: 100%;
                gap: 8px;
            }}

            .race-badges-group {{
                display: flex;
                align-items: center;
                gap: 6px;
                flex-wrap: wrap;
            }}

            .pickup-badge, .pickup2-badge, .reason-badge {{
                padding: 4px 10px;
                font-size: 0.75rem;
                gap: 4px;
            }}

            .accordion-toggle-btn {{
                margin-left: auto;
                width: 30px;
                height: 30px;
                flex-shrink: 0;
            }}
        }}

        .race-id {{
            font-size: 1.2rem;
            font-weight: 700;
            letter-spacing: 1px;
        }}

        .race-meta {{
            font-size: 0.85rem;
            color: var(--text-muted);
            background: rgba(255, 255, 255, 0.05);
            padding: 4px 10px;
            border-radius: 20px;
        }}

        .horse-row {{
            display: flex;
            align-items: center;
            padding: 12px 0;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }}

        .horse-row:first-of-type {{
            border-top: none;
        }}

        .horse-num {{
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 1.1rem;
            margin-right: 14px;
            background: rgba(255, 255, 255, 0.1);
            flex-shrink: 0;
        }}

        /* Top 3 coloring */
        .rank-1 .horse-num {{ background: linear-gradient(135deg, #fbbf24, #f59e0b); color: #000; box-shadow: 0 0 10px rgba(251, 191, 36, 0.5); }}
        .rank-2 .horse-num {{ background: linear-gradient(135deg, #94a3b8, #64748b); color: #fff; box-shadow: 0 0 10px rgba(148, 163, 184, 0.5); }}
        .rank-3 .horse-num {{ background: linear-gradient(135deg, #b45309, #78350f); color: #fff; box-shadow: 0 0 10px rgba(180, 83, 9, 0.5); }}

        .horse-details {{
            flex-grow: 1;
        }}

        .horse-name {{
            font-size: 1.05rem;
            font-weight: 600;
            margin-bottom: 4px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .horse-score-bar-bg {{
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 3px;
            overflow: hidden;
        }}

        .horse-score-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #4ade80);
            border-radius: 3px;
            transition: width 1s cubic-bezier(0.1, 0.8, 0.2, 1);
        }}

        .horse-score-val {{
            font-size: 1.1rem;
            font-weight: 800;
            color: var(--primary);
            min-width: 60px;
            text-align: right;
            margin-left: 10px;
            padding-right: 12px;
        }}

        @keyframes fadeInDown {{
            from {{ opacity: 0; transform: translateY(-20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        @keyframes slideUp {{
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        #auth-overlay {{
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: radial-gradient(circle at top right, #1a2333, #0b0f19);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .auth-box {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 30px;
            text-align: center;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            width: 90%;
            max-width: 340px;
        }}

        .auth-box h2 {{ margin-bottom: 20px; font-size: 1.4rem; }}
        .auth-box input {{
            width: 100%;
            padding: 14px;
            margin-bottom: 15px;
            border-radius: 8px;
            border: 1px solid var(--card-border);
            background: rgba(255,255,255,0.05);
            color: #fff;
            outline: none;
            font-size: 1rem;
        }}
        .auth-box input:focus {{ border-color: var(--primary); }}
        .auth-box button {{
            width: 100%;
            padding: 14px;
            border-radius: 8px;
            border: none;
            background: var(--primary);
            color: #000;
            font-weight: 800;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        .auth-box button:hover {{
            box-shadow: 0 0 15px var(--primary-glow);
        }}
        #login-error {{ color: #ef4444; margin-top: 10px; font-size: 0.9rem; display: none; font-weight: 600; }}

        /* AI Recommendation Modal Styles */
        .pickup-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: linear-gradient(135deg, rgba(74, 222, 128, 0.25), rgba(59, 130, 246, 0.25));
            border: 1px solid rgba(74, 222, 128, 0.5);
            color: #4ade80;
            padding: 5px 14px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 900;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4), 0 0 15px rgba(74, 222, 128, 0.2);
            letter-spacing: 0.05em;
            flex-shrink: 0;
        }}

        .pickup-badge:hover {{
            background: linear-gradient(135deg, rgba(74, 222, 128, 0.4), rgba(59, 130, 246, 0.4));
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5), 0 0 30px rgba(74, 222, 128, 0.5);
            transform: translateY(-2px) scale(1.06);
        }}

        /* AI Recommendation Modal Styles (Strategy 2 / PICKUP 2) */
        .pickup2-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.3), rgba(245, 158, 11, 0.35));
            border: 1px solid rgba(249, 115, 22, 0.6);
            color: #fb923c;
            padding: 5px 14px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 900;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4), 0 0 15px rgba(249, 115, 22, 0.25);
            letter-spacing: 0.05em;
            flex-shrink: 0;
        }}

        .pickup2-badge:hover {{
            background: linear-gradient(135deg, rgba(251, 146, 60, 0.45), rgba(245, 158, 11, 0.55));
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5), 0 0 30px rgba(249, 115, 22, 0.5);
            transform: translateY(-2px) scale(1.06);
        }}

        /* Recommendation Modal Tabs */
        .rec-tab-group {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 22px;
            flex-wrap: wrap;
        }}

        .rec-tab-btn {{
            padding: 8px 18px;
            font-size: 0.85rem;
            font-weight: 800;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(255, 255, 255, 0.04);
            color: var(--text-muted);
            cursor: pointer;
            transition: all 0.25s ease;
            letter-spacing: 0.03em;
        }}

        .rec-tab-btn:hover {{
            background: rgba(255, 255, 255, 0.08);
            color: #fff;
        }}

        .rec-tab-btn.active {{
            background: linear-gradient(135deg, rgba(74, 222, 128, 0.25), rgba(59, 130, 246, 0.25));
            border-color: #4ade80;
            color: #fff;
            box-shadow: 0 0 16px rgba(74, 222, 128, 0.35);
        }}

        .rec-tab-btn.strat2.active {{
            background: linear-gradient(135deg, rgba(249, 115, 22, 0.35), rgba(245, 158, 11, 0.4));
            border-color: #fb923c;
            color: #fff;
            box-shadow: 0 0 16px rgba(249, 115, 22, 0.4);
        }}

        /* Evaluation Reasons Badge Styles */
        .reason-badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: linear-gradient(135deg, rgba(168, 85, 247, 0.25), rgba(59, 130, 246, 0.25));
            border: 1px solid rgba(168, 85, 247, 0.5);
            color: #c084fc;
            padding: 5px 14px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 900;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4), 0 0 15px rgba(168, 85, 247, 0.2);
            letter-spacing: 0.05em;
            flex-shrink: 0;
        }}

        .reason-badge:hover {{
            background: linear-gradient(135deg, rgba(168, 85, 247, 0.4), rgba(59, 130, 246, 0.4));
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5), 0 0 30px rgba(168, 85, 247, 0.5);
            transform: translateY(-2px) scale(1.06);
        }}

        .reason-card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 14px;
            transition: all 0.2s ease;
        }}
        .reason-card:hover {{
            background: rgba(255, 255, 255, 0.05);
            border-color: rgba(168, 85, 247, 0.3);
        }}
        .reason-horse-title {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.1rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 10px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }}
        .reason-item {{
            display: flex;
            align-items: flex-start;
            gap: 8px;
            font-size: 0.85rem;
            margin-bottom: 8px;
            color: #cbd5e1;
            line-height: 1.5;
        }}
        .reason-item:last-child {{
            margin-bottom: 0;
        }}
        .reason-tag {{
            background: rgba(168, 85, 247, 0.2);
            color: #c084fc;
            border: 1px solid rgba(168, 85, 247, 0.4);
            border-radius: 6px;
            padding: 1px 7px;
            font-size: 0.72rem;
            font-weight: 700;
            flex-shrink: 0;
            margin-top: 2px;
        }}

        /* Modal Overlay */
        #recommend-modal, #reasons-modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.85);
            z-index: 1000;
            backdrop-filter: blur(10px);
            align-items: center;
            justify-content: center;
            animation: fadeIn 0.3s ease;
        }}

        .modal-content {{
            background: linear-gradient(165deg, #1e293b, #0f172a);
            width: 90%;
            max-width: 600px;
            max-height: 85vh;
            border-radius: 24px;
            border: 1px solid rgba(74, 222, 128, 0.3);
            position: relative;
            padding: 30px;
            overflow-y: auto;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8), 0 0 40px rgba(74, 222, 128, 0.1);
            animation: modalPop 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        }}

        @keyframes modalPop {{
            from {{ transform: scale(0.9); opacity: 0; }}
            to {{ transform: scale(1); opacity: 1; }}
        }}

        .modal-close {{
            position: absolute;
            top: 20px;
            right: 20px;
            font-size: 1.5rem;
            color: var(--text-muted);
            cursor: pointer;
            transition: color 0.2s;
        }}
        .modal-close:hover {{ color: #ffffff; }}

        .bet-eyes-box {{
            background: rgba(0, 0, 0, 0.4);
            padding: 20px;
            border-radius: 16px;
            margin: 15px 0;
            border: 2px solid rgba(74, 222, 128, 0.3);
            text-align: center;
            box-shadow: inset 0 0 20px rgba(74, 222, 128, 0.05);
            position: relative;
        }}

        .bet-eyes-text {{
            font-size: 1.8rem;
            font-weight: 900;
            color: #4ade80;
            font-family: 'Space Mono', monospace;
            text-shadow: 0 0 15px rgba(74, 222, 128, 0.4);
            letter-spacing: 0.1em;
            white-space: normal;
            word-break: break-word;
            line-height: 1.35;
        }}

        .strategy-item-modal {{
            margin-bottom: 25px;
            padding-bottom: 20px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .strategy-item-modal:last-child {{ border-bottom: none; }}
@keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

        .stat-value.positive {{ color: #4ade80; }}

        /* Highlighting Styles */
        .is-jiku {{
            background: rgba(74, 222, 128, 0.12) !important;
            border-left: 5px solid #4ade80 !important;
            box-shadow: inset 0 0 20px rgba(74, 222, 128, 0.08);
        }}
        .is-jiku .horse-name {{
            color: #4ade80;
            font-weight: 800;
            font-size: 1.15rem;
        }}
        .is-partner {{
            background: rgba(254, 243, 199, 0.08) !important;
            border-left: 5px solid #fbbf24 !important;
        }}
        .is-partner .horse-name {{
            color: #fde68a;
            font-weight: 600;
        }}
        .is-jiku-strat2 {{
            background: rgba(249, 115, 22, 0.14) !important;
            border-left: 5px solid #f97316 !important;
            box-shadow: inset 0 0 20px rgba(249, 115, 22, 0.1);
        }}
        .is-jiku-strat2 .horse-name {{
            color: #fb923c;
            font-weight: 800;
            font-size: 1.15rem;
        }}
        .is-partner-strat2 {{
            background: rgba(251, 191, 36, 0.08) !important;
            border-left: 5px solid #f59e0b !important;
        }}
        .is-partner-strat2 .horse-name {{
            color: #fcd34d;
            font-weight: 600;
        }}

        .race-card {{
            position: relative;
            overflow: hidden; /* Ensure highlighting doesn't overflow rounded corners */
        }}
        .smappy-btn {{
            background: linear-gradient(135deg, #10b981, #059669);
            border: none;
            color: #fff;
            padding: 8px 14px;
            border-radius: 10px;
            font-size: 0.78rem;
            font-weight: 800;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }}
        .smappy-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(16, 185, 129, 0.45);
        }}
        .ipat-btn {{
            background: linear-gradient(135deg, #6366f1, #4f46e5);
            border: none;
            color: #fff;
            padding: 8px 14px;
            border-radius: 10px;
            font-size: 0.78rem;
            font-weight: 800;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }}
        .ipat-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(99, 102, 241, 0.45);
        }}
        .umaca-btn {{
            background: linear-gradient(135deg, #a855f7, #7c3aed);
            border: none;
            color: #fff;
            padding: 8px 14px;
            border-radius: 10px;
            font-size: 0.78rem;
            font-weight: 800;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 12px rgba(168, 85, 247, 0.3);
        }}
        .umaca-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(168, 85, 247, 0.45);
        }}
        .smappy-popup {{
            background: #1e293b;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 16px;
            margin-top: 15px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            animation: slideUp 0.4s ease;
        }}
        .smappy-tabs {{
            display: flex;
            gap: 4px;
            background: rgba(0, 0, 0, 0.2);
            padding: 4px;
            border-radius: 10px;
            margin-bottom: 12px;
        }}
        .smappy-tab {{
            flex: 1;
            padding: 6px;
            text-align: center;
            font-size: 0.75rem;
            font-weight: 700;
            cursor: pointer;
            border-radius: 8px;
            color: var(--text-muted);
        }}
        .smappy-tab.active {{
            background: #334155;
            color: #fff;
        }}
        /* --- PICKUP Summary Panel --- */
        .pickup-summary-panel {{
            background: linear-gradient(165deg, #1e293b, #0f172a);
            border: 1px solid rgba(74, 222, 128, 0.25);
            border-radius: 16px;
            margin-bottom: 18px;
            overflow: hidden;
        }}
        .pickup-summary-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 14px 16px;
            cursor: pointer;
            user-select: none;
        }}
        .pickup-summary-title {{
            font-weight: 900;
            font-size: 0.95rem;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .pickup-summary-count-badge {{
            background: rgba(74, 222, 128, 0.15);
            color: #4ade80;
            border: 1px solid rgba(74, 222, 128, 0.35);
            border-radius: 999px;
            padding: 1px 9px;
            font-size: 0.75rem;
            font-weight: 800;
        }}
        #pickup-summary-chevron {{
            transition: transform 0.25s ease;
            color: #94a3b8;
        }}
        .pickup-summary-body {{
            padding: 0 16px 16px 16px;
        }}
        .pickup-summary-toolbar {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 12px;
        }}
        .pickup-summary-toolbar button.btn-toggle-all {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.12);
            color: #cbd5e1;
            padding: 7px 12px;
            border-radius: 8px;
            font-size: 0.72rem;
            font-weight: 700;
            cursor: pointer;
        }}
        .btn-bulk-smappy {{
            margin-left: auto;
            background: linear-gradient(135deg, #10b981, #059669);
            border: none;
            color: #fff;
            padding: 9px 16px;
            border-radius: 10px;
            font-size: 0.78rem;
            font-weight: 800;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }}
        .pickup-summary-list {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            max-height: 420px;
            overflow-y: auto;
        }}
        .pickup-summary-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 10px;
            padding: 10px 12px;
            cursor: pointer;
        }}
        .pickup-summary-item input.pickup-check {{
            flex-shrink: 0;
            width: 18px;
            height: 18px;
            accent-color: #10b981;
            cursor: pointer;
        }}
        .pickup-summary-item-main {{
            flex: 1;
            min-width: 0;
        }}
        .pickup-summary-item-top {{
            display: flex;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 4px;
        }}
        .pickup-summary-tag {{
            font-size: 0.65rem;
            font-weight: 900;
            padding: 1px 7px;
            border-radius: 5px;
            border: 1px solid;
            letter-spacing: 0.04em;
        }}
        .pickup-summary-race {{
            font-size: 0.78rem;
            font-weight: 800;
            color: #e2e8f0;
        }}
        .pickup-summary-type {{
            font-size: 0.72rem;
            font-weight: 700;
            color: #94a3b8;
        }}
        .pickup-summary-eyes {{
            font-size: 0.88rem;
            font-weight: 800;
            color: #fff;
            overflow-x: auto;
            white-space: nowrap;
        }}
        /* --- Smappy Bulk-Vote Queue Modal --- */
        .smappy-queue-modal-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.85);
            z-index: 1100;
            backdrop-filter: blur(10px);
            display: flex;
            align-items: center;
            justify-content: center;
            animation: fadeIn 0.3s ease;
            padding: 16px;
        }}
        .smappy-queue-popup.smappy-popup {{
            margin-top: 0;
            width: 100%;
            max-width: 420px;
            max-height: 90vh;
            overflow-y: auto;
        }}
        .smappy-queue-progress {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 10px;
            font-size: 0.72rem;
            font-weight: 800;
            color: #cbd5e1;
        }}
        .smappy-queue-nav {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.15);
            color: #fff;
            padding: 6px 10px;
            border-radius: 8px;
            font-size: 0.72rem;
            font-weight: 800;
            cursor: pointer;
            flex-shrink: 0;
        }}
        .smappy-queue-nav:disabled {{
            opacity: 0.35;
            cursor: default;
        }}
        .smappy-queue-eyes {{
            text-align: center;
            font-size: 1.1rem;
            font-weight: 900;
            color: #4ade80;
            background: rgba(74, 222, 128, 0.08);
            border: 1px solid rgba(74, 222, 128, 0.25);
            border-radius: 10px;
            padding: 8px;
            margin-bottom: 12px;
        }}
        .step-box {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }}
        .step-title {{
            font-size: 0.65rem;
            text-transform: uppercase;
            font-weight: 800;
            color: #4ade80;
            margin-bottom: 6px;
            display: block;
        }}
        .step-desc {{
            font-size: 0.72rem;
            color: #cbd5e1;
            line-height: 1.4;
        }}

        /* Offline Status Banner */
        #offline-banner {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background: linear-gradient(135deg, #f59e0b, #d97706);
            color: #ffffff;
            text-align: center;
            padding: 8px 12px;
            font-size: 0.85rem;
            font-weight: 700;
            z-index: 100000;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.4);
            letter-spacing: 0.03em;
        }}
    </style>
</head>
<body>
    <div id="offline-banner">⚡ オフライン表示中 (1日以内の保存データから自動読み込み中)</div>
    <div id="auth-overlay">
        <div class="auth-box">
            <h2>Keiba AI Login</h2>
            <input type="password" id="auth-pw" placeholder="Password" onkeydown="if(event.key==='Enter') checkAuth()" />
            <button onclick="checkAuth()">Enter</button>
            <div id="login-error">Invalid credentials</div>
        </div>
    </div>

    <div id="app-content" style="display: none;">
        <header>
            <h1>Keiba AI</h1>
            <div style="font-size: 0.8rem; color: var(--text-muted); font-weight: 600;">STATIC HOSTED (v4.0)</div>
        </header>

        <div class="controls-grid">
            <select id="filter-date" onchange="onDateChange()">
                <!-- Options populated by JS -->
            </select>
            <select id="filter-place" onchange="renderRaces()">
                <option value="ALL">All Places</option>
            </select>
            <select id="filter-round" onchange="renderRaces()">
                <option value="ALL">All Races</option>
            </select>
            <select id="filter-pickup" onchange="renderRaces()">
                <option value="ALL">All Status</option>
                <option value="PICKUP1">🎯 PICKUP 1</option>
                <option value="PICKUP2">🚀 PICKUP 2</option>
                <option value="ANY_PICKUP">✨ PICKUP (いずれか)</option>
            </select>
            <select id="sort-select" onchange="renderRaces()">
                <option value="score">Sort by AI Score</option>
                <option value="odds">Sort by Odds</option>
                <option value="horse_number">Sort by Horse Number</option>
            </select>
            <select id="model-select" onchange="renderRaces()" style="display: none;">
                <option value="Ensemble">Ensemble</option>
                <option value="AutoGluon">AutoGluon</option>
                <option value="LightGBM">LightGBM</option>
                <option value="XGBoost">XGBoost</option>
                <option value="CatBoost">CatBoost</option>
                <option value="LSTM">LSTM</option>
                <option value="RandomForest">RandomForest</option>
                <option value="DecisionTree">DecisionTree</option>
                <option value="Transformer">Transformer</option>
                <option value="TabNet">TabNet</option>
            </select>
        </div>

        <div class="pickup-summary-panel">
            <div class="pickup-summary-header" onclick="togglePickupSummaryPanel()">
                <div class="pickup-summary-title">
                    📌 選択日のPICKUP馬券一覧
                    <span id="pickup-summary-count" class="pickup-summary-count-badge">0</span>
                </div>
                <svg id="pickup-summary-chevron" viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
            </div>
            <div id="pickup-summary-body" class="pickup-summary-body">
                <div class="pickup-summary-toolbar">
                    <button type="button" class="btn-toggle-all" onclick="togglePickupCheckAll(true)">☑ 全選択</button>
                    <button type="button" class="btn-toggle-all" onclick="togglePickupCheckAll(false)">☐ 全解除</button>
                    <span style="align-self:center; font-size:0.72rem; color:var(--text-muted); font-weight:800;">選択中 <span id="pickup-selected-count">0</span> 件 →</span>
                    <button type="button" class="btn-bulk-smappy" onclick="startBulkVote('smappy')">📌 スマッピー投票</button>
                    <button type="button" class="btn-bulk-smappy" style="background: linear-gradient(135deg, #10b981, #059669);" onclick="startBulkVote('ipat')">🟢 即PAT投票</button>
                    <button type="button" class="btn-bulk-smappy" style="background: linear-gradient(135deg, #a855f7, #7c3aed); box-shadow: 0 4px 12px rgba(168, 85, 247, 0.3);" onclick="startBulkVote('umaca')">🟣 UMACA投票</button>
                    <span style="align-self:center; font-size:0.72rem; color:var(--text-muted); font-weight:800; margin-left:4px;">1点:</span>
                    <input type="number" id="global-unit-amount" value="100" step="100" min="100" title="全馬券の1点金額" style="width:68px; padding:4px 6px; background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.15); color:#fff; border-radius:6px; font-size:0.75rem; text-align:right; font-weight:800;" oninput="window._globalUnitAmount = parseInt(this.value)||100;">
                    <span style="align-self:center; font-size:0.72rem; color:var(--text-muted);">円</span>
                </div>
                <div id="pickup-summary-list" class="pickup-summary-list"></div>
            </div>
        </div>

        <div style="display: flex; justify-content: flex-end; gap: 8px; margin-top: -12px; margin-bottom: 20px;">
            <button type="button" class="btn-toggle-all" onclick="toggleAllRaces(true)">▲ 全て縮小</button>
            <button type="button" class="btn-toggle-all" onclick="toggleAllRaces(false)">▼ 全て展開</button>
        </div>

        <div id="races-container" class="race-list"></div>

        <!-- Floating Next Race Button -->
        <button id="btn-next-race" onclick="scrollToNextRace()" class="next-race-btn" title="次の発走レースへ移動">
            🏇 次の発走
        </button>
    </div>

    <!-- Recommendation Modal -->
    <div id="recommend-modal" onclick="if(event.target===this) closeRecommendation()">
        <div class="modal-content">
            <span class="modal-close" onclick="closeRecommendation()">&times;</span>
            <div id="modal-body"></div>
        </div>
    </div>

    <!-- Evaluation Reasons Modal -->
    <div id="reasons-modal" onclick="if(event.target===this) closeReasons()">
        <div class="modal-content" style="border-color: rgba(168, 85, 247, 0.4); box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8), 0 0 40px rgba(168, 85, 247, 0.15);">
            <span class="modal-close" onclick="closeReasons()">&times;</span>
            <div id="reasons-modal-body"></div>
        </div>
    </div>

    <script>
        console.log("[DEBUG] Keiba AI Script Initializing...");
        let currentData = {{}};
        window.strategies2 = {strategies2_json};

        // Service Worker の登録とオフライン状態の監視
        if ('serviceWorker' in navigator) {{
            window.addEventListener('load', () => {{
                navigator.serviceWorker.register('./sw.js?v=9')
                    .then(reg => console.log('[SW] ServiceWorker registered with scope:', reg.scope))
                    .catch(err => console.warn('[SW] ServiceWorker registration failed:', err));
            }});
        }}

        function updateOnlineStatus() {{
            const banner = document.getElementById('offline-banner');
            if (!banner) return;
            if (!navigator.onLine) {{
                banner.style.display = 'block';
            }} else {{
                banner.style.display = 'none';
            }}
        }}

        window.addEventListener('online', updateOnlineStatus);
        window.addEventListener('offline', updateOnlineStatus);

        let raceCollapseState = {{}};
        try {{
            const saved = localStorage.getItem('keiba_collapsed_races');
            if (saved) {{
                raceCollapseState = JSON.parse(saved);
            }}
        }} catch (e) {{
            console.warn("Failed to load race collapse state:", e);
        }}

        function saveRaceCollapseState() {{
            try {{
                localStorage.setItem('keiba_collapsed_races', JSON.stringify(raceCollapseState));
            }} catch (e) {{
                console.warn("Failed to save race collapse state:", e);
            }}
        }}

        function toggleRaceCard(raceId) {{
            const card = document.getElementById('race-card-' + raceId);
            if (card) {{
                const isCollapsed = card.classList.toggle('collapsed');
                raceCollapseState[raceId] = isCollapsed;
                saveRaceCollapseState();
            }}
        }}

        function toggleAllRaces(collapse) {{
            document.querySelectorAll('.race-card').forEach(card => {{
                const rid = card.dataset.raceId || card.id.replace('race-card-', '');
                if (collapse) {{
                    card.classList.add('collapsed');
                    if (rid) raceCollapseState[rid] = true;
                }} else {{
                    card.classList.remove('collapsed');
                    if (rid) raceCollapseState[rid] = false;
                }}
            }});
            saveRaceCollapseState();
        }}

        function scrollToNextRace() {{
            const cards = Array.from(document.querySelectorAll('.race-card'));
            if (cards.length === 0) return;

            const now = new Date();
            const currentHHMM = String(now.getHours()).padStart(2, '0') + ':' + String(now.getMinutes()).padStart(2, '0');

            let targetCard = null;
            for (const card of cards) {{
                const startTime = card.dataset.startTime;
                if (startTime && startTime >= currentHHMM) {{
                    targetCard = card;
                    break;
                }}
            }}

            // 全てのレースが現在時刻より前の場合は最後のレースへ
            if (!targetCard && cards.length > 0) {{
                targetCard = cards[cards.length - 1];
            }}

            if (targetCard) {{
                if (targetCard.classList.contains('collapsed')) {{
                    targetCard.classList.remove('collapsed');
                    const rid = targetCard.dataset.raceId || targetCard.id.replace('race-card-', '');
                    if (rid) raceCollapseState[rid] = false;
                    saveRaceCollapseState();
                }}
                targetCard.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                targetCard.classList.add('highlight-target');
                setTimeout(() => {{
                    targetCard.classList.remove('highlight-target');
                }}, 3000);
            }}
        }}

        async function checkAuth() {{
            console.log("[DEBUG] checkAuth called");
            const input = document.getElementById('auth-pw');
            const pw = (input ? input.value : "").trim();
            const errDiv = document.getElementById('login-error');
            
            if (pw === 'tohshin20') {{
                console.log("[DEBUG] Password correct, initializing app...");
                try {{
                    localStorage.setItem('keiba_auth_time', new Date().getTime());
                }} catch (e) {{
                    console.warn("localStorage is not available:", e);
                }}
                
                const isLocal = window.location.protocol === 'file:';
                if (isLocal) {{
                    if (errDiv) {{
                        errDiv.style.display = 'block';
                        errDiv.style.color = '#fbbf24';
                        errDiv.style.fontSize = '0.8rem';
                        errDiv.style.lineHeight = '1.5';
                        errDiv.style.textAlign = 'left';
                        errDiv.style.marginTop = '12px';
                        errDiv.innerHTML = `
                            パスワードは正しいですが、ファイルを直接開いているため（file://）ブラウザのセキュリティでデータ読み込みがブロックされます。<br><br>
                            <strong>【解決手順】</strong><br>
                            1. ターミナルで <code>python -m http.server 8000</code> を実行<br>
                            2. ブラウザで <a href="http://localhost:8000" target="_blank" style="color: #4ade80; text-decoration: underline;">http://localhost:8000</a> を開く
                        `;
                    }}
                    return;
                }}
                
                // オーバーレイを完全に削除（Safariのdisplay:flex優先バグ等を回避）
                const overlay = document.getElementById('auth-overlay');
                if (overlay) overlay.remove();
                
                document.getElementById('app-content').style.display = 'block';
                loadData();
            }} else {{
                if (errDiv) {{
                    errDiv.style.display = 'block';
                    errDiv.style.color = '#ef4444';
                    errDiv.innerText = 'パスワードが違います (Invalid password)';
                }}
            }}
        }}

        // ページ読み込み時に認証チェック
        window.onload = function() {{
            updateOnlineStatus();
            let isAuthenticated = false;
            try {{
                const authTime = localStorage.getItem('keiba_auth_time');
                if (authTime) {{
                    const now = new Date().getTime();
                    const diffHours = (now - parseInt(authTime)) / (1000 * 60 * 60);
                    if (diffHours < 24) {{
                        isAuthenticated = true;
                    }}
                }}
            }} catch (e) {{
                console.warn("localStorage is not available for auth check:", e);
            }}
            
            const isLocal = window.location.protocol === 'file:';
            if (isAuthenticated && !isLocal) {{
                const overlay = document.getElementById('auth-overlay');
                if (overlay) overlay.remove();
                document.getElementById('app-content').style.display = 'block';
                loadData();
            }} else {{
                // 認証が必要な場合
                const overlay = document.getElementById('auth-overlay');
                if (overlay) overlay.style.display = 'flex';
                if (isLocal) {{
                    const errDiv = document.getElementById('login-error');
                    if (errDiv) {{
                        errDiv.style.display = 'block';
                        errDiv.style.color = '#fbbf24';
                        errDiv.style.fontSize = '0.75rem';
                        errDiv.style.lineHeight = '1.4';
                        errDiv.style.marginTop = '10px';
                        errDiv.innerHTML = `⚠️ file:// で開いているため画面上のログイン後はローカルサーバー (http://localhost:8000) が必要です`;
                    }}
                }}
            }}
        }};

        async function loadData() {{
            const container = document.getElementById('races-container');
            container.innerHTML = '<div style="text-align:center; padding: 40px;"><p>Loading metadata...</p></div>';

            try {{
                // 1. メタデータ (日付リスト) を取得
                const metaRes = await fetch('jsons/meta.json?t=' + new Date().getTime());
                if (!metaRes.ok) throw new Error('Metadata fetch failed');
                const metaData = await metaRes.json();
                
                // フィルタの初期化 (日付リストをセット)
                initDateFilter(metaData.dates, metaData.latest);

                // 2. 最新日付または選択された日付のデータを読み込む
                await fetchDailyData(metaData.latest);
                
                // 単勝オッズデータの読み込み (これは共通)
                const tanshoRes = await fetch('jsons/tansho_data.json?t=' + new Date().getTime());
                if (tanshoRes.ok) {{
                    window.tanshoData = await tanshoRes.json();
                }} else {{
                    console.warn("tansho_data.json not found, using empty data.");
                    window.tanshoData = {{}};
                }}
                
                renderRaces();
            }} catch (error) {{
                console.error("Fetch error details: ", error);
                const isLocal = window.location.protocol === 'file:';
                container.innerHTML = `
                    <div style="text-align:center; padding: 40px; color: #ef4444;">
                        <p style="font-weight: 800; font-size: 1.2rem; margin-bottom: 15px;">Data Load Error</p>
                        <p style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 20px;">
                            ${{error.message}}<br>
                            ${{isLocal ? '【重要】ローカルファイルとして直接開いているため、ブラウザのセキュリティ制限（CORS）により読み込みがブロックされています。' : ''}}
                        </p>
                        <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; text-align: left; display: inline-block;">
                            <p style="font-size: 0.8rem; font-weight: 800; margin-bottom: 8px;">解決方法:</p>
                            <ol style="font-size: 0.8rem; color: #f8fafc; padding-left: 20px;">
                                <li>VSCode の Live Server 拡張機能を使用する</li>
                                <li>ターミナルで python -m http.server を実行し、localhost:8000 にアクセスする</li>
                                <li>serve.bat を作成して実行する</li>
                            </ol>
                        </div>
                    </div>
                `;
            }}
        }}

        function initDateFilter(dates, latest) {{
            const dp = document.getElementById('filter-date');
            dp.innerHTML = ''; // クリア
            
            dates.forEach(d => {{
                const opt = document.createElement('option');
                opt.value = d; opt.innerText = d;
                dp.appendChild(opt);
            }});
            
            if (latest) {{
                dp.value = latest;
            }}
            
            // Round フィルタの初期化 (1R-12R)
            const rp = document.getElementById('filter-round');
            rp.innerHTML = '<option value="ALL">All Races</option>';
            for (let i = 1; i <= 12; i++) {{
                const opt = document.createElement('option');
                opt.value = String(i);
                opt.innerText = i + "R";
                rp.appendChild(opt);
            }}
        }}

        async function fetchDailyData(date) {{
            const container = document.getElementById('races-container');
            container.innerHTML = '<div style="text-align:center; padding: 40px;"><p>Loading race data for ' + date + '...</p></div>';
            
            try {{
                const dataRes = await fetch(`jsons/data_${{date}}.json?t=` + new Date().getTime());
                if (!dataRes.ok) throw new Error('Failed to fetch daily data for ' + date);
                currentData = await dataRes.json();
                
                updatePlacesForDate();
            }} catch (error) {{
                console.error("Daily data fetch error:", error);
                throw error;
            }}
        }}

        async function onDateChange() {{
            const fDate = document.getElementById('filter-date').value;
            try {{
                await fetchDailyData(fDate);
                renderRaces();
            }} catch (e) {{
                alert("データの読み込みに失敗しました: " + fDate);
            }}
        }}

        function updatePlacesForDate() {{
            const fDate = document.getElementById('filter-date').value;
            const pp = document.getElementById('filter-place');
            const prevValue = pp.value;
            pp.innerHTML = '<option value="ALL">All Places</option>';
            
            const placesForDate = [];
            for (const [rid, rdata] of Object.entries(currentData)) {{
                if (rdata.date === fDate && !placesForDate.includes(rdata.place)) {{
                    placesForDate.push(rdata.place);
                }}
            }}
            
            placesForDate.sort().forEach(p => {{
                const opt = document.createElement('option');
                opt.value = p;
                opt.innerText = p;
                pp.appendChild(opt);
            }});
            
            // Try to restore previous selection if valid
            if (placesForDate.includes(prevValue)) {{
                pp.value = prevValue;
            }}
        }}

        function renderRaces() {{
            // 既存カードの折りたたみ状態を同期
            document.querySelectorAll('.race-card').forEach(c => {{
                const rid = c.dataset.raceId || c.id.replace('race-card-', '');
                if (rid) {{
                    raceCollapseState[rid] = c.classList.contains('collapsed');
                }}
            }});
            saveRaceCollapseState();

            const container = document.getElementById('races-container');
            container.innerHTML = '';
            
            const sortBy = document.getElementById('sort-select').value;
            const sortModel = document.getElementById('model-select').value;
            const mSelect = document.getElementById('model-select');
            const fDate = document.getElementById('filter-date').value;
            const fPlace = document.getElementById('filter-place').value;
            const fRound = document.getElementById('filter-round').value;
            const fPickup = document.getElementById('filter-pickup') ? document.getElementById('filter-pickup').value : 'ALL';

            // Show model selector only when sorting by score
            if (sortBy === 'score') {{
                mSelect.style.display = 'inline-block';
            }} else {{
                mSelect.style.display = 'none';
            }}

            // Sort races by start_time (発走時刻), then round, then place
            const sortedRaces = Object.values(currentData).sort((a, b) => {{
                const timeA = a.start_time || "00:00";
                const timeB = b.start_time || "00:00";
                if (timeA !== timeB) return timeA.localeCompare(timeB);
                const roundA = parseInt(a.round) || 0;
                const roundB = parseInt(b.round) || 0;
                if (roundA !== roundB) return roundA - roundB;
                return (a.place || "").localeCompare(b.place || "");
            }});

            for (const raceData of sortedRaces) {{
                const raceId = raceData.race_id;
                
                // Filtering
                if (fDate !== 'ALL' && raceData.date !== fDate) continue;
                if (fPlace !== 'ALL' && raceData.place !== fPlace) continue;
                if (fRound !== 'ALL' && String(raceData.round) !== String(fRound)) continue;

                // --- 2. Strategy Highlighting & PICKUP Calculation (Kelly2 High-Confidence Logic) ---
                const kellyResult = evaluateKelly2Strategies(raceData, raceId);
                const jikuSet = kellyResult.jikuSet;
                const partnerSet = kellyResult.partnerSet;
                const hasPickup1 = kellyResult.validStrategies && kellyResult.validStrategies.length > 0;

                // --- Strategy 2 (Common Purchase Strategy / PICKUP 2 Logic) ---
                const strat2Result = evaluateStrategy2(raceData, raceId);
                const hasPickup2 = strat2Result.validStrategies && strat2Result.validStrategies.length > 0;

                // Filter by PICKUP status
                if (fPickup === 'PICKUP1' && !hasPickup1) continue;
                if (fPickup === 'PICKUP2' && !hasPickup2) continue;
                if (fPickup === 'ANY_PICKUP' && !hasPickup1 && !hasPickup2) continue;

                // Sort horses
                let sortedHorses = [...raceData.horses];
                
                const getWinOdds = (h) => {{
                    const rIdShort = String(raceId).length === 12 ? String(raceId).substring(2) : raceId;
                    const rOdds = window.tanshoData ? (window.tanshoData[raceId] || window.tanshoData[rIdShort]) : null;
                    if (rOdds) {{
                        const hO = rOdds.find(o => o[0] == h.horse_number);
                        if (hO) return parseFloat(hO[1]) || 999;
                    }}
                    return 999;
                }};

                if (sortBy === 'score') {{
                    const mKey = (sortModel === 'Ensemble') ? 'Ensemble' : sortModel + '_raw';
                    sortedHorses.sort((a, b) => (parseFloat(b[mKey]) || 0)  - (parseFloat(a[mKey]) || 0));
                }} else if (sortBy === 'odds') {{
                    sortedHorses.sort((a, b) => getWinOdds(a) - getWinOdds(b));
                }} else {{
                    sortedHorses.sort((a, b) => (parseInt(a.horse_number) || 0)  - (parseInt(b.horse_number) || 0));
                }}

                // Decide main score vs sub scores display key for the whole race
                const mainModelKey = (sortBy === 'score') ? ((sortModel === 'Ensemble') ? 'Ensemble' : sortModel + '_raw') : 'Ensemble';

                // --- 1. Calculate Z-Scores for Each Model Per Race ---
                const scoreModels = ['LightGBM_raw', 'XGBoost_raw', 'CatBoost_raw', 'LSTM_raw', 'RandomForest_raw', 'DecisionTree_raw', 'Transformer_raw', 'TabNet_raw', 'Ensemble', 'AutoGluon_raw'];
                const raceStats = {{}};
                scoreModels.forEach(m => {{
                    const vals = raceData.horses.map(h => parseFloat(h[m]) || 0);
                    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                    const variance = vals.map(v => Math.pow(v - mean, 2)).reduce((a, b) => a + b, 0) / Math.max(1, vals.length - 1);
                    const std = Math.sqrt(variance) || 1.0;
                    raceStats[m] = {{ mean, std }};
                }});

                // Helper to get Z-Score
                const getZ = (h, m) => {{
                    const stats = raceStats[m];
                    const val = parseFloat(h[m]) || 0;
                    return (val - stats.mean) / stats.std;
                }};

                // Helper for EV calculation
                const getPWin = (h, m) => {{
                    const stats = raceStats[m];
                    const zAdj = raceData.horses.map(horse => (parseFloat(horse[m]) - stats.mean) / stats.std * 2.0);
                    const maxZ = Math.max(...zAdj);
                    const expZ = zAdj.map(z => Math.exp(z - maxZ));
                    const sumExpZ = expZ.reduce((a, b) => a + b, 0);
                    const hZAdj = (parseFloat(h[m]) - stats.mean) / stats.std * 2.0;
                    return Math.exp(hZAdj - maxZ) / sumExpZ;
                }};

                const getEV = (h, m) => {{
                    const pw = getPWin(h, m);
                    const rIdShort = String(raceId).length === 12 ? String(raceId).substring(2) : raceId;
                    const rOdds = window.tanshoData ? (window.tanshoData[raceId] || window.tanshoData[rIdShort]) : null;
                    if (rOdds) {{
                        const hO = rOdds.find(o => o[0] == h.horse_number);
                        if (hO) return pw * Math.log1p(parseFloat(hO[1]) || 0);
                    }}
                    return 0;
                }};

                // --- Calculate Softmax Probabilities (KV_z_peak using selected model) ---
                const allResultsForKV = [...raceData.horses];
                const zAdjScores = allResultsForKV.map(h => getZ(h, mainModelKey) * 2.0);
                const maxZ = Math.max(...zAdjScores);
                const expZ = zAdjScores.map(z => Math.exp(z - maxZ));
                const sumExpZ = expZ.reduce((a, b) => a + b, 0);
                
                allResultsForKV.forEach((h, idx) => {{
                    h.pWin = expZ[idx] / sumExpZ;
                }});

                // calculate max score for bar formatting based on main model
                const allMainScores = sortedHorses.map(h => parseFloat(h[mainModelKey]) || 0);
                const maxScore = Math.max(...allMainScores, 0.1);
                const minScore = Math.min(...allMainScores, 0);

                const isCollapsed = raceCollapseState[raceId] === true;
                const card = document.createElement('div');
                card.className = 'race-card' + (isCollapsed ? ' collapsed' : '');
                card.id = 'race-card-' + raceId;
                card.dataset.startTime = raceData.start_time || '';
                card.dataset.raceId = raceId;
                card.dataset.hasPickup1 = hasPickup1 ? '1' : '0';
                card.dataset.hasPickup2 = hasPickup2 ? '1' : '0';

                let horsesHtml = '';
                sortedHorses.forEach((horse, index) => {{
                    const hNum = horse.horse_number;
                    const hName = horse.horse_name;
                    const pWin = horse.pWin || 0;
                    
                    // Normalize width for bar
                    let widthPct = 0;
                    const mainScoreVal = parseFloat(horse[mainModelKey]) || 0;
                    if(maxScore > 0) {{
                        widthPct = Math.max(5, ((mainScoreVal - Math.min(0, minScore)) / (maxScore - Math.min(0, minScore))) * 100);
                    }}
                    
                    // Get Win Odds and calculate KV
                    // Support both 12-digit (2026...) and 10-digit (26...) keys
                    const raceIdShort = String(raceId).length === 12 ? String(raceId).substring(2) : raceId;
                    const raceWinOdds = window.tanshoData ? (window.tanshoData[raceId] || window.tanshoData[raceIdShort]) : null;
                    let winOdds = "-";
                    let kv = 0;
                    if (raceWinOdds) {{
                        const horseOdds = raceWinOdds.find(o => o[0] == hNum);
                        if (horseOdds) {{
                            winOdds = horseOdds[1];
                            kv = pWin * Math.log1p(parseFloat(winOdds));
                        }}
                    }}
                    
                    let rankClass = '';
                    if(sortBy === 'score') {{
                        if(index === 0) rankClass = 'rank-1';
                        else if(index === 1) rankClass = 'rank-2';
                        else if(index === 2) rankClass = 'rank-3';
                    }}

                    // Highlighting
                    let highlightClass = '';
                    if (jikuSet && jikuSet.has(String(hNum))) {{
                        highlightClass = 'is-jiku';
                    }} else if (partnerSet && partnerSet.has(String(hNum))) {{
                        highlightClass = 'is-partner';
                    }} else if (strat2Result.jikuSet && strat2Result.jikuSet.has(String(hNum))) {{
                        highlightClass = 'is-jiku-strat2';
                    }} else if (strat2Result.partnerSet && strat2Result.partnerSet.has(String(hNum))) {{
                        highlightClass = 'is-partner-strat2';
                    }}

                    // Decide main score vs sub scores display
                    const subModels = scoreModels.filter(m => m !== mainModelKey);
                    const modelShortNames = {{
                        'Ensemble': 'Ens',
                        'LightGBM_raw': 'LGBM',
                        'XGBoost_raw': 'XGB',
                        'CatBoost_raw': 'CB',
                        'LSTM_raw': 'LSTM',
                        'RandomForest_raw': 'RF',
                        'DecisionTree_raw': 'DT',
                        'Transformer_raw': 'TF',
                        'TabNet_raw': 'TN',
                        'AutoGluon_raw': 'AG'
                    }};

                    let subScoresHtml = '';
                    subModels.forEach(m => {{
                        subScoresHtml += `<span style="background: rgba(255,255,255,0.05); padding: 2px 6px; border-radius: 4px;">${{modelShortNames[m] || m}}: ${{getZ(horse, m).toFixed(4)}}</span> `;
                    }});

                    const winOddsNum = parseFloat(winOdds) || 999;
                    const probVal = pWin * 100;
                    
                    const oddsColor = winOddsNum <= 1.9 ? '#4ade80' : '#f8fafc';
                    const probColor = probVal >= 50 ? '#4ade80' : '#f8fafc';
                    const kvColor = kv >= 1.3 ? '#4ade80' : '#f8fafc';

                    const blockStyle = (color) => `background: rgba(255,255,255,0.05); padding: 2px 8px; border-radius: 4px; color: ${{color}}; border: 1px solid ${{color === '#4ade80' ? 'rgba(74, 222, 128, 0.2)' : 'transparent'}};`;

                    horsesHtml += `
                        <div class="horse-row ${{rankClass}} ${{highlightClass}}" style="flex-wrap: wrap; padding-left: 8px;">
                            <div style="display: flex; width: 100%; align-items: center; margin-bottom: 6px;">
                                <div class="horse-num">${{hNum}}</div>
                                <div class="horse-details" style="min-width: 0;">
                                    <div class="horse-name">${{hName}}</div>
                                    <div class="horse-score-bar-bg">
                                        <div class="horse-score-bar-fill" style="width: 0%" data-target="${{widthPct}}%"></div>
                                    </div>
                                </div>
                                <div class="horse-score-val" title="${{mainModelKey}} Z-Score">${{getZ(horse, mainModelKey).toFixed(4)}}</div>
                            </div>
                            <div style="display: flex; width: 100%; justify-content: flex-end; gap: 6px; font-size: 0.72rem; color: var(--text-muted); flex-wrap: wrap; margin-left: 50px; margin-bottom: 4px;">
                                ${{subScoresHtml}}
                            </div>
                            <div style="display: flex; width: 100%; gap: 6px; font-size: 0.75rem; font-weight: 700; overflow-x: auto; white-space: nowrap; scrollbar-width: none; -ms-overflow-style: none; padding-bottom: 4px; margin-left: 50px;">
                                <span style="${{blockStyle(oddsColor)}}">単勝: ${{winOdds}}</span>
                                <span style="${{blockStyle(probColor)}}">勝率予測: ${{probVal.toFixed(1)}}%</span>
                                <span style="${{blockStyle(kvColor)}}">期待値(EV): ${{kv > 0 ? kv.toFixed(2) : '-'}}</span>
                            </div>
                        </div>
                    `;
                }});

                const raceTitle = `${{raceData.place || ''}}${{raceData.round || ''}}R ${{raceData.start_time || ''}}`.trim() || raceData.title;

                const pickupBadgeHtml = ( () => {{
                    if (!hasPickup1) return '';
                    return `
                        <div class="pickup-badge" onclick="event.stopPropagation(); showRecommendation('${{raceId}}', 'strat1')">
                            <span style="font-size: 0.6rem; opacity: 0.8; font-weight: 400; color: #fff;">INFO</span>
                            <div style="font-weight: 900; letter-spacing: 0.05em; color: #fff;">PICKUP</div>
                        </div>
                    `;
                }})();

                const pickup2BadgeHtml = ( () => {{
                    if (!hasPickup2) return '';
                    return `
                        <div class="pickup2-badge" onclick="event.stopPropagation(); showRecommendation('${{raceId}}', 'strat2')">
                            <span style="font-size: 0.6rem; opacity: 0.85; font-weight: 700; color: #e9d5ff;">STRAT</span>
                            <div style="font-weight: 900; letter-spacing: 0.05em; color: #fff;">PICKUP 2</div>
                        </div>
                    `;
                }})();

                const reasonBadgeHtml = ( () => {{
                    if (!raceData.reasons || Object.keys(raceData.reasons).length === 0) return '';
                    return `
                        <div class="reason-badge" onclick="event.stopPropagation(); showReasons('${{raceId}}')">
                            <span style="font-size: 0.6rem; opacity: 0.8; font-weight: 400; color: #fff;">AI</span>
                            <div style="font-weight: 900; letter-spacing: 0.05em; color: #fff;">評価理由</div>
                        </div>
                    `;
                }})();

                card.innerHTML = `
                    <div class="race-info-header" onclick="toggleRaceCard('${{raceId}}')">
                        <div class="race-header-row-top">
                            <span class="race-title-text">${{raceTitle}}</span>
                            <div class="race-ext-links">
                                <a href="https://race.netkeiba.com/race/shutuba.html?race_id=${{raceData.race_id}}" target="_blank" onclick="event.stopPropagation()" class="race-link-btn web-link">🌐 Web</a>
                                <a href="https://netkeiba.onelink.me/Wmzg?af_xp=custom&af_dp=jp.co.netdreamers.netkeiba%3A%2F%2F&deep_link_value=https%3A%2F%2Frace.netkeiba.com%2Frace%2Fshutuba.html%3Frace_id%3D${{raceData.race_id}}&rf=race_toggle_menu" onclick="event.stopPropagation()" class="race-link-btn app-link">🏇 App</a>
                            </div>
                        </div>
                        <div class="race-header-row-bottom">
                            <div class="race-badges-group">
                                ${{pickupBadgeHtml}}
                                ${{pickup2BadgeHtml}}
                                ${{reasonBadgeHtml}}
                            </div>
                            <button class="accordion-toggle-btn" aria-label="Toggle race" onclick="event.stopPropagation(); toggleRaceCard('${{raceId}}')">
                                <svg class="chevron-svg" viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round">
                                    <polyline points="6 9 12 15 18 9"></polyline>
                                </svg>
                            </button>
                        </div>
                    </div>
                    <div class="race-body">
                        ${{horsesHtml}}
                    </div>
                `;
                container.appendChild(card);
            }}

            setTimeout(() => {{
                document.querySelectorAll('.horse-score-bar-fill').forEach(bar => {{
                    bar.style.width = bar.getAttribute('data-target');
                }});
            }}, 50);

            renderPickupSummary();
        }}

        // ==========================================================================
        // PICKUP 一括表示 & スマッピー一括投票 (選択日の PICKUP / PICKUP2 買い目をまとめて表示)
        // ==========================================================================
        let pickupSummaryItems = [];
        let pickupSummaryCollapsed = false;

        function togglePickupSummaryPanel() {{
            pickupSummaryCollapsed = !pickupSummaryCollapsed;
            const body = document.getElementById('pickup-summary-body');
            const chev = document.getElementById('pickup-summary-chevron');
            if (body) body.style.display = pickupSummaryCollapsed ? 'none' : 'block';
            if (chev) chev.style.transform = pickupSummaryCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)';
        }}

        function renderPickupSummary() {{
            const listEl = document.getElementById('pickup-summary-list');
            if (!listEl) return;

            const fDate = document.getElementById('filter-date') ? document.getElementById('filter-date').value : 'ALL';
            pickupSummaryItems = [];

            const sortedRaces = Object.values(currentData).sort((a, b) => {{
                const timeA = a.start_time || "00:00";
                const timeB = b.start_time || "00:00";
                if (timeA !== timeB) return timeA.localeCompare(timeB);
                const roundA = parseInt(a.round) || 0;
                const roundB = parseInt(b.round) || 0;
                if (roundA !== roundB) return roundA - roundB;
                return (a.place || "").localeCompare(b.place || "");
            }});

            sortedRaces.forEach(raceData => {{
                if (fDate !== 'ALL' && raceData.date !== fDate) return;
                const raceId = raceData.race_id;
                const raceTitle = `${{raceData.place || ''}}${{raceData.round || ''}}R ${{raceData.start_time || ''}}`.trim() || raceData.title;

                // PICKUP (戦略1 / Kelly2)
                const kellyResult = evaluateKelly2Strategies(raceData, raceId);
                (kellyResult.validStrategies || []).forEach(item => {{
                    if (!item.bettingEyesText) return;
                    pickupSummaryItems.push({{
                        raceId: raceId,
                        raceTitle: raceTitle,
                        tag: 'PICKUP',
                        tagColor: '#4ade80',
                        rawType: item.rawType,
                        bettingEyesText: item.bettingEyesText,
                        round: raceData.round,
                        place: raceData.place,
                        weekday: raceData.weekday,
                        axisCount: item.axis2Num ? 2 : 1
                    }});
                }});

                // PICKUP 2 (戦略2 / Kelly3)
                const strat2Result = evaluateStrategy2(raceData, raceId);
                (strat2Result.validStrategies || []).forEach(s2 => {{
                    if (s2.sub_items && s2.sub_items.length > 0) {{
                        s2.sub_items.forEach(sub => {{
                            if (!sub.bettingEyesText) return;
                            pickupSummaryItems.push({{
                                raceId: raceId,
                                raceTitle: raceTitle,
                                tag: 'PICKUP 2',
                                tagColor: '#fb923c',
                                rawType: sub.rawType,
                                bettingEyesText: sub.bettingEyesText,
                                round: raceData.round,
                                place: raceData.place,
                                weekday: raceData.weekday,
                                axisCount: sub.axis2Num ? 2 : (sub.axis1Num ? 1 : 0)
                            }});
                        }});
                    }} else if (s2.bettingEyesText) {{
                        pickupSummaryItems.push({{
                            raceId: raceId,
                            raceTitle: raceTitle,
                            tag: 'PICKUP 2',
                            tagColor: '#fb923c',
                            rawType: s2.rawType,
                            bettingEyesText: s2.bettingEyesText,
                            round: raceData.round,
                            place: raceData.place,
                            weekday: raceData.weekday,
                            axisCount: s2.axis2Num ? 2 : 1
                        }});
                    }}
                }});
            }});

            if (pickupSummaryItems.length === 0) {{
                listEl.innerHTML = `<div style="padding: 20px; text-align: center; color: var(--text-muted); font-size: 0.8rem;">この日付の PICKUP 馬券はありません</div>`;
            }} else {{
                listEl.innerHTML = pickupSummaryItems.map((it, idx) => `
                    <label class="pickup-summary-item">
                        <input type="checkbox" class="pickup-check" data-idx="${{idx}}" onchange="updatePickupSelectedCount()">
                        <div class="pickup-summary-item-main">
                            <div class="pickup-summary-item-top">
                                <span class="pickup-summary-tag" style="color: ${{it.tagColor}}; border-color: ${{it.tagColor}}55; background: ${{it.tagColor}}18;">${{it.tag}}</span>
                                <span class="pickup-summary-race">${{it.raceTitle}}</span>
                                <span class="pickup-summary-type">${{it.rawType}}</span>
                            </div>
                            <div class="pickup-summary-eyes">${{it.bettingEyesText}}</div>
                        </div>
                        <div style="display:flex; gap:4px; flex-shrink:0;">
                            <button type="button" class="smappy-btn" style="padding: 7px 9px;" title="スマッピー投票" onclick="event.preventDefault(); event.stopPropagation(); startSingleVote(${{idx}}, 'smappy');">📌</button>
                            <button type="button" class="ipat-btn" style="padding: 7px 9px;" title="即PAT投票" onclick="event.preventDefault(); event.stopPropagation(); startSingleVote(${{idx}}, 'ipat');">🟢</button>
                            <button type="button" class="umaca-btn" style="padding: 7px 9px;" title="UMACA投票" onclick="event.preventDefault(); event.stopPropagation(); startSingleVote(${{idx}}, 'umaca');">🟣</button>
                        </div>
                    </label>
                `).join('');
            }}

            const countEl = document.getElementById('pickup-summary-count');
            if (countEl) countEl.textContent = pickupSummaryItems.length;
            updatePickupSelectedCount();
        }}

        function togglePickupCheckAll(state) {{
            document.querySelectorAll('.pickup-check').forEach(cb => {{ cb.checked = state; }});
            updatePickupSelectedCount();
        }}

        function updatePickupSelectedCount() {{
            const n = document.querySelectorAll('.pickup-check:checked').length;
            const el = document.getElementById('pickup-selected-count');
            if (el) el.textContent = n;
        }}

        function calcBetPoints(siki, hou, axes, partners, isMulti) {{
            axes = axes || [];
            partners = partners || [];
            const nA = axes.length;
            const nP = partners.length;
            if (siki === '1' || siki === '2') return 1;
            if (siki === '3' || siki === '4' || siki === '5') {{
                if (hou === '1' || hou === 'box') {{
                    const allH = nA + nP;
                    return (allH * (allH - 1)) / 2;
                }}
                return nP;
            }}
            if (siki === '6') {{
                if (hou === '1' || hou === 'box') {{
                    const allH = nA + nP;
                    return allH * (allH - 1);
                }}
                return isMulti ? (nP * 2) : nP;
            }}
            if (siki === '7') {{
                if (hou === '1' || hou === 'box') {{
                    const allH = nA + nP;
                    return Math.floor((allH * (allH - 1) * (allH - 2)) / 6);
                }}
                if (nA >= 2) return nP;
                return Math.floor((nP * (nP - 1)) / 2);
            }}
            if (siki === '8') {{
                if (hou === '1' || hou === 'box') {{
                    const allH = nA + nP;
                    return allH * (allH - 1) * (allH - 2);
                }}
                if (nA >= 2) {{
                    return isMulti ? (nP * 6) : nP;
                }}
                return isMulti ? (nP * (nP - 1) * 3) : (nP * (nP - 1));
            }}
            return 1;
        }}

        function convertItemToBets(item, unitVal, todayPlaces) {{
            const siki = getSmappySiki(item.rawType);
            const hou = getSmappyHou(item.rawType, item.axisCount || 1);
            const parsed = parseSmappyEyes(item.bettingEyesText, item.rawType);
            const isMulti = (item.rawType || "").indexOf('マルチ') >= 0;
            const placeName = item.place;
            const weekday = item.weekday || "";
            const round = item.round;
            let vIdx = todayPlaces.indexOf(placeName);
            if (vIdx < 0) vIdx = 0;
            const vStr = String(vIdx);

            if (parsed.is2Touri && parsed.axes && parsed.axes.length >= 2 && parsed.partners && parsed.partners.length >= 1) {{
                const h1 = String(parsed.axes[0]);
                const h2 = String(parsed.axes[1]);
                const h3 = String(parsed.partners[0]);
                return [
                    {{
                        steps: [vStr, round, siki, "0", h1, h2, h3],
                        venueName: placeName,
                        placeName: placeName,
                        round: round,
                        raceNo: round,
                        siki: siki,
                        hou: "0",
                        axes: [h1, h2],
                        partners: [h3],
                        isMulti: false,
                        weekday: weekday,
                        unitAmount: unitVal,
                        totalAmount: unitVal
                    }},
                    {{
                        steps: [vStr, round, siki, "0", h2, h1, h3],
                        venueName: placeName,
                        placeName: placeName,
                        round: round,
                        raceNo: round,
                        siki: siki,
                        hou: "0",
                        axes: [h2, h1],
                        partners: [h3],
                        isMulti: false,
                        weekday: weekday,
                        unitAmount: unitVal,
                        totalAmount: unitVal
                    }}
                ];
            }} else if (parsed.isOrikaeshi && parsed.axes && parsed.axes.length >= 2) {{
                // 馬単-折り返し: 通常の2点買い目として展開
                const fa = String(parsed.axes[0]);
                const fb = String(parsed.axes[1]);
                return [
                    {{
                        steps: [vStr, round, siki, "0", fa, fb],
                        venueName: placeName,
                        placeName: placeName,
                        round: round,
                        raceNo: round,
                        siki: siki,
                        hou: "0",
                        axes: [fa],
                        partners: [fb],
                        isMulti: false,
                        weekday: weekday,
                        unitAmount: unitVal,
                        totalAmount: unitVal
                    }},
                    {{
                        steps: [vStr, round, siki, "0", fb, fa],
                        venueName: placeName,
                        placeName: placeName,
                        round: round,
                        raceNo: round,
                        siki: siki,
                        hou: "0",
                        axes: [fb],
                        partners: [fa],
                        isMulti: false,
                        weekday: weekday,
                        unitAmount: unitVal,
                        totalAmount: unitVal
                    }}
                ];
            }} else {{
                const rawSteps = [vStr, round, siki];
                const simple = (siki === '1' || siki === '2' || siki === '9');
                if (!simple && hou) rawSteps.push(hou);
                (parsed.axes || []).forEach(a => rawSteps.push(String(a)));
                (parsed.partners || []).forEach(pt => rawSteps.push(String(pt)));

                const pts = calcBetPoints(siki, hou, parsed.axes, parsed.partners, isMulti);
                const tot = Math.max(1, pts) * unitVal;

                return [{{
                    steps: rawSteps,
                    venueName: placeName,
                    placeName: placeName,
                    round: round,
                    raceNo: round,
                    siki: siki,
                    hou: hou,
                    axes: (parsed.axes || []).map(String),
                    partners: (parsed.partners || []).map(String),
                    isMulti: isMulti,
                    weekday: weekday,
                    unitAmount: unitVal,
                    totalAmount: tot
                }}];
            }}
        }}

        // serviceType: 'smappy' | 'ipat' | 'umaca'
        function startBulkVote(serviceType) {{
            const idxs = Array.from(document.querySelectorAll('.pickup-check:checked')).map(cb => parseInt(cb.getAttribute('data-idx')));
            if (idxs.length === 0) {{
                alert('投票する買い目にチェックを入れてください');
                return;
            }}
            startBetQueue(idxs, serviceType);
        }}

        function startSingleVote(idx, serviceType) {{
            startBetQueue([idx], serviceType);
        }}

        let _betQueueItems = [];
        let _betQueueType = 'smappy';

        function syncGlobalUnitAmount() {{
            const v = window._globalUnitAmount || 100;
            const gInp = document.getElementById('global-unit-amount');
            if (gInp && parseInt(gInp.value) !== v) gInp.value = v;
            ['ipat-unit-amount', 'umaca-unit-amount', 'smappy-unit-amount'].forEach(function(id) {{
                const el = document.getElementById(id);
                if (el && parseInt(el.value) !== v) el.value = v;
            }});
        }}

        function startBetQueue(idxs, serviceType) {{
            _betQueueItems = idxs.map(i => pickupSummaryItems[i]).filter(Boolean);
            _betQueueType = serviceType || 'smappy';
            if (_betQueueItems.length === 0) return;
            renderBetQueueModal();
        }}

        function updateModalTotal() {{
            const svc = _betQueueType;
            const unitInp = document.getElementById(svc + '-unit-amount');
            const u = unitInp ? (parseInt(unitInp.value) || 100) : 100;
            window._globalUnitAmount = u;
            syncGlobalUnitAmount();

            let grandTotal = 0;
            _betQueueItems.forEach(it => {{
                const siki = getSmappySiki(it.rawType);
                const hou = getSmappyHou(it.rawType, it.axisCount || 1);
                const parsed = parseSmappyEyes(it.bettingEyesText, it.rawType);
                const isMulti = (it.rawType || "").indexOf('マルチ') >= 0;
                let pts = 1;
                if (parsed.is2Touri || parsed.isOrikaeshi) pts = 2;
                else pts = calcBetPoints(siki, hou, parsed.axes, parsed.partners, isMulti);
                grandTotal += Math.max(1, pts) * u;
            }});

            const dispEl = document.getElementById(svc + '-total-disp');
            if (dispEl) dispEl.innerText = grandTotal.toLocaleString() + '円';
        }}

        function launchBulkVote(serviceType) {{
            const items = _betQueueItems && _betQueueItems.length > 0 ? _betQueueItems : [];
            if (items.length === 0) {{
                alert('投票する買い目がありません');
                return;
            }}

            const unitInp = document.getElementById(serviceType + '-unit-amount') || document.getElementById('global-unit-amount');
            const unitVal = unitInp ? (parseInt(unitInp.value) || 100) : (window._globalUnitAmount || 100);

            const vCodes = {{ "札幌":"01","函館":"02","福島":"03","新潟":"04","東京":"05","中山":"06","中京":"07","京都":"08","阪神":"09","小倉":"10" }};
            let todayPlaces = [];
            for (const k in currentData) {{
                const p = currentData[k].place;
                if (!todayPlaces.includes(p)) todayPlaces.push(p);
            }}
            todayPlaces.sort((a, b) => (vCodes[a] || "99") - (vCodes[b] || "99"));

            let allBets = [];
            let grandTotal = 0;

            items.forEach(it => {{
                const bList = convertItemToBets(it, unitVal, todayPlaces);
                bList.forEach(b => {{
                    allBets.push(b);
                    grandTotal += (b.totalAmount || unitVal);
                }});
            }});

            const payload = {{
                bets: allBets,
                unitAmount: unitVal,
                totalAmount: grandTotal
            }};

            const jsonStr = JSON.stringify(payload);

            try {{
                const t = document.createElement('textarea');
                t.value = jsonStr;
                document.body.appendChild(t);
                t.select();
                document.execCommand('copy');
                document.body.removeChild(t);
            }} catch(e) {{}}

            const scriptName = serviceType === 'ipat' ? '即PAT' : (serviceType === 'umaca' ? 'UMACA' : 'スマッピー');
            const scriptableUrl = "scriptable:///run?scriptName=" + encodeURIComponent(scriptName) + "&data=" + encodeURIComponent(jsonStr);
            const aTag = document.createElement('a');
            aTag.href = scriptableUrl;
            aTag.style.display = 'none';
            document.body.appendChild(aTag);
            aTag.click();
            setTimeout(function() {{ document.body.removeChild(aTag); }}, 1000);
        }}

        function renderBetQueueModal() {{
            const prevModal = document.getElementById('bet-queue-modal');
            if (prevModal) prevModal.remove();

            const items = _betQueueItems;
            if (!items || items.length === 0) return;

            const svc = _betQueueType;
            const _initUnit = window._globalUnitAmount || 100;
            const color = svc === 'ipat' ? '#10b981' : (svc === 'umaca' ? '#c084fc' : '#38bdf8');
            const grad = svc === 'ipat' ? 'linear-gradient(135deg, #10b981, #059669)' : (svc === 'umaca' ? 'linear-gradient(135deg, #a855f7, #7c3aed)' : 'linear-gradient(135deg, #0284c7, #0369a1)');
            const svcLabel = svc === 'ipat' ? '即PAT' : (svc === 'umaca' ? 'UMACA' : 'スマッピー');
            const unitId = svc + '-unit-amount';
            const totalId = svc + '-total-disp';

            let grandTotal = 0;
            const itemsListHtml = items.map((it, idx) => {{
                const siki = getSmappySiki(it.rawType);
                const hou = getSmappyHou(it.rawType, it.axisCount || 1);
                const parsed = parseSmappyEyes(it.bettingEyesText, it.rawType);
                const isMulti = (it.rawType || "").indexOf('マルチ') >= 0;
                let pts = 1;
                if (parsed.is2Touri || parsed.isOrikaeshi) pts = 2;
                else pts = calcBetPoints(siki, hou, parsed.axes, parsed.partners, isMulti);
                const subTot = Math.max(1, pts) * _initUnit;
                grandTotal += subTot;

                return `
                    <div style="display: flex; align-items: center; justify-content: space-between; padding: 6px 8px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 6px; margin-bottom: 4px; font-size: 0.72rem;">
                        <div>
                            <span style="font-weight: 800; color: #f1f5f9; margin-right: 6px;">${{it.raceTitle}}</span>
                            <span style="color: #94a3b8; font-size: 0.68rem; margin-right: 6px;">${{it.rawType}}</span>
                            <span style="color: #38bdf8; font-weight: 700;">${{it.bettingEyesText}}</span>
                        </div>
                        <div style="text-align: right; color: #cbd5e1; font-weight: 700; white-space: nowrap; margin-left: 8px;">
                            ${{pts}}点
                        </div>
                    </div>
                `;
            }}).join('');

            const modal = document.createElement('div');
            modal.id = 'bet-queue-modal';
            modal.className = 'smappy-queue-modal-overlay';
            modal.onclick = (e) => {{ if (e.target === modal) closeBetQueue(); }};
            modal.innerHTML = `
                <div class="smappy-popup smappy-queue-popup" onclick="event.stopPropagation()" style="max-width: 420px; width: 92%;">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom: 8px;">
                        <span style="font-weight: 800; font-size: 0.95rem; color: ${{color}};">
                            🚀 ${{svcLabel}} 一括自動投票
                        </span>
                        <span style="font-size: 0.72rem; color: #94a3b8; font-weight: 700;">
                            選択中: <strong style="color: #fff;">${{items.length}}件</strong>
                        </span>
                    </div>

                    <div style="max-height: 150px; overflow-y: auto; margin-bottom: 12px; padding-right: 4px;">
                        ${{itemsListHtml}}
                    </div>

                    <div style="margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between; background: rgba(255,255,255,0.04); padding: 8px 12px; border-radius: 8px;">
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <label style="font-size: 0.72rem; color: var(--text-muted); font-weight: 800;">1点金額:</label>
                            <input type="number" id="${{unitId}}" value="${{_initUnit}}" step="100" min="100" style="width: 75px; padding: 4px 6px; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.15); color: #fff; border-radius: 6px; font-size: 0.78rem; text-align: right; font-weight: 700;" oninput="updateModalTotal();">
                            <span style="font-size: 0.72rem; color: #cbd5e1;">円</span>
                        </div>
                        <div style="text-align: right; font-size: 0.75rem; color: #94a3b8;">
                            合計: <strong id="${{totalId}}" style="color: ${{color}}; font-size: 1rem; font-weight: 800; margin-left: 4px;">${{grandTotal.toLocaleString()}}円</strong>
                        </div>
                    </div>

                    <button onclick="launchBulkVote('${{svc}}')" style="width: 100%; padding: 13px; background: ${{grad}}; color: #fff; border: none; border-radius: 8px; font-weight: 800; font-size: 0.9rem; cursor: pointer; box-shadow: 0 4px 14px rgba(0,0,0,0.3); margin-bottom: 8px; display: flex; align-items: center; justify-content: center; gap: 6px;">
                        🚀 ${{svcLabel}}で一括投票を実行 (${{items.length}}件)
                    </button>

                    <div style="font-size: 0.62rem; color: #94a3b8; text-align: center; line-height: 1.4; margin-bottom: 10px;">
                        ※ Scriptableが起動し全買い目を自動セットします。最終確認画面（金額入力済）で停止します。
                    </div>

                    <div style="text-align:center;">
                        <button type="button" onclick="closeBetQueue()" style="background:none; border:none; color:var(--text-muted); font-size:0.72rem; cursor:pointer; text-decoration:underline;">閉じる</button>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);
        }}

        function closeBetQueue() {{
            const modal = document.getElementById('bet-queue-modal');
            if (modal) modal.remove();
        }}

        function evaluateKelly2Strategies(raceData, raceId) {{
            const jikuSet = new Set();
            const partnerSet = new Set();
            if (raceData && raceData.kelly2 && Array.isArray(raceData.kelly2.bets)) {{
                const validStrategies = raceData.kelly2.bets.map(b => {{
                    if (b.axis1Num) jikuSet.add(String(b.axis1Num));
                    if (b.axis2Num) jikuSet.add(String(b.axis2Num));
                    (b.partnerNums || []).forEach(n => partnerSet.add(String(n)));
                    return {{
                        rawType: b.rawType,
                        model: b.model,
                        strategyId: b.strategy_id,
                        s: {{ axis_count: b.axis2Num ? 2 : 1 }},
                        strat: {{ axis_count: b.axis2Num ? 2 : 1 }},
                        combs: b.combs,
                        cost: b.cost,
                        unit: b.unit,
                        bettingEyesText: b.bettingEyesText,
                        axis1Num: b.axis1Num,
                        axis2Num: b.axis2Num,
                        partnerNums: b.partnerNums || [],
                        roi: b.roi,
                        hitRate: b.hitRate,
                        h1Info: b.h1Info,
                        h1PopRank: b.h1PopRank
                    }};
                }});
                return {{ validStrategies, jikuSet, partnerSet, horseConf: {{}}, popRanks: {{}}, source: 'Kelly2.ipynb' }};
            }}

            return {{ validStrategies: [], jikuSet: new Set(), partnerSet: new Set(), horseConf: {{}}, popRanks: {{}} }};
        }}

        function evaluateStrategy2(raceData, raceId) {{
            if (!raceData) {{
                return {{ validStrategies: [], isPickup: false, jikuSet: new Set(), partnerSet: new Set() }};
            }}

            const jikuSet = new Set();
            const partnerSet = new Set();
            const s2 = raceData.strat2;

            if (s2) {{
                if (s2.axis1Num) jikuSet.add(String(s2.axis1Num));
                if (s2.axis2Num) jikuSet.add(String(s2.axis2Num));
                if (s2.partnerNums && Array.isArray(s2.partnerNums)) {{
                    s2.partnerNums.forEach(n => partnerSet.add(String(n)));
                }}

                // PICKUP 2 の対象: サーバ側で生成した買い目(lines)がある、または
                // action_id > 0 (見送りSKIP以外) の場合
                const isPickup = !!s2.lines && s2.lines.length > 0
                    || s2.is_pickup === true || (s2.action_id > 0);
                return {{
                    validStrategies: isPickup ? [s2] : [],
                    isPickup: isPickup,
                    strat: s2,
                    jikuSet,
                    partnerSet
                }};
            }}

            return {{
                validStrategies: [],
                isPickup: false,
                strat: null,
                jikuSet,
                partnerSet
            }};
        }}

        let currentActiveRecTab = 'strat1';

        function showRecommendation(raceId, initialTab) {{
            const raceData = currentData[raceId];
            if (!raceData) return;

            if (initialTab) {{
                currentActiveRecTab = initialTab;
            }}

            const modal = document.getElementById('recommend-modal');
            const body = document.getElementById('modal-body');
            
            const kellyResult = evaluateKelly2Strategies(raceData, raceId);
            const strat1List = kellyResult.validStrategies || [];

            const strat2Result = evaluateStrategy2(raceData, raceId);
            const strat2List = strat2Result.validStrategies || [];

            // もし初期タブが指定されておらず、片方しか該当しない場合は該当するタブを優先
            if (!initialTab) {{
                if (strat1List.length === 0 && strat2List.length > 0) {{
                    currentActiveRecTab = 'strat2';
                }} else {{
                    currentActiveRecTab = 'strat1';
                }}
            }}

            let html = `
                <div style="text-align: center; margin-bottom: 20px; position: relative;">
                    <div style="font-size: 0.8rem; color: ${{currentActiveRecTab === 'strat2' ? '#fb923c' : '#4ade80'}}; font-weight: 800; text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 8px;">
                        ${{currentActiveRecTab === 'strat2' ? 'Kelly3 Portfolio (Kelly3.ipynb)' : 'Kelly2 (Kelly2.ipynb)'}}
                    </div>
                    <h2 style="margin: 0; font-size: 1.8rem; color: #fff;">${{raceData.title}}</h2>
                    <button onclick="event.stopPropagation(); fetchRaceResults('${{raceId}}', true)" 
                            style="position: absolute; top: 0; right: 0; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #fff; border-radius: 8px; width: 32px; height: 32px; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s; z-index: 30;"
                            title="Refresh Results">
                        🔄
                    </button>
                </div>

                <div class="rec-tab-group">
                    <button class="rec-tab-btn ${{currentActiveRecTab === 'strat1' ? 'active' : ''}}" onclick="switchRecTab('${{raceId}}', 'strat1')">
                        🎯 戦略1 (Kelly2)${{strat1List.length > 0 ? ` <span style="opacity:0.9; font-size:0.75rem;">(${{strat1List.length}})</span>` : ''}}
                    </button>
                    <button class="rec-tab-btn strat2 ${{currentActiveRecTab === 'strat2' ? 'active' : ''}}" onclick="switchRecTab('${{raceId}}', 'strat2')">
                        🚀 戦略2 (Kelly3)${{strat2List.length > 0 ? ` <span style="opacity:0.9; font-size:0.75rem;">(推奨)</span>` : ''}}
                    </button>
                </div>
            `;

            if (currentActiveRecTab === 'strat1') {{
                if (strat1List.length > 0) {{
                    strat1List.forEach(item => {{
                        const s = item.strat || item.s || {{}};
                        const displayType = item.rawType;
                        const popDisp = item.h1PopRank ? `単勝 ${{item.h1PopRank}}番人気` : '';
                        const h1 = item.h1Info || {{}};
                        const confDisp = (typeof h1.avgRank === 'number') ? `4モデル平均 ${{h1.avgRank.toFixed(1)}}位 (Top3支持: ${{h1.top3Count ?? 0}}モデル)` : '';

                        html += `
                            <div class="strategy-item-modal" data-strategy-type="${{item.rawType}}">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                    <div style="font-weight: 900; color: #fbbf24; font-size: 1.1rem;">
                                        ${{displayType}} 
                                        <span style="font-size: 0.75rem; color: #60a5fa; margin-left:8px; font-weight:700; background: rgba(96, 165, 250, 0.1); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(96, 165, 250, 0.2);">${{item.model}}</span>
                                        <span style="font-size: 0.75rem; color: #4ade80; margin-left:6px; font-weight:700; background: rgba(74, 222, 128, 0.1); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(74, 222, 128, 0.2);">${{item.combs}}点</span>
                                    </div>
                                </div>
                                <div class="bet-eyes-box">
                                    <div style="font-size: 0.7rem; color: var(--text-muted); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.1em;">Recommended Combination</div>
                                    <div class="bet-eyes-text">${{item.rawType.includes('2通り') ? item.bettingEyesText.replace(', ', '<br>') : item.bettingEyesText}}</div>
                                </div>
                                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">
                                    <div style="color: #94a3b8;">
                                        ${{item.axis1Num && item.axis2Num ? `軸馬: <strong>${{String(item.axis1Num).padStart(2, '0')}}番, ${{String(item.axis2Num).padStart(2, '0')}}番</strong>` : `軸馬: <strong>${{String(item.axis1Num).padStart(2, '0')}}番</strong>`}}
                                        ${{item.partnerNums && item.partnerNums.length > 0 ? ` | 相手: <strong>${{item.partnerNums.map(n => String(n).padStart(2, '0')).join(', ')}}</strong>` : ''}}
                                        (${{confDisp}}${{popDisp ? ' / ' + popDisp : ''}})
                                    </div>
                                    <div>
                                        ROI: <strong style="color: #4ade80;">${{item.roi}}%</strong> | 的中率: <strong style="color: #60a5fa;">${{item.hitRate}}%</strong>
                                    </div>
                                </div>
                                <div class="bet-result-details"></div>
                                <div style="margin-top: 10px; text-align: right; display: flex; justify-content: flex-end; gap: 8px; flex-wrap: wrap;">
                                    <button class="ipat-btn" data-eyes="${{item.bettingEyesText}}" data-type="${{item.rawType}}" data-round="${{raceData.round}}" data-axis="${{item.axis2Num ? 2 : (s.axis_count || 1)}}" data-place="${{raceData.place}}" data-weekday="${{raceData.weekday}}" onclick="event.stopPropagation(); showIpat(this)">🟢 即PAT</button>
                                    <button class="umaca-btn" data-eyes="${{item.bettingEyesText}}" data-type="${{item.rawType}}" data-round="${{raceData.round}}" data-axis="${{item.axis2Num ? 2 : (s.axis_count || 1)}}" data-place="${{raceData.place}}" data-weekday="${{raceData.weekday}}" onclick="event.stopPropagation(); showUmaca(this)">🟣 UMACA</button>
                                    <button class="smappy-btn" data-eyes="${{item.bettingEyesText}}" data-type="${{item.rawType}}" data-round="${{raceData.round}}" data-axis="${{item.axis2Num ? 2 : (s.axis_count || 1)}}" data-place="${{raceData.place}}" data-weekday="${{raceData.weekday}}" onclick="event.stopPropagation(); showSmappy(this)">📌 スマッピー</button>
                                </div>
                            </div>
                        `;
                    }});
                }} else {{
                    html += `
                        <div style="padding: 40px 20px; text-align: center; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px dashed rgba(255,255,255,0.1); color: var(--text-muted); margin-bottom: 20px;">
                            <div style="font-size: 1.5rem; margin-bottom: 10px;">📋</div>
                            <div style="font-size: 0.9rem; font-weight: 800; color: #fff; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.1em;">No High-Confidence Recommendations</div>
                            <div style="font-weight: 700; font-size: 0.8rem;">Kelly2 高信頼度条件を満たす買い目はありません</div>
                        </div>
                    `;
                }}
            }} else {{
                // 戦略2 (Kelly3 / PICKUP 2): Kelly3.ipynb の中央値重視ポートフォリオ
                const s2 = raceData.strat2;
                if (s2) {{
                    const isPickup = s2.is_pickup === true || (s2.action_id > 0);
                    const actionBadgeColor = isPickup ? '#fb923c' : '#94a3b8';
                    const actionBadgeBg = isPickup ? 'rgba(249, 115, 22, 0.15)' : 'rgba(255, 255, 255, 0.05)';
                    const borderLeftColor = isPickup ? '#fb923c' : '#64748b';
                    const statusText = isPickup ? '🚀 PICKUP 2 (積極購入推奨)' : '見送り (SKIP)';
                    const pad = (n) => String(n).padStart(2, '0');

                    // 1. ハイブリッド等で複数券種に分割されている場合
                    if (s2.sub_items && s2.sub_items.length > 0) {{
                        html += `
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding: 12px 16px; background: rgba(249, 115, 22, 0.08); border-radius: 10px; border: 1px solid rgba(249, 115, 22, 0.25);">
                                <div>
                                    <div style="font-weight: 900; color: #fb923c; font-size: 1.1rem;">
                                        ${{s2.action_name}}
                                        <span style="font-size: 0.75rem; color: #60a5fa; margin-left:8px; font-weight:700; background: rgba(96, 165, 250, 0.1); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(96, 165, 250, 0.2);">${{s2.model || 'Kelly3'}}</span>
                                    </div>
                                    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">
                                        合計: <strong style="color: #4ade80;">${{s2.combs}}点 (${{s2.cost.toLocaleString()}}円)</strong> / Kelly3.ipynb 中央値重視ポートフォリオ
                                    </div>
                                </div>
                                <div style="font-size: 0.7rem; color: #fb923c; font-weight: 700; background: rgba(249, 115, 22, 0.2); padding: 4px 10px; border-radius: 4px; border: 1px solid rgba(249, 115, 22, 0.4);">
                                    ${{statusText}}
                                </div>
                            </div>
                        `;

                        s2.sub_items.forEach((sub, subIdx) => {{
                            const subColor = sub.type === 'SANRENPUKU' ? '#fbbf24' : '#fb923c';
                            const subBg = sub.type === 'SANRENPUKU' ? 'rgba(251, 191, 36, 0.06)' : 'rgba(249, 115, 22, 0.06)';
                            const subBorder = sub.type === 'SANRENPUKU' ? 'rgba(251, 191, 36, 0.3)' : 'rgba(249, 115, 22, 0.3)';
                            const subLabel = sub.bet_type_jp || sub.rawType.split('-')[0];

                            html += `
                                <div class="strategy-item-modal" data-strategy-type="${{sub.rawType}}" style="border-left: 4px solid ${{subColor}}; margin-bottom: 14px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                        <div style="font-weight: 900; color: ${{subColor}}; font-size: 1.05rem;">
                                            ${{sub.rawType}}
                                            <span style="font-size: 0.75rem; color: #4ade80; margin-left:8px; font-weight:700; background: rgba(74, 222, 128, 0.1); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(74, 222, 128, 0.2);">${{sub.combs}}点 (${{sub.cost.toLocaleString()}}円)</span>
                                        </div>
                                        <div style="font-size: 0.7rem; color: ${{subColor}}; font-weight: 700; background: ${{subBg}}; padding: 2px 8px; border-radius: 4px; border: 1px solid ${{subBorder}};">
                                            ${{subLabel}}
                                        </div>
                                    </div>
                                    <div class="bet-eyes-box" style="border-color: ${{subBorder}}; background: ${{subBg}};">
                                        <div style="font-size: 0.7rem; color: ${{subColor}}; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700;">
                                            Recommended Combination
                                        </div>
                                        <div class="bet-eyes-text" style="color: #fff; font-size: 1.15rem;">${{sub.bettingEyesText}}</div>
                                    </div>
                                    <div class="bet-result-details"></div>
                                    <div style="margin-top: 10px; text-align: right; display: flex; justify-content: flex-end; gap: 8px; flex-wrap: wrap;">
                                        <button class="ipat-btn" data-eyes="${{sub.bettingEyesText}}" data-type="${{sub.rawType}}" data-round="${{raceData.round}}" data-axis="${{sub.axis2Num ? 2 : (sub.axis1Num ? 1 : 0)}}" data-place="${{raceData.place}}" data-weekday="${{raceData.weekday}}" onclick="event.stopPropagation(); showIpat(this)">🟢 即PAT (${{sub.rawType.split('-')[0]}})</button>
                                        <button class="umaca-btn" data-eyes="${{sub.bettingEyesText}}" data-type="${{sub.rawType}}" data-round="${{raceData.round}}" data-axis="${{sub.axis2Num ? 2 : (sub.axis1Num ? 1 : 0)}}" data-place="${{raceData.place}}" data-weekday="${{raceData.weekday}}" onclick="event.stopPropagation(); showUmaca(this)">🟣 UMACA (${{sub.rawType.split('-')[0]}})</button>
                                        <button class="smappy-btn" data-eyes="${{sub.bettingEyesText}}" data-type="${{sub.rawType}}" data-round="${{raceData.round}}" data-axis="${{sub.axis2Num ? 2 : (sub.axis1Num ? 1 : 0)}}" data-place="${{raceData.place}}" data-weekday="${{raceData.weekday}}" onclick="event.stopPropagation(); showSmappy(this)">📌 スマッピー (${{sub.rawType.split('-')[0]}})</button>
                                    </div>
                                </div>
                            `;
                        }});
                    }} else {{
                        // 2. 通常の単一買い目
                        let jikuDisp = '';
                        if (s2.axis1Num && s2.axis2Num) {{
                            jikuDisp = `軸馬: <strong>${{pad(s2.axis1Num)}}番, ${{pad(s2.axis2Num)}}番</strong>`;
                        }} else if (s2.axis1Num) {{
                            jikuDisp = `軸馬: <strong>${{pad(s2.axis1Num)}}番</strong>`;
                        }}
                        let partnerDisp = '';
                        if (s2.partnerNums && s2.partnerNums.length > 0) {{
                            partnerDisp = `相手馬: <strong>${{s2.partnerNums.map(pad).join(', ')}}</strong>`;
                        }}

                        html += `
                            <div class="strategy-item-modal" data-strategy-type="${{s2.rawType}}" style="border-left: 4px solid ${{borderLeftColor}};">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
                                    <div style="font-weight: 900; color: ${{actionBadgeColor}}; font-size: 1.1rem;">
                                        ${{s2.rawType}} 
                                        <span style="font-size: 0.75rem; color: #60a5fa; margin-left:8px; font-weight:700; background: rgba(96, 165, 250, 0.1); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(96, 165, 250, 0.2);">${{s2.model || 'Kelly3'}}</span>
                                        <span style="font-size: 0.75rem; color: #4ade80; margin-left:6px; font-weight:700; background: rgba(74, 222, 128, 0.1); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(74, 222, 128, 0.2);">${{s2.combs}}点 (${{s2.cost.toLocaleString()}}円)</span>
                                    </div>
                                    <div style="font-size: 0.7rem; color: ${{actionBadgeColor}}; font-weight: 700; background: ${{actionBadgeBg}}; padding: 3px 10px; border-radius: 4px; border: 1px solid ${{actionBadgeColor}}50;">
                                        ${{statusText}}
                                    </div>
                                </div>
                                <div class="bet-eyes-box" style="border-color: ${{actionBadgeColor}}60; background: ${{actionBadgeBg}};">
                                    <div style="font-size: 0.7rem; color: ${{actionBadgeColor}}; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700;">
                                        Kelly3 推奨買い目: ${{s2.action_name}}
                                    </div>
                                    <div class="bet-eyes-text" style="color: #fff; font-size: 1.05rem;">${{s2.bettingEyesText || '買い目なし'}}</div>
                                </div>
                                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">
                                    <div style="color: #cbd5e1;">
                                        ${{jikuDisp}}${{jikuDisp && partnerDisp ? ' | ' : ''}}${{partnerDisp}}
                                    </div>
                                    <div style="color: #94a3b8;">
                                        Kelly3 中央値重視ポートフォリオ (Kelly3.ipynb)
                                    </div>
                                </div>
                                <div class="bet-result-details"></div>
                                <div style="margin-top: 10px; text-align: right; display: flex; justify-content: flex-end; gap: 8px; flex-wrap: wrap;">
                                    <button class="ipat-btn" data-eyes="${{s2.bettingEyesText}}" data-type="${{s2.rawType}}" data-round="${{raceData.round}}" data-axis="${{s2.axis2Num ? 2 : 1}}" data-place="${{raceData.place}}" data-weekday="${{raceData.weekday}}" onclick="event.stopPropagation(); showIpat(this)">🟢 即PAT</button>
                                    <button class="umaca-btn" data-eyes="${{s2.bettingEyesText}}" data-type="${{s2.rawType}}" data-round="${{raceData.round}}" data-axis="${{s2.axis2Num ? 2 : 1}}" data-place="${{raceData.place}}" data-weekday="${{raceData.weekday}}" onclick="event.stopPropagation(); showUmaca(this)">🟣 UMACA</button>
                                    <button class="smappy-btn" data-eyes="${{s2.bettingEyesText}}" data-type="${{s2.rawType}}" data-round="${{raceData.round}}" data-axis="${{s2.axis2Num ? 2 : 1}}" data-place="${{raceData.place}}" data-weekday="${{raceData.weekday}}" onclick="event.stopPropagation(); showSmappy(this)">📌 スマッピー</button>
                                </div>
                            </div>
                        `;
                    }}
                }} else {{
                    html += `
                        <div style="padding: 40px 20px; text-align: center; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px dashed rgba(255,255,255,0.1); color: var(--text-muted); margin-bottom: 20px;">
                            <div style="font-size: 1.5rem; margin-bottom: 10px;">📋</div>
                            <div style="font-size: 0.9rem; font-weight: 800; color: #fff; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.1em;">No Kelly3 Recommendation</div>
                            <div style="font-weight: 700; font-size: 0.8rem;">Kelly3 (Kelly3.ipynb) の買い目データがありません</div>
                        </div>
                    `;
                }}
            }}

            body.innerHTML = html;
            modal.style.display = 'flex';
            document.body.style.overflow = 'hidden';

            fetchRaceResults(raceId);
        }}

        function switchRecTab(raceId, tabName) {{
            currentActiveRecTab = tabName;
            showRecommendation(raceId, tabName);
        }}

        async function fetchRaceResults(raceId) {{
            console.log("[DEBUG] fetchRaceResults entry, raceId:", raceId);
            const container = document.getElementById('modal-body');
            
            // 既存の結果表示があれば削除
            const existing = document.getElementById('race-results-container');
            if (existing) existing.remove();

            const resultDiv = document.createElement('div');
            resultDiv.id = 'race-results-container';
            resultDiv.style.marginBottom = '15px'; // Reduce bottom margin
            resultDiv.style.padding = '8px 12px';  // Tighten padding
            resultDiv.style.background = 'rgba(74, 222, 128, 0.03)'; // Darker/lower background
            resultDiv.style.border = 'none'; // Remove border
            resultDiv.style.borderRadius = '12px';
            resultDiv.innerHTML = '<div style="text-align:center; font-size:0.8rem; color:var(--text-muted);">Fetching results...</div>';
            
            // モーダルの先頭に挿入
            container.insertBefore(resultDiv, container.firstChild);

            try {{
                const targetUrl = "https://race.sp.netkeiba.com/?pid=race_result&race_id=" + raceId;
                const proxyUrl = "https://cors.toshin-toshin1.workers.dev/" + targetUrl;
                
                const response = await fetch(proxyUrl);
                if (!response.ok) throw new Error('Proxy response not OK');
                
                const buffer = await response.arrayBuffer();
                const decoder = new TextDecoder('utf-8');
                const html = decoder.decode(buffer);
                
                const parser = new DOMParser();
                const doc = parser.parseFromString(html, 'text/html');
                
                const payoutData = {{ nums: {{}}, pays: {{}} }};
                const payoutTables = doc.querySelectorAll('.Payout_Detail_Table, .Pay_Table_01, .pay_table_01');
                
                payoutTables.forEach(table => {{
                    table.querySelectorAll('tr').forEach(tr => {{
                        const type = tr.querySelector('th')?.innerText.trim();
                        const resultCell = tr.querySelector('td.Result');
                        const payoutCell = tr.querySelector('td.Payout');
                        
                        if (type && resultCell && payoutCell) {{
                            // より確実に数字のみを抽出
                            const allNums = (resultCell.innerText.match(/\\d+/g) || [])
                                    .map(n => n.replace(/^0+/, ''));
                            
                            // 馬連などで "7 7 10 10" となるのを防ぐため基本はSetで重複排除するが、
                            // ワイドは "7-10, 10-4" のように同じ数字が別ペアで出ることがあるため重複を許容する
                            const numbers = (type === 'ワイド' || type.includes('ワイド')) ? allNums : [...new Set(allNums)];
                            
                            // 払戻金のパース（"110円110円140円" のような結合を解消）
                            const payRaw = payoutCell.innerText.trim();
                            const payTexts = payRaw.match(/[\\d,]+円/g) || [];
                            
                            if (numbers.length > 0) {{
                                payoutData.nums[type] = numbers;
                                payoutData.pays[type] = payTexts;
                            }}
                        }}
                    }});
                }});

                if (Object.keys(payoutData.nums).length === 0) {{
                    resultDiv.innerHTML = '<div style="text-align:center; font-size:0.8rem; color:#ffcc00;">Results not yet available.</div>';
                    return;
                }}

                // 結果表示の構築 (サマリー形式・グループ化)
                let htmlRes = `
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="color:#4ade80; font-weight:800; font-size:0.7rem; text-transform:uppercase;">Confirmed Results</span>
                        <span style="font-size:0.65rem; color:var(--text-muted);">${{doc.title.split('|')[0].trim()}}</span>
                    </div>
                    <div style="display:flex; flex-direction:column; gap:4px;">
                `;

                const renderBlock = (type) => {{
                    const nums = payoutData.nums[type] || payoutData.nums[type.replace('3連', '三連')];
                    if (!nums) return '';
                    
                    const displayType = type.replace('3連', '3連');
                    let separator = ',';
                    if (type.includes('単') || type.includes('枠連')) separator = '→';
                    else if (type.includes('3連複') || type.includes('馬連') || type.includes('ワイド')) separator = '-';
                    
                    let numbersHtml = '';
                    if (type === 'ワイド' && nums.length >= 2) {{
                        const pairs = [];
                        for (let i = 0; i < nums.length; i += 2) {{
                            if (nums[i+1]) pairs.push(`${{nums[i]}}-${{nums[i+1]}}`);
                        }}
                        numbersHtml = pairs.map(p => `<span style="color:#4ade80; font-size:0.72rem; font-weight:900;">${{p}}</span>`).join('<span style="color:var(--text-muted); font-size:0.55rem; margin:0 1px;">,</span>');
                    }} else {{
                        numbersHtml = nums.map((n, i) => `
                            <span style="color:#4ade80; font-size:0.72rem; font-weight:900;">${{n}}</span>
                            ${{i < nums.length - 1 ? `<span style="color:var(--text-muted); font-size:0.55rem; font-weight:bold;">${{separator}}</span>` : ''}}
                        `).join('');
                    }}

                    return `<div style="background:rgba(255,255,255,0.03); padding:4px 8px; border-radius:6px; flex:1; min-width:60px;">
                        <div style="font-size:0.55rem; color:var(--text-muted); margin-bottom:1px;">${{displayType}}</div>
                        <div style="display:flex; gap:2px; flex-wrap:wrap; align-items:center;">${{numbersHtml}}</div>
                    </div>`;
                }};

                const groups = [
                    ['単勝', '複勝'],
                    ['馬連', '馬単', '枠連'],
                    ['ワイド', '3連複', '3連単']
                ];

                groups.forEach(g => {{
                    const rowHtml = g.map(t => renderBlock(t)).join('');
                    if (rowHtml) htmlRes += `<div style="display:flex; gap:4px;">${{rowHtml}}</div>`;
                }});
                htmlRes += '</div>';
                resultDiv.innerHTML = htmlRes;

                // 当たり判定の実行
                checkHits(payoutData);

            }} catch (error) {{
                console.error("[ERROR]", error);
                resultDiv.innerHTML = '<div style="text-align:center; font-size:0.8rem; color:#ef4444;">Failed to load results/status.</div>';
            }}
        }}

        function checkHits(payoutData) {{
            const strategyItems = document.querySelectorAll('.strategy-item-modal');
            strategyItems.forEach(item => {{
                const type = item.getAttribute('data-strategy-type');
                const eyesText = item.querySelector('.bet-eyes-text')?.innerText.trim();
                if (!eyesText || eyesText === '--') return;

                const resultArea = item.querySelector('.bet-result-details');
                if (resultArea) resultArea.innerHTML = ''; // Reset

                const eyesElem = item.querySelector('.bet-eyes-text');
                const eyesBox = item.querySelector('.bet-eyes-box');
                if (eyesElem) eyesElem.innerHTML = eyesText; // Clear previous HIT mark

                // 券種に応じたキーを抽出 (例: "3連単-2頭軸マルチ" -> "3連単")
                let baseType = "";
                const types = ["単勝", "複勝", "枠連", "枠単", "馬連", "馬単", "ワイド", "3連複", "3連単"];
                for (const t of types) {{
                    if (type.includes(t)) {{
                        baseType = t;
                        break;
                    }}
                }}
                
                if (!baseType) return;

                const normBaseType = baseType.replace('3連', '三連');
                const winNums = (payoutData.nums[baseType] || payoutData.nums[normBaseType] || []);
                const winPays = (payoutData.pays[baseType] || payoutData.pays[normBaseType] || []);
                
                let isHit = false;
                let totalPay = 0;
                let eyesCount = 0;

                // 買い目数の正確な計算 (正規化して3連/3連を統一)
                const normType = type;
                
                // 軸馬と相手馬のパース (マルチの '↔'、流しの '→' の両方に対応)
                const sep = eyesText.includes('↔') ? '↔' : (eyesText.includes('→') ? '→' : '');
                let parsedAxes = [];
                let parsedPartners = [];
                if (sep) {{
                    const parts = eyesText.split(sep);
                    parsedPartners = parts.pop().split(',').map(s => s.trim().replace(/^0+/, '')).filter(Boolean);
                    parsedAxes = parts.join(',').split(',').map(s => s.trim().replace(/^0+/, '')).filter(Boolean);
                }}
                const axisCount = parsedAxes.length;
                const partnersCount = parsedPartners.length;

                // Kelly3 ポートフォリオ表示用の券種も扱う
                if (normType.includes("フォーメーション") && eyesText.includes("→")) {{
                    // 軸1頭 → 2着候補 → 3着候補 (3連単)
                    const stages = eyesText.split("→").map(s =>
                        s.split(",").map(t => String(t).trim().replace(/^0+/, "")).filter(Boolean)
                    );
                    if (stages.length === 3 && stages[1].length > 0 && stages[2].length > 0) {{
                        eyesCount = stages[1].length * stages[2].length
                            - stages[1].filter(v => stages[2].includes(v)).length;
                        parsedAxes = stages[0];
                        parsedPartners = [];
                    }}
                }} else if (normType.includes("折り返し")) {{
                    // 馬単折り返し (a  b)  2点
                    const parts = eyesText.split("↔");
                    parsedAxes = parts[0].split(",").map(s => String(s).trim().replace(/^0+/, "")).filter(Boolean);
                    parsedPartners = parts[1].split(",").map(s => String(s).trim().replace(/^0+/, "")).filter(Boolean);
                    eyesCount = parts.length === 2 ? 2 : partnersCount;
                }} else if (normType.includes("単勝") || normType.includes("複勝")) {{
                    eyesCount = 1;
                }}
                
                if (normType.includes("2通り")) {{
                    eyesCount = 2;
                }} else if (normType.includes("単勝") || normType.includes("複勝")) {{
                    eyesCount = 1;
                }} else if (normType.includes("BOX")) {{
                    const n = eyesText.split(',').length;
                    if (normType.includes("3連単")) eyesCount = n * (n-1) * (n-2);
                    else if (normType.includes("3連複")) eyesCount = n * (n-1) * (n-2) / 6;
                    else if (normType.includes("馬単")) eyesCount = n * (n-1);
                    else if (normType.includes("馬連") || normType.includes("ワイド")) eyesCount = n * (n-1) / 2;
                }} else if (normType.includes("マルチ")) {{
                    if (normType.includes("3連単")) {{
                        if (axisCount === 1) eyesCount = 3 * partnersCount * (partnersCount - 1);
                        else eyesCount = 6 * partnersCount;
                    }} else if (normType.includes("馬単")) {{
                        eyesCount = 2 * partnersCount;
                    }} else {{
                        eyesCount = 2 * partnersCount;
                    }}
                }} else {{
                    // 流し
                    if (normType.includes("3連単")) {{
                        if (axisCount === 1) eyesCount = partnersCount * (partnersCount - 1);
                        else eyesCount = partnersCount;
                    }} else if (normType.includes("3連複")) {{
                        if (axisCount === 1) eyesCount = (partnersCount * (partnersCount - 1)) / 2;
                        else eyesCount = partnersCount;
                    }} else {{
                        eyesCount = partnersCount;
                    }}
                }}

                if (winNums.length > 0) {{
                    const predictedSet = eyesText.split(/[→↔,]/).map(s => s.trim().replace(/^0+/, '')).filter(Boolean);
                    const isMulti = normType.includes("マルチ") || normType.includes("BOX") || normType.includes("3連複") || normType.includes("馬連") || normType.includes("ワイド");

                    if (normType.includes("2通り")) {{
                        const m = eyesText.match(/\d+/g);
                        if (m && m.length >= 3 && winNums.length >= 3) {{
                            const a1 = String(parseInt(m[0]));
                            const a2 = String(parseInt(m[1]));
                            const a3 = String(parseInt(m[2]));
                            isHit = (winNums[0] === a1 && winNums[1] === a2 && winNums[2] === a3) ||
                                    (winNums[0] === a2 && winNums[1] === a1 && winNums[2] === a3);
                        }}
                    }} else if (baseType === "単勝") {{
                        isHit = (predictedSet[0] === winNums[0]);
                    }} else if (baseType === "複勝") {{
                        isHit = winNums.some(n => predictedSet.includes(n));
                    }} else if (isMulti) {{
                        if (baseType === "ワイド") {{
                            for (let i = 0; i < winNums.length; i += 2) {{
                                if (predictedSet.includes(winNums[i]) && predictedSet.includes(winNums[i+1])) {{
                                    isHit = true; break;
                                }}
                            }}
                        }} else if (normType.includes("マルチ")) {{
                            const axes = parsedAxes;
                            const partners = parsedPartners;
                            
                            const hasAllAxes = axes.every(a => winNums.includes(a));
                            const remainingWinNums = winNums.filter(n => !axes.includes(n));
                            const allRemainingInPartners = remainingWinNums.every(n => partners.includes(n));
                            
                            isHit = hasAllAxes && allRemainingInPartners && (remainingWinNums.length + axes.length === winNums.length);
                        }} else {{
                            isHit = winNums.every(n => predictedSet.includes(n));
                        }}
                    }} else {{
                        // Nagashi (Flow) logic
                        const axes = parsedAxes;
                        const partners = parsedPartners;
                        
                        const axesMatch = axes.every((a, i) => i < winNums.length && winNums[i] === a);
                        const remainingWinNums = winNums.slice(axes.length);
                        const partnersMatch = remainingWinNums.every(n => partners.includes(n));
                        
                        isHit = axesMatch && partnersMatch && (axes.length + remainingWinNums.length === winNums.length);
                    }}

                    if (isHit) {{
                        if (baseType === "複勝") {{
                            const hitIdx = winNums.indexOf(predictedSet[0]);
                            totalPay = parseInt((winPays[hitIdx] || winPays[0] || '0').replace(/,/g, '')) || 0;
                        }} else if (baseType === "ワイド") {{
                            for (let i = 0; i < winNums.length; i += 2) {{
                                if (predictedSet.includes(winNums[i]) && predictedSet.includes(winNums[i+1])) {{
                                    totalPay += parseInt((winPays[i/2] || '0').replace(/,/g, '')) || 0;
                                }}
                            }}
                        }} else {{
                            totalPay = parseInt((winPays[0] || '0').replace(/,/g, '')) || 0;
                        }}
                    }}
                }}

                const investment = eyesCount * 100;
                const profit = totalPay - investment;

                if (resultArea) {{
                    resultArea.innerHTML = `
                        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.05); display: flex; justify-content: space-between; font-size: 0.85rem;">
                            <div><span style="color:var(--text-muted);">Bets:</span> <span style="color:#fff; font-weight:700;">${{eyesCount}}点(${{investment}}円)</span></div>
                            <div><span style="color:var(--text-muted);">Payout:</span> <span style="color:#fff; font-weight:700;">${{totalPay.toLocaleString()}}円</span></div>
                            <div><span style="color:var(--text-muted);">Profit:</span> <span style="color:${{profit >= 0 ? '#4ade80' : '#ef4444'}}; font-weight:800;">${{profit < 0 ? '-' : ''}}${{Math.abs(profit).toLocaleString()}}円</span></div>
                        </div>
                    `;
                }}

                if (isHit) {{
                    if (eyesBox) {{
                        const hitBadge = document.createElement('div');
                        hitBadge.innerHTML = '🎯 HIT';
                        hitBadge.style.cssText = 'position:absolute; top:8px; right:8px; background:#4ade80; color:#064e3b; font-size:0.75rem; font-weight:900; padding:2px 8px; border-radius:12px; box-shadow:0 2px 10px rgba(0,0,0,0.3); z-index:10;';
                        eyesBox.appendChild(hitBadge);
                    }}
                }}
                // --- Dynamic Font Scaling For Betting Eyes (Already declared eyesElem/eyesBox) ---
                if (eyesElem && eyesBox) {{
                    let fontSize = 1.8; // rem
                    eyesElem.style.fontSize = fontSize + 'rem';
                    // モーダル表示後に幅が確定するため、少し待つ必要がある
                    setTimeout(() => {{
                        while (eyesElem.scrollWidth > eyesBox.clientWidth - 40 && fontSize > 0.6) {{
                            fontSize -= 0.1;
                            eyesElem.style.fontSize = fontSize + 'rem';
                        }}
                    }}, 0);
                }}
            }});
        }}


        function getSmappySiki(type) {{
            if (type.includes('3連単')) return '8';
            if (type.includes('3連複')) return '7';
            if (type.includes('馬単')) return '6';
            if (type.includes('ワイド')) return '5';
            if (type.includes('馬連')) return '4';
            if (type.includes('枠連')) return '3';
            if (type.includes('複勝')) return '2';
            return '1';
        }}

        function getSmappyHou(type, axisCount) {{
            if (type.includes('BOX')) return '2';
            if (type.includes('マルチ') || type.includes('MULTI')) return '7';
            if (type.includes('2通り') || (type.includes('3連単') && axisCount >= 2)) return '6';
            if (type.includes('3連単')) return '3';
            return '3';
        }}

        function parseSmappyEyes(text, stratType) {{
            text = text.trim();
            if (stratType.includes('2通り')) {{
                var matches = text.match(/\d+/g);
                if (matches && matches.length >= 3) {{
                    var h1 = parseInt(matches[0]);
                    var h2 = parseInt(matches[1]);
                    var h3 = parseInt(matches[2]);
                    return {{axes: [h1, h2], partners: [h3], is2Touri: true}};
                }}
            }}
            // 「01 ↔ 12 → 04」形式: 3連単-2通り（↔と→の両方を含む新表示形式）
            if (text.indexOf(' ↔ ') >= 0 && (text.indexOf(' → ') >= 0 || text.indexOf('→') >= 0)) {{
                var arrowIdx = text.indexOf(' → ');
                if (arrowIdx < 0) arrowIdx = text.indexOf('→');
                var leftPart = text.slice(0, arrowIdx).trim(); // 「01 ↔ 12」
                var rightPart = text.slice(arrowIdx).replace(/^\s*→\s*/, '').trim(); // 「04」
                var axNums = leftPart.split(' ↔ ').map(function(s){{ return parseInt(s.trim()); }}).filter(function(n){{ return !isNaN(n); }});
                var h3 = parseInt(rightPart.replace(/[^0-9]/g, ''));
                if (axNums.length >= 2 && !isNaN(h3)) {{
                    return {{axes: axNums, partners: [h3], is2Touri: true}};
                }}
            }}
            // 「05 ↔ 10」形式: 馬単-折り返し（↔のみ、→なし）
            if ((text.indexOf(' ↔ ') >= 0 || text.indexOf('↔') >= 0) && text.indexOf('→') < 0) {{
                var nums = text.match(/\d+/g);
                if (nums && nums.length === 2) {{
                    var fa = parseInt(nums[0]);
                    var fb = parseInt(nums[1]);
                    return {{axes: [fa, fb], partners: [], isOrikaeshi: true}};
                }}
            }}
            if (stratType.includes('BOX')) {{
                var clean = text.replace(/BOX/gi, '').trim();
                var all = clean.split(',').map(function(s){{ return parseInt(s.trim()); }}).filter(function(n){{ return !isNaN(n); }});
                return {{axes: all, partners: []}};
            }}
            var axesStr, partnersStr;
            if (text.indexOf(' ― ') >= 0) {{
                var dp = text.split(' ― ');
                partnersStr = dp.pop();
                axesStr = dp.join(' ― ');
            }} else if (text.indexOf(' - ') >= 0) {{
                var dp = text.split(' - ');
                partnersStr = dp.pop();
                axesStr = dp.join(' - ');
            }} else if (text.indexOf(' \u2192 ') >= 0) {{
                var ap = text.split(' \u2192 ');
                partnersStr = ap.pop();
                axesStr = ap.join(' \u2192 ');
            }} else if (text.indexOf('→') >= 0) {{
                var ap = text.split('→');
                partnersStr = ap.pop();
                axesStr = ap.join('→');
            }} else {{
                return {{axes: [parseInt(text)], partners: []}};
            }}
            var partners = partnersStr.split(',').map(function(s){{ return parseInt(s.trim()); }}).filter(function(n){{ return !isNaN(n); }});
            var axes;
            if (axesStr.indexOf(' \u2192 ') >= 0 || axesStr.indexOf('→') >= 0) {{
                axes = axesStr.split(/ \u2192 |→/).map(function(s){{ return parseInt(s.trim()); }}).filter(function(n){{ return !isNaN(n); }});
            }} else if (axesStr.indexOf(' ↔ ') >= 0 || axesStr.indexOf('↔') >= 0) {{
                axes = axesStr.split(/ ↔ |↔/).map(function(s){{ return parseInt(s.trim()); }}).filter(function(n){{ return !isNaN(n); }});
            }} else if (axesStr.indexOf(',') >= 0) {{
                axes = axesStr.split(',').map(function(s){{ return parseInt(s.trim()); }}).filter(function(n){{ return !isNaN(n); }});
            }} else {{
                axes = [parseInt(axesStr.trim())];
            }}
            return {{axes: axes, partners: partners}};
        }}

        function genSmappyBml(venueCode, venueName, weekday, raceRound, siki, hou, axes, partners) {{
            var rawSteps = [venueCode, raceRound, siki];
            var simple = (siki === '1' || siki === '2' || siki === '9');
            if (!simple && hou) rawSteps.push(hou);
            (axes || []).forEach(function(a) {{ rawSteps.push(String(a)); }});
            (partners || []).forEach(function(p) {{ rawSteps.push(String(p)); }});
            var stepsJSON = JSON.stringify(rawSteps);
            var venueJSON = JSON.stringify(venueName || "");
            var weekdayJSON = JSON.stringify(weekday || "");
            var part1 = "javascript:void((function(){{ var s=" + stepsJSON + "; var vn=" + venueJSON + "; var wd=" + weekdayJSON + "; ";
            var part2 = {_smappy_part2_js_json};
            // part2 は単独のIIFE文として終わっているため、
            // 末尾の ";" を取り除いてから void() を正しく閉じる
            return part1 + part2.slice(0, -1) + ")";
        }}

        function showSmappy(btn) {{
            var prev = document.querySelector('.smappy-popup');
            if (prev) prev.remove();
            
            var eyes = btn.getAttribute('data-eyes');
            var type = btn.getAttribute('data-type');
            var round = btn.getAttribute('data-round');
            var axisCount = parseInt(btn.getAttribute('data-axis')) || 1;
            if (!eyes || eyes === '--') {{ alert('買い目がありません'); return; }}
            
            var siki = getSmappySiki(type);
            var hou = getSmappyHou(type, axisCount);
            var parsed = parseSmappyEyes(eyes, type);
            
            // 会場情報をボタンから直接取得する（確実な判定）
            var currentPlace = btn.getAttribute('data-place') || "";
            var weekday = btn.getAttribute('data-weekday') || "";
            
            var vCodes = {{ "札幌":"01","函館":"02","福島":"03","新潟":"04","東京":"05","中山":"06","中京":"07","京都":"08","阪神":"09","小倉":"10" }};
            var todayPlaces = [];
            for (var k in currentData) {{
                var p = currentData[k].place;
                if (!todayPlaces.includes(p)) todayPlaces.push(p);
            }}
            todayPlaces.sort(function(a, b) {{ return (vCodes[a] || "99") - (vCodes[b] || "99"); }});
            window._smappyPlaces = todayPlaces;
            var vIdx = todayPlaces.indexOf(currentPlace);
            if (vIdx < 0) vIdx = 0;

            var popup = document.createElement('div');
            popup.className = 'smappy-popup';
            popup.innerHTML = `
                <div class="smappy-tabs">
                    <div id="tab-ios" class="smappy-tab active" onclick="switchSmappyTab('ios')">📱 iPhone (Scriptable)</div>
                    <div id="tab-pc" class="smappy-tab" onclick="switchSmappyTab('pc')">💻 PC / Android</div>
                </div>

                <div style="margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                    <label style="font-size: 0.7rem; color: var(--text-muted); font-weight: 800;">会場判定:</label>
                    <select id="smappy-venue" style="flex: 1; padding: 4px 8px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #fff; border-radius: 6px; font-size: 0.75rem;">
                        <option value="0" ${{vIdx==0?'selected':''}}>0 (1場目: ${{todayPlaces[0] || '?'}})</option>
                        <option value="1" ${{vIdx==1?'selected':''}}>1 (2場目: ${{todayPlaces[1] || '?'}})</option>
                        <option value="2" ${{vIdx==2?'selected':''}}>2 (3場目: ${{todayPlaces[2] || '?'}})</option>
                    </select>
                </div>

                <div id="panel-pc" style="display: none;">
                    <div class="step-box" style="background: rgba(99, 102, 241, 0.08); border-left: 3px solid #6366f1; padding: 8px 10px; margin-bottom: 10px; border-radius: 4px;">
                        <span class="step-title" style="color: #818cf8; font-weight: 800; font-size: 0.75rem;">🌟 コピー＆ブックマークで自動入力</span>
                        <div class="step-desc" style="font-size: 0.7rem; color: #94a3b8; margin-top: 2px;">
                            下のボタンで買い目をコピーし、JRA画面でブックマークを押すだけ！
                        </div>
                    </div>
                    <button onclick="copySmappyPayload()" style="width: 100%; padding: 13px; background: linear-gradient(135deg, #6366f1, #4f46e5); color: #fff; border: none; border-radius: 8px; font-weight: 800; font-size: 0.88rem; cursor: pointer; box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35); margin-bottom: 10px; display: flex; align-items: center; justify-content: center; gap: 6px;">
                        📋 買い目をコピーしてJRAへ
                    </button>
                    <details style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 8px; font-size: 0.7rem;">
                        <summary style="cursor: pointer; color: #94a3b8; font-weight: 700; outline: none; user-select: none;">📌 自動入力ブックマーク（初回1回のみ登録）</summary>
                        <div style="margin-top: 8px; line-height: 1.6; color: #cbd5e1;">
                            <b>1度登録すれば全てのレースで使えます！</b><br>
                            下のボタンをブックマークバーにドラッグ登録（またはコードをコピーしてブックマーク追加）してください。
                        </div>
                        <div style="display: flex; gap: 6px; margin-top: 8px;">
                            <a id="smappy-fixed-bml-link" href="#" style="flex: 1; text-align: center; padding: 8px; background: #334155; color: #fff; font-weight: 700; font-size: 0.75rem; border-radius: 6px; text-decoration: none;">📌 ドラッグ登録</a>
                            <button onclick="copyFixedBml()" style="padding: 8px 10px; background: #1e293b; color: #cbd5e1; border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; font-weight: 600; font-size: 0.72rem; cursor: pointer;">📋 コードコピー</button>
                        </div>
                    </details>
                </div>

                <div id="panel-ios" style="display: block;">
                    <div class="step-box" style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10b981; padding: 8px 10px; margin-bottom: 10px; border-radius: 4px;">
                        <span class="step-title" style="color: #10b981; font-weight: 800; font-size: 0.75rem;">🌟 ワンタップで自動入力</span>
                        <div class="step-desc" style="font-size: 0.7rem; color: #94a3b8; margin-top: 2px;">
                            下のボタンを押すとScriptableが起動し、JRAスマッピーで自動選択が完了します！
                        </div>
                    </div>
                    <button onclick="launchSmappyScriptable()" style="width: 100%; padding: 13px; background: linear-gradient(135deg, #10b981, #059669); color: #fff; border: none; border-radius: 8px; font-weight: 800; font-size: 0.88rem; cursor: pointer; box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35); margin-bottom: 10px; display: flex; align-items: center; justify-content: center; gap: 6px;">
                        🚀 Scriptableで投票を起動
                    </button>
                    <details style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 8px; font-size: 0.7rem;">
                        <summary style="cursor: pointer; color: #94a3b8; font-weight: 700; outline: none; user-select: none;">⚙️ 初回設定（1分・初回のみ）</summary>
                        <div style="margin-top: 8px; line-height: 1.6; color: #cbd5e1;">
                            1. App Storeで <b>Scriptable</b> アプリ（無料）をインストール<br>
                            2. 下のボタンでスクリプトコードをコピー<br>
                            3. Scriptableで右上の「<b>＋</b>」を押し、貼り付けて名前を「<b>スマッピー</b>」で保存
                        </div>
                        <button onclick="copyScriptableAppCode()" style="margin-top: 8px; width: 100%; padding: 8px; background: #334155; color: #f8fafc; border: none; border-radius: 6px; font-weight: 700; font-size: 0.75rem; cursor: pointer;">📋 Scriptable用コードをコピー</button>
                    </details>
                </div>
                
                <div style="font-size: 0.55rem; color: var(--text-muted); margin-top: 10px; text-align: center; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 8px;">
                    JRAスマッピー投票（QR作成）に自動連携します
                </div>
            `;
            btn.parentElement.appendChild(popup);

            window._smappyParsed = {{weekday: weekday, round: round, siki: siki, hou: hou, axes: parsed.axes, partners: parsed.partners, is2Touri: parsed.is2Touri, isOrikaeshi: parsed.isOrikaeshi}};
            
            var fixedLink = document.getElementById('smappy-fixed-bml-link');
            if (fixedLink) {{
                fixedLink.href = {_smappy_fixed_bml_json};
            }}

            function updateBml() {{
                var v = document.getElementById('smappy-venue').value;
                var placeName = (window._smappyPlaces && window._smappyPlaces[v]) || "";
                var bml = genSmappyBml(v, placeName, weekday, round, siki, hou, parsed.axes, parsed.partners);
                var linkEl = document.getElementById('smappy-bml-link');
                if (linkEl) linkEl.href = bml;
                window._smappyBml = bml;
            }}
            document.getElementById('smappy-venue').addEventListener('change', updateBml);
            updateBml();
        }}

        function switchSmappyTab(tab) {{
            document.getElementById('tab-pc').className = 'smappy-tab' + (tab === 'pc' ? ' active' : '');
            document.getElementById('tab-ios').className = 'smappy-tab' + (tab === 'ios' ? ' active' : '');
            document.getElementById('panel-pc').style.display = (tab === 'pc' ? 'block' : 'none');
            document.getElementById('panel-ios').style.display = (tab === 'ios' ? 'block' : 'none');
        }}

        function copySmappyPayload() {{
            var venueEl = document.getElementById('smappy-venue');
            if (!venueEl || !window._smappyParsed) return;
            var v = venueEl.value;
            var p = window._smappyParsed;
            var placeName = (window._smappyPlaces && window._smappyPlaces[v]) || "";

            var payload;
            if (p.is2Touri && p.axes && p.axes.length >= 2 && p.partners && p.partners.length >= 1) {{
                var h1 = String(p.axes[0]);
                var h2 = String(p.axes[1]);
                var h3 = String(p.partners[0]);
                payload = {{
                    bets: [
                        {{
                            steps: [v, p.round, p.siki, "0", h1, h2, h3],
                            venueName: placeName,
                            weekday: p.weekday || ""
                        }},
                        {{
                            steps: [v, p.round, p.siki, "0", h2, h1, h3],
                            venueName: placeName,
                            weekday: p.weekday || ""
                        }}
                    ]
                }};
            }} else if (p.isOrikaeshi && p.axes && p.axes.length >= 2) {{
                // 馬単-折り返し: 通常の2点買い目として展開
                var fa = String(p.axes[0]);
                var fb = String(p.axes[1]);
                payload = {{
                    bets: [
                        {{
                            steps: [v, p.round, p.siki, "0", fa, fb],
                            venueName: placeName,
                            weekday: p.weekday || ""
                        }},
                        {{
                            steps: [v, p.round, p.siki, "0", fb, fa],
                            venueName: placeName,
                            weekday: p.weekday || ""
                        }}
                    ]
                }};
            }} else {{
                var rawSteps = [v, p.round, p.siki];
                var simple = (p.siki === '1' || p.siki === '2' || p.siki === '9');
                if (!simple && p.hou) rawSteps.push(p.hou);
                (p.axes || []).forEach(function(a) {{ rawSteps.push(String(a)); }});
                (p.partners || []).forEach(function(pt) {{ rawSteps.push(String(pt)); }});

                payload = {{
                    steps: rawSteps,
                    venueName: placeName,
                    weekday: p.weekday || ""
                }};
            }}

            var jsonStr = JSON.stringify(payload);
            var t = document.createElement('textarea');
            t.value = jsonStr;
            document.body.appendChild(t);
            t.select();
            document.execCommand('copy');
            document.body.removeChild(t);

            if (confirm('買い目データをコピーしました！\\nJRA通常投票（会場画面）を開きますか？\\n（開いた画面で登録したブックマークを押してください）')) {{
                window.open('https://qrcode.jra.go.jp/pw_982_i.cgi', '_blank');
            }}
        }}

        function copyFixedBml() {{
            var bml = {_smappy_fixed_bml_json};
            var t = document.createElement('textarea');
            t.value = bml;
            document.body.appendChild(t);
            t.select();
            document.execCommand('copy');
            document.body.removeChild(t);
            alert('固定ブックマーク用コードをコピーしました！\\nブラウザのブックマーク登録画面のURL欄に貼り付けて保存してください。');
        }}

        function showIpat(btn) {{
            var prev = document.querySelector('.smappy-popup, .ipat-popup');
            if (prev) prev.remove();

            var eyes = btn.getAttribute('data-eyes');
            var type = btn.getAttribute('data-type');
            var round = btn.getAttribute('data-round');
            var axisCount = parseInt(btn.getAttribute('data-axis')) || 1;
            var unitAmount = parseInt(btn.getAttribute('data-unit')) || window._globalUnitAmount || 100;
            var totalAmount = parseInt(btn.getAttribute('data-total')) || unitAmount;
            if (!eyes || eyes === '--') {{ alert('買い目がありません'); return; }}

            var siki = getSmappySiki(type);
            var hou = getSmappyHou(type, axisCount);
            var parsed = parseSmappyEyes(eyes, type);
            var currentPlace = btn.getAttribute('data-place') || "";
            var weekday = btn.getAttribute('data-weekday') || "";

            var vCodes = {{ "札幌":"01","函館":"02","福島":"03","新潟":"04","東京":"05","中山":"06","中京":"07","京都":"08","阪神":"09","小倉":"10" }};
            var todayPlaces = [];
            for (var k in currentData) {{
                var p = currentData[k].place;
                if (!todayPlaces.includes(p)) todayPlaces.push(p);
            }}
            todayPlaces.sort(function(a, b) {{ return (vCodes[a] || "99") - (vCodes[b] || "99"); }});
            window._ipatPlaces = todayPlaces;
            var vIdx = todayPlaces.indexOf(currentPlace);
            if (vIdx < 0) vIdx = 0;

            var popup = document.createElement('div');
            popup.className = 'smappy-popup';
            popup.innerHTML = `
                <div style="font-weight: 800; font-size: 0.9rem; color: #10b981; margin-bottom: 10px; display: flex; align-items: center; gap: 6px;">
                    🟢 JRA即PAT 自動投票 (Scriptable連携)
                </div>

                <div style="margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                    <label style="font-size: 0.7rem; color: var(--text-muted); font-weight: 800;">会場判定:</label>
                    <select id="ipat-venue" style="flex: 1; padding: 4px 8px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #fff; border-radius: 6px; font-size: 0.75rem;">
                        <option value="0" ${{vIdx==0?'selected':''}}>0 (1場目: ${{todayPlaces[0] || '?'}})</option>
                        <option value="1" ${{vIdx==1?'selected':''}}>1 (2場目: ${{todayPlaces[1] || '?'}})</option>
                        <option value="2" ${{vIdx==2?'selected':''}}>2 (3場目: ${{todayPlaces[2] || '?'}})</option>
                    </select>
                </div>

                <div style="margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                    <label style="font-size: 0.7rem; color: var(--text-muted); font-weight: 800;">1点金額:</label>
                    <input type="number" id="ipat-unit-amount" value="${{unitAmount}}" step="100" min="100" style="width: 80px; padding: 4px 8px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #fff; border-radius: 6px; font-size: 0.75rem; text-align: right;" oninput="window._globalUnitAmount=parseInt(this.value)||100; syncGlobalUnitAmount(); updateIpatTotal();">
                    <span style="font-size: 0.75rem; color: #cbd5e1;">円</span>
                    <span style="flex: 1; text-align: right; font-size: 0.75rem; color: #94a3b8;">
                        合計: <strong id="ipat-total-disp" style="color: #4ade80;">${{totalAmount.toLocaleString()}}円</strong>
                    </span>
                </div>

                <div class="step-box" style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10b981; padding: 8px 10px; margin-bottom: 10px; border-radius: 4px;">
                    <span class="step-title" style="color: #10b981; font-weight: 800; font-size: 0.75rem;">🌟 ワンタップで自動ログイン＆入力</span>
                    <div class="step-desc" style="font-size: 0.7rem; color: #94a3b8; margin-top: 2px;">
                        下のボタンを押すとScriptableが起動し、即PATへのログイン・買い目セット・確認画面まで自動で進みます！
                    </div>
                </div>

                <button onclick="launchIpatScriptable()" style="width: 100%; padding: 13px; background: linear-gradient(135deg, #10b981, #059669); color: #fff; border: none; border-radius: 8px; font-weight: 800; font-size: 0.88rem; cursor: pointer; box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35); margin-bottom: 10px; display: flex; align-items: center; justify-content: center; gap: 6px;">
                    🚀 即PATで自動投票を実行
                </button>

                <details style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 8px; font-size: 0.7rem; margin-bottom: 8px;">
                    <summary style="cursor: pointer; color: #94a3b8; font-weight: 700; outline: none; user-select: none;">⚙️ 初回設定（1分・初回のみ）</summary>
                    <div style="margin-top: 8px; line-height: 1.6; color: #cbd5e1;">
                        1. App Storeで <b>Scriptable</b> アプリ（無料）をインストール<br>
                        2. 下のボタンで即PAT用スクリプトコードをコピー<br>
                        3. Scriptableで右上の「<b>＋</b>」を押し、貼り付けて名前を「<b>即PAT</b>」で保存<br>
                        4. 初回実行時のみ、加入者番号・暗証番号・P-ARS番号を入力（iOSのKeychainに安全に暗号化保存されます）
                    </div>
                    <button onclick="copyIpatAppCode()" style="margin-top: 8px; width: 100%; padding: 8px; background: #334155; color: #f8fafc; border: none; border-radius: 6px; font-weight: 700; font-size: 0.75rem; cursor: pointer;">📋 即PAT用Scriptableコードをコピー</button>
                </details>

                <div style="font-size: 0.65rem; color: #94a3b8; text-align: center; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 8px; line-height: 1.4;">
                    ※ 誤投票防止のため、最終確認画面（金額・暗証番号入力済）で停止します。<br>内容を目視確認して【投票】ボタンを押してください。
                </div>
            `;
            btn.parentElement.appendChild(popup);

            window._ipatParsed = {{
                placeName: currentPlace,
                weekday: weekday,
                round: round,
                siki: siki,
                hou: hou,
                axes: parsed.axes,
                partners: parsed.partners,
                is2Touri: parsed.is2Touri,
                isOrikaeshi: parsed.isOrikaeshi,
                isMulti: (type || "").indexOf('マルチ') >= 0,
                baseUnit: unitAmount,
                baseTotal: totalAmount
            }};
        }}

        function updateIpatTotal() {{
            if (!window._ipatParsed) return;
            var inp = document.getElementById('ipat-unit-amount');
            var disp = document.getElementById('ipat-total-disp');
            if (!inp || !disp) return;
            var u = parseInt(inp.value) || 100;
            var baseU = window._ipatParsed.baseUnit || 100;
            var ratio = u / baseU;
            var tot = Math.round((window._ipatParsed.baseTotal || u) * ratio);
            disp.innerText = tot.toLocaleString() + '円';
        }}

        function launchIpatScriptable() {{
            var venueEl = document.getElementById('ipat-venue');
            if (!venueEl || !window._ipatParsed) return;
            var v = venueEl.value;
            var p = window._ipatParsed;
            var placeName = (window._ipatPlaces && window._ipatPlaces[v]) || p.placeName || "";

            var unitInp = document.getElementById('ipat-unit-amount');
            var unitVal = unitInp ? (parseInt(unitInp.value) || 100) : 100;
            var baseU = p.baseUnit || 100;
            var ratio = unitVal / baseU;
            var totalVal = Math.round((p.baseTotal || unitVal) * ratio);

            var payload;
            if (p.is2Touri && p.axes && p.axes.length >= 2 && p.partners && p.partners.length >= 1) {{
                var h1 = p.axes[0];
                var h2 = p.axes[1];
                var h3 = p.partners[0];
                var bet1 = {{
                    steps: [v, p.round, p.siki, "0", String(h1), String(h2), String(h3)],
                    venueName: placeName,
                    placeName: placeName,
                    round: p.round,
                    raceNo: p.round,
                    siki: p.siki,
                    hou: "0",
                    axes: [h1, h2],
                    partners: [h3],
                    isMulti: false,
                    weekday: p.weekday || "",
                    unitAmount: unitVal,
                    totalAmount: unitVal
                }};
                var bet2 = {{
                    steps: [v, p.round, p.siki, "0", String(h2), String(h1), String(h3)],
                    venueName: placeName,
                    placeName: placeName,
                    round: p.round,
                    raceNo: p.round,
                    siki: p.siki,
                    hou: "0",
                    axes: [h2, h1],
                    partners: [h3],
                    isMulti: false,
                    weekday: p.weekday || "",
                    unitAmount: unitVal,
                    totalAmount: unitVal
                }};
                payload = {{
                    bets: [bet1, bet2],
                    unitAmount: unitVal,
                    totalAmount: totalVal
                }};
            }} else {{
                var rawSteps = [v, p.round, p.siki];
                var simple = (p.siki === '1' || p.siki === '2' || p.siki === '9');
                if (!simple && p.hou) rawSteps.push(p.hou);
                (p.axes || []).forEach(function(a) {{ rawSteps.push(String(a)); }});
                (p.partners || []).forEach(function(pt) {{ rawSteps.push(String(pt)); }});

                payload = {{
                    steps: rawSteps,
                    venueName: placeName,
                    placeName: placeName,
                    round: p.round,
                    raceNo: p.round,
                    siki: p.siki,
                    hou: p.hou,
                    axes: p.axes || [],
                    partners: p.partners || [],
                    isMulti: p.isMulti || false,
                    weekday: p.weekday || "",
                    unitAmount: unitVal,
                    totalAmount: totalVal
                }};
            }}

            var jsonStr = JSON.stringify(payload);

            try {{
                var t = document.createElement('textarea');
                t.value = jsonStr;
                document.body.appendChild(t);
                t.select();
                document.execCommand('copy');
                document.body.removeChild(t);
            }} catch(e) {{}}

            var scriptableUrl = "scriptable:///run?scriptName=" + encodeURIComponent("即PAT") + "&data=" + encodeURIComponent(jsonStr);
            // window.location.href だと scriptable:// を解釈できない環境で画面が真っ白になるため
            // <a> タグをクリックする方式で現在ページを維持したまま起動する
            var aTag = document.createElement('a');
            aTag.href = scriptableUrl;
            aTag.style.display = 'none';
            document.body.appendChild(aTag);
            aTag.click();
            setTimeout(function() {{ document.body.removeChild(aTag); }}, 1000);
        }}

        function copyIpatAppCode() {{
            var code = {_scriptable_ipat_code_json};
            if (!code) {{
                alert('スクリプトコードが見つかりません');
                return;
            }}
            var t = document.createElement('textarea');
            t.value = code;
            document.body.appendChild(t);
            t.select();
            document.execCommand('copy');
            document.body.removeChild(t);
            alert('即PAT用コードをコピーしました！\\n\\n【次の手順】\\n1. Scriptableアプリを開く\\n2. 右上の「＋」を押して貼り付け\\n3. スクリプト名を「即PAT」にして保存');
        }}

        function showUmaca(btn) {{
            var prev = document.querySelector('.smappy-popup');
            if (prev) prev.remove();

            var eyes = btn.getAttribute('data-eyes');
            var type = btn.getAttribute('data-type');
            var round = btn.getAttribute('data-round');
            var axisCount = parseInt(btn.getAttribute('data-axis')) || 1;
            var unitAmount = parseInt(btn.getAttribute('data-unit')) || window._globalUnitAmount || 100;
            var totalAmount = parseInt(btn.getAttribute('data-total')) || unitAmount;
            if (!eyes || eyes === '--') {{ alert('買い目がありません'); return; }}

            var siki = getSmappySiki(type);
            var hou = getSmappyHou(type, axisCount);
            var parsed = parseSmappyEyes(eyes, type);

            var currentPlace = btn.getAttribute('data-place') || "";
            var weekday = btn.getAttribute('data-weekday') || "";

            var vCodes = {{ "札幌":"01","函館":"02","福島":"03","新潟":"04","東京":"05","中山":"06","中京":"07","京都":"08","阪神":"09","小倉":"10" }};
            var todayPlaces = [];
            for (var k in currentData) {{
                var p = currentData[k].place;
                if (!todayPlaces.includes(p)) todayPlaces.push(p);
            }}
            todayPlaces.sort(function(a, b) {{ return (vCodes[a] || "99") - (vCodes[b] || "99"); }});
            window._umacaPlaces = todayPlaces;
            var vIdx = todayPlaces.indexOf(currentPlace);
            if (vIdx < 0) vIdx = 0;

            var popup = document.createElement('div');
            popup.className = 'smappy-popup';
            popup.innerHTML = `
                <div style="font-weight: 800; font-size: 0.9rem; color: #c084fc; margin-bottom: 10px; display: flex; align-items: center; gap: 6px;">
                    🟣 JRA UMACAスマート 自動投票 (Scriptable連携)
                </div>

                <div style="margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                    <label style="font-size: 0.7rem; color: var(--text-muted); font-weight: 800;">会場判定:</label>
                    <select id="umaca-venue" style="flex: 1; padding: 4px 8px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #fff; border-radius: 6px; font-size: 0.75rem;">
                        <option value="0" ${{vIdx==0?'selected':''}}>0 (1場目: ${{todayPlaces[0] || '?'}})</option>
                        <option value="1" ${{vIdx==1?'selected':''}}>1 (2場目: ${{todayPlaces[1] || '?'}})</option>
                        <option value="2" ${{vIdx==2?'selected':''}}>2 (3場目: ${{todayPlaces[2] || '?'}})</option>
                    </select>
                </div>

                <div style="margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
                    <label style="font-size: 0.7rem; color: var(--text-muted); font-weight: 800;">1点金額:</label>
                    <input type="number" id="umaca-unit-amount" value="${{unitAmount}}" step="100" min="100" style="width: 80px; padding: 4px 8px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #fff; border-radius: 6px; font-size: 0.75rem; text-align: right;" oninput="window._globalUnitAmount=parseInt(this.value)||100; syncGlobalUnitAmount(); updateUmacaTotal();">
                    <span style="font-size: 0.75rem; color: #cbd5e1;">円</span>
                    <span style="flex: 1; text-align: right; font-size: 0.75rem; color: #94a3b8;">
                        合計: <strong id="umaca-total-disp" style="color: #c084fc;">${{totalAmount.toLocaleString()}}円</strong>
                    </span>
                </div>

                <div class="step-box" style="background: rgba(168, 85, 247, 0.08); border-left: 3px solid #a855f7; padding: 8px 10px; margin-bottom: 10px; border-radius: 4px;">
                    <span class="step-title" style="color: #c084fc; font-weight: 800; font-size: 0.75rem;">🌟 ワンタップで自動ログイン＆入力</span>
                    <div class="step-desc" style="font-size: 0.7rem; color: #94a3b8; margin-top: 2px;">
                        ボタンを押すとScriptableが起動し、UMACAスマートへのログイン・買い目セット・確認画面まで自動で進みます！
                    </div>
                </div>

                <button onclick="launchUmacaScriptable()" style="width: 100%; padding: 13px; background: linear-gradient(135deg, #a855f7, #7c3aed); color: #fff; border: none; border-radius: 8px; font-weight: 800; font-size: 0.88rem; cursor: pointer; box-shadow: 0 4px 14px rgba(168, 85, 247, 0.35); margin-bottom: 10px; display: flex; align-items: center; justify-content: center; gap: 6px;">
                    🚀 UMACAで自動投票を実行
                </button>

                <details style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 6px; padding: 8px; font-size: 0.7rem; margin-bottom: 8px;">
                    <summary style="cursor: pointer; color: #94a3b8; font-weight: 700; outline: none; user-select: none;">⚙️ 初回設定（1分・初回のみ）</summary>
                    <div style="margin-top: 8px; line-height: 1.6; color: #cbd5e1;">
                        1. App Storeで <b>Scriptable</b> アプリ（無料）をインストール<br>
                        2. 下のボタンでUMACA用スクリプトコードをコピー<br>
                        3. Scriptableで右上の「<b>＋</b>」を押し、貼り付けて名前を「<b>UMACA</b>」で保存<br>
                        4. 初回実行時のみ、カード番号(12桁)・生年月日(8桁)・暗証番号(4桁)を入力（iOSのKeychainに安全に暗号化保存されます）
                    </div>
                    <button onclick="copyUmacaAppCode()" style="margin-top: 8px; width: 100%; padding: 8px; background: #334155; color: #f8fafc; border: none; border-radius: 6px; font-weight: 700; font-size: 0.75rem; cursor: pointer;">📋 UMACA用Scriptableコードをコピー</button>
                </details>

                <div style="font-size: 0.65rem; color: #94a3b8; text-align: center; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 8px; line-height: 1.4;">
                    ※ 誤投票防止のため、最終確認画面（金額・暗証番号入力済）で停止します。<br>内容を目視確認して【投票】ボタンを押してください。
                </div>
            `;
            btn.parentElement.appendChild(popup);

            window._umacaParsed = {{
                placeName: currentPlace,
                weekday: weekday,
                round: round,
                siki: siki,
                hou: hou,
                axes: parsed.axes,
                partners: parsed.partners,
                is2Touri: parsed.is2Touri,
                isOrikaeshi: parsed.isOrikaeshi,
                isMulti: (type || "").indexOf('マルチ') >= 0,
                baseUnit: unitAmount,
                baseTotal: totalAmount
            }};
        }}

        function updateUmacaTotal() {{
            if (!window._umacaParsed) return;
            var inp = document.getElementById('umaca-unit-amount');
            var disp = document.getElementById('umaca-total-disp');
            if (!inp || !disp) return;
            var u = parseInt(inp.value) || 100;
            var baseU = window._umacaParsed.baseUnit || 100;
            var ratio = u / baseU;
            var tot = Math.round((window._umacaParsed.baseTotal || u) * ratio);
            disp.innerText = tot.toLocaleString() + '円';
        }}

        function launchUmacaScriptable() {{
            var venueEl = document.getElementById('umaca-venue');
            if (!venueEl || !window._umacaParsed) return;
            var v = venueEl.value;
            var p = window._umacaParsed;
            var placeName = (window._umacaPlaces && window._umacaPlaces[v]) || p.placeName || "";

            var unitInp = document.getElementById('umaca-unit-amount');
            var unitVal = unitInp ? (parseInt(unitInp.value) || 100) : 100;
            var baseU = p.baseUnit || 100;
            var ratio = unitVal / baseU;
            var totalVal = Math.round((p.baseTotal || unitVal) * ratio);

            var payload;
            if (p.is2Touri && p.axes && p.axes.length >= 2 && p.partners && p.partners.length >= 1) {{
                var h1 = p.axes[0];
                var h2 = p.axes[1];
                var h3 = p.partners[0];
                var bet1 = {{
                    steps: [v, p.round, p.siki, "0", String(h1), String(h2), String(h3)],
                    venueName: placeName,
                    placeName: placeName,
                    round: p.round,
                    raceNo: p.round,
                    siki: p.siki,
                    hou: "0",
                    axes: [h1, h2],
                    partners: [h3],
                    isMulti: false,
                    weekday: p.weekday || "",
                    unitAmount: unitVal,
                    totalAmount: unitVal
                }};
                var bet2 = {{
                    steps: [v, p.round, p.siki, "0", String(h2), String(h1), String(h3)],
                    venueName: placeName,
                    placeName: placeName,
                    round: p.round,
                    raceNo: p.round,
                    siki: p.siki,
                    hou: "0",
                    axes: [h2, h1],
                    partners: [h3],
                    isMulti: false,
                    weekday: p.weekday || "",
                    unitAmount: unitVal,
                    totalAmount: unitVal
                }};
                payload = {{
                    bets: [bet1, bet2],
                    unitAmount: unitVal,
                    totalAmount: totalVal
                }};
            }} else if (p.isOrikaeshi && p.axes && p.axes.length >= 2) {{
                // 馬単-折り返し: 通常の2点買い目として展開
                var fa = p.axes[0];
                var fb = p.axes[1];
                var betA = {{
                    steps: [v, p.round, p.siki, "0", String(fa), String(fb)],
                    venueName: placeName,
                    placeName: placeName,
                    round: p.round,
                    raceNo: p.round,
                    siki: p.siki,
                    hou: "0",
                    axes: [fa],
                    partners: [fb],
                    isMulti: false,
                    weekday: p.weekday || "",
                    unitAmount: unitVal,
                    totalAmount: unitVal
                }};
                var betB = {{
                    steps: [v, p.round, p.siki, "0", String(fb), String(fa)],
                    venueName: placeName,
                    placeName: placeName,
                    round: p.round,
                    raceNo: p.round,
                    siki: p.siki,
                    hou: "0",
                    axes: [fb],
                    partners: [fa],
                    isMulti: false,
                    weekday: p.weekday || "",
                    unitAmount: unitVal,
                    totalAmount: unitVal
                }};
                payload = {{
                    bets: [betA, betB],
                    unitAmount: unitVal,
                    totalAmount: totalVal
                }};
            }} else {{
                var rawSteps = [v, p.round, p.siki];
                var simple = (p.siki === '1' || p.siki === '2' || p.siki === '9');
                if (!simple && p.hou) rawSteps.push(p.hou);
                (p.axes || []).forEach(function(a) {{ rawSteps.push(String(a)); }});
                (p.partners || []).forEach(function(pt) {{ rawSteps.push(String(pt)); }});

                payload = {{
                    steps: rawSteps,
                    venueName: placeName,
                    placeName: placeName,
                    round: p.round,
                    raceNo: p.round,
                    siki: p.siki,
                    hou: p.hou,
                    axes: p.axes || [],
                    partners: p.partners || [],
                    isMulti: p.isMulti || false,
                    weekday: p.weekday || "",
                    unitAmount: unitVal,
                    totalAmount: totalVal
                }};
            }}

            var jsonStr = JSON.stringify(payload);

            try {{
                var t = document.createElement('textarea');
                t.value = jsonStr;
                document.body.appendChild(t);
                t.select();
                document.execCommand('copy');
                document.body.removeChild(t);
            }} catch(e) {{}}

            var scriptableUrl = "scriptable:///run?scriptName=" + encodeURIComponent("UMACA") + "&data=" + encodeURIComponent(jsonStr);
            var aTag = document.createElement('a');
            aTag.href = scriptableUrl;
            aTag.style.display = 'none';
            document.body.appendChild(aTag);
            aTag.click();
            setTimeout(function() {{ document.body.removeChild(aTag); }}, 1000);
        }}

        function copyUmacaAppCode() {{
            var code = {_scriptable_umaca_code_json};
            if (!code) {{
                alert('スクリプトコードが見つかりません');
                return;
            }}
            var t = document.createElement('textarea');
            t.value = code;
            document.body.appendChild(t);
            t.select();
            document.execCommand('copy');
            document.body.removeChild(t);
            alert('UMACA用コードをコピーしました！\\n\\n【次の手順】\\n1. Scriptableアプリを開く\\n2. 右上の「＋」を押して貼り付け\\n3. スクリプト名を「UMACA」にして保存');
        }}

        function launchSmappyScriptable() {{
            var venueEl = document.getElementById('smappy-venue');
            if (!venueEl || !window._smappyParsed) return;
            var v = venueEl.value;
            var p = window._smappyParsed;
            var placeName = (window._smappyPlaces && window._smappyPlaces[v]) || "";

            var payload;
            var uVal = (window._globalUnitAmount || 100);
            if (p.is2Touri && p.axes && p.axes.length >= 2 && p.partners && p.partners.length >= 1) {{
                var h1 = String(p.axes[0]);
                var h2 = String(p.axes[1]);
                var h3 = String(p.partners[0]);
                payload = {{
                    bets: [
                        {{
                            steps: [v, p.round, p.siki, "0", h1, h2, h3],
                            venueName: placeName,
                            weekday: p.weekday || "",
                            unitAmount: uVal
                        }},
                        {{
                            steps: [v, p.round, p.siki, "0", h2, h1, h3],
                            venueName: placeName,
                            weekday: p.weekday || "",
                            unitAmount: uVal
                        }}
                    ],
                    unitAmount: uVal,
                    totalAmount: uVal * 2
                }};
            }} else if (p.isOrikaeshi && p.axes && p.axes.length >= 2) {{
                // 馬単-折り返し: 通常の2点買い目として展開
                var fa = String(p.axes[0]);
                var fb = String(p.axes[1]);
                payload = {{
                    bets: [
                        {{
                            steps: [v, p.round, p.siki, "0", fa, fb],
                            venueName: placeName,
                            weekday: p.weekday || "",
                            unitAmount: uVal
                        }},
                        {{
                            steps: [v, p.round, p.siki, "0", fb, fa],
                            venueName: placeName,
                            weekday: p.weekday || "",
                            unitAmount: uVal
                        }}
                    ],
                    unitAmount: uVal,
                    totalAmount: uVal * 2
                }};
            }} else {{
                var rawSteps = [v, p.round, p.siki];
                var simple = (p.siki === '1' || p.siki === '2' || p.siki === '9');
                if (!simple && p.hou) rawSteps.push(p.hou);
                (p.axes || []).forEach(function(a) {{ rawSteps.push(String(a)); }});
                (p.partners || []).forEach(function(pt) {{ rawSteps.push(String(pt)); }});

                payload = {{
                    bets: [{{
                        steps: rawSteps,
                        venueName: placeName,
                        weekday: p.weekday || "",
                        unitAmount: uVal
                    }}],
                    unitAmount: uVal,
                    totalAmount: uVal
                }};
            }}

            var jsonStr = JSON.stringify(payload);

            // クリップボードにもフォールバック用にコピー
            try {{
                var t = document.createElement('textarea');
                t.value = jsonStr;
                document.body.appendChild(t);
                t.select();
                document.execCommand('copy');
                document.body.removeChild(t);
            }} catch(e) {{}}

            // Scriptable URLスキームを起動
            var scriptableUrl = "scriptable:///run?scriptName=" + encodeURIComponent("スマッピー") + "&data=" + encodeURIComponent(jsonStr);
            var aTag = document.createElement('a');
            aTag.href = scriptableUrl;
            aTag.style.display = 'none';
            document.body.appendChild(aTag);
            aTag.click();
            setTimeout(function() {{ document.body.removeChild(aTag); }}, 1000);
        }}

        function copyScriptableAppCode() {{
            var code = {_scriptable_app_code_json};
            if (!code) {{
                alert('スクリプトコードが見つかりません');
                return;
            }}
            var t = document.createElement('textarea');
            t.value = code;
            document.body.appendChild(t);
            t.select();
            document.execCommand('copy');
            document.body.removeChild(t);
            alert('Scriptable用コードをコピーしました！\\n\\n【次の手順】\\n1. Scriptableアプリを開く\\n2. 右上の「＋」を押して貼り付け\\n3. スクリプト名を「スマッピー」にして保存');
        }}

        function copySmappyShortcutJS() {{
            var venueEl = document.getElementById('smappy-venue');
            if (!venueEl || !window._smappyParsed) return;
            var v = venueEl.value;
            var p = window._smappyParsed;
            var placeName = (window._smappyPlaces && window._smappyPlaces[v]) || "";
            
            // そのままJSとして実行できるコードをクリップボードにコピー
            var rawJS = genSmappyShortcutJS(v, placeName, p.weekday, p.round, p.siki, p.hou, p.axes, p.partners);
            
            var t = document.createElement('textarea');
            t.value = rawJS;
            document.body.appendChild(t);
            t.select();
            document.execCommand('copy');
            document.body.removeChild(t);
            
            if (confirm('コードをコピーしました！\\nこのままショートカット「スマッピー」を起動しますか？')) {{
                window.location.href = "shortcuts://run-shortcut?name=" + encodeURIComponent("スマッピー");
            }}
        }}

        function copySmappyBml() {{
            if (!window._smappyBml) return;
            var t = document.createElement('textarea');
            t.value = window._smappyBml;
            document.body.appendChild(t);
            t.select();
            document.execCommand('copy');
            document.body.removeChild(t);
            alert('コピーしました！JRA通常投票の「会場選択」画面で実行してください');
        }}

        function genSmappyShortcutJS(venueCode, venueName, weekday, raceRound, siki, hou, axes, partners) {{
            var rawSteps = [venueCode, raceRound, siki];
            var simple = (siki === '1' || siki === '2' || siki === '9');
            if (!simple && hou) rawSteps.push(hou);
            (axes || []).forEach(function(a) {{ rawSteps.push(String(a)); }});
            (partners || []).forEach(function(p) {{ rawSteps.push(String(p)); }});
            var stepsJSON = JSON.stringify(rawSteps);
            var venueJSON = JSON.stringify(venueName || "");
            var weekdayJSON = JSON.stringify(weekday || "");
            var part1 = "(function(){{ var s=" + stepsJSON + "; var vn=" + venueJSON + "; var wd=" + weekdayJSON + "; ";
            var part2 = {_smappy_part2_js_json};
            return part1 + part2;
        }}

        function closeRecommendation() {{
            document.getElementById('recommend-modal').style.display = 'none';
            document.body.style.overflow = 'auto';
        }}

        function showReasons(raceId) {{
            const raceData = currentData[raceId] || currentData[String(raceId)];
            if (!raceData) {{
                console.warn("Race data not found for ID:", raceId);
                return;
            }}

            const modal = document.getElementById('reasons-modal');
            const body = document.getElementById('reasons-modal-body');
            if (!modal || !body) {{
                console.error("Reasons modal DOM elements not found!");
                return;
            }}
            
            let html = `
                <div style="text-align: center; margin-bottom: 25px;">
                    <div style="font-size: 0.8rem; color: #c084fc; font-weight: 800; text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 8px;">AI Evaluation Reasons</div>
                    <h2 style="margin: 0; font-size: 1.6rem; color: #fff;">${{raceData.title || ''}} 評価理由</h2>
                    <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 6px;">LightGBMモデルの特徴量寄与度 (SHAP値) 分析</div>
                </div>
            `;

            const reasons = raceData.reasons || {{}};
            const horseNames = Object.keys(reasons);

            if (horseNames.length === 0) {{
                html += `
                    <div style="padding: 40px 20px; text-align: center; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px dashed rgba(255,255,255,0.1); color: var(--text-muted); margin-bottom: 20px;">
                        <div style="font-size: 1.5rem; margin-bottom: 10px;">📊</div>
                        <div style="font-weight: 700; font-size: 0.85rem; color: #fff;">このレースの評価理由データはありません</div>
                    </div>
                `;
            }} else {{
                horseNames.forEach(hName => {{
                    const horseData = raceData.horses ? raceData.horses.find(h => h.horse_name === hName) : null;
                    const hNum = horseData ? horseData.horse_number : '';
                    const hNumHtml = hNum ? `<span class="horse-num" style="width: 28px; height: 28px; font-size: 0.9rem; margin-right: 0; background: rgba(168, 85, 247, 0.2); color: #c084fc;">${{hNum}}</span>` : '';
                    const horseReasons = reasons[hName] || {{}};

                    html += `
                        <div class="reason-card">
                            <div class="reason-horse-title">
                                ${{hNumHtml}}
                                <span>${{hName}}</span>
                            </div>
                            <div>
                    `;

                    for (const [rKey, rText] of Object.entries(horseReasons)) {{
                        let formattedText = rText;
                        const kiyodoMatch = rText.match(/寄与度:\s*([+\-]?\d+(?:\.\d+)?)/);
                        if (kiyodoMatch) {{
                            const kVal = parseFloat(kiyodoMatch[1]);
                            const kColor = kVal >= 0 ? '#4ade80' : '#ef4444';
                            formattedText = rText.replace(kiyodoMatch[0], `<span style="color:${{kColor}}; font-weight:700;">${{kiyodoMatch[0]}}</span>`);
                        }}

                        html += `
                            <div class="reason-item">
                                <span class="reason-tag">${{rKey}}</span>
                                <div style="flex: 1;">${{formattedText}}</div>
                            </div>
                        `;
                    }}

                    html += `
                            </div>
                        </div>
                    `;
                }});
            }}

            body.innerHTML = html;
            modal.style.display = 'flex';
            document.body.style.overflow = 'hidden';
        }}

        function closeReasons() {{
            document.getElementById('reasons-modal').style.display = 'none';
            document.body.style.overflow = 'auto';
        }}

    </script>
</body>
</html>
"""

    # HTML の保存先
    output_html_paths = [
        r"C:\Users\kyoui\tohshin_keiba\index.html",
        r"C:\Users\kyoui\tohshin_keiba\deploy_tmp\index.html"
    ]
    
    for out_html in output_html_paths:
        try:
            with open(out_html, "w", encoding="utf-8") as f:
                f.write(html_template)
            logger.info(f"Successfully generated HTML at {out_html}")
        except Exception as e:
            logger.error(f"Failed to write HTML to {out_html}: {e}")
    
    # sw.js の deploy_tmp 同期処理
    sw_src = r"C:\Users\kyoui\tohshin_keiba\sw.js"
    sw_dst = r"C:\Users\kyoui\tohshin_keiba\deploy_tmp\sw.js"
    if os.path.exists(sw_src):
        try:
            import shutil
            os.makedirs(os.path.dirname(sw_dst), exist_ok=True)
            shutil.copy2(sw_src, sw_dst)
            logger.info(f"Successfully copied sw.js to {sw_dst}")
        except Exception as e:
            logger.error(f"Failed to copy sw.js to {sw_dst}: {e}")

    # Git 更新処理 (tohshin_keiba のみ)
    try:
        repo_dir = r"C:\Users\kyoui\tohshin_keiba"
        logger.info(f"Starting Git update process for {repo_dir}...")
        
        # 1. git add
        files_to_add = ["index.html", "generate_html.py", "sw.js", "jsons/meta.json", "jsons/tansho_data.json"]
        import glob
        files_to_add.extend(glob.glob("jsons/data_*.json"))
        subprocess.run(["git", "add"] + files_to_add, cwd=repo_dir, check=True)
        
        # 2. git commit (ステージされた変更がある場合のみ)
        has_staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_dir).returncode != 0
        if has_staged:
            subprocess.run(["git", "commit", "-m", "Auto-update race data and HTML (Fixed Corruption)"], cwd=repo_dir, check=True)
            logger.info("Successfully committed changes.")
            
            # 3. git push
            try:
                subprocess.run(["git", "push", "origin", "main"], cwd=repo_dir, check=True)
                logger.info("Successfully pushed changes to origin/main.")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Git push failed: {e}. Changes are committed locally.")
        else:
            logger.info("No changes to commit (tohshin_keiba).")
            
    except Exception as e:
        logger.error(f"Error during Git update for tohshin_keiba: {e}")

if __name__ == "__main__":
    generate_static_html()