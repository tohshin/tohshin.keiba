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

# Kelly3 v12 強化学習ハイブリッドモジュールのインポート (kelly3_auto_runner.py 準拠)
if r"C:\Users\kyoui\keiba" not in sys.path:
    sys.path.append(r"C:\Users\kyoui\keiba")

try:
    from modules.rl_betting import (
        load_rl_agent, extract_static_features, get_kelly3_state,
        predict_action, generate_rl_bet_lines, ACTIONS, VENUE_MAP, VENUE_NAME_MAP
    )
    rl_agent_available = True
    logger.info("Successfully loaded modules.rl_betting (Kelly3 RL Hybrid)")
except Exception as e:
    logger.warning(f"Could not load modules.rl_betting: {e}")
    rl_agent_available = False

try:
    from modules.training.predict_residual import (
        DEFAULT_KELLY2_PLAN_B_CONFIG,
        generate_kelly2_plan_b_race_bets,
        generate_kelly3_plan2_race_bets,
        load_daily_feature_csv,
        predict_residual_ensemble,
        scores_from_margins_and_odds,
    )
    residual_v1_available = True
    logger.info("Successfully loaded modules.training.predict_residual (Kelly2 Plan B / Kelly3 Plan 2)")
except Exception as e:
    logger.warning(f"Could not load modules.training.predict_residual: {e}")
    residual_v1_available = False

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
# Kelly2 / Kelly3 本家準拠の買い目生成
#  - PICKUP     : Kelly2.ipynb (改善案B: 強弱メリハリ型 毎R 3連単・3連複100% + 厳選単勝補完)
#  - PICKUP 2   : Kelly3.ipynb / kelly3_auto_runner.py (新案2: v12合議×市場残差統合 3レーン利確縮小継続)
# ==========================================================================
KELLY_CONFIG_DIR = r"C:\Users\kyoui\keiba\config"
KELLY_STRATEGIES_V13_CSV = os.path.join(KELLY_CONFIG_DIR, "winning_strategies_v13.csv")
KELLY_TCSV_DIR = r"C:\keibasoftcom\KSCAutoBetPlus\TCSV\v12"
KELLY_TCSV_FALLBACK_DIR = r"C:\keibasoftcom\KSCAutoBetPlus\TCSV"
KELLY_TCSV_V13_DIR = r"C:\keibasoftcom\KSCAutoBetPlus\TCSV\v13"
KELLY_TANSHO_DB_URL = "postgresql://postgres:zatenn@localhost/postgres"

# Kelly2 改善案B (7モデル市場残差アンサンブル Plan B) 定数
KELLY2_RESIDUAL_VERSION = "residual_v1_plan_b2"
KELLY3_PLAN2_VERSION = "k3_plan2_v12_residual_v2"
KELLY2_RECENT_RECALC_DAYS = 7  # 直近N日以内（および未来日）は毎回再計算、それ以前の生成済み過去日は既存JSONを再利用
KELLY2_RESIDUAL_MODEL_DIR = r"C:\Users\kyoui\keiba\models\residual_v1"
KELLY2_FEATURE_TMP_DIR = r"C:\Users\kyoui\keiba\data\tmp"

# Kelly3 強化学習ハイブリッド定数 (kelly3_auto_runner.py と同一デフォルト)
KELLY3_BASE_MODEL = r"C:\Users\kyoui\keiba\models\rl\rl_agent_kelly3_rebuilt_v12.pth"
KELLY3_SANREN_MODEL = r"C:\Users\kyoui\keiba\models\rl\rl_agent_kelly3_stop5000_sanren_cap100k_v12.pth"
KELLY3_BASE_ACTIONS = [0, 15, 18, 19, 20, 21]
KELLY3_SANREN_ACTIONS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 22, 23, 24, 25]
KELLY3_BASE_MULT = 5
KELLY3_SANREN_MULT = 1
KELLY3_BASE_LANE_STOP = 1 * KELLY3_BASE_MULT          # +5円 (100円基準 +1円)
KELLY3_SANREN_LANE_STOP = 5000 * KELLY3_SANREN_MULT   # +5,000円
KELLY3_MAX_BUDGET = 50000
KELLY3_AGREE_MODELS = ['Ensemble', 'LightGBM', 'CatBoost', 'XGBoost', 'TabNet']
_KELLY3_MODELS_CACHE = None
_KELLY2_RESIDUAL_CACHE = None
_KELLY_RESIDUAL_DAY_CACHE = {}
_KELLY_DB_ENGINE = None
_KELLY_ODDS_YEAR_CACHE = {}
_KELLY_HARAIMODOSHI_YEAR_CACHE = {}
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
# Kelly2.ipynb の race_meta が持つキー (これ以外では戦略/EXCLUDE判定しない)
KELLY2_META_KEYS = ('venue_name', 'venue_code', 'class_cat', 'track_type',
                    'venue_track', 'class_venue', 'class_venue_track')


def _softmax_np(x):
    """Kelly2.ipynb と同一のソフトマックス (EV算出用)"""
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum(axis=0) + 1e-12)


def _get_kelly_db_engine():
    global _KELLY_DB_ENGINE
    if _KELLY_DB_ENGINE is None:
        from sqlalchemy import create_engine
        _KELLY_DB_ENGINE = create_engine(KELLY_TANSHO_DB_URL)
    return _KELLY_DB_ENGINE


def _load_tansho_odds_df(day):
    """odds1_tansho から当日の単勝オッズ(実値)・人気を取得する。
    年単位で一括キャッシュし、日次ループでの重複DB接続・クエリを回避する。"""
    empty = pd.DataFrame(columns=['race_code', 'umaban', 'odds_val', 'ninki'])
    nen = str(day)[:4]
    gappi = str(day)[4:8]
    try:
        if nen not in _KELLY_ODDS_YEAR_CACHE:
            engine = _get_kelly_db_engine()
            sql = ("SELECT kaisai_gappi, race_code, umaban, odds, ninki FROM odds1_tansho "
                   f"WHERE kaisai_nen = '{nen}';")
            df_y = pd.read_sql(sql, engine)
            by_gappi = {}
            if not df_y.empty:
                df_y['kaisai_gappi'] = df_y['kaisai_gappi'].astype(str).str.zfill(4)
                df_y['race_code'] = df_y['race_code'].astype(str)
                df_y['umaban'] = pd.to_numeric(df_y['umaban'], errors='coerce')
                df_y['odds_val'] = pd.to_numeric(df_y['odds'], errors='coerce') / 10.0
                df_y['ninki'] = pd.to_numeric(df_y['ninki'], errors='coerce')
                df_y = df_y.dropna(subset=['umaban'])
                for gp, grp in df_y.groupby('kaisai_gappi'):
                    by_gappi[str(gp)] = grp[['race_code', 'umaban', 'odds_val', 'ninki']].reset_index(drop=True)
            _KELLY_ODDS_YEAR_CACHE[nen] = by_gappi
            logger.info(f"Cached odds1_tansho for year {nen}: {len(by_gappi)} days")

        df_o = _KELLY_ODDS_YEAR_CACHE[nen].get(gappi)
        if df_o is None or df_o.empty:
            logger.warning(f"No tansho odds found in DB for {day}")
            return empty
        return df_o
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


def _fast_load_daily_feature_csv(csv_path):
    """load_daily_feature_csv と同一仕様だが、同一ヘッダの連続/複数レースブロックをまとめて
    pd.read_csv することでパース回数を 36回→1〜2回 に削減する高速版。"""
    import io
    with open(csv_path, encoding='utf-8') as f:
        lines = f.readlines()

    # ヘッダごとにデータ行をまとめつつ、順序保持のため連続同一ヘッダを結合する
    groups = []  # [(header_line, [data_lines...])]
    cur_header = None
    cur_rows = []
    for line in lines:
        if line.startswith(',horse_id,') or line.startswith('race_id,') or ',馬番,' in line[:250]:
            if line == cur_header:
                continue
            if cur_header is not None and cur_rows:
                groups.append((cur_header, cur_rows))
            cur_header = line
            cur_rows = []
        else:
            if cur_header is not None:
                cur_rows.append(line)
    if cur_header is not None and cur_rows:
        groups.append((cur_header, cur_rows))

    dfs = []
    for hdr, rows in groups:
        df_b = pd.read_csv(io.StringIO(hdr + ''.join(rows)), low_memory=False)
        first_col = df_b.columns[0]
        if first_col in ('Unnamed: 0', '', 'index'):
            df_b = df_b.rename(columns={first_col: 'race_id'})
        dfs.append(df_b)

    if not dfs:
        return pd.DataFrame()

    df_all = dfs[0] if len(dfs) == 1 else pd.concat(dfs, ignore_index=True, sort=False)
    df_all = df_all.copy()
    df_all['race_id'] = df_all['race_id'].astype(str).str.strip()
    df_all['umaban'] = pd.to_numeric(df_all['馬番'], errors='coerce').fillna(0).astype(int)
    df_all = df_all.drop_duplicates(subset=['race_id', 'umaban'], keep='last').reset_index(drop=True)
    return df_all


def _load_kelly2_residual_bundle():
    """Kelly2 residual_v1 の7モデル・メタ情報・カラムマッピングを1回だけロードしてキャッシュする。"""
    global _KELLY2_RESIDUAL_CACHE
    if _KELLY2_RESIDUAL_CACHE is not None:
        return _KELLY2_RESIDUAL_CACHE

    meta_pkl_path = os.path.join(KELLY2_RESIDUAL_MODEL_DIR, "meta_residual_v1.pkl")
    if not os.path.exists(meta_pkl_path):
        logger.warning(f"Kelly2: residual model not found: {meta_pkl_path}")
        return None

    import pickle
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostClassifier

    with open(meta_pkl_path, 'rb') as f:
        meta = pickle.load(f)

    m_lgb_w = lgb.Booster(model_file=os.path.join(KELLY2_RESIDUAL_MODEL_DIR, 'lgb_win.txt'))
    m_lgb_wp = lgb.Booster(model_file=os.path.join(KELLY2_RESIDUAL_MODEL_DIR, 'lgb_win_pure.txt'))
    m_xgb_w = xgb.Booster()
    m_xgb_w.load_model(os.path.join(KELLY2_RESIDUAL_MODEL_DIR, 'xgb_win.json'))
    m_cb_w = CatBoostClassifier()
    m_cb_w.load_model(os.path.join(KELLY2_RESIDUAL_MODEL_DIR, 'cb_win.cbm'))

    m_lgb_p = lgb.Booster(model_file=os.path.join(KELLY2_RESIDUAL_MODEL_DIR, 'lgb_place.txt'))
    m_xgb_p = xgb.Booster()
    m_xgb_p.load_model(os.path.join(KELLY2_RESIDUAL_MODEL_DIR, 'xgb_place.json'))
    m_cb_p = CatBoostClassifier()
    m_cb_p.load_model(os.path.join(KELLY2_RESIDUAL_MODEL_DIR, 'cb_place.cbm'))

    # config/column_mapping.csv も1回だけロード
    mapping_path = os.path.join(KELLY_CONFIG_DIR, 'column_mapping.csv')
    mapping_pairs = []
    if os.path.exists(mapping_path):
        try:
            mdf = pd.read_csv(mapping_path, encoding='utf-8')
        except UnicodeDecodeError:
            mdf = pd.read_csv(mapping_path, encoding='cp932')
        mdf = mdf.dropna(subset=['japanese', 'english'])
        mapping = dict(zip(mdf['japanese'].astype(str), mdf['english'].astype(str)))
        mapping_pairs = [(k, mapping[k]) for k in sorted(mapping.keys(), key=len, reverse=True)]

    _KELLY2_RESIDUAL_CACHE = {
        'feature_cols': meta['feature_cols'],
        'pure_cols': meta['pure_cols'],
        'm_lgb_w': m_lgb_w,
        'm_lgb_wp': m_lgb_wp,
        'm_xgb_w': m_xgb_w,
        'm_cb_w': m_cb_w,
        'm_lgb_p': m_lgb_p,
        'm_xgb_p': m_xgb_p,
        'm_cb_p': m_cb_p,
        'mapping_pairs': mapping_pairs,
    }
    logger.info("Kelly2 residual_v1: cached 7 models and column_mapping")
    return _KELLY2_RESIDUAL_CACHE


def _predict_kelly2_residual_fast(df_feat, odds_map, day_str, bundle):
    """キャッシュ済み7モデルで predict_residual_ensemble と同一の推論を実行する。"""
    import unicodedata
    import xgboost as xgb
    from scipy.special import expit, logit
    from modules.training.predict_residual import compute_implied_probs

    df = df_feat.copy()
    df['racecode'] = day_str + df['race_id'].astype(str).str[4:]

    if odds_map is not None and not odds_map.empty:
        om = odds_map[['race_code', 'umaban', 'odds_val']].drop_duplicates(subset=['race_code', 'umaban'], keep='last')
        df = df.merge(om, left_on=['racecode', 'umaban'], right_on=['race_code', 'umaban'], how='left')
    else:
        df['odds_val'] = np.nan

    if df['odds_val'].isna().any() and '単勝' in df.columns:
        df['odds_val'] = df['odds_val'].fillna(pd.to_numeric(df['単勝'], errors='coerce'))
    df['odds_val'] = df['odds_val'].fillna(50.0)

    # カラム名の NFKC正規化 + 重複統合 + 英語置換 + サニタイズ
    norm_groups = {}
    for idx, col in enumerate(df.columns):
        nk = unicodedata.normalize('NFKC', str(col))
        norm_groups.setdefault(nk, []).append(idx)

    sanitize_pat = re.compile(r'[\[\]\{\},:\"\s\'\(\)]')
    en_cols = {}
    for nk, idxs in norm_groups.items():
        c_en = nk
        for jp_w, en_w in bundle['mapping_pairs']:
            if jp_w in c_en:
                c_en = c_en.replace(jp_w, en_w)
        c_en = sanitize_pat.sub('_', c_en)
        if c_en not in en_cols:
            en_cols[c_en] = idxs
        else:
            en_cols[c_en].extend(idxs)

    feature_cols = bundle['feature_cols']
    n_rows = len(df)
    data = {}
    for c in feature_cols:
        idxs = en_cols.get(c)
        if idxs:
            col = df.iloc[:, idxs[0]]
            for i in idxs[1:]:
                s_other = df.iloc[:, i]
                if col.isna().any() and not s_other.isna().all():
                    col = col.combine_first(s_other)
            if isinstance(col.dtype, pd.CategoricalDtype):
                data[c] = col.cat.codes.astype(np.float32).values
            elif col.dtype == bool:
                data[c] = col.astype(np.float32).values
            elif col.dtype == 'object':
                s = col.astype(str)
                if s.isin(['True', 'False', 'nan', 'None']).all():
                    data[c] = (s == 'True').astype(np.float32).values
                else:
                    data[c] = pd.to_numeric(col, errors='coerce').astype(np.float32).values
            else:
                data[c] = pd.to_numeric(col, errors='coerce').astype(np.float32).values
        else:
            data[c] = np.full(n_rows, np.nan, dtype=np.float32)

    X = pd.DataFrame(data, index=df.index)
    X_cb = X.fillna(-999)
    dmat = xgb.DMatrix(X)

    imp_win, imp_place = compute_implied_probs(df['racecode'].values, df['odds_val'].values)

    mw_ens = (
        bundle['m_lgb_w'].predict(X, raw_score=True)
        + bundle['m_lgb_wp'].predict(X[bundle['pure_cols']], raw_score=True)
        + bundle['m_xgb_w'].predict(dmat, output_margin=True)
        + bundle['m_cb_w'].predict(X_cb, prediction_type='RawFormulaVal')
    ) / 4.0

    mp_ens = (
        bundle['m_lgb_p'].predict(X, raw_score=True)
        + bundle['m_xgb_p'].predict(dmat, output_margin=True)
        + bundle['m_cb_p'].predict(X_cb, prediction_type='RawFormulaVal')
    ) / 3.0

    pw_raw = expit(logit(imp_win) + mw_ens)
    p_win = pw_raw / pd.Series(pw_raw).groupby(df['racecode']).transform('sum').values
    ev_win = p_win * df['odds_val'].values

    pp_raw = expit(logit(imp_place) + mp_ens)
    n_h = df.groupby('racecode')['umaban'].transform('count').values
    t_sum = np.where(n_h <= 7, 2.0, 3.0)
    p_place = np.clip(pp_raw * (t_sum / pd.Series(pp_raw).groupby(df['racecode']).transform('sum').values), 1e-4, 0.99)
    place_edge = p_place / imp_place

    return pd.DataFrame({
        'racecode': df['racecode'].values,
        'race_id': df['race_id'].values,
        'umaban': df['umaban'].values,
        'odds_val': df['odds_val'].values,
        'imp_win': imp_win,
        'imp_place': imp_place,
        'mw_ens': mw_ens,
        'mp_ens': mp_ens,
        'p_win': p_win,
        'ev_win': ev_win,
        'p_place': p_place,
        'place_edge': place_edge,
    })


def _get_or_compute_residual_for_day(day_str):
    """当日の residual_v1 推論結果 (df_pred) と新馬レースID集合をキャッシュ付きで返す。
    Kelly2 と Kelly3 の両方で共有し、1日あたり1回だけ推論する。"""
    if day_str in _KELLY_RESIDUAL_DAY_CACHE:
        return _KELLY_RESIDUAL_DAY_CACHE[day_str]

    feature_csv_path = os.path.join(KELLY2_FEATURE_TMP_DIR, f"{day_str}.csv")
    if not os.path.exists(feature_csv_path):
        logger.debug(f"Kelly2/3: feature CSV not found for {day_str} ({feature_csv_path})")
        _KELLY_RESIDUAL_DAY_CACHE[day_str] = None
        return None

    bundle = _load_kelly2_residual_bundle()
    if bundle is None:
        _KELLY_RESIDUAL_DAY_CACHE[day_str] = None
        return None

    odds_df = _load_tansho_odds_df(day_str)
    odds_map = pd.DataFrame()
    if not odds_df.empty:
        odds_map = odds_df[['race_code', 'umaban', 'odds_val']].dropna().copy()
        odds_map['umaban'] = odds_map['umaban'].astype(int)

    try:
        df_feat = _fast_load_daily_feature_csv(feature_csv_path)
        if df_feat.empty:
            logger.warning(f"Kelly2/3: feature CSV is empty for {day_str}")
            _KELLY_RESIDUAL_DAY_CACHE[day_str] = None
            return None

        # 新馬戦判定マップ (df_feat 側のフォールバック用)
        shinba_rids = set()
        for sc_col in ('shinba', 'レースクラス_新馬', '新馬'):
            if sc_col in df_feat.columns:
                for rid_k, grp in df_feat.groupby('race_id'):
                    v = pd.to_numeric(grp[sc_col], errors='coerce').fillna(0).max()
                    if v > 0 or grp[sc_col].astype(str).isin(['True', '1', '1.0']).any():
                        shinba_rids.add(str(rid_k).strip())

        df_pred = _predict_kelly2_residual_fast(df_feat, odds_map, day_str, bundle)
        # マージンマップ (Kelly3 新案2 で再利用)
        fallback_odds_map = {}
        for oc in ('単勝', 'オッズ', 'odds'):
            if oc in df_feat.columns:
                fb_vals = pd.to_numeric(df_feat[oc], errors='coerce').values
                for rid_k, ub_k, fbo in zip(df_feat['race_id'].astype(str), df_feat['umaban'].astype(int), fb_vals):
                    fallback_odds_map[(rid_k, ub_k)] = float(fbo) if pd.notnull(fbo) else np.nan
                break
        margin_map = {}
        for rid_k, ub_k, mw_v, mp_v in zip(
            df_pred['race_id'].astype(str),
            df_pred['umaban'].astype(int),
            df_pred['mw_ens'].values,
            df_pred['mp_ens'].values,
        ):
            margin_map[(rid_k, ub_k)] = (float(mw_v), float(mp_v), fallback_odds_map.get((rid_k, ub_k), np.nan))

        pack = {
            'df_pred': df_pred,
            'shinba_rids': shinba_rids,
            'margin_map': margin_map,
        }
        _KELLY_RESIDUAL_DAY_CACHE[day_str] = pack
        return pack
    except Exception as e:
        logger.error(f"Kelly2/3: residual_v1 inference failed for {day_str}: {e}")
        _KELLY_RESIDUAL_DAY_CACHE[day_str] = None
        return None


def _parse_kelly2_plan_b_line(line, bt):
    """Kelly2 改善案B の ipatgo 行から UI 表示用の bettingEyesText / axis / partners を生成する。"""
    parts = line.split(',')
    code = parts[3].strip() if len(parts) > 3 else ''
    sel_mode = parts[4].strip() if len(parts) > 4 else ''
    nums_str = parts[6].strip() if len(parts) > 6 else ''
    pad = lambda n: f"{int(n):02d}"

    axis1 = None
    axis2 = None
    partners = []
    eyes = nums_str

    if code == 'TANSYO':
        u = int(nums_str)
        axis1 = u
        eyes = pad(u)
    elif sel_mode == 'BOX':
        nums = [int(nums_str[i:i + 2]) for i in range(0, len(nums_str), 2) if nums_str[i:i + 2].isdigit()]
        axis1 = nums[0] if len(nums) >= 1 else None
        axis2 = nums[1] if len(nums) >= 2 else None
        partners = nums[2:] if len(nums) > 2 else []
        eyes = f"{', '.join(pad(x) for x in nums)} BOX"
    elif sel_mode == 'WHEEL1' and '-' in nums_str:
        ax_s, rest_s = nums_str.split('-', 1)
        ax = int(ax_s)
        oth = [int(rest_s[i:i + 2]) for i in range(0, len(rest_s), 2) if rest_s[i:i + 2].isdigit()]
        axis1 = ax
        partners = oth
        sep = ' → ' if code == 'SANRENTAN' else ' - '
        eyes = f"{pad(ax)}{sep}{', '.join(pad(x) for x in oth)}"
    elif sel_mode == 'NORMAL' and '-' in nums_str:
        nums = [int(x) for x in nums_str.split('-') if x.isdigit()]
        if code == 'SANRENTAN' and len(nums) == 3:
            axis1, axis2, partners = nums[0], nums[1], [nums[2]]
            eyes = f"{pad(nums[0])} → {pad(nums[1])} → {pad(nums[2])}"
        elif code == 'SANRENPUKU' and len(nums) == 3:
            axis1, partners = nums[0], [nums[1], nums[2]]
            eyes = f"{pad(nums[0])} - {pad(nums[1])}, {pad(nums[2])}"
        elif nums:
            axis1 = nums[0]
            partners = nums[1:]
            eyes = ' - '.join(pad(x) for x in nums)

    return axis1, axis2, partners, eyes


def build_kelly2_bets_for_day(day, races_of_day):
    """Kelly2.ipynb (改善案B: 強弱メリハリ型 毎R 3連単・3連複100% + 厳選単勝補完) と同一ロジックで
    当日の買い目を生成する。"""
    if not residual_v1_available:
        logger.warning("Kelly2: predict_residual モジュールが利用できないため生成をスキップ")
        return None

    day_str = str(day)
    pack = _get_or_compute_residual_for_day(day_str)
    if pack is None:
        return None
    df_pred = pack['df_pred']
    shinba_rids = pack['shinba_rids']
    if df_pred.empty:
        return {}

    odds_df = _load_tansho_odds_df(day_str)
    ninki_map = {}
    if not odds_df.empty:
        for rc, ub, nk in zip(odds_df['race_code'], odds_df['umaban'], odds_df['ninki']):
            if pd.notnull(nk):
                ninki_map[(str(rc), int(ub))] = int(nk)

    model_label = "市場残差アンサンブル (Plan B)"
    candidates = []

    for r_code, g in df_pred.groupby('racecode'):
        if len(g) < 5:
            continue
        r_code = str(r_code)
        s_r12 = (r_code[:4] + r_code[8:16]) if len(r_code) == 16 else r_code
        date_str = r_code[:8]
        track_name = KELLY_TRACK_MAPPING.get(r_code[8:10], 'UNKNOWN')
        race_no = int(r_code[-2:])

        r_meta = (races_of_day.get(s_r12) or {}).get('meta') or {}
        c_cat = r_meta.get('class_cat', 'その他')
        if c_cat != '新馬' and s_r12 in shinba_rids:
            c_cat = '新馬'

        r_bets = generate_kelly2_plan_b_race_bets(
            date_str=date_str,
            track_name=track_name,
            race_no=race_no,
            g_race=g,
            class_cat=c_cat,
            config=DEFAULT_KELLY2_PLAN_B_CONFIG,
        )

        row_by_u = {int(r['umaban']): r for _, r in g.iterrows()}

        # TierA:3連単1着固定(4点) は 1着固定ながし(6点)ではなく
        # 2着2頭絞りフォーメーション (h1 → o0, o1 → o0, o1, o2 = 4点) のため1件にまとめて表記する
        tier_a_tan_bets = [b for b in r_bets if b['bt'] == 'TierA:3連単1着固定(4点)']
        other_bets = [b for b in r_bets if b['bt'] != 'TierA:3連単1着固定(4点)']

        if tier_a_tan_bets:
            triples = []
            for b in tier_a_tan_bets:
                p_nums = b['line'].split(',')[6].split('-')
                if len(p_nums) == 3 and all(x.isdigit() for x in p_nums):
                    triples.append((int(p_nums[0]), int(p_nums[1]), int(p_nums[2])))
            if triples:
                s1 = list(dict.fromkeys(t[0] for t in triples))
                s2 = list(dict.fromkeys(t[1] for t in triples))
                s3 = list(dict.fromkeys(t[2] for t in triples))
                unit = int(tier_a_tan_bets[0]['unit'])
                combs = len(triples)
                bt_disp = f"TierA:3連単フォーメーション({combs}点)"
                eyes = (
                    f"{', '.join(f'{x:02d}' for x in s1)} → "
                    f"{', '.join(f'{x:02d}' for x in s2)} → "
                    f"{', '.join(f'{x:02d}' for x in s3)}"
                )
                axis1 = s1[0]
                partners = sorted((set(s2) | set(s3)) - set(s1))
                detail_parts = [bt_disp]
                if axis1 in row_by_u:
                    r1 = row_by_u[axis1]
                    detail_parts.append(
                        f"軸{axis1:02d}(ev {r1['ev_win']:.2f} / edge {r1['place_edge']:.2f} / {r1['odds_val']:.1f}倍)"
                    )
                candidates.append({
                    'line': f"{date_str},{track_name},{race_no},SANRENTAN,FORMATION,,{eyes},{unit}",
                    'lines': [b['line'] for b in tier_a_tan_bets],
                    'race_id': s_r12,
                    'strategy_id': bt_disp,
                    'model': model_label,
                    'bet_type': bt_disp,
                    'rawType': bt_disp,
                    'unit': unit,
                    'combs': combs,
                    'cost': unit * combs,
                    'axis1Num': axis1,
                    'axis2Num': None,
                    'partnerNums': partners,
                    'bettingEyesText': eyes,
                    'detailText': " | ".join(detail_parts),
                    'h1PopRank': ninki_map.get((r_code, axis1)),
                })

        for b in other_bets:
            line = b['line']
            bt = b['bt']
            unit = int(b['unit'])
            combs = int(b['combs'])
            axis1, axis2, partners, eyes = _parse_kelly2_plan_b_line(line, bt)

            # 根拠サマリ
            detail_parts = [bt]
            if axis1 in row_by_u:
                r1 = row_by_u[axis1]
                detail_parts.append(
                    f"軸{axis1:02d}(ev {r1['ev_win']:.2f} / edge {r1['place_edge']:.2f} / {r1['odds_val']:.1f}倍)"
                )
            detail_text = " | ".join(detail_parts)

            candidates.append({
                'line': line,
                'race_id': s_r12,
                'strategy_id': bt,
                'model': model_label,
                'bet_type': bt,
                'rawType': bt,
                'unit': unit,
                'combs': combs,
                'cost': unit * combs,
                'axis1Num': axis1,
                'axis2Num': axis2,
                'partnerNums': partners,
                'bettingEyesText': eyes,
                'detailText': detail_text,
                'h1PopRank': ninki_map.get((r_code, axis1)) if axis1 is not None else None,
            })

    if not candidates:
        logger.info(f"Kelly2 (Plan B): 0 bets for {day_str}")
        return {}

    # 同一券種・同一買い方・同一組合わせの重複排除 (Kelly2.ipynb と同じく単価が高い方を優先保持)
    candidates.sort(key=lambda c: c['unit'], reverse=True)
    deduped = []
    seen_keys = set()
    for c in candidates:
        dedup_key = ','.join(c['line'].split(',')[:7])
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        deduped.append(c)

    deduped.sort(key=lambda c: (c['race_id'], c['bet_type'], c['bettingEyesText']))

    by_race = {}
    for c in deduped:
        by_race.setdefault(c['race_id'], []).append(c)

    result = {}
    total_cost = 0
    for rid, bets in by_race.items():
        r_cost = sum(b['cost'] for b in bets)
        total_cost += r_cost
        result[rid] = {'bets': bets, 'total_cost': r_cost, 'version': KELLY2_RESIDUAL_VERSION}
    logger.info(f"Kelly2 (Plan B): {len(deduped)} bets / {len(result)} races / total_cost={total_cost:,}円 for {day_str}")
    return result


def _load_kelly3_score_df(day):
    """TCSV\\v12 のモデルスコアCSV（無い場合は TCSV 直下にフォールバック）を読み込み、
    race_id / horse_num を付与した DataFrame を返す。
    Ensemble を優先し、無い場合は他モデルの平均で補完する。"""
    score_dfs = {}
    for m in ['Ensemble', 'AutoGluon', 'LightGBM', 'CatBoost', 'XGBoost', 'TabNet']:
        name = f"Ensemble_{day}.csv" if m == 'Ensemble' else f"{m}_raw_{day}.csv"
        fp = os.path.join(KELLY_TCSV_DIR, name)
        if not os.path.exists(fp):
            fp_fb = os.path.join(KELLY_TCSV_FALLBACK_DIR, name)
            if os.path.exists(fp_fb):
                fp = fp_fb
            else:
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


def _load_kelly3_models():
    """Kelly3 強化学習ハイブリッド用モデル (ベース + 3連系) をロードしてキャッシュする。"""
    global _KELLY3_MODELS_CACHE
    if _KELLY3_MODELS_CACHE is not None:
        return _KELLY3_MODELS_CACHE
    if not rl_agent_available:
        return None
    try:
        net_base, device = load_rl_agent(KELLY3_BASE_MODEL)
        net_sanren = None
        if os.path.exists(KELLY3_SANREN_MODEL):
            net_sanren, _ = load_rl_agent(KELLY3_SANREN_MODEL, device=device)
        else:
            logger.warning(f"Kelly3: 3連系モデル未検出 ({KELLY3_SANREN_MODEL}) -> ベース単体で動作")
        _KELLY3_MODELS_CACHE = (net_base, net_sanren, device)
        logger.info(f"Kelly3 RL Hybrid models loaded (device={device}, sanren={net_sanren is not None})")
        return _KELLY3_MODELS_CACHE
    except Exception as e:
        logger.warning(f"Kelly3: RLモデルロード失敗: {e}")
        return None


def _load_haraimodoshi_map(day):
    """DB (haraimodoshi) から確定済み払戻金マップを取得する (kelly3_auto_runner.py 準拠)。
    年単位でキャッシュして日次ループでの重複DB取得を回避する。"""
    nen = str(day)[:4]
    if nen in _KELLY_HARAIMODOSHI_YEAR_CACHE:
        return _KELLY_HARAIMODOSHI_YEAR_CACHE[nen]
    try:
        engine = _get_kelly_db_engine()
        df = pd.read_sql(f"SELECT * FROM haraimodoshi WHERE kaisai_nen = '{nen}';", engine)
        if df.empty:
            _KELLY_HARAIMODOSHI_YEAR_CACHE[nen] = {}
            return {}

        def _ci(v):
            if pd.isna(v):
                return None
            s = str(v).strip()
            return int(s) if s.isdigit() else None

        p_map = {}
        for _, r in df.iterrows():
            rid = (
                str(r['kaisai_nen'])
                + str(r['keibajo_code']).zfill(2)
                + str(r['kaisai_kaiji']).zfill(2)
                + str(r['kaisai_nichiji']).zfill(2)
                + str(r['race_bango']).zfill(2)
            )
            p_dict = {k: {} for k in ('tansho', 'fukusho', 'umaren', 'wide', 'umatan', 'sanrenpuku', 'sanrentan')}
            for i in range(1, 4):
                u, amt = _ci(r.get(f'tansho{i}_umaban')), _ci(r.get(f'tansho{i}_haraimodoshikin'))
                if u and amt:
                    p_dict['tansho'][u] = float(amt)
            for i in range(1, 6):
                u, amt = _ci(r.get(f'fukusho{i}_umaban')), _ci(r.get(f'fukusho{i}_haraimodoshikin'))
                if u and amt:
                    p_dict['fukusho'][u] = float(amt)
            for i in range(1, 4):
                u1, u2, amt = _ci(r.get(f'umaren{i}_kumiban1')), _ci(r.get(f'umaren{i}_kumiban2')), _ci(r.get(f'umaren{i}_haraimodoshikin'))
                if u1 and u2 and amt:
                    p_dict['umaren'][tuple(sorted([u1, u2]))] = float(amt)
            for i in range(1, 8):
                u1, u2, amt = _ci(r.get(f'wide{i}_kumiban1')), _ci(r.get(f'wide{i}_kumiban2')), _ci(r.get(f'wide{i}_haraimodoshikin'))
                if u1 and u2 and amt:
                    p_dict['wide'][tuple(sorted([u1, u2]))] = float(amt)
            for i in range(1, 7):
                u1, u2, amt = _ci(r.get(f'umatan{i}_kumiban1')), _ci(r.get(f'umatan{i}_kumiban2')), _ci(r.get(f'umatan{i}_haraimodoshikin'))
                if u1 and u2 and amt:
                    p_dict['umatan'][(u1, u2)] = float(amt)
            for i in range(1, 4):
                u1, u2, u3, amt = _ci(r.get(f'sanrenpuku{i}_kumiban1')), _ci(r.get(f'sanrenpuku{i}_kumiban2')), _ci(r.get(f'sanrenpuku{i}_kumiban3')), _ci(r.get(f'sanrenpuku{i}_haraimodoshikin'))
                if u1 and u2 and u3 and amt:
                    p_dict['sanrenpuku'][tuple(sorted([u1, u2, u3]))] = float(amt)
            for i in range(1, 7):
                u1, u2, u3, amt = _ci(r.get(f'sanrentan{i}_kumiban1')), _ci(r.get(f'sanrentan{i}_kumiban2')), _ci(r.get(f'sanrentan{i}_kumiban3')), _ci(r.get(f'sanrentan{i}_haraimodoshikin'))
                if u1 and u2 and u3 and amt:
                    p_dict['sanrentan'][(u1, u2, u3)] = float(amt)
            p_map[rid] = p_dict
        _KELLY_HARAIMODOSHI_YEAR_CACHE[nen] = p_map
        logger.info(f"Cached haraimodoshi for year {nen}: {len(p_map)} races")
        return p_map
    except Exception as e:
        logger.warning(f"Kelly3: haraimodoshi load failed for {day}: {e}")
        return {}


def _calc_kelly3_line_payout(line, race_payout):
    """ipatgo形式の1買い目行に対する払戻額と結果確定フラグを返す。"""
    if not race_payout:
        return 0.0, False
    parts = line.split(',')
    if len(parts) < 8:
        return 0.0, False
    btype = parts[3].strip()
    comb_str = parts[6].strip()
    amt = float(parts[7].strip()) if parts[7].strip().isdigit() else 100.0
    scale = amt / 100.0
    payout = 0.0
    has_result = False

    if btype in ('TANSYO', 'TANSHO'):
        u1 = int(comb_str)
        if u1 in race_payout['tansho']:
            payout = race_payout['tansho'][u1] * scale
        has_result = len(race_payout['tansho']) > 0
    elif btype in ('FUKUSYO', 'FUKUSHO'):
        u1 = int(comb_str)
        if u1 in race_payout['fukusho']:
            payout = race_payout['fukusho'][u1] * scale
        has_result = len(race_payout['fukusho']) > 0
    elif btype == 'UMAREN':
        c = comb_str.replace('-', '')
        tup = tuple(sorted([int(c[:2]), int(c[2:4])]))
        if tup in race_payout['umaren']:
            payout = race_payout['umaren'][tup] * scale
        has_result = len(race_payout['umaren']) > 0
    elif btype == 'UMATAN':
        c = comb_str.replace('-', '')
        tup = (int(c[:2]), int(c[2:4]))
        if tup in race_payout['umatan']:
            payout = race_payout['umatan'][tup] * scale
        has_result = len(race_payout['umatan']) > 0
    elif btype == 'WIDE':
        c = comb_str.replace('-', '')
        tup = tuple(sorted([int(c[:2]), int(c[2:4])]))
        if tup in race_payout['wide']:
            payout = race_payout['wide'][tup] * scale
        has_result = len(race_payout['wide']) > 0
    elif btype == 'SANRENPUKU':
        c = comb_str.replace('-', '')
        tup = tuple(sorted([int(c[:2]), int(c[2:4]), int(c[4:6])]))
        if tup in race_payout['sanrenpuku']:
            payout = race_payout['sanrenpuku'][tup] * scale
        has_result = len(race_payout['sanrenpuku']) > 0
    elif btype == 'SANRENTAN':
        c = comb_str.replace('-', '')
        tup = (int(c[:2]), int(c[2:4]), int(c[4:6]))
        if tup in race_payout['sanrentan']:
            payout = race_payout['sanrentan'][tup] * scale
        has_result = len(race_payout['sanrentan']) > 0
    return payout, has_result


def _kelly3_action_to_sub_item(action_id, h_list, combs, cost, mult, agent_key):
    """RLアクションIDと順位リストから画面表示・投票連携用の sub_item を構築する。"""
    act = ACTIONS[action_id]
    mode = act.get('mode', '')
    code = act.get('type', '')
    jp = KELLY_BET_CODE_JP.get(code, code)
    u = [int(h[0]) for h in h_list]
    pad = lambda n: f"{int(n):02d}"
    unit = 100 * mult
    agent_label = f"現行 (x{mult}: {unit}円/点)" if agent_key == 'base' else f"3連系 (x{mult}: {unit}円/点)"

    axis1 = axis2 = None
    partners = []
    eyes = ''
    raw_type = jp

    if mode == 'tansho_1':
        axis1 = u[0]
        eyes = pad(u[0])
        raw_type = '単勝-1点'
    elif mode == 'tansho_2':
        partners = sorted(u[:2])
        eyes = ', '.join(pad(x) for x in partners)
        raw_type = '単勝-2点'
    elif mode == 'umaren_1':
        axis1, partners = u[0], [u[1]]
        eyes = f"{pad(u[0])} - {pad(u[1])}"
        raw_type = '馬連-1点'
    elif mode == 'umaren_3':
        axis1 = u[0]
        partners = sorted(u[1:min(4, len(u))])
        eyes = f"{pad(axis1)} - {', '.join(pad(x) for x in partners)}"
        raw_type = f"馬連-1頭軸{len(partners)}頭ながし"
    elif mode == 'umatan_1':
        axis1, partners = u[0], [u[1]]
        eyes = f"{pad(u[0])} → {pad(u[1])}"
        raw_type = '馬単-1点'
    elif mode == 'umatan_3':
        axis1 = u[0]
        partners = sorted(u[1:min(4, len(u))])
        eyes = f"{pad(axis1)} → {', '.join(pad(x) for x in partners)}"
        raw_type = f"馬単-1頭軸{len(partners)}頭ながし"
    elif mode == 'wide_box3':
        partners = sorted(u[:min(3, len(u))])
        eyes = f"{', '.join(pad(x) for x in partners)} BOX"
        raw_type = f"ワイド-{len(partners)}頭BOX"
    elif mode == 'sanrenpuku_2jiku_3p':
        axis1, axis2 = u[0], u[1]
        partners = sorted(u[2:5])
        eyes = f"{pad(axis1)}, {pad(axis2)} - {', '.join(pad(x) for x in partners)}"
        raw_type = '3連複-2頭軸3頭ながし'
    elif mode == 'sanrenpuku_2jiku_5p':
        axis1, axis2 = u[0], u[1]
        partners = sorted(u[2:min(7, len(u))])
        eyes = f"{pad(axis1)}, {pad(axis2)} - {', '.join(pad(x) for x in partners)}"
        raw_type = f"3連複-2頭軸{len(partners)}頭ながし"
    elif mode == 'sanrenpuku_1jiku_3p':
        axis1 = u[0]
        partners = sorted(u[1:4])
        eyes = f"{pad(axis1)} - {', '.join(pad(x) for x in partners)}"
        raw_type = '3連複-1頭軸3頭ながし'
    elif mode == 'sanrenpuku_1jiku_4p':
        axis1 = u[0]
        partners = sorted(u[1:min(5, len(u))])
        eyes = f"{pad(axis1)} - {', '.join(pad(x) for x in partners)}"
        raw_type = f"3連複-1頭軸{len(partners)}頭ながし"
    elif mode == 'sanrenpuku_1jiku_5p':
        axis1 = u[0]
        partners = sorted(u[1:min(6, len(u))])
        eyes = f"{pad(axis1)} - {', '.join(pad(x) for x in partners)}"
        raw_type = f"3連複-1頭軸{len(partners)}頭ながし"
    elif mode == 'sanrenpuku_box4':
        partners = sorted(u[:min(4, len(u))])
        eyes = f"{', '.join(pad(x) for x in partners)} BOX"
        raw_type = f"3連複-{len(partners)}頭BOX"
    elif mode == 'sanrenpuku_box5':
        partners = sorted(u[:min(5, len(u))])
        eyes = f"{', '.join(pad(x) for x in partners)} BOX"
        raw_type = f"3連複-{len(partners)}頭BOX"
    elif mode == 'sanrentan_2way':
        axis1, axis2, partners = u[0], u[1], [u[2]]
        eyes = f"{pad(u[0])} ↔ {pad(u[1])} → {pad(u[2])}"
        raw_type = '3連単-2通り'
    elif mode == 'sanrentan_box3':
        partners = sorted(u[:3])
        eyes = f"{', '.join(pad(x) for x in partners)} BOX"
        raw_type = '3連単-3頭BOX'
    elif mode == 'sanrentan_box4':
        partners = sorted(u[:min(4, len(u))])
        eyes = f"{', '.join(pad(x) for x in partners)} BOX"
        raw_type = f"3連単-{len(partners)}頭BOX"
    elif mode == 'sanrentan_form6':
        axis1 = u[0]
        s2 = sorted(u[1:3])
        s3 = sorted(u[1:5])
        partners = s3
        eyes = f"{pad(axis1)} → {', '.join(pad(x) for x in s2)} → {', '.join(pad(x) for x in s3)}"
        raw_type = '3連単-フォーメーション'
    elif mode == 'sanrentan_form8':
        axis1, axis2 = u[0], u[1]
        s12 = sorted([u[0], u[1]])
        s3 = sorted(u[2:min(6, len(u))])
        partners = s3
        eyes = f"{', '.join(pad(x) for x in s12)} → {', '.join(pad(x) for x in s12)} → {', '.join(pad(x) for x in s3)}"
        raw_type = '3連単-フォーメーション'
    elif mode == 'sanrentan_form12':
        axis1 = u[0]
        s2 = sorted(u[1:4])
        s3 = sorted(u[1:min(6, len(u))])
        partners = s3
        eyes = f"{pad(axis1)} → {', '.join(pad(x) for x in s2)} → {', '.join(pad(x) for x in s3)}"
        raw_type = '3連単-フォーメーション'
    elif mode == 'sanrentan_2jiku_multi3':
        axis1, axis2 = u[0], u[1]
        partners = sorted(u[2:min(5, len(u))])
        eyes = f"{pad(axis1)}, {pad(axis2)} ↔ {', '.join(pad(x) for x in partners)}"
        raw_type = f"3連単-2頭軸{len(partners)}頭マルチ"
    elif mode == 'sanrentan_1jiku_multi3':
        axis1 = u[0]
        partners = sorted(u[1:4])
        eyes = f"{pad(axis1)} ↔ {', '.join(pad(x) for x in partners)}"
        raw_type = '3連単-1頭軸3頭マルチ'
    elif mode == 'sanrentan_1jiku_multi4':
        axis1 = u[0]
        partners = sorted(u[1:5])
        eyes = f"{pad(axis1)} ↔ {', '.join(pad(x) for x in partners)}"
        raw_type = '3連単-1頭軸4頭マルチ'
    elif mode == 'sanrentan_1jiku_3p':
        axis1 = u[0]
        partners = sorted(u[1:4])
        eyes = f"{pad(axis1)} → {', '.join(pad(x) for x in partners)}"
        raw_type = '3連単-1頭軸3頭流し'
    elif mode == 'sanrentan_1jiku_4p':
        axis1 = u[0]
        partners = sorted(u[1:5])
        eyes = f"{pad(axis1)} → {', '.join(pad(x) for x in partners)}"
        raw_type = '3連単-1頭軸4頭流し'
    else:
        partners = sorted(u[:min(4, len(u))])
        eyes = ', '.join(pad(x) for x in partners)
        raw_type = act.get('name', jp)

    return {
        'type': code,
        'rawType': raw_type,
        'bet_type_jp': jp,
        'action_id': int(action_id),
        'action_name': act.get('name', raw_type),
        'agent': agent_key,
        'agent_label': agent_label,
        'unit': int(unit),
        'combs': int(combs),
        'cost': int(cost),
        'bettingEyesText': eyes,
        'axis1Num': axis1,
        'axis2Num': axis2,
        'partnerNums': partners,
    }


def build_kelly3_portfolio_for_day(day, races_of_day):
    """Kelly3.ipynb / kelly3_auto_runner.py と同一ロジック (K3新案2: v12合議×市場残差統合・3レーン利確縮小継続) で
    当日の買い目を生成し、画面表示用の strat2 (sub_items 付き) を返す。
    - 全レースで3連単・3連複を購入 (SKIPなし)
    - レーン利確 (+5,000円) 到達後は停止せず 200〜300円の最小3連系へ縮小して完走
    - 1日予算目安 50,000円 (残り4,000円以下で budget_tight 縮小)
    """
    if not residual_v1_available:
        logger.warning("Kelly3: predict_residual モジュールが利用できないため生成をスキップ")
        return {}

    day_str = str(day)
    loaded = _load_kelly3_score_df(day_str)
    if loaded is None:
        logger.warning(f"Kelly3: no score CSV for {day_str} -> PICKUP 2 は生成しません")
        return {}
    df_scores, score_models = loaded

    # --- オッズ (DB) ---
    odds_df = _load_tansho_odds_df(day_str)
    odds_lk = {}
    if not odds_df.empty:
        for rc, ub, ov in zip(odds_df['race_code'], odds_df['umaban'], odds_df['odds_val']):
            rc_s = str(rc)
            if len(rc_s) >= 16 and pd.notnull(ov) and float(ov) > 0:
                rid_o = rc_s[:4] + rc_s[8:16]
                odds_lk[(rid_o, int(ub))] = float(ov)

    # --- residual_v1 マージン (Kelly2 のキャッシュを再利用) ---
    res_pack = _get_or_compute_residual_for_day(day_str)
    res_margin_map = res_pack['margin_map'] if res_pack else {}
    shinba_rids = res_pack['shinba_rids'] if res_pack else set()

    # --- レースごとのグループ化 ---
    from collections import Counter
    score_by_race = {str(rid): grp for rid, grp in df_scores.groupby('race_id')}

    lane_names = ['Lane_A', 'Lane_B', 'Lane_C']
    lane_names_jp = ['レーンA', 'レーンB', 'レーンC']
    sorted_rids = sorted(
        [str(rid) for rid in races_of_day.keys() if str(rid) in score_by_race and len(score_by_race[str(rid)]) >= 5],
        key=lambda x: (races_of_day[x].get('start_time', '99:99'), x),
    )
    if not sorted_rids:
        return {}

    races_list = []
    for idx, rid_str in enumerate(sorted_rids):
        r_info = races_of_day[rid_str]
        meta = dict(r_info.get('meta') or {})
        round_val = str(r_info.get('round', '1'))
        race_num = int(round_val) if round_val.isdigit() else (int(rid_str[10:12]) if len(rid_str) >= 12 and rid_str[10:12].isdigit() else 1)
        meta['race_num'] = race_num
        meta['track_name'] = meta.get('track_name') or KELLY_TRACK_MAPPING.get(rid_str[4:6], 'UNKNOWN')
        class_cat = meta.get('class_cat', 'その他')
        if class_cat != '新馬' and rid_str in shinba_rids:
            class_cat = '新馬'
        meta['class_cat'] = class_cat

        lid = idx % 3
        races_list.append({
            'race_id': rid_str,
            'time': r_info.get('start_time', '99:99'),
            'lane_id': lid,
            'lane': lane_names[lid],
            'lane_jp': lane_names_jp[lid],
            'meta': meta,
            'df_g': score_by_race[rid_str],
        })

    # 払戻金マップ (確定済みレースがあれば利確縮小判定に反映、開催前は空)
    p_map = _load_haraimodoshi_map(day_str)

    sanren_stopped = [False, False, False]
    purchased_history = []
    total_spent_today = 0

    time_order_keys = []
    time_buckets = {}
    for r in races_list:
        t = r['time']
        if t not in time_buckets:
            time_order_keys.append(t)
            time_buckets[t] = []
        time_buckets[t].append(r)

    result = {}
    for r_time in time_order_keys:
        g_races = time_buckets[r_time]

        # レーン別収支集計 & 利確縮小判定 (+5,000円)
        s_spents, s_rets, s_has_pay = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [False, False, False]
        for b in purchased_history:
            lid_b = b['lane_id']
            rid_b = str(b['race_id'])
            pval, hres = _calc_kelly3_line_payout(b['line'], p_map.get(rid_b)) if rid_b in p_map else (0.0, False)
            s_spents[lid_b] += b['cost']
            s_rets[lid_b] += pval
            if hres:
                s_has_pay[lid_b] = True

        for lid_c in range(3):
            if not sanren_stopped[lid_c] and s_has_pay[lid_c]:
                if (s_rets[lid_c] - s_spents[lid_c]) >= KELLY3_SANREN_LANE_STOP:
                    sanren_stopped[lid_c] = True

        bucket_bets = []
        for r in g_races:
            lid = r['lane_id']
            m_info = r['meta']
            rid_str = r['race_id']
            g = r['df_g']

            # 5モデル1位合議度
            firsts = [
                int(g.sort_values(m, ascending=False)['horse_num'].iloc[0])
                for m in KELLY3_AGREE_MODELS
                if m in g.columns and len(g) > 0
            ]
            agree = Counter(firsts).most_common(1)[0][1] / float(len(firsts)) if firsts else 0.5

            gs = g.sort_values('Ensemble', ascending=False).reset_index(drop=True)
            sv = gs['Ensemble'].values.astype(float)
            zv = (sv - sv.mean()) / (sv.std() if len(sv) > 1 and sv.std() > 0 else 1.0)
            gs['z_v12'] = zv
            gs['rk_v12'] = np.arange(1, len(gs) + 1)

            mw_l, mp_l, od_l = [], [], []
            for u_val, z_val in zip(gs['horse_num'].astype(int), gs['z_v12'].values):
                mw_v, mp_v, fb_o = res_margin_map.get((rid_str, u_val), (0.25 * float(z_val), 0.25 * float(z_val), np.nan))
                o_v = odds_lk.get((rid_str, u_val), fb_o if pd.notna(fb_o) and fb_o > 0 else 20.0)
                mw_l.append(mw_v)
                mp_l.append(mp_v)
                od_l.append(o_v)

            sc_df = scores_from_margins_and_odds(
                np.full(len(gs), rid_str),
                gs['horse_num'].astype(int).values,
                np.array(od_l, dtype=float),
                np.array(mw_l, dtype=float),
                np.array(mp_l, dtype=float),
            )
            sc_df['z_v12'] = gs['z_v12'].values
            sc_df['rk_v12'] = gs['rk_v12'].values

            is_budget_tight = (total_spent_today + sum(x['cost'] for x in bucket_bets)) >= (KELLY3_MAX_BUDGET - 4000)
            plan = generate_kelly3_plan2_race_bets(
                date_str=day_str,
                track_name=m_info['track_name'],
                race_no=m_info['race_num'],
                g_race=sc_df,
                agree=agree,
                class_cat=m_info.get('class_cat', 'その他'),
                lane_locked=sanren_stopped[lid],
                budget_tight=is_budget_tight,
            )

            sub_items = plan.get('sub_items') or []
            race_lines = plan.get('lines') or []
            if not sub_items or not race_lines:
                continue

            for ln in race_lines:
                parts = ln.split(',')
                l_cost = int(parts[-1]) if parts[-1].isdigit() else 100
                bucket_bets.append({
                    'race_id': rid_str,
                    'lane_id': lid,
                    'agent': 'sanren',
                    'cost': l_cost,
                    'line': ln,
                })

            axis1_u = plan.get('axis1')
            partners_u = list(plan.get('partners') or [])
            top_list = [axis1_u] + partners_u[:3] if axis1_u is not None else gs['horse_num'].astype(int).tolist()[:4]

            result[rid_str] = {
                'action_id': sub_items[0]['action_id'],
                'action_name': ' + '.join(si['action_name'] for si in sub_items),
                'is_pickup': True,
                'cost': int(sum(si['cost'] for si in sub_items)),
                'combs': int(sum(si['combs'] for si in sub_items)),
                'axis1Num': axis1_u,
                'axis2Num': None,
                'partnerNums': partners_u,
                'bettingEyesText': ' / '.join(si['bettingEyesText'] for si in sub_items),
                'rawType': 'Kelly3-v12×残差統合',
                'lines': race_lines,
                'model': 'Kelly3 v12×Residual Hybrid (Plan 2)',
                'sub_items': sub_items,
                'lane': r['lane'],
                'lane_jp': r['lane_jp'],
                'top_horses': '-'.join(str(x) for x in top_list if x is not None),
                'version': KELLY3_PLAN2_VERSION,
            }

        if bucket_bets:
            total_spent_today += sum(x['cost'] for x in bucket_bets)
            purchased_history.extend(bucket_bets)

    logger.info(f"Kelly3 (Plan 2 v12×residual): {len(result)} races / total_spent={total_spent_today:,}円 for {day_str}")
    return result


def generate_static_html():
    eval_dir = r"C:\Users\kyoui\keiba\data\eval"
    output_html_path = r"C:\Users\kyoui\tohshin_keiba\index.html"
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

    # AutoGluon スコアの読み込み (C:\keibasoftcom\KSCAutoBetPlus\TCSV\v12 のみ参照)
    tcsv_dir = r"C:\keibasoftcom\KSCAutoBetPlus\TCSV\v12"
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
                    df_ag['AutoGluon_raw_v12'] = pd.to_numeric(df_ag['score'], errors='coerce')
                    valid_ag = df_ag[['ag_race_id', 'ag_horse_num', 'AutoGluon_raw_v12']].dropna(subset=['ag_race_id', 'ag_horse_num'])
                    ag_dfs.append(valid_ag)
                except Exception as e:
                    logger.error(f"Error reading {fp_ag}: {e}")
        if ag_dfs:
            all_ag = pd.concat(ag_dfs, ignore_index=True).drop_duplicates(subset=['ag_race_id', 'ag_horse_num'], keep='last')
            df['temp_rid'] = df[race_id_col].astype(str)
            df['temp_hnum'] = pd.to_numeric(df[horse_num_col], errors='coerce')
            df = df.reset_index(drop=True)
            df = pd.merge(df, all_ag, left_on=['temp_rid', 'temp_hnum'], right_on=['ag_race_id', 'ag_horse_num'], how='left')
            if 'AutoGluon_raw' in df.columns:
                df['AutoGluon_raw'] = df['AutoGluon_raw_v12'].combine_first(pd.to_numeric(df['AutoGluon_raw'], errors='coerce')).fillna(0.0)
            else:
                df['AutoGluon_raw'] = df['AutoGluon_raw_v12'].fillna(0.0)
            df.drop(columns=['temp_rid', 'temp_hnum', 'ag_race_id', 'ag_horse_num', 'AutoGluon_raw_v12'], errors='ignore', inplace=True)
            logger.info(f"Merged AutoGluon v12 scores: {(df['AutoGluon_raw'] > 0).sum()} valid entries")
        elif 'AutoGluon_raw' not in df.columns:
            df['AutoGluon_raw'] = 0.0
    elif 'AutoGluon_raw' not in df.columns:
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

                            dist_val = f_row.get('course_len', f_row.get('distance', 0))
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

    # 各日付ごとに発走時刻を付与・タイトル更新 & PICKUP生成
    jsons_dir = r"C:\Users\kyoui\tohshin_keiba\jsons"
    recalc_cutoff_str = (datetime.now() - timedelta(days=KELLY2_RECENT_RECALC_DAYS)).strftime('%Y-%m-%d')
    reused_dates = []
    updated_dates = []

    def _is_existing_residual_v1_json(json_path, expected_rids):
        if not os.path.exists(json_path):
            return False
        try:
            with open(json_path, "r", encoding="utf-8") as rf:
                ex = json.load(rf)
            if not isinstance(ex, dict) or not expected_rids.issubset(ex.keys()):
                return False
            has_k2 = False
            has_k3 = False
            for r_obj in ex.values():
                k2 = r_obj.get('kelly2')
                if isinstance(k2, dict) and k2.get('version') == KELLY2_RESIDUAL_VERSION:
                    has_k2 = True
                s2 = r_obj.get('strat2')
                if isinstance(s2, dict) and (
                    s2.get('version') == KELLY3_PLAN2_VERSION
                    or 'Plan 2' in str(s2.get('model', ''))
                ):
                    has_k3 = True
            return has_k2 and has_k3
        except Exception:
            return False

    for d, d_races in sorted(dates_data.items()):
        out_json = os.path.join(jsons_dir, f"data_{d}.json")
        # 直近N日より前の過去日で、すでに residual_v1 のJSONが存在する場合は再計算・書き出しをスキップ
        if d < recalc_cutoff_str and _is_existing_residual_v1_json(out_json, set(d_races.keys())):
            reused_dates.append(d)
            continue

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
        # PICKUP (戦略1) : Kelly2.ipynb (residual_v1) と同一ロジックの買い目
        # PICKUP 2 (戦略2): Kelly3 (RLハイブリッド) の買い目
        # ==================================================================
        day_str = d.replace('-', '')
        kelly2_map = build_kelly2_bets_for_day(day_str, d_races)
        if kelly2_map is not None:
            for k_rid in d_races:
                d_races[k_rid]['kelly2'] = kelly2_map.get(
                    k_rid, {'bets': [], 'total_cost': 0, 'version': KELLY2_RESIDUAL_VERSION}
                )

        kelly3_map = build_kelly3_portfolio_for_day(day_str, d_races)
        if kelly3_map:
            for k_rid, strat2_item in kelly3_map.items():
                if k_rid in d_races:
                    d_races[k_rid]['strat2'] = strat2_item

        # 1. 再計算した日付のみ JSON を保存
        try:
            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(d_races, f, ensure_ascii=False)
            updated_dates.append(d)
            logger.info(f"Generated daily JSON: {out_json}")
        except Exception as e:
            logger.error(f"Failed to write daily JSON {out_json}: {e}")

    logger.info(
        f"Daily JSON summary (cutoff={recalc_cutoff_str}): "
        f"reused={len(reused_dates)} past dates, updated={len(updated_dates)} dates"
    )

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
                <option value="PICKUP1">🎯 PICKUP_v13</option>
                <option value="PICKUP2">🚀 PICKUP_v12</option>
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
                            <div style="font-weight: 900; letter-spacing: 0.05em; color: #fff;">PICKUP_v13</div>
                        </div>
                    `;
                }})();

                const pickup2BadgeHtml = ( () => {{
                    if (!hasPickup2) return '';
                    return `
                        <div class="pickup2-badge" onclick="event.stopPropagation(); showRecommendation('${{raceId}}', 'strat2')">
                            <span style="font-size: 0.6rem; opacity: 0.85; font-weight: 700; color: #e9d5ff;">STRAT</span>
                            <div style="font-weight: 900; letter-spacing: 0.05em; color: #fff;">PICKUP_v12</div>
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
                        tag: 'PICKUP_v13',
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
                                tag: 'PICKUP_v12',
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
                            tag: 'PICKUP_v12',
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
                        h1PopRank: b.h1PopRank,
                        detailText: b.detailText || ''
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
                        ${{currentActiveRecTab === 'strat2' ? 'Kelly3 新案2 (v12合議×市場残差統合)' : 'Kelly2 改善案B (市場残差アンサンブル)'}}
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
                        🎯 PICKUP_v13 (Kelly2)${{strat1List.length > 0 ? ` <span style="opacity:0.9; font-size:0.75rem;">(${{strat1List.length}})</span>` : ''}}
                    </button>
                    <button class="rec-tab-btn strat2 ${{currentActiveRecTab === 'strat2' ? 'active' : ''}}" onclick="switchRecTab('${{raceId}}', 'strat2')">
                        🚀 PICKUP_v12 (Kelly3)${{strat2List.length > 0 ? ` <span style="opacity:0.9; font-size:0.75rem;">(推奨)</span>` : ''}}
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
                        const unitPrice = item.unit || 100;
                        const totalCost = item.cost || (unitPrice * (item.combs || 1));

                        let leftFooter = '';
                        let rightFooter = '';
                        if (item.detailText) {{
                            leftFooter = `${{item.detailText}}${{popDisp ? ' / ' + popDisp : ''}}`;
                            rightFooter = `投資額: <strong style="color: #4ade80;">¥${{totalCost.toLocaleString()}}</strong> (${{unitPrice.toLocaleString()}}円 × ${{item.combs}}点)`;
                        }} else {{
                            const axisLabel = item.axis1Num && item.axis2Num
                                ? `軸馬: <strong>${{String(item.axis1Num).padStart(2, '0')}}番, ${{String(item.axis2Num).padStart(2, '0')}}番</strong>`
                                : (item.axis1Num ? `軸馬: <strong>${{String(item.axis1Num).padStart(2, '0')}}番</strong>` : '');
                            const partnerLabel = item.partnerNums && item.partnerNums.length > 0
                                ? `${{axisLabel ? ' | ' : ''}}相手: <strong>${{item.partnerNums.map(n => String(n).padStart(2, '0')).join(', ')}}</strong>`
                                : '';
                            leftFooter = `${{axisLabel}}${{partnerLabel}}${{(confDisp || popDisp) ? ` (${{confDisp}}${{confDisp && popDisp ? ' / ' : ''}}${{popDisp}})` : ''}}`;
                            rightFooter = `ROI: <strong style="color: #4ade80;">${{item.roi}}%</strong> | 的中率: <strong style="color: #60a5fa;">${{item.hitRate}}%</strong>`;
                        }}

                        html += `
                            <div class="strategy-item-modal" data-strategy-type="${{item.rawType}}" data-unit="${{unitPrice}}">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                                    <div style="font-weight: 900; color: #fbbf24; font-size: 1.1rem;">
                                        ${{displayType}} 
                                        <span style="font-size: 0.75rem; color: #60a5fa; margin-left:8px; font-weight:700; background: rgba(96, 165, 250, 0.1); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(96, 165, 250, 0.2);">${{item.model}}</span>
                                        <span style="font-size: 0.75rem; color: #4ade80; margin-left:6px; font-weight:700; background: rgba(74, 222, 128, 0.1); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(74, 222, 128, 0.2);">${{item.combs}}点 / ¥${{totalCost.toLocaleString()}}</span>
                                    </div>
                                </div>
                                <div class="bet-eyes-box">
                                    <div style="font-size: 0.7rem; color: var(--text-muted); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.1em;">Recommended Combination</div>
                                    <div class="bet-eyes-text">${{item.rawType.includes('2通り') ? item.bettingEyesText.replace(', ', '<br>') : item.bettingEyesText}}</div>
                                </div>
                                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; font-size: 0.75rem; color: var(--text-muted); margin-top: 8px;">
                                    <div style="color: #94a3b8;">
                                        ${{leftFooter}}
                                    </div>
                                    <div>
                                        ${{rightFooter}}
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
                // 戦略2 (Kelly3 / PICKUP 2): kelly3_auto_runner.py 強化学習ハイブリッド (現行x5 + 3連系x1)
                const s2 = raceData.strat2;
                if (s2) {{
                    const isPickup = s2.is_pickup === true || (s2.action_id > 0);
                    const actionBadgeColor = isPickup ? '#fb923c' : '#94a3b8';
                    const actionBadgeBg = isPickup ? 'rgba(249, 115, 22, 0.15)' : 'rgba(255, 255, 255, 0.05)';
                    const borderLeftColor = isPickup ? '#fb923c' : '#64748b';
                    const statusText = isPickup ? '🚀 PICKUP_v12 (積極購入推奨)' : '見送り (SKIP)';
                    const pad = (n) => String(n).padStart(2, '0');

                    // 1. ハイブリッド等で複数券種に分割されている場合
                    if (s2.sub_items && s2.sub_items.length > 0) {{
                        html += `
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; padding: 12px 16px; background: rgba(249, 115, 22, 0.08); border-radius: 10px; border: 1px solid rgba(249, 115, 22, 0.25);">
                                <div>
                                    <div style="font-weight: 900; color: #fb923c; font-size: 1.05rem;">
                                        ${{s2.action_name}}
                                        <span style="font-size: 0.75rem; color: #60a5fa; margin-left:8px; font-weight:700; background: rgba(96, 165, 250, 0.1); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(96, 165, 250, 0.2);">${{s2.model || 'Kelly3 RL Hybrid'}}</span>
                                    </div>
                                    <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 4px;">
                                        合計: <strong style="color: #4ade80;">${{s2.combs}}点 (${{s2.cost.toLocaleString()}}円)</strong> / ${{s2.lane_jp || ''}} (強化学習ハイブリッド: 現行500円・3連系100円)
                                    </div>
                                </div>
                                <div style="font-size: 0.7rem; color: #fb923c; font-weight: 700; background: rgba(249, 115, 22, 0.2); padding: 4px 10px; border-radius: 4px; border: 1px solid rgba(249, 115, 22, 0.4);">
                                    ${{statusText}}
                                </div>
                            </div>
                        `;

                        s2.sub_items.forEach((sub, subIdx) => {{
                            const isSanren = sub.agent === 'sanren' || sub.type === 'SANRENPUKU' || sub.type === 'SANRENTAN';
                            const subColor = isSanren ? '#fbbf24' : '#38bdf8';
                            const subBg = isSanren ? 'rgba(251, 191, 36, 0.06)' : 'rgba(56, 189, 248, 0.06)';
                            const subBorder = isSanren ? 'rgba(251, 191, 36, 0.3)' : 'rgba(56, 189, 248, 0.3)';
                            const subLabel = sub.agent_label || sub.bet_type_jp || sub.rawType.split('-')[0];

                            html += `
                                <div class="strategy-item-modal" data-strategy-type="${{sub.rawType}}" data-unit="${{sub.unit || 100}}" style="border-left: 4px solid ${{subColor}}; margin-bottom: 14px;">
                                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 6px;">
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
                                        <span style="font-size: 0.75rem; color: #60a5fa; margin-left:8px; font-weight:700; background: rgba(96, 165, 250, 0.1); padding: 2px 8px; border-radius: 4px; border: 1px solid rgba(96, 165, 250, 0.2);">${{s2.model || 'Kelly3 RL Hybrid'}}</span>
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
                                        Kelly3 強化学習ハイブリッド (kelly3_auto_runner.py)
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
                            <div style="font-weight: 700; font-size: 0.8rem;">Kelly3 (強化学習ハイブリッド) は見送りまたは利確済です</div>
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

                // Kelly3 ポートフォリオ/RLハイブリッド表示用の券種も扱う
                let formationStages = null;
                if (normType.includes("フォーメーション") && eyesText.includes("→")) {{
                    // 1着候補 → 2着候補 → 3着候補 (3連単フォーメーション)
                    const stages = eyesText.split("→").map(s =>
                        s.split(",").map(t => String(t).trim().replace(/^0+/, "")).filter(Boolean)
                    );
                    if (stages.length === 3 && stages[0].length > 0 && stages[1].length > 0 && stages[2].length > 0) {{
                        formationStages = stages;
                        let cnt = 0;
                        stages[0].forEach(a => stages[1].forEach(b => {{
                            if (b === a) return;
                            stages[2].forEach(c => {{
                                if (c !== a && c !== b) cnt++;
                            }});
                        }}));
                        eyesCount = cnt;
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
                
                if (formationStages) {{
                    // フォーメーション点数は上で計算済み
                }} else if (normType.includes("2通り")) {{
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
                    const predictedSet = eyesText.replace(/BOX/gi, '').split(/[→↔,-]/).map(s => s.trim().replace(/^0+/, '')).filter(Boolean);
                    const isMulti = normType.includes("マルチ") || normType.includes("BOX") || normType.includes("3連複") || normType.includes("馬連") || normType.includes("ワイド");

                    if (formationStages && winNums.length >= 3) {{
                        isHit = formationStages[0].includes(winNums[0])
                            && formationStages[1].includes(winNums[1])
                            && formationStages[2].includes(winNums[2]);
                    }} else if (normType.includes("2通り")) {{
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

                const unitAmt = parseInt(item.getAttribute('data-unit') || '100', 10) || 100;
                if (unitAmt !== 100 && totalPay > 0) {{
                    totalPay = Math.round(totalPay * (unitAmt / 100));
                }}
                const investment = eyesCount * unitAmt;
                const profit = totalPay - investment;

                if (resultArea) {{
                    resultArea.innerHTML = `
                        <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.05); display: flex; justify-content: space-between; font-size: 0.85rem;">
                            <div><span style="color:var(--text-muted);">Bets:</span> <span style="color:#fff; font-weight:700;">${{eyesCount}}点(${{investment.toLocaleString()}}円)</span></div>
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