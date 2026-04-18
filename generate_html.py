import os
import json
import logging
import pandas as pd
import numpy as np
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def generate_static_html():
    eval_dir = r"C:\Users\kyoui\keiba\data\eval"
    output_html_path = r"C:\Users\kyoui\tohshin_keiba\index.html"
    strategies_csv_path = r"C:\Users\kyoui\keiba\config\winning_strategies.csv"
    
    # 戦略データの読み込み
    strategies_dict = {}
    if os.path.exists(strategies_csv_path):
        try:
            sdf = pd.read_csv(strategies_csv_path)
            # NaNをNoneに置き換える (JSONでnullとして出力される)
            # 数値型のままだとNoneがnp.nanに戻ってしまうため、一度物オブジェクト型に変換する
            sdf = sdf.astype(object).where(pd.notnull(sdf), None)
            
            # 会場名または venue_code でマッピングできるように準備
            # venue_name をキーにしたリストを作成
            for _, row in sdf.iterrows():
                v_name = str(row['venue_name'])
                if v_name not in strategies_dict:
                    strategies_dict[v_name] = []
                strategies_dict[v_name].append(row.to_dict())
            logger.info(f"Loaded {len(sdf)} strategies from {strategies_csv_path}")
        except Exception as e:
            logger.error(f"Error loading strategies CSV: {e}")
    else:
        logger.warning(f"Strategies CSV not found: {strategies_csv_path}")

    logger.info(f"Loading all picke files from {eval_dir}...")
    
    if not os.path.exists(eval_dir):
        logger.error(f"Directory not found: {eval_dir}")
        return

    import glob
    pickle_files = glob.glob(os.path.join(eval_dir, "*.pickle"))
    if not pickle_files:
        logger.error(f"No pickle files found in {eval_dir}")
        return
    
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
            
            all_dfs.append(df_part)
        
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
        'Ensemble': 'Python'
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

    req_scores = ['LightGBM_raw', 'XGBoost_raw', 'CatBoost_raw', 'LSTM_raw', 'RandomForest_raw', 'DecisionTree_raw', 'Transformer_raw', 'TabNet_raw', 'Ensemble']
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
            
        if date_val and place_name and round_int:
            race_title = f"{date_val} {place_name} {round_int}R"
        elif place_name and round_int:
            race_title = f"{place_name} {round_int}R"
        else:
            race_title = f"Race {race_id_str}"
            
        # この会場に対応する推奨戦略を取得
        race_strategies = strategies_dict.get(place_name, []) + strategies_dict.get('全場', [])
            
        races[race_id_str] = {
            "race_id": race_id_str,
            "title": race_title,
            "date": date_val,
            "place": place_name,
            "round": str(round_int),
            "horses": records,
            "strategies": race_strategies
        }

    # データを日付ごとにグループ化
    dates_data = {}
    for r_id, r_info in races.items():
        d = r_info.get('date', 'unknown')
        if d not in dates_data:
            dates_data[d] = {}
        dates_data[d][r_id] = r_info

    # 1. 各日付のデータを保存
    for d, d_races in dates_data.items():
        out_json = os.path.join(r"C:\Users\kyoui\tohshin_keiba\jsons", f"data_{d}.json")
        try:
            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(d_races, f, ensure_ascii=False)
            logger.info(f"Generated daily JSON: {out_json}")
        except Exception as e:
            logger.error(f"Failed to write daily JSON {out_json}: {e}")

    # 2. メタデータ（日付リスト）を保存
    meta_data = {
        "dates": sorted(list(dates_data.keys())),
        "latest": max(dates_data.keys()) if dates_data else ""
    }
    meta_json_path = r"C:\Users\kyoui\tohshin_keiba\jsons\meta.json"
    try:
        with open(meta_json_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, ensure_ascii=False)
        logger.info(f"Generated meta.json at {meta_json_path}")
    except Exception as e:
        logger.error(f"Failed to write meta.json: {e}")

    # (互換性維持) 全データをJSON文字列化して保存
    # 注意: ファイルサイズが巨大化(100MB超)するため、本番運用では生成・Gitプッシュを停止しました。
    # json_data = json.dumps(races, ensure_ascii=False)
    # output_json_paths = [
    #     r"C:\Users\kyoui\tohshin_keiba\jsons\data.json"
    # ]
    # for out_json in output_json_paths:
    #     try:
    #         os.makedirs(os.path.dirname(out_json), exist_ok=True)
    #         with open(out_json, "w", encoding="utf-8") as f:
    #             f.write(json_data)
    #         logger.info(f"Successfully generated full JSON data at {out_json}")
    #     except Exception as e:
    #         logger.error(f"Failed to write full JSON to {out_json}: {e}")

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

        .race-list {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .race-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 20px;
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
            cursor: pointer;
            animation: slideUp 0.6s ease forwards;
            opacity: 0;
            transform: translateY(20px);
        }}

        .race-card:hover {{
            transform: translateY(-4px) scale(1.02);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
            border-color: rgba(255, 255, 255, 0.15);
        }}

        .race-info-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
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
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 20;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: linear-gradient(135deg, rgba(74, 222, 128, 0.25), rgba(59, 130, 246, 0.25));
            border: 1px solid rgba(74, 222, 128, 0.5);
            color: #4ade80;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 900;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            backdrop-filter: blur(12px);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4), 0 0 15px rgba(74, 222, 128, 0.2);
            letter-spacing: 0.05em;
        }}

        .pickup-badge:hover {{
            background: linear-gradient(135deg, rgba(74, 222, 128, 0.4), rgba(59, 130, 246, 0.4));
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5), 0 0 30px rgba(74, 222, 128, 0.5);
            transform: translateY(-2px) scale(1.08);
        }}

        /* Modal Overlay */
        #recommend-modal {{
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
            white-space: nowrap;
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

        .race-card {{
            position: relative;
            overflow: hidden; /* Ensure highlighting doesn't overflow rounded corners */
        }}
        .smappy-btn {{
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            border: none;
            color: #fff;
            padding: 8px 16px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 800;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }}
        .smappy-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
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
    </style>
</head>
<body>
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
            <select id="sort-select" onchange="renderRaces()">
                <option value="score">Sort by AI Score</option>
                <option value="odds">Sort by Odds</option>
                <option value="horse_number">Sort by Horse Number</option>
            </select>
            <select id="model-select" onchange="renderRaces()" style="display: none;">
                <option value="Ensemble">Ensemble</option>
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

        <div id="races-container" class="race-list"></div>
    </div>

    <!-- Recommendation Modal -->
    <div id="recommend-modal" onclick="if(event.target===this) closeRecommendation()">
        <div class="modal-content">
            <span class="modal-close" onclick="closeRecommendation()">&times;</span>
            <div id="modal-body"></div>
        </div>
    </div>

    <script>
        console.log("[DEBUG] Keiba AI Script Initializing...");
        let currentData = {{}};

        
        
        function checkAuth() {{
            var input = document.getElementById('auth-pw');
            if (!input) {{ alert('Input not found!'); return; }}
            var pw = (input.value || "").trim();
            if (pw === '8888') {{
                try {{
                    localStorage.setItem('keiba_auth_time', new Date().getTime());
                }} catch (e) {{}}
                var overlay = document.getElementById('auth-overlay');
                if (overlay) overlay.remove();
                document.getElementById('app-content').style.display = 'block';
                if (typeof loadData === 'function') loadData();
            }} else {{
                alert('パスワードが違います');
                document.getElementById('login-error').style.display = 'block';
            }}
        }}

        window.onload = function() {{
            var isAuthenticated = false;
            try {{
                var authTime = localStorage.getItem('keiba_auth_time');
                if (authTime) {{
                    var now = new Date().getTime();
                    var diffHours = (now - parseInt(authTime)) / (1000 * 60 * 60);
                    if (diffHours < 24) isAuthenticated = true;
                }}
            }} catch (e) {{}}
            
            if (isAuthenticated) {{
                var overlay = document.getElementById('auth-overlay');
                if (overlay) overlay.remove();
                document.getElementById('app-content').style.display = 'block';
                if (typeof loadData === 'function') loadData();
            }} else {{
                document.getElementById('auth-overlay').style.display = 'flex';
                document.getElementById('app-content').style.display = 'none';
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
            const container = document.getElementById('races-container');
            container.innerHTML = '';
            
            const sortBy = document.getElementById('sort-select').value;
            const sortModel = document.getElementById('model-select').value;
            const mSelect = document.getElementById('model-select');
            const fDate = document.getElementById('filter-date').value;
            const fPlace = document.getElementById('filter-place').value;
            const fRound = document.getElementById('filter-round').value;

            // Show model selector only when sorting by score
            if (sortBy === 'score') {{
                mSelect.style.display = 'inline-block';
            }} else {{
                mSelect.style.display = 'none';
            }}

            let delay = 0;

            for (const [raceId, raceData] of Object.entries(currentData)) {{
                
                // Filtering
                if (fDate !== 'ALL' && raceData.date !== fDate) continue;
                if (fPlace !== 'ALL' && raceData.place !== fPlace) continue;
                if (fRound !== 'ALL' && String(raceData.round) !== String(fRound)) continue;

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
                const scoreModels = ['LightGBM_raw', 'XGBoost_raw', 'CatBoost_raw', 'LSTM_raw', 'RandomForest_raw', 'DecisionTree_raw', 'Transformer_raw', 'TabNet_raw', 'Ensemble'];
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

                // --- 2. Strategy Highlighting Calculation (Z-Score + EV Based Thresholding) ---
                const jikuSet = new Set();
                const partnerSet = new Set();

                (raceData.strategies || []).forEach(s => {{
                    const scoreKey = s.model === 'Ensemble' ? 'Ensemble' : s.model + "_raw";
                    const allSorted = [...raceData.horses].sort((a,b) => getZ(b, scoreKey) - getZ(a, scoreKey));
                    if (allSorted.length === 0) return;

                    const sTh = s.score_th === null ? -9.9 : parseFloat(s.score_th);
                    const a2Th = (s.axis2_score_th === null && s.partner_score_th === null) ? -9.9 : parseFloat(s.axis2_score_th || s.partner_score_th);
                    const pTh = s.partner_score_th === null ? -9.9 : parseFloat(s.partner_score_th);
                    
                    const evTh = s.EV_th === null ? -1 : parseFloat(s.EV_th);
                    const a2EvTh = (s.axis2_EV_th === null && s.partner_EV_th === null) ? -1 : parseFloat(s.axis2_EV_th || s.partner_EV_th);
                    const pEvTh = s.partner_EV_th === null ? -1 : parseFloat(s.partner_EV_th);

                    if (s.type === "単勝") {{
                        const h = allSorted[0];
                        if (getZ(h, scoreKey) >= sTh && getEV(h, scoreKey) >= evTh) {{
                            jikuSet.add(String(h.horse_number));
                        }}
                    }} else if (s.type.includes("BOX")) {{
                        const count = parseInt(s.partners) || 5;
                        const valid = allSorted.filter(h => getZ(h, scoreKey) >= sTh && getEV(h, scoreKey) >= evTh).slice(0, count);
                        if (valid.length >= (s.type.includes("3連") ? 3 : 2)) {{
                            valid.forEach(h => partnerSet.add(String(h.horse_number)));
                        }}
                    }} else {{
                        const axisCount = parseInt(s.axis_count) || 1;
                        const partnerCount = parseInt(s.partners) || 5;
                        
                        // 1軸目の判定
                        const axes1 = allSorted.filter(h => getZ(h, scoreKey) >= sTh && getEV(h, scoreKey) >= evTh).slice(0, 1);
                        if (axes1.length > 0) {{
                            let finalAxes = [...axes1];
                            let others = allSorted.filter(h => h.horse_number !== axes1[0].horse_number);
                            
                            // 2軸目がある場合
                            if (axisCount >= 2) {{
                                const axis2Candidate = others.filter(h => getZ(h, scoreKey) >= a2Th && getEV(h, scoreKey) >= a2EvTh).slice(0, axisCount - 1);
                                if (axis2Candidate.length < axisCount - 1) {{
                                    return; // 2軸目が条件を満たさない場合はこの戦略はスキップ
                                }}
                                finalAxes = finalAxes.concat(axis2Candidate);
                                const axis2Numbers = new Set(axis2Candidate.map(ax => ax.horse_number));
                                others = others.filter(h => !axis2Numbers.has(h.horse_number));
                            }}

                            const partners = others.filter(h => getZ(h, scoreKey) >= pTh && getEV(h, scoreKey) >= pEvTh).slice(0, partnerCount);
                            const minTotal = s.type.includes("3連") ? 3 : 2;
                            if (finalAxes.length + partners.length >= minTotal) {{
                                finalAxes.forEach(h => jikuSet.add(String(h.horse_number)));
                                partners.forEach(h => partnerSet.add(String(h.horse_number)));
                            }}
                        }}
                    }}
                }});


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

                const card = document.createElement('div');
                card.className = 'race-card';
                card.style.animationDelay = `${{delay}}ms`;
                delay += 50;

                let horsesHtml = '';
                sortedHorses.forEach((horse, index) => {{
                    const hNum = horse.horse_number;
                    const hName = horse.horse_name;
                    const ensScore = parseFloat(horse.Ensemble) || 0;
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

                    // Strategy highlighting class
                    const isJiku = jikuSet.has(String(hNum));
                    const isPartner = partnerSet.has(String(hNum)) && !isJiku;
                    const highlightClass = isJiku ? 'is-jiku' : (isPartner ? 'is-partner' : '');

                    // Decide main score vs sub scores display
                    // mainModelKey decided above
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
                        'TabNet_raw': 'TN'
                    }};

                    let subScoresHtml = '';
                    subModels.forEach(m => {{
                        subScoresHtml += `<span style="background: rgba(255,255,255,0.05); padding: 2px 6px; border-radius: 4px;">${{modelShortNames[m]}}: ${{getZ(horse, m).toFixed(4)}}</span> `;
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

                card.innerHTML = `
                    <div class="race-info-header" style="padding-right: 140px;">
                        <div class="race-id">
                            <span style="color:var(--primary);">${{raceData.title}}</span>
                            <div style="display: inline-flex; gap: 8px; margin-left: 10px; font-size: 0.8rem; align-items: center;">
                                <a href="https://race.netkeiba.com/race/shutuba.html?race_id=${{raceData.race_id}}" target="_blank" style="color:var(--text-muted); text-decoration:none; background:rgba(255,255,255,0.05); padding: 2px 8px; border-radius: 6px;">🌐 Web</a>
                                <a href="https://netkeiba.onelink.me/Wmzg?af_xp=custom&af_dp=jp.co.netdreamers.netkeiba%3A%2F%2F&deep_link_value=https%3A%2F%2Frace.netkeiba.com%2Frace%2Fshutuba.html%3Frace_id%3D${{raceData.race_id}}&rf=race_toggle_menu" style="color:var(--primary); text-decoration:none; background:rgba(74, 222, 128, 0.1); padding: 2px 8px; border-radius: 6px; border: 1px solid var(--primary);">🏇 App</a>
                            </div>
                        </div>
                        ${{ ( () => {{
                            if (!raceData.strategies || raceData.strategies.length === 0) return '';
                            return `
                                <div class="pickup-badge" onclick="event.stopPropagation(); showRecommendation('${{raceId}}')">
                                    <span style="font-size: 0.6rem; opacity: 0.8; font-weight: 400; color: #fff;">INFO</span>
                                    <div style="font-weight: 900; letter-spacing: 0.05em; color: #fff;">PICKUP</div>
                                </div>
                            `;
                        }})()}}
                    </div>
                    <div>
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
        }}

        function generateBettingEyes(horses, strategy, stats, raceId) {{
            const scoreKey = strategy.model === 'Ensemble' ? 'Ensemble' : strategy.model + "_raw";
            
            const getZ = (h) => {{
                const s = stats[scoreKey];
                return ((parseFloat(h[scoreKey]) || 0) - s.mean) / s.std;
            }};

            const getPWin = (h) => {{
                const s = stats[scoreKey];
                const zAdj = horses.map(horse => (parseFloat(horse[scoreKey]) - s.mean) / s.std * 2.0);
                const maxZ = Math.max(...zAdj);
                const expZ = zAdj.map(z => Math.exp(z - maxZ));
                const sumExpZ = expZ.reduce((a, b) => a + b, 0);
                const hZAdj = (parseFloat(h[scoreKey]) - s.mean) / s.std * 2.0;
                return Math.exp(hZAdj - maxZ) / sumExpZ;
            }};

            const getEV = (h) => {{
                const pw = getPWin(h);
                const rIdShort = String(raceId).length === 12 ? String(raceId).substring(2) : raceId;
                const rOdds = window.tanshoData ? (window.tanshoData[raceId] || window.tanshoData[rIdShort]) : null;
                if (rOdds) {{
                    const hO = rOdds.find(o => o[0] == h.horse_number);
                    if (hO) return pw * Math.log1p(parseFloat(hO[1]) || 0);
                }}
                return 0;
            }};
            
            const allSorted = [...horses].sort((a,b) => getZ(b) - getZ(a));
            const pad = (n) => String(n).padStart(2, '0');
            if (allSorted.length === 0) return "--";

            const sTh = strategy.score_th === null ? -9.9 : parseFloat(strategy.score_th);
            const a2Th = (strategy.axis2_score_th === null && strategy.partner_score_th === null) ? -9.9 : parseFloat(strategy.axis2_score_th || strategy.partner_score_th);
            const pTh = strategy.partner_score_th === null ? -9.9 : parseFloat(strategy.partner_score_th);
            
            const evTh = strategy.EV_th === null ? -1 : parseFloat(strategy.EV_th);
            const a2EvTh = (strategy.axis2_EV_th === null && strategy.partner_EV_th === null) ? -1 : parseFloat(strategy.axis2_EV_th || strategy.partner_EV_th);
            const pEvTh = strategy.partner_EV_th === null ? -1 : parseFloat(strategy.partner_EV_th);

            // 単勝
            if (strategy.type === "単勝") {{
                const h = allSorted[0];
                if (getZ(h) < sTh || getEV(h) < evTh) return "--";
                return pad(h.horse_number);
            }}
            
            // BOX
            if (strategy.type.includes("BOX")) {{
                const count = parseInt(strategy.partners) || 5;
                const valid = allSorted.filter(h => getZ(h) >= sTh && getEV(h) >= evTh);
                if (valid.length < count) return "--";
                return valid.slice(0, count).map(h => pad(h.horse_number)).join(', ');
            }}
            
            // 流し / マルチ
            const axisCount = parseInt(strategy.axis_count) || 1;
            const partnerCount = parseInt(strategy.partners) || 5;
            
            const axes1 = allSorted.filter(h => getZ(h) >= sTh && getEV(h) >= evTh).slice(0, 1);
            if (axes1.length === 0) return "--";
            
            let finalAxes = [...axes1];
            let others = allSorted.filter(h => h.horse_number !== axes1[0].horse_number);
            
            if (axisCount >= 2) {{
                const axes2 = others.filter(h => getZ(h) >= a2Th && getEV(h) >= a2EvTh).slice(0, axisCount - 1);
                if (axes2.length < axisCount - 1) return "--";
                finalAxes = finalAxes.concat(axes2);
                const axes2Nums = new Set(axes2.map(h => h.horse_number));
                others = others.filter(h => !axes2Nums.has(h.horse_number));
            }}
            
            const partners = others.filter(h => getZ(h) >= pTh && getEV(h) >= pEvTh).slice(0, partnerCount);
            const minTotal = strategy.type.includes("3連") ? 3 : 2;
            if (finalAxes.length + partners.length < minTotal) return "--";
            
            const separator = (strategy.type.includes("BOX") || strategy.type.includes("複") || strategy.type.includes("連複") || strategy.type.includes("連連") || strategy.type.includes("ワイド")) ? ' - ' : ' → ';
            return finalAxes.map(h => pad(h.horse_number)).join(' → ') + separator + partners.map(h => pad(h.horse_number)).join(', ');
        }}

        function showRecommendation(raceId) {{
            const raceData = currentData[raceId];
            if (!raceData || !raceData.strategies) return;

            const modal = document.getElementById('recommend-modal');
            const body = document.getElementById('modal-body');
            
            // Need to pass stats to generateBettingEyes
            const scoreModels = ['LightGBM_raw', 'XGBoost_raw', 'CatBoost_raw', 'LSTM_raw', 'RandomForest_raw', 'DecisionTree_raw', 'Transformer_raw', 'TabNet_raw', 'Ensemble'];
            const raceStats = {{}};
            scoreModels.forEach(m => {{
                const vals = raceData.horses.map(h => parseFloat(h[m]) || 0);
                const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
                const variance = vals.map(v => Math.pow(v - mean, 2)).reduce((a, b) => a + b, 0) / Math.max(1, raceData.horses.length - 1);
                const std = Math.sqrt(variance) || 1.0;
                raceStats[m] = {{ mean, std }};
            }});

            let html = `
                <div style="text-align: center; margin-bottom: 25px; position: relative;">
                    <div style="font-size: 0.8rem; color: #4ade80; font-weight: 800; text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 8px;">AI Prediction</div>
                    <h2 style="margin: 0; font-size: 1.8rem; color: #fff;">${{raceData.title}}</h2>
                    <button onclick="event.stopPropagation(); fetchRaceResults('${{raceId}}', true)" 
                            style="position: absolute; top: 0; right: 0; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #fff; border-radius: 8px; width: 32px; height: 32px; cursor: pointer; display: flex; align-items: center; justify-content: center; transition: all 0.2s; z-index: 30;"
                            title="Refresh Results">
                        🔄
                    </button>
                </div>
            `;

            let hasValidRec = false;
            raceData.strategies.forEach(s => {{
                const bettingEyes = generateBettingEyes(raceData.horses, s, raceStats, raceId);
                if (bettingEyes === '--') return;

                hasValidRec = true;
                const displayType = s.type;
                const axis2Disp = s.axis_count >= 2 ? ` / Jiku2 > ${{s.axis2_score_th === null ? 'なし' : s.axis2_score_th || s.partner_score_th}}${{s.axis2_EV_th !== null && parseFloat(s.axis2_EV_th) > 0 ? (' (EV > ' + s.axis2_EV_th + ')') : ''}}` : '';
                const evDisp = s.EV_th !== null && parseFloat(s.EV_th) > 0 ? (' (EV > ' + s.EV_th + ')') : '';
                const pEvDisp = s.partner_EV_th !== null && parseFloat(s.partner_EV_th) > 0 ? (' (EV > ' + s.partner_EV_th + ')') : '';

                html += `
                    <div class="strategy-item-modal" data-strategy-type="${{s.type}}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <div style="font-weight: 900; color: #fbbf24; font-size: 1.1rem;">${{displayType}} <span style="font-size: 0.7rem; color: var(--text-muted); margin-left:8px; font-weight:400;">by ${{s.model}}</span></div>
                        </div>
                        <div class="bet-eyes-box">
                            <div style="font-size: 0.7rem; color: var(--text-muted); margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.1em;">Recommended Combination</div>
                            <div class="bet-eyes-text">${{bettingEyes}}</div>
                        </div>
                        <div style="font-size: 0.75rem; color: var(--text-muted); text-align: right; margin-top: 8px;">
                            Z-Score: Jiku1 > ${{s.score_th}}${{evDisp}}${{axis2Disp}} / Partner > ${{s.partner_score_th}}${{pEvDisp}} 　ROI ${{s.roi}}% | Hit ${{s.hit_rate}}%
                        </div>
                        <div class="bet-result-details"></div>
                        <div style="margin-top: 8px; text-align: right;">
                            <button class="smappy-btn" data-eyes="${{bettingEyes}}" data-type="${{s.type}}" data-round="${{raceData.round}}" data-axis="${{s.axis_count || 1}}" data-place="${{raceData.place}}" onclick="event.stopPropagation(); showSmappy(this)">📌 スマッピー</button>
                        </div>
                    </div>
                `;
            }});

            if (!hasValidRec) {{
                html += `
                    <div style="padding: 40px 20px; text-align: center; background: rgba(255,255,255,0.02); border-radius: 12px; border: 1px dashed rgba(255,255,255,0.1); color: var(--text-muted); margin-bottom: 20px;">
                        <div style="font-size: 1.5rem; margin-bottom: 10px;">📋</div>
                        <div style="font-size: 0.9rem; font-weight: 800; color: #fff; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.1em;">No Recommendations Available</div>
                        <div style="font-weight: 700; font-size: 0.8rem;">オススメの買い目はありません</div>
                    </div>
                `;
            }}

            body.innerHTML = html;
            modal.style.display = 'flex';
            document.body.style.overflow = 'hidden';

            fetchRaceResults(raceId);
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
                const decoder = new TextDecoder('euc-jp');
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
                const axisCount = (eyesText.match(/→/g) || []).length; 
                const partners = eyesText.split('→').pop().split(',').length;
                
                if (normType.includes("単勝") || normType.includes("複勝")) {{
                    eyesCount = 1;
                }} else if (normType.includes("BOX")) {{
                    const n = eyesText.split(',').length;
                    if (normType.includes("3連単")) eyesCount = n * (n-1) * (n-2);
                    else if (normType.includes("3連複")) eyesCount = n * (n-1) * (n-2) / 6;
                    else if (normType.includes("馬単")) eyesCount = n * (n-1);
                    else if (normType.includes("馬連") || normType.includes("ワイド")) eyesCount = n * (n-1) / 2;
                }} else if (normType.includes("マルチ")) {{
                    if (normType.includes("3連単")) {{
                        if (axisCount === 1) eyesCount = 3 * partners * (partners - 1);
                        else eyesCount = 6 * partners;
                    }} else if (normType.includes("馬単")) {{
                        eyesCount = 2 * partners;
                    }}
                }} else {{
                    // 流し
                    if (normType.includes("3連単")) {{
                        if (axisCount === 1) eyesCount = partners * (partners - 1);
                        else eyesCount = partners;
                    }} else if (normType.includes("3連複")) {{
                        if (axisCount === 1) eyesCount = (partners * (partners - 1)) / 2;
                        else eyesCount = partners;
                    }} else {{
                        eyesCount = partners;
                    }}
                }}

                if (winNums.length > 0) {{
                    const predictedSet = eyesText.split(/[→,]/).map(s => s.trim().replace(/^0+/, ''));
                    const isMulti = normType.includes("マルチ") || normType.includes("BOX") || normType.includes("3連複") || normType.includes("馬連") || normType.includes("ワイド");

                    if (baseType === "単勝") {{
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
                            const parts = eyesText.split(' → ');
                            const partners = parts.pop().split(',').map(s => s.trim().replace(/^0+/, ''));
                            const axes = parts.map(s => s.trim().replace(/^0+/, ''));
                            
                            const hasAllAxes = axes.every(a => winNums.includes(a));
                            const remainingWinNums = winNums.filter(n => !axes.includes(n));
                            const allRemainingInPartners = remainingWinNums.every(n => partners.includes(n));
                            
                            isHit = hasAllAxes && allRemainingInPartners && (remainingWinNums.length + axes.length === winNums.length);
                        }} else {{
                            isHit = winNums.every(n => predictedSet.includes(n));
                        }}
                    }} else {{
                        // Nagashi (Flow) logic
                        const parts = eyesText.split(' → ');
                        const partners = parts.pop().split(',').map(s => s.trim().replace(/^0+/, ''));
                        const axes = parts.map(s => s.trim().replace(/^0+/, ''));
                        
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
            if (type.includes('3連単') && axisCount >= 2) return '6';
            if (type.includes('3連単')) return '3';
            return '3';
        }}

        function parseSmappyEyes(text, stratType) {{
            text = text.trim();
            if (stratType.includes('BOX')) {{
                var all = text.split(',').map(function(s){{ return parseInt(s.trim()); }}).filter(function(n){{ return !isNaN(n); }});
                return {{axes: all, partners: []}};
            }}
            var axesStr, partnersStr;
            if (text.indexOf(' - ') >= 0) {{
                var dp = text.split(' - ');
                partnersStr = dp.pop();
                axesStr = dp.join(' - ');
            }} else if (text.indexOf(' \u2192 ') >= 0) {{
                var ap = text.split(' \u2192 ');
                partnersStr = ap.pop();
                axesStr = ap.join(' \u2192 ');
            }} else {{
                return {{axes: [parseInt(text)], partners: []}};
            }}
            var partners = partnersStr.split(',').map(function(s){{ return parseInt(s.trim()); }}).filter(function(n){{ return !isNaN(n); }});
            var axes;
            if (axesStr.indexOf(' \u2192 ') >= 0) {{
                axes = axesStr.split(' \u2192 ').map(function(s){{ return parseInt(s.trim()); }}).filter(function(n){{ return !isNaN(n); }});
            }} else {{
                axes = [parseInt(axesStr.trim())];
            }}
            return {{axes: axes, partners: partners}};
        }}

        function genSmappyBml(venue, raceRound, siki, hou, axes, partners) {{
            var raceVal = String(parseInt(raceRound) - 1);
            var simple = (siki === '1' || siki === '2' || siki === '9');
            var steps = "'" + venue + "','" + raceVal + "','" + siki + "'";
            if (!simple) steps += ",'" + hou + "'";
            axes.forEach(function(h) {{ steps += ",'" + h + "'"; }});
            partners.forEach(function(h) {{ steps += ",'" + h + "'"; }});
            return "javascript:void((function(){{" +
                "var steps=[" + steps + "];" +
                "var idx=0;var retries=0;" +
                "function tap(el){{" +
                    "var r=el.getBoundingClientRect();" +
                    "var x=r.left+r.width/2;" +
                    "var y=r.top+r.height/2;" +
                    "var opt={{bubbles:true,cancelable:true,clientX:x,clientY:y,view:window}};" +
                    "el.dispatchEvent(new MouseEvent('mousedown',opt));" +
                    "el.dispatchEvent(new MouseEvent('mouseup',opt));" +
                    "el.dispatchEvent(new MouseEvent('click',opt));" +
                "}}" +
                "function clickNext(){{" +
                    "var btns=document.querySelectorAll('a,button');" +
                    "for(var i=0;i<btns.length;i++){{" +
                        "var t=btns[i].textContent;" +
                        "var b=btns[i].getBoundingClientRect();" +
                        "if(b.width>0&&b.height>0&&(t.indexOf('金額')>=0||t.indexOf('セット')>=0||t.indexOf('次へ')>=0)){{" +
                            "tap(btns[i]);return;" +
                        "}}" +
                    "}}" +
                "}}" +
                "function next(){{" +
                    "if(idx>=steps.length){{" +
                        "setTimeout(clickNext,500);" +
                        "setTimeout(function(){{alert('Complete!')}},1500);" +
                        "return;" +
                    "}}" +
                    "var val=steps[idx];" +
                    "var els=document.querySelectorAll('a[data-value=\\\"'+val+'\\\"]');" +
                    "var found=false;" +
                    "for(var j=0;j<els.length;j++){{" +
                        "var b=els[j].getBoundingClientRect();" +
                        "if(b.width>0&&b.height>0){{" +
                            "tap(els[j]);idx++;retries=0;" +
                            "setTimeout(next,800);found=true;break;" +
                        "}}" +
                    "}}" +
                    "if(!found){{" +
                        "var btns=document.querySelectorAll('a,button');" +
                        "var clickedAdvance=false;" +
                        "for(var k=0;k<btns.length;k++){{" +
                            "var t=btns[k].textContent; var b=btns[k].getBoundingClientRect();" +
                            "if(b.width>0&&b.height>0&&(t.indexOf('通常投票')>=0)){{tap(btns[k]); clickedAdvance=true; break;}}" +
                        "}}" +
                        "if(clickedAdvance) retries=0;" +
                        "retries++;if(retries>500){{alert('Not found: '+steps[idx]);return;}}setTimeout(next, clickedAdvance ? 800 : 100);" +
                    "}}" +
                "}}" +
                "next();" +
            "}})())";
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
            
            var vCodes = {{ "札幌":"01","函館":"02","福島":"03","新潟":"04","東京":"05","中山":"06","中京":"07","京都":"08","阪神":"09","小倉":"10" }};
            var todayPlaces = [];
            for (var k in currentData) {{
                var p = currentData[k].place;
                if (!todayPlaces.includes(p)) todayPlaces.push(p);
            }}
            todayPlaces.sort(function(a, b) {{ return (vCodes[a] || "99") - (vCodes[b] || "99"); }});
            var vIdx = todayPlaces.indexOf(currentPlace);
            if (vIdx < 0) vIdx = 0;

            var popup = document.createElement('div');
            popup.className = 'smappy-popup';
            popup.innerHTML = `
                <div class="smappy-tabs">
                    <div id="tab-pc" class="smappy-tab" onclick="switchSmappyTab('pc')">💻 PC / Android</div>
                    <div id="tab-ios" class="smappy-tab active" onclick="switchSmappyTab('ios')">🍎 iPhone (Shortcuts)</div>
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
                    <div class="step-box">
                        <span class="step-title">使い方</span>
                        <div class="step-desc">下のボタンをブックマークバーにドラッグ登録して、JRAの会場画面で開くだけ！</div>
                    </div>
                    <div style="display: flex; gap: 6px;">
                        <a id="smappy-bml-link" href="#" style="flex: 1; text-align: center; padding: 10px; background: linear-gradient(135deg, #6366f1, #8b5cf6); color: #fff; font-weight: 800; font-size: 0.8rem; border-radius: 8px; text-decoration: none;">📌 ドラッグ登録</a>
                        <button onclick="copySmappyBml()" style="padding: 10px 12px; background: #334155; color: #fff; border: none; border-radius: 8px; font-weight: 700; font-size: 0.8rem; cursor: pointer;">📋 コピー</button>
                    </div>
                </div>

                <div id="panel-ios" style="display: block;">
                    <div class="step-box">
                        <span class="step-title">1. 初回設定 (1分)</span>
                        <div class="step-desc">
                            1. iOSショートカットアプリで新規作成<br>
                            2. <b>「クリップボードを取得」</b>アクションを追加<br>
                            3. <b>「WebページでJavaScriptを実行」</b>を追加<br>
                            4. JavaScriptの中身を <b>eval(クリップボード)</b> にして完了（※クリップボードの部分は変数で選択）
                        </div>
                    </div>
                    <div class="step-box">
                        <span class="step-title">2. 使い方</span>
                        <div class="step-desc">下のボタンで「コード」をコピー。JRA画面で共有ボタン(⬆️)からそのショートカットを押すだけ！</div>
                    </div>
                    <button onclick="copySmappyShortcutJS()" style="width: 100%; padding: 12px; background: #10b981; color: #fff; border: none; border-radius: 8px; font-weight: 800; font-size: 0.85rem; cursor: pointer; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);">🍎 実行用のコードをコピー</button>
                </div>
                
                <div style="font-size: 0.55rem; color: var(--text-muted); margin-top: 10px; text-align: center; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 8px;">
                    JRAスマッピー → 通常投票 → 会場選択画面で実行してください
                </div>
            `;
            btn.parentElement.appendChild(popup);

            window._smappyParsed = {{round: round, siki: siki, hou: hou, axes: parsed.axes, partners: parsed.partners}};
            
            function updateBml() {{
                var v = document.getElementById('smappy-venue').value;
                var bml = genSmappyBml(v, round, siki, hou, parsed.axes, parsed.partners);
                document.getElementById('smappy-bml-link').href = bml;
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

                
        function copySmappyShortcutJS() {{
            var venueEl = document.getElementById('smappy-venue');
            if (!venueEl || !window._smappyParsed) return;
            var v = venueEl.value;
            var p = window._smappyParsed;
            var vName = venueEl.options[venueEl.selectedIndex].text.split(' ').pop();
            var rawJS = genSmappyShortcutJS(v, vName, p.round, p.siki, p.hou, p.axes, p.partners);
            var t = document.createElement('textarea');
            t.value = rawJS;
            document.body.appendChild(t);
            t.select();
            document.execCommand('copy');
            document.body.removeChild(t);
            if (confirm('コードをコピーしました！このままショートカット「スマッピー」を起動しますか？')) {{
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
            alert('コピーしました！JRA→通常投票→会場選択画面で実行');
        }}

        
        
        
        function genSmappyShortcutJS(venueCode, venueName, raceRound, siki, hou, axes, partners) {{
            var rawSteps = [venueCode, raceRound, siki];
            var simple = (siki === '1' || siki === '2' || siki === '9');
            if (!simple && hou) rawSteps.push(hou);
            (axes || []).forEach(function(a) {{ rawSteps.push(a); }});
            (partners || []).forEach(function(p) {{ rawSteps.push(p); }});

            var stepsJSON = JSON.stringify(rawSteps);
            var venueJSON = JSON.stringify(venueName);
            
            var part1 = "(function(){{ var s=" + stepsJSON + "; var vn=" + venueJSON + "; ";
            var part2 = "var sn={{'1':'単勝','2':'複勝','3':'枠連','4':'馬連','5':'ワイド','6':'馬単','7':'3連複','8':'3連単'}}; var i=0,r=0,d=false,T=Date.now(); function dg(m){{ var x=document.getElementById('smappy-diag'); if(!x){{ x=document.createElement('div'); x.id='smappy-diag'; x.style='position:fixed;top:0;left:0;width:100%;z-index:100000;background:rgba(0,0,0,0.9);color:#0f0;font-size:10px;padding:4px;pointer-events:none;'; document.body.appendChild(x); }} x.innerText=m; }} function fi(ok){{ if(d)return; d=true; dg('FINISH:'+ok); completion(ok); }} function tp(e){{ var r=e.getBoundingClientRect(); var x=r.left+r.width/2; var y=r.top+r.height/2; var o={{bubbles:true,cancelable:true,clientX:x,clientY:y,view:window}}; try {{ var t=new Touch({{identifier:Date.now(),target:e,clientX:x,clientY:y,radiusX:2,radiusY:2}}); var to={{bubbles:true,cancelable:true,touches:[t],targetTouches:[t],changedTouches:[t],view:window}}; e.dispatchEvent(new TouchEvent('touchstart',to)); e.dispatchEvent(new TouchEvent('touchend',to)); }} catch(err){{}} e.dispatchEvent(new MouseEvent('mousedown',o)); e.dispatchEvent(new MouseEvent('mouseup',o)); e.dispatchEvent(new MouseEvent('click',o)); }} function cf(){{ var k=['金額','セット','次へ']; var a=document.querySelectorAll('a,button'); for(var j=0;j<a.length;j++){{ var b=a[j].getBoundingClientRect(); if(b.width>0&&b.height>0){{ for(var l=0;l<k.length;l++){{ if(a[j].textContent.indexOf(k[l])>=0){{ tp(a[j]); return; }} }} }} }} }} function nx(){{ try {{ if(Date.now()-T>22000){{ fi(false); return; }} var c=(document.body.innerText||'').split('\\n').join(' ').trim(); var p=''; if(c.indexOf('会場')>=0||c.indexOf('開催')>=0||c.indexOf('場所')>=0) p='V'; if(c.indexOf('レース')>=0||c.indexOf('締切')>=0||c.indexOf('状況')>=0||c.indexOf('回次')>=0) p='R'; if(c.indexOf('式別')>=0) p='S'; if(c.indexOf('方式')>=0) p='M'; if(i>=s.length){{ dg('Done'); cf(); fi(true); return; }} var v=s[i]; var f=false; var vs=[v]; var n=parseInt(v); if(!isNaN(n)){{ vs=[v,String(n),(n<10?'0'+n:String(n)),String(n-1),(n-1<10?'0'+(n-1):String(n-1))]; }} dg('S'+i+':'+v+' r:'+r+' p:'+p+' txt:'+c.substring(0,10)); var okP=(i===0&&(p==='V'||p===''))||(i===1&&(p==='R'||p==='V'||p===''))||(i===2&&p==='S')||(i===3&&(p==='M'||p==='S'))||(i>3); if(okP){{ for(var k=0;k<vs.length;k++){{ var es=document.querySelectorAll('a[data-value=\"'+vs[k]+'\"]'); for(var j=0;j<es.length;j++){{ var b=es[j].getBoundingClientRect(); if(b.width>3&&b.height>3){{ tp(es[j]); i++; r=0; setTimeout(nx,1500); f=true; break; }} }} if(f)break; }} if(!f){{ var bs=document.querySelectorAll('a,button'); for(var k2=0;k2<bs.length;k2++){{ var b2=bs[k2].getBoundingClientRect(); if(b2.width<=4||b2.height<=4)continue; var t=(bs[k2].innerText||bs[k2].textContent||'').trim(); if(i===0&&vn&&t.indexOf(vn)>=0){{ tp(bs[k2]); i++; r=0; setTimeout(nx,1500); f=true; break; }} if(i===1&&(t===v+'R'||t===v+'レース')){{ tp(bs[k2]); i++; r=0; setTimeout(nx,1500); f=true; break; }} if(i===2&&sn[v]&&(t.indexOf(sn[v])>=0||t.indexOf(sn[v].replace('3','３'))>=0)){{ tp(bs[k2]); i++; r=0; setTimeout(nx,1500); f=true; break; }} }} }} }} if(!f){{ var bs3=document.querySelectorAll('a,button'); for(var k3=0;k3<bs3.length;k3++){{ var b3=bs3[k3].getBoundingClientRect(); if(b3.width>4&&b3.height>4&&bs3[k3].innerText.indexOf('通常投票')>=0){{ tp(bs3[k3]); r=0; break; }} }} r++; setTimeout(nx,200); }} }} catch(err){{ dg('E:' + err.message); fi(false); }} }} nx(); }})();";
            return part1 + part2;
        }}

        function closeRecommendation() {{
            document.getElementById('recommend-modal').style.display = 'none';
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
    
    # Git 更新処理 (tohshin_keiba のみ)
    try:
        repo_dir = r"C:\Users\kyoui\tohshin_keiba"
        logger.info(f"Starting Git update process for {repo_dir}...")
        
        # 1. git add
        subprocess.run(["git", "add", "index.html", "generate_html.py", "jsons/meta.json", "jsons/tansho_data.json"], cwd=repo_dir, check=True)
        # 日次JSONも追加
        subprocess.run(["git", "add", "jsons/data_*.json"], cwd=repo_dir, check=True)
        
        # 2. git commit (変更がある場合のみ)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True)
        if status.stdout.strip():
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
