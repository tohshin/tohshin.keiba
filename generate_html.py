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

    # Get scores
    req_scores = ['LightGBM_raw', 'XGBoost_raw', 'CatBoost_raw', 'LSTM_raw', 'Ensemble']
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
        place_code = race_id_str[4:6] if len(race_id_str) >= 12 else ''
        round_no = race_id_str[10:12] if len(race_id_str) >= 12 else ''
        
        place_name = REVERSE_PLACE_DICT.get(place_code, place_code)
        
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
            
        races[race_id_str] = {
            "race_id": race_id_str,
            "title": race_title,
            "date": date_val,
            "place": place_name,
            "round": str(round_int),
            "horses": records
        }

    # データをJSON文字列化して別ファイルに保存
    json_data = json.dumps(races, ensure_ascii=False)
    
    # 保存先パスの定義
    output_json_paths = [
        r"C:\Users\kyoui\tohshin_keiba\jsons\data.json"
    ]
    
    for out_json in output_json_paths:
        try:
            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                f.write(json_data)
            logger.info(f"Successfully generated JSON data at {out_json}")
        except Exception as e:
            logger.error(f"Failed to write JSON to {out_json}: {e}")

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
            <select id="filter-date" onchange="renderRaces()">
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
                <option value="horse_number">Sort by Horse Number</option>
            </select>
        </div>

        <div id="races-container" class="race-list"></div>
    </div>

    <script>
        let currentData = {{}};

        async function checkAuth() {{
            const pw = document.getElementById('auth-pw').value;
            if (pw === 'tohshi') {{
                document.getElementById('auth-overlay').style.display = 'none';
                document.getElementById('app-content').style.display = 'block';
                
                const container = document.getElementById('races-container');
                container.innerHTML = '<div style="text-align:center; padding: 40px;"><p>Loading data...</p></div>';

                try {{
                    // data.json と tansho_data.json を取得
                    const [dataRes, tanshoRes] = await Promise.all([
                        fetch('jsons/data.json?t=' + new Date().getTime()),
                        fetch('jsons/tansho_data.json?t=' + new Date().getTime())
                    ]);

                    if (!dataRes.ok) throw new Error('Data fetch failed');
                    currentData = await dataRes.json();
                    
                    if (tanshoRes.ok) {{
                        window.tanshoData = await tanshoRes.json();
                    }} else {{
                        console.warn("tansho_data.json not found, using empty data.");
                        window.tanshoData = {{}};
                    }}
                    
                    initFilters();
                    renderRaces();
                }} catch (error) {{
                    console.error("Fetch error: ", error);
                    container.innerHTML = '<div style="text-align:center; padding: 40px; color: #ef4444;"><p>Failed to load data.json or tansho_data.json. Note: This requires a web server.</p></div>';
                }}
            }} else {{
                document.getElementById('login-error').style.display = 'block';
            }}
        }}

        function initFilters() {{
            const dateSet = new Set();
            const placeSet = new Set();
            const roundSet = new Set();
            
            for (const race of Object.values(currentData)) {{
                if(race.date) dateSet.add(race.date);
                if(race.place && race.place.search(/\\S/) !== -1 && !/^\\d+$/.test(race.place)) placeSet.add(race.place);
                if(race.round) roundSet.add(race.round);
            }}
            
            const dp = document.getElementById('filter-date');
            const datesArr = Array.from(dateSet).sort();
            datesArr.forEach(d => {{
                const opt = document.createElement('option');
                opt.value = d; opt.innerText = d;
                dp.appendChild(opt);
            }});
            if (datesArr.length > 0) {{
                dp.value = datesArr[datesArr.length - 1]; // Default to latest date
            }}
            
            const pp = document.getElementById('filter-place');
            Array.from(placeSet).sort().forEach(p => {{
                const opt = document.createElement('option');
                opt.value = p; opt.innerText = p;
                pp.appendChild(opt);
            }});
            const rp = document.getElementById('filter-round');
            Array.from(roundSet).sort((a,b)=>parseInt(a)-parseInt(b)).forEach(r => {{
                const opt = document.createElement('option');
                opt.value = r; opt.innerText = parseInt(r) + "R";
                rp.appendChild(opt);
            }});
        }}

        function renderRaces() {{
            const container = document.getElementById('races-container');
            container.innerHTML = '';
            
            const sortBy = document.getElementById('sort-select').value;
            const fDate = document.getElementById('filter-date').value;
            const fPlace = document.getElementById('filter-place').value;
            const fRound = document.getElementById('filter-round').value;
 
            let delay = 0;

            for (const [raceId, raceData] of Object.entries(currentData)) {{
                
                // Filtering
                if (fDate !== 'ALL' && raceData.date !== fDate) continue;
                if (fPlace !== 'ALL' && raceData.place !== fPlace) continue;
                if (fRound !== 'ALL' && String(raceData.round) !== String(fRound)) continue;

                // Sort horses based on Ensemble score or number
                let sortedHorses = [...raceData.horses];
                if (sortBy === 'score') {{
                    sortedHorses.sort((a, b) => (parseFloat(b.Ensemble) || 0)  - (parseFloat(a.Ensemble) || 0));
                }} else {{
                    sortedHorses.sort((a, b) => (parseInt(a.horse_number) || 0)  - (parseInt(b.horse_number) || 0));
                }}

                // --- Calculate Softmax Probabilities ---
                const allScores = sortedHorses.map(h => parseFloat(h.Ensemble) || 0);
                const maxEns = allScores.length > 0 ? Math.max(...allScores) : 0;
                const expScores = allScores.map(s => Math.exp(s - maxEns));
                const sumExp = expScores.reduce((a, b) => a + b, 0);
                
                sortedHorses.forEach((h, idx) => {{
                    h.pWin = expScores[idx] / sumExp;
                }});

                // calculate max score for bar formatting
                const allEnsembleScores = sortedHorses.map(h => parseFloat(h.Ensemble) || 0);
                const maxScore = Math.max(...allEnsembleScores, 0.1);
                const minScore = Math.min(...allEnsembleScores, 0);

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
                    if(maxScore > 0) {{
                        widthPct = Math.max(5, ((ensScore - Math.min(0, minScore)) / (maxScore - Math.min(0, minScore))) * 100);
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
                            kv = pWin * parseFloat(winOdds);
                        }}
                    }}
                    
                    let rankClass = '';
                    if(sortBy === 'score') {{
                        if(index === 0) rankClass = 'rank-1';
                        else if(index === 1) rankClass = 'rank-2';
                        else if(index === 2) rankClass = 'rank-3';
                    }}

                    horsesHtml += `
                        <div class="horse-row ${{rankClass}}" style="flex-wrap: wrap;">
                            <div style="display: flex; width: 100%; align-items: center; margin-bottom: 6px;">
                                <div class="horse-num">${{hNum}}</div>
                                <div class="horse-details">
                                    <div style="display: flex; justify-content: space-between; align-items: baseline;">
                                        <div class="horse-name">${{hName}}</div>
                                        <div style="font-size: 0.85rem; font-weight: 600; color: #fbbf24;">単勝: ${{winOdds}} | KV: <span style="color: ${{kv >= 1.5 ? '#4ade80' : (kv >= 1.0 ? '#f8fafc' : '#94a3b8')}}">${{kv > 0 ? kv.toFixed(2) : '-'}}</span></div>
                                    </div>
                                    <div class="horse-score-bar-bg">
                                        <div class="horse-score-bar-fill" style="width: 0%" data-target="${{widthPct}}%"></div>
                                    </div>
                                </div>
                                <div class="horse-score-val">${{ensScore.toFixed(3)}}</div>
                            </div>
                            <div style="display: flex; width: 100%; justify-content: flex-end; gap: 6px; font-size: 0.72rem; color: var(--text-muted); flex-wrap: wrap; margin-left: 50px;">
                                <span style="background: rgba(255,255,255,0.05); padding: 2px 6px; border-radius: 4px;">LGBM: ${{parseFloat(horse.LightGBM_raw || 0).toFixed(3)}}</span>
                                <span style="background: rgba(255,255,255,0.05); padding: 2px 6px; border-radius: 4px;">XGB: ${{parseFloat(horse.XGBoost_raw || 0).toFixed(3)}}</span>
                                <span style="background: rgba(255,255,255,0.05); padding: 2px 6px; border-radius: 4px;">CB: ${{parseFloat(horse.CatBoost_raw || 0).toFixed(3)}}</span>
                                <span style="background: rgba(255,255,255,0.05); padding: 2px 6px; border-radius: 4px;">LSTM: ${{parseFloat(horse.LSTM_raw || 0).toFixed(3)}}</span>
                            </div>
                        </div>
                    `;
                }});

                card.innerHTML = `
                    <div class="race-info-header">
                        <div class="race-id">
                            <span style="color:var(--primary);">${{raceData.title}}</span>
                            <div style="display: inline-flex; gap: 8px; margin-left: 10px; font-size: 0.8rem;">
                                <a href="https://race.sp.netkeiba.com/race/shutuba.html?race_id=${{raceData.race_id}}" target="_blank" style="color:var(--text-muted); text-decoration:none; background:rgba(255,255,255,0.05); padding: 2px 8px; border-radius: 6px;">🌐 Web</a>
                                <a href="https://netkeiba.onelink.me/Wmzg?af_xp=custom&af_dp=jp.co.netdreamers.netkeiba%3A%2F%2F&deep_link_value=https%3A%2F%2Frace.sp.netkeiba.com%2Frace%2Fshutuba.html%3Frace_id%3D${{raceData.race_id}}&rf=race_toggle_menu" style="color:var(--primary); text-decoration:none; background:rgba(74, 222, 128, 0.1); padding: 2px 8px; border-radius: 6px; border: 1px solid var(--primary);">🏇 App</a>
                            </div>
                        </div>
                        <div class="race-meta">${{sortedHorses.length}} Horses</div>
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

    </script>
</body>
</html>
"""

    # HTML の保存先
    output_html_paths = [
        r"C:\Users\kyoui\tohshin_keiba\index.html"
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
        # インデックス作成に時間がかかる場合があるため、明示的に指定
        subprocess.run(["git", "add", "index.html", "jsons/data.json", "generate_html.py"], cwd=repo_dir, check=True)
        
        # 2. git commit (変更がある場合のみ)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True)
        if status.stdout.strip():
            subprocess.run(["git", "commit", "-m", "Auto-update race data and HTML"], cwd=repo_dir, check=True)
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
            
    except Exception as e:
        logger.error(f"Error during Git update: {e}")

if __name__ == "__main__":
    generate_static_html()
