f_path = r"c:\Users\kyoui\tohshin_keiba\deploy_tmp\index.html"
with open(f_path, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()
    print(f"File length: {len(content)}")
    if "fetchRaceResults" in content:
        print("FOUND fetchRaceResults")
    else:
        print("NOT FOUND fetchRaceResults")
