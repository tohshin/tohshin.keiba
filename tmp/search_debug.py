import os

files = [
    r"c:\Users\kyoui\tohshin_keiba\index.html",
    r"c:\Users\kyoui\tohshin_keiba\deploy_tmp\index.html",
    r"c:\Users\kyoui\tohshin_keiba\generate_html.py"
]

for f_path in files:
    if os.path.exists(f_path):
        print(f"--- File: {f_path} ---")
        with open(f_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f, 1):
                if "DEBUG" in line or "fetch" in line or "cors" in line:
                    print(f"{i}: {line.strip()}")
    else:
        print(f"--- Missing: {f_path} ---")
