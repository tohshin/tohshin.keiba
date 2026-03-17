import subprocess

commit_hash = "a47f3d4a4952ef28906083fc7ac2ad111318bd40"
file_path = "index.html"

result = subprocess.run(["git", "show", f"{commit_hash}:{file_path}"], capture_output=True, text=True, encoding='utf-8')
content = result.stdout

# fetchRaceResults 関数を抽出
start_marker = "async function fetchRaceResults"
if start_marker in content:
    start_idx = content.find(start_marker)
    # 簡易的に次の関数の手前かスクリプト終了までを取得
    end_idx = content.find("function ", start_idx + len(start_marker))
    if end_idx == -1:
        end_idx = content.find("</script>", start_idx)
    
    print("--- START ---")
    print(content[start_idx:end_idx])
    print("--- END ---")
else:
    print("Not found")
