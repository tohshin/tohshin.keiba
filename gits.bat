@echo off
cd /d C:\Users\kyoui\tohshin_keiba
echo === 1. Pull Latest Changes for tohshin_keiba ===
git pull https://github.com/tohshin/tohshin.keiba.git main --rebase --autostash
echo === 2. Generate Latest Data ===
call C:\Users\kyoui\anaconda3\Scripts\activate.bat C:\Users\kyoui\anaconda3
call conda activate new
python generate_html.py
echo === 3. Sync to GitHub ===
git add index.html jsons/*.json
git commit -m "Update predictions and strategies"
git push origin main
echo === Done! ===