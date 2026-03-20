@echo off
call C:\Users\kyoui\anaconda3\Scripts\activate.bat C:\Users\kyoui\anaconda3

call conda activate new

cd /d C:\Users\kyoui\tohshin_keiba

python generate_html.py
