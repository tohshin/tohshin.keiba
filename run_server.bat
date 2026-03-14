@echo off
setlocal

echo ==========================================
echo  東信競馬 AI 予想 - ローカルサーバー
echo ==========================================
echo.
echo サーバーを http://localhost:8000 で起動しています...
echo.
echo [重要] 
echo 1. このウィンドウは閉じずに、そのままにしておいてください。
echo 2. ブラウザが自動的に開かない場合は、以下のアドレスをブラウザに入力してください。
echo    http://localhost:8000
echo.

:: ブラウザを先に起動（サーバー起動の直前に呼び出す）
start http://localhost:8000

:: Pythonの実行コマンドを確認して起動
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [実行中] 'python' を使用しています...
    python -m http.server 8000
    goto end
)

py --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [実行中] 'py' を使用しています...
    py -m http.server 8000
    goto end
)

python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [実行中] 'python3' を使用しています...
    python3 -m http.server 8000
    goto end
)

echo.
echo [エラー] Pythonが見つかりませんでした。
echo Pythonがインストールされているか、PATHが通っているか確認してください。
echo.
pause

:end
endlocal
