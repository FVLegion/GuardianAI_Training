@echo off
echo Guardian AI Local Pipeline Listener
echo =====================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python and add it to PATH.
    pause
    exit /b 1
)

echo SUCCESS: Python found
echo.

REM Install required dependencies if not present
echo Checking dependencies...
pip show requests flask gitpython clearml >nul 2>&1
if errorlevel 1 (
    echo Installing required dependencies...
    pip install requests flask gitpython clearml
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo SUCCESS: Dependencies ready
echo.

REM Ask user for mode
echo Choose execution mode:
echo 1. Polling mode (checks GitHub every 60 seconds)
echo 2. Webhook mode (listens for real-time triggers)
echo.
set /p mode="Enter choice (1 or 2): "

if "%mode%"=="1" (
    echo Starting in polling mode...
    python local_pipeline_listener.py
) else if "%mode%"=="2" (
    echo Starting webhook server...
    echo Webhook URL: http://localhost:8080/webhook
    echo Status URL: http://localhost:8080/status
    echo.
    python local_pipeline_listener.py --webhook
) else (
    echo Invalid choice. Defaulting to polling mode...
    python local_pipeline_listener.py
)

echo.
echo Pipeline listener stopped.
pause 