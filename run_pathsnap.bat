@echo off
setlocal
cd /d "%~dp0"

echo.
echo PathSnap launcher
echo 1. Start API and model service (Laptop 1)
echo 2. Start UI (Laptop 2)
echo.
set /p ROLE=Choose 1 or 2: 

if "%ROLE%"=="1" goto API
if "%ROLE%"=="2" goto UI
echo Please enter 1 or 2.
exit /b 1

:API
echo.
echo Preparing Python environment...
if not exist ".venv\Scripts\activate.bat" (
  python -m venv .venv
  if errorlevel 1 goto ERROR
)
call ".venv\Scripts\activate.bat"
python -m pip install -r requirements.txt
if errorlevel 1 goto ERROR

if not exist "artifacts\final_model.joblib" (
  echo No trained model artifact found. Training and tuning now; this can take a while.
  python src\features\build_table.py
  if errorlevel 1 goto ERROR
  python src\models\run_experiments.py
  if errorlevel 1 goto ERROR
  python src\tuning\optuna_hgb.py --trials 10 --folds 5
  if errorlevel 1 goto ERROR
  python src\models\export_tuned_model.py --out artifacts\final_model.joblib
  if errorlevel 1 goto ERROR
)

set /p TRIPS_DIR=Optional evaluator trips folder (press Enter to use included student-test trips): 
if not "%TRIPS_DIR%"=="" set "PATHSNAP_TRIPS_DIR=%TRIPS_DIR%"
echo Starting API at http://0.0.0.0:8000
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
goto END

:UI
if not exist "frontend\package.json" (
  echo Run this batch file from the PathSnap project root.
  goto ERROR
)
set /p BACKEND_IP=Enter Laptop 1 IP address (or localhost for same computer): 
if "%BACKEND_IP%"=="" set "BACKEND_IP=localhost"
set "VITE_API_BASE_URL=http://%BACKEND_IP%:8000"
pushd frontend
if not exist "node_modules" (
  call npm ci
  if errorlevel 1 goto ERROR_POP
)
echo UI will use %VITE_API_BASE_URL%
call npm run dev -- --host 0.0.0.0
popd
goto END

:ERROR_POP
popd
:ERROR
echo.
echo PathSnap could not start. Check the error above, and confirm Python and Node.js are installed.
exit /b 1

:END
endlocal
