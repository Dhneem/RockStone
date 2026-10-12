@echo off
rem Rockstone CT Workbench - one-click launcher (Windows).
rem Starts the server on 127.0.0.1:8770 (unless already running) and opens
rem http://127.0.0.1:8770 in the default browser.
rem Usage: start.bat [port]
cd /d "%~dp0"
setlocal

set "PORT=8770"
if not "%~1"=="" set "PORT=%~1"
set "URL=http://127.0.0.1:%PORT%"

rem --- pick a Python interpreter: project venv first, then PATH -----------
set "PY="
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
if defined PY goto havepy
if exist ".venv\bin\python.exe" set "PY=.venv\bin\python.exe"
if defined PY goto havepy
where python >nul 2>nul
if errorlevel 1 goto nopython
set "PY=python"
:havepy

rem --- dependencies installed? --------------------------------------------
"%PY%" -c "import uvicorn" >nul 2>&1
if errorlevel 1 goto nodeps

rem --- server already answering? then just open the browser ---------------
"%PY%" -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('%URL%/api/health',timeout=2).status==200 else 1)" >nul 2>&1
if not errorlevel 1 goto already

rem --- start the server in its own console window -------------------------
echo Starting Rockstone CT Workbench server on %URL% ...
start "Rockstone CT Workbench" cmd /k "%PY% -m uvicorn server.app:app --host 127.0.0.1 --port %PORT%"

rem --- wait for /api/health (max 60 s) ------------------------------------
set /a TRIES=0
:wait
"%PY%" -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('%URL%/api/health',timeout=2).status==200 else 1)" >nul 2>&1
if not errorlevel 1 goto ready
set /a TRIES=%TRIES%+1
if %TRIES% geq 60 goto failed
ping -n 2 127.0.0.1 >nul
goto wait

:already
echo Server already running at %URL%
start "" "%URL%"
exit /b 0

:ready
echo Server ready - opening %URL%
start "" "%URL%"
exit /b 0

:failed
echo.
echo ERROR: the server did not become ready within 60 seconds.
echo Check the "Rockstone CT Workbench" server window for errors.
pause
exit /b 1

:nopython
echo.
echo ERROR: Python not found on PATH and no .venv exists.
echo Install Python 3.11+, then run:
echo     python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
pause
exit /b 1

:nodeps
echo.
echo ERROR: dependencies are not installed for %PY%.
echo Run:
echo     "%PY%" -m pip install -r requirements.txt
pause
exit /b 1
