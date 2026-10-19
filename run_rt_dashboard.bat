@echo off
setlocal
cd /d "%~dp0"
python -m rt_dashboard.server --open --warmup
endlocal
