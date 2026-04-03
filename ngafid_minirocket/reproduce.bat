@echo off
setlocal
cd /d "%~dp0"
pip install -r requirements.txt
python scripts\run_cv.py %*
