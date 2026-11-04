@echo off
title AgriGuard AI - Enterprise Deep Learning Suite
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo ========================================================
echo       AGRIGUARD AI: AGRICULTURAL PATHOLOGY SUITE
echo ========================================================
echo.
echo Starting Streamlit Production Server...
echo Web UI will open at: http://localhost:8501
echo.
streamlit run app/app.py
pause

