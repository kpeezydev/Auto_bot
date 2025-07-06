@echo off
echo ========================================
echo AUTO_BOT ENHANCED STRATEGY TESTING
echo ========================================

echo.
echo Finding Python installation...

:: Try different Python commands
set PYTHON_CMD=

:: Check for python in PATH
python --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=python
    goto :found_python
)

:: Check for py launcher
py --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=py
    goto :found_python
)

:: Check common installation paths
if exist "C:\Python39\python.exe" (
    set PYTHON_CMD=C:\Python39\python.exe
    goto :found_python
)

if exist "C:\Python310\python.exe" (
    set PYTHON_CMD=C:\Python310\python.exe
    goto :found_python
)

if exist "C:\Python311\python.exe" (
    set PYTHON_CMD=C:\Python311\python.exe
    goto :found_python
)

if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python39\python.exe" (
    set PYTHON_CMD=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python39\python.exe
    goto :found_python
)

if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe" (
    set PYTHON_CMD=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\python.exe
    goto :found_python
)

if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe" (
    set PYTHON_CMD=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python311\python.exe
    goto :found_python
)

echo ERROR: Python not found!
echo Please install Python or add it to your PATH.
echo You can download Python from: https://www.python.org/downloads/
pause
exit /b 1

:found_python
echo Found Python: %PYTHON_CMD%
%PYTHON_CMD% --version

echo.
echo ========================================
echo PHASE 1: QUICK FUNCTIONALITY TEST
echo ========================================
echo Running quick test with sample data...
%PYTHON_CMD% quick_test.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Quick test failed!
    echo Check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo PHASE 2: STRATEGY COMPARISON
echo ========================================
echo Comparing different strategies...

echo.
echo Testing Enhanced Confluence Strategy (Conservative)...
%PYTHON_CMD% run_enhanced_confluence_backtester.py --symbol BTC/USDT --confluence-threshold 0.7 --min-adx 30 --risk-reward-ratio 2.5 --max-trades-per-day 2 --limit 1000 --no-plot

echo.
echo Testing Enhanced Confluence Strategy (Balanced)...
%PYTHON_CMD% run_enhanced_confluence_backtester.py --symbol BTC/USDT --confluence-threshold 0.6 --min-adx 25 --risk-reward-ratio 2.0 --max-trades-per-day 3 --limit 1000 --no-plot

echo.
echo Testing Enhanced Confluence Strategy (Aggressive)...
%PYTHON_CMD% run_enhanced_confluence_backtester.py --symbol BTC/USDT --confluence-threshold 0.5 --min-adx 20 --risk-reward-ratio 1.5 --max-trades-per-day 4 --limit 1000 --no-plot

echo.
echo ========================================
echo PHASE 3: MULTI-SYMBOL TESTING
echo ========================================

echo.
echo Testing on ETH/USDT...
%PYTHON_CMD% run_enhanced_confluence_backtester.py --symbol ETH/USDT --confluence-threshold 0.6 --limit 1000 --no-plot

echo.
echo Testing on SOL/USDT...
%PYTHON_CMD% run_enhanced_confluence_backtester.py --symbol SOL/USDT --confluence-threshold 0.6 --limit 1000 --no-plot

echo.
echo ========================================
echo TESTING COMPLETED!
echo ========================================
echo.
echo All tests have been completed.
echo Check the output above for performance results.
echo.
echo Key files created:
echo - Enhanced indicators: src/enhanced_indicators.py
echo - Market structure: src/market_structure.py  
echo - Enhanced strategy: src/strategy_enhanced_confluence.py
echo - Optimization script: optimize_strategy.py
echo.
echo Next steps:
echo 1. Review the test results above
echo 2. Run parameter optimization: python optimize_strategy.py
echo 3. Paper trade the best performing configuration
echo 4. Integrate with your main trading bot
echo.
pause