@echo off
echo ========================================
echo RUNNING ENHANCED CONFLUENCE STRATEGY
echo ========================================

echo.
echo Finding Python installation...

:: Try to find Python
set PYTHON_CMD=

if exist "C:\Users\keith\AppData\Local\Programs\Python\Python311\python.exe" (
    set PYTHON_CMD=C:\Users\keith\AppData\Local\Programs\Python\Python311\python.exe
    goto :found_python
)

if exist "C:\Users\keith\AppData\Local\Programs\Python\Python313\python.exe" (
    set PYTHON_CMD=C:\Users\keith\AppData\Local\Programs\Python\Python313\python.exe
    goto :found_python
)

echo ERROR: Python not found!
echo Please run the command manually using your Python installation.
echo.
echo Command to run:
echo python main.py --strategy enhanced_confluence --mode paper --pair BTC/USDT --run-once
pause
exit /b 1

:found_python
echo Found Python: %PYTHON_CMD%

echo.
echo ========================================
echo TESTING INTEGRATION
echo ========================================
echo Running integration test...
%PYTHON_CMD% test_integration.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Integration test failed!
    pause
    exit /b 1
)

echo.
echo ========================================
echo RUNNING ENHANCED STRATEGY
echo ========================================
echo.
echo Running enhanced confluence strategy in paper mode...
echo Command: %PYTHON_CMD% main.py --strategy enhanced_confluence --mode paper --pair BTC/USDT --run-once
echo.

%PYTHON_CMD% main.py --strategy enhanced_confluence --mode paper --pair BTC/USDT --run-once

echo.
echo ========================================
echo STRATEGY RUN COMPLETED
echo ========================================
echo.
echo If you see trading signals and performance metrics above,
echo the enhanced strategy is working correctly!
echo.
echo Next steps:
echo 1. Monitor the win rate and profit factor
echo 2. Run for longer periods to gather more data
echo 3. Optimize parameters if needed
echo 4. Consider live trading with small positions
echo.
pause