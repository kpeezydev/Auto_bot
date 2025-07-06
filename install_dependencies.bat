@echo off
echo ========================================
echo INSTALLING AUTO_BOT DEPENDENCIES
echo ========================================

echo.
echo Installing required Python packages...

pip install scipy
pip install matplotlib
pip install scikit-learn

echo.
echo Verifying installations...
python -c "import scipy; print('✅ scipy installed:', scipy.__version__)"
python -c "import matplotlib; print('✅ matplotlib installed:', matplotlib.__version__)"
python -c "import pandas; print('✅ pandas installed:', pandas.__version__)"
python -c "import numpy; print('✅ numpy installed:', numpy.__version__)"
python -c "import pandas_ta; print('✅ pandas_ta installed:', pandas_ta.__version__)"

echo.
echo ========================================
echo DEPENDENCIES INSTALLATION COMPLETED
echo ========================================
echo.
echo You can now run the enhanced strategy:
echo python main.py --strategy enhanced_confluence --mode paper --pair BTC/USDT --run-once
echo.
pause