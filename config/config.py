"""
Configuration settings for the trading bot.
"""

import logging

# Trading Parameters
TRADING_PAIR = 'AVAX/USDT'  # Default trading pair
TIMEFRAME = '1h'           # Default timeframe for analysis
EXCHANGE = 'binance'       # Default exchange
RISK_PER_TRADE = 0.01      # Default risk percentage per trade (e.g., 0.01 = 1%)

# Logging Configuration
LOG_LEVEL = logging.INFO   # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_FILE = 'logs/trading_bot.log' # Path to the log file (relative to project root)

# Add any other shared configuration variables here
# e.g., API endpoints, specific strategy parameters if not passed via command line