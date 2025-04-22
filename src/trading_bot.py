import logging
import ccxt
import time
import schedule
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from src.data_fetcher import DataFetcher
from src.indicators import TechnicalIndicators
from src.strategy import Strategy, SignalType
from src.execution import ExecutionEngine
from src.utils import load_env_vars, setup_logging
from config.config import TRADING_PAIR, TIMEFRAME, EXCHANGE, LOG_LEVEL, LOG_FILE

# Configure logging
logger = logging.getLogger(__name__)

class TradingBot:
    """
    Main trading bot class that orchestrates the entire trading process.
    """
    
    def __init__(self, strategy: Strategy, exchange_id: str = EXCHANGE, 
                 trading_pair: str = TRADING_PAIR, timeframe: str = TIMEFRAME,
                 paper_trading: bool = True):
        """
        Initialize the trading bot.
        
        Args:
            strategy: Trading strategy to use
            exchange_id: ID of the exchange to use
            trading_pair: Trading pair to trade
            timeframe: Timeframe for analysis
            paper_trading: Whether to use paper trading
        """
        self.strategy = strategy
        self.exchange_id = exchange_id
        self.trading_pair = trading_pair
        self.timeframe = timeframe
        self.paper_trading = paper_trading
        
        # Load environment variables
        env_vars = load_env_vars()
        
        # Override paper_trading from environment if provided
        if 'PAPER_TRADING' in env_vars:
            self.paper_trading = env_vars['PAPER_TRADING']
        
        # Initialize components
        self.exchange = ccxt.binance()
        
        self.execution_engine = ExecutionEngine(
            exchange_id=exchange_id,
            api_key=env_vars.get('PIONEX_API_KEY'),
            api_secret=env_vars.get('PIONEX_API_SECRET'),
            paper_trading=self.paper_trading
        )

        # Initialize DataFetcher
        self.data_fetcher = DataFetcher(exchange_id=exchange_id)
        
        logger.info(f"Initialized trading bot with {strategy.name} strategy")
        logger.info(f"Trading {trading_pair} on {exchange_id} using {timeframe} timeframe")
        logger.info(f"Paper trading: {'Enabled' if self.paper_trading else 'Disabled'}")
    
    def fetch_and_analyze(self) -> 'Tuple[pd.DataFrame, SignalType]':
        """
        Fetch market data and analyze it to generate a trading signal.
        
        Returns:
            Tuple of (dataframe_with_indicators_and_signals, latest_signal)
        """
        try:
            logger.info(f"Fetching and analyzing market data for {self.trading_pair}")
            
            # Fetch OHLCV data
            df = self.data_fetcher.fetch_ohlcv(
                symbol=self.trading_pair,
                timeframe=self.timeframe,
                limit=100  # Fetch enough data for indicators
            )
            
            # Calculate indicators
            df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)
            
            # Generate signals
            df_with_signals = self.strategy.generate_signals(df_with_indicators)
            
            # Get latest signal
            latest_signal = self.strategy.get_latest_signal(df_with_signals)
            
            logger.info(f"Analysis complete. Latest signal: {latest_signal.value}")
            
            return df_with_signals, latest_signal
            
        except Exception as e:
            logger.error(f"Error in fetch_and_analyze: {str(e)}")
            raise
    
    def execute_trading_cycle(self) -> Dict[str, Any]:
        """
        Execute a complete trading cycle: fetch data, analyze, and execute signal.
        
        Returns:
            Dictionary with execution results
        """
        try:
            logger.info("Starting trading cycle")
            
            # Fetch and analyze data
            df_with_signals, latest_signal = self.fetch_and_analyze()
            
            # Execute the signal
            execution_result = self.execution_engine.execute_signal(
                signal=latest_signal,
                symbol=self.trading_pair
            )
            
            logger.info(f"Trading cycle completed: {execution_result['message']}")
            
            return {
                'timestamp': datetime.now(),
                'signal': latest_signal.value,
                'execution_result': execution_result,
                'data': df_with_signals.tail(1)  # Just the latest row
            }
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")
            return {
                'timestamp': datetime.now(),
                'error': str(e)
            }
    
    def run_once(self) -> Dict[str, Any]:
        """
        Run a single trading cycle.
        
        Returns:
            Dictionary with execution results
        """
        return self.execute_trading_cycle()
    
    def run_scheduled(self, interval_minutes: int = 60) -> None:
        """
        Run the trading bot on a schedule.
        
        Args:
            interval_minutes: Interval between trading cycles in minutes
        """
        try:
            logger.info(f"Starting scheduled trading bot with {interval_minutes} minute interval")
            
            # Run once immediately
            self.execute_trading_cycle()
            
            # Schedule regular runs
            schedule.every(interval_minutes).minutes.do(self.execute_trading_cycle)
            
            logger.info(f"Trading bot scheduled to run every {interval_minutes} minutes")
            
            # Keep the script running
            while True:
                schedule.run_pending()
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
        except Exception as e:
            logger.error(f"Error in scheduled run: {str(e)}")
            raise

# Example usage
