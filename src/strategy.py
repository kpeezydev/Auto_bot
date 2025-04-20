import pandas as pd
import logging
from typing import Dict, Any, Tuple, List
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Enum for different types of trading signals."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"

class Strategy:
    """
    Base class for trading strategies.
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        """
        Initialize the strategy.
        
        Args:
            name: Name of the strategy
        """
        self.name = name
        logger.info(f"Initialized {name} strategy")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the strategy.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with added signal column
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement generate_signals method")
    
    def get_latest_signal(self, df: pd.DataFrame) -> SignalType:
        """
        Get the latest trading signal from the dataframe.
        
        Args:
            df: DataFrame with signals
            
        Returns:
            Latest signal
        """
        try:
            # Make sure signals have been generated
            if 'signal' not in df.columns:
                df = self.generate_signals(df)
            
            # Get the latest signal
            latest_signal = df['signal'].iloc[-1]
            
            # Convert to SignalType enum
            if isinstance(latest_signal, str):
                return SignalType(latest_signal)
            else:
                # Handle numeric signals if used
                signal_map = {
                    1: SignalType.BUY,
                    -1: SignalType.SELL,
                    0: SignalType.HOLD
                }
                return signal_map.get(latest_signal, SignalType.HOLD)
                
        except Exception as e:
            logger.error(f"Error getting latest signal: {str(e)}")
            # Default to HOLD in case of error
            return SignalType.HOLD

class MovingAverageCrossStrategy(Strategy):
    """
    Strategy based on Moving Average Crossover with RSI filter.
    """
    
    def __init__(self, fast_ma_period: int = 20, slow_ma_period: int = 50, 
                 rsi_period: int = 14, rsi_overbought: int = 70, rsi_oversold: int = 30):
        """
        Initialize the Moving Average Cross strategy.
        
        Args:
            fast_ma_period: Period for the fast moving average
            slow_ma_period: Period for the slow moving average
            rsi_period: Period for RSI calculation
            rsi_overbought: RSI threshold for overbought condition
            rsi_oversold: RSI threshold for oversold condition
        """
        super().__init__(name="MA Cross with RSI Filter")
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        
        logger.info(f"Strategy parameters: Fast MA={fast_ma_period}, Slow MA={slow_ma_period}, "
                   f"RSI Period={rsi_period}, RSI Overbought={rsi_overbought}, RSI Oversold={rsi_oversold}")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Moving Average Crossover with RSI filter.
        
        Buy signal: Fast MA crosses above Slow MA and RSI is below overbought
        Sell signal: Fast MA crosses below Slow MA and RSI is above oversold
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with added signal column
        """
        try:
            logger.info("Generating signals using MA Cross with RSI Filter strategy")
            
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Check if required columns exist
            fast_ma_col = f'sma_{self.fast_ma_period}'
            slow_ma_col = f'sma_{self.slow_ma_period}'
            rsi_col = f'rsi_{self.rsi_period}'
            
            required_cols = [fast_ma_col, slow_ma_col, rsi_col]
            missing_cols = [col for col in required_cols if col not in result_df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Initialize signal column with HOLD
            result_df['signal'] = SignalType.HOLD.value
            
            # Calculate crossover
            result_df['ma_crossover'] = (
                (result_df[fast_ma_col] > result_df[slow_ma_col]) & 
                (result_df[fast_ma_col].shift(1) <= result_df[slow_ma_col].shift(1))
            )
            
            result_df['ma_crossunder'] = (
                (result_df[fast_ma_col] < result_df[slow_ma_col]) & 
                (result_df[fast_ma_col].shift(1) >= result_df[slow_ma_col].shift(1))
            )
            
            # Generate signals
            # Buy when fast MA crosses above slow MA and RSI is not overbought
            buy_condition = (
                result_df['ma_crossover'] & 
                (result_df[rsi_col] < self.rsi_overbought)
            )
            
            # Sell when fast MA crosses below slow MA and RSI is not oversold
            sell_condition = (
                result_df['ma_crossunder'] & 
                (result_df[rsi_col] > self.rsi_oversold)
            )
            
            # Apply signals
            result_df.loc[buy_condition, 'signal'] = SignalType.BUY.value
            result_df.loc[sell_condition, 'signal'] = SignalType.SELL.value
            
            # Clean up temporary columns
            result_df = result_df.drop(['ma_crossover', 'ma_crossunder'], axis=1)
            
            logger.info(f"Generated {buy_condition.sum()} BUY signals and {sell_condition.sum()} SELL signals")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create sample data with indicators
    import numpy as np
    from indicators import TechnicalIndicators
    
    # Create sample price data
    dates = pd.date_range('20210101', periods=100)
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Add indicators
    df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)
    
    # Create strategy instance
    strategy = MovingAverageCrossStrategy()
    
    # Generate signals
    df_with_signals = strategy.generate_signals(df_with_indicators)
    
    # Print signals
    print(df_with_signals[['close', 'sma_20', 'sma_50', 'rsi_14', 'signal']].tail(10))
    
    # Get latest signal
    latest_signal = strategy.get_latest_signal(df_with_signals)
    print(f"Latest signal: {latest_signal}")