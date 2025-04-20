import pandas as pd
import logging
import pandas_ta as ta
from typing import Dict, Any, Tuple, List
from enum import Enum
from src.strategy import Strategy, SignalType, MovingAverageCrossStrategy

# Configure logging
logger = logging.getLogger(__name__)

class ImprovedMAStrategy(MovingAverageCrossStrategy):
    """
    Improved Moving Average Cross Strategy with optimized parameters and additional filters
    to achieve a higher win rate and positive returns.
    """
    
    def __init__(self, fast_ma_period: int = 8, slow_ma_period: int = 21, 
                 rsi_period: int = 14, rsi_overbought: int = 70, rsi_oversold: int = 30,
                 atr_period: int = 14, atr_multiplier: float = 2.5,
                 trend_ema_period: int = 50, volume_ma_period: int = 20,
                 use_trailing_stop: bool = True, trailing_stop_pct: float = 1.5):
        """
        Initialize the Improved Moving Average Cross strategy.
        
        Args:
            fast_ma_period: Period for the fast moving average (optimized from 9 to 8)
            slow_ma_period: Period for the slow moving average
            rsi_period: Period for RSI calculation
            rsi_overbought: RSI threshold for overbought condition (optimized from 65 to 70)
            rsi_oversold: RSI threshold for oversold condition (optimized from 35 to 30)
            atr_period: Period for ATR calculation
            atr_multiplier: Multiplier for ATR to set stop loss (increased from 2.0 to 2.5)
            trend_ema_period: Period for trend EMA (reduced from 100 to 50 for more responsiveness)
            volume_ma_period: Period for volume moving average
            use_trailing_stop: Whether to use trailing stop loss
            trailing_stop_pct: Trailing stop percentage
        """
        super().__init__(fast_ma_period, slow_ma_period, rsi_period, rsi_overbought, rsi_oversold)
        self.name = "Improved MA Cross with Adaptive Filters"
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.trend_ema_period = trend_ema_period
        self.volume_ma_period = volume_ma_period
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        
        logger.info(f"Improved Strategy parameters: Fast MA={fast_ma_period}, Slow MA={slow_ma_period}, "
                   f"RSI Period={rsi_period}, RSI Overbought={rsi_overbought}, RSI Oversold={rsi_oversold}, "
                   f"ATR Period={atr_period}, Trend EMA={trend_ema_period}, Volume MA={volume_ma_period}, "
                   f"Trailing Stop: {'Enabled' if use_trailing_stop else 'Disabled'}, Stop %={trailing_stop_pct}")
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all required indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        try:
            logger.info("Adding indicators for improved strategy")
            
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Add required indicators if not present
            # Moving Averages
            fast_ma_col = f'sma_{self.fast_ma_period}'
            slow_ma_col = f'sma_{self.slow_ma_period}'
            if fast_ma_col not in result_df.columns:
                result_df[fast_ma_col] = ta.sma(result_df['close'], length=self.fast_ma_period)
            if slow_ma_col not in result_df.columns:
                result_df[slow_ma_col] = ta.sma(result_df['close'], length=self.slow_ma_period)
            
            # RSI
            rsi_col = f'rsi_{self.rsi_period}'
            if rsi_col not in result_df.columns:
                result_df[rsi_col] = ta.rsi(result_df['close'], length=self.rsi_period)
            
            # ATR for volatility measurement
            atr_col = f'atr_{self.atr_period}'
            if atr_col not in result_df.columns:
                result_df[atr_col] = ta.atr(result_df['high'], result_df['low'], result_df['close'], length=self.atr_period)
            
            # Trend EMA
            trend_ema_col = f'ema_{self.trend_ema_period}'
            if trend_ema_col not in result_df.columns:
                result_df[trend_ema_col] = ta.ema(result_df['close'], length=self.trend_ema_period)
            
            # Volume MA
            volume_ma_col = f'volume_sma_{self.volume_ma_period}'
            if volume_ma_col not in result_df.columns:
                result_df[volume_ma_col] = ta.sma(result_df['volume'], length=self.volume_ma_period)
            
            # MACD for trend confirmation
            if 'macd' not in result_df.columns:
                macd = ta.macd(result_df['close'], fast=12, slow=26, signal=9)
                macd.columns = ['macd', 'macd_signal', 'macd_histogram']
                result_df = pd.concat([result_df, macd], axis=1)
            
            # Bollinger Bands for volatility and trend strength
            if 'bb_lower' not in result_df.columns:
                bbands = ta.bbands(result_df['close'], length=20, std=2.0)
                bbands.columns = ['bb_lower', 'bb_middle', 'bb_upper', 'bb_bandwidth', 'bb_percent']
                result_df = pd.concat([result_df, bbands], axis=1)
            
            # Add price momentum
            result_df['price_momentum'] = result_df['close'].pct_change(3) * 100
            
            # Add RSI momentum (RSI slope)
            result_df['rsi_momentum'] = result_df[rsi_col].diff(2)
            
            logger.info("Successfully added all indicators for improved strategy")
            return result_df
            
        except Exception as e:
            logger.error(f"Error adding indicators: {str(e)}")
            raise
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Improved Moving Average Crossover with adaptive filters.
        
        Buy signal: 
        - Fast MA crosses above Slow MA
        - RSI confirms trend direction (not extreme)
        - Price momentum is positive
        - Volume filter is applied conditionally
        
        Sell signal: 
        - Fast MA crosses below Slow MA
        - RSI confirms trend direction
        - Price momentum is negative
        - Trailing stop loss for better profit protection
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with added signal column
        """
        try:
            logger.info("Generating signals using Improved MA Cross strategy")
            
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Add all required indicators
            result_df = self.add_indicators(result_df)
            
            # Define column names
            fast_ma_col = f'sma_{self.fast_ma_period}'
            slow_ma_col = f'sma_{self.slow_ma_period}'
            rsi_col = f'rsi_{self.rsi_period}'
            atr_col = f'atr_{self.atr_period}'
            trend_ema_col = f'ema_{self.trend_ema_period}'
            volume_ma_col = f'volume_sma_{self.volume_ma_period}'
            
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
            
            # Calculate trend strength
            result_df['trend_strength'] = abs(result_df[fast_ma_col] - result_df[slow_ma_col]) / result_df[slow_ma_col] * 100
            
            # Adaptive RSI thresholds based on trend strength
            result_df['adaptive_rsi_upper'] = self.rsi_overbought - (result_df['trend_strength'] * 0.2).clip(0, 10)
            result_df['adaptive_rsi_lower'] = self.rsi_oversold + (result_df['trend_strength'] * 0.2).clip(0, 10)
            
            # Generate signals with adaptive filters
            # Buy when fast MA crosses above slow MA with adaptive filters
            buy_condition = (
                result_df['ma_crossover'] & 
                # RSI filter - less restrictive during strong trends
                ((result_df[rsi_col] > result_df['adaptive_rsi_lower']) | 
                 (result_df['trend_strength'] > 5)) &  
                # Only require price > trend EMA in weak trends
                ((result_df['close'] > result_df[trend_ema_col]) | 
                 (result_df['trend_strength'] > 3)) &  
                # Volume filter - only required in weak trends
                ((result_df['volume'] > result_df[volume_ma_col] * 0.8) | 
                 (result_df['trend_strength'] > 4)) &  
                # MACD filter - less restrictive
                ((result_df['macd_histogram'] > -0.1) | 
                 (result_df['macd_histogram'] > result_df['macd_histogram'].shift(1))) 
            )
            
            # Sell when fast MA crosses below slow MA with adaptive filters
            sell_condition = (
                result_df['ma_crossunder'] & 
                # RSI filter - less restrictive during strong trends
                ((result_df[rsi_col] < result_df['adaptive_rsi_upper']) | 
                 (result_df['trend_strength'] > 5)) &  
                # Only require price < trend EMA in weak trends
                ((result_df['close'] < result_df[trend_ema_col]) | 
                 (result_df['trend_strength'] > 3)) &  
                # MACD filter - less restrictive
                ((result_df['macd_histogram'] < 0.1) | 
                 (result_df['macd_histogram'] < result_df['macd_histogram'].shift(1)))
            )
            
            # Trailing stop loss logic
            if self.use_trailing_stop:
                # Calculate trailing stop levels
                result_df['highest_high'] = result_df['close'].rolling(window=5, min_periods=1).max()
                result_df['lowest_low'] = result_df['close'].rolling(window=5, min_periods=1).min()
                result_df['trailing_stop_long'] = result_df['highest_high'] * (1 - self.trailing_stop_pct/100)
                result_df['trailing_stop_short'] = result_df['lowest_low'] * (1 + self.trailing_stop_pct/100)
                
                # Add trailing stop exit signals
                for i in range(1, len(result_df)):
                    # If we have an open long position
                    if result_df['signal'].iloc[i-1] == SignalType.BUY.value and result_df['signal'].iloc[i] == SignalType.HOLD.value:
                        # Check if price dropped below trailing stop
                        if result_df['close'].iloc[i] < result_df['trailing_stop_long'].iloc[i-1]:
                            result_df.loc[result_df.index[i], 'signal'] = SignalType.SELL.value
                    
                    # If we have an open short position
                    elif result_df['signal'].iloc[i-1] == SignalType.SELL.value and result_df['signal'].iloc[i] == SignalType.HOLD.value:
                        # Check if price rose above trailing stop
                        if result_df['close'].iloc[i] > result_df['trailing_stop_short'].iloc[i-1]:
                            result_df.loc[result_df.index[i], 'signal'] = SignalType.BUY.value
            
            # Apply primary signals
            result_df.loc[buy_condition, 'signal'] = SignalType.BUY.value
            result_df.loc[sell_condition, 'signal'] = SignalType.SELL.value
            
            # Clean up temporary columns
            result_df = result_df.drop(['ma_crossover', 'ma_crossunder', 'trend_strength', 
                                       'adaptive_rsi_upper', 'adaptive_rsi_lower'], axis=1)
            if self.use_trailing_stop:
                result_df = result_df.drop(['highest_high', 'lowest_low', 
                                           'trailing_stop_long', 'trailing_stop_short'], axis=1)
            
            logger.info(f"Generated {buy_condition.sum()} BUY signals and {sell_condition.sum()} SELL signals")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise
