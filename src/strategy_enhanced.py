import pandas as pd
import logging
from typing import Dict, Any, Tuple, List
from enum import Enum
import pandas_ta as ta
from pandas_ta.trend import increasing, decreasing
from pandas_ta.volatility import kc
from src.strategy import Strategy, SignalType, MovingAverageCrossStrategy

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedMAStrategy(MovingAverageCrossStrategy):
    """
    Enhanced Moving Average Cross Strategy with Squeeze Pro indicator and additional filters
    to achieve a higher win rate.
    """
    
    def __init__(self, fast_ma_period: int = 20, slow_ma_period: int = 50, 
                 rsi_period: int = 14, rsi_overbought: int = 70, rsi_oversold: int = 30,
                 atr_period: int = 14, atr_multiplier: float = 2.0,
                 bb_length: int = 20, bb_std: float = 2.0,
                 kc_length: int = 20, kc_scalar: float = 1.5,
                 mom_length: int = 12, mom_smooth: int = 6):
        """
        Initialize the Enhanced Moving Average Cross strategy with Squeeze Pro.
        
        Args:
            fast_ma_period: Period for the fast moving average
            slow_ma_period: Period for the slow moving average
            rsi_period: Period for RSI calculation
            rsi_overbought: RSI threshold for overbought condition
            rsi_oversold: RSI threshold for oversold condition
            atr_period: Period for ATR calculation
            atr_multiplier: Multiplier for ATR to set stop loss
            bb_length: Bollinger Bands period for Squeeze Pro
            bb_std: Bollinger Bands standard deviation for Squeeze Pro
            kc_length: Keltner Channel period for Squeeze Pro
            kc_scalar: Keltner Channel scalar for Squeeze Pro
            mom_length: Momentum period for Squeeze Pro
            mom_smooth: Momentum smoothing period for Squeeze Pro
        """
        super().__init__(fast_ma_period, slow_ma_period, rsi_period, rsi_overbought, rsi_oversold)
        self.name = "Enhanced MA Cross with Squeeze Pro"
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.bb_length = bb_length
        self.bb_std = bb_std
        self.kc_length = kc_length
        self.kc_scalar = kc_scalar
        self.mom_length = mom_length
        self.mom_smooth = mom_smooth
        
        logger.info(f"Enhanced Strategy parameters: Fast MA={fast_ma_period}, Slow MA={slow_ma_period}, "
                   f"RSI Period={rsi_period}, RSI Overbought={rsi_overbought}, RSI Oversold={rsi_oversold}, "
                   f"ATR Period={atr_period}, ATR Multiplier={atr_multiplier}, "
                   f"Squeeze Pro: BB Length={bb_length}, BB Std={bb_std}, KC Length={kc_length}, KC Scalar={kc_scalar}")
    
    def add_squeeze_pro(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Squeeze Pro indicator to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added Squeeze Pro indicator
        """
        try:
            logger.info("Adding Squeeze Pro indicator")
            
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Calculate Bollinger Bands
            bb = ta.bbands(result_df['close'], length=self.bb_length, std=self.bb_std)
            
            # Calculate Keltner Channels manually
            typical_price = (result_df['high'] + result_df['low'] + result_df['close']) / 3
            atr = ta.atr(result_df['high'], result_df['low'], result_df['close'], length=self.kc_length)
            
            # Calculate different Keltner Channel widths
            kc_middle = ta.ema(typical_price, length=self.kc_length)
            kc_wide_upper = kc_middle + atr * (self.kc_scalar + 0.5)
            kc_wide_lower = kc_middle - atr * (self.kc_scalar + 0.5)
            kc_normal_upper = kc_middle + atr * self.kc_scalar
            kc_normal_lower = kc_middle - atr * self.kc_scalar
            kc_narrow_upper = kc_middle + atr * (self.kc_scalar - 0.5)
            kc_narrow_lower = kc_middle - atr * (self.kc_scalar - 0.5)
            
            # Calculate if squeeze is on (Bollinger Bands inside Keltner Channel)
            result_df['SQZPRO_ON'] = ((bb['BBL_' + str(self.bb_length) + '_' + str(self.bb_std)] > kc_normal_lower) & 
                                      (bb['BBU_' + str(self.bb_length) + '_' + str(self.bb_std)] < kc_normal_upper)).astype(int)
            result_df['SQZPRO_OFF'] = 1 - result_df['SQZPRO_ON']
            
            # Calculate momentum
            momentum = ta.mom(result_df['close'], length=self.mom_length)
            result_df['SQZPRO_MOMENTUM'] = ta.ema(momentum, length=self.mom_smooth)
            
            # Determine if momentum is increasing or decreasing
            result_df['SQZPRO_INC'] = result_df['SQZPRO_MOMENTUM'].diff() > 0
            result_df['SQZPRO_DEC'] = result_df['SQZPRO_MOMENTUM'].diff() < 0
            
            logger.info("Successfully added Squeeze Pro indicator")
            return result_df
            
        except Exception as e:
            logger.error(f"Error adding Squeeze Pro indicator: {str(e)}")
            raise
    
    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Average True Range (ATR) to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added ATR
        """
        try:
            logger.info(f"Calculating ATR with period {self.atr_period}")
            
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Calculate ATR
            result_df[f'atr_{self.atr_period}'] = ta.atr(result_df['high'], result_df['low'], result_df['close'], length=self.atr_period)
            
            logger.info("Successfully calculated ATR")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            raise
    
    def add_trend_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend filter to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added trend filter
        """
        try:
            logger.info("Adding trend filter")
            
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Calculate EMA for trend direction
            ema_200 = ta.ema(result_df['close'], length=200)
            result_df['ema_200'] = ema_200
            
            # Determine trend direction
            result_df['uptrend'] = result_df['close'] > result_df['ema_200']
            result_df['downtrend'] = result_df['close'] < result_df['ema_200']
            
            logger.info("Successfully added trend filter")
            return result_df
            
        except Exception as e:
            logger.error(f"Error adding trend filter: {str(e)}")
            raise
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Enhanced Moving Average Crossover with Squeeze Pro and additional filters.
        
        Buy signal: 
        - Fast MA crosses above Slow MA
        - RSI is below overbought
        - Squeeze Pro is firing (squeeze off and momentum increasing)
        - Price is in an uptrend (above 200 EMA)
        
        Sell signal: 
        - Fast MA crosses below Slow MA
        - RSI is above oversold
        - Squeeze Pro momentum is decreasing
        - Price is in a downtrend (below 200 EMA)
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with added signal column
        """
        try:
            logger.info("Generating signals using Enhanced MA Cross with Squeeze Pro strategy")
            
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Add required indicators if not present
            if f'sma_{self.fast_ma_period}' not in result_df.columns or f'sma_{self.slow_ma_period}' not in result_df.columns:
                fast_ma_col = f'sma_{self.fast_ma_period}'
                slow_ma_col = f'sma_{self.slow_ma_period}'
                result_df[fast_ma_col] = ta.sma(result_df['close'], length=self.fast_ma_period)
                result_df[slow_ma_col] = ta.sma(result_df['close'], length=self.slow_ma_period)
            
            if f'rsi_{self.rsi_period}' not in result_df.columns:
                rsi_col = f'rsi_{self.rsi_period}'
                result_df[rsi_col] = ta.rsi(result_df['close'], length=self.rsi_period)
            
            if f'atr_{self.atr_period}' not in result_df.columns:
                result_df = self.add_atr(result_df)
            
            # Add Squeeze Pro if not present
            sqz_cols = [col for col in result_df.columns if 'SQZPRO' in col]
            if not sqz_cols:
                result_df = self.add_squeeze_pro(result_df)
            
            # Add trend filter if not present
            if 'ema_200' not in result_df.columns:
                result_df = self.add_trend_filter(result_df)
            
            # Check if required columns exist
            fast_ma_col = f'sma_{self.fast_ma_period}'
            slow_ma_col = f'sma_{self.slow_ma_period}'
            rsi_col = f'rsi_{self.rsi_period}'
            
            required_cols = [fast_ma_col, slow_ma_col, rsi_col, 'SQZPRO_OFF', 'SQZPRO_INC', 'ema_200']
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
            
            # Generate signals with enhanced filters
            # Buy when:
            # 1. Fast MA crosses above slow MA
            # 2. RSI is not overbought
            # 3. Squeeze Pro is firing (squeeze off and momentum increasing)
            # 4. Price is in an uptrend (above 200 EMA)
            buy_condition = (
                result_df['ma_crossover'] & 
                (result_df[rsi_col] < self.rsi_overbought) &
                (result_df['SQZPRO_OFF'] == 1) &  # Squeeze is off (volatility expanding)
                (result_df['SQZPRO_INC'].notna()) &  # Momentum is increasing
                (result_df['uptrend'])  # Price is in uptrend
            )
            
            # Sell when:
            # 1. Fast MA crosses below slow MA
            # 2. RSI is not oversold
            # 3. Squeeze Pro momentum is decreasing
            # 4. Price is in a downtrend (below 200 EMA)
            sell_condition = (
                result_df['ma_crossunder'] & 
                (result_df[rsi_col] > self.rsi_oversold) &
                (result_df['SQZPRO_DEC'].notna()) &  # Momentum is decreasing
                (result_df['downtrend'])  # Price is in downtrend
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
    from src.indicators import TechnicalIndicators
    
    # Create sample price data
    dates = pd.date_range('20210101', periods=200)
    df = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 102,
        'low': np.random.randn(200).cumsum() + 98,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 200)
    }, index=dates)
    
    # Create strategy instance
    strategy = EnhancedMAStrategy()
    
    # Add indicators
    df_with_indicators = TechnicalIndicators.calculate_all_indicators(df)
    
    # Add Squeeze Pro indicator
    df_with_indicators = strategy.add_squeeze_pro(df_with_indicators)
    
    # Add trend filter
    df_with_indicators = strategy.add_trend_filter(df_with_indicators)
    
    # Generate signals
    df_with_signals = strategy.generate_signals(df_with_indicators)
    
    # Print results
    print(df_with_signals[['close', 'signal']].tail(20))