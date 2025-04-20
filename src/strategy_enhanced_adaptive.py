import pandas as pd
import logging
import pandas_ta as ta
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from enum import Enum
from src.strategy import Strategy, SignalType, MovingAverageCrossStrategy

# Configure logging
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Enum to represent different market regimes"""
    TRENDING_UP = 'trending_up'
    TRENDING_DOWN = 'trending_down'
    RANGING = 'ranging'
    VOLATILE = 'volatile'
    UNKNOWN = 'unknown'

class EnhancedAdaptiveStrategy(MovingAverageCrossStrategy):
    """
    Enhanced Adaptive Strategy that fixes issues with the original adaptive strategy
    to achieve better performance across different market conditions.
    """
    
    def __init__(self, 
                 # Base parameters with defaults
                 fast_ma_period: int = 12,        # Slower MA
                 slow_ma_period: int = 26,        # Slower MA
                 rsi_period: int = 14,
                 rsi_overbought: int = 70,        # Standard RSI level
                 rsi_oversold: int = 30,        # Standard RSI level
                 atr_period: int = 14,
                 atr_multiplier: float = 2.0,
                 trend_ema_period: int = 50,
                 volume_ma_period: int = 20,
                 # Adaptive parameters
                 use_adaptive_parameters: bool = False, # Disabled adaptive parameters
                 volatility_lookback: int = 50,
                 regime_lookback: int = 30,
                 # Risk management
                 use_trailing_stop: bool = True,
                 trailing_stop_pct: float = 2.5,  # Slightly wider stop
                 # Trade filtering
                 min_trade_duration: int = 4,
                 max_trades_per_day: int = 2,
                 # BTC-specific optimization
                 btc_specific_optimization: bool = False):   # Disabled BTC optimization
        """
        Initialize the Enhanced Adaptive strategy.
        
        Args:
            fast_ma_period: Base period for the fast moving average
            slow_ma_period: Base period for the slow moving average
            rsi_period: Base period for RSI calculation
            rsi_overbought: Base RSI threshold for overbought condition
            rsi_oversold: Base RSI threshold for oversold condition
            atr_period: Period for ATR calculation
            atr_multiplier: Multiplier for ATR to set stop loss
            trend_ema_period: Period for trend EMA
            volume_ma_period: Period for volume moving average
            use_adaptive_parameters: Whether to adapt parameters to market conditions
            volatility_lookback: Lookback period for volatility calculation
            regime_lookback: Lookback period for market regime detection
            use_trailing_stop: Whether to use trailing stop loss
            trailing_stop_pct: Trailing stop percentage
            min_trade_duration: Minimum number of candles to hold a trade
            max_trades_per_day: Maximum number of trades per day
        """
        super().__init__(fast_ma_period, slow_ma_period, rsi_period, rsi_overbought, rsi_oversold)
        self.name = "Enhanced Adaptive Strategy with Improved Filters"
        
        # Store base parameters
        self.base_fast_ma_period = fast_ma_period
        self.base_slow_ma_period = slow_ma_period
        self.base_rsi_period = rsi_period
        self.base_rsi_overbought = rsi_overbought
        self.base_rsi_oversold = rsi_oversold
        
        # Additional parameters
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.trend_ema_period = trend_ema_period
        self.volume_ma_period = volume_ma_period
        
        # Adaptive parameters
        self.use_adaptive_parameters = use_adaptive_parameters
        self.volatility_lookback = volatility_lookback
        self.regime_lookback = regime_lookback
        
        # Current trading pair
        self.current_symbol = None
        
        # Risk management
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        
        # Trade filtering
        self.min_trade_duration = min_trade_duration
        self.max_trades_per_day = max_trades_per_day
        
        # BTC-specific optimization
        self.btc_specific_optimization = btc_specific_optimization
        
        # Market state tracking
        self.current_regime = MarketRegime.UNKNOWN
        self.historical_volatility = None
        
        # BTC-specific parameters (will be used if trading BTC and optimization is enabled)
        self.btc_min_trade_duration = 8  # Longer hold time for BTC
        self.btc_max_trades_per_day = 1  # Fewer trades for BTC
        self.btc_trend_strength_threshold = 25  # Higher trend strength requirement
        self.btc_trailing_stop_pct = 3.5  # Wider trailing stop for BTC's volatility
        self.btc_bull_market_threshold = 0.15  # 15% increase over 30 days indicates bull market
        
        logger.info(f"Initialized {self.name} with enhanced parameters")
    
    def detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """
        Detect the current market regime based on price action and indicators.
        Improved to be more accurate and less prone to false signals.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            MarketRegime enum value
        """
        try:
            # Ensure we have enough data
            if len(df) < self.regime_lookback:
                return MarketRegime.UNKNOWN
            
            # Get recent data for regime detection
            recent_df = df.iloc[-self.regime_lookback:].copy()
            
            # Calculate price direction using multiple timeframes
            short_term_direction = recent_df['close'].iloc[-1] > recent_df['close'].iloc[-5]
            medium_term_direction = recent_df['close'].iloc[-1] > recent_df['close'].iloc[-15]
            long_term_direction = recent_df['close'].iloc[-1] > recent_df['close'].iloc[-self.regime_lookback]
            
            # Calculate trend strength using multiple indicators
            ema_col = f'ema_{self.trend_ema_period}'
            if ema_col not in recent_df.columns:
                recent_df[ema_col] = ta.ema(recent_df['close'], length=self.trend_ema_period)
            
            # Price relative to EMA
            price_vs_ema = recent_df['close'] / recent_df[ema_col]
            ema_trend = price_vs_ema.mean() > 1.0
            
            # ADX for trend strength
            trend_strength = False
            if 'adx' in recent_df.columns:
                trend_strength = recent_df['adx'].iloc[-1] > 20
            
            # MACD for trend confirmation
            macd_trend = False
            if 'macd' in recent_df.columns and 'macd_signal' in recent_df.columns:
                macd_trend = recent_df['macd'].iloc[-1] > recent_df['macd_signal'].iloc[-1]
            
            # Calculate volatility using ATR and price
            if 'atr' in recent_df.columns:
                volatility = recent_df['atr'].iloc[-1] / recent_df['close'].iloc[-1]
            else:
                volatility = recent_df['close'].pct_change().rolling(window=10).std().iloc[-1] * np.sqrt(10)
            
            high_volatility = volatility > self.historical_volatility * 1.3 if self.historical_volatility else volatility > 0.025
            
            # Determine market regime with improved logic
            if trend_strength and (ema_trend or macd_trend):
                if medium_term_direction and long_term_direction:
                    return MarketRegime.TRENDING_UP
                elif not medium_term_direction and not long_term_direction:
                    return MarketRegime.TRENDING_DOWN
                else:
                    return MarketRegime.VOLATILE if high_volatility else MarketRegime.RANGING
            elif high_volatility:
                return MarketRegime.VOLATILE
            else:
                return MarketRegime.RANGING
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return MarketRegime.UNKNOWN
    
    def calculate_historical_volatility(self, df: pd.DataFrame) -> float:
        """
        Calculate the historical volatility of the asset.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Historical volatility value
        """
        try:
            if len(df) < self.volatility_lookback:
                return 0.02  # Default value if not enough data
            
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            
            # Use exponentially weighted standard deviation to give more weight to recent volatility
            volatility = returns.ewm(span=self.volatility_lookback).std().iloc[-1] * np.sqrt(365)
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating historical volatility: {str(e)}")
            return 0.02  # Default value on error
            
    def is_btc_bull_market(self, df: pd.DataFrame) -> bool:
        """
        Detect if BTC is in a bull market phase where buy & hold might outperform active trading.
        This is a specialized function for BTC-specific optimization.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Boolean indicating if BTC is in a bull market
        """
        try:
            if len(df) < self.regime_lookback:
                return False
                
            # Calculate 30-day price change
            days_30_ago_idx = max(0, len(df) - 30)
            price_30d_ago = df['close'].iloc[days_30_ago_idx]
            current_price = df['close'].iloc[-1]
            price_change_30d = (current_price - price_30d_ago) / price_30d_ago
            
            # Check if we're in a strong uptrend (bull market)
            is_bull = price_change_30d > self.btc_bull_market_threshold
            
            # Additional bull market indicators
            if 'adx' in df.columns and 'macd' in df.columns:
                # Strong trend with positive MACD
                trend_strength = df['adx'].iloc[-1] > 25
                positive_momentum = df['macd'].iloc[-1] > 0 and df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]
                is_bull = is_bull and trend_strength and positive_momentum
            
            if is_bull:
                logger.info(f"BTC bull market detected: 30-day change: {price_change_30d:.2%}")
                
            return is_bull
            
        except Exception as e:
            logger.error(f"Error detecting BTC bull market: {str(e)}")
            return False
    
    def adapt_parameters_to_regime(self, regime: MarketRegime) -> None:
        """
        Adapt strategy parameters to the current market regime.
        Improved to make more moderate adjustments and avoid extreme parameter values.
        Includes BTC-specific optimizations when trading BTC pairs.
        
        Args:
            regime: Current market regime
        """
        try:
            logger.info(f"Adapting parameters to {regime.value} market regime")
            
            # Store current regime
            self.current_regime = regime
            
            # Check if we're trading BTC and should apply specific optimizations
            is_btc_pair = False
            if hasattr(self, 'current_symbol') and self.current_symbol and self.current_symbol.startswith('BTC/') and self.btc_specific_optimization:
                is_btc_pair = True
                logger.info(f"Applying BTC-specific parameter adaptations for {self.current_symbol}")
            
            # Reset parameters to base values first
            self.fast_ma_period = self.base_fast_ma_period
            self.slow_ma_period = self.base_slow_ma_period
            self.rsi_overbought = self.base_rsi_overbought
            self.rsi_oversold = self.base_rsi_oversold
            
            # Adjust parameters based on regime - more moderate adjustments
            if regime == MarketRegime.TRENDING_UP:
                # In uptrends, be more selective with entries but give trades room to run
                if is_btc_pair:
                    # BTC-specific adjustments for uptrends - much more conservative
                    self.fast_ma_period = max(10, self.fast_ma_period + 2)  # Slower fast MA for BTC
                    self.slow_ma_period = max(25, self.slow_ma_period + 4)  # Slower slow MA for BTC
                    self.rsi_oversold = max(40, self.rsi_oversold + 5)  # Higher oversold threshold for BTC
                    self.trailing_stop_pct = min(4.0, self.btc_trailing_stop_pct * 1.1)  # Wider stops for BTC
                else:
                    # Standard adjustments for other pairs
                    self.fast_ma_period = max(5, self.fast_ma_period - 1)
                    self.slow_ma_period = max(15, self.slow_ma_period - 2)
                    self.rsi_oversold = max(30, self.rsi_oversold - 5)  # Less extreme adjustment
                    self.trailing_stop_pct = min(3.0, self.trailing_stop_pct * 1.2)  # More moderate increase
                
            elif regime == MarketRegime.TRENDING_DOWN:
                # In downtrends, be more conservative
                if is_btc_pair:
                    # BTC-specific adjustments for downtrends - extremely selective
                    self.fast_ma_period = min(15, self.fast_ma_period + 7)  # Much slower fast MA for BTC
                    self.slow_ma_period = min(35, self.slow_ma_period + 14)  # Much slower slow MA for BTC
                    self.rsi_overbought = min(60, self.rsi_overbought - 5)  # Lower overbought for early exits
                    self.trailing_stop_pct = max(2.0, self.btc_trailing_stop_pct * 0.8)  # Tighter stops in downtrends
                else:
                    # Standard adjustments for other pairs
                    self.fast_ma_period = min(12, self.fast_ma_period + 2)
                    self.slow_ma_period = min(30, self.slow_ma_period + 3)
                    self.rsi_overbought = min(75, self.rsi_overbought + 5)  # Less extreme adjustment
                    self.trailing_stop_pct = max(1.5, self.trailing_stop_pct * 0.9)  # More moderate decrease
                
            elif regime == MarketRegime.RANGING:
                # In ranging markets, use faster indicators to catch swings
                if is_btc_pair:
                    # BTC-specific adjustments for ranging markets - avoid excessive trading
                    self.fast_ma_period = max(8, self.fast_ma_period)  # Keep fast MA moderate for BTC
                    self.slow_ma_period = max(21, self.slow_ma_period)  # Keep slow MA moderate for BTC
                    self.rsi_overbought = max(70, self.rsi_overbought + 5)  # Higher overbought threshold
                    self.rsi_oversold = min(30, self.rsi_oversold - 5)  # Lower oversold threshold
                    self.trailing_stop_pct = max(3.0, self.btc_trailing_stop_pct)  # Wider stops for BTC
                else:
                    # Standard adjustments for other pairs
                    self.fast_ma_period = max(5, self.fast_ma_period - 2)
                    self.slow_ma_period = max(15, self.slow_ma_period - 3)
                    self.rsi_overbought = max(60, self.rsi_overbought - 5)
                    self.rsi_oversold = min(40, self.rsi_oversold + 5)
                    self.trailing_stop_pct = max(1.5, self.trailing_stop_pct * 0.8)
                
            elif regime == MarketRegime.VOLATILE:
                # In volatile markets, use wider stops but be more selective with entries
                if is_btc_pair:
                    # BTC-specific adjustments for volatile markets - extremely selective
                    self.fast_ma_period = min(15, self.fast_ma_period + 7)  # Much slower fast MA for BTC
                    self.slow_ma_period = min(40, self.slow_ma_period + 19)  # Much slower slow MA for BTC
                    self.rsi_overbought = min(80, self.rsi_overbought + 15)  # Much higher overbought threshold
                    self.rsi_oversold = max(20, self.rsi_oversold - 15)  # Much lower oversold threshold
                    self.trailing_stop_pct = min(5.0, self.btc_trailing_stop_pct * 1.5)  # Much wider stops
                else:
                    # Standard adjustments for other pairs
                    self.fast_ma_period = min(10, self.fast_ma_period + 1)
                    self.slow_ma_period = min(25, self.slow_ma_period + 2)
                    self.rsi_overbought = min(75, self.rsi_overbought + 10)
                    self.rsi_oversold = max(25, self.rsi_oversold - 10)
                    self.trailing_stop_pct = min(3.5, self.trailing_stop_pct * 1.5)
                
            # Log the adapted parameters
            logger.info(f"Adapted parameters for {regime.value} regime: "
                       f"Fast MA={self.fast_ma_period}, Slow MA={self.slow_ma_period}, "
                       f"RSI Overbought={self.rsi_overbought}, RSI Oversold={self.rsi_oversold}, "
                       f"Trailing Stop %={self.trailing_stop_pct}")
                
        except Exception as e:
            logger.error(f"Error adapting parameters to regime {regime}: {str(e)}")
            # Revert to base parameters on error
            self.fast_ma_period = self.base_fast_ma_period
            self.slow_ma_period = self.base_slow_ma_period
            self.rsi_overbought = self.base_rsi_overbought
            self.rsi_oversold = self.base_rsi_oversold
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all required indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        try:
            logger.info("Adding indicators for enhanced adaptive strategy")
            
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Calculate historical volatility
            self.historical_volatility = self.calculate_historical_volatility(result_df)
            logger.info(f"Historical volatility: {self.historical_volatility:.4f}")
            
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
            
            # ADX for trend strength
            if 'adx' not in result_df.columns:
                adx = ta.adx(result_df['high'], result_df['low'], result_df['close'], length=14)
                adx.columns = ['adx', 'dmp', 'dmn']
                result_df = pd.concat([result_df, adx], axis=1)
            
            # Bollinger Bands for volatility and trend strength
            if 'bb_lower' not in result_df.columns:
                bbands = ta.bbands(result_df['close'], length=20, std=2.0)
                bbands.columns = ['bb_lower', 'bb_middle', 'bb_upper', 'bb_bandwidth', 'bb_percent']
                result_df = pd.concat([result_df, bbands], axis=1)
            
            # Add price momentum
            result_df['price_momentum'] = result_df['close'].pct_change(3) * 100
            
            # Add RSI momentum (RSI slope)
            result_df['rsi_momentum'] = result_df[rsi_col].diff(2)
            
            # Calculate trend strength
            result_df['trend_strength'] = abs(result_df[fast_ma_col] - result_df[slow_ma_col]) / result_df[slow_ma_col] * 100
            
            # Detect market regime
            if len(result_df) >= self.regime_lookback:
                regime = self.detect_market_regime(result_df)
                if self.use_adaptive_parameters:
                    self.adapt_parameters_to_regime(regime)
                    
                    # Recalculate moving averages with adapted parameters
                    fast_ma_col = f'sma_{self.fast_ma_period}'
                    slow_ma_col = f'sma_{self.slow_ma_period}'
                    if fast_ma_col not in result_df.columns:
                        result_df[fast_ma_col] = ta.sma(result_df['close'], length=self.fast_ma_period)
                    if slow_ma_col not in result_df.columns:
                        result_df[slow_ma_col] = ta.sma(result_df['close'], length=self.slow_ma_period)
            
            logger.info("Successfully added all indicators for enhanced adaptive strategy")
            return result_df
            
        except Exception as e:
            logger.error(f"Error adding indicators: {str(e)}")
            raise
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with improved filters to reduce false signals
        and increase win rate.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added signal column
        """
        try:
            logger.info("Generating signals using Enhanced Adaptive strategy")
            
            # Create a copy to avoid modifying the original
            result_df = df.copy()
            
            # Check if we're trading BTC and should apply specific optimizations
            is_btc_pair = False
            if hasattr(df, 'attrs') and 'symbol' in df.attrs:
                symbol = df.attrs['symbol']
                is_btc_pair = symbol.startswith('BTC/') and self.btc_specific_optimization
                if is_btc_pair:
                    logger.info(f"Applying BTC-specific optimizations for {symbol}")
                    
                    # Apply BTC-specific parameters
                    original_min_trade_duration = self.min_trade_duration
                    original_max_trades_per_day = self.max_trades_per_day
                    original_trailing_stop_pct = self.trailing_stop_pct
                    
                    self.min_trade_duration = self.btc_min_trade_duration
                    self.max_trades_per_day = self.btc_max_trades_per_day
                    self.trailing_stop_pct = self.btc_trailing_stop_pct
            
            # Add all required indicators
            result_df = self.add_indicators(result_df)
            
            # Detect market regime
            if len(result_df) >= self.regime_lookback:
                regime = self.detect_market_regime(result_df)
                
                # For BTC in bull markets, be more conservative with trading
                if is_btc_pair and self.is_btc_bull_market(result_df):
                    logger.info("Detected BTC bull market - using more conservative trading parameters")
                    regime = MarketRegime.TRENDING_UP  # Force trending up regime for more selective trading
                    self.max_trades_per_day = max(1, self.btc_max_trades_per_day - 1)  # Even fewer trades in bull market
                
                if self.use_adaptive_parameters:
                    self.adapt_parameters_to_regime(regime)
                    
                    # Recalculate moving averages with adapted parameters
                    fast_ma_col = f'sma_{self.fast_ma_period}'
                    slow_ma_col = f'sma_{self.slow_ma_period}'
                    if fast_ma_col not in result_df.columns:
                        result_df[fast_ma_col] = ta.sma(result_df['close'], length=self.fast_ma_period)
                    if slow_ma_col not in result_df.columns:
                        result_df[slow_ma_col] = ta.sma(result_df['close'], length=self.slow_ma_period)
            
            # Define column names
            fast_ma_col = f'sma_{self.fast_ma_period}'
            slow_ma_col = f'sma_{self.slow_ma_period}'
            rsi_col = f'rsi_{self.rsi_period}'
            atr_col = f'atr_{self.atr_period}'
            trend_ema_col = f'ema_{self.trend_ema_period}'
            volume_ma_col = f'volume_sma_{self.volume_ma_period}'
            
            # Initialize signal column with HOLD
            result_df['signal'] = SignalType.HOLD.value
            
            # Calculate crossover with confirmation
            result_df['ma_crossover'] = (
                (result_df[fast_ma_col] > result_df[slow_ma_col]) & 
                (result_df[fast_ma_col].shift(1) <= result_df[slow_ma_col].shift(1))
            )
            
            result_df['ma_crossunder'] = (
                (result_df[fast_ma_col] < result_df[slow_ma_col]) & 
                (result_df[fast_ma_col].shift(1) >= result_df[slow_ma_col].shift(1))
            )
            
            # Calculate normalized volatility
            result_df['norm_atr'] = result_df[atr_col] / result_df['close']
            
            # Trade count filter - limit number of trades per day
            result_df['date'] = pd.to_datetime(result_df.index).date
            result_df['daily_trade_count'] = 0
            
            # Generate base signals based on market regime
            if self.current_regime == MarketRegime.TRENDING_UP:
                # In uptrends, focus on trend following with stronger confirmation
                buy_condition = (
                    result_df['ma_crossover'] &
                    (result_df[rsi_col] > self.rsi_oversold) &
                    # (result_df['close'] > result_df[trend_ema_col]) & # Reverted Trend Filter
                    (result_df['macd_histogram'] > -0.1 * result_df['norm_atr']) &
                    (result_df['adx'] > 20) &  # Re-added ADX filter
                    (result_df['price_momentum'] > 0)  # Require positive momentum
                )
                
                sell_condition = (
                    (result_df['ma_crossunder'] & (result_df[rsi_col] > self.rsi_overbought * 0.9)) | # Reverted Trend Filter
                    # Add profit-taking exit when RSI gets very high
                    (result_df[rsi_col] > self.rsi_overbought + 10) |
                    # Exit when trend weakens significantly
                    ((result_df['adx'] < 15) & (result_df['price_momentum'] < 0))
                )
                
            elif self.current_regime == MarketRegime.TRENDING_DOWN:
                # In downtrends, be very selective with entries
                buy_condition = (
                    result_df['ma_crossover'] &
                    (result_df[rsi_col] > self.rsi_oversold * 1.2) &
                    # (result_df['close'] > result_df[trend_ema_col]) & # Reverted Trend Filter
                    (result_df['macd_histogram'] > 0.1 * result_df['norm_atr']) &
                    (result_df['volume'] > result_df[volume_ma_col] * 1.2) &
                    (result_df['price_momentum'] > 0) &
                    (result_df['adx'] > 25)  # Re-added ADX filter
                )
                
                sell_condition = (
                    result_df['ma_crossunder'] | # Reverted Trend Filter
                    (result_df[rsi_col] > self.rsi_overbought * 0.8) |
                    (result_df['price_momentum'] < -1.0)  # Exit quickly on momentum shift
                )
                
            elif self.current_regime == MarketRegime.RANGING:
                # In ranging markets, focus on oscillator extremes
                buy_condition = (
                    (result_df[rsi_col] < self.rsi_oversold * 1.1) &
                    (result_df[rsi_col] > result_df[rsi_col].shift(1)) &
                    (result_df[rsi_col] < self.rsi_oversold * 1.1) &
                    (result_df[rsi_col] > result_df[rsi_col].shift(1)) &
                    (result_df['bb_percent'] < 0.2) &
                    (result_df['volume'] > result_df[volume_ma_col] * 0.8) &
                    (result_df['adx'] > 20) # Re-added ADX filter
                )
                
                sell_condition = (
                    (result_df[rsi_col] > self.rsi_overbought * 0.9) &
                    (result_df[rsi_col] < result_df[rsi_col].shift(1)) &
                    (result_df['bb_percent'] > 0.8) &
                    (result_df['adx'] > 20) # Re-added ADX filter
                )
                
            elif self.current_regime == MarketRegime.VOLATILE:
                # In volatile markets, be very selective and use wider stops
                buy_condition = (
                    result_df['ma_crossover'] &
                    (result_df[rsi_col] < self.rsi_oversold * 1.2) &
                    (result_df['adx'] > 25) &
                    (result_df['volume'] > result_df[volume_ma_col] * 1.5) &
                    (result_df['price_momentum'] > 1.0) &
                    (result_df['macd_histogram'] > result_df['macd_histogram'].shift(1)) &  # Increasing momentum
                    (result_df['adx'] > 25) # Re-added ADX filter
                )
                
                sell_condition = (
                    result_df['ma_crossunder'] &
                    (result_df[rsi_col] > self.rsi_overbought * 0.7) &
                    (result_df['adx'] > 20) # Re-added ADX filter
                )
                
            else:  # UNKNOWN or default
                # Use more conservative conditions
                buy_condition = (
                    result_df['ma_crossover'] &
                    (result_df[rsi_col] > self.rsi_oversold) &
                    (result_df['macd_histogram'] > 0) &
                    (result_df['adx'] > 20) # Re-added ADX filter
                )
                
                sell_condition = (
                    result_df['ma_crossunder'] &
                    (result_df[rsi_col] < self.rsi_overbought) &
                    (result_df['macd_histogram'] < 0) &
                    (result_df['adx'] > 20) # Re-added ADX filter
                )
            
            # Apply minimum trade duration filter
            for i in range(1, len(result_df)):
                # If we have an open position
                if result_df['signal'].iloc[i-1] == SignalType.BUY.value:
                    # Count how long we've been in the trade
                    trade_duration = 1
                    for j in range(i-2, -1, -1):
                        if result_df['signal'].iloc[j] == SignalType.BUY.value:
                            trade_duration += 1
                        else:
                            break
                    
                    # If we haven't held the trade for minimum duration, don't sell yet
                    if trade_duration < self.min_trade_duration and sell_condition.iloc[i]:
                        sell_condition.iloc[i] = False
            
            # Apply maximum trades per day filter
            current_date = None
            daily_trade_count = 0
            
            for i in range(len(result_df)):
                date = pd.to_datetime(result_df.index[i]).date()
                
                # Reset counter on new day
                if current_date != date:
                    current_date = date
                    daily_trade_count = 0
                
                # If we're about to enter a new trade
                if buy_condition.iloc[i]:
                    # Check if we've reached the daily limit
                    if daily_trade_count >= self.max_trades_per_day:
                        buy_condition.iloc[i] = False
                    else:
                        daily_trade_count += 1
                        result_df['daily_trade_count'].iloc[i] = daily_trade_count
            
            # Trailing stop loss logic
            if self.use_trailing_stop:
                # Calculate trailing stop levels
                result_df['highest_high'] = result_df['close'].rolling(window=5, min_periods=1).max()
                result_df['lowest_low'] = result_df['close'].rolling(window=5, min_periods=1).min()
                
                # Adjust trailing stop based on volatility
                volatility_factor = result_df['norm_atr'] / self.historical_volatility if self.historical_volatility else 1.0
                result_df['trailing_stop_pct'] = self.trailing_stop_pct * volatility_factor.clip(0.7, 1.5)  # Less extreme adjustment
                
                result_df['trailing_stop_long'] = result_df['highest_high'] * (1 - result_df['trailing_stop_pct']/100)
                
                # Add trailing stop exit signals
                for i in range(1, len(result_df)):
                    # If we have an open long position
                    if result_df['signal'].iloc[i-1] == SignalType.BUY.value and result_df['signal'].iloc[i] == SignalType.HOLD.value:
                        # Check if price dropped below trailing stop
                        if result_df['close'].iloc[i] < result_df['trailing_stop_long'].iloc[i-1]:
                            result_df.loc[result_df.index[i], 'signal'] = SignalType.SELL.value
            
            # Apply primary signals
            result_df.loc[buy_condition, 'signal'] = SignalType.BUY.value
            result_df.loc[sell_condition, 'signal'] = SignalType.SELL.value
            
            # Apply additional BTC-specific filters if needed
            if is_btc_pair:
                # For BTC, require stronger trend confirmation for entries
                btc_trend_filter = (result_df['adx'] > self.btc_trend_strength_threshold) & \
                                   (result_df['trend_strength'] > 1.5) & \
                                   (result_df['volume'] > result_df[volume_ma_col] * 1.2)
                
                # Only keep buy signals that pass the additional BTC filter
                result_df.loc[(result_df['signal'] == SignalType.BUY.value) & ~btc_trend_filter, 'signal'] = SignalType.HOLD.value
                
                # Restore original parameters if we modified them
                if 'original_min_trade_duration' in locals():
                    self.min_trade_duration = original_min_trade_duration
                    self.max_trades_per_day = original_max_trades_per_day
                    self.trailing_stop_pct = original_trailing_stop_pct
            
            # Clean up temporary columns
            cols_to_drop = ['ma_crossover', 'ma_crossunder', 'norm_atr', 'date', 'daily_trade_count']
            if self.use_trailing_stop:
                cols_to_drop.extend(['highest_high', 'lowest_low', 'trailing_stop_pct', 'trailing_stop_long'])
            
            result_df = result_df.drop(cols_to_drop, axis=1, errors='ignore')
            
            logger.info(f"Generated {buy_condition.sum()} BUY signals and {sell_condition.sum()} SELL signals")
            logger.info(f"Current market regime: {self.current_regime.value}")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise
