"""
Market Structure Analysis Module for Auto_bot
Provides support/resistance detection and market structure analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)

class MarketStructure:
    """
    Market structure analysis for better entry/exit timing and stop loss placement.
    """
    
    @staticmethod
    def find_swing_points(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Find swing highs and lows in the price data.
        
        Args:
            df: DataFrame with OHLCV data
            window: Window size for swing point detection
            
        Returns:
            DataFrame with swing high/low columns added
        """
        try:
            result_df = df.copy()
            
            # Find local maxima (swing highs)
            high_indices = argrelextrema(result_df['high'].values, np.greater, order=window)[0]
            result_df['swing_high'] = np.nan
            result_df.iloc[high_indices, result_df.columns.get_loc('swing_high')] = result_df.iloc[high_indices]['high']
            
            # Find local minima (swing lows)
            low_indices = argrelextrema(result_df['low'].values, np.less, order=window)[0]
            result_df['swing_low'] = np.nan
            result_df.iloc[low_indices, result_df.columns.get_loc('swing_low')] = result_df.iloc[low_indices]['low']
            
            logger.info(f"Found {len(high_indices)} swing highs and {len(low_indices)} swing lows")
            return result_df
            
        except Exception as e:
            logger.error(f"Error finding swing points: {str(e)}")
            raise
    
    @staticmethod
    def identify_support_resistance(df: pd.DataFrame, lookback: int = 50, 
                                  tolerance_pct: float = 0.5) -> Dict[str, List[float]]:
        """
        Identify key support and resistance levels.
        
        Args:
            df: DataFrame with swing points
            lookback: Number of periods to look back for levels
            tolerance_pct: Tolerance percentage for grouping similar levels
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            # Get recent swing points
            recent_df = df.tail(lookback)
            
            # Extract swing highs and lows (remove NaN values)
            swing_highs = recent_df['swing_high'].dropna().values
            swing_lows = recent_df['swing_low'].dropna().values
            
            # Group similar levels
            def group_levels(levels, tolerance_pct):
                if len(levels) == 0:
                    return []
                
                grouped = []
                sorted_levels = np.sort(levels)
                
                current_group = [sorted_levels[0]]
                
                for level in sorted_levels[1:]:
                    # Check if level is within tolerance of current group average
                    group_avg = np.mean(current_group)
                    if abs(level - group_avg) / group_avg <= tolerance_pct / 100:
                        current_group.append(level)
                    else:
                        # Start new group
                        grouped.append(np.mean(current_group))
                        current_group = [level]
                
                # Add the last group
                if current_group:
                    grouped.append(np.mean(current_group))
                
                return grouped
            
            resistance_levels = group_levels(swing_highs, tolerance_pct)
            support_levels = group_levels(swing_lows, tolerance_pct)
            
            logger.info(f"Identified {len(resistance_levels)} resistance and {len(support_levels)} support levels")
            
            return {
                'resistance': resistance_levels,
                'support': support_levels
            }
            
        except Exception as e:
            logger.error(f"Error identifying support/resistance: {str(e)}")
            return {'resistance': [], 'support': []}
    
    @staticmethod
    def calculate_trend_strength(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Calculate trend strength using multiple methods.
        
        Args:
            df: DataFrame with price data
            period: Period for calculations
            
        Returns:
            DataFrame with trend strength indicators
        """
        try:
            result_df = df.copy()
            
            # Method 1: EMA slope
            ema = result_df['close'].ewm(span=period).mean()
            ema_slope = (ema - ema.shift(period)) / period
            result_df['ema_slope'] = ema_slope
            
            # Method 2: Higher highs and lower lows count
            def count_hh_ll(series, window):
                hh_count = 0
                ll_count = 0
                for i in range(len(series) - window, len(series)):
                    if i > 0:
                        if series.iloc[i] > series.iloc[i-1]:
                            hh_count += 1
                        elif series.iloc[i] < series.iloc[i-1]:
                            ll_count += 1
                return hh_count, ll_count
            
            # Calculate trend consistency
            result_df['trend_consistency'] = 0.0
            for i in range(period, len(result_df)):
                window_data = result_df['close'].iloc[i-period:i]
                hh, ll = count_hh_ll(window_data, period)
                total = hh + ll
                if total > 0:
                    # Positive for uptrend, negative for downtrend
                    result_df.iloc[i, result_df.columns.get_loc('trend_consistency')] = (hh - ll) / total
            
            # Method 3: Price position relative to moving averages
            sma_20 = result_df['close'].rolling(window=20).mean()
            sma_50 = result_df['close'].rolling(window=50).mean()
            
            result_df['price_above_sma20'] = (result_df['close'] > sma_20).astype(int)
            result_df['price_above_sma50'] = (result_df['close'] > sma_50).astype(int)
            result_df['sma20_above_sma50'] = (sma_20 > sma_50).astype(int)
            
            # Combined trend score (-1 to 1)
            result_df['trend_score'] = (
                result_df['price_above_sma20'] + 
                result_df['price_above_sma50'] + 
                result_df['sma20_above_sma50'] - 1.5
            ) / 1.5
            
            logger.info("Successfully calculated trend strength indicators")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating trend strength: {str(e)}")
            raise
    
    @staticmethod
    def get_dynamic_stop_loss(df: pd.DataFrame, direction: str, current_price: float,
                            atr_multiplier: float = 2.0) -> float:
        """
        Calculate dynamic stop loss based on market structure and ATR.
        
        Args:
            df: DataFrame with market data and swing points
            direction: 'long' or 'short'
            current_price: Current entry price
            atr_multiplier: ATR multiplier for stop distance
            
        Returns:
            Stop loss price
        """
        try:
            # Get recent ATR
            if 'atr_14' in df.columns:
                current_atr = df['atr_14'].iloc[-1]
            else:
                # Calculate ATR if not available
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                current_atr = true_range.rolling(window=14).mean().iloc[-1]
            
            # ATR-based stop
            atr_stop_distance = current_atr * atr_multiplier
            
            if direction == 'long':
                atr_stop = current_price - atr_stop_distance
                
                # Find nearest swing low for structure-based stop
                recent_swing_lows = df['swing_low'].dropna().tail(10)
                if len(recent_swing_lows) > 0:
                    structure_stop = recent_swing_lows.min()
                    # Use the lower of ATR stop or structure stop (more conservative)
                    final_stop = min(atr_stop, structure_stop)
                else:
                    final_stop = atr_stop
                    
            else:  # short
                atr_stop = current_price + atr_stop_distance
                
                # Find nearest swing high for structure-based stop
                recent_swing_highs = df['swing_high'].dropna().tail(10)
                if len(recent_swing_highs) > 0:
                    structure_stop = recent_swing_highs.max()
                    # Use the higher of ATR stop or structure stop (more conservative)
                    final_stop = max(atr_stop, structure_stop)
                else:
                    final_stop = atr_stop
            
            logger.info(f"Dynamic stop loss for {direction} position: ${final_stop:.4f}")
            return final_stop
            
        except Exception as e:
            logger.error(f"Error calculating dynamic stop loss: {str(e)}")
            # Fallback to simple ATR-based stop
            if direction == 'long':
                return current_price * 0.98  # 2% stop
            else:
                return current_price * 1.02  # 2% stop
    
    @staticmethod
    def calculate_position_size(capital: float, risk_pct: float, entry_price: float,
                              stop_loss: float) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            capital: Available capital
            risk_pct: Risk percentage per trade (e.g., 1.0 for 1%)
            entry_price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Position size
        """
        try:
            risk_amount = capital * (risk_pct / 100)
            price_diff = abs(entry_price - stop_loss)
            
            if price_diff == 0:
                return 0
            
            position_size = risk_amount / price_diff
            
            # Ensure position doesn't exceed available capital
            max_position = capital * 0.95  # Use max 95% of capital
            position_value = position_size * entry_price
            
            if position_value > max_position:
                position_size = max_position / entry_price
            
            logger.info(f"Calculated position size: {position_size:.6f} (Risk: ${risk_amount:.2f})")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0