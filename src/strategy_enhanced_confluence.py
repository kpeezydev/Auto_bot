"""
Enhanced Confluence Strategy for Auto_bot
Multi-indicator confluence system with improved risk management
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional
from src.strategy import Strategy, SignalType
from src.enhanced_indicators import EnhancedIndicators
try:
    from src.market_structure import MarketStructure
except ImportError:
    # Fallback to simplified version if scipy is not available
    from src.market_structure_simple import MarketStructure
from src.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

class EnhancedConfluenceStrategy(Strategy):
    """
    Enhanced strategy using multi-indicator confluence for higher winning rate.
    
    Key Features:
    - Multi-indicator confluence scoring
    - Dynamic risk management
    - Market structure analysis
    - Trend strength filtering
    - Volume confirmation
    """
    
    def __init__(self,
                 # Confluence parameters
                 min_confluence_score: float = 0.6,
                 trend_strength_threshold: float = 0.3,
                 volume_confirmation: bool = True,
                 
                 # Risk management
                 risk_per_trade_pct: float = 1.0,
                 min_risk_reward_ratio: float = 2.0,
                 use_dynamic_stops: bool = True,
                 atr_stop_multiplier: float = 2.0,
                 
                 # Indicator parameters
                 fast_ma_period: int = 12,
                 slow_ma_period: int = 26,
                 rsi_period: int = 14,
                 adx_period: int = 14,
                 stoch_period: int = 14,
                 
                 # Market regime filters
                 min_adx_strength: float = 25.0,
                 max_trades_per_day: int = 3,
                 min_trade_spacing_hours: int = 4):
        """
        Initialize the Enhanced Confluence Strategy.
        """
        super().__init__(name="Enhanced Confluence Strategy")
        
        # Confluence parameters
        self.min_confluence_score = min_confluence_score
        self.trend_strength_threshold = trend_strength_threshold
        self.volume_confirmation = volume_confirmation
        
        # Risk management
        self.risk_per_trade_pct = risk_per_trade_pct
        self.min_risk_reward_ratio = min_risk_reward_ratio
        self.use_dynamic_stops = use_dynamic_stops
        self.atr_stop_multiplier = atr_stop_multiplier
        
        # Indicator parameters
        self.fast_ma_period = fast_ma_period
        self.slow_ma_period = slow_ma_period
        self.rsi_period = rsi_period
        self.adx_period = adx_period
        self.stoch_period = stoch_period
        
        # Market regime filters
        self.min_adx_strength = min_adx_strength
        self.max_trades_per_day = max_trades_per_day
        self.min_trade_spacing_hours = min_trade_spacing_hours
        
        # Trade tracking
        self.last_trade_time = None
        self.trades_today = 0
        self.current_day = None
        
        logger.info(f"Initialized {self.name} with confluence threshold: {min_confluence_score}")
    
    def calculate_confluence_score(self, df: pd.DataFrame, index: int) -> Tuple[float, Dict[str, float]]:
        """
        Calculate confluence score based on multiple indicators.
        
        Returns:
            Tuple of (total_score, individual_scores)
        """
        try:
            row = df.iloc[index]
            scores = {}
            
            # 1. Moving Average Confluence (Weight: 0.2)
            ma_score = 0.0
            if f'ema_{self.fast_ma_period}' in df.columns and f'ema_{self.slow_ma_period}' in df.columns:
                fast_ma = row[f'ema_{self.fast_ma_period}']
                slow_ma = row[f'ema_{self.slow_ma_period}']
                price = row['close']
                
                if price > fast_ma > slow_ma:
                    ma_score = 1.0  # Strong bullish
                elif price > fast_ma and fast_ma < slow_ma:
                    ma_score = 0.5  # Weak bullish
                elif price < fast_ma < slow_ma:
                    ma_score = -1.0  # Strong bearish
                elif price < fast_ma and fast_ma > slow_ma:
                    ma_score = -0.5  # Weak bearish
            
            scores['ma_confluence'] = ma_score * 0.2
            
            # 2. RSI Momentum (Weight: 0.15)
            rsi_score = 0.0
            if f'rsi_{self.rsi_period}' in df.columns:
                rsi = row[f'rsi_{self.rsi_period}']
                if 30 < rsi < 70:
                    rsi_score = 0.5  # Neutral zone, good for trend following
                elif rsi > 70:
                    rsi_score = -0.3  # Overbought, bearish bias
                elif rsi < 30:
                    rsi_score = 0.3  # Oversold, bullish bias
            
            scores['rsi_momentum'] = rsi_score * 0.15
            
            # 3. Stochastic Confirmation (Weight: 0.1)
            stoch_score = 0.0
            if f'stoch_k_{self.stoch_period}' in df.columns:
                stoch_k = row[f'stoch_k_{self.stoch_period}']
                stoch_d = row[f'stoch_d_{self.stoch_period}']
                
                if stoch_k > stoch_d and stoch_k < 80:
                    stoch_score = 0.5  # Bullish crossover, not overbought
                elif stoch_k < stoch_d and stoch_k > 20:
                    stoch_score = -0.5  # Bearish crossover, not oversold
            
            scores['stochastic'] = stoch_score * 0.1
            
            # 4. MACD Trend (Weight: 0.15)
            macd_score = 0.0
            if 'macd' in df.columns and 'macd_signal' in df.columns:
                macd = row['macd']
                macd_signal = row['macd_signal']
                macd_hist = row['macd_histogram']
                
                if macd > macd_signal and macd_hist > 0:
                    macd_score = 1.0  # Strong bullish
                elif macd > macd_signal:
                    macd_score = 0.5  # Weak bullish
                elif macd < macd_signal and macd_hist < 0:
                    macd_score = -1.0  # Strong bearish
                elif macd < macd_signal:
                    macd_score = -0.5  # Weak bearish
            
            scores['macd_trend'] = macd_score * 0.15
            
            # 5. Volume Confirmation (Weight: 0.1)
            volume_score = 0.0
            if self.volume_confirmation and 'volume_ratio' in df.columns:
                volume_ratio = row['volume_ratio']
                if volume_ratio > 1.5:
                    volume_score = 0.5  # High volume confirmation
                elif volume_ratio > 1.2:
                    volume_score = 0.3  # Moderate volume
                elif volume_ratio < 0.8:
                    volume_score = -0.2  # Low volume, reduce confidence
            
            scores['volume_confirmation'] = volume_score * 0.1
            
            # 6. Trend Strength (Weight: 0.15)
            trend_score = 0.0
            if 'trend_score' in df.columns:
                trend_strength = row['trend_score']
                trend_score = trend_strength * 0.8  # Scale down slightly
            
            scores['trend_strength'] = trend_score * 0.15
            
            # 7. Williams %R (Weight: 0.1)
            williams_score = 0.0
            if f'williams_r_{self.rsi_period}' in df.columns:
                williams = row[f'williams_r_{self.rsi_period}']
                if -50 < williams < -20:
                    williams_score = 0.3  # Moderate bullish
                elif williams > -20:
                    williams_score = -0.3  # Overbought
                elif williams < -80:
                    williams_score = 0.3  # Oversold, bullish
            
            scores['williams_r'] = williams_score * 0.1
            
            # 8. CCI Confirmation (Weight: 0.05)
            cci_score = 0.0
            if 'cci_20' in df.columns:
                cci = row['cci_20']
                if -100 < cci < 100:
                    cci_score = 0.2  # Normal range
                elif cci > 100:
                    cci_score = -0.2  # Overbought
                elif cci < -100:
                    cci_score = 0.2  # Oversold
            
            scores['cci_confirmation'] = cci_score * 0.05
            
            # Calculate total score
            total_score = sum(scores.values())
            
            return total_score, scores
            
        except Exception as e:
            logger.error(f"Error calculating confluence score: {str(e)}")
            return 0.0, {}
    
    def check_market_conditions(self, df: pd.DataFrame, index: int) -> bool:
        """
        Check if market conditions are suitable for trading.
        """
        try:
            row = df.iloc[index]
            
            # Check ADX for trend strength
            if f'adx_{self.adx_period}' in df.columns:
                adx = row[f'adx_{self.adx_period}']
                if adx < self.min_adx_strength:
                    logger.debug(f"ADX too low: {adx:.2f} < {self.min_adx_strength}")
                    return False
            
            # Check daily trade limit
            current_time = df.index[index]
            current_date = current_time.date() if hasattr(current_time, 'date') else None
            
            if current_date != self.current_day:
                self.current_day = current_date
                self.trades_today = 0
            
            if self.trades_today >= self.max_trades_per_day:
                logger.debug(f"Daily trade limit reached: {self.trades_today}")
                return False
            
            # Check minimum spacing between trades
            if self.last_trade_time is not None:
                time_diff = current_time - self.last_trade_time
                hours_diff = time_diff.total_seconds() / 3600 if hasattr(time_diff, 'total_seconds') else 0
                
                if hours_diff < self.min_trade_spacing_hours:
                    logger.debug(f"Trade spacing too small: {hours_diff:.1f}h < {self.min_trade_spacing_hours}h")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking market conditions: {str(e)}")
            return False
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using multi-indicator confluence.
        """
        try:
            logger.info("Generating signals using Enhanced Confluence Strategy")
            
            # Ensure we have all required indicators
            df = self._ensure_indicators(df)
            
            # Add market structure analysis
            df = MarketStructure.find_swing_points(df)
            df = MarketStructure.calculate_trend_strength(df)
            
            result_df = df.copy()
            result_df['signal'] = SignalType.HOLD.value
            result_df['confluence_score'] = 0.0
            result_df['signal_strength'] = 0.0
            
            # Generate signals for each row
            for i in range(50, len(result_df)):  # Start after enough data for indicators
                try:
                    # Check market conditions
                    if not self.check_market_conditions(result_df, i):
                        continue
                    
                    # Calculate confluence score
                    score, individual_scores = self.calculate_confluence_score(result_df, i)
                    result_df.iloc[i, result_df.columns.get_loc('confluence_score')] = score
                    
                    # Generate signal based on confluence score
                    if score >= self.min_confluence_score:
                        result_df.iloc[i, result_df.columns.get_loc('signal')] = SignalType.BUY.value
                        result_df.iloc[i, result_df.columns.get_loc('signal_strength')] = score
                        self.trades_today += 1
                        self.last_trade_time = result_df.index[i]
                        
                        logger.info(f"BUY signal generated at {result_df.index[i]} with score {score:.3f}")
                        
                    elif score <= -self.min_confluence_score:
                        result_df.iloc[i, result_df.columns.get_loc('signal')] = SignalType.SELL.value
                        result_df.iloc[i, result_df.columns.get_loc('signal_strength')] = abs(score)
                        self.trades_today += 1
                        self.last_trade_time = result_df.index[i]
                        
                        logger.info(f"SELL signal generated at {result_df.index[i]} with score {score:.3f}")
                
                except Exception as e:
                    logger.error(f"Error processing row {i}: {str(e)}")
                    continue
            
            # Count signals
            buy_signals = (result_df['signal'] == SignalType.BUY.value).sum()
            sell_signals = (result_df['signal'] == SignalType.SELL.value).sum()
            
            logger.info(f"Generated {buy_signals} BUY and {sell_signals} SELL signals")
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise
    
    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all required indicators are calculated.
        """
        try:
            result_df = df.copy()
            
            # Add basic indicators if not present
            if f'sma_{self.fast_ma_period}' not in result_df.columns:
                result_df = TechnicalIndicators.calculate_all_indicators(result_df)
            
            # Add enhanced indicators
            result_df = EnhancedIndicators.calculate_all_enhanced_indicators(result_df)
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error ensuring indicators: {str(e)}")
            raise
    
    def get_risk_management_params(self, df: pd.DataFrame, signal: SignalType, 
                                 entry_price: float) -> Dict[str, float]:
        """
        Get risk management parameters for a trade.
        """
        try:
            if self.use_dynamic_stops:
                direction = 'long' if signal == SignalType.BUY else 'short'
                stop_loss = MarketStructure.get_dynamic_stop_loss(
                    df, direction, entry_price, self.atr_stop_multiplier
                )
            else:
                # Fixed percentage stop
                if signal == SignalType.BUY:
                    stop_loss = entry_price * 0.98  # 2% stop
                else:
                    stop_loss = entry_price * 1.02  # 2% stop
            
            # Calculate take profit based on risk-reward ratio
            risk = abs(entry_price - stop_loss)
            if signal == SignalType.BUY:
                take_profit = entry_price + (risk * self.min_risk_reward_ratio)
            else:
                take_profit = entry_price - (risk * self.min_risk_reward_ratio)
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'risk_amount': risk,
                'reward_amount': risk * self.min_risk_reward_ratio
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk management params: {str(e)}")
            return {
                'stop_loss': entry_price * 0.98 if signal == SignalType.BUY else entry_price * 1.02,
                'take_profit': entry_price * 1.04 if signal == SignalType.BUY else entry_price * 0.96,
                'risk_amount': entry_price * 0.02,
                'reward_amount': entry_price * 0.04
            }