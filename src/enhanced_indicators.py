"""
Enhanced Technical Indicators Module for Auto_bot
Provides additional indicators for multi-confluence signal generation
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class EnhancedIndicators:
    """
    Enhanced technical indicators for improved signal quality and confluence.
    """
    
    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average Directional Index (ADX) for trend strength measurement.
        ADX > 25 indicates strong trend, < 20 indicates weak/ranging market.
        """
        try:
            result_df = df.copy()
            adx_data = ta.adx(result_df['high'], result_df['low'], result_df['close'], length=period)
            
            result_df[f'adx_{period}'] = adx_data[f'ADX_{period}']
            result_df[f'dmp_{period}'] = adx_data[f'DMP_{period}']  # Positive Directional Movement
            result_df[f'dmn_{period}'] = adx_data[f'DMN_{period}']  # Negative Directional Movement
            
            logger.info(f"Successfully calculated ADX with period {period}")
            return result_df
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            raise
    
    @staticmethod
    def add_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """
        Add Stochastic Oscillator for momentum confirmation.
        Values > 80 indicate overbought, < 20 indicate oversold.
        """
        try:
            result_df = df.copy()
            stoch = ta.stoch(result_df['high'], result_df['low'], result_df['close'], 
                           k=k_period, d=d_period)
            
            result_df[f'stoch_k_{k_period}'] = stoch[f'STOCHk_{k_period}_{d_period}_{d_period}']
            result_df[f'stoch_d_{k_period}'] = stoch[f'STOCHd_{k_period}_{d_period}_{d_period}']
            
            logger.info(f"Successfully calculated Stochastic with periods K={k_period}, D={d_period}")
            return result_df
        except Exception as e:
            logger.error(f"Error calculating Stochastic: {str(e)}")
            raise
    
    @staticmethod
    def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Williams %R for momentum confirmation.
        Values > -20 indicate overbought, < -80 indicate oversold.
        """
        try:
            result_df = df.copy()
            result_df[f'williams_r_{period}'] = ta.willr(result_df['high'], result_df['low'], 
                                                        result_df['close'], length=period)
            
            logger.info(f"Successfully calculated Williams %R with period {period}")
            return result_df
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {str(e)}")
            raise
    
    @staticmethod
    def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Add Commodity Channel Index (CCI) for trend and momentum analysis.
        Values > 100 indicate overbought, < -100 indicate oversold.
        """
        try:
            result_df = df.copy()
            result_df[f'cci_{period}'] = ta.cci(result_df['high'], result_df['low'], 
                                               result_df['close'], length=period)
            
            logger.info(f"Successfully calculated CCI with period {period}")
            return result_df
        except Exception as e:
            logger.error(f"Error calculating CCI: {str(e)}")
            raise
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average True Range (ATR) for volatility measurement.
        """
        try:
            result_df = df.copy()
            result_df[f'atr_{period}'] = ta.atr(result_df['high'], result_df['low'], 
                                               result_df['close'], length=period)
            
            logger.info(f"Successfully calculated ATR with period {period}")
            return result_df
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            raise
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Add volume-based indicators for confirmation.
        """
        try:
            result_df = df.copy()
            
            # Volume Moving Average
            result_df[f'volume_ma_{period}'] = ta.sma(result_df['volume'], length=period)
            
            # Volume Ratio (current volume vs average)
            result_df['volume_ratio'] = result_df['volume'] / result_df[f'volume_ma_{period}']
            
            # On Balance Volume (OBV)
            result_df['obv'] = ta.obv(result_df['close'], result_df['volume'])
            
            # Volume Weighted Average Price (VWAP) - approximation
            result_df['vwap'] = ta.vwap(result_df['high'], result_df['low'], 
                                       result_df['close'], result_df['volume'])
            
            logger.info(f"Successfully calculated volume indicators with period {period}")
            return result_df
        except Exception as e:
            logger.error(f"Error calculating volume indicators: {str(e)}")
            raise
    
    @staticmethod
    def add_ema_suite(df: pd.DataFrame, periods: list = [9, 21, 50, 200]) -> pd.DataFrame:
        """
        Add multiple EMAs for trend analysis and confluence.
        """
        try:
            result_df = df.copy()
            
            for period in periods:
                result_df[f'ema_{period}'] = ta.ema(result_df['close'], length=period)
            
            logger.info(f"Successfully calculated EMAs for periods {periods}")
            return result_df
        except Exception as e:
            logger.error(f"Error calculating EMAs: {str(e)}")
            raise
    
    @staticmethod
    def calculate_all_enhanced_indicators(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Calculate all enhanced indicators for comprehensive analysis.
        """
        if config is None:
            config = {
                'adx_period': 14,
                'stoch_k': 14,
                'stoch_d': 3,
                'williams_period': 14,
                'cci_period': 20,
                'atr_period': 14,
                'volume_period': 20,
                'ema_periods': [9, 21, 50, 200]
            }
        
        try:
            logger.info("Calculating all enhanced technical indicators")
            result_df = df.copy()
            
            # Add all enhanced indicators
            result_df = EnhancedIndicators.add_adx(result_df, config['adx_period'])
            result_df = EnhancedIndicators.add_stochastic(result_df, config['stoch_k'], config['stoch_d'])
            result_df = EnhancedIndicators.add_williams_r(result_df, config['williams_period'])
            result_df = EnhancedIndicators.add_cci(result_df, config['cci_period'])
            result_df = EnhancedIndicators.add_atr(result_df, config['atr_period'])
            result_df = EnhancedIndicators.add_volume_indicators(result_df, config['volume_period'])
            result_df = EnhancedIndicators.add_ema_suite(result_df, config['ema_periods'])
            
            # Drop NaN values
            result_df = result_df.dropna()
            
            logger.info("Successfully calculated all enhanced indicators")
            return result_df
            
        except Exception as e:
            logger.error(f"Error calculating enhanced indicators: {str(e)}")
            raise